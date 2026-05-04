"""OpenClaw Agent — faithful reproduction of OpenClaw's agent-side architecture for CUA.

Adapts OpenClaw's context management (system prompt construction, compaction pipeline,
memory recall, tool loop, session persistence) to CUA's constraints:
  - instructions= for persistent context (only content never truncated)
  - Trajectory-based observation (CUA reasoning in trajectory files, not conversation)

US-OC-017: Uses OpenClawComputerAgent subclass for mid-loop compaction instead of
the stop-compact-resume pattern. Compaction happens in-place inside run() — no
agent rebuild needed.

Subclassing surface (US-OC-073):
  Wrappers that need to specialize behavior (e.g. orchestration's
  ``openclaw-cua`` adapter that nests logs under ``origin_log/``, pre-builds a
  custom computer handler, writes an interaction log, etc.) should subclass
  ``OpenClawAgent`` and override the protected hooks below rather than
  recreating ``perform_task`` from scratch. This prevents the wrapper from
  silently drifting when fixes land here:

    - ``_AGENT_CLASS``                 — agent class to instantiate
    - ``_LOOP_EXIT_FAILURE_MODE``      — failure mode for loop exit w/o done-signal
    - ``_default_lightweight_model``   — auto-sibling lookup for ``lightweight_model``
    - ``_default_summary_model``       — fallback when ``summary_model`` kwarg absent
    - ``_default_gui_model``           — fallback when ``gui_model`` kwarg absent
    - ``_resolve_paths``               — trajectory_dir / memory_base / session_base / task_id
    - ``_resolve_workspace_root``      — VM workspace root for FS tools
    - ``_make_computer_handler``       — pre-built CUA computer handler (else auto)
    - ``_filter_tools``                — post-filter on tool list (e.g. disabled_tools)
    - ``_after_run_finally``           — finalization hook (log writers, etc.)

  Subclasses re-register under their own name via ``@register_agent(...)``;
  the upstream ``"openclaw-agent"`` registration here is independent.

References:
  - docs/openclaw-source-analysis.md — OpenClaw source code analysis
  - docs/openclaw-context-flow.html — interactive visual pipeline
  - openclaw/docs/concepts/ — component-level docs
  - architecture.md — AgentHLE system architecture
"""

import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from . import register_agent
from .base import AgentResult, BaseAgent, FailureMode
from .openclaw.agent_loop import OpenClawComputerAgent, has_done_signal
from .openclaw.model_config import resolve_model

if TYPE_CHECKING:
    from ..computers import DesktopSession


_LIGHTWEIGHT_SIBLINGS: dict[str, str] = {
    "gpt-5.4": "gpt-5.4-mini",
}


def _derive_lightweight_model(model: str) -> str | None:
    """Map a main-agent model string to its cheaper sibling, when known.

    Lookup is suffix-based on the segment after the last ``/`` so the same
    rule works for direct provider IDs (``openai/gpt-5.4``) and routed IDs
    (``openrouter/openai/gpt-5.4``). Returns ``None`` when no sibling is
    registered — the delegate tools then expose only the default.
    """
    if not model:
        return None
    suffix = model.rsplit("/", 1)[-1]
    sibling_suffix = _LIGHTWEIGHT_SIBLINGS.get(suffix)
    if sibling_suffix is None:
        return None
    if "/" not in model:
        return sibling_suffix
    prefix = model.rsplit("/", 1)[0]
    return f"{prefix}/{sibling_suffix}"


@register_agent("openclaw-agent")
class OpenClawAgent(BaseAgent):
    """OpenClaw agent reproduction for CUA benchmark framework.

    Subclasses can override the ``_*`` hook methods listed in the module
    docstring to specialize behavior without recreating ``perform_task``.
    """

    # ------------------------------------------------------------------
    # Subclass-overridable surface (see module docstring)
    # ------------------------------------------------------------------
    _AGENT_CLASS: ClassVar[type] = OpenClawComputerAgent

    # FailureMode reported when the agent loop exits without a done-signal
    # AND without hitting max_steps. Upstream treats this as success
    # (NONE); some wrappers want UNKNOWN so eval metrics don't conflate
    # this with a real success.
    _LOOP_EXIT_FAILURE_MODE: ClassVar[FailureMode] = FailureMode.NONE

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = kwargs.get("model", "openrouter/anthropic/claude-sonnet-4-20250514")
        self.max_steps = kwargs.get("max_steps", 100)
        self.max_history_turns = kwargs.get("max_history_turns", None)  # None = all
        # When True, the main agent has no direct ``computer`` tool — all GUI
        # work must go through ``delegate_gui``. Used to validate the
        # post-delegation screenshot injection path end-to-end (US-SUB-006).
        self.disable_main_computer = bool(kwargs.get("disable_main_computer", False))
        # When True, ``DelegateGUITool`` is omitted from the tool list — the
        # main ``computer`` tool handles all GUI work. Mirrors OpenClaw's
        # filterToolsByPolicy pattern; absence is the signal.
        self.disable_delegate_gui = bool(kwargs.get("disable_delegate_gui", False))
        if self.disable_main_computer and self.disable_delegate_gui:
            raise ValueError(
                "Both disable_main_computer and disable_delegate_gui set — "
                "the agent has no way to interact with the VM."
            )
        # Image retention mode (US-OC-072). "openclaw" (default) keeps all
        # images from last N completed turns (OpenClaw-parity); "cua" keeps
        # last N images by count (CUA-default). Both modes use sticky
        # placeholder replacement. Default flipped from "count" to "openclaw"
        # after on-task verification (cache-thrash-image-retention.md).
        self.image_retention_mode = kwargs.get("image_retention_mode", "openclaw")
        # Optional lightweight sibling exposed to delegate tools as the
        # second enum option. Explicit override wins; otherwise resolve via
        # the ``_default_lightweight_model`` hook.
        self.lightweight_model = kwargs.get("lightweight_model") or self._default_lightweight_model(
            self.model
        )
        # Separate model for summarization and memory flush. Explicit override
        # wins; otherwise resolve via the ``_default_summary_model`` hook
        # (subclasses can default to ``lightweight_model`` for cost savings).
        self.summary_model = kwargs.get("summary_model") or self._default_summary_model()
        # GUI subagent model. Explicit override wins; otherwise resolve via
        # the ``_default_gui_model`` hook.
        self.gui_model = kwargs.get("gui_model") or self._default_gui_model()

        # Thinking level configuration (US-OC-019)
        # CLI --thinking-level overrides auto-detection; omitting uses model default.
        from .openclaw.thinking import (ThinkingConfig, ThinkLevel,
                                        resolve_thinking_default)

        thinking_level_str = kwargs.get("thinking_level")
        if thinking_level_str is not None:
            level = ThinkLevel(thinking_level_str)
        else:
            level = resolve_thinking_default(self.model)
        flush_level_str = kwargs.get("flush_thinking_level")
        compaction_level_str = kwargs.get("compaction_thinking_level")
        vision_level_str = kwargs.get("vision_thinking_level")
        gui_level_str = kwargs.get("gui_thinking_level")
        flush_level = ThinkLevel(flush_level_str) if flush_level_str is not None else level
        compaction_level = (
            ThinkLevel(compaction_level_str) if compaction_level_str is not None else level
        )
        vision_level = (
            ThinkLevel(vision_level_str) if vision_level_str is not None else ThinkLevel.OFF
        )
        gui_level = ThinkLevel(gui_level_str) if gui_level_str is not None else ThinkLevel.OFF
        self.thinking_config = ThinkingConfig(
            level=level,
            flush_level=flush_level,
            compaction_level=compaction_level,
            vision_level=vision_level,
            gui_level=gui_level,
        )

    @staticmethod
    def name() -> str:
        return "openclaw-agent"

    # ------------------------------------------------------------------
    # Hook defaults
    # ------------------------------------------------------------------

    def _default_lightweight_model(self, model: str) -> str | None:
        """Map ``model`` to a cheaper sibling, or None when none is registered."""
        return _derive_lightweight_model(model)

    def _default_summary_model(self) -> str:
        """Fallback when ``summary_model`` kwarg is absent. Defaults to main model."""
        return self.model

    def _default_gui_model(self) -> str | None:
        """Fallback when ``gui_model`` kwarg is absent. Defaults to None (= main model)."""
        return None

    def _resolve_paths(self, logging_dir: Path | None) -> dict[str, Any]:
        """Resolve filesystem layout for this run.

        Returns a dict with:
          - ``trajectory_dir``: where per-turn API payloads are written.
          - ``memory_base``:    base dir for ``MemoryStore`` (None → default).
          - ``session_base``:   base dir for ``SessionManager`` (None → default).
          - ``task_id``:        memory/session keying string.

        Default: trajectories sit under ``logging_dir/trajectories``, memory and
        session use upstream defaults, ``task_id`` is the parent directory
        name. Subclasses can nest under ``origin_log/``, wipe-on-entry, walk
        the path for a smarter task_id, etc.
        """
        if logging_dir is None:
            return {
                "trajectory_dir": None,
                "memory_base": None,
                "session_base": None,
                "task_id": "default",
            }
        trajectory_dir = logging_dir / "trajectories"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        return {
            "trajectory_dir": trajectory_dir,
            "memory_base": None,
            "session_base": None,
            "task_id": logging_dir.parent.name,
        }

    def _resolve_workspace_root(self, session: "DesktopSession") -> str | None:
        """Workspace root passed to FS tools. None → permissive (no bound).

        Default: Windows-style path under ``REMOTE_ROOT_DIR`` /
        ``TASK_CATEGORY`` / ``TASK_TAG`` (matches the Windows VM image used
        for CUA bench tasks). When ``TASK_TAG`` is unset, returns None.
        Subclasses can detect Linux vs Windows etc.
        """
        task_tag = os.environ.get("TASK_TAG", "").strip()
        if not task_tag:
            return None
        root_dir = os.environ.get("REMOTE_ROOT_DIR", r"C:\Users\User\Desktop")
        category = os.environ.get("TASK_CATEGORY", "tasks")
        return f"{root_dir}\\{category}\\{task_tag}"

    def _make_computer_handler(self, session: "DesktopSession"):
        """Optional pre-built computer handler passed to ``build_tools``.

        Default: return None — let ``build_tools`` auto-instantiate the
        upstream CUA handler. Subclasses can return a pre-initialized
        custom ``AsyncComputerHandler`` (avoids module-level monkey-patches).
        """
        return None

    def _filter_tools(self, tools: list) -> list:
        """Optional post-filter on the assembled tool list. Default: identity."""
        return tools

    def _after_run_finally(
        self,
        *,
        logging_dir: Path | None,
        instruction: str,
        total_usage: dict,
        started_at: float,
    ) -> None:
        """Finalization hook called from ``perform_task``'s ``finally``.

        Default: no-op. Subclasses can write an interaction log, sync
        artifacts, etc. Always called, even if the agent raises.
        """

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def perform_task(
        self,
        task_description: str,
        session: "DesktopSession",
        logging_dir: Path | None = None,
        tracer=None,
    ) -> AgentResult:
        """Perform a task using the OpenClawComputerAgent with mid-loop compaction.

        Uses ``OpenClawComputerAgent`` (US-OC-017) which handles compaction
        in-place inside ``run()`` — no stop-compact-resume pattern needed.
        """
        try:
            from agent import \
                ComputerAgent  # noqa: F401 — validate package is installed
        except ImportError as e:
            raise RuntimeError(
                f"{self.name()} requires the CUA `agent` package. " "Run: uv sync --reinstall"
            ) from e

        instruction = self._render_instruction(task_description)

        # Filesystem layout (subclass-controlled).
        paths = self._resolve_paths(logging_dir)
        trajectory_dir: Path | None = paths.get("trajectory_dir")
        memory_base = paths.get("memory_base")
        session_base = paths.get("session_base")
        task_id: str = paths.get("task_id") or "default"

        # Build structured system prompt via PromptBuilder (US-OC-001)
        from .openclaw import (ContextFile, ContextOverflowCallback,
                               MemoryStore, PromptBuilder, SessionManager,
                               SubagentRegistry, ToolLoggingCallback,
                               build_replay_messages,
                               build_system_prompt_report, build_tools,
                               convert_to_responses_api_items,
                               get_tool_summaries, limit_history_turns,
                               sanitize_history)

        # Initialize memory store (US-OC-002).
        if memory_base is not None:
            memory_store = MemoryStore(task_id=task_id, base_dir=memory_base)
        else:
            memory_store = MemoryStore(task_id=task_id)
        memory_store.init_session()

        # Initialize session persistence (US-OC-004).
        if session_base is not None:
            session_mgr = SessionManager(task_id=task_id, base_dir=session_base)
        else:
            session_mgr = SessionManager(task_id=task_id)
        session_mgr.init_session(model=self.model)

        # Cross-run continuity (US-OC-012): replay prior transcript as messages
        # so the agent sees actual conversation history from previous runs.
        prior_entries = session_mgr.load_history()
        replay_messages: list[dict[str, Any]] = []
        if prior_entries:
            replay_messages = build_replay_messages(prior_entries)
            replay_messages = sanitize_history(replay_messages)
            replay_messages = limit_history_turns(replay_messages, self.max_history_turns)
            # Re-sanitize after truncation (may orphan tool results at cut point)
            replay_messages = sanitize_history(replay_messages)
            # Unnest Chat Completions messages into Responses API items (US-OC-022)
            replay_messages = convert_to_responses_api_items(replay_messages)
            if replay_messages:
                print(f"[Replay] Loaded {len(replay_messages)} items from prior transcript")

        resolved_model = resolve_model(self.model)
        resolved_summary_model = (
            resolved_model
            if self.summary_model == self.model
            else resolve_model(self.summary_model)
        )

        # Subagent registry (US-SUB-005/007) — disk-backed when session is active.
        persist_path = session_mgr.task_dir / "subagent-runs.jsonl"
        registry = SubagentRegistry(persist_path=persist_path)
        registry.restore()

        # Resolve context window up-front so tools (adaptive paging in
        # ReadFileTool) and the later ContextOverflowCallback share the
        # same number. Honors CONTEXT_WINDOW_OVERRIDE for testing.
        from .openclaw.context import (DEFAULT_CONTEXT_TOKENS,
                                       resolve_context_window)

        ctx_override = os.environ.get("CONTEXT_WINDOW_OVERRIDE")
        if ctx_override:
            context_window_tokens = int(ctx_override)
        else:
            context_window_tokens = (
                resolved_model.context_window
                or resolve_context_window(self.model)
                or DEFAULT_CONTEXT_TOKENS
            )

        # Workspace root for FS-tool path policy (US-OC-055).
        workspace_root = self._resolve_workspace_root(session)

        # Host workspace root for `target='host'` on read/write/edit. Operator
        # override via OPENCLAW_HOST_WORKSPACE; otherwise pin per-task to the
        # MemoryStore task dir so the agent's relative writes
        # (memory/session-NNN.md, TASK_MEMORY.md) round-trip into the host
        # memory store rather than landing at the repo root or the VM's
        # cua-server cwd.
        host_override = os.environ.get("OPENCLAW_HOST_WORKSPACE", "").strip()
        if host_override:
            host_workspace_root: str | None = str(Path(host_override).resolve())
        else:
            host_workspace_root = str(memory_store.task_dir.resolve())

        # Tool assembly (US-OC-007 + US-SUB-005 delegation tools + US-OC-055 fs tools)
        thinking_api_params = self.thinking_config.to_api_params(self.model)
        gui_model_str = self.gui_model or self.model
        gui_thinking_params = self.thinking_config.gui_params(gui_model_str)

        # Pre-built computer handler (subclass-controlled). When None,
        # build_tools auto-instantiates the upstream CUA handler.
        computer_handler = None
        if not self.disable_main_computer:
            computer_handler = self._make_computer_handler(session)
            if computer_handler is not None and hasattr(computer_handler, "_initialize"):
                # Pre-initialize so build_tools' isinstance() short-circuit
                # accepts it (the SDK's make_computer_handler() returns
                # AsyncComputerHandler instances as-is).
                await computer_handler._initialize()

        tools = build_tools(
            session,
            memory_store,
            summary_model=self.summary_model,
            vision_thinking_params=self.thinking_config.vision_params(
                self.summary_model,
                runtime=resolved_summary_model,
            ),
            registry=registry,
            parent_session_dir=session_mgr.task_dir,
            default_model=self.model,
            lightweight_model=self.lightweight_model,
            thinking_params=thinking_api_params,
            gui_thinking_params=gui_thinking_params,
            disable_main_computer=self.disable_main_computer,
            disable_delegate_gui=self.disable_delegate_gui,
            gui_model=self.gui_model,
            workspace_root=workspace_root,
            host_workspace_root=host_workspace_root,
            context_window_tokens=context_window_tokens,
            computer_handler=computer_handler,
        )
        tools = self._filter_tools(tools)
        tool_summaries = get_tool_summaries(tools)

        # AGENTS.md ships next to the openclaw subpackage. Locate via the
        # imported package so wrappers don't need to know the path.
        from . import openclaw as _openclaw_pkg

        agents_md = (Path(_openclaw_pkg.__file__).parent / "AGENTS.md").read_text()

        # Build context files, injecting TASK_MEMORY.md if it exists.
        # Note: task description is NOT injected here — it's passed separately
        # via agent.run(instruction) to avoid duplication in context.
        context_files = [
            ContextFile(path="AGENTS.md", content=agents_md),
        ]
        bootstrap = memory_store.get_bootstrap_context()
        if bootstrap:
            context_files.append(ContextFile(path="TASK_MEMORY.md", content=bootstrap))

        builder = PromptBuilder()
        instructions = builder.build(
            tool_summaries=tool_summaries,
            context_files=context_files,
        )

        # System prompt report for observability (US-OC-008)
        report = build_system_prompt_report(
            system_prompt=instructions,
            context_files=context_files,
            tool_summaries=tool_summaries,
            tools=tools,
        )
        session_mgr.set_system_prompt_report(report)

        # Context overflow detection (US-OC-005).
        overflow_cb = ContextOverflowCallback(
            model=self.model,
            context_window=context_window_tokens,
            instructions_tokens=len(instructions) // 4,
            resolved_model=resolved_model,
        )

        # Persist resolved context window in session state (matches OpenClaw's contextTokens)
        if session_mgr._state is not None:
            session_mgr._state.contextTokens = overflow_cb.context_window
            session_mgr.save_state()

        # Create OpenClawComputerAgent with mid-loop compaction support (US-OC-017)
        # overflow_cb is auto-injected into callbacks by OpenClawComputerAgent (US-OC-028)
        # Thinking params flow to ComputerAgent's additional_generation_kwargs (US-OC-019)
        tool_logging_cb = ToolLoggingCallback()
        # use_prompt_caching=True trips the gate in
        # ``UnifiedAgentConfig.predict_step`` that calls
        # ``apply_openclaw_cache_markers`` on Anthropic-family models.
        agent = self._AGENT_CLASS(
            # ComputerAgent params
            model=self.model,
            tools=tools,
            only_n_most_recent_images=3,
            trajectory_dir=trajectory_dir,
            instructions=instructions,
            use_prompt_caching=True,
            callbacks=[tool_logging_cb],
            # Re-injected as a user message after each compaction (US-OC-070
            # post-compaction context refresh — mirrors OpenClaw default).
            context_files=context_files,
            # US-OC-072 — see OpenClawComputerAgent docstring for modes.
            image_retention_mode=self.image_retention_mode,
            # Only the explicit ``screenshot`` action returns an image —
            # click/type/keypress/etc. return their tool result as text.
            auto_screenshot=False,
            # OpenClaw compaction params
            overflow_cb=overflow_cb,
            session_mgr=session_mgr,
            memory_store=memory_store,
            summary_model=self.summary_model,
            # Thinking config (US-OC-019/020)
            thinking_config=self.thinking_config,
            resolved_model=resolved_model,
            summary_runtime=resolved_summary_model,
            # Subagent delegation (US-SUB-005)
            registry=registry,
            # Provider-specific thinking kwargs → ComputerAgent additional_generation_kwargs
            **thinking_api_params,
        )
        print("OpenClaw Agent initialized with model:", self.model)
        if self.lightweight_model:
            print("  Lightweight model:", self.lightweight_model)
        if self.summary_model != self.model:
            print("  Summary/flush model:", self.summary_model)
        if self.gui_model and self.gui_model != self.model:
            print("  GUI subagent model:", self.gui_model)
        if self.disable_main_computer:
            print("  Main computer tool DISABLED — GUI work must go through delegate_gui")
        if self.disable_delegate_gui:
            print("  delegate_gui DISABLED — GUI work goes through the main computer tool")
        if self.thinking_config.level.value != "off":
            print("  Thinking level:", self.thinking_config.level.value)
        if self.thinking_config.flush_level != self.thinking_config.level:
            print("  Flush thinking level:", self.thinking_config.flush_level.value)
        if self.thinking_config.compaction_level != self.thinking_config.level:
            print("  Compaction thinking level:", self.thinking_config.compaction_level.value)
        if self.thinking_config.vision_level.value != "off":
            print("  Vision thinking level:", self.thinking_config.vision_level.value)
        if self.thinking_config.gui_level.value != "off":
            print("  GUI thinking level:", self.thinking_config.gui_level.value)
        # Always print the resolved mode so users can confirm which threshold
        # is in effect — both have meaningful trade-offs and the default
        # changed at US-OC-072 verification time.
        print(f"  Image retention mode: {self.image_retention_mode}")

        # Hoist out of the try so the finally block can pass them to the
        # finalization hook even if the agent raises before assigning.
        total_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "response_cost": 0.0,
        }
        agent_run_start = time.time()

        # Single-loop execution (US-OC-017).
        # Compaction happens in-place inside OpenClawComputerAgent.run() — no
        # stop-compact-resume pattern needed. Reactive overflow is also handled
        # inside the custom run() via try/except around predict_step().
        try:
            step = 0
            step_offset = session_mgr.get_step_count()
            task_completed = False

            # Pass replay messages + instruction as the messages arg (US-OC-012).
            # When no prior history, replay_messages is empty and this is
            # equivalent to agent.run(instruction).
            run_input = (
                replay_messages + [{"role": "user", "content": instruction}]
                if replay_messages
                else instruction
            )

            async for result in agent.run(run_input):
                sys.stdout.flush()
                step += 1
                for k in total_usage:
                    total_usage[k] += result["usage"].get(k, 0)

                # Session persistence tracking (US-OC-004)
                step_input = result["usage"].get("input_tokens", 0)
                step_output = result["usage"].get("output_tokens", 0)
                session_mgr.update_step_count(step_offset + step)
                session_mgr.update_tokens(step_input, step_output)

                # Tracer recording (optional)
                if tracer:
                    await _record_tracer_step(
                        tracer, session, step, self.name(), self.model, result
                    )

                if step >= self.max_steps:
                    print(f"\n[Max steps reached] Stopped at step {step}/{self.max_steps}")
                    break

                task_completed = has_done_signal(result.get("output", []))
                if task_completed:
                    print(f"\n[Task completed] Agent indicated completion at step {step}")
                    break

            print(f"\nTotal usage: {total_usage}")
            print(f"Steps completed: {step}/{self.max_steps}")
            if agent.compaction_count > 0:
                print(f"Compactions performed: {agent.compaction_count}")

            # Determine failure mode
            if task_completed:
                failure_mode = FailureMode.NONE
            elif step >= self.max_steps:
                failure_mode = FailureMode.MAX_STEPS_EXCEEDED
            else:
                failure_mode = self._LOOP_EXIT_FAILURE_MODE

            return AgentResult(
                total_input_tokens=total_usage.get("input_tokens", 0),
                total_output_tokens=total_usage.get("output_tokens", 0),
                failure_mode=failure_mode,
            )
        except Exception as e:
            print(f"Agent execution failed: {e}")
            import traceback

            traceback.print_exc()
            return AgentResult(
                total_input_tokens=total_usage.get("input_tokens", 0),
                total_output_tokens=total_usage.get("output_tokens", 0),
                failure_mode=FailureMode.UNKNOWN,
            )
        finally:
            self._after_run_finally(
                logging_dir=logging_dir,
                instruction=instruction,
                total_usage=total_usage,
                started_at=agent_run_start,
            )


async def _record_tracer_step(
    tracer, session, step: int, agent_name: str, model: str, result: dict
) -> None:
    """Record an agent step to the tracer (optional observability)."""
    try:
        screenshot = await session.screenshot()
        tracer.record(
            "agent_step",
            {
                "step": step,
                "agent": agent_name,
                "model": model,
                "usage": result["usage"],
                "output": result["output"],
            },
            [screenshot],
        )
    except Exception as e:
        print(f"Warning: Failed to record agent step to tracer: {e}")
