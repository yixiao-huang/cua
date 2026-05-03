"""Tests for has_done_signal — the OpenClaw task-completion detector.

The detector must match the convention in
``cua_bench/agents/openclaw/AGENTS.md``: DONE on its own line ends the run.
Incidental substrings like "JOB DONE" inside running commentary must NOT
terminate the loop (regression: silicon_bse_absorption mid-task termination
on the line `no "JOB DONE" message in BerkeleyGW kernel`).
"""

import pytest

# Importing the openclaw agent_loop transitively pulls in the cua-agent SDK
# (`agent.agent`, `agent.computers.*`, etc.). The cua-bench package's own
# default test env does not install that SDK, so skip these tests there.
# In benchmark envs (and the agenthle orchestration env) where the SDK is
# present, the tests run and exercise the real has_done_signal.
pytest.importorskip("agent.agent")
from cua_bench.agents.openclaw.agent_loop import has_done_signal  # noqa: E402


def _msg(text):
    return [{"type": "message", "content": [{"type": "output_text", "text": text}]}]


def _msg_str(text):
    return [{"type": "message", "content": text}]


class TestHasDoneSignalPositive:
    def test_done_alone(self):
        assert has_done_signal(_msg("DONE"))

    def test_done_with_trailing_summary(self):
        assert has_done_signal(_msg("DONE: task complete"))

    def test_done_after_context(self):
        assert has_done_signal(_msg("Verified the file exists.\nDONE"))

    def test_done_bolded(self):
        assert has_done_signal(_msg("**DONE**"))

    def test_done_lowercase(self):
        assert has_done_signal(_msg("done"))

    def test_done_string_content(self):
        assert has_done_signal(_msg_str("All steps verified.\nDONE"))

    def test_done_in_later_block(self):
        msg = [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "Some intro text"},
                    {"type": "output_text", "text": "DONE"},
                ],
            }
        ]
        assert has_done_signal(msg)

    def test_done_with_inline_summary(self):
        # `DONE my task is complete` — space-separated trailing text.
        assert has_done_signal(_msg("DONE the task is complete"))

    def test_done_with_leading_whitespace(self):
        assert has_done_signal(_msg("   DONE"))


class TestHasDoneSignalNegative:
    def test_job_done_in_sentence(self):
        # Regression: BerkeleyGW kernel discussion ended a run mid-task.
        text = (
            'Kernel completed (no "JOB DONE" message in BerkeleyGW kernel - '
            "just timing). The `bsedmat` and `bsexmat` files are created."
        )
        assert not has_done_signal(_msg(text))

    def test_done_inside_word(self):
        assert not has_done_signal(_msg("The kernel is DONEISH but not finished"))

    def test_done_inline_in_sentence(self):
        assert not has_done_signal(_msg("Once we are DONE we will move on."))

    def test_inline_done_with_following_text(self):
        # "Now let me set up..." mid-sentence — DONE not at line start.
        assert not has_done_signal(
            _msg("Status: DONE on the kernel side, but absorption pending.")
        )

    def test_no_message_items(self):
        assert not has_done_signal([{"type": "function_call", "name": "exec"}])

    def test_empty_output(self):
        assert not has_done_signal([])


class TestRealWorldDonePatterns:
    """Patterns observed in past openclaw runs (audited across 985
    transcripts). The dominant real-world completion is `Done — <summary>`
    with em-dash. The case-sensitive substring check missed all of these."""

    def test_done_emdash_summary(self):
        # The most common Cat C pattern in past runs.
        text = (
            "Done — I tested every available tool I could safely check, "
            "and wrote the JSON report here:\n\n"
            "`/path/to/report.json`"
        )
        assert has_done_signal(_msg(text))

    def test_done_capitalized_with_path(self):
        text = "Done — I wrote the required file here:\n\n`/path/output.json`"
        assert has_done_signal(_msg(text))

    def test_done_followed_by_full_path_listing(self):
        # robotics URDF case from past run.
        text = (
            "Done — I reconstructed the URDF and saved it here:\n\n"
            "`/media/user/data/.../base/output/submission.urdf`\n\n"
            "I also verified:\n- valid XML parse\n- all required 12 links present"
        )
        assert has_done_signal(_msg(text))


class TestRealWorldFalsePositives:
    """False-positive patterns observed in past runs that previously
    triggered the buggy substring match. Each must NOT terminate."""

    def test_quoted_done_in_console_observation(self):
        # Past run: agent observing the literal string "DONE" in a screenshot.
        text = (
            'The script executed successfully - I can see "DONE" in the '
            "console and the form fields on the left show data (SSN: "
            "487-61-3295, ZIP: 92101, US home checked). Let me close "
            "DevTools and verify the form."
        )
        assert not has_done_signal(_msg(text))


class TestRegressionFromTrajectory:
    """Verify against the exact assistant outputs captured in the failed
    silicon_bse_absorption run (turn_147 + earlier 'X done.' commentary)."""

    def test_silicon_bse_turn_147_does_not_terminate(self):
        # Verbatim from
        # .logs/simprun/openclaw_cua__claude-sonnet-4-6/tasks/
        #   materials_science__silicon_bse_absorption/variants/v0/
        #   20260502_215542/debug/trajectories/.../turn_147/0572_agent_response.json
        # Pre-fix this triggered premature termination.
        output = [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            'Kernel completed (no "JOB DONE" message in '
                            "BerkeleyGW kernel - just timing). The `bsedmat` "
                            "and `bsexmat` files are created. Now let me set "
                            "up the absorption calculation:"
                        ),
                    }
                ],
            },
            {
                "type": "function_call",
                "call_id": "toolu_bdrk_01DZmhDbC9KUAKkoaHEnZ5e2",
                "name": "exec",
                "arguments": "{}",
            },
        ]
        assert not has_done_signal(output)

    def test_intermediate_done_commentary_does_not_terminate(self):
        # Other assistant lines from the same run that mention "done"
        # in passing — none should terminate the loop.
        for text in [
            "SCF done. Now let me create the NSCF input for BerkeleyGW WFN.",
            "NSCF done. Now create the NSCF for q-shifted grid (WFNq):",
            "Bands calculation done. Now let me run pw2bgw.",
        ]:
            assert not has_done_signal(_msg(text)), text
