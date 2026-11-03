"""Unit tests for tool-call error reporting."""

import json

from rogueone.llm.agents.core.tool_error_reporting import ToolErrorReporter


def test_tool_error_reporter_writes_jsonl_and_counts(tmp_path):
    reporter = ToolErrorReporter(
        agent_name="TesterAgent",
        output_dir=tmp_path,
        session_name="tester_step_1",
    )

    try:
        raise ValueError("invalid tool input")
    except Exception as exc:
        reporter.report(
            tool_name="generic_python_executor_tool",
            message="Tool failed while executing user code",
            exception=exc,
            step=1,
            context={"code": "result = 1/0"},
        )

    assert reporter.total_errors == 1
    assert reporter.counts_by_tool == {"generic_python_executor_tool": 1}
    assert reporter.file_path.exists()

    lines = reporter.file_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    payload = json.loads(lines[0])
    assert payload["agent_name"] == "TesterAgent"
    assert payload["tool_name"] == "generic_python_executor_tool"
    assert payload["exception_type"] == "ValueError"
    assert payload["step"] == 1
    assert payload["session_name"] == "tester_step_1"


def test_tool_error_reporter_sanitizes_large_payloads(tmp_path):
    reporter = ToolErrorReporter(agent_name="ScientistAgent", output_dir=tmp_path)

    huge_error = RuntimeError("failed with filtered.input=" + ("x" * 5000))
    reporter.report(
        tool_name="runner",
        message="Runner crashed with filtered.input=" + ("y" * 5000),
        exception=huge_error,
        step=2,
    )

    payload = json.loads(reporter.file_path.read_text(encoding="utf-8").strip())

    assert "filtered.input=<omitted large payload>" in payload["message"]
    assert "filtered.input=<omitted large payload>" in payload["exception_message"]
    assert len(payload["message"]) <= 2100
    assert len(payload["exception_message"]) <= 2100
