from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import traceback
from collections import Counter
from typing import Any

from rogueone.utils import instrumentation


_MAX_ERROR_LEN = 2000
_FILTERED_INPUT_MARKER = "filtered.input="

# A07 error-site taxonomy: the site recorded when the caller does not pass one
# (the ``_report_tool_error`` helper of each agent: h1 Scientist, h4 Extractor,
# h9 Tester). Runner exceptions are not tool errors and get no site.
_DEFAULT_SITE_BY_AGENT = {
    "ScientistAgent": "h1",
    "ExtractorAgent": "h4",
    "TesterAgent": "h9",
}


@dataclass(frozen=True)
class ToolCallErrorRecord:
    """Structured representation of a failed agent tool call."""

    timestamp_utc: str
    agent_name: str
    tool_name: str
    message: str
    exception_type: str
    exception_message: str
    traceback: str
    step: int | None = None
    session_name: str | None = None
    context: dict[str, Any] | None = None

    @property
    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "agent_name": self.agent_name,
            "tool_name": self.tool_name,
            "message": self.message,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "traceback": self.traceback,
            "step": self.step,
            "session_name": self.session_name,
            "context": self.context,
        }


class ToolErrorReporter:
    """Collect and persist tool-call errors for post-run analysis."""

    def __init__(
        self,
        *,
        agent_name: str,
        output_dir: Path | str | None,
        session_name: str | None = None,
        file_name: str = "agent_toolcall_errors.jsonl",
    ):
        base_dir = Path(output_dir) if output_dir is not None else Path("./output")
        base_dir.mkdir(parents=True, exist_ok=True)

        self.agent_name = agent_name
        self.session_name = session_name
        self.file_path = base_dir / file_name
        self._records: list[ToolCallErrorRecord] = []
        # E9f (set_phased): the SET phase (S, E, T-A, T-B); None under three_agent.
        self.phase: str | None = None

    @staticmethod
    def _sanitize(text: str) -> str:
        sanitized = text or ""

        if _FILTERED_INPUT_MARKER in sanitized:
            prefix = sanitized.split(_FILTERED_INPUT_MARKER, 1)[0].rstrip()
            sanitized = (
                f"{prefix} {_FILTERED_INPUT_MARKER}<omitted large payload>".strip()
            )

        if len(sanitized) > _MAX_ERROR_LEN:
            return sanitized[:_MAX_ERROR_LEN] + "... [truncated]"

        return sanitized

    def report(
        self,
        *,
        tool_name: str,
        message: str,
        exception: Exception,
        step: int | None = None,
        context: dict[str, Any] | None = None,
        kind: str | None = None,
    ) -> ToolCallErrorRecord:
        # E3: tag every tool-error record with its A07 site (``context.site``).
        if tool_name not in ("runner", "runner_fallback"):
            site = (context or {}).get("site") or _DEFAULT_SITE_BY_AGENT.get(self.agent_name)
            if site:
                context = {**(context or {}), "site": site}
        if self.phase is not None:  # E9f only: the shared session name no longer names the role
            context = {**(context or {}), "phase": self.phase}
        record = ToolCallErrorRecord(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            agent_name=self.agent_name,
            tool_name=tool_name,
            message=self._sanitize(message),
            exception_type=type(exception).__name__,
            exception_message=self._sanitize(str(exception)),
            traceback=self._sanitize(
                "".join(
                    traceback.format_exception(
                        type(exception), exception, exception.__traceback__
                    )
                )
            ),
            step=step,
            session_name=self.session_name,
            context=context,
        )

        self._records.append(record)

        # Single counting point for the E3 error counters (summary: err_site_*, ...).
        instrumentation.note_tool_error(
            self.agent_name,
            tool_name,
            site=(context or {}).get("site"),
            exception_type=type(exception).__name__,
            message=message,
            kind=kind,
        )

        try:
            with self.file_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record.as_dict, ensure_ascii=True) + "\n")
        except Exception:
            # Never let error reporting break the main agent flow.
            pass

        return record

    @property
    def records(self) -> list[ToolCallErrorRecord]:
        return list(self._records)

    @property
    def total_errors(self) -> int:
        return len(self._records)

    @property
    def counts_by_tool(self) -> dict[str, int]:
        return dict(Counter(record.tool_name for record in self._records))


def report_bypass(
    reporter: "ToolErrorReporter | None",
    *,
    site: str,
    tool_name: str,
    message: str,
    step: int | None = None,
    code: str | None = None,
    kind: str | None = None,
    agent_name: str | None = None,
) -> None:
    """Record an error site that bypasses ``_report_tool_error`` (A07: h2, h3, h5-h8, v1, v3).

    Writes one JSONL record with ``context.site`` and counts it. The agent-facing
    return value of the calling tool is not touched. Never raises.
    """
    try:
        context: dict[str, Any] = {"site": site}
        if code is not None:
            context["code"] = code
        if reporter is not None:
            reporter.report(
                tool_name=tool_name,
                message=message,
                exception=ValueError(message),
                step=step,
                context=context,
                kind=kind,
            )
        else:
            instrumentation.note_tool_error(
                agent_name or "", tool_name, site=site,
                exception_type="ValueError", message=message, kind=kind,
            )
    except Exception:
        pass
