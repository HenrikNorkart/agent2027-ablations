"""Query-level audit trail for every retrieval the agents perform.

baGY Q1 asked for a query-level audit trail and the rebuttal conceded that none
could be produced for the reported runs ("This is fixed going forward."). This
module is that fix: a append-only JSONL transcript of every retrieval call made
by any agent, written alongside the run's other artifacts.

Two design constraints drive the shape of this module.

*It must never change a result.* Every public entry point swallows its own
exceptions and returns ``None``; a broken transcript degrades to no transcript,
never to a failed run. Nothing here is on a path that feeds the evaluator.

*It must be testable without the retrieval stack.* Like
``rogueone.utils.console``, this lives in ``utils`` rather than beside the
agents so the record format can be exercised without importing ``agents``,
``langchain`` or ``chroma``. The agents keep only the call.

Destination resolution, in precedence order:

1. an explicit :func:`configure` call,
2. ``$ROGUE_ONE_RETRIEVAL_TRANSCRIPT`` (a file path),
3. :func:`configure_from_output_dir`, i.e. ``<output_dir>/retrieval_transcript.jsonl``.

When none of those has happened the recorder is a no-op, so importing it from a
unit test costs nothing and writes nothing.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from logging import getLogger
from pathlib import Path
from typing import Any

logger = getLogger(__name__)

#: Environment variable naming an explicit transcript file.
ENV_TRANSCRIPT_PATH = "ROGUE_ONE_RETRIEVAL_TRANSCRIPT"

#: Set to "0"/"false" to store only a preview of each retrieved result.
ENV_TRANSCRIPT_FULL = "ROGUE_ONE_RETRIEVAL_TRANSCRIPT_FULL"

#: Filename used when the destination is derived from a run's ``output_dir``.
TRANSCRIPT_FILENAME = "retrieval_transcript.jsonl"

#: Hard ceiling on a single stored result, so one pathological response cannot
#: grow the transcript without bound. Preview length when full storage is off.
_MAX_RESULT_CHARS = 20_000
_PREVIEW_CHARS = 500

_lock = threading.Lock()
_path: Path | None = None
_seq = 0


def _truthy(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def configure(path: str | Path | None) -> Path | None:
    """Point the recorder at ``path``, creating its parent directory.

    Passing ``None`` disables recording. Returns the resolved path, or ``None``
    if recording is now disabled or the destination could not be prepared.
    """
    global _path
    if path is None:
        with _lock:
            _path = None
        return None
    try:
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            _path = resolved
        logger.info("Retrieval transcript: %s", resolved)
        return resolved
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never break a run
        logger.warning("Could not open retrieval transcript at %s: %s", path, exc)
        with _lock:
            _path = None
        return None


def configure_from_output_dir(output_dir: str | Path) -> Path | None:
    """Write the transcript to ``<output_dir>/retrieval_transcript.jsonl``.

    An explicit ``$ROGUE_ONE_RETRIEVAL_TRANSCRIPT`` wins over ``output_dir`` so a
    single run can be redirected without touching its config.
    """
    override = os.getenv(ENV_TRANSCRIPT_PATH)
    if override:
        return configure(override)
    return configure(Path(output_dir) / TRANSCRIPT_FILENAME)


def transcript_path() -> Path | None:
    """Return the active transcript path, resolving the env var on first use."""
    global _path
    with _lock:
        if _path is not None:
            return _path
    override = os.getenv(ENV_TRANSCRIPT_PATH)
    if override:
        return configure(override)
    return None


def is_enabled() -> bool:
    """True when a destination is resolvable and records will be written."""
    return transcript_path() is not None


def reset_for_testing() -> None:
    """Drop all recorder state. Intended for tests, not for run code."""
    global _path, _seq
    with _lock:
        _path = None
        _seq = 0


def _clip(text: str) -> tuple[str, bool]:
    """Return the stored form of ``text`` and whether it was truncated."""
    keep_full = _truthy(os.getenv(ENV_TRANSCRIPT_FULL), default=True)
    limit = _MAX_RESULT_CHARS if keep_full else _PREVIEW_CHARS
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def record(
    *,
    channel: str,
    event: str,
    caller: str,
    query: str | None = None,
    result: Any = None,
    ok: bool | None = None,
    error: str | None = None,
    latency_s: float | None = None,
    **extra: Any,
) -> dict[str, Any] | None:
    """Append one retrieval record. Returns the record written, or ``None``.

    ``channel`` is the retrieval mechanism (``"web_search"``, ``"vector_db"``),
    ``event`` the stage (``"query"`` for an individual lookup, ``"answer"`` for
    an agent's synthesised reply). ``caller`` names the agent that asked.

    ``result`` is stringified; ``None`` is preserved distinctly from ``""``
    because "the tool returned nothing" and "the tool returned an empty string"
    are different failures and the distinction is the point of this transcript.
    """
    global _seq
    path = transcript_path()
    if path is None:
        return None

    try:
        rec: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "channel": channel,
            "event": event,
            "caller": caller,
        }
        if query is not None:
            rec["query"] = str(query)
            rec["query_chars"] = len(str(query))
        if result is None:
            rec["result"] = None
            rec["result_chars"] = 0
            rec["result_is_none"] = True
        else:
            text = str(result)
            stored, truncated = _clip(text)
            rec["result"] = stored
            rec["result_chars"] = len(text)
            rec["result_is_none"] = False
            if truncated:
                rec["result_truncated"] = True
        if ok is not None:
            rec["ok"] = bool(ok)
        if error is not None:
            rec["error"] = str(error)
        if latency_s is not None:
            rec["latency_s"] = round(float(latency_s), 4)
        for key, value in extra.items():
            if value is not None:
                rec[key] = value

        with _lock:
            _seq += 1
            rec["seq"] = _seq
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        return rec
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never break a run
        logger.debug("Retrieval transcript record failed: %s", exc)
        return None


class timed_retrieval:
    """Context manager recording one retrieval, with latency and outcome.

    Set :attr:`result` inside the block; on exit the record is written. An
    exception propagates unchanged after being recorded as a failure, so this
    cannot swallow a real error from the surrounding code.
    """

    def __init__(self, *, channel: str, event: str, caller: str, query: str | None = None, **extra: Any):
        self._fields = dict(channel=channel, event=event, caller=caller, query=query)
        self._extra = extra
        self.result: Any = None
        self._t0 = 0.0

    def __enter__(self) -> "timed_retrieval":
        self._t0 = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = time.monotonic() - self._t0
        record(
            **self._fields,
            result=self.result,
            ok=exc_type is None,
            error=None if exc is None else f"{exc_type.__name__}: {exc}",
            latency_s=elapsed,
            **self._extra,
        )
        return False
