"""Rendered-prompt dump (AGENT 2027 E5).

With ``ROGUE_ONE_PROMPT_DUMP=1`` every agent call writes its fully rendered
system prompt, first user message and tool schemas to ``<output_dir>/prompts/``
so that two runs/commits can be compared byte-for-byte with
``scripts/agent2027/prompt_diff.py``.

File naming (flat, one triple per agent call)::

    <role>__<phase>__iter<NNN>__c<SEQ>.system.txt
    <role>__<phase>__iter<NNN>__c<SEQ>.user.txt
    <role>__<phase>__iter<NNN>__c<SEQ>.tools.json

``SEQ`` counts calls with the same (role, phase, iteration) within the process
(two-digit, from 01). Files are written as UTF-8 with no newline translation.
The dump is a pure side effect: it never raises into the caller.

Directory resolution: ``ROGUE_ONE_PROMPT_DUMP_DIR`` if set, else the explicit
``output_dir`` argument + ``/prompts``, else ``./output/prompts``.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Optional

FLAG = "ROGUE_ONE_PROMPT_DUMP"
DIR_ENV = "ROGUE_ONE_PROMPT_DUMP_DIR"

_lock = threading.Lock()
_counters: dict[tuple[str, str, int], int] = {}


def is_enabled() -> bool:
    return os.getenv(FLAG, "0").strip().lower() in ("1", "true", "yes", "on")


def reset_counters() -> None:
    """Reset the per-process call counters (used by tests)."""
    with _lock:
        _counters.clear()


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", str(text)).strip("-").lower() or "x"


def _resolve_dir(output_dir: Optional[Any]) -> Path:
    env = os.getenv(DIR_ENV)
    if env:
        return Path(env)
    if output_dir is not None:
        return Path(output_dir) / "prompts"
    return Path("output") / "prompts"


def _tools_payload(tools: Optional[list]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tool in tools or []:
        out.append(
            {
                "name": getattr(tool, "name", type(tool).__name__),
                "description": getattr(tool, "description", None),
                "params_json_schema": getattr(tool, "params_json_schema", None),
            }
        )
    return out


def _write(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def dump_prompt(
    role: str,
    phase: str,
    iteration: Optional[int],
    system: Any,
    user: Any,
    tools: Optional[list] = None,
    output_dir: Optional[Any] = None,
    once: bool = False,
) -> Optional[Path]:
    """Write one rendered prompt triple. No-op unless the flag is on.

    ``once=True`` writes only the first call per (role, phase, iteration) in
    the process; used for the web-search helper, whose call count is
    data-dependent while its rendered prompt is static.
    """
    if not is_enabled():
        return None
    try:
        it = int(iteration or 0)
        key = (_slug(role), _slug(phase), it)
        with _lock:
            if once and key in _counters:
                return None
            _counters[key] = _counters.get(key, 0) + 1
            seq = _counters[key]
        directory = _resolve_dir(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"{key[0]}__{key[1]}__iter{it:03d}__c{seq:02d}"
        _write(directory / f"{stem}.system.txt", "" if system is None else str(system))
        _write(directory / f"{stem}.user.txt", "" if user is None else str(user))
        _write(
            directory / f"{stem}.tools.json",
            json.dumps(_tools_payload(tools), indent=2, sort_keys=True, ensure_ascii=False),
        )
        return directory / stem
    except Exception:  # never break an experiment because of the dump
        return None
