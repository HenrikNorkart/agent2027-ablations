"""E9f (AGENT 2027 A16 RT2): ``ROGUE_ONE_TOPOLOGY=three_agent|set_phased``.

``three_agent`` (default; unset or empty means ``three_agent``)
    Full-v2. Every role builds its own ``Agent``, system prompt and ``SQLiteSession``.
    Nothing in this module is called on that path.
``set_phased``
    SET-phased (A16 §2, A16-T5…T7). The same calls in the same order, with:

    1. **one shared session per iteration** (:class:`SharedIterationSession`): one
       ``SQLiteSession("set_phased_trial_{index}")`` is created in ``_forward_pass``
       after the fold reset and handed (``shared_session=``) to ``determine_focus``,
       every Extractor round, and Tester Stage A and Stage B;
    2. **one neutral system prompt** (``rogueone.llm.prompts.set_phased``), identical in
       every phase of the iteration;
    3. **role rules in the phase messages**: each role's workflow and hard rules go
       verbatim (after the frozen substitution table) into the first message of its
       phase, behind ``### PHASE k/4 — <NAME>. …``. Retries and Extractor rounds 2..R
       send only the phase's trailing input ("header-once" rule).

    Each phase keeps its own client, ``ModelSettings`` and tool list; the generating
    phases (S, E) never hold the Tester's labelled executor. The Scientist's no-tool
    fallback keeps its own fresh session; every fallback is counted per phase.

    Counters (summary keys only under ``set_phased``): ``offphase_tool_calls`` (caught
    ``Tool … not found in agent …`` exceptions) and per-phase context errors, calls and
    fallbacks; ``output/set_phased_log.jsonl`` holds one row per iteration with the
    per-phase values and the S2 "exhausted by context errors" flag.

A16 never combines with another switch: under ``set_phased`` every other AGENT 2027
switch must sit at its Full-v2 value (:func:`check_exclusive`), else config load
raises. Standard library only (the SDK session is imported lazily).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

TOPOLOGY_ENV = "ROGUE_ONE_TOPOLOGY"
THREE_AGENT = "three_agent"
SET_PHASED = "set_phased"
TOPOLOGIES: tuple[str, ...] = (THREE_AGENT, SET_PHASED)

#: Phase ids (error-record ``phase`` field) -> (k, NAME) of the frozen phase header.
PHASE_S, PHASE_E, PHASE_TA, PHASE_TB = "S", "E", "T-A", "T-B"
PHASES: tuple[str, ...] = (PHASE_S, PHASE_E, PHASE_TA, PHASE_TB)
#: Phase id -> suffix used in counter / summary key names.
PHASE_KEY = {PHASE_S: "s", PHASE_E: "e", PHASE_TA: "ta", PHASE_TB: "tb"}

SESSION_NAME = "set_phased_trial_{index}"
LOG_FILE = "set_phased_log.jsonl"

#: SDK 0.9.1 ``run_internal/turn_resolution.py``: ``ModelBehaviorError(f"Tool {name}
#: not found in agent {agent.name}")`` (0.3.3 ``_run_impl.py:558-559``, same text).
OFFPHASE_RE = re.compile(r"Tool \S+ not found in agent ")

#: Other switches and their Full-v2 values; under set_phased they must hold these.
_EXCLUSIVE_FULL_VALUES: dict[str, tuple[str, ...]] = {
    "ROGUE_ONE_PLANNER_MODE": ("", "llm"),
    "ROGUE_ONE_MEMORY_MODE": ("", "full"),
    "ROGUE_ONE_ASSESSMENT_MODE": ("", "llm"),
    "ROGUE_ONE_PRUNING_MODE": ("", "llm"),
    "ROGUE_ONE_PRUNING_SHADOW": ("", "0", "false", "off"),
    "ROGUE_ONE_RESTART_MODE": ("", "off"),
    "ROGUE_ONE_FEEDBACK_LEVEL": ("", "full"),
    "ROGUE_ONE_KNOWLEDGE_BACKEND": ("", "web"),
}


def parse_topology(value: Optional[str]) -> str:
    """Validate ``ROGUE_ONE_TOPOLOGY`` (unset/empty = ``three_agent``)."""
    mode = (value if value is not None else "").strip().lower() or THREE_AGENT
    if mode not in TOPOLOGIES:
        raise ValueError(f"{TOPOLOGY_ENV} must be one of {TOPOLOGIES}, got {value!r}")
    return mode


def topology(environ: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if environ is None else environ
    return parse_topology(env.get(TOPOLOGY_ENV))


def is_set_phased(environ: Optional[Mapping[str, str]] = None) -> bool:
    return topology(environ) == SET_PHASED


def check_exclusive(mode: str, environ: Optional[Mapping[str, str]] = None) -> None:
    """Under ``set_phased`` every other switch must be at its Full-v2 value (A16 §4)."""
    if mode != SET_PHASED:
        return
    env = os.environ if environ is None else environ
    bad = {
        k: env.get(k)
        for k, ok in _EXCLUSIVE_FULL_VALUES.items()
        if (env.get(k) or "").strip().lower() not in ok
    }
    if bad:
        raise ValueError(
            f"{TOPOLOGY_ENV}=set_phased does not combine with other switches (A16 §4); "
            f"set these to their Full-v2 values: {bad}"
        )


def is_offphase_error(exc: BaseException | str) -> bool:
    return bool(OFFPHASE_RE.search(str(exc)))


def strip_reasoning_and_special_tokens(data: Any) -> Any:
    """The S/E ``call_model_input_filter`` (``sci/agent.py`` ``_strip_reasoning``).

    Under ``set_phased`` the T phases use it too (A16-T7), because they now see S and
    E content in the shared history. Full's Tester keeps its own filter.
    """
    from agents.run import ModelInputData  # lazy: SDK only at run time

    filtered = []
    for item in data.model_data.input:
        if isinstance(item, dict) and item.get("type") == "reasoning":
            continue
        if isinstance(item, dict) and isinstance(item.get("content"), str):
            cleaned_item = dict(item)
            cleaned_item["content"] = re.sub(r"<\|[^|]+\|>", " ", cleaned_item["content"])
            filtered.append(cleaned_item)
        else:
            filtered.append(item)
    return ModelInputData(input=filtered, instructions=data.model_data.instructions)


def _default_session_factory(name: str) -> Any:
    from agents.memory.sqlite_session import SQLiteSession  # lazy

    return SQLiteSession(name)


class SharedIterationSession:
    """Everything the four phases of one SET iteration share.

    ``session``: the one ``SQLiteSession`` of the iteration (empty when created).
    ``system_prompt``: the neutral system prompt, identical for every phase.
    :meth:`input_for` implements the header-once rule.
    """

    def __init__(
        self,
        index: int,
        system_prompt: str,
        *,
        output_dir: Any = None,
        session_factory: Callable[[str], Any] = _default_session_factory,
    ):
        from rogueone.utils import instrumentation

        self.index = int(index)
        self.session_name = SESSION_NAME.format(index=self.index)
        self.session = session_factory(self.session_name)
        self.system_prompt = system_prompt
        self.output_dir = output_dir
        self._opened: set[str] = set()
        self.messages_sent: dict[str, int] = {p: 0 for p in PHASES}
        self.offphase: dict[str, int] = {p: 0 for p in PHASES}
        self.fallbacks: dict[str, list[str]] = {p: [] for p in PHASES}
        self._ctx0 = {p: instrumentation.get_counter(ctx_counter(p)) for p in PHASES}
        self._calls0 = {p: instrumentation.get_counter(calls_counter(p)) for p in PHASES}
        self.closed = False

    # -------------------------------------------------------------- phase messages
    def input_for(self, phase: str, phase_message: str, trailing: str) -> str:
        """First call of ``phase``: the full phase message; afterwards ``trailing`` only
        (S retries, E rounds 2..R, T-A/T-B attempts 2..20 re-send what Full re-sends)."""
        from rogueone.utils import instrumentation

        instrumentation.set_phase(phase)
        self.messages_sent[phase] += 1
        if phase in self._opened:
            return trailing
        self._opened.add(phase)
        return phase_message

    def enter(self, phase: str) -> None:
        """Tag the calls made from now on with ``phase`` (context errors, trace)."""
        from rogueone.utils import instrumentation

        instrumentation.set_phase(phase)

    def opened(self, phase: str) -> bool:
        return phase in self._opened

    # -------------------------------------------------------------- counters
    def note_runner_exception(self, phase: str, exc: BaseException) -> bool:
        """Count an off-phase tool call (``Tool … not found in agent …``)."""
        from rogueone.utils import instrumentation

        if not is_offphase_error(exc):
            return False
        self.offphase[phase] += 1
        instrumentation.count(offphase_counter(phase))
        return True

    def note_fallback(self, phase: str, kind: str) -> None:
        from rogueone.utils import instrumentation

        self.fallbacks[phase].append(kind)
        instrumentation.count(fallback_counter(phase))

    def phase_stats(self) -> dict[str, dict[str, Any]]:
        from rogueone.utils import instrumentation

        out: dict[str, dict[str, Any]] = {}
        for p in PHASES:
            ctx = instrumentation.get_counter(ctx_counter(p)) - self._ctx0[p]
            calls = instrumentation.get_counter(calls_counter(p)) - self._calls0[p]
            fb = list(self.fallbacks[p])
            out[p] = {
                "messages": self.messages_sent[p],
                "llm_calls": calls,
                "context_errors": ctx,
                "offphase_tool_calls": self.offphase[p],
                "fallbacks": fb,
                # A16 §4 S2: the phase ended in its fallback AND >= 1 context error.
                "exhausted": bool(fb) and ctx > 0,
            }
        return out

    def close(self, fold_run_id: Optional[int] = None, iteration_in_fold: Optional[int] = None) -> dict:
        """End of iteration: log per-phase values, count S2 exhaustion. Idempotent."""
        from rogueone.utils import instrumentation

        if self.closed:
            return {}
        self.closed = True
        instrumentation.set_phase(None)
        stats = self.phase_stats()
        n_exh = sum(1 for v in stats.values() if v["exhausted"])
        if n_exh:
            instrumentation.count("set_phase_exhausted", n_exh)
        row = {
            "index": self.index,
            "fold_run_id": fold_run_id,
            "iteration_in_fold": iteration_in_fold,
            "session_name": self.session_name,
            "phases": stats,
        }
        try:
            base = Path(self.output_dir) if self.output_dir is not None else Path("output")
            base.mkdir(parents=True, exist_ok=True)
            with open(base / LOG_FILE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return row


def offphase_counter(phase: str) -> str:
    return f"set_offphase_{PHASE_KEY[phase]}"


def ctx_counter(phase: str) -> str:
    return f"set_context_errors_{PHASE_KEY[phase]}"


def calls_counter(phase: str) -> str:
    return f"set_llm_calls_{PHASE_KEY[phase]}"


def fallback_counter(phase: str) -> str:
    return f"set_fallback_{PHASE_KEY[phase]}"


def summary_keys(counters: Mapping[str, Any], environ: Optional[Mapping[str, str]] = None) -> dict:
    """A16 keys for ``compute_summary``; empty unless ``set_phased`` (Full unchanged)."""
    try:
        if not is_set_phased(environ):
            return {}
    except ValueError:
        return {}
    g = lambda k: int(counters.get(k, 0))  # noqa: E731
    s: dict[str, Any] = {"topology": SET_PHASED}
    for p in PHASES:
        k = PHASE_KEY[p]
        s[f"offphase_tool_calls_{k}"] = g(offphase_counter(p))
        s[f"context_errors_{k}"] = g(ctx_counter(p))
        s[f"llm_calls_phase_{k}"] = g(calls_counter(p))
        s[f"fallbacks_{k}"] = g(fallback_counter(p))
    s["offphase_tool_calls"] = sum(s[f"offphase_tool_calls_{PHASE_KEY[p]}"] for p in PHASES)
    s["set_phase_exhausted"] = g("set_phase_exhausted")
    return s
