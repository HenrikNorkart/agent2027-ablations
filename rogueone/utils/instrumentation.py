"""Run instrumentation (AGENT 2027 E3): tokens, LLM traces, counters, focus log.

Everything here is a pure side channel. No entry point may raise into the
experiment (each one is guarded), nothing here changes a prompt, a tool return
value or any control flow, and with ``ROGUE_ONE_LLM_TRACE`` unset no trace file
is written. The module is stdlib only so that it can be imported (and unit
tested) without the agent stack.

What it records
---------------
* Tokens per role (prompt, completion, reasoning, cached), per tester stage and
  per (fold, iteration), **including discarded re-rolls** (splice / empty
  responses inside ``_streaming_create_with_fallback``). Web-search calls are
  logged by their own client as ``WebSearchAgent``; they are *charged to the
  calling role* through :func:`set_caller` (a context variable).
* One trace record per logical LLM call (``ROGUE_ONE_LLM_TRACE=1``) in
  ``<output>/llm_traces.jsonl.gz`` (gzip, sync-flushed after every record so a
  killed run still leaves a readable file).
* Counters: rescues per model family, fallbacks, tool-error sites (A07),
  appends, report reads, context errors, floor activations, Runner calls.
* Focus log, per-iteration log, wall clock per role.
* :func:`compute_summary` (R13 names) and :func:`finalize_run` (summary JSON,
  W&B summary, ``run-output`` artifact).

Definitions fixed here (docs/ablations/01_shared_infrastructure.md, 4.6, R13)
-----------------------------------------------------------------------------
* ``tok_compl_<role>`` / ``tok_prompt_<role>``: final responses plus discarded
  attempts, with search charged to the caller. ``tok_search_<role>`` is the
  search part of ``tok_compl_<role>``. ``tok_compl_total`` sums all buckets
  (search with no known caller lands in ``unattributed`` and is still in the
  total).
* ``llm_calls_total`` = logical calls through the client wrapper = number of
  trace records when tracing is on. ``rerolls`` = discarded attempts.
* ``silent_rejections`` = NaN/Inf, non-numeric or duplicate append rejections
  (4.6). The A07 variant (h8 + v1) is derivable from ``err_site_h8`` and
  ``err_site_v1``.
* ``report_read_rate`` = Scientist ``generic_pandas_tool`` calls whose command
  mentions ``feature_assessment`` / all Scientist ``generic_pandas_tool`` calls.
* ``fallback_share_<role>`` = fallback outputs / all outputs of that role
  (Extractor: features; Scientist: focuses; Tester: assessment reports);
  ``fallback_share`` pools the three.
* ``energy_j`` is not produced here: it is apportioned post hoc
  (scripts/agent2027/analysis/energy_apportion.py).
"""
from __future__ import annotations

import contextlib
import contextvars
import gzip
import json
import logging
import math
import os
import re
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

TRACE_FLAG = "ROGUE_ONE_LLM_TRACE"
TRACE_FILE = "llm_traces.jsonl.gz"
FOCUS_FILE = "focus_log.jsonl"
ITERATION_FILE = "iteration_log.jsonl"
SUMMARY_FILE = "instrumentation_summary.json"
ARTIFACT_NAME = "run-output"

ROLES = ("scientist", "extractor", "tester")
UNATTRIBUTED = "unattributed"

#: client ``agent_name`` -> role (None = a helper whose cost is charged to its caller).
AGENT_ROLE: dict[str, Optional[str]] = {
    "ScientistAgent": "scientist",
    "ExtractorAgent": "extractor",
    "TesterAgent": "tester",
    "WebSearchAgent": None,
    "KnowledgeAgent": None,
}

#: Sites of the A07 error taxonomy: h* return None under ROGUE_ONE_TOOL_ERROR_FEEDBACK=none,
#: v* are visible in both modes. v2 (auto-unprune) and v4 (fallback strings) are not errors.
ERROR_SITES = ("h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "v1", "v3")
_DEFAULT_SITE_BY_AGENT = {
    "ScientistAgent": "h1",
    "ExtractorAgent": "h4",
    "TesterAgent": "h9",
}
_NON_TOOL_NAMES = {"runner", "runner_fallback"}
#: rejection kinds that make up ``silent_rejections`` (4.6)
SILENT_KINDS = ("nan", "non_numeric", "duplicate")

#: N_active (billions) for the FLOP proxy (A23 2.3). Matched on the model id.
_ACTIVE_PARAMS_FIXED = (("gpt-oss-120b", 5.1), ("gpt-oss-20b", 3.6))

_lock = threading.RLock()

#: E9f phase id -> counter suffix (mirrors ``rogueone.utils.topology.PHASE_KEY``).
_SET_PHASE_KEY = {"S": "s", "E": "e", "T-A": "ta", "T-B": "tb"}

# --------------------------------------------------------------------------- context
_ITER: contextvars.ContextVar = contextvars.ContextVar("ro_iteration", default=None)
_CALLER: contextvars.ContextVar = contextvars.ContextVar("ro_caller", default=None)
_STAGE: contextvars.ContextVar = contextvars.ContextVar("ro_stage", default=None)
_SESSION: contextvars.ContextVar = contextvars.ContextVar("ro_session", default=None)
_CALL: contextvars.ContextVar = contextvars.ContextVar("ro_call", default=None)
#: E9f (A16): SET phase id (S, E, T-A, T-B); always None under three_agent (Full).
_PHASE: contextvars.ContextVar = contextvars.ContextVar("ro_set_phase", default=None)


def _guard(fn):
    """Decorator: an instrumentation failure must never reach the experiment."""

    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:  # pragma: no cover - exercised via tests that force failures
            logger.debug("instrumentation.%s failed", fn.__name__, exc_info=True)
            return None

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def trace_enabled() -> bool:
    return os.getenv(TRACE_FLAG, "0").strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------- state
class _State:
    def __init__(self) -> None:
        self.output_dir: Optional[Path] = None
        self.run_name: Optional[str] = None
        self.t0 = time.monotonic()
        self.wall_epoch0 = time.time()
        self.buckets: dict[str, Counter] = defaultdict(Counter)
        self.tester_stage_compl: Counter = Counter()
        self.per_iter: dict[int, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
        self.families: dict[str, Counter] = defaultdict(Counter)
        self.counters: Counter = Counter()
        self.max_prompt_tokens = 0
        self.wall: Counter = Counter()  # role -> seconds
        self.iter_wall: dict[int, Counter] = defaultdict(Counter)
        self.session_seq: Counter = Counter()
        self.seq = 0
        self.last_error_sig: dict[tuple, tuple] = {}
        self.focus_source: Optional[str] = None
        self.fallback_features: set[str] = set()
        self.inner_cv_scores: Optional[list[float]] = None
        self.trace_writer: Optional[_GzipJsonlWriter] = None
        self.trace_path: Optional[Path] = None
        self.trace_disabled_logged = False


_STATE = _State()


class _GzipJsonlWriter:
    """Append-only gzip JSONL writer, sync-flushed after every record."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = gzip.open(path, "wb")
        self._lock = threading.Lock()

    def write(self, obj: dict) -> None:
        line = (json.dumps(obj, ensure_ascii=False, default=str) + "\n").encode("utf-8")
        with self._lock:
            self._fh.write(line)
            self._fh.flush()  # GzipFile.flush -> Z_SYNC_FLUSH + underlying flush

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except Exception:
                pass


def reset() -> None:
    """Drop all accumulated state (tests; a new network in the same process)."""
    global _STATE
    with _lock:
        if _STATE.trace_writer is not None:
            _STATE.trace_writer.close()
        _STATE = _State()
    for var in (_ITER, _CALLER, _STAGE, _SESSION, _CALL, _PHASE):
        var.set(None)


@_guard
def configure(output_dir: Any, run_name: Optional[str] = None) -> None:
    """Start a fresh measurement window for one run (call once per network)."""
    reset()
    with _lock:
        _STATE.output_dir = Path(output_dir)
        _STATE.run_name = run_name
        _STATE.output_dir.mkdir(parents=True, exist_ok=True)
        for name in (FOCUS_FILE, ITERATION_FILE):
            try:
                (_STATE.output_dir / name).write_text("", encoding="utf-8")
            except Exception:
                pass


def _out_dir() -> Path:
    return _STATE.output_dir if _STATE.output_dir is not None else Path("output")


# --------------------------------------------------------------------------- context setters
@_guard
def set_iteration(index: int, fold_run_id: int, iteration_in_fold: int) -> None:
    _ITER.set(
        {"index": index, "fold_run_id": fold_run_id, "iteration_in_fold": iteration_in_fold}
    )
    _STAGE.set(None)
    _SESSION.set(None)
    _PHASE.set(None)


@_guard
def set_caller(role: Optional[str]) -> None:
    """Role that web-search (helper) calls made from now on are charged to."""
    _CALLER.set(role)


def current_caller() -> Optional[str]:
    """Role that web-search (helper) calls are currently charged to (None = unknown)."""
    return _CALLER.get()


@contextlib.contextmanager
def caller_scope(role: Optional[str]):
    """Charge helper calls inside the block to ``role``, restoring the previous caller."""
    token = _CALLER.set(role)
    try:
        yield
    finally:
        _CALLER.reset(token)


@_guard
def set_stage(stage: Optional[str], session_name: Optional[str] = None) -> None:
    _STAGE.set(stage)
    if session_name is not None:
        _SESSION.set(session_name)


@_guard
def set_session(session_name: Optional[str]) -> None:
    _SESSION.set(session_name)


@_guard
def set_phase(phase: Optional[str]) -> None:
    """E9f: SET phase of the calls made from now on (never set under three_agent)."""
    _PHASE.set(phase)


def current_phase() -> Optional[str]:
    return _PHASE.get()


def current_iteration() -> dict:
    return _ITER.get() or {"index": None, "fold_run_id": None, "iteration_in_fold": None}


# --------------------------------------------------------------------------- counters
@_guard
def count(name: str, n: int = 1) -> None:
    with _lock:
        _STATE.counters[name] += n


def get_counter(name: str) -> int:
    return int(_STATE.counters.get(name, 0))


@_guard
def note_focus_source(source: str) -> None:
    _STATE.focus_source = source


def pop_focus_source(default: str = "llm") -> str:
    with _lock:
        src, _STATE.focus_source = _STATE.focus_source, None
    return src or default


@_guard
def note_fallback_features(names) -> None:
    with _lock:
        _STATE.fallback_features.update(str(n) for n in names)


def fallback_feature_names() -> set[str]:
    return set(_STATE.fallback_features)


@_guard
def reset_fold_scope() -> None:
    """Fallback-feature names are fold scoped (the attribute pool is reset per fold)."""
    with _lock:
        _STATE.fallback_features.clear()


@_guard
def note_scientist_pandas(command: Any) -> None:
    """Scientist ``generic_pandas_tool`` call; counts reads of ``feature_assessment``."""
    with _lock:
        _STATE.counters["scientist_pandas_calls"] += 1
        if isinstance(command, str) and "feature_assessment" in command:
            _STATE.counters["report_reads"] += 1


@_guard
def note_inner_cv(scores: Optional[list]) -> None:
    with _lock:
        _STATE.inner_cv_scores = None if scores is None else [float(s) for s in scores]


def pop_inner_cv_scores() -> Optional[list[float]]:
    with _lock:
        s, _STATE.inner_cv_scores = _STATE.inner_cv_scores, None
    return s


_NORM_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")
_NORM_DIGITS = re.compile(r"\d+")


def _normalise_message(msg: str) -> str:
    return _NORM_DIGITS.sub("N", _NORM_QUOTED.sub("Q", msg or ""))[:300]


def _role_of_agent(agent_name: Optional[str]) -> str:
    role = AGENT_ROLE.get(agent_name or "")
    if role:
        return role
    low = (agent_name or "").lower()
    for r in ROLES:
        if low.startswith(r):
            return r
    return UNATTRIBUTED


@_guard
def note_tool_error(
    agent_name: str,
    tool_name: str,
    site: Optional[str] = None,
    exception_type: str = "",
    message: str = "",
    kind: Optional[str] = None,
) -> None:
    """Count one tool error (single counting point: ``ToolErrorReporter.report``)."""
    role = _role_of_agent(agent_name)
    with _lock:
        if tool_name in _NON_TOOL_NAMES:
            _STATE.counters["runner_errors"] += 1
            return
        site = site or _DEFAULT_SITE_BY_AGENT.get(agent_name or "", "h?")
        c = _STATE.counters
        c["n_tool_errors"] += 1
        c[f"err_site_{site}"] += 1
        c[f"err_role_{role}"] += 1
        if kind in SILENT_KINDS:
            c["silent_rejections"] += 1
        it = current_iteration()
        key = (role, it["fold_run_id"], it["iteration_in_fold"])
        sig = (tool_name, exception_type, _normalise_message(message))
        if _STATE.last_error_sig.get(key) == sig:
            c["err_repeats"] += 1
        _STATE.last_error_sig[key] = sig


# --------------------------------------------------------------------------- wall clock
def stage_begin(role: str) -> Optional[tuple]:
    try:
        return (role, time.monotonic(), current_iteration().get("index"))
    except Exception:
        return None


@_guard
def stage_end(token: Optional[tuple]) -> None:
    if not token:
        return
    role, t0, idx = token
    dt = time.monotonic() - t0
    with _lock:
        _STATE.wall[role] += dt
        if idx is not None:
            _STATE.iter_wall[idx][role] += dt


# --------------------------------------------------------------------------- model helpers
def model_family(model_id: Optional[str]) -> str:
    m = (model_id or "").lower()
    if "gpt-oss" in m or "gptoss" in m:
        return "gpt-oss"
    if "qwen" in m:
        return "qwen"
    slug = re.sub(r"[^a-z0-9.\-]+", "-", m.split("/")[-1]).strip("-")
    return slug or "unknown"


def active_params_b(model_id: Optional[str]) -> Optional[float]:
    """Active parameters in billions for ``model_id`` (None if unknown)."""
    m = (model_id or "").lower()
    for token, value in _ACTIVE_PARAMS_FIXED:
        if token in m:
            return value
    active = re.search(r"-a(\d+(?:\.\d+)?)b", m)
    if active:
        return float(active.group(1))
    size = re.search(r"(\d+(?:\.\d+)?)b(?![a-z])", m)
    if size and "qwen" in m:
        return float(size.group(1))
    return None


# --------------------------------------------------------------------------- usage helpers
def _get(obj: Any, *path: str) -> Any:
    for p in path:
        if obj is None:
            return None
        obj = obj.get(p) if isinstance(obj, dict) else getattr(obj, p, None)
    return obj


def usage_of(response: Any) -> dict[str, int]:
    usage = _get(response, "usage")
    out = {
        "prompt_tokens": int(_get(usage, "prompt_tokens") or 0),
        "completion_tokens": int(_get(usage, "completion_tokens") or 0),
        "total_tokens": int(_get(usage, "total_tokens") or 0),
        "reasoning_tokens": int(_get(usage, "completion_tokens_details", "reasoning_tokens") or 0),
        "cached_tokens": int(_get(usage, "prompt_tokens_details", "cached_tokens") or 0),
    }
    if not out["total_tokens"]:
        out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    return out


def _jsonable(obj: Any, _depth: int = 0) -> Any:
    if _depth > 40:
        return str(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return _jsonable(obj.model_dump(exclude_none=True), _depth + 1)
        except Exception:
            return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v, _depth + 1) for v in obj]
    return str(obj)


# --------------------------------------------------------------------------- LLM call lifecycle
class CallRecord:
    """Mutable record of one logical call through ``chat.completions.create``."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.t_start = time.time()
        self.t_mono = time.monotonic()
        self.attempts: list[dict] = []
        self.request: Optional[dict] = None
        self.rescue: Optional[str] = None
        self.raw_content_before_rescue: Optional[str] = None
        self.reasoning_chars = 0


def begin_call(agent_name: str) -> CallRecord:
    return CallRecord(agent_name)


async def traced_call(agent_name: str, fn, args: tuple, kwargs: dict):
    """Run ``fn(*args, **kwargs)`` as one metered/traced logical call."""
    rec = None
    token = None
    try:
        rec = begin_call(agent_name)
        token = _CALL.set(rec)
    except Exception:
        rec = None
    try:
        response = await fn(*args, **kwargs)
    except BaseException as exc:
        if rec is not None:
            end_call(rec, None, exc)
        raise
    finally:
        if token is not None:
            try:
                _CALL.reset(token)
            except Exception:
                pass
    if rec is not None:
        end_call(rec, response, None)
    return response


def _current_call() -> Optional[CallRecord]:
    return _CALL.get()


@_guard
def note_request(kwargs: dict) -> None:
    rec = _current_call()
    if rec is None:
        return
    req = dict(kwargs)
    if isinstance(req.get("messages"), list):
        req["messages"] = list(req["messages"])
    rec.request = req


@_guard
def note_attempt(response: Any, extra: Optional[dict] = None) -> None:
    """One completed attempt inside the streaming fallback loop (kept or discarded)."""
    rec = _current_call()
    if rec is None:
        return
    u = usage_of(response)
    choice = None
    try:
        choice = response.choices[0]
    except Exception:
        pass
    n_tc = 0
    try:
        n_tc = len(choice.message.tool_calls or [])
    except Exception:
        pass
    att = {
        **u,
        "finish_reason": getattr(choice, "finish_reason", None),
        "n_tool_calls": n_tc,
        "discard_reason": None,
    }
    if extra:
        att.update(extra)
    rec.attempts.append(att)


@_guard
def mark_discard(reason: str) -> None:
    """Mark the latest attempt as discarded (``splice`` or ``empty``)."""
    rec = _current_call()
    if rec is None or not rec.attempts:
        return
    rec.attempts[-1]["discard_reason"] = reason


@_guard
def note_stream_extras(reasoning_chars: int = 0) -> None:
    rec = _current_call()
    if rec is not None:
        rec.reasoning_chars += int(reasoning_chars or 0)


@_guard
def note_rescue(kind: str, raw_content: Optional[str] = None, model_id: Optional[str] = None) -> None:
    """Text-format tool-call rescue (``kind='text_tool_call'``)."""
    rec = _current_call()
    if rec is not None:
        rec.rescue = kind
        rec.raw_content_before_rescue = raw_content
    fam = model_family(model_id or ((rec.request or {}).get("model") if rec else None))
    with _lock:
        _STATE.families[fam]["rescue_text"] += 1


def _is_helper(agent_name: str) -> bool:
    """Search/knowledge clients: their cost is charged to the calling role."""
    return agent_name in AGENT_ROLE and AGENT_ROLE[agent_name] is None


def _charged_role(agent_name: str) -> tuple[str, Optional[str]]:
    """(charged role, caller) for a client name; helpers are charged to the caller."""
    if _is_helper(agent_name):
        caller = _CALLER.get()
        return (caller if caller in ROLES else UNATTRIBUTED), caller
    return _role_of_agent(agent_name), None


def _tester_stage_key(stage: Optional[str]) -> str:
    s = (stage or "").lower()
    if s.startswith("assess"):
        return "assess"
    if s.startswith("prun"):
        return "prune"
    return "other"


@_guard
def end_call(rec: CallRecord, response: Any, exc: Optional[BaseException]) -> None:
    it = current_iteration()
    role, caller = _charged_role(rec.agent_name)
    is_search = _is_helper(rec.agent_name)
    ctx_stage = _STAGE.get()
    req = rec.request or {}
    model_id = req.get("model")
    fam = model_family(model_id)

    final = usage_of(response) if response is not None else usage_of(None)
    attempts = rec.attempts
    # The last attempt is the one returned; if the call raised, every attempt was lost.
    discarded = list(attempts) if response is None else attempts[:-1]
    disc_prompt = sum(a["prompt_tokens"] for a in discarded)
    disc_compl = sum(a["completion_tokens"] for a in discarded)
    n_discard = len(discarded)
    splice_discards = sum(1 for a in discarded if a.get("discard_reason") == "splice")
    empty_discards = sum(1 for a in discarded if a.get("discard_reason") == "empty")

    finish = None
    message = None
    resp_model = None
    if response is not None:
        try:
            choice = response.choices[0]
            finish = choice.finish_reason
            message = choice.message
        except Exception:
            pass
        resp_model = _get(response, "model")

    status = getattr(exc, "status_code", None) if exc is not None else None
    with _lock:
        b = _STATE.buckets[role]
        b["prompt"] += final["prompt_tokens"] + disc_prompt
        b["compl"] += final["completion_tokens"] + disc_compl
        b["reasoning"] += final["reasoning_tokens"] + sum(a["reasoning_tokens"] for a in discarded)
        b["cached"] += final["cached_tokens"]
        b["calls"] += 1
        b["rerolls"] += n_discard
        b["disc_prompt"] += disc_prompt
        b["disc_compl"] += disc_compl
        if is_search:
            b["search_prompt"] += final["prompt_tokens"] + disc_prompt
            b["search_compl"] += final["completion_tokens"] + disc_compl
            b["search_calls"] += 1
        if role == "tester":
            _STATE.tester_stage_compl[_tester_stage_key(ctx_stage)] += (
                final["completion_tokens"] + disc_compl
            )
        if it["index"] is not None:
            pi = _STATE.per_iter[it["index"]][role]
            pi["prompt"] += final["prompt_tokens"] + disc_prompt
            pi["compl"] += final["completion_tokens"] + disc_compl
            pi["calls"] += 1
        f = _STATE.families[fam]
        f["calls"] += 1
        f["rerolls_splice"] += splice_discards
        f["rerolls_empty"] += empty_discards
        _STATE.max_prompt_tokens = max(_STATE.max_prompt_tokens, final["prompt_tokens"])
        if final["reasoning_tokens"] > final["completion_tokens"] > 0:
            _STATE.counters["reasoning_exceeds_completion_calls"] += 1
        if finish == "length":
            _STATE.counters["context_errors_length"] += 1
        if exc is not None:
            _STATE.counters["llm_call_errors"] += 1
            if status == 400:
                _STATE.counters["context_errors_http400"] += 1
        set_phase_id = _PHASE.get()
        if set_phase_id is not None:  # E9f only; Full never sets a phase
            _pk = _SET_PHASE_KEY.get(set_phase_id, "x")
            _STATE.counters[f"set_llm_calls_{_pk}"] += 1
            if finish == "length" or (exc is not None and status == 400):
                _STATE.counters[f"set_context_errors_{_pk}"] += 1
        _STATE.seq += 1
        seq = _STATE.seq

    if not trace_enabled():
        return
    session = "web_search_agent" if is_search else (_SESSION.get() or _default_session(role, it))
    with _lock:
        _STATE.session_seq[session] += 1
        call_seq = _STATE.session_seq[session]
    stage = "search" if is_search else (
        ctx_stage if role == "tester" and ctx_stage else _default_stage(role)
    )
    record: dict[str, Any] = {
        "seq": seq,
        "call_seq": call_seq,
        "session_name": session,
        "agent_name": rec.agent_name,
        "role": role,
        "caller": caller if is_search else None,
        "stage": stage,
        "caller_stage": ctx_stage if is_search else None,
        "index": it["index"],
        "fold_run_id": it["fold_run_id"],
        "test_fold_id": it["fold_run_id"],
        "iteration_in_fold": it["iteration_in_fold"],
        "ts_start": rec.t_start,
        "ts_end": time.time(),
        "duration_s": round(time.monotonic() - rec.t_mono, 4),
    }
    for k in _REQUEST_FIELDS:  # always present, null when not sent (e.g. seed unset)
        record[k] = None
    for k, v in req.items():
        record[k] = _jsonable(v)
    record["response_model"] = resp_model
    record["message"] = _jsonable(message) if message is not None else None
    record["finish_reason"] = finish
    record["usage"] = dict(final)
    record["attempts"] = attempts
    record["rerolls"] = n_discard
    record["discarded_usage"] = {"prompt_tokens": disc_prompt, "completion_tokens": disc_compl}
    record["rescue"] = rec.rescue
    if set_phase_id is not None:  # E9f only: the shared session name no longer names the role
        record["phase"] = set_phase_id
    if rec.raw_content_before_rescue is not None:
        record["raw_content_before_rescue"] = rec.raw_content_before_rescue
    if rec.reasoning_chars:
        record["reasoning_chars"] = rec.reasoning_chars
    if exc is not None:
        record["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000], "status_code": status}
    _write_trace(record)


_REQUEST_FIELDS = (
    "messages", "tools", "tool_choice", "extra_body", "temperature", "max_tokens", "model", "seed",
)


def _default_session(role: str, it: dict) -> str:
    return f"{role}_agent_session_step_{it['index'] if it['index'] is not None else 0}"


def _default_stage(role: str) -> str:
    return {"scientist": "focus", "extractor": "append", "tester": "tester"}.get(role, role)


def _write_trace(record: dict) -> None:
    try:
        with _lock:
            if _STATE.trace_writer is None:
                path = _out_dir() / TRACE_FILE
                _STATE.trace_writer = _GzipJsonlWriter(path)
                _STATE.trace_path = path
            writer = _STATE.trace_writer
        writer.write(record)
        count("trace_records")
    except Exception:
        count("trace_write_errors")
        logger.debug("trace write failed", exc_info=True)


# --------------------------------------------------------------------------- jsonl logs
def _append_jsonl(name: str, obj: dict) -> None:
    path = _out_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


@_guard
def log_focus(index: int, fold_run_id: int, iteration_in_fold: int, focus: Any, source: str) -> None:
    """One row per (fold, iteration) in ``focus_log.jsonl``."""
    count("focus_total")
    if source == "heuristic":
        count("fallback_scientist_focus")
    _append_jsonl(
        FOCUS_FILE,
        {
            "index": index,
            "fold_run_id": fold_run_id,
            "test_fold_id": fold_run_id,
            "iteration_in_fold": iteration_in_fold,
            "focus": focus if isinstance(focus, str) else str(focus),
            "source": source,
            "ts": time.time(),
        },
    )


_ITER_TOKEN_KEYS = ("prompt", "compl", "calls")


@_guard
def log_iteration(index: int, fold_run_id: int, iteration_in_fold: int, status: Any, extra: Optional[dict] = None) -> None:
    """One row per forward pass in ``iteration_log.jsonl`` (tokens, wall, active features)."""
    count("iterations_total")
    with _lock:
        per_role = {r: dict(_STATE.per_iter.get(index, {}).get(r, {})) for r in (*ROLES, UNATTRIBUTED)}
        wall = {r: round(float(v), 3) for r, v in _STATE.iter_wall.get(index, {}).items()}
    row: dict[str, Any] = {
        "index": index,
        "fold_run_id": fold_run_id,
        "test_fold_id": fold_run_id,
        "iteration_in_fold": iteration_in_fold,
        "status": status,
        "tokens": {
            r: {k: int(v.get(k, 0)) for k in _ITER_TOKEN_KEYS} for r, v in per_role.items()
        },
        "wall_s": wall,
        "ts": time.time(),
    }
    if extra:
        row.update(extra)
    _append_jsonl(ITERATION_FILE, row)


# --------------------------------------------------------------------------- summary
def _ratio(num: float, den: float) -> Optional[float]:
    return (num / den) if den else None


def compute_summary(models: Optional[dict] = None, active_params: Optional[dict] = None) -> dict[str, Any]:
    """Flat dict of R13 cost / process metrics (JSON- and W&B-summary-safe scalars)."""
    with _lock:
        s: dict[str, Any] = {}
        c = _STATE.counters
        buckets = {r: _STATE.buckets.get(r, Counter()) for r in (*ROLES, UNATTRIBUTED)}
        for r, b in buckets.items():
            s[f"tok_compl_{r}"] = int(b["compl"])
            s[f"tok_prompt_{r}"] = int(b["prompt"])
            s[f"llm_calls_{r}"] = int(b["calls"])
            if r == UNATTRIBUTED:
                continue
            s[f"tok_search_{r}"] = int(b["search_compl"])
            s[f"tok_search_prompt_{r}"] = int(b["search_prompt"])
            s[f"llm_calls_search_{r}"] = int(b["search_calls"])
            s[f"tok_reasoning_{r}"] = int(b["reasoning"])
            s[f"tok_cached_{r}"] = int(b["cached"])
            s[f"rerolls_{r}"] = int(b["rerolls"])
            s[f"tok_compl_discarded_{r}"] = int(b["disc_compl"])
            s[f"tok_prompt_discarded_{r}"] = int(b["disc_prompt"])
            s[f"wall_s_{r}"] = round(float(_STATE.wall.get(r, 0.0)), 3)
            s[f"runner_runs_{r}"] = int(c.get(f"runner_runs_{r}", 0))
        s["tok_compl_total"] = sum(int(b["compl"]) for b in buckets.values())
        s["tok_prompt_total"] = sum(int(b["prompt"]) for b in buckets.values())
        s["llm_calls_total"] = sum(int(b["calls"]) for b in buckets.values())
        s["rerolls"] = sum(int(b["rerolls"]) for b in buckets.values())
        s["tok_compl_tester_assess"] = int(_STATE.tester_stage_compl.get("assess", 0))
        s["tok_compl_tester_prune"] = int(_STATE.tester_stage_compl.get("prune", 0))
        s["tok_compl_tester_other"] = int(_STATE.tester_stage_compl.get("other", 0))
        s["max_prompt_tokens"] = int(_STATE.max_prompt_tokens)
        s["context_errors_http400"] = int(c.get("context_errors_http400", 0))
        s["context_errors_length"] = int(c.get("context_errors_length", 0))
        s["context_errors"] = s["context_errors_http400"] + s["context_errors_length"]
        s["llm_call_errors"] = int(c.get("llm_call_errors", 0))
        s["reasoning_exceeds_completion_calls"] = int(c.get("reasoning_exceeds_completion_calls", 0))
        s["wall_s"] = round(time.monotonic() - _STATE.t0, 3)

        # rescues per family (always report the two programme families)
        fams = set(_STATE.families) | {"gpt-oss", "qwen"}
        for fam in sorted(fams):
            f = _STATE.families.get(fam, Counter())
            rescues = int(f["rescue_text"]) + int(f["rerolls_splice"])
            s[f"rescues_per100_{fam}"] = (100.0 * rescues / f["calls"]) if f["calls"] else 0.0
            s[f"rescue_text_{fam}"] = int(f["rescue_text"])
            s[f"rescue_splice_{fam}"] = int(f["rerolls_splice"])
            s[f"rerolls_empty_{fam}"] = int(f["rerolls_empty"])
            s[f"llm_calls_family_{fam}"] = int(f["calls"])

        # fallbacks
        fb_ext_feat = int(c.get("fallback_extractor_features", 0))
        fb_sci = int(c.get("fallback_scientist_focus", 0))
        fb_tst = int(c.get("fallback_tester_report", 0))
        for name in (
            "fallback_extractor",
            "fallback_extractor_features",
            "fallback_extractor_restorations",
            "fallback_scientist_focus",
            "fallback_scientist_agent",
            "fallback_tester_report",
            "floor_activations",
            "floor_restored_features",
            "focus_total",
            "iterations_total",
            "tester_assessments",
        ):
            s[name] = int(c.get(name, 0))
        accepted = int(c.get("append_accepted", 0))
        ext_out = accepted + fb_ext_feat
        sci_out = int(c.get("focus_total", 0))
        tst_out = int(c.get("tester_assessments", 0))
        s["fallback_share_extractor"] = _ratio(fb_ext_feat, ext_out)
        s["fallback_share_scientist"] = _ratio(fb_sci, sci_out)
        s["fallback_share_tester"] = _ratio(fb_tst, tst_out)
        s["fallback_share"] = _ratio(fb_ext_feat + fb_sci + fb_tst, ext_out + sci_out + tst_out)

        # tool errors (A07 sites) and appends
        for site in ERROR_SITES:
            s[f"err_site_{site}"] = int(c.get(f"err_site_{site}", 0))
        s["err_site_other"] = int(
            sum(v for k, v in c.items() if k.startswith("err_site_") and k[9:] not in ERROR_SITES)
        )
        for r in (*ROLES, UNATTRIBUTED):
            s[f"err_role_{r}"] = int(c.get(f"err_role_{r}", 0))
        n_err = int(c.get("n_tool_errors", 0))
        attempts = int(c.get("append_attempts", 0))
        s["n_tool_errors"] = n_err
        s["n_accepted"] = accepted
        s["err_per_accepted"] = _ratio(n_err, accepted)
        s["err_repeat_rate"] = _ratio(int(c.get("err_repeats", 0)), n_err)
        s["runner_errors"] = int(c.get("runner_errors", 0))
        s["silent_rejections"] = int(c.get("silent_rejections", 0))
        s["append_attempts"] = attempts
        s["append_accepted"] = accepted
        s["append_restored"] = int(c.get("append_restored", 0))
        s["restore_pruned_count"] = s["append_restored"]
        s["append_success"] = _ratio(accepted, attempts)

        # report reads
        s["scientist_pandas_calls"] = int(c.get("scientist_pandas_calls", 0))
        s["report_reads"] = int(c.get("report_reads", 0))
        s["report_read_rate"] = _ratio(s["report_reads"], s["scientist_pandas_calls"])

        s["trace_records"] = int(c.get("trace_records", 0))
        s["trace_write_errors"] = int(c.get("trace_write_errors", 0))
        # E4: upstream web-search calls (any provider; R13 keeps the Serper name), the
        # share that came back empty or failed, and the query cache.
        s["serper_calls"] = int(c.get("serper_calls", 0))
        s["serper_ok_false"] = int(c.get("serper_ok_false", 0))
        s["serper_ok_false_rate"] = _ratio(s["serper_ok_false"], s["serper_calls"])
        s["search_cache_hits"] = int(c.get("search_cache_hits", 0))
        s["search_cache_misses"] = int(c.get("search_cache_misses", 0))
        # E9d: FX keys, only when an FX switch is active (Full-v2 summary unchanged).
        s.update(_fx_summary(c))
        # E9f: A16 keys, only under ROGUE_ONE_TOPOLOGY=set_phased.
        s.update(_topology_summary(c))

    # FLOP proxy (A23 2.3): 2 * N_active * (prompt + completion), search at the caller's model.
    flop = flop_c = 0.0
    known = True
    for r in ROLES:
        model = (models or {}).get(r)
        n_b = (active_params or {}).get(model) if active_params else None
        if n_b is None:
            n_b = active_params_b(model)
        if n_b is None:
            if s[f"tok_prompt_{r}"] or s[f"tok_compl_{r}"]:
                known = False
            continue
        flop += 2.0 * n_b * 1e9 * (s[f"tok_prompt_{r}"] + s[f"tok_compl_{r}"])
        flop_c += 2.0 * n_b * 1e9 * s[f"tok_compl_{r}"]
    s["flop_proxy"] = flop if known else None
    s["flop_proxy_compl"] = flop_c if known else None
    return s


def _fx_summary(counters: Any) -> dict[str, Any]:
    try:
        from rogueone.utils import fx_switches

        return fx_switches.summary_keys(counters)
    except Exception:
        return {}


def _topology_summary(counters: Any) -> dict[str, Any]:
    try:
        from rogueone.utils import topology

        return topology.summary_keys(counters)
    except Exception:
        return {}


def role_models() -> dict[str, Optional[str]]:
    """Configured model id per role (lazy import: the config needs LLM_MODEL etc.)."""
    try:
        from rogueone.utils import llm_cfg

        return {r: llm_cfg.model_for(r) for r in ROLES}
    except Exception:
        return {}


def close_trace() -> None:
    with _lock:
        if _STATE.trace_writer is not None:
            _STATE.trace_writer.close()
            _STATE.trace_writer = None


def _scalars(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            continue
        if v is None or isinstance(v, (int, float, str, bool)):
            out[k] = v
    return out


def finalize_run(output_dir: Any = None, upload: bool = True) -> Optional[dict]:
    """Write the summary JSON; log the W&B summary and the ``run-output`` artifact.

    Call before ``wandb.finish()``. Never raises.
    """
    try:
        out = Path(output_dir) if output_dir is not None else _out_dir()
        summary = compute_summary(models=role_models())
        close_trace()
        try:
            out.mkdir(parents=True, exist_ok=True)
            (out / SUMMARY_FILE).write_text(
                json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
            )
        except Exception:
            logger.debug("summary write failed", exc_info=True)
        if upload:
            _log_to_wandb(summary, out)
        return summary
    except Exception:
        logger.debug("finalize_run failed", exc_info=True)
        return None


def _log_to_wandb(summary: dict, out: Path) -> None:
    try:
        import wandb
    except Exception:
        return
    run = getattr(wandb, "run", None)
    if run is None:
        return
    try:
        for k, v in _scalars(summary).items():
            run.summary[k] = v
    except Exception:
        logger.debug("wandb summary update failed", exc_info=True)
    try:
        artifact = wandb.Artifact(name=ARTIFACT_NAME, type=ARTIFACT_NAME)
        artifact.add_dir(str(out))
        wandb.log_artifact(artifact)
    except Exception:
        logger.debug("run-output artifact failed", exc_info=True)
