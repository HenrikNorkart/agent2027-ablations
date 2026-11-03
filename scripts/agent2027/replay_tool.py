"""X2 replay tool: re-send recorded LLM prompts single-turn and score tool-call validity.

01 §3 X2, A04 §2/§4 check (b), A04-T7 step 1, A18-T3.

What it does
------------
* Reads prompt records from a local JSONL(.gz) file or a W&B artifact. Two record
  shapes are accepted:

  - frozen prompt-set records (``stage0_prompts.py``, artifact ``stage0-prompts``):
    ``{"prompt_id", "request": {messages, tools, ...}, "max_prompt_tokens", "kv_probe", ...}``;
  - raw E3 trace records (``llm_traces.jsonl.gz``, ``rogueone/utils/instrumentation.py``),
    with the request fields at the top level. ``--role`` filters them.

* Sends each recorded prompt **single-turn** (the recorded ``messages`` as they are,
  no tool execution, no follow-up turn) with the recorded tool schemas, temperature
  0.7 (``--temperature``) and ``seed = --seed`` (a recorded ``seed`` never survives),
  through Rogue One's own client wrapper ``VLLMCompatibleAsyncOpenAI``
  (``rogueone/llm/clients.py``): streamed tool calls, empty-response retries, splice
  re-rolls, name/argument normalisation and the JSON-only text rescue all apply
  exactly as in a run.

* Captures **attempt 1** without editing ``clients.py`` by wrapping the module-level
  ``clients._stream_to_chat_completion`` in this process (A18-T3). The wrapper also
  tees the raw SSE stream so that the *server's* ``finish_reason`` is kept (the
  aggregator rewrites it to ``tool_calls`` whenever any tool-call delta arrived).

* Scores every call (A04 §2 definitions, pre-registered in X10):

  - **valid, raw**: attempt 1, before any retry or rescue, has
    ``finish_reason == "tool_calls"`` (server value), at least one structured call,
    every call names one of the request's tools, and every call's arguments parse
    as a JSON object carrying the schema's required keys;
  - **valid, rescued**: the same test on the wrapper's final response (what the
    agent sees);
  - a request that errors is invalid under both.

  "Required keys" are checked at every object level of the schema (``$ref`` into
  ``$defs`` is resolved), so ``{"args": {"name": ..., "command": ...}}`` without
  the required ``argumentation`` is invalid. The top-level-only reading is kept in
  the ``*_valid_top`` columns and summary keys for comparison.

* Writes ``<out>/replay_calls.jsonl`` (full raw and final responses),
  ``<out>/replay_calls.csv`` (one row per call) and ``<out>/replay_summary.json``,
  prints the summary, and with ``--wandb`` logs the run (default name
  ``A04__<arm_id>__replay50__f42__r0``, group ``A04``, job_type ``<arm_id>``),
  the per-call table, the summary keys (``stage0_valid_call_raw``,
  ``stage0_valid_call_rescued``, ``stage0_finish_length_share``, ...) and the
  replies as artifact ``--replies-artifact`` (default ``stage0-replies-<arm_id>``).

* ``--kv-probe N`` (A04 §4 (a), SR3): sends the N longest prompts (by recorded
  ``max_prompt_tokens``) **concurrently** and reports per-call errors; the default
  run name becomes ``A04__<arm_id>__kvprobe<N>__f42__r0``.

Error classes: ``context`` (HTTP 400 that names the context length / prompt length),
``timeout`` (client timeout, ``--timeout`` s, default 600), ``infra`` (HTTP 5xx,
connection errors), ``other``. Infra errors are re-queued once at the end with the
same seed (``--no-requeue-infra`` disables this; A18-T7 step 2).

``--clip-max-tokens L``: before sending, the prompt is tokenised by the server
(vLLM ``/tokenize``; falls back to the recorded ``max_prompt_tokens``), and if
``prompt + max_tokens > L`` then ``max_tokens`` is clipped to ``L - prompt`` and the
turn is flagged ``clipped``.

Parser exceptions (A04 §2: a server-log ``Traceback``/``ERROR`` from a tool or
reasoning parser) are only visible in the server log. Pass ``--server-log FILE`` to
count them here (``stage0_parser_exceptions``); otherwise A04-T9 counts them from
``~/ro27_a04_logs/``. The client-side count of unparseable tool-call arguments is
``args_parse_errors_raw`` / ``args_parse_errors_rescued``.

Invocation on CAIR (A04-T7 step 1; containers only, CPU only)
---------------------------------------------------------------
The image's ``rogueone/`` supplies the wrapper under test; the script directory of
the checked-out branch is mounted over ``/workspace/scripts/agent2027``. With the
server at port ``P`` serving model ``M`` (its HF id) and its key in ``$API_KEY``::

    docker run --rm --name ro27-<sid>-replay50 --network host --cpus 4 \\
      --env-file /raid/home/henrbr16/code/PlanBench/.env \\
      -e LLM_MODEL=M -e LLM_PORT=P -e LLM_API_KEY=$API_KEY -e ROGUE_ONE_LLM_SEED=42 \\
      -v $HOME/code/RougeOne/scripts/agent2027:/workspace/scripts/agent2027:ro \\
      -v $HOME/ro27_a04_logs:/out \\
      --entrypoint python3 rogue_one:agent2027-pre.<N> \\
      /workspace/scripts/agent2027/replay_tool.py \\
        --prompts stage0-prompts:v0 --model M --port P --seed 42 --concurrency 1 \\
        --arm-id stage0_<rung> --session-id <SID> --image-tag agent2027-pre.<N> \\
        --out /out/replay_<SID>_<rung> --wandb

``--model``/``--port`` default to ``$LLM_MODEL``/``$LLM_PORT`` and ``--seed`` to
``$ROGUE_ONE_LLM_SEED`` (else 42), so they may be omitted. ``--project`` defaults to
``RogueOne-AGENT2027-prefreeze`` (session C: ``--project RogueOne-AGENT2027``); the
prompt artifact is always read from ``--prompts-project`` (default the prefreeze
project). The ``.env`` sets ``WANDB_PROJECT`` to an unrelated project, so the
project is always passed explicitly here. Q122 KV probe (same container, second
call)::

    ... replay_tool.py --prompts stage0-prompts:v0 --kv-probe 6 \\
        --arm-id stage0_Q122 --session-id <SID> --out /out/kvprobe_<SID>_q122 --wandb

A local file works too: ``--prompts /out/stage0-prompts-v0.jsonl`` (mount it).
"""
from __future__ import annotations

import argparse
import asyncio
import contextvars
import copy
import csv
import gzip
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ENTITY = "PlanBench"
DEFAULT_PROJECT = "RogueOne-AGENT2027-prefreeze"
DEFAULT_PROMPTS = "stage0-prompts:v0"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TIMEOUT_S = 600.0

#: request fields forwarded from a record (everything else -- extra_headers, store,
#: metadata, Omit placeholders -- is dropped). model/seed/temperature are overridden.
FORWARDED_FIELDS = (
    "messages", "tools", "tool_choice", "parallel_tool_calls", "max_tokens", "top_p",
    "extra_body", "response_format", "stop", "frequency_penalty", "presence_penalty",
)

_CONTEXT_ERROR_RE = re.compile(
    r"context length|maximum context|context window|too many tokens|prompt is too long|"
    r"max_model_len|longer than the maximum|exceeds the max|input tokens",
    re.IGNORECASE,
)
_PARSER_LOG_RE = re.compile(r"(tool_parser|tool_parsers|reasoning_parser|reasoning|harmony|qwen3_coder)", re.I)


# =========================================================================== validity
def _is_placeholder(value: Any) -> bool:
    """Trace placeholders for unset SDK params (``<openai.Omit object at ...>``)."""
    return isinstance(value, str) and (
        value.startswith("<openai.") or value in ("NOT_GIVEN", "NotGiven", "Omit")
    )


def _resolve_ref(schema: Any, root: dict) -> Any:
    seen = 0
    while isinstance(schema, dict) and "$ref" in schema and seen < 20:
        ref = schema["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/"):
            return schema
        node: Any = root
        for part in ref[2:].split("/"):
            node = node.get(part) if isinstance(node, dict) else None
        if node is None:
            return schema
        schema = node
        seen += 1
    return schema


def _is_object_schema(schema: Any) -> bool:
    return isinstance(schema, dict) and (schema.get("type") == "object" or "properties" in schema
                                         or "required" in schema)


def decode_json_string_object(value: Any, schema: Any, root: dict) -> Any:
    """HQ #315: ``value`` decoded when it is a JSON *string* holding an object where ``schema``
    (after ``$ref`` resolution) expects an object; otherwise ``value`` unchanged.

    vLLM 0.22's ``qwen3_coder`` parser does not follow ``$ref`` (pre-registration Amendment 17
    (i)), so object arguments arrive as stringified JSON.
    """
    if not isinstance(value, str) or not _is_object_schema(_resolve_ref(schema, root)):
        return value
    try:
        dec = json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return value
    return dec if isinstance(dec, dict) else value


def missing_required_keys(args: Any, schema: Optional[dict], nested: bool = True,
                          _root: Optional[dict] = None, _path: str = "", *,
                          decode_json_strings: bool = False) -> list[str]:
    """Required keys (as dotted paths) absent from ``args`` under JSON schema ``schema``.

    Nested objects are checked only when the value is a dict. With ``decode_json_strings``
    (HQ #315 scoring sensitivity; default off = the pre-registered behaviour) a property value
    that is a JSON string holding an object, where the schema expects an object, is decoded
    first and its required keys are checked too.
    """
    if not isinstance(schema, dict):
        return []
    root = _root if _root is not None else schema
    schema = _resolve_ref(schema, root)
    if not isinstance(schema, dict) or not isinstance(args, dict):
        return []
    missing = [f"{_path}{k}" for k in (schema.get("required") or []) if k not in args]
    if nested:
        for key, sub in (schema.get("properties") or {}).items():
            val = args.get(key)
            if key in args and decode_json_strings:
                val = decode_json_string_object(val, sub, root)
            if key in args and isinstance(val, dict):
                missing += missing_required_keys(val, sub, True, root, f"{_path}{key}.",
                                                 decode_json_strings=decode_json_strings)
    return missing


def tool_schemas(tools: list) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for t in tools or []:
        if isinstance(t, dict) and t.get("type") == "function" and isinstance(t.get("function"), dict):
            fn = t["function"]
            if isinstance(fn.get("name"), str):
                out[fn["name"]] = fn.get("parameters") or {}
    return out


def classify_tool_calls(finish_reason: Optional[str], tool_calls: Optional[list],
                        tools: list, nested: bool = True, *,
                        decode_json_strings: bool = False) -> dict[str, Any]:
    """Apply the A04 §2 valid-call test to one completion.

    ``tool_calls`` is a list of ``{"name": str, "arguments": str}``. Returns
    ``{"valid": bool, "reason": str, "args_parse_errors": int, "missing": [...]}``;
    ``reason`` is ``ok`` or the first failing condition. ``decode_json_strings``: HQ #315
    sensitivity (see :func:`missing_required_keys`); default off.
    """
    schemas = tool_schemas(tools)
    calls = list(tool_calls or [])
    res: dict[str, Any] = {"valid": False, "reason": "ok", "args_parse_errors": 0, "missing": []}
    parse_errors = 0
    first_bad: Optional[str] = None
    for tc in calls:
        name = tc.get("name")
        if name not in schemas:
            first_bad = first_bad or f"unknown_tool:{name}"
            continue
        raw = tc.get("arguments")
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            parse_errors += 1
            first_bad = first_bad or "args_not_json"
            continue
        if not isinstance(parsed, dict):
            first_bad = first_bad or "args_not_object"
            continue
        miss = missing_required_keys(parsed, schemas[name], nested=nested,
                                     decode_json_strings=decode_json_strings)
        if miss:
            res["missing"] += miss
            first_bad = first_bad or "missing_required:" + ",".join(miss)
    res["args_parse_errors"] = parse_errors
    if finish_reason != "tool_calls":
        res["reason"] = f"finish_reason={finish_reason}"
    elif not calls:
        res["reason"] = "no_tool_call"
    elif first_bad:
        res["reason"] = first_bad
    else:
        res["valid"] = True
    return res


#: A18 §2: an answer is a well-formed move when its text is longer than the role's own
#: acceptance threshold (S: ``len(final_output) > 10``, sci/agent.py; E, T, W: > 0).
ANSWER_MIN_CHARS = {"S": 10, "E": 0, "T": 0, "W": 0}


def classify_move(finish_reason: Optional[str], tool_calls: Optional[list], content: Optional[str],
                  tools: list, *, answer_min_chars: int = 0, nested: bool = True,
                  decode_json_strings: bool = False) -> dict[str, Any]:
    """A18 §2 valid-move test (``rp1_valid_call_*``), judged on the move actually made.

    * **Action** (at least one structured call): the A04 §2 test of
      :func:`classify_tool_calls` (``finish_reason == "tool_calls"``, every call names a
      recorded tool, arguments parse to a JSON object with the schema's required keys).
    * **Answer** (no structured call): ``finish_reason == "stop"`` and
      ``len(text) > answer_min_chars``.

    Returns the :func:`classify_tool_calls` dict plus ``move`` (``action``/``answer``).
    ``decode_json_strings``: HQ #315 sensitivity (see :func:`missing_required_keys`); default off.
    """
    calls = list(tool_calls or [])
    if calls:
        res = classify_tool_calls(finish_reason, calls, tools, nested=nested,
                                  decode_json_strings=decode_json_strings)
        res["move"] = "action"
        return res
    res: dict[str, Any] = {"valid": False, "reason": "ok", "args_parse_errors": 0, "missing": [],
                           "move": "answer"}
    text = content if isinstance(content, str) else ""
    if finish_reason != "stop":
        res["reason"] = f"finish_reason={finish_reason}"
    elif len(text) <= answer_min_chars:
        res["reason"] = "answer_too_short"
    else:
        res["valid"] = True
    return res


def calls_from_message(message: Any) -> list[dict]:
    """``[{"name", "arguments"}]`` from an OpenAI message object or dict."""
    if message is None:
        return []
    tcs = message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
    out = []
    for tc in tcs or []:
        fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
        if fn is None:
            continue
        name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
        args = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", None)
        out.append({"name": name, "arguments": args})
    return out


def _call_key(calls: list[dict]) -> list:
    """Comparable form of a call list: names plus *parsed* arguments (whitespace-blind)."""
    key = []
    for t in calls:
        args = t.get("arguments")
        try:
            args = json.loads(args) if isinstance(args, str) else args
        except (json.JSONDecodeError, TypeError):
            pass
        key.append((t.get("name"), json.dumps(args, sort_keys=True, default=str)))
    return key


def classify_error(exc: BaseException) -> tuple[str, Optional[int]]:
    """(kind, http status) for a failed request: context | timeout | infra | other."""
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    msg = str(exc)
    if "Timeout" in name or isinstance(exc, asyncio.TimeoutError):
        return "timeout", status
    if status == 400 and _CONTEXT_ERROR_RE.search(msg):
        return "context", status
    if (status is not None and status >= 500) or "Connection" in name or "RemoteProtocol" in name:
        return "infra", status
    if status is None and _CONTEXT_ERROR_RE.search(msg) and "400" in msg:
        return "context", 400
    return "other", status


# =========================================================================== records
def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8")


def read_records(path: Path) -> list[dict]:
    rows = []
    with _open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalise_record(rec: dict, i: int) -> dict:
    """Bring a prompt-set record or a raw E3 trace record to one shape."""
    if isinstance(rec.get("request"), dict):
        req = dict(rec["request"])
        out = dict(rec)
    else:  # raw E3 trace record
        req = {k: rec.get(k) for k in FORWARDED_FIELDS + ("model", "seed", "temperature")}
        usage = rec.get("usage") or {}
        mpt = max([int(usage.get("prompt_tokens") or 0)]
                  + [int(a.get("prompt_tokens") or 0) for a in rec.get("attempts") or []])
        out = {
            "prompt_id": rec.get("prompt_id") or f"trace-{rec.get('seq', i)}",
            "role": rec.get("role"),
            "agent_name": rec.get("agent_name"),
            "max_prompt_tokens": mpt,
            "kv_probe": False,
            "provenance": {k: rec.get(k) for k in (
                "seq", "call_seq", "session_name", "index", "fold_run_id", "iteration_in_fold", "stage")},
        }
    out["request"] = {k: v for k, v in req.items() if v is not None and not _is_placeholder(v)}
    out.setdefault("prompt_id", f"p{i:03d}")
    out.setdefault("max_prompt_tokens", 0)
    return out


def resolve_prompts(spec: str, entity: str, project: str,
                    download_root: Optional[Path] = None) -> tuple[Path, str, Optional[str]]:
    """Local path, or a W&B artifact ref (``name:alias`` / ``entity/project/name:alias``).

    Returns (file, human-readable source, artifact ref or None).
    """
    p = Path(spec)
    if p.exists():
        return (p if p.is_file() else _pick_file(p)), str(p.resolve()), None
    ref = spec[len("wandb:"):] if spec.startswith("wandb:") else spec
    if ref.count("/") == 0:
        ref = f"{entity}/{project}/{ref}"
    if ":" not in ref.rsplit("/", 1)[-1]:
        ref += ":latest"
    import wandb  # noqa: WPS433 (lazy: only needed for artifacts)

    import tempfile  # noqa: WPS433

    art = wandb.Api().artifact(ref)
    root = art.download(root=str(download_root or tempfile.mkdtemp(prefix="replay_prompts_")))
    return _pick_file(Path(root)), f"{ref} ({getattr(art, 'version', '?')})", ref


def _pick_file(d: Path) -> Path:
    for pat in ("*.jsonl", "*.jsonl.gz", "**/*.jsonl", "**/*.jsonl.gz"):
        hits = sorted(d.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no .jsonl(.gz) under {d}")


def select_kv_probe(records: list[dict], n: int) -> list[dict]:
    return sorted(records, key=lambda r: (-int(r.get("max_prompt_tokens") or 0), str(r["prompt_id"])))[:n]


# =========================================================================== attempt capture
_ATTEMPTS: contextvars.ContextVar = contextvars.ContextVar("replay_attempts", default=None)


class _TeeStream:
    """Async-iterates an upstream stream, remembering the server's finish_reason."""

    def __init__(self, stream: Any, sink: dict):
        self._stream = stream
        self._sink = sink

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        async for chunk in self._stream:
            for ch in getattr(chunk, "choices", None) or []:
                fr = getattr(ch, "finish_reason", None)
                if fr:
                    self._sink["server_finish_reason"] = fr
            yield chunk


def _snapshot(completion: Any) -> dict:
    choice = completion.choices[0]
    usage = getattr(completion, "usage", None)
    return {
        "finish_reason": choice.finish_reason,
        "content": choice.message.content,
        "tool_calls": calls_from_message(choice.message),
        "usage": {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "reasoning_tokens": int(getattr(getattr(usage, "completion_tokens_details", None),
                                            "reasoning_tokens", 0) or 0),
        },
    }


def install_attempt_capture(clients_mod: Any) -> None:
    """Wrap ``clients._stream_to_chat_completion`` (idempotent) to record every attempt."""
    original = getattr(clients_mod._stream_to_chat_completion, "_replay_original", None)
    if original is not None:
        return
    original = clients_mod._stream_to_chat_completion

    async def capturing(create_fn, args, kwargs, model_id):
        sink: dict[str, Any] = {"server_finish_reason": None}

        async def teeing_create(*a, **kw):
            return _TeeStream(await create_fn(*a, **kw), sink)

        attempts = _ATTEMPTS.get()
        try:
            completion = await original(teeing_create, args, kwargs, model_id)
        except BaseException as exc:
            if attempts is not None:
                attempts.append({"error": f"{type(exc).__name__}: {str(exc)[:500]}",
                                 "server_finish_reason": sink["server_finish_reason"]})
            raise
        if attempts is not None:
            snap = _snapshot(completion)  # before the wrapper mutates the object
            snap["server_finish_reason"] = sink["server_finish_reason"]
            snap["temperature"] = kwargs.get("temperature")
            attempts.append(snap)
        return completion

    capturing._replay_original = original  # type: ignore[attr-defined]
    clients_mod._stream_to_chat_completion = capturing


# =========================================================================== one call
def build_request(rec: dict, model: str, seed: int, temperature: float) -> dict:
    req = copy.deepcopy(rec["request"])
    kwargs = {k: req[k] for k in FORWARDED_FIELDS if k in req}
    kwargs["model"] = model
    kwargs["seed"] = int(seed)  # a recorded seed must not survive (A18-T3)
    kwargs["temperature"] = float(temperature)
    return kwargs


async def count_prompt_tokens(http_base: str, api_key: str, kwargs: dict, timeout: float = 60.0) -> Optional[int]:
    """Prompt length via vLLM ``/tokenize`` (chat form, with tools); None if unavailable."""
    try:
        import httpx
    except Exception:  # pragma: no cover
        return None
    body = {"model": kwargs["model"], "messages": kwargs["messages"], "add_generation_prompt": True}
    if kwargs.get("tools"):
        body["tools"] = kwargs["tools"]
    tmpl = (kwargs.get("extra_body") or {}).get("chat_template_kwargs")
    if tmpl:
        body["chat_template_kwargs"] = tmpl
    try:
        async with httpx.AsyncClient(timeout=timeout) as cl:
            r = await cl.post(f"{http_base}/tokenize", json=body,
                              headers={"Authorization": f"Bearer {api_key}"})
            if r.status_code == 200:
                data = r.json()
                return int(data.get("count") or len(data.get("tokens") or []))
    except Exception:
        return None
    return None


async def server_healthy(http_base: str, timeout: float = 10.0) -> Optional[bool]:
    """GET ``/health`` == 200 (None if httpx is unavailable)."""
    try:
        import httpx
    except Exception:  # pragma: no cover
        return None
    try:
        async with httpx.AsyncClient(timeout=timeout) as cl:
            r = await cl.get(f"{http_base}/health")
            return r.status_code == 200
    except Exception:
        return False


#: record metadata copied into every replay row (A18 prompt sets carry them)
ROW_META_FIELDS = ("stage", "stratum", "recorded_move", "caller", "agent_name")


async def replay_one(client: Any, rec: dict, *, model: str, seed: int, temperature: float,
                     nested: bool = True, clip_max_tokens: Optional[int] = None,
                     http_base: Optional[str] = None, api_key: str = "EMPTY",
                     validity: str = "a04", clip_min_tokens: int = 1) -> dict:
    """Send one recorded prompt through the wrapper and score it.

    ``validity="a18"``: answers are scored too (:func:`classify_move`, threshold from
    :data:`ANSWER_MIN_CHARS` by the record's ``role``), and a client timeout is an
    ``infra`` error only if the server's ``/health`` is not 200 straight afterwards
    (A18 §4: a timeout on a healthy server is an invalid move).
    """
    kwargs = build_request(rec, model, seed, temperature)
    tools = kwargs.get("tools") or []
    row: dict[str, Any] = {
        "prompt_id": rec["prompt_id"],
        "role": rec.get("role"),
        **{k: rec.get(k) for k in ROW_META_FIELDS if k in rec},
        "kv_probe": bool(rec.get("kv_probe")),
        "recorded_max_prompt_tokens": int(rec.get("max_prompt_tokens") or 0),
        "model": model,
        "seed": int(seed),
        "temperature": float(temperature),
        "max_tokens_recorded": kwargs.get("max_tokens"),
        "clipped": False,
        "prompt_tokens_est": None,
    }
    if clip_max_tokens:
        est = None
        if http_base:
            est = await count_prompt_tokens(http_base, api_key, kwargs)
        row["prompt_tokens_est_source"] = "tokenize" if est is not None else "recorded"
        if est is None:
            est = row["recorded_max_prompt_tokens"] or None
        row["prompt_tokens_est"] = est
        mt = kwargs.get("max_tokens")
        if est is not None and mt is not None and est + int(mt) > clip_max_tokens:
            kwargs["max_tokens"] = max(int(clip_min_tokens), int(clip_max_tokens) - int(est))
            row["clipped"] = True
    row["max_tokens_sent"] = kwargs.get("max_tokens")

    attempts: list[dict] = []
    token = _ATTEMPTS.set(attempts)
    t0 = time.monotonic()
    final = None
    try:
        final = await client.chat.completions.create(**kwargs)
        row["error_kind"] = None
        row["error_status"] = None
        row["error_message"] = None
    except Exception as exc:  # every failure is a scored, invalid call
        kind, status = classify_error(exc)
        row["error_kind"] = kind
        row["error_status"] = status
        row["error_message"] = f"{type(exc).__name__}: {str(exc)[:1000]}"
    finally:
        _ATTEMPTS.reset(token)
    row["latency_s"] = round(time.monotonic() - t0, 3)
    if validity == "a18" and row["error_kind"] == "timeout" and http_base:
        healthy = await server_healthy(http_base)
        row["server_healthy_after_timeout"] = healthy
        if healthy is False:
            row["error_kind"] = "infra"
    row["context_error"] = row["error_kind"] == "context"
    min_chars = ANSWER_MIN_CHARS.get(str(rec.get("role")), 0)
    return score_call(row, attempts, final, tools, nested=nested, validity=validity,
                      answer_min_chars=min_chars)


def _classify(validity: str, fr: Optional[str], calls: list, content: Optional[str], tools: list,
              nested: bool, answer_min_chars: int) -> dict[str, Any]:
    if validity == "a18":
        return classify_move(fr, calls, content, tools, answer_min_chars=answer_min_chars, nested=nested)
    return classify_tool_calls(fr, calls, tools, nested=nested)


def score_call(row: dict, attempts: list[dict], final: Any, tools: list, nested: bool = True,
               validity: str = "a04", answer_min_chars: int = 0) -> dict:
    """Fill the raw/rescued validity columns of ``row`` (pure; unit-tested).

    ``validity="a04"``: only a well-formed tool call is valid (A04 §2).
    ``validity="a18"``: the A18 §2 move test (:func:`classify_move`); also fills
    ``raw_move`` / ``final_move`` (``action``/``answer``).
    """
    ok_attempts = [a for a in attempts if "error" not in a]
    row["attempts"] = len(attempts)
    row["rerolls"] = max(0, len(attempts) - 1)
    a1 = attempts[0] if attempts else None
    if a1 is None or "error" in a1:
        raw = {"valid": False, "reason": "error", "args_parse_errors": 0}
        raw_top = dict(raw)
        row["raw_finish_reason"] = None
        row["raw_finish_reason_aggregated"] = None
        row["raw_n_tool_calls"] = 0
        row["raw_tool_names"] = ""
        row["raw_move"] = None
    else:
        fr = a1.get("server_finish_reason") or a1.get("finish_reason")
        raw = _classify(validity, fr, a1["tool_calls"], a1.get("content"), tools, nested, answer_min_chars)
        raw_top = _classify(validity, fr, a1["tool_calls"], a1.get("content"), tools, False, answer_min_chars)
        row["raw_move"] = "action" if a1["tool_calls"] else "answer"
        row["raw_finish_reason"] = fr
        row["raw_finish_reason_aggregated"] = a1.get("finish_reason")
        row["raw_n_tool_calls"] = len(a1["tool_calls"])
        row["raw_tool_names"] = ";".join(str(t.get("name")) for t in a1["tool_calls"])
    row["raw_valid"] = bool(raw["valid"])
    row["raw_valid_top"] = bool(raw_top["valid"])
    row["raw_reason"] = raw["reason"]
    row["args_parse_errors_raw"] = raw["args_parse_errors"]

    if final is None:
        fin = {"valid": False, "reason": "error", "args_parse_errors": 0}
        fin_top = dict(fin)
        row.update(final_finish_reason=None, final_n_tool_calls=0, final_tool_names="",
                   prompt_tokens=0, completion_tokens=0, reasoning_tokens=0, response_model=None,
                   rescue_kind=None, final_move=None)
        row["_final_message"] = None
    else:
        choice = final.choices[0]
        fcalls = calls_from_message(choice.message)
        fin = _classify(validity, choice.finish_reason, fcalls, choice.message.content, tools, nested,
                        answer_min_chars)
        fin_top = _classify(validity, choice.finish_reason, fcalls, choice.message.content, tools, False,
                            answer_min_chars)
        row["final_move"] = "action" if fcalls else "answer"
        usage = getattr(final, "usage", None)
        row["final_finish_reason"] = choice.finish_reason
        row["final_n_tool_calls"] = len(fcalls)
        row["final_tool_names"] = ";".join(str(t.get("name")) for t in fcalls)
        row["prompt_tokens"] = int(getattr(usage, "prompt_tokens", 0) or 0)
        row["completion_tokens"] = int(getattr(usage, "completion_tokens", 0) or 0)
        row["reasoning_tokens"] = int(getattr(getattr(usage, "completion_tokens_details", None),
                                              "reasoning_tokens", 0) or 0)
        row["response_model"] = getattr(final, "model", None)
        last = ok_attempts[-1] if ok_attempts else None
        if last is not None and not last["tool_calls"] and fcalls:
            row["rescue_kind"] = "text_tool_call"
        elif last is not None and fcalls and _call_key(last["tool_calls"]) != _call_key(fcalls):
            row["rescue_kind"] = "normalised"
        else:
            row["rescue_kind"] = None
        row["_final_message"] = {"content": choice.message.content, "tool_calls": fcalls}
    row["rescued_valid"] = bool(fin["valid"])
    row["rescued_valid_top"] = bool(fin_top["valid"])
    row["rescued_reason"] = fin["reason"]
    row["args_parse_errors_rescued"] = fin["args_parse_errors"]
    row["discarded_completion_tokens"] = sum(
        int((a.get("usage") or {}).get("completion_tokens") or 0) for a in ok_attempts[:-1]
    ) if ok_attempts else 0
    row["_attempts"] = attempts
    return row


# =========================================================================== summary
def summarise_a18(rows: list[dict], prefix: str = "rp1") -> dict[str, Any]:
    """A18-T7 online summary keys (per role S/E/T/W, and per stage/stratum).

    ``<prefix>_valid_call_raw__<role>`` / ``..._rescued__<role>`` are **rates** over the
    role's prompts at this target that ended in neither a context error nor an infra
    error (this target's own scored set; the cross-target common set is applied
    offline by ``rp1_score.py``). Counts are under ``..._n__<role>``.
    """
    s: dict[str, Any] = {}

    def scored(rr: list[dict]) -> list[dict]:
        return [r for r in rr if r.get("error_kind") not in ("context", "infra")]

    def block(key: str, rr: list[dict]) -> None:
        sc = scored(rr)
        k = len(sc)
        raw = sum(bool(r.get("raw_valid")) for r in sc)
        resc = sum(bool(r.get("rescued_valid")) for r in sc)
        s[f"{prefix}_valid_call_raw__{key}"] = raw / k if k else None
        s[f"{prefix}_valid_call_rescued__{key}"] = resc / k if k else None
        s[f"{prefix}_valid_call_raw_n__{key}"] = raw
        s[f"{prefix}_valid_call_rescued_n__{key}"] = resc
        s[f"n_scored__{key}"] = k
        s[f"n_prompts__{key}"] = len(rr)
        acted = [r for r in sc if r.get("recorded_move") == "action" and r.get("final_move") is not None]
        s[f"answer_instead_of_action__{key}"] = (
            sum(r.get("final_move") == "answer" for r in acted) / len(acted) if acted else None)
        moved = [r for r in sc if r.get("final_move") == "action"]
        s[f"multi_call_share__{key}"] = (
            sum(int(r.get("final_n_tool_calls") or 0) > 1 for r in moved) / len(moved) if moved else None)
        for kind in ("context", "infra", "timeout"):
            s[f"{kind}_errors__{key}" if kind != "infra" else f"infra_error__{key}"] = sum(
                1 for r in rr if r.get("error_kind") == kind)
        s[f"finish_length__{key}"] = sum(1 for r in sc if r.get("final_finish_reason") == "length")
        s[f"clipped__{key}"] = sum(1 for r in rr if r.get("clipped"))

    for role in sorted({str(r.get("role")) for r in rows if r.get("role")}):
        block(role, [r for r in rows if str(r.get("role")) == role])
    for stratum in sorted({str(r.get("stratum")) for r in rows if r.get("stratum")}):
        rr = [r for r in rows if str(r.get("stratum")) == stratum]
        sc = scored(rr)
        s[f"{prefix}_valid_call_rescued__stratum_{stratum}"] = (
            sum(bool(r.get("rescued_valid")) for r in sc) / len(sc) if sc else None)
        s[f"{prefix}_valid_call_raw__stratum_{stratum}"] = (
            sum(bool(r.get("raw_valid")) for r in sc) / len(sc) if sc else None)
        s[f"n_prompts__stratum_{stratum}"] = len(rr)
    s["infra_error_final"] = sum(1 for r in rows if r.get("error_kind") == "infra")
    return s


def summarise(rows: list[dict], prefix: str = "stage0", wall_s: Optional[float] = None,
              validity: str = "a04") -> dict[str, Any]:
    n = len(rows)
    raw = sum(r["raw_valid"] for r in rows)
    resc = sum(r["rescued_valid"] for r in rows)
    length = sum(1 for r in rows if r.get("final_finish_reason") == "length"
                 or r.get("raw_finish_reason") == "length")
    final_length = sum(1 for r in rows if r.get("final_finish_reason") == "length")
    s: dict[str, Any] = {
        "n_prompts": n,
        f"{prefix}_valid_call_raw": raw,
        f"{prefix}_valid_call_rescued": resc,
        f"{prefix}_valid_call_raw_rate": raw / n if n else None,
        f"{prefix}_valid_call_rescued_rate": resc / n if n else None,
        f"{prefix}_valid_call_raw_top": sum(r["raw_valid_top"] for r in rows),
        f"{prefix}_valid_call_rescued_top": sum(r["rescued_valid_top"] for r in rows),
        f"{prefix}_finish_length_share": final_length / n if n else None,
        f"{prefix}_any_attempt_length_share": length / n if n else None,
        "args_parse_errors_raw": sum(r["args_parse_errors_raw"] for r in rows),
        "args_parse_errors_rescued": sum(r["args_parse_errors_rescued"] for r in rows),
        "rescues_text_tool_call": sum(1 for r in rows if r.get("rescue_kind") == "text_tool_call"),
        "rescues_normalised": sum(1 for r in rows if r.get("rescue_kind") == "normalised"),
        "raw_stop_with_calls": sum(1 for r in rows if r.get("raw_finish_reason") == "stop"
                                   and r.get("raw_n_tool_calls")),
        "rerolls": sum(r["rerolls"] for r in rows),
        "context_errors": sum(1 for r in rows if r.get("error_kind") == "context"),
        "infra_error": sum(1 for r in rows if r.get("error_kind") == "infra"),
        "timeout": sum(1 for r in rows if r.get("error_kind") == "timeout"),
        "other_errors": sum(1 for r in rows if r.get("error_kind") == "other"),
        "clipped": sum(1 for r in rows if r.get("clipped")),
        "max_prompt_tokens": max([int(r.get("prompt_tokens") or 0) for r in rows] or [0]),
        "tok_prompt_total": sum(int(r.get("prompt_tokens") or 0) for r in rows),
        "tok_compl_total": sum(int(r.get("completion_tokens") or 0) for r in rows)
        + sum(int(r.get("discarded_completion_tokens") or 0) for r in rows),
    }
    if wall_s is not None:
        s["replay_wall_s"] = round(wall_s, 2)
    roles = sorted({r.get("role") for r in rows if r.get("role")})
    if validity == "a18":
        s.update(summarise_a18(rows, prefix))
    elif len(roles) > 1:
        for role in roles:
            rr = [r for r in rows if r.get("role") == role]
            s[f"{prefix}_valid_call_raw__{role}"] = sum(r["raw_valid"] for r in rr)
            s[f"{prefix}_valid_call_rescued__{role}"] = sum(r["rescued_valid"] for r in rr)
            s[f"n_prompts__{role}"] = len(rr)
    return s


def count_parser_exceptions(log_path: Path) -> int:
    """Server-log ``Traceback``/``ERROR`` lines attributable to a tool/reasoning parser."""
    n = 0
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines):
        if "ERROR" in line and _PARSER_LOG_RE.search(line):
            n += 1
        elif line.startswith("Traceback"):
            window = "\n".join(lines[i:i + 40])
            if _PARSER_LOG_RE.search(window):
                n += 1
    return n


CSV_COLUMNS = (
    "prompt_id role kv_probe recorded_max_prompt_tokens model seed temperature "
    "max_tokens_recorded max_tokens_sent clipped prompt_tokens_est attempts rerolls "
    "raw_finish_reason raw_finish_reason_aggregated raw_n_tool_calls raw_tool_names raw_valid "
    "raw_valid_top raw_reason args_parse_errors_raw final_finish_reason final_n_tool_calls "
    "final_tool_names rescued_valid rescued_valid_top rescued_reason args_parse_errors_rescued "
    "rescue_kind prompt_tokens completion_tokens reasoning_tokens discarded_completion_tokens "
    "response_model error_kind error_status context_error error_message latency_s "
    "stage stratum recorded_move raw_move final_move server_healthy_after_timeout requeued"
).split()


def write_outputs(out: Path, rows: list[dict], summary: dict) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = {"jsonl": out / "replay_calls.jsonl", "csv": out / "replay_calls.csv",
             "summary": out / "replay_summary.json"}
    with open(paths["jsonl"], "w", encoding="utf-8") as fh:
        for r in rows:
            rec = {k: v for k, v in r.items() if not k.startswith("_")}
            rec["attempt_details"] = r.get("_attempts")
            rec["final_message"] = r.get("_final_message")
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    with open(paths["csv"], "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CSV_COLUMNS})
    paths["summary"].write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return paths


# =========================================================================== driver
async def run_replay(records: list[dict], client: Any, *, model: str, seed: int, temperature: float,
                     concurrency: int, nested: bool, clip_max_tokens: Optional[int],
                     http_base: Optional[str], api_key: str, requeue_infra: bool,
                     progress: bool = True, validity: str = "a04",
                     clip_min_tokens: int = 1) -> list[dict]:
    sem = asyncio.Semaphore(max(1, concurrency))
    done = {"n": 0}

    async def one(rec):
        async with sem:
            row = await replay_one(client, rec, model=model, seed=seed, temperature=temperature,
                                   nested=nested, clip_max_tokens=clip_max_tokens,
                                   http_base=http_base, api_key=api_key, validity=validity,
                                   clip_min_tokens=clip_min_tokens)
        done["n"] += 1
        if progress:
            print(f"[replay] {done['n']}/{len(records)} {row['prompt_id']} raw={int(row['raw_valid'])} "
                  f"rescued={int(row['rescued_valid'])} fin={row.get('final_finish_reason')} "
                  f"err={row.get('error_kind')} {row['latency_s']}s", flush=True)
        return row

    rows = list(await asyncio.gather(*(one(r) for r in records)))
    if requeue_infra:
        by_id = {r["prompt_id"]: r for r in records}
        retry_idx = [i for i, r in enumerate(rows) if r.get("error_kind") == "infra"]
        if retry_idx:
            print(f"[replay] re-queueing {len(retry_idx)} infra error(s) once", flush=True)
            for i in retry_idx:
                new = await one(by_id[rows[i]["prompt_id"]])
                new["requeued"] = True
                if new.get("error_kind") == "infra":
                    new["infra_error_final"] = True
                rows[i] = new
    return rows


def _import_wrapper(model: str, port: str, api_key: str):
    """Import ``rogueone.llm.clients`` with the env its config module insists on."""
    os.environ.setdefault("LLM_MODEL", model)
    os.environ.setdefault("LLM_PORT", str(port))
    os.environ.setdefault("LLM_API_KEY", api_key)
    for k in ("EMBEDD_MODEL", "EMBEDD_PORT", "EMBEDD_API_KEY"):
        os.environ.setdefault(k, "unused")
    os.environ["ROGUE_ONE_LLM_TRACE"] = "0"  # the replay writes its own records
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from rogueone.llm import clients  # noqa: WPS433

    install_attempt_capture(clients)
    return clients


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", default=DEFAULT_PROMPTS,
                    help="local .jsonl(.gz) or W&B artifact ref (name:alias, entity/project/name:alias)")
    ap.add_argument("--prompts-project", default=DEFAULT_PROJECT)
    ap.add_argument("--model", default=os.getenv("LLM_MODEL"))
    ap.add_argument("--port", default=os.getenv("LLM_PORT"))
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--api-key", default=None, help="default $LLM_API_KEY (never printed)")
    ap.add_argument("--seed", type=int, default=int(os.getenv("ROGUE_ONE_LLM_SEED") or 42))
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--clip-max-tokens", type=int, default=None)
    ap.add_argument("--clip-min-tokens", type=int, default=1,
                    help="floor for a clipped max_tokens (A18 §4: 1024)")
    ap.add_argument("--validity", choices=("a04", "a18"), default="a04",
                    help="a04: only a well-formed tool call is valid; a18: A18 §2 move test "
                         "(answers scored with the role threshold), timeout on a healthy server = invalid move")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--limit", type=int, default=None, help="first N records only")
    ap.add_argument("--role", default=None, help="filter raw trace records by role (e.g. extractor)")
    ap.add_argument("--kv-probe", type=int, default=0, metavar="N",
                    help="send the N longest prompts concurrently (Q122 SR3 probe)")
    ap.add_argument("--top-level-only", action="store_true",
                    help="primary validity checks top-level required keys only")
    ap.add_argument("--no-requeue-infra", action="store_true")
    ap.add_argument("--server-log", default=None, help="vLLM server log to count parser exceptions")
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary-prefix", default="stage0")
    # W&B
    ap.add_argument("--wandb", action="store_true", help="log run, table, summary and replies artifact")
    ap.add_argument("--entity", default=os.getenv("WANDB_ENTITY") or DEFAULT_ENTITY)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--ablation-id", default="A04")
    ap.add_argument("--group", default=None, help="default = --ablation-id")
    ap.add_argument("--arm-id", default=None, help="e.g. stage0_Q9 (default stage0_<model tail>)")
    ap.add_argument("--rung", default=None)
    ap.add_argument("--dataset-slot", default=None, help="run-name slot (default replay<N> / kvprobe<N>)")
    ap.add_argument("--fold-seed", type=int, default=42)
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--session-id", default=os.getenv("SESSION_ID"))
    ap.add_argument("--image-tag", default=os.getenv("ROGUE_ONE_IMAGE_TAG"))
    ap.add_argument("--tags", default="", help="extra comma-separated tags")
    ap.add_argument("--replies-artifact", default=None)
    ap.add_argument("--run-id", default=None, help="W&B run id (lets a later step attach files to this run)")
    ap.add_argument("--task-type", default="replay", help="config task_type (A18: mixed)")
    ap.add_argument("--job-type", default=None, help="default = --arm-id")
    args = ap.parse_args(argv)
    if not args.model or not args.port:
        ap.error("--model and --port are required (or set LLM_MODEL / LLM_PORT)")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    api_key = args.api_key or os.getenv("LLM_API_KEY") or "EMPTY"
    nested = not args.top_level_only
    arm_id = args.arm_id or f"stage0_{args.model.split('/')[-1]}"
    group = args.group or args.ablation_id

    path, source, art_ref = resolve_prompts(args.prompts, args.entity, args.prompts_project)
    raw_records = read_records(path)
    if args.role:
        raw_records = [r for r in raw_records if r.get("role") == args.role or "request" in r]
    records = [normalise_record(r, i) for i, r in enumerate(raw_records)]
    records = [r for r in records if r["request"].get("messages")]
    if args.limit:
        records = records[: args.limit]
    mode = "replay"
    concurrency = args.concurrency
    if args.kv_probe:
        records = select_kv_probe(records, args.kv_probe)
        concurrency = len(records)
        mode = "kvprobe"
    n = len(records)
    slot = args.dataset_slot or f"{mode}{n}"
    run_name = args.run_name or f"{args.ablation_id}__{arm_id}__{slot}__f{args.fold_seed}__r{args.replicate}"
    out = Path(args.out or f"./replay_out/{run_name}")
    print(f"[replay] {mode}: {n} prompts from {source}; model={args.model} port={args.port} "
          f"seed={args.seed} T={args.temperature} concurrency={concurrency} "
          f"clip={args.clip_max_tokens} (min {args.clip_min_tokens}) validity={args.validity} "
          f"nested_required={nested} run={run_name}", flush=True)

    clients = _import_wrapper(args.model, args.port, api_key)
    base_http = f"http://{args.host}:{args.port}"
    client = clients.VLLMCompatibleAsyncOpenAI(
        base_url=f"{base_http}/v1", api_key=api_key, agent_name="ReplayTool",
        timeout=args.timeout, max_retries=0,
    )
    run = None
    if args.wandb:
        import wandb  # noqa: WPS433

        base_tags = [args.session_id, args.image_tag, mode]
        if args.validity == "a04":
            base_tags.insert(1, "smoke")
        tags = [t for t in (base_tags + [t.strip() for t in args.tags.split(",")]) if t]
        init_kw = {"id": args.run_id} if args.run_id else {}
        run = wandb.init(entity=args.entity, project=args.project, group=group,
                         job_type=args.job_type or arm_id, name=run_name, tags=tags, **init_kw)
        if art_ref:
            run.use_artifact(art_ref)  # lineage: prompts -> this run
        run.config.update({
            "ablation_id": args.ablation_id, "arm_id": arm_id, "dataset_id": slot, "mode": mode,
            "task_type": args.task_type, "fold_seed": args.fold_seed, "llm_seed": args.seed,
            "replicate": args.replicate, "is_sentinel": 0, "session_id": args.session_id,
            "image_tag": args.image_tag, "rung_S": args.rung, "rung_E": args.rung, "rung_T": args.rung,
            "model": args.model, "model_S": args.model, "model_E": args.model, "model_T": args.model,
            "model_W": args.model, "port": args.port, "temperature": args.temperature,
            "parallel": concurrency, "clip_max_tokens": args.clip_max_tokens,
            "clip_min_tokens": args.clip_min_tokens, "timeout_s": args.timeout, "validity": args.validity,
            "prompts_source": source, "n_prompts": n, "nested_required_keys": nested,
            "flags/ROGUE_ONE_LLM_SEED": str(args.seed), "replay_tool": "scripts/agent2027/replay_tool.py",
        }, allow_val_change=True)

    t0 = time.monotonic()
    rows = asyncio.run(run_replay(
        records, client, model=args.model, seed=args.seed, temperature=args.temperature,
        concurrency=concurrency, nested=nested, clip_max_tokens=args.clip_max_tokens,
        http_base=base_http, api_key=api_key, requeue_infra=not args.no_requeue_infra,
        validity=args.validity, clip_min_tokens=args.clip_min_tokens,
    ))
    wall = time.monotonic() - t0
    summary = summarise(rows, prefix=args.summary_prefix, wall_s=wall, validity=args.validity)
    summary.update(mode=mode, model=args.model, seed=args.seed, prompts_source=source, run_name=run_name)
    if mode == "kvprobe":
        errs = [r for r in rows if r.get("error_kind")]
        summary.update(kv_probe_n=n, kv_probe_errors=len(errs), kv_probe_ok=n - len(errs),
                       kv_probe_longest_prompt_tokens=max([r["recorded_max_prompt_tokens"] for r in rows] or [0]),
                       kv_probe_error_messages=[f"{r['prompt_id']}: {r['error_message']}" for r in errs])
    if args.server_log:
        summary[f"{args.summary_prefix}_parser_exceptions"] = count_parser_exceptions(Path(args.server_log))
    paths = write_outputs(out, rows, summary)

    print(json.dumps(summary, indent=2, default=str))
    print(f"[replay] raw valid {summary[f'{args.summary_prefix}_valid_call_raw']}/{n}, rescued valid "
          f"{summary[f'{args.summary_prefix}_valid_call_rescued']}/{n}, finish=length share "
          f"{summary[f'{args.summary_prefix}_finish_length_share']}, context errors "
          f"{summary['context_errors']}, wall {wall:.1f}s -> {out}", flush=True)

    if run is not None:
        import wandb  # noqa: WPS433

        table = wandb.Table(columns=list(CSV_COLUMNS),
                            data=[[_cell(r.get(c)) for c in CSV_COLUMNS] for r in rows])
        run.log({f"{mode}_calls": table})
        run.summary.update({k: v for k, v in summary.items() if not isinstance(v, (list, dict))})
        art_name = args.replies_artifact or f"stage0-replies-{arm_id}".replace(".", "_")
        if mode == "kvprobe":
            art_name = args.replies_artifact or f"stage0-kvprobe-{arm_id}".replace(".", "_")
        art = wandb.Artifact(art_name, type="replay-replies", metadata=artifact_metadata(summary))
        for p in paths.values():
            art.add_file(str(p))
        run.log_artifact(art)
        run.finish()
    return 0


#: W&B refuses an artifact with more than 100 metadata keys.
ARTIFACT_METADATA_MAX_KEYS = 100


def artifact_metadata(summary: dict) -> dict:
    """Scalar summary keys for the replies artifact, within W&B's 100-key limit.

    The full summary is always in the run summary and in ``replay_summary.json`` inside
    the artifact; when it is too long (A18's per-role and per-stratum keys contain
    ``__``), the artifact metadata keeps only the top-level keys.
    """
    meta = {k: v for k, v in summary.items() if not isinstance(v, (list, dict))}
    if len(meta) > ARTIFACT_METADATA_MAX_KEYS:
        meta = {k: v for k, v in meta.items() if "__" not in k}
        meta = dict(list(meta.items())[:ARTIFACT_METADATA_MAX_KEYS - 1])
        meta["metadata_truncated"] = True
    return meta


def _cell(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


if __name__ == "__main__":
    sys.exit(main())
