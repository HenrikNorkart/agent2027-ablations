"""E3: token meter, discarded re-rolls, trace records, counters, summary names.

The client wrapper is driven end to end with a fake ``chat.completions.create``
that yields streaming chunks, so the discarded splice/empty attempts, the text
rescue, reasoning tokens and the trace record are exercised through the real
``VLLMCompatibleAsyncOpenAI`` code path.
"""

import asyncio
import gzip
import json
import os
import zlib

for _k, _v in {"EMBEDD_API_KEY": "x", "EMBEDD_MODEL": "x", "EMBEDD_PORT": "1", "LLM_MODEL": "x"}.items():
    os.environ.setdefault(_k, _v)

import pytest

from rogueone.llm.clients import VLLMCompatibleAsyncOpenAI
from rogueone.utils import instrumentation as ins

pytestmark = [pytest.mark.unit]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "append_new_attribute",
            "parameters": {"type": "object", "properties": {"args": {"type": "object"}}, "required": ["args"]},
        },
    }
]


@pytest.fixture(autouse=True)
def _clean(tmp_path, monkeypatch):
    monkeypatch.delenv(ins.TRACE_FLAG, raising=False)
    ins.configure(tmp_path / "output")
    yield
    ins.reset()


# --------------------------------------------------------------------------- fakes


class _Delta:
    def __init__(self, content=None, tool_calls=None, reasoning=None):
        self.content = content
        self.tool_calls = tool_calls
        if reasoning is not None:
            self.reasoning_content = reasoning


class _Choice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class _Fn:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class _TC:
    def __init__(self, index, id=None, name=None, arguments=None):
        self.index = index
        self.id = id
        self.function = _Fn(name, arguments)


class _Details:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Usage:
    def __init__(self, prompt, completion, reasoning=None, cached=None):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion
        self.completion_tokens_details = (
            _Details(reasoning_tokens=reasoning) if reasoning is not None else None
        )
        self.prompt_tokens_details = _Details(cached_tokens=cached) if cached is not None else None


class _Chunk:
    def __init__(self, choices=None, usage=None):
        self.choices = choices or []
        if usage is not None:
            self.usage = usage


async def _aiter(items):
    for it in items:
        yield it


def _tool_stream(arguments, usage, reasoning=None):
    return _aiter(
        [
            _Chunk([_Choice(_Delta(reasoning=reasoning, tool_calls=[_TC(0, "c1", "append_new_attribute", arguments)]))]),
            _Chunk([_Choice(_Delta(), "tool_calls")]),
            _Chunk(usage=usage),
        ]
    )


def _text_stream(content, usage, finish="stop"):
    return _aiter(
        [_Chunk([_Choice(_Delta(content=content))]), _Chunk([_Choice(_Delta(), finish)]), _Chunk(usage=usage)]
    )


def _client(agent_name, script):
    """Real wrapper around a scripted ``create``: ``script`` is a list of items,
    each either a stream or an exception to raise."""
    client = VLLMCompatibleAsyncOpenAI(base_url="http://localhost:1/v1", api_key="x", agent_name=agent_name)
    queue = list(script)

    async def fake_create(*args, **kwargs):
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    client.chat.completions.create = fake_create
    client._wrap_completions()
    return client


def _call(client, **extra):
    kwargs = dict(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": "hi"}],
        tools=TOOLS,
        temperature=0.7,
        max_tokens=100,
    )
    kwargs.update(extra)
    return asyncio.run(client.chat.completions.create(**kwargs))


GOOD = json.dumps({"args": {"name": "a", "command": "result = 1"}})
SPLICED = json.dumps({"args": {}, "command": "x = 'abc"})  # unterminated string literal


# --------------------------------------------------------------------------- meter


def test_final_call_tokens_per_role_with_reasoning():
    ins.set_iteration(3, 0, 3)
    client = _client("ExtractorAgent", [_tool_stream(GOOD, _Usage(100, 40, reasoning=25, cached=10))])
    _call(client)
    s = ins.compute_summary()
    assert s["tok_compl_extractor"] == 40
    assert s["tok_prompt_extractor"] == 100
    assert s["tok_reasoning_extractor"] == 25
    assert s["tok_cached_extractor"] == 10
    assert s["llm_calls_extractor"] == 1 and s["llm_calls_total"] == 1
    assert s["tok_compl_scientist"] == 0 and s["tok_compl_tester"] == 0
    assert s["tok_compl_total"] == 40
    assert s["max_prompt_tokens"] == 100
    assert s["reasoning_exceeds_completion_calls"] == 0


def test_reasoning_larger_than_completion_is_flagged():
    client = _client("ScientistAgent", [_tool_stream(GOOD, _Usage(10, 5, reasoning=9))])
    _call(client)
    assert ins.compute_summary()["reasoning_exceeds_completion_calls"] == 1


def test_discarded_splice_reroll_is_metered():
    client = _client(
        "ExtractorAgent",
        [_tool_stream(SPLICED, _Usage(100, 50)), _tool_stream(GOOD, _Usage(100, 30))],
    )
    _call(client)
    s = ins.compute_summary()
    assert s["tok_compl_extractor"] == 80  # 50 discarded + 30 final
    assert s["tok_prompt_extractor"] == 200
    assert s["tok_compl_discarded_extractor"] == 50
    assert s["rerolls"] == 1 and s["rerolls_extractor"] == 1
    assert s["llm_calls_total"] == 1  # one logical call
    assert s["rescue_splice_gpt-oss"] == 1
    assert s["rescues_per100_gpt-oss"] == 100.0


def test_discarded_empty_reroll_is_metered_but_not_a_rescue():
    client = _client(
        "TesterAgent",
        [_text_stream("", _Usage(10, 2)), _text_stream("", _Usage(10, 3)), _tool_stream(GOOD, _Usage(10, 7))],
    )
    _call(client)
    s = ins.compute_summary()
    assert s["tok_compl_tester"] == 12 and s["rerolls"] == 2
    assert s["rerolls_empty_gpt-oss"] == 2
    assert s["rescues_per100_gpt-oss"] == 0.0


def test_text_tool_call_rescue_counted_per_family():
    body = json.dumps({"name": "append_new_attribute", "arguments": {"name": "a", "command": "result = 1"}})
    client = _client("ExtractorAgent", [_text_stream(body, _Usage(10, 5))])
    resp = _call(client, model="Qwen/Qwen3.5-9B")
    assert resp.choices[0].message.tool_calls  # rescued
    s = ins.compute_summary()
    assert s["rescue_text_qwen"] == 1 and s["rescues_per100_qwen"] == 100.0
    assert s["rescues_per100_gpt-oss"] == 0.0


def test_search_tokens_charged_to_caller_and_tester_stages():
    ins.set_iteration(0, 0, 0)
    ins.set_caller("scientist")
    _call(_client("WebSearchAgent", [_tool_stream(GOOD, _Usage(70, 20))]))
    ins.set_caller("tester")
    ins.set_stage("assessment", "tester_agent_assessment_step_0")
    _call(_client("TesterAgent", [_tool_stream(GOOD, _Usage(10, 4))]))
    _call(_client("WebSearchAgent", [_tool_stream(GOOD, _Usage(10, 6))]))
    ins.set_stage("pruning", "tester_agent_pruning_step_0")
    _call(_client("TesterAgent", [_tool_stream(GOOD, _Usage(10, 8))]))
    s = ins.compute_summary()
    assert s["tok_compl_scientist"] == 20 and s["tok_search_scientist"] == 20
    assert s["llm_calls_scientist"] == 1 and s["llm_calls_search_scientist"] == 1
    assert s["tok_compl_tester"] == 18 and s["tok_search_tester"] == 6
    assert s["tok_compl_tester_assess"] == 10 and s["tok_compl_tester_prune"] == 8
    assert s["tok_compl_total"] == 38
    assert s["tok_compl_unattributed"] == 0


def test_search_without_caller_is_unattributed_but_in_total():
    _call(_client("WebSearchAgent", [_tool_stream(GOOD, _Usage(5, 9))]))
    s = ins.compute_summary()
    assert s["tok_compl_unattributed"] == 9 and s["tok_compl_total"] == 9


def test_context_errors_http400_and_length():
    class BadRequest(Exception):
        status_code = 400

    client = _client("ExtractorAgent", [BadRequest("context too long")])
    with pytest.raises(BadRequest):
        _call(client, tools=None)  # no tools -> non-streaming path
    s = ins.compute_summary()
    assert s["context_errors_http400"] == 1 and s["llm_call_errors"] == 1
    assert s["llm_calls_extractor"] == 1  # failed calls still count as calls


def test_finish_reason_length_counts_as_context_error():
    client = _client("ScientistAgent", [_text_stream("cut off", _Usage(10, 99), finish="length")])
    _call(client)  # tools present, no tool call, finish=length (not 'stop') -> accepted as is
    assert ins.compute_summary()["context_errors_length"] == 1


def test_per_iteration_tokens_and_iteration_log(tmp_path):
    ins.set_iteration(5, 1, 2)
    _call(_client("ExtractorAgent", [_tool_stream(GOOD, _Usage(11, 22))]))
    ins.log_iteration(5, 1, 2, "completed", {"active_features": 4})
    rows = [json.loads(l) for l in (tmp_path / "output" / ins.ITERATION_FILE).read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["tokens"]["extractor"] == {"prompt": 11, "compl": 22, "calls": 1}
    assert rows[0]["active_features"] == 4 and rows[0]["iteration_in_fold"] == 2


# --------------------------------------------------------------------------- trace


def _read_trace(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


A18_FIELDS = (
    "agent_name caller stage index fold_run_id iteration_in_fold session_name call_seq "
    "messages tools tool_choice extra_body temperature max_tokens model seed "
    "message finish_reason usage"
).split()


def test_trace_off_by_default_writes_nothing(tmp_path):
    _call(_client("ExtractorAgent", [_tool_stream(GOOD, _Usage(1, 1))]))
    assert not (tmp_path / "output" / ins.TRACE_FILE).exists()
    assert ins.compute_summary()["llm_calls_total"] == 1


def test_trace_one_record_per_call_with_a18_fields(tmp_path, monkeypatch):
    monkeypatch.setenv(ins.TRACE_FLAG, "1")
    ins.set_iteration(7, 0, 7)
    ins.set_stage("assessment", "tester_agent_assessment_step_0")
    client = _client("TesterAgent", [_tool_stream(GOOD, _Usage(9, 4, reasoning=2)), _tool_stream(GOOD, _Usage(9, 5))])
    _call(client, tool_choice="auto")
    _call(client)
    ins.set_caller("tester")
    _call(_client("WebSearchAgent", [_tool_stream(GOOD, _Usage(3, 3))]))
    ins.close_trace()
    recs = _read_trace(tmp_path / "output" / ins.TRACE_FILE)
    assert len(recs) == ins.compute_summary()["llm_calls_total"] == 3
    for r in recs:
        for f in A18_FIELDS:
            assert f in r, f
    t0, t1, w = recs
    assert t0["agent_name"] == "TesterAgent" and t0["stage"] == "assessment"
    assert t0["session_name"] == "tester_agent_assessment_step_0"
    assert (t0["call_seq"], t1["call_seq"]) == (1, 2)  # per-session counter
    assert (t0["index"], t0["fold_run_id"], t0["iteration_in_fold"]) == (7, 0, 7)
    assert t0["tool_choice"] == "auto" and t0["tools"] == TOOLS
    assert t0["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False  # as sent
    assert t0["messages"] == [{"role": "user", "content": "hi"}]
    assert t0["usage"]["reasoning_tokens"] == 2 and t0["usage"]["completion_tokens"] == 4
    assert t0["finish_reason"] == "tool_calls"
    assert t0["message"]["tool_calls"][0]["function"]["name"] == "append_new_attribute"
    assert w["agent_name"] == "WebSearchAgent" and w["caller"] == "tester" and w["stage"] == "search"
    assert w["role"] == "tester" and w["session_name"] == "web_search_agent"


def test_trace_keeps_discarded_attempts_and_rescue_raw_content(tmp_path, monkeypatch):
    monkeypatch.setenv(ins.TRACE_FLAG, "1")
    body = json.dumps({"name": "append_new_attribute", "arguments": {"name": "a", "command": "result = 1"}})
    client = _client(
        "ExtractorAgent",
        [_tool_stream(SPLICED, _Usage(10, 6)), _text_stream(body, _Usage(10, 4))],
    )
    _call(client)
    ins.close_trace()
    (rec,) = _read_trace(tmp_path / "output" / ins.TRACE_FILE)
    assert rec["rerolls"] == 1
    assert [a["discard_reason"] for a in rec["attempts"]] == ["splice", None]
    assert rec["discarded_usage"] == {"prompt_tokens": 10, "completion_tokens": 6}
    assert rec["rescue"] == "text_tool_call" and rec["raw_content_before_rescue"] == body
    assert rec["usage"]["completion_tokens"] == 4  # the final attempt, not the sum


def test_trace_is_readable_before_close(tmp_path, monkeypatch):
    """Sync flush: a killed run must still leave complete records on disk."""
    monkeypatch.setenv(ins.TRACE_FLAG, "1")
    _call(_client("ExtractorAgent", [_tool_stream(GOOD, _Usage(1, 1))]))
    _call(_client("ExtractorAgent", [_tool_stream(GOOD, _Usage(1, 1))]))
    raw = (tmp_path / "output" / ins.TRACE_FILE).read_bytes()  # writer still open
    text = zlib.decompressobj(wbits=31).decompress(raw).decode("utf-8")
    assert len(text.strip().splitlines()) == 2
    ins.close_trace()


def test_trace_records_errors(tmp_path, monkeypatch):
    monkeypatch.setenv(ins.TRACE_FLAG, "1")

    class Boom(Exception):
        status_code = 400

    with pytest.raises(Boom):
        _call(_client("ExtractorAgent", [Boom("ctx")]))
    ins.close_trace()
    (rec,) = _read_trace(tmp_path / "output" / ins.TRACE_FILE)
    assert rec["error"]["status_code"] == 400 and rec["message"] is None


# --------------------------------------------------------------------------- summary pieces


def test_summary_contains_every_required_counter_with_zeros():
    s = ins.compute_summary(models={"scientist": "openai/gpt-oss-120b"})
    required = (
        "tok_compl_scientist tok_compl_extractor tok_compl_tester tok_compl_total "
        "tok_search_scientist tok_search_extractor tok_search_tester tok_prompt_total "
        "tok_prompt_scientist llm_calls_scientist llm_calls_total rerolls wall_s wall_s_scientist "
        "wall_s_extractor wall_s_tester max_prompt_tokens context_errors flop_proxy "
        "tok_compl_tester_assess tok_compl_tester_prune fallback_share rescues_per100_gpt-oss "
        "rescues_per100_qwen floor_activations silent_rejections append_success err_per_accepted "
        "err_repeat_rate report_read_rate fallback_extractor fallback_scientist_focus "
        "fallback_tester_report runner_runs_extractor"
    ).split()
    required += [f"err_site_{x}" for x in ins.ERROR_SITES]
    for k in required:
        assert k in s, k


def test_wall_clock_per_role():
    ins.set_iteration(0, 0, 0)
    tok = ins.stage_begin("scientist")
    import time

    time.sleep(0.02)
    ins.stage_end(tok)
    s = ins.compute_summary()
    assert s["wall_s_scientist"] >= 0.02 and s["wall_s_extractor"] == 0 and s["wall_s"] >= s["wall_s_scientist"]


def test_flop_proxy_uses_active_params():
    ins.set_iteration(0, 0, 0)
    _call(_client("ScientistAgent", [_tool_stream(GOOD, _Usage(1000, 500))]))
    s = ins.compute_summary(models={"scientist": "openai/gpt-oss-120b", "extractor": "x", "tester": "x"})
    assert s["flop_proxy"] == pytest.approx(2 * 5.1e9 * 1500)
    assert s["flop_proxy_compl"] == pytest.approx(2 * 5.1e9 * 500)
    # a role with tokens but an unknown model makes the proxy undefined, not wrong
    _call(_client("ExtractorAgent", [_tool_stream(GOOD, _Usage(1, 1))]))
    assert ins.compute_summary(models={"scientist": "openai/gpt-oss-120b", "extractor": "mystery"})["flop_proxy"] is None


@pytest.mark.parametrize(
    "model, expected",
    [
        ("openai/gpt-oss-120b", 5.1),
        ("openai/gpt-oss-20b", 3.6),
        ("Qwen/Qwen3.5-122B-A10B", 10.0),
        ("Qwen/Qwen3-4B", 4.0),
        ("Qwen/Qwen3.5-0.8B", 0.8),
        ("llama", None),
    ],
)
def test_active_params(model, expected):
    assert ins.active_params_b(model) == expected


@pytest.mark.parametrize(
    "model, fam",
    [("openai/gpt-oss-120b", "gpt-oss"), ("Qwen/Qwen3-4B", "qwen"), ("meta/Llama-3", "llama-3")],
)
def test_model_family(model, fam):
    assert ins.model_family(model) == fam


def test_instrumentation_failures_never_raise(monkeypatch):
    # a response object that explodes on every attribute access
    class Evil:
        def __getattr__(self, name):
            raise RuntimeError("boom")

    rec = ins.begin_call("ExtractorAgent")
    ins.end_call(rec, Evil(), None)  # guarded: swallows the RuntimeError
    ins.note_attempt(Evil())
    ins.mark_discard("splice")
    ins.count("x")
    assert ins.finalize_run(upload=False) is not None


# --------------------------------------------------------------------------- stream aggregation


def test_stream_aggregation_usage_unchanged_without_details_and_keeps_details_with():
    from rogueone.llm.clients import _stream_to_chat_completion

    async def create(*a, **k):
        return _tool_stream(GOOD, _Usage(10, 4))

    plain = asyncio.run(_stream_to_chat_completion(create, (), {"messages": []}, "m"))
    assert plain.usage.completion_tokens_details is None and plain.usage.prompt_tokens_details is None
    assert (plain.usage.prompt_tokens, plain.usage.completion_tokens, plain.usage.total_tokens) == (10, 4, 14)

    async def create2(*a, **k):
        return _tool_stream(GOOD, _Usage(10, 4, reasoning=3, cached=2))

    rich = asyncio.run(_stream_to_chat_completion(create2, (), {"messages": []}, "m"))
    assert rich.usage.completion_tokens_details.reasoning_tokens == 3
    assert rich.usage.prompt_tokens_details.cached_tokens == 2


def test_reasoning_text_length_is_recorded_in_the_trace(tmp_path, monkeypatch):
    monkeypatch.setenv(ins.TRACE_FLAG, "1")
    _call(_client("ExtractorAgent", [_tool_stream(GOOD, _Usage(10, 40, reasoning=30), reasoning="x" * 120)]))
    ins.close_trace()
    (rec,) = _read_trace(tmp_path / "output" / ins.TRACE_FILE)
    assert rec["reasoning_chars"] == 120 and rec["usage"]["reasoning_tokens"] == 30
