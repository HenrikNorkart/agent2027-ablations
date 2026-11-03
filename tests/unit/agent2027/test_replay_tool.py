"""X2 replay tool: A04 §2 valid-call classification (raw vs rescued) through the real
``VLLMCompatibleAsyncOpenAI`` wrapper, with the upstream stream mocked."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from rogueone.llm import clients
from rogueone.utils import instrumentation as ins
from scripts.agent2027 import replay_tool as rt
from scripts.agent2027 import stage0_prompts as sp

pytestmark = [pytest.mark.unit]

PANDAS_ARGS = {
    "$defs": {
        "PandasCommandArgs": {
            "properties": {"command": {"type": "string"}, "explanation": {"type": "string"}},
            "required": ["command", "explanation"],
            "type": "object",
        }
    },
    "properties": {"args": {"$ref": "#/$defs/PandasCommandArgs"}},
    "required": ["args"],
    "type": "object",
}
TOOLS = [
    {"type": "function", "function": {"name": "generic_pandas_tool", "parameters": PANDAS_ARGS}},
    {"type": "function", "function": {"name": "append_new_attribute", "parameters": {
        "properties": {"args": {"type": "object", "required": ["name", "command", "argumentation"],
                                "properties": {"name": {}, "command": {}, "argumentation": {}}}},
        "required": ["args"], "type": "object"}}},
    {"type": "function", "function": {"name": "list_known_attributes_tool",
                                      "parameters": {"properties": {}, "required": [], "type": "object"}}},
    {"type": "function", "function": {"name": "search_in_literature_tool", "parameters": {
        "properties": {"query": {"type": "string"}}, "required": ["query"], "type": "object"}}},
]
GOOD_ARGS = json.dumps({"args": {"command": "result = df['a'].mean()", "explanation": "mean"}})
RECORD = {
    "prompt_id": "s0-000",
    "role": "extractor",
    "max_prompt_tokens": 1234,
    "request": {
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}],
        "tools": TOOLS,
        "temperature": 0.2,
        "max_tokens": 32768,
        "seed": 999,
        "model": "openai/gpt-oss-120b",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "tool_choice": "<openai.Omit object at 0x1>",
    },
}


# --------------------------------------------------------------------------- pure classification
def test_valid_json_call():
    r = rt.classify_tool_calls("tool_calls", [{"name": "generic_pandas_tool", "arguments": GOOD_ARGS}], TOOLS)
    assert r["valid"] and r["reason"] == "ok"


def test_wrong_tool_name():
    r = rt.classify_tool_calls("tool_calls", [{"name": "delete_everything", "arguments": "{}"}], TOOLS)
    assert not r["valid"] and r["reason"].startswith("unknown_tool")


def test_missing_required_key_nested_and_top_level():
    args = json.dumps({"args": {"command": "result = 1"}})  # no 'explanation'
    nested = rt.classify_tool_calls("tool_calls", [{"name": "generic_pandas_tool", "arguments": args}], TOOLS)
    top = rt.classify_tool_calls("tool_calls", [{"name": "generic_pandas_tool", "arguments": args}], TOOLS,
                                 nested=False)
    assert not nested["valid"] and nested["missing"] == ["args.explanation"]
    assert top["valid"]
    bare = rt.classify_tool_calls("tool_calls", [{"name": "search_in_literature_tool", "arguments": "{}"}], TOOLS)
    assert not bare["valid"] and bare["reason"] == "missing_required:query"


def test_finish_length_and_unparseable_args():
    r = rt.classify_tool_calls("length", [{"name": "generic_pandas_tool", "arguments": '{"args": {"comm'}], TOOLS)
    assert not r["valid"] and r["reason"] == "finish_reason=length" and r["args_parse_errors"] == 1


def test_no_call_and_empty_list_tool():
    assert rt.classify_tool_calls("tool_calls", [], TOOLS)["reason"] == "no_tool_call"
    assert rt.classify_tool_calls("tool_calls", [{"name": "list_known_attributes_tool", "arguments": "{}"}],
                                  TOOLS)["valid"]


def test_classify_error_kinds():
    class E(Exception):
        def __init__(self, msg, status):
            super().__init__(msg)
            self.status_code = status

    assert rt.classify_error(E("This model's maximum context length is 131072 tokens", 400)) == ("context", 400)
    assert rt.classify_error(E("bad gateway", 502)) == ("infra", 502)
    assert rt.classify_error(type("APITimeoutError", (Exception,), {})("t"))[0] == "timeout"
    assert rt.classify_error(E("invalid tool_choice", 400)) == ("other", 400)


def test_normalise_record_drops_placeholders_and_build_request_overrides():
    rec = rt.normalise_record(RECORD, 0)
    assert "tool_choice" not in rec["request"]
    kw = rt.build_request(rec, "M", 42, 0.7)
    assert kw["seed"] == 42 and kw["model"] == "M" and kw["temperature"] == 0.7
    assert kw["max_tokens"] == 32768 and kw["tools"] == TOOLS


def test_trace_record_shape_and_kv_probe_selection():
    trace = {"seq": 7, "role": "extractor", "agent_name": "ExtractorAgent",
             "messages": RECORD["request"]["messages"], "tools": TOOLS, "max_tokens": 10,
             "usage": {"prompt_tokens": 50}, "attempts": [{"prompt_tokens": 60}],
             "stream": "<openai.Omit object at 0x2>"}
    rec = rt.normalise_record(trace, 3)
    assert rec["max_prompt_tokens"] == 60 and rec["prompt_id"] == "trace-7"
    recs = [dict(prompt_id=f"p{i}", max_prompt_tokens=t) for i, t in enumerate([5, 90, 7, 90, 1])]
    assert [r["prompt_id"] for r in rt.select_kv_probe(recs, 3)] == ["p1", "p3", "p2"]


# --------------------------------------------------------------------------- through the wrapper
def _chunk(content=None, tool_calls=None, finish=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)], usage=None)


def _usage_chunk(p=100, c=20):
    return SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=p, completion_tokens=c,
                                                            total_tokens=p + c))


def _tc(name, args, idx=0):
    return [SimpleNamespace(index=idx, id=f"call_{idx}", function=SimpleNamespace(name=name, arguments=args))]


def _stream(chunks):
    async def gen():
        for c in chunks:
            yield c
    return gen()


@pytest.fixture
def replay(monkeypatch):
    """Run ``replay_one`` on RECORD against a scripted sequence of upstream streams."""
    monkeypatch.delenv(ins.TRACE_FLAG, raising=False)
    rt.install_attempt_capture(clients)
    from openai.resources.chat.completions import AsyncCompletions

    def run(streams):
        sent = []
        queue = list(streams)

        async def fake_create(self, *args, **kwargs):
            sent.append(kwargs)
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return _stream(item)

        monkeypatch.setattr(AsyncCompletions, "create", fake_create)
        client = clients.VLLMCompatibleAsyncOpenAI(base_url="http://127.0.0.1:9/v1", api_key="x",
                                                   agent_name="ReplayTool", max_retries=0)
        rec = rt.normalise_record(RECORD, 0)
        row = asyncio.run(rt.replay_one(client, rec, model="M", seed=42, temperature=0.7))
        return row, sent

    yield run
    ins.reset()


def test_wrapper_valid_json_call(replay):
    row, sent = replay([[_chunk(tool_calls=_tc("generic_pandas_tool", GOOD_ARGS)),
                         _chunk(finish="tool_calls"), _usage_chunk()]])
    assert row["raw_valid"] and row["rescued_valid"] and row["attempts"] == 1
    assert row["rescue_kind"] is None and row["prompt_tokens"] == 100
    assert sent[0]["seed"] == 42 and sent[0]["temperature"] == 0.7 and sent[0]["stream"] is True


def test_wrapper_wrong_tool_name(replay):
    row, _ = replay([[_chunk(tool_calls=_tc("delete_everything", "{}")), _chunk(finish="tool_calls"),
                      _usage_chunk()]])
    assert not row["raw_valid"] and not row["rescued_valid"]
    assert row["raw_reason"].startswith("unknown_tool")


def test_wrapper_garbled_name_is_rescued_by_normalisation(replay):
    row, _ = replay([[_chunk(tool_calls=_tc("generic_pandas_tool<|channel|>commentary", GOOD_ARGS)),
                      _chunk(finish="tool_calls"), _usage_chunk()]])
    assert not row["raw_valid"] and row["rescued_valid"] and row["rescue_kind"] == "normalised"


def test_wrapper_missing_required_key(replay):
    args = json.dumps({"args": {"name": "f", "command": "result = 1"}})  # no argumentation
    row, _ = replay([[_chunk(tool_calls=_tc("append_new_attribute", args)), _chunk(finish="tool_calls"),
                      _usage_chunk()]])
    assert not row["raw_valid"] and not row["rescued_valid"]
    assert row["raw_valid_top"] and row["rescued_valid_top"]
    assert row["rescued_reason"] == "missing_required:args.argumentation"


def test_wrapper_finish_length(replay):
    row, _ = replay([[_chunk(content="Let me first think about the"), _chunk(finish="length"),
                      _usage_chunk(c=32768)]])
    assert row["raw_finish_reason"] == "length" and row["final_finish_reason"] == "length"
    assert not row["raw_valid"] and not row["rescued_valid"] and row["attempts"] == 1
    s = rt.summarise([row])
    assert s["stage0_valid_call_raw"] == 0 and s["stage0_finish_length_share"] == 1.0


def test_wrapper_length_inside_tool_call_keeps_server_finish_reason(replay):
    cut = [_chunk(tool_calls=_tc("generic_pandas_tool", '{"args": {"command": "res')),
           _chunk(finish="length"), _usage_chunk(c=32768)]
    # unparseable arguments look like splice damage: the wrapper re-rolls twice, then passes it on
    row, sent = replay([list(cut), list(cut), list(cut)])
    assert len(sent) == 3 and row["attempts"] == 3
    # the aggregator rewrites finish_reason to tool_calls; the replay keeps the server's value
    assert row["raw_finish_reason"] == "length" and row["raw_finish_reason_aggregated"] == "tool_calls"
    assert not row["raw_valid"] and not row["rescued_valid"]
    assert row["args_parse_errors_rescued"] == 1 and row["rescued_reason"] == "args_not_json"


def test_wrapper_text_only_json_rescued(replay):
    text = json.dumps({"name": "generic_pandas_tool",
                       "arguments": {"command": "result = df['a'].mean()", "explanation": "mean"}})
    row, _ = replay([[_chunk(content=text), _chunk(finish="stop"), _usage_chunk()]])
    assert not row["raw_valid"] and row["raw_reason"] == "finish_reason=stop"
    assert row["rescued_valid"] and row["rescue_kind"] == "text_tool_call"
    assert row["final_finish_reason"] == "tool_calls" and row["attempts"] == 1


def test_wrapper_empty_then_valid_counts_retry(replay):
    row, sent = replay([[_chunk(content=""), _chunk(finish="stop"), _usage_chunk()],
                        [_chunk(tool_calls=_tc("generic_pandas_tool", GOOD_ARGS)), _chunk(finish="tool_calls"),
                         _usage_chunk()]])
    assert row["attempts"] == 2 and row["rerolls"] == 1 and len(sent) == 2
    assert not row["raw_valid"] and row["rescued_valid"]


def test_wrapper_context_error(replay):
    class BadRequest(Exception):
        status_code = 400

    row, _ = replay([BadRequest("maximum context length is 131072 tokens; you requested 140000")])
    assert row["context_error"] and row["error_kind"] == "context"
    assert not row["raw_valid"] and not row["rescued_valid"] and row["raw_reason"] == "error"
    s = rt.summarise([row])
    assert s["context_errors"] == 1 and s["stage0_valid_call_rescued"] == 0


def test_write_outputs(tmp_path, replay):
    row, _ = replay([[_chunk(tool_calls=_tc("generic_pandas_tool", GOOD_ARGS)), _chunk(finish="tool_calls"),
                      _usage_chunk()]])
    paths = rt.write_outputs(tmp_path, [row], rt.summarise([row]))
    line = json.loads(paths["jsonl"].read_text().splitlines()[0])
    assert line["attempt_details"][0]["server_finish_reason"] == "tool_calls"
    assert "raw_valid" in paths["csv"].read_text().splitlines()[0]
    assert json.loads(paths["summary"].read_text())["stage0_valid_call_raw"] == 1


def test_cli_end_to_end_with_mocked_server(tmp_path, monkeypatch):
    """main(): local prompt file -> replay (seed/model override) -> CSV/JSONL/summary; and --kv-probe."""
    monkeypatch.setenv(ins.TRACE_FLAG, "0")  # main() pins it to 0; monkeypatch restores it
    from openai.resources.chat.completions import AsyncCompletions

    sent = []

    async def fake_create(self, *args, **kwargs):
        sent.append(kwargs)
        return _stream([_chunk(tool_calls=_tc("generic_pandas_tool", GOOD_ARGS)), _chunk(finish="tool_calls"),
                        _usage_chunk()])

    monkeypatch.setattr(AsyncCompletions, "create", fake_create)
    prompts = tmp_path / "p.jsonl"
    with open(prompts, "w") as fh:
        for i, t in enumerate([10, 30, 20]):
            fh.write(json.dumps({**RECORD, "prompt_id": f"s0-{i:03d}", "max_prompt_tokens": t}) + "\n")
    out = tmp_path / "out"
    assert rt.main(["--prompts", str(prompts), "--model", "M", "--port", "9", "--seed", "7",
                    "--arm-id", "stage0_X", "--out", str(out)]) == 0
    summary = json.loads((out / "replay_summary.json").read_text())
    assert summary["stage0_valid_call_raw"] == 3 and summary["n_prompts"] == 3
    assert summary["run_name"] == "A04__stage0_X__replay3__f42__r0"
    assert {k["seed"] for k in sent} == {7} and {k["model"] for k in sent} == {"M"}
    sent.clear()
    out2 = tmp_path / "kv"
    assert rt.main(["--prompts", str(prompts), "--model", "M", "--port", "9", "--kv-probe", "2",
                    "--arm-id", "stage0_X", "--out", str(out2)]) == 0
    s2 = json.loads((out2 / "replay_summary.json").read_text())
    assert s2["kv_probe_n"] == 2 and s2["kv_probe_errors"] == 0 and s2["kv_probe_longest_prompt_tokens"] == 30
    assert s2["run_name"] == "A04__stage0_X__kvprobe2__f42__r0" and len(sent) == 2
    ins.reset()


# --------------------------------------------------------------------------- prompt-set sampler
def test_stage0_sampler_eligibility_and_determinism():
    def rec(seq, **kw):
        base = {"agent_name": "ExtractorAgent", "seq": seq, "response_model": sp.O120,
                "message": {"tool_calls": [{"function": {"name": "x", "arguments": "{}"}}]},
                "usage": {"prompt_tokens": seq * 10}, "attempts": []}
        base.update(kw)
        return base

    assert sp.is_eligible(rec(1))
    assert not sp.is_eligible(rec(1, rescue="text_tool_call"))
    assert not sp.is_eligible(rec(1, message={"content": "hi"}))
    assert not sp.is_eligible(rec(1, agent_name="TesterAgent"))
    assert not sp.is_eligible(rec(1, response_model="openai/gpt-oss-20b"))

    from pathlib import Path
    prim = [sp.make_record(rec(i), "S1", "A02-901", Path("/r/S1/A02-901/tasks/regression/d/output/t.gz"),
                           Path("/r"), "G0a") for i in range(1, 4)]
    sec = [sp.make_record(rec(i), "S2", "A02-001", Path("/r/S2/A02-001/tasks/regression/d/output/t.gz"),
                          Path("/r"), "A02") for i in range(1, 11)]
    a = sp.sample([dict(r) for r in prim], [dict(r) for r in sec], 5, 42, 2)
    b = sp.sample([dict(r) for r in prim], [dict(r) for r in sec], 5, 42, 2)
    assert [r["provenance"]["seq"] for r in a] == [r["provenance"]["seq"] for r in b]
    assert sum(r["provenance"]["source_group"] == "G0a" for r in a) == 3
    assert sum(r["kv_probe"] for r in a) == 2
    assert max(r["max_prompt_tokens"] for r in a if not r["kv_probe"]) <= min(
        r["max_prompt_tokens"] for r in a if r["kv_probe"])


def test_artifact_metadata_respects_wandb_key_limit():

    small = {"a": 1, "b": "x", "lst": [1]}
    assert rt.artifact_metadata(small) == {"a": 1, "b": "x"}
    big = {f"k{i}": i for i in range(40)}
    big.update({f"rp1_valid_call_raw__stratum_{i}": 1.0 for i in range(120)})
    meta = rt.artifact_metadata(big)
    assert len(meta) <= rt.ARTIFACT_METADATA_MAX_KEYS
    assert meta["metadata_truncated"] is True and "k0" in meta
    assert not any("__" in k for k in meta)
