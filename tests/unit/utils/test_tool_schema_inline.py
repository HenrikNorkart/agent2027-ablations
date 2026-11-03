"""ROGUE_ONE_INLINE_TOOL_SCHEMA_REFS: tool schemas without "$ref" for vLLM's qwen3_coder parser."""
import json

from pydantic import BaseModel, Field

from rogueone.llm import clients


class PandasCommandArgs(BaseModel):
    command: str = Field(..., description="pandas command")


class Wrapper(BaseModel):
    args: PandasCommandArgs = Field(..., description="the payload")


def _tools():
    params = Wrapper.model_json_schema()
    assert "$defs" in params and "$ref" in json.dumps(params)
    return [{"type": "function", "function": {"name": "generic_pandas_tool", "parameters": params}},
            {"type": "function", "function": {"name": "plain", "parameters": {"type": "object", "properties": {}}}}]


def test_inline_removes_refs_and_types_args_as_object():
    original = _tools()
    before = json.dumps(original)
    kwargs = {"tools": original}
    clients._inline_tool_schemas(kwargs)
    params = kwargs["tools"][0]["function"]["parameters"]
    assert "$ref" not in json.dumps(params) and "$defs" not in params
    args = params["properties"]["args"]
    assert args["type"] == "object" and args["properties"]["command"]["type"] == "string"
    assert args["required"] == ["command"]
    assert kwargs["tools"][1]["function"]["parameters"] == {"type": "object", "properties": {}}
    # the caller's tool list is not mutated
    assert json.dumps(original) == before


def test_recursive_definition_stops_after_one_level():
    schema = {"$defs": {"Node": {"type": "object", "properties": {"child": {"$ref": "#/$defs/Node"}}}},
              "type": "object", "properties": {"root": {"$ref": "#/$defs/Node"}}}
    out = clients._inline_schema_refs(schema)
    assert out["properties"]["root"]["type"] == "object"
    assert out["properties"]["root"]["properties"]["child"] == {"$ref": "#/$defs/Node"}


def test_tester_code_input_accepts_json_object_string(monkeypatch):
    from rogueone.llm.agents.core.tester.agent import CodeInput
    from rogueone.utils import rouge_one_cfg

    payload = json.dumps({"code": "print('a\\nb')", "reasoning": "why"})
    monkeypatch.setattr(rouge_one_cfg, "inline_tool_schema_refs", False)
    assert CodeInput.model_validate(payload).code == payload  # unchanged default behaviour
    monkeypatch.setattr(rouge_one_cfg, "inline_tool_schema_refs", True)
    ci = CodeInput.model_validate(payload)
    assert ci.code == "print('a\\nb')" and ci.reasoning == "why"
    assert CodeInput.model_validate("x = 1").code == "x = 1"
    assert CodeInput.model_validate("{'not': json}").code == "{'not': json}"
