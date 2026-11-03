"""E2 flags: config parsing and reasoning_effort passthrough in the vLLM client."""

import asyncio
import os

for _k, _v in {"EMBEDD_API_KEY": "x", "EMBEDD_MODEL": "x", "EMBEDD_PORT": "1", "LLM_MODEL": "x"}.items():
    os.environ.setdefault(_k, _v)

import pytest

from rogueone.llm.clients import VLLMCompatibleAsyncOpenAI
from rogueone.utils import rouge_one_cfg
from rogueone.utils.config import rogueoneConfig

pytestmark = [pytest.mark.unit]

_ENV = (
    "ROGUE_ONE_TOOL_ERROR_FEEDBACK",
    "ROGUE_ONE_TOOL_OUTPUT_MAX_CHARS",
    "ROGUE_ONE_REASONING_EFFORT_PASSTHROUGH",
    "ROGUE_ONE_EXTRACTOR_ROUNDS",
)


def _fresh_cfg():
    cfg = object.__new__(rogueoneConfig)
    cfg.__init__()
    return cfg


def test_defaults_keep_current_behaviour(monkeypatch):
    for k in _ENV:
        monkeypatch.delenv(k, raising=False)
    cfg = _fresh_cfg()
    assert cfg.tool_error_feedback == "message"
    assert cfg.tool_output_max_chars == 0
    assert cfg.reasoning_effort_passthrough is False
    assert cfg.extractor_rounds == 10


def test_flags_parse_from_env(monkeypatch):
    monkeypatch.setenv("ROGUE_ONE_TOOL_ERROR_FEEDBACK", " None ")
    monkeypatch.setenv("ROGUE_ONE_TOOL_OUTPUT_MAX_CHARS", "8000")
    monkeypatch.setenv("ROGUE_ONE_REASONING_EFFORT_PASSTHROUGH", "1")
    monkeypatch.setenv("ROGUE_ONE_EXTRACTOR_ROUNDS", "3")
    cfg = _fresh_cfg()
    assert cfg.tool_error_feedback == "none"
    assert cfg.tool_output_max_chars == 8000
    assert cfg.reasoning_effort_passthrough is True
    assert cfg.extractor_rounds == 3


def test_invalid_tool_error_feedback_rejected(monkeypatch):
    monkeypatch.setenv("ROGUE_ONE_TOOL_ERROR_FEEDBACK", "hidden")
    with pytest.raises(ValueError):
        _fresh_cfg()


@pytest.mark.parametrize("passthrough", [False, True])
def test_reasoning_effort_passthrough(monkeypatch, passthrough):
    monkeypatch.setattr(rouge_one_cfg, "reasoning_effort_passthrough", passthrough)
    client = VLLMCompatibleAsyncOpenAI(api_key="x", base_url="http://localhost:1/v1")
    seen = {}

    async def fake_create(*args, **kwargs):
        seen.update(kwargs)
        return object()

    client.chat.completions.create = fake_create
    client._wrap_completions()
    asyncio.run(
        client.chat.completions.create(
            model="m", messages=[], reasoning_effort="high", logprobs=True, stream=True
        )
    )
    assert ("reasoning_effort" in seen) is passthrough
    assert "logprobs" not in seen
