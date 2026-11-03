"""Unit tests for the vLLM streaming client wrapper, focused on the
include_usage injection that restores token-usage capture.

Background: every CAR run between 2026-04 and 2026-05-05 showed
``LLM_tokens/*`` cumulative values of 0 despite hundreds of LLM calls per
run — confirmed for run m4i2r4v9 (834 history rows, 649 ExtractorAgent
token-log entries, every value 0). The cause was that
``_stream_to_chat_completion`` stripped any caller-supplied
``stream_options`` and never re-added ``{"include_usage": True}``, so vLLM
never emitted a usage chunk, ``usage_data`` stayed None, and the wrapper
recorded (0, 0, 0) for every call.
"""

from __future__ import annotations

import asyncio

import pytest

from rogueone.llm.clients import _stream_to_chat_completion


pytestmark = [pytest.mark.unit]


class _FakeChoice:
    """Minimal stand-in for an OpenAI streaming choice object."""

    class _Delta:
        def __init__(self, content=None, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls

    def __init__(self, content=None, finish_reason=None):
        self.delta = _FakeChoice._Delta(content=content)
        self.finish_reason = finish_reason


class _FakeUsage:
    def __init__(self, prompt_tokens=0, completion_tokens=0, total_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _FakeChunk:
    """Mimics one chunk returned by the OpenAI/vLLM streaming endpoint."""

    def __init__(self, *, choices=None, usage=None):
        self.choices = choices or []
        if usage is not None:
            self.usage = usage


async def _async_iter(items):
    for item in items:
        yield item


def test_stream_to_chat_completion_injects_include_usage():
    """The streaming wrapper must always set stream_options.include_usage so
    vLLM emits a final usage chunk and tokens land in the run."""

    captured: dict = {}

    async def fake_create(*args, **kwargs):
        captured["kwargs"] = kwargs
        return _async_iter(
            [
                _FakeChunk(choices=[_FakeChoice(content="hello", finish_reason="stop")]),
                _FakeChunk(
                    usage=_FakeUsage(
                        prompt_tokens=10, completion_tokens=2, total_tokens=12
                    )
                ),
            ]
        )

    response = asyncio.run(
        _stream_to_chat_completion(
            fake_create, args=(), kwargs={"messages": []}, model_id="vllm-test"
        )
    )

    assert captured["kwargs"]["stream"] is True
    assert captured["kwargs"]["stream_options"] == {"include_usage": True}

    # The aggregated response carries the usage chunk through.
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 2
    assert response.usage.total_tokens == 12


def test_stream_to_chat_completion_preserves_caller_stream_options():
    """If a caller already requested other stream_options, include_usage must
    be added without dropping the caller's keys."""

    captured: dict = {}

    async def fake_create(*args, **kwargs):
        captured["kwargs"] = kwargs
        return _async_iter([])

    asyncio.run(
        _stream_to_chat_completion(
            fake_create,
            args=(),
            kwargs={"messages": [], "stream_options": {"some_other_flag": True}},
            model_id="vllm-test",
        )
    )

    options = captured["kwargs"]["stream_options"]
    assert options["include_usage"] is True
    assert options["some_other_flag"] is True
