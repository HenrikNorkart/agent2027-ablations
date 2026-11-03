"""Unit tests for Finding 18: tool-call arguments damaged in transit.

Background (T-1.6 in paper/reviewer_feedback/CAMERA_READY_EXPERIMENT_PLAN.md).
`audit_extractor_corruption.py` found the splice signature in 100 of 102 completed
run directories, 7,131 events, destroying 25-50% of the Extractor's attribute
proposals in every reported wave. `analyze_splice_signature.py` then recovered 16
exact deleted spans by re-inserting a candidate span into the damaged statement and
re-parsing: the spans are short, contiguous, and land on model-token boundaries
(`('` seven times, always straight after `groupby`), never on the JSON-closing
punctuation a bad partial-JSON diff would produce.

These tests pin down what each layer can and cannot be responsible for:

  * `_stream_to_chat_completion` concatenates deltas faithfully, including across a
    multi-call burst -- so it does not manufacture damage, and
  * a delta dropped upstream reproduces the observed corruption exactly, which our
    layer cannot repair, only detect.
"""

from __future__ import annotations

import asyncio
from itertools import zip_longest

import pytest

from rogueone.llm.clients import (
    _is_spliced_python,
    _stream_to_chat_completion,
    _tool_call_was_spliced,
)

pytestmark = [pytest.mark.unit]


class _FnDelta:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class _ToolCallDelta:
    def __init__(self, index, name=None, arguments=None, id=None):
        self.index = index
        self.id = id
        self.function = _FnDelta(name=name, arguments=arguments)


class _Choice:
    class _Delta:
        def __init__(self, content=None, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls

    def __init__(self, content=None, tool_calls=None, finish_reason=None):
        self.delta = _Choice._Delta(content=content, tool_calls=tool_calls)
        self.finish_reason = finish_reason


class _Chunk:
    def __init__(self, *, choices=None, usage=None):
        self.choices = choices or []
        if usage is not None:
            self.usage = usage


async def _aiter(items):
    for item in items:
        yield item


def _run_stream(chunks):
    async def fake_create(*args, **kwargs):
        return _aiter(chunks)

    return asyncio.run(
        _stream_to_chat_completion(
            fake_create, args=(), kwargs={"messages": []}, model_id="vllm-test"
        )
    )


def _arg_chunks(index, pieces, name="append_new_attribute"):
    chunks = [_Chunk(choices=[_Choice(
        tool_calls=[_ToolCallDelta(index, name=name, id=f"call_{index}")])])]
    chunks += [_Chunk(choices=[_Choice(
        tool_calls=[_ToolCallDelta(index, arguments=p)])]) for p in pieces]
    return chunks


# --- the aggregator is a faithful concatenator -------------------------------

def test_aggregator_reassembles_arguments_exactly():
    """Whatever vLLM sends, we must return byte-identical concatenation."""
    intended = '{"name": "w", "command": "w = df.groupby(\'id\')[\'Weight\'].first()"}'
    pieces = [intended[i:i + 3] for i in range(0, len(intended), 3)]

    response = _run_stream(_arg_chunks(0, pieces))

    calls = response.choices[0].message.tool_calls
    assert len(calls) == 1
    assert calls[0].function.arguments == intended
    assert not _tool_call_was_spliced(calls[0].function.arguments)


def test_aggregator_keeps_a_multi_call_burst_separate():
    """The Extractor emits several append_new_attribute calls per completion.
    Arguments must not bleed between indices, and order must be preserved."""
    first = '{"name": "a", "command": "result = df.groupby(\'id\')[\'A\'].first()"}'
    second = '{"name": "b", "command": "result = df.groupby(\'id\')[\'B\'].mean()"}'

    chunks = []
    # Interleave the two calls to make index handling do real work. zip_longest, not
    # zip: the two argument strings differ in length, so plain zip drops the tail
    # chunk of the longer one and the test then asserts against a truncated
    # expectation -- which is indistinguishable from the very splice under test.
    for pair in zip_longest(
        _arg_chunks(0, [first[i:i + 5] for i in range(0, len(first), 5)]),
        _arg_chunks(1, [second[i:i + 5] for i in range(0, len(second), 5)]),
    ):
        chunks.extend([chunk for chunk in pair if chunk is not None])

    response = _run_stream(chunks)
    calls = response.choices[0].message.tool_calls

    assert [c.function.arguments for c in calls] == [first, second]
    assert all(not _tool_call_was_spliced(c.function.arguments) for c in calls)


# --- a dropped upstream delta reproduces the observed damage -----------------

def test_dropped_delta_reproduces_the_observed_splice():
    """Deleting one delta from the stream yields exactly the corruption recorded
    in paper/extractor_corruption_audit.csv -- `df.groupbyid')` from
    `df.groupby('id')`, a two-character, token-aligned, mid-string deletion whose
    suffix is intact. This is what a token-budget truncation cannot produce, and
    it is why the fault was misread as truncation for two sessions."""
    pieces = ['{"name": "w", "command": "w = df.groupby',
              "('",                       # the token vLLM drops
              "id')['Weight'].first()\"}"]

    intact = _run_stream(_arg_chunks(0, pieces))
    lossy = _run_stream(_arg_chunks(0, [p for p in pieces if p != "('"]))

    assert not _tool_call_was_spliced(intact.choices[0].message.tool_calls[0]
                                      .function.arguments)

    damaged = lossy.choices[0].message.tool_calls[0].function.arguments
    assert "df.groupbyid')" in damaged
    # The JSON is still well-formed: only the Python inside it is destroyed, which
    # is why nothing in the pipeline noticed this for the whole life of the project.
    assert _tool_call_was_spliced(damaged)


# --- the detector's precision ------------------------------------------------

@pytest.mark.parametrize("code", [
    "w = df.groupbyid')['Weight'].first()",          # deleted "('"
    "s = df.groupby')['QRSduration'].first()",       # deleted "('id"
    "r = df_features['id.map(df.groupby('id')['x'])", # deleted "']."
])
def test_detector_flags_spliced_code(code):
    assert _is_spliced_python(code)
    assert _tool_call_was_spliced('{"name": "f", "command": %s}' % _json(code))


@pytest.mark.parametrize("code", [
    "result = df.groupby('id')['Weight'].first()",
    "result = df_features['id'].map(m).fillna(0)",
    "result = df.no_such_method('id')",              # runtime error, not a splice
    "result = df.groupby('id')[  'x' ].first( )",
])
def test_detector_ignores_healthy_code(code):
    assert not _is_spliced_python(code)
    assert not _tool_call_was_spliced('{"name": "f", "command": %s}' % _json(code))


def test_detector_ignores_ordinary_model_syntax_errors():
    """A model writing bad-but-complete code must not be counted as damage --
    otherwise the fix would re-roll completions for the model's own mistakes."""
    assert not _is_spliced_python("result = df.groupby('id')['x'].first() +")
    assert not _is_spliced_python("for x in df:\nresult = x")


def test_unparseable_json_counts_as_damage():
    """Sometimes the deleted span takes a quote with it and the JSON itself breaks."""
    assert _tool_call_was_spliced('{"name": "w", "command": "w = df.groupby(')


def _json(s):
    import json
    return json.dumps(s)
