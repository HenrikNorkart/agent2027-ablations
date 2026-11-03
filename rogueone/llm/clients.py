"""Custom OpenAI client factory for vLLM compatibility.

Provides a factory for creating AsyncOpenAI clients pointed at a local vLLM endpoint.
All RougeOne agents use OpenAIChatCompletionsModel which calls chat.completions.create,
so the wrapper below targets the correct API surface.
"""

import ast
import contextvars
import json
import logging
import re
import time
import uuid
from collections import defaultdict
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.completion_usage import (
    CompletionTokensDetails,
    CompletionUsage,
    PromptTokensDetails,
)

from rogueone.utils import instrumentation, rouge_one_cfg
from rogueone.utils.wandb_utils import log_llm_token_usage

logger = logging.getLogger(__name__)


_AGENT_INPUT_TOKENS: defaultdict[str, int] = defaultdict(int)
_AGENT_OUTPUT_TOKENS: defaultdict[str, int] = defaultdict(int)
_AGENT_TOTAL_TOKENS: defaultdict[str, int] = defaultdict(int)


def _resolve_text_tool_call(content: str, tools: list) -> tuple[str, dict] | None:
    """Parse a text-format tool call emitted by vLLM and return (tool_name, args_dict).

    vLLM's gpt-oss-20b emits tool calls as plain JSON text instead of using the
    structured tool_calls API.  Two observed formats:

    1. Envelope format (primary): {"name":"tool_name","arguments":{...args...}}
    2. Direct-args format (fallback): {"command":"..."} / {"query":"..."} etc.

    Returns None when the content is not parseable JSON or no tool name can be
    resolved.
    """
    content = content.strip()
    data = _extract_first_json_object(content)
    if data is None:
        # Some model profiles leak special header tokens. Try once more with them stripped.
        data = _extract_first_json_object(_strip_special_tokens(content))
    if not isinstance(data, dict):
        return None

    # --- Format 1: {"name":"<tool>","arguments":{...}} ---
    tool_name = _normalize_tool_name(data.get("name"), tools)
    arguments = data.get("arguments")
    if isinstance(arguments, str):
        parsed_arguments = _extract_first_json_object(arguments)
        arguments = parsed_arguments if parsed_arguments is not None else arguments

    if isinstance(tool_name, str) and isinstance(arguments, dict):
        return tool_name, arguments

    # --- Format 2: direct args dict — match by required-key/schema overlap ---
    arg_keys = set(data.keys())
    best_name: str | None = None
    best_score = -1

    for tool_def in tools:
        if tool_def.get("type") != "function":
            continue
        fn = tool_def["function"]
        schema = fn.get("parameters", {})
        required = set(schema.get("required", []))
        props = schema.get("properties", {})

        score = -1
        if required and required.issubset(arg_keys):
            score = len(required)

            # Prefer closer nested arg shape matches when the top-level contract is `{args: ...}`.
            if "args" in props and isinstance(props.get("args"), dict):
                nested = props["args"]
                nested_props = set((nested.get("properties") or {}).keys())
                nested_payload = data.get("args")
                if isinstance(nested_payload, dict) and nested_props:
                    score += len(set(nested_payload.keys()) & nested_props)
        elif required == {"args"} and "args" in props:
            # vLLM often emits only the inner payload without wrapping under `args`.
            nested = props["args"] if isinstance(props["args"], dict) else {}
            nested_props = set((nested.get("properties") or {}).keys())
            if nested_props:
                overlap = len(arg_keys & nested_props)
                if overlap > 0:
                    score = overlap

        if score < 0:
            continue
        if score > best_score:
            best_score = score
            best_name = fn["name"]

    if best_name is None:
        return None
    return best_name, data


def _inject_tool_call(response, tool_name: str, tool_args: dict):
    """Mutate a chat-completion response to carry a synthetic function tool call."""
    tc = ChatCompletionMessageToolCall(
        id=f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=Function(name=tool_name, arguments=json.dumps(tool_args)),
    )
    choice = response.choices[0]
    choice.message.tool_calls = [tc]
    choice.message.content = None
    choice.finish_reason = "tool_calls"
    return response


def _normalize_tool_arguments(tool_name: str, tool_args: dict, tools: list) -> dict:
    """Adjust extracted args to match the declared JSON schema for the selected tool.

    Many @function_tool handlers in RougeOne use a single parameter named "args"
    that contains the real payload object.  vLLM text envelopes typically emit just
    the inner payload, so we wrap it when the tool schema requires top-level "args".
    """
    for tool_def in tools:
        if tool_def.get("type") != "function":
            continue
        fn = tool_def["function"]
        if fn.get("name") != tool_name:
            continue
        required = set(fn.get("parameters", {}).get("required", []))
        if "args" in required and "args" not in tool_args:
            return {"args": tool_args}
        return tool_args
    return tool_args


def _extract_first_json_object(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _strip_special_tokens(text: str) -> str:
    cleaned = re.sub(r"<\|[^|]+\|>", " ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def _tool_names(tools: list) -> list[str]:
    return [
        t["function"]["name"]
        for t in tools
        if t.get("type") == "function" and "function" in t and "name" in t["function"]
    ]


def _normalize_tool_name(raw_name: Any, tools: list) -> str | None:
    if not isinstance(raw_name, str):
        return None

    known = _tool_names(tools)
    if raw_name in known:
        return raw_name

    cleaned = _strip_special_tokens(raw_name)
    cleaned = re.sub(r"[^A-Za-z0-9_ ]", " ", cleaned).strip()
    candidate = cleaned.split()[0] if cleaned else ""

    if candidate in known:
        return candidate

    lower_raw = raw_name.lower()
    lower_candidate = candidate.lower()
    for name in known:
        lower_name = name.lower()
        if lower_candidate and (
            lower_candidate.startswith(lower_name) or lower_name.startswith(lower_candidate)
        ):
            return name
        if lower_name in lower_raw:
            return name
    return None


#: E9a restart seed rule (AGENT 2027 A10-T5): an offset added to ``ROGUE_ONE_LLM_SEED``.
#: Only ``forward_pass`` under ``ROGUE_ONE_RESTART_MODE=every_iteration`` sets it (to the
#: global trial index); everywhere else it stays 0, so the seed sent is unchanged.
_LLM_SEED_OFFSET: contextvars.ContextVar[int] = contextvars.ContextVar(
    "rogueone_llm_seed_offset", default=0
)


def set_llm_seed_offset(offset: int) -> contextvars.Token:
    """Set the seed offset for this context; returns the token for :func:`reset_llm_seed_offset`."""
    return _LLM_SEED_OFFSET.set(int(offset))


def reset_llm_seed_offset(token: contextvars.Token) -> None:
    _LLM_SEED_OFFSET.reset(token)


def current_llm_seed_offset() -> int:
    return _LLM_SEED_OFFSET.get()


def _merge_extra_body_for_vllm(kwargs: dict[str, Any]) -> None:
    existing = kwargs.get("extra_body")
    extra_body = dict(existing) if isinstance(existing, dict) else {}
    template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
    template_kwargs.setdefault("enable_thinking", False)
    extra_body["chat_template_kwargs"] = template_kwargs
    kwargs["extra_body"] = extra_body
    # Seed the sampler here rather than in each agent's ModelSettings: every request
    # path (streaming, non-streaming, and the retry paths) funnels through this
    # function, so one place covers all three agents. An explicit caller-supplied seed
    # always wins.
    seed = rouge_one_cfg.llm_seed
    if seed is not None and kwargs.get("seed") is None:
        kwargs["seed"] = int(seed) + _LLM_SEED_OFFSET.get()


def _inline_schema_refs(schema: Any) -> Any:
    """Return ``schema`` with every local ``#/$defs/<name>`` reference replaced by its definition.

    vLLM's qwen3_coder tool parser types each parameter from its own schema node and does
    not follow ``$ref``, so a parameter declared as ``{"$ref": "#/$defs/X"}`` is parsed as a
    string. Inlining gives it ``"type": "object"``. Sibling keys of a ``$ref`` (for example
    ``description``) are kept; recursive definitions are left as a ``$ref`` after one level.
    """
    if not isinstance(schema, dict):
        return schema
    defs = schema.get("$defs") or {}

    def resolve(node: Any, seen: tuple[str, ...]) -> Any:
        if isinstance(node, list):
            return [resolve(v, seen) for v in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref[len("#/$defs/"):]
            if name in defs and name not in seen:
                merged = dict(resolve(defs[name], seen + (name,)))
                merged.update({k: resolve(v, seen) for k, v in node.items() if k != "$ref"})
                return merged
        return {k: resolve(v, seen) for k, v in node.items() if k != "$defs"}

    return resolve(schema, ())


def _inline_tool_schemas(kwargs: dict[str, Any]) -> None:
    """Apply :func:`_inline_schema_refs` to the ``parameters`` of every function tool in ``kwargs``."""
    tools = kwargs.get("tools")
    if not tools:
        return
    out = []
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(fn, dict) and isinstance(fn.get("parameters"), dict) and "$defs" in fn["parameters"]:
            tool = {**tool, "function": {**fn, "parameters": _inline_schema_refs(fn["parameters"])}}
        out.append(tool)
    kwargs["tools"] = out


LOOP_GUARD_NOTE = (
    "You have made exactly the same tool call {n} times in a row and it returns the same "
    "result every time. Do not call any tool again. Use what you already have and give "
    "your final answer now."
)


def _repeated_tool_calls(messages: Any) -> int:
    """Number of consecutive, most recent assistant turns whose tool calls are identical.

    Counting stops at the latest user message: a new round's prompt starts afresh, even
    when the session history ends in a loop from the previous round (smoke 4).
    """
    if not isinstance(messages, list):
        return 0
    last = None
    count = 0
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user":
            break
        if msg.get("role") != "assistant":
            continue
        calls = msg.get("tool_calls")
        if not calls:
            break
        key = json.dumps(
            [(c.get("function") or {}) for c in calls if isinstance(c, dict)], sort_keys=True
        )
        if last is not None and key != last:
            break
        last = key
        count += 1
    return count


def _apply_loop_guard(kwargs: dict[str, Any], repeats: int) -> None:
    """After ``repeats`` identical tool calls, ask for an answer with no tools offered.

    The tools are removed rather than sent with tool_choice="none": vLLM's qwen3_coder
    path still returned tool calls under "none" (A17 Q4 smoke 3, 31 guarded repeats).
    """
    if repeats <= 0 or not kwargs.get("tools"):
        return
    n = _repeated_tool_calls(kwargs.get("messages"))
    if n < repeats:
        return
    for key in ("tools", "tool_choice", "parallel_tool_calls"):
        kwargs.pop(key, None)
    kwargs["messages"] = list(kwargs["messages"]) + [
        {"role": "user", "content": LOOP_GUARD_NOTE.format(n=n)}
    ]
    instrumentation.count("loop_guard")


def _is_retryable_vllm_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "unexpected tokens remaining in message header" in msg
        or "validation errors for validatoriterator" in msg
    )


def _is_empty_tool_response(choice: Any) -> bool:
    """Detect responses where vLLM failed to emit tool calls in non-streaming mode.

    vLLM's non-streaming path sometimes drops tool calls and instead returns
    finish_reason='stop' with empty or near-empty content (e.g. '{}', '{"": {}}',
    or blank).  This helper identifies those responses so the caller can retry
    via streaming.
    """
    if getattr(choice, "finish_reason", None) != "stop":
        return False
    if choice.message.tool_calls:
        return False
    content = (choice.message.content or "").strip()
    if not content:
        return True
    parsed = _extract_first_json_object(content)
    if parsed is not None and not any(
        v for v in parsed.values() if v and v != {}
    ):
        return True
    return False


# --- Finding 18: detecting tool-call arguments damaged in transit -------------
#
# vLLM 0.11.0's harmony streaming path drops whole model tokens out of the middle
# of a tool call's `arguments` string (see T-1.6 in the camera-ready plan). The
# aggregator below concatenates exactly what it is handed, so it cannot repair the
# damage -- by the time the deltas arrive the tokens are already gone. What we can
# do is *notice*: a span deleted mid-token leaves the generated Python with a
# string or bracket opened and never closed, which produces a small, distinctive
# set of SyntaxError messages that a model writing bad-but-complete code
# essentially never produces.
#
# Keep this list in sync with SPLICE_PATTERNS in
# paper/scripts/audit_extractor_corruption.py, which applies the same
# classification to historical runs.
_SPLICE_SYNTAX_MARKERS = (
    "unterminated string literal",
    "unterminated triple-quoted string literal",
    "was never closed",
    "unexpected EOF while parsing",
    "invalid syntax. Perhaps you forgot a comma",
)

# Tool-argument fields whose value is Python source we can validate by parsing.
_CODE_ARGUMENT_FIELDS = ("command", "code")


def _is_spliced_python(source: str) -> bool:
    """True if ``source`` fails to parse in the specific way a mid-token cut does."""
    try:
        ast.parse(source)
    except SyntaxError as exc:
        message = str(getattr(exc, "msg", "") or exc)
        return any(marker in message for marker in _SPLICE_SYNTAX_MARKERS)
    except Exception:
        # ValueError on embedded NULs and friends: not the splice signature.
        return False
    return False


def _tool_call_was_spliced(arguments: str) -> bool:
    """True if a tool call's raw ``arguments`` show the in-transit splice signature.

    Two shapes count. Usually the deleted span sits inside a JSON string value, so
    the JSON still parses and only the Python inside it is broken. Occasionally the
    span takes a quote or brace with it and the JSON itself will not parse -- also
    damage, and also not something the model does on its own, because the arguments
    are emitted by a constrained serializer rather than written freehand.
    """
    if not arguments:
        return False
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return True
    if not isinstance(parsed, dict):
        return False
    for field in _CODE_ARGUMENT_FIELDS:
        value = parsed.get(field)
        if isinstance(value, str) and value.strip() and _is_spliced_python(value):
            return True
    return False


def _stream_usage_dict(usage: Any) -> dict[str, Any]:
    """Usage ints from a stream usage chunk, keeping reasoning/cached tokens (E3).

    ``reasoning_tokens`` / ``cached_tokens`` are None when the server reports no
    details, so that the rebuilt ``CompletionUsage`` stays identical to the old one.
    """
    comp_details = getattr(usage, "completion_tokens_details", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    return {
        "prompt_tokens": usage.prompt_tokens or 0,
        "completion_tokens": usage.completion_tokens or 0,
        "total_tokens": usage.total_tokens or 0,
        "reasoning_tokens": getattr(comp_details, "reasoning_tokens", None),
        "cached_tokens": getattr(prompt_details, "cached_tokens", None),
    }


async def _stream_to_chat_completion(
    create_fn: Any,
    args: tuple,
    kwargs: dict[str, Any],
    model_id: str,
) -> ChatCompletion:
    """Call the chat completions API in streaming mode and aggregate into a ChatCompletion.

    vLLM's tool parser works correctly in streaming mode but sometimes fails in
    non-streaming mode, dropping tool_calls from the response.  This function
    transparently streams and reconstructs the non-streaming response shape.
    """
    stream_kwargs = dict(kwargs)
    stream_kwargs["stream"] = True
    # Without include_usage the upstream stream never yields a usage chunk, so
    # `usage_data` stays None and every token-usage record we emit downstream
    # is (0,0,0). That's why all CAR runs show LLM_tokens/* == 0 in W&B
    # despite hundreds of model calls — see runs 71g183ic / o5zyks6e /
    # m4i2r4v9. Merge with any existing stream_options instead of clobbering.
    existing_stream_options = stream_kwargs.get("stream_options")
    merged_stream_options = (
        dict(existing_stream_options)
        if isinstance(existing_stream_options, dict)
        else {}
    )
    merged_stream_options.setdefault("include_usage", True)
    stream_kwargs["stream_options"] = merged_stream_options

    stream = await create_fn(*args, **stream_kwargs)

    content_parts: list[str] = []
    tool_calls_data: dict[int, dict[str, str]] = {}
    finish_reason: str | None = None
    usage_data: dict[str, Any] | None = None
    reasoning_chars = 0

    async for chunk in stream:
        if not chunk.choices:
            # Usage-only chunk at the end of the stream
            if getattr(chunk, "usage", None):
                usage_data = _stream_usage_dict(chunk.usage)
            continue

        delta = chunk.choices[0].delta
        reasoning_chars += len(
            getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None) or ""
        )
        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason

        if delta.content:
            content_parts.append(delta.content)

        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tool_calls_data:
                    tool_calls_data[idx] = {"id": "", "name": "", "arguments": ""}
                if tc_delta.id:
                    tool_calls_data[idx]["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        tool_calls_data[idx]["name"] = tc_delta.function.name
                    if tc_delta.function.arguments:
                        tool_calls_data[idx]["arguments"] += tc_delta.function.arguments

        if getattr(chunk, "usage", None):
            usage_data = _stream_usage_dict(chunk.usage)

    instrumentation.note_stream_extras(reasoning_chars=reasoning_chars)

    # Build tool_calls list
    assembled_tool_calls: list[ChatCompletionMessageToolCall] | None = None
    if tool_calls_data:
        assembled_tool_calls = []
        for idx in sorted(tool_calls_data):
            tc = tool_calls_data[idx]
            assembled_tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=tc["id"] or f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=Function(
                        name=tc["name"],
                        arguments=tc["arguments"],
                    ),
                )
            )

    content = "".join(content_parts) if content_parts else None
    # If we got tool calls, set content to None and finish_reason to tool_calls
    if assembled_tool_calls:
        content = None
        finish_reason = "tool_calls"

    message = ChatCompletionMessage(
        role="assistant",
        content=content,
        tool_calls=assembled_tool_calls,
    )

    usage = CompletionUsage(
        prompt_tokens=usage_data["prompt_tokens"] if usage_data else 0,
        completion_tokens=usage_data["completion_tokens"] if usage_data else 0,
        total_tokens=usage_data["total_tokens"] if usage_data else 0,
    ) if usage_data else CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    if usage_data and usage_data.get("reasoning_tokens") is not None:
        usage.completion_tokens_details = CompletionTokensDetails(
            reasoning_tokens=usage_data["reasoning_tokens"]
        )
    if usage_data and usage_data.get("cached_tokens") is not None:
        usage.prompt_tokens_details = PromptTokensDetails(
            cached_tokens=usage_data["cached_tokens"]
        )

    return ChatCompletion(
        id=f"chatcmpl-stream-{uuid.uuid4().hex[:12]}",
        choices=[
            Choice(
                finish_reason=finish_reason or "stop",
                index=0,
                message=message,
            )
        ],
        created=int(time.time()),
        model=model_id,
        object="chat.completion",
        usage=usage,
    )


def _sanitize_messages_for_retry(messages: Any, tools: list) -> Any:
    if not isinstance(messages, list):
        return messages

    sanitized: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            sanitized.append(message)
            continue

        msg = dict(message)
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = _strip_special_tokens(content)

        if isinstance(msg.get("tool_calls"), list):
            fixed_tool_calls = []
            for tool_call in msg["tool_calls"]:
                if not isinstance(tool_call, dict):
                    fixed_tool_calls.append(tool_call)
                    continue
                tc = dict(tool_call)
                fn = dict(tc.get("function") or {})
                normalized_name = _normalize_tool_name(fn.get("name"), tools)
                if normalized_name is not None:
                    fn["name"] = normalized_name
                arguments = fn.get("arguments")
                if arguments is not None and not isinstance(arguments, str):
                    fn["arguments"] = json.dumps(arguments)
                tc["function"] = fn
                fixed_tool_calls.append(tc)
            msg["tool_calls"] = fixed_tool_calls

        sanitized.append(msg)
    return sanitized


class VLLMCompatibleAsyncOpenAI(AsyncOpenAI):
    """AsyncOpenAI subclass that strips unsupported parameters for vLLM chat completions
    and transparently converts text-format tool calls to proper API tool_calls.

    vLLM's gpt-oss-20b emits tool calls as plain JSON text rather than using the OpenAI
    function-calling protocol.  The wrapper detects those responses and injects synthetic
    tool_calls entries so the Agents SDK Runner can execute them normally.
    """

    def __init__(self, *args: Any, agent_name: str = "unknown_agent", **kwargs: Any):
        self._agent_name = agent_name
        super().__init__(*args, **kwargs)
        self._wrap_completions()

    def _extract_token_usage(self, response: Any) -> tuple[int, int, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0, 0, 0

        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

        input_tokens = int(prompt_tokens or 0)
        output_tokens = int(completion_tokens or 0)
        total = int(total_tokens or (input_tokens + output_tokens))
        return input_tokens, output_tokens, total

    def _record_token_usage(self, response: Any) -> None:
        input_tokens, output_tokens, total_tokens = self._extract_token_usage(response)

        _AGENT_INPUT_TOKENS[self._agent_name] += input_tokens
        _AGENT_OUTPUT_TOKENS[self._agent_name] += output_tokens
        _AGENT_TOTAL_TOKENS[self._agent_name] += total_tokens

        log_llm_token_usage(
            agent_name=self._agent_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cumulative_input_tokens=_AGENT_INPUT_TOKENS[self._agent_name],
            cumulative_output_tokens=_AGENT_OUTPUT_TOKENS[self._agent_name],
            cumulative_total_tokens=_AGENT_TOTAL_TOKENS[self._agent_name],
        )

    def _wrap_completions(self) -> None:
        original_create = self.chat.completions.create

        async def _create_untraced(*args: Any, **kwargs: Any) -> Any:
            # Strip parameters that vLLM's gpt-oss models do not support:
            # - logprobs / top_logprobs: returned as null, causing Pydantic errors
            # - reasoning_effort: causes text-format tool calls on complex prompts
            kwargs.pop("logprobs", None)
            kwargs.pop("top_logprobs", None)
            if not rouge_one_cfg.reasoning_effort_passthrough:
                kwargs.pop("reasoning_effort", None)
            if rouge_one_cfg.inline_tool_schema_refs:
                _inline_tool_schemas(kwargs)
            if rouge_one_cfg.loop_guard_repeats:
                _apply_loop_guard(kwargs, rouge_one_cfg.loop_guard_repeats)

            # Don't interfere with caller-requested streaming.
            if kwargs.get("stream"):
                _merge_extra_body_for_vllm(kwargs)
                instrumentation.note_request(kwargs)
                return await original_create(*args, **kwargs)

            _merge_extra_body_for_vllm(kwargs)
            instrumentation.note_request(kwargs)
            tools = kwargs.get("tools") or []
            model_id = kwargs.get("model") or "unknown"

            # When tools are present, use streaming internally.
            # vLLM's non-streaming path drops tool_calls for some model
            # profiles, while the streaming path processes them correctly.
            if tools:
                response = await self._streaming_create_with_fallback(
                    original_create, args, kwargs, tools, model_id,
                )
            else:
                try:
                    response = await original_create(*args, **kwargs)
                except Exception as exc:
                    if not _is_retryable_vllm_error(exc):
                        raise
                    instrumentation.count("retryable_vllm_error_retries")
                    retry_kwargs = dict(kwargs)
                    retry_kwargs["messages"] = _sanitize_messages_for_retry(
                        retry_kwargs.get("messages"), tools,
                    )
                    if isinstance(retry_kwargs.get("temperature"), (float, int)):
                        retry_kwargs["temperature"] = min(
                            float(retry_kwargs["temperature"]), 0.2,
                        )
                    response = await original_create(*args, **retry_kwargs)

            self._record_token_usage(response)
            return response

        async def _create(*args: Any, **kwargs: Any) -> Any:
            # E3: one metered (and, with ROGUE_ONE_LLM_TRACE=1, traced) record per call.
            return await instrumentation.traced_call(
                self._agent_name, _create_untraced, args, kwargs,
            )

        self.chat.completions.create = _create

    async def _streaming_create_with_fallback(
        self,
        original_create: Any,
        args: tuple,
        kwargs: dict[str, Any],
        tools: list,
        model_id: str,
        _max_empty_retries: int = 3,
        _max_splice_retries: int = 2,
    ) -> Any:
        """Use streaming to obtain a response when tools are present.

        Attempts streaming first (which preserves tool calls on vLLM).
        If the stream yields no tool calls and an empty/stub response, retries
        up to ``_max_empty_retries`` times before giving up.  Text-format tool
        call detection is applied as a final safety net.
        """
        last_response: ChatCompletion | None = None
        splice_retries = 0

        for attempt in range(_max_empty_retries + _max_splice_retries + 1):
            try:
                response = await _stream_to_chat_completion(
                    original_create, args, kwargs, model_id,
                )
            except Exception as exc:
                if not _is_retryable_vllm_error(exc):
                    raise
                instrumentation.count("retryable_vllm_error_retries")
                retry_kwargs = dict(kwargs)
                retry_kwargs["messages"] = _sanitize_messages_for_retry(
                    retry_kwargs.get("messages"), tools,
                )
                if isinstance(retry_kwargs.get("temperature"), (float, int)):
                    retry_kwargs["temperature"] = min(
                        float(retry_kwargs["temperature"]), 0.2,
                    )
                response = await _stream_to_chat_completion(
                    original_create, args, retry_kwargs, model_id,
                )

            last_response = response
            instrumentation.note_attempt(response)  # E3: every attempt is metered
            choice = response.choices[0]

            # If we got tool calls or meaningful content, accept the response --
            # unless the arguments arrived damaged. vLLM 0.11.0's harmony streaming
            # path silently deletes whole tokens from the middle of a tool call's
            # arguments (Finding 18), which historically destroyed 25-50% of the
            # Extractor's proposals: the call reached the tool, raised SyntaxError,
            # and cost a proposal out of the iteration budget. The loss is
            # length-biased -- longer generated code offers more to lose -- so it
            # falls hardest on exactly the multi-statement features the method
            # exists to produce. We cannot repair a spliced string (the tokens are
            # gone before we see them), but re-rolling the completion is cheap and
            # removes the bias.
            if choice.message.tool_calls:
                spliced = [
                    tc for tc in choice.message.tool_calls
                    if _tool_call_was_spliced(getattr(tc.function, "arguments", "") or "")
                ]
                if spliced and splice_retries < _max_splice_retries:
                    splice_retries += 1
                    logger.warning(
                        "Discarding %d/%d tool call(s) with in-transit splice damage "
                        "on attempt %d; re-rolling completion (%d/%d)",
                        len(spliced), len(choice.message.tool_calls), attempt + 1,
                        splice_retries, _max_splice_retries,
                    )
                    instrumentation.mark_discard("splice")
                    continue
                if spliced:
                    # Out of re-rolls. Hand the damaged call on rather than dropping
                    # the turn: the tool will raise SyntaxError and the agent retries,
                    # which is the pre-fix behaviour and strictly no worse.
                    logger.warning(
                        "Tool call still spliced after %d re-rolls; passing it through",
                        splice_retries,
                    )
                break
            if not _is_empty_tool_response(choice):
                break

            # Empty response with tools present — model failed to emit a tool
            # call.  Retry with the same parameters.
            instrumentation.mark_discard("empty")
            logger.debug(
                "vLLM returned empty response on attempt %d/%d, retrying",
                attempt + 1,
                _max_empty_retries + 1,
            )

        response = last_response  # type: ignore[assignment]
        choice = response.choices[0]

        # Normalise tool names that may be garbled by the model.
        if choice.message.tool_calls:
            for tool_call in choice.message.tool_calls:
                fn = getattr(tool_call, "function", None)
                if fn is None:
                    continue
                normalized_name = _normalize_tool_name(fn.name, tools)
                if normalized_name is not None:
                    fn.name = normalized_name
                # Wrap bare args for tools that expect an "args" envelope.
                try:
                    parsed_args = json.loads(fn.arguments)
                    if isinstance(parsed_args, dict):
                        normalized = _normalize_tool_arguments(
                            fn.name, parsed_args, tools,
                        )
                        fn.arguments = json.dumps(normalized)
                except (json.JSONDecodeError, TypeError):
                    pass

        # Text-format fallback: if streaming still didn't produce tool_calls.
        if (
            choice.finish_reason == "stop"
            and not choice.message.tool_calls
            and choice.message.content
        ):
            match = _resolve_text_tool_call(choice.message.content, tools)
            if match:
                instrumentation.note_rescue(
                    "text_tool_call", raw_content=choice.message.content, model_id=model_id,
                )
                tool_name, tool_args = match
                normalized_args = _normalize_tool_arguments(
                    tool_name, tool_args, tools,
                )
                response = _inject_tool_call(response, tool_name, normalized_args)

        return response


def create_openai_client_for_vllm(
    base_url: str,
    api_key: str,
    agent_name: str = "unknown_agent",
    **kwargs: Any,
) -> VLLMCompatibleAsyncOpenAI:
    """Create a vLLM-compatible async OpenAI client for use with chat-completions agents.

    Args:
        base_url: The vLLM endpoint (e.g., "http://localhost:8000/v1").
        api_key: API key for authentication.
        agent_name: Logical name for token accounting/logging.
        **kwargs: Extra arguments forwarded to VLLMCompatibleAsyncOpenAI.

    Returns:
        VLLMCompatibleAsyncOpenAI configured for the given endpoint.
    """
    return VLLMCompatibleAsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        agent_name=agent_name,
        **kwargs,
    )
