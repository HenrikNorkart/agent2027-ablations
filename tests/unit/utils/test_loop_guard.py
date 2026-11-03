"""ROGUE_ONE_LOOP_GUARD: stop a model that repeats one identical tool call."""
from rogueone.llm import clients


def _call(cmd):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": "generic_pandas_tool", "arguments": cmd}}]}


def _tool():
    return {"role": "tool", "tool_call_id": "x", "content": "same"}


def _kwargs(cmds):
    msgs = [{"role": "user", "content": "go"}]
    for c in cmds:
        msgs += [_call(c), _tool()]
    return {"messages": msgs, "tools": [{"type": "function", "function": {"name": "generic_pandas_tool"}}]}


def test_counts_only_the_latest_identical_run():
    assert clients._repeated_tool_calls(_kwargs(["a", "b", "b", "b"])["messages"]) == 3
    assert clients._repeated_tool_calls(_kwargs(["b", "b", "a"])["messages"]) == 1
    assert clients._repeated_tool_calls([{"role": "user", "content": "go"}]) == 0
    # a new round's prompt resets the count
    msgs = _kwargs(["a", "a", "a"])["messages"] + [{"role": "user", "content": "Round 2"}]
    assert clients._repeated_tool_calls(msgs) == 0
    assert clients._repeated_tool_calls(msgs + [_call("a"), _tool()]) == 1


def test_guard_removes_tools_after_n_repeats():
    kw = _kwargs(["a", "a", "a"])
    kw["tool_choice"] = "auto"
    n_before = len(kw["messages"])
    clients._apply_loop_guard(kw, 3)
    assert "tools" not in kw and "tool_choice" not in kw
    assert len(kw["messages"]) == n_before + 1 and kw["messages"][-1]["role"] == "user"


def test_guard_leaves_request_alone_below_threshold_or_off():
    for cmds, n in ((["a", "a"], 3), (["a", "a", "a"], 0), (["a", "b", "a"], 2)):
        kw = _kwargs(cmds)
        before = repr(kw)
        clients._apply_loop_guard(kw, n)
        assert repr(kw) == before
