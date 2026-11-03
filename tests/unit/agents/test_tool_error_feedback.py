"""E2: ROGUE_ONE_TOOL_ERROR_FEEDBACK return sites, bounded tool output, extractor rounds.

Tools are exercised by running the real agent method with a fake ``Runner`` that
invokes the tool callables the agent built.
"""

import asyncio
import collections
import json
import os
from pathlib import Path
from types import SimpleNamespace

for _k, _v in {"EMBEDD_API_KEY": "x", "EMBEDD_MODEL": "x", "EMBEDD_PORT": "1", "LLM_MODEL": "x"}.items():
    os.environ.setdefault(_k, _v)

import pandas as pd
import pytest
from agents.tool_context import ToolContext

import rogueone.llm.agents.core.extractor.agent as ext_mod
import rogueone.llm.agents.core.scientist.agent as sci_mod
from rogueone.utils import console_cfg, rouge_one_cfg
from rogueone.utils.console import ConsoleManager, tool_feedback, truncate_tool_output

pytestmark = [pytest.mark.unit, pytest.mark.agent]

_FLAGS = ("tool_error_feedback", "tool_output_max_chars", "extractor_rounds")


@pytest.fixture(autouse=True)
def _restore_flags():
    saved = {k: getattr(rouge_one_cfg, k) for k in _FLAGS}
    saved_disable = console_cfg.disable
    yield
    for k, v in saved.items():
        setattr(rouge_one_cfg, k, v)
    console_cfg.disable = saved_disable


def _async_value(v):
    async def f(*a, **k):
        return v

    return f


def _cfg(tmp_path):
    return SimpleNamespace(
        output_dir=Path(tmp_path),
        experiment_description="d",
        context_and_goal="g",
        experiment_name="e",
        task_type="classification",
        modality="tabular",
    )


def _tools_by_name(agent):
    return {t.name: t for t in agent.tools}


async def _inv(tools, name, **kw):
    payload = json.dumps(kw)
    ctx = ToolContext(
        context=None, tool_name=name, tool_call_id="call_1", tool_arguments=payload
    )
    return await tools[name].on_invoke_tool(ctx, payload)


def _counters(agent):
    for n in ("_tool_call_counter", "_tool_call_counter_success", "_tool_call_counter_failed"):
        setattr(agent, n, collections.defaultdict(int))
    agent._tool_error_reporter = None


# --------------------------------------------------------------------------- console


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("mode", ["message", "none"])
def test_console_returns_message_and_flag_decides(disabled, mode):
    console_cfg.disable = disabled
    rouge_one_cfg.tool_error_feedback = mode
    assert ConsoleManager.console_print("hello") == "hello"
    assert ConsoleManager.console_error_print("bad", code="x = 1") == "bad"
    expected = None if mode == "none" else "bad"
    assert tool_feedback(ConsoleManager.console_error_print("bad")) == expected


def test_truncate_tool_output():
    big = "x" * 100_000
    assert truncate_tool_output(big, 0) == big
    assert truncate_tool_output("short", 10) == "short"
    cut = truncate_tool_output(big, 8000)
    assert cut.startswith("x" * 8000) and not cut.startswith("x" * 8001)
    assert "truncated" in cut and len(cut) < 8300
    rouge_one_cfg.tool_output_max_chars = 50
    assert "truncated" in truncate_tool_output(big)


# --------------------------------------------------------------------------- extractor


def _make_extractor(tmp_path):
    agent = ext_mod.ExtractorAgent.__new__(ext_mod.ExtractorAgent)
    agent.cfg_experiment = _cfg(tmp_path)
    agent.knowledge_agent = SimpleNamespace(explain_query=_async_value("   "))
    _counters(agent)
    return agent


def _run_extractor(agent, monkeypatch, scenario):
    """Run generate_patient_attributes with a fake Runner."""
    state = {"calls": 0}

    class FakeRunner:
        async def run(self, a, *args, **kwargs):
            state["calls"] += 1
            if state["calls"] == 1 and scenario is not None:
                try:
                    state["results"] = await scenario(_tools_by_name(a))
                except BaseException as e:  # the agent swallows Runner errors
                    state["error"] = repr(e)
                    raise
            return SimpleNamespace(final_output="")

    monkeypatch.setattr(ext_mod, "Runner", FakeRunner)
    df = pd.DataFrame({"id": [1, 1, 2, 2], "v": [1.0, 2.0, 3.0, 4.0]})
    feats = pd.DataFrame({"id": [1, 2], "known": [0.0, 1.0]})
    expl = pd.DataFrame({"Attribute": ["known"], "Status": ["Active"]})
    asyncio.run(agent.generate_patient_attributes(df, feats, expl, "focus", step=0))
    return state


async def _extractor_sites(tools):
    async def pandas(cmd):
        return await _inv(tools, "generic_pandas_tool", args={"command": cmd, "explanation": "e"})

    async def append(name, cmd):
        return await _inv(
            tools, "append_new_attribute",
            args={"name": name, "command": cmd, "argumentation": "a"},
        )

    return {
        "h5_empty_search": await _inv(tools, "search_in_literature_tool", query="q"),
        "h6_illegal": await pandas("x = 1"),
        "h7_no_result": await pandas("y = df.shape"),
        "h4_exception": await pandas("result = df['nope']"),
        "h8_no_result": await append("n1", "z = df.groupby('id')['v'].mean()"),
        "h8_nan": await append("n2", "result = df_features['id'].map({1: float('nan'), 2: 1.0}).values"),
        "h8_nonnumeric": await append("n3", "result = df_features['id'].astype(str).values"),
        "v1_duplicate": await append("known", "result = df_features['id'].values"),
    }


H_SITES = [
    "h5_empty_search",
    "h6_illegal",
    "h7_no_result",
    "h4_exception",
    "h8_no_result",
    "h8_nan",
    "h8_nonnumeric",
]
DUP = "Attribute known already exists and is active."


@pytest.mark.parametrize("disabled", [False, True])
def test_extractor_sites_message_mode(tmp_path, monkeypatch, disabled):
    console_cfg.disable = disabled
    rouge_one_cfg.tool_error_feedback = "message"
    res = _run_extractor(_make_extractor(tmp_path), monkeypatch, _extractor_sites)["results"]
    for site in H_SITES:
        assert isinstance(res[site], str) and res[site] and res[site] != "None", site
    assert "ILLEGAL COMMAND" in res["h6_illegal"]
    assert res["v1_duplicate"] == DUP


@pytest.mark.parametrize("disabled", [False, True])
def test_extractor_sites_none_mode(tmp_path, monkeypatch, disabled):
    console_cfg.disable = disabled
    rouge_one_cfg.tool_error_feedback = "none"
    res = _run_extractor(_make_extractor(tmp_path), monkeypatch, _extractor_sites)["results"]
    # The SDK may render a None tool return as the text "None".
    for site in H_SITES:
        assert res[site] in (None, "None"), (site, res[site])
    assert res["v1_duplicate"] == DUP


def test_extractor_pandas_output_truncated(tmp_path, monkeypatch):
    rouge_one_cfg.tool_error_feedback = "message"

    async def scenario(tools):
        return await _inv(
            tools, "generic_pandas_tool",
            args={"command": "result = 'y' * 100000 if len(df) else ''", "explanation": "e"},
        )

    rouge_one_cfg.tool_output_max_chars = 0
    unbounded = _run_extractor(_make_extractor(tmp_path), monkeypatch, scenario)["results"]
    assert len(unbounded) > 100_000, unbounded
    rouge_one_cfg.tool_output_max_chars = 8000
    cut = _run_extractor(_make_extractor(tmp_path), monkeypatch, scenario)["results"]
    assert len(cut) < 8300 and "truncated" in cut


@pytest.mark.parametrize("rounds", [1, 3, 10])
def test_extractor_rounds_knob(tmp_path, monkeypatch, rounds):
    rouge_one_cfg.extractor_rounds = rounds
    state = _run_extractor(_make_extractor(tmp_path), monkeypatch, None)
    assert state["calls"] == rounds


# --------------------------------------------------------------------------- scientist


def test_scientist_model_settings_max_tokens():
    cfg = SimpleNamespace(temperature=0.7, reasoning=None, max_tokens=None)
    assert sci_mod._build_scientist_model_settings(cfg).max_tokens is None
    cfg.max_tokens = 16384
    assert sci_mod._build_scientist_model_settings(cfg).max_tokens == 16384


def test_scientist_round_label():
    f = sci_mod.ScientistAgent._round_label
    assert f(12, 10, 1) == "Round 2/10"
    assert f(12, 10, 9) == "Round 10/10"
    assert f(12, 10, None) == "Round 12/10"  # legacy behaviour when not supplied


def _make_scientist(tmp_path):
    agent = sci_mod.ScientistAgent.__new__(sci_mod.ScientistAgent)
    agent.cfg_experiment = _cfg(tmp_path)
    agent.knowledge_agent = SimpleNamespace()
    agent.web_search_agent = SimpleNamespace(explain_query=_async_value("  "))
    agent._note_book = ""
    _counters(agent)
    return agent


def _run_scientist(agent, monkeypatch, scenario, iteration_in_fold):
    state = {"prompts": []}

    class FakeRunner:
        async def run(self, a, prompt, **kwargs):
            state["prompts"].append(prompt)
            if scenario is not None and "results" not in state:
                try:
                    state["results"] = await scenario(_tools_by_name(a))
                except BaseException as e:  # the agent swallows Runner errors
                    state["error"] = repr(e)
                    raise
            return SimpleNamespace(final_output="focus text")

    monkeypatch.setattr(sci_mod, "Runner", FakeRunner)
    df = pd.DataFrame({"id": [1, 2], "v": [1.0, 2.0]})
    expl = pd.DataFrame({"Attribute": ["v"], "Status": ["Active"]})
    asyncio.run(
        agent.determine_focus(
            df, None, expl, ["previous focus"], index=12, max_index=10,
            iteration_in_fold=iteration_in_fold,
        )
    )
    return state


def test_scientist_prompt_renders_round_in_fold(tmp_path, monkeypatch):
    state = _run_scientist(_make_scientist(tmp_path), monkeypatch, None, 1)
    assert state["prompts"][0].startswith("Round 2/10:")


async def _scientist_sites(tools):
    async def pandas(cmd):
        return await _inv(tools, "generic_pandas_tool", command=cmd)

    return {
        "h2_empty_search": await _inv(tools, "search_tool", query="q"),
        "h3_no_result": await pandas("y = df.shape"),
        "h1_exception": await pandas("result = df['nope']"),
    }


@pytest.mark.parametrize("mode", ["message", "none"])
def test_scientist_sites(tmp_path, monkeypatch, mode):
    rouge_one_cfg.tool_error_feedback = mode
    state = _run_scientist(_make_scientist(tmp_path), monkeypatch, _scientist_sites, 0)
    assert "results" in state, state.get("error")
    for site, val in state["results"].items():
        if mode == "none":
            assert val in (None, "None"), site
        else:
            assert isinstance(val, str) and val and val != "None", site
