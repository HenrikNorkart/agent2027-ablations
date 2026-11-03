"""E9b: ROGUE_ONE_FEEDBACK_LEVEL=full|l0, the open-loop planner (AGENT 2027 A11-T7).

``full`` (default) must leave the Scientist's rendered prompt byte-identical to
``agent2027-v2.0`` (golden dumps in ``tests/unit/agent2027/fixtures/e9_full_v2``, rendered
from the v2.0 code by the same harness); ``l0`` must hide every ``train_*`` / L1 /
``feature_assessment`` value and the pruning columns from the Scientist, and change the
Scientist's prompt by exactly the seven A11-T6 edits.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import sys
from pathlib import Path

for _k, _v in {"EMBEDD_API_KEY": "x", "EMBEDD_MODEL": "x", "EMBEDD_PORT": "1", "LLM_MODEL": "x",
               "LLM_API_KEY": "x", "LLM_PORT": "1"}.items():
    os.environ.setdefault(_k, _v)

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests" / "unit" / "agent2027"))

import e9_scientist_harness as H  # noqa: E402
from rogueone.llm.agents.core.scientist import agent as sci  # noqa: E402
from rogueone.llm.prompts.agent_roles import AgentRoles  # noqa: E402
from rogueone.utils import flag_registry as fr  # noqa: E402
from rogueone.utils import loop_switches as ls  # noqa: E402
from rogueone.utils import rouge_one_cfg  # noqa: E402
from rogueone.utils import scientist_signal as ss  # noqa: E402
from rogueone.utils.config import rogueoneConfig  # noqa: E402

pytestmark = [pytest.mark.unit]

GOLDEN = ROOT / "tests" / "unit" / "agent2027" / "fixtures" / "e9_full_v2"
FORBIDDEN = ("test results", "tested attributes", "previously pruned", "feedback to the scientist",
             "train_", "feature_assessment", "cv_accuracy", "cv_rmse", "accuracy")
SECRETS = ("SECRET_REPORT", "0.731", "0.874", "SECRET_PRUNE_RATIONALE", "Pruned Iteration",
           "Prune Rationale")


@pytest.fixture
def level(monkeypatch):
    monkeypatch.setattr(rouge_one_cfg, "feedback_level", "full")
    monkeypatch.setattr(rouge_one_cfg, "restart_mode", "off")

    def _set(lvl, signal="inner_cv_l1"):
        monkeypatch.setattr(rouge_one_cfg, "feedback_level", lvl)
        monkeypatch.setattr(rouge_one_cfg, "scientist_signal", signal)

    return _set


def _render(tmp_path, **kw):
    return asyncio.run(H.render_scientist(tmp_path, **kw))


def _tool_text(tools) -> dict:
    return {n: t.description for n, t in tools.items()}


# --------------------------------------------------------------------------- flag
def test_flag_default_and_validation(monkeypatch):
    assert fr.FLAG_DEFAULTS["ROGUE_ONE_FEEDBACK_LEVEL"] == "full"
    assert ls.parse_feedback_level(None) == "full"
    assert ls.parse_feedback_level("") == "full"
    assert ls.parse_feedback_level("l0") == "l0"
    assert ls.parse_feedback_level("L0 ") == "l0"
    for bad in ("l1", "none", "off", "0"):
        with pytest.raises(ValueError):
            ls.parse_feedback_level(bad)
    c = rogueoneConfig()
    monkeypatch.delenv("ROGUE_ONE_FEEDBACK_LEVEL", raising=False)
    c.__init__()
    assert c.feedback_level == "full"
    monkeypatch.setenv("ROGUE_ONE_FEEDBACK_LEVEL", "L0 ")
    c.__init__()
    assert c.feedback_level == "l0"
    monkeypatch.setenv("ROGUE_ONE_FEEDBACK_LEVEL", "bogus")
    with pytest.raises(ValueError):
        c.__init__()
    monkeypatch.delenv("ROGUE_ONE_FEEDBACK_LEVEL", raising=False)
    c.__init__()


def test_v2_defaults_and_registry_carry_the_flag():
    env = (ROOT / "scripts" / "agent2027" / "v2_defaults.env").read_text(encoding="utf-8")
    assert "ROGUE_ONE_FEEDBACK_LEVEL=full" in env.splitlines()
    assert "ROGUE_ONE_FEEDBACK_LEVEL" in fr.FLAG_KEYS


# --------------------------------------------------------------------------- full = v2.0
def test_helpers_are_identity_under_full():
    pool = H.example_feedback_pool()
    view = sci._scientist_view_test_results(pool, "full")
    assert view is not pool and view.equals(pool)
    assert sci._scientist_view_test_results(None, "full").equals(pd.DataFrame())
    att = H.example_attributes()
    assert sci._scientist_view_attributes(att, "full") is att
    text = "anything"
    assert sci.apply_feedback_level_to_instructions(text, "full") is text
    tool = object()
    assert sci.apply_feedback_level_to_pandas_tool(tool, "full") is tool


def _golden(scenario: str) -> dict[str, bytes]:
    d = GOLDEN / scenario / "prompts"
    return {f.name: f.read_bytes() for f in sorted(d.iterdir())}


def _dump(tmp: Path) -> dict[str, bytes]:
    d = tmp / "prompts"
    return {f.name: f.read_bytes() for f in sorted(d.iterdir())}


def _tools_names_descriptions(raw: bytes) -> list:
    return [(t["name"], t["description"]) for t in json.loads(raw.decode("utf-8"))]


SCENARIOS = {
    # name: (scientist_signal, render kwargs)
    "train_fit_iter0": ("train_fit", dict(index=0, iteration_in_fold=0)),
    "l1_iter0": ("inner_cv_l1", dict(index=0, iteration_in_fold=0)),
    "l1_iter5": ("inner_cv_l1", dict(index=5, iteration_in_fold=5,
                                     focus_history=["focus 0", "focus 1", "focus 2", "focus 3",
                                                    "focus 4"])),
}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
@pytest.mark.parametrize("env_value", [None, "full"])
def test_full_renders_v2_0_prompt_byte_identical(tmp_path, level, monkeypatch, scenario, env_value):
    """Guard 2 of 01 §2.2: unset/full renders exactly what agent2027-v2.0 rendered."""
    signal, kw = SCENARIOS[scenario]
    if env_value is None:
        monkeypatch.delenv("ROGUE_ONE_FEEDBACK_LEVEL", raising=False)
    else:
        monkeypatch.setenv("ROGUE_ONE_FEEDBACK_LEVEL", env_value)
    level(ls.parse_feedback_level(os.environ.get("ROGUE_ONE_FEEDBACK_LEVEL")), signal)
    _render(tmp_path, dump=True, **kw)
    got, want = _dump(tmp_path), _golden(scenario)
    assert sorted(got) == sorted(want)
    for name in want:
        if name.endswith(".tools.json"):
            # Names and descriptions byte-identical (schemas depend on the openai-agents pin).
            assert _tools_names_descriptions(got[name]) == _tools_names_descriptions(want[name])
        else:
            assert got[name] == want[name], name


# --------------------------------------------------------------------------- l0: no leaks
def _l0_tool_script():
    cmds = [
        "result = df_test_results['feature_assessment']",
        "result = df_test_results['train_accuracy']",
        "result = df_test_results['cv_accuracy']",
        "result = df_test_results['cv_RMSE']",
        "result = df_test_results['status']",
        "result = df_test_results.to_string()",
        "result = list(df_test_results.columns)",
    ]
    script = [("generic_pandas_tool", {"command": c}) for c in cmds]
    script += [
        ("attribute_lookup_tool", {"attribute_names_list": ["a_div_b", "c_sq"]}),
        ("list_all_attributes", {}),
    ]
    return script


@pytest.mark.parametrize("signal", ["inner_cv_l1", "train_fit"])
def test_l0_scientist_cannot_read_feedback_values(tmp_path, level, signal):
    level("l0", signal)
    seen = _render(
        tmp_path, feedback_pool=H.example_feedback_pool(signal),
        attributes=H.example_attributes(), focus_history=["f0", "f1"], index=2,
        iteration_in_fold=2, tool_script=_l0_tool_script(),
    )
    outs = {o[1].get("command", o[0]): str(o[2]) for o in seen["tool_outputs"]}
    for col in ("feature_assessment", "train_accuracy", "cv_accuracy", "cv_RMSE", "status"):
        out = outs[f"result = df_test_results['{col}']"]
        assert "Error executing command" in out and col in out
    assert "Trial 0" in outs["result = df_test_results.to_string()"]
    assert outs["result = list(df_test_results.columns)"] == "['id']"
    lookup = outs["attribute_lookup_tool"]
    assert "a_div_b" in lookup and "c_sq" in lookup
    for col in sci.L0_HIDDEN_ATTRIBUTE_COLUMNS:
        assert col not in lookup
    assert "c_sq" in outs["list_all_attributes"]  # pruned names stay listed
    for out in outs.values():
        for secret in SECRETS:
            assert secret not in out, (secret, out)
    for marker in ("Pruned", "Active"):
        assert marker not in lookup


def test_full_scientist_does_read_them(tmp_path, level):
    """Sensitivity check for the test above: under full the same calls do leak."""
    level("full", "inner_cv_l1")
    seen = _render(
        tmp_path, feedback_pool=H.example_feedback_pool(), attributes=H.example_attributes(),
        focus_history=["f0", "f1"], index=2, iteration_in_fold=2, tool_script=_l0_tool_script(),
    )
    outs = {o[1].get("command", o[0]): str(o[2]) for o in seen["tool_outputs"]}
    assert "SECRET_REPORT_ZERO" in outs["result = df_test_results['feature_assessment']"]
    assert "0.874" in outs["result = df_test_results['cv_accuracy']"]
    assert "SECRET_PRUNE_RATIONALE" in outs["attribute_lookup_tool"]


def test_l0_views():
    pool = H.example_feedback_pool()
    v = sci._scientist_view_test_results(pool, "l0")
    assert list(v.columns) == ["id"] and list(v["id"]) == ["Trial 0", "Trial 1"]
    assert list(sci._scientist_view_test_results(pd.DataFrame(), "l0").columns) == ["id"]
    assert list(sci._scientist_view_test_results(None, "l0").columns) == ["id"]
    att = sci._scientist_view_attributes(H.example_attributes(), "l0")
    assert list(att.columns) == ["Attribute", "Description", "Pandas Command", "Added"]


# --------------------------------------------------------------------------- l0: prompt edits
def test_each_edit_applies_exactly_once(tmp_path, level):
    level("full")
    full = _render(tmp_path)["instructions"]
    for old, _new in sci.L0_PROMPT_EDITS:
        assert full.count(old) == 1, old[:60]
    edited = sci._apply_l0_prompt_edits(full)
    with pytest.raises(RuntimeError):
        sci._apply_l0_prompt_edits(edited)  # a second application finds no ``old``
    # Edit 3 drops exactly the two TESTER sentences about feedback to the Scientist.
    removed = set(AgentRoles.TESTER.value.split("\n")) - set(sci._TESTER_L0_TEXT.split("\n"))
    assert len(removed) == 2
    assert all("Scientist Agent" in line for line in removed)
    # The shared enum is untouched (the Extractor renders it).
    assert "detailed feedback to the Scientist Agent" in AgentRoles.TESTER.value


@pytest.mark.parametrize("signal", ["inner_cv_l1", "train_fit"])
def test_l0_prompt_differs_by_exactly_the_seven_edits(tmp_path, level, signal):
    level("full", signal)
    full = _render(tmp_path / "full", dump=True)
    level("l0", signal)
    l0 = _render(tmp_path / "l0", dump=True)

    # Edits 1-6: the instructions are full's with the six replacements and nothing else.
    expected = full["instructions"]
    for old, new in sci.L0_PROMPT_EDITS:
        expected = expected.replace(old, new)
    assert l0["instructions"] == expected
    changed = [ln for ln in difflib.ndiff(full["instructions"].splitlines(),
                                          l0["instructions"].splitlines())
               if ln[:2] in ("- ", "+ ")]
    # Edits 1, 2, 4, 5 and 6 each rewrite one line in place; edit 3 deletes two lines.
    assert sum(1 for ln in changed if ln.startswith("- ")) == 7
    assert sum(1 for ln in changed if ln.startswith("+ ")) == 5
    # Edit 7: only generic_pandas_tool's description; names, other tools, schemas unchanged.
    ft, lt = full["tools"], l0["tools"]
    assert list(ft) == list(lt)
    for name in ft:
        assert ft[name].params_json_schema == lt[name].params_json_schema
        if name != "generic_pandas_tool":
            assert ft[name].description == lt[name].description
    assert lt["generic_pandas_tool"].description == sci.l0_pandas_tool_description(
        ft["generic_pandas_tool"].description)
    assert lt["generic_pandas_tool"].description != ft["generic_pandas_tool"].description
    # The first user message is unchanged.
    assert l0["prompt"] == full["prompt"]
    # The rendered dumps differ only in the Scientist system prompt and tools file.
    fd, ld = _dump(tmp_path / "full"), _dump(tmp_path / "l0")
    assert sorted(fd) == sorted(ld)
    diff = sorted(n for n in fd if fd[n] != ld[n])
    assert diff == ["scientist__focus__iter000__c01.system.txt",
                    "scientist__focus__iter000__c01.tools.json"]


@pytest.mark.parametrize("signal", ["inner_cv_l1", "train_fit"])
def test_l0_prompt_has_no_feedback_vocabulary(tmp_path, level, signal):
    level("l0", signal)
    seen = _render(tmp_path)
    texts = [seen["instructions"], seen["tools"]["generic_pandas_tool"].description]
    for text in texts:
        low = text.lower()
        for phrase in FORBIDDEN:
            assert phrase not in low, phrase
    desc = seen["tools"]["generic_pandas_tool"].description
    assert "trial-index dataframe" in desc and "only the column `id`" in desc
    assert "result = df_raw_data.describe()" in desc
    assert "Use this tool to explore the data." in desc
    assert seen["tools"]["generic_pandas_tool"].name == "generic_pandas_tool"


def test_l0_tool_description_rejects_unknown_text():
    with pytest.raises(RuntimeError):
        sci.l0_pandas_tool_description("a description that has none of the anchors")
    assert ss.TRAIN_FIT_COLUMNS_LINE and ss.L1_COLUMNS_LINE  # anchors exist
