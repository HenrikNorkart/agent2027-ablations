"""E9d FX switches (A12-T4…T8): P− (SE|T digest), A− (template report), J−a (rules
pruning) with shadow mode, M− (amnesic planner).

Guards checked here (01 §2 post-freeze rule, guard 2): with every FX flag unset or at
its Full-v2 value, the code takes the unchanged path (same prompts, same tool returns,
same table, no FX file, no FX summary key). Agents run with a fake ``Runner``.
"""

import asyncio
import importlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

for _k, _v in {"EMBEDD_API_KEY": "x", "EMBEDD_MODEL": "x", "EMBEDD_PORT": "1", "LLM_MODEL": "x"}.items():
    os.environ.setdefault(_k, _v)

import numpy as np
import pandas as pd
import pytest
from agents.tool_context import ToolContext

import rogueone.llm.agents.core.extractor.agent as ext_mod
import rogueone.llm.agents.core.tester.agent as tst_mod
from rogueone.llm.agents.core.tester.metrics import FeatureMetricProfile, _compute_vif
from rogueone.utils import console_cfg, fx_switches as fx, rouge_one_cfg
from rogueone.utils import flag_registry as fr
from rogueone.utils import instrumentation as ins

pytestmark = [pytest.mark.unit]

FX_FLAGS = tuple(fx.FX_FLAGS)
FULL_V2 = {
    "ROGUE_ONE_PLANNER_MODE": "llm",
    "ROGUE_ONE_MEMORY_MODE": "full",
    "ROGUE_ONE_ASSESSMENT_MODE": "llm",
    "ROGUE_ONE_PRUNING_MODE": "llm",
    "ROGUE_ONE_PRUNING_SHADOW": "0",
}
CELLS = {  # A12 §4 run matrix
    "fx_pAJM": {"ROGUE_ONE_PLANNER_MODE": "none", "ROGUE_ONE_PRUNING_SHADOW": "1"},
    "fx_PaJM": {"ROGUE_ONE_ASSESSMENT_MODE": "template", "ROGUE_ONE_PRUNING_SHADOW": "1"},
    "fx_PAjM": {"ROGUE_ONE_PRUNING_MODE": "rules"},
    "fx_PAJm": {"ROGUE_ONE_MEMORY_MODE": "amnesic", "ROGUE_ONE_PRUNING_SHADOW": "1"},
    "fx_pajM": {"ROGUE_ONE_PLANNER_MODE": "none", "ROGUE_ONE_ASSESSMENT_MODE": "template",
                "ROGUE_ONE_PRUNING_MODE": "rules"},
    "fx_paJm": {"ROGUE_ONE_PLANNER_MODE": "none", "ROGUE_ONE_ASSESSMENT_MODE": "template",
                "ROGUE_ONE_MEMORY_MODE": "amnesic", "ROGUE_ONE_PRUNING_SHADOW": "1"},
    "fx_pAjm": {"ROGUE_ONE_PLANNER_MODE": "none", "ROGUE_ONE_PRUNING_MODE": "rules",
                "ROGUE_ONE_MEMORY_MODE": "amnesic"},
    "fx_Pajm": {"ROGUE_ONE_ASSESSMENT_MODE": "template", "ROGUE_ONE_PRUNING_MODE": "rules",
                "ROGUE_ONE_MEMORY_MODE": "amnesic"},
}


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    for k in FX_FLAGS:
        monkeypatch.delenv(k, raising=False)
    saved = (rouge_one_cfg.extractor_rounds, rouge_one_cfg.tool_output_max_chars, console_cfg.disable)
    console_cfg.disable = True
    ins.configure(tmp_path / "out")
    yield
    rouge_one_cfg.extractor_rounds, rouge_one_cfg.tool_output_max_chars, console_cfg.disable = saved
    ins.reset()


def _set(monkeypatch, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)


def _jsonl(tmp_path, name):
    p = Path(tmp_path) / "out" / name
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# =========================================================================== flags
def test_unset_and_full_v2_values_are_inactive(monkeypatch):
    assert not fx.any_active()
    assert fx.validate_all() == FULL_V2
    _set(monkeypatch, FULL_V2)
    assert not fx.any_active()
    assert fx.summary_keys({"fx_pruned": 3}) == {}


def test_full_v2_values_match_registry_and_v2_defaults():
    for k, v in FULL_V2.items():
        assert fr.FLAG_DEFAULTS[k] == v
    text = (Path(__file__).resolve().parents[3] / "scripts" / "agent2027" / "v2_defaults.env").read_text()
    for k, v in FULL_V2.items():
        assert f"\n{k}={v}\n" in text


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_every_a12_cell_is_active_and_valid(monkeypatch, cell):
    _set(monkeypatch, CELLS[cell])
    assert fx.any_active()
    fx.validate_all()


def test_bad_values_raise(monkeypatch):
    for k, bad in (("ROGUE_ONE_PLANNER_MODE", "nope"), ("ROGUE_ONE_MEMORY_MODE", "half"),
                   ("ROGUE_ONE_ASSESSMENT_MODE", "x"), ("ROGUE_ONE_PRUNING_MODE", "rule"),
                   ("ROGUE_ONE_PRUNING_SHADOW", "2")):
        monkeypatch.setenv(k, bad)
        with pytest.raises(fx.FxFlagError):
            fx.validate_all()
        monkeypatch.delenv(k)


def test_e9c_planner_modes_are_recognised_but_not_fx(monkeypatch):
    for mode in ("oracle", "misdirected"):
        monkeypatch.setenv("ROGUE_ONE_PLANNER_MODE", mode)
        assert fx.planner_mode() == mode and not fx.planner_off() and not fx.any_active()


def test_shadow_only_next_to_llm_pruner(monkeypatch):
    monkeypatch.setenv("ROGUE_ONE_PRUNING_SHADOW", "1")
    assert fx.shadow_active()
    monkeypatch.setenv("ROGUE_ONE_PRUNING_MODE", "rules")
    assert not fx.shadow_active()


# =========================================================================== M−
def test_m_minus_history_and_notebook(monkeypatch):
    hist = ["a", "b", "c"]
    assert fx.scientist_history_view(hist) is hist  # Full: the same object
    agent = SimpleNamespace(_note_book="Trail 1:\nnote\n")
    fx.reset_scientist_memory(agent)
    assert agent._note_book.startswith("Trail 1")
    monkeypatch.setenv("ROGUE_ONE_MEMORY_MODE", "amnesic")
    assert fx.scientist_history_view(hist) == ["c"] and hist == ["a", "b", "c"]
    assert fx.scientist_history_view([]) == []
    fx.reset_scientist_memory(agent)
    assert agent._note_book == ""


# =========================================================================== P− units
def _pool():
    return pd.DataFrame([{
        "id": "Trial 0", "status": "completed", "dataset_id": "synp_p3", "fold_seed": 42,
        "aborted_fold_id": None, "skip_reason": None, "feature_assessment": "REPORT-TEXT",
        "cv_accuracy": 0.81, "cv_RMSE": None,
    }])


def _attr_table():
    return pd.DataFrame({
        "Attribute": ["a", "b"], "Status": ["Active", "Pruned"],
        "Pruned Iteration": [pd.NA, "Trial 0"], "Prune Rationale": [pd.NA, "ja_vif_pfi"],
    })


def test_digest_iteration0_and_sections(monkeypatch):
    assert fx.build_digest(pd.DataFrame(), _attr_table(), [])[0] == fx.NO_TRIAL_DIGEST
    d, meta = fx.build_digest(_pool(), _attr_table(), ["f one", "f two"])
    assert "REPORT-TEXT" in d and "cv_accuracy: 0.81" in d and "status: completed" in d
    assert "cv_RMSE" not in d and "dataset_id" not in d
    assert "| b | Pruned | Trial 0 | ja_vif_pfi |" in d
    assert "1. f one" in d and "2. f two" in d and meta["focus_lines"] == 2
    monkeypatch.setenv("ROGUE_ONE_MEMORY_MODE", "amnesic")
    d, meta = fx.build_digest(_pool(), _attr_table(), ["f one", "f two"])
    assert "f one" not in d and "1. f two" in d and meta["focus_lines"] == 1


def test_digest_sections_are_capped(monkeypatch):
    monkeypatch.setenv("ROGUE_ONE_TOOL_OUTPUT_MAX_CHARS", "50")
    pool = _pool()
    pool.loc[0, "feature_assessment"] = "x" * 5000
    d, meta = fx.build_digest(pool, _attr_table(), ["f"])
    assert "truncated to 50" in d and meta["chars_feedback"] < 120


@pytest.mark.parametrize("text,focus,missing", [
    ("FOCUS: ratios of income\nthen more", "ratios of income", False),
    ("Some intro\n**FOCUS:** log transforms", "log transforms", False),
    ("focus: lower case works", "lower case works", False),
    ("No focus line here at all", "No focus line here at all", True),
])
def test_parse_focus(text, focus, missing):
    assert fx.parse_focus([text]) == (focus, missing)


def test_parse_focus_fallback_is_first_300_chars():
    f, missing = fx.parse_focus(["y" * 1000])
    assert missing and f == "y" * 300
    assert fx.parse_focus([]) == (fx.FOCUS_FALLBACK, True)


def test_self_focus_rounds():
    sf = fx.SelfFocus(message="DIGEST")
    assert sf.message_for_round(0) == "DIGEST"
    sf.observe(0, SimpleNamespace(final_output="FOCUS: build ratios\nok", new_items=[]))
    assert sf.message_for_round(1) == "System now wants to focus on: build ratios."
    sf.observe(1, SimpleNamespace(final_output="FOCUS: other", new_items=[]))  # ignored
    assert sf.finalize() == "build ratios" and ins.get_counter("focus_line_missing") == 0


def test_self_focus_missing_line_is_counted():
    sf = fx.SelfFocus(message="DIGEST")
    sf.observe(0, SimpleNamespace(final_output="I added three features.", new_items=[]))
    assert sf.finalize() == "I added three features." and ins.get_counter("focus_line_missing") == 1


def test_self_focus_reads_message_items_not_only_final_output():
    from agents import MessageOutputItem
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    msg = ResponseOutputMessage(
        id="m", type="message", role="assistant", status="completed",
        content=[ResponseOutputText(type="output_text", text="FOCUS: from first message", annotations=[])],
    )
    from agents import Agent

    item = MessageOutputItem(agent=Agent(name="x"), raw_item=msg)
    sf = fx.SelfFocus(message="D")
    sf.observe(0, SimpleNamespace(final_output="Done; summary table.", new_items=[item]))
    assert sf.finalize() == "from first message"


# =========================================================================== P− extractor
def _make_extractor(tmp_path):
    agent = ext_mod.ExtractorAgent.__new__(ext_mod.ExtractorAgent)
    agent.cfg_experiment = SimpleNamespace(
        output_dir=Path(tmp_path), experiment_description="d", context_and_goal="g",
        experiment_name="e", task_type="classification", modality="tabular",
    )
    agent.knowledge_agent = SimpleNamespace()
    return agent


def _run_extractor(tmp_path, monkeypatch, self_focus=None, rounds=3):
    rouge_one_cfg.extractor_rounds = rounds
    seen = []

    class FakeRunner:
        async def run(self, a, inp, *args, **kwargs):
            seen.append((a.instructions, inp, tuple(t.name for t in a.tools)))
            return SimpleNamespace(final_output="FOCUS: interaction terms\nadded", new_items=[])

    monkeypatch.setattr(ext_mod, "Runner", FakeRunner)
    df = pd.DataFrame({"id": [1, 2], "v": [1.0, 2.0]})
    feats = pd.DataFrame({"id": [1, 2], "known": [0.0, 1.0]})
    expl = pd.DataFrame({"Attribute": ["known"], "Status": ["Active"]})
    kw = {"self_focus": self_focus} if self_focus is not None else {}
    asyncio.run(_make_extractor(tmp_path).generate_patient_attributes(df, feats, expl, "THE FOCUS", step=4, **kw))
    return seen


def test_extractor_default_messages_unchanged(tmp_path, monkeypatch):
    seen = _run_extractor(tmp_path, monkeypatch)
    assert [s[1] for s in seen] == ["System now wants to focus on: THE FOCUS."] * 3


def test_extractor_p_minus_digest_in_round_one_only(tmp_path, monkeypatch):
    base = _run_extractor(tmp_path, monkeypatch)
    sf = fx.SelfFocus(message=fx.P_MINUS_PREAMBLE + "\n\nDIGEST-BODY")
    seen = _run_extractor(tmp_path, monkeypatch, self_focus=sf)
    msgs = [s[1] for s in seen]
    assert msgs[0].startswith(fx.P_MINUS_PREAMBLE) and "DIGEST-BODY" in msgs[0]
    assert msgs[1:] == ["System now wants to focus on: interaction terms."] * 2
    assert sum("DIGEST-BODY" in m for m in msgs) == 1
    # declared: the system prompt and tool list are unchanged under P−
    assert {s[0] for s in seen} == {s[0] for s in base}
    assert {s[2] for s in seen} == {s[2] for s in base}
    assert sf.finalize() == "interaction terms"


def test_begin_self_focus_logs_one_digest_record(tmp_path, monkeypatch):
    assert fx.begin_self_focus(_pool(), _attr_table(), [], index=0, fold_run_id=0, iteration_in_fold=0) is None
    assert _jsonl(tmp_path, fx.DIGEST_FILE) is None
    monkeypatch.setenv("ROGUE_ONE_PLANNER_MODE", "none")
    sf = fx.begin_self_focus(pd.DataFrame(), _attr_table(), [], index=0, fold_run_id=0, iteration_in_fold=0)
    assert sf.message.endswith(fx.NO_TRIAL_DIGEST)
    (rec,) = _jsonl(tmp_path, fx.DIGEST_FILE)
    assert rec["digest"] == fx.NO_TRIAL_DIGEST and rec["iteration_in_fold"] == 0


# =========================================================================== P−/M− main
_stubbed = importlib.util.find_spec("git") is None
if _stubbed:
    sys.modules["git"] = types.SimpleNamespace(Tree=object)
try:
    rom = importlib.import_module("rogue_one_main")
finally:
    if _stubbed:
        sys.modules.pop("git", None)
        sys.modules.pop("rogue_one_main", None)

COLS = ["Attribute", "Description", "Pandas Command", "Status", "Added", "Pruned Iteration", "Prune Rationale"]


def _network(tmp_path):
    net = object.__new__(rom.RogueOneAgentNetwork)
    net.cfg = SimpleNamespace(output_dir=Path(tmp_path) / "out", experiment_name="e", task_type="classification")
    net.enable_wandb = False
    net.df = pd.DataFrame({"id": [1, 2], "v": [1.0, 2.0]})
    net.df_entities_attributes = pd.DataFrame({"id": [1, 2]})
    net.df_attributes_explanations = pd.DataFrame(columns=COLS)
    net.df_scientist_feedback_pool = _pool()
    net.focus_history = ["earlier own focus"]
    net.df_test_pool = None
    net._scored_before_test = None
    return net


def _drive(net, calls):
    async def determine_focus(df, pool, attrs, history, **k):
        calls.append(("scientist", list(history), net.scientist_agent._note_book))
        return "scientist focus"

    async def generate(df, feats, attrs, focus, step=None, **kw):
        sf = kw.get("self_focus")
        calls.append(("extractor", focus, sf))
        if sf is not None:
            for r in range(3):
                msg = sf.message_for_round(r)
                calls.append(("msg", msg))
                sf.observe(r, SimpleNamespace(final_output="FOCUS: my own focus\n...", new_items=[]))
        return net.df_entities_attributes, pd.DataFrame()  # nothing produced -> iteration_skipped

    net.scientist_agent = SimpleNamespace(determine_focus=determine_focus, _note_book="old notes")
    net.extractor_agent = SimpleNamespace(generate_patient_attributes=generate)
    return asyncio.run(net.forward_pass(1))


def test_main_full_path_calls_scientist_with_full_history(tmp_path):
    net, calls = _network(tmp_path), []
    assert _drive(net, calls) == "iteration_skipped"
    assert calls[0] == ("scientist", ["earlier own focus"], "old notes")
    assert calls[1][0] == "extractor" and calls[1][1] == "scientist focus" and calls[1][2] is None
    (row,) = _jsonl(tmp_path, ins.FOCUS_FILE)
    assert row["focus"] == "scientist focus"
    assert _jsonl(tmp_path, fx.DIGEST_FILE) is None


def test_main_m_minus_truncates_history_and_clears_notebook(tmp_path, monkeypatch):
    monkeypatch.setenv("ROGUE_ONE_MEMORY_MODE", "amnesic")
    net, calls = _network(tmp_path), []
    net.focus_history = ["f0", "f1", "f2"]
    _drive(net, calls)
    assert calls[0] == ("scientist", ["f2"], "")
    assert net.focus_history == ["f0", "f1", "f2", "scientist focus"]


def test_main_p_minus_skips_scientist_and_logs_own_focus(tmp_path, monkeypatch):
    monkeypatch.setenv("ROGUE_ONE_PLANNER_MODE", "none")
    net, calls = _network(tmp_path), []
    assert _drive(net, calls) == "iteration_skipped"
    assert not [c for c in calls if c[0] == "scientist"]
    msgs = [c[1] for c in calls if c[0] == "msg"]
    assert msgs[0].startswith(fx.P_MINUS_PREAMBLE) and "REPORT-TEXT" in msgs[0]
    assert "earlier own focus" in msgs[0]
    assert msgs[1:] == ["System now wants to focus on: my own focus."] * 2
    (row,) = _jsonl(tmp_path, ins.FOCUS_FILE)
    assert row["focus"] == "my own focus" and row["source"] == "extractor_self"
    assert net.focus_history[-1] == "my own focus"
    assert len(_jsonl(tmp_path, fx.DIGEST_FILE)) == 1
    s = ins.compute_summary()
    assert s["fx_digests"] == 1 and s["focus_line_missing"] == 0 and s["llm_calls_scientist"] == 0


# =========================================================================== J−a rules
def _profile(name, vif, shap, mi=0.1, pfi_m=0.05, pfi_s=0.01):
    return FeatureMetricProfile(name, mi, vif, pfi_m, pfi_s, 0.0, shap)


def _trap_X(n=200, seed=0):
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, 3, size=n)
    d = np.eye(3)[cat]  # d0 + d1 + d2 == 1: perfect collinearity with the intercept
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    return np.column_stack([d, x1, x2]), ["d0", "d1", "d2", "x1", "x2"]


def test_ja_dummy_trap_prunes_exactly_one_low_shap_member():
    X, names = _trap_X()
    vifs = _compute_vif(X, names)
    assert all(v >= 1e10 for v in vifs[:3]) and all(v < 10 for v in vifs[3:])
    profs = [_profile("d0", vifs[0], 0.05), _profile("d1", vifs[1], 0.02), _profile("d2", vifs[2], 0.08),
             _profile("x1", vifs[3], 0.5), _profile("x2", vifs[4], 0.01, pfi_m=0.0)]
    dec = fx.ja_rules(profs, X, names)
    assert dec == [fx.JaDecision("d1", fx.RULE_CLUSTER)]


def test_ja_tie_breaks_on_mi():
    X, names = _trap_X()
    vifs = _compute_vif(X, names)
    profs = [_profile("d0", vifs[0], 0.05, mi=0.3), _profile("d1", vifs[1], 0.05, mi=0.1),
             _profile("d2", vifs[2], 0.05, mi=0.2), _profile("x1", 1.0, 0.5), _profile("x2", 1.0, 0.5)]
    assert fx.ja_rules(profs, X, names) == [fx.JaDecision("d1", fx.RULE_CLUSTER)]


def test_ja_high_shap_collinear_pair_is_not_pruned():
    rng = np.random.default_rng(1)
    a = rng.normal(size=100)
    X = np.column_stack([a, 2 * a, rng.normal(size=100)])
    profs = [_profile("a", 1e12, 0.4), _profile("a2", 1e12, 0.3), _profile("z", 1.1, 0.01, pfi_m=0.0)]
    assert fx.ja_rules(profs, X, ["a", "a2", "z"]) == []


def test_ja_finite_vif_zero_pfi_low_shap_is_pruned_and_vif_below_10_never():
    profs = [
        _profile("f", 25.0, 0.03, pfi_m=0.0, pfi_s=0.01),     # pruned (ja_vif_pfi)
        _profile("g", 25.0, 0.03, pfi_m=0.05, pfi_s=0.01),    # PFI clearly > 1 SD: kept
        _profile("h", 25.0, 0.1, pfi_m=0.0, pfi_s=0.01),      # SHAP not < 0.1: kept
        _profile("low", 9.99, 0.0, pfi_m=0.0, pfi_s=0.01),    # VIF < 10: never pruned
        _profile("nan", None, 0.0, pfi_m=0.0, pfi_s=0.01),    # missing VIF: never pruned
        _profile("nanshap", 25.0, None, pfi_m=0.0, pfi_s=0.01),
    ]
    X = np.random.default_rng(2).normal(size=(50, 6))
    names = ["f", "g", "h", "low", "nan", "nanshap"]
    assert fx.ja_rules(profs, X, names) == [fx.JaDecision("f", fx.RULE_PFI)]


def test_ja_protected_seeded_rows_are_never_pruned():
    X, names = _trap_X()
    vifs = _compute_vif(X, names)
    profs = [_profile(n, v, s) for n, v, s in zip(names, vifs, (0.05, 0.02, 0.08, 0.5, 0.01))]
    dec = fx.ja_rules(profs, X, names, protected={"d1"})
    assert dec == [fx.JaDecision("d0", fx.RULE_CLUSTER)]
    assert fx.ja_rules(profs, X, names, protected={"d0", "d1", "d2"}) == []


# =========================================================================== Tester runs
FEATS = ["d0", "d1", "d2", "x1", "x2", "x3"]


def _folds(k=5, n=40, seed=0):
    rng = np.random.default_rng(seed)
    folds, nid = [], 0
    for _ in range(k):
        cat = rng.integers(0, 3, size=n)
        d = pd.DataFrame(np.eye(3)[cat], columns=["d0", "d1", "d2"])
        d["x1"] = rng.normal(size=n)
        d["x2"] = rng.normal(size=n)
        d["x3"] = d["x1"] * 0.9 + rng.normal(size=n) * 0.1
        d["target"] = (d["x1"] > 0).astype(int)
        d["id"] = range(nid, nid + n)
        nid += n
        folds.append(d)
    return folds


def _expl():
    return pd.DataFrame({
        "Attribute": FEATS, "Description": [""] * 6, "Pandas Command": ["r"] * 6,
        "Status": ["Active"] * 6, "Added": ["Trial 7"] * 3 + ["Trial 3"] * 3,
        "Pruned Iteration": [pd.NA] * 6, "Prune Rationale": [pd.NA] * 6,
    })


def _fake_metrics(X_train, y_train, feature_names, **kw):
    """Controlled profiles: a dummy trap (d0-d2), a signal (x1), a VIF<10 noise (x2) and a
    finite-VIF zero-PFI low-SHAP feature (x3). Values per name, so the table is fixed."""
    table = {
        "d0": (1e12, 0.05, 0.1, 0.02, 0.01), "d1": (1e12, 0.02, 0.1, 0.0, 0.01),
        "d2": (1e12, 0.08, 0.1, 0.03, 0.01), "x1": (12.0, 0.9, 0.6, 0.3, 0.02),
        "x2": (1.1, 0.01, 0.0, 0.0, 0.01), "x3": (12.0, 0.03, 0.3, 0.0, 0.02),
    }
    out = []
    for n in feature_names:
        vif, shap, mi, pm, ps = table[n]
        out.append(FeatureMetricProfile(n, mi, vif, pm, ps, 0.0, shap))
    return out


def _run_tester(tmp_path, monkeypatch, env, llm_prune=("x2",), include_report=True):
    for k in FX_FLAGS:
        monkeypatch.delenv(k, raising=False)
    _set(monkeypatch, env)
    ins.configure(tmp_path / "out")
    monkeypatch.setattr(tst_mod, "compute_feature_metrics", _fake_metrics)
    prompts = []

    class FakeRunner:
        @staticmethod
        async def run(a, inp, *args, **kwargs):
            tools = {t.name: t for t in a.tools}
            rec = {"agent": a.name, "instructions": a.instructions, "input": inp,
                   "tools": sorted(tools), "tool_out": []}
            prompts.append(rec)
            if "attribute_pruning_tool" in tools:
                payload = json.dumps({"attribute_names_list": list(llm_prune)})
                ctx = ToolContext(context=None, tool_name="attribute_pruning_tool",
                                  tool_call_id="c", tool_arguments=payload)
                rec["tool_out"].append(await tools["attribute_pruning_tool"].on_invoke_tool(ctx, payload))
                for name in ("get_precomputed_metrics_tool", "get_notes_tool"):
                    ctx = ToolContext(context=None, tool_name=name, tool_call_id="c", tool_arguments="{}")
                    rec["tool_out"].append(await tools[name].on_invoke_tool(ctx, "{}"))
                return SimpleNamespace(final_output="LLM pruning summary")
            return SimpleNamespace(final_output="# LLM assessment\nfine")

    monkeypatch.setattr(tst_mod, "Runner", FakeRunner)
    cfg = SimpleNamespace(output_dir=Path(tmp_path) / "out", experiment_description="d", context_and_goal="g",
                          experiment_name="e", task_type="classification", modality="tabular")
    agent = tst_mod.TesterAgent(cfg, include_report=include_report, enable_wandb=False)
    res, expl = asyncio.run(agent.test_hypotheses(_folds(), _expl(), step=0, trial_index=7))
    return SimpleNamespace(res=res, expl=expl, prompts=prompts, agent=agent, summary=ins.compute_summary())


def _pruned(expl):
    return sorted(expl.loc[expl["Status"] == "Pruned", "Attribute"])


def test_tester_full_unset_equals_full_v2_values_and_writes_no_fx(tmp_path, monkeypatch):
    a = _run_tester(tmp_path / "a", monkeypatch, {})
    b = _run_tester(tmp_path / "b", monkeypatch, FULL_V2)
    assert a.prompts == b.prompts and a.res.report == b.res.report
    pd.testing.assert_frame_equal(a.expl, b.expl)
    assert _pruned(a.expl) == ["x2"]
    assert not any(k.startswith("fx_") for k in b.summary)
    for d in (tmp_path / "a", tmp_path / "b"):
        assert not (d / "out" / fx.SHADOW_FILE).exists() and not (d / "out" / fx.DIGEST_FILE).exists()
    assert [p["agent"] for p in a.prompts] == ["Tester Agent (Assessment)", "Tester Agent (Pruning)"]


def test_j_minus_a_prunes_by_rules_without_stage_b_llm_call(tmp_path, monkeypatch):
    r = _run_tester(tmp_path, monkeypatch, CELLS["fx_PAjM"])
    assert [p["agent"] for p in r.prompts] == ["Tester Agent (Assessment)"]  # no Stage B call
    assert _pruned(r.expl) == ["d1", "x3"]
    rat = r.expl.set_index("Attribute")
    assert rat.loc["d1", "Prune Rationale"] == fx.RULE_CLUSTER
    assert rat.loc["x3", "Prune Rationale"] == fx.RULE_PFI
    assert rat.loc["d1", "Pruned Iteration"] == "Trial 7"
    assert r.agent._pruning_stage_status == "rules"
    assert {d["rationale"] for d in r.agent._pruning_decisions} == {fx.RULE_CLUSTER, fx.RULE_PFI}
    assert r.summary["fx_ja_vif_cluster"] == 1 and r.summary["fx_ja_vif_pfi"] == 1
    assert r.res.report == "# LLM assessment\nfine"  # A+ report untouched


def test_prune_rate_manipulation_differs_between_j_plus_and_j_minus_a(tmp_path, monkeypatch):
    jp = _run_tester(tmp_path / "jp", monkeypatch, CELLS["fx_PAJm"])     # J+ (LLM prunes x2), shadow on
    ja = _run_tester(tmp_path / "ja", monkeypatch, CELLS["fx_PAjM"])     # J−a
    assert _pruned(jp.expl) == ["x2"] and _pruned(ja.expl) == ["d1", "x3"]
    assert jp.summary["fx_prune_candidates"] == ja.summary["fx_prune_candidates"] == 6
    assert jp.summary["fx_prune_rate"] == pytest.approx(1 / 6)
    assert ja.summary["fx_prune_rate"] == pytest.approx(2 / 6)
    assert jp.summary["fx_prune_rate"] != ja.summary["fx_prune_rate"]
    # the shadow in the J+ cell records what J−a would have done
    assert jp.summary["fx_shadow_prune_rate"] == pytest.approx(2 / 6)
    assert jp.summary["fx_shadow_jaccard_mean"] == 0.0


def test_shadow_decisions_are_logged_and_never_reach_a_prompt(tmp_path, monkeypatch):
    off = _run_tester(tmp_path / "off", monkeypatch, {})
    on = _run_tester(tmp_path / "on", monkeypatch, {"ROGUE_ONE_PRUNING_SHADOW": "1"})
    # byte-identical prompts, tool returns and report; identical table (shadow mutates nothing)
    assert on.prompts == off.prompts and on.res.report == off.res.report
    pd.testing.assert_frame_equal(on.expl, off.expl)
    blob = json.dumps(on.prompts) + on.res.report + on.expl.to_csv()
    for token in ("ja_vif", "shadow", "J-a", "J−a"):
        assert token not in blob
    (rec,) = _jsonl(tmp_path / "on", fx.SHADOW_FILE)
    assert rec["shadow_pruned"] == ["d1", "x3"] and rec["llm_pruned"] == ["x2"]
    assert {d["rule"] for d in rec["shadow_decisions"]} == {fx.RULE_CLUSTER, fx.RULE_PFI}
    assert rec["jaccard"] == 0.0 and rec["pre_active"] == 6
    assert on.summary["floor_activations"] == off.summary["floor_activations"] == 0
    assert not (tmp_path / "off" / "out" / fx.SHADOW_FILE).exists()


def test_shadow_is_off_under_rules(tmp_path, monkeypatch):
    _run_tester(tmp_path, monkeypatch, {"ROGUE_ONE_PRUNING_MODE": "rules", "ROGUE_ONE_PRUNING_SHADOW": "1"})
    assert _jsonl(tmp_path, fx.SHADOW_FILE) is None


def test_a_minus_template_no_stage_a_call_and_j_plus_prunes_through_the_gate(tmp_path, monkeypatch):
    r = _run_tester(tmp_path, monkeypatch, CELLS["fx_PaJM"])
    assert [p["agent"] for p in r.prompts] == ["Tester Agent (Pruning)"]  # 0 Stage-A calls
    stage_b = r.prompts[0]
    assert stage_b["tool_out"][0] == "Attributes pruned successfully."  # gate passed
    assert _pruned(r.expl) == ["x2"]
    # Stage B received sections 1–3 only
    assert "## 1. Feature counts" in stage_b["input"] and "## 4." not in stage_b["input"]
    assert "Active features assessed: 6" in stage_b["input"]
    assert "New features this iteration: 3" in stage_b["input"]
    assert "| 1 | x1 | 0.9 |" in stage_b["input"]
    # declared: Stage B's notebook is empty under A−
    assert stage_b["tool_out"][2] == ""
    # the planner gets sections 1–4; section 4 lists the LLM's prunes
    assert "## 4. Pruned this iteration" in r.res.report and "| x2 | llm |" in r.res.report
    assert r.agent._assessment_stage_status == "template"
    md = (tmp_path / "out" / "feature_assessment_step_0.md").read_text(encoding="utf-8")
    assert md == r.res.report
    assert r.summary["fx_template_reports"] == 1


def test_minimal_harness_cell_has_no_tester_llm_call(tmp_path, monkeypatch):
    r = _run_tester(tmp_path, monkeypatch, CELLS["fx_pajM"])
    assert r.prompts == []
    assert _pruned(r.expl) == ["d1", "x3"]
    assert "| d1 | ja_vif_cluster |" in r.res.report and "| x3 | ja_vif_pfi |" in r.res.report


def test_ja_floor_restores_like_stage_b(tmp_path, monkeypatch):
    """Only 3 agent features, J−a wants 2 pruned -> floor max(3, ceil(0.6)) restores."""
    t = pd.DataFrame({"Attribute": ["a", "b", "c"], "Status": ["Active"] * 3,
                      "Pruned Iteration": [pd.NA] * 3, "Prune Rationale": [pd.NA] * 3})
    dec = [fx.JaDecision("a", fx.RULE_PFI), fx.JaDecision("b", fx.RULE_PFI)]
    out = fx.apply_ja_pruning(t, dec, trial_label="Trial 1", floor_fn=tst_mod._enforce_pruning_floor,
                              shap_lookup={"a": 0.01, "b": 0.02}, mi_lookup={})
    assert sorted(out.restored) == ["a", "b"] and out.newly_pruned == []
    assert list(t["Status"]) == ["Active"] * 3 and out.stage_status == "rules_floor_enforced"
    assert ins.get_counter("floor_activations") == 1


def test_simulate_never_mutates_and_never_counts():
    t = pd.DataFrame({"Attribute": ["a", "b", "c", "d", "e"], "Status": ["Active"] * 5,
                      "Pruned Iteration": [pd.NA] * 5, "Prune Rationale": [pd.NA] * 5})
    before = t.copy(deep=True)
    out = fx.simulate_ja(t, [fx.JaDecision("a", fx.RULE_PFI)], floor_fn=tst_mod._enforce_pruning_floor,
                         shap_lookup={}, mi_lookup={})
    assert out.newly_pruned == ["a"]
    pd.testing.assert_frame_equal(t, before)
    assert ins.get_counter("fx_ja_vif_pfi") == 0


def test_summary_keys_only_when_active(monkeypatch):
    assert "fx_prune_rate" not in ins.compute_summary()
    monkeypatch.setenv("ROGUE_ONE_PRUNING_MODE", "rules")
    s = ins.compute_summary()
    assert s["fx_pruning_mode"] == "rules" and s["fx_prune_rate"] is None and s["focus_line_missing"] == 0


def test_start_run_empties_stale_fx_logs_only_if_present(tmp_path, monkeypatch):
    out = tmp_path / "out"
    fx.start_run()
    assert not (out / fx.DIGEST_FILE).exists()
    (out / fx.SHADOW_FILE).write_text('{"old": 1}\n', encoding="utf-8")
    fx.start_run()
    assert (out / fx.SHADOW_FILE).read_text(encoding="utf-8") == ""
    monkeypatch.setenv("ROGUE_ONE_PRUNING_MODE", "bogus")
    with pytest.raises(fx.FxFlagError):
        fx.start_run()
