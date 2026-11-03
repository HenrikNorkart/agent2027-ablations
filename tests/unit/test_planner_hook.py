"""E9c / A09-T5: planner hook (ROGUE_ONE_PLANNER_MODE=oracle|misdirected).

Done-when (A09-T5): under ``llm`` ``determine_focus`` is called; under ``oracle`` and
``misdirected`` it never is and the Extractor receives the file string for each
(fold, iteration); a missing key or a dataset mismatch raises. Plus the switch-off guard
(01 §2.2 guard 2): the default path passes exactly the Full-v2 arguments.
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

for _k, _v in {"EMBEDD_API_KEY": "x", "EMBEDD_MODEL": "x", "EMBEDD_PORT": "1", "LLM_MODEL": "x",
           "LLM_API_KEY": "x", "LLM_PORT": "1"}.items():
    os.environ.setdefault(_k, _v)

import pandas as pd
import pytest

_stubbed = importlib.util.find_spec("git") is None
if _stubbed:
    sys.modules["git"] = types.SimpleNamespace(Tree=object)
try:
    rom = importlib.import_module("rogue_one_main")
finally:
    if _stubbed:
        sys.modules.pop("git", None)
        sys.modules.pop("rogue_one_main", None)

from rogueone.utils import console_cfg, rouge_one_cfg
from rogueone.utils import instrumentation as ins
from rogueone.utils import planner_mode as pm

pytestmark = [pytest.mark.unit]

COLS = ["Attribute", "Description", "Pandas Command", "Status", "Added", "Pruned Iteration", "Prune Rationale"]
K, N = 5, 10


@pytest.fixture(autouse=True)
def _env(tmp_path):
    saved = (rouge_one_cfg.num_iterations, rouge_one_cfg.k_folds, console_cfg.disable,
             rouge_one_cfg.planner_mode, rouge_one_cfg.planner_focus_file, list(rouge_one_cfg.dataset_filter))
    rouge_one_cfg.num_iterations, rouge_one_cfg.k_folds = N, K
    console_cfg.disable = True
    ins.configure(tmp_path / "out")
    yield
    (rouge_one_cfg.num_iterations, rouge_one_cfg.k_folds, console_cfg.disable,
     rouge_one_cfg.planner_mode, rouge_one_cfg.planner_focus_file, rouge_one_cfg.dataset_filter) = saved
    ins.reset()


def _focus_doc(dataset="synp_p3", arm="oracle", fn=lambda f, i: f"focus f{f} i{i}"):
    return {"dataset_id": dataset, "arm_id": arm,
            "focus": {f"{f}:{i}": fn(f, i) for f in range(K) for i in range(N)}}


def _write(tmp_path, doc, name="focus.json"):
    p = Path(tmp_path) / name
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _plan(tmp_path, mode="oracle", **kw):
    p = _write(tmp_path, _focus_doc(arm=mode, **kw))
    return pm.load_focus_plan(mode, str(p), dataset_filter=["synp_p3"], k_folds=K, num_iterations=N)


def _network(tmp_path, plan=None, *, scientist_calls=None, extractor_focus=None):
    net = object.__new__(rom.RogueOneAgentNetwork)
    net.cfg = SimpleNamespace(output_dir=Path(tmp_path) / "out", experiment_name="e", task_type="classification")
    net.enable_wandb = False
    net.df = pd.DataFrame({"id": [1, 2], "v": [1.0, 2.0]})
    net.df_entities_attributes = pd.DataFrame({"id": [1, 2]})
    net.df_attributes_explanations = pd.DataFrame(columns=COLS)
    net.df_scientist_feedback_pool = pd.DataFrame()
    net.focus_history = []
    net.df_test_pool = None
    net._scored_before_test = None
    if plan is not ...:
        net._planner_focus_plan = plan

    async def determine_focus(*a, **k):
        if scientist_calls is None:
            raise AssertionError("the Scientist must not be called under a file planner mode")
        scientist_calls.append((a, k))
        return "llm focus string"

    async def generate(df, ents, expl, focus, step=None):
        extractor_focus.append((step, focus))
        return net.df_entities_attributes, pd.DataFrame()  # nothing produced -> skipped

    net.scientist_agent = SimpleNamespace(determine_focus=determine_focus, _note_book="")
    net.extractor_agent = SimpleNamespace(generate_patient_attributes=generate)
    return net


def _focus_rows(tmp_path):
    p = Path(tmp_path) / "out" / ins.FOCUS_FILE
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------- mode parsing
@pytest.mark.parametrize("raw,want", [(None, "llm"), ("", "llm"), ("  ", "llm"), ("llm", "llm"),
                                      ("LLM ", "llm"), ("oracle", "oracle"), ("Misdirected", "misdirected")])
def test_parse_planner_mode(raw, want):
    assert pm.parse_planner_mode(raw) == want


def test_parse_planner_mode_rejects_unknown_and_accepts_e9d_none():
    with pytest.raises(ValueError):
        pm.parse_planner_mode("generic")
    # E9d merged: ``none`` (P−) is implemented by fx_switches; it is not a file mode.
    assert pm.parse_planner_mode("none") == "none"
    assert "none" in pm.IMPLEMENTED_MODES and "none" not in pm.FILE_MODES


def test_config_reads_planner_env(monkeypatch):
    from rogueone.utils.config import rogueoneConfig

    def fresh():
        obj = object.__new__(rogueoneConfig)  # bypass the singleton
        rogueoneConfig.__init__(obj)
        return obj

    monkeypatch.delenv(pm.MODE_ENV, raising=False)
    monkeypatch.delenv(pm.FILE_ENV, raising=False)
    c = fresh()
    assert (c.planner_mode, c.planner_focus_file) == ("llm", None)
    monkeypatch.setenv(pm.MODE_ENV, "oracle")
    monkeypatch.setenv(pm.FILE_ENV, "/workspace/focus/focus.json")
    c = fresh()
    assert (c.planner_mode, c.planner_focus_file) == ("oracle", "/workspace/focus/focus.json")
    monkeypatch.setenv(pm.MODE_ENV, "bogus")
    with pytest.raises(ValueError):
        fresh()


# --------------------------------------------------------------------------- llm (switch off)
@pytest.mark.parametrize("plan_attr", [None, ...])  # explicit None, and attribute absent
def test_llm_mode_calls_determine_focus_with_full_v2_arguments(tmp_path, plan_attr):
    calls, seen = [], []
    net = _network(tmp_path, plan_attr, scientist_calls=calls, extractor_focus=seen)
    assert asyncio.run(net.forward_pass(13)) == "iteration_skipped"  # fold 1, iteration 3
    (args, kwargs), = calls
    assert args == (net.df, net.df_scientist_feedback_pool, net.df_attributes_explanations, net.focus_history)
    assert kwargs == {"index": 13, "max_index": N, "iteration_in_fold": 3}
    assert seen == [(13, "llm focus string")]
    assert net.focus_history == ["llm focus string"]
    (row,) = _focus_rows(tmp_path)
    assert row["source"] == "llm" and row["focus"] == "llm focus string"
    assert (row["fold_run_id"], row["iteration_in_fold"]) == (1, 3)


def test_llm_mode_init_loads_no_plan(monkeypatch):
    monkeypatch.setattr(rouge_one_cfg, "planner_mode", "llm")
    monkeypatch.setattr(rouge_one_cfg, "planner_focus_file", "/does/not/exist.json")
    assert pm.load_focus_plan("llm", "/does/not/exist.json", dataset_filter=["a", "b"],
                              k_folds=K, num_iterations=N) is None


# --------------------------------------------------------------------------- oracle / misdirected
@pytest.mark.parametrize("mode", ["oracle", "misdirected"])
def test_file_mode_never_calls_the_scientist_and_feeds_the_file_string(tmp_path, mode):
    plan = _plan(tmp_path, mode)
    seen = []
    net = _network(tmp_path, plan, scientist_calls=None, extractor_focus=seen)
    indices = [1, 2, 9, 11, 25, 49]
    for idx in indices:
        assert asyncio.run(net.forward_pass(idx)) == "iteration_skipped"
    assert seen == [(idx, f"focus f{idx // N} i{idx % N}") for idx in indices]
    rows = _focus_rows(tmp_path)
    assert [r["source"] for r in rows] == [mode] * len(indices)
    assert [r["focus"] for r in rows] == [s for _, s in seen]
    assert net.focus_history == [s for _, s in seen]
    assert ins.compute_summary().get("focus_total") == len(indices)


def test_oracle_same_string_every_iteration(tmp_path):
    plan = _plan(tmp_path, "oracle", fn=lambda f, i: "Build exactly this feature: oracle_feature = x.")
    seen = []
    net = _network(tmp_path, plan, extractor_focus=seen)
    for idx in (1, 2, 3, 17, 48):
        asyncio.run(net.forward_pass(idx))
    assert {s for _, s in seen} == {"Build exactly this feature: oracle_feature = x."}


# --------------------------------------------------------------------------- validation
def test_missing_key_raises(tmp_path):
    doc = _focus_doc()
    del doc["focus"]["4:9"]
    p = _write(tmp_path, doc)
    with pytest.raises(pm.PlannerFocusError, match="4:9"):
        pm.load_focus_plan("oracle", str(p), dataset_filter=["synp_p3"], k_folds=K, num_iterations=N)


def test_extra_key_raises(tmp_path):
    doc = _focus_doc()
    doc["focus"]["5:0"] = "x"
    p = _write(tmp_path, doc)
    with pytest.raises(pm.PlannerFocusError, match="5:0"):
        pm.load_focus_plan("oracle", str(p), dataset_filter=["synp_p3"], k_folds=K, num_iterations=N)


def test_dataset_mismatch_raises(tmp_path):
    p = _write(tmp_path, _focus_doc(dataset="synp_p4"))
    with pytest.raises(pm.PlannerFocusError, match="dataset_id"):
        pm.load_focus_plan("oracle", str(p), dataset_filter=["synp_p3"], k_folds=K, num_iterations=N)


def test_arm_mismatch_empty_string_and_dataset_count_raise(tmp_path):
    p = _write(tmp_path, _focus_doc(arm="misdirected"))
    with pytest.raises(pm.PlannerFocusError, match="arm_id"):
        pm.load_focus_plan("oracle", str(p), dataset_filter=["synp_p3"], k_folds=K, num_iterations=N)
    p = _write(tmp_path, _focus_doc(fn=lambda f, i: "" if (f, i) == (2, 2) else "x"))
    with pytest.raises(pm.PlannerFocusError, match="2:2"):
        pm.load_focus_plan("oracle", str(p), dataset_filter=["synp_p3"], k_folds=K, num_iterations=N)
    p = _write(tmp_path, _focus_doc())
    for filt in ([], ["synp_p3", "synp_p4"]):
        with pytest.raises(pm.PlannerFocusError, match="exactly one dataset"):
            pm.load_focus_plan("oracle", str(p), dataset_filter=filt, k_folds=K, num_iterations=N)


def test_missing_or_unset_file_raises(tmp_path):
    with pytest.raises(pm.PlannerFocusError):
        pm.load_focus_plan("oracle", None, dataset_filter=["synp_p3"], k_folds=K, num_iterations=N)
    with pytest.raises(pm.PlannerFocusError, match="does not exist"):
        pm.load_focus_plan("misdirected", str(tmp_path / "nope.json"), dataset_filter=["synp_p3"],
                           k_folds=K, num_iterations=N)
    bad = Path(tmp_path) / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(pm.PlannerFocusError, match="JSON"):
        pm.load_focus_plan("oracle", str(bad), dataset_filter=["synp_p3"], k_folds=K, num_iterations=N)


def test_network_init_validates_before_wandb_or_agents(tmp_path, monkeypatch):
    """A bad focus file stops __init__ before W&B, the data read or any agent is built."""
    doc = _focus_doc()
    del doc["focus"]["0:0"]
    p = _write(tmp_path, doc)
    monkeypatch.setattr(rouge_one_cfg, "planner_mode", "oracle")
    monkeypatch.setattr(rouge_one_cfg, "planner_focus_file", str(p))
    monkeypatch.setattr(rouge_one_cfg, "dataset_filter", ["synp_p3"])

    def boom(*a, **k):
        raise AssertionError("reached past focus-file validation")

    monkeypatch.setattr(rom, "init_wandb_run", boom)
    monkeypatch.setattr(rom, "ScientistAgent", boom)
    cfg = SimpleNamespace(output_dir=Path(tmp_path) / "out", experiment_name="e",
                          feature_file_path=Path(tmp_path) / "missing.csv")
    with pytest.raises(pm.PlannerFocusError, match="0:0"):
        rom.RogueOneAgentNetwork(cfg, enable_wandb=True)


def test_network_init_keeps_a_valid_plan(tmp_path, monkeypatch):
    p = _write(tmp_path, _focus_doc(arm="misdirected"))
    monkeypatch.setattr(rouge_one_cfg, "planner_mode", "misdirected")
    monkeypatch.setattr(rouge_one_cfg, "planner_focus_file", str(p))
    monkeypatch.setattr(rouge_one_cfg, "dataset_filter", ["synp_p3"])

    class Stop(Exception):
        pass

    def stop(*a, **k):
        raise Stop

    monkeypatch.setattr(rom.pd, "read_csv", stop)  # first step after validation + logging setup
    cfg = SimpleNamespace(output_dir=Path(tmp_path) / "out", experiment_name="e",
                          feature_file_path=Path(tmp_path) / "x.csv")
    net = object.__new__(rom.RogueOneAgentNetwork)
    with pytest.raises(Stop):
        net.__init__(cfg, enable_wandb=False)
    assert net._planner_focus_plan.mode == "misdirected"
    assert net._planner_focus_plan.focus_for(4, 9) == "focus f4 i9"
