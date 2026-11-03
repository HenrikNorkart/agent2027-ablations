"""E3 orchestration hooks in rogue_one_main: iteration context, focus log, iteration log,
per-trial tables (``Scored`` copied before the Tester), inner-CV scores in the trial row."""

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

import pandas as pd
import pytest

# rogue_one_main imports GitPython for an unused name. When it is missing (dev venv)
# stub it just for the import, and do not leave the stub (or the module) behind:
# test_fold_lifecycle.py skips itself when rogue_one_main cannot be imported.
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

pytestmark = [pytest.mark.unit]

COLS = ["Attribute", "Description", "Pandas Command", "Status", "Added", "Pruned Iteration", "Prune Rationale"]


@pytest.fixture(autouse=True)
def _env(tmp_path):
    saved = (rouge_one_cfg.num_iterations, console_cfg.disable)
    rouge_one_cfg.num_iterations = 10
    console_cfg.disable = True  # progress-bar calls assert a task id otherwise
    ins.configure(tmp_path / "out")
    yield
    rouge_one_cfg.num_iterations, console_cfg.disable = saved
    ins.reset()


def _network(tmp_path):
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
    return net


def _read(tmp_path, name):
    p = Path(tmp_path) / "out" / name
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_forward_pass_sets_iteration_context_and_logs_iteration(tmp_path):
    net = _network(tmp_path)
    seen = {}

    async def impl(index):
        seen["ctx"] = ins.current_iteration()
        net.df_attributes_explanations = pd.DataFrame(
            {"Attribute": ["a", "b", "c"], "Status": ["Active", "Pruned", "Active"]}
        )
        return "completed"

    net._forward_pass = impl
    status = asyncio.run(net.forward_pass(23))
    assert status == "completed"
    assert seen["ctx"] == {"index": 23, "fold_run_id": 2, "iteration_in_fold": 3}
    (row,) = _read(tmp_path, ins.ITERATION_FILE)
    assert (row["index"], row["fold_run_id"], row["iteration_in_fold"]) == (23, 2, 3)
    assert row["status"] == "completed"
    assert row["active_features"] == 2 and row["pruned_features"] == 1 and row["pool_rows"] == 3


def test_iteration_is_logged_even_when_the_pass_raises(tmp_path):
    net = _network(tmp_path)

    async def boom(index):
        raise RuntimeError("x")

    net._forward_pass = boom
    with pytest.raises(RuntimeError):
        asyncio.run(net.forward_pass(1))
    (row,) = _read(tmp_path, ins.ITERATION_FILE)
    assert row["status"] is None and row["index"] == 1


def test_real_forward_pass_logs_focus_and_stage_timers(tmp_path):
    """Drive the real _forward_pass down the cheap ``iteration_skipped`` path."""
    net = _network(tmp_path)

    async def determine_focus(*a, **k):
        assert ins._CALLER.get() == "scientist"
        return "try ratios"

    async def generate(*a, **k):
        assert ins._CALLER.get() == "extractor"
        return net.df_entities_attributes, pd.DataFrame()  # nothing produced

    net.scientist_agent = SimpleNamespace(determine_focus=determine_focus, _note_book="")
    net.extractor_agent = SimpleNamespace(generate_patient_attributes=generate)
    ins.note_focus_source("heuristic")  # as the Scientist's fallback would
    status = asyncio.run(net.forward_pass(1))  # fold 0, iteration 1 -> iteration_skipped
    assert status == "iteration_skipped"
    (focus,) = _read(tmp_path, ins.FOCUS_FILE)
    assert focus["focus"] == "try ratios" and focus["source"] == "heuristic"
    assert (focus["fold_run_id"], focus["iteration_in_fold"], focus["index"]) == (0, 1, 1)
    s = ins.compute_summary()
    assert s["focus_total"] == 1 and s["fallback_scientist_focus"] == 1
    assert s["iterations_total"] == 1
    assert s["wall_s_scientist"] >= 0 and s["wall_s_extractor"] >= 0 and s["wall_s_tester"] == 0


def test_focus_source_defaults_to_planner_mode_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ROGUE_ONE_PLANNER_MODE", "oracle")
    net = _network(tmp_path)

    async def determine_focus(*a, **k):
        return "f"

    async def generate(*a, **k):
        return net.df_entities_attributes, pd.DataFrame()

    net.scientist_agent = SimpleNamespace(determine_focus=determine_focus, _note_book="")
    net.extractor_agent = SimpleNamespace(generate_patient_attributes=generate)
    asyncio.run(net.forward_pass(1))
    assert _read(tmp_path, ins.FOCUS_FILE)[0]["source"] == "oracle"


def test_trial_tables_scored_is_copied_before_the_tester_mutates_the_table(tmp_path):
    net = _network(tmp_path)
    net.df_attributes_explanations = pd.DataFrame(
        {
            "Attribute": ["a", "b", "c"],
            "Description": ["", "", ""],
            "Pandas Command": ["r=1", "r=2", "r=3"],
            "Status": ["Active", "Pruned", "Active"],  # b pruned in an earlier iteration
        }
    )
    snapshot = net._snapshot_scored()
    assert snapshot == {"a": 1, "b": 0, "c": 1}
    net._scored_before_test = snapshot
    # the Tester prunes ``c`` in place during test_hypotheses
    net.df_attributes_explanations.loc[net.df_attributes_explanations["Attribute"] == "c", "Status"] = "Pruned"
    ins.note_fallback_features(["a"])
    net.df_scientist_feedback_pool = pd.DataFrame({"id": ["Trial 13"], "feature_assessment": ["ok"]})

    asyncio.run(net.dump_data(13))
    d = Path(tmp_path) / "out" / "intermediate_steps"
    assert (d / "df_entities_attributes_trail_13.csv").exists()
    attrs = pd.read_csv(d / "df_attributes_explanations_trail_13.csv").set_index("Attribute")
    assert list(attrs.index) == ["a", "b", "c"]  # every row, pruned included
    assert list(attrs["Status"]) == ["Active", "Pruned", "Pruned"]  # status after pruning
    assert list(attrs["Scored"]) == [1, 0, 1]  # not Pruned *before* test_hypotheses
    assert list(attrs["Scored_post_prune"]) == [1, 0, 0]  # what the P4 fix actually scores
    assert list(attrs["Fallback"]) == [1, 0, 0]
    assert list(attrs["Seeded"]) == [0, 0, 0]  # default until E6 sets the marker
    assert (attrs["fold_run_id"] == 1).all() and (attrs["iteration_in_fold"] == 3).all()
    fb = pd.read_csv(d / "df_scientist_feedback_trail_13.csv")
    assert list(fb["id"]) == ["Trial 13"]


def test_trial_tables_written_on_paths_without_a_tester_run(tmp_path):
    net = _network(tmp_path)
    net.df_attributes_explanations = pd.DataFrame(
        {"Attribute": ["a"], "Pandas Command": ["r=1"], "Status": ["Active"]}
    )
    net._scored_before_test = None  # aborted / skipped trial
    asyncio.run(net.dump_data(0))
    d = Path(tmp_path) / "out" / "intermediate_steps"
    attrs = pd.read_csv(d / "df_attributes_explanations_trail_0.csv")
    assert list(attrs["Scored"]) == [0] and list(attrs["Scored_post_prune"]) == [0]
    assert (d / "df_scientist_feedback_trail_0.csv").exists()


def test_seeded_column_is_kept_when_present(tmp_path):
    net = _network(tmp_path)
    net.df_attributes_explanations = pd.DataFrame(
        {"Attribute": ["a"], "Pandas Command": ["r=1"], "Status": ["Active"], "Seeded": [True]}
    )
    asyncio.run(net.dump_data(0))
    attrs = pd.read_csv(Path(tmp_path) / "out" / "intermediate_steps" / "df_attributes_explanations_trail_0.csv")
    assert bool(attrs["Seeded"].iloc[0]) is True


def test_reset_fold_state_clears_fallback_names(tmp_path):
    net = _network(tmp_path)
    net.scientist_agent = SimpleNamespace(_note_book="x")
    ins.note_fallback_features(["a"])
    net._reset_fold_state()
    assert ins.fallback_feature_names() == set()
