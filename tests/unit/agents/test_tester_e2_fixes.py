"""E2 Tester fixes: Pruned Iteration = Trial <trial_index>, P4 score-after-pruning,
bounded executor output, and the h9 `_report_tool_error` return site in both modes."""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

for _k, _v in {"EMBEDD_API_KEY": "x", "EMBEDD_MODEL": "x", "EMBEDD_PORT": "1", "LLM_MODEL": "x"}.items():
    os.environ.setdefault(_k, _v)

import numpy as np
import pandas as pd
import pytest
from agents.tool_context import ToolContext

import rogueone.llm.agents.core.tester.agent as tst_mod
from rogueone.utils import rouge_one_cfg

pytestmark = [pytest.mark.unit, pytest.mark.agent]

FEATURES = ["f0", "f1", "f2", "f3"]


@pytest.fixture(autouse=True)
def _restore_flags():
    saved = (rouge_one_cfg.tool_error_feedback, rouge_one_cfg.tool_output_max_chars)
    yield
    rouge_one_cfg.tool_error_feedback, rouge_one_cfg.tool_output_max_chars = saved


def test_rebuild_after_pruning_slices_to_active_set():
    rng = np.random.default_rng(0)
    X_test_full = rng.normal(size=(5, 4))
    df_after = pd.DataFrame({"f0": [1.0, 2.0], "f2": [3.0, 4.0], "target": [0, 1]})
    X_prev = np.zeros((2, 3))
    X_tr, y_tr, X_te = tst_mod.TesterAgent._rebuild_after_pruning(
        df_after, X_test_full, FEATURES, X_prev, np.array([9, 9])
    )
    assert X_tr.shape == (2, 2) and list(y_tr) == [0, 1]
    assert X_te.shape == (5, 2)
    np.testing.assert_array_equal(X_te, X_test_full[:, [0, 2]])


def test_rebuild_after_pruning_keeps_previous_when_nothing_active():
    X_prev, y_prev, X_test = np.zeros((2, 3)), np.array([0, 1]), np.zeros((1, 3))
    out = tst_mod.TesterAgent._rebuild_after_pruning(
        pd.DataFrame({"target": [0, 1]}), X_test, FEATURES, X_prev, y_prev
    )
    assert out[0] is X_prev and out[1] is y_prev and out[2] is X_test


# --------------------------------------------------------------------------- end to end


def _make_folds(k=5, n=12, seed=0):
    rng = np.random.default_rng(seed)
    folds, next_id = [], 0
    for _ in range(k):
        d = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
        d["target"] = rng.integers(0, 2, size=n)
        d["id"] = range(next_id, next_id + n)
        next_id += n
        folds.append(d)
    return folds


def _make_tester(tmp_path):
    cfg = SimpleNamespace(
        output_dir=Path(tmp_path),
        experiment_description="d",
        context_and_goal="g",
        experiment_name="e",
        task_type="classification",
        modality="tabular",
    )
    agent = tst_mod.TesterAgent(cfg, include_report=False, enable_wandb=False)
    return agent


def _run_test_hypotheses(agent, monkeypatch, prune, expl, trial_index):
    """Run test_hypotheses, pruning ``prune`` during the pruning stage; return
    (result, expl, shapes scored by the inner-CV metric)."""
    scored = []
    real_inner = tst_mod._inner_cv_score

    def spy(X_train, *a, **k):
        scored.append(int(np.asarray(X_train).shape[1]))
        return real_inner(X_train, *a, **k)

    monkeypatch.setattr(tst_mod, "_inner_cv_score", spy)

    class FakeRunner:
        @staticmethod
        async def run(a, *args, **kwargs):
            tools = {t.name: t for t in a.tools}
            if "attribute_pruning_tool" in tools:
                payload = json.dumps({"attribute_names_list": prune})
                ctx = ToolContext(
                    context=None, tool_name="attribute_pruning_tool",
                    tool_call_id="c", tool_arguments=payload,
                )
                await tools["attribute_pruning_tool"].on_invoke_tool(ctx, payload)
            return SimpleNamespace(final_output="pruned")

    monkeypatch.setattr(tst_mod, "Runner", FakeRunner)
    res, out_expl = asyncio.run(
        agent.test_hypotheses(_make_folds(), expl, step=1, trial_index=trial_index)
    )
    return res, out_expl, scored


def _expl():
    return pd.DataFrame(
        {
            "Attribute": FEATURES,
            "Status": ["Active"] * len(FEATURES),
            "Pruned Iteration": [pd.NA] * len(FEATURES),
            "Prune Rationale": [pd.NA] * len(FEATURES),
        }
    )


def test_pruned_iteration_is_trial_index_and_scores_active_set(tmp_path, monkeypatch):
    agent = _make_tester(tmp_path)
    res, out, scored = _run_test_hypotheses(
        agent, monkeypatch, ["f1"], _expl(), trial_index=7
    )
    pruned = out[out["Status"] == "Pruned"]
    assert list(pruned["Attribute"]) == ["f1"]
    assert list(pruned["Pruned Iteration"]) == ["Trial 7"]  # not fold_run_id (step=1)
    active = int((out["Status"] != "Pruned").sum())
    assert active == 3
    # every scored matrix (inner CV is called once per outer fold) has the active count
    assert scored and set(scored) == {active}
    assert res.status == "completed"


def test_pruned_iteration_falls_back_to_step(tmp_path, monkeypatch):
    agent = _make_tester(tmp_path)
    _, out, _ = _run_test_hypotheses(agent, monkeypatch, ["f2"], _expl(), trial_index=None)
    assert list(out.loc[out["Status"] == "Pruned", "Pruned Iteration"]) == ["Trial 1"]


# --------------------------------------------------------------------------- h9 / executor


def test_tester_report_tool_error_and_executor_truncation(tmp_path, monkeypatch):
    """Capture the Tester tools from the assessment/pruning agents and exercise
    generic_python_executor_tool (truncation) and an exception path (h9)."""
    agent = _make_tester(tmp_path)
    captured = {}

    class FakeRunner:
        @staticmethod
        async def run(a, *args, **kwargs):
            tools = {t.name: t for t in a.tools}
            if "generic_python_executor_tool" in tools and "tools" not in captured:
                captured["tools"] = tools
                for key, code in (
                    ("big", "output = 'z' * 100000"),
                    ("boom", "raise ValueError('boom')"),
                ):
                    payload = json.dumps(
                        {"code_input": {"code": code, "reasoning": "r"}}
                    )
                    ctx = ToolContext(
                        context=None, tool_name="generic_python_executor_tool",
                        tool_call_id="c", tool_arguments=payload,
                    )
                    for mode, cap in (("message", 0), ("message", 8000), ("none", 0)):
                        rouge_one_cfg.tool_error_feedback = mode
                        rouge_one_cfg.tool_output_max_chars = cap
                        captured[(key, mode, cap)] = await tools[
                            "generic_python_executor_tool"
                        ].on_invoke_tool(ctx, payload)
            return SimpleNamespace(final_output="ok")

    monkeypatch.setattr(tst_mod, "Runner", FakeRunner)
    asyncio.run(agent.test_hypotheses(_make_folds(), _expl(), step=0, trial_index=0))

    assert len(captured[("big", "message", 0)]) >= 100_000
    cut = captured[("big", "message", 8000)]
    assert len(cut) < 8300 and "truncated" in cut
    assert "boom" in captured[("boom", "message", 0)]
    assert captured[("boom", "none", 0)] in (None, "None")
