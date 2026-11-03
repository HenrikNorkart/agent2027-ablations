"""E3 agent hooks: A07 bypass-site records and counters, fallback / floor / report-read
counters, Runner-call counter, per-inner-fold scores, Tester stage context.

The agents are driven with a fake ``Runner`` exactly as in test_tool_error_feedback.py.
"""

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

import rogueone.llm.agents.core.extractor.agent as ext_mod
import rogueone.llm.agents.core.scientist.agent as sci_mod
import rogueone.llm.agents.core.tester.agent as tst_mod
from rogueone.llm.agents.core.tool_error_reporting import ToolErrorReporter, report_bypass
from rogueone.utils import console_cfg, rouge_one_cfg
from rogueone.utils import instrumentation as ins
from tests.unit.agents.test_tool_error_feedback import (
    _cfg,
    _extractor_sites,
    _inv,
    _make_extractor,
    _make_scientist,
    _run_extractor,
    _run_scientist,
    _tools_by_name,
)

pytestmark = [pytest.mark.unit, pytest.mark.agent]


@pytest.fixture(autouse=True)
def _clean(tmp_path, monkeypatch):
    saved = (rouge_one_cfg.tool_error_feedback, rouge_one_cfg.extractor_rounds, console_cfg.disable)
    ins.configure(Path(tmp_path) / "out")
    yield
    rouge_one_cfg.tool_error_feedback, rouge_one_cfg.extractor_rounds, console_cfg.disable = saved
    ins.reset()


def _records(tmp_path):
    path = Path(tmp_path) / "agent_toolcall_errors.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _sites(tmp_path):
    return sorted(r["context"]["site"] for r in _records(tmp_path) if r["context"])


# --------------------------------------------------------------------------- reporter


def test_reporter_default_site_by_agent_and_counters(tmp_path):
    for agent, site in (("ScientistAgent", "h1"), ("ExtractorAgent", "h4"), ("TesterAgent", "h9")):
        ToolErrorReporter(agent_name=agent, output_dir=tmp_path).report(
            tool_name="generic_pandas_tool", message="m", exception=ValueError("x"),
            context={"code": "c"},
        )
    recs = _records(tmp_path)
    assert [r["context"]["site"] for r in recs] == ["h1", "h4", "h9"]
    assert recs[0]["context"]["code"] == "c"  # caller context preserved
    s = ins.compute_summary()
    assert (s["err_site_h1"], s["err_site_h4"], s["err_site_h9"], s["n_tool_errors"]) == (1, 1, 1, 3)


def test_runner_errors_are_not_tool_errors(tmp_path):
    ToolErrorReporter(agent_name="ExtractorAgent", output_dir=tmp_path).report(
        tool_name="runner", message="m", exception=RuntimeError("x"), context={"session_name": "s"}
    )
    rec = _records(tmp_path)[0]
    assert "site" not in rec["context"]
    s = ins.compute_summary()
    assert s["n_tool_errors"] == 0 and s["runner_errors"] == 1


def test_report_bypass_writes_one_record_and_counts(tmp_path):
    reporter = ToolErrorReporter(agent_name="ExtractorAgent", output_dir=tmp_path)
    report_bypass(reporter, site="h8", tool_name="append_new_attribute", message="nan!", kind="nan")
    report_bypass(None, site="v1", tool_name="append_new_attribute", message="dup", kind="duplicate",
                  agent_name="ExtractorAgent")  # no reporter: still counted
    assert _sites(tmp_path) == ["h8"]
    s = ins.compute_summary()
    assert s["err_site_h8"] == 1 and s["err_site_v1"] == 1 and s["silent_rejections"] == 2


def test_repeat_rate_normalises_digits_and_quoted_names(tmp_path):
    ins.set_iteration(0, 0, 0)
    r = ToolErrorReporter(agent_name="ExtractorAgent", output_dir=tmp_path)
    for msg in ("Attribute 'a1' failed at row 3", "Attribute 'b2' failed at row 99", "other problem"):
        r.report(tool_name="t", message=msg, exception=ValueError(msg))
    s = ins.compute_summary()
    assert s["n_tool_errors"] == 3 and s["err_repeat_rate"] == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- extractor


async def _extractor_sites_plus(tools):
    """The E2 site scenario plus two more reachable non-numeric rejections."""
    out = await _extractor_sites(tools)

    async def append(name, cmd):
        return await _inv(
            tools, "append_new_attribute",
            args={"name": name, "command": cmd, "argumentation": "a"},
        )

    out["h8_scalar_string"] = await append("n4", "result = 'abc'")
    out["h8_bool_array"] = await append("n5", "result = (df_features['id'] > 0).values")
    return out


@pytest.mark.parametrize("mode", ["message", "none"])
@pytest.mark.parametrize("disabled", [False, True])
def test_extractor_bypass_sites_each_one_record(tmp_path, monkeypatch, mode, disabled):
    console_cfg.disable = disabled
    rouge_one_cfg.tool_error_feedback = mode
    _run_extractor(_make_extractor(tmp_path), monkeypatch, _extractor_sites_plus)
    # h5, h6, h7, h4 x2 (a raised exception and a str array: the isnan check raises, as
    # before E3), h8 x4 (no result, NaN, scalar string, bool array), v1 (duplicate)
    assert _sites(tmp_path) == ["h4", "h4", "h5", "h6", "h7", "h8", "h8", "h8", "h8", "v1"]
    s = ins.compute_summary()
    assert s["err_site_h8"] == 4 and s["err_site_v1"] == 1 and s["n_tool_errors"] == 10
    assert s["silent_rejections"] == 4  # NaN + 2 non-numeric + duplicate (4.6)
    assert s["append_attempts"] == 6 and s["append_accepted"] == 0 and s["append_success"] == 0.0
    assert s["err_role_extractor"] == 10


def test_extractor_append_accepted_and_restored(tmp_path, monkeypatch):
    async def scenario(tools):
        ok = await _inv(tools, "append_new_attribute", args={"name": "new1", "command": "result = df_features['id'].values * 2.0", "argumentation": "a"})
        back = await _inv(tools, "append_new_attribute", args={"name": "known", "command": "x", "argumentation": "a"})
        return ok, back

    agent = _make_extractor(tmp_path)

    class FakeRunner:
        calls = 0

        async def run(self, a, *args, **kwargs):
            FakeRunner.calls += 1
            if FakeRunner.calls == 1:
                FakeRunner.res = await scenario(_tools_by_name(a))
            return SimpleNamespace(final_output="")

    monkeypatch.setattr(ext_mod, "Runner", FakeRunner)
    rouge_one_cfg.extractor_rounds = 3
    df = pd.DataFrame({"id": [1, 1, 2, 2], "v": [1.0, 2.0, 3.0, 4.0]})
    feats = pd.DataFrame({"id": [1, 2], "known": [0.0, 1.0]})
    expl = pd.DataFrame({"Attribute": ["known"], "Status": ["Pruned"]})
    asyncio.run(agent.generate_patient_attributes(df, feats, expl, "focus", step=0))
    s = ins.compute_summary()
    assert s["append_attempts"] == 2 and s["append_accepted"] == 1 and s["append_restored"] == 1
    assert s["restore_pruned_count"] == 1 and s["append_success"] == 0.5
    assert s["n_tool_errors"] == 0
    assert s["runner_runs_extractor"] == 3  # E2 done-when: R=3 gives exactly 3 Runner.run calls
    assert s["fallback_extractor"] == 0


def test_extractor_fallback_counters_and_names(tmp_path, monkeypatch):
    rouge_one_cfg.extractor_rounds = 1
    _run_extractor(_make_extractor(tmp_path), monkeypatch, None)  # no append at all
    s = ins.compute_summary()
    assert s["fallback_extractor"] == 1 and s["fallback_extractor_features"] >= 1
    assert s["fallback_share_extractor"] == 1.0
    assert "v" in ins.fallback_feature_names()


# --------------------------------------------------------------------------- scientist


def test_scientist_bypass_sites_and_report_reads(tmp_path, monkeypatch):
    async def scenario(tools):
        async def pandas(cmd):
            return await _inv(tools, "generic_pandas_tool", command=cmd)

        return {
            "h2": await _inv(tools, "search_tool", query="q"),
            "h3": await pandas("y = df_raw_data.shape"),  # sets no `result`
            "h1": await pandas("result = df_raw_data['nope']"),  # raises
            "read": await pandas("result = 'feature_assessment' in str(df_test_results.columns)"),
        }

    rouge_one_cfg.tool_error_feedback = "message"
    state = _run_scientist(_make_scientist(tmp_path), monkeypatch, scenario, 0)
    assert "results" in state, state.get("error")
    assert _sites(tmp_path) == ["h1", "h2", "h3"]
    s = ins.compute_summary()
    assert s["err_site_h2"] == 1 and s["err_site_h3"] == 1 and s["err_site_h1"] == 1
    # 3 Scientist pandas calls (h3, h1, read) -> denominator 3, one of them reads the report
    assert s["scientist_pandas_calls"] == 3 and s["report_reads"] == 1
    assert s["report_read_rate"] == pytest.approx(1 / 3)
    assert s["runner_runs_scientist"] >= 1


def test_scientist_heuristic_focus_source(tmp_path, monkeypatch):
    agent = _make_scientist(tmp_path)

    class EmptyRunner:
        async def run(self, a, prompt, **kwargs):
            return SimpleNamespace(final_output="")

    monkeypatch.setattr(sci_mod, "Runner", EmptyRunner)
    df = pd.DataFrame({"id": [1, 2], "v": [1.0, 2.0]})
    expl = pd.DataFrame({"Attribute": ["v"], "Status": ["Active"]})
    focus = asyncio.run(agent.determine_focus(df, None, expl, [], index=0, max_index=10, iteration_in_fold=0))
    assert focus  # heuristic text
    assert ins.pop_focus_source() == "heuristic"
    assert ins.pop_focus_source() == "llm"  # consumed


# --------------------------------------------------------------------------- tester

FEATURES = ["f0", "f1", "f2", "f3"]


def _folds(k=5, n=12, seed=0):
    rng = np.random.default_rng(seed)
    out, nid = [], 0
    for _ in range(k):
        d = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
        d["target"] = rng.integers(0, 2, size=n)
        d["id"] = range(nid, nid + n)
        nid += n
        out.append(d)
    return out


def _expl():
    return pd.DataFrame(
        {
            "Attribute": FEATURES,
            "Status": ["Active"] * 4,
            "Pruned Iteration": [pd.NA] * 4,
            "Prune Rationale": [pd.NA] * 4,
        }
    )


def _tester(tmp_path, include_report):
    cfg = SimpleNamespace(
        output_dir=Path(tmp_path), experiment_description="d", context_and_goal="g",
        experiment_name="e", task_type="classification", modality="tabular",
    )
    return tst_mod.TesterAgent(cfg, include_report=include_report, enable_wandb=False)


def _run_tester(agent, monkeypatch, prune, include_stage_probe=None, final_output="ok", block_prune_gate=False):
    seen = []

    class FakeRunner:
        @staticmethod
        async def run(a, *args, **kwargs):
            seen.append((ins._STAGE.get(), ins._SESSION.get()))
            tools = {t.name: t for t in a.tools}
            if "attribute_pruning_tool" in tools:
                if block_prune_gate:
                    agent._assessment_report_produced = False
                payload = json.dumps({"attribute_names_list": prune})
                ctx = ToolContext(context=None, tool_name="attribute_pruning_tool", tool_call_id="c", tool_arguments=payload)
                await tools["attribute_pruning_tool"].on_invoke_tool(ctx, payload)
            return SimpleNamespace(final_output=final_output)

    monkeypatch.setattr(tst_mod, "Runner", FakeRunner)
    res, out = asyncio.run(agent.test_hypotheses(_folds(), _expl(), step=1, trial_index=4))
    return res, out, seen


def test_tester_inner_cv_scores_recorded_and_mean_matches(tmp_path, monkeypatch):
    res, _out, _ = _run_tester(_tester(tmp_path, False), monkeypatch, ["f1"])
    scores = ins.pop_inner_cv_scores()
    assert scores is not None and 2 <= len(scores) <= 4
    assert all(np.isfinite(scores))
    assert abs(float(np.mean(scores)) - res.inner_cv_accuracy) < 1e-9
    assert ins.pop_inner_cv_scores() is None  # consumed


def test_tester_floor_activation_counted(tmp_path, monkeypatch):
    _res, out, _ = _run_tester(_tester(tmp_path, False), monkeypatch, FEATURES)  # prune everything
    s = ins.compute_summary()
    assert s["floor_activations"] == 1 and s["floor_restored_features"] == 3
    assert int((out["Status"] != "Pruned").sum()) == 3


def test_tester_stage_context_and_runner_counter(tmp_path, monkeypatch):
    _res, _out, seen = _run_tester(_tester(tmp_path, True), monkeypatch, ["f1"])
    assert [st for st, _ in seen] == ["assessment", "pruning"]
    assert seen[0][1].startswith("tester_agent_assessment_step_")
    assert seen[1][1].startswith("tester_agent_pruning_step_")
    s = ins.compute_summary()
    assert s["runner_runs_tester"] == 2 and s["tester_assessments"] == 1
    assert s["fallback_tester_report"] == 0


def test_tester_fallback_report_counted(tmp_path, monkeypatch):
    _run_tester(_tester(tmp_path, True), monkeypatch, [], final_output="")
    s = ins.compute_summary()
    assert s["fallback_tester_report"] == 1 and s["tester_assessments"] == 1
    assert s["fallback_share_tester"] == 1.0


def test_tester_pruning_blocked_v3_one_record(tmp_path, monkeypatch):
    rouge_one_cfg.tool_error_feedback = "message"
    _run_tester(_tester(tmp_path, False), monkeypatch, ["f1"], block_prune_gate=True)
    assert _sites(tmp_path) == ["v3"]
    assert ins.compute_summary()["err_site_v3"] == 1
