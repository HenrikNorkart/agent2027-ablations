"""E3 scripts: energy sampler parsing/loop, energy apportionment (A23 2.3), run-output
fetch/check, and finalize_run (summary JSON, W&B summary, ``run-output`` artifact)."""

import asyncio
import gzip
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from rogueone.utils import instrumentation as ins
from scripts.agent2027 import energy_sampler as es
from scripts.agent2027.analysis import energy_apportion as ea
from scripts.agent2027.analysis import fetch_wandb as fw

pytestmark = [pytest.mark.unit]


@pytest.fixture(autouse=True)
def _clean_instrumentation(monkeypatch):
    monkeypatch.delenv(ins.TRACE_FLAG, raising=False)
    yield
    ins.reset()


# --------------------------------------------------------------------------- sampler


def test_parse_smi_output_skips_unsupported_lines():
    text = "0, 312.45, GPU-aaa\n1, [N/A], GPU-bbb\n2, 98.10, GPU-ccc\nnonsense\n"
    assert es.parse_smi_output(text) == [(0, 312.45, "GPU-aaa"), (2, 98.10, "GPU-ccc")]


def test_parse_gpu_list():
    assert es.parse_gpu_list("0, 1,3") == [0, 1, 3] and es.parse_gpu_list("") == []


def test_trapezoid_energy():
    assert es.trapezoid_energy_j([(0.0, 100.0), (10.0, 100.0)]) == 1000.0
    assert es.trapezoid_energy_j([(0.0, 0.0), (10.0, 100.0), (20.0, 100.0)]) == pytest.approx(500 + 1000)
    assert es.trapezoid_energy_j([(0.0, 5.0)]) == 0.0


def test_sampler_loop_writes_csv_at_one_hertz(tmp_path):
    now = {"t": 1000.0}

    class FakeBackend:
        name = "fake"

        def read(self):
            return [(0, 200.0, "GPU-a"), (1, 300.0, "GPU-b")]

    def clock():
        return now["t"]

    def sleep(d):
        now["t"] += d

    out = tmp_path / "S__energy.csv"
    stats = es.run_sampler(FakeBackend(), out, interval=1.0, max_samples=11, clock=clock, sleep=sleep)
    rows = out.read_text().splitlines()
    assert rows[0] == "t_epoch,gpu_index,power_w,uuid" and len(rows) == 1 + 11 * 2
    assert stats["samples"] == 11 and stats["gpus"] == [0, 1]
    assert stats["median_interval_s"] == pytest.approx(1.0)
    assert stats["energy_j_gpu0"] == pytest.approx(2000.0)  # 200 W * 10 s
    assert stats["energy_j_total_sampled"] == pytest.approx(5000.0)


def test_sampler_survives_a_failed_read(tmp_path):
    calls = {"n": 0}

    class Flaky:
        name = "flaky"

        def read(self):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("nvml hiccup")
            return [(0, 100.0, "u")]

    now = {"t": 0.0}
    stats = es.run_sampler(Flaky(), tmp_path / "o.csv", max_samples=4, clock=lambda: now["t"],
                           sleep=lambda d: now.__setitem__("t", now["t"] + d))
    assert stats["samples"] == 4
    assert len((tmp_path / "o.csv").read_text().splitlines()) == 1 + 3  # one tick missed


def test_smi_backend_builds_query_and_parses():
    seen = {}

    def runner(cmd):
        seen["cmd"] = cmd
        return "0, 100.0, GPU-a\n"

    b = es.SmiBackend([0, 2], runner=runner)
    assert b.read() == [(0, 100.0, "GPU-a")]
    assert seen["cmd"][-2:] == ["-i", "0,2"]


# --------------------------------------------------------------------------- apportionment


def _power(gpus, t_from, t_to, watts=100.0, skip=()):
    return {
        g: [(float(t), watts) for t in range(t_from, t_to + 1) if not any(a <= t < b for a, b in skip)]
        for g in gpus
    }


def test_gpu_bin_energy_splits_a_segment_at_bin_edges():
    e = ea.gpu_bin_energy([(0.0, 0.0), (120.0, 120.0)], t0=0.0, bin_s=60.0, nbins=2)
    assert e == [pytest.approx(1800.0), pytest.approx(5400.0)] and sum(e) == pytest.approx(7200.0)


def test_apportion_by_token_share_and_equal_split_otherwise():
    power = _power([0, 1], 0, 300)  # 2 GPUs x 100 W x 300 s = 60 kJ on one server
    runs = [ea.Run("A", 0, 300, ("s1",)), ea.Run("B", 0, 300, ("s1",))]
    calls = [ea.Call("A", "s1", t_end=30.0, compl_tokens=75), ea.Call("B", "s1", t_end=40.0, compl_tokens=25)]
    r = ea.apportion_energy(power, {0: "s1", 1: "s1"}, runs, calls)
    assert r.session_energy_j == pytest.approx(60_000)
    # bin 0 (12 kJ) is split 75/25, the other four bins equally among the alive runs
    assert r.run_energy_j["A"] == pytest.approx(9000 + 24000)
    assert r.run_energy_j["B"] == pytest.approx(3000 + 24000)
    assert r.overhead_j == 0 and abs(r.residual_j) < 1e-6
    eq = ea.apportion_energy(power, {0: "s1", 1: "s1"}, runs, calls, equal_split=True)
    assert eq.run_energy_j["A"] == pytest.approx(30_000) == pytest.approx(eq.run_energy_j["B"])


def test_apportion_overhead_when_no_run_is_alive():
    power = _power([0], 0, 300)
    r = ea.apportion_energy(power, {0: "s"}, [ea.Run("A", 0, 120, ("s",))], [])
    assert r.run_energy_j["A"] == pytest.approx(12_000)  # bins 0 and 1
    assert r.overhead_j == pytest.approx(18_000) and abs(r.residual_j) < 1e-6


def test_tokens_spread_uniformly_over_call_interval():
    c = ea.Call("A", "s", t_start=30.0, t_end=90.0, compl_tokens=60)
    assert ea.spread_tokens(c, t0=0.0, bin_s=60.0, nbins=3) == {0: pytest.approx(30), 1: pytest.approx(30)}
    c2 = ea.Call("A", "s", t_end=95.0, compl_tokens=10)  # no start: the bin holding the end
    assert ea.spread_tokens(c2, 0.0, 60.0, 3) == {1: 10}


def test_mixed_run_sums_over_servers_and_servers_are_independent():
    power = {**_power([0], 0, 120, watts=100.0), **_power([1], 0, 120, watts=50.0)}
    runs = [ea.Run("mix", 0, 120, ("s1", "s2")), ea.Run("solo", 0, 120, ("s1",))]
    calls = [ea.Call("mix", "s1", 10.0, 10), ea.Call("solo", "s1", 20.0, 30), ea.Call("mix", "s2", 70.0, 5)]
    r = ea.apportion_energy(power, {0: "s1", 1: "s2"}, runs, calls)
    s1_bin0, s1_bin1 = 6000.0, 6000.0
    s2_bin0, s2_bin1 = 3000.0, 3000.0
    expect_mix = s1_bin0 * 0.25 + s1_bin1 * 0.5 + s2_bin0 * 1.0 + s2_bin1 * 1.0
    # s1 bin1 has no tokens -> equal split between the two alive runs; s2 bin0 has no
    # tokens -> only ``mix`` uses s2; s2 bin1 has the 5 tokens of mix
    assert r.run_energy_j["mix"] == pytest.approx(expect_mix)
    assert r.run_energy_j["solo"] == pytest.approx(s1_bin0 * 0.75 + s1_bin1 * 0.5)
    assert abs(r.residual_j) < 1e-6


def test_coverage_requires_every_gpu_of_the_servers_within_two_seconds():
    power = _power([0, 1], 0, 100, skip=[(40, 60)])  # both GPUs silent for 20 s
    runs = [ea.Run("A", 0, 100, ("s",))]
    r = ea.apportion_energy(power, {0: "s", 1: "s"}, runs, [])
    # samples at 39 and 60 cover [37,41] and [58,62]: the hole is (41, 58) = 17 s
    assert r.coverage["A"] == pytest.approx(1 - 17 / 100)
    assert r.valid["A"] is False
    full = ea.apportion_energy(_power([0], 0, 100), {0: "s"}, runs, [])
    assert full.coverage["A"] == 1.0 and full.valid["A"] is True


def test_calls_from_trace_charges_search_to_the_callers_server(tmp_path):
    p = tmp_path / "t.jsonl.gz"
    recs = [
        {"role": "scientist", "ts_start": 1.0, "ts_end": 3.0, "usage": {"completion_tokens": 10},
         "discarded_usage": {"completion_tokens": 4}},
        {"role": "tester", "ts_start": 4.0, "ts_end": 5.0, "usage": {"completion_tokens": 7}},
        {"role": "unattributed", "ts_start": 4.0, "ts_end": 5.0, "usage": {"completion_tokens": 7}},
    ]
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    calls = ea.calls_from_trace(p, "run1", {"scientist": "big", "tester": "small"})
    assert [(c.server, c.compl_tokens) for c in calls] == [("big", 14.0), ("small", 7.0)]


def test_read_power_csv_roundtrip(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text("t_epoch,gpu_index,power_w,uuid\n2.0,0,10,u\n1.0,0,20,u\n1.0,1,5,v\nbad,row\n")
    assert ea.read_power_csv(p) == {0: [(1.0, 20.0), (2.0, 10.0)], 1: [(1.0, 5.0)]}


# --------------------------------------------------------------------------- run-output


def _resp(prompt, compl):
    msg = SimpleNamespace(content=None, tool_calls=[])
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="stop", message=msg)],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=compl, total_tokens=prompt + compl,
                              completion_tokens_details=None, prompt_tokens_details=None),
        model="m",
    )


def _call(agent, prompt=10, compl=5):
    async def fn(*a, **k):
        return _resp(prompt, compl)

    return asyncio.run(
        ins.traced_call(agent, fn, (), {"model": "m", "messages": [], "tools": [], "temperature": 0.7})
    )


def _make_run_output(out: Path, monkeypatch):
    monkeypatch.setenv(ins.TRACE_FLAG, "1")
    ins.configure(out)
    for i, role in enumerate(["ScientistAgent", "ExtractorAgent", "TesterAgent"]):
        ins.set_iteration(i, 0, i)
        _call(role)
        ins.log_focus(i, 0, i, f"focus {i}", "llm")
        ins.log_iteration(i, 0, i, "completed", {"active_features": i})
        steps = out / "intermediate_steps"
        steps.mkdir(exist_ok=True)
        (steps / f"df_attributes_explanations_trail_{i}.csv").write_text("Attribute\n")
        (steps / f"df_scientist_feedback_trail_{i}.csv").write_text("id\n")
    return ins.finalize_run(out, upload=False)


def test_finalize_writes_summary_and_closed_trace(tmp_path, monkeypatch):
    out = tmp_path / "output"
    summary = _make_run_output(out, monkeypatch)
    assert summary["llm_calls_total"] == 3 and summary["tok_compl_total"] == 15
    assert json.loads((out / ins.SUMMARY_FILE).read_text())["tok_compl_extractor"] == 5
    with gzip.open(out / ins.TRACE_FILE, "rt") as fh:  # cleanly closed gzip
        assert len(fh.read().strip().splitlines()) == 3


def test_run_output_reads_and_passes_the_e3_contract(tmp_path, monkeypatch):
    out = tmp_path / "output"
    _make_run_output(out, monkeypatch)
    data = fw.read_run_output(out)
    assert len(data["traces"]) == 3 and len(data["focus"]) == 3 and len(data["iterations"]) == 3
    assert sorted(data["trial_tables"]) == [0, 1, 2]
    assert fw.check_run_output(data) == []
    assert fw.main(["--read", str(out), "--check"]) == 0


def test_check_flags_missing_pieces(tmp_path, monkeypatch):
    out = tmp_path / "output"
    _make_run_output(out, monkeypatch)
    # drop one trace record: count no longer equals llm_calls_total
    with gzip.open(out / ins.TRACE_FILE, "rt") as fh:
        lines = fh.read().splitlines()
    with gzip.open(out / ins.TRACE_FILE, "wt") as fh:
        fh.write("\n".join(lines[:-1]) + "\n")
    problems = fw.check_run_output(fw.read_run_output(out))
    assert any("llm_calls_total" in p for p in problems)
    (out / ins.FOCUS_FILE).unlink()
    assert any("focus_log" in p for p in fw.check_run_output(fw.read_run_output(out)))


def test_check_flags_zero_role_tokens(tmp_path, monkeypatch):
    out = tmp_path / "output"
    monkeypatch.setenv(ins.TRACE_FLAG, "1")
    ins.configure(out)
    ins.set_iteration(0, 0, 0)
    _call("ExtractorAgent")
    ins.finalize_run(out, upload=False)
    problems = fw.check_run_output(fw.read_run_output(out))
    assert "tok_compl_scientist is zero" in problems and "tok_compl_tester is zero" in problems


def test_finalize_logs_summary_and_run_output_artifact_to_wandb(tmp_path, monkeypatch):
    out = tmp_path / "output"
    calls = {"added": [], "logged": []}

    class Artifact:
        def __init__(self, name, type):
            self.name, self.type = name, type

        def add_dir(self, d):
            calls["added"].append(d)

    fake = types.ModuleType("wandb")
    fake.run = SimpleNamespace(summary={})
    fake.Artifact = Artifact
    fake.log_artifact = lambda a: calls["logged"].append((a.name, a.type))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    _make_run_output(out, monkeypatch)  # upload=False here
    ins.finalize_run(out, upload=True)
    assert fake.run.summary["tok_compl_total"] == 15 and fake.run.summary["llm_calls_total"] == 3
    assert calls["added"] == [str(out)] and calls["logged"] == [("run-output", "run-output")]
    assert "tok_compl_scientist" in fake.run.summary


def test_finalize_without_wandb_run_is_a_noop(tmp_path, monkeypatch):
    fake = types.ModuleType("wandb")
    fake.run = None
    monkeypatch.setitem(sys.modules, "wandb", fake)
    ins.configure(tmp_path / "o")
    assert ins.finalize_run(tmp_path / "o", upload=True) is not None
