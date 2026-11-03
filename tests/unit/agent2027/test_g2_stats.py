"""Unit tests for the pure G2 statistics (E11a, g2_stats.py) on hand-computed toy examples."""

import math

import numpy as np
import pandas as pd
import pytest

from scripts.agent2027.analysis import g2_stats as g

pytestmark = pytest.mark.unit


def test_s_d_and_signed_delta():
    assert g.s_d(0.02, 0.04) == pytest.approx(0.03)
    assert g.signed_delta(0.8, 0.7, "classification") == pytest.approx(0.1)
    assert g.signed_delta(0.20, 0.25, "regression") == pytest.approx(0.05)  # lower nRMSE = better


def test_sigma_aa_sigma_p_and_z_hand_example():
    a, b, c, s = [1.0, 2.0], [0.0, 2.0], [0.0, 0.0], [1.0, 1.0]
    saa2 = g.sigma_aa2(a, b, s)
    assert saa2 == pytest.approx(0.25)          # mean([1, 0]) / 2
    sp2 = g.sigma_p2(a, b, c, s, saa2)
    assert sp2 == pytest.approx(0.875)          # (mean([0.25, 4]) - 0.375) / 2
    assert g.z_share(saa2, sp2) == pytest.approx(0.25 / 1.125)
    assert g.sigma_p2([0.0], [0.0], [0.0], [1.0], 1.0) == 0.0  # truncated at 0
    assert math.isnan(g.z_share(0.0, 0.0))


def test_sigma_d_floor_and_pooling():
    assert g.sigma_d_raw(1.0, 0.0, 0.5) == pytest.approx(0.5)
    assert g.floor_sigma(0.5, 10.0) == pytest.approx(1.0)
    assert g.floor_sigma(0.5, 1.0) == pytest.approx(0.5)
    assert g.pooled_sigma(0.03, 0.04, 0.1) == pytest.approx(math.sqrt(0.00125))
    assert g.pooled_sigma(0.0, 0.0, 0.1) == pytest.approx(0.01)


def test_f_bounds_match_preregistration():
    lo, hi = g.f_bounds(3, 6)
    assert lo == pytest.approx(0.068, abs=5e-4) and hi == pytest.approx(6.60, abs=5e-3)


def test_homogeneity_pool_and_reject():
    s = {"housing": 0.01, "insurance": 0.01, "crab": 0.01}
    aa = {"housing": (0.20, 0.21), "insurance": (0.30, 0.31), "crab": (0.10, 0.11)}  # each ((0.01/0.01)^2)/2 = 0.5
    samp = {"housing": [0.01, -0.01], "insurance": [0.01, 0.01], "crab": [-0.01, 0.01]}
    h = g.homogeneity(aa, samp, s)
    assert h["F"] == pytest.approx(1.0) and h["df1"] == 3 and h["df2"] == 6 and h["pool"]
    big = {"housing": (0.20, 0.30), "insurance": (0.30, 0.40), "crab": (0.10, 0.20)}  # F = 100
    assert not g.homogeneity(big, samp, s)["pool"]
    assert not g.homogeneity(aa, samp, s, drift=True)["pool"]
    assert not g.homogeneity(aa, {}, s)["pool"]  # k = 0


def test_a03_sigma_hat_and_readout():
    sh = g.a03_sigma_hat("wine", 0.05)                 # A01 absent, t17 0.0010, floor 0.005
    assert sh["sigma_hat"] == pytest.approx(0.005) and sh["decided_by"] == "floor"
    assert g.a03_sigma_hat("junglechess", 0.02)["sigma_hat"] == pytest.approx(0.070)
    pairs = [{"dataset": f"d{i}", "task_type": "classification", "aug": 0.80, "rep": 0.80, "sigma_hat": 0.01}
             for i in range(8)]
    r = g.a03_readout(pairs)
    assert r["evaluable"] and r["crit_a"] and r["crit_b"] and r["outcome"] == "augment stands"
    bad = pairs[:7] + [{"dataset": "d7", "task_type": "regression", "aug": 0.30, "rep": 0.25, "sigma_hat": 0.01}]
    rb = g.a03_readout(bad)  # Delta = 0.25 - 0.30 = -0.05 -> z = -3.54 < -2.5
    assert not rb["crit_b"] and rb["outcome"].startswith("FAILS")
    few = g.a03_readout(pairs[:7])
    assert not few["evaluable"] and "augment stands" in few["outcome"]
    fail = pairs[:7] + [{"dataset": "d7", "task_type": "classification", "aug": None, "rep": 0.8,
                         "sigma_hat": 0.01, "aug_failed": True}]
    assert g.a03_readout(fail)["aug_fail_where_rep_ok"] == ["d7"]


def test_informative_and_sentinel_ties():
    assert g.informative_tasks({"a": 0, "b": 7, "c": 15}) == ["b"]
    rec10 = {"synp_p1": 10, "synp_p2": 4, "synp_p3": 6, "synp_p4": 0, "synp_p5": 0, "synp_p6a": 10, "synp_p6b": 1}
    rec15 = {"synp_p1": 15, "synp_p2": 6, "synp_p3": 9, "synp_p4": 0, "synp_p5": 0, "synp_p6a": 15, "synp_p6b": 1}
    s = g.sentinel_task(rec10, rec15)
    assert s["task"] == "synp_p2" and s["Z_of_10"] == 4  # tie with p3 at both levels -> lower index
    rec15b = dict(rec15, synp_p3=8)                       # p3 now closer at 15 episodes
    assert g.sentinel_task(rec10, rec15b)["task"] == "synp_p3"


def test_a06_band():
    sm = g.a06_sigma_aa_m([1.0, 2.0], [0.0, 2.0])
    assert sm == pytest.approx(0.5)
    b = g.a06_band(1.0, sm, 0.1)
    assert b["half_width"] == pytest.approx(3 * math.sqrt(1.5) * 0.5)
    assert g.a06_band(0.0, 0.0, 0.1)["half_width"] == pytest.approx(0.1)  # floor
    assert g.a06_transform(0.0, "log1p") == 0.0 and math.isnan(g.a06_transform(0.0, "log"))


def test_tau_branch_headroom():
    assert g.a19_tau(0.45) == 0.60 and g.a19_tau(0.50) == 0.60 and g.a19_tau(0.62) == pytest.approx(0.72)
    assert g.a12_branch(0.35, 0.40, True)["branch"].startswith("default")
    assert g.a12_branch(0.20, 0.40, True)["branch"].startswith("(i)")
    assert g.a12_branch(0.20, 0.20, False)["branch"].startswith("(ii)")
    h = g.a09_headroom({"p2": 0.4, "p3": 0.1, "p4": 0.9}, ["p2", "p3", "p4"])
    assert h["n"] == 2 and h["flag"]


def test_nogo_verdict():
    assert g.nogo_verdict(0, 0, 0, 0, 0, 0, None)["verdict"] == "GO"
    assert g.nogo_verdict(1, 0, 0, 0, 0, 0, None)["verdict"].startswith("NO-GO (unconditional)")
    assert g.nogo_verdict(0, 0, 0, 0, 0, 3, True)["verdict"].startswith("NO-GO (conditional")
    assert g.nogo_verdict(0, 0, 0, 3, 0, 0, False)["verdict"].startswith("GO (conditional")
    assert g.nogo_verdict(0, 0, 0, 2, 5, 1, None)["verdict"] == "GO"


def test_anova_icc_and_glmm():
    perfect = pd.DataFrame({"task": list("AAABBB"), "p": [0, 0, 0, 1, 1, 1]})
    assert g.anova_icc(perfect)["icc"] == pytest.approx(1.0)
    none = pd.DataFrame({"task": list("AAABBB"), "p": [0.2, 0.4, 0.6, 0.2, 0.4, 0.6]})
    assert g.anova_icc(none)["icc"] < 0
    rng = np.random.default_rng(1)
    rows = []
    for t, p in enumerate([0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]):
        for run, seed in enumerate([42, 42, 4]):
            for f in range(5):
                rows.append({"task": f"t{t}", "seed": seed, "run": f"t{t}r{run}", "rec_sel": int(rng.random() < p)})
    fit = g.glmm_icc(pd.DataFrame(rows))
    assert 0.2 < fit["icc"] < 1.0 and fit["n_obs"] == 105
    flat = pd.DataFrame([{"task": f"t{t}", "seed": 42, "run": f"t{t}r{r}", "rec_sel": int(f < 2)}
                         for t in range(7) for r in range(3) for f in range(5)])
    assert g.glmm_icc(flat)["icc"] < 0.05


def test_scaled_mde_provisional():
    assert g.scaled_mde(0.26, 6, 0.0) == pytest.approx(0.26)
    assert g.scaled_mde(0.26, 3, 0.25) == pytest.approx(0.26 * math.sqrt(2) * math.sqrt(2))
    assert g.scaled_mde(0.26, 0, 0.1) == float("inf")


def test_scan_container_log(tmp_path):
    log = tmp_path / "container.log"
    log.write_text(
        "[TesterAgent step=2 trial=25 +  0.00s] test_hypotheses: start\n"
        "  Error: Error running experiment for config /workspace/tasks/regression/insurance/config.yml:\n"
        "  XGBoostError: [21:04:29] /__w/xgboost/src/data/gradient_index.h:99:\n"
        "  Check failed: valid: Input data contains `inf` or a value too large\n"
        "inner-CV unavailable; falling back to the held-out metric for iteration 3\n", encoding="utf-8")
    s = g.scan_container_log(log)
    assert s["inner_cv_fallback"] == 1 and len(s["exceptions"]) == 1
    e = s["exceptions"][0]
    assert e["type"] == "XGBoostError" and "inf" in e["signature"] and e["context"].endswith("trial=25")
