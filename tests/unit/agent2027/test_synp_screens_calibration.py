"""Pairwise screen and headroom calibration tests (E8a).  Reduced n keeps them fast;
the full-size results are in the calibration / screen reports."""

import numpy as np
import pytest

from scripts.agent2027.synp import calibration as C
from scripts.agent2027.synp import screen_pairwise as S
from scripts.agent2027.synp import tasks as T

pytestmark = pytest.mark.unit


def test_screen_flags_bmi_like_and_passes_reynolds():
    bmi = T.generate_task("p6a", n=600, p6a_candidate="bmi_like")
    r = S.screen_task(bmi.data, bmi.phi, "spearman")
    assert not r["passed"] and r["max_abs_spearman"] >= 0.95  # w/h^2 ~ rank-identical to w/h
    rey = T.generate_task("p6a", n=600, p6a_candidate="reynolds")
    r = S.screen_task(rey.data, rey.phi, "spearman")
    assert r["passed"] and r["max_abs_spearman"] < 0.9


def test_screen_flags_plain_ratio_task():
    p1 = T.generate_task("p1", n=400)
    r = S.screen_task(p1.data, p1.phi, "spearman")
    assert not r["passed"] and r["max_abs_spearman"] > 0.99


def test_p7_screen_and_extras():
    g = T.generate_task("p7", n=1000, p7_secret="unit-test-secret")
    r = S.screen_task(g.data, g.phi, "rule")
    assert r["passed"], r["top_by_spearman"][:2]
    ex = S.screen_p7_extras(g.data, g.meta)
    assert ex["ratio_alone_passes_screen"] and not ex["guess_grid_any_meets"] and ex["passed"]


def test_p7_extras_detect_a_guessable_threshold():
    g = T.generate_task("p7", n=1000, p7_secret="x")
    c = g.data["corrosion_depth_mm"].to_numpy()
    meta = dict(g.meta)
    meta["phi"] = {**g.meta["phi"], "consts": {"thr": float(np.quantile(c, 0.5))}}
    assert not S.screen_p7_extras(g.data, meta)["passed"]


def test_headroom_positive_for_planted_feature_and_zero_for_p0():
    C.N_JOBS = 2
    g = T.generate_task("p1", n=500, params={"noise": 0.1})
    h = C.headroom(g.data, g.phi, g.labels["target"].to_numpy(), "regression", seeds=(42,))
    assert h["mean"] > 0.15 and h["with_phi_mean"] < h["raw_mean"]
    g0 = T.generate_task("p0", n=500)
    h0 = C.headroom(g0.data, g0.phi, g0.labels["target"].to_numpy(), "regression", seeds=(42,))
    assert abs(h0["mean"]) < 0.05  # trees are invariant to log x


def test_calibrate_task_picks_weakest_signal_reaching_aim(monkeypatch):
    calls = []
    real_generate = T.generate_task

    def fake_generate(key, params=None, **kw):
        calls.append(params)
        return real_generate(key, n=60, params=params)

    def fake_headroom(data, phi, y, task_type, seeds=C.FOLD_SEEDS, extra=None):
        h = 0.4 - 0.2 * calls[-1]["noise"]  # grows as noise shrinks
        return {"mean": h, "min": h, "raw_mean": 1.0, "with_phi_mean": 0.8, "per_seed": []}

    monkeypatch.setattr(C.T, "generate_task", fake_generate)
    monkeypatch.setattr(C, "headroom", fake_headroom)
    r = C.calibrate_task("p1")
    # grid is weakest-first: 2.5, 2.0, 1.5, 1.2, 1.0 -> 0.4-0.2*1.0 = 0.2 >= aim 0.18
    assert r.value == 1.0 and r.passed and r.scan[-1]["noise"] == 1.0 and r.target == 0.15


def test_compute_target_classification_base_rate_and_regression_standardised():
    rng = np.random.default_rng(0)
    phi = rng.lognormal(size=5000)
    eps, u = rng.standard_normal(5000), rng.random(5000)
    y = T.compute_target("classification", phi, {"beta": 1000.0}, eps, u)
    assert abs(y.mean() - 0.5) < 0.01
    y = T.compute_target("classification", phi, {"beta": 1000.0, "prevalence": 0.1}, eps, u)
    assert abs(y.mean() - 0.1) < 0.01
    yb = T.compute_target("classification", (phi > 1.5).astype(float), {"beta": 1000.0}, eps, u)  # binary phi
    assert (yb == (phi > 1.5)).all()
    yr = T.compute_target("regression", phi, {"noise": 0.5}, eps, u)
    assert yr.std() == pytest.approx(1.0) and yr.mean() == pytest.approx(T.TARGET_OFFSET)
