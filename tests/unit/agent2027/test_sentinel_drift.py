"""A06 sentinel drift (E11b, sentinel_drift.py) on hand-computed toy examples.

Covers the A06-T8 "done when" cases (2 perturbed channels -> FLAG; only context_errors perturbed ->
WARN; M0 r1 against r0 alone -> PASS; a changed flags/ value -> FLAG) and the A06-T10 pooled tests.
"""

import math

import pandas as pd
import pytest

from scripts.agent2027.analysis import sentinel_drift as sd

pytestmark = pytest.mark.unit

K = 3 * math.sqrt(1.5)  # 3.674


def test_band_sigma_and_q4_sigma():
    b = sd.band(1.0, 0.1, 0.1)
    assert b["half_width"] == pytest.approx(0.3674, abs=1e-4)
    assert sd.band(1.0, 0.01, 0.1)["half_width"] == 0.1          # floor wins
    assert sd.band(0.0, float("nan"), 0.05)["half_width"] == 0.05  # no sigma -> floor
    assert sd.sigma_aa_m([1.0, 2.0], [0.0, 2.0]) == pytest.approx(0.5)  # sqrt((1 + 0) / (2 * 2))
    assert sd.q4_sigma(0.1, 1.0, 2.0) == pytest.approx(1 / math.sqrt(2))
    assert sd.q4_sigma(0.9, 1.0, 2.0) == pytest.approx(0.9)
    assert sd.transform(0, "log1p") == 0.0 and math.isnan(sd.transform(0, "log"))


# ---- a toy G2 band table: r0 = (tok 1e6, calls 2000, wall 9000 s, epa 0.05, ctx 0, serper 0.0,
#      score 0.07 / s 0.0064); sigma_AA,m chosen by hand
SIG = {"tok_compl_total": 0.05, "llm_calls_total": 0.04, "wall_s": 0.09, "err_per_accepted": 0.015,
       "context_errors": 0.0, "serper_ok_false_rate": 0.004, "scaled_score": 1.8}
S_TASK = 0.0064
R0 = {"tok_compl_total": 1e6, "llm_calls_total": 2000, "wall_s": 9000, "err_per_accepted": 0.05,
      "context_errors": 0, "serper_ok_false_rate": 0.0}
R0_SCORE = 0.07


def toy_bands(centre_summary=R0, centre_score=R0_SCORE):
    vals = sd.metric_values(centre_summary, centre_score, S_TASK)
    g2 = []
    for ch, m, scale in sd.CHANNEL_METRICS:
        g2.append({"channel": ch, "metric": m, "scale": scale, "sigma_AA_m": SIG[m],
                   **sd.band(vals[m], SIG[m], sd.FLOORS[scale])})
    g2.append({"channel": "execution", "metric": "execute_rate", "scale": "rate", "sigma_AA_m": float("nan"),
               "centre": None, "half_width": None, "lo": None, "hi": None})  # dropped by Amendment 6
    return g2


def row(summary, score, **kw):
    r = {"row_id": kw.pop("row_id", "A06-001"), "arm_id": kw.pop("arm_id", "sentinel"), "final_status": "OK",
         "cap_hit": False, "inner_cv_ok": True, "flag_diffs": [], "vllm_image_id": "sha256:v", "ref_vllm_image_id": "sha256:v",
         "serper_ok_false_rate": summary.get("serper_ok_false_rate"), "verdict_basis": "evaluated",
         "values": sd.metric_values(summary, score, S_TASK), "session": kw.pop("session", "S1"),
         "image_tag": kw.pop("image_tag", "v2.4"), "start_epoch": kw.pop("start_epoch", 1)}
    r.update(kw)
    return r


def judge(rows, bands, is_q4=False, q4_ref_capped=False):
    for r in rows:
        sd.score_row(r, bands, is_q4=is_q4, q4_ref_capped=q4_ref_capped)
    sd.finalise_verdicts(rows)
    return [r["verdict"] for r in rows]


def test_bands_from_g2_drop_execute_rate():
    b = sd.bands_from_g2(toy_bands())
    assert "execute_rate" not in b and len(b) == 7
    # log(1e6) = 13.8155 +- 3.674 * 0.05
    assert b["tok_compl_total"]["lo"] == pytest.approx(math.log(1e6) - K * 0.05)
    # context_errors: sigma 0 -> floor 0.1 on log1p
    assert b["context_errors"]["half_width"] == pytest.approx(0.1)


def test_m0_r1_against_r0_alone_passes():
    bands = sd.bands_from_g2(toy_bands())
    r1 = {"tok_compl_total": 1.08e6, "llm_calls_total": 2100, "wall_s": 9900, "err_per_accepted": 0.07,
          "context_errors": 0, "serper_ok_false_rate": 0.01}
    # log(1.08) = 0.077 < 0.184; log(1.05) = 0.049 < 0.147; log(1.1) = 0.095 < 0.331;
    # log1p(.07) - log1p(.05) = 0.019 < 0.1; 0.01 < 0.05; score 0.08 vs 0.07 -> 1.56 < 6.61
    assert judge([row(r1, 0.08)], bands) == ["PASS"]


def test_two_perturbed_channels_flag():
    bands = sd.bands_from_g2(toy_bands())
    pert = dict(R0, tok_compl_total=1.5e6, wall_s=15000)  # log 1.5 = 0.405 > 0.184; log 1.667 = 0.51 > 0.331
    r = row(pert, R0_SCORE)
    assert judge([r], bands) == ["FLAG"]
    assert r["channels_out"] == ["cost", "time"]


def test_only_context_errors_warn():
    bands = sd.bands_from_g2(toy_bands())
    r = row(dict(R0, context_errors=1), R0_SCORE)  # log1p(1) = 0.693 > floor 0.1
    assert judge([r], bands) == ["WARN"]
    assert r["channels_out"] == ["execution"]
    assert r["channels"]["execution"]["metrics_outside"] == ["context_errors"]


def test_changed_flag_flags():
    ref = {"ROGUE_ONE_PLANNER_MODE": "llm", "ROGUE_ONE_LLM_SEED": "42", "LLM_PORT": "8010",
           "ROGUE_ONE_IMAGE_TAG": "agent2027-v2.0"}
    run = {"ROGUE_ONE_PLANNER_MODE": "none", "ROGUE_ONE_LLM_SEED": "5001", "LLM_PORT": "8023",
           "ROGUE_ONE_IMAGE_TAG": "agent2027-v2.4"}
    diffs = sd.flag_diff(ref, run, {})
    assert [d["key"] for d in diffs] == ["ROGUE_ONE_PLANNER_MODE"]  # seed, port, image tag exempt
    bands = sd.bands_from_g2(toy_bands())
    r = row(R0, R0_SCORE, flag_diffs=diffs)
    assert judge([r], bands) == ["FLAG"]
    assert r["hard"][0].startswith("5 flags differ")


def test_post_v20_flags_exempt_only_at_full_value():
    ref = {"ROGUE_ONE_SANDBOX": "on"}
    full = {"ROGUE_ONE_FALLBACKS": "on"}
    assert sd.flag_diff(ref, {"ROGUE_ONE_SANDBOX": "on", "ROGUE_ONE_FALLBACKS": "on"}, full) == []
    d = sd.flag_diff(ref, {"ROGUE_ONE_SANDBOX": "on", "ROGUE_ONE_FALLBACKS": "off"}, full)
    assert d[0]["reason"] == "post-v2.0 flag not at its Full value"
    d = sd.flag_diff(ref, {"ROGUE_ONE_SANDBOX": "on", "ROGUE_ONE_INF_GUARD": "1"}, full)
    assert d[0]["key"] == "ROGUE_ONE_INF_GUARD" and "no Full value" in d[0]["reason"]
    # a registry key absent from both envs is equal (<unset>)
    assert sd.flag_diff(ref, ref, full, registry_keys=["ROGUE_ONE_SMOKE_FOLDS"]) == []


def test_hard_indicators():
    bands = sd.bands_from_g2(toy_bands())
    assert judge([row(R0, R0_SCORE, cap_hit=True)], bands) == ["FLAG"]
    assert judge([row(R0, R0_SCORE, final_status="FAILED", values={})], bands) == ["FLAG"]
    assert judge([row(dict(R0, serper_ok_false_rate=0.5), R0_SCORE)], bands) == ["FLAG"]
    assert judge([row(R0, R0_SCORE, inner_cv_ok=False)], bands) == ["FLAG"]
    assert judge([row(R0, R0_SCORE, vllm_image_id="sha256:other")], bands) == ["FLAG"]
    # Q4 cap hit is descriptive when a Q4 reference run capped
    r = row(R0, R0_SCORE, cap_hit=True, arm_id="sentinel_Q4")
    assert judge([r], bands, is_q4=True, q4_ref_capped=True) == ["PASS"]
    assert "descriptively" in r["notes"][0]


def test_q4_needs_three_channels():
    bands = sd.bands_from_g2(toy_bands())
    pert = dict(R0, tok_compl_total=1.5e6, wall_s=15000)
    assert judge([row(pert, R0_SCORE, arm_id="sentinel_Q4")], bands, is_q4=True) == ["WARN"]
    pert3 = dict(pert, serper_ok_false_rate=0.2)
    assert judge([row(pert3, R0_SCORE, arm_id="sentinel_Q4")], bands, is_q4=True) == ["FLAG"]


def test_q4_bands_from_anchor_and_k1():
    g2 = toy_bands()
    a = sd.metric_values(R0, R0_SCORE, S_TASK)
    k1 = sd.metric_values(dict(R0, wall_s=9000 * math.e), R0_SCORE, S_TASK)  # log wall differs by 1.0
    qb = sd.q4_bands(g2, a, k1)
    assert qb["wall_s"]["centre"] == pytest.approx(math.log(9000) + 0.5)
    assert qb["wall_s"]["half_width"] == pytest.approx(K / math.sqrt(2))  # max(0.09, 1/sqrt2)
    assert qb["tok_compl_total"]["half_width"] == pytest.approx(max(K * 0.05, 0.1))
    assert sd.q4_bands(g2, {}, k1) == {}  # anchor missing (capped) -> not evaluable


def test_same_channel_twice_in_a_row_flags():
    bands = sd.bands_from_g2(toy_bands())
    out = dict(R0, serper_ok_false_rate=0.2)  # search channel outside only
    rows = [row(out, R0_SCORE, row_id="A06-001", session="S1", start_epoch=1),
            row(R0, R0_SCORE, row_id="A06-002", session="S2", start_epoch=2, image_tag="v2.3"),  # other image
            row(out, R0_SCORE, row_id="A06-003", session="S3", start_epoch=3)]
    assert judge(rows, bands) == ["WARN", "PASS", "FLAG"]
    assert rows[2]["repeated_channels"] == ["search"]


def test_pending_and_reference_rows():
    bands = sd.bands_from_g2(toy_bands())
    rows = [row(R0, R0_SCORE, verdict_basis="pending"), row(R0, R0_SCORE, verdict_basis="reference"),
            row(R0, R0_SCORE, verdict_basis="not_evaluable", cap_hit=True)]
    assert judge(rows, bands) == ["PENDING", "REFERENCE", "FLAG"]


# ------------------------------------------------------------------ pooled tests (A06 4.8)
def test_fisher_matches_power_statement():
    # Z = 5/10; at Y = 30 only rates <= 0.13 or >= 0.87 reject (A06 4.8, rounded): x <= 4 (0.133,
    # p = 0.029) or x >= 26 rejects, x = 5 (0.167, p = 0.085) does not
    assert sd.fisher_two_sided(15, 30, 5, 10)["p"] == pytest.approx(1.0)
    assert sd.fisher_two_sided(4, 30, 5, 10)["p"] == pytest.approx(0.0286, abs=1e-4)
    assert sd.fisher_two_sided(3, 30, 5, 10)["p"] < 0.05
    assert sd.fisher_two_sided(27, 30, 5, 10)["p"] < 0.05
    assert sd.fisher_two_sided(5, 30, 5, 10)["p"] > 0.05


def test_stratified_exact_hand_example():
    # one stratum (1 of 1 sentinel success, 0 of 1 M0): X in {0, 1} with 0.5 each, E = 0.5 -> p = 1
    assert sd.stratified_exact([(1, 1, 0, 1)])["p"] == pytest.approx(1.0)
    # two such strata: X in {0,1,2} with .25/.5/.25, E = 1, observed 2 -> p = .25 + .25
    r = sd.stratified_exact([(1, 1, 0, 1), (1, 1, 0, 1)])
    assert r["p"] == pytest.approx(0.5) and r["expected"] == pytest.approx(1.0)


def test_run_permutation_hand_example():
    # 5 runs, C(5,2) = 10 labellings; observed (M0 = the two zeros) is the unique maximum -> 1/10 one-sided
    r = sd.run_permutation([0.0, 0.0], [1.0, 1.0, 1.0])
    assert r["n_perm"] == 10 and r["p_upper"] == pytest.approx(0.1) and r["p"] == pytest.approx(0.2)
    assert r["min_p"] == pytest.approx(0.2)
    # A06 4.8: min p = 2 / C(K+2, 2): 0.056 at K = 7, 0.044 at K = 8
    assert sd.run_permutation([0, 0], [1] * 7)["p"] == pytest.approx(2 / 36)
    assert sd.run_permutation([0, 0], [1] * 8)["p"] == pytest.approx(2 / 45)


def test_ca_trend_exact_hand_example():
    # groups (score, n, x): (0,2,0), (1,2,2); S = 2; T = x1 with weights C(2,2-x1) C(2,x1) = 1, 4, 1
    r = sd.ca_trend_exact([(0, 2, 0), (1, 2, 2)])
    assert r["t_obs"] == 2 and r["expected"] == pytest.approx(1.0)
    assert r["p"] == pytest.approx(1 / 3)
    flat = sd.ca_trend_exact([(0, 10, 5), (1, 5, 2), (2, 5, 3)])
    assert 0 < flat["p"] <= 1


def test_ffh_monte_carlo():
    r = sd.ffh_mc([[5, 0], [0, 5]], draws=20000, seed=1)
    assert r["p"] == pytest.approx(2 / 252, abs=0.004)  # exact two-sided Fisher p
    assert sd.ffh_mc([[5, 0], [5, 0], [5, 0]])["p"] == 1.0  # all zeros: homogeneous


def test_final_tests_toy():
    eps = []
    for rid, hits in (("M0a", [0, 1, 1, 1, 0]), ("M0b", [0, 1, 1, 0, 0]),
                      ("S1", [0, 0, 0, 0, 0]), ("S2", [0, 0, 0, 0, 0]), ("S3", [0, 0, 0, 0, 0])):
        eps += [{"row_id": rid, "fold": f, "rec_sel": h} for f, h in enumerate(hits)]
    sent = pd.DataFrame({"row_id": ["S1", "S2", "S3"], "session": ["A", "B", "C"], "start_epoch": [1, 2, 3]})
    q4 = pd.DataFrame({"row_id": ["S1"], "session": ["Q"], "start_epoch": [1], "func_success": [0]})
    res = sd.final_tests(pd.DataFrame(eps), ["M0a", "M0b"], sent, q4, draws=200)
    f = res["primary_fisher"]
    assert (f["x"], f["y"], f["z"], f["n0"]) == (0, 15, 5, 10)
    assert f["p"] < 0.05 and res["primary_rejects"]
    assert res["period_sensitivity"].startswith("REQUIRED")
    assert res["sens_run_permutation"]["n_perm"] == 10
    assert res["desc_trend_ca_exact"]["session_order"] == ["A", "B", "C"]
    assert res["desc_q4_homogeneity"]["rec_sel"]["p"] == 1.0
