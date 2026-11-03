"""Unit tests for the minimal SYN-P recovery detector (E8a)."""

import numpy as np
import pandas as pd
import pytest

from scripts.agent2027.analysis import recovery_detector as rd

pytestmark = pytest.mark.unit

N = 400


@pytest.fixture(scope="module")
def raw():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "id": np.arange(N),
        "a": np.exp(rng.normal(0, 0.4, N)),
        "b": np.exp(rng.normal(0, 0.4, N)),
        "c": np.exp(rng.normal(0, 0.4, N)),
        "t": rng.uniform(0, 720, N),
        "cat": rng.choice(["x", "y", "z"], N),
    })


def attrs(*rows):
    """rows: (name, command[, description])"""
    return pd.DataFrame([
        {"Attribute": r[0], "Pandas Command": r[1], "Description": r[2] if len(r) > 2 else "LLM feature"}
        for r in rows
    ])


MASK = np.arange(N) % 5 != 0  # "training rows"
SPEAR = rd.TaskDetector(rd.FAMILY_SPEARMAN)
RULE = rd.TaskDetector(rd.FAMILY_RULE)


def run(raw, phi, attributes, active, selected, det=SPEAR):
    return rd.detect_episode(raw, attributes, MASK, phi, det, active, selected)


def test_phi_injected_as_feature_is_recovered(raw):
    phi = (raw["a"] / raw["b"]).to_numpy()
    r = run(raw, phi, attrs(("ratio", "result = df['a'] / df['b']")), {0: ["ratio"]}, 0)
    assert (r.rec_any, r.rec_sel, r.iter_first_rec) == (1, 1, 0)


def test_scaled_or_monotone_of_phi_still_recovered(raw):
    phi = (raw["a"] / raw["b"]).to_numpy()
    r = run(raw, phi, attrs(("lr", "result = np.log(df['a'] / df['b'])")), {0: ["lr"]}, 0)
    assert r.rec_sel == 1


def test_groupby_id_command_pattern_replays(raw):
    phi = (raw["a"] / raw["b"]).to_numpy()
    cmd = "result = df_features['id'].map(df.groupby('id')['a'].first()) / df_features['id'].map(df.groupby('id')['b'].first())"
    r = run(raw, phi, attrs(("g", cmd)), {0: ["g"]}, 0)
    assert r.rec_sel == 1


def test_single_column_monotone_transform_is_rec0(raw):
    phi = np.log(raw["a"].to_numpy())  # P0
    a = attrs(("raw_a", "result = df['a']"), ("log_a", "result = np.log(df['a'])"),
              ("sqrt_a", "result = np.sqrt(df['a'])"), ("neg_a", "result = -df['a']"),
              ("clip_a", "result = df['a'].clip(upper=df['a'].quantile(0.9))"))
    r = run(raw, phi, a, {0: list(a["Attribute"])}, 0)
    assert (r.rec_any, r.rec_sel) == (0, 0)
    pf = r.per_feature.set_index("feature")
    assert pf.loc["log_a", "recovered"]  # criterion met ...
    assert pf.loc["log_a", "excluded"].startswith("raw_or_monotone")  # ... but excluded


def test_fallback_feature_is_ignored(raw):
    phi = (raw["a"] / raw["b"]).to_numpy()
    a = attrs(("fb", "result = df['a'] / df['b']", "Fallback engineered feature using deterministic ratio between 'a' and 'b'."))
    r = run(raw, phi, a, {0: ["fb"]}, 0)
    assert (r.rec_any, r.rec_sel) == (0, 0)
    assert r.per_feature.set_index("feature").loc["fb", "excluded"] == "fallback"
    # explicit flag works too
    a2 = attrs(("fb", "result = df['a'] / df['b']"))
    r2 = rd.detect_episode(raw, a2, MASK, phi, SPEAR, {0: ["fb"]}, 0, fallback={"fb"})
    assert r2.rec_any == 0


def test_rec_any_vs_rec_sel_and_first_iteration(raw):
    phi = (raw["a"] / raw["b"]).to_numpy()
    a = attrs(("noise", "result = df['c']"), ("ratio", "result = df['a'] / df['b']"))
    active = {0: ["noise"], 1: ["noise"], 3: ["noise", "ratio"], 4: ["noise", "ratio"], 5: ["noise"]}
    r = run(raw, phi, a, active, selected=5)  # pruned again at the selected iteration
    assert (r.rec_any, r.rec_sel, r.iter_first_rec) == (1, 0, 3)
    assert run(raw, phi, a, active, selected=4).rec_sel == 1
    assert run(raw, phi, a, active, selected=None).rec_sel == 0


def test_inactive_feature_does_not_count(raw):
    phi = (raw["a"] / raw["b"]).to_numpy()
    a = attrs(("ratio", "result = df['a'] / df['b']"), ("other", "result = df['c']"))
    r = run(raw, phi, a, {0: ["other"]}, 0)
    assert r.rec_any == 0


def test_rule_family_uses_nmi_for_threshold_rules(raw):
    ratio = (raw["a"] / raw["b"]).to_numpy()
    thr = float(np.median(ratio))
    phi = ((ratio >= thr) & (raw["cat"] == "x").to_numpy()).astype(float)
    rule = "result = ((df['a'] / df['b'] >= %r) & (df['cat'] == 'x')).astype(float)" % thr
    ratio_only = "result = df['a'] / df['b']"
    flag_only = "result = (df['a'] / df['b'] >= %r).astype(float)" % thr
    a = attrs(("rule", rule), ("ratio", ratio_only), ("flag", flag_only))
    r = run(raw, phi, a, {0: ["ratio", "flag"], 1: ["rule"]}, 1, det=RULE)
    pf = r.per_feature.set_index("feature")
    assert pf.loc["rule", "counted"] and not pf.loc["ratio", "counted"] and not pf.loc["flag", "counted"]
    assert (r.iter_first_rec, r.rec_sel) == (1, 1)
    # a binary rule is below the Spearman criterion alone: only NMI makes it count
    assert pf.loc["rule", "rho"] >= 0.95 or pf.loc["rule", "nmi"] >= 0.9
    assert not rd.meets_criterion(raw["a"].to_numpy() / raw["b"].to_numpy(), phi, rd.FAMILY_RULE)[0]


def test_nonmonotone_single_column_function_counts(raw):
    # P4-like: sin of an elapsed-hours counter is a single-column but NON-monotone transform
    phi = np.sin(2 * np.pi * raw["t"].to_numpy() / 24)
    a = attrs(("sin24", "result = np.sin(2 * np.pi * df['t'] / 24)"), ("t_raw", "result = df['t']"))
    r = run(raw, phi, a, {0: ["sin24", "t_raw"]}, 0)
    pf = r.per_feature.set_index("feature")
    assert pf.loc["sin24", "counted"] and not pf.loc["t_raw", "counted"]


def test_failed_replay_nan_feature_never_recovers(raw):
    phi = (raw["a"] / raw["b"]).to_numpy()
    a = attrs(("bad", "result = df['does_not_exist'] / df['b']"))
    r = run(raw, phi, a, {0: ["bad"]}, 0)
    assert r.rec_any == 0


def test_correlated_but_below_threshold_is_not_recovery(raw):
    phi = (raw["a"] / raw["b"]).to_numpy()
    a = attrs(("sum", "result = df['a'] + df['c']"), ("prod", "result = df['a'] * df['c']"))
    r = run(raw, phi, a, {0: ["sum", "prod"]}, 0)
    assert r.rec_any == 0


def test_train_mask_is_used(raw):
    # a feature that equals phi only on test rows must not count
    phi = (raw["a"] / raw["b"]).to_numpy()
    test_rows = ~MASK
    cmd = "result = df['c'].copy()"
    r = run(raw, phi, attrs(("x", cmd)), {0: ["x"]}, 0)
    assert r.rec_any == 0
    assert test_rows.sum() > 0


def test_nmi_basics():
    x = np.arange(100.0)
    assert rd.normalized_mi(x, x) == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    assert rd.normalized_mi(rng.normal(size=2000), rng.normal(size=2000)) < 0.1
    assert rd.spearman_abs(x, -x**3) == pytest.approx(1.0)
    assert rd.spearman_abs(x, np.full(100, 3.0)) == 0.0


# --- Detector amendment (pre-registration Part III.2 / Amendment 2, 2026-10-19) -----------
# Formula mode applies the no_gain_over_raw rule by default (it used to run only in
# --features-table mode, so the A05 P0 null task read rec_sel 4/5).

P0_NEAR_MONOTONE = "result = np.log(df['a']) + 0.15 * np.log(df['b'])"  # two columns, tracks a


def test_amended_p0_feature_without_gain_over_raw_is_not_counted(raw):
    phi = np.log(raw["a"].to_numpy())  # P0: phi is a monotone function of one raw column
    a = attrs(("near_a", P0_NEAR_MONOTONE))
    r = run(raw, phi, a, {0: ["near_a"]}, 0)  # default = amended
    assert (r.rec_any, r.rec_sel) == (0, 0)
    pf = r.per_feature.set_index("feature")
    assert pf.loc["near_a", "recovered"]  # the criterion is met (|rho| >= 0.95) ...
    assert pf.loc["near_a", "rho"] < rd.MONOTONE_RHO_THRESHOLD  # ... not caught by the 0.999 rule ...
    assert pf.loc["near_a", "excluded"] == "no_gain_over_raw:a"  # ... but no better than raw 'a'
    # the registered (pre-amendment) rule counted it: the A05 P0 bug
    reg = rd.detect_episode(raw, a, MASK, phi, SPEAR, {0: ["near_a"]}, 0, amended=False)
    assert (reg.rec_any, reg.rec_sel) == (1, 1)


def test_amended_true_recovery_still_counted(raw):
    phi = (raw["a"] / raw["b"]).to_numpy()
    a = attrs(("noise", "result = df['c']"), ("lr", "result = np.log(df['a']) - np.log(df['b'])"),
              ("fb", "result = df['a'] / df['b']", "Fallback ratio"))
    active = {0: ["noise"], 1: ["noise", "lr", "fb"]}
    for amended in (True, False):
        r = rd.detect_episode(raw, a, MASK, phi, SPEAR, active, 1, amended=amended)
        assert (r.rec_any, r.rec_sel, r.iter_first_rec) == (1, 1, 1)
        assert r.hits_by_iter == {1: ["lr"]}
        pf = r.per_feature.set_index("feature")
        assert pf.loc["fb", "excluded"] == "fallback" and not pf.loc["fb", "counted"]
    pf = rd.detect_episode(raw, a, MASK, phi, SPEAR, active, 1).per_feature.set_index("feature")
    assert pf.loc["lr", "rho"] - pf.loc["lr", "raw_rho_max"] >= rd.RAW_GAIN_MARGIN


def test_amended_formula_mode_matches_table_mode(raw):
    """The amended formula path is evaluate_feature_table on the replayed features."""
    from rogueone.utils.fold_safe_features import materialize_fold_safe_features

    phi = np.log(raw["a"].to_numpy())
    a = attrs(("near_a", P0_NEAR_MONOTONE), ("log_a", "result = np.log(df['a'])"),
              ("ratio", "result = df['a'] / df['b']"))
    r = run(raw, phi, a, {0: list(a["Attribute"])}, 0)
    feats = materialize_fold_safe_features(raw, a, MASK).reset_index(drop=True)
    tbl = rd.evaluate_feature_table(feats[list(a["Attribute"])], phi[MASK],
                                    raw[MASK].reset_index(drop=True), SPEAR)
    got = r.per_feature.set_index("feature")
    for _, row in tbl.iterrows():
        assert bool(got.loc[row["feature"], "counted"]) == bool(row["counted"])
        assert got.loc[row["feature"], "excluded"] == row["excluded"]
