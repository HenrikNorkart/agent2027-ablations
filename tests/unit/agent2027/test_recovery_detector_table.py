"""Detector ``--features-table`` mode (A08-T8): brute-force comparators' dumped
training-row feature tables are scored with the same phi criteria as formula mode."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.agent2027.analysis import recovery_detector as rd
from scripts.agent2027.synp import generate_synp as G

pytestmark = pytest.mark.unit

N = 400
SPEAR = rd.TaskDetector(rd.FAMILY_SPEARMAN)


@pytest.fixture(scope="module")
def raw():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "id": np.arange(100, 100 + N),
        "a": np.exp(rng.normal(0, 0.4, N)),
        "b": np.exp(rng.normal(0, 0.4, N)),
        "c": np.exp(rng.normal(0, 0.4, N)),
        "cat": rng.choice(["x", "y", "z"], N),
    })


def _table(raw, mask, **cols):
    t = pd.DataFrame({"id": raw.loc[mask, "id"].to_numpy()})
    for k, v in cols.items():
        t[k] = np.asarray(v)[mask]
    return t.sample(frac=1.0, random_state=1).reset_index(drop=True)  # row order must not matter


MASK = np.arange(N) % 5 != 0


def test_injected_phi_column_is_recovered(raw):
    phi = (raw["a"] / raw["b"]).to_numpy()
    t = _table(raw, MASK, noise=raw["c"] * raw["a"], log_a=np.log(raw["a"]), autoFE_f_7=phi)
    s, pf = rd.score_feature_table(t, raw, phi, SPEAR)
    assert (s["rec_any"], s["rec_sel"]) == (1, 1)
    assert s["best_feature"] == "autoFE_f_7" and s["best_score"] == pytest.approx(1.0)
    assert s["first_rec_rank"] == 3 and s["n_features"] == 3
    assert s["n_excluded_monotone"] == 1  # log_a
    assert pf.set_index("feature").loc["log_a", "excluded"].startswith("monotone_rho:a")


def test_log_of_single_raw_column_is_rec0(raw):
    phi = np.log(raw["a"].to_numpy())  # P0
    t = _table(raw, MASK, log_a=phi, a=raw["a"], neg_sqrt_a=-np.sqrt(raw["a"]),
               lin_a=3.0 * raw["a"] + 1.0)
    s, pf = rd.score_feature_table(t, raw, phi, SPEAR)
    assert (s["rec_any"], s["rec_sel"]) == (0, 0)
    assert s["n_excluded_monotone"] == 4 and s["n_eligible"] == 0
    pf = pf.set_index("feature")
    assert pf.loc["log_a", "recovered"] and not pf.loc["log_a", "counted"]  # criterion met but excluded
    assert s["best_feature"] is None


def test_value_based_rule_excludes_near_monotone_noise(raw):
    # 0.999 rule: tiny jitter breaks exact monotonicity but is still "one raw column"
    phi = np.log(raw["a"].to_numpy())
    jit = np.log(raw["a"].to_numpy()) + 1e-4 * np.random.default_rng(3).normal(size=N)
    s, pf = rd.score_feature_table(_table(raw, MASK, jit=jit), raw, phi, SPEAR)
    assert s["rec_any"] == 0
    assert pf.loc[0, "excluded"].startswith("monotone_rho:a")


def test_two_column_feature_dominated_by_one_raw_column_is_rec0(raw):
    # The A08 screen's p0 false positive: log(c)/a (|rho| ~0.96 with log a) uses two
    # columns but tracks phi = log(a) no better than a alone does.
    phi = np.log(raw["a"].to_numpy())
    f = np.log(raw["c"].to_numpy()) / raw["a"].to_numpy() * 0.05 - raw["a"].to_numpy()
    s, pf = rd.score_feature_table(_table(raw, MASK, autoFE_f_1=f), raw, phi, SPEAR)
    assert pf.loc[0, "recovered"] and pf.loc[0, "raw_rho_max"] < rd.MONOTONE_RHO_THRESHOLD
    assert s["rec_any"] == 0
    assert pf.loc[0, "excluded"].startswith("no_gain_over_raw:a")


def test_categorical_codes_count_as_raw_columns(raw):
    codes = raw["cat"].map({"x": 0, "y": 1, "z": 2}).to_numpy(dtype=float)
    s, pf = rd.score_feature_table(_table(raw, MASK, cat_code=codes * 2), raw, codes, SPEAR)
    assert s["rec_any"] == 0 and pf.loc[0, "excluded"].startswith(("monotone_rho:cat", "raw_or_monotone:cat"))


def test_raw_arm_table_with_id_only(raw):
    s, pf = rd.score_feature_table(_table(raw, MASK), raw, raw["a"].to_numpy(), SPEAR)
    assert (s["rec_any"], s["n_features"], s["best_feature"]) == (0, 0, None)


def test_unknown_ids_raise(raw):
    t = _table(raw, MASK, f=raw["a"])
    t.loc[0, "id"] = -5
    with pytest.raises(ValueError):
        rd.score_feature_table(t, raw, raw["a"].to_numpy(), SPEAR)


def test_discover_maps_seed_dataset_arm_fold(tmp_path):
    for rel in ["f42/features/synp_p1_openfe_fold0.csv.gz", "f4/features/synp_p6a_autofeat_fold3.csv.gz",
                "f5/features/synp_p1_openfe_exprs.json", "loose/synp_p3_raw_fold2.csv"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("id\n")
    got = {(d["dataset_id"], d["arm"], d["fold_seed"], d["fold"]) for d in
           rd.discover_feature_tables(tmp_path, default_fold_seed=7)}
    assert got == {("synp_p1", "openfe", 42, 0), ("synp_p6a", "autofeat", 4, 3), ("synp_p3", "raw", 7, 2)}
    one = rd.discover_feature_tables(tmp_path / "f4" / "features" / "synp_p6a_autofeat_fold3.csv.gz")
    assert one[0]["fold_seed"] == 4  # single file: seed still taken from its f<seed> parent


# --------------------------------------------------------------------------- #
# Table mode == formula mode on generated SYN-P tasks
# --------------------------------------------------------------------------- #
KEYS = ("p0", "p1", "p3", "p4", "p5")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("tbl")
    out = G.build_all(root / "tasks", root / "meta", n=500, p7_secret="tbl-secret", keys=KEYS)
    return root, out


def _attrs(meta, df):
    cols = [c for c in df.columns if c != "id" and pd.api.types.is_numeric_dtype(df[c])]
    c0, c1 = cols[0], cols[1]
    rows = [
        ("oracle", "result = " + meta["phi_pandas_expr"]),
        ("raw0", f"result = df[{c0!r}]"),
        ("log0", f"result = np.log(df[{c0!r}].abs() + 1)"),
        ("prod01", f"result = df[{c0!r}] * df[{c1!r}]"),
        ("ratio01", f"result = df[{c0!r}] / (df[{c1!r}].abs() + 1)"),
    ]
    return pd.DataFrame([{"Attribute": n, "Pandas Command": c, "Description": "LLM"} for n, c in rows])


def test_table_mode_equals_formula_mode(world, tmp_path):
    from rogueone.utils.fold_safe_features import materialize_fold_safe_features

    root, out = world
    expected = {}
    for key, g in out.items():
        meta = json.loads((root / "meta" / f"{g.task_id}.json").read_text())
        df_raw = pd.read_csv(root / "tasks" / g.task_type / g.task_id / "data.csv")
        phi, det = rd.load_phi(root / "meta", g.task_id, df_raw)
        attrs = _attrs(meta, df_raw)
        names = list(attrs["Attribute"])
        for seed, fold in [(42, 0), (42, 3), (4, 1)]:
            mask = (np.random.default_rng(seed).permutation(len(df_raw)) % 5) != fold
            res = rd.detect_episode(df_raw, attrs, mask, phi, det, {0: names}, 0)
            expected[(g.task_id, seed, fold)] = res
            feats = materialize_fold_safe_features(df_raw, attrs, mask)
            tbl = pd.concat([df_raw.loc[mask, ["id"]].reset_index(drop=True),
                             feats[[n for n in names if n in feats.columns]].reset_index(drop=True)], axis=1)
            dest = tmp_path / "in" / f"f{seed}" / "features" / f"{g.task_id}_openfe_fold{fold}.csv.gz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            tbl.to_csv(dest, index=False)

    ep, pf = rd.run_feature_tables(tmp_path / "in", root / "tasks", root / "meta")
    assert len(ep) == len(expected) and (ep["status"] == "ok").all(), ep["error"].tolist()
    assert (ep["rec_sel"] == ep["rec_any"]).all()
    for _, row in ep.iterrows():
        res = expected[(row["dataset_id"], row["fold_seed"], row["fold"])]
        assert row["rec_any"] == res.rec_any, (row["dataset_id"], row["fold_seed"], row["fold"])
        # per-feature: the same columns count in both modes
        f_counted = set(res.per_feature.loc[res.per_feature["counted"], "feature"])
        sub = pf[(pf["dataset_id"] == row["dataset_id"]) & (pf["fold_seed"] == row["fold_seed"])
                 & (pf["fold"] == row["fold"])]
        assert set(sub.loc[sub["counted"], "feature"]) == f_counted
        if row["dataset_id"] == "synp_p0":
            assert row["rec_any"] == 0  # P0 control: phi is a monotone transform of one column
        else:
            assert row["rec_any"] == 1 and row["best_feature"] == "oracle"

    # CLI writes CSV + JSON (+ per-feature) to --out
    rd.main(["--features-table", str(tmp_path / "in"), "--tasks", str(root / "tasks"),
             "--meta", str(root / "meta"), "--out", str(tmp_path / "out")])
    csv = pd.read_csv(tmp_path / "out" / "episodes.csv")
    js = json.loads((tmp_path / "out" / "episodes.json").read_text())
    assert len(csv) == len(js) == len(expected)
    assert list(csv.columns) == rd.EPISODE_COLUMNS
    assert (tmp_path / "out" / "per_feature.csv.gz").exists()


def test_missing_meta_is_itt_error_row(world, tmp_path):
    root, _ = world
    d = tmp_path / "f42" / "features"
    d.mkdir(parents=True)
    pd.DataFrame({"id": [0, 1], "x": [1.0, 2.0]}).to_csv(d / "synp_nope_openfe_fold0.csv.gz", index=False)
    ep, _ = rd.run_feature_tables(tmp_path, root / "tasks", root / "meta")
    assert ep.loc[0, ["status", "rec_any", "rec_sel"]].tolist() == ["error", 0, 0]
