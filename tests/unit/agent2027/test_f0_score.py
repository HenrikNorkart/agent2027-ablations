"""E11d / A17-T8 F0 scorer: best-of-k, validation, Stage C params, and a hand-made toy pool."""

import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.agent2027 import f0_generate as f0
from scripts.agent2027 import f0_score as fs

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parents[3]
TESTER = REPO / "rogueone" / "llm" / "agents" / "core" / "tester" / "agent.py"


# --------------------------------------------------------------------------- Stage C params
def _block(src: str, start: str, end: str) -> str:
    i = src.index(start)
    return src[i:src.index(end, i)]


@pytest.mark.parametrize("task_type,start,end", [
    ("regression", "def get_regression_metrics", "mses = []"),
    ("classification", "def get_classification_metrics_tabular", "accs = []"),
])
def test_heldout_params_equal_tester_source(task_type, start, end):
    block = _block(TESTER.read_text(encoding="utf-8"), start, end)
    params = fs.heldout_params(task_type, fold_seed=4)
    for k, v in params.items():
        if k in ("n_jobs", "random_state"):
            continue
        lit = f'"{v}"' if isinstance(v, str) else repr(v)
        assert re.search(rf'"{k}":\s*{re.escape(lit)}', block), (k, v)
    assert '"random_state": self.fold_seed' in block
    assert params["random_state"] == 4


# --------------------------------------------------------------------------- best-of-k
def test_best_of_k_handmade_all_orderings():
    perms = np.array(list(itertools.permutations(range(3))))
    curve = fs.best_of_k([-math.inf, 0.5, 0.7], [None, 0.6, 0.8], [0, 0, 1], [10, 20, 30],
                         0.4, perms)
    assert [c["k"] for c in curve] == [1, 2, 3]
    # k=1: first draw of the ordering; draw 0 failed -> ITT (base 0.4, rec 0)
    assert curve[0]["heldout_bestk"] == pytest.approx((2 * 0.4 + 2 * 0.6 + 2 * 0.8) / 6)
    assert curve[0]["rec_sel_bestk"] == pytest.approx(1 / 3)
    assert curve[0]["itt_share"] == pytest.approx(1 / 3)
    assert curve[0]["tok_compl_cum"] == pytest.approx(20)
    # k=2: (0,1)->1 (0,2)->2 (1,0)->1 (1,2)->2 (2,0)->2 (2,1)->2
    assert curve[1]["heldout_bestk"] == pytest.approx((0.6 * 2 + 0.8 * 4) / 6)
    assert curve[1]["rec_sel_bestk"] == pytest.approx(4 / 6)
    assert curve[1]["itt_share"] == 0
    assert curve[2]["heldout_bestk"] == pytest.approx(0.8)
    assert curve[2]["rec_sel_bestk"] == 1
    assert curve[2]["tok_compl_cum"] == pytest.approx(60)


def test_best_of_k_ties_go_to_first_in_ordering():
    curve = fs.best_of_k([0.5, 0.5], [0.1, 0.2], [1, 0], [1, 1], 0.0, np.array([[0, 1], [1, 0]]))
    assert curve[1]["rec_sel_bestk"] == pytest.approx(0.5)
    assert curve[1]["heldout_bestk"] == pytest.approx(0.15)


def test_best_of_k_all_failed_is_itt():
    perms = fs.orderings(4, 10, seed=1)
    curve = fs.best_of_k([-math.inf] * 4, [None] * 4, [0] * 4, [5] * 4, 0.33, perms)
    for c in curve:
        assert c["itt_share"] == 1 and c["heldout_bestk"] == pytest.approx(0.33)
        assert c["rec_sel_bestk"] == 0
    # unknown recovery (no phi) stays None
    assert fs.best_of_k([0.1], [0.2], [None], [1], 0.0, np.array([[0]]))[0]["rec_sel_bestk"] is None


def test_orderings_seeded_permutations():
    a, b = fs.orderings(128), fs.orderings(128)
    assert a.shape == (fs.N_ORDERINGS, 128)
    assert (a == b).all()
    assert all(sorted(r) == list(range(128)) for r in a)


def test_selection_score_direction():
    assert fs.selection_score(0.8, "classification") == 0.8
    assert fs.selection_score(2.0, "regression") == -2.0
    assert fs.selection_score(None, "regression") == -math.inf


# --------------------------------------------------------------------------- parse + validate
def test_parse_cuts_over_limit_to_first_ten():
    items = [{"name": f"f{i}", "pandas_command": "result = 1"} for i in range(12)]
    p = fs.parse_draw({"raw_text": "```json\n" + json.dumps(items) + "\n```"})
    assert p["parse_ok"] and p["fenced"] and p["over_limit"] and p["n_items"] == 12
    assert [it["name"] for it in p["items"]] == [f"f{i}" for i in range(10)]
    assert not fs.parse_draw({"raw_text": "nope"})["parse_ok"]
    assert not fs.parse_draw(None)["parse_ok"]


def test_validate_items_extractor_rules():
    df = pd.DataFrame({"id": [0, 0, 1, 2], "a": [1.0, 2.0, 3.0, 4.0], "s": ["x", "y", "x", "z"]})
    ok = "result = df_features['id'].map(df.groupby('id')['a'].sum())"
    items = [
        {"name": "asum", "pandas_command": ok},
        {"name": "again", "pandas_command": "result = df_features['asum'] * 2"},  # uses accepted item
        {"name": "asum", "pandas_command": ok},                                   # duplicate
        {"name": "a", "pandas_command": ok},                                      # raw column name
        {"name": "target", "pandas_command": ok},                                 # reserved
        {"name": "nan", "pandas_command": "result = df_features['id'].map({0: 1.0})"},
        {"name": "txt", "pandas_command": "result = df_features['id'].map(df.groupby('id')['s'].first())"},
        {"name": "len", "pandas_command": "result = df['a'].values"},             # 4 values, 3 ids
        {"name": "key", "pandas_command": "result = df['nope']"},
        {"name": "none", "pandas_command": "x = 1"},
    ]
    accepted, status = fs.validate_items(items, df)
    assert [a["name"] for a in accepted] == ["asum", "again"]
    st = dict(s.split(":", 1) for s in status[2:])
    assert status[2] == "asum:duplicate" and st["a"] == "duplicate" and st["target"] == "reserved_name"
    assert st["nan"] == "nan_or_inf" and st["txt"] != "ok"  # reason depends on the pandas string dtype
    assert st["len"] == "ValueError" and st["key"] == "KeyError" and st["none"] == "no_result"


def test_validate_items_augment_frame_sees_raw_columns():
    df = pd.DataFrame({"id": [0, 1, 2], "a": [1.0, 2.0, 3.0]})
    item = [{"name": "a2", "pandas_command": "result = df_features['a'] * 2"}]
    assert fs.validate_items(item, df)[0] == []  # id-only frame (A17-T8 as written)
    frame = pd.DataFrame({"id": [0, 1, 2], "a": [1.0, 2.0, 3.0]})
    acc, _ = fs.validate_items(item, df, df_features=frame)
    assert [a["name"] for a in acc] == ["a2"] and list(frame.columns) == ["id", "a"]


def test_covtype_pool_refused(tmp_path):
    (tmp_path / "folds.json").write_text(json.dumps(
        {"row_id": "A17-999", "dataset_id": "covtype", "fold_seed": 42, "test_ids": [[1]]}))
    with pytest.raises(ValueError, match="covtype"):
        fs.load_pool(tmp_path)


# --------------------------------------------------------------------------- toy pool end to end
GOOD = "tmp = df.groupby('id')[['a', 'b']].first()\nresult = df_features['id'].map(tmp['a'] * tmp['b'])"
BAD = "result = df_features['id'].map(df.groupby('id')['zzz'].first())"


@pytest.fixture()
def toy(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    a, b, c = rng.uniform(0.5, 1.5, n), rng.uniform(0.5, 1.5, n), rng.uniform(0, 1, n)
    task = tmp_path / "tasks" / "regression" / "synp_toy"
    task.mkdir(parents=True)
    df = pd.DataFrame({"id": range(n), "a": a, "b": b, "c": c})
    labels = pd.DataFrame({"id": range(n), "target": a * b + 0.01 * rng.normal(size=n)})
    df.to_csv(task / "data.csv", index=False)
    labels.to_csv(task / "labels.csv", index=False)
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "synp_toy.json").write_text(json.dumps(
        {"phi": {"expr": "a * b", "cols": {"a": "a", "b": "b"}}, "detector": {"family": "spearman"}}))
    pool = tmp_path / "pools" / "A17-999"
    pool.mkdir(parents=True)
    folds = f0.outer_folds(df, labels, "regression", 42, False)
    (pool / "folds.json").write_text(json.dumps(
        {"row_id": "A17-999", "dataset_id": "synp_toy", "fold_seed": 42, "stratified": False,
         "test_ids": [f["test_ids"] for f in folds]}, default=int))
    (pool / "summary.json").write_text(json.dumps(
        {"row_id": "A17-999", "run_name": "A17__f0pool_Q4__synp_toy__f42__r0", "complete": True}))
    texts = ["not json at all",
             json.dumps([{"name": "bad", "pandas_command": BAD, "argumentation": "x"}]),
             "```json\n" + json.dumps([{"name": "ab", "pandas_command": GOOD, "argumentation": "a*b"}]) + "\n```"]
    with open(pool / "draws.jsonl", "w", encoding="utf-8") as fh:
        for f in range(5):
            for j, t in enumerate(texts):
                fh.write(json.dumps({"fold": f, "draw": j, "seed": 42000 + 128 * f + j, "raw_text": t,
                                     "completion_tokens": 100 * (j + 1), "error_kind": None}) + "\n")
    f2 = tmp_path / "f2.csv"
    pd.DataFrame([{"rung": "Q4", "dataset_id": "synp_toy", "fold_seed": 42, "fold": f,
                   "tok_compl_total": 450} for f in range(5)]).to_csv(f2, index=False)
    return tmp_path, f2


def _run(root, f2, out, *extra):
    argv = ["--pools", str(root / "pools"), "--tasks", str(root / "tasks"), "--meta", str(root / "meta"),
            "--manifest", str(root / "none.csv"), "--out", str(out), "--n-draws", "3",
            "--f2-tokens", str(f2), *extra]
    assert fs.main(argv) == 0
    return (pd.read_csv(out / "draws.csv"), pd.read_csv(out / "curves.csv"),
            pd.read_csv(out / "fold_summary.csv"))


def test_toy_pool_end_to_end(toy):
    root, f2 = toy
    draws, curves, summ = _run(root, f2, root / "out")
    assert len(draws) == 15 and len(summ) == 5 and len(curves) == 15
    d = draws.set_index(["fold", "draw"])
    for f in range(5):
        assert d.loc[(f, 0), "fail_reason"] == "parse" and not d.loc[(f, 0), "parse_ok"]
        assert d.loc[(f, 1), "fail_reason"] == "exec" and "bad:KeyError" in d.loc[(f, 1), "item_errors"]
        assert d.loc[(f, 2), "scored"] and d.loc[(f, 2), "rec_sel"] == 1 and d.loc[(f, 2), "hits"] == "ab"
        assert d.loc[(f, 2), "n_features"] == 4  # seeds a, b, c + ab (augment)
    s = summ.iloc[0]
    assert s["f0_parse_rate"] == pytest.approx(2 / 3)
    assert s["f0_exec_rate"] == pytest.approx(1 / 2)
    assert s["f0_parse_exec_rate"] == pytest.approx(1 / 3)
    # F0 = draw 1 (index 0) failed -> ITT at base_no_fe, rec_sel 0
    assert s["f0_itt"] == 1 and s["f0_rec_sel"] == 0
    assert s["f0_heldout"] == pytest.approx(s["base_no_fe_heldout"])
    assert s["rec_any_pool"] == 1 and s["rec_sel_best128"] == 1  # k = n = 3 here
    # the planted feature helps: its held-out nRMSE beats the seed-only pool
    assert d.loc[(0, 2), "heldout"] < s["base_no_fe_heldout"]
    # k = 1 against an independent count over the same orderings (all-failed prefix = ITT)
    perms = fs.orderings(3)
    first_is_good = perms[:, 0] == 2
    c1 = curves[(curves["fold"] == 0) & (curves["k"] == 1)].iloc[0]
    assert c1["rec_sel_bestk"] == pytest.approx(first_is_good.mean())
    assert c1["itt_share"] == pytest.approx(1 - first_is_good.mean())
    exp_h = np.where(first_is_good, d.loc[(0, 2), "heldout"], s["base_no_fe_heldout"]).mean()
    assert c1["heldout_bestk"] == pytest.approx(exp_h)
    # k_matched = 450 / mean(100, 200, 300) = 2.25 -> k_used 2
    assert s["k_matched"] == pytest.approx(2.25) and s["k_used"] == 2
    assert s["tok_ratio_f2_f0"] == pytest.approx(0.75) and bool(s["k_matched_le_128"])
    c2 = curves[(curves["fold"] == 0) & (curves["k"] == 2)].iloc[0]
    assert s["rec_sel_bestk"] == pytest.approx(c2["rec_sel_bestk"])


def test_toy_pool_deterministic_and_cached(toy):
    root, f2 = toy
    a = _run(root, f2, root / "o1")
    b = _run(root, f2, root / "o2", "--cpus", "2")
    c = _run(root, f2, root / "o1")  # from cache/
    drop = ["wall_s"]
    pd.testing.assert_frame_equal(a[0].drop(columns=drop), b[0].drop(columns=drop))
    pd.testing.assert_frame_equal(a[0].drop(columns=drop), c[0].drop(columns=drop))
    for i in (1, 2):
        pd.testing.assert_frame_equal(a[i], b[i])
        pd.testing.assert_frame_equal(a[i], c[i])
