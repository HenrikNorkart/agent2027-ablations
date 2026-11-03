"""E9c RT3 tooling (A09-T4): σ map, renaming rule, focus-log extraction, build + verify, and
the A09 manifest (A09-T7). No W&B: source focus logs come from a local cache."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parents[3]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


rt3 = _load("rt3_focus", REPO / "scripts" / "agent2027" / "rt3_focus.py")
pm = _load("planner_mode_by_path", REPO / "rogueone" / "utils" / "planner_mode.py")
mm = _load("make_manifest", REPO / "scripts" / "agent2027" / "make_manifest.py")

SIGMA = REPO / "scripts" / "agent2027" / "rt3_sigma_map.yaml"
A09_CSV = REPO / "manifests" / "agent2027" / "A09_RT3.csv"
A05_CSV = REPO / "manifests" / "agent2027" / "A05_M0.csv"


# --------------------------------------------------------------------------- σ map
def test_sigma_map_is_the_preregistered_one():
    s = rt3.load_sigma(SIGMA)
    got = {t: (e["source"], e["fallback"]) for t, e in s["map"].items()}
    assert got == {  # pre-registration Part V.9 / A09 §4.2
        "synp_p1": ("synp_p4", "synp_p5"), "synp_p2": ("synp_p4", "synp_p5"),
        "synp_p3": ("synp_p4", "synp_p5"), "synp_p4": ("synp_p2", "synp_p1"),
        "synp_p5": ("synp_p1", "synp_p2"), "synp_p6a": ("synp_p5", "synp_p4"),
        "synp_p6b": ("synp_p5", "synp_p4"), "synp_p7": ("synp_p4", "synp_p5"),
        "junglechess": ("housing", None),
    }
    assert (s["source_ablation"], s["source_arm"], s["source_fold_seed"]) == ("A05", "full", 42)
    assert s["source_replicates"] == [0, 1] and s["min_rows_per_fold"] == 10
    assert rt3.source_run_name(s, "synp_p4", 1) == "A05__full__synp_p4__f42__r1"
    ratio, other = {"synp_p1", "synp_p2", "synp_p3", "synp_p6a", "synp_p6b", "synp_p7"}, {"synp_p4", "synp_p5"}
    for t, (src, fb) in got.items():
        if t in ratio:
            assert {src, fb} <= other
        if t in other:
            assert {src, fb} <= ratio - {"synp_p7"}


def test_sigma_rejects_self_map_and_p7_source(tmp_path):
    base = SIGMA.read_text(encoding="utf-8")
    for bad in ("synp_p1: {source: synp_p1, fallback: synp_p5}", "synp_p1: {source: synp_p7, fallback: synp_p5}"):
        p = tmp_path / "s.yaml"
        p.write_text(base.replace("synp_p1: {source: synp_p4, fallback: synp_p5}", bad), encoding="utf-8")
        with pytest.raises(rt3.RT3Error):
            rt3.load_sigma(p)


# --------------------------------------------------------------------------- rule 4
def test_rename_boundaries_longest_first_single_pass():
    m = {"LW": "f04", "LD": "f09", "f1": "zz", "site_altitude_m": "a", "site_altitude_m_x": "b"}
    assert rt3.apply_rename("LW_x_LD and LW*LD", m) == "f04_x_f09 and f04*f09"
    assert rt3.apply_rename("f13 f1 xf1 f1a f1_", m) == "f13 zz xf1 f1a zz_"
    assert rt3.apply_rename("`site_altitude_m_x` site_altitude_m", m) == "`b` a"
    assert rt3.apply_rename("lw LW2 LW 2.5", m) == "lw LW2 f04 2.5"  # case-sensitive; numbers kept
    assert rt3.apply_rename("a b", {"a": "b", "b": "a"}) == "b a"  # swap in one pass
    assert rt3.apply_rename("Left-Weight*Left-Distance", {"Left-Weight": "f1", "Left-Distance": "f2"}) == "f1*f2"


def test_names_in_first_occurrence_order():
    assert rt3.names_in("use b then a then b_c and b", ["a", "b", "b_c"]) == ["b", "a", "b_c"]


# --------------------------------------------------------------------------- rules 1-3
def test_rank_decoys_spearman_and_ties():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    df = pd.DataFrame({
        "near": x + rng.normal(scale=0.1, size=500),  # high |rho| with the input
        "phi_in": x,
        "tie1": np.ones(500),  # constant -> rho NaN -> 0
        "far": rng.normal(size=500),
        "tie2": np.ones(500),
        "cat": np.where(x > 0, "b", "a"),  # ordinal codes, correlated
    })
    order = rt3.rank_decoys(df, ["phi_in"])
    assert order[:2] == ["tie1", "tie2"]  # ties keep column order
    assert order.index("far") < order.index("cat") < order.index("near")
    assert rt3.rank_decoys(df, []) == ["near", "phi_in", "tie1", "far", "tie2", "cat"]


def test_build_rename_map_types_fallback_and_cycle():
    d_t = ["n1", "c1", "n2"]
    t_types = {"n1": False, "c1": True, "n2": False}
    s_types = {"A": False, "B": True, "C": True, "D": False, "E": False}
    m = rt3.build_rename_map(["A", "B", "C", "D", "E"], s_types, d_t, t_types)
    # A->n1 (numeric), B->c1 (cat), C: no unused cat -> any unused -> n2, D: all used -> cycle -> n1, E -> n2
    assert m == {"A": "n1", "B": "c1", "C": "n2", "D": "n1", "E": "n2"}


# --------------------------------------------------------------------------- fixtures
K, N = 5, 10


def _task(root: Path, ds: str, ttype: str, df: pd.DataFrame):
    d = root / ttype / ds
    d.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df.insert(0, "id", range(len(df)))
    df.to_csv(d / "data.csv", index=False)


@pytest.fixture
def world(tmp_path):
    rng = np.random.default_rng(1)
    n = 300
    tasks, meta = tmp_path / "tasks", tmp_path / "meta"
    meta.mkdir()
    # target T = synp_p1 (φ = a_in / b_in), source S = synp_p4 (φ = sin(hours)), fallback synp_p5
    a = rng.uniform(1, 5, n)
    _task(tasks, "synp_p1", "regression", pd.DataFrame({
        "a_in": a, "b_in": rng.uniform(1, 5, n), "dec_close": a + rng.normal(0, 0.05, n),
        "dec_far": rng.normal(size=n), "dec_cat": rng.choice(["x", "y"], n), "dec_mid": a + rng.normal(0, 2, n)}))
    _task(tasks, "synp_p4", "classification", pd.DataFrame({
        "hours": rng.uniform(0, 100, n), "temp_c": rng.normal(size=n), "region": rng.choice(["n", "s"], n)}))
    _task(tasks, "synp_p5", "classification", pd.DataFrame({
        "LW": rng.integers(1, 5, n), "LD": rng.integers(1, 5, n), "noise": rng.normal(size=n)}))
    metas = {
        "synp_p1": {"task_id": "synp_p1", "phi": {"expr": "a / b", "cols": {"a": "a_in", "b": "b_in"}},
                    "phi_pandas_expr": "df['a_in'] / df['b_in']",
                    "oracle_focus": "Build exactly this feature with `append_new_attribute`: oracle_feature = df['a_in'] / df['b_in']."},
        "synp_p4": {"task_id": "synp_p4", "phi": {"expr": "np.sin(a)", "cols": {"a": "hours"}},
                    "phi_pandas_expr": "np.sin(df['hours'])", "oracle_focus": "Build exactly: oracle_feature = np.sin(df['hours'])."},
        "synp_p5": {"task_id": "synp_p5", "phi": {"expr": "a * b", "cols": {"a": "LW", "b": "LD"}},
                    "phi_pandas_expr": "df['LW'] * df['LD']", "oracle_focus": "Build exactly: oracle_feature = df['LW'] * df['LD']."},
    }
    for k, v in metas.items():
        (meta / f"{k}.json").write_text(json.dumps(v), encoding="utf-8")
    cache = tmp_path / "cache"

    def write_log(run, folds, text=lambda f, i: f"f{f}i{i}: try `temp_c` * hours, region dummies; 24.0"):
        d = cache / run
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "focus_log.jsonl", "w", encoding="utf-8") as fh:
            for f in folds:
                for i in range(N):
                    fh.write(json.dumps({"fold_run_id": f, "iteration_in_fold": i, "focus": text(f, i),
                                         "source": "heuristic" if (f, i) == (0, 5) else "llm"}) + "\n")

    manifest = tmp_path / "A09_RT3.csv"
    rows = [
        {"row_id": "A09-000", "arm_id": "full", "dataset_id": "synp_p1", "env_overrides": "{}"},
        {"row_id": "A09-001", "arm_id": "oracle", "dataset_id": "synp_p1",
         "env_overrides": json.dumps({"ROGUE_ONE_PLANNER_MODE": "oracle"})},
        {"row_id": "A09-009", "arm_id": "misdirected", "dataset_id": "synp_p1",
         "env_overrides": json.dumps({"ROGUE_ONE_PLANNER_MODE": "misdirected"})},
    ]
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return {"tasks": tasks, "meta": meta, "cache": cache, "write_log": write_log, "manifest": manifest,
            "out": tmp_path / "out", "sigma": rt3.load_sigma(SIGMA)}


def _build(world, *extra):
    return rt3.main(["build", "--manifest", str(world["manifest"]), "--sigma", str(SIGMA),
                     "--meta", str(world["meta"]), "--tasks", str(world["tasks"]), "--out", str(world["out"]),
                     "--cache", str(world["cache"]), "--no-fetch", *extra])


# --------------------------------------------------------------------------- build + verify
def test_build_verify_end_to_end_and_hook_accepts_files(world):
    world["write_log"]("A05__full__synp_p4__f42__r0", range(K))
    assert _build(world, "--verify") == 0
    out = world["out"]
    assert sorted(p.name for p in out.glob("*.json")) == ["A09-001.json", "A09-009.json"]
    oracle = json.loads((out / "A09-001.json").read_text(encoding="utf-8"))
    assert set(oracle) == {"dataset_id", "arm_id", "focus"}
    assert set(oracle["focus"].values()) == {json.loads((world["meta"] / "synp_p1.json").read_text())["oracle_focus"]}
    mis = json.loads((out / "A09-009.json").read_text(encoding="utf-8"))
    side = json.loads((out / "meta" / "A09-009.json").read_text(encoding="utf-8"))
    ren = side["rename"]["synp_p4"]
    assert ren["inputs_t"] == ["a_in", "b_in"]
    assert ren["decoys_t_order"][-1] == "dec_close"  # most correlated with φ's inputs goes last
    assert ren["c_s"] == ["hours", "temp_c", "region"]  # φ_S input first, then first occurrence
    assert ren["map"]["region"] == "dec_cat"  # type-matched
    assert side["heuristic_focus_count"] == 1 and side["source_counts"] == {"heuristic": 1, "llm": 49}
    m = ren["map"]
    assert mis["focus"]["3:7"] == f"f3i7: try `{m['temp_c']}` * {m['hours']}, region dummies; 24.0".replace(
        "region dummies", f"{m['region']} dummies")
    for s in mis["focus"].values():
        assert "a_in" not in s and "b_in" not in s and "hours" not in s
    # The minimal file is exactly what the E9c hook loads.
    for rid, mode in (("A09-001", "oracle"), ("A09-009", "misdirected")):
        plan = pm.load_focus_plan(mode, str(out / f"{rid}.json"), dataset_filter=["synp_p1"], k_folds=K,
                                  num_iterations=N)
        assert plan.focus_for(4, 9)
    assert "A09-001" in json.loads((out / "meta" / "build_report.json").read_text())["files"]


def test_missing_source_folds_use_the_fallback_chain(world):
    world["write_log"]("A05__full__synp_p4__f42__r0", [0, 1, 4])  # folds 2, 3 missing
    world["write_log"]("A05__full__synp_p4__f42__r1", [0, 2], text=lambda f, i: f"r1 f{f}i{i} hours")
    world["write_log"]("A05__full__synp_p5__f42__r0", [3], text=lambda f, i: f"fb f{f}i{i} LW LD noise")
    assert _build(world, "--verify") == 0
    side = json.loads((world["out"] / "meta" / "A09-009.json").read_text(encoding="utf-8"))
    steps = {f: (v["run"], v["chain_step"], v["fallback_used"]) for f, v in side["folds"].items()}
    assert steps["0"] == ("A05__full__synp_p4__f42__r0", 0, False)
    assert steps["2"] == ("A05__full__synp_p4__f42__r1", 1, True)
    assert steps["3"] == ("A05__full__synp_p5__f42__r0", 2, True)
    mis = json.loads((world["out"] / "A09-009.json").read_text(encoding="utf-8"))
    m5 = side["rename"]["synp_p5"]["map"]
    assert mis["focus"]["3:0"] == f"fb f3i0 {m5['LW']} {m5['LD']} {m5['noise']}"
    assert mis["focus"]["2:4"].startswith("r1 f2i4 ")


def test_no_complete_source_fold_is_an_input_error(world):
    world["write_log"]("A05__full__synp_p4__f42__r0", [0, 1, 2, 3])  # fold 4 nowhere
    assert _build(world) == 2


def test_partial_fold_counts_as_missing(world, tmp_path):
    world["write_log"]("A05__full__synp_p4__f42__r0", range(K))
    p = world["cache"] / "A05__full__synp_p4__f42__r0" / "focus_log.jsonl"
    lines = [l for l in p.read_text(encoding="utf-8").splitlines() if '"fold_run_id": 1, "iteration_in_fold": 9' not in l]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    world["write_log"]("A05__full__synp_p4__f42__r1", range(K), text=lambda f, i: f"r1 f{f}i{i}")
    assert _build(world) == 0
    side = json.loads((world["out"] / "meta" / "A09-009.json").read_text(encoding="utf-8"))
    assert side["folds"]["1"]["run"].endswith("r1") and side["folds"]["0"]["run"].endswith("r0")


def test_rename_refuses_to_leave_a_phi_input_name(world):
    # The source string names T's φ input (not a column of S): renaming cannot remove it.
    world["write_log"]("A05__full__synp_p4__f42__r0", range(K), text=lambda f, i: "compute a_in from hours")
    assert _build(world) == 2


def test_verify_catches_tampering(world):
    world["write_log"]("A05__full__synp_p4__f42__r0", range(K))
    assert _build(world) == 0
    args = ["verify", "--manifest", str(world["manifest"]), "--sigma", str(SIGMA), "--meta", str(world["meta"]),
            "--tasks", str(world["tasks"]), "--out", str(world["out"]), "--cache", str(world["cache"]), "--no-fetch"]
    assert rt3.main(args) == 0
    p = world["out"] / "A09-009.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["focus"]["2:2"] = "use b_in"  # φ input leak + differs from the renamed source
    p.write_text(json.dumps(doc), encoding="utf-8")
    po = world["out"] / "A09-001.json"
    odoc = json.loads(po.read_text(encoding="utf-8"))
    odoc["focus"]["0:0"] = "Build exactly: oracle_feature = np.sin(df['hours'])."  # another task's φ
    po.write_text(json.dumps(odoc), encoding="utf-8")
    fails = rt3.verify_files(rt3.focus_rows_of_manifest(rt3.read_manifest(world["manifest"])), world["out"],
                             world["sigma"], rt3.LogStore(world["cache"], fetch=False),
                             tasks_root=world["tasks"], meta_dir=world["meta"])
    text = "\n".join(fails)
    assert "A09-009: misdirected strings differ" in text and "A09-009: φ input name(s) ['b_in']" in text
    assert "A09-001: oracle strings differ" in text and "A09-001: contains the φ of synp_p4" in text
    assert rt3.main(args) == 1


def test_parse_focus_log_last_row_wins(tmp_path):
    p = tmp_path / "focus_log.jsonl"
    p.write_text("\n".join(json.dumps({"fold_run_id": 0, "iteration_in_fold": 0, "focus": s, "source": "llm"})
                           for s in ("first", "retry")) + "\n", encoding="utf-8")
    assert rt3.parse_focus_log(p) == {"0:0": {"focus": "retry", "source": "llm"}}


# --------------------------------------------------------------------------- A09 manifest (A09-T7)
def _csv(path):
    return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))


def test_a09_manifest_matches_the_spec_and_validates():
    rows = mm.validate_rows(mm.read_csv(A09_CSV))
    assert [r["row_id"] for r in rows] == [f"A09-{n:03d}" for n in range(18)]
    by = {r["row_id"]: r for r in rows}
    assert all(r["image_tag"] == "agent2027-v2.1" for r in rows)
    assert all((r["rung_S"], r["rung_E"], r["rung_T"]) == ("O120",) * 3 for r in rows)
    assert all((r["port_S"], r["port_E"], r["port_T"]) == ("8010",) * 3 for r in rows)
    assert all((r["fold_seed"], r["llm_seed"], r["replicate"], r["cap_hours"]) == ("42", "42", "0", "6") for r in rows)
    assert [by[f"A09-{n:03d}"]["dataset_id"] for n in range(1, 9)] == list(mm.SYNP8)
    assert [by[f"A09-{n:03d}"]["dataset_id"] for n in range(9, 16)] == list(mm.SYNP7)
    assert {by[f"A09-{n:03d}"]["arm_id"] for n in range(1, 9)} == {"oracle"}
    assert {by[f"A09-{n:03d}"]["arm_id"] for n in range(9, 18)} == {"misdirected"}
    assert (by["A09-016"]["dataset_id"], by["A09-017"]["dataset_id"]) == ("junglechess", "synp_p7")
    smoke = by["A09-000"]
    assert (smoke["arm_id"], smoke["dataset_id"], smoke["notes"], smoke["expected_u"]) == ("full", "synp_p3", "smoke", "0.17")
    assert json.loads(smoke["env_overrides"]) == {"ROGUE_ONE_SMOKE_FOLDS": "1", "ROGUE_ONE_STRATIFIED_FOLDS": "True"}
    assert by["A09-016"]["expected_u"] == "1.0"
    names = {f"A09__{r['arm_id']}__{r['dataset_id']}__f{r['fold_seed']}__r{r['replicate']}" for r in rows}
    assert len(names) == len(rows)
    # Stratified-fold flag as in A05's f42 r0 row of the same task; planner keys on arm rows only.
    a05 = {r["dataset_id"]: json.loads(r["env_overrides"]) for r in _csv(A05_CSV)
           if r["fold_seed"] == "42" and r["replicate"] == "0" and int(r["row_id"][4:]) < 900}
    for r in rows:
        ov = json.loads(r["env_overrides"])
        assert ov.get("ROGUE_ONE_STRATIFIED_FOLDS") == a05[r["dataset_id"]].get("ROGUE_ONE_STRATIFIED_FOLDS")
        if r["arm_id"] in ("oracle", "misdirected"):
            assert ov["ROGUE_ONE_PLANNER_MODE"] == r["arm_id"]
            assert ov["ROGUE_ONE_PLANNER_FOCUS_FILE"] == "/workspace/focus/focus.json"
        else:
            assert "ROGUE_ONE_PLANNER_MODE" not in ov and "ROGUE_ONE_PLANNER_FOCUS_FILE" not in ov
    assert [r["row_id"] for r in rt3.focus_rows_of_manifest(rt3.read_manifest(A09_CSV))] == \
        [f"A09-{n:03d}" for n in range(1, 18)]


def test_a09_manifest_is_reproducible(tmp_path):
    out = mm.make_ablation("A09", None, tmp_path / "A09_RT3.csv")
    # Compare with line endings normalised (Windows checkouts may hold CRLF).
    assert out.read_bytes().replace(b"\r\n", b"\n") == A09_CSV.read_bytes().replace(b"\r\n", b"\n")
