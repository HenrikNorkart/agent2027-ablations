"""A02 (M4) read-out, A02-T12: extractor rounds R in {1, 3, 10}.

Spec: ``docs/ablations/A02_M4_extractor_rounds.md`` (sections 2, 4, 5.4, 6) and the
binding rules in ``docs/agent2027/preregistration.md`` section "A02 M4" (committed
before launch; Amendment 1: sessions S1013c/S1013d on ``agent2027-pre.3``, the
aborted S1013a/S1013b runs are excluded).

R rule (pre-registered; outcome ``rec_sel``, reference ``R10``, all 8 tasks / 40
fold-episodes). Adopt R=3 iff all four hold, otherwise R=10; R=1 is never adopted:

1. sum rec_sel(R10) - sum rec_sel(R3) <= 4 (of 40), the net paired loss;
2. no task falls from >= 3/5 under R10 to 0/5 under R3;
3. success count(R3) >= success count(R10) (of 8);
4. mean u_run(R3) <= 0.75 * mean u_run(R10) (8 rows per arm, ITT, a cap hit = 6 h).

ITT: a failed row (no ``fold_aggregation_summary.csv`` after its retry) or a cap
hit scores ``rec_sel = rec_any = 0`` on all 5 episodes and ``success = 0``; its
time still counts in ``u_run``. Fallback features never count as recovery.

Inputs (all read-only; the agent run directories hold the same ``output/`` folder
that the ``run-output`` artifact uploads):

* ``--runs DIR [DIR ...]`` session run roots (``<root>/<row_id>/tasks/<type>/<ds>/output``,
  plus the launcher's ``status.tsv``). ``<row_id>.attempt<k>.*`` directories are
  earlier attempts: only their wall clock is used.
* ``--manifest`` ``manifests/agent2027/A02_M4.csv`` (rows A02-001..024; smoke rows dropped).
* ``--tasks`` canonical SYN-P task folders, ``--meta`` phi metadata (off-mount).
* ``--a08`` A08 CPU screen output root (``f42/arm_<ds>_raw_stratified.json`` gives
  ``base_no_fe``); ``--a08-episodes`` the table-mode episodes CSV (M5S labels).
* W&B (optional, ``--wandb``): configs and summaries of group ``A02`` in
  ``PlanBench/RogueOne-AGENT2027-prefreeze``; used for ``wall_s``, the flag
  invariant and cross-checks. Runs tagged S1013a/S1013b are excluded.

The rule functions at the top are pure and unit-tested
(``tests/unit/agent2027/test_a02_readout.py``).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

ARMS = ("R10", "R3", "R1")
TASKS = ("synp_p1", "synp_p2", "synp_p3", "synp_p4", "synp_p5", "synp_p6a", "synp_p6b", "synp_p7")
CLASSIFICATION = {"synp_p3", "synp_p4", "synp_p5", "synp_p7"}
N_FOLDS = 5
CAP_S = 6 * 3600.0
LOSS_MAX = 4               # criterion 1
DROP_FROM = 3              # criterion 2: >= 3/5 under R10 ...
DROP_TO = 0                # ... to 0/5 under R3
U_RATIO_MAX = 0.75         # criterion 4
SUCCESS_SD_MULT = 2.0      # success: no worse than base_no_fe by > 2 fold SD
EXCLUDED_SESSION_TAGS = {"S1013a", "S1013b"}
VALID_SESSION_TAGS = {"S1013c", "S1013d"}
W_REF_A01_FALLBACK_S = 7886.0  # A01 session plan: median wall_s over the 16 W0 runs (planning only)
INVARIANT_EXEMPT = {"flags/ROGUE_ONE_EXTRACTOR_ROUNDS", "flags/ROGUE_ONE_STRATIFIED_FOLDS"}
# Differ by design, reported but not a violation: the task itself, and the vLLM port of
# the 2xTP2 deviation (S1013c :8010, S1013d :8011; session plan S1013a.md).
INVARIANT_BY_DESIGN = {"flags/ROGUE_ONE_DATASETS": "dataset under test",
                       "flags/LLM_PORT": "2xTP2 servers (S1013c :8010, S1013d :8011)"}


# =========================================================================== rule (pure)
def criterion1(sum_r10: int, sum_r3: int, loss_max: int = LOSS_MAX) -> dict[str, Any]:
    loss = int(sum_r10) - int(sum_r3)
    return {"id": 1, "name": "net paired loss sum rec_sel(R10) - sum rec_sel(R3) <= 4",
            "value": loss, "threshold": loss_max, "pass": bool(loss <= loss_max),
            "detail": f"R10 {sum_r10}/40, R3 {sum_r3}/40, loss {loss}"}


def criterion2(per_task_r10: Mapping[str, int], per_task_r3: Mapping[str, int]) -> dict[str, Any]:
    drops = sorted(t for t in per_task_r10
                   if int(per_task_r10[t]) >= DROP_FROM and int(per_task_r3.get(t, 0)) <= DROP_TO)
    return {"id": 2, "name": "no task falls from >= 3/5 (R10) to 0/5 (R3)",
            "value": len(drops), "threshold": 0, "pass": not drops,
            "detail": ("drops: " + ", ".join(f"{t} {per_task_r10[t]}->{per_task_r3.get(t, 0)}" for t in drops))
            if drops else "no drop"}


def criterion3(success_r3: int, success_r10: int) -> dict[str, Any]:
    return {"id": 3, "name": "success count R3 >= R10 (of 8)", "value": int(success_r3) - int(success_r10),
            "threshold": 0, "pass": bool(int(success_r3) >= int(success_r10)),
            "detail": f"R3 {success_r3}/8, R10 {success_r10}/8"}


def criterion4(mean_u_r3: float, mean_u_r10: float, ratio_max: float = U_RATIO_MAX) -> dict[str, Any]:
    ratio = float(mean_u_r3) / float(mean_u_r10) if mean_u_r10 else float("inf")
    return {"id": 4, "name": "mean u_run(R3) <= 0.75 x mean u_run(R10)", "value": ratio,
            "threshold": ratio_max, "pass": bool(float(mean_u_r3) <= ratio_max * float(mean_u_r10)),
            "detail": f"mean u R3 {mean_u_r3:.4f}, R10 {mean_u_r10:.4f}, ratio {ratio:.4f}"}


def decide_r(criteria: Sequence[Mapping[str, Any]]) -> int:
    """R=3 iff all four criteria pass, else R=10. R=1 is never a possible output."""
    ids = sorted(int(c["id"]) for c in criteria)
    if ids != [1, 2, 3, 4]:
        raise ValueError(f"need exactly criteria 1-4, got {ids}")
    return 3 if all(bool(c["pass"]) for c in criteria) else 10


def apply_r_rule(task_arm: pd.DataFrame, rows: pd.DataFrame) -> dict[str, Any]:
    """``task_arm``: columns arm, dataset_id, rec_sel (0..5, ITT already applied).
    ``rows``: one row per manifest row with arm, success (0/1), u_run.
    Only R10 and R3 enter; R1 rows are ignored by construction."""
    ta = task_arm[task_arm["arm"].isin(["R10", "R3"])]
    per = {a: ta[ta["arm"] == a].set_index("dataset_id")["rec_sel"].astype(int).to_dict() for a in ("R10", "R3")}
    rr = rows[rows["arm"].isin(["R10", "R3"])]
    succ = rr.groupby("arm")["success"].sum().astype(int).to_dict()
    u = rr.groupby("arm")["u_run"].mean().to_dict()
    crit = [
        criterion1(sum(per["R10"].values()), sum(per["R3"].values())),
        criterion2(per["R10"], per["R3"]),
        criterion3(succ.get("R3", 0), succ.get("R10", 0)),
        criterion4(u.get("R3", float("nan")), u.get("R10", float("nan"))),
    ]
    return {"criteria": crit, "decision_R": decide_r(crit)}


def itt_episode_rows(row_ok: bool, episodes: Optional[pd.DataFrame], n_folds: int = N_FOLDS) -> pd.DataFrame:
    """ITT: a failed / capped row contributes ``n_folds`` zero episodes; an OK row
    keeps its scored episodes, and any missing fold is added as a zero episode."""
    base = pd.DataFrame({"fold": range(n_folds), "rec_sel": 0, "rec_any": 0, "itt_imputed": 1})
    if not row_ok or episodes is None or not len(episodes):
        return base
    ep = episodes.copy()
    ep["itt_imputed"] = 0
    missing = sorted(set(range(n_folds)) - set(ep["fold"].astype(int)))
    if missing:
        ep = pd.concat([ep, base[base["fold"].isin(missing)]], ignore_index=True)
    return ep.sort_values("fold").reset_index(drop=True)


def is_success(task_type: str, score: Optional[float], base_mean: Optional[float], base_sd: Optional[float],
               finished_in_cap: bool, mult: float = SUCCESS_SD_MULT) -> int:
    """Run finished within the cap AND held-out score no worse than base_no_fe by
    more than ``mult`` x base fold SD. Direction-aware: regression (nRMSE) worse =
    higher; classification (balanced accuracy) worse = lower."""
    if not finished_in_cap or score is None or base_mean is None or base_sd is None:
        return 0
    if any(isinstance(v, float) and math.isnan(v) for v in (score, base_mean, base_sd)):
        return 0
    if task_type == "regression":
        return int(float(score) <= float(base_mean) + mult * float(base_sd))
    return int(float(score) >= float(base_mean) - mult * float(base_sd))


def u_run(attempt_walls: Iterable[Optional[float]], cap_hits: Iterable[bool], w_ref: float) -> float:
    """Sum of wall clock over the row's attempts / W_ref; a cap hit counts as 6 h."""
    total = 0.0
    for w, cap in zip(attempt_walls, cap_hits):
        total += CAP_S if cap else float(w or 0.0)
    return total / float(w_ref)


def m5s_gate(episodes: pd.DataFrame, min_hits: int = 14) -> dict[str, Any]:
    """A08 table-mode episodes -> M5S gate. A task is non-discriminating if OpenFE
    or AutoFeat (judged separately) recovers in >= ``min_hits`` of its 15 episodes;
    families out of 7 (P6 solved if P6a or P6b); p0 must have 0 recoveries."""
    ep = episodes[episodes["arm"].isin(["openfe", "autofeat"])]
    hits = ep.groupby(["dataset_id", "arm"])["rec_any"].agg(["sum", "count"]).reset_index()
    nondisc_tasks = sorted(set(hits.loc[(hits["sum"] >= min_hits) & (hits["dataset_id"] != "synp_p0"),
                                        "dataset_id"]))
    fam = {t: ("P6" if t in ("synp_p6a", "synp_p6b") else t.replace("synp_p", "P")) for t in nondisc_tasks}
    families = sorted(set(fam.values()))
    p0 = int(episodes.loc[episodes["dataset_id"] == "synp_p0", "rec_any"].sum())
    return {"hits": hits, "nondisc_tasks": nondisc_tasks, "nondisc_families": families,
            "n_nondisc_families": len(families), "p0_recoveries": p0,
            "gate_pass": bool(len(families) < 4 and p0 == 0)}


# =========================================================================== run-dir reading
_TRAIL_RE = re.compile(r"df_attributes_explanations_trail_(\d+)\.csv$")


def _status_table(run_roots: Sequence[Path]) -> pd.DataFrame:
    frames = []
    for root in run_roots:
        p = Path(root) / "status.tsv"
        if p.exists():
            frames.append(pd.read_csv(p, sep="\t").assign(session=Path(root).name))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def find_output_dir(run_roots: Sequence[Path], row_id: str) -> tuple[Optional[Path], Optional[str]]:
    for root in run_roots:
        hits = sorted((Path(root) / row_id).glob("tasks/*/*/output"))
        if hits:
            return hits[0], Path(root).name
    return None, None


def read_trails(out_dir: Path) -> list[dict[str, Any]]:
    trails = []
    for f in (out_dir / "intermediate_steps").glob("df_attributes_explanations_trail_*.csv"):
        m = _TRAIL_RE.search(f.name)
        df = pd.read_csv(f)
        if not len(df):
            continue
        trails.append({
            "index": int(m.group(1)), "trial": str(df["Trial"].iloc[0]),
            "fold_run_id": int(df["fold_run_id"].iloc[0]), "iteration_in_fold": int(df["iteration_in_fold"].iloc[0]),
            "attributes": df,
            "active": df.loc[df["Status"].astype(str) == "Active", "Attribute"].astype(str).tolist(),
        })
    return sorted(trails, key=lambda t: t["index"])


def _attr_key(df: pd.DataFrame) -> list[tuple[str, str]]:
    return list(zip(df["Attribute"].astype(str), df["Pandas Command"].astype(str)))


def build_row_episodes(out_dir: Path) -> list[dict[str, Any]]:
    """One dict per outer fold: the attribute table(s), active sets per iteration
    and the inner-CV-selected iteration, from the run's own per-trial tables."""
    trails = read_trails(out_dir)
    it_log = [json.loads(l) for l in (out_dir / "iteration_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    run_to_test = {int(r["fold_run_id"]): int(r["test_fold_id"]) for r in it_log}
    best = pd.read_csv(out_dir / "fold_aggregation_best_iterations.csv")
    episodes = []
    for fold in range(N_FOLDS):
        b = best[best["test_fold_id"] == fold]
        sel_trail = None
        if len(b) and str(b["status"].iloc[0]) == "completed":
            sel_trail = next((t for t in trails if t["trial"] == str(b["id"].iloc[0])), None)
        if sel_trail is not None:
            fold_run = sel_trail["fold_run_id"]
        else:
            runs = sorted(r for r, tf in run_to_test.items() if tf == fold)
            fold_run = runs[-1] if runs else None
        ft = [t for t in trails if t["fold_run_id"] == fold_run]
        n_runs_for_fold = sum(1 for tf in run_to_test.values() if tf == fold)
        episodes.append({
            "fold": fold, "fold_run_id": fold_run, "trails": ft,
            "selected_iter": None if sel_trail is None else sel_trail["iteration_in_fold"],
            "selected_trial": None if sel_trail is None else sel_trail["trial"],
            "n_fold_runs": n_runs_for_fold,
        })
    return episodes


def score_episode(ep: Mapping[str, Any], df_raw: pd.DataFrame, labels: pd.DataFrame, phi: np.ndarray,
                  det, fold_seed: int, stratified: bool, *, amended: bool = True) -> dict[str, Any]:
    """Formula-mode detection for one fold-episode (recovery_detector.detect_episode).
    ``amended`` (default) applies the Part III.2 raw-column exclusions
    (``no_gain_over_raw`` etc.); ``False`` is the registered pre-amendment rule that
    produced the original A02 read-out.
    If every iteration's attribute table is a prefix of the fold's last table (the
    append-only pool), one replay of the last table scores all iterations; else
    each iteration is replayed with its own table."""
    from scripts.agent2027.analysis import recovery_detector as rd  # noqa: PLC0415

    trails = ep["trails"]
    mask = rd.train_mask_for_fold(df_raw, labels, fold_seed, ep["fold"], stratified=stratified)
    if not trails:
        return {"rec_any": 0, "rec_sel": 0, "iter_first_rec": None, "mode": "no_trails", "hits_sel": "",
                "best_active_sel_score": None, "n_attributes": 0, "n_active_sel": 0, "n_fallback": 0}
    last = trails[-1]["attributes"]
    last_key = _attr_key(last)
    names_unique = len(set(last["Attribute"].astype(str))) == len(last)
    prefix_ok = names_unique and all(_attr_key(t["attributes"]) == last_key[:len(t["attributes"])] for t in trails)
    active = {t["iteration_in_fold"]: t["active"] for t in trails}
    sel = ep["selected_iter"]
    if prefix_ok:
        res = rd.detect_episode(df_raw, last, mask, phi, det, active, sel, amended=amended)
        hits, pf = res.hits_by_iter, res.per_feature.set_index("feature")
        pfs = {it: pf for it in active}
        mode = "superset"
    else:
        hits, pfs = {}, {}
        for t in trails:
            r = rd.detect_episode(df_raw, t["attributes"], mask, phi, det,
                                  {t["iteration_in_fold"]: t["active"]}, t["iteration_in_fold"],
                                  amended=amended)
            hits.update(r.hits_by_iter)
            pfs[t["iteration_in_fold"]] = r.per_feature.set_index("feature")
        mode = "per_trail"
    best = None
    if sel is not None and sel in pfs:
        pf = pfs[sel]
        act = [a for a in active.get(sel, []) if a in pf.index]
        if act:
            sc = pf.loc[act].apply(lambda r: max(float(r["rho"]), 0.0 if pd.isna(r["nmi"]) else float(r["nmi"])), axis=1)
            best = float(sc.max())
    fb = last.get("Fallback")
    return {
        "rec_any": int(bool(hits)), "rec_sel": int(sel is not None and int(sel) in hits),
        "iter_first_rec": min(hits) if hits else None, "mode": mode,
        "hits_sel": ";".join(hits.get(int(sel), [])) if sel is not None else "",
        "hits_any": ";".join(sorted({h for v in hits.values() for h in v})),
        "best_active_sel_score": best, "n_attributes": int(len(last)),
        "n_active_sel": len(active.get(sel, [])) if sel is not None else 0,
        "n_fallback": int(pd.to_numeric(fb, errors="coerce").fillna(0).astype(bool).sum()) if fb is not None else 0,
    }


def held_out_score(out_dir: Path, task_type: str) -> Optional[float]:
    p = out_dir / "fold_aggregation_summary.csv"
    if not p.exists():
        return None
    s = pd.read_csv(p)
    metric = "accuracy" if task_type == "classification" else "n_RMSE"
    hit = s.loc[s["metric"] == metric, "mean"]
    return float(hit.iloc[0]) if len(hit) else None


def base_no_fe(a08_root: Path, dataset_id: str, task_type: str, seed: int = 42) -> dict[str, Any]:
    """A08 raw arm at seed 42. Regression: nRMSE (sqrt(MSE)/target mean, as the
    agent's n_RMSE). Classification: ``accw_balanced`` (balanced-sample-weight fit,
    balanced accuracy), the estimator matched to A02's ROGUE_ONE_DO_BALANCING=True
    run, whose ``accuracy`` is accuracy_score with balanced test weights = balanced accuracy."""
    p = Path(a08_root) / f"f{seed}" / f"arm_{dataset_id}_raw_stratified.json"
    if not p.exists():
        return {"base_mean": None, "base_sd": None, "base_field": None, "base_folds": None}
    d = json.loads(p.read_text(encoding="utf-8"))
    key = "nrmse" if task_type == "regression" else "accw_balanced"
    return {"base_mean": d.get(key), "base_sd": d.get(f"{key}_sd"), "base_field": key,
            "base_folds": d.get("folds_done"), "base_fold_scheme": d.get("fold_scheme_effective")}


def search_queries(out_dir: Path) -> int:
    p = out_dir / "retrieval_transcript.jsonl"
    if not p.exists():
        return 0
    n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if '"event": "query"' in line:
            n += 1
    return n


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =========================================================================== W&B
def fetch_wandb_runs(entity: str, project: str, group: str) -> list[dict[str, Any]]:
    import wandb  # noqa: PLC0415

    api = wandb.Api(timeout=180)
    out = []
    for r in api.runs(f"{entity}/{project}", filters={"group": group}):
        out.append({"id": r.id, "name": r.name, "state": r.state, "tags": list(r.tags or []),
                    "created_at": str(r.created_at), "config": dict(r.config),
                    "summary": {k: v for k, v in dict(r.summary).items() if not str(k).startswith("_")}})
    return out


def match_wandb(wb: list[dict[str, Any]], manifest: pd.DataFrame, status: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Map W&B runs (valid sessions only) to manifest rows by (arm, dataset);
    attempts ordered by created_at. Returns per-attempt frame and problems."""
    problems = []
    keep = []
    for r in wb:
        tags = set(r["tags"])
        if r["config"].get("code_phase") != "prefreeze":
            continue
        if tags & EXCLUDED_SESSION_TAGS:
            continue
        if not tags & VALID_SESSION_TAGS:
            problems.append(f"W&B run {r['id']} {r['name']} has no S1013c/S1013d tag (tags {sorted(tags)}); ignored")
            continue
        keep.append(r)
    recs = []
    for _, m in manifest.iterrows():
        rs = sorted([r for r in keep if r["config"].get("arm_id") == m["arm_id"]
                     and r["config"].get("dataset_id") == m["dataset_id"]], key=lambda r: r["created_at"])
        st = status[status["row_id"] == m["row_id"]].sort_values("attempt")
        if len(rs) != len(st):
            problems.append(f"{m['row_id']}: {len(rs)} W&B runs vs {len(st)} launcher attempts")
        for k, r in enumerate(rs, start=1):
            recs.append({"row_id": m["row_id"], "attempt": k, "wandb_id": r["id"], "wandb_name": r["name"],
                         "wandb_state": r["state"], "wandb_tags": ",".join(r["tags"]),
                         "wandb_created_at": r["created_at"], "wandb_wall_s": r["summary"].get("wall_s"),
                         "wandb_session": r["config"].get("session_id"),
                         "wandb_image_id": r["config"].get("image_id"), "wandb_git_sha": r["config"].get("git_sha"),
                         "wandb_image_tag": r["config"].get("image_tag"),
                         "flags_rounds": r["config"].get("flags/ROGUE_ONE_EXTRACTOR_ROUNDS"),
                         "wb_fold_agg_nrmse": r["summary"].get("fold_aggregation/n_RMSE_mean"),
                         "wb_fold_agg_acc": r["summary"].get("fold_aggregation/accuracy_mean"),
                         "_config": r["config"]})
    return pd.DataFrame(recs), problems


def flag_invariant(final_configs: Mapping[str, Mapping[str, Any]], manifest: pd.DataFrame) -> dict[str, Any]:
    """Every flags/* value identical across rows except ROUNDS (by arm) and
    STRATIFIED_FOLDS (by task type); those two must match the manifest."""
    keys = sorted({k for c in final_configs.values() for k in c if str(k).startswith("flags/")})
    diffs, by_design = {}, {}
    for k in keys:
        if k in INVARIANT_EXEMPT:
            continue
        vals = {rid: str(c.get(k)) for rid, c in final_configs.items()}
        if len(set(vals.values())) > 1:
            if k in INVARIANT_BY_DESIGN:
                by_design[k] = {"reason": INVARIANT_BY_DESIGN[k], "values": sorted(set(vals.values()))}
            else:
                diffs[k] = vals
    exp_bad = []
    for _, m in manifest.iterrows():
        c = final_configs.get(m["row_id"])
        if c is None:
            continue
        ov = json.loads(m["env_overrides"])
        if str(c.get("flags/ROGUE_ONE_EXTRACTOR_ROUNDS")) != ov["ROGUE_ONE_EXTRACTOR_ROUNDS"]:
            exp_bad.append(f"{m['row_id']} rounds {c.get('flags/ROGUE_ONE_EXTRACTOR_ROUNDS')}")
        want_strat = ov.get("ROGUE_ONE_STRATIFIED_FOLDS", "False")
        if str(c.get("flags/ROGUE_ONE_STRATIFIED_FOLDS")).lower() != want_strat.lower():
            exp_bad.append(f"{m['row_id']} stratified {c.get('flags/ROGUE_ONE_STRATIFIED_FOLDS')} != {want_strat}")
    return {"n_flag_keys": len(keys), "n_rows": len(final_configs), "differing": diffs,
            "differing_by_design": by_design, "by_arm_mismatch": exp_bad,
            "pass": not diffs and not exp_bad and len(final_configs) == len(manifest)}


def w_ref_from_a03(entity: str, projects: Sequence[str]) -> tuple[Optional[float], str]:
    try:
        import wandb  # noqa: PLC0415
        api = wandb.Api(timeout=180)
    except Exception as exc:  # noqa: BLE001
        return None, f"wandb unavailable ({exc})"
    notes = []
    for proj in projects:
        try:
            runs = list(api.runs(f"{entity}/{proj}", filters={"group": "A03"}))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{proj}: {type(exc).__name__}")
            continue
        rep = [r for r in runs if "replace" in str(r.config.get("arm_id", "")).lower()
               and r.config.get("code_phase") != "smoke" and "smoke" not in (r.tags or [])]
        fin = [r.summary.get("wall_s") for r in rep if r.state == "finished" and r.summary.get("wall_s")]
        running = [r for r in rep if r.state == "running"]
        notes.append(f"{proj}: {len(rep)} A03 replace runs, {len(fin)} finished, {len(running)} running")
        if fin and not running:
            return float(np.median(fin)), f"A03 replace median wall_s over {len(fin)} runs ({proj})"
    return None, "; ".join(notes) or "no A03 runs"


# =========================================================================== main
def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - I/O driver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--a08", required=True)
    ap.add_argument("--a08-episodes", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold-seed", type=int, default=42)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--entity", default="PlanBench")
    ap.add_argument("--project", default="RogueOne-AGENT2027-prefreeze")
    args = ap.parse_args(argv)

    from scripts.agent2027.analysis import recovery_detector as rd  # noqa: PLC0415

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    roots = [Path(r) for r in args.runs]
    problems: list[str] = []
    manifest = pd.read_csv(args.manifest, dtype=str)
    manifest = manifest[(manifest["notes"].fillna("") != "smoke")].reset_index(drop=True)
    status = _status_table(roots)
    status["attempt"] = status["attempt"].astype(int)

    # ---------------------------------------------------------------- W&B
    wb_att = pd.DataFrame()
    final_cfg: dict[str, dict] = {}
    if args.wandb:
        try:
            wb = fetch_wandb_runs(args.entity, args.project, "A02")
            wb_att, p = match_wandb(wb, manifest, status)
            problems += p
            for rid, g in wb_att.groupby("row_id"):
                final_cfg[rid] = g.sort_values("attempt").iloc[-1]["_config"]
            n_excl = sum(1 for r in wb if set(r["tags"]) & EXCLUDED_SESSION_TAGS)
            (out / "wandb_note.txt").write_text(
                f"group A02 runs: {len(wb)}; excluded S1013a/S1013b-tagged: {n_excl}; smoke: "
                f"{sum(1 for r in wb if r['config'].get('code_phase') == 'smoke')}\n", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"W&B fetch failed: {type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------- per row
    cache: dict[str, Any] = {}
    row_recs, ep_recs = [], []
    for _, m in manifest.iterrows():
        rid, arm, ds, ttype = m["row_id"], m["arm_id"], m["dataset_id"], m["task_type"]
        stratified = json.loads(m["env_overrides"]).get("ROGUE_ONE_STRATIFIED_FOLDS", "False") == "True"
        st = status[status["row_id"] == rid].sort_values("attempt")
        out_dir, session = find_output_dir(roots, rid)
        final = st.iloc[-1] if len(st) else None
        row_ok = bool(final is not None and final["status"] == "OK" and out_dir is not None
                      and (out_dir / "fold_aggregation_summary.csv").exists())
        cap_flags = [str(s).upper() == "CAP_HIT" for s in st["status"]]
        if not len(st):
            problems.append(f"{rid}: no launcher status line")
        # wall clock per attempt: W&B wall_s, else launcher wall_s
        walls, wall_src = [], []
        for _, a in st.iterrows():
            w = None
            if len(wb_att):
                hit = wb_att[(wb_att["row_id"] == rid) & (wb_att["attempt"] == int(a["attempt"]))]
                if len(hit) and hit.iloc[0]["wandb_wall_s"] is not None and not pd.isna(hit.iloc[0]["wandb_wall_s"]):
                    w = float(hit.iloc[0]["wandb_wall_s"])
                    wall_src.append("wandb")
            if w is None:
                w = float(a["wall_s"])
                wall_src.append("launcher")
            walls.append(w)
        # detector
        if ds not in cache:
            tdir = Path(args.tasks) / ttype / ds
            df_raw = pd.read_csv(tdir / "data.csv")
            labels = pd.read_csv(tdir / "labels.csv")
            phi, det = rd.load_phi(args.meta, ds, df_raw)
            cache[ds] = (df_raw, labels, phi, det, sha256(tdir / "data.csv"))
        df_raw, labels, phi, det, sha_canon = cache[ds]
        scored = None
        if out_dir is not None:
            run_copy = out_dir.parent / "data.csv"
            if run_copy.exists() and sha256(run_copy) != sha_canon:
                problems.append(f"{rid}: run-dir data.csv differs from canonical tasks/{ttype}/{ds}/data.csv")
        if row_ok:
            eps = []
            for ep in build_row_episodes(out_dir):
                r = score_episode(ep, df_raw, labels, phi, det, args.fold_seed, stratified)
                if ep["n_fold_runs"] > 1:
                    problems.append(f"{rid} fold {ep['fold']}: {ep['n_fold_runs']} fold runs (fold retry); "
                                    f"rec_any uses the selected fold run only")
                if ep["selected_iter"] is None:
                    problems.append(f"{rid} fold {ep['fold']}: no completed inner-CV selection -> rec_sel 0")
                eps.append({"fold": ep["fold"], "selected_iter": ep["selected_iter"],
                            "selected_trial": ep["selected_trial"], **r})
            scored = pd.DataFrame(eps)
        ep_df = itt_episode_rows(row_ok, scored)
        for _, e in ep_df.iterrows():
            ep_recs.append({"row_id": rid, "arm": arm, "dataset_id": ds, **e.to_dict()})
        # success
        score = held_out_score(out_dir, ttype) if row_ok else None
        base = base_no_fe(Path(args.a08), ds, ttype, args.fold_seed)
        finished_in_cap = row_ok and not any(cap_flags) and (walls[-1] if walls else CAP_S) <= CAP_S
        succ = is_success(ttype, score, base["base_mean"], base["base_sd"], finished_in_cap)
        summ = {}
        if out_dir is not None and (out_dir / "instrumentation_summary.json").exists():
            summ = json.loads((out_dir / "instrumentation_summary.json").read_text(encoding="utf-8"))
        rec = {
            "row_id": rid, "arm": arm, "dataset_id": ds, "task_type": ttype, "session": session,
            "n_attempts": len(st), "final_status": None if final is None else final["status"],
            "row_ok": int(row_ok), "cap_hit": int(any(cap_flags)), "itt_imputed": int(not row_ok),
            "attempt_walls_s": ";".join(f"{w:.1f}" for w in walls), "wall_sources": ";".join(wall_src),
            "wall_total_s": float(sum(CAP_S if c else w for w, c in zip(walls, cap_flags))),
            "launcher_wall_total_s": float(st["wall_s"].astype(float).sum()) if len(st) else None,
            "rec_sel": int(ep_df["rec_sel"].sum()), "rec_any": int(ep_df["rec_any"].sum()),
            "score_metric": "balanced_accuracy" if ttype == "classification" else "nRMSE",
            "score": score, **base, "success": succ,
        }
        for k in ("wall_s_extractor", "tok_compl_scientist", "tok_compl_extractor", "tok_compl_tester",
                  "tok_prompt_scientist", "tok_prompt_extractor", "tok_prompt_tester", "max_prompt_tokens",
                  "context_errors", "runner_errors", "fallback_share", "fallback_extractor",
                  "runner_runs_extractor", "llm_calls_total"):
            rec[k] = summ.get(k)
        rec["search_queries"] = search_queries(out_dir) if out_dir is not None else None
        if len(wb_att):
            fin = wb_att[wb_att["row_id"] == rid].sort_values("attempt")
            if len(fin):
                f = fin.iloc[-1]
                rec.update({"wandb_id": f["wandb_id"], "wandb_ids_all": ";".join(fin["wandb_id"]),
                            "wandb_state": f["wandb_state"], "flags_rounds": f["flags_rounds"],
                            "wandb_image_id": f["wandb_image_id"], "wandb_git_sha": f["wandb_git_sha"]})
                wb_metric = f["wb_fold_agg_acc"] if ttype == "classification" else f["wb_fold_agg_nrmse"]
                if score is not None and wb_metric is not None and not pd.isna(wb_metric) \
                        and abs(float(wb_metric) - score) > 1e-6:
                    problems.append(f"{rid}: W&B fold_aggregation mean {wb_metric} != local {score}")
                if summ.get("wall_s") is not None and f["wandb_wall_s"] is not None and not pd.isna(f["wandb_wall_s"]) \
                        and abs(float(f["wandb_wall_s"]) - float(summ["wall_s"])) > 1.0:
                    problems.append(f"{rid}: W&B wall_s {f['wandb_wall_s']} != local {summ['wall_s']}")
        row_recs.append(rec)

    rows = pd.DataFrame(row_recs)
    episodes = pd.DataFrame(ep_recs)

    # ---------------------------------------------------------------- W_ref
    w_ref, w_src = (None, "W&B not queried")
    if args.wandb:
        w_ref, w_src = w_ref_from_a03(args.entity, [args.project, "RogueOne-AGENT2027"])
    if w_ref is None:
        w_ref, w_src = W_REF_A01_FALLBACK_S, f"A01 median wall_s 7886 s (PLANNING ONLY; A03 replace rows not available: {w_src})"
    rows["u_run"] = rows["wall_total_s"] / w_ref
    rows["u_run_launcher"] = rows["launcher_wall_total_s"] / w_ref

    # ---------------------------------------------------------------- task x arm, rule
    task_arm = episodes.groupby(["arm", "dataset_id"]).agg(
        rec_sel=("rec_sel", "sum"), rec_any=("rec_any", "sum"), itt_imputed=("itt_imputed", "sum")).reset_index()
    task_arm = task_arm.merge(rows[["arm", "dataset_id", "row_id", "success", "u_run", "score"]],
                              on=["arm", "dataset_id"], how="left")
    rule = apply_r_rule(task_arm, rows)
    arm_summary = rows.groupby("arm").agg(
        rec_sel=("rec_sel", "sum"), rec_any=("rec_any", "sum"), success=("success", "sum"),
        mean_u_run=("u_run", "mean"), mean_u_run_launcher=("u_run_launcher", "mean"),
        mean_wall_s=("wall_total_s", "mean"), mean_wall_s_extractor=("wall_s_extractor", "mean"),
        sum_tok_compl_extractor=("tok_compl_extractor", "sum"), sum_tok_prompt_extractor=("tok_prompt_extractor", "sum"),
        sum_context_errors=("context_errors", "sum"), sum_runner_errors=("runner_errors", "sum"),
        sum_search_queries=("search_queries", "sum"), rows_itt=("itt_imputed", "sum"),
    ).reindex(list(ARMS)).reset_index()
    u = arm_summary.set_index("arm")["mean_u_run"]
    ratios = {"R3/R10": float(u["R3"] / u["R10"]), "R1/R10": float(u["R1"] / u["R10"])}

    # descriptive: without M5S non-discriminating tasks
    gate = None
    desc = None
    if args.a08_episodes and Path(args.a08_episodes).exists():
        gate = m5s_gate(pd.read_csv(args.a08_episodes))
        nd = set(gate["nondisc_tasks"])
        ta2 = task_arm[~task_arm["dataset_id"].isin(nd)]
        r2 = rows[~rows["dataset_id"].isin(nd)]
        per = {a: ta2[ta2["arm"] == a].set_index("dataset_id")["rec_sel"].astype(int).to_dict() for a in ("R10", "R3")}
        s2 = r2.groupby("arm")["success"].sum().to_dict()
        u2 = r2.groupby("arm")["u_run"].mean().to_dict()
        n_ep = 5 * len(per["R10"])
        desc = {"excluded_tasks": sorted(nd), "n_episodes_per_arm": n_ep,
                "criteria": [criterion1(sum(per["R10"].values()), sum(per["R3"].values())),
                             criterion2(per["R10"], per["R3"]),
                             criterion3(int(s2.get("R3", 0)), int(s2.get("R10", 0))),
                             criterion4(u2.get("R3"), u2.get("R10"))]}

    # descriptive sensitivity of criterion 4 (the rule uses u_run above)
    def _final_wall(s: str) -> float:
        return float(str(s).split(";")[-1])
    rows["wall_final_attempt_s"] = rows["attempt_walls_s"].map(_final_wall)
    mean = rows.groupby("arm")[["u_run_launcher", "wall_final_attempt_s", "tok_prompt_extractor",
                                "tok_compl_extractor", "wall_s_extractor"]].mean()
    sens = {
        "launcher_wall_only": float(mean.loc["R3", "u_run_launcher"] / mean.loc["R10", "u_run_launcher"]),
        "final_attempt_only_not_ITT": float(mean.loc["R3", "wall_final_attempt_s"] / mean.loc["R10", "wall_final_attempt_s"]),
        "wall_s_extractor": float(mean.loc["R3", "wall_s_extractor"] / mean.loc["R10", "wall_s_extractor"]),
        "tok_prompt_extractor": float(mean.loc["R3", "tok_prompt_extractor"] / mean.loc["R10", "tok_prompt_extractor"]),
        "tok_compl_extractor": float(mean.loc["R3", "tok_compl_extractor"] / mean.loc["R10", "tok_compl_extractor"]),
    }
    provenance = {}
    if len(wb_att):
        provenance = {"git_sha": sorted(set(map(str, wb_att["wandb_git_sha"]))),
                      "image_id": sorted(set(map(str, wb_att["wandb_image_id"]))),
                      "image_tag": sorted(set(map(str, wb_att["wandb_image_tag"]))),
                      "n_wandb_runs_valid": int(len(wb_att))}

    p7 = episodes[episodes["dataset_id"] == "synp_p7"]
    canary = {"rec_any_total": int(p7["rec_any"].sum()), "rec_sel_total": int(p7["rec_sel"].sum()),
              "rec_any_R10_R3": int(p7.loc[p7["arm"].isin(["R10", "R3"]), "rec_any"].sum()),
              "by_arm": p7.groupby("arm")[["rec_any", "rec_sel"]].sum().to_dict(orient="index")}
    inv = flag_invariant(final_cfg, manifest) if final_cfg else {"pass": None, "note": "W&B configs not available"}

    # ---------------------------------------------------------------- write
    rows.to_csv(out / "a02_rows.csv", index=False)
    episodes.to_csv(out / "a02_episodes.csv", index=False)
    task_arm.to_csv(out / "a02_task_arm.csv", index=False)
    arm_summary.to_csv(out / "a02_arm_summary.csv", index=False)
    if len(wb_att):
        wb_att.drop(columns=["_config"]).to_csv(out / "a02_wandb_attempts.csv", index=False)
    result = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "decision_R": rule["decision_R"], "criteria": rule["criteria"],
        "w_ref_s": w_ref, "w_ref_source": w_src, "u_ratios": ratios,
        "mean_u_per_arm": u.to_dict(), "criterion4_sensitivity_R3_over_R10": sens, "provenance": provenance,
        "leak_canary_p7": canary, "flag_invariant": inv,
        "descriptive_without_nondisc": desc,
        "m5s_gate": None if gate is None else {k: v for k, v in gate.items() if k != "hits"},
        "retier_trigger": {"R3_not_adopted": rule["decision_R"] != 3,
                           "R10_mean_u_gt_0.85": bool(u["R10"] > 0.85),
                           "apply_CL1_CL4": bool(rule["decision_R"] != 3 or u["R10"] > 0.85)},
        "problems": problems,
    }
    (out / "a02_readout.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("decision_R", "criteria", "w_ref_source", "u_ratios",
                                             "leak_canary_p7", "retier_trigger")}, indent=1, default=str))
    print(task_arm.pivot(index="dataset_id", columns="arm", values="rec_sel").to_string())
    print(f"problems: {len(problems)}")
    for p in problems:
        print("  -", p)
    return 0


if __name__ == "__main__":  # pragma: no cover
    root = str(Path(__file__).resolve().parents[3])
    if root not in sys.path:
        sys.path.insert(0, root)
    sys.exit(main())
