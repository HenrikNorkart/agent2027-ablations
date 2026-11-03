"""A23 CAP-6 cost Pareto front (E11b; docs/ablations/A23_CAP6_cost_pareto.md, prereg Part V.23). Descriptive only.

Points (A23 section 4.1, prereg V.23): one configuration = one point, y = mean ``rec_sel`` over its fold-episodes in
the population, x = mean cost per SYN-P run. Population = SYN-P 7 minus P1 (Part III.2) at fold seed 42, r0 plus r1
where it exists (4.2). Panel A (O120): ``full``, ``set_phased``, the 8 ``fx_*`` fraction cells, the 4 RV3 foldover
cells (RV3 triggered at G4, 4.3 "added to A"), ``restarts``, ``l0``, ``llmfe``; references ``oracle``,
``misdirected``, ``caafe`` (own task draw, matched Full marker), ``openfe``/``autofeat`` lines. Panel B: ``full``
(= O120), ``f2_<rung>``, ``mix_esmall``, ``mix_1b``, ``fx_pajM_Q4`` (V.23 replaces ``fx_pajM_O20``), ``fx_pajM``.
Reference points never enter a front (R-A23-6). Panels C-E and the F0 curves are not built here.

Rows (4.2, prereg amendments): manifest rows without ``smoke``/``pilot`` in ``notes``; the final attempt is the datum.
Rows of record: A05/A09/A10/A11/A12/A16 rows < 100; **A17-4xx** (Amendment 17 replaces the first wave), **A19-4xx**
(Amendment 18), **A20-0xx** (Amendment 10: "the 6 h rows A20-001…007 stay the pre-registered result"; Amendment 18's
A20-4xx re-run the 12 h sensitivity rows, so they are the labelled sensitivity ``a20_relaunch``). Labelled
sensitivities only: power replicates 8xx (Amendments 11/13/14), inf-guard re-runs 5xx (Amendment 19, each replaces
the row it re-runs; the latest re-run wins). Excluded: A17/A19-0xx first waves, A20-9xx, 9xx smoke/probe rows,
A12-7xx (real data).

ITT (4.6): a failed or capped run has 5 episodes at ``rec_sel`` = 0 (from the episode tables); its cost is what it
spent (``iteration_log.jsonl`` per-iteration tokens; unreached folds cost 0); a run with no output has cost NaN.

Cost axes (2.3): ``tok_compl_total`` (panel A primary), ``flop_proxy`` = sum over roles of
2 * N_active(rung_role) * (prompt + completion) (panel B primary), ``flop_proxy_compl`` (sensitivity d), ``energy_j``
and ``energy_j_equal`` (sensitivity e; only with ``--energy``; energy is valid for a run only at coverage >= 0.95).
Fold cost = the run's per-iteration tokens summed over the fold (4.5); runs without per-fold cost use run cost / 5.

Bootstrap (4.5): paired two-stage cluster bootstrap, seed 20261016, B = 2000: stage 1 resamples the panel's tasks
(same draw for every configuration), stage 2 the 5 fold indices within each drawn task (same draw for all). Per
configuration and axis: 95% CIs, ``p_front`` (share of resamples in which it is non-dominated in its panel; dominance
= rec >= and cost <= with one strict), cost ratio vs Full with 95% CI, paired rec difference vs Full with 90%/95% CIs,
cost per recovered episode. Rules R-A23-1..4 and the section 3 panel classification are applied mechanically; the
token-validity checks behind "measurement failure" (A23-T1/T13, R-A23-5) are not checked here (``--token-invalid``).

    python pareto.py build-points --episodes <csv...> --manifest <csv...> --runs <session dirs...> --out <dir>
    python pareto.py front --out <dir> [--delta 0.167] [--B 2000] [--plot]
    python pareto.py all   (= build-points + front)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

SEED = 20261016
B_DEFAULT = 2000
DELTA = 0.167  # prereg Amendment 12 (A09-T15)
#: N_active per rung (A23 2.3); parameters, not billions, so flop_proxy matches E3's logged value.
ACTIVE_PARAMS = {"Q0.8": 0.8e9, "Q2": 2e9, "Q4": 4e9, "Q9": 9e9, "Q122": 10e9, "O20": 3.6e9, "O120": 5.1e9}
ROLES = ("scientist", "extractor", "tester")
POP_TASKS = ("synp_p2", "synp_p3", "synp_p4", "synp_p5", "synp_p6a", "synp_p6b")
ALL7 = ("synp_p1",) + POP_TASKS
CLASSIFICATION = {"synp_p3", "synp_p4", "synp_p5", "synp_p7"}
AXES = ("tok_compl_total", "flop_proxy", "flop_proxy_compl", "energy_j", "energy_j_equal")
PRIMARY = {"A": "tok_compl_total", "B": "flop_proxy", "AB": "flop_proxy", "caafe": "tok_compl_total"}
FX_FRACTION = ("fx_pAJM", "fx_PaJM", "fx_PAjM", "fx_PAJm", "fx_paJm", "fx_pAjm", "fx_Pajm", "fx_pajM")
FX_FOLDOVER = ("fx_paJM", "fx_pAjM", "fx_pAJm", "fx_pajm")  # Amendment 15 / G4: RV3 triggered
RUNGS = ("Q0.8", "Q2", "Q4", "Q9", "Q122")

# config -> (panels, ablation, arm_id, role). role "front" enters its panels' fronts; "reference" never does.
CONFIGS: dict[str, tuple[tuple[str, ...], str, str, str]] = {
    "full": (("A", "B"), "A05", "full", "front"),
    "set_phased": (("A",), "A16", "set_phased", "front"),
    "restarts": (("A",), "A10", "restarts", "front"),
    "l0": (("A",), "A11", "l0", "front"),
    "llmfe": (("A",), "A08", "llmfe", "front"),
    "oracle": (("A",), "A09", "oracle", "reference"),
    "misdirected": (("A",), "A09", "misdirected", "reference"),
    "caafe": (("caafe",), "A08", "caafe", "reference"),
    "openfe": (("A",), "A08", "openfe", "reference"),
    "autofeat": (("A",), "A08", "autofeat", "reference"),
    **{c: (("A", "B") if c == "fx_pajM" else ("A",), "A12", c, "front") for c in FX_FRACTION + FX_FOLDOVER},
    **{f"f2_{r}": (("B",), "A17", f"f2_{r}", "front") for r in RUNGS},
    "mix_esmall": (("B",), "A19", "mix_esmall", "front"),
    "mix_1b": (("B",), "A19", "mix_1b", "front"),
    "fx_pajM_Q4": (("B",), "A20", "fx_pajM_Q4", "front"),
}
NOT_EVALUATED = {
    "c_fallback_only": "needs a per-run 'all selected features are fallback' flag (process_metrics.py)",
    "f_drift": "needs A06 drift verdicts (--drift session_id,drift_flag)",
    "g_func_success": "needs per-run func_success",
}


# --------------------------------------------------------------------------- row selection
def row_tag(row_id: str, notes: str = "") -> str:
    """'record', 'power', 'infguard', 'a20_relaunch' or 'excluded' (module docstring)."""
    if re.search(r"smoke|pilot", notes or "", re.I):
        return "excluded"
    abl, num = row_id.split("-")[0], int(row_id.split("-")[1])
    blk = num // 100
    if abl in ("A17", "A19"):
        return {4: "record", 5: "infguard"}.get(blk, "excluded")
    if abl == "A20":
        return {0: "record", 4: "a20_relaunch", 5: "infguard"}.get(blk, "excluded")
    if blk == 0:
        return "record"
    if blk == 8:
        return "power"
    return "excluded"


def rerun_of(notes: str) -> Optional[str]:
    m = re.search(r"re-run of (A\d\d-\d+)", notes or "")
    return m.group(1) if m else None


def config_of(ablation: str, arm: str, tag: str) -> Optional[str]:
    if ablation == "A20" and arm == "fx_pajM_Q4_cap12" and tag in ("a20_relaunch", "infguard"):
        return "fx_pajM_Q4"
    for c, (_, abl, a, _) in CONFIGS.items():
        if abl == ablation and a == arm:
            return c
    return None


# --------------------------------------------------------------------------- costs
def _rung_params(rung: str) -> float:
    return ACTIVE_PARAMS.get(str(rung), np.nan)


def role_params(m: Mapping[str, str]) -> dict[str, float]:
    """N_active per role from the manifest rungs; 'unattributed' calls go to the global LLM_MODEL."""
    p = {r: _rung_params(m.get(f"rung_{r[0].upper()}", "")) for r in ROLES}
    env = json.loads(m.get("env_overrides") or "{}") if isinstance(m.get("env_overrides"), str) else {}
    glob = env.get("LLM_MODEL", "")
    p["unattributed"] = ACTIVE_PARAMS["Q122"] if "122B" in glob else p["scientist"]
    return p


def flop(prompt: Mapping[str, float], compl: Mapping[str, float], params: Mapping[str, float],
         compl_only: bool = False) -> float:
    return float(sum(2.0 * params.get(r, np.nan) * ((0 if compl_only else prompt.get(r, 0)) + compl.get(r, 0))
                     for r in set(prompt) | set(compl) if (prompt.get(r, 0) or compl.get(r, 0))))


def run_costs(out_dir: Optional[Path], m: Mapping[str, str], n_folds: int = 5) -> tuple[dict[str, Any], np.ndarray]:
    """Run-level cost dict and a (n_folds, 3) fold-cost array [tok_compl, flop, flop_compl]."""
    params = role_params(m)
    folds = np.zeros((n_folds, 3))
    res: dict[str, Any] = {"cost_source": "missing", "tok_compl_total": np.nan, "tok_prompt_total": np.nan,
                           "flop_proxy": np.nan, "flop_proxy_compl": np.nan}
    if out_dir is None:
        return res, folds * np.nan
    s = {}
    if (out_dir / "instrumentation_summary.json").exists():
        s = json.loads((out_dir / "instrumentation_summary.json").read_text(encoding="utf-8"))
    log = out_dir / "iteration_log.jsonl"
    it = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()] if log.exists() else []
    for r in it:
        f = int(r.get("test_fold_id", r.get("fold_run_id", 0)))
        pr = {k: v.get("prompt", 0) for k, v in r.get("tokens", {}).items()}
        co = {k: v.get("compl", 0) for k, v in r.get("tokens", {}).items()}
        if 0 <= f < n_folds:
            folds[f] += [sum(co.values()), flop(pr, co, params), flop(pr, co, params, True)]
    if s:
        pr = {r: s.get(f"tok_prompt_{r}", 0) for r in ROLES + ("unattributed",)}
        co = {r: s.get(f"tok_compl_{r}", 0) for r in ROLES + ("unattributed",)}
        res.update(cost_source="summary", tok_compl_total=float(s.get("tok_compl_total", sum(co.values()))),
                   tok_prompt_total=float(s.get("tok_prompt_total", sum(pr.values()))),
                   flop_proxy=flop(pr, co, params), flop_proxy_compl=flop(pr, co, params, True),
                   flop_proxy_logged=s.get("flop_proxy"), wall_s=s.get("wall_s"),
                   llm_calls_total=s.get("llm_calls_total"), fallback_share=s.get("fallback_share"),
                   iterations_total=s.get("iterations_total"))
    elif it:
        res.update(cost_source="iteration_log", tok_compl_total=float(folds[:, 0].sum()),
                   tok_prompt_total=float(sum(v.get("prompt", 0) for r in it for v in r["tokens"].values())),
                   flop_proxy=float(folds[:, 1].sum()), flop_proxy_compl=float(folds[:, 2].sum()),
                   iterations_total=len(it))
    if not it and s:  # no per-iteration tokens: run cost / 5
        folds = np.tile([res["tok_compl_total"], res["flop_proxy"], res["flop_proxy_compl"]], (n_folds, 1)) / n_folds
        res["fold_cost_source"] = "run/5"
    else:
        res["fold_cost_source"] = "iteration_log" if it else "missing"
    if it and s:
        res["iter_tok_coverage"] = float(folds[:, 0].sum() / max(res["tok_compl_total"], 1))
    return res, folds


def final_status(root: Path, row_id: str) -> str:
    st = root / "status.tsv"
    last = "MISSING"
    if st.exists():
        for r in csv.DictReader(st.open(encoding="utf-8"), delimiter="\t"):
            if r.get("row_id") == row_id:
                last = r.get("status", "?")
    return last


def locate(roots: Sequence[Path], rid: str) -> tuple[Optional[Path], str]:
    have = [r for r in roots if (r / rid).exists()]
    root = next((r for r in have if final_status(r, rid) == "OK"), have[-1] if have else None)
    if root is None:
        return None, "MISSING"
    outs = sorted((root / rid).glob("tasks/*/*/output"))
    return (outs[0] if outs else None), final_status(root, rid)


# --------------------------------------------------------------------------- build-points
def build_points(episodes: Sequence[str], manifests: Sequence[str], runs: Sequence[str],
                 external: Sequence[str] = (), energy: Optional[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(points: one row per run, units: one row per run x fold) for every configured row with episodes."""
    man: dict[str, dict[str, str]] = {}
    for f in manifests:
        for r in csv.DictReader(open(f, newline="", encoding="utf-8")):
            man[r["row_id"]] = r
    eps = {}
    for f in episodes:  # later files override earlier ones row by row
        df = pd.read_csv(f)
        for rid, g in df.groupby("row_id"):
            eps[rid] = g
    en = pd.read_csv(energy).set_index("row_id") if energy else None
    roots = [Path(r) for r in runs]
    pts, units = [], []
    for rid, g in eps.items():
        m = man.get(rid)
        if m is None:
            continue
        tag = row_tag(rid, m.get("notes", ""))
        cfg = config_of(m["ablation_id"], m["arm_id"], tag)
        if cfg is None or tag == "excluded":
            continue
        if "status" in g and str(g["status"].iloc[-1]) in ("MISSING", "PENDING"):
            continue  # not run (yet): a source that is still running or was cut leaves no point (4.6)
        out_dir, status = locate(roots, rid) if roots else (None, "?")
        cost, fc = run_costs(out_dir, m)
        st = str(g["status"].iloc[-1]) if "status" in g else status
        p = {"row_id": rid, "config": cfg, "tag": tag, "rerun_of": rerun_of(m.get("notes", "")),
             "ablation_id": m["ablation_id"], "arm_id": m["arm_id"], "dataset_id": m["dataset_id"],
             "fold_seed": int(m["fold_seed"]), "replicate": int(m["replicate"]), "status": st,
             "itt_imputed": int(g.get("itt_imputed", pd.Series([0])).astype(int).max()),
             "n_episodes": len(g), "rec_sel_sum": int(g["rec_sel"].sum()), "rec_sel_mean": float(g["rec_sel"].mean()),
             "rung_S": m.get("rung_S"), "rung_E": m.get("rung_E"), "rung_T": m.get("rung_T"), **cost}
        e_fold = np.full(5, np.nan)
        e_fold_eq = np.full(5, np.nan)
        if en is not None and rid in en.index:
            r = en.loc[rid]
            ok = float(r.get("energy_coverage", 1.0)) >= 0.95  # R-A23-5
            p["energy_coverage"] = r.get("energy_coverage")
            p["energy_j"] = float(r["energy_j"]) if ok else np.nan
            p["energy_j_equal"] = float(r.get("energy_j_equal", np.nan)) if ok else np.nan
            e_fold[:], e_fold_eq[:] = p["energy_j"] / 5, p["energy_j_equal"] / 5
        pts.append(p)
        for _, e in g.iterrows():
            f = int(e["fold"])
            units.append({"row_id": rid, "config": cfg, "tag": tag, "dataset_id": m["dataset_id"],
                          "fold_seed": int(m["fold_seed"]), "replicate": int(m["replicate"]), "fold": f,
                          "rec_sel": int(e["rec_sel"]), "tok_compl_total": fc[f, 0], "flop_proxy": fc[f, 1],
                          "flop_proxy_compl": fc[f, 2], "energy_j": e_fold[f], "energy_j_equal": e_fold_eq[f]})
    for f in external:  # external arms: config,dataset_id,fold_seed,replicate,fold,rec_sel[,run costs]
        x = pd.read_csv(f)
        for (cfg, ds, seed, rep), g in x.groupby(["config", "dataset_id", "fold_seed", "replicate"]):
            rid = f"{cfg}:{ds}:f{seed}:r{rep}"
            c = {a: float(g[a].iloc[0]) if a in g else np.nan for a in AXES}
            pts.append({"row_id": rid, "config": cfg, "tag": "record", "ablation_id": CONFIGS.get(cfg, ("", "A08"))[1],
                        "arm_id": cfg, "dataset_id": ds, "fold_seed": int(seed), "replicate": int(rep),
                        "status": "OK", "itt_imputed": 0, "n_episodes": len(g), "rec_sel_sum": int(g.rec_sel.sum()),
                        "rec_sel_mean": float(g.rec_sel.mean()), "cost_source": "external", "fold_cost_source": "run/5",
                        **c})
            for _, e in g.iterrows():
                units.append({"row_id": rid, "config": cfg, "tag": "record", "dataset_id": ds, "fold_seed": int(seed),
                              "replicate": int(rep), "fold": int(e["fold"]), "rec_sel": int(e["rec_sel"]),
                              **{a: c[a] / 5 for a in AXES}})
    return pd.DataFrame(pts), pd.DataFrame(units)


# --------------------------------------------------------------------------- selection for one analysis variant
def select(units: pd.DataFrame, points: pd.DataFrame, variant: str = "primary") -> pd.DataFrame:
    """Units entering one analysis (population, seeds, tags) per A23 4.2 and the labelled sensitivities."""
    tasks = ALL7 if variant == "a_all7" else POP_TASKS
    seeds = (42, 4) if variant == "b_allseeds" else (42,)
    u = units[units.dataset_id.isin(tasks) & units.fold_seed.isin(seeds)]
    tags = {"record"}
    if variant == "power":
        tags.add("power")
    if variant == "a20_relaunch":
        u = u[~((u.config == "fx_pajM_Q4") & (u.tag == "record"))]
        tags.add("a20_relaunch")
    base = u[u.tag.isin(tags)]
    if variant == "infguard":
        # each row of record with an inf-guard re-run (Amendment 19/19b) is replaced by its latest re-run that has
        # episodes; A20-5xx re-run A20-4xx (not rows of record), so they never enter here
        rer = points[(points.tag == "infguard") & points.rerun_of.isin(base.row_id.unique())]
        latest = rer.sort_values("row_id").groupby("rerun_of").row_id.last()
        base = pd.concat([base[~base.row_id.isin(latest.index)], u[u.row_id.isin(latest.values)]])
    return base


# --------------------------------------------------------------------------- dominance and bootstrap
def front_mask(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Non-dominated set: a dominates b iff y_a >= y_b and x_a <= x_b with one strict. NaN points are never on it."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    ok = np.isfinite(y) & np.isfinite(x)
    on = ok.copy()
    for i in np.flatnonzero(ok):
        dom = ok & (y >= y[i]) & (x <= x[i]) & ((y > y[i]) | (x < x[i]))
        if dom.any():
            on[i] = False
    return on


def config_arrays(u: pd.DataFrame, configs: Sequence[str], tasks: Sequence[str], axes: Sequence[str]):
    """Per config: rec (T, M, 5) and cost[axis] (T, M, 5) arrays, NaN-padded; M = max runs per task."""
    out = {}
    for c in configs:
        uc = u[u.config == c]
        runs = {t: sorted(uc[uc.dataset_id == t].row_id.unique()) for t in tasks}
        M = max([len(v) for v in runs.values()] + [1])
        R = np.full((len(tasks), M, 5), np.nan)
        C = {a: np.full((len(tasks), M, 5), np.nan) for a in axes}
        for ti, t in enumerate(tasks):
            for mi, rid in enumerate(runs[t]):
                for _, e in uc[uc.row_id == rid].iterrows():
                    R[ti, mi, int(e.fold)] = e.rec_sel
                    for a in axes:
                        C[a][ti, mi, int(e.fold)] = e[a]
        out[c] = (R, C)
    return out


def _stat(R, C, td, fd):
    """y and x (= 5 x mean fold cost) for one draw; td (T,), fd (T, 5); None draws = the observed sample."""
    if td is not None:
        R = np.take_along_axis(R[td], fd[:, None, :].repeat(R.shape[1], 1), 2)
        C = {a: np.take_along_axis(v[td], fd[:, None, :].repeat(v.shape[1], 1), 2) for a, v in C.items()}
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            y = np.nanmean(R) if np.isfinite(R).any() else np.nan
            x = {a: (5 * np.nanmean(v[np.isfinite(R)]) if np.isfinite(v[np.isfinite(R)]).any() else np.nan)
                 for a, v in C.items()}
    return y, x


def _ci(v, lo, hi):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi))) if len(v) else (np.nan, np.nan)


def wording(p: float) -> str:
    """R-A23-1."""
    if not np.isfinite(p):
        return "n/a"
    return "on the front" if p >= 0.8 else ("near the front" if p >= 0.5 else "dominated")


def panel_front(u: pd.DataFrame, points: pd.DataFrame, panel: str, *, B: int = B_DEFAULT, seed: int = SEED,
                delta: float = DELTA, axes: Sequence[str] = AXES, ref: str = "full",
                tasks: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """One panel: point estimates, bootstrap CIs, p_front per axis and the R-A23 columns."""
    if panel == "AB":
        members = [c for c, v in CONFIGS.items() if set(v[0]) & {"A", "B"}]
    else:
        members = [c for c, v in CONFIGS.items() if panel in v[0]]
    members = [c for c in members if (u.config == c).any()]
    if panel == "caafe":
        tasks = tasks or [t for t in POP_TASKS if t in CLASSIFICATION]
    tasks = list(tasks or sorted(set(u.dataset_id) & set(ALL7)))
    tasks = [t for t in tasks if (u.dataset_id == t).any()]
    if not members or not tasks:
        return pd.DataFrame()
    # caafe and its matched Full marker are reference only (R-A23-6): no front in that sub-panel
    front = [] if panel == "caafe" else [c for c in members if CONFIGS[c][3] == "front"]
    arr = config_arrays(u, members, tasks, axes)
    rng = np.random.default_rng(seed)
    T = len(tasks)
    ys = {c: np.full(B, np.nan) for c in members}
    xs = {c: {a: np.full(B, np.nan) for a in axes} for c in members}
    onf = {c: {a: np.zeros(B, bool) for a in axes} for c in members}
    for b in range(B):
        td = rng.integers(0, T, T)
        fd = rng.integers(0, 5, (T, 5))
        for c in members:
            y, x = _stat(*arr[c], td, fd)
            ys[c][b] = y
            for a in axes:
                xs[c][a][b] = x[a]
        for a in axes:
            if front:
                fm = front_mask([ys[c][b] for c in front], [xs[c][a][b] for c in front])
                for c, f in zip(front, fm):
                    onf[c][a][b] = f
    obs = {c: _stat(*arr[c], None, None) for c in members}
    pt = points.set_index("row_id")
    rows = []
    for a in axes:
        fm_pt = dict(zip(front, front_mask([obs[c][0] for c in front], [obs[c][1][a] for c in front]))) if front else {}
        for c in members:
            y0, x0 = obs[c][0], obs[c][1][a]
            defined = np.isfinite(ys[c]) & np.isfinite(xs[c][a])
            r: dict[str, Any] = {"panel": panel, "axis": a, "primary_axis": a == PRIMARY.get(panel),
                                 "config": c, "role": "front" if c in front else "reference",
                                 "n_runs": int(u[u.config == c].row_id.nunique()), "n_episodes": int((u.config == c).sum()),
                                 "tasks": int(u[u.config == c].dataset_id.nunique()),
                                 "rec_sel": y0, "rec_lo95": _ci(ys[c], 2.5, 97.5)[0], "rec_hi95": _ci(ys[c], 2.5, 97.5)[1],
                                 "cost": x0, "cost_lo95": _ci(xs[c][a], 2.5, 97.5)[0],
                                 "cost_hi95": _ci(xs[c][a], 2.5, 97.5)[1]}
            if c in front:
                r["on_front_point"] = bool(fm_pt.get(c, False))
                r["p_front"] = float(onf[c][a][defined].mean()) if defined.any() else np.nan
                r["n_boot_defined"] = int(defined.sum())
                r["wording"] = wording(r["p_front"])
            else:
                r.update(on_front_point=np.nan, p_front=np.nan, n_boot_defined=int(defined.sum()),
                         wording="reference (never in a front)")
            if ref in members and c != ref:
                d = ys[c] - ys[ref]
                ratio = xs[c][a] / xs[ref][a]
                r.update(diff_vs_full=y0 - obs[ref][0], diff_lo90=_ci(d, 5, 95)[0], diff_hi90=_ci(d, 5, 95)[1],
                         diff_lo95=_ci(d, 2.5, 97.5)[0], diff_hi95=_ci(d, 2.5, 97.5)[1],
                         ratio_vs_full=x0 / obs[ref][1][a] if obs[ref][1][a] else np.nan,
                         ratio_lo95=_ci(ratio, 2.5, 97.5)[0], ratio_hi95=_ci(ratio, 2.5, 97.5)[1])
                r["equal_band"] = bool(np.isfinite(r["diff_lo90"]) and r["diff_lo90"] >= -delta and r["diff_hi90"] <= delta)
                r["cheaper_at_equal"] = bool(r["equal_band"] and r["ratio_vs_full"] <= 0.70 and r["ratio_hi95"] < 1.0)
                r["diff95_excludes_0"] = bool(np.isfinite(r["diff_lo95"]) and (r["diff_lo95"] > 0 or r["diff_hi95"] < 0))
            pc = pt[pt.index.isin(u[u.config == c].row_id.unique())]
            n_rec = int(u[u.config == c].rec_sel.sum())
            tot = float(pc[a].sum()) if a in pc else np.nan
            r["cost_per_recovered"] = tot / n_rec if n_rec else "no recovery"
            itt = float(pc.itt_imputed.mean()) if len(pc) else np.nan
            fb = float(pd.to_numeric(pc.get("fallback_share"), errors="coerce").mean()) if "fallback_share" in pc else np.nan
            r.update(itt_share=itt, mean_fallback_share=fb,
                     caveat=bool((itt > 0.20) if np.isfinite(itt) else False) or bool(fb > 0.10 if np.isfinite(fb) else False))
            if r["caveat"] and r.get("wording") == "on the front":
                r["wording"] = "on the front (caveat R-A23-4)"
            rows.append(r)
    return pd.DataFrame(rows)


def classify(front: pd.DataFrame, panel: str, token_invalid: bool = False) -> dict[str, Any]:
    """A23 section 3, first matching row in table order (panels A and B)."""
    f = front[(front.panel == panel) & front.primary_axis]
    if f.empty:
        return {"panel": panel, "outcome": "no data"}
    if token_invalid:
        return {"panel": panel, "outcome": "Measurement failure", "basis": "--token-invalid"}

    def meets(c):
        r = f[f.config == c]
        return bool(len(r) and r.cheaper_at_equal.fillna(False).iloc[0])

    def word(c):
        r = f[f.config == c]
        return str(r.wording.iloc[0]) if len(r) else "absent"

    if panel == "A":
        opp = [c for c in ("llmfe", "fx_pajM", "set_phased") if meets(c)]
        cf = front[(front.panel == "caafe") & front.primary_axis & (front.config == "caafe")]
        if len(cf) and bool(cf.cheaper_at_equal.fillna(False).iloc[0]):
            opp.append("caafe")
        if word("full") == "dominated" or opp:
            return {"panel": "A", "outcome": "Opposite", "meets_R-A23-3": opp, "full": word("full")}
        if word("full").startswith("on the front"):
            return {"panel": "A", "outcome": "Hoped (A)", "full": word("full")}
        return {"panel": "A", "outcome": "Null (A)", "full": word("full")}
    if meets("mix_1b") and word("mix_1b").startswith("on the front"):
        return {"panel": "B", "outcome": "Hoped (B)"}
    f2 = [c for c in f.config if c.startswith("f2_") and meets(c) and word(c).startswith("on the front")]
    if f2:
        return {"panel": "B", "outcome": "Hoped variant (B)", "configs": f2}
    any_meet = [c for c in f.config if meets(c)]
    if not any_meet and not f.diff95_excludes_0.fillna(False).any():
        return {"panel": "B", "outcome": "Null B"}
    return {"panel": "B", "outcome": "Null A", "meets_R-A23-3": any_meet}


def run_front(points: pd.DataFrame, units: pd.DataFrame, *, B: int, delta: float, variant: str = "primary",
              token_invalid: bool = False) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    u = select(units, points, variant)
    frames = [panel_front(u, points, p, B=B, delta=delta) for p in ("A", "B")]
    if variant == "primary":
        frames.append(panel_front(u, points, "AB", B=B, delta=delta, axes=("flop_proxy",)))  # sensitivity (h)
        cu = select(units, points, "b_allseeds")
        cu = cu[cu.config.isin(["caafe"]) | ((cu.config == "full") & cu.row_id.isin(
            points[(points.config == "full") & (((points.fold_seed == 42) & points.replicate.isin([0, 1]))
                                                 | ((points.fold_seed == 4) & (points.replicate == 0)))].row_id))]
        if (cu.config == "caafe").any():
            frames.append(panel_front(cu, points, "caafe", B=B, delta=delta, axes=("tok_compl_total",)))
    fr = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else pd.DataFrame()
    fr.insert(0, "variant", variant)
    outcomes = [dict(classify(fr, p, token_invalid), variant=variant) for p in ("A", "B")] if len(fr) else []
    return fr, outcomes


def sensitivity_changes(primary: pd.DataFrame, other: pd.DataFrame, variant: str) -> pd.DataFrame:
    key = ["panel", "axis", "config"]
    a = primary[primary.primary_axis][key + ["wording", "cheaper_at_equal"]]
    b = other[other.primary_axis][key + ["wording", "cheaper_at_equal"]]
    m = a[a.panel.isin(set(b.panel))].merge(b, on=key, how="outer", suffixes=("_primary", "_sens"))
    w = m[["wording_primary", "wording_sens"]].fillna("absent")
    c = m[["cheaper_at_equal_primary", "cheaper_at_equal_sens"]].astype(object).fillna("n/a").astype(str)
    ch = m[(w.wording_primary != w.wording_sens) | (c.cheaper_at_equal_primary != c.cheaper_at_equal_sens)]
    return ch.assign(sensitivity=variant)


def outcome_changes(outcomes: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Section 3 panel outcomes of every variant that differ from the primary analysis."""
    prim = {o["panel"]: o["outcome"] for o in outcomes if o.get("variant") == "primary"}
    rows = [{"panel": o["panel"], "axis": "section 3 outcome", "config": "", "wording_primary": prim.get(o["panel"]),
             "wording_sens": o["outcome"], "sensitivity": o["variant"]}
            for o in outcomes if o.get("variant") != "primary" and o["outcome"] != prim.get(o["panel"])]
    return pd.DataFrame(rows)


def axis_changes(front: pd.DataFrame, base: str, alt: str, label: str) -> pd.DataFrame:
    key = ["panel", "config"]
    a = front[front.axis == base][key + ["wording"]]
    b = front[front.axis == alt][key + ["wording"]]
    m = a.merge(b, on=key, suffixes=("_primary", "_sens"))
    m = m[m.wording_sens.notna() & (m.wording_primary != m.wording_sens)]
    return m.assign(sensitivity=label, axis=f"{base}->{alt}")


def plot(front: pd.DataFrame, out: Path) -> list[str]:  # pragma: no cover - figure only
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    for (panel, axis), f in front[front.variant == "primary"].groupby(["panel", "axis"]):
        f = f[np.isfinite(pd.to_numeric(f.cost, errors="coerce"))]
        if f.empty or panel == "AB":
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for _, r in f[f.config.isin(["openfe", "autofeat"])].iterrows():  # 0 LLM tokens: not placeable on log-x
            ax.axhline(r.rec_sel, ls="--", lw=0.8, color="0.4")
            ax.annotate(f"{r.config} (brute force, 0 LLM tokens)", (0.01, r.rec_sel), xycoords=("axes fraction", "data"),
                        fontsize=7, va="bottom")
        for _, r in f[~f.config.isin(["openfe", "autofeat"]) & (f.cost > 0)].iterrows():
            hollow = r.role == "reference"
            ax.errorbar(r.cost, r.rec_sel, xerr=[[max(r.cost - r.cost_lo95, 0)], [max(r.cost_hi95 - r.cost, 0)]],
                        yerr=[[max(r.rec_sel - r.rec_lo95, 0)], [max(r.rec_hi95 - r.rec_sel, 0)]],
                        fmt="o", mfc="none" if hollow else None, color="0.5" if hollow else "C0", capsize=2, lw=0.8)
            ax.annotate(r.config, (r.cost, r.rec_sel), fontsize=7, xytext=(3, 3), textcoords="offset points")
        fp = f[f.on_front_point == True].sort_values("cost")  # noqa: E712
        ax.step(fp.cost, fp.rec_sel, where="post", color="C3", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel(f"{axis} per SYN-P run (log)")
        ax.set_ylabel("rec_sel")
        ax.set_title(f"A23 panel {panel} ({axis})")
        p = out / f"pareto_{panel}_{axis}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        files.append(str(p))
    return files


def external_units(llmfe_episodes: Optional[str], llmfe_totals: Sequence[str], screen_episodes: Optional[str],
                   llmfe_selection: str = "best3") -> pd.DataFrame:
    """External-arm units for ``--external``: LLM-FE (X6 detector episodes of arm ``llmfe-r<rep>-<selection>``;
    best-3 is primary, decision of 2026-10-26; X5 proxy totals = run cost on O120, no per-fold cost -> run/5) and the
    A08 CPU screen ``openfe``/``autofeat`` at seed 42 (0 LLM tokens; reference lines only)."""
    rows = []
    tot: dict[str, Any] = {}
    for f in llmfe_totals:
        tot.update(json.loads(Path(f).read_text(encoding="utf-8")))
    if llmfe_episodes:
        e = pd.read_csv(llmfe_episodes)
        e = e[e.arm.str.fullmatch(rf"llmfe-r\d+-{llmfe_selection}")]
        for _, r in e.iterrows():
            rep = int(re.search(r"-r(\d+)-", r.arm).group(1))
            t = tot.get(f"a08-llmfe-{r.dataset_id}-r{rep}")
            c = {"tok_compl_total": np.nan, "flop_proxy": np.nan, "flop_proxy_compl": np.nan}
            if t:
                n = ACTIVE_PARAMS["O120"]
                c = {"tok_compl_total": t["completion_tokens"],
                     "flop_proxy": 2 * n * (t["prompt_tokens"] + t["completion_tokens"]),
                     "flop_proxy_compl": 2 * n * t["completion_tokens"]}
            rows.append({"config": "llmfe", "dataset_id": r.dataset_id, "fold_seed": int(r.fold_seed),
                         "replicate": rep, "fold": int(r.fold), "rec_sel": int(r.rec_sel), **c})
    if screen_episodes:
        s = pd.read_csv(screen_episodes)
        s = s[s.arm.isin(["openfe", "autofeat"]) & (s.fold_seed == 42)]
        for _, r in s.iterrows():
            rows.append({"config": r.arm, "dataset_id": r.dataset_id, "fold_seed": 42, "replicate": 0,
                         "fold": int(r.fold), "rec_sel": int(r.rec_sel), "tok_compl_total": 0.0, "flop_proxy": 0.0,
                         "flop_proxy_compl": 0.0})
    return pd.DataFrame(rows)


def energy_csv(jsons: Sequence[str], jsons_equal: Sequence[str], out: str) -> pd.DataFrame:
    """Merge ``energy_apportion.py`` result JSONs (run names = row ids) into the ``--energy`` CSV."""
    rows: dict[str, dict[str, Any]] = {}
    for f in jsons:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        for k, v in d.get("run_energy_j", {}).items():
            rows.setdefault(k, {"row_id": k})["energy_j"] = v
            rows[k]["energy_coverage"] = d.get("coverage", {}).get(k)
    for f in jsons_equal:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        for k, v in d.get("run_energy_j", {}).items():
            rows.setdefault(k, {"row_id": k})["energy_j_equal"] = v
    df = pd.DataFrame(list(rows.values()))
    df.to_csv(out, index=False)
    return df


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("build-points", "external", "energy", "front", "plot", "all"))
    ap.add_argument("--episodes", nargs="*", default=[], help="fold-episode CSVs (row_id, fold, rec_sel, status, itt_imputed)")
    ap.add_argument("--manifest", nargs="*", default=[])
    ap.add_argument("--runs", nargs="*", default=[], help="session folders holding <row>/tasks/*/*/output")
    ap.add_argument("--external", nargs="*", default=[], help="external-arm units CSV (llmfe, caafe, openfe, autofeat)")
    ap.add_argument("--llmfe-episodes", default=None, help="external: X6 comparator detector episodes")
    ap.add_argument("--llmfe-totals", nargs="*", default=[], help="external: X5 proxy totals.json files")
    ap.add_argument("--screen-episodes", default=None, help="external: A08 CPU screen detector episodes")
    ap.add_argument("--llmfe-selection", default="best3")
    ap.add_argument("--energy", default=None, help="CSV row_id, energy_j, energy_coverage[, energy_j_equal]")
    ap.add_argument("--energy-json", nargs="*", default=[])
    ap.add_argument("--energy-json-equal", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=B_DEFAULT)
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--token-invalid", action="store_true", help="R-A23-5 token validation failed (A23-T1/T13)")
    ap.add_argument("--variants", nargs="*",
                    default=["a_all7", "b_allseeds", "power", "infguard", "a20_relaunch"])
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "external":
        external_units(a.llmfe_episodes, a.llmfe_totals, a.screen_episodes, a.llmfe_selection).to_csv(
            out / "external_units.csv", index=False)
        return 0
    if a.cmd == "energy":
        energy_csv(a.energy_json, a.energy_json_equal, str(out / "energy.csv"))
        return 0
    if a.cmd in ("build-points", "all"):
        pts, units = build_points(a.episodes, a.manifest, a.runs, a.external, a.energy)
        pts.to_csv(out / "pareto_points.csv", index=False)
        units.to_csv(out / "pareto_units.csv", index=False)
        print(pts.groupby(["config", "tag"]).size().to_string(), file=sys.stderr)
    if a.cmd in ("front", "all"):
        pts = pd.read_csv(out / "pareto_points.csv")
        units = pd.read_csv(out / "pareto_units.csv")
        fr, outcomes = run_front(pts, units, B=a.B, delta=a.delta, token_invalid=a.token_invalid)
        sens, changes = [fr], []
        for v in a.variants:
            fv, ov = run_front(pts, units, B=a.B, delta=a.delta, variant=v, token_invalid=a.token_invalid)
            if len(fv):
                sens.append(fv)
                changes.append(sensitivity_changes(fr, fv, v))
            outcomes += ov
        changes.append(axis_changes(fr[fr.panel == "B"], "flop_proxy", "flop_proxy_compl", "d_flop_proxy_compl"))
        changes.append(axis_changes(fr, "energy_j", "energy_j_equal", "e_equal_split_energy"))
        changes.append(outcome_changes(outcomes))
        allf = pd.concat(sens, ignore_index=True)
        allf.to_csv(out / "pareto_front.csv", index=False)
        ch = pd.concat([c for c in changes if len(c)], ignore_index=True) if any(len(c) for c in changes) else pd.DataFrame()
        ch.to_csv(out / "pareto_sensitivity_changes.csv", index=False)
        summary = {"seed": SEED, "B": a.B, "delta": a.delta, "population": list(POP_TASKS),
                   "outcomes": outcomes, "not_evaluated": NOT_EVALUATED,
                   "missing_configs": sorted(c for c, v in CONFIGS.items() if not (pts.config == c).any()),
                   "token_validation": "not checked by pareto.py (A23-T1/T13); pass --token-invalid if it failed"}
        (out / "pareto_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        cols = ["panel", "config", "n_runs", "rec_sel", "cost", "p_front", "wording", "ratio_vs_full", "cheaper_at_equal"]
        pd.set_option("display.width", 250)
        print(fr[fr.primary_axis][[c for c in cols if c in fr]].to_string(index=False))
        print(json.dumps(outcomes, indent=1, default=str))
        if a.plot:
            plot(allf, out)
    if a.cmd == "plot":
        plot(pd.read_csv(out / "pareto_front.csv"), out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
