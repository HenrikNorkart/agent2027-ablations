"""E11a: G2 statistics for AGENT2027 (Mon 19 Oct 2026, hard go/no-go).

Binding rules: ``docs/agent2027/preregistration.md`` Part IV, Part V.3 (A03 fallback),
V.5 (A05 formulas), V.6 (A06 sentinel rule and band), V.7 (A07), V.9 (A09 headroom),
V.13 (A13), V.15 (A15 leak rule), V.19 (A19 tau) and Part VI.2 (G2). Formulas restated
from ``docs/ablations/A05_M0_full_v2_reference.md`` section 2 and 6, A06 section 4.7,
A19 section 2/6, A12-T3, A21 section 6 (G2 validity) and 01 section 4.7.

Two stages, both run inside a container on CAIR (``rogue_one:agent2027-v2.0``; never on
the host):

* ``collect``: reads the session run roots (read-only), the manifests, the A08 /
  A05-T6 ``base_no_fe`` JSONs, the off-mount phi metadata and W&B (configs, summaries,
  ``run-output`` artifact presence); runs the recovery detector on A05's SYN-P rows;
  scans logs for the NO-GO triggers; runs the A15 P6b leak checks. Writes
  ``g2_rows.csv``, ``g2_episodes.csv``, ``g2_iterations.csv``, ``g2_attempts.csv``,
  ``g2_leak.json`` to ``--out``.
* ``analyze``: pure computation of every G2 constant from those files; writes
  ``g2_constants.json`` and small CSV tables.

The functions in the first section are pure and unit-tested
(``tests/unit/agent2027/test_g2_stats.py``). Where a pre-registered definition is
missing, the code says so (``NOT_DEFINED``) instead of inventing one; the one
computation that needs an assumption (power) is labelled ``provisional``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import gzip
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- constants
CORE10 = {  # dataset -> task type, in A05 section 2 order
    "balance-scale": "classification", "junglechess": "classification", "vehicle": "classification",
    "heart": "classification", "credit-g": "classification", "myocardial": "classification",
    "housing": "regression", "insurance": "regression", "crab": "regression", "wine": "regression",
}
SYNP8 = ["synp_p1", "synp_p2", "synp_p3", "synp_p4", "synp_p5", "synp_p6a", "synp_p6b", "synp_p7"]
SYNP7 = SYNP8[:-1]
SYNP_CLASSIFICATION = {"synp_p3", "synp_p4", "synp_p5", "synp_p7"}
GATED_OUT = {"synp_p1"}  # Part III.2: non-discriminating (M5S), excluded from confirmatory populations
POOLABLE_R = ("housing", "insurance", "crab")
# A05 section 2 (paper/results_seed_sweep_t17.csv, sd_across_seeds), plain accuracy for C.
T17_SD = {"balance-scale": 0.000, "junglechess": 0.070, "vehicle": 0.029, "heart": 0.019, "credit-g": 0.0055,
          "myocardial": 0.0076, "housing": 0.021, "insurance": 0.063, "crab": 0.024, "wine": 0.0010}
# Part V.3 table: A01 sigma_samp,d (no W0 pairs for balance-scale and wine).
A01_SIGMA_SAMP = {"junglechess": 0.0245, "vehicle": 0.0037, "heart": 0.0082, "credit-g": 0.0141,
                  "myocardial": 0.0170, "housing": 0.0151, "insurance": 0.0501, "crab": 0.0153}
A01_DRIFT = False            # Part III.1 rule 2: no drift
M_MARGIN = 1.0               # Part IV.2 / VIII.1 item 8
FLOOR_MULT = 0.1             # sigma_d floor = 0.1 x s_d
CAP_S = 6 * 3600.0
N_FOLDS = 5
NI_SUCCESS_MULT = 2.0
JUNGLECHESS_RULE = 0.95
A03_MEAN_MIN = -0.75         # A03 (a)
A03_PAIR_MIN = -2.5          # A03 (b): delta / (sqrt(2) sigma) < -2.5 fails
A03_MIN_PAIRS = 8
A06_FLOORS = {"log": 0.1, "log1p": 0.1, "rate": 0.05, "scaled": 0.1}
A06_K = 3.0 * math.sqrt(1.5)
A12_STAGE_MIN_SHARE = 0.30
A19_TAU_BASE, A19_TAU_SE_CUT, A19_TAU_ADD = 0.60, 0.50, 0.10
A09_HEADROOM_RATE, A09_HEADROOM_MIN_TASKS = 0.4, 3
A21_TOK_TOL, A21_TOK_SHARE = 0.01, 0.95
NOGO_CAP_HITS, NOGO_ITT, NOGO_SAME_EXC = 3, 6, 2
WORDING_INFORMATIVE_MIN, WORDING_ICC_MAX, WORDING_Z_MAX = 4, 0.5, 0.85
INNER_CV_FALLBACK_TEXT = "falling back to the held-out metric"
NOT_DEFINED = "NOT_DEFINED"
# 01 section 4.7 planning MDEs (unclustered, ~6 informative tasks).
PLANNING_MDE = {
    "C1a": (0.265, "about 0.23-0.30"), "C2a/C2b/C2c": (0.11, "90% CI about +-0.10-0.12"),
    "RT3 (A09, outside Holm)": (0.26, "about 0.26"), "C3": (0.425, "about 40-45 pp at 0.05"),
    "C4a": (0.35, "about 30-40 pp at 6 seeds"), "C4b": (0.60, "about 60 pp, 15 pairs"),
}


# =========================================================================== pure math
def s_d(s42: float, s4: float) -> float:
    """A05: s_d = mean(s_{d,42}, s_{d,4})."""
    return 0.5 * (float(s42) + float(s4))


def signed_delta(arm: float, full: float, task_type: str) -> float:
    """Delta = arm - Full, sign-flipped for nRMSE: Delta > 0 always means the arm is better."""
    d = float(arm) - float(full)
    return -d if task_type == "regression" else d


def sigma_aa2(a: Sequence[float], b: Sequence[float], s: Sequence[float]) -> float:
    """sigma^2_AA = mean_d[((a - b)/s_d)^2] / 2."""
    a, b, s = (np.asarray(x, dtype=float) for x in (a, b, s))
    return float(np.mean(((a - b) / s) ** 2) / 2.0)


def sigma_p2(a, b, c, s, saa2: float) -> float:
    """sigma^2_p = max(0, (mean_d[(((a + b)/2 - c)/s_d)^2] - 1.5 sigma^2_AA) / 2)."""
    a, b, c, s = (np.asarray(x, dtype=float) for x in (a, b, c, s))
    m = float(np.mean((((a + b) / 2.0 - c) / s) ** 2))
    return max(0.0, (m - 1.5 * float(saa2)) / 2.0)


def z_share(saa2: float, sp2: float) -> float:
    """Z = sigma^2_AA / (sigma^2_AA + sigma^2_p)."""
    den = float(saa2) + float(sp2)
    return float("nan") if den <= 0 else float(saa2) / den


def sigma_d_raw(a: float, b: float, c: float) -> float:
    """SD of {a, b, c} in raw units, 2 df (ddof=1), before the floor."""
    return float(np.std([a, b, c], ddof=1))


def floor_sigma(sigma: float, s: float, mult: float = FLOOR_MULT) -> float:
    return max(float(sigma), mult * float(s))


def f_bounds(df1: int, df2: int) -> tuple[float, float]:
    from scipy import stats  # noqa: PLC0415
    return float(stats.f.ppf(0.025, df1, df2)), float(stats.f.ppf(0.975, df1, df2))


def homogeneity(aa_pairs: Mapping[str, tuple[float, float]], samp_deltas: Mapping[str, Sequence[float]],
                s: Mapping[str, float], drift: bool = A01_DRIFT) -> dict[str, Any]:
    """A05 homogeneity rule on housing/insurance/crab.

    ``aa_pairs``: dataset -> (a, b) raw held-out nRMSE; ``samp_deltas``: dataset -> list of
    A01 W0-t17 raw deltas; ``s``: dataset -> A05 s_d. F = sigma^2_AA,R / sigma^2_samp,R
    against the central 95% of F(3, k), k = number of usable pairs (no pooling if k = 0
    or if A01 reports drift)."""
    ds = [d for d in POOLABLE_R if d in aa_pairs and d in s]
    aa = [((aa_pairs[d][0] - aa_pairs[d][1]) / s[d]) ** 2 / 2.0 for d in ds]
    sm = [(float(x) / s[d]) ** 2 / 2.0 for d in ds for x in samp_deltas.get(d, []) if x is not None
          and not (isinstance(x, float) and math.isnan(x))]
    k = len(sm)
    out: dict[str, Any] = {"datasets": ds, "df1": len(aa), "df2": k, "drift": bool(drift)}
    if not aa or k == 0:
        out.update({"F": None, "bounds": None, "pool": False,
                    "reason": "no usable W0 pairs" if k == 0 else "no A05 A/A pairs"})
        return out
    saa = float(np.mean(aa))
    ssm = float(np.mean(sm))
    F = saa / ssm if ssm > 0 else float("inf")
    lo, hi = f_bounds(len(aa), k)
    inside = bool(lo <= F <= hi)
    out.update({"sigma2_AA_R": saa, "sigma2_samp_R": ssm, "F": F, "bounds": [lo, hi], "inside": inside,
                "pool": bool(inside and not drift),
                "reason": ("F inside central 95%" if inside else ("F above upper bound (v2 noisier)" if F > hi
                                                                    else "F below lower bound"))
                + ("; A01 drift" if drift else "")})
    return out


def pooled_sigma(sigma_raw: float, t17_sd: float, s: float) -> float:
    """If pooled: sigma^2_d = (sigma^2_d + SD^2_t17,d) / 2, then the same floor."""
    return floor_sigma(math.sqrt((sigma_raw ** 2 + t17_sd ** 2) / 2.0), s)


def a03_sigma_hat(dataset: str, base_sd_fold: float) -> dict[str, Any]:
    """Part V.3: sigma_hat_d = max(A01 sigma_samp,d, t17 SD_d, 0.1 x base_no_fe fold SD_d)."""
    terms = {"a01": A01_SIGMA_SAMP.get(dataset), "t17": T17_SD.get(dataset),
             "floor": FLOOR_MULT * float(base_sd_fold)}
    vals = [v for v in terms.values() if v is not None]
    decided = max(terms, key=lambda k: -1 if terms[k] is None else terms[k])
    return {"sigma_hat": float(max(vals)), "terms": terms, "decided_by": decided}


def a03_readout(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Part V.3 rule. ``pairs``: dataset, task_type, aug (A05 full f42 r0 score or None),
    rep (A03 replace score or None), sigma_hat, aug_failed, rep_failed.
    (a) mean delta_std >= -0.75 over complete pairs (>= 8 of 10);
    (b) no complete pair with Delta/(sqrt 2 sigma) < -2.5, and no dataset with augment
    failing/capping where replace completed. < 8 complete pairs: descriptive, augment stands."""
    rows, complete = [], []
    for p in pairs:
        ok = p.get("aug") is not None and p.get("rep") is not None and not p.get("aug_failed") \
            and not p.get("rep_failed")
        r = dict(p)
        if ok:
            # Delta_d = score_aug - score_rep (C); nRMSE_rep - nRMSE_aug (R); > 0 = augment better
            d = signed_delta(p["aug"], p["rep"], p["task_type"])
            r.update({"delta": d, "delta_std": d / p["sigma_hat"],
                      "pair_z": d / (math.sqrt(2.0) * p["sigma_hat"])})
            complete.append(r)
        rows.append(r)
    n = len(complete)
    aug_fail_where_rep_ok = [p["dataset"] for p in pairs if p.get("aug_failed") and not p.get("rep_failed")]
    mean_std = float(np.mean([r["delta_std"] for r in complete])) if complete else float("nan")
    worst = min((r["pair_z"] for r in complete), default=float("nan"))
    a = bool(complete and mean_std >= A03_MEAN_MIN)
    b = bool(all(r["pair_z"] >= A03_PAIR_MIN for r in complete) and not aug_fail_where_rep_ok)
    evaluable = n >= A03_MIN_PAIRS
    outcome = ("augment stands" if (a and b) else "FAILS: Henrik chooses revert-or-limitation (D6)") \
        if evaluable else "not evaluable (< 8 complete pairs): read descriptively, augment stands"
    return {"n_complete": n, "evaluable": evaluable, "mean_delta_std": mean_std, "worst_pair_z": worst,
            "crit_a": a, "crit_b": b, "aug_fail_where_rep_ok": aug_fail_where_rep_ok,
            "crit_c": "untested (SYN-P add-on could only run before the freeze)", "outcome": outcome,
            "pairs": rows}


def informative_tasks(rec_by_task: Mapping[str, int], n_episodes: int = 15) -> list[str]:
    """Full's rec_sel strictly between 0/n and n/n."""
    return [t for t, k in rec_by_task.items() if 0 < int(k) < n_episodes]


def sentinel_task(seed42_rec10: Mapping[str, int], rec15: Mapping[str, int],
                  candidates: Sequence[str] = SYNP7) -> dict[str, Any]:
    """A06 rule: closest seed-42 r0+r1 rate to 5/10; ties by the 15-episode rate closest to
    0.5, then the lower task index (candidate order)."""
    table = []
    for i, t in enumerate(candidates):
        k10, k15 = int(seed42_rec10.get(t, 0)), int(rec15.get(t, 0))
        table.append({"task": t, "rec10": k10, "rec15": k15, "dist10": abs(k10 / 10.0 - 0.5),
                      "dist15": abs(k15 / 15.0 - 0.5), "index": i})
    table.sort(key=lambda r: (round(r["dist10"], 12), round(r["dist15"], 12), r["index"]))
    for rank, r in enumerate(table, start=1):
        r["rank"] = rank
    chosen = table[0]
    informative = [r["task"] for r in table if 0 < r["rec15"] < 15]
    return {"task": chosen["task"], "Z_of_10": chosen["rec10"], "table": table,
            "any_informative_candidate": bool(informative)}


def a06_sigma_aa_m(r0: Sequence[float], r1: Sequence[float]) -> float:
    """sigma_AA,m = sqrt(sum d_i^2 / 2n), d_i = r0 - r1 on the metric's scale."""
    d = np.asarray(r0, dtype=float) - np.asarray(r1, dtype=float)
    d = d[~np.isnan(d)]
    return float(math.sqrt(np.sum(d ** 2) / (2.0 * len(d)))) if len(d) else float("nan")


def a06_band(centre: float, sigma_aa_m: float, floor: float) -> dict[str, float]:
    hw = max(A06_K * float(sigma_aa_m), float(floor)) if not math.isnan(sigma_aa_m) else float(floor)
    return {"centre": float(centre), "half_width": hw, "lo": float(centre) - hw, "hi": float(centre) + hw}


def a06_transform(x: float, scale: str) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return float("nan")
    if scale == "log":
        return math.log(float(x)) if float(x) > 0 else float("nan")
    if scale == "log1p":
        return math.log1p(float(x))
    return float(x)


def a19_tau(s_e_median: float) -> float:
    """tau = 0.60 if median s_E <= 0.50, else s_E + 0.10."""
    return A19_TAU_BASE if float(s_e_median) <= A19_TAU_SE_CUT else round(float(s_e_median) + A19_TAU_ADD, 4)


def a12_branch(share_assess: float, share_prune: float, separable: bool) -> dict[str, Any]:
    """Part IV.2 token co-primary: default >= 30% fewer tok_compl_total; if a Tester stage is
    < 30% of tok_compl_total, (i) the replaced stage's own tokens, or (ii) the joint A-J- vs
    A+J+ contrast if the stages are not separable."""
    small = {k: v for k, v in (("assessment", share_assess), ("pruning", share_prune)) if v < A12_STAGE_MIN_SHARE}
    if not small:
        branch = "default: >= 30% fewer tok_compl_total"
    elif separable:
        branch = "(i) replaced stage's own tokens"
    else:
        branch = "(ii) joint A-J- vs A+J+ contrast"
    return {"stages_below_30pct": sorted(small), "separable": bool(separable), "branch": branch}


def a09_headroom(rec_rate_by_task: Mapping[str, float], population: Sequence[str]) -> dict[str, Any]:
    """Headroom flag: fewer than 3 gate-passing tasks with Full rec_sel >= 0.4 -> noted."""
    ok = [t for t in population if rec_rate_by_task.get(t, 0.0) >= A09_HEADROOM_RATE]
    return {"tasks_with_rate_ge_0.4": ok, "n": len(ok), "flag": len(ok) < A09_HEADROOM_MIN_TASKS}


def nogo_verdict(n_inner_cv_fallback: int, n_zero_role: int, n_missing_artifacts: int, n_cap: int,
                 n_itt: int, max_same_exception_runs: int, fixable_cause: Optional[bool]) -> dict[str, Any]:
    """Part VI.2. Unconditional triggers -> NO-GO. Conditional triggers -> NO-GO only with a
    fixable defect identified by 10:00 (``fixable_cause`` True); None = not judged."""
    uncond = {"inner_cv_fallback": n_inner_cv_fallback > 0, "zero_tok_compl_role": n_zero_role > 0,
              "missing_traces_scores_or_run_output": n_missing_artifacts > 0}
    cond = {"cap_hits_ge_3": n_cap >= NOGO_CAP_HITS, "itt_ge_6": n_itt >= NOGO_ITT,
            "same_exception_ge_2_runs": max_same_exception_runs >= NOGO_SAME_EXC}
    if any(uncond.values()):
        verdict = "NO-GO (unconditional)"
    elif any(cond.values()):
        verdict = {True: "NO-GO (conditional trigger + fixable defect)",
                   False: "GO (conditional trigger is a finding: no fixable defect)",
                   None: "conditional trigger fired: verdict depends on the fixable-defect judgement"}[fixable_cause]
    else:
        verdict = "GO"
    return {"unconditional": uncond, "conditional": cond, "verdict": verdict}


# ---- recovery ICC
def _laplace_loglik(theta: np.ndarray, y: np.ndarray, Z: np.ndarray, groups: Sequence[slice]) -> tuple[float, np.ndarray]:
    mu, log_sd = theta[0], theta[1:]
    sd = np.concatenate([np.full(g.stop - g.start, math.exp(ls)) for g, ls in zip(groups, log_sd)])
    ZL = Z * sd  # b = sd * v, v ~ N(0, I)
    v = np.zeros(Z.shape[1])
    for _ in range(100):
        eta = mu + ZL @ v
        p = 1.0 / (1.0 + np.exp(-eta))
        w = p * (1 - p)
        g = ZL.T @ (y - p) - v
        H = ZL.T @ (ZL * w[:, None]) + np.eye(len(v))
        step = np.linalg.solve(H, g)
        v = v + step
        if np.max(np.abs(step)) < 1e-10:
            break
    eta = mu + ZL @ v
    ll = float(np.sum(y * eta - np.logaddexp(0, eta)))
    p = 1.0 / (1.0 + np.exp(-eta))
    w = p * (1 - p)
    H = ZL.T @ (ZL * w[:, None]) + np.eye(len(v))
    sign, logdet = np.linalg.slogdet(H)
    return ll - 0.5 * float(v @ v) - 0.5 * logdet, v


def glmm_icc(episodes: pd.DataFrame, task_col: str = "task", seed_col: str = "seed", run_col: str = "run",
             y_col: str = "rec_sel", sd_floor: float = 1e-2) -> dict[str, Any]:
    """Logistic GLMM rec ~ 1 + (1|task) + (1|task:seed) + (1|run) by Laplace ML;
    ICC = s2_task / (s2_task + s2_task:seed + s2_run + pi^2/3). Singular if any SD < sd_floor."""
    from scipy import optimize  # noqa: PLC0415
    df = episodes.copy()
    df["_ts"] = df[task_col].astype(str) + ":" + df[seed_col].astype(str)
    levels = [sorted(df[task_col].astype(str).unique()), sorted(df["_ts"].unique()),
              sorted(df[run_col].astype(str).unique())]
    cols = [df[task_col].astype(str), df["_ts"], df[run_col].astype(str)]
    blocks, groups, start = [], [], 0
    for lev, col in zip(levels, cols):
        idx = {l: i for i, l in enumerate(lev)}
        B = np.zeros((len(df), len(lev)))
        B[np.arange(len(df)), col.map(idx).to_numpy()] = 1.0
        blocks.append(B)
        groups.append(slice(start, start + len(lev)))
        start += len(lev)
    Z = np.hstack(blocks)
    y = df[y_col].astype(float).to_numpy()
    ybar = min(max(y.mean(), 1e-3), 1 - 1e-3)
    best = None
    for init in ([math.log(ybar / (1 - ybar)), 0.5, -0.5, -0.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, -1.0]):
        res = optimize.minimize(lambda th: -_laplace_loglik(np.clip(th, [-30, -6, -6, -6], [30, 3.5, 3.5, 3.5]),
                                                            y, Z, groups)[0],
                                np.array(init, dtype=float), method="Nelder-Mead",
                                options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
        if best is None or res.fun < best.fun:
            best = res
    th = np.clip(best.x, [-30, -6, -6, -6], [30, 3.5, 3.5, 3.5])
    sds = np.exp(th[1:])
    var = sds ** 2
    icc = float(var[0] / (var.sum() + math.pi ** 2 / 3.0))
    singular = bool(np.any(sds < sd_floor))
    at_upper = bool(np.any(th[1:] >= 3.5 - 1e-6))
    return {"mu": float(th[0]), "sd_task": float(sds[0]), "sd_task_seed": float(sds[1]), "sd_run": float(sds[2]),
            "icc": icc, "singular": singular, "at_upper_bound": at_upper, "loglik": float(-best.fun),
            "n_obs": int(len(y)), "converged": bool(best.success)}


def anova_icc(run_props: pd.DataFrame, task_col: str = "task", p_col: str = "p") -> dict[str, Any]:
    """One-way random-effects ICC(1) on run-level proportions grouped by task (fallback when
    the GLMM is singular). Unbalanced groups use the usual n0 correction."""
    g = run_props.groupby(task_col)[p_col]
    k = g.count()
    N, a = int(k.sum()), int(len(k))
    grand = run_props[p_col].mean()
    ssb = float((k * (g.mean() - grand) ** 2).sum())
    ssw = float(((run_props[p_col] - run_props.groupby(task_col)[p_col].transform("mean")) ** 2).sum())
    msb, msw = ssb / (a - 1), ssw / (N - a)
    n0 = (N - float((k ** 2).sum()) / N) / (a - 1)
    den = msb + (n0 - 1) * msw
    icc = float((msb - msw) / den) if den > 0 else float("nan")
    return {"icc": icc, "msb": msb, "msw": msw, "n0": n0, "n_runs": N, "n_tasks": a}


def scaled_mde(planning_mde: float, n_informative: int, icc: float, m: int = 5,
               planning_informative: int = 6) -> float:
    """PROVISIONAL (not defined in the pre-registration): planning MDE scaled by
    sqrt(6 / n_informative) and by the design effect sqrt(1 + (m - 1) ICC), m = 5 episodes
    per run. ``inf`` when no task is informative."""
    if n_informative <= 0:
        return float("inf")
    icc = 0.0 if icc is None or math.isnan(icc) else max(0.0, icc)
    return float(planning_mde * math.sqrt(planning_informative / n_informative) * math.sqrt(1 + (m - 1) * icc))


# =========================================================================== collect (I/O)
_TRIAL_RE = re.compile(r"Error running experiment for config (\S+): ([A-Za-z_][\w.]*): ?(.*)")


def _read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _iter_lines(p: Path) -> Iterable[str]:
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            yield line


def scan_container_log(log: Path) -> dict[str, Any]:
    """Inner-CV fallback lines, run-level exceptions (``run_experiments`` swallows them) and
    the Tester step/trial context of the last ``test_hypotheses`` start before each one."""
    out = {"inner_cv_fallback": 0, "exceptions": [], "tracebacks": 0}
    last_ctx = None
    buf: list[str] = []
    for line in _iter_lines(log):
        if INNER_CV_FALLBACK_TEXT in line:
            out["inner_cv_fallback"] += 1
        if "Traceback (most recent call last)" in line:
            out["tracebacks"] += 1
        m = re.search(r"\[TesterAgent step=(\d+) trial=(\d+)", line)
        if m:
            last_ctx = f"TesterAgent step={m.group(1)} trial={m.group(2)}"
        buf.append(line.rstrip("\n"))
        if len(buf) > 8:
            buf.pop(0)
        if "Error running experiment for config" in line:
            out["_pending"] = (len(out["exceptions"]), last_ctx)
            out["exceptions"].append({"raw": line.strip(), "context": last_ctx, "type": None, "message": None})
            out["_collect"] = 6
            continue
        if out.get("_collect"):
            out["_collect"] -= 1
            e = out["exceptions"][-1]
            e["raw"] += " " + line.strip()
            if e["type"] is None:
                mm = re.search(r"\b([A-Z][A-Za-z]*(?:Error|Exception))\b:?\s*(.*)", line)
                if mm:
                    e["type"], e["message"] = mm.group(1), mm.group(2).strip()
    for e in out["exceptions"]:
        if e["type"] is None:
            mm = re.search(r"\b([A-Z][A-Za-z]*(?:Error|Exception))\b", e["raw"])
            e["type"] = mm.group(1) if mm else "unknown"
        msg = e["raw"]
        if "Input data contains `inf`" in msg or "Input data contains inf" in msg:
            e["signature"] = f"{e['type']}: input contains inf (Tester test_hypotheses, XGBoost DMatrix)"
        else:
            e["signature"] = f"{e['type']}: {(e['message'] or '')[:80]}"
    out.pop("_pending", None)
    out.pop("_collect", None)
    return out


def inner_fold_scores_check(out_dir: Path) -> dict[str, Any]:
    """Every completed trial in ``df_test_pool_final.csv`` must carry parseable per-inner-fold
    scores and an inner-CV score."""
    p = out_dir / "df_test_pool_final.csv"
    if not p.exists():
        return {"ok": False, "n_completed": 0, "n_missing": None, "n_skipped": None, "reason": "no df_test_pool_final.csv"}
    df = pd.read_csv(p)
    comp = df[df["status"].astype(str) == "completed"]
    inner_col = "inner_cv_accuracy" if "inner_cv_accuracy" in df.columns else None
    if "inner_cv_RMSE" in df.columns and (inner_col is None or comp[inner_col].isna().all()):
        inner_col = "inner_cv_RMSE"
    missing = 0
    for _, r in comp.iterrows():
        raw = r.get("inner_cv_fold_scores")
        ok = False
        if isinstance(raw, str) and raw.strip():
            try:
                vals = json.loads(raw)
                ok = isinstance(vals, list) and len(vals) >= 2 and all(v is not None for v in vals)
            except Exception:  # noqa: BLE001
                ok = len(re.findall(r"[-\d.eE]+", raw)) >= 2
        if not ok or (inner_col and pd.isna(r.get(inner_col))):
            missing += 1
    return {"ok": missing == 0 and len(comp) > 0, "n_completed": int(len(comp)), "n_missing": int(missing),
            "n_skipped": int((df["status"].astype(str) == "iteration_skipped").sum()),
            "n_aborted": int((df["status"].astype(str) == "aborted").sum()), "inner_col": inner_col}


def held_out_score(out_dir: Path, task_type: str) -> Optional[float]:
    p = out_dir / "fold_aggregation_summary.csv"
    if not p.exists():
        return None
    s = pd.read_csv(p)
    metric = "accuracy" if task_type == "classification" else "n_RMSE"
    hit = s.loc[s["metric"] == metric, "mean"]
    return float(hit.iloc[0]) if len(hit) else None


def iteration_tokens(out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for line in _iter_lines(out_dir / "iteration_log.jsonl"):
        if not line.strip():
            continue
        r = json.loads(line)
        tok = r.get("tokens") or {}
        rec = {"index": r.get("index"), "fold_run_id": r.get("fold_run_id"), "test_fold_id": r.get("test_fold_id"),
               "iteration_in_fold": r.get("iteration_in_fold"), "status": r.get("status"),
               "raw_coverage": r.get("raw_coverage")}
        tot = 0
        for role in ("scientist", "extractor", "tester", "unattributed"):
            c = int((tok.get(role) or {}).get("compl") or 0)
            rec[f"compl_{role}"] = c
            tot += c
        rec["compl_total"] = tot
        rows.append(rec)
    return rows


def k10_pick_reproduces(out_dir: Path, task_type: str) -> Optional[bool]:
    """A21 validity: argmax inner-CV over the fold's completed trials (ties: earliest, as
    ``compute_fold_aggregation``) equals the ``fold_aggregation_best_iterations.csv`` pick."""
    tp, bp = out_dir / "df_test_pool_final.csv", out_dir / "fold_aggregation_best_iterations.csv"
    if not tp.exists() or not bp.exists():
        return None
    # round_trip: the default C parser is not bit-exact, which breaks near-ties
    df, best = pd.read_csv(tp, float_precision="round_trip"), pd.read_csv(bp, float_precision="round_trip")
    col = "inner_cv_accuracy" if task_type == "classification" else "inner_cv_RMSE"
    if col not in df.columns:
        return None
    df = df[df["status"].astype(str) == "completed"]
    for _, b in best.iterrows():
        if str(b.get("status")) != "completed":
            continue
        f = df[df["test_fold_id"] == b["test_fold_id"]].copy()
        f[col] = pd.to_numeric(f[col], errors="coerce")
        f = f.dropna(subset=[col])
        if not len(f):
            return False
        pick = f.loc[f[col].idxmax() if task_type == "classification" else f[col].idxmin(), "id"]
        if str(pick) != str(b["id"]):
            return False
    return True


def _attempt_dirs(root: Path, row_id: str) -> list[Path]:
    """Earlier attempts live in ``<row_id>.attempt<k>.<epoch>``; the final one in ``<row_id>``."""
    olds = sorted(root.glob(f"{row_id}.attempt*"), key=lambda p: int(re.search(r"attempt(\d+)", p.name).group(1)))
    final = root / row_id
    return olds + ([final] if final.exists() else [])


def _output_dir(run_dir: Path) -> Optional[Path]:
    hits = sorted(run_dir.glob("tasks/*/*/output"))
    return hits[0] if hits else None


def collect(args: argparse.Namespace) -> int:  # pragma: no cover - I/O driver
    root = Path(args.repo)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.agent2027.analysis import a02_readout as a02  # noqa: PLC0415
    from scripts.agent2027.analysis import recovery_detector as rd  # noqa: PLC0415

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs_roots = [Path(r) for r in args.runs]
    man = pd.concat([pd.read_csv(m, dtype=str) for m in args.manifest], ignore_index=True)
    man = man[man["notes"].fillna("") != "smoke"]
    man = man[man["row_id"].str.match(r"^A0[357]-0\d\d$")]
    status = pd.concat([pd.read_csv(r / "status.tsv", sep="\t").assign(session=r.name)
                        for r in runs_roots if (r / "status.tsv").exists()], ignore_index=True)
    problems: list[str] = []

    # ---- base_no_fe (A05-T6 Core-10 + A08 SYN-P)
    base: dict[tuple[str, int], dict] = {}
    for f in args.base_jsonl:
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                base[(d["dataset"], int(d["fold_seed"]))] = d

    def base_vals(ds: str, ttype: str, seed: int) -> tuple[Optional[float], Optional[float]]:
        d = base.get((ds, seed))
        if d is None:
            return None, None
        key = "nrmse" if ttype == "regression" else "accw_balanced"
        return d.get(key), d.get(f"{key}_sd")

    # ---- W&B
    wb: dict[str, list[dict]] = {}
    if args.wandb:
        try:
            import wandb  # noqa: PLC0415
            api = wandb.Api(timeout=180)
            for grp in ("A05", "A07", "A03"):
                for r in api.runs(f"{args.entity}/{args.project}", filters={"group": grp}):
                    cfg = dict(r.config)
                    if cfg.get("code_phase") in ("smoke",):
                        continue
                    arts = []
                    try:
                        arts = [a.type for a in r.logged_artifacts()]
                    except Exception as exc:  # noqa: BLE001
                        problems.append(f"W&B artifacts of {r.id}: {type(exc).__name__}")
                    summ = {k: v for k, v in dict(r.summary).items()
                            if not str(k).startswith(("_", "feature_metrics", "feature_metric_warnings"))
                            and not isinstance(v, (dict, list))}
                    rec = {"id": r.id, "name": r.name, "state": r.state, "tags": list(r.tags or []),
                           "created_at": str(r.created_at), "group": grp,
                           "config": {k: cfg.get(k) for k in ("manifest_row", "arm_id", "dataset_id", "fold_seed",
                                                              "replicate", "llm_seed", "session_id", "image_id",
                                                              "image_tag", "git_sha", "code_phase", "prereg_tag",
                                                              "vllm_image_id")},
                           "flags": {k: v for k, v in cfg.items() if str(k).startswith("flags/")},
                           "summary": summ, "has_run_output": "run-output" in arts}
                    wb.setdefault(str(cfg.get("manifest_row") or ""), []).append(rec)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"W&B fetch failed: {type(exc).__name__}: {exc}")
    (out / "g2_wandb.json").write_text(json.dumps(wb, indent=1, default=str), encoding="utf-8")

    # ---- per row
    det_cache: dict[str, Any] = {}
    row_recs, ep_recs, it_recs, att_recs = [], [], [], []
    for _, m in man.iterrows():
        rid, ds, ttype = m["row_id"], m["dataset_id"], m["task_type"]
        seed, rep = int(m["fold_seed"]), int(m["replicate"])
        st = status[status["row_id"] == rid].copy()
        st["attempt"] = st["attempt"].astype(int)
        st = st.sort_values("attempt")
        session = st["session"].iloc[-1] if len(st) else None
        sroot = next((r for r in runs_roots if r.name == session), None)
        dirs = _attempt_dirs(sroot, rid) if sroot else []
        exc_all, fb_all = [], 0
        for k, d in enumerate(dirs, start=1):
            sc = scan_container_log(d / "logs" / "container.log")
            fb_all += sc["inner_cv_fallback"]
            a_st = st[st["attempt"] == k]
            att_recs.append({"row_id": rid, "attempt": k, "dir": d.name, "session": session,
                             "status": a_st["status"].iloc[0] if len(a_st) else None,
                             "rc": a_st["rc"].iloc[0] if len(a_st) else None,
                             "wall_s": float(a_st["wall_s"].iloc[0]) if len(a_st) else None,
                             "reason": a_st["reason"].iloc[0] if len(a_st) else None,
                             "inner_cv_fallback_lines": sc["inner_cv_fallback"],
                             "n_run_exceptions": len(sc["exceptions"]),
                             "exception_signature": ";".join(e["signature"] for e in sc["exceptions"]),
                             "exception_context": ";".join(str(e["context"]) for e in sc["exceptions"]),
                             "exception_raw": " || ".join(e["raw"][:400] for e in sc["exceptions"])})
            exc_all += sc["exceptions"]
        final = st.iloc[-1] if len(st) else None
        out_dir = _output_dir(sroot / rid) if sroot and (sroot / rid).exists() else None
        row_ok = bool(final is not None and final["status"] == "OK" and out_dir is not None
                      and (out_dir / "fold_aggregation_summary.csv").exists())
        cap = bool((st["status"].astype(str).str.upper() == "CAP_HIT").any())
        summ = _read_json(out_dir / "instrumentation_summary.json") if out_dir else None
        summ = summ or {}
        sig = _read_json(out_dir / "scientist_signal_summary.json") if out_dir else None
        rep_v = _read_json(out_dir / "replay_validity_summary.json") if out_dir else None
        ifs = inner_fold_scores_check(out_dir) if out_dir else {"ok": False}
        traces = out_dir / "llm_traces.jsonl.gz" if out_dir else None
        traces_ok = bool(traces and traces.exists() and summ.get("trace_records") == summ.get("llm_calls_total")
                         and (summ.get("trace_write_errors") or 0) == 0)
        its = iteration_tokens(out_dir) if out_dir else []
        for it in its:
            it_recs.append({"row_id": rid, "dataset_id": ds, **it})
        cov0 = [it["raw_coverage"] for it in its if it["iteration_in_fold"] == 0 and it["raw_coverage"] is not None]
        b_mean42, b_sd42 = base_vals(ds, ttype, seed)
        score = held_out_score(out_dir, ttype) if row_ok else None
        wbr = sorted(wb.get(rid, []), key=lambda r: r["created_at"])
        wfin = wbr[-1] if wbr else None
        rec = {"row_id": rid, "ablation": m["ablation_id"], "arm": m["arm_id"], "dataset_id": ds, "task_type": ttype,
               "fold_seed": seed, "replicate": rep, "llm_seed": int(m["llm_seed"]), "session": session,
               "n_attempts": int(len(st)), "attempt_statuses": ";".join(st["status"].astype(str)),
               "final_status": None if final is None else final["status"], "row_ok": int(row_ok), "cap_hit": int(cap),
               "itt": int(not row_ok or cap), "wall_s_final": None if final is None else float(final["wall_s"]),
               "wall_s_total": float(st["wall_s"].astype(float).sum()) if len(st) else None,
               "inner_cv_fallback_lines": fb_all, "n_run_exceptions_all_attempts": len(exc_all),
               "heldout_score": score, "base_mean": b_mean42, "base_sd": b_sd42,
               "traces_ok": int(traces_ok), "inner_fold_scores_ok": int(bool(ifs.get("ok"))),
               "n_completed_trials": ifs.get("n_completed"), "n_trials_missing_inner": ifs.get("n_missing"),
               "n_iteration_skipped": ifs.get("n_skipped"), "n_aborted_trials": ifs.get("n_aborted"),
               "raw_coverage_iter0_min": min(cov0) if cov0 else None,
               "replay_check_ran": (rep_v or {}).get("replay_check_ran"),
               "replay_nonrepro_folds": (rep_v or {}).get("replay_nonrepro_folds"),
               "l1_split_disjoint": (sig or {}).get("l1_split_disjoint"),
               "l1_split_failures": (sig or {}).get("l1_split_failures"),
               "k10_pick_reproduces": k10_pick_reproduces(out_dir, ttype) if (out_dir and row_ok) else None,
               "iter_compl_sum": int(sum(it["compl_total"] for it in its)),
               "n_iterations_logged": len(its),
               "wandb_ids": ";".join(r["id"] for r in wbr), "wandb_n_runs": len(wbr),
               "wandb_state_final": wfin["state"] if wfin else None,
               "wandb_run_output": int(bool(wfin and wfin["has_run_output"])),
               "wandb_image_id": (wfin or {}).get("config", {}).get("image_id"),
               "wandb_vllm_image_id": (wfin or {}).get("config", {}).get("vllm_image_id"),
               "wandb_prereg_tag": (wfin or {}).get("config", {}).get("prereg_tag")}
        for k in ("tok_compl_scientist", "tok_compl_extractor", "tok_compl_tester", "tok_compl_total",
                  "tok_compl_tester_assess", "tok_compl_tester_prune", "tok_compl_tester_other",
                  "tok_compl_unattributed", "tok_search_scientist", "tok_search_extractor", "tok_search_tester",
                  "llm_calls_total", "trace_records", "trace_write_errors", "wall_s", "err_per_accepted",
                  "n_tool_errors", "n_accepted", "context_errors", "serper_calls", "serper_ok_false_rate",
                  "fallback_share", "max_prompt_tokens", "runner_errors", "llm_call_errors"):
            rec[k] = summ.get(k)
        if wfin:
            ws = wfin["summary"]
            for k in ("fun_proposed", "fun_executes", "wall_s", "energy_j"):
                rec[f"wb_{k}"] = ws.get(k)
            rec["wb_fun_keys"] = ";".join(sorted(k for k in ws if str(k).startswith("fun_")))
        # recovery (A05 SYN-P rows)
        if ds.startswith("synp_") and m["ablation_id"] == "A05":
            if ds not in det_cache:
                tdir = Path(args.tasks) / ttype / ds
                df_raw = pd.read_csv(tdir / "data.csv")
                labels = pd.read_csv(tdir / "labels.csv")
                phi, det = rd.load_phi(args.meta, ds, df_raw)
                det_cache[ds] = (df_raw, labels, phi, det)
            df_raw, labels, phi, det = det_cache[ds]
            stratified = json.loads(m["env_overrides"] or "{}").get("ROGUE_ONE_STRATIFIED_FOLDS", "False") == "True"
            scored = None
            if row_ok:
                eps = []
                for ep in a02.build_row_episodes(out_dir):
                    # registered (pre-amendment) detector, kept for the both-detectors table
                    try:
                        r = a02.score_episode(ep, df_raw, labels, phi, det, seed, stratified, amended=False)
                    except Exception as exc:  # noqa: BLE001
                        problems.append(f"{rid} fold {ep['fold']}: detector error {type(exc).__name__}: {exc}")
                        r = {"rec_any": 0, "rec_sel": 0, "mode": "detector_error"}
                    # detector of record: recovery_detector's amended path (Part III.2, Amendment 2)
                    try:
                        ra = a02.score_episode(ep, df_raw, labels, phi, det, seed, stratified, amended=True)
                        r.update({"rec_any_amended": ra["rec_any"], "rec_sel_amended": ra["rec_sel"],
                                  "hits_sel_amended": ra["hits_sel"]})
                    except Exception as exc:  # noqa: BLE001
                        problems.append(f"{rid} fold {ep['fold']}: amended detector error {type(exc).__name__}: {exc}")
                        r.update({"rec_any_amended": 0, "rec_sel_amended": 0, "hits_sel_amended": "detector_error"})
                    if ep["selected_iter"] is None:
                        problems.append(f"{rid} fold {ep['fold']}: no completed inner-CV selection -> rec_sel 0")
                    eps.append({"fold": ep["fold"], "selected_iter": ep["selected_iter"],
                                "n_fold_runs": ep["n_fold_runs"], **r})
                scored = pd.DataFrame(eps)
            ep_df = a02.itt_episode_rows(row_ok and not cap, scored)
            for c in ("rec_sel_amended", "rec_any_amended"):
                ep_df[c] = (ep_df[c] if c in ep_df.columns else pd.Series(0, index=ep_df.index)).fillna(0).astype(int)
            for _, e in ep_df.iterrows():
                ep_recs.append({"row_id": rid, "dataset_id": ds, "fold_seed": seed, "replicate": rep, **e.to_dict()})
            rec["rec_sel"] = int(ep_df["rec_sel"].sum())
            rec["rec_any"] = int(ep_df["rec_any"].sum())
            rec["rec_sel_amended"] = int(ep_df["rec_sel_amended"].sum())
            rec["rec_any_amended"] = int(ep_df["rec_any_amended"].sum())
        row_recs.append(rec)
        print(f"[collect] {rid} {ds} ok={row_ok} attempts={len(st)}", file=sys.stderr, flush=True)

    pd.DataFrame(row_recs).to_csv(out / "g2_rows.csv", index=False)
    pd.DataFrame(ep_recs).to_csv(out / "g2_episodes.csv", index=False)
    pd.DataFrame(it_recs).to_csv(out / "g2_iterations.csv", index=False)
    pd.DataFrame(att_recs).to_csv(out / "g2_attempts.csv", index=False)

    # ---- A15 leak checks on A05 P6b runs
    leak = a15_leak_checks(man, status, runs_roots, Path(args.tasks), Path(args.meta), problems)
    (out / "g2_leak.json").write_text(json.dumps(leak, indent=1, default=str), encoding="utf-8")
    (out / "g2_collect_problems.json").write_text(json.dumps(problems, indent=1), encoding="utf-8")
    return 0


# ---- A15 leak audit (A15 section 6; A15-T22 checks a-d; the script a15_leak_audit.py does not exist)
CODE_READ_PATTERNS = ["open(", "read_csv", "read_json", "read_table", "read_parquet", "sqlite", "serper_cache",
                      "/workspace", "os.listdir", "os.walk", "glob", "os.environ", "wandb", "subprocess",
                      "requests", "urllib", "socket"]


def p6_terms(meta_dir: Path, tasks: Path) -> dict[str, list[str]]:
    """The frozen term list ``synp_meta/p6_domain_terms.txt`` (A15-T7) is not on CAIR.
    Derived here (disclosed): strong = the formula name and its parts; other = every P6a
    column name and its underscore tokens (>= 4 chars, generic words dropped) and every
    P6a level string."""
    m = json.loads((meta_dir / "synp_p6b.json").read_text(encoding="utf-8"))
    strong = ["reynolds", "kinematic viscosity", "viscosity", "flow velocity", "pipe diameter", "pipe-flow",
              "pipe flow"]
    cols = list(pd.read_csv(tasks / "regression" / "synp_p6a" / "data.csv", nrows=1).columns)
    # Plain English words that also occur in templates, metric names or agent prose are not
    # P6a-specific; they are dropped (A15-T7 excludes generic words such as "ratio").
    generic = {"total", "count", "days", "hours", "months", "rate", "level", "site", "with", "from", "ratio",
               "index", "value", "type", "flow", "noise", "power", "relative", "length", "line", "factor",
               "interval", "region", "shift", "supply", "operating", "operator", "inspection", "maintenance",
               "sensor", "ambient", "temperature", "pressure", "concentration", "vendor"}
    other = set()
    for c in cols:
        if c == "id":
            continue
        other.add(c.lower())
        other.update(t for t in c.lower().split("_") if len(t) >= 4 and t not in generic and not t.isdigit())
    lm = m.get("level_map") or {}
    for col, levels in lm.items():
        other.add(col.lower())
        other.update(str(lv).lower() for lv in levels if "_" in str(lv))  # vendor_a..d; not north/day/...
    return {"strong": sorted(strong), "other": sorted(other - set(strong))}


def _is_input_prompt(name: str) -> bool:
    """A15 leak scope: system prompts, tool docstrings and first user messages (iteration 0,
    first call). Later user messages carry agent-written text ('guessed domain')."""
    return name.endswith(".system.txt") or name.endswith(".tools.json") or (
        name.endswith(".user.txt") and "__iter000__c01" in name)


def _term_hits(text: str, terms: Sequence[str]) -> list[str]:
    return [t for t in terms if re.search(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![A-Za-z0-9])", text, re.I)]


def a15_leak_checks(man: pd.DataFrame, status: pd.DataFrame, roots: Sequence[Path], tasks: Path,
                    meta: Path, problems: list[str]) -> dict[str, Any]:  # pragma: no cover - I/O
    terms = p6_terms(meta, tasks)
    all_terms = terms["strong"] + terms["other"]
    a05 = man[(man["ablation_id"] == "A05") & man["dataset_id"].isin(["synp_p6a", "synp_p6b"])]
    res: dict[str, Any] = {"terms_source": "derived (A15-T7 p6_domain_terms.txt absent)", "terms": terms, "runs": []}

    def outdir(rid: str) -> Optional[Path]:
        st = status[status["row_id"] == rid]
        if not len(st):
            return None
        sroot = next((r for r in roots if r.name == st["session"].iloc[-1]), None)
        return _output_dir(sroot / rid) if sroot else None

    for _, m in a05[a05["dataset_id"] == "synp_p6b"].iterrows():
        twin = a05[(a05["dataset_id"] == "synp_p6a") & (a05["fold_seed"] == m["fold_seed"])
                   & (a05["replicate"] == m["replicate"])]
        ob, oa = outdir(m["row_id"]), outdir(twin["row_id"].iloc[0]) if len(twin) else None
        r: dict[str, Any] = {"row_id": m["row_id"], "twin": twin["row_id"].iloc[0] if len(twin) else None}
        # (a) prompt hits not shared verbatim with the P6a dump of the same role and call
        leaks, template, guessed = [], [], 0
        if ob is not None and (ob / "prompts").exists():
            for f in sorted((ob / "prompts").glob("*.txt")) + sorted((ob / "prompts").glob("*.json")):
                if not _is_input_prompt(f.name):
                    guessed += sum(1 for ln in f.read_text(encoding="utf-8", errors="replace").splitlines()
                                   if _term_hits(ln, all_terms))
                    continue
                twin_lines = set()
                if oa is not None and (oa / "prompts" / f.name).exists():
                    twin_lines = set((oa / "prompts" / f.name).read_text(encoding="utf-8", errors="replace").splitlines())
                for ln in f.read_text(encoding="utf-8", errors="replace").splitlines():
                    h = _term_hits(ln, all_terms)
                    if h:
                        (template if ln in twin_lines else leaks).append({"file": f.name, "terms": h, "line": ln[:200]})
        r["a_prompt_leaks"] = leaks
        r["a_later_message_hits_guessed_domain"] = guessed
        r["a_template_constant_hits"] = len(template)
        r["a_template_terms"] = sorted({t for x in template for t in x["terms"]})
        # (b) feedback pool dataset_id
        ids = set()
        if ob is not None:
            for f in (ob / "intermediate_steps").glob("df_scientist_feedback_trail_*.csv"):
                ids.update(pd.read_csv(f, usecols=["dataset_id"])["dataset_id"].astype(str).unique())
        r["b_feedback_dataset_ids"] = sorted(ids)
        r["b_ok"] = ids == {"synp_p6b"}
        # (c) agent-written code that reads something other than in-memory frames
        hits = []
        tr = ob / "llm_traces.jsonl.gz" if ob is not None else None
        if tr is not None and tr.exists():
            with gzip.open(tr, "rt", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    rec = json.loads(line)
                    msg = rec.get("message") or {}
                    for tc in (msg.get("tool_calls") or []):
                        argstr = str(((tc or {}).get("function") or {}).get("arguments") or "")
                        h = [p for p in CODE_READ_PATTERNS if p in argstr]
                        if h:
                            hits.append({"call": i, "agent": rec.get("agent_name"), "patterns": h, "args": argstr[:300]})
        r["c_code_hits"] = hits
        # (d) fold test sets identical across twins (same rows, id, labels)
        try:
            da = pd.read_csv(tasks / "regression" / "synp_p6a" / "data.csv", usecols=["id"])
            db = pd.read_csv(tasks / "regression" / "synp_p6b" / "data.csv", usecols=["id"])
            la = (tasks / "regression" / "synp_p6a" / "labels.csv").read_bytes()
            lb = (tasks / "regression" / "synp_p6b" / "labels.csv").read_bytes()
            from rogueone.utils.train_test_splitter import deterministic_kfold_indices  # noqa: PLC0415
            import inspect  # noqa: PLC0415
            sig = inspect.signature(deterministic_kfold_indices)
            r["d_ids_equal"] = bool(da["id"].equals(db["id"]))
            r["d_labels_equal"] = la == lb
            r["d_splitter_signature"] = str(sig)
            r["d_ok"] = r["d_ids_equal"] and r["d_labels_equal"]
        except Exception as exc:  # noqa: BLE001
            r["d_ok"] = None
            r["d_error"] = f"{type(exc).__name__}: {exc}"
        r["input_leak"] = bool(leaks) or not r["b_ok"]
        res["runs"].append(r)
    res["any_input_leak"] = any(r["input_leak"] for r in res["runs"])
    res["any_code_hits_for_review"] = sum(len(r["c_code_hits"]) for r in res["runs"])
    return res


# =========================================================================== analyze
def analyze(args: argparse.Namespace) -> int:  # pragma: no cover - driver; math is unit-tested above
    src = Path(args.inp)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(src / "g2_rows.csv")
    eps = pd.read_csv(src / "g2_episodes.csv")
    reg = None
    if "rec_sel_amended" in eps.columns:  # detector of record = amended (Part VI.2 detector-bug rule)
        reg = eps.copy()
        eps["rec_sel_registered"], eps["rec_any_registered"] = eps["rec_sel"], eps["rec_any"]
        eps["rec_sel"], eps["rec_any"] = eps["rec_sel_amended"], eps["rec_any_amended"]
    its =pd.read_csv(src / "g2_iterations.csv")
    att = pd.read_csv(src / "g2_attempts.csv")
    leak = json.loads((src / "g2_leak.json").read_text(encoding="utf-8"))
    base = {}
    for f in args.base_jsonl:
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                base[(d["dataset"], int(d["fold_seed"]))] = d
    pairs_w0 = pd.read_csv(args.w0_pairs)
    C: dict[str, Any] = {"generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    a05 = rows[rows["ablation"] == "A05"].copy()

    def bsd(ds: str, seed: int) -> float:
        tt = CORE10.get(ds, "classification" if ds in SYNP_CLASSIFICATION else "regression")
        key = "nrmse" if tt == "regression" else "accw_balanced"
        return float(base[(ds, seed)][f"{key}_sd"])

    def bmean(ds: str, seed: int) -> float:
        tt = CORE10.get(ds, "classification" if ds in SYNP_CLASSIFICATION else "regression")
        key = "nrmse" if tt == "regression" else "accw_balanced"
        return float(base[(ds, seed)][key])

    # ---- ITT imputation (Part IV.3: base_no_fe) and success
    def score_itt(r) -> float:
        if int(r["itt"]) == 1 or pd.isna(r["heldout_score"]):
            return bmean(r["dataset_id"], int(r["fold_seed"]))
        return float(r["heldout_score"])
    rows["score_itt"] = rows.apply(score_itt, axis=1)

    def success(r) -> int:
        if int(r["itt"]) == 1 or pd.isna(r["heldout_score"]):
            return 0
        m, s = bmean(r["dataset_id"], int(r["fold_seed"])), bsd(r["dataset_id"], int(r["fold_seed"]))
        sc = float(r["heldout_score"])
        ok = sc <= m + NI_SUCCESS_MULT * s if r["task_type"] == "regression" else sc >= m - NI_SUCCESS_MULT * s
        if r["dataset_id"] == "junglechess":
            ok = ok and sc >= JUNGLECHESS_RULE
        return int(ok)
    rows["success"] = rows.apply(success, axis=1)
    a05 = rows[rows["ablation"] == "A05"].copy()

    # ---- s_{d,s}, s_d
    sd_tab = []
    for ds in list(CORE10) + SYNP8 + ["synp_p0"]:
        r = {"dataset_id": ds}
        for s in (42, 4):
            if (ds, s) in base:
                r[f"base_mean_f{s}"] = bmean(ds, s)
                r[f"s_f{s}"] = bsd(ds, s)
        if "s_f42" in r and "s_f4" in r:
            r["s_d"] = s_d(r["s_f42"], r["s_f4"])
        sd_tab.append(r)
    sd_df = pd.DataFrame(sd_tab).set_index("dataset_id")

    # ---- Core-10 a, b, c
    def cell(ds, seed, rep):
        x = a05[(a05["dataset_id"] == ds) & (a05["fold_seed"] == seed) & (a05["replicate"] == rep)]
        return float(x["score_itt"].iloc[0]) if len(x) else float("nan")
    core = []
    for ds, tt in CORE10.items():
        a, b, c = cell(ds, 42, 0), cell(ds, 42, 1), cell(ds, 4, 0)
        core.append({"dataset_id": ds, "task_type": tt, "a_f42r0": a, "b_f42r1": b, "c_f4r0": c,
                     "s_d": float(sd_df.loc[ds, "s_d"])})
    core = pd.DataFrame(core)
    saa2 = sigma_aa2(core["a_f42r0"], core["b_f42r1"], core["s_d"])
    sp2 = sigma_p2(core["a_f42r0"], core["b_f42r1"], core["c_f4r0"], core["s_d"], saa2)
    Z = z_share(saa2, sp2)
    C["sigma2_AA"], C["sigma_AA"], C["sigma2_p"], C["Z"] = saa2, math.sqrt(saa2), sp2, Z
    C["Z_ge_0.85_wording"] = bool(Z >= WORDING_Z_MAX)

    # homogeneity
    w0 = pairs_w0[pairs_w0["dataset_id"].isin(POOLABLE_R) & pairs_w0["complete"].astype(str).str.lower().eq("true")]
    hom = homogeneity({d: (float(core.set_index("dataset_id").loc[d, "a_f42r0"]),
                           float(core.set_index("dataset_id").loc[d, "b_f42r1"])) for d in POOLABLE_R},
                      {d: w0.loc[w0["dataset_id"] == d, "delta_raw"].astype(float).tolist() for d in POOLABLE_R},
                      {d: float(sd_df.loc[d, "s_d"]) for d in POOLABLE_R})
    C["homogeneity"] = hom
    sig_rows = []
    for _, r in core.iterrows():
        raw = sigma_d_raw(r["a_f42r0"], r["b_f42r1"], r["c_f4r0"])
        fl = floor_sigma(raw, r["s_d"])
        pooled = r["dataset_id"] in POOLABLE_R and hom["pool"]
        sig = pooled_sigma(raw, T17_SD[r["dataset_id"]], r["s_d"]) if pooled else fl
        sig_rows.append({**r.to_dict(), "sigma_d_raw": raw, "floor": FLOOR_MULT * r["s_d"],
                         "floored": bool(raw < FLOOR_MULT * r["s_d"]), "pooled_with_t17": bool(pooled),
                         "sigma_d": sig, "M_d": M_MARGIN * sig})
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(out / "G2_sigma_d.csv", index=False)
    C["sigma_d"] = {r["dataset_id"]: r["sigma_d"] for r in sig_rows}
    C["M"] = M_MARGIN
    C["M_d"] = {r["dataset_id"]: r["M_d"] for r in sig_rows}
    C["A13_M"] = {"M": M_MARGIN, "note": "A13 no-harm: mean of 10 Delta_std, Var(Delta_std) ~ 1.5, bounds mean -+ 0.637; "
                  "'no harm beyond M' if lower bound > -1.0"}

    # ---- recovery: per task
    syn = eps.copy()
    syn["run"] = syn["row_id"]
    by_task = syn.groupby("dataset_id")["rec_sel"].sum().astype(int).to_dict()
    by_task_any = syn.groupby("dataset_id")["rec_any"].sum().astype(int).to_dict()
    s42 = syn[syn["fold_seed"] == 42].groupby("dataset_id")["rec_sel"].sum().astype(int).to_dict()
    rec_tab = pd.DataFrame([{"task": t, "rec_sel_15": by_task.get(t, 0), "rec_any_15": by_task_any.get(t, 0),
                             "rec_sel_f42_10": s42.get(t, 0),
                             "f42r0": int(syn[(syn.dataset_id == t) & (syn.fold_seed == 42) & (syn.replicate == 0)].rec_sel.sum()),
                             "f42r1": int(syn[(syn.dataset_id == t) & (syn.fold_seed == 42) & (syn.replicate == 1)].rec_sel.sum()),
                             "f4r0": int(syn[(syn.dataset_id == t) & (syn.fold_seed == 4)].rec_sel.sum()),
                             "rec_sel_15_registered_detector": int(syn.loc[syn.dataset_id == t, "rec_sel_registered"].sum())
                             if "rec_sel_registered" in syn else None,
                             "rec_any_15_registered_detector": int(syn.loc[syn.dataset_id == t, "rec_any_registered"].sum())
                             if "rec_any_registered" in syn else None}
                            for t in SYNP8 + ["synp_p0"]])
    rec_tab.to_csv(out / "G2_recovery_by_task.csv", index=False)
    inf7 = informative_tasks({t: by_task.get(t, 0) for t in SYNP7})
    inf_pop = [t for t in inf7 if t not in GATED_OUT]
    C["informative_tasks_synp7"] = inf7
    C["informative_tasks_confirmatory_pop"] = inf_pop
    C["informative_lt_4_wording"] = bool(len(inf7) < WORDING_INFORMATIVE_MIN)
    C["informative_by_contrast"] = {
        "C1a/C3(pooled)/C5b/A09/A12/A16/A20 (SYN-P 7 minus P1)": inf_pop,
        "C1b (discriminating SYN-P 7: p2,p3,p4,p5,p6a,p6b)": inf_pop,
        "C3 D3 population {p3,p5,p2}": [t for t in ("synp_p3", "synp_p5", "synp_p2") if t in inf7],
        "C4a (synp_p7)": "P7 is C4a's only task; boundary allowed (Full expected at 0)",
        "C4b (p6a, p6b)": [t for t in ("synp_p6a", "synp_p6b") if t in inf7],
        "C5a (SYN-P 7, not gated; Qwen rungs)": inf7,
    }
    # ICC
    pop7 = syn[syn["dataset_id"].isin(SYNP7)].rename(columns={"dataset_id": "task", "fold_seed": "seed"})
    pop8 = syn[syn["dataset_id"].isin(SYNP8)].rename(columns={"dataset_id": "task", "fold_seed": "seed"})
    icc = {}
    for name, pop in (("SYN-P 7 (primary)", pop7), ("SYN-P 8 (sensitivity)", pop8)):
        g = glmm_icc(pop)
        rp = pop.groupby(["task", "run"])["rec_sel"].mean().reset_index().rename(columns={"rec_sel": "p"})
        an = anova_icc(rp)
        icc[name] = {"glmm": g, "anova_run_props": an,
                     "icc_of_record": an["icc"] if g["singular"] else g["icc"],
                     "source": "ANOVA on run proportions (GLMM singular)" if g["singular"] else "GLMM (Laplace ML)"}
    C["icc_recovery"] = icc
    icc_rec = icc["SYN-P 7 (primary)"]["icc_of_record"]
    C["icc_gt_0.5_wording"] = bool(icc_rec is not None and not math.isnan(icc_rec) and icc_rec > WORDING_ICC_MAX)

    # sentinel
    sent = sentinel_task({t: s42.get(t, 0) for t in SYNP7}, {t: by_task.get(t, 0) for t in SYNP7})
    C["A06_sentinel"] = sent
    if reg is not None:
        rbt = reg.groupby("dataset_id")["rec_sel"].sum().astype(int).to_dict()
        r42 = reg[reg["fold_seed"] == 42].groupby("dataset_id")["rec_sel"].sum().astype(int).to_dict()
        C["sensitivity_registered_detector"] = {
            "rec_sel_by_task": {t: rbt.get(t, 0) for t in SYNP8 + ["synp_p0"]},
            "informative_synp7": informative_tasks({t: rbt.get(t, 0) for t in SYNP7}),
            "sentinel": sentinel_task({t: r42.get(t, 0) for t in SYNP7}, {t: rbt.get(t, 0) for t in SYNP7})["task"]}
    # A06 band table (8 SYN-P A/A pairs at seed 42)
    a05s = a05[a05["dataset_id"].isin(SYNP8) & (a05["fold_seed"] == 42)]
    r0 = a05s[a05s["replicate"] == 0].set_index("dataset_id").reindex(SYNP8)
    r1 = a05s[a05s["replicate"] == 1].set_index("dataset_id").reindex(SYNP8)
    s_task = {t: bsd(t, 42) for t in SYNP8}
    metrics = [("cost", "tok_compl_total", "log"), ("cost", "llm_calls_total", "log"), ("time", "wall_s", "log"),
               ("execution", "err_per_accepted", "log1p"), ("execution", "execute_rate", "rate"),
               ("execution", "context_errors", "log1p"), ("search", "serper_ok_false_rate", "rate"),
               ("score", "scaled_score", "scaled")]
    band = []
    st = sent["task"]
    for ch, mname, scale in metrics:
        def val(df, t):
            if mname == "execute_rate":
                fp, fe = df.loc[t, "wb_fun_proposed"] if "wb_fun_proposed" in df else np.nan, \
                    df.loc[t, "wb_fun_executes"] if "wb_fun_executes" in df else np.nan
                return float(fe) / float(fp) if fp and not pd.isna(fp) and not pd.isna(fe) else float("nan")
            if mname == "scaled_score":
                return float(df.loc[t, "score_itt"]) / s_task[t]
            return a06_transform(df.loc[t, mname], scale)
        v0 = [val(r0, t) for t in SYNP8]
        v1 = [val(r1, t) for t in SYNP8]
        sm = a06_sigma_aa_m(v0, v1)
        cen = (val(r0, st) + val(r1, st)) / 2.0
        b = a06_band(cen, sm, A06_FLOORS[scale]) if not math.isnan(cen) else {"centre": None, "half_width": None,
                                                                             "lo": None, "hi": None}
        band.append({"channel": ch, "metric": mname, "scale": scale, "sigma_AA_m": sm, **b,
                     "n_pairs": int(np.sum(~np.isnan(np.asarray(v0) - np.asarray(v1)))),
                     "note": "" if not math.isnan(cen) else "metric not logged (fun_* absent from W&B summary): NOT_DEFINED"})
    pd.DataFrame(band).to_csv(out / "G2_A06_band.csv", index=False)
    C["A06_band"] = band
    C["A06_s_task"] = s_task

    # ---- B_d, role and stage shares
    it_ok = its[its["row_id"].isin(a05["row_id"])]
    bd = it_ok[it_ok["status"].astype(str) == "completed"].groupby("dataset_id")["compl_total"].agg(["median", "count"])
    bd_all = it_ok.groupby("dataset_id")["compl_total"].agg(["median", "count"])
    C["B_d"] = {d: {"median_compl_per_iteration": float(bd_all.loc[d, "median"]), "n_iterations": int(bd_all.loc[d, "count"]),
                    "median_completed_only": float(bd.loc[d, "median"]) if d in bd.index else None}
                for d in bd_all.index}
    a05["s_S"] = a05["tok_compl_scientist"] / a05["tok_compl_total"]
    a05["s_E"] = a05["tok_compl_extractor"] / a05["tok_compl_total"]
    a05["s_T"] = a05["tok_compl_tester"] / a05["tok_compl_total"]
    a05["s_T_assess"] = a05["tok_compl_tester_assess"] / a05["tok_compl_total"]
    a05["s_T_prune"] = a05["tok_compl_tester_prune"] / a05["tok_compl_total"]
    shares = a05.groupby("dataset_id")[["s_S", "s_E", "s_T", "s_T_assess", "s_T_prune"]].median()
    shares.to_csv(out / "G2_token_shares.csv")
    syn7 = a05[a05["dataset_id"].isin(SYNP7)]
    sh_syn7 = syn7[["s_S", "s_E", "s_T", "s_T_assess", "s_T_prune"]].median().to_dict()
    sh_all = a05[["s_S", "s_E", "s_T", "s_T_assess", "s_T_prune"]].median().to_dict()
    C["token_shares"] = {"median_over_21_synp7_runs": sh_syn7, "median_over_55_runs": sh_all,
                         "n_synp7_runs": int(len(syn7))}
    C["A19_tau"] = {"s_E_median_synp7": sh_syn7["s_E"], "tau": a19_tau(sh_syn7["s_E"]),
                    "rule": "tau = 0.60 if median s_E <= 0.50 else s_E + 0.10 (O120 shares; reference Q122, disclosed)"}
    separable = bool((a05["tok_compl_tester_assess"].fillna(0) + a05["tok_compl_tester_prune"].fillna(0)
                      + a05["tok_compl_tester_other"].fillna(0) - a05["tok_compl_tester"]).abs().max() <= 1)
    C["A12_token_branch"] = {**a12_branch(sh_all["s_T_assess"], sh_all["s_T_prune"], separable),
                             "share_assess_median55": sh_all["s_T_assess"], "share_prune_median55": sh_all["s_T_prune"],
                             "share_assess_median_synp7": sh_syn7["s_T_assess"],
                             "share_prune_median_synp7": sh_syn7["s_T_prune"],
                             "A12-T3_note": "(i) is degenerate for FX (replaced stage has 0 tokens by construction); "
                                            "A12-T3 recommends judging A and J on tok_compl_tester; Henrik decides"}
    # u_SYN-P
    w_syn = a05[a05["dataset_id"].isin(SYNP8 + ["synp_p0"])]["wall_s_final"].median()
    w_core = a05[a05["dataset_id"].isin(CORE10)]["wall_s_final"].median()
    C["u_SYNP"] = {"median_synp_wall_s": float(w_syn), "median_core10_wall_s": float(w_core),
                   "u_SYNP": float(w_syn / w_core)}
    # A09 headroom
    rate = {t: by_task.get(t, 0) / 15.0 for t in SYNP7}
    C["A09_headroom"] = a09_headroom(rate, [t for t in SYNP7 if t not in GATED_OUT])
    # A21 validity
    a05ok = a05[a05["itt"] == 0]
    tok_ok = ((a05ok["iter_compl_sum"] - a05ok["tok_compl_total"]).abs() <= A21_TOK_TOL * a05ok["tok_compl_total"])
    C["A21_validity"] = {"tokens_within_1pct_share": float(tok_ok.mean()), "tokens_pass": bool(tok_ok.mean() >= A21_TOK_SHARE),
                         "tokens_fail_rows": a05ok.loc[~tok_ok, "row_id"].tolist(),
                         "k10_pick_reproduces_all": bool(a05ok["k10_pick_reproduces"].astype(str).eq("True").all()),
                         "k10_fail_rows": a05ok.loc[~a05ok["k10_pick_reproduces"].astype(str).eq("True"), "row_id"].tolist(),
                         "inner_fold_scores_all_present": bool(a05ok["inner_fold_scores_ok"].astype(int).eq(1).all())}
    C["A10_T18"] = {"per_iteration_active_feature_log": "present (df_attributes_explanations_trail_<i>.csv with Status, "
                    "Fallback, Seeded per iteration)", "option_needed": False}

    # ---- power (provisional)
    n_inf = len(inf_pop)
    C["power_recomputed"] = {
        "status": "provisional: method NOT_DEFINED in the pre-registration (01 4.7 says only 'recomputed at G2 from "
                  "M0's ICC and the number of informative tasks'); MDE_G2 = MDE_plan x sqrt(6/n_inf) x sqrt(1 + 4 ICC)",
        "n_informative_pop": n_inf, "icc_used": icc_rec,
        "rows": {k: {"planning": v[1], "planning_mid": v[0],
                     "mde_g2": scaled_mde(v[0], n_inf if k not in ("C4a", "C4b") else 6, icc_rec if k not in ("C4a", "C4b") else 0.0)}
                 for k, v in PLANNING_MDE.items()},
        "note": "C4a/C4b are not rescaled (their populations are fixed: P7 seeds; 15 McNemar pairs)"}

    # ---- A03 fallback read-out
    a03 = rows[rows["ablation"] == "A03"]
    pairs = []
    for _, r in a03.iterrows():
        ds = r["dataset_id"]
        full = a05[(a05["dataset_id"] == ds) & (a05["fold_seed"] == 42) & (a05["replicate"] == 0)]
        sh = a03_sigma_hat(ds, bsd(ds, 42))
        f = full.iloc[0] if len(full) else None
        pairs.append({"dataset": ds, "task_type": r["task_type"], "aug_row": None if f is None else f["row_id"],
                      "rep_row": r["row_id"], "aug": None if f is None or int(f["itt"]) else float(f["heldout_score"]),
                      "rep": None if int(r["itt"]) else float(r["heldout_score"]),
                      "aug_failed": bool(f is None or int(f["itt"])), "rep_failed": bool(int(r["itt"])),
                      "sigma_hat": sh["sigma_hat"], "sigma_terms": sh["terms"], "sigma_decided_by": sh["decided_by"]})
    C["A03_readout"] = a03_readout(pairs)
    pd.DataFrame(C["A03_readout"]["pairs"]).drop(columns=["sigma_terms"]).to_csv(out / "G2_A03_pairs.csv", index=False)
    # A05-T6 floor values (A03 sigma-hat table)
    C["A05_T6_floor"] = {ds: {"base_sd_f42": bsd(ds, 42), "floor_0.1x": FLOOR_MULT * bsd(ds, 42)} for ds in CORE10}

    # ---- NO-GO checks (A05's 55 runs; A07/A03 beside)
    def nogo_counts(df: pd.DataFrame, at: pd.DataFrame) -> dict[str, Any]:
        roles_zero = df[(df[["tok_compl_scientist", "tok_compl_extractor", "tok_compl_tester"]].fillna(0) <= 0).any(axis=1)
                        & (df["row_ok"] == 1)]
        missing = df[(df["row_ok"] == 1) & ((df["traces_ok"] == 0) | (df["inner_fold_scores_ok"] == 0)
                                            | (df["wandb_run_output"] == 0))]
        a = at[at["row_id"].isin(df["row_id"])]
        sigs = a[a["exception_signature"].fillna("") != ""]
        exc_counts: dict[str, list[str]] = {}
        for _, x in sigs.iterrows():
            for s in str(x["exception_signature"]).split(";"):
                exc_counts.setdefault(s, []).append(f"{x['row_id']}#a{x['attempt']}")
        return {"n_rows": int(len(df)), "n_attempts": int(len(a)),
                "inner_cv_fallback_rows": df.loc[df["inner_cv_fallback_lines"] > 0, "row_id"].tolist(),
                "zero_role_rows": roles_zero["row_id"].tolist(),
                "missing_artifact_rows": missing[["row_id", "traces_ok", "inner_fold_scores_ok", "wandb_run_output"]]
                .to_dict(orient="records"),
                "cap_hits": df.loc[df["cap_hit"] == 1, "row_id"].tolist(),
                "itt_failures": df.loc[df["itt"] == 1, "row_id"].tolist(),
                "retried_rows": df.loc[df["n_attempts"] > 1, "row_id"].tolist(),
                "exceptions_by_signature": exc_counts,
                "max_same_exception_runs": max((len(v) for v in exc_counts.values()), default=0)}
    ng = {"A05": nogo_counts(a05, att), "A07": nogo_counts(rows[rows["ablation"] == "A07"], att),
          "A03": nogo_counts(rows[rows["ablation"] == "A03"], att)}
    x = ng["A05"]
    ng["verdict_rule"] = nogo_verdict(len(x["inner_cv_fallback_rows"]), len(x["zero_role_rows"]),
                                      len(x["missing_artifact_rows"]), len(x["cap_hits"]), len(x["itt_failures"]),
                                      x["max_same_exception_runs"], None)
    C["nogo"] = ng
    # health
    C["health"] = {
        "A05_success": int(a05["success"].sum()), "A05_n": int(len(a05)),
        "raw_coverage_iter0_all_1": bool((a05["raw_coverage_iter0_min"] == 1.0).all()),
        "replay_check_ran_all": bool((a05["replay_check_ran"] == 1).all()),
        "replay_nonrepro_folds_total": int(a05["replay_nonrepro_folds"].fillna(0).sum()),
        "replay_rows_with_nonrepro": int((a05["replay_nonrepro_folds"].fillna(0) > 0).sum()),
        "l1_split_disjoint_all": bool((a05["l1_split_disjoint"] == 1).all()),
        "serper_ok_false_rate_max": float(a05["serper_ok_false_rate"].max()),
        "serper_ok_false_rate_rows_ge_5pct": a05.loc[a05["serper_ok_false_rate"] >= 0.05, "row_id"].tolist(),
        "context_errors_total": int(a05["context_errors"].fillna(0).sum()),
        "image_ids": sorted(set(map(str, rows["wandb_image_id"].dropna()))),
        "prereg_tags": sorted(set(map(str, rows["wandb_prereg_tag"].dropna()))),
    }
    p0 = a05[a05["dataset_id"] == "synp_p0"]
    p0r = p0.iloc[0] if len(p0) else None
    C["P0_check"] = None if p0r is None else {
        "rec_sel": int(eps.loc[eps["dataset_id"] == "synp_p0", "rec_sel"].sum()),
        "heldout": float(p0r["heldout_score"]), "base": bmean("synp_p0", 42), "s42": bsd("synp_p0", 42),
        "within_2s": bool(abs(float(p0r["heldout_score"]) - bmean("synp_p0", 42)) <= 2 * bsd("synp_p0", 42))}
    p7 = eps[eps["dataset_id"] == "synp_p7"]
    C["P7_canary"] = {"rec_any": int(p7["rec_any"].sum()), "rec_sel": int(p7["rec_sel"].sum()), "episodes": int(len(p7)),
                      "fires": bool(p7["rec_any"].sum() > 0)}
    C["A15_leak"] = {"any_input_leak": leak.get("any_input_leak"), "code_hits_for_review": leak.get("any_code_hits_for_review"),
                     "C4b_status": "invalid (enters Holm with p = 1)" if leak.get("any_input_leak") else "valid"}
    rows.to_csv(out / "G2_rows_scored.csv", index=False)
    (out / "g2_constants.json").write_text(json.dumps(C, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: C[k] for k in ("sigma_AA", "Z", "nogo")}, indent=1, default=str)[:4000])
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--manifest", nargs="+", required=True)
    c.add_argument("--tasks", required=True)
    c.add_argument("--meta", required=True)
    c.add_argument("--base-jsonl", nargs="+", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    c.add_argument("--wandb", action="store_true")
    c.add_argument("--entity", default="PlanBench")
    c.add_argument("--project", default="RogueOne-AGENT2027")
    a = sub.add_parser("analyze")
    a.add_argument("--inp", required=True)
    a.add_argument("--base-jsonl", nargs="+", required=True)
    a.add_argument("--w0-pairs", required=True)
    a.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return collect(args) if args.cmd == "collect" else analyze(args)


if __name__ == "__main__":  # pragma: no cover
    _root = str(Path(__file__).resolve().parents[3])
    if _root not in sys.path:
        sys.path.insert(0, _root)
    sys.exit(main())
