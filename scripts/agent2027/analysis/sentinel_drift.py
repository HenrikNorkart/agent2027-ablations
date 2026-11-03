"""A06 session-sentinel drift test (E11b: A06-T8 ``--session``, A06-T10/T16/T17 ``--final``).

Binding rules: ``docs/agent2027/preregistration.md`` Part V.6, restating
``docs/ablations/A06_M0S_session_sentinel.md`` sections 4.7 (per-session check), 4.8 (pooled tests)
and 6 (gates), as amended by Amendment 2 (sentinel task ``synp_p2``, band frozen at G2),
Amendment 3 (Q4 channel not evaluable when the Q4 reference runs capped) and Amendment 6
(execute-rate metric dropped: no ``fun_*`` keys in v2.0).

Per-session check (A06 section 4.7). Five channels; a channel is outside when any of its metrics
is outside its band. The O120 bands are the frozen G2 band table (``g2_constants.json``
``A06_band``: centre = mean of A05 r0/r1 on ``synp_p2``, half-width = max(3 sqrt(1.5) sigma_AA,m,
floor_m)); they are never recomputed here. Q4 bands: centre = mean of the Q4 anchor and
``sentinel_Q4`` k = 1, sigma = max(sigma_AA,m, |anchor - k1| / sqrt 2). Hard indicators (any ->
FLAG): persistent crash after the solo retry; cap hit (Q4: only if neither Q4 reference capped);
search-failure rate >= 0.5; inner-CV column missing or all-NaN; an R3/R4 flag differing from the
reference's (identity, model and port keys exempt; post-v2.0 flags exempt at their Full value);
a ``vllm_image_id`` differing from the reference's. Verdict: PASS (no channel outside, no hard
indicator), WARN (exactly one channel outside), FLAG (>= 2 channels outside, >= 3 for Q4; any hard
indicator; or the same channel outside in two consecutive sentinel-bearing sessions of the same arm
on the same image). ``rec_sel`` is descriptive only.

Pooled tests (A06 section 4.8; alpha = 0.05, outside Holm): primary two-sided Fisher exact test of
pooled O120 sentinel fold-episodes X/Y against M0's Z/10; sensitivities: exact test stratified by
outer fold (labels permuted within each fold), exact run-level permutation test (M0's 2 runs vs
the K sentinels); descriptive: exact Cochran-Armitage trend over session order (M0 = 0) and the Q4
homogeneity of ``rec_sel`` / ``func_success`` across sessions (Fisher-Freeman-Halton, 1e5 Monte
Carlo draws). ``--interim`` labels the same computation as the operational G4 read (A06-T16).

Data source. The run folders (``<runs>/<SESSION>/<ROW>/``: ``status.tsv``, ``env.list``,
``tasks/*/*/output/``) are read directly; ``env.list`` holds the launcher-set flags and
``ROGUE_ONE_VLLM_IMAGE_ID``. ``--wandb`` additionally compares the effective ``flags/*`` from the
W&B config (post-dotenv; the system of record). Recovery (``rec_sel``) comes from an episodes CSV
written by ``recovery_rows.py`` (detector of record, Amendment 2).

    # per session (all sentinel rows found are scored; --session filters the report)
    python sentinel_drift.py session --runs <runroot>/S* --manifest ~/ro27_manifests/*.csv \
        --g2-constants g2_constants.json --defaults scripts/agent2027/v2_defaults.env \
        --flag-registry rogueone/utils/flag_registry.py [--session S1024c] --out-dir <dir>
    # pooled tests (final, or --interim at G4)
    python sentinel_drift.py final --rows <dir>/sentinel_rows.csv --episodes eps.csv [--interim] --out-dir <dir>

Run on CAIR only inside a CPU container with the run folders mounted read-only.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import importlib.util
import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- constants
K_BAND = 3.0 * math.sqrt(1.5)
FLOORS = {"log": 0.1, "log1p": 0.1, "rate": 0.05, "scaled": 0.1}
#: (channel, metric, scale). execute_rate dropped by Amendment 6.
CHANNEL_METRICS = [
    ("cost", "tok_compl_total", "log"), ("cost", "llm_calls_total", "log"),
    ("time", "wall_s", "log"),
    ("execution", "err_per_accepted", "log1p"), ("execution", "context_errors", "log1p"),
    ("search", "serper_ok_false_rate", "rate"),
    ("score", "scaled_score", "scaled"),
]
CHANNELS = ["cost", "time", "execution", "search", "score"]
SEARCH_FAIL_HARD = 0.5
O120_FLAG_CHANNELS, Q4_FLAG_CHANNELS = 2, 3
SENTINEL_TASK = "synp_p2"
N_FOLDS = 5
INNER_CV_FALLBACK_TEXT = "falling back to the held-out metric"
#: Keys exempt from the flags/* comparison (A06 4.7 hard indicator 5): identity, model, port,
#: secrets and per-row seeds. Everything else in R3/R4 (and any other ROGUE_ONE_* key) is compared.
EXEMPT_EXACT = {
    "LLM_MODEL", "LLM_PORT", "LLM_API_KEY", "WANDB_API_KEY", "SERPER_API_KEY", "WANDB_PROJECT_NAME",
    "ROGUE_ONE_LLM_SEED", "ROGUE_ONE_ABLATION_ID", "ROGUE_ONE_ARM_ID", "ROGUE_ONE_REPLICATE",
    "ROGUE_ONE_SESSION_ID", "ROGUE_ONE_MANIFEST_ROW", "ROGUE_ONE_IS_SENTINEL", "ROGUE_ONE_CODE_PHASE",
    "ROGUE_ONE_IMAGE_TAG", "ROGUE_ONE_GIT_SHA", "ROGUE_ONE_SERVER_TOPOLOGY", "ROGUE_ONE_RUNG_S",
    "ROGUE_ONE_RUNG_E", "ROGUE_ONE_RUNG_T", "ROGUE_ONE_IMAGE_ID", "ROGUE_ONE_VLLM_IMAGE_ID",
    "ROGUE_ONE_PREREG_TAG", "ROGUE_ONE_PARALLEL",
}
EXEMPT_RE = re.compile(r"^ROGUE_ONE_(SCIENTIST|EXTRACTOR|TESTER)_(MODEL|PORT)$")
UNSET = "<unset>"


# =========================================================================== pure: bands
def transform(x: Any, scale: str) -> float:
    """Value on the channel scale: log, log1p, raw rate, or the already-scaled score."""
    if x is None:
        return float("nan")
    try:
        x = float(x)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(x):
        return x
    if scale == "log":
        return math.log(x) if x > 0 else float("nan")
    if scale == "log1p":
        return math.log1p(x) if x > -1 else float("nan")
    return x


def sigma_aa_m(r0: Sequence[float], r1: Sequence[float]) -> float:
    """sigma_AA,m = sqrt(sum d_i^2 / 2n), d_i = r0 - r1 on the metric's scale (NaN pairs dropped)."""
    d = np.asarray(r0, float) - np.asarray(r1, float)
    d = d[~np.isnan(d)]
    return float(math.sqrt(np.sum(d ** 2) / (2.0 * len(d)))) if len(d) else float("nan")


def band(centre: float, sigma: float, floor: float) -> dict[str, float]:
    """Half-width = max(3 sqrt(1.5) sigma, floor); a NaN sigma falls back to the floor."""
    hw = float(floor) if sigma is None or math.isnan(sigma) else max(K_BAND * float(sigma), float(floor))
    return {"centre": float(centre), "half_width": hw, "lo": float(centre) - hw, "hi": float(centre) + hw}


def q4_sigma(sigma_aa: float, anchor: float, k1: float) -> float:
    """sigma_Q4,m = max(sigma_AA,m, |anchor - k1| / sqrt 2)."""
    spread = abs(float(anchor) - float(k1)) / math.sqrt(2.0)
    if sigma_aa is None or math.isnan(sigma_aa):
        return spread
    return max(float(sigma_aa), spread)


def bands_from_g2(g2_band: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """The frozen O120 band table (G2) keyed by metric; rows without a centre (execute_rate) dropped."""
    out = {}
    for b in g2_band:
        if b.get("centre") is None:
            continue
        out[b["metric"]] = {"channel": b["channel"], "scale": b["scale"], "centre": float(b["centre"]),
                            "half_width": float(b["half_width"]), "lo": float(b["lo"]), "hi": float(b["hi"]),
                            "sigma_AA_m": float(b["sigma_AA_m"])}
    return out


def q4_bands(g2_band: Sequence[Mapping[str, Any]], anchor_vals: Mapping[str, float],
             k1_vals: Mapping[str, float]) -> dict[str, dict[str, Any]]:
    """Q4 band per metric; a metric whose anchor or k1 value is missing gets no band (not evaluable)."""
    sig = {b["metric"]: b.get("sigma_AA_m") for b in g2_band}
    out = {}
    for ch, m, scale in CHANNEL_METRICS:
        a, k = anchor_vals.get(m, float("nan")), k1_vals.get(m, float("nan"))
        if a is None or k is None or math.isnan(a) or math.isnan(k):
            continue
        s = sig.get(m)
        s = float("nan") if s is None else float(s)
        out[m] = {"channel": ch, "scale": scale, "sigma_AA_m": s, "sigma_Q4_m": q4_sigma(s, a, k),
                  **band((a + k) / 2.0, q4_sigma(s, a, k), FLOORS[scale])}
    return out


def metric_values(summary: Mapping[str, Any], heldout: Optional[float], s_task: float) -> dict[str, float]:
    """Channel metrics on their scales from ``instrumentation_summary.json`` and the held-out score."""
    vals = {}
    for _, m, scale in CHANNEL_METRICS:
        if m == "scaled_score":
            vals[m] = float("nan") if heldout is None else float(heldout) / float(s_task)
        else:
            vals[m] = transform(summary.get(m), scale)
    return vals


def channels_outside(vals: Mapping[str, float], bands: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Per channel: outside (any metric outside its band), the metrics outside, and the metrics that
    could not be evaluated (missing value or band). A channel with no evaluable metric is ``None``."""
    res: dict[str, Any] = {}
    for ch in CHANNELS:
        mets = [m for c, m, _ in CHANNEL_METRICS if c == ch]
        out_m, not_eval, evaluated = [], [], []
        for m in mets:
            b, v = bands.get(m), vals.get(m, float("nan"))
            if b is None or v is None or math.isnan(v):
                not_eval.append(m)
                continue
            evaluated.append(m)
            if v < b["lo"] or v > b["hi"]:
                out_m.append(m)
        res[ch] = {"outside": (bool(out_m) if evaluated else None), "metrics_outside": out_m,
                   "not_evaluable": not_eval}
    return res


def verdict(n_outside: int, hard: Sequence[str], repeated: Sequence[str], is_q4: bool) -> str:
    """PASS / WARN / FLAG (A06 4.7)."""
    if hard or repeated:
        return "FLAG"
    if n_outside >= (Q4_FLAG_CHANNELS if is_q4 else O120_FLAG_CHANNELS):
        return "FLAG"
    return "WARN" if n_outside >= 1 else "PASS"


# =========================================================================== pure: flags
def parse_env(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def is_exempt(key: str) -> bool:
    return key in EXEMPT_EXACT or bool(EXEMPT_RE.match(key))


def flag_diff(ref: Mapping[str, str], run: Mapping[str, str], full_defaults: Mapping[str, str],
              registry_keys: Iterable[str] = ()) -> list[dict[str, str]]:
    """Hard indicator 5. Compared keys: the R3/R4 registry keys plus every ``ROGUE_ONE_*`` key seen in
    either env, minus the exempt identity/model/port keys. A key the reference does not carry is a
    post-v2.0 flag: exempt iff the run has it at its Full value (``v2_defaults.env``); a post-v2.0
    flag with no Full value on record is reported (no Full value to be "at")."""
    keys = set(registry_keys) | {k for k in list(ref) + list(run) if k.startswith("ROGUE_ONE_")}
    diffs = []
    for k in sorted(keys):
        if is_exempt(k):
            continue
        rv, sv = ref.get(k, UNSET), run.get(k, UNSET)
        if rv == sv:
            continue
        if k not in ref:
            full = full_defaults.get(k)
            if full is not None and sv == full:
                continue
            reason = "post-v2.0 flag not at its Full value" if full is not None else \
                "post-v2.0 flag with no Full value in v2_defaults.env"
        else:
            reason = "differs from reference"
        diffs.append({"key": k, "reference": rv, "run": sv, "reason": reason})
    return diffs


# =========================================================================== pure: row verdict
def hard_indicators(row: Mapping[str, Any], *, is_q4: bool, q4_ref_capped: bool) -> tuple[list[str], list[str]]:
    """Return (hard indicators, descriptive notes) for one sentinel row."""
    hard, notes = [], []
    st = str(row.get("final_status") or "")
    if st == "FAILED":
        hard.append("1 persistent crash after the solo retry")
    if row.get("cap_hit"):
        if is_q4 and q4_ref_capped:
            notes.append("Q4 cap hit reported descriptively (a Q4 reference run capped)")
        else:
            hard.append("2 cap_hit")
    sfr = row.get("serper_ok_false_rate")
    if sfr is not None and not (isinstance(sfr, float) and math.isnan(sfr)) and float(sfr) >= SEARCH_FAIL_HARD:
        hard.append(f"3 serper_ok_false_rate {float(sfr):.3f} >= 0.5")
    if st == "OK" and row.get("inner_cv_ok") is False:
        hard.append("4 inner-CV column missing or all-NaN")
    if row.get("flag_diffs"):
        hard.append("5 flags differ: " + ", ".join(d["key"] for d in row["flag_diffs"]))
    vr, vs = row.get("ref_vllm_image_id"), row.get("vllm_image_id")
    if vr and vs and vr != vs:
        hard.append("6 vllm_image_id differs from reference")
    elif not vs:
        notes.append("vllm_image_id not logged")
    return hard, notes


def apply_consecutive(rows: list[dict[str, Any]]) -> None:
    """FLAG rule 3: the same channel outside in two consecutive sentinel-bearing sessions of the same
    arm on the same image (rows ordered by sentinel start time). Sets ``repeated_channels``."""
    by = {}
    for r in rows:
        by.setdefault((r["arm_id"], r.get("image_tag")), []).append(r)
    for grp in by.values():
        grp.sort(key=lambda r: (r.get("start_epoch") or 0, r["row_id"]))
        prev = None
        for r in grp:
            r["repeated_channels"] = []
            if r.get("verdict_basis") != "evaluated":
                continue
            if prev is not None and prev.get("session") != r.get("session"):
                r["repeated_channels"] = sorted(set(prev["channels_out"]) & set(r["channels_out"]))
            prev = r


def score_row(row: dict[str, Any], bands: Mapping[str, Mapping[str, Any]], *, is_q4: bool,
              q4_ref_capped: bool) -> dict[str, Any]:
    """Channels and hard indicators for one row (no consecutive rule yet)."""
    vals = row.get("values") or {}
    ch = channels_outside(vals, bands)
    row["channels"] = ch
    row["channels_out"] = [c for c, v in ch.items() if v["outside"]]
    row["channels_not_evaluable"] = [c for c, v in ch.items() if v["outside"] is None]
    row["hard"], row["notes"] = hard_indicators(row, is_q4=is_q4, q4_ref_capped=q4_ref_capped)
    return row


def finalise_verdicts(rows: list[dict[str, Any]]) -> None:
    """Verdict per row. ``verdict_basis``: evaluated -> PASS/WARN/FLAG; not_evaluable (Q4 without a
    usable reference) -> FLAG on a hard indicator, else NOT_EVALUABLE; reference (Q4 k = 1) ->
    REFERENCE; anything else (running, retry pending) -> PENDING."""
    apply_consecutive(rows)
    for r in rows:
        basis = r.get("verdict_basis")
        if basis == "evaluated":
            r["verdict"] = verdict(len(r["channels_out"]), r["hard"], r["repeated_channels"],
                                   r["arm_id"] == "sentinel_Q4")
        elif basis == "not_evaluable":
            r["verdict"] = "FLAG" if r["hard"] else "NOT_EVALUABLE"
        elif basis == "reference":
            r["verdict"] = "REFERENCE"
        else:
            r["verdict"] = "PENDING"


# =========================================================================== pure: pooled tests
def fisher_two_sided(x: int, y: int, z: int, n0: int) -> dict[str, float]:
    """Two-sided Fisher exact test: sentinels x/y vs M0 z/n0 (scipy convention)."""
    from scipy.stats import fisher_exact
    odds, p = fisher_exact([[x, y - x], [z, n0 - z]], alternative="two-sided")
    return {"x": int(x), "y": int(y), "z": int(z), "n0": int(n0), "rate_sentinel": x / y if y else float("nan"),
            "rate_m0": z / n0 if n0 else float("nan"), "odds_ratio": float(odds), "p": float(p)}


def _hypergeom_pmf(N: int, K: int, n: int) -> dict[int, float]:
    """P(X = k) of the number of successes among n draws (the sentinel group) from N with K successes."""
    tot = math.comb(N, n)
    return {k: math.comb(K, k) * math.comb(N - K, n - k) / tot for k in range(max(0, n - (N - K)), min(K, n) + 1)}


def stratified_exact(strata: Sequence[tuple[int, int, int, int]], tol: float = 1e-9) -> dict[str, float]:
    """Exact test stratified by outer fold. Each stratum = (x_sent, n_sent, x_m0, n_m0); labels are
    permuted within strata, so X = sum x_sent has the convolution of hypergeometric laws. Two-sided
    p = P(|X - E X| >= |x_obs - E X|)."""
    dist = {0: 1.0}
    x_obs = 0
    for xs, ns, xm, nm in strata:
        pmf = _hypergeom_pmf(ns + nm, xs + xm, ns)
        new: dict[int, float] = {}
        for a, pa in dist.items():
            for b, pb in pmf.items():
                new[a + b] = new.get(a + b, 0.0) + pa * pb
        dist = new
        x_obs += xs
    ex = sum(k * p for k, p in dist.items())
    d_obs = abs(x_obs - ex)
    p = sum(pk for k, pk in dist.items() if abs(k - ex) >= d_obs - tol)
    return {"x_obs": int(x_obs), "expected": ex, "p": min(1.0, p), "n_strata": len(strata)}


def run_permutation(m0_runs: Sequence[float], sent_runs: Sequence[float], tol: float = 1e-12) -> dict[str, Any]:
    """Exact run-level permutation test: statistic mean(sentinels) - mean(M0) over every choice of
    which len(m0_runs) runs are labelled M0. Two-sided p = min(1, 2 * smaller one-sided tail), the
    reading under which A06 4.8's "minimum two-sided p is 2 / C(K+2, 2)" holds."""
    allr = list(m0_runs) + list(sent_runs)
    n0, n = len(m0_runs), len(allr)
    obs = float(np.mean(sent_runs) - np.mean(m0_runs))
    tot = ge = le = 0
    s_all = sum(allr)
    for idx in itertools.combinations(range(n), n0):
        s0 = sum(allr[i] for i in idx)
        stat = (s_all - s0) / (n - n0) - s0 / n0
        tot += 1
        ge += stat >= obs - tol
        le += stat <= obs + tol
    p = min(1.0, 2.0 * min(ge, le) / tot)
    return {"stat": obs, "p": p, "p_upper": ge / tot, "p_lower": le / tot, "n_perm": tot,
            "min_p": 2 / tot if tot else float("nan")}


def ca_trend_exact(groups: Sequence[tuple[float, int, int]], tol: float = 1e-9) -> dict[str, Any]:
    """Exact Cochran-Armitage trend test. ``groups`` = (score, n, x). Conditional on the total
    number of successes, T = sum score * x has P proportional to prod C(n_g, x_g). Scores must be
    non-negative integers (session order). Two-sided p = P(|T - E T| >= |t_obs - E T|)."""
    S = sum(x for _, _, x in groups)
    tmax = int(sum(int(s) * n for s, n, _ in groups))
    W = np.zeros((S + 1, tmax + 1))
    W[0, 0] = 1.0
    t_obs = 0
    for s, n, x in groups:
        s = int(s)
        new = np.zeros_like(W)
        for k in range(0, min(n, S) + 1):
            c = math.comb(n, k)
            sh = s * k
            if sh > tmax:
                continue
            new[k:, sh:] += c * W[: S + 1 - k, : tmax + 1 - sh]
        W = new
        t_obs += s * x
    row = W[S]
    tot = row.sum()
    pmf = row / tot
    ts = np.arange(tmax + 1)
    et = float((ts * pmf).sum())
    p = float(pmf[np.abs(ts - et) >= abs(t_obs - et) - tol].sum())
    return {"t_obs": int(t_obs), "expected": et, "p": min(1.0, p), "n_groups": len(groups), "successes": int(S)}


def ffh_mc(table: Sequence[Sequence[int]], draws: int = 100_000, seed: int = 20261101) -> dict[str, Any]:
    """Fisher-Freeman-Halton test for an r x c table by Monte Carlo (fixed margins): p = share of
    permuted tables with probability <= the observed one (plus one, as (b+1)/(B+1))."""
    t = np.asarray(table, int)
    if t.size == 0 or t.sum() == 0:
        return {"p": 1.0, "draws": 0, "note": "empty table"}
    rows, cols = t.sum(1), t.sum(0)
    if (cols > 0).sum() < 2 or (rows > 0).sum() < 2:
        return {"p": 1.0, "draws": 0, "note": "a single non-empty row or column: homogeneous by construction"}

    def logp(tab):
        return float(sum(math.lgamma(v + 1) for v in tab.sum(1)) + sum(math.lgamma(v + 1) for v in tab.sum(0))
                     - math.lgamma(tab.sum() + 1) - sum(math.lgamma(v + 1) for v in tab.ravel()))
    lp_obs = logp(t)
    labels = np.repeat(np.arange(len(cols)), cols)
    bounds = np.cumsum(rows)[:-1]
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(draws):
        perm = rng.permutation(labels)
        tab = np.array([np.bincount(g, minlength=len(cols)) for g in np.split(perm, bounds)])
        hits += logp(tab) <= lp_obs + 1e-9
    return {"p": (hits + 1) / (draws + 1), "draws": draws, "seed": seed}


def final_tests(eps: pd.DataFrame, m0_rows: Sequence[str], sent_rows: pd.DataFrame,
                q4_rows: Optional[pd.DataFrame] = None, draws: int = 100_000) -> dict[str, Any]:
    """All A06 4.8 tests. ``eps``: episodes (row_id, fold, rec_sel; ITT zeros already present);
    ``sent_rows``: O120 sentinel rows (row_id, session, start_epoch) in any order."""
    e = eps.copy()
    e["rec_sel"] = e["rec_sel"].astype(int)
    m0 = e[e.row_id.isin(m0_rows)]
    sr = sent_rows.sort_values(["start_epoch", "row_id"])
    s = e[e.row_id.isin(sr.row_id)]
    missing = sorted(set(sr.row_id) - set(s.row_id))
    out: dict[str, Any] = {"m0_rows": list(m0_rows), "sentinel_rows": sr.row_id.tolist(),
                           "sentinel_rows_without_episodes": missing}
    out["primary_fisher"] = fisher_two_sided(int(s.rec_sel.sum()), int(len(s)), int(m0.rec_sel.sum()), int(len(m0)))
    strata = []
    for f in range(N_FOLDS):
        a, b = s[s.fold == f], m0[m0.fold == f]
        strata.append((int(a.rec_sel.sum()), int(len(a)), int(b.rec_sel.sum()), int(len(b))))
    out["sens_stratified_by_fold"] = stratified_exact(strata)
    m0_run = m0.groupby("row_id").rec_sel.mean().tolist()
    s_run = s.groupby("row_id").rec_sel.mean().tolist()
    out["sens_run_permutation"] = run_permutation(m0_run, s_run) if len(s_run) else None
    # trend over session order: M0 = 0; sentinel sessions numbered 1.. in order of first sentinel start
    order = list(dict.fromkeys(sr.session.tolist()))
    groups = [(0, int(len(m0)), int(m0.rec_sel.sum()))]
    for i, sid in enumerate(order, start=1):
        rids = sr[sr.session == sid].row_id
        g = s[s.row_id.isin(rids)]
        groups.append((i, int(len(g)), int(g.rec_sel.sum())))
    out["desc_trend_ca_exact"] = {**ca_trend_exact(groups), "session_order": order}
    if q4_rows is not None and len(q4_rows):
        q = e[e.row_id.isin(q4_rows.row_id)].merge(q4_rows[["row_id", "session"]], on="row_id")
        tab = [[int((g.rec_sel == 0).sum()), int((g.rec_sel == 1).sum())] for _, g in q.groupby("session")]
        fs = q4_rows.dropna(subset=["func_success"]) if "func_success" in q4_rows else q4_rows.iloc[0:0]
        tab_f = [[int((g.func_success == 0).sum()), int((g.func_success == 1).sum())] for _, g in fs.groupby("session")]
        out["desc_q4_homogeneity"] = {"rec_sel": ffh_mc(tab, draws), "rec_sel_table": tab,
                                      "func_success": ffh_mc(tab_f, draws), "func_success_table": tab_f,
                                      "rec_sel_by_row": q.groupby("row_id").rec_sel.sum().astype(int).to_dict()}
    p = out["primary_fisher"]["p"]
    out["primary_rejects"] = bool(p < 0.05)
    out["period_sensitivity"] = ("REQUIRED (A06 6): refit each confirmatory GLMM with the sentinels as Full "
                                 "observations, rec_sel ~ arm + period + (1|task) + (1|task:seed); not run by this "
                                 "script") if p < 0.05 else "not triggered (primary p >= 0.05)"
    return out


# =========================================================================== I/O
def _read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def load_flag_registry(path: Optional[str]) -> list[str]:
    if not path or not Path(path).exists():
        return []
    spec = importlib.util.spec_from_file_location("_ro27_flag_registry", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # standard-library only at import time (module docstring)
    return list(getattr(mod, "FLAG_KEYS", ()))


def status_rows(run_roots: Sequence[Path]) -> pd.DataFrame:
    frames = []
    for r in run_roots:
        p = Path(r) / "status.tsv"
        if p.exists():
            frames.append(pd.read_csv(p, sep="\t", dtype=str).assign(session=Path(r).name, root=str(r)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def output_dir(run_dir: Path) -> Optional[Path]:
    hits = sorted(run_dir.glob("tasks/*/*/output"))
    return hits[0] if hits else None


def inner_cv_ok(out: Path) -> Optional[bool]:
    """Hard indicator 4: the inner-CV column exists and is not all-NaN among completed trials."""
    p = out / "df_test_pool_final.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    cols = [c for c in ("inner_cv_accuracy", "inner_cv_RMSE") if c in df.columns]
    if not cols:
        return False
    comp = df[df["status"].astype(str) == "completed"] if "status" in df.columns else df
    return bool(any(pd.to_numeric(comp[c], errors="coerce").notna().any() for c in cols))


def held_out(out: Path, task_type: str) -> Optional[float]:
    p = out / "fold_aggregation_summary.csv"
    if not p.exists():
        return None
    s = pd.read_csv(p)
    hit = s.loc[s["metric"] == ("accuracy" if task_type == "classification" else "n_RMSE"), "mean"]
    return float(hit.iloc[0]) if len(hit) else None


def collect_row(st: pd.DataFrame, row_id: str, task_type: str, s_task: float,
                job_log_dir: Optional[Path] = None) -> dict[str, Any]:
    """Status, env, summary metrics and checks of one row from its session folder(s)."""
    s = st[st.row_id == row_id].copy()
    rec: dict[str, Any] = {"row_id": row_id}
    if not len(s):
        rec.update({"final_status": "MISSING", "verdict_basis": "pending"})
        return rec
    s["attempt"] = s["attempt"].astype(int)
    s = s.sort_values(["session", "attempt"])
    final = s.iloc[-1]
    root = Path(final["root"])
    rdir = root / row_id
    rec.update({"session": final["session"], "n_attempts": int(len(s)),
                "attempt_statuses": ";".join(s["status"].astype(str)), "final_status": str(final["status"]),
                "cap_hit": bool((s["status"].astype(str) == "CAP_HIT").any()),
                "start_epoch": int(float(s.iloc[0]["start_epoch"])) if str(s.iloc[0].get("start_epoch")) not in ("", "nan") else None,
                "status_wall_s": float(final["wall_s"]) if str(final.get("wall_s")) not in ("", "nan") else None})
    env = parse_env((rdir / "env.list").read_text(encoding="utf-8", errors="replace")) if (rdir / "env.list").exists() else {}
    rec["env"] = env
    rec["image_tag"] = env.get("ROGUE_ONE_IMAGE_TAG")
    rec["vllm_image_id"] = env.get("ROGUE_ONE_VLLM_IMAGE_ID")
    rec["server_topology"] = env.get("ROGUE_ONE_SERVER_TOPOLOGY")
    out = output_dir(rdir) if rdir.exists() else None
    summ = (_read_json(out / "instrumentation_summary.json") if out else None) or {}
    rec["has_summary"] = bool(summ)
    rec["serper_ok_false_rate"] = summ.get("serper_ok_false_rate")
    ho = held_out(out, task_type) if (out and rec["final_status"] == "OK") else None
    rec["heldout_score"] = ho
    rec["values"] = metric_values(summ, ho, s_task) if summ else {}
    rec["inner_cv_ok"] = inner_cv_ok(out) if (out and rec["final_status"] == "OK") else None
    log = rdir / "logs" / "container.log"
    rec["inner_cv_fallback_lines"] = (log.read_text(encoding="utf-8", errors="replace").count(INNER_CV_FALLBACK_TEXT)
                                      if log.exists() else None)
    rec["out_dir"] = str(out) if out else None
    fs = rec["final_status"]
    if fs in ("OK", "FAILED", "CAP_HIT"):
        session_done = True
        if fs == "FAILED" and job_log_dir is not None:  # a solo retry may still follow
            jl = Path(job_log_dir) / f"ro27_{final['session']}.job.log"
            session_done = (not jl.exists()) or (f"RO27_SESSION_END {final['session']} " in
                                                  jl.read_text(encoding="utf-8", errors="replace"))
        rec["verdict_basis"] = "evaluated" if session_done else "pending"
    else:
        rec["verdict_basis"] = "pending"
    return rec


def load_manifests(paths: Sequence[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p, dtype=str)
        except Exception:  # noqa: BLE001
            continue
        if "row_id" in df.columns:
            frames.append(df)
    return pd.concat(frames, ignore_index=True).drop_duplicates("row_id", keep="last") if frames else pd.DataFrame()


def run_session(a: argparse.Namespace) -> int:  # pragma: no cover - I/O driver; logic unit-tested above
    C = json.loads(Path(a.g2_constants).read_text(encoding="utf-8"))
    g2_band = C["A06_band"]
    task = C.get("A06_sentinel", {}).get("task", SENTINEL_TASK)
    s_task = float(C["A06_s_task"][task])
    o_bands = bands_from_g2(g2_band)
    man = load_manifests(a.manifest)
    st = status_rows([Path(r) for r in a.runs])
    defaults = parse_env(Path(a.defaults).read_text(encoding="utf-8")) if a.defaults else {}
    registry = load_flag_registry(a.flag_registry)
    job_logs = Path(a.job_logs) if a.job_logs else None

    sent = man[man.get("is_sentinel", pd.Series(dtype=str)).astype(str) == "1"] if len(man) else man
    sent = sent[sent.row_id.isin(set(st.row_id))] if len(st) else sent.iloc[0:0]
    ref_o = {r: collect_row(st, r, "regression", s_task) for r in a.m0_rows}
    q4_k1 = man[(man.arm_id == "sentinel_Q4") & (man.replicate.astype(str) == "1")].row_id.tolist() if len(man) else []
    ref_q = {r: collect_row(st, r, "regression", s_task) for r in [a.q4_anchor_row] + q4_k1[:1]}
    anchor = ref_q.get(a.q4_anchor_row, {})
    k1 = ref_q.get(q4_k1[0], {}) if q4_k1 else {}
    q4_ref_capped = bool(anchor.get("cap_hit") or k1.get("cap_hit"))
    qb = q4_bands(g2_band, anchor.get("values") or {}, k1.get("values") or {})
    wb_flags = fetch_wandb_flags(a) if a.wandb else {}

    rows = []
    for m in sent.itertuples():
        is_q4 = m.arm_id == "sentinel_Q4"
        rec = collect_row(st, m.row_id, m.task_type, s_task, job_logs)
        rec.update({"arm_id": m.arm_id, "dataset_id": m.dataset_id, "replicate": int(m.replicate)})
        ref = anchor if is_q4 else ref_o.get(a.m0_rows[0], {})
        rec["reference_row"] = a.q4_anchor_row if is_q4 else a.m0_rows[0]
        rec["ref_vllm_image_id"] = ref.get("vllm_image_id")
        renv = ref.get("env") or {}
        rec["flag_diffs"] = flag_diff(renv, rec.get("env") or {}, defaults, registry) if renv and rec.get("env") else []
        if not renv:
            rec.setdefault("notes_pre", []).append("reference env.list not found: flags not compared")
        if a.wandb and m.row_id in wb_flags and rec["reference_row"] in wb_flags:
            wd = flag_diff(wb_flags[rec["reference_row"]], wb_flags[m.row_id], defaults, registry)
            rec["flag_diffs"] += [dict(d, source="wandb") for d in wd]
        bands = qb if is_q4 else o_bands
        if is_q4 and m.row_id in q4_k1[:1]:
            rec["verdict_basis"] = "reference" if rec.get("verdict_basis") == "evaluated" else rec.get("verdict_basis")
        if is_q4 and rec.get("verdict_basis") == "evaluated" and not qb:
            rec["verdict_basis"] = "not_evaluable"
            rec.setdefault("notes_pre", []).append(
                "Q4 reference has no metrics (anchor/k1 missing or capped): Q4 channel not evaluable (Amendment 3)")
        score_row(rec, bands, is_q4=is_q4, q4_ref_capped=q4_ref_capped)
        rec["notes"] = rec.get("notes_pre", []) + rec["notes"]
        rows.append(rec)
    finalise_verdicts(rows)

    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    flat = []
    for r in rows:
        f = {k: r.get(k) for k in ("row_id", "arm_id", "session", "image_tag", "server_topology", "replicate",
                                    "start_epoch", "attempt_statuses", "final_status", "cap_hit", "verdict",
                                    "heldout_score", "inner_cv_ok", "inner_cv_fallback_lines", "serper_ok_false_rate")}
        for _, mname, _ in CHANNEL_METRICS:
            f[f"v_{mname}"] = (r.get("values") or {}).get(mname)
        f["channels_out"] = ";".join(r.get("channels_out", []))
        f["channels_not_evaluable"] = ";".join(r.get("channels_not_evaluable", []))
        f["repeated_channels"] = ";".join(r.get("repeated_channels", []))
        f["hard"] = " | ".join(r.get("hard", []))
        f["notes"] = " | ".join(r.get("notes", []))
        flat.append(f)
    df = pd.DataFrame(flat)
    df.to_csv(od / "sentinel_rows.csv", index=False)
    sel = [r for r in rows if not a.session or r.get("session") == a.session]
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d")
    report = {"name": f"ANALYSIS__sentinel_drift__{stamp}", "session_filter": a.session, "sentinel_task": task,
              "s_task": s_task, "bands_o120": o_bands, "bands_q4": qb, "q4_reference": {
                  "anchor_row": a.q4_anchor_row, "anchor_status": anchor.get("final_status"),
                  "k1_row": q4_k1[0] if q4_k1 else None, "k1_status": k1.get("final_status"),
                  "any_capped": q4_ref_capped},
              "m0_reference": {r: {"status": v.get("final_status"), "values": v.get("values")} for r, v in ref_o.items()},
              "verdict_counts": df.groupby(["arm_id", "verdict"]).size().reset_index().values.tolist() if len(df) else [],
              "rows": [{k: v for k, v in r.items() if k not in ("env", "notes_pre")} for r in sel]}
    (od / ("sentinel_session_" + (a.session or "all") + ".json")).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(df[["row_id", "arm_id", "session", "image_tag", "final_status", "verdict", "channels_out", "hard"]].to_string(index=False))
    return 0


def fetch_wandb_flags(a: argparse.Namespace) -> dict[str, dict[str, str]]:  # pragma: no cover - network
    try:
        import wandb  # noqa: PLC0415
        api = wandb.Api(timeout=180)
        out = {}
        for grp in ("A06", "A05", "A17"):
            for r in api.runs(f"{a.entity}/{a.project}", filters={"group": grp}):
                cfg = dict(r.config)
                rid = str(cfg.get("manifest_row") or "")
                if rid:
                    out[rid] = {k[len("flags/"):]: str(v) for k, v in cfg.items() if str(k).startswith("flags/")}
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"[sentinel_drift] W&B flags not fetched: {type(exc).__name__}: {exc}", file=sys.stderr)
        return {}


def run_final(a: argparse.Namespace) -> int:  # pragma: no cover - I/O driver
    rows = pd.read_csv(a.rows)
    eps = pd.concat([pd.read_csv(p) for p in a.episodes], ignore_index=True).drop_duplicates(["row_id", "fold"], keep="last")
    done = rows[rows.final_status.isin(["OK", "FAILED", "CAP_HIT"])]
    o120 = done[done.arm_id == "sentinel"][["row_id", "session", "start_epoch"]]
    q4 = done[done.arm_id == "sentinel_Q4"][["row_id", "session", "start_epoch"]].copy()
    if len(q4):
        # func_success: a capped/failed run is 0 (ITT, A17 4); an OK Q4 run needs the A17 fold rule,
        # which this script does not evaluate -> NaN (reported as missing)
        q4["func_success"] = [0 if s != "OK" else np.nan for s in done[done.arm_id == "sentinel_Q4"].final_status]
    res = final_tests(eps, a.m0_rows, o120, q4, a.draws)
    res["label"] = ("INTERIM (A06-T16, G4): operational only, not a reported test" if a.interim
                    else "FINAL (A06-T17)")
    res["verdicts"] = rows.groupby(["arm_id", "verdict"]).size().reset_index().values.tolist()
    res["rec_sel_by_sentinel"] = eps[eps.row_id.isin(o120.row_id)].groupby("row_id").rec_sel.sum().astype(int).to_dict()
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d")
    name = f"ANALYSIS__sentinel_drift__{stamp}" + ("__interim" if a.interim else "__final")
    (od / f"{name}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("label", "primary_fisher", "sens_stratified_by_fold", "sens_run_permutation",
                                          "desc_trend_ca_exact", "primary_rejects")}, indent=1, default=str))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("session", help="A06 4.7 per-session verdicts (A06-T8)")
    s.add_argument("--runs", nargs="+", required=True, help="session run folders (each with status.tsv)")
    s.add_argument("--manifest", nargs="+", required=True)
    s.add_argument("--g2-constants", required=True)
    s.add_argument("--defaults", help="scripts/agent2027/v2_defaults.env (Full values of post-v2.0 flags)")
    s.add_argument("--flag-registry", help="rogueone/utils/flag_registry.py (R3/R4 keys)")
    s.add_argument("--session", help="report only this session (all sentinels are still scored for the consecutive rule)")
    s.add_argument("--m0-rows", nargs=2, default=["A05-032", "A05-048"], help="M0 r0 r1 on the sentinel task")
    s.add_argument("--q4-anchor-row", default="A17-031")
    s.add_argument("--job-logs", help="directory with ro27_<SID>.job.log (pending solo retries)")
    s.add_argument("--wandb", action="store_true", help="also compare W&B flags/* (effective values)")
    s.add_argument("--entity", default="PlanBench")
    s.add_argument("--project", default="RogueOne-AGENT2027")
    s.add_argument("--out-dir", required=True)
    f = sub.add_parser("final", help="A06 4.8 pooled tests (A06-T10/T17; --interim = A06-T16)")
    f.add_argument("--rows", required=True, help="sentinel_rows.csv from the session step")
    f.add_argument("--episodes", nargs="+", required=True, help="recovery_rows.py CSV(s) with M0 and sentinel rows")
    f.add_argument("--m0-rows", nargs=2, default=["A05-032", "A05-048"])
    f.add_argument("--interim", action="store_true")
    f.add_argument("--draws", type=int, default=100_000)
    f.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    return run_session(a) if a.cmd == "session" else run_final(a)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
