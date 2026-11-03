"""A18 (RP1) A18-T18: per-role floors and the RP1-D1…D5 verdicts from ``rp1_score.py score`` output.

Pre-registration: Part V.18 (RP1-D1 floor, floor-stability threshold 0.6, D3b minimum denominator 30;
"D2–D6 as A18 §6"); ``docs/ablations/A18_RP1_rolewise_replay.md`` §2 *Denominators*, §4 *Failure
handling* (sensitivity analyses), §5.4 A18-T18 and §6 (RP1-D1…D6). Exploratory, outside Holm (R14).

Input: ``rp1_scored.csv`` (one row per (target, prompt), incl. the pseudo-target ``recorded_O120``;
written by ``rp1_score.py score``). Optional second input: the same table scored with
``rp1_score.py score --decode-json-string-args --out-subdir scored_315`` (HQ #315 sensitivity:
JSON-string object arguments decoded before the required-key check); it is evaluated with the
primary population rules and reported side by side with the primary verdicts.

What is computed (A18-T18 bullets, in order):

* **Rates.** Per variant × unit × target × score; denominators as A18 §2: every common-set turn of
  the role (stage) for ``rp1_valid_call_raw``/``_rescued``/``rp1_usable`` (rows whose score is
  empty, e.g. ``recorded_O120`` raw or ``--no-exec`` usable, are left out, as in
  ``rp1_score.rate_table``); ``rp1_executes`` only over rows where it is set and not
  ``auto_unprune``. Role rates pool the role's strata unweighted (T pools assessment and pruning;
  the two stages are also reported as units ``T/assessment``, ``T/pruning``). Wilson 95% CIs.
* **Ratio to O120** (``replay_O120``, target ``O120``) with a cluster bootstrap over the source
  fold-episodes (column ``fold_episode``; 2,000 resamples, seed 42; one draw of clusters shared by
  every target, unit and score, so the comparison stays paired) and percentile 95% CIs.
* **RP1-D1 floor.** A target passes when rate ≥ 0.8 × O120's rate (same unit, score, population;
  point estimates). Floor = the smallest rung r on the measured Qwen ladder such that r and every
  larger rung pass; coded by position (``≤ first`` = 1, …, ``> last`` = n+1). Floor-stability
  share = share of bootstrap resamples whose floor equals the point floor. Off-ladder targets (O20)
  are reported pass/fail. Primary score ``rp1_usable``; transport floor on ``rp1_valid_call_rescued``.
* **Exact McNemar** between adjacent ladder rungs, paired by prompt (exploratory, unadjusted).
* **T Jaccards** (prune list vs O120's recorded list and vs J−a), mean per target.
* **Sensitivities** (A18 §4): ``no_common_set`` (each target's own context errors count as invalid
  moves; only its own infra/missing rows are dropped), ``drop_clipped`` and
  ``drop_suspicious_partial``; plus ``hq315`` from the second input.
* **Verdicts** RP1-D2 (order; "uninformative (unstable floors)" when any of S/E/T's stability
  share < 0.6), D3a (transport- vs code/planning-limited at failing rungs), D3b (E transport
  bottleneck, per family), D4 (A19 1B-cell prediction), D5(a) (O120 self-replay rescued ≥ 0.9 per
  role, re-checked on the scored rows) and D5(b) (``recorded_O120`` vs O120 > 10 pp), D6 (transport-
  only fallback when no usable scores exist; κ < 0.6 substitutes S's usable by the joint D1 pass
  of rescued and executes).

Containers only on CAIR, offline (A18-T18: same form as ``score``)::

    docker run --rm --network none --cpus 4 -v $HOME/ro27_a18_work:/work \\
      -v <scripts/agent2027>:/workspace/scripts/agent2027:ro --entrypoint python3 \\
      rogue_one:agent2027-v2.<N> /workspace/scripts/agent2027/analysis/rp1_verdicts.py \\
        --scored /work/scored/rp1_scored.csv --scored-315 /work/scored_315/rp1_scored.csv \\
        --out /work/floors
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

LADDER = ("Q0.8", "Q2", "Q4", "Q9", "Q122")  # A18 §6 RP1-D1 (Qwen ladder, small to large)
REFERENCE = "O120"  # replay_O120, the self-replay control
RECORDED = "recorded_O120"  # pseudo-target: the logged M0 responses
SCORES = ("rp1_valid_call_raw", "rp1_valid_call_rescued", "rp1_executes", "rp1_usable")
PRIMARY = "rp1_usable"
TRANSPORT = "rp1_valid_call_rescued"
ORDER_ROLES = ("S", "E", "T")  # D2: S, E and T enter the order verdict; W is descriptive
UNITS = ("S", "E", "T", "W", "T/assessment", "T/pruning")
PASS_FRACTION = 0.8  # D1
STABILITY_MIN = 0.6  # D2
D3B_MIN_DEN = 30  # D3b
D3B_RUNGS = ("Q0.8", "Q2", "Q4")
D5A_MIN = 0.9  # D5(a)
D5B_MAX_PP = 0.10  # D5(b)
KAPPA_MIN = 0.6  # D6
N_BOOT = 2000
SEED = 42
Z95 = 1.959963984540054
VARIANTS = ("primary", "no_common_set", "drop_clipped", "drop_suspicious_partial")


# =========================================================================== input
def _bool(v: Any) -> Optional[bool]:
    """CSV cell -> bool / None (``rp1_score`` writes True/False and '' for None)."""
    if isinstance(v, bool) or v is None:
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def read_scored(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in SCORES + ("common_set", "context_error", "infra_error", "timeout", "clipped",
                           "auto_unprune", "suspicious"):
            r[k] = _bool(r.get(k))
    return rows


def units_of(row: dict) -> list[str]:
    out = [row["role"]]
    if row["role"] == "T" and row.get("stage") in ("assessment", "pruning"):
        out.append(f"T/{row['stage']}")
    return out


def in_population(row: dict, variant: str) -> bool:
    """A18 §4 populations. ``primary``/``hq315``: the common set."""
    if variant == "no_common_set":
        return not (row.get("infra_error") or row.get("error_kind") == "missing")
    if not row.get("common_set"):
        return False
    if variant == "drop_clipped":
        return not row.get("clipped")
    if variant == "drop_suspicious_partial":
        return not (row.get("suspicious") or "reconstruction_partial" in str(row.get("flags") or ""))
    return True


def score_value(row: dict, score: str) -> Optional[bool]:
    """The row's score, or None when the row is outside that score's denominator (A18 §2)."""
    v = row.get(score)
    if v is None or (score == "rp1_executes" and row.get("auto_unprune")):
        return None
    return bool(v)


# =========================================================================== statistics
def wilson(k: int, n: int, z: float = Z95) -> tuple[Optional[float], Optional[float]]:
    if n <= 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p (binomial on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def d1_pass(rate: Any, ref: Any) -> Any:
    """RP1-D1 on arrays or scalars: 1.0 pass, 0.0 fail, nan if either rate is undefined."""
    rate, ref = np.asarray(rate, dtype=float), np.asarray(ref, dtype=float)
    with np.errstate(invalid="ignore"):
        out = np.where(rate >= PASS_FRACTION * ref, 1.0, 0.0)
    return np.where(np.isnan(rate) | np.isnan(ref), np.nan, out)


def floor_positions(passes: np.ndarray) -> np.ndarray:
    """Floor positions from a (..., n_rungs) pass matrix (1/0/nan) over the ladder, small to large.

    Position = (index of the smallest rung with it and every larger rung passing) + 1, so the
    smallest rung ("≤ first") is 1 and "no rung passes / largest fails" is n+1. nan if any
    rung is undefined.
    """
    p = np.asarray(passes, dtype=float)
    n = p.shape[-1]
    suffix_all = np.flip(np.cumprod(np.flip(np.nan_to_num(p, nan=0.0), -1), -1), -1)
    pos = n - suffix_all.sum(-1) + 1
    return np.where(np.isnan(p).any(-1), np.nan, pos)


def floor_label(pos: Optional[float], ladder: tuple) -> Optional[str]:
    if pos is None or (isinstance(pos, float) and math.isnan(pos)):
        return None
    pos = int(pos)
    if pos == 1:
        return f"<= {ladder[0]}"
    if pos == len(ladder) + 1:
        return f"> {ladder[-1]}"
    return ladder[pos - 1]


def _num(x: Any) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else x


# =========================================================================== core
class Counts:
    """Per-cluster numerators/denominators for every (unit, target, score) of one population."""

    def __init__(self, rows: list[dict], variant: str, clusters: list[str]):
        self.cidx = {c: i for i, c in enumerate(clusters)}
        k = len(clusters)
        self.num: dict[tuple, np.ndarray] = {}
        self.den: dict[tuple, np.ndarray] = {}
        for r in rows:
            if not in_population(r, variant):
                continue
            ci = self.cidx[r["fold_episode"]]
            for u in units_of(r):
                for s in SCORES:
                    v = score_value(r, s)
                    if v is None:
                        continue
                    key = (u, r["target"], s)
                    if key not in self.den:
                        self.num[key] = np.zeros(k)
                        self.den[key] = np.zeros(k)
                    self.den[key][ci] += 1
                    self.num[key][ci] += v

    def point(self, key: tuple) -> tuple[int, int]:
        if key not in self.den:
            return 0, 0
        return int(self.num[key].sum()), int(self.den[key].sum())

    def rate(self, key: tuple, weights: Optional[np.ndarray] = None) -> Any:
        """Point rate (weights None) or bootstrap rates (weights (B, K)); nan where den = 0."""
        if key not in self.den:
            return np.nan if weights is None else np.full(weights.shape[0], np.nan)
        if weights is None:
            n, d = self.num[key].sum(), self.den[key].sum()
        else:
            n, d = weights @ self.num[key], weights @ self.den[key]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(d > 0, n / np.where(d > 0, d, 1), np.nan)


def bootstrap_weights(n_clusters: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """(B, K) multiplicities of a cluster bootstrap (K clusters drawn with replacement)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    w = np.zeros((n_boot, n_clusters))
    np.add.at(w, (np.repeat(np.arange(n_boot), n_clusters), idx.ravel()), 1.0)
    return w


def ladder_of(rows: list[dict], ladder: Iterable[str] = LADDER) -> tuple:
    """The measured Qwen ladder (A18 §6 D1, third bullet: only measured rungs)."""
    present = {r["target"] for r in rows}
    return tuple(t for t in ladder if t in present)


def evaluate(rows: list[dict], variant: str, *, clusters: list[str], ladder: tuple,
             weights: Optional[np.ndarray], reference: str = REFERENCE, kappa: Optional[float] = None,
             population: Optional[str] = None) -> dict[str, Any]:
    """Rates, floors and the D1–D6 verdicts for one variant (population ``population or variant``)."""
    pop = population or variant
    cnt = Counts(rows, pop, clusters)
    targets = sorted({r["target"] for r in rows})
    off_ladder = [t for t in targets if t not in ladder and t not in (reference, RECORDED)]

    rates: list[dict] = []
    pass_point: dict[tuple, float] = {}
    pass_boot: dict[tuple, np.ndarray] = {}
    for u in UNITS:
        for s in SCORES:
            ref_key = (u, reference, s)
            ref_pt = cnt.rate(ref_key)
            ref_b = cnt.rate(ref_key, weights) if weights is not None else None
            for t in targets:
                key = (u, t, s)
                k, n = cnt.point(key)
                if n == 0 and t != reference:
                    continue
                lo, hi = wilson(k, n)
                pt = cnt.rate(key)
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio = pt / ref_pt if ref_pt and not np.isnan(ref_pt) else np.nan
                row = {"variant": variant, "unit": u, "target": t, "score": s, "num": k, "den": n,
                       "rate": _num(pt), "wilson_lo": lo, "wilson_hi": hi, "ratio_to_ref": _num(ratio),
                       "ratio_lo": None, "ratio_hi": None, "d1_pass": None}
                if t not in (reference,):
                    pp = float(d1_pass(pt, ref_pt))
                    row["d1_pass"] = None if math.isnan(pp) else bool(pp)
                    pass_point[key] = pp
                if weights is not None and t != reference:
                    tb = cnt.rate(key, weights)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        rb = np.where(ref_b > 0, tb / np.where(ref_b > 0, ref_b, 1), np.nan)
                    ok = rb[~np.isnan(rb)]
                    if ok.size:
                        row["ratio_lo"], row["ratio_hi"] = (float(np.percentile(ok, 2.5)),
                                                            float(np.percentile(ok, 97.5)))
                    pass_boot[key] = d1_pass(tb, ref_b)
                rates.append(row)

    def pass_vec(u: str, s: str) -> tuple[np.ndarray, Optional[np.ndarray]]:
        pt = np.array([pass_point.get((u, t, s), np.nan) for t in ladder])
        if weights is None:
            return pt, None
        b = np.stack([pass_boot.get((u, t, s), np.full(weights.shape[0], np.nan)) for t in ladder], -1)
        return pt, b

    # D6 (κ rule): S's usable replaced by the joint D1 pass of rescued and executes
    d6_kappa = kappa is not None and kappa < KAPPA_MIN
    usable_any = any(r.get(PRIMARY) is not None for r in rows if r["target"] != RECORDED)
    floors: list[dict] = []
    floor_pos: dict[tuple, Optional[float]] = {}
    stab: dict[tuple, Optional[float]] = {}
    score_set = list(SCORES) + (["S_joint_rescued_executes"] if d6_kappa else [])
    for u in UNITS:
        for s in score_set:
            if s == "S_joint_rescued_executes":
                if u != "S":
                    continue
                a_pt, a_b = pass_vec(u, TRANSPORT)
                e_pt, e_b = pass_vec(u, "rp1_executes")
                pt, b = np.minimum(a_pt, e_pt), (None if a_b is None else np.minimum(a_b, e_b))
            else:
                pt, b = pass_vec(u, s)
            pos = _num(floor_positions(pt)) if len(ladder) else None
            share = None
            if pos is not None and b is not None:
                share = float(np.mean(floor_positions(b) == pos))
            floor_pos[(u, s)] = pos
            stab[(u, s)] = share
            floors.append({"variant": variant, "unit": u, "score": s,
                           "floor": floor_label(pos, ladder), "position": pos,
                           "stability_share": share, "ladder": ",".join(ladder),
                           "passes": ";".join(f"{t}={'' if math.isnan(p) else int(p)}"
                                              for t, p in zip(ladder, pt))})

    def primary_key(role: str) -> tuple:
        return (role, "S_joint_rescued_executes") if (role == "S" and d6_kappa) else (role, PRIMARY)

    verdicts: dict[str, Any] = {"variant": variant, "ladder": list(ladder),
                                "d6_transport_only": not usable_any, "d6_kappa_substitute_S": d6_kappa}

    # ---------------------------------------------------------------- D2
    if not usable_any:
        verdicts["D2"] = "not evaluated (RP1-D6: no usable scores; transport floors only)"
        verdicts["D2_point"] = None
    else:
        fs = {r: floor_pos[primary_key(r)] for r in ORDER_ROLES}
        point = order_verdict(fs["S"], fs["E"], fs["T"])
        shares = {r: stab[primary_key(r)] for r in ORDER_ROLES}
        unstable = [r for r, v in shares.items() if v is not None and v < STABILITY_MIN]
        verdicts["D2_point"] = point
        verdicts["D2_floor_positions"] = fs
        verdicts["D2_stability"] = shares
        verdicts["D2"] = (f"uninformative (unstable floors) [{point}]" if unstable and point else point)

    # ---------------------------------------------------------------- D3a
    d3a: dict[str, dict[str, str]] = {}
    for role in ORDER_ROLES + ("W",):
        per = {}
        for t in ladder:
            pk = primary_key(role)
            if pk[1] == "S_joint_rescued_executes":
                pu = min(pass_point.get((role, t, TRANSPORT), np.nan),
                         pass_point.get((role, t, "rp1_executes"), np.nan))
            else:
                pu = pass_point.get((role, t, PRIMARY), np.nan)
            pr = pass_point.get((role, t, TRANSPORT), np.nan)
            if math.isnan(pu):
                per[t] = "not evaluable"
            elif pu == 1.0:
                per[t] = "pass"
            elif math.isnan(pr):
                per[t] = "fail (transport not evaluable)"
            elif pr == 0.0:
                per[t] = "transport-limited"
            else:
                per[t] = "code/planning-limited"
        d3a[role] = per
    verdicts["D3a"] = d3a

    # ---------------------------------------------------------------- D3b
    d3b: dict[str, Any] = {}
    qwen = [t for t in D3B_RUNGS if t in ladder]
    d3b["Qwen"] = d3b_verdict(cnt, pass_point, qwen, reference)
    gpt = [t for t in off_ladder if t.startswith("O")]
    d3b["gpt-oss"] = (d3b_verdict(cnt, pass_point, gpt, reference) if gpt else
                      {"verdict": "not evaluable (no gpt-oss target besides the O120 reference; "
                                  "replay_O20 dropped, Part VIII.1 item 15)"})
    verdicts["D3b"] = d3b

    # ---------------------------------------------------------------- D4
    if not usable_any:
        verdicts["D4"] = "not evaluated (RP1-D6)"
    else:
        fs_, ft = floor_pos[primary_key("S")], floor_pos[primary_key("T")]
        if fs_ is None or ft is None:
            verdicts["D4"] = "not evaluable (floor undefined)"
        else:
            verdicts["D4"] = ("1B cell predicted to pass" if fs_ == 1 and ft == 1
                              else "1B cell predicted to fail")

    # ---------------------------------------------------------------- D5
    d5a = {}
    for role in ("S", "E", "T", "W"):
        v = _num(cnt.rate((role, reference, TRANSPORT)))
        d5a[role] = {"rate": v, "ok": None if v is None else v >= D5A_MIN}
    verdicts["D5a"] = {"by_role": d5a, "flag": any(x["ok"] is False for x in d5a.values())}
    d5b, affected = {}, []
    for role in ("S", "E", "T", "W"):
        for s in (TRANSPORT, "rp1_executes", PRIMARY):
            a, b = _num(cnt.rate((role, RECORDED, s))), _num(cnt.rate((role, reference, s)))
            diff = None if a is None or b is None else a - b
            hit = diff is not None and abs(diff) > D5B_MAX_PP + 1e-12
            d5b[f"{role}:{s}"] = {"recorded_O120": a, "O120": b, "diff": diff, "flag": hit}
            if hit and role not in affected:
                affected.append(role)
    verdicts["D5b"] = {"by_role_score": d5b, "affected_roles": affected, "flag": bool(affected)}
    excl = [r for r in affected if r in ORDER_ROLES]
    verdicts["D2_without_affected"] = (verdicts.get("D2") if not excl else
                                       f"not evaluable without role(s) {','.join(excl)} (D5(b))")

    # ---------------------------------------------------------------- off-ladder pass/fail (O20)
    verdicts["off_ladder"] = {t: {f"{u}:{s}": (None if math.isnan(pass_point.get((u, t, s), np.nan))
                                              else bool(pass_point[(u, t, s)]))
                                  for u in ("S", "E", "T", "W") for s in (PRIMARY, TRANSPORT)}
                              for t in off_ladder}
    return {"rates": rates, "floors": floors, "verdicts": verdicts}


def order_verdict(fs: Optional[float], fe: Optional[float], ft: Optional[float]) -> Optional[str]:
    """RP1-D2 on floor positions (A18 §6 table; the four cases are disjoint)."""
    if fs is None or fe is None or ft is None:
        return None
    if fe == fs == ft:
        return "Null"
    if fe >= fs >= ft and fe - ft >= 2:
        return "Hoped"
    if fe < max(fs, ft):
        return "Opposite"
    return "Partial"


def d3b_verdict(cnt: Counts, pass_point: dict, rungs: list[str], reference: str) -> dict[str, Any]:
    """RP1-D3b: at every served rung, E's rescued fails D1 while E's executes passes D1; the
    executes comparison is evaluable only where its denominator is ≥ 30 at the rung and at O120."""
    if not rungs:
        return {"verdict": "not evaluable (no rung served)", "rungs": {}}
    ref_n = cnt.point(("E", reference, "rp1_executes"))[1]
    per, status = {}, []
    for t in rungs:
        tr = pass_point.get(("E", t, TRANSPORT), np.nan)
        n = cnt.point(("E", t, "rp1_executes"))[1]
        ex = pass_point.get(("E", t, "rp1_executes"), np.nan)
        evaluable = n >= D3B_MIN_DEN and ref_n >= D3B_MIN_DEN and not math.isnan(ex)
        per[t] = {"rescued_pass": None if math.isnan(tr) else bool(tr), "executes_den": n,
                  "ref_executes_den": ref_n, "executes_pass": (bool(ex) if evaluable else None)}
        if math.isnan(tr):
            status.append("ne")
        elif tr == 1.0:
            status.append("no")
        elif not evaluable:
            status.append("ne")
        else:
            status.append("yes" if ex == 1.0 else "no")
    verdict = ("no" if "no" in status else "not evaluable" if "ne" in status else "yes")
    return {"verdict": {"yes": "transport-bottlenecked", "no": "not transport-bottlenecked",
                        "not evaluable": "not evaluable"}[verdict], "rungs": per}


def mcnemar_table(rows: list[dict], variant: str, ladder: tuple) -> list[dict]:
    """Exact McNemar between adjacent ladder rungs, paired by prompt (exploratory, unadjusted)."""
    idx: dict[tuple, Optional[bool]] = {}
    for r in rows:
        if in_population(r, variant):
            for u in units_of(r):
                for s in SCORES:
                    idx[(u, r["target"], s, r["prompt_id"])] = score_value(r, s)
    prompts = sorted({r["prompt_id"] for r in rows})
    out = []
    for u in UNITS:
        for s in SCORES:
            for lo, hi in zip(ladder, ladder[1:]):
                b = c = n = 0
                for p in prompts:
                    x, y = idx.get((u, lo, s, p)), idx.get((u, hi, s, p))
                    if x is None or y is None:
                        continue
                    n += 1
                    b += x and not y
                    c += y and not x
                if n:
                    out.append({"variant": variant, "unit": u, "score": s, "lower": lo, "upper": hi,
                                "n_pairs": n, "lower_only": b, "upper_only": c,
                                "p_exact": mcnemar_exact(b, c)})
    return out


def jaccard_table(rows: list[dict], variant: str) -> list[dict]:
    acc: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        if r["role"] != "T" or not in_population(r, variant):
            continue
        for col in ("jaccard_O120", "jaccard_Ja"):
            v = r.get(col)
            if v not in (None, ""):
                acc.setdefault(r["target"], {}).setdefault(col, []).append(float(v))
    out = []
    for t, d in sorted(acc.items()):
        for col, vals in sorted(d.items()):
            out.append({"variant": variant, "target": t, "measure": col, "n": len(vals),
                        "mean": sum(vals) / len(vals)})
    return out


def run_all(primary_rows: list[dict], rows_315: Optional[list[dict]] = None, *,
            n_boot: int = N_BOOT, seed: int = SEED, ladder: Iterable[str] = LADDER,
            reference: str = REFERENCE, kappa: Optional[float] = None) -> dict[str, Any]:
    """Every variant, side by side. One cluster draw (seed) shared by all variants."""
    if not any(r["target"] == reference for r in primary_rows):
        raise ValueError(f"reference target {reference!r} not in the scored rows")
    lad = ladder_of(primary_rows, ladder)
    clusters = sorted({r["fold_episode"] for r in primary_rows}
                      | {r["fold_episode"] for r in (rows_315 or [])})
    weights = bootstrap_weights(len(clusters), n_boot, seed) if n_boot else None
    res: dict[str, Any] = {"meta": {"ladder": list(lad), "reference": reference, "n_boot": n_boot,
                                    "seed": seed, "n_clusters": len(clusters), "kappa": kappa,
                                    "pass_fraction": PASS_FRACTION, "stability_min": STABILITY_MIN,
                                    "d3b_min_den": D3B_MIN_DEN,
                                    "common_set_prompts": len({r["prompt_id"] for r in primary_rows
                                                               if r.get("common_set")})},
                           "variants": {}, "mcnemar": [], "jaccard": []}
    for v in VARIANTS:
        res["variants"][v] = evaluate(primary_rows, v, clusters=clusters, ladder=lad, weights=weights,
                                      reference=reference, kappa=kappa)
    res["mcnemar"] += mcnemar_table(primary_rows, "primary", lad)
    res["jaccard"] += jaccard_table(primary_rows, "primary")
    if rows_315 is not None:
        res["variants"]["hq315"] = evaluate(rows_315, "hq315", clusters=clusters, ladder=lad,
                                            weights=weights, reference=reference, kappa=kappa,
                                            population="primary")
        res["mcnemar"] += [dict(m, variant="hq315") for m in mcnemar_table(rows_315, "primary", lad)]
        res["jaccard"] += [dict(j, variant="hq315") for j in jaccard_table(rows_315, "primary")]
    return res


SIDE_BY_SIDE_KEYS = ("D2", "D2_point", "D2_without_affected", "D4", "D3b.Qwen", "D3b.gpt-oss",
                     "D5a.flag", "D5b.flag", "D5b.affected_roles", "d6_transport_only",
                     "d6_kappa_substitute_S")


def _get(d: dict, dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else None
    if isinstance(cur, dict) and "verdict" in cur:
        cur = cur["verdict"]
    return cur


def side_by_side(res: dict) -> list[dict]:
    """One row per verdict / floor, one column per variant (primary and #315 next to each other)."""
    names = list(res["variants"])
    order = ["primary"] + (["hq315"] if "hq315" in names else []) + [n for n in names
                                                                   if n not in ("primary", "hq315")]
    out = []
    for key in SIDE_BY_SIDE_KEYS:
        out.append({"item": key, **{n: json.dumps(_get(res["variants"][n]["verdicts"], key),
                                                  ensure_ascii=False) for n in order}})
    for u in UNITS:
        for s in (PRIMARY, TRANSPORT):
            row = {"item": f"floor {u} {s}"}
            for n in order:
                f = next((x for x in res["variants"][n]["floors"] if x["unit"] == u and x["score"] == s), None)
                row[n] = (f"{f['floor']} (stab {f['stability_share']:.2f})"
                          if f and f["floor"] is not None and f["stability_share"] is not None
                          else (f["floor"] if f else None))
            out.append(row)
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_outputs(res: dict, out: Path) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / "rp1_floors.json", "rates": out / "rp1_floor_rates.csv",
             "floors": out / "rp1_floors.csv", "mcnemar": out / "rp1_mcnemar.csv",
             "jaccard": out / "rp1_jaccard.csv", "side_by_side": out / "rp1_verdicts.csv"}
    paths["json"].write_text(json.dumps(res, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    _write_csv(paths["rates"], [r for v in res["variants"].values() for r in v["rates"]])
    _write_csv(paths["floors"], [r for v in res["variants"].values() for r in v["floors"]])
    _write_csv(paths["mcnemar"], res["mcnemar"])
    _write_csv(paths["jaccard"], res["jaccard"])
    _write_csv(paths["side_by_side"], side_by_side(res))
    return paths


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scored", required=True, help="rp1_scored.csv (primary scoring)")
    ap.add_argument("--scored-315", default=None,
                    help="rp1_scored.csv from `score --decode-json-string-args` (HQ #315 sensitivity)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--ladder", default=",".join(LADDER), help="Qwen ladder, small to large")
    ap.add_argument("--reference", default=REFERENCE)
    ap.add_argument("--kappa", type=float, default=None,
                    help="A18-T17 κ(rule vs consensus); < 0.6 applies RP1-D6's S substitute")
    args = ap.parse_args(argv)
    rows = read_scored(Path(args.scored))
    rows_315 = read_scored(Path(args.scored_315)) if args.scored_315 else None
    res = run_all(rows, rows_315, n_boot=args.n_boot, seed=args.seed,
                  ladder=tuple(x for x in args.ladder.split(",") if x), reference=args.reference,
                  kappa=args.kappa)
    paths = write_outputs(res, Path(args.out))
    for r in side_by_side(res):
        print(" | ".join(f"{k}={v}" for k, v in r.items()))
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
