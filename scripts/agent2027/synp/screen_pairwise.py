"""CPU pairwise-operation screen for SYN-P (E8a).

Gate (01 E8a): *no unary or pairwise operation of the raw columns may reach
|rho| >= 0.95 (NMI >= 0.9) with phi* on **P6a and P7**.  The candidate set is

* unary:    identity, log, sqrt, square, reciprocal (where defined) of each column
            (categoricals enter as ordinal codes, identity only);
* pairwise: f(x_i) op g(x_j) for f, g in the unary set and op in
            {+, -, *, /, reversed /} over every pair of numeric columns.

(The unary-then-pairwise composition is what catches a BMI-like w / h^2, which is
nearly rank-identical to w / h.)  |rho| and NMI come from the *detector's own*
criterion functions, so the screen and the detector can never disagree.

P7 extra checks (``screen_p7_extras``): the ratio component alone must not meet the
detector criterion, and no ``ratio * 1[c >= q-quantile]`` guess on a grid of round
quantiles and round numbers may meet it either (the threshold constant must be
unguessable).

Informational runs on the other tasks use the same code (``gate=False``).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.agent2027.analysis.recovery_detector import (
    FAMILY_RULE,
    NMI_THRESHOLD,
    SPEARMAN_THRESHOLD,
    meets_criterion,
    normalized_mi,
    spearman_abs,
)
from scripts.agent2027.synp import tasks as T

GATED_KEYS = ("p6a", "p7")
_OPS = ("+", "-", "*", "/", "r/")


def _unary_variants(x: np.ndarray) -> dict[str, np.ndarray]:
    out = {"id": x, "sq": x * x}
    if np.all(x > 0):
        out["log"] = np.log(x)
    if np.all(x >= 0):
        out["sqrt"] = np.sqrt(x)
    if np.all(x != 0):
        out["inv"] = 1.0 / x
    return out


def _binary(op: str, a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    with np.errstate(all="ignore"):
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return a / b if np.all(b != 0) else None
        return b / a if np.all(a != 0) else None


def screen_task(
    df: pd.DataFrame, phi: np.ndarray, family: str = "spearman", *, top_k: int = 5,
    spearman_threshold: float = SPEARMAN_THRESHOLD, nmi_threshold: float = NMI_THRESHOLD,
) -> dict[str, Any]:
    """Screen every unary / pairwise op of the raw columns against ``phi``."""
    feats = df.drop(columns=[c for c in ("id",) if c in df.columns])
    num = {c: feats[c].to_numpy(dtype=float) for c in feats.columns if pd.api.types.is_numeric_dtype(feats[c])}
    cat = [c for c in feats.columns if c not in num]
    phi = np.asarray(phi, dtype=float)
    use_nmi = family == FAMILY_RULE
    best: list[tuple[float, float, str]] = []  # (rho, nmi, description)
    n_cand = 0

    def consider(desc: str, v: np.ndarray | None):
        nonlocal n_cand
        if v is None or not np.all(np.isfinite(v)):
            return
        n_cand += 1
        rho = spearman_abs(v, phi)
        nmi = normalized_mi(v, phi) if use_nmi else float("nan")
        best.append((rho, nmi, desc))

    for c in cat:  # ordinal codes, identity only
        codes = pd.Categorical(feats[c], categories=sorted(feats[c].unique())).codes.astype(float)
        consider(f"code({c})", codes)
    un = {c: _unary_variants(x) for c, x in num.items()}
    for c, variants in un.items():
        for uname, v in variants.items():
            consider(f"{uname}({c})", v)
    for ci, cj in itertools.combinations(num, 2):
        for (fi, vi), (gj, vj) in itertools.product(un[ci].items(), un[cj].items()):
            for op in _OPS:
                if op == "r/":
                    consider(f"{gj}({cj}) / {fi}({ci})", _binary(op, vi, vj))
                else:
                    consider(f"{fi}({ci}) {op} {gj}({cj})", _binary(op, vi, vj))

    best.sort(key=lambda t: -t[0])
    top_rho = best[:top_k]
    max_rho = best[0][0] if best else 0.0
    out: dict[str, Any] = {
        "n_candidates": n_cand, "max_abs_spearman": max_rho,
        "top_by_spearman": [{"rho": r, "nmi": m, "op": d} for r, m, d in top_rho],
        "spearman_threshold": spearman_threshold,
    }
    ok = max_rho < spearman_threshold
    if use_nmi:
        by_nmi = sorted(best, key=lambda t: -t[1])
        out["max_nmi"] = by_nmi[0][1] if by_nmi else 0.0
        out["top_by_nmi"] = [{"rho": r, "nmi": m, "op": d} for r, m, d in by_nmi[:top_k]]
        out["nmi_threshold"] = nmi_threshold
        ok = ok and out["max_nmi"] < nmi_threshold
    out["passed"] = bool(ok)
    return out


def screen_p7_extras(df: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    """Ratio-alone and guess-grid checks for P7 (see module docstring)."""
    spec = meta["phi"]
    phi = T.eval_phi(spec, df)
    a, b, c = (df[spec["cols"][v]].to_numpy(float) for v in "abc")
    ratio = a / b
    ratio_ok = not meets_criterion(ratio, phi, FAMILY_RULE)[0]
    guesses = []
    qs = list(T.P7_ROUND_QUANTILES)
    rounds = sorted({float(f"{np.quantile(c, q):.1g}") for q in np.linspace(0.1, 0.9, 17)}
                    | {float(round(np.quantile(c, q), 1)) for q in qs})
    worst = ("", 0.0, 0.0)
    for label, thr in [(f"q={q:.3f}", float(np.quantile(c, q))) for q in qs] + [(f"round={r}", r) for r in rounds]:
        v = ratio * (c >= thr)
        hit, rho, nmi = meets_criterion(v, phi, FAMILY_RULE)
        guesses.append({"guess": label, "thr": thr, "rho": rho, "nmi": nmi, "meets": bool(hit)})
        if max(rho, nmi) > max(worst[1], worst[2]):
            worst = (label, rho, nmi)
    return {
        "ratio_alone_rho": spearman_abs(ratio, phi), "ratio_alone_nmi": normalized_mi(ratio, phi),
        "ratio_alone_passes_screen": bool(ratio_ok),
        "guess_grid_any_meets": any(g["meets"] for g in guesses),
        "guess_grid_worst": {"guess": worst[0], "rho": worst[1], "nmi": worst[2]},
        "passed": bool(ratio_ok and not any(g["meets"] for g in guesses)),
    }


def screen_from_folders(tasks_root: Path, meta_dir: Path, keys=GATED_KEYS, gate_keys=GATED_KEYS) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for key in keys:
        task_id = f"synp_{key}"
        meta = json.loads((meta_dir / f"{task_id}.json").read_text(encoding="utf-8"))
        df = pd.read_csv(tasks_root / meta["task_type"] / task_id / "data.csv")
        phi = T.phi_from_meta(meta, df)
        res = screen_task(df, phi, meta["detector"]["family"])
        res["gated"] = key in gate_keys
        if key == "p7":
            res["p7_extras"] = screen_p7_extras(df, meta)
            res["passed"] = bool(res["passed"] and res["p7_extras"]["passed"])
        report[task_id] = res
    return report


def main(argv=None) -> int:  # pragma: no cover - thin CLI
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks-root", required=True)
    ap.add_argument("--meta-dir", required=True)
    ap.add_argument("--keys", nargs="*", default=list(GATED_KEYS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rep = screen_from_folders(Path(a.tasks_root), Path(a.meta_dir), tuple(a.keys))
    text = json.dumps(rep, indent=2, default=float)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    return 0 if all(r["passed"] for r in rep.values() if r["gated"]) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
