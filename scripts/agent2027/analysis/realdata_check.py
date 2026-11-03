"""Amendment 16 real-data check: role knockouts vs Full on Core-10 (exploratory, outside Holm).

Pairs each arm row A12-701..780 with the A05 Full row named in its notes ("pairs Full A05-xxx"). Outcome
Delta = arm - Full on ``heldout_score`` (balanced accuracy; nRMSE sign-flipped so higher is better), standardised
by sigma_d (G2_sigma_d.csv, the A05 NI sigma_d). Per arm: mean standardised Delta over the 20 (dataset, seed) pairs
with t 90%/95% CIs and a dataset-cluster bootstrap 90% CI. Reading: "non-inferior" iff the t 90% lower bound > -M
(M = 1.0); "worse" iff the t 95% upper bound < 0; else "inconclusive". Tokens: per-pair log ratio of
``tok_compl_total`` arm/Full, mean with t 90% CI, back-transformed (pairing makes dataset FE implicit).
ITT: an arm row without a held-out score (final status not OK) is imputed at the paired Full row's ``base_mean``.

    python realdata_check.py --runs <S1025h> <S1025i> <S1025j> <S1025k> --manifest A12_FX.csv \
        --g2-rows G2_rows.csv --g2-sigma G2_sigma_d.csv --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a02_readout import _status_table, find_output_dir, held_out_score  # noqa: E402

M = 1.0
ARMS = ["fx_pAJM", "fx_PaJM", "fx_PAjM", "fx_PajM"]


def tok_total(out_dir: Path | None) -> float | None:
    if out_dir is None:
        return None
    p = out_dir / "instrumentation_summary.json"
    if not p.exists():
        return None
    v = json.loads(p.read_text(encoding="utf-8")).get("tok_compl_total")
    return float(v) if v is not None else None


JOB_LOG_DIR = Path.home()


def session_ended(sid: str) -> bool:
    p = JOB_LOG_DIR / f"ro27_{sid}.job.log"
    return p.exists() and f"RO27_SESSION_END {sid} " in p.read_text(encoding="utf-8", errors="replace")


def build(runs, manifest, g2_rows, g2_sigma) -> pd.DataFrame:
    man = pd.read_csv(manifest)
    man = man[man.row_id.str.match(r"^A12-7[0-8][0-9]$") & man.notes.str.contains("Amendment 16", na=False)]
    full = pd.read_csv(g2_rows).set_index("row_id")
    sig = pd.read_csv(g2_sigma).set_index("dataset_id").sigma_d
    status = _status_table([Path(r) for r in runs])
    final = status.groupby("row_id").tail(1).set_index("row_id") if len(status) else pd.DataFrame()
    out = []
    for r in man.itertuples():
        fid = re.search(r"pairs Full (A05-\d+)", r.notes).group(1)
        f = full.loc[fid]
        od, sess = find_output_dir([Path(x) for x in runs], r.row_id)
        st = str(final.loc[r.row_id, "status"]) if r.row_id in final.index and "status" in final.columns else "MISSING"
        ended = sess is not None and session_ended(sess)
        if st == "MISSING" or (st != "OK" and not ended):  # still running, or a retry may follow
            out.append({"row_id": r.row_id, "arm": r.arm_id, "dataset_id": r.dataset_id, "status": "PENDING"})
            continue
        score = held_out_score(od, r.task_type) if od is not None and st == "OK" else None
        itt = score is None
        if itt:
            score = float(f.base_mean)
        sign = -1.0 if r.task_type == "regression" else 1.0
        d = sign * (score - float(f.heldout_score))
        out.append({"row_id": r.row_id, "arm": r.arm_id, "dataset_id": r.dataset_id, "task_type": r.task_type,
                    "fold_seed": r.fold_seed, "full_row": fid, "status": st, "itt": int(itt),
                    "arm_score": score, "full_score": float(f.heldout_score), "delta": d,
                    "sigma_d": float(sig[r.dataset_id]), "delta_std": d / float(sig[r.dataset_id]),
                    "tok_arm": tok_total(od), "tok_full": float(f.tok_compl_total)})
    return pd.DataFrame(out)


def t_ci(x, level):
    x = np.asarray(x, float)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(0.5 + level / 2, len(x) - 1) * se
    return m, m - h, m + h


def cluster_boot(df, col, n=10000, seed=1):
    rng = np.random.default_rng(seed)
    groups = [g[col].to_numpy() for _, g in df.groupby("dataset_id")]
    means = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        means.append(np.concatenate([groups[i] for i in pick]).mean())
    return np.percentile(means, [5, 95])


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    res = []
    for arm in ARMS:
        a = rows[(rows.arm == arm) & (rows.status != "PENDING")]
        if a.empty:
            continue
        m, lo90, hi90 = t_ci(a.delta_std, 0.90)
        _, lo95, hi95 = t_ci(a.delta_std, 0.95)
        b = cluster_boot(a, "delta_std")
        verdict = "non-inferior" if lo90 > -M else ("worse" if hi95 < 0 else "inconclusive")
        tk = a.dropna(subset=["tok_arm"])
        lr = np.log(tk.tok_arm / tk.tok_full)
        tm, tlo, thi = t_ci(lr, 0.90) if len(lr) > 1 else (np.nan, np.nan, np.nan)
        res.append({"arm": arm, "pairs": len(a), "pending": int(((rows.arm == arm) & (rows.status == "PENDING")).sum()),
                    "itt": int(a.itt.sum()), "mean_delta_std": m,
                    "t90": [lo90, hi90], "t95": [lo95, hi95], "boot90": list(b), "verdict": verdict,
                    "mean_delta_raw": a.delta.mean(), "median_delta_std": a.delta_std.median(),
                    "tok_ratio": float(np.exp(tm)), "tok_ratio_90": [float(np.exp(tlo)), float(np.exp(thi))],
                    "tok_pairs": int(len(lr))})
    return pd.DataFrame(res)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--g2-rows", required=True)
    ap.add_argument("--g2-sigma", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--job-logs", default=str(Path.home()), help="directory holding ro27_<SID>.job.log")
    a = ap.parse_args(argv)
    global JOB_LOG_DIR
    JOB_LOG_DIR = Path(a.job_logs)
    rows = build(a.runs, a.manifest, a.g2_rows, a.g2_sigma)
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    rows.to_csv(od / "realdata_rows.csv", index=False)
    s = summarise(rows)
    s.to_json(od / "realdata_summary.json", orient="records", indent=2)
    pd.set_option("display.width", 250)
    print(rows.groupby(["arm", "status"]).size().to_string())
    print(s.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
