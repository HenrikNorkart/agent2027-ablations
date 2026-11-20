"""AGENT2027 final read-out (2026-11-02): one metrics row per manifest row, from the run folders.

Collects, for every row of the given manifests that has a status entry in any session: final status (the last
attempt by end time; ITT = final status not OK or any CAP_HIT), held-out score (``fold_aggregation_summary.csv``),
per-fold held-out scores of the selected trial (``fold_aggregation_best_iterations.csv``), the seed-matched
``base_no_fe`` mean and fold SD (``G2_base_no_fe_ref.csv``), ``success`` (A05 definition), and the E3
instrumentation counters (tokens per role, calls, tool errors, accepted appends, loop-guard and inf-guard fires,
context errors, fallback share). Descriptive collector only: no test is computed here.

    python final_rows.py --runs <S* dirs> --manifest <csv...> --base G2_base_no_fe_ref.csv --out rows.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

KEYS = ["tok_compl_scientist", "tok_compl_extractor", "tok_compl_tester", "tok_compl_total",
        "tok_compl_tester_assess", "tok_compl_tester_prune", "tok_prompt_total", "llm_calls_total",
        "n_tool_errors", "n_accepted", "err_per_accepted", "context_errors", "fallback_share",
        "loop_guard", "inf_guard", "max_prompt_tokens", "serper_calls", "llm_calls_extractor", "llm_calls_scientist",
        "llm_calls_tester", "append_accepted", "append_attempts", "iterations_total", "flop_proxy", "silent_rejections",
        "err_repeat_rate", "err_role_extractor", "runner_errors"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    roots = [Path(r) for r in a.runs]
    frames = []
    for r in roots:
        p = r / "status.tsv"
        if p.exists():
            frames.append(pd.read_csv(p, sep="\t", dtype=str).assign(session=r.name))
    st = pd.concat(frames, ignore_index=True)
    st["end_epoch"] = pd.to_numeric(st["end_epoch"], errors="coerce")
    base = pd.read_csv(a.base)
    bkey = {(r.dataset_id, int(r.fold_seed)): (float(r["mean"]), float(r.fold_sd)) for _, r in base.iterrows()}
    man = pd.concat([pd.read_csv(m, dtype=str) for m in a.manifest], ignore_index=True)

    out = []
    for _, m in man.iterrows():
        rid = m["row_id"]
        s = st[st["row_id"] == rid].sort_values("end_epoch")
        if not len(s):
            continue
        fin = s.iloc[-1]
        status = str(fin["status"])
        cap = bool((s["status"].str.upper() == "CAP_HIT").any())
        root = next((r for r in roots if r.name == fin["session"]), None)
        hits = sorted((root / rid).glob("tasks/*/*/output")) if root else []
        od = hits[0] if hits else None
        ttype, ds, seed = m["task_type"], m["dataset_id"], int(m["fold_seed"])
        metric = "accuracy" if ttype == "classification" else "n_RMSE"
        rec = {"row_id": rid, "ablation_id": m["ablation_id"], "arm_id": m["arm_id"], "dataset_id": ds,
               "task_type": ttype, "fold_seed": seed, "replicate": m["replicate"], "llm_seed": m.get("llm_seed"),
               "notes": m.get("notes", ""), "env_overrides": m.get("env_overrides", ""),
               "session": fin["session"], "n_attempts": len(s), "attempt_statuses": ";".join(s["status"]),
               "final_status": status, "final_reason": fin.get("reason", ""), "cap_hit": int(cap),
               "first_status": str(s.iloc[0]["status"])}
        ok = status == "OK" and od is not None and (od / "fold_aggregation_summary.csv").exists()
        rec["row_ok"] = int(ok)
        rec["itt"] = int((not ok) or cap)
        bm, bsd = bkey.get((ds, seed), (None, None))
        rec["base_mean"], rec["base_sd"] = bm, bsd
        score = None
        if ok:
            sm = pd.read_csv(od / "fold_aggregation_summary.csv")
            hit = sm.loc[sm["metric"] == metric, "mean"]
            score = float(hit.iloc[0]) if len(hit) else None
        rec["heldout_score"] = score
        if score is not None and bm is not None:
            rec["success"] = int(score >= bm - 2 * bsd) if ttype == "classification" else int(score <= bm + 2 * bsd)
            if ds == "junglechess":
                rec["success"] = int(rec["success"] and score >= 0.95)
        else:
            rec["success"] = 0
        if od is not None and (od / "fold_aggregation_best_iterations.csv").exists():
            b = pd.read_csv(od / "fold_aggregation_best_iterations.csv")
            for f in range(5):
                bb = b[(b["test_fold_id"] == f) & (b["status"].astype(str) == "completed")]
                rec[f"fold{f}_score"] = float(bb[metric].iloc[0]) if len(bb) and metric in bb else None
        js = od / "instrumentation_summary.json" if od is not None else None
        if js is not None and js.exists():
            d = json.loads(js.read_text(encoding="utf-8"))
            for k in KEYS:
                rec[k] = d.get(k)
        out.append(rec)
        print(f"[final_rows] {rid} {status} ok={int(ok)} score={score}", file=sys.stderr, flush=True)
    pd.DataFrame(out).to_csv(a.out, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
