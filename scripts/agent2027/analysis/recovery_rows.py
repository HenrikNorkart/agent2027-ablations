"""Score recovery (rec_sel / rec_any per fold-episode) for arbitrary manifest rows.

Same episode construction and detector of record as G2 (``g2_stats.collect``): episodes from
``a02_readout.build_row_episodes`` on the row's final attempt, scored with the amended
detector (Part III.2, Amendment 2). ITT: a row whose final status in its session's
``status.tsv`` is not OK contributes 5 zero episodes (``a02_readout.itt_episode_rows``).

  python scripts/agent2027/analysis/recovery_rows.py --runs <session dirs...> \
      --manifest manifests/agent2027/A09_RT3.csv ... --rows A09-002 ... \
      --tasks <tasks root> --meta <synp meta> --out episodes.csv

Run on CAIR only inside a rogue_one container with the run folders mounted read-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd


def final_status(root: Path, row_id: str) -> str:
    st = root / "status.tsv"
    last = "MISSING"
    if st.exists():
        for r in csv.DictReader(st.open(encoding="utf-8"), delimiter="\t"):
            if r.get("row_id") == row_id:
                last = r.get("status", "?")
    return last


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    a = ap.parse_args(argv)
    if a.repo not in sys.path:
        sys.path.insert(0, a.repo)
    from scripts.agent2027.analysis import a02_readout as a02  # noqa: PLC0415
    from scripts.agent2027.analysis import recovery_detector as rd  # noqa: PLC0415

    man = {}
    for f in a.manifest:
        for r in csv.DictReader(open(f, newline="", encoding="utf-8")):
            man[r["row_id"]] = r
    roots = [Path(r) for r in a.runs]
    cache, out = {}, []
    for rid in a.rows:
        m = man[rid]
        ds, ttype, seed = m["dataset_id"], m["task_type"], int(m["fold_seed"])
        # a row re-run in a later session: prefer the session where it ended OK, else the last one that has it
        have = [r for r in roots if (r / rid).exists()]
        root = next((r for r in have if final_status(r, rid) == "OK"), have[-1] if have else None)
        status = final_status(root, rid) if root else "MISSING"
        ok = status == "OK"
        scored = None
        if ok:
            out_dir = sorted((root / rid).glob("tasks/*/*/output"))[0]
            if ds not in cache:
                tdir = Path(a.tasks) / ttype / ds
                df_raw = pd.read_csv(tdir / "data.csv")
                labels = pd.read_csv(tdir / "labels.csv")
                phi, det = rd.load_phi(a.meta, ds, df_raw)
                cache[ds] = (df_raw, labels, phi, det)
            df_raw, labels, phi, det = cache[ds]
            strat = json.loads(m["env_overrides"] or "{}").get("ROGUE_ONE_STRATIFIED_FOLDS", "False") == "True"
            eps = []
            for ep in a02.build_row_episodes(out_dir):
                try:
                    r = a02.score_episode(ep, df_raw, labels, phi, det, seed, strat, amended=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"{rid} fold {ep['fold']}: detector error {exc}", file=sys.stderr)
                    r = {"rec_any": 0, "rec_sel": 0}
                eps.append({"fold": ep["fold"], "rec_sel": int(r["rec_sel"]), "rec_any": int(r["rec_any"])})
            scored = pd.DataFrame(eps)
        ep_df = a02.itt_episode_rows(ok, scored)
        for _, e in ep_df.iterrows():
            out.append({"row_id": rid, "ablation_id": m["ablation_id"], "arm_id": m["arm_id"], "dataset_id": ds,
                        "fold_seed": seed, "replicate": m["replicate"], "status": status,
                        "fold": int(e["fold"]), "rec_sel": int(e["rec_sel"]), "rec_any": int(e["rec_any"]),
                        "itt_imputed": int(e.get("itt_imputed", 0))})
        print(f"[recovery_rows] {rid} {ds} status={status} rec_sel={int(ep_df['rec_sel'].sum())}", file=sys.stderr, flush=True)
    pd.DataFrame(out).to_csv(a.out, index=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
