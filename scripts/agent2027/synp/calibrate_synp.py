"""Run the headroom calibration for every SYN-P task and write a report (E8a).

    python scripts/agent2027/synp/calibrate_synp.py --out <dir> [--keys p1 p2 ...]

Writes ``<out>/calibration.json`` (machine-readable, incl. the chosen signal
parameter per task: paste into ``tasks.SIGNAL_DEFAULTS``) and
``<out>/calibration_report.md``.  Also checks P0 (no headroom by design,
|headroom| < 0.02 on nRMSE), P6b (identical to P6a, so same data and labels) and
P7's phi_wrong (XGB(raw + phi_wrong) - XGB(raw) < 0.01 balanced accuracy).
The report never contains P7's constant or phi_wrong.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.agent2027.synp import calibration as C  # noqa: E402
from scripts.agent2027.synp import tasks as T  # noqa: E402


def _run_one(args):
    key, kw, jobs = args
    C.N_JOBS = jobs
    if key == "p0":
        g = T.generate_task("p0", **kw)
        h = C.headroom(g.data, g.phi, g.labels["target"].to_numpy(), "regression")
        return key, {"key": key, "param": "noise", "value": T.SIGNAL_DEFAULTS["p0"]["noise"],
                     "target": None, "achieved_mean": h["mean"], "achieved_min": h["min"],
                     "seeds": list(C.FOLD_SEEDS), "passed": bool(abs(h["mean"]) < 0.02 and abs(h["min"]) < 0.02),
                     "note": "no headroom by design: |headroom| < 0.02", "raw_score": h["raw_mean"],
                     "phi_score": h["with_phi_mean"]}
    r = C.calibrate_task(key, **kw).to_dict()
    if key == "p7":  # phi_wrong must add nothing
        kw2 = dict(kw)
        g = T.generate_task("p7", params={r["param"]: r["value"]}, **kw2)
        spec = g.meta["phi_wrong"]
        wrong = T.eval_phi(spec, g.data)
        hw = C.headroom(g.data, g.phi, g.labels["target"].to_numpy(), "classification", extra=wrong)
        r["phi_wrong_headroom_mean"] = hw["mean"]
        r["phi_wrong_ok"] = bool(hw["mean"] < 0.01)
    return key, r


def run(keys, kw, workers: int = 8, jobs: int = 3):
    jobs_list = [(k, kw, jobs) for k in keys]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return dict(ex.map(_run_one, jobs_list))


def write_report(results: dict, out: Path, master_seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    lines = [
        "# SYN-P headroom calibration (E8a)", "",
        f"Master seed {master_seed}; n = {T.N_ROWS}; XGBoost with the Tester's assessment hyper-parameters, "
        "5-fold CV, fold seeds {42, 4, 5} (mean; min in brackets).", "",
        "Headroom: classification = balanced-accuracy gain (target >= 0.06); regression = relative RMSE "
        "reduction (target >= 0.15).", "",
        "| task | type | signal param | value | target | achieved mean (min) | pass |", "|---|---|---|---|---|---|---|",
    ]
    for key, r in results.items():
        tgt = "n/a (|h|<0.02)" if r["target"] is None else f"{r['target']:.2f}"
        lines.append(f"| synp_{key} | {T.TASK_TYPES[key]} | {r['param']} | {r['value']} | {tgt} | "
                     f"{r['achieved_mean']:.4f} ({r['achieved_min']:.4f}) | {'yes' if r['passed'] else 'NO'} |")
    if "p7" in results and "phi_wrong_headroom_mean" in results["p7"]:
        lines += ["", f"P7 phi_wrong headroom (must be < 0.01): {results['p7']['phi_wrong_headroom_mean']:.4f} "
                      f"-> {'ok' if results['p7']['phi_wrong_ok'] else 'FAIL'}"]
    lines += ["", "P6b is a transformation of P6a (same rows, same labels), so its headroom is P6a's by construction "
                  "(XGBoost is invariant to the hidden positive rescaling up to histogram binning)."]
    (out / "calibration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:  # pragma: no cover - thin CLI
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--keys", nargs="*", default=list(T.GENERATED_KEYS))
    ap.add_argument("--master-seed", type=int, default=T.MASTER_SEED)
    ap.add_argument("--p7-secret", default=os.environ.get("SYNP_P7_SECRET", ""))
    ap.add_argument("--p6a-candidate", default=T.DEFAULT_P6A)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    kw = dict(master_seed=a.master_seed, p7_secret=a.p7_secret, p6a_candidate=a.p6a_candidate)
    res = run(a.keys, kw, workers=a.workers)
    write_report(res, a.out, a.master_seed)
    print((a.out / "calibration_report.md").read_text())
    return 0 if all(r["passed"] for r in res.values()) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
