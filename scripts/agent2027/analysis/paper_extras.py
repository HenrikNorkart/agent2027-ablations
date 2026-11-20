"""Reproduce the paper numbers that are not part of the confirmatory read-out (final_tests.py).

    python paper_extras.py [--res docs/agent2027/results]

A1   tokens with / without the Planner at seed 42 (Results, A1)   <- final_20261102/pareto_front.csv
A2   discovery without / with memory, exact test stratified by task, runs with >= 1 discovery,
     and focus revisits with / without memory (Results, A2)       <- RV3_A12_runs.csv, final_20261112/episodes_final.csv,
                                                                     final_20261102/process_a12_rows.csv
F0   sampling baseline (gpt-oss-120b, best of k one-shot lists at the complete system's token budget) vs the
     complete system on P2-P6b, paired 90% CI (Discussion)        <- final_20261102/f0_fold_summary.csv, episodes, rows
Baselines: OpenFE (screen, and vs the complete system at seed 42) and LLM-FE under its three scorings (Design, Table III) <- A08_screen_episodes.csv, X6_A08_llmfe_episodes.csv
Real knockouts of A1, A4, A5 on the real datasets (Discussion)    <- A12_realdata_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

from scipy import stats

from final_tests import exact_stratified, newcombe_unpaired

T6 = ["synp_p2", "synp_p3", "synp_p4", "synp_p5", "synp_p6a", "synp_p6b"]


def rows_of(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=str(Path(__file__).resolve().parents[3] / "docs/agent2027/results"))
    R = Path(ap.parse_args().res)
    F02, F12 = R / "final_20261102", R / "final_20261112"

    # A1 tokens
    pf = {x["config"]: x for x in rows_of(F02 / "pareto_front.csv") if x["variant"] == "primary" and x["panel"] == "A" and x["axis"] == "tok_compl_total"}
    print(f"A1 tokens: complete {float(pf['full']['cost'])/1e6:.2f}M, without Planner {float(pf['fx_pAJM']['cost'])/1e6:.2f}M "
          f"(ratio {float(pf['fx_pAJM']['ratio_vs_full']):.2f})")
    # A1 removed alone: the factorial cell without the Planner (one run per task, seed 42) vs the complete system
    # at seed 42 (both runs), 90% Newcombe CI
    a1 = [x for x in rows_of(R / "RV3_A12_runs.csv") if x["arm_id"] == "fx_pAJM"]
    k1, n1 = sum(int(x["k"]) for x in a1), sum(int(x["n"]) for x in a1)
    k0, n0 = round(float(pf["full"]["rec_sel"]) * 60), 60
    dd, lo, hi = newcombe_unpaired(k1, n1, k0, n0, 0.90)
    print(f"A1 removed alone: {k1}/{n1} vs {k0}/{n0}, {dd:+.2f} 90% [{lo:+.2f}, {hi:+.2f}]")

    # A2 discovery and revisits on the 72 factorial runs
    fx = {x["row_id"]: x for x in rows_of(R / "RV3_A12_runs.csv")}
    ep = {(x["row_id"], x["fold"]): x for x in rows_of(F12 / "episodes_final.csv") if x["row_id"] in fx}
    tab = defaultdict(lambda: [0, 0, 0, 0])  # memory off: k, n; memory on: k, n
    run_any: dict[str, int] = defaultdict(int)
    for (rid, _), x in ep.items():
        off = fx[rid]["M"] == "-1.0"
        hit = int(float(x["rec_any"] or 0))
        t = tab[fx[rid]["task"]]
        t[0 if off else 2] += hit
        t[1 if off else 3] += 1
        run_any[rid] = max(run_any[rid], hit)
    k_off = sum(t[0] for t in tab.values()); k_on = sum(t[2] for t in tab.values())
    ex = exact_stratified([tuple(t) for _, t in sorted(tab.items())])
    r_off = sum(v for r, v in run_any.items() if fx[r]["M"] == "-1.0")
    r_on = sum(v for r, v in run_any.items() if fx[r]["M"] == "1.0")
    print(f"A2 discovery: {k_off}/180 without vs {k_on}/180 with memory, exact stratified p={float(ex['p_two_sided']):.3f}; "
          f"runs with >=1 discovery {r_off} without vs {r_on} with memory, of 36 each")
    pr = [x for x in rows_of(F02 / "process_a12_rows.csv") if x["row_id"] in fx and x["focus_novelty"] != ""]
    on = [float(x["focus_novelty"]) for x in pr if fx[x["row_id"]]["M"] == "1.0"]
    off = [float(x["focus_novelty"]) for x in pr if fx[x["row_id"]]["M"] == "-1.0"]
    print(f"A2 focus revisits: {100*st.mean(on):.2f}% with vs {100*st.mean(off):.2f}% without memory, "
          f"Mann-Whitney p={stats.mannwhitneyu(on, off).pvalue:.4f} (n={len(on)}, {len(off)})")

    # Sampling baseline vs complete system, paired over folds of P2-P6b at seeds 42 and 4
    rows = {x["row_id"]: x for x in rows_of(F12 / "rows_final.csv")}
    f0 = {(x["dataset_id"], x["fold_seed"], x["fold"]): float(x["rec_sel_bestk"])
          for x in rows_of(F02 / "f0_fold_summary.csv") if x["rung"] == "O120" and x["dataset_id"] in T6}
    cs = {(x["dataset_id"], x["fold_seed"], x["fold"]): float(x["rec_sel"])
          for x in rows_of(F12 / "episodes_final.csv")
          if x["ablation_id"] == "A05" and x["arm_id"] == "full" and int(x["row_id"].split("-")[1]) < 100
          and x["dataset_id"] in T6 and rows[x["row_id"]]["llm_seed"] == x["fold_seed"]}
    keys = sorted(set(f0) & set(cs))
    d = [f0[k] - cs[k] for k in keys]
    m, se = st.mean(d), st.stdev(d) / math.sqrt(len(d))
    print(f"F0 sampling: {st.mean(f0[k] for k in keys):.2f} vs complete {st.mean(cs[k] for k in keys):.2f} "
          f"({m:+.2f}, paired 90% CI [{m-1.645*se:.2f}, {m+1.645*se:.2f}], {len(keys)} fold-episodes)")

    # A4/A5 equivalence on the two informative tasks (P2, P6b): runs of record vs with the power replicates (8xx)
    import pandas as pd
    import glmm_holm as g
    epi = pd.read_csv(F12 / "episodes_final.csv").drop_duplicates(["row_id", "fold"], keep="last")
    a12 = epi[epi.row_id.str.startswith("A12-")]
    num = a12.row_id.str.slice(4).astype(int)
    cells = {**g.CELLS, **g.RV3_CELLS}
    base = num.between(1, 56) | num.between(66, 93)
    for label, keep in (("runs of record", base), ("with replicates", base | num.between(800, 899))):
        sub = a12[keep & a12.arm_id.isin(cells) & a12.dataset_id.isin(["synp_p2", "synp_p6b"])]
        run = sub.groupby(["row_id", "arm_id", "dataset_id"], as_index=False).agg(k=("rec_sel", "sum"), n=("rec_sel", "size"))
        for i, f in enumerate("PAJM"):
            run[f] = run.arm_id.map(lambda a, i=i: 1.0 if cells[a][i] == "+" else -1.0)
        run["Intercept"] = 1.0
        terms = ["Intercept", "P", "A", "J", "M"]
        fit = g.fit_betabin_glmm(run.k, run.n, run[terms].to_numpy(), run.dataset_id, terms)
        for t, name in (("A", "A4"), ("J", "A5")):
            dlt, se = g.delta_x(fit, t)  # component on minus off; report ablated minus complete
            print(f"{name} on P2+P6b, {label} ({len(run)} runs): ablated-complete {-dlt:+.3f} "
                  f"90% [{-dlt-1.6449*se:+.3f}, {-dlt+1.6449*se:+.3f}], TOST p={g.tost_p(dlt, se, 0.167):.4f}")

    # Baselines: OpenFE from the A08 CPU screen. OpenFE is deterministic (fixed internal seed), so its seed-42
    # fold-episodes are paired with both complete-system runs at seed 42 (LLM seeds 42, 1042), the LLM-FE reference;
    # 90% Newcombe CI and exact test stratified by task, as for the single ablations
    scr = [x for x in rows_of(R / "A08_screen_episodes.csv") if x["arm"] == "openfe"]
    p1 = [int(x["rec_sel"]) for x in scr if x["dataset_id"] == "synp_p1"]
    rest = [int(x["rec_sel"]) for x in scr if x["dataset_id"] in T6 + ["synp_p7"]]
    print(f"OpenFE screen: P1 {sum(p1)}/{len(p1)}, P2-P7 {sum(rest)}/{len(rest)} (fold seeds 42, 4, 5)")
    of = {(x["dataset_id"], x["fold"]): int(x["rec_sel"]) for x in scr if x["dataset_id"] in T6 and x["fold_seed"] == "42"}
    f42 = [x for x in rows_of(F12 / "episodes_final.csv") if x["ablation_id"] == "A05" and x["arm_id"] == "full"
           and int(x["row_id"].split("-")[1]) < 100 and x["dataset_id"] in T6 and x["fold_seed"] == "42"]
    k1 = sum(of[(x["dataset_id"], x["fold"])] for x in f42); k0 = sum(int(float(x["rec_sel"])) for x in f42); n = len(f42)
    tabs = [(sum(of[(x["dataset_id"], x["fold"])] for x in f42 if x["dataset_id"] == t), sum(x["dataset_id"] == t for x in f42),
             sum(int(float(x["rec_sel"])) for x in f42 if x["dataset_id"] == t), sum(x["dataset_id"] == t for x in f42)) for t in T6]
    dd, lo, hi = newcombe_unpaired(k1, n, k0, n, 0.90)
    print(f"OpenFE vs complete (seed 42, both runs): {k1}/{n} vs {k0}/{n}, {dd:+.2f} 90% [{lo:+.2f}, {hi:+.2f}], "
          f"exact stratified p={exact_stratified(tabs)['p_two_sided']:.2g}")
    lf = [x for x in rows_of(R / "X6_A08_llmfe_episodes.csv") if x["dataset_id"] in T6]
    for mode in ("best3", "notebook", "any"):
        v = [int(x["rec_sel"]) for x in lf if x["arm"].endswith(mode)]
        print(f"LLM-FE {mode}: {sum(v)}/{len(v)} on P2-P6b (seed 42, r0 + r1); best3 is primary (decision of 2026-10-26)")
    # Table III baseline rows: LLM-FE (best3) against the same reference with the same CI and test as OpenFE
    b3 = [x for x in lf if x["arm"].endswith("best3")]
    k1 = sum(int(x["rec_sel"]) for x in b3)
    tabs = [(sum(int(x["rec_sel"]) for x in b3 if x["dataset_id"] == t), sum(x["dataset_id"] == t for x in b3),
             sum(int(float(x["rec_sel"])) for x in f42 if x["dataset_id"] == t), sum(x["dataset_id"] == t for x in f42)) for t in T6]
    dd, lo, hi = newcombe_unpaired(k1, len(b3), k0, n, 0.90)
    print(f"LLM-FE vs complete (seed 42, both runs): {k1}/{len(b3)} vs {k0}/{n}, {dd:+.2f} 90% [{lo:+.2f}, {hi:+.2f}], "
          f"exact stratified p={exact_stratified(tabs)['p_two_sided']:.2g} (confirmatory C1b: final_tests.py)")

    # Real-data knockouts
    for a in json.loads((R / "A12_realdata_summary.json").read_text()):
        print(f"real data {a['arm']}: verdict {a['verdict']}, delta_std {a['mean_delta_std']:+.2f} "
              f"90% [{a['t90'][0]:+.2f}, {a['t90'][1]:+.2f}], tokens x{a['tok_ratio']:.2f}")


if __name__ == "__main__":
    main()
