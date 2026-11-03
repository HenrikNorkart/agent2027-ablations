"""Descriptive arm-vs-Full read-out for A10, A11, A16 (SYN-P 7 minus P1) and A14 (P7). PRELIM, LPM with task FE."""
import sys
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

D = Path(sys.argv[1])
ep = pd.read_csv(D / "episodes_all.csv")
run = ep.groupby(["row_id", "ablation_id", "arm_id", "dataset_id", "fold_seed", "replicate", "status"], as_index=False) \
    .agg(k=("rec_sel", "sum"), n=("rec_sel", "size"))
run["y"] = run.k / run.n
run["pre"] = ~run.row_id.str.contains(r"-8\d\d$")
full = run[run.ablation_id == "A05"]


def contrast(abl, arm, label, pre_only):
    a = run[(run.ablation_id == abl) & (run.arm_id == arm) & (run.pre | (not pre_only))]
    seeds = set(a.fold_seed)
    f = full[full.fold_seed.isin(seeds) & full.dataset_id.isin(set(a.dataset_id)) & (full.pre | (not pre_only))]
    df = pd.concat([a.assign(g=1), f.assign(g=0)])
    df["blk"] = df.fold_seed.astype(str)
    m = smf.ols("y ~ g + C(dataset_id) + C(blk)", data=df).fit()
    b, se, dfr = m.params["g"], m.bse["g"], m.df_resid
    t = stats.t.ppf(0.975, dfr)
    by = df.groupby(["dataset_id", "g"]).y.mean().unstack().round(2)
    return {"contrast": label, "rows": "pre-reg" if pre_only else "+replicates", "arm k/n": f"{int(a.k.sum())}/{int(a.n.sum())}",
            "Full k/n": f"{int(f.k.sum())}/{int(f.n.sum())}", "Δ": round(b, 3), "95% CI": f"[{b - t * se:.3f}, {b + t * se:.3f}]",
            "p": round(2 * stats.t.sf(abs(b / se), dfr), 4), "ITT": int((a.status != "OK").sum()),
            "by task (arm-Full)": " ".join(f"{d[5:]}:{r[1] - r[0]:+.2f}" for d, r in by.iterrows())}


rows = []
for pre in (True, False):
    rows += [contrast("A16", "set_phased", "C1a SET-phased − Full", pre),
             contrast("A10", "restarts", "C3 restarts − Full", pre),
             contrast("A11", "l0", "A11 L0 feedback − Full", pre)]
print(pd.DataFrame(rows).to_markdown(index=False))
p7 = run[run.ablation_id == "A14"].groupby("arm_id").agg(k=("k", "sum"), n=("n", "sum"), runs=("k", "size"), itt=("status", lambda s: int((s != "OK").sum())))
print("\nA14 on synp_p7 (rec_sel k/n per arm):")
print(p7.to_markdown())
print("\nFull by task (all A05 Full rows on the population):")
print(full.groupby("dataset_id").agg(k=("k", "sum"), n=("n", "sum")).to_markdown())
