"""A12 PRELIM read-out (descriptive + linear-probability model). The pre-registered beta-binomial GLMM is E11b (Tue 27)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

D = Path(sys.argv[1])
DELTA = 0.167
POP = ["synp_p2", "synp_p3", "synp_p4", "synp_p5", "synp_p6a", "synp_p6b"]
CELLS = {"fx_pAJM": "-+++", "fx_PaJM": "+-++", "fx_PAjM": "++-+", "fx_PAJm": "+++-",
         "fx_pajM": "---+", "fx_paJm": "--+-", "fx_pAjm": "-+--", "fx_Pajm": "+---"}

ep = pd.concat([pd.read_csv(D / "a12_episodes.csv"), pd.read_csv(D / "A09_delta_episodes.csv").query("ablation_id == 'A05'")])
run = ep.groupby(["row_id", "ablation_id", "arm_id", "dataset_id", "fold_seed", "replicate", "status"], as_index=False) \
    .agg(k=("rec_sel", "sum"), n=("rec_sel", "size"), k_any=("rec_any", "sum"))
run["y"] = run.k / run.n
num = run.row_id.str.slice(4).astype(int)
fx = run[(run.ablation_id == "A12")].copy()
fx["prereg"] = num[fx.index] <= 56
for i, f in enumerate("PAJM"):
    fx[f] = fx.arm_id.map(lambda a: 1 if CELLS[a][i] == "+" else -1)
fx["PA"], fx["PJ"], fx["PM"] = fx.P * fx.A, fx.P * fx.J, fx.P * fx.M
tok = pd.read_csv(D / "a12_tokens.tsv", sep="\t")
fx = fx.merge(tok, on="row_id", how="left")
out = []
P = lambda *a: out.append(" ".join(str(x) for x in a))

pre = fx[fx.prereg & fx.dataset_id.isin(POP)]
P("## Confirmatory rows:", len(pre), "runs,", int(pre.n.sum()), "episodes; ITT:", int((pre.status != "OK").sum()))
tab = pre.pivot_table(index="arm_id", columns="dataset_id", values="k", aggfunc="sum").reindex(list(CELLS))[POP]
tab["pooled"] = pre.groupby("arm_id").k.sum() / pre.groupby("arm_id").n.sum()
P("\n### rec_sel k/5 per cell x task (pre-registered, SYN-P 7 minus P1)\n")
P(tab.to_markdown())
full = run[(run.ablation_id == "A05") & run.dataset_id.isin(POP)]


def lpm(df, extra=""):
    m = smf.ols(f"y ~ C(dataset_id) + P + A + J + M + PA + PJ + PM{extra}", data=df).fit()
    rows = []
    for f in ["P", "A", "J", "M", "PA", "PJ", "PM"]:
        b, se, dfr = 2 * m.params[f], 2 * m.bse[f], m.df_resid
        t = stats.t.ppf(0.95, dfr)
        lo90, hi90 = b - t * se, b + t * se
        t975 = stats.t.ppf(0.975, dfr)
        p2 = 2 * stats.t.sf(abs(b / se), dfr)
        p_tost = max(stats.t.sf((b + DELTA) / se, dfr), stats.t.cdf((b - DELTA) / se, dfr))
        rows.append({"effect": f, "Delta": round(b, 3), "90% CI": f"[{lo90:.3f}, {hi90:.3f}]",
                     "95% CI": f"[{b - t975 * se:.3f}, {b + t975 * se:.3f}]", "p two-sided": round(p2, 4),
                     "p TOST(±0.167)": round(p_tost, 4), "df": int(dfr)})
    return pd.DataFrame(rows)


P("\n### Effects (Delta = mean X+ minus X-, LPM with task FE; pre-registered rows)\n")
P(lpm(pre).to_markdown(index=False))

# conservative: per-task contrasts, t over tasks (df = n_tasks - 1)
def per_task(df):
    rows = []
    for f in ["P", "A", "J", "M", "PA", "PJ", "PM"]:
        d = df.groupby("dataset_id").apply(lambda g: g[g[f] == 1].y.mean() - g[g[f] == -1].y.mean())
        n = len(d); b = d.mean(); se = d.std(ddof=1) / np.sqrt(n); t = stats.t.ppf(0.95, n - 1)
        pt = max(stats.t.sf((b + DELTA) / se, n - 1), stats.t.cdf((b - DELTA) / se, n - 1)) if se > 0 else float("nan")
        rows.append({"effect": f, "Delta": round(b, 3), f"90% CI (t, df={n - 1})": f"[{b - t * se:.3f}, {b + t * se:.3f}]",
                     "p two-sided": round(2 * stats.t.sf(abs(b / se), n - 1), 4) if se > 0 else float("nan"),
                     "p TOST": round(pt, 4), "per task": " ".join(f"{v:+.2f}" for v in d.values)})
    return pd.DataFrame(rows)


P("\n### Conservative: per-task contrasts, t over the 6 tasks\n")
P(per_task(pre).to_markdown(index=False))

# falsifier: fx_pajM vs Full f42 r0+r1
fz = pd.concat([pre[pre.arm_id == "fx_pajM"].assign(g=1), full[(full.fold_seed == 42) & full.replicate.isin([0, 1])].assign(g=0)])
m = smf.ols("y ~ g + C(dataset_id)", data=fz).fit()
b, se, dfr = m.params["g"], m.bse["g"], m.df_resid
t = stats.t.ppf(0.95, dfr)
P(f"\n### Falsifier fx_pajM - Full (f42 r0+r1): Delta={b:.3f}, 90% CI [{b - t * se:.3f}, {b + t * se:.3f}], df={int(dfr)}; "
  f"fx_pajM {int(pre[pre.arm_id == 'fx_pajM'].k.sum())}/30, Full {int(fz[fz.g == 0].k.sum())}/60")
fpt = fz.groupby(["dataset_id", "g"]).y.mean().unstack()
P(fpt.round(2).to_markdown())

# tokens
tk = pre.dropna(subset=["tok_compl_total"]).copy()
P(f"\n### Tokens (log tok_compl_total, task FE; {len(tk)} runs with instrumentation)\n")
rows = []
for col in ["tok_compl_total", "tok_compl_tester", "tok_compl_tester_assess", "tok_compl_tester_prune", "tok_compl_scientist"]:
    tk["lt"] = np.log(tk[col].astype(float) + 1)
    m = smf.ols("lt ~ C(dataset_id) + P + A + J + M + PA + PJ + PM", data=tk).fit()
    for f in "PAJM":
        r = np.exp(-2 * m.params[f]); t = stats.t.ppf(0.95, m.df_resid)
        rows.append({"outcome": col, "factor": f, "ratio X-/X+": round(r, 3),
                     "90% CI": f"[{np.exp(-2 * (m.params[f] + t * m.bse[f])):.3f}, {np.exp(-2 * (m.params[f] - t * m.bse[f])):.3f}]"})
P(pd.DataFrame(rows).to_markdown(index=False))
P("\nMean tok_compl_total per cell (M):")
P((tk.groupby("arm_id").tok_compl_total.mean() / 1e6).reindex(list(CELLS)).round(2).to_markdown())

# sensitivity: + Amendment 11 rows (f4 r0) on p2, p4, p6b, with a seed block
sens = fx[fx.dataset_id.isin(POP)].copy()
sens["blk"] = sens.fold_seed.astype(str)
P(f"\n### Sensitivity: + A12-801..824 (f4 r0), {len(sens)} runs, block = fold seed\n")
P(lpm(sens, " + C(blk)").to_markdown(index=False))
s8 = fx[~fx.prereg]
P("\nA12-801..824 per cell (k/5):")
P(s8.pivot_table(index="arm_id", columns="dataset_id", values="k", aggfunc="sum").reindex(list(CELLS)).to_markdown())
P("\nFull f4 r0 / r1 / f42 r2 on the same tasks (k/5):")
P(full[full.fold_seed.eq(4) | full.replicate.eq(2)][["row_id", "dataset_id", "fold_seed", "replicate", "k"]].to_markdown(index=False))
(D / "A12_prelim.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
