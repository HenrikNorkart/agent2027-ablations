"""Figures for the aspect-by-aspect AGENT 2027 paper (IEEE, v2 structure).

Every number in these figures is read from the final read-out (docs/agent2027/results/final_20261102); nothing is typed
by hand. Writes <paper>/figures_v2/{planner,knowledge,cross}.tex.

    python make_ieee_v2.py --results <final_20261102 dir> --paper <paper dir>

Sources
  pareto_front.csv  (registered cost analysis, A23 / prereg V.23): panel A, axis tok_compl_total. Population = the six
                    confirmatory SYN-P tasks at fold seed 42; Full = two runs per task, every other configuration one.
                    Paired two-stage cluster bootstrap (tasks, then folds), B = 2000: 95% CI of each recovery rate,
                    90% CI of the difference vs Full, 90% CI of the completion-token ratio vs Full.
  episodes_final.csv  fold-episode recovery (knowledge probe on P7; P6a vs its anonymised twin P6b).
"""
import argparse
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--paper", required=True)
a = ap.parse_args()
R, P = Path(a.results), Path(a.paper)
(P / "figures_v2").mkdir(exist_ok=True)

pf = pd.read_csv(R / "pareto_front.csv")
pa = pf[(pf.variant == "primary") & (pf.panel == "A") & (pf.axis == "tok_compl_total")].set_index("config")
ep = pd.read_csv(R / "episodes_final.csv")
DELTA = 0.167


def f3(x):
    return f"{x:.3f}"


# ---------------------------------------------------------------- planner: quality of the hypothesis source
rows = [("misdirected", "misdirected"), ("fx_pAJM", "no planner"), ("full", "LLM planner (Full)"), ("oracle", "oracle")]
coords = " ".join(f"({i},{f3(pa.loc[c, 'rec_sel'])}) +- (0,0)" for i, (c, _) in enumerate(rows))
err = " ".join(
    f"({i},{f3(pa.loc[c, 'rec_sel'])}) += (0,{f3(pa.loc[c, 'rec_hi95'] - pa.loc[c, 'rec_sel'])}) -= (0,{f3(pa.loc[c, 'rec_sel'] - pa.loc[c, 'rec_lo95'])})"
    for i, (c, _) in enumerate(rows))
(P / "figures_v2" / "planner.tex").write_text(r"""\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[width=\columnwidth,height=3.6cm,ybar,bar width=14pt,ymin=0,ymax=1.08,ylabel={recovery},
  xtick={0,1,2,3},xticklabels={""" + ",".join("{" + l + r"\\" + f"{pa.loc[c, 'rec_sel']:.2f}" + "}" for c, l in rows) + r"""},
  xticklabel style={font=\scriptsize,align=center},yticklabel style={font=\scriptsize},enlarge x limits=0.15,grid=major,grid style={gray!20}]
\addplot[fill=blue!35,draw=blue!60!black,error bars/.cd,y dir=both,y explicit] coordinates {""" + err + r"""};
\end{axis}
\end{tikzpicture}
\caption{Planner (A1): the loop recovers what its planner points at. Six tasks, fold seed 42, one run per task (two for Full); exploratory; 95\% bootstrap CI.}
\label{fig:v2_planner}
\end{figure}
""", encoding="utf-8")

# ---------------------------------------------------------------- knowledge: P7 corpus arms, named vs anonymised
kb = ep[ep.ablation_id == "A14"].copy()
kb = kb[kb.row_id.str.match(r"A14-0")]  # rows of record (smoke/probe rows excluded)
kk = kb.groupby("arm_id").rec_sel.agg(["sum", "count"])
f6 = ep[(ep.ablation_id == "A05") & (ep.arm_id == "full") & ep.dataset_id.isin(["synp_p6a", "synp_p6b"])]
runs6a = f6[f6.dataset_id == "synp_p6a"][["fold_seed", "replicate"]].drop_duplicates()
f6 = f6.merge(runs6a, on=["fold_seed", "replicate"])
k6 = f6.groupby("dataset_id").rec_sel.agg(["sum", "count"])
bars = [("kb_off", "no retrieval"), ("kb_parametric", "recall only"), ("kb_local_wrong", "wrong doc."), ("kb_local_correct", "correct doc.")]
kc = " ".join(f"({i},{f3(kk.loc[c, 'sum'] / kk.loc[c, 'count'])})" for i, (c, _) in enumerate(bars))
kn = " ".join(f"({i},{int(kk.loc[c, 'sum'])}/{int(kk.loc[c, 'count'])})" for i, (c, _) in enumerate(bars))
nc = f"(0,{f3(k6.loc['synp_p6b', 'sum'] / k6.loc['synp_p6b', 'count'])}) (1,{f3(k6.loc['synp_p6a', 'sum'] / k6.loc['synp_p6a', 'count'])})"
lab = lambda c, k: f"{int(k.loc[c, 'sum'])}/{int(k.loc[c, 'count'])}"
(P / "figures_v2" / "knowledge.tex").write_text(r"""\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[name=l,width=0.64\columnwidth,height=3.5cm,ybar,bar width=10pt,ymin=0,ymax=1.22,ytick={0,0.5,1},ylabel={recovery},
  xtick={0,1,2,3},xticklabels={""" + ",".join(l for _, l in bars) + r"""},xticklabel style={font=\tiny,align=center,text width=1.0cm},
  yticklabel style={font=\scriptsize},enlarge x limits=0.15,title={\scriptsize (a) rule only in a document (P7)},title style={yshift=-4pt},grid=major,grid style={gray!20}]
\addplot[fill=teal!40,draw=teal!60!black] coordinates {""" + kc + r"""};
""" + "".join(rf"\node[font=\tiny,above] at (axis cs:{i},{f3(kk.loc[c, 'sum'] / kk.loc[c, 'count'])}) {{{lab(c, kk)}}};" + "\n" for i, (c, _) in enumerate(bars)) + r"""\end{axis}
\begin{axis}[at={(l.east)},anchor=west,xshift=0.25cm,width=0.44\columnwidth,height=3.5cm,ybar,bar width=10pt,ymin=0,ymax=1.22,ytick={0,0.5,1},
  xtick={0,1},xticklabels={anonymized,named},xticklabel style={font=\tiny},yticklabels={},enlarge x limits=0.4,
  title={\scriptsize (b) column names (P6)},title style={yshift=-4pt},grid=major,grid style={gray!20}]
\addplot[fill=orange!45,draw=orange!70!black] coordinates {""" + nc + r"""};
\node[font=\tiny,above] at (axis cs:0,""" + f3(k6.loc['synp_p6b', 'sum'] / k6.loc['synp_p6b', 'count']) + r""") {""" + lab("synp_p6b", k6) + r"""};
\node[font=\tiny,above] at (axis cs:1,""" + f3(k6.loc['synp_p6a', 'sum'] / k6.loc['synp_p6a', 'count']) + r""") {""" + lab("synp_p6a", k6) + r"""};
\end{axis}
\end{tikzpicture}
\caption{Domain knowledge (A3). (a) The rule is recovered only with the correct document. (b) Domain-specific column names increase recovery (matched runs of the complete system).}
\label{fig:v2_knowledge}
\end{figure}
""", encoding="utf-8")

# ---------------------------------------------------------------- cross-ablation: one row per single-aspect ablation
cross = [("fx_pAJM", "A1 no planner"), ("fx_PAJm", "A2 no memory"), ("fx_PaJM", "A4 template report"),
         ("fx_PAjM", "A5 rule-based selection"), ("restarts", "A6 independent restarts"), ("l0", "A6 open-loop planner"),
         ("set_phased", "A7 single agent"), ("llmfe", "LLM-FE (evolutionary)"), ("fx_Pajm", "planner only")]
n = len(cross)
ylab = ",".join("{" + l + "}" for _, l in reversed(cross))
pt = []
pr = []
for j, (c, _) in enumerate(reversed(cross)):
    r = pa.loc[c]
    pt.append(f"({f3(r.diff_vs_full)},{j}) -= ({f3(r.diff_vs_full - r.diff_lo90)},0) += ({f3(r.diff_hi90 - r.diff_vs_full)},0)")
    pr.append(f"({f3(r.ratio_vs_full)},{j}) -= ({f3(r.ratio_vs_full - r.ratio_lo90)},0) += ({f3(r.ratio_hi90 - r.ratio_vs_full)},0)")
sep = len([x for x in cross if not x[1].startswith("A")]) - 0.5  # reference points are the bottom rows
yh = n - 0.5
(P / "figures_v2" / "cross.tex").write_text(r"""\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[name=d,width=0.56\columnwidth,height=5.4cm,scale only axis=false,xmin=-0.5,xmax=0.5,ymin=-0.6,ymax=""" + f"{yh:.1f}" + r""",
  ytick={""" + ",".join(str(j) for j in range(n)) + r"""},yticklabels={""" + ylab + r"""},yticklabel style={font=\scriptsize},
  xtick={-0.4,-0.2,0,0.2,0.4},xticklabel style={font=\tiny},
  xlabel={$\Delta$ recovery},xlabel style={font=\scriptsize},grid=major,grid style={gray!20}]
\fill[green!10] (axis cs:-0.167,-0.6) rectangle (axis cs:0.167,""" + f"{yh:.1f}" + r""");
\draw[gray] (axis cs:0,-0.6) -- (axis cs:0,""" + f"{yh:.1f}" + r""");
\draw[gray,dashed] (axis cs:-0.5,""" + f"{sep:.1f}" + r""") -- (axis cs:0.5,""" + f"{sep:.1f}" + r""");
\addplot[only marks,mark=*,mark size=1.6pt,blue!70!black,error bars/.cd,x dir=both,x explicit] coordinates {""" + " ".join(pt) + r"""};
\end{axis}
\begin{axis}[at={(d.east)},anchor=west,xshift=0.2cm,width=0.40\columnwidth,height=5.4cm,xmode=log,xmin=0.08,xmax=2,ymin=-0.6,ymax=""" + f"{yh:.1f}" + r""",
  ytick={""" + ",".join(str(j) for j in range(n)) + r"""},yticklabels={},xtick={0.1,0.5,1,2},xticklabels={0.1,0.5,1,2},xticklabel style={font=\tiny},
  xlabel={relative tokens},xlabel style={font=\scriptsize},grid=major,grid style={gray!20}]
\draw[gray] (axis cs:1,-0.6) -- (axis cs:1,""" + f"{yh:.1f}" + r""");
\draw[gray,dashed] (axis cs:0.08,""" + f"{sep:.1f}" + r""") -- (axis cs:2,""" + f"{sep:.1f}" + r""");
\addplot[only marks,mark=square*,mark size=1.4pt,orange!80!black,error bars/.cd,x dir=both,x explicit] coordinates {""" + " ".join(pr) + r"""};
\end{axis}
\end{tikzpicture}
\caption{Single-aspect ablations, descriptive (A1, A2, A4--A7; reference points below the dashed line). Left: difference in recovery from the complete system, 90\% CI; the band marks $\pm\delta$ for scale only. Right: completion tokens relative to the complete system, 90\% CI, log scale. Fold seed 42: 60 fold-episodes for the complete system and LLM-FE, 30 per other configuration. Planner only: report, selection and memory all replaced.}
\label{fig:v2_cross}
\end{figure}
""", encoding="utf-8")
print("figures_v2 written")
print(kk)
print(k6)
print(pa.loc[[c for c, _ in cross] + ["full", "misdirected", "oracle"], ["rec_sel", "rec_lo95", "rec_hi95", "diff_vs_full", "diff_lo90", "diff_hi90", "ratio_vs_full", "ratio_lo95", "ratio_hi95", "cost"]].round(3).to_string())

# ladder (A8): written by make_ladder_fig.py from the latest read-out (final_20261112); not generated here.

