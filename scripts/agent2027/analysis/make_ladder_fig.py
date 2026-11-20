"""Fig. 4 of the AGENT 2027 IEEE paper (v2): the Qwen3.5 model-size ladder (A8 / A17), with re-runs (Amendment 19h).

Both plotted series use the exploratory crash-guarded re-runs (Amendments 19-19g): every A17 4xx row of record whose
run crashed or hit the cap is replaced by its latest re-run (final_tests.rerun_map, as recorded in
final_readout.json -> contrasts.C5a.inf_guard.replaced). This is a display decision only (Amendment 19h): the
confirmatory C5a test and the Holm family stay on the rows of record, which the text reports.

  functional success  contrasts.C5a.inf_guard.by_rung (run rule 3 of 5 folds)
  recovery            rec_sel over the fold-episodes of the same rows (episodes_final.csv, latest episode per fold)
  red line            gpt-oss-120b (A05 Full, replicate 0) recovery on the same seven SYN-P tasks

    python make_ladder_fig.py --in docs/agent2027/results/final_20261110 --paper paper/agent2027
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="ind", required=True)
ap.add_argument("--paper", required=True)
a = ap.parse_args()
I, P = Path(a.ind), Path(a.paper)

SYNP7 = [f"synp_p{t}" for t in ("1", "2", "3", "4", "5", "6a", "6b")]
RUNGS = ["f2_Q0.8", "f2_Q2", "f2_Q4", "f2_Q9", "f2_Q122"]
LAB = {"f2_Q0.8": "0.8B", "f2_Q2": "2B", "f2_Q4": "4B", "f2_Q9": "9B", "f2_Q122": "122B-A10B"}

c5 = json.loads((I / "final_readout.json").read_text())["contrasts"]["C5a"]
ig = c5["inf_guard"]
rep, fs = ig["replaced"], ig["by_rung"]
lad = c5.get("seed4_ladder")  # Amendments 19i/19j: Q2 and Q9 at fold seed 4 as well, 14 runs per rung
if lad:
    fs = lad["by_rung"]
ep = pd.read_csv(I / "episodes_final.csv").drop_duplicates(["row_id", "fold"], keep="last")
num = ep.row_id.str.split("-").str[1].astype(int)
rec17 = ep[(ep.ablation_id == "A17") & num.between(400, 499) & ep.dataset_id.isin(SYNP7)]

fsp, rsp, nruns, out = [], [], {}, {}
for x, r in enumerate(RUNGS):
    k, n = map(int, fs[r].split("/"))
    if lad:
        sel_rows = lad["rows"][r]
    else:
        sel_rows = [rep.get(i, i) for i in sorted(set(rec17[rec17.arm_id == r].row_id))]
    rows = sel_rows
    e = ep[ep.row_id.isin(sel_rows)]
    assert len(rows) == n and len(e) == 5 * n, (r, len(rows), n, len(e))
    nruns[r] = n
    fsp.append(f"({x},{k / n:.3f})")
    rsp.append(f"({x},{e.rec_sel.mean():.3f})")
    out[r] = (f"{k}/{n}", f"{int(e.rec_sel.sum())}/{len(e)}")

full = ep[(ep.ablation_id == "A05") & (ep.arm_id == "full") & (ep.replicate == 0) & ~ep.row_id.str.contains("-9")
          & ep.dataset_id.isin(SYNP7)].rec_sel.mean()
CAPTION_HEAD = ("In each run, the Planner, the Coder, and the Tester all run on the same Qwen3.5 model; the mixed configurations are not shown. Functional success (an accepted LLM feature in at least 3 of 5 folds) and recovery per model; the red line marks the recovery of gpt-oss-120b in every role (the complete system)." if lad else "Functional success (an accepted LLM feature in at least 3 of 5 folds) increases with the number of active parameters; recovery remains below that of gpt-oss-120b (red line).")
CAPTION_TAIL = ("Seven tasks (P1 included) at fold seeds 42 and 4, 14 runs per model, including exploratory runs." if lad else "Runs that crashed or hit the cap are replaced by their exploratory crash-guarded re-runs (all seven 9B runs crashed in the runs of record; see text). Seven tasks, P1 included.")
ticks = ",".join(f"{LAB[r]} ({nruns[r]})" for r in RUNGS)

tex = r"""\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[width=\columnwidth,height=4.0cm,xlabel={Qwen3.5 model of all three agents (runs), ordered by active parameters},ylabel={rate},ymin=0,ymax=1.05,
  xmin=-0.4,xmax=4.4,xtick={0,1,2,3,4},xticklabels={""" + ticks + r"""},xticklabel style={font=\scriptsize},
  legend style={font=\scriptsize,at={(0.5,-0.42)},anchor=north,legend columns=3},grid=major,grid style={gray!20}]
\addplot+[mark=*,thick] coordinates {""" + " ".join(fsp) + r"""};
\addlegendentry{functional success}
\addplot+[mark=square*,thick,dashed] coordinates {""" + " ".join(rsp) + r"""};
\addlegendentry{recovery}
\addplot[red,thick,densely dotted,domain=-0.4:4.4] {""" + f"{full:.3f}" + r"""};
\addlegendentry{gpt-oss-120b recovery}
\end{axis}
\end{tikzpicture}
\caption{Model size (A8). """ + CAPTION_HEAD + " " + CAPTION_TAIL + r"""}
\label{fig:v2_ladder}
\end{figure}
"""
(P / "figures_v2" / "ladder.tex").write_text(tex, encoding="utf-8")
print("ladder (functional success, recovery) with re-runs:", out, f"gpt-oss-120b {full:.3f}")
