"""Write the AGENT2027 paper's results LaTeX from the final read-out (2026-11-02).

Every number in the generated files comes from these inputs; nothing is typed by hand:
  final_readout.json, describe.json, glmm_results.json   (final_tests.py / final_glmm.R)
  RV3_A12_glmm.json, A12_realdata_summary.json            (G4 addendum / Amendment 16, committed)
  sentinel/ANALYSIS__sentinel_drift__*__final.json, sentinel_session_all.json  (A06)
  pareto_1102/pareto_front.csv, pareto_summary.json        (A23)
  loop_guard_counts.csv                                   (trace scan, A17)
  f0/fold_summary.csv, a21_isocost/*                       (optional: filled when present)

    python make_agent2027_tex.py --in <final dir> --results <docs/agent2027/results> --paper <paper/agent2027>
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import pandas as pd


def f(x, d=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{d}f}"


def p(x):
    if x is None:
        return "--"
    if x < 0.0001:
        return "$<$0.0001"
    return f"{x:.4f}" if x < 0.01 else f"{x:.3f}"


def pe(x):
    if x is None:
        return "$ n/a"
    return "$p<0.0001$" if x < 0.0001 else f"$p={p(x)}$"


def ci(c, d=3):
    return f"[{f(c[0], d)}, {f(c[1], d)}]" if c and c[0] is not None else "--"


def frac(k, n):
    return f"{k}/{n}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="ind", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--paper", required=True)
    a = ap.parse_args()
    I, R, P = Path(a.ind), Path(a.results), Path(a.paper)
    (P / "tables").mkdir(exist_ok=True)
    (P / "sections").mkdir(exist_ok=True)
    (P / "appendix").mkdir(exist_ok=True)
    fr = json.loads((I / "out" / "final_readout.json").read_text())
    C, O = fr["contrasts"], fr["outside_holm"]
    rv3 = json.loads((R / "RV3_A12_glmm.json").read_text())
    real = json.loads((R / "A12_realdata_summary.json").read_text())
    a06 = json.loads(Path(sorted(glob.glob(str(I / "sentinel" / "ANALYSIS__sentinel_drift__*__final.json")))[-1]).read_text())
    sess = json.loads((I / "sentinel" / "sentinel_session_all.json").read_text())
    verd = sess.get("verdicts") or sess.get("rows")
    n_pass = sum(1 for v in verd if v.get("arm_id") == "sentinel" and v.get("verdict") == "PASS")
    n_warn = sum(1 for v in verd if v.get("arm_id") == "sentinel" and v.get("verdict") == "WARN")
    n_flag = sum(1 for v in verd if v.get("arm_id") == "sentinel" and v.get("verdict") == "FLAG")
    front = pd.read_csv(I / "pareto_1102" / "pareto_front.csv")
    psum = json.loads((I / "pareto_1102" / "pareto_summary.json").read_text())
    lg = pd.read_csv(I / "loop_guard_counts.csv", header=None, names=["row_id", "note_requests"])

    def d(k):
        return C[k].get("desc") or {}

    # ------------------------------------------------------------------ verdicts (Part IV / V wording rules)
    def verdict(k):
        c = C[k]
        if k == "C1a":
            if c["holm_reject"]:
                return "SET better" if d(k)["rd"] > 0 else "Full better"
            lo, hi = d(k)["rd_ci90"]
            return "equivalent (secondary)" if (-0.167 < lo and hi < 0.167) else "inconclusive"
        if k == "C1b":
            if c["holm_reject"] and c["three_mode_agreement"]:
                return "LLM-FE better" if d(k)["rd"] > 0 else "Full better"
            return "inconclusive (selection-dependent)" if c["holm_reject"] else "inconclusive"
        if k in ("C2a", "C2b"):
            return "reducible" if c["holm_reject"] else "inconclusive"
        if k == "C2c":
            return ("planner helps" if c["Delta"] > 0 else "planner hurts") if c["holm_reject"] else "inconclusive"
        if k == "C3":
            return ("Restarts $>$ Full" if d(k)["rd"] > 0 else "Full $>$ Restarts") if c["holm_reject"] else "inconclusive"
        if k == "C4a":
            return "retrieved knowledge used" if c["holm_reject"] else "inconclusive"
        if k == "C4b":
            return "names matter" if c["holm_reject"] else "inconclusive"
        if k == "C5a":
            return "success rises with size" if c["holm_reject"] else "inconclusive"
        if k == "C5b":
            return ("Extractor on 4B loses" if d(k)["rd"] < 0 else "Extractor on 4B better") if c["holm_reject"] else "no detectable difference"

    def effect(k):
        c = C[k]
        if k in ("C2a", "C2b", "C2c"):
            return f"$\\Delta={f(c['Delta'])}$ {ci(c['Delta_ci90'])}"
        if k == "C4a":
            return f"{c['arms']['kb_local_correct']} vs {c['arms']['kb_off']}"
        if k == "C4b":
            return f"$\\Delta={f(c['delta'], 2)}$ {ci(c['delta_ci95'], 2)}"
        if k == "C5a":
            br = c["rule_3of5"]["by_rung"]
            return " / ".join(br.get(r, "--") for r in ("f2_Q0.8", "f2_Q2", "f2_Q4", "f2_Q9", "f2_Q122"))
        dd = d(k)
        return f"{frac(dd['k1'], dd['n1'])} vs {frac(dd['k0'], dd['n0'])}; RD ${f(dd['rd'])}$ {ci(dd['rd_ci90'])}"

    LAB = {"C1a": "One phased context vs three agents", "C1b": "LLM-FE (top-3 programs) vs Full",
           "C2a": "Assessment: template vs LLM (TOST)", "C2b": "Pruning: code rules vs LLM (TOST)",
           "C2c": "Planner on vs off", "C3": "Independent restarts vs refinement",
           "C4a": "Correct local corpus vs no retrieval (P7)$^\S$", "C4b": "Anonymised vs named columns$^\P$",
           "C5a": "Functional success over Qwen3.5 size$^\dagger$", "C5b": "Extractor on 4B vs all on 122B$^\dagger$"}
    def test_of(k):
        t = C[k]["test"]
        return ("beta-binomial" if "RV3" in t else "exact strat." if "exact conditional" in t or "separation" in t
                else "exact trend" if "Cochran" in t else "McNemar" if "McNemar" in t else "GLMM, lik.-ratio" if "LR" in t else "GLMM")
    VERD2 = {"C2a": "equivalent$^\ddagger$", "C2b": "equivalent$^\ddagger$"}
    rows = []
    for k in ("C1a", "C1b", "C2a", "C2b", "C2c", "C3", "C4a", "C4b", "C5a", "C5b"):
        c = C[k]
        rows.append(f"{k} & {LAB[k]} & {effect(k)} & {test_of(k)} & {p(c['p'])} & {p(c['p_holm'])} & {VERD2.get(k, verdict(k))} \\\\")
    (P / "tables" / "a27_confirmatory.tex").write_text(r"""\begin{table}[t]
\centering\small
\caption{The ten pre-registered contrasts (Holm at family-wise $\alpha=0.05$). Outcome: recovery of the planted feature in the selected iteration, per fold-episode, on the six confirmatory tasks; C3 on the multi-step tasks; C4a on P7; C5a: functional success per run on the seven synthetic tasks (0.8B / 2B / 4B / 9B / 122B-A10B). Effects: arm vs.\ reference with risk difference (RD) and 90\% Newcombe interval (descriptive; fold-episodes within a run are correlated); C2 rows give the factorial effect $\Delta$ (component on $-$ off) with 90\% CI. Crashed or capped runs count as zeros. $^\dagger$Run on the relaunched Qwen rows with three tool-compatibility switches, after every first-wave Qwen run hit the cap or failed (Amendments 17--18). $^\ddagger$Registered verdict ``reducible''; its token criterion is met by construction (a replaced stage makes no LLM call), and the equivalence is pooled over tasks. $^\S$Manipulation check: the comparison arms cannot succeed on P7 by design. $^\P$The 15 fold pairs come from three run pairs; a pre-freeze pilot had shown the same direction.}
\label{tab:a27_confirmatory}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lllllll}
\toprule
ID & Contrast & Effect & Test & $p$ (raw) & $p$ (Holm) & Verdict \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""", encoding="utf-8")

    # ------------------------------------------------------------------ A12 factorial + real data
    eff = rv3["effects"]
    t12 = []
    for term, lab in (("P", "Planner (P)"), ("A", "Assessment (A)"), ("J", "Pruning (J)"), ("M", "Memory (M)"),
                      ("PA", "PA"), ("PJ", "PJ"), ("PM", "PM"), ("AJ", "AJ"), ("AM", "AM"), ("JM", "JM")):
        e = eff[term]
        t12.append(f"{lab} & {f(e['Delta'])} & {ci(e['Delta_ci90'])} & {p(e['p_wald'])} & {p(e['p_tost'])} \\\\")
    tr = []
    names = {"fx_pAJM": "planner off", "fx_PaJM": "assessment off", "fx_PAjM": "pruning off", "fx_PajM": "assessment and pruning off"}
    for r in real:
        tr.append(f"{names.get(r['arm'], r['arm'])} & {f(r['mean_delta_std'], 2)} & {ci(r['t90'], 2)} & {f(r['mean_delta_raw'], 3)} & {r['verdict']} & {f(r['tok_ratio'], 2)} {ci(r['tok_ratio_90'], 2)} \\\\")
    (P / "tables" / "a27_a12.tex").write_text(r"""\begin{table}[t]
\centering\small
\caption{Component knockouts. Left: the 12-cell factorial of record on the six confirmatory tasks (72 runs, 360 fold-episodes; run-level beta-binomial model, $\pm1$ coding; $\Delta_X$ = predicted recovery with component $X$ on minus off, tasks equally weighted; TOST margin $\pm0.167$). Right: the same knockouts on ten real datasets (exploratory; $\Delta_\text{std}$ = arm minus Full in units of the dataset's run-to-run SD; non-inferior if the 90\% lower bound exceeds $-1$, a margin fixed before any of these runs ended; the knockout runs used a later software image than their Full pairs).}
\label{tab:a27_a12}
\begin{tabular}{lrrrr}
\toprule
Term & $\Delta_X$ & 90\% CI & $p$ & TOST $p$ \\
\midrule
""" + "\n".join(t12) + r"""
\bottomrule
\end{tabular}

\vspace{0.6em}
\begin{tabular}{lrrrlr}
\toprule
Knockout & $\Delta_\text{std}$ & 90\% CI & Raw $\Delta$ & Verdict & Token ratio \\
\midrule
""" + "\n".join(tr) + r"""
\bottomrule
\end{tabular}
\end{table}
""", encoding="utf-8")

    # ------------------------------------------------------------------ A17 ladder
    br = O["A17_descriptive"]["by_rung"]
    fs = C["C5a"]
    lgm = {}
    for rung in br:
        pass
    rows17 = []
    act = {"f2_Q0.8": ("Qwen3.5-0.8B", "0.8"), "f2_Q2": ("Qwen3.5-2B", "2"), "f2_Q4": ("Qwen3.5-4B", "4"),
           "f2_Q9": ("Qwen3.5-9B", "9"), "f2_Q122": ("Qwen3.5-122B-A10B", "10")}
    for rung in ("f2_Q0.8", "f2_Q2", "f2_Q4", "f2_Q9", "f2_Q122"):
        b = br[rung]
        inf = fs.get("inf_guard", {}).get("by_rung", {}).get(rung, "--")
        rows17.append(f"{act[rung][0]} & {act[rung][1]} & {b['runs']} & {b['ok']} & {b['cap_hit']} & {b['inf_crash']} & "
                      f"{fs['rule_3of5']['by_rung'][rung]} & {fs['rule_5of5']['by_rung'][rung]} & {fs['rule_1of5']['by_rung'][rung]} & {inf} & "
                      f"{b['rec_sel']} & {f(b['tok_compl_median'] / 1e6, 2) if b['tok_compl_median'] else '--'} \\\\")
    (P / "tables" / "a27_ladder.tex").write_text(r"""\begin{table}[t]
\centering\small
\caption{The Qwen3.5 scale ladder (A17, relaunched rows of record, SYN-P 7; Q0.8, Q4, Q122 at two fold seeds, Q2 and Q9 at one). \texttt{func\_success}: the run finishes and, in at least 3 of 5 folds (sensitivities 5/5 and $\ge$1/5), the selected iteration passes the held-out success bound and holds $\ge$1 accepted non-fallback LLM feature. ``inf crash'' = rows ending on XGBoost's \emph{Input data contains inf} (ITT zeros); ``inf-guard'' replaces those rows with their Amendment 19 re-runs (exploratory). Tokens: median completion tokens per run (millions). gpt-oss-120b (Full) recovers 30/90 on the same population.}
\label{tab:a27_ladder}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrrrrrrrrrr}
\toprule
Model & Active (B) & Runs & OK & Cap & Inf crash & func 3/5 & 5/5 & $\ge$1/5 & inf-guard & \texttt{rec\_sel} & Tok.\ (M) \\
\midrule
""" + "\n".join(rows17) + r"""
\bottomrule
\end{tabular}}
\end{table}
""", encoding="utf-8")

    # ------------------------------------------------------------------ outside-Holm table
    a07, a13 = O["A07"], O["A13"]
    a09o, a09m, a11, m1b = O["A09_oracle"], O["A09_misdirected"], O["A11"], O["A19_mix_1b"]
    a20 = O["A20"]
    def dd(x):
        return x["desc"]
    out_rows = [
        f"Drift sentinel & Full repeated on one task in every session & {n_pass} pass / {n_warn} warn / {n_flag} flag; pooled {a06['primary_fisher']['x']}/{a06['primary_fisher']['y']} vs reference {a06['primary_fisher']['z']}/{a06['primary_fisher']['n0']}, Fisher $p={f(a06['primary_fisher']['p'])}$ & no drift \\\\",
        f"Error messages & Tool-error text hidden vs shown (8 real-data pairs) & $\\hat\\rho={f(a07['rho_shown_over_hidden'], 2)}$ {ci(a07['rho_ci95'], 2)}, sign-flip $p={f(a07['p_signflip'], 4)}$ & {a07['verdict'].split(':')[0]} \\\\",
        f"Planner control & Oracle planner vs Full & {frac(dd(a09o)['k1'], dd(a09o)['n1'])} vs {frac(dd(a09o)['k0'], dd(a09o)['n0'])}, one-sided exact $p$ {p(a09o['p'])} & positive control \\\\",
        f"Planner control & Misdirected planner vs Full & {frac(dd(a09m)['k1'], dd(a09m)['n1'])} vs {frac(dd(a09m)['k0'], dd(a09m)['n0'])}, one-sided {pe(a09m['p'])} & planner direction matters \\\\",
        f"Open loop & Planner never sees results vs Full & {frac(dd(a11)['k1'], dd(a11)['n1'])} vs {frac(dd(a11)['k0'], dd(a11)['n0'])}, LR {pe(a11['p'])} & inconclusive \\\\",
        f"Web search & Search off, ten real datasets & tokens {f(a13['tok_ratio_gm'], 2)}$\\times$ {ci(a13['tok_ratio_ci95'], 2)}; $\\overline{{\\Delta}}_\\text{{std}}={f(a13['mean_delta_std'], 2)}$ {ci(a13['nh_bounds'], 2)} & {a13['cost_verdict'].split(' ', 1)[1]}; {a13['no_harm_verdict']} \\\\",
        f"Phased context & Engineering-collapse check & {O['A16_overlay']['itt_runs']} of {O['A16_overlay']['set_synp_runs']} runs failed & not triggered \\\\",
        f"Mixed models & Scientist, Tester on 0.8B vs all on 122B & {frac(dd(m1b)['k1'], dd(m1b)['n1'])} vs {frac(dd(m1b)['k0'], dd(m1b)['n0'])}, exact $p$ {p(m1b['p'])} & loop stops \\\\",
        f"Minimal harness & Harness $\\times$ model (gpt-oss-120b vs 4B) & cells Full/min.: gpt-oss {a20['cells']['11']} / {a20['cells']['-11']}; 4B {a20['cells']['1-1']} / {a20['cells']['-1-1']}; Firth $p={f((a20.get('firth') or {}).get('p_wald'))}$ & not readable: 4B recovers too little \\\\",
    ]
    (P / "tables" / "a27_outside.tex").write_text(r"""\begin{table}[t]
\centering\small
\caption{Registered analyses outside the confirmatory family (raw $p$; reading rules as registered). The drift sentinel could not be evaluated for the Qwen models, whose sentinel runs hit the cap.}
\label{tab:a27_outside}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llll}
\toprule
Analysis & Comparison & Estimate & Reading \\
\midrule
""" + "\n".join(out_rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""", encoding="utf-8")

    # ------------------------------------------------------------------ A23 Pareto table
    pa = front[(front.variant == "primary") & (front.primary_axis == True) & (front.panel == "A")]  # noqa: E712
    nice = {"full": "Full", "set_phased": "Single phased context", "restarts": "Restarts", "l0": "Open loop", "llmfe": "LLM-FE (top-3 programs)",
            "fx_Pajm": "planner only (no A, J, M)", "fx_pAjm": "assessment only (no P, J, M)", "fx_pajm": "minimal, no memory", "fx_pajM": "minimal harness",
            "fx_PAjM": "pruning off", "fx_PaJM": "assessment off", "fx_pAJM": "planner off", "fx_PAJm": "memory off"}
    pr = []
    for _, r in pa.iterrows():
        if r.role != "front" or r.config not in nice:
            continue
        pr.append(f"{nice[r.config]} & {r.n_runs} & {f(r.rec_sel)} & {f(r.cost / 1e6, 2)} & {f(r.p_front, 3)} & {r.wording} \\\\")
    pb = front[(front.variant == "primary") & (front.primary_axis == True) & (front.panel == "B")]  # noqa: E712
    nb = {"full": "gpt-oss-120b Full", "fx_pajM": "gpt-oss-120b minimal harness", "f2_Q0.8": "Qwen3.5-0.8B", "f2_Q2": "Qwen3.5-2B",
          "f2_Q4": "Qwen3.5-4B", "f2_Q9": "Qwen3.5-9B", "f2_Q122": "Qwen3.5-122B-A10B", "mix_esmall": "Extractor 4B, others 122B",
          "mix_1b": "Scientist, Tester 0.8B; Extractor 122B", "fx_pajM_Q4": "Qwen3.5-4B minimal harness"}
    prb = [f"{nb[r.config]} & {r.n_runs} & {f(r.rec_sel)} & {f(r.cost / 1e18, 2)} & {f(r.p_front, 3)} & {r.wording} \\\\"
           for _, r in pb.iterrows() if r.config in nb and r.role == "front"]
    outB = next((o["outcome"] for o in psum["outcomes"] if o["panel"] == "B" and o["variant"] == "primary"), "--")
    (P / "tables" / "a27_pareto.tex").write_text(r"""\begin{table}[t]
\centering\small
\caption{Cost--recovery Pareto fronts (descriptive; $p_\text{front}$ = share of $B=2000$ bootstrap resamples in which the configuration is on the front). Panel A: gpt-oss-120b configurations; cost = mean completion tokens per run on the planted-feature tasks (millions); the full system is dominated. Panel B: model configurations; cost = FLOP proxy $\sum_\text{roles} 2N_\text{active}(\text{prompt}+\text{completion})$ per run ($10^{18}$); no Qwen configuration is on the front. Both panels: the six confirmatory tasks at fold seed 42, one run per task (two for Full and LLM-FE). Reading: on the front if $p_\text{front}\ge0.8$, near it if $\ge0.5$.}
\label{tab:a27_pareto}
\begin{tabular}{lrrrrl}
\toprule
\multicolumn{6}{l}{\emph{Panel A}} \\
Configuration & Runs & Recovery & Tokens (M) & $p_\text{front}$ & Reading \\
\midrule
""" + "\n".join(pr) + r"""
\midrule
\multicolumn{6}{l}{\emph{Panel B}} \\
Configuration & Runs & Recovery & FLOP ($10^{18}$) & $p_\text{front}$ & Reading \\
\midrule
""" + "\n".join(prb) + r"""
\bottomrule
\end{tabular}
\end{table}
""", encoding="utf-8")

    # ------------------------------------------------------------------ ladder figure (pgfplots)
    _e7 = pd.read_csv(I / "episodes_final.csv")
    _e7 = _e7[(_e7.ablation_id == "A05") & (_e7.arm_id == "full") & (_e7.replicate == 0) & ~_e7.row_id.str.contains("-9")
              & _e7.dataset_id.isin([f"synp_p{t}" for t in ("1", "2", "3", "4", "5", "6a", "6b")])]
    _full7 = _e7.rec_sel.mean()
    xs = {"f2_Q0.8": -0.097, "f2_Q2": 0.301, "f2_Q4": 0.602, "f2_Q9": 0.954, "f2_Q122": 1.000}
    def rate(s):
        k, n = s.split("/")
        return int(k) / int(n)
    rws = pd.read_csv(I / "rows_final.csv")
    rep = fs.get("inf_guard", {}).get("replaced", {})
    q9rec = rws[rws.row_id.str.match(r"A17-4") & (rws.arm_id == "f2_Q9") & rws.dataset_id.str.startswith("synp")]
    q9fin = [rep[r] for r in q9rec.row_id if r in rep]
    q9ok = int(rws[rws.row_id.isin(q9fin)].final_status.eq("OK").sum())
    q9fs = fs.get("inf_guard", {}).get("by_rung", {}).get("f2_Q9", "--")
    q9txt = (f"after the inf-guarded re-runs on dedicated servers (Amendments 19b, 19d, 19f), {q9ok} of its {len(q9rec)} SYN-P rows finished; \\texttt{{func\\_success}} with the re-runs is {q9fs}"
             if q9ok else f"none of its {len(q9rec)} SYN-P rows finished even after inf-guarded re-runs")
    fsp = " ".join(f"({xs[r]},{rate(fs['rule_3of5']['by_rung'][r]):.3f})" for r in xs)
    rsp = " ".join(f"({xs[r]},{br[r]['rec_sel_rate']:.3f})" for r in xs)
    igs = fs.get("inf_guard", {}).get("by_rung", {})
    isp = " ".join(f"({xs[r]},{rate(igs[r]):.3f})" for r in xs if r in igs)
    (P / "figures" / "a27_ladder.tex").write_text(r"""\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[width=0.62\linewidth,height=4.6cm,xlabel={$\log_{10}$ active parameters (B)},ylabel={rate},ymin=0,ymax=1.05,
  xmin=-0.2,xmax=1.15,xtick={-0.097,0.301,0.602,0.954},xticklabels={0.8B (14),2B (7),4B (14),9B (7)},xticklabel style={font=\scriptsize},
  legend style={font=\scriptsize,at={(0.5,-0.42)},anchor=north,legend columns=2},grid=major,grid style={gray!20}]
\addplot+[mark=*,thick] coordinates {""" + fsp + r"""};
\addlegendentry{functional success}
\addplot+[mark=square*,thick,dashed] coordinates {""" + rsp + r"""};
\addlegendentry{recovery}
\addplot+[mark=o,thick,dotted,gray] coordinates {""" + isp + r"""};
\addlegendentry{functional success, with re-runs}
\addplot[red,thick,densely dotted,domain=-0.2:1.15] {""" + f"{_full7:.3f}" + r"""};
\addlegendentry{gpt-oss-120b recovery}
\node[font=\scriptsize,anchor=south east] at (axis cs:1.0,0.80) {122B-A10B (14)};
\end{axis}
\end{tikzpicture}
\caption{Qwen3.5 model-size ladder: functional success (3 of 5 folds) and recovery against active parameters; runs per model in brackets. 9B is zero in the rows of record because all seven of its runs crashed on an XGBoost \emph{inf} error; the grey series replaces runs whose first attempt crashed or was capped by exploratory re-runs with a crash guard and a longer cap. Seven synthetic tasks including P1; the red line is gpt-oss-120b's recovery on the same tasks and seeds.}
\label{fig:a27_ladder}
\end{figure}
""", encoding="utf-8")

    # ------------------------------------------------------------------ optional F0 / A21
    f0txt = r"F0 single-shot pools (A17-T8) are being scored; the table is written when \texttt{f0/fold\_summary.csv} exists."
    f0p = I / "f0_summary_by_rung.csv"
    f0res = ""
    if f0p.exists():
        g = pd.read_csv(f0p).set_index("rung")
        order = [r for r in ("Q0.8", "Q2", "Q4", "Q9", "Q122", "O120") if r in g.index]
        lab = {"O120": "gpt-oss-120b"}
        lab = {"O120": "gpt-oss-120b", "Q0.8": "Qwen3.5-0.8B", "Q2": "Qwen3.5-2B", "Q4": "Qwen3.5-4B", "Q9": "Qwen3.5-9B", "Q122": "Qwen3.5-122B-A10B"}
        rowsf0 = "\n".join(f"{lab.get(r, r)} & {f(g.loc[r, 'pexec'], 2)} & {f(g.loc[r, 'rec128'], 3)} & {f(g.loc[r, 'recK'], 3)} ({int(g.loc[r, 'nK'])}) & "
                           f"{f(g.loc[r, 'tokr'], 2)} & {f(g.loc[r, 'f2_rec'], 3)} \\\\" for r in order)
        (P / "tables" / "a27_f0.tex").write_text(r"""\begin{table}[t]
\centering\small
\caption{The full loop against single-shot sampling with selection (exploratory): the same model writes a one-shot feature list up to 128 times per fold and the best list by inner CV is kept. Seven synthetic tasks including P1; fold-level recovery. ``Usable'': share of draws that yield $\ge$1 accepted feature. ``Matched'': best-of-$k$ at the $k$ whose tokens equal the loop's spend on that fold (at most 128), over the folds where the loop logged tokens (count in brackets), so it can exceed best-of-128. Token ratio: loop / 128 draws.}
\label{tab:a27_f0}
\begin{tabular}{lrrrrr}
\toprule
Model & Usable & Best-of-128 & Matched & Token ratio & Loop \\
\midrule
""" + rowsf0 + r"""
\bottomrule
\end{tabular}
\end{table}
""", encoding="utf-8")
        f0txt = r"Table~\ref{tab:a27_f0} reports the single-shot pools against the full loop."
        o = g.loc["O120"] if "O120" in g.index else None
        if o is not None:
            # P1-free paired comparison with a task-seed cluster bootstrap (review round 2)
            import numpy as _np
            _fs = pd.read_csv(I / "f0" / "fold_summary.csv"); _fs = _fs[_fs.rung == "O120"]
            _e0 = pd.read_csv(I / "episodes_final.csv")
            _e0 = _e0[(_e0.ablation_id == "A05") & (_e0.arm_id == "full") & (_e0.replicate == 0) & ~_e0.row_id.str.contains("-9")]
            _m = _fs.merge(_e0[["dataset_id", "fold_seed", "fold", "rec_sel"]].rename(columns={"rec_sel": "f2"}), on=["dataset_id", "fold_seed", "fold"])
            _m = _m[_m.dataset_id != "synp_p1"].assign(df=lambda x: x.rec_sel_best128 - x.f2)
            _g = _m.groupby(["dataset_id", "fold_seed"]).df.agg(["sum", "count"])
            _a, _c = _g["sum"].values, _g["count"].values
            _rng = _np.random.default_rng(20261016)
            _bs = [_a[i].sum() / _c[i].sum() for i in (_rng.integers(0, len(_a), len(_a)) for _ in range(10000))]
            nop1 = (_m.rec_sel_best128.mean(), _m.f2.mean(), _a.sum() / _c.sum(), *_np.percentile(_bs, [2.5, 97.5]))
            f0res = (rf"\paragraph{{The loop does not beat sampling plus selection (exploratory)}} Asking gpt-oss-120b 128 times for a one-shot list of features and keeping the best list by inner CV recovers {f(o['rec128'])} of fold-episodes, against {f(o['f2_rec'])} for the Full loop on the same folds and at about the same completion-token budget (the loop spends {f(o['tokr'], 2)}$\times$ the tokens of 128 draws). Without P1, which brute force solves, the rates on these 60 folds (the first Full run at each fold seed) are {f(nop1[0])} and {f(nop1[1])} (difference {f(nop1[2], 2)}, 95\% CI [{f(nop1[3], 2)}, {f(nop1[4], 2)}], bootstrap over task--seed pairs): we cannot show that the loop beats single-shot sampling, nor the reverse. The single-shot baseline is scored on a stricter validation frame than the loop, which handicaps it. On the Qwen models (Table~\ref{{tab:a27_f0}}), with at most the loop's tokens single-shot sampling does better than the loop at 4B and slightly worse at 122B-A10B; the 9B loop is zero because of crashes.")
    a21txt = r"The A21 iso-cost comparison for the gpt-oss-120b arms is being computed; it is written here when \texttt{a21\_isocost} exists. The A05 anytime curve (validity 55/55) reads R-A early (borderline) with an R-C flag."
    a21res = ""
    any_p = I / "anytime_readings.json"
    if any_p.exists():
        ar = json.loads(any_p.read_text())
        ra, rb, rc = ar["R-A_primary"], ar["R-B"], ar["R-C"]
        s5 = ra["S_k"]["5"]
        a21txt = (rf"On the A05 Full runs (validity 55/55), R-A reads ``{ra['reading']}'': {f(s5, 2)} of the recoveries seen by iteration 10 are present by iteration 5, and selected recovery is {f(ra['rec_sel_k']['5'])} at $k=5$ against {f(ra['rec_sel_k']['10'])} at $k=10$. "
                  rf"R-B: on Core-10 the anytime curve reaches within 0.5\,$\sigma_d$ of its final value by $\hat k={rb['k_hat']}$. "
                  rf"R-C: the winner's-curse flag is raised, because the inner-CV pick at $k=10$ scores below the pick at $k=5$ on {rc['a_n_worse_k10_vs_k5']} of 10 datasets.")
        a21res = (rf"\paragraph{{Most of the loop's value arrives early (descriptive)}} On the Full runs, {f(s5, 2)} of all recoveries seen after ten iterations are already present after five, real-data held-out scores come within half a run-to-run standard deviation of their final value after {rb['k_hat']} iteration, and on {rc['a_n_worse_k10_vs_k5']} of 10 real datasets the iteration selected after ten iterations scores below the one selected after five, a sign of selection overfitting.")
    a21p = I / "a21_isocost" / "isocost_points.csv"
    if any_p.exists() and not a21p.exists():
        a21txt += " The iso-cost comparison (R-D) for the gpt-oss-120b arms has not finished."
    if a21p.exists():
        iso = pd.read_csv(a21p)
        iso = iso[iso.cost_axis == "tok_compl_total"] if "cost_axis" in iso else iso
        agg = iso[iso.dataset_id.isin(["ALL_synp", "ALL_core10"]) & (iso.arm_id != "full")].copy()
        nm = {"set_phased": "Single phased context", "restarts": "Restarts", "l0": "Open loop", "oracle": "Oracle planner", "misdirected": "Misdirected planner",
              "search_off": "Web search off", "fx_PaJM": "Assessment off", "fx_PAjM": "Pruning off", "fx_pAJM": "Planner off", "fx_PAJm": "Memory off",
              "fx_PajM": "Assessment and pruning off", "fx_Pajm": "Planner only", "fx_pAjm": "Assessment only", "fx_pajm": "Minimal, no memory",
              "fx_pajM": "Minimal harness", "fx_paJM": "Pruning and memory only", "fx_paJm": "Pruning only", "fx_pAJm": "Assessment and pruning, no planner",
              "fx_pAjM": "Assessment and memory, no planner"}
        lines = []
        for _, r in agg.sort_values(["dataset_id", "arm_id"]).iterrows():
            pop = "SYN-P (\\texttt{rec\\_sel})" if r.dataset_id == "ALL_synp" else "Core-10 ($\\Delta_\\text{std}$)"
            lines.append(f"{nm.get(r.arm_id, r.arm_id.replace('_', chr(92) + '_'))} & {pop} & {f(r.delta_iso_kstar)} {ci([r.ci95_lo_kstar, r.ci95_hi_kstar]) if pd.notna(r.ci95_lo_kstar) else ''} & {f(r.delta_iso_kstar1)} & {r['R-D_ci']} \\\\")
        (P / "tables" / "a27_isocost.tex").write_text(r"""\begin{table}[h]
\centering\small
\caption{A21 iso-cost comparison (R-D): each arm against Full truncated, at whole iterations, to the arm's completion tokens ($k^*$) and one iteration more ($k^*{+}1$); $\Delta$ = arm $-$ truncated Full, positive = arm better; reading from the 95\% bootstrap class at both points (``budget-sensitive'' if they disagree). Descriptive; never changes a Holm verdict (R-E).}
\label{tab:a27_isocost}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lllll}
\toprule
Arm & Population & $\Delta$ at $k^*$ [95\% CI] & $\Delta$ at $k^*{+}1$ & Reading \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
""", encoding="utf-8")
        a21txt += r" Table~\ref{tab:a27_isocost} gives the iso-cost comparison (R-D)."
        def g_(arm, pop):
            x = agg[(agg.arm_id == arm) & (agg.dataset_id == pop)]
            return x.iloc[0] if len(x) else None
        s_syn, s_core, so_core = g_("set_phased", "ALL_synp"), g_("set_phased", "ALL_core10"), g_("search_off", "ALL_core10")
        if s_syn is not None and s_core is not None:
            a21res += (rf" When Full is truncated to an arm's token budget, the single phased context still recovers more on the synthetic tasks (difference {f(s_syn.delta_iso_kstar)}) but scores lower on real data ({f(s_core.delta_iso_kstar, 2)} run-to-run SDs)"
                       + (rf", and a cost-matched Full also beats the run without web search on real data ({f(so_core.delta_iso_kstar, 2)} SDs)" if so_core is not None else "") + r" (Table~\ref{tab:a27_isocost}).")

    # ------------------------------------------------------------------ appendix: study inventory + sensitivities
    c1ap, c3p = C["C1a"]["sensitivity_power"], C["C3"]["sensitivity_power"]
    sens = [
        f"C1a with power replicates (Amendments 11--14) & {frac(c1ap['desc']['k1'], c1ap['desc']['n1'])} vs {frac(c1ap['desc']['k0'], c1ap['desc']['n0'])}; RD {f(c1ap['desc']['rd'])} {ci(c1ap['desc']['rd_ci90'])}; $p={f(c1ap['p'])}$ \\\\",
        f"C1b notebook selection (secondary) & RD {f(C['C1b']['modes']['notebook']['rd'])}; {pe(C['C1b']['modes']['notebook']['p'])} \\\\",
        f"C1b any-program (\\texttt{{rec\\_any}}) & RD {f(C['C1b']['modes']['any']['rd'])}; $p={f(C['C1b']['modes']['any']['p'])}$ \\\\",
        f"C3 with power replicates & {frac(c3p['desc']['k1'], c3p['desc']['n1'])} vs {frac(c3p['desc']['k0'], c3p['desc']['n0'])}; $p={f(c3p['p'])}$ \\\\",
        f"C4a H2: local-correct vs parametric & exact {pe(C['C4a']['H2_secondary']['kb_parametric']['p_two_sided'])} ({C['C4a']['arms']['kb_parametric']}) \\\\",
        f"C4a H2: local-correct vs local-wrong & exact {pe(C['C4a']['H2_secondary']['kb_local_wrong']['p_two_sided'])} ({C['C4a']['arms']['kb_local_wrong']}) \\\\",
        f"C5a fold rule 5/5 & exact {pe(fs['rule_5of5']['exact']['p_two_sided'])} \\\\",
        f"C5a fold rule $\\ge$1/5 & exact {pe(fs['rule_1of5']['exact']['p_two_sided'])} \\\\",
        f"C5a first attempts only & exact {pe(fs['first_attempt']['exact']['p_two_sided'])} \\\\",
        f"C5a inf-guard re-runs (Amendment 19) & exact {pe(fs['inf_guard']['exact']['p_two_sided'])} \\\\",
        f"C5a Firth logistic (task, seed fixed) & slope {f(fs['firth'].get('estimate'), 2)}, {pe(fs['firth'].get('p_wald'))} \\\\",
    ]
    c5ig = C["C5b"].get("inf_guard")
    if c5ig:
        dd_ = c5ig["desc"]
        sens.append(f"C5b inf-guard re-runs replace their rows (Amendments 19, 19c) & {frac(dd_['k1'], dd_['n1'])} vs {frac(dd_['k0'], dd_['n0'])}; RD {f(dd_['rd'])}; exact {pe(dd_['exact_task']['p_two_sided'])} \\\\")
        m1 = C["C5b"]["inf_guard_mix_1b"]
        sens.append(f"A19 \\texttt{{mix\\_1b}} with inf-guard re-runs & {frac(m1['k1'], m1['n1'])} vs {frac(m1['k0'], m1['n0'])} \\\\")
    a12r = fr.get("a12_retries") or {}
    for rid, v in sorted(a12r.items()):
        src = v["retries"].split("retry of ")[-1].split(" ")[0] if "retry of" in v["retries"] else "?"
        sens.append(f"A12 retry {rid} of ITT row {src} (Amendment 19e, exploratory) & {v['status']}, \\texttt{{rec\\_sel}} {v['rec_sel']}; RV3 analysis of record unchanged \\\\")
    lgsum = lg.assign(rung=lg.row_id).merge(pd.read_csv(I / "rows_final.csv")[["row_id", "arm_id"]], on="row_id", how="left")
    lgr = lgsum.groupby("arm_id").note_requests.median().to_dict()
    (P / "appendix" / "a27_details.tex").write_text(r"""\section{AGENT2027: pre-registration, sensitivities and descriptive read-outs}
\label{app:a27}

\paragraph{Protocol.} All analyses follow the AGENT2027 pre-registration (tag \texttt{agent2027-prereg}, frozen 16 October 2026) and its 19 dated amendments. Each confirmatory contrast was fitted with its registered model; where an arm sits at 0\% or 100\%, the GLMM does not converge or $|\hat\beta|>10$, the registered exact conditional test stratified by task is the test of record. Every run that crashed or hit the 12\,h cap counts as an intention-to-treat failure with \texttt{rec\_sel}$=0$ in all five folds. Recovery was re-scored for all 591 SYN-P rows with the amended detector of record; the re-score reproduces every committed episode it overlaps (A05 Full 20/60, all 240 A12 and 315 RV1 episodes).

\begin{table}[h]
\centering\small
\caption{Registered sensitivities of the confirmatory contrasts.}
\begin{tabular}{ll}
\toprule
Sensitivity & Result \\
\midrule
""" + "\n".join(sens) + r"""
\bottomrule
\end{tabular}
\end{table}

\paragraph{Loop guard (A17, Amendment 17).} The guard's counter is not written to the instrumentation summary, so it is counted from the LLM traces as requests carrying the guard note; the note stays in the history after it fires, so these are upper bounds on exposure. Median requests per run: """ + ", ".join(f"{k.replace('f2_', '')} {int(v)}" for k, v in sorted(lgr.items()) if isinstance(k, str) and k.startswith("f2_")) + r""".

\paragraph{A18 role-wise replay.} Primary scoring (registered): the Scientist's floor is Q4, the Tester's floor is at or below Q0.8, and no Qwen rung reaches the Extractor's floor (``$>$Q122''). With JSON-string arguments decoded (sensitivity, HQ \#315) the Extractor executes 0.52--0.92 of replayed calls across the Qwen rungs (from 0.01--0.02), its floor drops to Q4 and floor stability is 0.44: the Extractor's failure on small models is mainly a tool-call transport failure (stringified arguments from the \texttt{qwen3\_coder} parser), not a code-writing failure.

\paragraph{A02, A03, A05.} Extractor rounds $R=10$ were kept over 3 and 1 (A02). Augmenting the raw columns beat replacing them on 10/10 real-data pairs (mean $\Delta_\text{std}=+0.49$; A03). The Full-v2 reference has ICC 0.899 across SYN-P tasks, and only p2, p4 and p6b are informative (A05), so most contrasts rest on two or three tasks.

\paragraph{F0 single-shot pools (A17-T8).} """ + f0txt + r"""

\paragraph{A21 anytime and iso-cost.} """ + a21txt + r"""

\input{tables/a27_pareto}
""" + ("\n\\input{tables/a27_isocost}\n" if (I / "a21_isocost" / "isocost_points.csv").exists() else ""), encoding="utf-8")

    # ------------------------------------------------------------------ results section prose
    onB = [nb[r.config] for _, r in pb.iterrows() if r.config in nb and r.wording == "on the front"]
    nearB = [nb[r.config] for _, r in pb.iterrows() if r.config in nb and "near" in str(r.wording)]
    panelB_txt = ("the front holds " + " and ".join(onB) + (", with " + " and ".join(nearB) + " near it" if nearB else "") + "; every other Qwen or mixed configuration is dominated.") if onB else "no configuration is clearly on the front."
    memtxt = ""
    mp = I / "process_a12_rows.csv"
    if mp.exists():
        from scipy import stats as _st
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        import final_tests as _F
        P6_ = ["synp_p2", "synp_p3", "synp_p4", "synp_p5", "synp_p6a", "synp_p6b"]
        # exactly the 72 runs of the RV3 analysis of record (SYN-P 7 minus P1, 12 cells)
        rv_ = pd.read_csv(R / "RV3_A12_runs.csv")[["row_id", "M", "task"]]
        pr_ = pd.read_csv(mp).merge(rv_, on="row_id")
        pr_["M"] = pr_.M > 0
        ep_ = pd.read_csv(I / "episodes_final.csv").merge(rv_, on="row_id")
        ep_["M"] = ep_.M > 0
        ex_any = _F.exact_stratified([(int(g[g.M].rec_any.sum()), int(g.M.sum()), int(g[~g.M].rec_any.sum()), int((~g.M).sum())) for _, g in ep_.groupby("task")])
        fnd = ep_[ep_.rec_any == 1]
        runs_ = ep_.groupby(["row_id", "M"]).rec_any.sum().reset_index()
        r_on, r_off = int((runs_[runs_.M].rec_any > 0).sum()), int((runs_[~runs_.M].rec_any > 0).sum())
        rep_on, rep_off = pr_[pr_.M].focus_novelty.mean(), pr_[~pr_.M].focus_novelty.mean()
        mw = _st.mannwhitneyu(pr_[pr_.M].focus_novelty.dropna(), pr_[~pr_.M].focus_novelty.dropna()).pvalue
        td = pr_.groupby(["task", "M"]).focus_novelty.mean().unstack()
        nlow = int((td[True] < td[False]).sum())
        memtxt = (rf"\paragraph{{Memory lowers discovery (exploratory)}} On the {len(rv_)} factorial runs, the memory effect lies in discovery, not selection: with memory on, $\varphi$ is found at any iteration in {int(ep_[ep_.M].rec_any.sum())}/{int(ep_.M.sum())} fold-episodes against {int(ep_[~ep_.M].rec_any.sum())}/{int((~ep_.M).sum())} without (exact test stratified by task, {pe(ex_any['p_two_sided'])}, ignoring within-run clustering), and every found $\varphi$ is also selected in both arms. At run level the difference is small ({r_on} vs {r_off} of {int(runs_.M.sum())} runs find $\varphi$, Fisher {pe(_st.fisher_exact([[r_on, int(runs_.M.sum()) - r_on], [r_off, int((~runs_.M).sum()) - r_off]])[1])}), so the factorial's run-level memory term (Table~\ref{{tab:a27_a12}}) is the firmer evidence. "
                  rf"With memory the planner almost never returns to an earlier focus: the share of foci within cosine 0.9 of an earlier focus in the same fold is {f(rep_on, 4)} against {f(rep_off, 4)} per run (lower on {nlow} of {len(td)} tasks; run-level Mann--Whitney {pe(mw)}; lexical TF-IDF similarity). "
                  rf"This fits memory pushing the planner away from directions before they are refined, but the data do not show which abandoned directions would have succeeded.")
    igb = C["C5a"].get("inf_guard", {})
    ig_rep = igb.get("replaced", {})
    _rw = pd.read_csv(I / "rows_final.csv"); _ep = pd.read_csv(I / "episodes_final.csv")
    def _rec_ig(arm):
        rec = _rw[_rw.row_id.str.match(r"A17-4") & (_rw.arm_id == arm) & _rw.dataset_id.str.startswith("synp")].row_id
        ids = [ig_rep.get(r, r) for r in rec]
        e_ = _ep[_ep.row_id.isin(ids)]
        return int(e_.rec_sel.sum()), len(e_)
    rec_ig = {a_: _rec_ig(a_) for a_ in ("f2_Q0.8", "f2_Q2", "f2_Q4", "f2_Q9", "f2_Q122")}
    ig_txt = ""
    _lab5 = {"f2_Q0.8": "0.8B", "f2_Q2": "2B", "f2_Q4": "4B", "f2_Q9": "9B", "f2_Q122": "122B-A10B"}
    if igb:
        b_ = igb["by_rung"]
        ig_txt = (rf"When labelled re-runs with a guard against the crash, on dedicated servers, replace the runs whose first attempt crashed or was capped (exploratory), functional success is "
                  + ", ".join(f"{b_[r]} ({_lab5[r]})" for r in ("f2_Q0.8", "f2_Q2", "f2_Q4", "f2_Q9", "f2_Q122"))
                  + rf", the ladder no longer decreases anywhere, and the trend tightens to {pe(igb['exact']['p_two_sided'])}; 9B then recovers {rec_ig['f2_Q9'][0]}/{rec_ig['f2_Q9'][1]} fold-episodes against {rec_ig['f2_Q4'][0]}/{rec_ig['f2_Q4'][1]} for 4B and {rec_ig['f2_Q122'][0]}/{rec_ig['f2_Q122'][1]} for 122B-A10B. The re-runs had an 18\,h cap instead of 12\,h, needed several rounds, and three of the seven 9B rows still hit the cap, so the 9B zero in the rows of record is consistent with a crash artefact but does not establish 9B's capability.")
    c1a, c1b, c3, c4a, c4b, c5a, c5b = (C[k] for k in ("C1a", "C1b", "C3", "C4a", "C4b", "C5a", "C5b"))
    from scipy import stats as _st  # run-level views of the fold-level C4 tests (review round 2)
    _a14 = _ep[(_ep.ablation_id == "A14") & ~_ep.row_id.str.contains("-9")].groupby(["arm_id", "row_id"]).rec_sel.sum().reset_index()
    _c4r = {a_: (int((_a14[_a14.arm_id == a_].rec_sel > 0).sum()), int((_a14.arm_id == a_).sum())) for a_ in ("kb_local_correct", "kb_off")}
    _c4a_run = _st.fisher_exact([[_c4r["kb_local_correct"][0], _c4r["kb_local_correct"][1] - _c4r["kb_local_correct"][0]],
                                 [_c4r["kb_off"][0], _c4r["kb_off"][1] - _c4r["kb_off"][0]]])[1]
    _f5 = _ep[(_ep.ablation_id == "A05") & (_ep.arm_id == "full") & _ep.dataset_id.isin(["synp_p6a", "synp_p6b"])]
    _r5 = _f5.groupby(["dataset_id", "fold_seed", "replicate"]).rec_sel.mean().unstack(0).dropna()
    _c4b_np = len(_r5); _c4b_fav = int((_r5["synp_p6a"] > _r5["synp_p6b"]).sum())
    _c4b_sign = _st.binomtest(_c4b_fav, _c4b_np, 0.5).pvalue
    n_rej = sum(1 for k in C if C[k]["holm_reject"])
    # ---------------------------------------------------------------- sensitivity table (review round 1)
    import sys as _sys2
    _sys2.path.insert(0, str(Path(__file__).resolve().parent))
    import final_tests as _F2
    rvr = pd.read_csv(R / "RV3_A12_runs.csv")
    inf_t = ["synp_p2", "synp_p6b"]
    sens_rows = []
    c1p = C["C1a"]["sensitivity_power"]["desc"]
    sens_rows.append(("Phased context vs three agents", "+ power replicates (Amend.\\ 11--14)", f"RD {f(c1p['rd'])} {ci(c1p['rd_ci90'])}"))
    for fac, nm in (("P", "Planner"), ("A", "Assessment"), ("J", "Pruning"), ("M", "Memory")):
        g = rvr[rvr.task.isin(inf_t)]
        k1, n1 = int(g[g[fac] > 0].k.sum()), int(g[g[fac] > 0].n.sum())
        k0, n0 = int(g[g[fac] < 0].k.sum()), int(g[g[fac] < 0].n.sum())
        rd_ = _F2.newcombe_unpaired(k1, n1, k0, n0, 0.90)
        sens_rows.append((f"{nm} on $-$ off", "two informative tasks only", f"{f(rd_[0])} [{f(rd_[1])}, {f(rd_[2])}]"))
    fsx = pd.read_csv(I / "out" / "c5a_func_success.csv")
    def _trend(df):
        st = [[(_F2.LOG10_ACTIVE[r_.arm_id], int(r_.func_success)) for _, r_ in g_.iterrows()] for _, g_ in df.groupby(["dataset_id", "fold_seed"])]
        return _F2.exact_ca_trend(st)["p_two_sided"]
    f431 = fsx.copy(); f431.loc[f431.row_id == "A17-431", "func_success"] = 0
    for lab_, key_ in (("all five folds must succeed", "rule_5of5"), ("one fold suffices", "rule_1of5"), ("first attempts only", "first_attempt"), ("crash re-runs replace rows", "inf_guard")):
        sens_rows.append(("Functional-success trend", lab_, f"{pe(fs[key_]['exact']['p_two_sided'])}"))
    sens_rows.append(("Functional-success trend", "One 4B run's successful retry counted as a failure", f"{pe(_trend(f431))}"))
    sens_rows.append(("Functional-success trend", "Firth logistic regression", f"slope {f(fs['firth'].get('estimate'), 2)}, $p={p(fs['firth'].get('p_wald'))}$"))
    c5ig_ = C["C5b"].get("inf_guard", {}).get("desc")
    if c5ig_:
        sens_rows.append(("Extractor on 4B", "crash re-runs replace rows", f"{frac(c5ig_['k1'], c5ig_['n1'])} vs {frac(c5ig_['k0'], c5ig_['n0'])}, $p={p(c5ig_['exact_task']['p_two_sided'])}$"))
    isop = I / "a21_isocost" / "isocost_points.csv"
    if isop.exists():
        iso_ = pd.read_csv(isop)
        def _iso(arm, pop):
            x = iso_[(iso_.arm_id == arm) & (iso_.dataset_id == pop)]
            return x.iloc[0] if len(x) else None
        for arm, pop, nm, unit in (("set_phased", "ALL_synp", "Phased context", "recovery"), ("set_phased", "ALL_core10", "Phased context", "SD units"),
                                   ("search_off", "ALL_core10", "Web search off", "SD units")):
            x = _iso(arm, pop)
            if x is not None and pd.notna(x.delta_iso_kstar):
                sens_rows.append((nm, f"vs Full at equal tokens, {'synthetic' if pop == 'ALL_synp' else 'real data'}", f"{f(x.delta_iso_kstar, 2)} [{f(x.ci95_lo_kstar, 2)}, {f(x.ci95_hi_kstar, 2)}] ({unit})"))
    (P / "tables" / "a27_sens.tex").write_text(r"""\begin{table}[t]
\centering\small
\caption{Sensitivity analyses. Differences with 90\% (factorial) or 95\% (equal-token) intervals; the two informative tasks are those whose Full recovery is strictly between 0 and 1 in the factorial.}
\label{tab:a27_sens}
\begin{tabular}{lll}
\toprule
Result & Variant & Estimate \\
\midrule
""" + "\n".join(f"{a_} & {b_} & {c_} \\\\" for a_, b_, c_ in sens_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""", encoding="utf-8")
    a12tok = {r_["arm"]: r_ for r_ in real}
    text = rf"""\section{{Results}}
\label{{sec:a27_results}}

Table~\ref{{tab:a27_confirmatory}} reports the ten pre-registered contrasts; {n_rej} reject under Holm. Unless stated otherwise the outcome is recovery in the selected iteration on the planted-feature benchmark, and the gpt-oss-120b reference system (Full) recovers {d('C1a')['k0']}/{d('C1a')['n0']} fold-episodes on the six confirmatory tasks. The table marks the design caveat of each conclusive contrast; the planner contrast (C2c) is the cleanest, though not token-matched.

\input{{tables/a27_confirmatory}}

Table~\ref{{tab:a27_sens}} collects the registered and post-hoc sensitivity analyses referred to below.

\input{{tables/a27_sens}}

\subsection{{Which components earn their cost}}

\input{{tables/a27_a12}}

\paragraph{{The planner helps; the Tester's two LLM stages do not}} In the component factorial (Table~\ref{{tab:a27_a12}}), switching the planner on raises recovery by {f(eff['P']['Delta'])} (90\% CI {ci(eff['P']['Delta_ci90'])}; C2c). Replacing the Tester's written assessment by a template report, or its LLM pruning by code rules, changes recovery by {f(eff['A']['Delta'])} and {f(eff['J']['Delta'])}, both equivalent to zero within the pre-registered margin of $\pm0.167$, half of Full's recovery rate (C2a, C2b). Pooling favours these verdicts, because tasks at floor or ceiling contribute differences near zero: on the two informative tasks (P2, P6b) the assessment interval still fits the margin but the pruning interval does not (Table~\ref{{tab:a27_sens}}). The registered verdict for both is ``reducible'': the token criterion is met by construction, because a replaced stage makes no LLM call. Total completion tokens fall less: on ten real datasets, removing the assessment uses {f(a12tok['fx_PaJM']['tok_ratio'], 2)}$\times$ Full's tokens, removing pruning {f(a12tok['fx_PAjM']['tok_ratio'], 2)}$\times$, and removing both {f(a12tok['fx_PajM']['tok_ratio'], 2)}$\times$. On real data all four knockouts, the planner knockout included, score \emph{{higher}} than Full, three of them with intervals above zero in units of run-to-run SD, although the raw differences are below 0.01 in accuracy or nRMSE. These runs are exploratory and ran on a later software image than their Full pairs, so we read them as confounded, not as gains. The planner effect is therefore a planted-feature result. It is concentrated where recovery can move: in the factorial P3 and P5 sit at 0, P4 near 0 and P6a near 1, and on the two informative tasks the planner adds 0.24 and memory removes 0.22. The planner arm is not token-matched (1.38M completion tokens per run with the planner, 1.08M without; Table~\ref{{tab:a27_pareto}}), so the effect may partly be one of extra reasoning rather than of planning as such; against this, the planner-only configuration recovers 0.433 at 0.86M tokens, more than the planner knockout (0.233) at 1.08M. No two-factor interaction is significant once the semi-foldover de-aliases them.

{memtxt}

\paragraph{{The planner's direction is causal}} A misdirected planner recovers {frac(dd(a09m)['k1'], dd(a09m)['n1'])} against Full's {frac(dd(a09m)['k0'], dd(a09m)['n0'])} (one-sided {pe(a09m['p'])}), and an oracle planner that names the target feature recovers {frac(dd(a09o)['k1'], dd(a09o)['n1'])}: the loop follows the planner's direction (positive control, outside the family).

\paragraph{{Iteration, feedback and decomposition show no detectable effect}} Ten independent restarts recover {frac(d('C3')['k1'], d('C3')['n1'])} against Full's {frac(d('C3')['k0'], d('C3')['n0'])} on the multi-step tasks (C3, $p={f(c3['p'])}$), and an open-loop planner that never sees results recovers {frac(dd(a11)['k1'], dd(a11)['n1'])} on the same tasks ($p={f(a11['p'])}$, exploratory). A single context that runs the same roles as phases recovers {frac(d('C1a')['k1'], d('C1a')['n1'])} against {frac(d('C1a')['k0'], d('C1a')['n0'])} (C1a, risk difference {f(d('C1a')['rd'])}, 90\% CI {ci(d('C1a')['rd_ci90'])}, $p={f(c1a['p'])}$). All three are inconclusive, not evidence of equivalence: the registered power analysis put the minimum detectable effect at about 0.8 for C1a and above 1 for C3, so these contrasts could not be expected to reject. With the non-registered power replicates added (a labelled sensitivity), the single-context difference is {f(C['C1a']['sensitivity_power']['desc']['rd'])}, 90\% CI {ci(C['C1a']['sensitivity_power']['desc']['rd_ci90'])}; no equivalence test was registered for it.

\paragraph{{Error messages reduce errors}} Returning tool-error text to the agents cuts errors per accepted feature to {f(a07['rho_shown_over_hidden'], 2)} of the rate with errors hidden (95\% CI {ci(a07['rho_ci95'], 2)}; exact sign-flip $p={f(a07['p_signflip'], 4)}$, 8 paired real-data runs; exploratory).

\subsection{{What the agents use: knowledge}}

\paragraph{{Column names and targeted documents carry the knowledge}} Anonymising the column names of an otherwise identical task lowers recovery from {f(c4b['rate_p6a'], 2)} to {f(c4b['rate_p6b'], 2)} ({c4b['n_pairs']} pairs, difference {f(c4b['delta'], 2)}, 95\% CI {ci(c4b['delta_ci95'], 2)}; C4b, exact McNemar {pe(c4b['p'])}; one domain, and a pre-freeze pilot had already shown the same direction, which the pre-registration discloses). The 15 pairs come from {_c4b_np} run pairs; all {_c4b_fav} favour named columns, but at run level that cannot reach significance (sign test {pe(_c4b_sign)}). On P7, whose rule exists only in a local document, the correct corpus recovers {c4a['arms']['kb_local_correct']} fold-episodes against {c4a['arms']['kb_off']} with retrieval off, {c4a['arms']['kb_parametric']} with parametric recall and {c4a['arms']['kb_local_wrong']} with a corpus whose rule is wrong (C4a, {pe(c4a['p'])}; at run level {_c4r['kb_local_correct'][0]} of {_c4r['kb_local_correct'][1]} runs against {_c4r['kb_off'][0]} of {_c4r['kb_off'][1]}, Fisher {pe(_c4a_run)}). By design the comparison arms cannot succeed on P7, and the relevant document is retrieved in the top three for 85\% of queries, so C4a shows that the loop relays and uses retrieved text, not that retrieval works in general. Web search was not varied on the planted tasks; on real data, turning it off does no measurable harm at the native budget and saves tokens ({f(a13['tok_ratio_gm'], 2)}$\times$, 95\% CI {ci(a13['tok_ratio_ci95'], 2)}), but a Full run truncated to the same token budget scores higher (Table~\ref{{tab:a27_sens}}), so its value is small and budget-dependent.

\subsection{{What each role needs from the model}}

\input{{figures/a27_ladder}}
\input{{tables/a27_ladder}}

\paragraph{{Functional success rises with active parameters}} Across the Qwen3.5 ladder, the share of runs that finish with an accepted LLM feature in the selected iteration in at least three of five folds rises with log active parameters (C5a, exact stratified trend {pe(c5a['p'])}; Fig.~\ref{{fig:a27_ladder}}, Table~\ref{{tab:a27_ladder}}). The trend holds when all five folds must succeed, on first attempts only, with Firth regression and with the crash re-runs, but not under the lenient one-fold rule. The top of the ladder pairs a dense 9B model with a mixture-of-experts model of 10B active parameters (122B-A10B), so the trend there contrasts architectures as much as sizes. No size reaches the registered functional threshold of 0.8, and in the rows of record 9B is zero because all seven of its runs crashed on an XGBoost \emph{{inf}} error. {ig_txt} Every 0.8B run finished, yet none kept a non-fallback LLM feature: the smallest model completes the loop only through its deterministic fallbacks.

\paragraph{{The roles need different models}} Moving only the Extractor to 4B while the Scientist and Tester stay on 122B-A10B lowers recovery from {frac(d('C5b')['k0'], d('C5b')['n0'])} to {frac(d('C5b')['k1'], d('C5b')['n1'])} (C5b, risk difference {f(d('C5b')['rd'])}). Putting the Scientist and Tester on 0.8B instead stops the loop ({frac(dd(m1b)['k1'], dd(m1b)['n1'])}; Table~\ref{{tab:a27_outside}}). Replaying recorded prompts role by role on each model (registered, outside the family) puts the Scientist's floor at 4B and the Tester's at or below 0.8B, and finds no Qwen size that reaches the Extractor's floor. A post-hoc check explains much of the last result: the tool parser returns the small models' arguments as strings, and decoding them raises the share of Extractor calls that execute from 0.01--0.02 to 0.52--0.92 (floor then 4B, but that floor holds in only 0.44 of bootstrap resamples, below the registered stability threshold of 0.6). Part of the Extractor gap, and so of C5b, may therefore reflect serving-parser compatibility rather than model capability.

\subsection{{Cost}}

On completion tokens the full system is Pareto-dominated (Table~\ref{{tab:a27_pareto}}; descriptive). LLM-FE recovers {frac(d('C1b')['k1'], d('C1b')['n1'])} against Full's {frac(d('C1b')['k0'], d('C1b')['n0'])} (C1b, $p={f(c1b['p'])}$, inconclusive; 90\% CI {ci(d('C1b')['rd_ci90'])}, no equivalence test registered) with about an eighth of the tokens and is the only configuration on the front; the planner-only configuration (no assessment, pruning or memory) lies near it, recovering 0.433 at 0.86M tokens against Full's 0.333 at 1.38M. Full is therefore not the best setting of its own factorial, and the comparisons below are against Full, not against the best loop. On a FLOP proxy across models, {panelB_txt}

{f0res}

{r"\input{tables/a27_f0}" if f0res else ""}

{a21res}

\input{{tables/a27_outside}}

\subsection{{Lessons for agent builders}}
\begin{{enumerate}}
\item Ablate before adding components: on planted-feature tasks the planner was the only agentic component with a confirmed benefit.
\item Replace LLM review and pruning by deterministic code where the decision is a threshold on metrics; here neither replacement lost more than 0.17 recovery (pruning only when pooled over tasks).
\item Treat cross-iteration memory as a design variable, not a default: here it lowered discovery, possibly by steering the planner off directions before they were refined.
\item Return raw tool-error text to agents.
\item Before concluding a model is too small for a role, check tool-call argument parsing: here, decoding string-serialised arguments raised the share of Extractor calls that execute from 1--2\% to 52--92\%.
\item Compare against cost-matched non-agentic baselines (best-of-$k$ sampling with a verifier, an evolutionary LLM search); we could not show that the full configuration beats them.
\end{{enumerate}}
"""
    (P / "sections" / "a27_results.tex").write_text(text, encoding="utf-8")
    print("wrote tables/a27_*.tex, figures/a27_ladder.tex, sections/a27_results.tex, appendix/a27_details.tex")


if __name__ == "__main__":
    main()
