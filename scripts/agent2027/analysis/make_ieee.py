"""Derive the IEEE two-column (AGENT 2027) variants of the generated results LaTeX.

Reads the files written by make_agent2027_tex.py in <paper>/sections, tables, figures and writes:
  sections/a27i_results.tex, tables_ieee/*.tex, figures/a27i_ladder.tex
Numbers are never edited; only float environments, widths and cross-references to material that lives in the
replication package (appendix tables of the long report) change.

    python make_ieee.py <paper dir>
"""
import re
import sys
from pathlib import Path

P = Path(sys.argv[1])
(P / "tables_ieee").mkdir(exist_ok=True)
BS = chr(92)
NL = chr(10)
WIDE = {"a27_confirmatory", "a27_outside", "a27_a12"}
for name in ("a27_confirmatory", "a27_a12", "a27_outside", "a27_f0", "a27_pareto", "a27_sens"):
    s = (P / "tables" / f"{name}.tex").read_text(encoding="utf-8")
    if name in WIDE:
        s = s.replace(BS + "begin{table}[t]", BS + "begin{table*}[t]").replace(BS + "end{table}", BS + "end{table*}")
    else:
        s = s.replace(BS + "centering" + BS + "small", BS + "centering" + BS + "scriptsize" + BS + "setlength{" + BS + "tabcolsep}{2.5pt}")
        s = re.sub(r"(\\begin\{tabular\}\{[^}]*\})", lambda m: BS + "resizebox{" + BS + "linewidth}{!}{%\n" + m.group(1), s, count=1)
        s = s.replace(BS + "end{tabular}", BS + "end{tabular}}", 1)
    s = s.replace(BS + "centering" + BS + "small", BS + "centering" + BS + "footnotesize")
    s = s.replace("Left: the 12-cell", "Top: the 12-cell").replace("Right: the same knockouts", "Bottom: the same knockouts")
    if name == "a27_outside":  # rows that only repeat the results text
        s = s.replace("no harm beyond M", "no harm beyond margin")
        s = NL.join(l for l in s.split(NL) if not l.startswith(("Planner control", "Error messages", "Open loop")))
    (P / "tables_ieee" / f"{name}.tex").write_text(s, encoding="utf-8")

f = (P / "figures" / "a27_ladder.tex").read_text(encoding="utf-8")
f = f.replace("width=0.62" + BS + "linewidth,height=4.6cm", "width=" + BS + "columnwidth,height=4.0cm")
(P / "figures" / "a27i_ladder.tex").write_text(f, encoding="utf-8")

r = (P / "sections" / "a27_results.tex").read_text(encoding="utf-8")
r = r.replace(BS + "input{tables/a27_ladder}\n", "")
r = r.replace(BS + "input{figures/a27_ladder}", BS + "input{figures/a27i_ladder}")
r = r.replace(BS + "input{tables/", BS + "input{tables_ieee/")
r = r.replace("Fig.~" + BS + "ref{fig:a27_ladder}, Table~" + BS + "ref{tab:a27_ladder}", "Fig.~" + BS + "ref{fig:a27_ladder}")
r = r.replace("Table~" + BS + "ref{tab:a27_ladder}", "Fig.~" + BS + "ref{fig:a27_ladder}")
for lab, txt in (("tab:a27_isocost", "Table~" + BS + "ref{tab:a27_sens}"),):
    r = r.replace("Table~" + BS + "ref{" + lab + "}", txt)
r = r.replace("Appendix~" + BS + "ref{app:a27}", "the replication package")
# the descriptive anytime paragraph shrinks to its one cost-matched finding
r = re.sub(r"\\paragraph\{Most of the loop's value arrives early \(descriptive\)\}.*?(When Full is truncated)", r"\1", r, count=1, flags=re.S)
NL = chr(10)
r = r.replace(BS + "subsection{Cost}" + NL, BS + "subsection{Cost}" + NL + NL + BS + "input{tables_ieee/a27_pareto}" + NL, 1)
(P / "sections" / "a27i_results.tex").write_text(r, encoding="utf-8")
left = re.findall(r"\\ref\{(tab:a27_(?:isocost|ladder)|app:a27)\}", r)
print("ieee variants written; dangling refs:", left)
