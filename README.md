# Replication package: component ablations of Rogue One (AGENT 2027)

Data, results and analysis code for

> H. Brådland et al., **"These Are Not the Agents You Are Looking For: Component Ablations in a Multi-Agent System for
> Automated Feature Engineering."** AGENT 2027 workshop at ICSE 2027.

The package holds what is needed to check every number in the paper: the protocol, the run manifests, the per-run
and per-fold results, the trace annotations, and the scripts that turn them into the paper's tests, tables and
figures. The Rogue One system under test is included so that runs can be repeated.

## Layout

| path | content |
|---|---|
| `docs/agent2027/preregistration.md` | study protocol with all dated amendments: contrasts, the confirmatory (Holm) family, margins, row-of-record rules |
| `docs/ablations/` | per-ablation specifications, which the protocol incorporates (Part I) |
| `docs/agent2027/results/final_20261112/` | **read-out of record**: per-run rows, fold-episode recovery, Holm table, full read-out (`README.md` there lists its provenance) |
| `docs/agent2027/results/final_20261102/` | inputs of the earlier read-out that the record reuses: base funnel, late funnel, cost/Pareto points, sampling-baseline folds |
| `docs/agent2027/results/*.csv, *.json, *.md` | factorial analysis of record (`RV3_A12_*`), power constants (`g2_constants.json`, `G2_*`), equivalence margin (`A09_delta*`), LLM-FE comparison (`X6_A08_*`), real datasets (`A12_realdata_*`) |
| `docs/agent2027/results/a5_prune_shadow/` | feature-selection shadow mode (A5) |
| `docs/agent2027/results/mast/` | trace analysis: codebook, 48 blinded packets, both annotators' labels, consensus, kappa (see its `README.md`) |
| `manifests/agent2027/` | run manifests per ablation and per session; `mflat4/` is the flat manifest set the read-out uses |
| `scripts/agent2027/analysis/` | analysis scripts (below) |
| `scripts/agent2027/synp/`, `corpora/`, `manifest_specs/`, `llmfe_specs/` | planted-task generator, knowledge corpora for the knowledge probe, manifest and LLM-FE specifications |
| `rogueone/`, `rogue_one_main.py`, `Dockerfile`, `docker/` | the Rogue One system (roles: Scientist = Planner, Extractor = Coder, Tester) and the LLM-FE images |

## Paper ablations and code identifiers

The paper numbers design aspects A1-A8; the code and the protocol use ablation ids A01-A23 and contrast ids C1-C5.

| paper | aspect | code ablation (contrast) |
|---|---|---|
| A1 | Planner | A12 factor P (C2c); oracle/misdirected Planner A09 |
| A2 | memory | A12 factor M |
| A3 | domain knowledge | A14 knowledge probe on P7 (C4a), A15 named vs anonymised P6a/P6b (C4b) |
| A4 | report | A12 factor A (C2a) |
| A5 | feature selection | A12 factor J (C2b); shadow mode in `a5_prune_shadow/` |
| A6 | refinement | A10 restarts vs refinement (C3), A11 open-loop Planner |
| A7 | separate agents | A16 single agent (C1a) |
| A8 | model size and role mixing | A17 scale ladder (C5a), A19 role mixing (C5b), A18 role-wise replay |
| baselines | OpenFE, LLM-FE, sampling | A08 CPU screen (`A08_screen_episodes.csv`); A08/X6 (C1b); sampling baseline in `final_20261102/f0_*` |

## Where the paper's numbers come from

| paper element | file(s) | script |
|---|---|---|
| Confirmatory contrasts, Holm-adjusted p-values | `final_20261112/holm_table.csv`, `final_readout.json` | `final_tests.py`, `final_glmm.R` |
| Table III, single ablations (A3, A6, A7, A8, LLM-FE) | `final_20261112/final_readout.json`, `episodes_final.csv` | `final_tests.py` |
| Table III, factorial (A1, A2, A4, A5): Δ, 90% CI, TOST | `RV3_A12_glmm.json`, `RV3_A12_runs.csv` | `glmm_holm.py` |
| Token ratios, cost and Pareto front | `final_20261102/pareto_front.csv` | `pareto.py` |
| Model-size figure (Fig. 4) | `final_20261112/final_readout.json` (`seed4_ladder`), `episodes_final.csv` | `make_ladder_fig.py` |
| Knowledge figure (A3) | `final_20261112/final_readout.json`, `episodes_final.csv` | `make_ieee_v2.py` |
| Real datasets | `A12_realdata_summary.json`, `A12_realdata_rows.csv`, `final_20261102/a21_isocost_points.csv` | `realdata_check.py` |
| Feature-selection shadow mode | `a5_prune_shadow/` | `prune_shadow.py` |
| A1 tokens, A2 discovery and focus revisits, A4/A5 equivalence on P2 and P6b, baselines (Design, Table III: OpenFE vs the complete system, LLM-FE scorings), sampling baseline (Discussion), real-data knockouts | `A08_screen_episodes.csv`, `X6_A08_llmfe_episodes.csv`, `final_20261102/{pareto_front,process_a12_rows,f0_fold_summary}.csv`, `RV3_A12_runs.csv`, `A12_realdata_summary.json`, `final_20261112/episodes_final.csv` | `paper_extras.py` |
| Power analysis, ICC | `g2_constants.json`, `G2_memo.md` | `g2_stats.py` |
| Equivalence margin δ = 0.167 | `A09_delta.md`, `A09_delta_episodes.csv` | — |
| Table IV (MAST), kappa | `mast/mast_consensus.csv`, `mast/annot_A/`, `mast/annot_B/` | `mast/kappa.py`, `mast/unblind.py` |
| Planner-pointing rates (A7), tool errors and prompt tokens per role (A7, A8) | `mast/flow/`, `mast/capacity/derived/` | scripts in those folders |

## Reproducing the confirmatory read-out

Python 3 with pandas, numpy and scipy; R 4.4 with lme4 and logistf only to refit the GLMMs.

```bash
R=docs/agent2027/results
mkdir -p in/mflat out/paper/figures_v2
cp $R/final_20261112/{episodes_final.csv,rows_final.csv,funnel_episodes_5xx.csv} in/
cp $R/final_20261102/{funnel_episodes.csv,funnel_episodes_late.csv} in/
cp manifests/agent2027/mflat4/*.csv in/mflat/
A=scripts/agent2027/analysis
python $A/final_tests.py prep   --in in --x6 $R/X6_A08_llmfe_episodes.csv --out out
Rscript $A/final_glmm.R out        # or: cp $R/final_20261112/{glmm_results.json,describe.json} out/
python $A/final_tests.py report --in in --x6 $R/X6_A08_llmfe_episodes.csv \
       --rv3 $R/RV3_A12_glmm.json --g2 $R/g2_constants.json --out out
python $A/make_ladder_fig.py --in $R/final_20261112 --paper out/paper   # writes figures_v2/ladder.tex
python $A/paper_extras.py --res $R              # the non-confirmatory numbers quoted in the text
```

With the stored GLMM fits, `out/holm_table.csv`, `out/c5a_func_success.csv` and `out/final_readout.json` are identical
to the files in `final_20261112/`, apart from the time stamp.

## Not included

Raw run outputs (LLM traces and per-iteration logs) are not in the package. The per-run and per-fold tables
derived from them are. The scripts that read raw outputs (`final_rows.py`, `recovery_rows.py`, `funnel.py`, `pareto.py`,
`realdata_check.py`, `prune_shadow.py`, and those in `mast/flow/` and `mast/capacity/`) are included to document how
the tables were built. Paths of the form `<CAIR_HOME>` and `~/ro27_*` in the protocol refer to the compute server the
study ran on.
