# A12 real-data check (Amendment 16), final read-out — Tue 27 Oct 2026

Exploratory, outside Holm. Script `scripts/agent2027/analysis/realdata_check.py`; sessions S1025h–k (all ended);
rows A12-701…780 paired with the A05 Full rows named in their notes; outputs `realdata_rows.csv`,
`realdata_summary.json` on CAIR in `~/ro27_analysis/realdata/final/` (git-ignored, as row data).

All 80 rows have a final OK attempt (20 pairs per arm, no ITT imputation, nothing pending). Δ = arm − Full on the
held-out score (balanced accuracy; nRMSE sign-flipped), standardised by σ_d; non-inferior iff t 90% lower bound > −1.

| Arm | Knockout | Mean Δ_std | t 90% CI | Dataset-bootstrap 90% | Raw Δ | Verdict | Token ratio (90% CI) |
|---|---|---|---|---|---|---|---|
| fx_pAJM | planner (P−) | +0.58 | [0.13, 1.03] | [0.04, 1.03] | +0.009 | non-inferior | 0.76 [0.72, 0.80] |
| fx_PaJM | assessment (A−) | +0.87 | [0.31, 1.44] | [0.30, 1.57] | +0.009 | non-inferior | 0.83 [0.78, 0.87] |
| fx_PAjM | pruning (J−) | +0.19 | [−0.21, 0.59] | [−0.20, 0.53] | +0.003 | non-inferior | 0.71 [0.68, 0.74] |
| fx_PajM | cheap (A−J−) | +0.52 | [0.12, 0.92] | [0.19, 0.87] | +0.007 | non-inferior | 0.54 [0.51, 0.57] |

Reading: on real data no knockout is worse than Full; every arm is non-inferior at M = 1 σ_d, and the cheap arm uses
about half of Full's completion tokens. Raw differences are under one point of balanced accuracy, so the positive
standardised means reflect small σ_d, not large gains. Change from the preliminary read (75/80 rows, 26 Oct): P−
+0.75 → +0.58, J− +0.26 → +0.19; all verdicts unchanged.

Caveat: Full rows are A05 (v2.0) and arm rows v2.3; the v2.0→v2.3 changes are default-off switches, so the version
confound should be nil, but the pairing is across sessions.
