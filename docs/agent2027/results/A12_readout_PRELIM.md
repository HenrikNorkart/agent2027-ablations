# A12 FX read-out: PRELIMINARY

**Status:** preliminary. The detector ran on CAIR at 2026-10-24 20:21:50Z, after δ was committed (Amendment 12,
`89647d8`, 20:20:42Z). Estimates use a **linear probability model with task fixed effects** (and, as a conservative check,
per-task contrasts with a t over the 6 tasks), **not** the pre-registered beta-binomial GLMM. That model and Holm are
E11b / A12-T16 (G4, Tue 27). Do not quote these numbers as final.

- Population: SYN-P 7 minus P1, 8 cells × 6 tasks = 48 runs (A12-001…048 minus the P1 rows A12-049…056), 240
  episodes. One ITT failure: A12-035 (`fx_*`, `synp_p6b`; rc 0 with no `fold_aggregation_summary`, twice).
  Smoke rows A12-057…065 are excluded.
- δ = 0.167 (Amendment 12). Token rule: branch (i) (Amendment 6), which the FX arms satisfy by construction. Total
  completion-token ratios are shown below as well.
- Sensitivity: adds A12-801…824 (f4 r0, Amendment 11) with a fold-seed block. Amendment 13 rows are still running.

## Headline (pre-registered rows)

| Contrast | Δ rec_sel (X+ − X−) | 90% CI, LPM | 90% CI, per-task t (df 5) | TOST p (±0.167) | Total tokens X−/X+ (90% CI) | Prelim verdict |
|---|---|---|---|---|---|---|
| C2a, A (LLM assessment) | −0.008 | [−0.077, 0.061] | [−0.039, 0.023] | 0.0002 / 0.0001 | 0.77 [0.74, 0.80] | equivalent; *reducible* under branch (i) |
| C2b, J (LLM pruning) | +0.042 | [−0.027, 0.111] | [0.011, 0.073] | 0.002 / 0.0002 | 0.61 [0.58, 0.64] | equivalent; *reducible* (also on total tokens) |
| C2c, P (planner) | +0.125 | [0.056, 0.194] | [0.001, 0.249] | two-sided p 0.004 / 0.098 | 0.65 [0.62, 0.67] | planner helps; Holm verdict depends on the final model |
| M (memory), outside Holm | −0.108 | [−0.177, −0.039] | [−0.249, 0.033] | – | 1.09 | memory **hurts**, driven by p6b (−0.45) |
| Falsifier `fx_pajM` − Full | −0.167 | [−0.325, −0.008] (df 11) | – | – | 0.41 M vs Full | inconclusive: the CI crosses −δ; the point estimate equals −δ |

Sensitivity with the f4 r0 rows: P +0.172 [0.106, 0.238], A −0.017 [−0.083, 0.050], J +0.061 [−0.005, 0.127],
M −0.072 [−0.138, −0.006], PA −0.094 [−0.161, −0.028]. The direction is the same throughout, and P strengthens.

**G4 note:** the alias pair PA (= −JM) has Wald p 0.075 (LPM, pre-registered rows) and 0.021 (sensitivity). The RV3
trigger (p < 0.05 in T16 **and** at least 28u of reserve) is decided on the GLMM on Tue 27.

## Full output
## Confirmatory rows: 48 runs, 240 episodes; ITT: 1

### rec_sel k/5 per cell x task (pre-registered, SYN-P 7 minus P1)

| arm_id   |   synp_p2 |   synp_p3 |   synp_p4 |   synp_p5 |   synp_p6a |   synp_p6b |   pooled |
|:---------|----------:|----------:|----------:|----------:|-----------:|-----------:|---------:|
| fx_pAJM  |         2 |         0 |         0 |         0 |          5 |          0 | 0.233333 |
| fx_PaJM  |         5 |         0 |         1 |         0 |          5 |          2 | 0.433333 |
| fx_PAjM  |         3 |         0 |         0 |         0 |          5 |          0 | 0.266667 |
| fx_PAJm  |         4 |         0 |         1 |         0 |          5 |          3 | 0.433333 |
| fx_pajM  |         1 |         0 |         0 |         0 |          4 |          0 | 0.166667 |
| fx_paJm  |         2 |         0 |         0 |         0 |          5 |          2 | 0.3      |
| fx_pAjm  |         3 |         0 |         0 |         0 |          5 |          3 | 0.366667 |
| fx_Pajm  |         4 |         0 |         1 |         0 |          5 |          3 | 0.433333 |

### Effects (Delta = mean X+ minus X-, LPM with task FE; pre-registered rows)

| effect   |   Delta | 90% CI           | 95% CI           |   p two-sided |   p TOST(±0.167) |   df |
|:---------|--------:|:-----------------|:-----------------|--------------:|-----------------:|-----:|
| P        |   0.125 | [0.056, 0.194]   | [0.042, 0.208]   |        0.0042 |           0.1552 |   35 |
| A        |  -0.008 | [-0.077, 0.061]  | [-0.091, 0.074]  |        0.8393 |           0.0002 |   35 |
| J        |   0.042 | [-0.027, 0.111]  | [-0.041, 0.124]  |        0.3142 |           0.0021 |   35 |
| M        |  -0.108 | [-0.177, -0.039] | [-0.191, -0.026] |        0.0118 |           0.0797 |   35 |
| PA       |  -0.075 | [-0.144, -0.006] | [-0.158, 0.008]  |        0.0745 |           0.0153 |   35 |
| PJ       |   0.042 | [-0.027, 0.111]  | [-0.041, 0.124]  |        0.3142 |           0.0021 |   35 |
| PM       |   0.025 | [-0.044, 0.094]  | [-0.058, 0.108]  |        0.544  |           0.0007 |   35 |

### Conservative: per-task contrasts, t over the 6 tasks

| effect   |   Delta | 90% CI (t, df=5)   |   p two-sided |   p TOST | per task                            |
|:---------|--------:|:-------------------|--------------:|---------:|:------------------------------------|
| P        |   0.125 | [0.001, 0.249]     |        0.0981 |   0.2627 | +0.40 +0.00 +0.15 +0.00 +0.05 +0.15 |
| A        |  -0.008 | [-0.039, 0.023]    |        0.6109 |   0.0001 | -0.00 +0.00 -0.05 +0.00 +0.05 -0.05 |
| J        |   0.042 | [0.011, 0.073]     |        0.0422 |   0.0002 | +0.10 +0.00 +0.05 +0.00 +0.05 +0.05 |
| M        |  -0.108 | [-0.249, 0.033]    |        0.1825 |   0.2202 | -0.10 +0.00 -0.05 +0.00 -0.05 -0.45 |
| PA       |  -0.075 | [-0.143, -0.007]   |        0.0756 |   0.0203 | -0.20 +0.00 -0.05 +0.00 -0.05 -0.15 |
| PJ       |   0.042 | [-0.019, 0.102]    |        0.2242 |   0.0044 | +0.10 +0.00 +0.05 +0.00 -0.05 +0.15 |
| PM       |   0.025 | [-0.018, 0.068]    |        0.2956 |   0.0006 | +0.10 +0.00 -0.05 +0.00 +0.05 +0.05 |

### Falsifier fx_pajM - Full (f42 r0+r1): Delta=-0.167, 90% CI [-0.325, -0.008], df=11; fx_pajM 5/30, Full 20/60
| dataset_id   |   0 |   1 |
|:-------------|----:|----:|
| synp_p2      | 0.5 | 0.2 |
| synp_p3      | 0   | 0   |
| synp_p4      | 0   | 0   |
| synp_p5      | 0   | 0   |
| synp_p6a     | 1   | 0.8 |
| synp_p6b     | 0.5 | 0   |

### Tokens (log tok_compl_total, task FE; 47 runs with instrumentation)

| outcome                 | factor   |   ratio X-/X+ | 90% CI         |
|:------------------------|:---------|--------------:|:---------------|
| tok_compl_total         | P        |         0.645 | [0.618, 0.674] |
| tok_compl_total         | A        |         0.769 | [0.737, 0.803] |
| tok_compl_total         | J        |         0.608 | [0.582, 0.635] |
| tok_compl_total         | M        |         1.092 | [1.045, 1.140] |
| tok_compl_tester        | P        |         0.985 | [0.948, 1.025] |
| tok_compl_tester        | A        |         0.002 | [0.001, 0.002] |
| tok_compl_tester        | J        |         0.001 | [0.001, 0.001] |
| tok_compl_tester        | M        |         1.018 | [0.979, 1.059] |
| tok_compl_tester_assess | P        |         0.99  | [0.969, 1.012] |
| tok_compl_tester_assess | A        |         0     | [0.000, 0.000] |
| tok_compl_tester_assess | J        |         1.028 | [1.006, 1.051] |
| tok_compl_tester_assess | M        |         1.015 | [0.994, 1.037] |
| tok_compl_tester_prune  | P        |         0.983 | [0.939, 1.028] |
| tok_compl_tester_prune  | A        |         1.156 | [1.104, 1.209] |
| tok_compl_tester_prune  | J        |         0     | [0.000, 0.000] |
| tok_compl_tester_prune  | M        |         1.018 | [0.973, 1.065] |
| tok_compl_scientist     | P        |         0     | [0.000, 0.000] |
| tok_compl_scientist     | A        |         1.236 | [1.173, 1.303] |
| tok_compl_scientist     | J        |         1.05  | [0.996, 1.106] |
| tok_compl_scientist     | M        |         0.968 | [0.919, 1.021] |

Mean tok_compl_total per cell (M):
| arm_id   |   tok_compl_total |
|:---------|------------------:|
| fx_pAJM  |              1.08 |
| fx_PaJM  |              1.36 |
| fx_PAjM  |              1.03 |
| fx_PAJm  |              1.48 |
| fx_pajM  |              0.41 |
| fx_paJm  |              0.93 |
| fx_pAjm  |              0.75 |
| fx_Pajm  |              0.86 |

### Sensitivity: + A12-801..824 (f4 r0), 72 runs, block = fold seed

| effect   |   Delta | 90% CI           | 95% CI           |   p two-sided |   p TOST(±0.167) |   df |
|:---------|--------:|:-----------------|:-----------------|--------------:|-----------------:|-----:|
| P        |   0.172 | [0.106, 0.238]   | [0.093, 0.252]   |        0.0001 |           0.5522 |   58 |
| A        |  -0.017 | [-0.083, 0.050]  | [-0.096, 0.063]  |        0.6757 |           0.0002 |   58 |
| J        |   0.061 | [-0.005, 0.127]  | [-0.018, 0.140]  |        0.1286 |           0.0049 |   58 |
| M        |  -0.072 | [-0.138, -0.006] | [-0.152, 0.007]  |        0.0736 |           0.01   |   58 |
| PA       |  -0.094 | [-0.161, -0.028] | [-0.174, -0.015] |        0.0205 |           0.0362 |   58 |
| PJ       |   0.028 | [-0.038, 0.094]  | [-0.052, 0.107]  |        0.4863 |           0.0004 |   58 |
| PM       |   0.05  | [-0.016, 0.116]  | [-0.029, 0.129]  |        0.2122 |           0.0023 |   58 |

A12-801..824 per cell (k/5):
| arm_id   |   synp_p2 |   synp_p4 |   synp_p6b |
|:---------|----------:|----------:|-----------:|
| fx_pAJM  |         3 |         0 |          0 |
| fx_PaJM  |         5 |         1 |          3 |
| fx_PAjM  |         4 |         0 |          1 |
| fx_PAJm  |         4 |         0 |          1 |
| fx_pajM  |         0 |         0 |          0 |
| fx_paJm  |         1 |         0 |          2 |
| fx_pAjm  |         1 |         0 |          2 |
| fx_Pajm  |         5 |         0 |          1 |

Full f4 r0 / r1 / f42 r2 on the same tasks (k/5):
| row_id   | dataset_id   |   fold_seed |   replicate |   k |
|:---------|:-------------|------------:|------------:|----:|
| A05-040  | synp_p2      |           4 |           0 |   3 |
| A05-042  | synp_p4      |           4 |           0 |   1 |
| A05-045  | synp_p6b     |           4 |           0 |   1 |
| A05-801  | synp_p2      |           4 |           1 |   4 |
| A05-802  | synp_p2      |          42 |           2 |   2 |
| A05-803  | synp_p4      |           4 |           1 |   1 |
| A05-804  | synp_p4      |          42 |           2 |   1 |
| A05-805  | synp_p6b     |           4 |           1 |   2 |
| A05-806  | synp_p6b     |          42 |           2 |   1 |
## Update 2026-10-25 13:15Z: + Amendment 13 rows (A12-825…872, f42 r1 and f4 r1; 120 runs in all)

LPM with task fixed effects and a fold-seed block, 106 residual df. A12-827 is ITT (XGBoost `inf` crash, twice).

| Effect | Δ | 90% CI | p two-sided | TOST p (±0.167) |
|---|---|---|---|---|
| P | +0.117 | [0.055, 0.178] | 0.002 | 0.09 |
| A | +0.017 | [−0.045, 0.078] | 0.66 | 0.0001 |
| J | +0.063 | [0.002, 0.125] | 0.09 | 0.003 |
| M | −0.043 | [−0.105, 0.018] | 0.25 | 0.0006 |
| PA (= −JM) | −0.110 | [−0.172, −0.048] | 0.004 | 0.06 |
| PJ | +0.043 | [−0.018, 0.105] | 0.25 | 0.0006 |
| PM | +0.003 | [−0.058, 0.065] | 0.93 | 0.0000 |

Reading: the planner effect holds, and A and J stay equivalent within ±δ. The memory effect shrinks towards 0: the p6b
signal in the pre-registered rows did not replicate. The PA/JM alias pair is now clearly non-zero, which is relevant
to the RV3 trigger at G4 (that trigger is decided on the GLMM, pre-registered rows only).
