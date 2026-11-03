# A01 W0 session plan (audit trail)

Spec: `docs/ablations/A01_W0_old_image_replicate.md`. One dated line per record; nothing here is a result
except the final hand-off block, which copies `paper/results_w0_t17rep_summary.json`.

## Deviations from the spec

- **Session IDs.** The spec's `S1007a` / `S1008a` were never launched (the box was held by T2.1/T1.2 work).
  The waves ran as **`S1009a`** (seed 42) and **`S1009b`** (seed 4), both on Fri 9 Oct, inside the §6 kill
  dates (seed 42 by Fri 9 Oct 18:00, seed 4 by Sun 11 Oct 08:00). Run dirs `~/rogueone-runs-agent2027/S1009a`
  and `S1009b`; W&B `brainflux/RogueOne_W0_T17Rep_Seed42` / `_Seed4` (unchanged).
- **A01-T7 (Serper credits) was not checked** before launch (dashboard only). Query count not recorded.
- **A01-T15 dropped** (D1); it was not needed anyway: no pair failed after the in-wave solo retry.
- **No optional rows** (`balance-scale`, `wine`); A03 uses the t17 SD for both.
- To free GPUs 0-1 for `S1009a`, Henrik's T1.2 gpt-oss-20b job was stopped (his OK) and relaunched after A01
  (Oct 10 03:25Z; finished Oct 12 04:44Z, 27/27).

## Records

- 2026-10-08 — A01-T1 / D9: W&B entity `brainflux` (HQ #252).
- 2026-10-08 — D11: the Satterthwaite variance rule is confirmed (HQ #254).
- 2026-10-09 08:28Z — A01-T2: `rogue_one:nestedcv9` id `sha256:27c81e5b8652…` = the `rogueone image=` id in both t17 job logs. PASS.
- 2026-10-09 08:28Z — A01-T3: `<VLLM_REF>` = `sha256:6b1c0c57fe1ab30f080e2c190352889c998ce1c0c62b7a26901d3c9912b5092f` (tags `latest`, `v0.22.0`), the same in both t17 logs; `vllm.__version__` = 0.22.0. PASS.
- 2026-10-09 08:28Z — A01-T4: `nested-cv-rebuttal` pushed; the six extracted files match the spec md5s. PASS.
- 2026-10-09 08:28Z — A01-T5: one snapshot, `refs/main` = `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` (snapshot 2026-08-03) = Hub `main`. PASS. Re-checked unchanged at each launch.
- 2026-10-09 08:28Z — A01-T6: no launcher key overridden; `ROGUE_ONE_DO_BALANCING=False`; `LLM_MODEL=openai/gpt-oss-120b`; W&B reachable (28 runs in `brainflux/RogueOne_NestedCV_T17_Seed42`). PASS.
- 2026-10-09 08:28Z — A01-T7: Serper credits **not checked**.
- 2026-10-09 08:28Z — A01-T8: task data `differences: 0`. PASS.
- 2026-10-09 08:28Z — A01-T11/T12: pre-flight clean (GPUs 0-3 < 2,000 MiB, no `ro27-`/`rogueone-` containers); `S1009a` launched, seed 42, GPUs 0-3, TP4, PARALLEL=6, CPUS=12, SOLO=' '.
- 2026-10-09 09:54Z — `S1009a`: crab first pass rc=0 but no `fold_aggregation_summary.csv` (stopped silently ~81 min in, global iter 36; W&B run `zhixt0e0` "finished", no traceback: the swallowed-exception path, `main.py:1590-1598`). Solo-retried at the end of the wave.
- 2026-10-09 13:23Z — A01-T13: `S1009a` ended: `=== still failing after solo retry: none ===`, `T16_RERUN_ALL_DONE seed=42`, `session end rc=0`; wall 4h55m. Tear-down clean.
- 2026-10-09 13:25Z — A01-T14: `refs/main` unchanged; `S1009b` launched, seed 4, same settings.
- 2026-10-09 17:01Z — `S1009b` ended: `=== no first-pass failures, no retry phase needed ===`, `T16_RERUN_ALL_DONE seed=4`, `session end rc=0`; wall 3h33m. 16/16 pairs have a W0 run.
- 2026-10-12 06:33Z — A01-T16: collection clean: no unexpected W0 problems; `t17 cells 16 max abs diff 0.0`.
- 2026-10-12 06:33Z — A01-T10: `scripts/agent2027/analysis/w0_sigma_samp.py`; `--selftest` PASSED locally in `rogue_one:nestedcv9` (ν1 = ν2 = 6.136, CI factor 5.69, MC flag rate 1.54%, coverage 0.969).
- 2026-10-12 06:34Z — A01-T17: analysis run; W&B `PlanBench/RogueOne-AGENT2027` run `ANALYSIS__w0_sigma_samp__20261012` (`h6tkeq0f`) with artifacts `a01-results` and `a01-run-outputs`.

## Result and hand-offs (A01-T18)

- **Usable data (rule 1):** 16/16 complete pairs → full analysis.
- **Drift (rule 2): none.** Wilcoxon on z = δ/s_d: p = 0.528 (W+ = 81, n = 16). Z_v1 = 0.372 < `var_inflation_threshold` 5.69. No pooling restriction from drift.
- **Verdict (rule 3): partial**, point estimate just above the "hoped" line: **Z_v1 = 0.372** (R_raw = 0.61), 95% CI **[0.065, 2.12]** at ν1 = ν2 = 6.13. Wording: "consistent with partial (CI spans 0.36 and 0.72)". Without junglechess: Z = 0.56, CI [0.072, 4.40] (ν = 4.64). Nominal 16/16-df sensitivity: CI [0.135, 1.03], threshold 2.76, no flag.
- **Per-dataset ρ_d** (descriptive, 2 df each): credit-g 2.56, heart 0.43, junglechess 0.35, myocardial 2.24, vehicle 0.13, crab 0.63, housing 0.71, insurance 0.80. ρ_d ≤ 0.6 on 3 of 8; ≥ 0.85 on 2 of 8 (credit-g and myocardial, the two datasets with the smallest t17 SD).
- **Independence (rule 4):** 0 of 16 pairs `pair_identical`; median command Jaccard 0.001; median `tester_identical_share` 0. σ_samp is not a lower bound.
- **To A03-T3 (σ̂ table):** σ_samp,d (raw units, 2 pairs each): credit-g 0.0141, heart 0.0082, junglechess 0.0245, myocardial 0.0170, vehicle 0.0037, crab 0.0153, housing 0.0151, insurance 0.0501. No drift, no independence flag. `balance-scale` and `wine` use the t17 SD. A03's rule max(A01 σ_samp,d, t17 SD, 0.1·s_d) applies.
- **To A05-T2 / A05-T14 (E11a):** `paper/results_w0_t17rep_pairs.csv`; `sigma_samp_pooled_sd_reg3` = 2.158 (6 df); `sigma_samp_pooled_sd` = 4.344 (16 df) and 1.434 without junglechess (14 df); Z_v1 and CI as above; drift: none.
- **To A02:** median `wall_s` over the 16 W0 runs = 7,886 s (2.19 h); no run > 6 h. Lower bound for a 6-client run (wave tails ran fewer clients).
- **Secondary.** junglechess folds ≥ 0.95: f42 W0 3 / t17 4; f4 W0 1 / t17 2. insurance seed 4: t17's outlier comes from folds 2 and 3 (t17 n_RMSE 0.644 and 0.599 vs W0 0.398 and 0.316 on the same partition), so it was a sampling draw, not a partition effect.
