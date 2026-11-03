# A02 · M4 read-out (A02-T12): extractor rounds R ∈ {1, 3, 10}

Read-out of sessions `S1013c`/`S1013d` (image `agent2027-pre.3`), run on CAIR on 2026-10-14 in a CPU-only
container. The rule, gates and analysis conventions are the binding text in
`docs/agent2027/preregistration.md` ("A02 M4", including Amendment 1). This file does not change them.

## Decision: **R = 10** (R=3 not adopted)

| # | Criterion (pre-registered) | Value | Threshold | Result |
|---|---|---|---|---|
| 1 | Σ`rec_sel`(R10) − Σ`rec_sel`(R3) ≤ 4 of 40 | 15 − 13 = **2** | ≤ 4 | PASS |
| 2 | No task falls from ≥ 3/5 (R10) to 0/5 (R3) | no drop (largest move: `synp_p6b` 1 → 0, `synp_p4` 2 → 1) | 0 drops | PASS |
| 3 | `success` count R3 ≥ R10 (of 8) | R3 **6/8**, R10 **4/8** | R3 ≥ R10 | PASS |
| 4 | mean `u_run`(R3) ≤ 0.75 × mean `u_run`(R10) | 0.9644 / 1.2852 = **0.7504** | ≤ 0.75 | **FAIL** |

Criterion 4 fails by a very small margin. To pass, R3's mean wall clock would have to be
≤ 0.75 × 10,135.0 s = 7,601.3 s. It is 7,605.5 s, which is 4.3 s per row (34 s over the 8 rows) too high.
The rule has no tolerance band, so R=3 is not adopted, and R=1 is never adopted under the rule. Criterion 4
also fails under every alternative wall-clock reading (descriptive only; the rule uses the ITT reading):

| Reading of R3/R10 cost | Ratio |
|---|---|
| **Rule (ITT): W&B `wall_s` per attempt, launcher wall clock for attempts without a W&B `wall_s`** | **0.7504** |
| Launcher wall clock for every attempt | 0.7517 |
| Final attempt only (drops the failed first attempts of A02-006 and A02-011; not ITT) | 0.7553 |
| `wall_s_extractor` | 0.607 |
| `tok_prompt_extractor` | 0.398 |
| `tok_compl_extractor` | 0.597 |

Cutting rounds from 10 to 3 cuts the Extractor's own time by about 40% and its prompt tokens by 60%. But the
Scientist and Tester time per iteration does not change, so the whole run gets only about 25% shorter.

**Re-tier (G1, canonical wording):** CL1–CL4 apply. Both triggers hold: R=3 is not adopted, and the mean u per
SYN-P run of `R10` is 1.285u, above 0.85u. `W_ref` is the planning-only fallback (see below), and the
1.285u depends on it. The decision itself does not, because criterion 4 is a ratio.

**Leak canary (`synp_p7`):** `rec_any` = 0/15 and `rec_sel` = 0/15 over all three arms (0/10 for R10+R3).
No P7 recovery, so there is nothing to investigate with X9.

`v2_defaults.env` and the pre-registration were **not** edited. A02-T13 writes R into the freeze merge and the
G1 entry.

## Task × arm (`rec_sel` of 5 fold-episodes, `success`, `u_run`)

| dataset_id | R10 rec_sel/5 | R3 rec_sel/5 | R1 rec_sel/5 | R10 success | R3 success | R1 success | R10 u | R3 u | R1 u |
|---|---|---|---|---|---|---|---|---|---|
| synp_p1 | 5 | 5 | 5 | 0 | 1 | 1 | 0.869 | 0.730 | 0.844 |
| synp_p2 | 2 | 2 | 1 | 0 | 1 | 1 | 1.208 | 0.782 | 0.835 |
| synp_p3 | 0 | 0 | 0 | 1 | 1 | 1 | 1.215 | 1.324 | 0.707 |
| synp_p4 | 2 | 1 | 1 | 1 | 1 | 1 | 1.389 | 1.113 | 1.028 |
| synp_p5 | 0 | 0 | 0 | 1 | 1 | 1 | 1.176 | 0.955 | 0.742 |
| synp_p6a | 5 | 5 | 5 | 1 | 1 | 1 | 1.811 | 0.761 | 0.682 |
| synp_p6b | 1 | 0 | 1 | 0 | 0 | 0 | 1.466 | 1.081 | 1.108 |
| synp_p7 | 0 | 0 | 0 | 0 | 0 | 0 | 1.147 | 0.969 | 0.836 |
| **Σ / mean** | **15/40** | **13/40** | **13/40** | **4/8** | **6/8** | **6/8** | **1.285** | **0.964** | **0.848** |

- `rec_any` equals `rec_sel` in every episode (15 / 13 / 13).
- u ratios: R3/R10 = 0.750, R1/R10 = 0.660.
- **Without the `M5S` non-discriminating task `synp_p1` (descriptive only; the rule uses all 40 episodes):**
  - R10 10/35 vs R3 8/35 (loss 2): PASS.
  - No drop: PASS.
  - Success R3 5/7 vs R10 4/7: PASS.
  - u ratio 0.742: PASS.

  On these 7 tasks all four criteria would pass, because R10's `synp_p1` row (0.869u) is cheaper than the
  R10 mean. This is reported, not applied.

## Per row

`u_run` = Σ wall clock over attempts ÷ `W_ref` (7,886 s). `score` is the held-out mean over the 5 outer folds
at the inner-CV-selected iteration: nRMSE for regression, balanced accuracy for classification. `base` is
A08's `base_no_fe` at seed 42 (mean ± fold SD). `success` = no worse than `base` by more than 2 SD, direction-aware.

| row_id | arm | dataset_id | session | attempts | wall s (per attempt) | u_run | rec_sel | rec_any | score | base | success | W&B run (final) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A02-001 | R10 | synp_p1 | S1013c | 1 | 6856.3 | 0.869 | 5 | 5 | 0.0521 | 0.0332 ± 0.0085 | 0 | ub5nf0e0 |
| A02-002 | R10 | synp_p2 | S1013d | 1 | 9526.3 | 1.208 | 2 | 2 | 0.0844 | 0.0711 ± 0.0064 | 0 | zgapfd7s |
| A02-003 | R10 | synp_p3 | S1013c | 1 | 9580.0 | 1.215 | 0 | 0 | 0.7922 | 0.6969 ± 0.0427 | 1 | 3tw7nepw |
| A02-004 | R10 | synp_p4 | S1013d | 1 | 10955.4 | 1.389 | 2 | 2 | 0.5315 | 0.4725 ± 0.0357 | 1 | 81vk8n18 |
| A02-005 | R10 | synp_p5 | S1013c | 1 | 9270.8 | 1.176 | 0 | 0 | 0.9572 | 0.6838 ± 0.0174 | 1 | ni1eic1u |
| A02-006 | R10 | synp_p6a | S1013c | 2 | 8558.0 (launcher) + 5723.1 | 1.811 | 5 | 5 | 0.0548 | 0.0698 ± 0.0044 | 1 | 6xxvripe |
| A02-007 | R10 | synp_p6b | S1013d | 1 | 11564.2 | 1.466 | 1 | 1 | 0.0790 | 0.0698 ± 0.0044 | 0 | p5p6o4ib |
| A02-008 | R10 | synp_p7 | S1013d | 1 | 9046.1 | 1.147 | 0 | 0 | 0.7139 | 0.9191 ± 0.0225 | 0 | jeldvplc |
| A02-009 | R3 | synp_p1 | S1013c | 1 | 5756.3 | 0.730 | 5 | 5 | 0.0302 | 0.0332 ± 0.0085 | 1 | fhxf2cfg |
| A02-010 | R3 | synp_p2 | S1013d | 1 | 6168.2 | 0.782 | 2 | 2 | 0.0696 | 0.0711 ± 0.0064 | 1 | ki4mdgy7 |
| A02-011 | R3 | synp_p3 | S1013c | 2 | 6066.0 (launcher) + 4372.2 | 1.324 | 0 | 0 | 0.7922 | 0.6969 ± 0.0427 | 1 | zr7a43dr |
| A02-012 | R3 | synp_p4 | S1013d | 1 | 8775.7 | 1.113 | 1 | 1 | 0.4937 | 0.4725 ± 0.0357 | 1 | thl4f13e |
| A02-013 | R3 | synp_p5 | S1013c | 1 | 7534.4 | 0.955 | 0 | 0 | 0.9993 | 0.6838 ± 0.0174 | 1 | ifo5tkum |
| A02-014 | R3 | synp_p6a | S1013c | 1 | 6001.8 | 0.761 | 5 | 5 | 0.0563 | 0.0698 ± 0.0044 | 1 | q1osj7i0 |
| A02-015 | R3 | synp_p6b | S1013d | 1 | 8525.9 | 1.081 | 0 | 0 | 0.1089 | 0.0698 ± 0.0044 | 0 | 9lv0le7w |
| A02-016 | R3 | synp_p7 | S1013d | 1 | 7643.8 | 0.969 | 0 | 0 | 0.7161 | 0.9191 ± 0.0225 | 0 | ilhybub5 |
| A02-017 | R1 | synp_p1 | S1013c | 1 | 6654.2 | 0.844 | 5 | 5 | 0.0269 | 0.0332 ± 0.0085 | 1 | ftbegppg |
| A02-018 | R1 | synp_p2 | S1013d | 1 | 6586.1 | 0.835 | 1 | 1 | 0.0580 | 0.0711 ± 0.0064 | 1 | uiyur4eu |
| A02-019 | R1 | synp_p3 | S1013c | 1 | 5573.8 | 0.707 | 0 | 0 | 0.7330 | 0.6969 ± 0.0427 | 1 | 2s6jfgkr |
| A02-020 | R1 | synp_p4 | S1013d | 1 | 8110.7 | 1.028 | 1 | 1 | 0.5135 | 0.4725 ± 0.0357 | 1 | ve33230h |
| A02-021 | R1 | synp_p5 | S1013c | 1 | 5855.1 | 0.742 | 0 | 0 | 0.8827 | 0.6838 ± 0.0174 | 1 | 4wc53uns |
| A02-022 | R1 | synp_p6a | S1013c | 2 | 2402.0 (launcher) + 2978.1 | 0.682 | 5 | 5 | 0.0547 | 0.0698 ± 0.0044 | 1 | tyu16j1u |
| A02-023 | R1 | synp_p6b | S1013d | 1 | 8735.4 | 1.108 | 1 | 1 | 0.1410 | 0.0698 ± 0.0044 | 0 | mg1uw9py |
| A02-024 | R1 | synp_p7 | S1013d | 1 | 6592.6 | 0.836 | 0 | 0 | 0.6449 | 0.9191 ± 0.0225 | 0 | rl4js4bl |

All 24 rows ended `OK`. No cap hit and no ITT imputation was needed. The first attempts of A02-006, A02-011 and
A02-022 failed with "rc 0, no `fold_aggregation_summary.csv`" (dataset abort) and were retried once, alone; only
their wall clock enters the analysis.

### Cost table (final attempt; instrumentation summary = W&B summary)

| row_id | arm | dataset_id | wall_s_E | tok_prompt_E (M) | tok_compl_E (M) | tok_compl_S (M) | tok_compl_T (M) | max_prompt_tokens | context_errors | runner_errors | search queries | fallback_share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A02-001 | R10 | synp_p1 | 2888 | 13.12 | 0.543 | 0.182 | 0.457 | 69551 | 0 | 1 | 324 | 0.0 |
| A02-002 | R10 | synp_p2 | 4263 | 12.64 | 0.612 | 0.171 | 0.501 | 21900 | 0 | 0 | 313 | 0.0 |
| A02-003 | R10 | synp_p3 | 4151 | 12.20 | 0.596 | 0.137 | 0.563 | 21939 | 0 | 2 | 290 | 0.0 |
| A02-004 | R10 | synp_p4 | 5581 | 17.00 | 0.835 | 0.157 | 0.551 | 49605 | 0 | 0 | 374 | 0.0 |
| A02-005 | R10 | synp_p5 | 3798 | 12.99 | 0.525 | 0.153 | 0.544 | 23126 | 0 | 0 | 332 | 0.0 |
| A02-006 | R10 | synp_p6a | 2285 | 12.23 | 0.579 | 0.181 | 0.510 | 17547 | 0 | 0 | 341 | 0.0 |
| A02-007 | R10 | synp_p6b | 5953 | 15.99 | 0.873 | 0.164 | 0.514 | 84792 | 0 | 2 | 418 | 0.0 |
| A02-008 | R10 | synp_p7 | 3747 | 12.28 | 0.583 | 0.174 | 0.549 | 22766 | 0 | 1 | 401 | 0.0 |
| A02-009 | R3 | synp_p1 | 1929 | 4.44 | 0.317 | 0.181 | 0.349 | 19678 | 0 | 0 | 352 | 0.0 |
| A02-010 | R3 | synp_p2 | 1793 | 3.80 | 0.255 | 0.178 | 0.357 | 34566 | 0 | 0 | 395 | 0.0 |
| A02-011 | R3 | synp_p3 | 1609 | 5.53 | 0.412 | 0.100 | 0.491 | 25770 | 0 | 0 | 177 | 0.0 |
| A02-012 | R3 | synp_p4 | 3703 | 7.15 | 0.517 | 0.139 | 0.523 | 28230 | 0 | 1 | 352 | 0.0 |
| A02-013 | R3 | synp_p5 | 2567 | 5.80 | 0.365 | 0.187 | 0.410 | 47627 | 0 | 0 | 401 | 0.0 |
| A02-014 | R3 | synp_p6a | 1892 | 4.15 | 0.265 | 0.185 | 0.316 | 45772 | 0 | 0 | 271 | 0.0 |
| A02-015 | R3 | synp_p6b | 3938 | 6.94 | 0.587 | 0.128 | 0.478 | 18754 | 0 | 2 | 227 | 0.0 |
| A02-016 | R3 | synp_p7 | 2400 | 5.30 | 0.351 | 0.211 | 0.444 | 23757 | 0 | 0 | 547 | 0.0 |
| A02-017 | R1 | synp_p1 | 2612 | 3.96 | 0.367 | 0.161 | 0.336 | 21719 | 0 | 0 | 285 | 0.0 |
| A02-018 | R1 | synp_p2 | 2142 | 3.46 | 0.315 | 0.197 | 0.351 | 18724 | 0 | 0 | 418 | 0.0 |
| A02-019 | R1 | synp_p3 | 1579 | 2.85 | 0.258 | 0.164 | 0.390 | 39937 | 0 | 1 | 416 | 0.0 |
| A02-020 | R1 | synp_p4 | 3384 | 5.81 | 0.475 | 0.145 | 0.456 | 27447 | 0 | 1 | 342 | 0.0 |
| A02-021 | R1 | synp_p5 | 1827 | 3.90 | 0.291 | 0.144 | 0.401 | 19263 | 0 | 0 | 267 | 0.0 |
| A02-022 | R1 | synp_p6a | 890 | 2.65 | 0.227 | 0.164 | 0.286 | 36865 | 1 | 0 | 148 | 0.0 |
| A02-023 | R1 | synp_p6b | 3807 | 5.70 | 0.551 | 0.149 | 0.471 | 20530 | 0 | 0 | 375 | 0.0 |
| A02-024 | R1 | synp_p7 | 1831 | 3.14 | 0.266 | 0.172 | 0.417 | 19433 | 0 | 0 | 377 | 0.0 |

"search queries" counts the `"event": "query"` lines in `retrieval_transcript.jsonl`. Since Amendment 1 these are
SearXNG queries, not Serper calls. "runner_errors" is the run total from the instrumentation summary. The W&B key
`Extractor_tool_usage_failed/runner` is per iteration, and only its last value is in the summary.

## `M5S` re-screen (A08 CPU screen, table mode, all 81 cells)

- **Completeness:** all 81 cells finished (9 tasks × {OpenFE, AutoFeat, raw} × seeds {42, 4, 5}).
  - Every cell has its JSON with `folds_done` = 5 and its 5 feature tables.
  - `run_a08_screen.log` ends `ALL_CELLS_DONE 2026-10-13T09:53:17Z`. The last cell was AutoFeat `synp_p5` f4, finished 09:53:16Z.
- **Rescore:** 405 episodes, 0 errors.
  - The preliminary table (`~/ro27_a08/screen_prelim/`, 391 episodes) lacked 14 AutoFeat episodes: 9 on `synp_p5` and 5 on `synp_p7`. All 14 score 0.
  - The 391 shared episodes are unchanged.

| dataset_id | OpenFE | AutoFeat | raw |
|---|---|---|---|
| synp_p0 | 0/15 | 0/15 | 0/15 |
| synp_p1 | **15/15** | **15/15** | 0/15 |
| synp_p2 | 0/15 | 0/15 | 0/15 |
| synp_p3 | 0/15 | 0/15 | 0/15 |
| synp_p4 | 0/15 | 0/15 | 0/15 |
| synp_p5 | 0/15 | 0/15 | 0/15 |
| synp_p6a | 0/15 | 0/15 | 0/15 |
| synp_p6b | 0/15 | 0/15 | 0/15 |
| synp_p7 | 0/15 | 0/15 | 0/15 |

**Gate: PASS.**
- **1 of 7** families is non-discriminating: P1 (OpenFE 15/15, AutoFeat 15/15). The gate threshold is ≥ 4.
- `synp_p0` has 0 recoveries in every arm.

This confirms the 07:50Z preliminary verdict in `sessions/S1013a.md`.

## Analysis conventions applied (and where the implementation had to choose)

- **Episodes.** For each row, the outer fold's selected iteration comes from `fold_aggregation_best_iterations.csv`
  (the `Trial` id).
  - The active set per iteration is the set of `Status == "Active"` attributes in that iteration's
    `intermediate_steps/df_attributes_explanations_trail_<n>.csv`.
  - Features are replayed on the fold's training rows by `recovery_detector.detect_episode` (formula mode). Training
    rows come from the same splitter call as `rogue_one_main.py`: stratified on p3/p4/p5/p7, KFold on the regression tasks.
  - All 120 scored episodes used one replay of the fold's last (append-only) table. The prefix check passed everywhere.
  - No fold retry, no fallback feature and no missing selection occurred.
  - Each run directory's `data.csv` is byte-identical to the canonical task folder.
- **`success` metric (classification).** The agent's logged `accuracy` is `accuracy_score` with balanced test
  sample weights, because A02 runs with `ROGUE_ONE_DO_BALANCING=True` (`tester/agent.py`). That makes it balanced
  accuracy, as the spec requires.
  - The matched A08 estimator is `accw_balanced` / `accw_balanced_sd` (balanced-weight fit, balanced accuracy).
  - For regression the base is `nrmse` / `nrmse_sd` (sqrt(MSE) / target mean), the same definition as the agent's `n_RMSE`.
- **`u_run` and `W_ref`.** Per attempt, the wall clock is W&B `wall_s`, or the launcher's `status.tsv` wall clock
  when the attempt has no W&B `wall_s` (the 3 failed first attempts).
  - `W_ref` = **7,886 s, A01's median `wall_s`: PLANNING ONLY**. No A03 `replace` rows exist yet in
    `RogueOne-AGENT2027-prefreeze` or `RogueOne-AGENT2027`.
  - Absolute u values (and the 0.85u re-tier trigger) will move when A03 lands. Criterion 4 and the decision will not.
- **Flag invariant: PASS.**
  - 73 `flags/*` keys over 24 rows. `ROUNDS` matches the arm and `STRATIFIED_FOLDS` matches the task type on every row.
  - Two keys differ by design and are reported, not counted as violations:
    - `flags/ROGUE_ONE_DATASETS`, the task itself;
    - `flags/LLM_PORT`, 8010 for S1013c and 8011 for S1013d (the 2×TP2 deviation).
- **Deviations from the A02-T12 recipe:**
  - The script reads each row's `output/` folder from the read-only run directories, which hold the same files the
    `run-output` artifact uploads. It does not download the artifacts. W&B supplies configs, summaries and `wall_s`,
    and is cross-checked against the local files with no mismatch.
  - No `ANALYSIS__a02_readout__*` W&B run was logged: this read-out lives in this file and the CSVs.
  - The container ran with `--cpus 8 --memory 32g` instead of `--cpus 16`.

## Data notes

- **W&B group `A02`** has 55 runs:
  - 4 smoke runs (S1012a/S1012b);
  - 24 tagged S1013a/S1013b, all excluded;
  - 27 valid runs (S1013c/S1013d): 24 final attempts plus the 3 failed first attempts `ne8z894y` (A02-006),
    `ns7awiij` (A02-011) and `5qtfz75h` (A02-022).
- **Failed first attempts:** W&B shows them as `finished` with no `wall_s`. The abort path exits rc 0, as §2 of the
  spec predicts. The W&B run count per row equals the launcher's attempt count on all 24 rows.
- **W&B config has no `row_id`.** Runs are matched to rows by (`arm_id`, `dataset_id`), with attempts ordered by `created_at`.
- **Excluded S1013a run `c2gjumk3`** (A02__R1__synp_p5) shows `finished`, with `wall_s` 5,336.6 s (created 08:00:42Z,
  so it ended at about 09:29Z). This contradicts Amendment 1's "no row finished". The run is excluded either way
  and was not read for this analysis, but the amendment's record should be checked.
- **`git_sha` in the run configs** is `2032f6b`, the CAIR checkout at launch. The image `agent2027-pre.3` was built
  from `b1c3518`. Between the two, only docs and manifests differ.

## Provenance

- **Runs:** CAIR `~/rogueone-runs-agent2027/S1013c` and `.../S1013d` (final attempts only). W&B
  `PlanBench/RogueOne-AGENT2027-prefreeze`, group `A02`; final-attempt run ids in the per-row table.
- **Image:** `rogue_one:agent2027-pre.3`, `sha256:29db4ceda59544935aa851beda20b9f186e4265930031948ce5db46d72c47940`
  (same as in every valid run's config). The analysis container mounted this repo's `scripts/agent2027`
  read-only over the image's copy.
- **Analysis code:** `scripts/agent2027/analysis/a02_readout.py` at `e617604`, on top of `agent2027/v2` @ `2032f6b`.
  - The detector `recovery_detector.py` is unchanged since `agent2027-pre.3`.
  - Unit tests in the image: `test_a02_readout.py` plus the detector tests, 48 passed. The full `tests/unit/agent2027`
    run has 1 known failure (`test_meta_dir_must_be_off_mount_and_off_repo`, which cannot raise without `.git`;
    see the session plan).
- **Commands** (CAIR, CPU only, no `--gpus`, read-only mounts; `S=~/ro27_a02_analysis/src`):
  - A08: `docker run --rm --cpus 8 --memory 32g ... -v ~/classicfe_a08/out:/a08:ro -v ~/code/RougeOne/tasks:/tasks:ro -v ~/rogueone-synp-meta:/meta:ro rogue_one:agent2027-pre.3 recovery_detector.py --features-table /a08 --tasks /tasks --meta /meta --out /out`
  - A02: `docker run --rm --cpus 8 --memory 32g --env-file .../PlanBench/.env ... -v ~/rogueone-runs-agent2027/S1013c:/runs/S1013c:ro -v .../S1013d:/runs/S1013d:ro ... a02_readout.py --runs /runs/S1013c /runs/S1013d --manifest /in/A02_M4.csv --tasks /tasks --meta /meta --a08 /a08 --a08-episodes /a08rescore/episodes.csv --out /out --wandb`
- **Outputs:**
  - `paper/results/agent2027/`: `a02_rows.csv`, `a02_task_arm.csv`, `a02_episodes.csv`, `a02_arm_summary.csv`,
    `a02_wandb_attempts.csv`, `a02_readout.json`, `a08_screen_episodes.csv`.
  - CAIR copies are in `~/ro27_a02_analysis/out/`, including the A08 `per_feature.csv.gz`.

## Detector amendment re-check (2026-10-19)

Henrik's G2 decision (pre-registration Amendment 2; `G2_memo.md` §2.2 and §7 item 2) made the
`no_gain_over_raw` rule (Part III.2) apply to every read-out. Until then `recovery_detector.py` ran it only in
`--features-table` mode, so this read-out used the registered formula-mode rule. Formula mode now scores the
replayed features with `evaluate_feature_table` by default. That adds three exclusions: a raw-column name,
`monotone_rho` ≥ 0.999, and a Spearman hit that beats its best single raw column by less than 0.01. Fallback
features are still excluded. `amended=False` (CLI `--registered-formula`) reproduces the old rule.

**Result: no A02 number that depends on the detector changes.**

| Quantity | Registered detector (2026-10-14) | Amended detector (2026-10-19) |
|---|---|---|
| Σ `rec_sel` R10 / R3 / R1 (of 40) | 15 / 13 / 13 | 15 / 13 / 13 |
| Σ `rec_any` R10 / R3 / R1 (of 40) | 15 / 13 / 13 | 15 / 13 / 13 |
| Per task `rec_sel` (all arms, of 15): p1 / p2 / p3 / p4 / p5 / p6a / p6b / p7 | 15 / 5 / 0 / 4 / 0 / 15 / 2 / 0 | 15 / 5 / 0 / 4 / 0 / 15 / 2 / 0 |
| Episodes where the two detectors differ | – | 0 of 120 (hit sets identical too: 0 features dropped) |
| Criterion 1 (net loss ≤ 4) | 2, PASS | 2, PASS |
| Criterion 2 (no ≥ 3/5 → 0/5 drop) | no drop, PASS | no drop, PASS |
| Criterion 3 (success R3 ≥ R10) | 6/8 vs 4/8, PASS | 6/8 vs 4/8, PASS (not detector-dependent) |
| Criterion 4 (u ratio ≤ 0.75) | 0.7504, FAIL | 0.7504, FAIL (not detector-dependent) |
| **Decision** | **R = 10** | **R = 10** |
| Without `synp_p1` (descriptive) | R10 10/35, R3 8/35 | R10 10/35, R3 8/35 |
| `synp_p7` leak canary | `rec_any` 0/15 | `rec_any` 0/15 |
| `M5S` (A08 table mode, 405 episodes) | p1 15/15 both methods, p0 0, gate PASS | identical `episodes.csv` (table mode unchanged), gate PASS |

How it was checked (CAIR, CPU-only containers, no `--gpus`, read-only run mounts):
- `a02_readout.py` was re-run end to end with the amended detector (image `agent2027-pre.3`, same mounts as
  in Provenance). `a02_episodes.csv` and `a02_wandb_attempts.csv` are byte-for-byte equal to the 2026-10-14 output.
- A side-by-side script scored all 120 episodes with both detectors. The registered scores reproduce the
  2026-10-14 `a02_episodes.csv` exactly.
- `recovery_detector.py --features-table` re-scored the A08 screen. Its `episodes.csv` equals the 2026-10-14 rescore.
- The same side-by-side check on the 125 A05 SYN-P episodes reproduces G2 exactly. Registered: P0 `rec_sel` 4/5,
  `rec_any` 5/5. Amended: P0 0/5. Every other SYN-P count is the same under both detectors.

**A change the detector did not cause (reported, not applied).** The re-run's W&B query found the 10 A03
`replace` rows in `RogueOne-AGENT2027`. So `W_ref` is now the A03 median `wall_s`, 9,782.9 s, instead of the
planning-only A01 fallback of 7,886 s that this read-out used. Only absolute u values move:

| | Before | After |
|---|---|---|
| R10 mean u | 1.285 | 1.036 |
| R3 mean u | 0.964 | 0.777 |
| R1 mean u | 0.848 | 0.683 |

The ratios do not move (R3/R10 0.7504, R1/R10 0.660), so criterion 4 and the decision stand. The G1 re-tier
trigger "R10 mean u per SYN-P run > 0.85u" still holds (1.036u). The "1.285u" figures above (and the pre-registration
disclosure "Scale of SYN-P runs") were computed on the planning-only `W_ref`. The read-out section above is left as written on 2026-10-14.
