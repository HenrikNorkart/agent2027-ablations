# G2 read-out memo (E11a; A05-T13/T14/T15) — DRAFT for Henrik

Gate G2, Mon 19 Oct 2026. Binding rules: `docs/agent2027/preregistration.md` Part VI.2, with Parts IV, V.3, V.5, V.6,
V.9, V.13, V.15 and V.19. Data: sessions `S1016a`/`b`/`c` (image `rogue_one:agent2027-v2.0` =
`sha256:910b6995…de78f` in every run config; `prereg_tag=agent2027-prereg`), 73 manifest rows (A05 55, A07 8, A03 10),
76 attempts. Computed by `scripts/agent2027/analysis/g2_stats.py` (`collect` and `analyze`), run in CPU-only
`rogue_one:agent2027-v2.0` containers on CAIR with the run folders, tasks and φ metadata mounted read-only, and W&B
`PlanBench/RogueOne-AGENT2027` (groups A05/A07/A03) for configs, summaries and `run-output` presence. Numbers were
computed 2026-10-19 08:20Z. No arm result was read for this memo.

**Timing.** The read-out was due by 10:00 Oslo. The cause of the repeated exception (§1) was identified from the
container logs at about 09:52 Oslo. The constants were computed after 10:00.

## 0. Verdict

**DECISION (Henrik, 2026-10-19): GO.** The repeated `inf` exception is a finding, not a re-freeze defect; v2.0
stays frozen, the one-retry rule applies to every arm, and the crash rate is a limitation (pre-registration
Amendment 2). Detector: amend `recovery_detector.py` and re-check A02.


**Under the pre-registered rule: NO-GO (conditional).** No unconditional trigger fired. The conditional trigger "the
same exception in ≥ 2 runs" fired: 3 A05 runs failed with the same exception. Its cause is a defect in Full-v2 code that
looks fixable in under 0.5 d and leaves every G1 decision unchanged.

**Henrik decides two points before this verdict is final:**
1. Does this count as a "fixable defect in Full-v2"? (§1 gives the case for and against.)
2. Does the 10:00 clause hold, given the cause was found at about 09:52 but the memo is late?

If either answer is no, the trigger is a finding and the verdict is **GO**. Every constant below is computed on the
current data either way. On a NO-GO, all of them are recomputed after the `v2.0r1` rerun.

| Rule (Part VI.2) | Count over A05's 55 runs (A07 / A03 beside them) | Fires? |
|---|---|---|
| Any inner-CV fallback ("falling back to the held-out metric") | 0 lines in all 76 attempt logs (0 / 0) | no |
| Any role with zero `tok_compl_*` (S, E, T) | 0 (0 / 0) | no |
| Missing traces (`trace_records` = `llm_calls_total`, 0 write errors), per-inner-fold scores (every completed trial has `inner_cv_fold_scores` and an inner-CV score) or `run-output` (W&B artifact) in a run that did not crash | 0 (0 / 0) | no |
| ≥ 3 cap hits | 0 (0 / 0) | no |
| ≥ 6 ITT failures | 0 (0 / 0) | no |
| Same exception in ≥ 2 runs | **3**: A05-028, A05-038, A05-044 (first attempts) (0 / 0) | **yes** |

## 1. The repeated exception (the three retried rows)

**What happened.** In each of these rows, attempt 1 exited with rc = 0 but wrote no `fold_aggregation_summary.csv`
(launcher reason `rc0-no-fold_aggregation_summary`). Under Part IV.3 that is a crash. Each row was retried once, alone,
at the end of its session, and attempt 2 finished OK. So none of them is an ITT failure, and ITT failures = 0. The three
failed first attempts are separate W&B runs: group A05 has 58 runs = 55 + 3.

| Row | Session | Task | Attempt 1 wall | Exception | Where (last Tester log line) |
|---|---|---|---|---|---|
| A05-028 | S1016a | `insurance` f42 r1 | 3,183 s | `XGBoostError: … Input data contains inf or a value too large` | `test_hypotheses`, step 2 trial 25, right after "prepared training matrices" |
| A05-038 | S1016b | `synp_p7` f42 r0 | 3,393 s | same | `test_hypotheses`, step 1 trial 16 |
| A05-044 | S1016b | `synp_p6a` f4 r0 | 1,021 s | same | `test_hypotheses`, step 0 trial 5 |

**Mechanism** (from the logs; the console does not print a Python traceback):
1. Agent features are replayed per fold by `rogueone/utils/fold_safe_features.py`. In all three runs the replay emitted
   `divide by zero` or `invalid value encountered in log1p`, so a feature took ±inf on a fold partition.
2. The append-time check (`ext/agent.py:524`, `:695`) rejects inf only on the full-training computation, not on the
   per-fold replay.
3. `tst/agent.py` `test_hypotheses` (around `:941-991`) passes the matrix to XGBoost with no inf→NaN step. XGBoost
   raises inside `QuantileDMatrix` creation.
4. `run_experiments` (`rogue_one_main.py:1887`) swallows the exception, so the container exits with rc 0 and no summary.

**Does it count under the rule?**
- It meets the trigger whether you count W&B runs (3) or rows (3, the A05 §6 wording). In all three the exception has
  the same type and message, inside the same Tester step. The exact innermost Rogue One line is inferred, not read from
  a traceback.
- **Case for "fixable defect":**
  - The fix is a few lines in Full-v2 code: map ±inf to NaN (XGBoost's `missing`) in the Tester matrices, and in the
    inner-CV and L1 fits.
  - It is under 0.5 d with a unit test.
  - It leaves R, the pool mode and D2 unchanged.
  - Today, any run whose matrix contains inf crashes. So the fix changes no non-crashing run.
- **Case for "finding":**
  - The retry absorbed all three crashes; ITT = 0 and no data was lost.
  - The rerun costs about 60–70u (A05 + A07 + A03, the 73 rows were 205 h of wall clock in total).
  - Without a fix the crash rate is 3 of 76 attempts (about 4%) and will recur in every arm. Any second crash of the
    same row becomes ITT.
- **If GO is chosen anyway:** the fix can only land after G2 as a default-off switch with the three merge guards
  (01 §2.2). Under Part VI.2, "after G2 such a fix becomes a limitation". Disclose the 3/76 crash-and-retry rate.

## 2. Frozen constants (computed on the current data)

### 2.1 s_{d,s}, σ_AA, Z, σ_d, homogeneity, M_d

`base_no_fe` reference (A05-T6; §6) in `G2_base_no_fe_ref.csv`. s_d = mean(s_{d,42}, s_{d,4}).

- **σ̂²_AA = 4.815 (σ_AA = 2.194 s_d units, 10 df). σ̂²_p = 0.450. Z = 0.915.**
  - Z ≥ 0.85, so a wording-only outcome applies: pairing by fold seed buys little. The paired analysis and the margin
    formula are unchanged.
- **Homogeneity (housing, insurance, crab):**
  - σ̂²_AA,R = 11.590 (3 df) and σ̂²_samp,R = 11.590 / 1.285 = 9.020 (6 W0–t17 pairs rescaled by A05's s_d).
  - **F = 1.285**, inside the central 95% of F(3, 6) = [0.068, 6.599]. A01 reports no drift. **Pooled.**
  - Classification datasets are never pooled.
- **M = 1.0, so M_d = σ_d** (Part IV.2). σ_d is the SD of {a = f42 r0, b = f42 r1, c = f4 r0}, 2 df, floored at 0.1 × s_d.
  For the pooled datasets, σ²_d = (σ̂²_d + SD²_t17) / 2, then the same floor.

| dataset | a | b | c | s_d | σ_d raw | floor | pooled | **σ_d = M_d** |
|---|---|---|---|---|---|---|---|---|
| balance-scale | 1.0000 | 1.0000 | 1.0000 | 0.02305 | 0.0000 | 0.002305 | – | **0.002305** (floor) |
| junglechess | 0.9723 | 0.9536 | 0.9818 | 0.0039 | 0.01438 | 0.00039 | – | **0.01438** |
| vehicle | 0.7520 | 0.7492 | 0.7763 | 0.02675 | 0.01493 | 0.00268 | – | **0.01493** |
| heart | 0.8196 | 0.8501 | 0.8497 | 0.0217 | 0.01748 | 0.00217 | – | **0.01748** |
| credit-g | 0.6552 | 0.6948 | 0.6805 | 0.03255 | 0.02005 | 0.00326 | – | **0.02005** |
| myocardial | 0.5783 | 0.5933 | 0.6173 | 0.0411 | 0.01967 | 0.00411 | – | **0.01967** |
| housing (nRMSE) | 0.2836 | 0.2426 | 0.2242 | 0.00525 | 0.03040 | 0.00053 | yes (t17 0.021) | **0.02613** |
| insurance (nRMSE) | 0.3628 | 0.3990 | 0.3704 | 0.0127 | 0.01911 | 0.00127 | yes (t17 0.063) | **0.04655** |
| crab (nRMSE) | 0.2380 | 0.2314 | 0.2343 | 0.0086 | 0.00334 | 0.00086 | yes (t17 0.024) | **0.01713** |
| wine (nRMSE) | 0.1194 | 0.1199 | 0.1193 | 0.00145 | 0.00028 | 0.000145 | – | **0.000282** |

- Pooling **raises** M_d on insurance (0.019 → 0.047) and crab (0.003 → 0.017), because t17's seed SD is larger there.
  This is the pre-registered formula.
- **A13's M:** M = 1.0 Δ_std. A13's no-harm bounds are mean ∓ 0.637 (Var(Δ_std) ≈ 1.5). "No harm beyond M" requires a
  lower bound > −1.0.

### 2.2 Recovery, informative tasks, ICC, sentinel, headroom

`rec_sel` out of 15 episodes (f42 r0 / f42 r1 / f4 r0, out of 5 each). `G2_recovery_by_task.csv` holds both detectors.

| task | rec_sel /15 | f42r0 | f42r1 | f4r0 | rec_any /15 |
|---|---|---|---|---|---|
| synp_p1 | 15 | 5 | 5 | 5 | 15 |
| synp_p2 | **8** | 3 | 2 | 3 | 9 |
| synp_p3 | 0 | 0 | 0 | 0 | 0 |
| synp_p4 | **1** | 0 | 0 | 1 | 1 |
| synp_p5 | 0 | 0 | 0 | 0 | 0 |
| synp_p6a | 15 | 5 | 5 | 5 | 15 |
| synp_p6b | **6** | 1 | 4 | 1 | 6 |
| synp_p7 | 0 | 0 | 0 | 0 | 0 |
| synp_p0 (f42 r0, /5) | 0 | 0 | – | – | 0 |

**Detector amendment (Part VI.2: fixed in analysis, never rerun).**
- The registered formula-mode detector gave **P0 `rec_sel` = 4/5** (`rec_any` 5/5): a detector bug.
- Part III.2 says the `no_gain_over_raw` fix (commit `a6b64ac`) applies to every arm. In `recovery_detector.py` it runs
  only in `--features-table` mode (the A08 screen), so agent runs, A02 included, never got it.
- `g2_stats.py` applies the table-mode exclusions (raw name, `monotone_rho` ≥ 0.999, `no_gain_over_raw` 0.01) to the
  replayed agent features. The result: **P0 = 0/5**, and **every SYN-P 8 count is unchanged**. All constants use the
  amended detector.
- `recovery_detector.py` itself was not changed: the A02 read-out was not reproduced under the fix. Henrik decides
  whether to amend it.

- **Informative tasks** (Full strictly between 0/15 and 15/15): **`synp_p2`, `synp_p4`, `synp_p6b` (3)**.
  - Fewer than 4, so a wording-only outcome applies. None is P1, so the confirmatory population has the same 3.
  - Per contrast:

    | Contrast | Population | Informative tasks |
    |---|---|---|
    | C1a, C1b, C5b, A09, A12, A16, A20 | SYN-P 7 minus P1 | p2, p4, p6b |
    | C3 | D3 {p3, p5, p2} | p2 only |
    | C4b | p6a, p6b | p6b only (p6a 15/15) |
    | C4a | P7 | P7 = 0/15 (boundary allowed) |
    | C5a | SYN-P 7, Qwen rungs | p2, p4, p6b |

- **Recovery ICC.** Primary: SYN-P 7, 105 episodes.
  - The GLMM is singular (Laplace ML: σ_task = 4.92, σ_task:seed = 0.002, σ_run = 0.47; GLMM ICC 0.873). So the
    pre-registered fallback applies: **ANOVA ICC on run proportions = 0.899**.
  - SYN-P 8 sensitivity: 0.909 (GLMM 0.895).
  - ICC > 0.5, so a wording-only outcome applies: effective n ≈ informative tasks × partitions, and "inconclusive" is
    expected and stated.
- **A06 sentinel task: `synp_p2`, Z = 5/10.**
  - `synp_p2` and `synp_p6b` both have seed-42 r0 + r1 = 5/10. The tie is broken by the 15-episode rate: p2 8/15
    (distance 0.033) beats p6b 6/15 (0.100).
  - Ranking of the 7 candidates: p2, p6b, p4, p1, p3, p5, p6a.
  - s_task (`base_no_fe` fold SD at seed 42): p2 0.0064.
- **A09 headroom flag:** 3 gate-passing tasks have Full `rec_sel` ≥ 0.4 (p2, p6a, p6b). That is not fewer than 3, so
  **no flag**.

### 2.3 A06 band table (sentinel `synp_p2`; σ_AA,m over the 8 SYN-P A/A pairs at seed 42, 8 df)

Half-width = max(3·√1.5·σ_AA,m, floor_m). The centre is the mean of A05-032 and A05-048 on the stated scale.

| channel | metric (scale) | σ_AA,m | centre | half-width | band |
|---|---|---|---|---|---|
| cost | `tok_compl_total` (log) | 0.0467 | 14.0428 | 0.1717 | [13.871, 14.214] |
| cost | `llm_calls_total` (log) | 0.0411 | 7.7256 | 0.1511 | [7.575, 7.877] |
| time | `wall_s` (log) | 0.0944 | 9.1418 | 0.3467 | [8.795, 9.489] |
| execution | `err_per_accepted` (log1p) | 0.0153 | 0.0470 | 0.1 (floor) | [−0.053, 0.147] |
| execution | execute rate `fun_executes/fun_proposed` | — | — | — | **not computable: no `fun_*` key is logged in v2.0 summaries** |
| execution | `context_errors` (log1p) | 1.399 | 0.0 | 5.140 | [−5.14, 5.14] |
| search | `serper_ok_false_rate` (raw) | 0.0043 | 0.0079 | 0.05 (floor) | [−0.042, 0.058] |
| score | `heldout_score`/s_task (nRMSE/0.0064) | 1.799 | 12.666 | 6.611 | [6.055, 19.277] |

- The `context_errors` band is very wide: one A/A pair differs by about 100 context errors. Several A05 runs log
  exactly 100–101 `context_errors` (for example A05-043, -038, -035, -022, -018, -010), and the maximum prompt was
  126,128 tokens against `MAX_MODEL_LEN` 131,072. This is descriptive, not a G2 trigger.
- The execute-rate metric has no data source at v2.0. Henrik decides whether to drop it or define it from the funnel
  (E11b).

### 2.4 Tokens: B_d, role and Tester-stage shares, A12 branch, A19 τ

- **B_d:** the median completion tokens per iteration over each dataset's 150 A05 iterations (50 for P0). Every
  iteration had status `completed`.

  | dataset | B_d | dataset | B_d |
  |---|---|---|---|
  | balance-scale | 19,163 | synp_p1 | 24,163 |
  | junglechess | 25,175 | synp_p2 | 23,704 |
  | vehicle | 32,962 | synp_p3 | 26,398 |
  | heart | 28,221 | synp_p4 | 28,813 |
  | credit-g | 27,214 | synp_p5 | 24,197 |
  | myocardial | 29,656 | synp_p6a | 24,874 |
  | housing | 22,852 | synp_p6b | 28,833 |
  | insurance | 21,130 | synp_p7 | 24,397 |
  | crab | 22,753 | synp_p0 | 24,875 |
  | wine | 22,688 | | |
- **Role shares** (`tok_compl_<role>`/`tok_compl_total`, search charged to the caller):

  | Population | s_S | s_E | s_T | Tester assessment | Tester pruning |
  |---|---|---|---|---|---|
  | Median over the 21 SYN-P 7 runs | 0.091 | **0.425** | 0.488 | 0.212 | 0.274 |
  | Median over all 55 runs | 0.086 | 0.428 | 0.478 | 0.211 | 0.261 |

  Per-dataset shares are in `G2_token_shares.csv`. The two stages are separable: assessment + pruning + other =
  `tok_compl_tester` on every run.
- **A12 token-rule branch:**
  - Both Tester stages are below 30% of `tok_compl_total`, and they are separable. So the pre-registered branch is
    **(i): the replaced stage's own tokens**.
  - A12-T3 notes that (i) is degenerate for FX: the replaced stage has 0 tokens by construction. It recommends judging
    A and J on `tok_compl_tester`. **Henrik decides:** (i) as written, or the A12-T3 reading. (ii) does not apply,
    because the stages are separable.
- **A19 τ (also the CAP-2 big-model token threshold):** the median s_E is 0.425 ≤ 0.50, so **τ = 0.60**. It comes from
  O120 shares while the reference is Q122 (disclosed). The dated addendum is due before A19 launches.

### 2.5 Other G2 items

- **u_SYN-P** = 9,279 s / 9,894 s = **0.938**. Medians over sessions: Core-10 2.75 h, SYN-P 2.58 h.
- **A21 validity (G2 rule).** Both parts pass, and A21 can use the token axis:
  - Per-iteration tokens sum to `tok_compl_total` within 1% on **55/55** runs.
  - The k = 10 inner-CV pick reproduces `fold_aggregation_best_iterations.csv` on **55/55** runs (ties need
    `float_precision="round_trip"`).
  - Per-inner-fold scores are present on every completed trial.
- **A10-T18:** a per-iteration active-feature log is present (`df_attributes_explanations_trail_<i>.csv` with `Status`,
  `Fallback`, `Seeded`). No option is needed.
- **Recomputed power (PROVISIONAL; the method is not defined).**
  - 01 §4.7 says only "recomputed at G2 from M0's ICC and the number of informative tasks".
  - Scaling used here: MDE = planning MDE × √(6/n_inf) × √(1 + 4·ICC), with n_inf = 3 and ICC = 0.899.
  - Results:

    | Contrast | Planning MDE | Scaled MDE |
    |---|---|---|
    | C1a | about 0.265 | 0.80 |
    | FX (C2a–c) | ±0.11 | ±0.33 |
    | RT3 | 0.26 | 0.79 |
    | C3 | 0.425 | > 1 (no effect detectable) |
    | C4a | 30–40 pp | 30–40 pp (fixed population) |
    | C4b | about 60 pp | about 60 pp (fixed population) |

  - The message is that only very large effects are detectable on SYN-P. Henrik should confirm or replace the scaling
    before it is quoted.
- **Run health (descriptive):**
  - `success` 48/55. The 7 misses: housing f42 r0 and r1; wine f42 r0, f4 r0 and f42 r1; synp_p6b f4 r0; synp_p2 f42
    r1. All are held-out scores more than 2 fold-SDs worse than `base_no_fe`, which is a finding about v2, not a
    trigger.
  - `raw_coverage` = 1.0 at iteration 0 in every fold of all 55 runs, so the augment manipulation check passes.
  - The replay check ran on 55/55 runs: 78 non-reproducing folds in 36 runs (the Finding-39 effect).
  - `l1_split_disjoint` = 1 on 55/55.
  - The maximum `serper_ok_false_rate` is 0.044, below 5%.
  - **P0:** `rec_sel` 0/5 (amended detector). `heldout_score` 0.0939 against `base_no_fe` 0.0911 ± 2 × 0.0036, so it is
    within 2 s.
  - `scan_leakage.py` was **not run**: it hard-codes the t17 run roots.

## 3. A03 fallback read-out (Part V.3)

All 10 pairs are complete: A05 `full` f42 r0 against A03 `replace`. Δ_d = augment − replace, with the sign flipped for
nRMSE, so Δ > 0 means augment is better. σ̂_d = max(A01 σ_samp,d, t17 SD_d, 0.1 × `base_no_fe` fold SD at seed 42).

| dataset | aug | rep | σ̂_d (term) | Δ_std | Δ/(√2σ̂) |
|---|---|---|---|---|---|
| balance-scale | 1.0000 | 1.0000 | 0.00247 (floor) | 0.000 | 0.000 |
| junglechess | 0.9723 | 0.9422 | 0.070 (t17) | 0.430 | 0.304 |
| vehicle | 0.7520 | 0.7707 | 0.029 (t17) | −0.646 | −0.457 |
| heart | 0.8196 | 0.7544 | 0.019 (t17) | 3.432 | 2.427 |
| credit-g | 0.6552 | 0.6506 | 0.0141 (A01) | 0.327 | 0.231 |
| myocardial | 0.5783 | 0.5591 | 0.0170 (A01) | 1.127 | 0.797 |
| housing | 0.2836 | 0.2492 | 0.021 (t17) | −1.636 | −1.157 |
| insurance | 0.3628 | 0.3689 | 0.063 (t17) | 0.097 | 0.068 |
| crab | 0.2380 | 0.2543 | 0.024 (t17) | 0.676 | 0.478 |
| wine | 0.1194 | 0.1205 | 0.0010 (t17) | 1.101 | 0.779 |

- **(a)** mean Δ_std = **+0.491** ≥ −0.75: holds.
- **(b)** worst pair −1.157 (housing) ≥ −2.5, and no augment failure where replace completed: holds.
- **(c)** untested (disclosed).
- **Outcome: augment stands.** No revert-or-limitation choice is needed.
- The sensitivity analysis (both arms ITT-imputed) is identical, because there are no failures.

## 4. Leak checks

- **P7 canary:** A05 `synp_p7` `rec_any` = **0/15** (`rec_sel` 0/15) under both detectors. **It does not fire**; A14's
  §6 rule is not triggered.
- **A15 leak rule on the A05 P6b runs** (A05-037, -045, -053, each against its P6a twin):
  - `a15_leak_audit.py` (A15-T22) does not exist, so `g2_stats.py` implements checks (a)–(d).
  - The frozen term list `synp_meta/p6_domain_terms.txt` (A15-T7) is **absent on CAIR**. A derived list was used:
    - **strong:** reynolds, viscosity, kinematic viscosity, flow velocity, pipe diameter, pipe flow;
    - **other:** every P6a column name and its distinctive tokens, and the vendor level strings; generic English words
      were dropped.
  - Results:

    | Check | Result |
    |---|---|
    | (a) P6b system prompts, tool docstrings and first user messages | **0** twin-specific hits; 0 template-constant hits; 0 hits in later agent-written messages |
    | (b) feedback-pool `dataset_id` | = {`synp_p6b`} in all 3 runs |
    | (c) agent code | 69 pattern hits: 66 are `read_csv(io.StringIO(<literal table>))` or the word "global". 3 are file-read attempts, `read_csv('')`, `read_csv('precomputed_metrics.csv')` and `read_csv('<<PLACEHOLDER>>')` (A05-037, A05-053); none targets P6a material, the cache, W&B or another run, so for review only |
    | (d) twin data | `id` column and `labels.csv` byte-identical, so the fold test sets are identical |

  - **No input leak. C4b stays valid.**
  - Henrik should confirm the derived term list, or supply A15-T7's list and its hash, and the audit is re-run.

## 5. Proposed box order, Thu 22 – Wed 28 Oct (D12; Henrik pins)

**Measured run lengths** (`PARALLEL=6` per O120 TP2 server):
- Median run: Core-10 2.75 h, SYN-P 2.58 h, A07 3.09 h.
- Session throughput: S1016a 26 attempts in 13.3 h, S1016b 26 in about 15 h, S1016c 24 in about 13 h. That is **about
  1.85 runs/h per TP2 server**, or about 3.7 runs/h for two servers (4 H100).
- 1u of wall clock is about 2.75 h per client slot.

**Assumptions:**
- At most 4 H100 (Part I.1; the weekend 8-GPU allowance covered A05 only).
- D12 lets rows of different ablations share one server as clients.
- Row counts come from the manifests now on `origin/agent2027/v2` (A12 57, A13 20, A14 18) and the ablation documents
  (A16 26, A19 28 under CL1, A20 7, A17 Q122).

| Slot | GPUs | Session | Rows | Estimate |
|---|---|---|---|---|
| Wed 21 ≈ 20:00 → Thu 22 ≈ 15:00 | 2 × O120 TP2 | `S1021a`: A12 block 1 (p3, p5, p2, p6a; 32) + A13 (20) + A14 (18) | 70 | ≈ 19 h |
| Thu 22 night (inside or right after `S1021a`) | same servers, as clients | A16 crash-only pilot `S1022b` (2 rows, v2.4) | 2 | ≈ 3 h; meets K-SET (Fri 23) |
| Fri 23 morning → Fri 23 night | 2 × O120 TP2 | `S1023a`: A12 block 2 (p6b, p4, p1; 24) + A16 SYN-P (14) | 38 | ≈ 10–11 h |
| Fri 23 night → Sun 25 | Q122 TP2 + Q4 TP1 + Q0.8 TP1 | A19 `S1022a` (28, CL1); A20 Q4 rows (7) and A18 small-rung replays as clients of A19's Q4/Q0.8 servers | 35 + replays | 20–39 h (A19 §1; Qwen throughput not yet measured) |
| Sun 25 → Wed 28 | Q122 TP2 ‖ O120 TP2 | A17 Q122 session ‖ A16 Core-10 (10) on one O120 server; Tue 27: A18 O120 self-replay and A17 O120 F0 pools as clients of that server | — | A16 Core-10 ≈ 5.5 h on one server |
| Wed 28 | as free | RV1/RV3 if triggered at G4, re-queues, KT3 latest-start slack | — | — |

- On a **NO-GO**, the `v2.0r1` rerun of A05 + A07 + A03 (73 rows ≈ 20 h on two servers) goes first, and every slot
  above shifts by about one day. A19 (latest full-matrix launch Tue 27) and A16 Core-10 (by L-LAST Thu 29) still fit.
  The A17 Q122 window shrinks.

## 6. Amendment text: A05-T6 `base-no-fe-ref` floor values (was due Sat 17 Oct; written Mon 19 Oct)

> **Amendment (2026-10-19, A05-T6; Part VIII.2 row "Sat 17 Oct").** The `base_no_fe` reference was not built on
> Sat 17 Oct. It was computed on Mon 19 Oct, before any arm result was read, with the A05-T6 command:
> `classicfe:4 classicfe_arm.py <ds> <task> raw --folds kfold --fold-seed <s>`, `--network none`, `--cpus=4`, in
> transient containers, one cell at a time, from `~/code/RougeOne` at `04a2225`. It covers Core-10 × {42, 4}.
>
> The SYN-P 8 + P0 values are taken from the A08 CPU screen raw arm. Its fold schemes match D4: stratified on p3, p4,
> p5 and p7, KFold elsewhere. The task-hash check named in A05-T6 was not repeated.
>
> Metric: `accw_balanced` (C) or `nrmse` (R); the fold SD is the population SD over 5 folds. The floor term of the
> Part V.3 σ̂_d table (0.1 × fold SD at seed 42) is applied mechanically:
>
> | dataset | fold SD f42 | **floor (0.1×)** | fold SD f4 | σ̂_d after floor (decided by) |
> |---|---|---|---|---|
> | balance-scale | 0.0247 | **0.00247** | 0.0214 | 0.00247 (floor) |
> | junglechess | 0.0048 | **0.00048** | 0.0030 | 0.070 (t17) |
> | vehicle | 0.0315 | **0.00315** | 0.0220 | 0.029 (t17) |
> | heart | 0.0318 | **0.00318** | 0.0116 | 0.019 (t17) |
> | credit-g | 0.0421 | **0.00421** | 0.0230 | 0.0141 (A01) |
> | myocardial | 0.0630 | **0.00630** | 0.0192 | 0.0170 (A01) |
> | housing | 0.0070 | **0.00070** | 0.0035 | 0.021 (t17) |
> | insurance | 0.0193 | **0.00193** | 0.0061 | 0.063 (t17) |
> | crab | 0.0096 | **0.00096** | 0.0076 | 0.024 (t17) |
> | wine | 0.0011 | **0.00011** | 0.0018 | 0.0010 (t17) |
>
> The floor decides σ̂_d only for `balance-scale`. For `wine`, t17's 0.0010 exceeds the floor 0.00011. The full
> reference (means, both seeds, SYN-P) is in `docs/agent2027/results/G2_base_no_fe_ref.csv`. The W&B artifact
> `base-no-fe-ref` (`ANALYSIS__base_no_fe__…`) has not been logged yet.

## 7. Open items for Henrik

1. **The verdict** (§0, §1): is it a fixable defect, and does the 10:00 timing clause hold?
2. **Detector:** formula mode lacks the Part III.2 raw-gain fix. Amend `recovery_detector.py`, and re-check A02?
   A02's counts would only change through hits that are no better than a raw column.
3. **A12 token branch:** (i) as written, or A12-T3's `tok_compl_tester`.
4. **A06 band, execute-rate metric:** `fun_*` is not logged in v2.0. Drop the metric or define it.
5. **Power:** the method is not defined; the scaling in §2.5 is provisional.
6. **A15 term list** (A15-T7) is absent; the audit used a derived list (§4).
7. **Not done:** the G2 memo and constants are not logged to W&B (`ANALYSIS__g2_stats__20261019`, group `ANALYSIS`);
   `base-no-fe-ref` is not logged; `a05-handoff` is not posted.
8. **Observed on CAIR at about 10:05:** session `S1019q` (`A06-101`, `sentinel_Q4`, `synp_p2`, Q4) had been running
   since about 09:30, before this memo. Its task is fixed by A06, not by the G2 sentinel rule. Part VI.2 says G2 is
   computed "before any arm launches".

## Files

`scripts/agent2027/analysis/g2_stats.py`; tests `tests/unit/agent2027/test_g2_stats.py` (13 passed in the
`agent2027-v2.0` image). Results in `docs/agent2027/results/`:

| File | Content |
|---|---|
| `g2_constants.json` | every constant |
| `G2_sigma_d.csv` | σ_d and M_d table |
| `G2_recovery_by_task.csv` | recovery per task, both detectors |
| `G2_A06_band.csv` | A06 band table |
| `G2_token_shares.csv` | role and Tester-stage shares per dataset |
| `G2_A03_pairs.csv` | A03 fallback pairs |
| `G2_base_no_fe_ref.csv` | `base_no_fe` reference |
| `G2_rows.csv` | per-row health and scores |
| `G2_attempts.csv` | per-attempt status and exceptions |

The fetched run data stays out of git.
