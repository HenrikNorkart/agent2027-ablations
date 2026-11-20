# A21 — M2: Anytime and iso-cost curves of the 10-iteration schedule (post-hoc)

| Field | Value |
|---|---|
| **ID** | A21 · M2 |
| **Claim / Holm contrast** | None of its own. Descriptive, outside Holm. It supplies the iso-cost reading for **C1** (the only budget-matched view of **C1b**, plus SET-phased for **C1a**), cost context for the "reducible" verdicts **C2a/C2b**, a descriptive **C3** reading ("does the loop keep improving?"), and the F2 anytime curves that A17 overlays on F0-best-of-k |
| **Tier** | T1-minimum ("free post-hoc analysis"; breakdown §2 row "M2 / M3": 0 runs, 0u, CPU, earliest start Mon 19 Oct) |
| **Owner** | Henrik |
| **Start / finish** | Data contract into E3 by **Thu 8 Oct** · checked on the G0a smoke **Mon 12 Oct** and the freeze smoke **Fri 16 Oct** · rules pre-registered **Fri 16 Oct 12:00** (X10) · code for the preliminary **Sat 17 – Sun 18 Oct** (needs D1 weekends) · preliminary on A05 **Mon 19 Oct** (G2) · full code and draft iso-cost points **Tue 27 Oct** (E11b, G4) · final **Sun 1 – Tue 3 Nov** · paper **Wed 4 – Fri 6 Nov** · package **Thu 12 Nov** |
| **Budget** | **0 new runs · 0u · 0 GPU-hours.** It reads 368–370 post-freeze runs and 44–51 pre-freeze runs that other ablations already paid for (§4, "Budget arithmetic"). CPU: ≲ 1 h per full pass on 8 CPUs (estimate; measure on the Mon 19 pass). Developer time (sum of §5 efforts): **≈ 2.35–2.5 d**, of which 0.6–0.75 d is code (0.35 d before G2: T6, T7, T9; 0.25–0.4 d inside E11b: T8) and ≈ 1.75 d is checks, CAIR passes, analysis and writing. ≈ 0.5–0.65 d of the total sits inside other budgets (T3 in A05-T3, T4 in X10, T8 in E11b); ≈ 0.75 d falls in the Nov 1–12 statistics/writing slots that the breakdown already reserves for M2 |
| **Models** | None are run. The data come from O120 (`openai/gpt-oss-120b`, all roles and W) and, for the cross-rung points, the A17/A19/A20 rungs (R9) |
| **Status** | Not started |
| **Depends on** | E3 (per-iteration tokens, per-inner-fold scores, per-iteration feature log, `run-output`; a minimal `scripts/agent2027/analysis/fetch_wandb.py` already exists at G0a because E3's done-when requires "the analysis fetch script reads it"), E1 (config keys), E2 (per-fold round counter), E8a/E11b (detector with per-iteration output; full `fetch_wandb.py`), X5 (comparator tokens), X10, A05 data and A05-T6 (`base-no-fe-ref`), A10-T18 (per-iteration active-feature log on the v2.0 smoke), G2 (σ_d), A08 gate labels (M5S), A23 (`flop_proxy` definition); the arm data from A07–A20; A02/A03 (pre-freeze panel) |
| **Feeds into** | Fig. 2 (right: Full's anytime line and the iso-cost points); the Table 2 cost/iso-cost reading; appendix "M2 curves"; the C1 sentence; A10, A11, A13, A16 (iso-cost secondaries; read-outs only, those documents do not depend on A21); A17 (F2 anytime curves; read-out only); A23 (Full front); the short paper (winner's-curse citation) |

*Every "hoped" result below is a hypothesis. Code references are to RougeOne `nested-cv-rebuttal` @ `07cbf93` and were re-verified line by line on Wed 7 Oct (the cited files are byte-identical at `1bdaf42`, the current branch tip). Short paths (01 §7 R0): `main.py` = `rogue_one_main.py`; `sci/agent.py`, `tst/agent.py` = `rogueone/llm/agents/core/{scientist,tester}/agent.py`; `clients.py` = `rogueone/llm/clients.py`; `config.py` = `rogueone/utils/config.py`; `wandb_utils.py` = `rogueone/utils/wandb_utils.py`. "01 §n" refers to `docs/ablations/01_shared_infrastructure.md`.*

---

## 1. What we test and why it matters

Suppose every logged run had stopped after k ∈ {1, …, 10} iterations per fold and then picked its reported iteration by inner CV from that prefix. How much recovery and held-out score would be lost, and how many tokens saved? A21 also truncates Full to the tokens each cheaper arm or external comparator actually spent, and asks whether truncated Full still recovers as much ("iso-cost"). With Solo-auto cut, this is the only budget-matched view of C1b. **It is not an iteration-budget ablation.** The Scientist sees "Round i/N" and switches between exploring and exploiting by round, so the paper must say "anytime curve of a 10-iteration schedule". True N = 3 runs (M2-N) are T2 and will not run. Tables and figures report the grid k ∈ {1, 2, 3, 5, 7, 10}; all ten k are computed, because `anytime_auc` and the iso-cost point k\* need every k.

## 2. Background you need

**How a run produces the data.**
- A run is 5 outer folds × 10 iterations. `forward_pass` (`main.py:536`) derives `fold_run_id` and `iteration_in_fold` (0-based, `:539-540`) and resets fold state at iteration 0 (`:543-544`).
- Each completed trial appends one row to `df_test_pool` (`:799-820`) with `status`, `test_fold_id`, `iteration_in_fold`, the held-out metrics and the inner-CV score.
- Rows with no Tester result are also appended:
  - an iteration with 0 productive features at `iteration_in_fold > 0` becomes `status="iteration_skipped"` (`:639-663`), and the run continues (`:1449-1454`);
  - at iteration 0 the whole dataset aborts (`:665-717`), which makes the run an ITT failure. Any other non-`completed` status also ends the run (`:1456-1471`).
  - Consequence: in a run that finishes, iteration 0 of every fold is `completed`, so every prefix k ≥ 1 has at least one completed row per fold. The only gaps inside a prefix are `iteration_skipped` rows.
- The pool is written to `output/df_test_pool_final.csv` (`:1388`, `:1394`). Feature values are dumped every iteration to `output/intermediate_steps/df_entities_attributes_trail_<index>.csv` (`dump_data`, `:839-847`, called at `:833`); per-iteration pandas commands are not.
- **Selection (the control rule).** `compute_fold_aggregation` (`:1074-1205`) uses completed rows only and needs all 5 folds (`:1101-1102`).
  - If `inner_cv_accuracy` (C) / `inner_cv_RMSE` (R) is missing or all-NaN across the run's completed rows, it falls back to the held-out metric and prints a warning (`:1114-1134`).
  - Otherwise, per fold, the argmax of `inner_cv_accuracy` (C) or the argmin of `inner_cv_RMSE` (R) (`:1137-1154`). Ties go to the first row (pandas `idxmax`/`idxmin`). If a fold's inner values are all NaN, the fold's last completed row is used (`:1151-1153`).
  - The reported score is the mean over the 5 selected rows (`:1166-1175`); the selected rows are written to `fold_aggregation_best_iterations.csv` (`:1353-1357`).
- **Inner CV.** `_inner_cv_score` (`tst/agent.py:194-305`) runs 4 inner folds (`:199`) and returns **only their mean** (`:304-305`); failed inner folds are dropped, so the mean can rest on fewer than 4 scores (`:292-299`). It is called at `:1722-1728` (R) and `:1990-1997` (C).
  - Classification inner CV fits with balanced weights but scores plain, unweighted accuracy (`:270-279`). The reported `accuracy` is balanced-weight accuracy under `ROGUE_ONE_DO_BALANCING=True` (`:1874-1878`, `:1915-1919`).
  - Regression inner CV uses raw RMSE. The reported `n_RMSE` = RMSE / mean(target) (`:359-361`, `:1788`). `RMSE` and `R2` are also in the trial row (`:410-424`).
- **Tokens today.** `_record_token_usage` (`clients.py:499-514`) accumulates per `agent_name` and logs to W&B via `log_llm_token_usage` (`wandb_utils.py:136-167`), one `wandb.log` per LLM call without a step, so every call advances the W&B step. There is **no fold or iteration index**. The streaming path requests usage chunks (`clients.py:324-336`).
  - The only per-iteration W&B markers are `extractor_iteration/*` (`main.py:958-980`, in `_log_extractor_iteration_activity` `:905`, called after extraction at `:594`) and `trial/*` (`tst/agent.py:2186-2211`, after testing). Both go through `safe_wandb_log` (`wandb_utils.py:39-65`), which never regresses the step, so the history order is call order.
- **Why "anytime", not "budget".** The round prompt is `sci/agent.py:472-475` (per-fold "Round i/N" after E2 item 2; today `main.py:553` passes the global index). The rule "exploratory strategies if … an early round" is `sci/agent.py:440`.

**What A21 needs from E3/E8a (the data contract, A21-T1).**
1. Every trace record in `output/llm_traces.jsonl.gz` is tagged with `test_fold_id`, `iteration_in_fold` and the charged role (search charged to the caller), and carries prompt and completion tokens. The tag comes from a context variable set in `forward_pass` right after `main.py:540`.
2. Each completed trial row carries its inner-fold scores (proposed column `inner_cv_fold_scores`, a JSON list of 2–4 floats whose mean is the row's `inner_cv_*` value).
3. A per-(fold, iteration) active-feature log (name, `Pandas Command`, `Status`, fallback flag). This is the log A10-T18 checks on the v2.0 smoke (re-checked by A10-T4 item 4). If it is missing, A21 uses A10-T18's G2 fallback: `rec_*@k` is scored from `df_entities_attributes_trail_<index>.csv` with every column counted active, labelled an upper bound. (A10-T5's extra `df_attributes_explanations_trail_<index>.csv` is written for A10 rows only and does not help A05.)
4. `scripts/agent2027/analysis/recovery_detector.py` writes one row per run × fold × iteration (`recovery_by_iteration.csv`, A21-T9).

**Instruments.**
- **Core-10:** `balance-scale`, `junglechess`, `vehicle`, `heart`, `credit-g`, `myocardial` (C); `housing`, `insurance`, `crab`, `wine` (R).
- **SYN-P 8:** `synp_p1`, `synp_p2`, `synp_p3`, `synp_p4`, `synp_p5`, `synp_p6a`, `synp_p6b`, `synp_p7`. SYN-P 7 = SYN-P 8 minus p7. Task types as confirmed by E8a in `synp_meta/registry.yaml`.
- **P0:** `synp_p0`, seed 42 only, held-out negative control (`rec` = 0 by construction, so P0 is left out of the recovery curves).
- **Brute-force gate (A08, M5S):** a task where OpenFE or AutoFeat recovers φ in ≥ 14/15 episodes (seeds {42, 4, 5}) is non-discriminating. It is shown in every curve but flagged, and it is excluded from the primary population of reading rule R-A (§6).
- φ stays off-mount in `~/rogueone-synp-meta/`. It is mounted read-only into **analysis** containers only, never into an agent container (01 §4.4).

**Metrics** (R13 names).
- **`rec_sel`:** an active non-fallback feature in the selected iteration has |Spearman| ≥ 0.95 with φ on training rows (NMI ≥ 0.9 for P3, P5 and P7); raw columns and single-column monotone transforms never count. The unit is the fold-episode. Per (fold, iteration) the detector writes `rec_iter` ∈ {0, 1}; `rec_sel@k(rule)` = `rec_iter` of the row the rule selects in the prefix.
- **`rec_any@k`** = max of `rec_iter` over the prefix (rule-independent); **`iter_first_rec`** (1–10, censored at 11); **`tok_to_first_rec`** = the fold's cumulative `tok_compl_total` through the first recovering iteration (censored at the fold's total).
- **`heldout_score`:** balanced accuracy (C; the trial's `accuracy` under balancing); `n_RMSE` on Core-10 R; **`RMSE` on SYN-P R**, because E8a standardises SYN-P regression targets (01 §3.3 E8a), so their mean is ≈ 0 and `n_RMSE` is ill-conditioned (`tst/agent.py:361`, `:1788`; A22 makes the same call). A21-T2 confirms the standardisation on the G0a smoke (OQ2).
- **`anytime_auc`:** iteration axis = the mean over k = 1..10 of the `inner_cv`-selected `heldout_score@k`. Token axis = the area of the step function c ↦ `heldout_score@k(c)` over [0, T_d] divided by T_d, where k(c) = the largest k with C(k) ≤ c, the score is the dataset's `base_no_fe` (A05-T6 artifact `base-no-fe-ref`) for c < C(1), the curve is held flat after C(10), and **T_d** = the median run `tok_compl_total` of the A05 runs on dataset d (for within-rung A17 curves: the median of that rung's runs on d). T_d is not the G2 quantity `B_d` (median per iteration).
- **Cost:** C(k) = the sum over the 5 folds of `tok_compl_total` of all (fold, iteration) with `iteration_in_fold < k`, `iteration_skipped` iterations included. Primary: `tok_compl_total` (completion tokens including reasoning and re-rolls, search charged to the caller). Also `llm_calls_total`, `wall_s`, and `flop_proxy` as defined by A23: Σ_roles 2 · N_active(model of the role) · (`tok_prompt_<role>` + `tok_compl_<role>`), W at its caller's model; `flop_proxy_compl` (completion only) as sensitivity. N_active from R9 (O120 5.1B, O20 3.6B, Q122 10B, dense Qwen = size).
- **`success`**, **`cap_hit`**.

## 3. Hypotheses and interpretation (all hypotheses)

| Outcome | Pattern | Meaning and reporting |
|---|---|---|
| **Hoped (H1)** | Most recoveries and most of the score gain arrive by k ≈ 3–5, then the curve plateaus (R-A reads "early"; R-B gives k̂ ≤ 5) | *"X% of recovered features are found by iteration 5, at Y% of the tokens."* X = S5 = Σ`rec_any@5` / Σ`rec_any@10`; Y = the median over A05 runs of C(5)/C(10) |
| **Hoped (H2)** | At LLM-FE's or CAAFE's token spend, truncated Full recovers at least as much (R-D: Δ_iso ≤ 0 at k\* and k\*+1) | *"At matched completion tokens, Full recovered a% vs b% for LLM-FE."* This is a co-reading of C1b and never a Holm verdict (R-E) |
| **Planning prior** | t17 seed 42 (replace mode; `paper/selection_policy_audit.csv`, columns `iter0` vs `inner_cv`; master plan §M2 "REUSE") had iteration 0 far below the selected iteration: heart 0.732 vs 0.866, junglechess 0.742 vs 0.971, vehicle 0.715 vs 0.773 | Under augment, iteration 0 starts near raw XGBoost, so the curve is expected to be flatter. Used for planning only; these are v1 results, not A21 data |
| **Null** | Flat after 1–2 iterations: S2 = Σ`rec_any@2` / Σ`rec_any@10` ≥ 0.8 (SYN-P) and/or k̂ ≤ 2 (Core-10, R-B); each panel is read separately | *"The anytime curve of the 10-iteration schedule saturates after k̂ iterations."* It hints that the loop is mostly best-of-few plus selection; cross-check with A10 (restarts) |
| **Opposite (1)** | Held-out score falls or selection gets worse as k grows (R-C flag) | The winner's curse of inner-CV selection over more candidates. Flagged here, analysed and cited from the short paper |
| **Opposite (2)** | Truncated Full loses to a comparator at equal tokens (R-D: Δ_iso > 0 at k\* and k\*+1) | This weakens C1 on cost. Report it, and the C1 sentence moves to the fallback narrative (bounded context, per-role allocation) |

## 4. Design

**Control and arms.** There are no new runs. The **control rule** is `inner_cv` (the reported protocol). **Comparison rules**, all computed per fold on the prefix `iteration_in_fold < k`, using `completed` rows only (an `iteration_skipped` row is never selected):

| Rule | Definition |
|---|---|
| `inner_cv` | exactly as `main.py:1114-1154` applied to the prefix: argmax `inner_cv_accuracy` (C) / argmin `inner_cv_RMSE` (R); first row on ties; the last completed row if the fold's inner values are all NaN; if no completed row of the run's prefix has a finite inner value, select on the held-out metric and flag the run `inner_cv_unavailable` (expected never under Full-v2). The legacy `policy_scores` (`paper/scripts/diag_selection_policy.py:82-87`) drops such folds instead, so the anytime mode must not reuse it |
| `iter0` | the completed row with the lowest `iteration_in_fold` (iteration 0 in every finished run) |
| `last` | the completed row with the highest `iteration_in_fold` < k |
| `mean_all` | held-out: the mean over the prefix's completed rows; recovery: the mean of their `rec_iter`. The expected value of a uniform random pick (the random-pick reference, as in `diag_selection_policy.py:16-18`) |
| `oracle` | the completed row with the best held-out score (first on ties); recovery = that row's `rec_iter`. It peeks at the test fold and is a **bound only** (`diag_selection_policy.py:19-22`) |
| `patience_2`, `patience_3` | Per fold, walk iterations 0, 1, … in order and stop after p consecutive iterations with no strict inner-CV gain (C: `inner_cv_accuracy` > best so far; R: `inner_cv_RMSE` < best so far; skipped or NaN iterations count as no gain), or at k − 1. Select the `inner_cv` pick among the iterations walked. The run's cost is the sum over folds of each fold's cumulative tokens through its stopping iteration |
| `one_se` | Per fold, the earliest completed iteration whose inner mean is within 1 SE of the prefix best (C: ≥ best − SE; R: ≤ best + SE), with SE = SD(the best row's `inner_cv_fold_scores`, ddof = 1)/√n, n = number of those scores (≥ 2). Computed only if `inner_cv_fold_scores` exists. "Simplest = earliest" is the proposed reading (OQ3, closed in X10) |

`heldout_score@k(rule)` = the mean over the 5 folds of the selected rows' held-out score; `rec_sel@k(rule)` is per fold-episode.

**Fixed settings of the data** (`scripts/agent2027/v2_defaults.env`; 01 §4.3):
- `ROGUE_ONE_NUM_ITERATIONS=10` (code default 3, `config.py:305`), `ROGUE_ONE_K_FOLDS=5` (`:309`);
- `ROGUE_ONE_SELECTION_SIGNAL=inner_cv` (`:317`), `ROGUE_ONE_ENABLE_ROLLBACK=False` (`:313`);
- `ROGUE_ONE_DO_BALANCING=True` (`:312`);
- `ROGUE_ONE_STRATIFIED_FOLDS` per row (`:323`; `True` on SYN-P classification rows if D4);
- `ROGUE_ONE_EXTRACTOR_ROUNDS` and `ROGUE_ONE_FEATURE_POOL_MODE` as frozen at G1;
- `ROGUE_ONE_LLM_TRACE=1`.

**Seeds:** fold seeds {42, 4}; `llm_seed` = `fold_seed` for r0 and `fold_seed + 1000` for the A/A (r1) (R11).

**Iso-cost rule** (01 §4.8 rule 2, made exact here; frozen in X10):
- An **instrument** i is a (`dataset_id`, `fold_seed`) pair on which both the arm and its reference have runs.
- Let C_F,i(k) = the reference's run cost C(k) on instrument i, averaged over all reference replicates present (for A05 `full`: f42 r0 and r1; f4 r0).
- Let C_A,i = the arm's run total on instrument i, averaged over its replicates.
- Then **k\*_i = max{k ∈ 1..10 : C_F,i(k) ≤ C_A,i}**. Whole iterations only, rounded down, so Full never gets more than the arm. k\*_i + 1 is reported as the bracket.
- If C_A,i < C_F,i(1), the instrument has **"no iso-cost point"** (this includes the 0-token CPU arms).
- If C_A,i ≥ C_F,i(10), then k\*_i = 10 and the point is flagged "arm ≥ Full's whole schedule"; it has no k\*+1 bracket.
- **Δ_iso** = the mean over the arm's instruments with a point of (arm `rec_sel` − reference `rec_sel@k*_i` under `inner_cv`) on SYN-P, or of Δ_std = (arm `heldout_score` − reference `heldout_score@k*_i`)/σ_d, sign-adjusted so that positive = arm better, on Core-10. Positive Δ_iso = the arm beats truncated Full.
- Its 95% CI comes from a task-stratified bootstrap (2,000 reps, RNG seed 20261016) over fold-episodes (SYN-P) or runs (Core-10). Descriptive only.

**Cost axis by arm class** (also the per-arm reference table the `--mode isocost` code applies):
- same backbone as Full (O120): `tok_compl_total`, reference A05 `full`;
- within one non-O120 rung (A20 vs A17 `f2_O20`; A17 F2 curves): that rung's `tok_compl_total`;
- cross-backbone (A17 rungs and A19 mixes vs O120 Full): `flop_proxy` ("iso-FLOP"), with A19's big-model tokens alongside;
- if token logging fails validation (§6, G2), fall back to `llm_calls_total` and `wall_s` (01 §4.8 rule 1).

**Run matrix** (data consumed; W&B entity `PlanBench`, project `RogueOne-AGENT2027` unless marked pre-freeze; run name `<AID>__<arm_id>__<dataset_id>__f<fold_seed>__r<replicate>`). "A05" in the reference column means all A05 `full` replicates present on the instrument (f42 r0 + r1, f4 r0):

| Source (arm_id) | Instruments × seeds × replicates | Runs | Reference / axis | Use |
|---|---|---|---|---|
| A05 `full` | Core-10 and SYN-P 8 × {f42 r0, f4 r0, f42 r1}; `synp_p0` × f42 r0 | 55 | is the reference / tokens | primary anytime curves, all 8 rules |
| A08 `llmfe` | SYN-P 7 × f42 × r0, r1 | 14 | A05 f42 (r0 + r1) / X5 tokens | iso-cost (C1b view; notebook primary, best3 and any-program as sensitivity) |
| A08 `caafe` | SYN-P 7 classification tasks × {42, 4} | 6 (≤ 8 budgeted) | A05 / X5 tokens | iso-cost |
| A08 `base_no_fe`, `openfe`, `autofeat` | CPU cells | (81 cells) | 0 tokens | "no iso-cost point" (A23 only) |
| A16 `set_phased` | SYN-P 7 × {42, 4}; Core-10 × 42 (pilot rows excluded) | 24 | A05 / tokens | iso-cost (C1a) |
| A12 8 FX cells | SYN-P 7 × 42 | 56 | A05 f42 (r0 + r1) / tokens | iso-cost (C2 context) |
| A10 `restarts`, A11 `l0` | SYN-P 7 × {42, 4} | 28 | A05 / tokens | iso-cost and curves (Restarts → L0 → Full) |
| A09 `oracle`, `misdirected` | SYN-P 8 / SYN-P 7 × 42 | 15 | A05 f42 / tokens | reference curves |
| A13 `search_off` | Core-10 × 42 | 10 | A05 f42 (mean of r0 + r1) / tokens | iso-cost on Δ_std |
| A07 `errors_hidden` | HC1-6 × {42, 4} | 12 | A05 / tokens | descriptive iso-cost |
| A14 `kb_*` (4 arms) | `synp_p7` × 3–6 seeds | 18 | A05 P7 at 42/4 only / tokens | cost points only (no Full reference at seeds 5, 1, 2, 3) |
| A17 `f2_<rung>` | SYN-P 7 × 42 (+ seed 4 for Q0.8, Q4, Q122); CAP real-4 × 42 | 87 | within rung (tokens); vs A05 (`flop_proxy`) | F2 curves for A17's F0 overlay; iso-FLOP |
| A19 `mix_esmall`, `mix_1b` | SYN-P 7 × {42, 4} + CAP real-4 × 42 | 36 | A05 / `flop_proxy` | iso-FLOP |
| A20 `fx_pajM_O20` | SYN-P 7 × 42 | 7 | A17 `f2_O20` / O20 tokens | within-O20 iso-cost |
| **Pre-freeze panel:** A02 `R10`/`R3`/`R1`; A03 `replace`/`augment` (+ add-on) | SYN-P 8 × 42; Core-10 × 42 (+ SYN-P 7) | 44 (+7) | within-pilot only (A02 `R10` truncated to `R3`/`R1` spend; A03 arms as anytime curves) | separate panel, project `RogueOne-AGENT2027-prefreeze`; **never** truncates frozen Full, never on the frozen front (01 §4.8 rule 6) |

**Not read:** A01 (v1 code, entity `brainflux`), A04 (smoke rows), A06 sentinels (drift guard; A21 uses only A06's drift verdicts as caveats), A18 (replays, no runs), the RT2 pilot rows (`notes=pilot`), and A15's optional AB2 rows (if run, they are shown on Full's curve as extra `full` points, outside the counts).

**Budget arithmetic.** New runs = 0, so 0u and 0 GPU-hours. Inputs = 55 + (14 + 6–8) + 24 + 56 + 28 + 15 + 10 + 12 + 18 + 87 + 36 + 7 = **368–370** post-freeze runs, plus 24 + 20 (+ 7) = **44–51** pre-freeze runs. All are charged to their own ablations. Reconciliation with R1: the post-freeze GPU rows of A05 and A07–A20 total 372 (with CAAFE at its budgeted 8) plus A06's 9–17; A21 drops the A06 sentinels and the 2 RT2 pilot rows, and counts CAAFE at 6–8, giving 368–370. This matches the breakdown's "M2: 0 runs, 0u (CPU)".

**Pairing.** Arms share Full's outer partition by `fold_seed` (with matching `ROGUE_ONE_STRATIFIED_FOLDS`). Where a comparator's partition differs, the comparison is unpaired at task level: LLM-FE generates on its native `StratifiedKFold(5, shuffle, 42)` (01 §4.2; its regression split is still to be verified in A08), and CAAFE runs `--folds stratified --fold-seed <s>` (A08-T11), which equals Full's partition only if D4 makes Full stratified on SYN-P classification rows.

**Failure handling.** A21 launches nothing, so there are no retries or caps of its own. It applies 01 §4.5 ITT at **every k**:
- a failed, aborted or 6 h-capped run gives `rec_sel@k` = `rec_any@k` = 0 for all 5 episodes;
- its `heldout_score@k` is imputed at `base_no_fe` (augment canonical) or the majority/mean predictor (replace canonical);
- such a run contributes no tokens to C_F,i(k) (it is not averaged in), and its cost is reported separately.

An "as-observed" curve, using the completed prefixes of partial runs where their `run-output` was uploaded, is a sensitivity analysis. Infra re-runs (`notes=infra-rerun`) replace their failed originals.

## 5. Tasks

### 5.1 Prerequisites
- [ ] **A21-T1** [type: decision] (0.1 d, due Thu 8 Oct) — Add the §2 data contract (items 1–4) to E3's and E8a's done-when lists before E3 is built (Thu 8 – Fri 9). *Done when:* the four items appear as tick-box lines in the E3 and E8a "Done when" lists of 01 §3.3.
- [ ] **A21-T2** [type: data] (0.1 d, due Mon 12 Oct, G0a) — On the G0a smoke run (`ROGUE_ONE_SMOKE_FOLDS=1`, project `RogueOne-AGENT2027-prefreeze`), fetch its `run-output` with the minimal `fetch_wandb.py` and run a short pandas check in a `rogue_one:agent2027-pre.N` container (CAIR, `--cpus 4`, 01 §5.2 CPU rules):
  - (a) completion tokens of `llm_traces.jsonl.gz`, summed per (`test_fold_id`, `iteration_in_fold`) and then in total, equal the W&B summary `tok_compl_total` within 1%;
  - (b) each of S, E and T has > 0 completion tokens, with search charged to the caller;
  - (c) every `iteration_skipped` row of `df_test_pool_final.csv` has ≥ 1 trace record with its (fold, iteration);
  - (d) every `completed` row has `inner_cv_fold_scores` with 2–4 finite values whose mean equals its `inner_cv_accuracy`/`inner_cv_RMSE` (|Δ| < 1e-9);
  - (e) on one SYN-P regression task folder (`<ROGUEONE_CAIR_PATH>/tasks/regression/synp_p1/`), the target in the file named by `config.yml` `label_file_path` (E8a writes `labels.csv`) has mean ≈ 0 and SD ≈ 1 (OQ2).

  *Done when:* the five results are pasted into the G0a notes, and every miss has an E3/E8a fix item due before Fri 16 Oct.
- [ ] **A21-T3** [type: data] (0.05 d, Fri 16 Oct afternoon, inside A05-T3) — Repeat T2 (a)–(d) on the `agent2027-v2.0` freeze smoke. *Done when:* it is ticked in the session plan, or the §6 G1 fallback is recorded there.
- [ ] **A21-T4** [type: writing] (0.2 d, due Fri 16 Oct 12:00, inside X10) — Write the "A21/M2" block of `docs/agent2027/preregistration.md`: the k grid (all of 1..10 computed), the 8 rules with the §4 definitions, ITT at every k, T_d, the cost axes by arm class and the `flop_proxy` definition, the iso-cost rule, reading rules R-A to R-E (§6), the wording constraint, OQ1–OQ3 as closed, and "descriptive, outside Holm". *Done when:* it is inside tag `agent2027-prereg`.
- [ ] **A21-T5** [type: data] (0.05 d, due Mon 19 Oct 08:00) — Gate check before the preliminary pass. *Done when:* (a) A05's 55 runs are `finished` or ITT with `run-output` (A05-T12/T13); (b) the G2 memo holds σ_d (A05-T14) and the A08 gate labels (M5S); (c) A21-T6, -T7 and -T9 are merged and pushed to `agent2027/v2`. If (c) fails, T11 runs only the T2 checks on A05 (§7).

### 5.2 Implementation (analysis code only; Full-v2 untouched)
- [ ] **A21-T6** [type: code] (0.2 d, due Sun 18 Oct; weekend work per D1) — Add `--mode anytime` to `paper/scripts/diag_selection_policy.py`. Keep the current path (`main()`, `:100-176`) as the default `--mode legacy`; its `--runs`/`--baselines` (`:102-104`) become required in legacy mode only. The new mode:
  - reads `<artifacts>/<run_name>/output/` and takes `task_type` from the W&B config, not from the hard-coded lists (`:40-48`);
  - for each k in `--k` applies `inner_cv` (re-implemented from `main.py:1114-1154`, §4), `iter0`, `last`, `mean_all` and `oracle` per fold on the prefix;
  - joins held-out scores, `recovery_by_iteration.csv` and per-iteration tokens from the tagged traces;
  - writes `anytime_long.csv` (run × fold × k × rule), `anytime_summary.csv`, and `validity.csv` (per run: the T2 (a)–(d) results and whether the k = 10 `inner_cv` pick reproduces `fold_aggregation_best_iterations.csv`).

  *Done when:* A21-T7 passes.
- [ ] **A21-T7** [type: code] (0.1 d, due Sun 18 Oct) — Write `tests/unit/test_diag_selection_policy.py` (load the script with `importlib`, as `tests/unit/utils/test_retrieval_audit.py:20-21` does) with three tests:
  1. a toy pool with ties, an all-NaN inner fold, a run with no finite inner value, and `iteration_skipped` rows;
  2. at k = 10 the `inner_cv` pick equals `fold_aggregation_best_iterations.csv`, `heldout_score@10` equals `fold_aggregation_summary.csv`, and `rec_sel@10` equals the detector's run-level `rec_sel`, on every A02 `R10` run (8 runs, available from Thu 15). The test reads the folder in env `A21_ARTIFACTS_DIR` and skips if it is unset;
  3. legacy-mode output is unchanged on a toy t17-style folder, and reproduces `paper/selection_policy_audit.csv` from `~/rogueone-runs-t17s42` (`paper/scripts/t17_rerun_launch.sh:46`, SEED=42) with `--baselines paper/results_matched_harness_baselines.csv` if that folder still exists (env `A21_T17_RUNS`; skip if unset).

  Run it on CAIR after fetching A02 `R10` (project `RogueOne-AGENT2027-prefreeze`) into `~/ro27_a21/artifacts_a02` and running the per-iteration detector on it (T10 pattern). First `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "ls -d ~/rogueone-runs-t17s42"`; drop the `/t17` mount and `A21_T17_RUNS` if it is absent (otherwise Docker creates an empty root-owned folder). Then:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker run --rm --name ro27-analysis-a21-test --cpus 4 -e PYTHONPATH=/repo -e PYTHONDONTWRITEBYTECODE=1 -e A21_ARTIFACTS_DIR=/work/artifacts_a02 -e A21_T17_RUNS=/t17 -v <ROGUEONE_CAIR_PATH>:/repo:ro -v ~/ro27_a21:/work -v ~/rogueone-runs-t17s42:/t17:ro --entrypoint bash rogue_one:agent2027-pre.N -c 'python3 -m pip install -q pytest==8.3.4 && cd /repo && python3 -m pytest -p no:cacheprovider -q tests/unit/test_diag_selection_policy.py'"
  ```
  (pytest is pinned in `requirements.txt:30` but not in `setup.py` `install_requires` (`:10-38`), and the image installs only `-e .` (`Dockerfile:19-20`), so it is installed inside the throwaway container.) *Done when:* all three tests pass with none skipped (test 3 may skip its t17 half only if the folder is gone; record that).
- [ ] **A21-T8** [type: code] (0.25–0.4 d, due Tue 27 Oct, inside E11b) — Add:
  - `patience_2`, `patience_3` and `one_se`;
  - `--mode isocost` with the §4 rule and the §4 per-arm reference/axis table; `--reference-group`/`--reference-arm` override it (pre-freeze panel only);
  - `--cost-axis tok_compl_total|llm_calls_total|wall_s|flop_proxy`;
  - `anytime_auc` on both axes, `iter_first_rec` and `tok_to_first_rec`;
  - `isocost_points.csv`;
  - figures `m2_anytime_iter.pdf` and `m2_anytime_tok.pdf`;
  - `--wandb-log` (with optional `--wandb-tag`), which writes run `ANALYSIS__diag_selection_policy__<YYYYMMDD>` (group `ANALYSIS`, project `RogueOne-AGENT2027`) with tables `m2_anytime` and `m2_isocost`, and uploads the `--out` folder (CSVs and PDFs) as artifact `a21-m2-outputs`.

  *Done when:* toy tests pass for each rule and for both edge cases ("no iso-cost point"; k\* = 10 flag), and a dry run on A05 + A10 writes every output.
- [ ] **A21-T9** [type: code] (0.05 d, due Sun 18 Oct) — Make `scripts/agent2027/analysis/recovery_detector.py` emit, under `--per-iteration`, `recovery_by_iteration.csv` (run, `test_fold_id`, `iteration_in_fold`, `rec_iter`, best |ρ| or NMI, feature). *Done when:* its value at the `inner_cv`-selected trial equals the run-level `rec_sel` on every A02 `R10` run (Sun 18) and on every A05 SYN-P run (Mon 19). If it is late, the Mon 19 preliminary covers held-out scores only.

### 5.3 Execution on CAIR (CPU only)
Standing rules (01 §5.1):
- **Containers only.** Every step runs in a container; the host issues only `docker`, plus `nvidia-smi`, `docker ps`, `tail`, `df`, `uptime` for checks and the git sync of 01 §5.2 step 0. Never `python …` on the host. Any `docker compose run` over SSH needs `-T`.
- **No GPU.** A21 never passes `--gpus`, so no GPU pre-flight is needed. Anything that ever needs a GPU first runs the full 01 §5.2 step-1 pre-flight (stop and report if the GPUs are not free) and uses ≤ 4 of the 8 H100s without Henrik's explicit approval in chat. `ro27-analysis-a21` matches the `ro27-*` pattern of other sessions' pre-flights but holds no GPU; those pre-flights may ignore it.
- **CPU cap** `--cpus 8` (≤ 16 per 01 §5.2). One A21 pass at a time (fixed container name).
- **W&B is the system of record.** Results live in the `ANALYSIS__…` run and its artifact; `~/ro27_a21/` is scratch. The W&B key comes from `<CAIR_HOME>/code/PlanBench/.env`.
- **Detached long jobs:** `nohup`, log in `~`, never in the root-owned run folders.
- `<ROGUEONE_CAIR_PATH>` = `<CAIR_HOME>/code/RougeOne` is **inferred; verify it** with `ssh CAIR-gpu14 "ls -d ~/code/RougeOne"` (from `paper/scripts/clients_t16_rerun.sh:59`, `:135`; OQ4).

- [ ] **A21-T10** [type: run] (≈ 0.1 d per pass: Mon 19 Oct 08:00, Tue 27 Oct, Sun 1 Nov; the pre-freeze pass runs right after the Sun 1 main pass) — Commit and push `agent2027/v2` locally, then:
  1. **Sync** (01 §5.2 step 0): `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15' git fetch git@github.com:HenrikNorkart/RougeOne.git agent2027/v2 && git checkout -B agent2027/v2 FETCH_HEAD && git log -1 --format='HEAD: %h %s'"`. Record the SHA in the pass notes (the analysis code runs from this checkout, mounted read-only).
  2. **CPU/disk check:** `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "nproc; uptime; df -h ~ | tail -1; docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'; docker image ls rogue_one --format '{{.Repository}}:{{.Tag}} {{.ID}}'"`. Proceed only if no `ro27-analysis-a21` container runs, the 1-min load average is below `nproc` − 8, and `~` has ≥ 50 GB free (placeholder; replace it with 1.5 × the measured A05 download size after Mon 19). Use the newest `rogue_one:agent2027-v2.N` listed (it only supplies the Python environment) and record its ID.
  3. **Launch detached** (Mon 19 form):
     ```
     ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "mkdir -p ~/ro27_a21 && nohup docker run --rm --name ro27-analysis-a21 --cpus 8 --env-file <CAIR_HOME>/code/PlanBench/.env -e WANDB_ENTITY=PlanBench -e WANDB_PROJECT_NAME=RogueOne-AGENT2027 -e PYTHONPATH=/repo -e PYTHONDONTWRITEBYTECODE=1 -v <ROGUEONE_CAIR_PATH>:/repo:ro -v ~/ro27_a21:/work -v ~/rogueone-synp-meta:/synp_meta:ro --entrypoint bash rogue_one:agent2027-v2.N -c 'cd /repo && python3 scripts/agent2027/analysis/fetch_wandb.py --project RogueOne-AGENT2027 --group A05 --out /work/artifacts && python3 scripts/agent2027/analysis/recovery_detector.py --per-iteration --meta /synp_meta --artifacts /work/artifacts --out /work/recovery_by_iteration.csv && python3 paper/scripts/diag_selection_policy.py --mode anytime --artifacts /work/artifacts --recovery /work/recovery_by_iteration.csv --groups A05 --k 1 2 3 4 5 6 7 8 9 10 --cost-axis tok_compl_total --out /work/out_<YYYYMMDD> --wandb-log' < /dev/null > ~/ro27_a21_<YYYYMMDD>.job.log 2>&1 & echo launched"
     ```
     The options of `fetch_wandb.py` and of the detector other than `--per-iteration` are whatever E3/E8a/E11b implement (check `--help` first; OQ5).
     - **Tue 27 and Sun 1:** fetch and detect `--group A05 A07 A08 A09 A10 A11 A12 A13 A14 A16 A17 A19 A20`, run `--mode anytime` on all of them, then a second `diag_selection_policy.py` call with `--mode isocost --groups A07 A08 A09 A10 A11 A12 A13 A14 A16 A17 A19 A20` (references and axes from the §4 table), both with `--wandb-log`.
     - **Pre-freeze pass (Sun 1, after the main pass ends):** `--project RogueOne-AGENT2027-prefreeze --group A02 A03`, `--mode anytime --groups A02 A03`, then `--mode isocost --groups A02 --reference-group A02 --reference-arm R10`, with `--wandb-tag prefreeze`.
  4. **Monitor:** `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "tail -20 ~/ro27_a21_<YYYYMMDD>.job.log; docker ps --format '{{.Names}}' | grep '^ro27-analysis-a21' || echo done"`. Exit code 255 means the network dropped, not the job: reconnect.

  *Done when:* for each pass, the W&B run `ANALYSIS__diag_selection_policy__<YYYYMMDD>` exists with its tables (and, from Tue 27, artifact `a21-m2-outputs`), the job log ends without a Python traceback, and no `ro27-analysis-a21` container is left.

### 5.4 Analysis
- [ ] **A21-T11** [type: analysis] (0.15 d, Mon 19 Oct; validity by 10:00 for the G2 memo, curves by end of day) — Read `validity.csv` from the Mon 19 pass against the §6 G2 rule, then produce the preliminary `rec_sel@k` and `heldout_score@k` for `inner_cv`, `last` and `oracle` on A05. *Done when:* the validity result is in the G2 memo and the curves are logged with tag "preliminary" in W&B. They are used for no decision.
- [ ] **A21-T12** [type: analysis] (0.15 d, due Tue 27 Oct) — Draft iso-cost points for every arm whose W&B runs are all `finished` or ITT by Tue 27 08:00 (expected: A07, A09, A10, A11, A13; A08, A12, A16 SYN-P, A17 small rungs, A19, A20 as available). *Done when:* `isocost_points.csv` (draft) is attached to the Tue 27 analysis run and its link is given to A23's draft and to A10, A13 and A16.
- [ ] **A21-T13** [type: analysis] (0.25 d, Sun 1 – Tue 3 Nov, after DF Sat 31 Oct 24:00) — Final pass on every group, with reading rules R-A to R-E, the as-observed sensitivity, `flop_proxy_compl`, and LLM-FE best3/any-program. *Done when:* the final `ANALYSIS__diag_selection_policy__2026110x` run holds `m2_anytime`, `m2_isocost` and artifact `a21-m2-outputs` with the figures, and the R-A to R-E readings are written in the analysis notes.

### 5.5 Paper outputs
- [ ] **A21-T14** [type: writing] (0.3 d, Wed 4 – Fri 6 Nov) — Produce four items:
  - the appendix figure "M2 curves": `rec_sel@k` against iterations and against cumulative tokens, with the rules overlaid; Core-10 Δ_std(k); the pre-freeze panel separate;
  - Full's anytime line and the iso-cost points for Fig. 2 (right), together with A23;
  - the iso-cost note in Table 2's cost column;
  - one Results sentence worded as in §3 and R-A/R-D, using "anytime curve of a 10-iteration schedule".

  *Done when:* all four are in the AGENT draft by Fri 6 Nov.
- [ ] **A21-T15** [type: writing] (0.1 d, Thu 12 Nov) — Disclose:
  - this is not an iteration budget (Round i/N, `sci/agent.py:440`, `:472-475`);
  - inner-CV accuracy is unweighted while the reported accuracy is balance-weighted;
  - SYN-P regression uses `RMSE`, not `n_RMSE`;
  - comparators ran at native budgets;
  - the winner's curse is cited to the short paper;
  - the pre-freeze panel is kept separate.

  Add the script, the tests and `isocost_points.csv` to the replication package. *Done when:* the six disclosures are in the AGENT draft and the three files are in the anonymised replication package.

## 6. Decision rules and gates (pre-registered in X10; descriptive)

| When | Rule |
|---|---|
| G0a, Mon 12 Oct | If any data-contract item (T2) is missing from the smoke, fix it in E3/E8a before the freeze |
| G1, Fri 16 Oct | If trace iteration tags are missing at the freeze smoke and the fix would delay the 18:00 A05 launch by > 2 h, launch anyway. A21 then uses the iteration axis for curves and reconstructs per-iteration tokens from W&B history order (`LLM_tokens/*` rows between the `extractor_iteration/*` and `trial/*` markers, read with `run.scan_history()`, not the sampled `run.history()`), labelled approximate. Never rerun A05 for A21 alone |
| G2, Mon 19 Oct | **Validity:** per-iteration tokens sum to `tok_compl_total` within 1% on ≥ 95% of runs; the k = 10 pick reproduces `fold_aggregation_best_iterations.csv` on all non-ITT runs. If tokens fail, the cost axis becomes `llm_calls_total` and `wall_s`. If the k = 10 pick fails on any run, it is a script bug: no A21 curve is published until it is fixed. If inner-fold scores are missing, that is a G2 defect for A05 first (01 §6.1); if A05 is not rerun, drop `one_se` |
| R-A (SYN-P) | Population: A05 SYN-P 8 fold-episodes (3 runs per task) minus brute-force-flagged tasks; all tasks as a sensitivity. S_k = Σ`rec_any@k` / Σ`rec_any@10`. **"Early"** if S5 ≥ 0.8 **and** `rec_sel@5` ≥ `rec_sel@10` − 0.05 (proportions of fold-episodes, `inner_cv`); **"late"** if S5 < 0.5; otherwise **"gradual"**. If Σ`rec_any@10` = 0, R-A reads "not applicable (no recoveries)". Report Wilson CIs, marked as not adjusted for clustering |
| R-B (Core-10) | Δ_std(k) = the mean over the 10 datasets of (`heldout_score@k` − `heldout_score@10`)/σ_d under `inner_cv`, each dataset averaged over its 3 A05 runs, sign-adjusted (positive = better). k̂ = the smallest k with Δ_std(k) ≥ −0.5 (k̂ ≤ 10 always). Wording: "the anytime curve reaches within 0.5 σ_d of its final value by k̂" — never "N = k̂ suffices" |
| R-C | Winner's-curse flag if (a) the `inner_cv` `heldout_score` at k = 10 is below k = 5 (sign-adjusted, averaged over the dataset's A05 runs) on ≥ 6/10 Core-10 datasets, **or** (b) the uncaptured headroom share 1 − (`inner_cv@k` − `mean_all@k`)/(`oracle@k` − `mean_all@k`), averaged over Core-10 datasets with positive headroom at both k, rises by ≥ 0.10 from k = 3 to k = 10. (The raw `oracle − inner_cv` gap is not used: it widens mechanically with more candidates.) Cite the short paper |
| R-D (iso-cost) | Classify Δ_iso at k\* and at k\*+1 as +, 0 or −. Write "at matched completion tokens …" only if both classes agree; otherwise write "budget-sensitive". A point with the k\* = 10 flag is read at k\* alone and labelled "arm ≥ Full's whole schedule". Report "no iso-cost point" as defined in §4 |
| R-E (C1b) | The iso-cost reading never creates or changes a Holm verdict. If its sign disagrees with A08's native-budget C1b, the C1 sentence says "budget-dependent" |
| G4, Tue 27 Oct | E11b is done, so A21-T8 must be merged; draft points go to A23 |
| DF, Sat 31 Oct 24:00 | The final pass starts after it. There is no kill date (0 GPU) |

## 7. Risks and cut levers
- **Per-iteration token attribution is missing in M0** (highest risk). Covered by A21-T1–T3 and the G1 fallback.
- **Developer time.** The 0.35 d of code before G2 (T6, T7, T9) lands on Sat 17 – Sun 18, which 01 §6.2 does not list (it holds E11a, E9c tooling, D5). It needs D1 = weekends. If D1 says no, or it slips, the Mon 19 preliminary shrinks to the T2 checks on A05 (validity only), and T6, T7 and T9 move into E11b (due Tue 27). **Internal cut order:**
  1. cross-backbone iso-FLOP points (left to A23);
  2. `patience_*`/`one_se` beyond A05;
  3. the pre-freeze panel (A02/A03 keep their own tables).

  **Never cut:** A05's `inner_cv` anytime curve and the iso-cost points for A08 `llmfe`, A16 (if SET survives K-SET), A10 and A12.
- **SYN-P regression metric.** E8a standardises the targets, so `n_RMSE` divides by ≈ 0 (`tst/agent.py:361`, `:1788`). A21 uses `RMSE` there; T2 (e) confirms the standardisation.
- **Disk.** ≈ 370 `run-output` artifacts with gzipped traces; the T10 step-2 check guards it. If space is short, fetch only `df_test_pool_final.csv`, `fold_aggregation_*.csv`, `llm_traces.jsonl.gz`, the feature log and `intermediate_steps/`.
- **Mixed code versions.** Pre-freeze points are kept in their own panel (01 §4.8 rule 6).
- **Misreading as a budget claim.** The wording rules R-B and A21-T15 guard against this.
- **Cut levers elsewhere** shrink A21's inputs only: CL1 removes A19's real rows, CL2 10 A17 small-rung real rows, CL3 4 A07 rows, CL4 the CAAFE points, CL6 some F0 overlays, CL7 the L0 point. ED3 (F0 dropped) removes the need for the A17 F2 overlay; ED4 (SET dropped at K-SET) removes the A16 point; ED6 removes the L0 point.

## 8. Definition of done
- [ ] The data contract is verified on the G0a and freeze smokes (or the G1 fallback is recorded); the A21 block is in `agent2027-prereg`.
- [ ] `diag_selection_policy.py` has the anytime and isocost modes, with green tests (k = 10 reproduces the protocol; the legacy mode is unchanged).
- [ ] Preliminary (Mon 19), draft (Tue 27) and final (Nov 1–3) `ANALYSIS__diag_selection_policy__*` runs exist with `m2_anytime`, `m2_isocost` and, from Tue 27, artifact `a21-m2-outputs`; the pre-freeze run carries tag `prefreeze`.
- [ ] Every post-freeze arm has an iso-cost point or "no iso-cost point"; the pre-freeze panel is separate.
- [ ] The appendix figure, the Fig. 2 inputs, the Table 2 note, the Results sentence, the disclosures and the replication-package files are in the drafts.

**Open questions.**
- OQ1 (resolved by A23 §2, to be frozen in X10): `flop_proxy` uses prompt + completion tokens; `flop_proxy_compl` (completion only) is the sensitivity.
- OQ2 (resolved by E8a's spec, check in T2 (e)): SYN-P regression targets are standardised, so `heldout_score` on SYN-P R is `RMSE`.
- OQ3: should `one_se`'s "simplest" mean the earliest iteration (proposed) or the fewest active features? Closed in X10 (T4); the default is "earliest".
- OQ4: `<ROGUEONE_CAIR_PATH>` is unverified (A05-T4 runs `ls -d ~/code/RougeOne`).
- OQ5: the CLI of `fetch_wandb.py` (`--project`, `--group`, `--out`) and of `recovery_detector.py` (`--meta`, `--artifacts`, `--out`) is assumed here; align the T10 commands with what E3/E8a/E11b implement.

---

**New names introduced** (not in the canonical registry):
- **Script options** for `diag_selection_policy.py`: `--mode legacy|anytime|isocost`, `--artifacts`, `--recovery`, `--groups`, `--reference-group`, `--reference-arm`, `--k`, `--cost-axis`, `--out`, `--wandb-log`, `--wandb-tag`.
- **Rule IDs:** `patience_2`, `patience_3`, `one_se`; run flag `inner_cv_unavailable`.
- **Output files, tables and artifacts:** `anytime_long.csv`, `anytime_summary.csv`, `validity.csv`, `isocost_points.csv`, `m2_anytime`, `m2_isocost`, `m2_anytime_iter.pdf`, `m2_anytime_tok.pdf`, artifact `a21-m2-outputs`.
- **Data-contract names:** the proposed trial column `inner_cv_fold_scores`; trace tags `test_fold_id` and `iteration_in_fold`; the detector output `recovery_by_iteration.csv` (column `rec_iter`) and its option `--per-iteration`; the test file `tests/unit/test_diag_selection_policy.py` and its env vars `A21_ARTIFACTS_DIR`, `A21_T17_RUNS`.
- **Notation:** "iso-FLOP"; instrument i, k\*_i, C(k), C_F,i(k), C_A,i, Δ_iso; T_d; S_k (S2, S5); reading rules R-A to R-E. `flop_proxy_compl` is A23's name.
- **CAIR names:** containers `ro27-analysis-a21` and `ro27-analysis-a21-test`, work dir `~/ro27_a21/`, log `~/ro27_a21_<YYYYMMDD>.job.log`.
