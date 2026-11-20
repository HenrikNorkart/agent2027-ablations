# A03 — M1 · Augment vs replace gate (+ optional SYN-P add-on)

*Rogue One v2 design study, ICSE AGENT 2027, T1-minimum. Written Tue 6 Oct 2026; code and data re-checked Wed 7 Oct. Every "hoped" result below is a hypothesis, not a finding.*

*Code refs are to RougeOne `nested-cv-rebuttal` @ `07cbf93` (`C:/code/PHD/RougeOne`), read line by line. The branch tip `1bdaf42` changes only `paper/`: in `paper/scripts/classicfe_arm.py` every line after 85 moves by +8; `paper/results_seed_sweep_t17.csv` is **new at the tip** (it does not exist at `07cbf93`, so read it at `nested-cv-rebuttal`); `paper/significance_analysis_v2.md` is regenerated (its −0.0385 becomes −0.0387). No other reference in this document moves.*

*Short paths: `main.py` = `rogue_one_main.py`; `sci/agent.py`, `ext/agent.py`, `tst/agent.py` = `rogueone/llm/agents/core/{scientist,extractor,tester}/agent.py`; `config.py` = `rogueone/utils/config.py`. There is no `cfg.py`. Shared doc = `docs/ablations/01_shared_infrastructure.md`, cited as "01 §n".*

| Field | Value |
|---|---|
| **ID** | A03 · M1 |
| **Claim / Holm contrast** | None. M1 is a **gate**: it sets freeze item 5 (`ROGUE_ONE_FEATURE_POOL_MODE`) and so defines Full-v2. It is outside Holm, is not an FX factor, and is reported as the first design result. |
| **Tier** | T1-minimum. The Core-10 part is on the never-cut list. The SYN-P add-on is add-back AB1, chosen by D6. |
| **Owner** | Henrik |
| **Start / finish** | Rule committed Tue 13 Oct → build + E6 smoke session `S1014e` **Wed 14 Oct from 08:00** (after A02's `S1013a` ends) → Core-10 session `S1014b` ≈ **10:30** → read-out **Thu 15 Oct, 12:00** → frozen at **G1, Fri 16 Oct** → appendix Nov 1–6 |
| **Budget** | Core-10: 20 runs = **20u**. Add-on: 7 runs = +4.9–7u. E6 smoke: 5 one-fold rows ≈ 1.0u (not counted in the registry, like A02's smoke rows). The main session is 10–19.3 h on one TP4 server (27u at 1.4 u/h; 20u at 2 u/h), about **40–77 H100-hours**, plus ≈ 1–1.5 h (4–6 H100-h) for the smoke session. The fallback path replaces all of this with 10 runs = 10u inside A05's sessions. |
| **Models** | O120 (`openai/gpt-oss-120b`, MXFP4, TP4, port 8010, 6 clients) for S, E, T and W |
| **Status** | Not started |
| **Depends on** | E1, E2 (R knob), E3, E5, E13 (or ED1); **E6 (gate G-E6, Tue 13 Oct, end of day)**; A01 (`sigma_samp`, hand-off Mon 12 Oct 18:00); D6; the `classicfe:4` image on CAIR (the default of `paper/scripts/classicfe_launch.sh:21`; reference for A03-T6a, existence checked there). The add-on also needs E8a with the recovery detector, the M5S read, and A02 `R10` rows A02-001…008. |
| **Feeds into** | G1 `v2_defaults.env`, and so A05 and every v2 arm; the ITT imputation rule; A02 (`W_ref` = median `wall_s` of A03 `replace`); A10 (re-seeding on reset); A21/A23 (pre-freeze panel); X10 pre-registration |

## 1. What we test

Rogue One today rebuilds its feature pool from nothing in every fold ("replace"). The final XGBoost sees a raw column only if an agent re-creates it.

"Augment" seeds every raw column, ordinal-encoded where it is categorical, as a protected pool row. We test whether augment removes replace's deficit to plain XGBoost without lowering planted-feature recovery.

On the nested-CV panel replace trails raw XGBoost (`base_no_fe`): the 3-seed mean oriented Δ is −0.0385 in classification (n = 18, Holm p = 0.0156) and −0.060 nRMSE in regression (n = 8, Holm p = 0.031) (`paper/significance_analysis_v2.md` @ `07cbf93`; 26 datasets, because `cmc` is dropped — the breakdown's "27-dataset panel" counts it). On Core-10 itself the gap is mixed (table in §2): classification mean +0.010, carried by `balance-scale`; regression mean −0.018. Whichever mode passes this gate becomes Full-v2, so every later number depends on it.

## 2. Background you need

**Architecture touched:**

- **Fold reset.** `forward_pass` (`main.py:536`) calls `_reset_fold_state` (`main.py:417-457`) at `iteration_in_fold == 0` (`main.py:543-544`).
  - The reset rebuilds the entity table to the `id` column (`main.py:432-437`) and empties the attribute table (`:439-449`).
  - Outer folds come from row positions of the entity table, computed once at start-up (`main.py:305-315`), so seeding may add columns but must never reorder rows.
- **Appending features.** Appends go through `append_new_attribute` (`ext/agent.py:389`; the append itself is `:538-548`). Names already in `df_features` are rejected (`:231`, `:465`, `:486-489`). Results with NaN or inf are rejected too (`:510-519`).
- **Baseline fallback.** The closure `_apply_baseline_fallback_attributes` (`ext/agent.py:599-844`) fires only when an Extractor turn appends nothing (`:996-1003`). It is unsuitable for seeding:
  - it caps at 8 features (`:660`, `:696`, `:734`, `:807`);
  - it skips columns with NaN (`:670-671`, `:704-708`, and `_ordinal_encode_categorical` `:42-82`, NaN exits at `:75-80`);
  - it adds interactions (`:726-799`);
  - its categorical command re-derives the categories from whichever partition it is replayed on (`:829-836`).
  - Note: this departs from 01 §3 E6, which says E6 "reuses" these lines. The reasons above make reuse unsafe.
- **Replay.** Every stored `Pandas Command` is replayed on the fold's train rows and then on its test rows (`main.py:752-758`; `rogueone/utils/fold_safe_features.py:37-187`).
  - A failed replay is NaN-filled (`:84-98`, `:117-124`), and NaN becomes 0 before XGBoost (`tst/agent.py:1694-1705`, `:1839-1850`).
  - Under augment, a broken agent feature no longer collapses the score, so replay failures (Finding 39) can be **masked**. That is why a per-run replay-validity check is mandatory.
  - The check is `paper/scripts/diag_finding39_authoring_frame.py`. `analyse()` (`:103`) rescores each fold's selected trial from its authoring frame and calls the fold reproduced if |Δ| ≤ `REPRO_TOL = 0.01` (`:78`). It compares against the pool's plain `accuracy` / `n_RMSE` column (`:120-121`).
  - It rebuilds folds **unstratified** (`_fold_test_ids`, `:93-100`; `stratify_labels=None` at `:98`) and assumes 10 iterations per fold (trail index `fold_id * 10 + it`, `:139`).
- **Tester.**
  - Pruning: `attribute_pruning_tool` (`tst/agent.py:1269-1327`) sets `Pruned` at `:1302-1310`. It is the only place the Tester writes `Pruned`.
  - Floor: `_enforce_pruning_floor` (`:119-163`, called at `:1617-1623`) keeps max(3, ⌈0.2·pre_active⌉) features, counting all rows (`:135-139`).
  - VIF rule: the prompt prunes members of perfectly collinear clusters (`:1545-1559`).
  - Mutual information returns None for **every** feature if any value is NaN (`rogueone/llm/agents/core/tester/metrics.py:57-72`), and it is computed on un-imputed X (`tst/agent.py:893`, `:967-974`).
  - Scoring: XGBoost with 400 trees (`:1673`, `:1815`), `random_state = fold_seed` (`:1671`, `:1813`).
  - Status merges back through `update_attribute_pool` (`main.py:319-381`).
- **`base_no_fe`.** In `paper/scripts/classicfe_arm.py`, `prep()` (`:102-116`) ordinal-encodes `object` columns alphabetically over the full column (`:112-115`), and `to_matrix` applies `nan_to_num` → 0 (`:138-141`). Augment's iteration-0 pool must equal this input.
  - **dtype trap.** The classicfe image pins pandas 2.2.3 (`docker/Dockerfile.classicfe:22`), where string columns are `object`. The Rogue One image installs pandas 3.0.0 (`setup.py`, `requirements.txt:8`), where `read_csv` gives them the `str` dtype, so a `dtype == object` test finds no categoricals. E6 must use `not pd.api.types.is_numeric_dtype(...)`, as `ext/agent.py:612-617` does.

**Instruments**

- **Core-10 × fold seed 42.**
- **Add-on: SYN-P 7** = `synp_p1`, `synp_p2`, `synp_p6a`, `synp_p6b` (R) and `synp_p3`, `synp_p4`, `synp_p5` (C; E8a confirms the types in `synp_meta/registry.yaml`). n = 2,000, p ≈ 20–30, one planted φ per task (01 §4.1).
  - P7 is excluded under the P7 policy (R10). The breakdown card's "SYN-P 8" (8 runs, +5.6–8u) is overridden by the registry (7 runs, +4.9–7u; 01 Appendix A item 15).
- **Tier B** (car, tic-tac-toe, eucalyptus, bike) is excluded, because there is no fit-on-train replay fix.

The t17 (replace) prior below compares the `mean_across_seeds` column of `paper/results_seed_sweep_t17.csv` (branch tip) with the `base_no_fe` kfold `value` and `value_sd` columns of `paper/results_matched_harness_baselines.csv`. A positive gap means Rogue One is better. Both are plain accuracy (C) or nRMSE (R). Recomputed Wed 7 Oct.

| `dataset_id` | Type | t17 − base | t17 seed SD | Gap / SD | Floor (0.1 × base fold SD) |
|---|---|---|---|---|---|
| `balance-scale` | C | +0.144 | 0.000 | ceiling (harm detector) | 0.0022 |
| `junglechess` | C | +0.025 | 0.070 | +0.4 | 0.0002 |
| `vehicle` | C | −0.018 | 0.029 | −0.6 | 0.0038 |
| `heart` | C | −0.012 | 0.019 | −0.7 | 0.0032 |
| `credit-g` | C | −0.026 | 0.0055 | **−4.7** | 0.0050 |
| `myocardial` | C | −0.052 | 0.0076 | **−6.9** | 0.0033 |
| `housing` | R | −0.017 | 0.021 | −0.8 | 0.0007 |
| `insurance` | R | −0.029 | 0.063 | −0.5 | 0.0019 |
| `crab` | R | −0.010 | 0.024 | −0.4 | 0.0010 |
| `wine` | R | −0.014 | 0.0010 | **−14.8** | 0.0001 |

**Metrics** (R13 names):

- **`heldout_score`.** The outer-fold score at the inner-CV-selected iteration, averaged over 5 folds: balanced accuracy (C) or nRMSE (R).
- **Δ_d.** Signed so that positive means augment is better: score_aug − score_rep for C, and nRMSE_rep − nRMSE_aug for R.
- **σ̂_d.** The per-dataset noise scale from the A03-T3 table, with the estimator fixed in §6.
- **`delta_std`** = Δ_d / σ̂_d.
- **`raw_coverage`.** The share of raw columns whose seeded row is `Active` in an iteration's pool. E6 logs it per iteration to the W&B history; the read-out takes the selected iteration. Under replace it is 0 by construction.
- **`rec_sel`.** An active, non-fallback feature with |Spearman| ≥ 0.95 with φ on training rows (NMI ≥ 0.9 for P3/P5), present in the selected iteration. Raw columns (so every seeded row) and single-column monotone transforms never count. Unit: the fold-episode.
- **`success`** = the run finishes within the cap and scores ≥ `base_no_fe` − 2 × fold SD (01 §4.6). **`func_success`** = `success` **and** at least 1 non-fallback LLM feature in the selected iteration. It matters here because augment can "succeed" on raw columns alone.
- **Net paired loss** (add-on) = Σ `rec_sel`(A02 `R10`) − Σ `rec_sel`(A03 `augment`) over the 35 paired fold-episodes (7 tasks × 5 folds).

## 3. Hypotheses and interpretation (all hypotheses)

- **Hoped.** Augment closes the deficit where it exceeds noise: wine (~15 SD), myocardial (~7 SD) and credit-g (~5 SD). The other gaps are under 1 SD and invisible at one seed. Recovery does not fall.
  - Sentence: *"Keeping raw columns in the pool removes the deficit to raw XGBoost on wine, myocardial and credit-g without lowering planted-feature recovery."* The last clause needs the add-on.
- **Null.** No detectable difference, which is likely on about 7 of 10 datasets. The no-harm rule then still passes with probability ≈ 0.80–0.90 (§6 error rates; (c) adds its own false-fail rate when the add-on runs), and augment is adopted as the **safety default** ("no measurable effect at one seed"). v2 ITT failures are then imputed at `base_no_fe`.
- **Opposite.** Augment hurts. Possible mechanisms:
  - the agents build fewer features;
  - the VIF rule prunes agent features that are collinear with protected rows;
  - long seeded pools crowd the Tester's metrics output (A03-T6b (vi));
  - balance-scale falls below 1.000.

  Then v2 keeps `replace`, the paper says it was kept deliberately, and ITT failures are imputed at the majority-class or mean predictor.

## 4. Design

**Arms** (R5): `replace` and `augment`, set by `ROGUE_ONE_FEATURE_POOL_MODE` on one commit. Everything else is identical.

**Fixed settings** (both arms; draft `scripts/agent2027/v2_defaults.env` at the A03 tag `agent2027-pre.N`, 01 §4.3):
- iterations, folds and selection: `ROGUE_ONE_NUM_ITERATIONS=10` (the code default is 3, `config.py:305`), `ROGUE_ONE_K_FOLDS=5`, `ROGUE_ONE_SELECTION_SIGNAL=inner_cv`, `ROGUE_ONE_ENABLE_ROLLBACK=False`, `ROGUE_ONE_FOLD_RETRY_LIMIT=1`;
- sampling: `LLM_TEMPERATURE` and every `*_AGENT_TEMPERATURE` = 0.7; reasoning effort not passed (`ROGUE_ONE_REASONING_EFFORT_PASSTHROUGH=0`); `EXTRACTOR_AGENT_MAX_TOKENS=32768`, `TESTER_AGENT_MAX_TOKENS=8192`, `SCIENTIST_AGENT_MAX_TOKENS` and `ROGUE_ONE_TOOL_OUTPUT_MAX_CHARS` at their draft values (proposed 16384 and 8000; pinned only at G1);
- E2 and E6 flags: `ROGUE_ONE_EXTRACTOR_ROUNDS=10`, `ROGUE_ONE_TOOL_ERROR_FEEDBACK=message`, `ROGUE_ONE_REPLAY_VALIDITY_CHECK=1`;
- logging and balancing: `ROGUE_ONE_LLM_TRACE=1`, `ROGUE_ONE_PROMPT_DUMP=1`, `ROGUE_ONE_DO_BALANCING=True`, `ROGUE_ONE_DO_LOGGING=True`, `CONSOLE_DISABLE=False`;
- search and sandbox: E4 lands only at G1, so `ROGUE_ONE_SEARCH_CACHE` does not exist on the pre-freeze image and **both arms use live, uncached Serper search**. `ROGUE_ONE_SANDBOX` = `on` if E13 merged at G0a, else `off`; the same in both arms;
- serving and launch: `vllm/vllm-openai:v0.22.0`, `PARALLEL=6`, `CPUS=12`, `RUN_CAP_HOURS=6`;
- seeds: `fold_seed = llm_seed = 42`, `replicate=0`.

**Folds:**
- Core-10 uses KFold (`ROGUE_ONE_STRATIFIED_FOLDS=False`), matching the `base_no_fe` kfold rows.
- Add-on rows copy the `ROGUE_ONE_STRATIFIED_FOLDS` value of the matching A02 `R10` row (A02 sets `True` on p3, p4, p5, p7), so the partitions match.

**Run matrix.** Project `RogueOne-AGENT2027-prefreeze`, group `A03`. Run name `A03__<arm_id>__<dataset_id>__f42__r0`.

| Rows | Arm | Instrument | Seeds (fold/LLM) | Runs | u | Image | Session | Paired with |
|---|---|---|---|---|---|---|---|---|
| A03-901, A03-903 (smoke) | `replace` | myocardial, credit-g; `ROGUE_ONE_SMOKE_FOLDS=1` | 42/42 | 2 | 0.4 | A03 `agent2027-pre.N` | `S1014e` | A03-902/904 |
| A03-902, A03-904 (smoke) | `augment` | myocardial, credit-g; `ROGUE_ONE_SMOKE_FOLDS=1` | 42/42 | 2 | 0.4 | A03 `agent2027-pre.N` | `S1014e` | A03-901/903 |
| A03-905 (smoke) | `replace` | myocardial; `ROGUE_ONE_SMOKE_FOLDS=1` | 42/42 | 1 | 0.2 | **A02's** pre-E6 tag | `S1014e` | A03-901 (prompt baseline) |
| A03-001…010 | `replace` | Core-10 | 42/42 | 10 | 10 | A03 `agent2027-pre.N` | `S1014b` | A03 `augment`, same dataset |
| A03-011…020 | `augment` | Core-10 | 42/42 | 10 | 10 | A03 `agent2027-pre.N` | `S1014b` | A03 `replace`, same dataset |
| A03-021…027 (AB1) | `augment` | SYN-P 7 | 42/42 | 7 | 4.9–7 | A03 `agent2027-pre.N` | `S1014b` | A02 `R10` rows A02-001…008 without `synp_p7` (0 new runs) |
| A03-031…040 (fallback only) | `replace` | Core-10 | 42/42 | 10 | 10 | `agent2027-v2.0`, project `RogueOne-AGENT2027` | A05's `S1016a` (and `S1016b` if D5 = TP2) | A05 `full` A05-001…010 (Core-10 × f42 × r0) |

Smoke rows carry `ROGUE_ONE_CODE_PHASE=smoke` and `notes=smoke`, and are excluded from every analysis. A03 uses the **next unused** `agent2027-pre.N` (one higher than A02's): never re-use a tag, because A02's end-of-session retries resolve `rogue_one:<image_tag>` when they start.

**Budget.** 2 × 10 × 1u = 20u, plus 7 × 0.7–1.0u = 4.9–7u, plus ≈ 1.0u of smoke. That is 21–28u of the 126u gross pre-freeze capacity (Tue 13 00:00 → Fri 16 18:00 = 90 h × 1.4 u/h), which is otherwise mostly idle. The master plan's post-freeze re-check (frozen replace × Core-10, 10u) is **not** in T1-minimum; it runs only on the fallback path (rows A03-031…040).

**Order.**
- `S1014e` (smoke) runs first and alone; `S1014b` launches only after A03-T6a and A03-T6b pass.
- In `S1014b` the 20 Core-10 rows run in the random `order_key` order that `make_manifest.py --plan-session` assigns (seeded by the session ID), so both arms see the same load.
- The add-on rows come last, so they never delay the Core-10 decision: A03-T8 adds 1000 to their `order_key` (the launcher runs rows in `order_key` order, 01 §5.3).
- There is no sentinel, because sentinels start Tue 20 Oct.

**Failures.**
- A crash (rc ≠ 0, or no `fold_aggregation_summary.csv`) is retried once, alone, at the end of the session. A 6 h cap hit is not retried.
- **Both arms follow the same rule.**
  - Primary: the decision uses complete pairs only, and failures are reported separately.
  - If augment fails or caps where replace completes, that dataset gets a harm flag (§6 (b)).
  - Sensitivity: both arms ITT-imputed at the majority-class or mean predictor.
- A failed or capped run scores `rec_sel = rec_any = 0` for all its episodes (A02 `R10` rows too).

## 5. Tasks

### 5.1 Prerequisites
- [ ] **A03-T1** [type: decision] (0.1 d, due Mon 12 Oct) — Decide **D6**:
  - run the add-on (AB1; recommended, because it is the only test of "recovery does not fall");
  - accept the E6 fallback (§6).

  If Henrik records nothing by Mon 12 Oct 24:00, the 01 §6.4 default applies: run the add-on and accept the fallback. *Done when:* both decisions are written into the draft `docs/agent2027/preregistration.md`.
- [ ] **A03-T2** [type: code] (0 d charged here; E1 0.75 + E2 1.1 + E3 1.25 + E5 0.5 + E13 0.5 = 4.1 d, booked in 01 §3; due Mon 12 Oct, G0a) — Get E1, E2 (with the R knob), E3, E5 and E13 (or ED1) merged on `agent2027/v2`. *Done when:*
  - the G0a smoke run is in W&B under its R12 name with non-zero `tok_compl_scientist`, `tok_compl_extractor` and `tok_compl_tester`, every `flags/<NAME>` key, and a `run-output` artifact with traces;
  - `git ls-tree agent2027/v2 .dev.env` prints nothing. **`.dev.env` is tracked at `07cbf93`** (it sets `LLM_PORT=8000` and `ROGUE_ONE_K_FOLDS=5`), and `config.py:24` loads it with `override=True`, so E1 must `git rm` it; otherwise the A03-T9 guard stops the build.
- [ ] **A03-T3** [type: data] (0.1 d, due Mon 12 Oct, 20:00) — Build the 10-row σ̂ table from three sources:
  - A01 `sigma_samp` for its 8 datasets (hand-off A01-T18, Mon 12 Oct 18:00; A01 names this task A03-T3);
  - the t17 `sd_across_seeds` for all 10 (`paper/results_seed_sweep_t17.csv` at the branch tip);
  - the floor from §2.

  Use t17 alone for any dataset (or all, if A01 flags pooled drift) where A01 reports drift from t17 (its drift verdict: mean shift and/or variance inflation, with the trigger named; see A01-T18) or if the hand-off has not arrived by Mon 12 Oct 18:00. Give σ̂_d under both estimators of §6. *Done when:* the table is committed in the draft `docs/agent2027/preregistration.md`, next to the rule.
- [ ] **A03-T4** [type: data] (0.1 d, add-on only; due Tue 13 Oct, 22:00, after the A03 tag exists) — Check four things:
  - the SYN-P 7 folders exist in `<ROGUEONE_CAIR_PATH>/tasks/<task_type>/synp_<id>/`, hold no φ metadata, and are listed in the `ROGUE_ONE_TASK_REGISTRY` YAML (the legacy list at `main.py:1536-1566` has no SYN-P);
  - `scripts/agent2027/analysis/recovery_detector.py` passes its unit test;
  - the M5S labels have been read (Tue 13 Oct, morning);
  - locally, `git diff --stat <A02 tag> <A03 tag> -- rogueone rogue_one_main.py` lists only E6 files (`rogue_one_main.py`, `tst/agent.py`, `rogueone/utils/fold_safe_features.py`, `config.py` / `rogueone/utils/config_keys.py` and the E1 flag registry for the two new flags). Changes outside `rogueone/` and `main.py` (launcher scripts, X1, X2, docs, manifests, tests, and E6's optional argument in `diag_finding39_authoring_frame.py`) do not change agent behaviour.

  *Done when:* all four hold, or the add-on is dropped and the drop is logged in the pre-registration.

### 5.2 Implementation
- [ ] **A03-T5** [type: code] (E6, 1–1.5 d; start Sat 10 or Mon 12; **merged by Tue 13 Oct, end of day**) — Implement `ROGUE_ONE_FEATURE_POOL_MODE=replace|augment` with these points:
  - **(a) Seeding.** Seed inside `_reset_fold_state`, after `main.py:439-449`, on every call (E9a relies on this). Put the seeding logic in one pure function (name recorded in the PR) that the reset calls and A03-T6a calls directly.
    - Add one column per raw column of `data.csv` except `id`, as `prep()` in `classicfe_arm.py` does.
    - Add one matching row to `df_attributes_explanations`: `Status="Active"`, `Added="Seed"`, and a replayable command.
    - Never reorder rows.
  - **(b) No fallback reuse.** Do not reuse the fallback or `_ordinal_encode_categorical` as they stand (§2).
  - **(c) Numeric columns.** Use the command `result = df_features['id'].map(df.groupby('id')['<col>'].first()).fillna(0.0)`, embedding the column name with `repr()`.
    - This is the canonical shape (`fold_safe_features.py:62-68`).
    - `fillna(0.0)` scores the same as `nan_to_num`, and it keeps mutual information computable.
  - **(d) Categorical columns.** A column is categorical if `not pd.api.types.is_numeric_dtype(...)` (the §2 dtype trap). Embed a **literal** alphabetical category→code map, built once over the full column as in `classicfe_arm.py:112-115`. Name the column `<col>_ord` (as at `ext/agent.py:810`) and add `.fillna(0.0)` (missing → 0, as `prep()` + `nan_to_num` gives).
  - **(e) Protection.**
    - `attribute_pruning_tool` skips seeded names and says so in its return text (`tst/agent.py:1302-1310`).
    - `_enforce_pruning_floor` counts only non-seeded rows (`:135-139`).
    - `update_attribute_pool` never sets `Pruned` on a seeded row (`main.py:325-352`).
  - **(f) Replay check.**
    - With `ROGUE_ONE_REPLAY_VALIDITY_CHECK=1`, call `analyse()` from `paper/scripts/diag_finding39_authoring_frame.py` at the end of the run, on the row's own task folder (outputs are in `<task folder>/output`). `paper/scripts` is not a package, so import the file by path (`/workspace/paper/scripts/…`).
    - Add an optional `fixed_test_ids=None` argument to `analyse()` (used instead of `_fold_test_ids` at `:127`; the default keeps today's output) and pass the run's own `_tabular_fixed_test_ids` (`main.py:313-315`). This covers stratified add-on rows, which `:98` would otherwise rebuild unstratified.
    - Log `replay_nonrepro_folds` (folds with `reproduced == False`) and `replay_nan_fills` (a run-total counter of the NaN fills at `fold_safe_features.py:89`, `:98`, `:124`) to the W&B summary.
  - **(g) Coverage log.** Log `raw_coverage` to the W&B history at every iteration.
  - **(h) Leave unchanged:**
    - the prompts (`sci/agent.py:437` still asks for a raw baseline first, and `:475` says round 1 has no attributes);
    - the iteration-0 abort (`main.py:606-717`). Under augment it can still fire only if the Extractor and its fallback both add nothing; the fallback's interactions over seeded columns (`ext/agent.py:726-799`) make that unlikely.

  *Done when:* the extended `tests/unit/test_fold_lifecycle.py` (re-seeding on every reset; the replace path stays id-only) and `tests/unit/agents/test_tester_pruning_floor.py` (seeded rows are unprunable and not counted) pass locally, `scripts/agent2027/e6_seed_check.py` (A03-T6a) is committed, and A03-T6a and A03-T6b pass.
- [ ] **A03-T6a** [type: test] (0.25 d, Wed 14 Oct, 08:30, right after the A03-T9 build; CPU only, in containers on CAIR) — Three steps:
  1. **Reference** (pandas 2.2.3, the real `base_no_fe` path) in the classicfe image:
     ```
     ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker image inspect classicfe:4 --format '{{.Id}}' && mkdir -p ~/ro27_e6check && docker run --rm --network none --cpus 4 -v <ROGUEONE_CAIR_PATH>/tasks:/tasks:ro -v <ROGUEONE_CAIR_PATH>/paper/scripts/classicfe_arm.py:/arm.py:ro -v ~/ro27_e6check:/out classicfe:4 python -c \"import sys; sys.path.insert(0, '/'); import arm, pandas as pd; [pd.DataFrame(arm.to_matrix(X), columns=X.columns).assign(id=pd.read_csv(f'/tasks/classification/{d}/data.csv')['id'].to_numpy()).to_csv(f'/out/{d}_prep.csv', index=False) for d in ('myocardial', 'credit-g', 'heart') for X in [arm.prep(d, 'classification', '/tasks')[0]]]\""
     ```
  2. **Seed check** in the A03 image: `scripts/agent2027/e6_seed_check.py` calls the A03-T5 (a) seeding function on each `data.csv`, compares every seeded column (`<col>` or `<col>_ord`) with the reference by `id` (max |diff| ≤ 1e-6 after float32), then replays every seed command with `materialize_fold_safe_features` on the train rows and the test rows of fold 0 (`deterministic_kfold_indices`, seed 42) and requires identical values to the authoring frame:
     ```
     ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker run --rm --network none --cpus 4 -v <ROGUEONE_CAIR_PATH>/tasks:/workspace/tasks:ro -v ~/ro27_e6check:/check:ro --entrypoint python3 rogue_one:agent2027-pre.N scripts/agent2027/e6_seed_check.py --tasks /workspace/tasks --ref /check --datasets myocardial,credit-g,heart"
     ```
  3. **Unit tests in the image.** pytest is not in `setup.py`'s `install_requires`, so the image lacks it unless E1–E5 added it; install the pins from `requirements.txt:30-31` in the throw-away container:
     ```
     ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker run --rm --cpus 8 --entrypoint bash rogue_one:agent2027-pre.N -c 'python3 -m pip install -q pytest==8.3.4 pytest-asyncio==0.24.0 && python3 -m pytest tests/unit -q'"
     ```

  *Done when:* the script prints `E6_SEED_CHECK PASS` for all three datasets (checks (i) seeded matrix = `prep()` output after `nan_to_num`, and (ii) seeds replay identically on both partitions), and pytest exits 0. On failure: fix, re-tag `pre.N+1`, rebuild (A03-T9), repeat; the launch kill (§6) applies.
- [ ] **A03-T6b** [type: run] (0.25 d, Wed 14 Oct, ≈ 08:45–10:15; session `S1014e`, launched by A03-T10) — Read the five smoke rows (one fold each). *Done when* all of the following hold:
  - **(i, live)** A03-902/904 log `raw_coverage` = 1.0 at iteration 0 in the W&B history; A03-901/903 have no `Added == "Seed"` row in `df_attributes_explanations_final.csv` (`run-output`; with one fold this file covers the whole run).
  - **(iii)** Under replace, `scripts/agent2027/prompt_diff.py` on A03-901 vs A03-905 (same dataset and seeds, pre-E6 image) reports the system prompts of S, E, T and W and the Scientist's iteration-0 user prompt byte-identical. Later prompts carry sampled text and are not compared.
  - **(iv)** A03-902 vs A03-901: the same prompts differ only in attribute listings.
  - **(v)** `replay_nonrepro_folds` and `replay_nan_fills` are in the W&B summary of A03-901…904.
  - **(vi)** On A03-902 (myocardial, augment), `context_errors` = 0 and the Tester's precomputed-metrics tool output in the trace carries no E2 truncation marker (`ROGUE_ONE_TOOL_OUTPUT_MAX_CHARS`).
  - **(vii)** Iteration-0 fallback activations under augment (the "Extractor fallback activated" line, `ext/agent.py:999-1002`) are counted and recorded.

  **On failure:**
  - (i, live), (iii), (iv), (v): an E6 bug. Fix, re-tag, rebuild, re-smoke as a new session (`S1014f`, …).
  - (vi): Henrik decides before `S1014b` between an augment-only compact listing of seeded rows in the Tester's metrics output (code change, re-smoke) and proceeding with the truncation disclosed. Default if he has not decided by Wed 14 Oct 12:00: proceed, disclosed.
  - (vii) > 0: Henrik decides before `S1014b` whether an augment-only wording fix (`sci/agent.py:437`, `:475`) becomes part of the `augment` arm (code change, re-smoke). Default if he has not decided by 12:00: no fix; disclosed through `fallback_share` (risk R7).
- [ ] **A03-T7** [type: decision] (0.25 d, due Tue 13 Oct, before any A03 run) — Commit the §6 rule verbatim to the draft pre-registration, with the σ̂ table, the failure rule, the no-decision default and the σ̂ estimator (§6 default unless Henrik picks the alternative). *Done when:* the commit time precedes the first A03 W&B `created_at` (smoke rows included).
- [ ] **A03-T8** [type: config] (0.2 d, due Tue 13 Oct, after A02-T8 has fixed A02's tag) — Run `scripts/agent2027/make_manifest.py --ablation A03` to write `manifests/agent2027/A03_M1.csv` (R7 columns), with these values:
  - `rung_S/E/T=O120`, `port_S/E/T=8010`, `cap_hours=6`, `is_sentinel=0`, `replicate=0`, `fold_seed=llm_seed=42`;
  - `image_tag` = the A03 tag, except A03-905 (A02's tag) and A03-031…040 (`agent2027-v2.0`);
  - `expected_u` = 1.0 (Core-10), 0.85 (SYN-P) or 0.2 (smoke);
  - `env_overrides` = `{"ROGUE_ONE_FEATURE_POOL_MODE":"<arm>","ROGUE_ONE_EXTRACTOR_ROUNDS":"10"}` on A03-001…027 and A03-901…904, plus the copied `"ROGUE_ONE_STRATIFIED_FOLDS":"True"` on classification add-on rows and `"ROGUE_ONE_SMOKE_FOLDS":"1"` on smoke rows; A03-905 = `{"ROGUE_ONE_EXTRACTOR_ROUNDS":"10","ROGUE_ONE_SMOKE_FOLDS":"1"}` (pre-E6 code is replace-only); fallback rows A03-031…040 = `{"ROGUE_ONE_FEATURE_POOL_MODE":"replace"}` only (R and everything else as frozen);
  - `notes`: `smoke` (A03-901…905), `pair A02-<nnn>` (add-on), `fallback` (A03-031…040).

  Then run `make_manifest.py --plan-session S1014e --rows A03-901 … A03-905` and `make_manifest.py --plan-session S1014b --rows A03-001 … A03-027`, and add 1000 to the `order_key` of A03-021…027 in `S1014b.csv`. A07's smoke rows A07-901/902 may be appended to the end of `S1014b` the same way if A07-T6 has not run yet; they are not analysed here. *Done when:* the files are committed and pushed; container names `ro27-s1014e-a03-90n` and `ro27-s1014b-a03-0nn` are unique; the add-on rows sort last.

### 5.3 Execution on CAIR

**Hard rules** (01 §5.1):
- containers only: every execution runs in Docker; the host runs nothing but `docker` / `docker compose` (with `-T` over SSH), the `~` launchers, and `nvidia-smi` / `docker ps` / `tail` / `grep` for monitoring — never `python`;
- GPU pre-flight before every GPU launch; if not enough GPUs are free, **stop and report** (per-GPU memory and utilisation, compute apps, running containers);
- at most 4 of the 8 H100s without Henrik's explicit approval in chat, and one Rogue One job at a time;
- W&B is the system of record (entity `PlanBench`): stateless runs, a `run-output` artifact per run, secrets from `.env` (`WANDB_API_KEY` from `<CAIR_HOME>/code/PlanBench/.env`);
- SSH is one-shot and non-interactive (`-o BatchMode=yes -o ConnectTimeout=30`); long jobs run detached with `nohup`, logging to `~`, never to the root-owned run folders;
- code reaches CAIR only by local commit + push, then `git fetch` over SSH.

`<ROGUEONE_CAIR_PATH>` = `<CAIR_HOME>/code/RougeOne` (**inferred; verify once** with `ssh CAIR-gpu14 "ls -d ~/code/RougeOne"`).

- [ ] **A03-T9** [type: run] (0.2 d; local part Tue 13 Oct evening, CAIR part **Wed 14 Oct, 08:00**) — Locally, after E6 merges: tag the next unused `agent2027-pre.N`, push the branch and the tag, and note the tag's commit SHA (`<SHA_A03>`). On CAIR, **only after `S1013a` has ended** (updating the checkout or `~` launchers while A02's session runs would change files it is executing):
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "grep 'RO27_SESSION_END' ~/ro27_S1013a.job.log || echo NOT_ENDED"
  ```
  If it prints `NOT_ENDED`, re-check hourly (the launch kill in §6 applies). Then:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15' git fetch git@github.com:HenrikNorkart/RougeOne.git agent2027/v2 && git checkout -B agent2027/v2 <SHA_A03> && git log -1 --format='HEAD: %H %s' && test ! -e .dev.env && docker build -t rogue_one:agent2027-pre.N . && docker image inspect rogue_one:agent2027-pre.N --format '{{.Id}}' && cp scripts/agent2027/ro27_session_launch.sh scripts/agent2027/ro27_clients_manifest.sh ~/ && cp paper/scripts/vllm_session.sh ~/ro27_vllm_session.sh && mkdir -p ~/ro27_manifests && cp -r manifests/agent2027/. ~/ro27_manifests/"
  ```
  *Done when:*
  - HEAD equals `<SHA_A03>`;
  - the image ID is recorded in the session plan;
  - there is no `.dev.env` (`config.py:24` loads it with `override=True`; see A03-T2);
  - A02's image tag still resolves to A02's image ID (the tags differ).
- [ ] **A03-T10** [type: run] (0.1 d; **Wed 14 Oct**, `S1014e` ≈ 08:30, `S1014b` ≈ 10:30 after A03-T6a and A03-T6b pass) — For **each** of the two sessions, run the pre-flight first:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader && echo '--- compute apps ---' && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader && echo '--- containers ---' && docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'"
  ```
  **Stop and report** unless:
  - 4 GPUs each show < 2,000 MiB used;
  - no `ro27-*`, `rogueone-vllm-*` (including A01's `rogueone-vllm-o120-<sid>` and the script default `rogueone-vllm-session`), `llmfe-*` or `caafe-*` container is running.

  Then launch detached with those 4 GPU ids (shown for `S1014e`; for the main session replace every `S1014e` / `s1014e` with `S1014b` / `s1014b`):
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "SESSION_ID=S1014e SESSION_FILE=~/ro27_manifests/sessions/S1014e.csv GPUS=<4 ids> NEED=4 PORT=8010 MODEL=openai/gpt-oss-120b SERVER_NAME=rogueone-vllm-o120-s1014e LOG=~/vllm_session_S1014e.log WAIT_MAX_MIN=0 IMAGE_RO=rogue_one:agent2027-pre.N RUNROOT=~/rogueone-runs-agent2027 PARALLEL=6 CPUS=12 RUN_CAP_HOURS=6 WANDB_PROJECT_NAME=RogueOne-AGENT2027-prefreeze nohup bash ~/ro27_session_launch.sh > ~/ro27_S1014e.job.log 2>&1 & echo launched; sleep 5; tail -5 ~/ro27_S1014e.job.log"
  ```
  Each row runs its own `image_tag` (01 §5.3 step 3), so A03-905 runs A02's image. Between the sessions, wait for `RO27_SESSION_END` in `~/ro27_S1014e.job.log` (A03-T11 watcher) and run the tear-down check. *Done when*, for each session, the job log shows:
  - vLLM 0.22.0;
  - both image IDs;
  - the energy sidecar `ro27-<session>-energy` (W&B run `<SESSION_ID>__energy`);
  - the first wave of `ro27-<session>-a03-*` containers (5 for `S1014e`, 6 for `S1014b`).
- [ ] **A03-T11** [type: run] (0.2 d, Wed 14 – Thu 15 Oct) — For each session, start a background watcher:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 CAIR-gpu14 "while ! grep -q 'RO27_SESSION_END' ~/ro27_S1014b.job.log 2>/dev/null; do sleep 120; done; tail -30 ~/ro27_S1014b.job.log"
  ```
  Exit code 255 means the network dropped, not the job; reconnect.
  - After 1 h of `S1014b`, check one `augment` run in W&B:
    - `flags/ROGUE_ONE_FEATURE_POOL_MODE=augment`;
    - iteration-0 `raw_coverage` = 1.0 in the history (otherwise stop: §6 manipulation check);
    - tags [`S1014b`, `prefreeze`, `agent2027-pre.N`].
  - A W&B run in state `running` with no live container is a zombie.
  - At the end, run the tear-down check:
    ```
    ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker ps --format '{{.Names}}' | grep -E '^(ro27-|rogueone-vllm-)' || echo none; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"
    ```

  *Done when:* every row of both sessions is `OK`, `CAP_HIT` or `FAILED` after retry, each is matched in W&B, and the tear-down check prints `none`.
- [ ] **A03-T12** [type: run] (0.1 d; **fallback only**; planned Fri 16 Oct 15:00, runs from 18:00) — Add rows A03-031…040 to the row list of A05-T5's `--plan-session S1016a` (alternating with `S1016b` if D5 = TP2), so they interleave with A05 `full` on `agent2027-v2.0`. They add 10u (≈ 7 h) to A05's sessions. *Done when:* all 10 rows have finished by Sun 18 Oct 24:00, in project `RogueOne-AGENT2027`, group `A03`, with tags [`S1016a`, `v2`, `agent2027-v2.0`].

### 5.4 Analysis
- [ ] **A03-T13** [type: analysis] (0.5 d, due **Thu 15 Oct, 12:00**) — Write `scripts/agent2027/analysis/a03_readout.py`, commit and push it, and run it on CAIR in a container. It is newer than the image, so fetch it without moving the server checkout:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15' git fetch git@github.com:HenrikNorkart/RougeOne.git agent2027/v2 && mkdir -p ~/ro27_analysis && git show FETCH_HEAD:scripts/agent2027/analysis/a03_readout.py > ~/ro27_analysis/a03_readout.py && docker run --rm --cpus 16 --env-file <CAIR_HOME>/code/PlanBench/.env -v ~/ro27_analysis:/analysis -v ~/rogueone-synp-meta:/meta:ro --entrypoint python3 rogue_one:agent2027-pre.N /analysis/a03_readout.py --meta /meta --out /analysis"
  ```
  The φ metadata is mounted only into this analysis container, never into an agent container. The script pulls group `A03` (smoke rows excluded) and A02 `R10` from W&B, with their `run-output` artifacts, and computes:
  - per dataset: Δ_d, σ̂_d (both estimators), `delta_std`, Δ_d/(√2·σ̂_d), failures, `raw_coverage`, the replay keys, and each arm's Δ vs `base_no_fe` (`value_balanced` for C, `value` for R, kfold rows of `paper/results_matched_harness_baselines.csv`);
  - the mean of `delta_std` over complete pairs, and a Wilcoxon signed-rank test across the datasets (descriptive only);
  - sensitivity: (a) and (b) recomputed on plain outer-fold accuracy (C), the unit of σ̂ (risk R6); and with non-reproduced folds removed (risk R3);
  - for the add-on: `rec_sel` per task for `augment` vs `R10`, the net paired loss, an exact McNemar test, and the same figures without M5S non-discriminating tasks;
  - per arm: `success` (`base_no_fe` from the same CSV for Core-10 and from A08's CPU-screen cells at seed 42 for SYN-P), `func_success`, `tok_compl_total`, `wall_s`, `fun_*`, `prune_rate`, `floor_activations`, `reproposal_ratio` and `fallback_share`.

  *Done when:* run `ANALYSIS__a03_readout__20261015` (project `RogueOne-AGENT2027`, group `ANALYSIS`) exists with the CSV as artifact `a03-readout` and PASS / FAIL / untested for each §6 criterion with its numbers.
- [ ] **A03-T14** [type: decision] (0.1 d; decided Thu 15 Oct, by 23:59 at the latest; applied Fri 16 Oct) — Apply §6 and write `ROGUE_ONE_FEATURE_POOL_MODE=<augment|replace>` into `scripts/agent2027/v2_defaults.env`. *Done when:* the value is in tag `agent2027-v2.0-freeze`, and the outcome of every criterion is recorded in the pre-registration.
- [ ] **A03-T15** [type: analysis] (0.25 d, due Mon 19 Oct, G2) — Then do one of two things:
  - if augment was adopted by the rule: an exploratory drift check, A03 `augment` vs A05 `full` (Core-10 × 42). This compares pre- and post-freeze code (cache, sandbox, R, D2, D7); it does not re-check the decision.
  - on the fallback path: the A03-031…040 read-out against §6 (a) and (b), with Δ = A05 `full` − A03 `replace`.

  *Done when:* the result is logged in the G2 notes.
- [ ] **A03-T16** [type: analysis] (0.25 d, Sun 1 – Tue 3 Nov) — Make the final appendix tables, and hand the pre-freeze points to A21/A23 as a separate panel. *Done when:* the tables are in the paper repo.

### 5.5 Paper outputs
- [ ] **A03-T17** [type: writing] (0.1 d, Fri 16 Oct, inside X10) — Have the pre-registration state three things:
  - the M1 rule as applied, including the σ̂ estimator;
  - that M1 ran on pre-freeze code (live search, R = 10, draft caps);
  - that M1's results were seen before the pre-registration commit.

  *Done when:* this is in tag `agent2027-prereg`.
- [ ] **A03-T18** [type: writing] (0.25 d, Wed 4 – Fri 6 Nov) — Write:
  - one design-section sentence on which pool mode Full-v2 uses and why, including the ITT consequence;
  - an appendix "M1 gate" table: per-dataset Δ, `delta_std`, flags, failures, `raw_coverage` and replay check, plus the add-on recovery table.

  *Done when:* both are in the draft.

## 6. Decision rules and gates (committed Tue 13 Oct, before any A03 run)

**Adopt `augment` iff (a) and (b) hold, and (c) holds or is "untested":**
- **(a)** The mean of `delta_std` over the complete Core-10 pairs (at least 8 of 10) is ≥ −0.75.
- **(b)** No complete pair has Δ_d/(√2·σ̂_d) < −2.5, and no dataset has augment failing or capping where replace completed.
- **(c)** Recovery does not fall, over all 7 add-on tasks (35 paired fold-episodes, as A02 uses all its tasks):
  - the net paired loss (§2) is ≤ 4;
  - no task falls from ≥ 3/5 under `R10` to 0/5 under `augment`.

  Without the add-on, or if its rows or A02's `R10` rows are not complete by Thu 15 Oct 23:59, (c) is recorded as **"untested"**, and that is disclosed. The version without M5S non-discriminating tasks is a sensitivity analysis only. (01 §4.1 excludes those tasks from *confirmatory* recovery contrasts; (c) is a within-system no-harm gate, so it keeps them.)

**σ̂_d (pre-registered default):** σ̂_d = max(A01 `sigma_samp`_d, t17 seed SD_d, 0.1 × the `base_no_fe` fold SD_d), with A01 omitted where A03-T3 voids it (always for `balance-scale` and `wine`). The alternative is the 01 §4.7 wording: A01 `sigma_samp`, falling back to the t17 seed SD, with the same floor. Henrik may choose the alternative only in A03-T7, before any A03 run; if he records nothing, the default applies.

**Error rates under a true null** (simulated, normal errors, 10 independent datasets): with σ known, about 5% for (a), 6% for (b), ≈ 10% jointly. With the default estimator, 10–21% jointly, depending on how much of t17's SD is sampling noise. With A01 alone (2 df), about 50% (risk R2). (c) adds a false-fail rate of roughly 10–17% (A02's figure for its identical criterion over 40 episodes; not re-simulated for 35).

**Manipulation check.** `raw_coverage` = 1.0 at iteration 0 in every augment fold. A failure is an E6 bug to fix and rerun, not evidence. Datasets that cannot be rerun before Thu 15 Oct 23:59 count as incomplete pairs.

**Outcomes:**
- **Pass:** freeze `augment`; v2 ITT is imputed at `base_no_fe`.
- **Fail:** freeze `replace`; ITT is imputed at the majority-class or mean predictor.
- Sensitivity analyses (A03-T13) are reported but never change the outcome.

**Gates and kill dates:**
- **G-E6, Tue 13 Oct, end of day.** If E6 has not merged, take the **fallback**:
  - freeze augment by design, with E6 merged by Fri 16;
  - run A03-031…040 against A05 at seed 42 (A03-T12). This is no longer a gate.
  - If that read-out fails (a) or (b) at **G2 (Mon 19 Oct)**, Henrik chooses between reverting to replace (an M0 + HC1 rerun, about 60–70u, which fits only then) and reporting it as a limitation.
- **Launch kill, Thu 15 Oct, 08:00.** If `S1014b` has not started by then, take the fallback.
- **No-decision default.** If fewer than 8 of the 10 Core-10 pairs are complete by Thu 15 Oct, 23:59, adopt augment by design and run the fallback rows.
- **Add-on rows** still running at that time are read later; they never hold up G1. If (c) is read after G1 and fails, the pool mode is **not** re-opened; the failure is reported as a limitation.

## 7. Risks and cut levers

| # | Risk | Mitigation |
|---|---|---|
| R1 | E6 slips: the G0a week is already over-full | Start E6 on Sat 10 or Mon 12; otherwise the pre-declared fallback |
| R2 | **σ̂ has only 2 df.** Simulated false-harm rate under a true null: about **50%** with A01 alone; 10–21% with the max estimator; about 10% with σ known | §6 default σ̂_d = max(A01 `sigma_samp`, t17 SD, floor); Henrik may switch only before any A03 run (A03-T7) |
| R3 | Augment masks replay failures (Finding 39) | Per-run replay check; a sensitivity analysis without non-reproduced folds |
| R4 | Pre-freeze code ≠ Full-v2: live search (E4 not merged), sandbox, R (A02 may freeze R=3), draft output and token caps, D2, D7 | Disclosed; G2 drift check (A03-T15) |
| R5 | The add-on pairs across sessions and across commits (A02 runs the pre-E6 tag) | Valid only if the A03-T4 diff is E6-only and A03-T6b (iii) passes; otherwise the add-on is reported as exploratory |
| R6 | σ̂ is on plain accuracy, but `heldout_score` is balanced accuracy | Disclosed; sensitivity on plain accuracy (A03-T13); matters most on imbalanced `myocardial` |
| R7 | Round-1 prompts ask for raw baselines, which are now duplicates, so the iteration-0 fallback fires under augment | Measured through `fallback_share` and A03-T6b (vii) |
| R8 | Box contention with A02's tail and A04 sessions A/B | One job at a time; A03 Core-10 has priority until Thu 15 |
| R9 | `.dev.env` is tracked in git and would override `-e LLM_PORT` and other flags | A03-T2 (removed from `agent2027/v2`); `test ! -e .dev.env` in A03-T9 |
| R10 | pandas 3.0 `str` dtype makes a `dtype == object` test miss every categorical | A03-T5 (d); A03-T6a compares with the pandas 2.2.3 reference |
| R11 | Updating the server checkout, the `~` launchers or a re-used image tag while A02's `S1013a` runs | A03-T9 waits for `RO27_SESSION_END`; A03 uses a new `pre.N` |
| R12 | Seeded pools (about 100 columns on `myocardial`) push the Tester's metrics output past `ROGUE_ONE_TOOL_OUTPUT_MAX_CHARS` | A03-T6b (vi) with a pre-declared default |

**Cut levers.** M1 on Core-10 is never cut. Only the add-on (AB1) is optional. Drop it if E6 merges late on Wed 14 or A03-T4 fails. AB1 can only run before the freeze, because it must pair with A02 on pre-freeze code. D6 on Mon 12 is therefore the binding decision.

## 8. Definition of done
- [ ] D6 recorded. The rule, the σ̂ table and the estimator committed before the first A03 run.
- [ ] `.dev.env` absent from `agent2027/v2`; E6 merged, with A03-T6a and A03-T6b checks (i)–(vii) passed.
- [ ] 20 Core-10 rows, plus 7 add-on rows if AB1 runs, finished or ITT-classified. All are in W&B with `run-output` artifacts. Both sessions torn down.
- [ ] `ANALYSIS__a03_readout__20261015` exists with artifact `a03-readout`.
- [ ] Pool mode frozen in `v2_defaults.env` (tag `agent2027-v2.0-freeze`), with the matching ITT rule.
- [ ] Pre-registration disclosure in `agent2027-prereg`.
- [ ] G2 drift check, or the fallback read-out, logged.
- [ ] Appendix table and design sentence drafted (Nov 4–6).

---

**New names introduced** (not in the canonical registry):
- `scripts/agent2027/analysis/a03_readout.py`: the read-out script (A03-T13), with W&B artifact `a03-readout`.
- `scripts/agent2027/e6_seed_check.py`: the CPU seed-equivalence check (A03-T6a).
- `Added="Seed"`: the marker for seeded rows in `df_attributes_explanations`.
- `replay_nonrepro_folds` and `replay_nan_fills`: W&B summary keys for E6's replay check.
- Host scratch folders `~/ro27_e6check/` and `~/ro27_analysis/`.
- Sessions `S1014e` (E6 smoke) and `S1014b` (main), and rows `A03-001…040` and `A03-901…905`, follow the R7/R8 patterns. `S1014a` stays reserved for A02's contingency, which runs after `S1014b`.
