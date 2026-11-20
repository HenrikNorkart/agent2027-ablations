# A22 — M3: Evaluator transfer (CPU re-scoring)

| Field | Value |
|---|---|
| **ID** | A22 · M3 |
| **Claim / Holm contrast** | None. This is a validity check outside the Holm family. It supports every sentence that rests on held-out score or on "the features are informative" (external validity of the last funnel stage). |
| **Tier** | T1-minimum, flagged optional. It is engineering drop **ED5**: the fifth item to go, after E13, X2, X7 and SET (01 §3.1; R2), and it then moves to the short paper. |
| **Owner** | Henrik |
| **Start / finish** | Prerequisites checked Mon 12 Oct (G0a; T2 at 18:00, T1 at 22:00) and Tue 13 – Thu 15 Oct (T3, T4) · Core-10 OpenFE re-run Wed 14 Oct, evening (latest launch Sun 25 Oct) · plan pre-registered Fri 16 Oct (G1) · M0 inputs confirmed Mon 19 Oct (G2) · replay and parity module inside E11b by Tue 27 Oct (G4) · score module done Sun 1 Nov, 18:00 · final scoring Mon 2 Nov, 09:00 · parity read Mon 2 Nov, 18:00 · analysis Tue 3 Nov · writing Wed 4 – Fri 6 Nov · replication package Thu 12 Nov |
| **Budget** | **0 new GPU runs, 0u, 0 GPU-hours** (matches 01 §4.9 and R1). It re-scores the 55 A05 runs (275 fold-episodes, 3,290 nominal CPU model fits) and adds 20 new CPU OpenFE cells (Core-10 × {42, 4}); up to 8 more if the partition check or D4 forces it (≤ 28 cells). CPU: ≤ 16 CPUs at a time. **Developer time: 2.35 d in total**, of which 1.05 d is implementation (§4, "Budget arithmetic"). |
| **Models** | No LLM. Re-scoring evaluators: XGBoost-400 (Tester-identical), L2 logistic / ridge, random forest. The features come from A05's O120 runs (`openai/gpt-oss-120b`). |
| **Status** | Not started (today is Wed 7 Oct) |
| **Depends on** | E1 (`make_manifest.py`), E3 (per-iteration attribute snapshot, which **must be in `agent2027-v2.0`**), E6 (replay-validity check, augment seed marker), E8a (SYN-P, detector, φ in `synp_meta`), X3 and A08-T2 (`--fold-seed`, `--dump-features`; A22-T2 extends it), A08-T7 (launcher variables `BASE`, `DATASETS_FILE`, `FOLD_SEEDS`, `DUMP_FEATURES`), A08 CPU screen (SYN-P OpenFE dumps, brute-force labels), D4, X10, A05 (all 55 runs, plus A05-T16 `a05-handoff`), E11a (σ_d), E11b (`fetch_wandb.py`) |
| **Feeds into** | The validity sentence in the main text; appendix table "Evaluator transfer"; the threats-to-validity paragraph. If ED5 fires, the short paper instead. |

*Every "hoped" result is a hypothesis. Code references: RougeOne `nested-cv-rebuttal` @ `07cbf93`, re-checked against that commit's source on 2026-10-07 (the local checkout is at `1bdaf42`, which differs from `07cbf93` only in `classicfe_arm.py`: AutoFeat knobs, +8 lines after `:87`; every `classicfe_arm.py` number below is for `07cbf93`). `main.py` = `rogue_one_main.py`; `ext/`, `tst/agent.py` = `rogueone/llm/agents/core/{extractor,tester}/agent.py`; `metrics.py` = `rogueone/llm/agents/core/tester/metrics.py`; `config.py` = `rogueone/utils/config.py`; `pandas_exec.py` = `rogueone/utils/pandas_exec.py`; `classicfe_arm.py`, `classicfe_launch.sh` = `paper/scripts/classicfe_arm.py`, `paper/scripts/classicfe_launch.sh`. "01 §n" = `01_shared_infrastructure.md`. `<P>` = `<ROGUEONE_CAIR_PATH>`.*

---

## 1. What we test and why it matters

Rogue One picks features with one evaluator: the Tester's XGBoost-400, which both selects the iteration (inner CV) and reports held-out score. We take the feature set that inner CV selected in every A05 (M0) fold-episode and re-score it, on the same folds, with an L2 linear model and a random forest. We compare it with raw columns alone and with an equal-size OpenFE set. If the features help only the evaluator that chose them, "the agents build informative features" overstates what we can say. Reviewers at an SE venue are likely to ask for this check. The answer fills one appendix table and one sentence, and no confirmatory claim depends on it.

Scope follows the T1-minimum breakdown ("re-score M0's features"). The master design (§4, M3) says "every v2 run"; A22 narrows it to A05, and T15 optionally adds A16 and A12 runs.

## 2. Background you need

**The evaluator being transferred from** (`tst/agent.py`):
- **Parameters.** Regression (`:1667-1679`) and classification (`:1809-1821`) fits are built as `xgb.XGBRegressor(**base_params)` (`:1707`) and `xgb.XGBClassifier(**base_params)` (`:1852`) with `objective` `reg:squarederror` (R) or `binary:logistic` (C), `eval_metric` `rmse` / `logloss`, `tree_method=hist`, `device=cpu`, `n_estimators=400`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.9`, `colsample_bytree=0.9`, **`random_state=self.fold_seed`** (`:1671`, `:1813`) and `n_jobs=_XGB_N_JOBS` (`:71`; = min(`ROGUE_ONE_N_JOBS`, else `OMP_NUM_THREADS`, else `OPENBLAS_NUM_THREADS`, else 8; 8 at most; `metrics.py:18-40`). Any wrapper behaviour for multi-class labels is inherited by building the model the same way. The assessment/SHAP model is a separate 300-tree model (`:927`, `:956`).
- **Matrix:** `np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)` on train and test (R `:1694-1705`; C `:1839-1850`). Classification labels go through a `LabelEncoder` fitted on train + test labels (`:1855-1859`).
- **Classification:** balanced sample weights at fit (`:1861-1870`) and at scoring (`:1874-1878`) when `ROGUE_ONE_DO_BALANCING=True` (`config.py:312`, Full-v2 value `True`). The logged `accuracy` (`:1915-1919`) is therefore balanced accuracy. AUROC is binary (positive class = encoded 1), or one-vs-rest weighted for multi-class (`:1939-1985`).
- **Regression:** one fold per Tester call (`selected_test_fold_id = (step or 0) % len(x_folds)`, `:854`). `n_RMSE` = RMSE / mean of the **training-portion** target (`:352-361`; `mean_val = df_attributes["target"].mean()` at `:1788`, where `df_attributes` is the training portion built at `:863-873`). `RMSE` and `R2` are logged as well (`:416-420`). Regression fits use no sample weights.
- **Matrix order.** Columns follow attribute-table order (creation order) with `id` dropped (`:844-849`), filtered to the attributes that are not `Pruned` at Tester entry (`:885-896`; the test matrix gets the same column filter at `:902-906`). Training rows are the non-test folds stacked in fold order (`:863-868`), each fold frame in `data.csv` row order (`fold_safe_features.py:184`). Because XGBoost subsamples rows and columns, a parity replay must keep **both orders** exactly.
- **Pruning lag.** The scored matrix `X_train` is fixed at `:895`, before the LLM pruning stage; `prune_attributes_in_df` at `:2142-2144` runs on a frame that is no longer used for scoring. E2 (freeze item 3) rebuilds it after pruning. A22 always re-scores the set that was actually scored (the `Scored` mask of A22-T1).

**Selection and where the selected sets live** (`main.py`):
- Per fold, among `completed` trials, the one with the best `inner_cv_accuracy` / `inner_cv_RMSE` is selected (`:1078-1154`; ties go to the first; if all are NaN, the fold's last row is taken). Inner CV is 4 folds on training rows, seeded by the fold seed (`tst/agent.py:194-202`, called with `seed=self.fold_seed` at `:1727` R).
- The selected rows are written to `fold_aggregation_best_iterations.csv` (`:1353-1357`), with columns `id` (`Trial <index>`, index = fold·10 + iteration, `:803`), `test_fold_id`, `iteration_in_fold` (`:801-802`), `accuracy`/`AUROC` or `n_RMSE`/`RMSE`/`R2`. **The file exists only when all 5 folds have completed rows** (`:1091-1102`, `:1159-1160`): a run that aborts, and a smoke run with `ROGUE_ONE_SMOKE_FOLDS=1`, has none. Every trial of every fold is in `df_test_pool_final.csv` (`:1385-1398`).
- **`fold_aggregation_active_features.csv` cannot be used.** It is built from the final `df_attributes_explanations` (`:1177-1187`, `:1359-1363`), and that table is emptied at every fold start (`_reset_fold_state`, `:417-457`, called at `:543-544`). So it holds fold 4's pool only.
- The existing per-step dump `intermediate_steps/df_entities_attributes_trail_{step}.csv` (`:839-847`) holds globally computed values without status and is not fold-safe.
- **A22 therefore needs E3's per-iteration attribute snapshot** (A22-T1). The Tester mutates the attribute table in place (`tst/agent.py:1597-1611`, returned at `:2393`; `main.py` passes `self.df_attributes_explanations` itself at `:789-794`), so by `dump_data` (`:833`) the `Status` column already holds the **post-pruning** statuses. The scored set is the pre-call status, which is why the snapshot needs a `Scored` mask captured **before** `test_hypotheses`. Existing rows never change their `Pandas Command` (`update_attribute_pool` only updates `Status` and the prune metadata, `:334-361`), so a snapshot replays every earlier iteration of its fold.

**Fold-safe replay.**
- Each fold's matrices are rebuilt by replaying every attribute's `Pandas Command` on the train partition and on the test partition separately (`main.py:744-758`; `rogueone/utils/fold_safe_features.py:37-126`, `:129-187`). The replay covers **every row of the attribute table, including `Pruned` ones**, in creation order, because later commands may read earlier columns from `df_features`.
- Inputs: `df_raw` = `pd.read_csv(<task>/data.csv)` (`main.py:197`) and the labels from `<task>/labels.csv` (`train_test_splitter.py:345-349`, with the `patient_id` → `id` rename of `main.py:740-742`; labels de-duplicated by `id`, `fold_safe_features.py:179`).
- A failed replay is NaN-filled, not raised (`fold_safe_features.py:84-99`, `:117-124`). Commands run through `exec` with full builtins (`pandas_exec.py:139-143`). The A22 replay therefore runs with `--network none` and read-only mounts.

**Folds.**
- Outer folds come from `compute_entity_duplicate_groups` + `deterministic_kfold_indices` (`main.py:299-315`; `rogueone/utils/train_test_splitter.py:111`, `:199`), over the entity ids in first-appearance order (`main.py:200-202`, `:312-315`). They are stratified only when `ROGUE_ONE_STRATIFIED_FOLDS=True` and the task is classification; the stratification labels are built at `main.py:260-281`.
- If a dataset has exact-duplicate raw rows, a group splitter is used. `classicfe_arm.build_folds` (`:119-135`) is plain (Stratified)KFold with `random_state=42` hard-coded and no groups. On such a dataset the two partitions differ, so A22 passes Rogue One's own partition to OpenFE (A22-T7, T8).

**The comparator code (`classicfe_arm.py`), and why A22 reuses only `prep`.**
- `prep` (`:102-116`) ordinal-encodes object columns alphabetically over the full column domain. This is the same scheme as the Extractor's `_ordinal_encode_categorical` (`ext/agent.py:42-82`).
- `evaluate` (`:144-180`) is **not** Tester-identical:
  - `XGB_PARAMS` fixes `random_state=42` (`:89-99`);
  - the nRMSE denominator is the full-dataset mean (`:320`, `:379`).

  The audited card's line "reuses classicfe's evaluate, so scoring is identical by construction" is therefore corrected here. The OpenFE cells' own scores are used only for the reproduction check of A22-T9.
- OpenFE (`:183-229`) renames columns to `f{i}` (`:206`) and keeps every feature it returns (`:221-228`). Only scores are written (`:352-387`); X3 adds the feature dump.
- Image `classicfe:4` (`classicfe_launch.sh:21`): OpenFE 0.0.12, XGBoost 2.1.3, scikit-learn 1.5.2, pandas 2.2.3, numpy 1.26.4 (`docker/Dockerfile.classicfe:20-30`).
- **Launcher at `07cbf93`** (what A08-T7 and A22-T8 change): it rewrites `$BASE/datasets.txt` from a heredoc (`:45-73`), mounts `$BASE/classicfe_arm.py` (`:123`), names the container `classicfe_<ds>_<arm>_<folds>` without a seed (`:104`), writes `arm_<ds>_<arm>_<folds>.json` without a seed (`:117`, `:131`) and logs to `$BASE/run_<TAG>.log` (`:28`). It has no `DATASETS_FILE`, `FOLD_SEEDS` or `DUMP_FEATURES`. Each cell runs `--network none --cpus=4` (`:118`) and the log ends with `ALL_CELLS_DONE` (`:147`). Defaults: `PARALLEL=16`, `CELL_TIMEOUT=10800`.
- The client image copies the whole repo (`Dockerfile:13`; base `vllm/vllm-openai:v0.11.0`, `:3`).

**Fallback features.** The Extractor's deterministic fallback (`ext/agent.py:599-844`, triggered at `:996-1002`) can put non-LLM features into the selected set: restorations of pruned attributes, per-id means of raw numeric columns, and ordinal codes of categorical columns. Most of these are copies of raw columns. A22 keeps them in `ro`, as scored, and runs a fallback-excluded sensitivity analysis.

**Instruments** (from A05, n and p as in 01 §4.1):
- **Core-10 headroom stratum (h\* ≥ 1):** `balance-scale` (at ceiling), `junglechess`, `vehicle`, `housing`, `crab`, `wine`. **No-harm stratum (h\* < 1):** `heart`, `credit-g`, `myocardial`, `insurance`. h\* = (best matched FE − `base_no_fe`) / fold SD of `base_no_fe`, from the matched harness (master design, §1 and §2.2).
- **SYN-P 8:** `synp_p1`, `synp_p2`, `synp_p6a`, `synp_p6b` (R); `synp_p3`, `synp_p4`, `synp_p5`, `synp_p7` (C). φ and seeds are stored in `synp_meta/<id>.json`, which on CAIR lives off the mount in `~/rogueone-synp-meta/` (01 §3.3 E8a). A22 mounts it `:ro` into **analysis containers only** (never into a Rogue One run container).
- **`synp_p0`** (log x, R): seed 42 only.
- **Brute-force label:** a task is non-discriminating if OpenFE or AutoFeat recovers φ in ≥ 14/15 episodes (A08 screen, seeds {42, 4, 5}).
- **A05 rows:** A05-001…055 as in A05 §4.3; the smoke rows (`A05-901`, `A05-902`; `code_phase=smoke`) share run names with A05-004 and A05-033 and are excluded.

**Metrics.**
- `rec_sel` = an active, non-fallback feature in the inner-CV-selected iteration has |Spearman| ≥ 0.95 with φ on training rows (NMI ≥ 0.9 for P3, P5, P7); raw columns and single-column monotone transforms never count (01 §4.6). A22 does not recompute it: it reads the per-episode `rec_sel` from the episode table in artifact `a05-handoff` (A05-T16).
- `heldout_score` = balanced accuracy (C) or nRMSE (R); `heldout_auroc` (R13).
- σ_d (`sigma_d`) = the dataset's Full-v2 run-to-run SD from G2 (E11a).
- The raw block of §4 is called `B` here; it is unrelated to R13's `B_d`.

## 3. Hypotheses and interpretation (all are hypotheses)

| Outcome | Pattern | Meaning and reporting |
|---|---|---|
| **Hoped** | On discriminating SYN-P tasks, episodes with `rec_sel=1` show gains for all three evaluators, and the linear gain is at least the XGBoost gain (a linear model cannot build ratios or thresholds itself). On the headroom datasets the gains have the same sign as XGBoost's, and label B of §6 does not fire (`raw_ro` is not behind `raw_ofe_k`). | *"The selected features transfer. Re-scored with an L2 linear model and a random forest, they raise held-out score by X and Y on episodes where the planted feature was recovered, close Z% of each evaluator's oracle gap, and do not trail an equal-size OpenFE set."* |
| **Null** | No gain on the no-harm datasets. This is expected, since h\* < 1. | Report on SYN-P and the headroom datasets only: *"no evaluator-specific artefacts detected"*. Never "the features help every model". |
| **Floor** | `xgb400`'s own gain is ≤ 0 on the headroom datasets: the features do not help even the evaluator that selected them. The master plan already reports Rogue One below raw XGBoost on real data under the corrected protocol, so this is a live possibility. | There is nothing to transfer on real data. Say so, and rest the validity sentence on the SYN-P episodes with `rec_sel=1`. |
| **Opposite** | Gains appear for XGBoost only; or the linear model is hurt (extreme values, unstable ratios); or `raw_ofe_k` beats `raw_ro` for every evaluator. | A limitation in the threats paragraph: the features are partly evaluator-specific, or no better than brute force. First rule out the preprocessing with the no-winsorisation sensitivity analysis. No confirmatory claim changes; any wording about agents "understanding" the data is removed (the paper does not make that claim). |
| **P0 note** | Trees should show a gain of about 0 on `synp_p0`; a linear model may gain from a log feature. | This is expected and is not evidence of discovery. It is reported as a sanity row and is in neither S nor H. |

## 4. Design

**Arms.** No Rogue One arm and no new LLM runs. The comparison is between **feature sets** (6) and **evaluators** (3) on fixed A05 fold-episodes. **Unit = fold-episode** = (A05 run, test fold). All gains are computed per fold-episode and then averaged as stated in T14 and §6.

| Feature set | Content per fold-episode |
|---|---|
| `raw` | The raw block `B`. Trees: `prep()` ordinal encoding. Linear: the numeric columns as `prep` returns them, plus a one-hot encoding of the original object columns read from `data.csv` (levels from the training fold; unseen → all zeros), so that the linear baseline is not handicapped by ordinal codes. |
| `ro` | The exact Tester matrix at the selected trial (snapshot replay, `Scored` mask, Tester row and column order). Scored with `xgb400` only: this is the **parity set**. |
| `raw_ro` | `B` + the **live** non-seeded `ro` features (LLM-authored and fallback; rows seeded by E6's augment mode are dropped, because `B` already contains them; a feature is live if its replayed training column is neither all-NaN nor constant). |
| `raw_ofe_k` | `B` + the first k OpenFE features in OpenFE's rank order, from the same (dataset, fold seed, fold). k = the number of features in `raw_ro` beyond `B` (logged as `et_k`; the dead ones as `et_k_dead`). If k = 0 the episode has no control and is listed. If OpenFE returned fewer than k, all of them, with the count logged. |
| `raw_ofe_all` | `B` + every OpenFE feature (the matched-harness OpenFE arm). Secondary. |
| `raw_phi` | SYN-P only: `B` + φ computed from `synp_meta`, with the φ code E8a's detector uses (to verify its API in E8a; never re-implemented). This is the per-evaluator ceiling. |

**Evaluators (pre-registered; no tuning on test rows).**
- **`xgb400`:** the Tester's `base_params`, copied verbatim (`tst/agent.py:1667-1679`, `:1809-1821`), with `random_state=fold_seed`, balanced weights on C, the Tester's `nan_to_num`, and `n_jobs` set explicitly and logged (a different thread count can move the last digits of a `hist` fit; the parity tolerance covers that).
- **`linear`:**
  - preprocessing: ±inf → NaN; NaN → training-fold median; winsorise at the training-fold 0.5/99.5 percentiles; `StandardScaler` fit on train; drop columns that are constant on train;
  - C: `LogisticRegressionCV(Cs=logspace(-3,3,7), cv=StratifiedKFold(4, shuffle=True, random_state=fold_seed), penalty='l2', class_weight='balanced', scoring='balanced_accuracy', max_iter=5000)`;
  - R: `RidgeCV(alphas=logspace(-3,3,7))`.
- **`rf`:** `RandomForest{Classifier,Regressor}(n_estimators=500, random_state=fold_seed, n_jobs=4)`, with `class_weight='balanced'` on C and other sklearn defaults, on the Tester's `nan_to_num` matrix.

**Scores (oriented so that positive = better; every score is per fold-episode).**
- C: balanced accuracy (primary), computed as the Tester does (`accuracy_score` with balanced test weights), and AUROC (always reported alongside, using the Tester's formula).
- Core-10 R: relative nRMSE reduction = 1 − nRMSE(set) / nRMSE(`raw`), with nRMSE as the Tester computes it (RMSE / training-portion target mean), plus ΔR².
- SYN-P R: **ΔR²** is primary. E8a standardises SYN-P regression targets, so the mean is about 0 and the Tester's nRMSE is ill-conditioned there.
- Transfer gain: `et_gain_<e>` = score(`raw_ro`) − score(`raw`) (relative reduction on Core-10 R). Control contrast: `et_gain_vs_ofe_<e>` = score(`raw_ro`) − score(`raw_ofe_k`). Ceiling share on SYN-P: `et_gap_closed_<e>` = mean gain(`raw_ro`) / mean gain(`raw_phi`), a ratio of means over the task's episodes, reported only where the `raw_phi` mean gain is > 0 (otherwise "n/a").

**Parity gate (must pass before any transfer number is read).**
- For each fold-episode, `xgb400` on `ro` must reproduce the logged selected-trial metric (from `fold_aggregation_best_iterations.csv`):
  - C: |Δ balanced accuracy| ≤ 0.002 and |Δ AUROC| ≤ 0.002;
  - R: |ΔRMSE| / RMSE ≤ 0.5%.
- A deterministic replay should give about 0. A larger gap is a replay defect (F39 semantics or NaN-filled replays; cross-check with E6's replay-validity result), not noise.
- An episode that fails is excluded from every gain computation and counted. An episode whose logged metric is NaN (a fold that failed after its retry) is "not re-scorable": listed, not counted as a parity failure. A dataset with < 3/5 passing episodes in a run is excluded from that run's row and named.
- **Gate:** ≥ 90% of the re-scorable fold-episodes pass (denominator = fold-episodes of A05 runs that have a `fold_aggregation_best_iterations.csv`, minus NaN-logged ones).

**Run matrix** (no new GPU runs; "runs" here are A05 runs re-scored or CPU cells).

| Group | Source rows | Instruments × fold_seed / replicate | Runs | Fold-episodes | New CPU cells | Names |
|---|---|---|---|---|---|---|
| Core-10 primary | A05-001…020 | Core-10 × 42/0, 4/0 | 20 | 100 | 0 | `A05__full__<dataset_id>__f{42,4}__r0` |
| Core-10 A/A | A05-021…030 | Core-10 × 42/1 | 10 | 50 | 0 | `A05__full__<dataset_id>__f42__r1` |
| SYN-P primary | A05-031…046 | SYN-P 8 × 42/0, 4/0 | 16 | 80 | 0 | `A05__full__synp_<id>__f{42,4}__r0` |
| SYN-P A/A | A05-047…054 | SYN-P 8 × 42/1 | 8 | 40 | 0 | `A05__full__synp_<id>__f42__r1` |
| P0 | A05-055 | `synp_p0` × 42/0 | 1 | 5 | 0 | `A05__full__synp_p0__f42__r0` |
| **Total re-scored** | A05-001…055 | | **55** | **275** | | |
| OpenFE control, Core-10 | A22-001…020 | Core-10 × {42, 4}, `kfold` + `--folds-file` | — | — | **20** | `A22__openfe__<dataset_id>__f<seed>__r0` |
| OpenFE control, SYN-P | A08 screen (reused) | SYN-P 8 × {42, 4} + P0 × 42 | — | — | 0 (17 of A08's 27 OpenFE cells) | `A08__openfe__<dataset_id>__f<seed>__r0` |
| Contingency | A22-021…028 | `synp_p3/p4/p5/p7` × {42, 4}, partition-mismatched or D4-rejected | — | — | 0–8 | `A22__openfe__…` |
| Analysis | — | — | — | — | — | `ANALYSIS__evaluator_transfer__<YYYYMMDD>` |

Check: 20 + 10 + 16 + 8 + 1 = 55 runs; 55 × 5 = 275 fold-episodes; 20 + 8 = 28 contingency-inclusive A22 cells; 8 SYN-P tasks × 2 seeds + P0 = 17 A08 cells.

**Budget arithmetic.**
- GPU: 0 runs, 0u, 0 GPU-hours.
- Fits (nominal; `LogisticRegressionCV` and `RidgeCV` do their inner fits internally):
  - run-specific: 275 episodes × (1 parity + 3 evaluators × {`raw_ro`, `raw_ofe_k`}) = 275 × 7 = 1,925;
  - cached per (dataset, seed, fold), shared by all runs of that seed: Core-10 100 × 6 (`raw`, `raw_ofe_all` × 3 evaluators) = 600; SYN-P 8 80 × 9 (`raw`, `raw_ofe_all`, `raw_phi` × 3) = 720; P0 5 × 9 = 45;
  - total = 1,925 + 600 + 720 + 45 = **3,290 fits**, an estimated 1–3 h wall clock in one `--cpus 16` container (junglechess and housing dominate; re-measure on the T10 parity run and update this line).
- OpenFE: 20 cells at 4 CPUs, `PARALLEL=4` (16 CPUs). Worst case 20 × 4 CPUs × 6 h = 480 CPU-hours. The datasets file is ordered largest-first, so the four slow cells (junglechess and housing × 2 seeds; `myocardial` has p = 91) start first and the wall clock is about one cell timeout (≤ 6 h) plus a short tail. T0.4's cell JSONs carry `secs` per cell (to verify under `~/classicfe_v3/out/`) and calibrate this.
- Developer time, by task (d):

| Group | Tasks | d |
|---|---|---|
| Prerequisite checks and decisions | T1 0.1, T2 0.05, T3 0.05, T4 0.1, T6 0.05 | 0.35 |
| Pre-registration (inside X10's 0.5 d) | T5 | 0.1 |
| Implementation | T7 0.1, T8 0.05, T10 0.4 (inside E11b's 1.75–2 d), T11 0.5 | 1.05 |
| Operation | T9 0.15, T12 0.1, T13 0.05 | 0.3 |
| Analysis | T14 | 0.25 |
| Writing | T16 0.25, T17 0.05 | 0.3 |
| **Total** | (T15, optional, 0.25 d is not included) | **2.35** |

- **Difference from the breakdown.** The breakdown budgets "about 1 developer-day" for M3 (1.0 d from Nov 1, optional). That figure matches the implementation line here (1.05 d). The other 1.3 d (checks, run operation, analysis, writing) is counted here and not there. Of the 2.35 d, 0.5 d (T5, T10) is already inside X10's and E11b's allowances. About 0.65 d (T1–T4, T6–T9) falls in Mon 12 – Mon 19 Oct, which 01 §6.2 does not schedule for A22 and which sits inside the 18–20.5 d load against 15.5 working days (01 §3.1). Levers: T7–T9 can slide to Sat 24 – Sun 25 Oct (latest launch), T3, T4 and T6 are read-outs, and ED5 removes the lot.

**Pairing.** Every feature set and evaluator is fitted on the identical outer partition and row order. The r0/r1 A/A pair at seed 42 gives two independent feature sets on the same folds, i.e. a transfer-reproducibility estimate. There is no randomisation: the analysis is deterministic given the seeds.

**Failure handling.**
- A05 runs that are ITT failures (a crash after the retry, or a 6 h cap hit) have no selected set. They are listed as "not re-scorable", counted, and not imputed. A22 is conditional on completion; A05 owns ITT.
- An OpenFE cell that times out is retried once, alone, with `CELL_TIMEOUT=43200`. If it is still missing, `raw_ofe_*` is marked "control unavailable" for that (dataset, seed).
- An OpenFE dump whose test ids differ from Rogue One's fold (the partition cross-check of A22-T4) is not used; its cell goes to the contingency list.
- A22's own containers are deterministic: a crashed step is simply rerun.

## 5. Tasks

### 5.1 Prerequisites (verify before the step that needs them)

- [ ] **A22-T1** [type: data] (0.1 d, due Mon 12 Oct, 22:00, G0a; re-check Fri 16 Oct, 16:00 with A10-T18 on the v2.0 smoke) — Check that E3 writes a per-(fold, iteration) attribute snapshot `intermediate_steps/df_attributes_explanations_trail_<index>.csv`. It must:
  - be written in `dump_data` (`main.py:839-847`), i.e. after the Tester's pruning statuses are applied (`:795-798`);
  - contain **every** row of the attribute table, including `Pruned` ones, in creation order, with `Attribute`, `Pandas Command`, `Status` (post-pruning), the fallback flag, E6's seed marker and a **`Scored`** column. `Scored` = 1 iff the attribute was not `Pruned` in the table as it stood **before** `test_hypotheses` (`main.py:789`), i.e. the columns of the matrix at `tst/agent.py:885-896`. It must be copied before the call, because the Tester mutates the table in place. E6's spec (01 §3.3) does not name the seed marker; if E6 sets none, the snapshot adds a boolean `Seeded`;
  - travel in the `run-output` artifact.

  If it is missing, it must land before tag `agent2027-v2.0` (Fri 16 Oct). It is logging-only, so Full's behaviour is unchanged. A10-T5's v2.2 fallback lands after M0 and is too late for A22. A10-T18 checks only four fields (name, `Pandas Command`, `Status`, fallback flag), so this check is additional.
  *Done when:* the G0a smoke artifact has one snapshot per iteration with all the columns above, and a throwaway container replay of the smoke's inner-CV-selected trial (the best `inner_cv_*` row of the smoke fold in `df_test_pool_final.csv`; the smoke has no `fold_aggregation_best_iterations.csv`, §2) reproduces its logged `accuracy` or `n_RMSE` within the parity tolerance. The replay code is the first step of T10 pulled forward, not extra work.
- [ ] **A22-T2** [type: code] (0.05 d, due Mon 12 Oct, 18:00, with X3) — Extend X3's `--dump-features DIR` so that, next to A08-T2's training-row file `<dataset_id>_openfe_fold<k>.csv.gz` (host path `<OUT>/f<S>/features/`, A08-T7), it writes the test rows as `<dataset_id>_openfe_fold<k>_test.csv.gz` (`id` first), with the same columns in OpenFE's rank order (the p renamed raw columns first, then the generated features), and lists the formulas in the same order in the JSON. The fold index base (0 or 1) is whatever A08-T2 merges (to verify).
  *Done when:* on one SYN-P task, both files exist for every fold, the train/test `id` sets are disjoint and cover the data, and the column order matches the JSON. If this misses the Monday screen, the 17 SYN-P OpenFE cells are re-dumped under the contingency (A22-T9 step 7; CPU only).
- [ ] **A22-T3** [type: data] (0.05 d, due Tue 13 Oct, morning, M5S) — From the A08 screen, record the non-discriminating task list and confirm OpenFE dumps (train and test files) for SYN-P 8 × {42, 4} + P0 × 42.
  *Done when:* both are in `docs/agent2027/a22_notes.md` (committed to `agent2027/v2`; notes are kept in the repo, not on CAIR) and in W&B group `A08`.
- [ ] **A22-T4** [type: decision] (0.1 d, due Thu 15 Oct) — Read D4. Run `evaluator_transfer.py folds` (A22-T7) for SYN-P 8 with `--stratified-classification` set to D4's value, and compare each task × seed's test-id sets with the test ids in the A08 dumps (A22-T2). Any mismatch, and every `synp_p3/p4/p5/p7` × {42, 4} cell if D4 rejects stratification, is queued as a `kfold` contingency cell (A22-T9 step 7).
  *Done when:* the D4 value, the comparison table and the queued cell list (possibly empty) are in the pre-registration draft.
- [ ] **A22-T5** [type: writing] (0.1 d, due Fri 16 Oct, X10) — Pre-register §4 verbatim in `docs/agent2027/preregistration.md`: feature sets (including the k rule), evaluators and hyperparameters, preprocessing, scores, parity tolerance and gate, the §6 read-out rules (S, H, T, X, B, N and their denominators), the T14 bootstrap settings, and the ED5 trigger.
  *Done when:* it is inside tag `agent2027-prereg`.
- [ ] **A22-T6** [type: data] (0.05 d, due Mon 19 Oct, G2; recheck Tue 20 Oct for late rows) — In W&B group `A05` (project `RogueOne-AGENT2027`; keep `config.code_phase = v2`, `state = finished`; drop the smoke rows `A05-901/902`), count the finished runs (55 expected). Confirm that each has `run-output` with snapshots, `fold_aggregation_best_iterations.csv` and the E6 replay-validity result, and that `a05-handoff` (A05-T16) is posted.
  *Done when:* a table of re-scorable vs ITT-failed runs is in `docs/agent2027/a22_notes.md`.

### 5.2 Implementation (repo `agent2027/v2`; never touches Full-v2 code paths)

- [ ] **A22-T7** [type: code] (0.1 d, due Wed 14 Oct) — Add the subcommand `scripts/agent2027/analysis/evaluator_transfer.py folds --tasks DIR --datasets core10|synp8|<ids> --seeds S… --stratified-classification 0|1 --out DIR`. It reads `<tasks>/<task_type>/<dataset_id>/{data.csv,labels.csv}` and writes `folds_<dataset_id>_f<seed>.json` (`dataset_id`, `task_type`, `fold_seed`, `stratified`, and 5 test-id lists) by calling `compute_entity_duplicate_groups` + `deterministic_kfold_indices` exactly as `main.py:299-315` does, with the stratification labels built as at `main.py:260-281` (only if `--stratified-classification 1` and the task is classification; Core-10 always 0). `core10` = the R10 list.
  *Done when:* there is a file for each of Core-10 × {42, 4}, `data.csv` ids are asserted unique, and a test lists the datasets whose partition differs from classicfe's plain `KFold(5, shuffle, 42)` (expected: those with duplicate rows).
- [ ] **A22-T8** [type: code] (0.05 d, due Wed 14 Oct, on top of A08-T7) — Add `classicfe_arm.py --folds-file PATH`, which replaces `build_folds` (`:119-135`) by mapping the ids to positions through `data.csv`'s `id` and records `fold_scheme_effective="folds_file"`; the output file name stays `arm_<ds>_<arm>_<folds>.json`. Add `FOLDS_DIR` to `classicfe_launch.sh`: when set, mount it `:ro` at `/folds` and pass `--folds-file /folds/folds_<ds>_f<seed>.json` (the seed variable is the one A08-T7 introduces; to verify its name). Unset, behaviour is unchanged.
  *Done when:* `bash -n` passes; a `raw` cell on `heart` without `--folds-file` gives `acc_plain` equal (4 decimals) to the `value` of the row (heart, `base_no_fe`, `kfold`) in `paper/results_matched_harness_baselines.csv`; and with `--folds-file` for `heart` × 42 it gives the same value if `heart`'s partition equals plain KFold (otherwise the difference is explained by the T7 test).
- [ ] **A22-T10** [type: code] (0.4 d, due Tue 27 Oct, inside E11b) — Add the subcommands `fetch` (wraps `analysis/fetch_wandb.py`; its CLI is to verify) and `score --parity-only`:
  - `fetch` downloads, for each finished `code_phase=v2` run of group `A05`, only `fold_aggregation_best_iterations.csv`, `df_test_pool_final.csv`, the snapshots `intermediate_steps/df_attributes_explanations_trail_*.csv`, E6's replay-validity result and the artifact `a05-handoff`; never the traces or `df_entities_attributes_trail_*.csv`;
  - read the selected trials (`fold_aggregation_best_iterations.csv`);
  - load the matching snapshot;
  - replay with `build_fold_safe_entities_folds`, from `data.csv` and `labels.csv` as in §2;
  - rebuild the Tester matrices in Tester row and column order (the `Scored` mask);
  - fit `xgb400` and write `et_parity_delta` and `et_parity_ok`.

  A unit test asserts that `xgb400`'s params equal the source lines.
  *Done when:* parity runs end to end on the A05 Core-10 seed-42 runs, and the test passes.
- [ ] **A22-T11** [type: code] (0.5 d, due Sun 1 Nov, 18:00) — Add to `score` the feature sets, the `linear` and `rf` evaluators, the OpenFE join (by `id`, with k-capping), `raw_phi` from `synp_meta`, the `rec_sel` join from `a05-handoff`, and the sensitivity variants (no winsorisation; fallback-only runs excluded). The script prints `=== A22_SCORE_DONE rc=<rc> ===` at the end. Add `log`, which writes the ANALYSIS run and the 20 `A22__openfe__…` cell runs (reuse A08-T10's logger for the cell runs if its CLI fits; to verify).
  *Done when:* toy tests pass (an injected φ gives a positive gain for all three evaluators; a duplicated raw column gives a gain of about 0), and the long-format table has one row per (run, fold, evaluator, feature set).

### 5.3 Execution on CAIR

**Rules** (01 §5.1, plus the SSH and detach conventions of 01 §5.2).
1. **Containers only.** The host issues only `docker` commands and bash orchestration scripts in `~`, plus the allowed checks and setup (`nvidia-smi`, `docker ps`, `nproc`, `df`, `cat`, `tail`, `grep`, `mkdir`, `cp`, `git fetch`). Never host Python. A22 uses plain `docker run`; any `docker compose run` over SSH needs `-T`.
2. **GPU pre-flight.** A22 never requests a GPU (no `--gpus`). The pre-flight below still records `nvidia-smi`. If any step ever needed a GPU, 01 §5.2 step 1 applies: STOP and report if not enough GPUs are free.
3. **Never more than 4 of the 8 H100s** without Henrik's explicit approval in chat. Not applicable here (0 GPUs), and A22 must not touch them.
4. **W&B (entity `PlanBench`) is the system of record.** Projects and groups: `RogueOne-AGENT2027`, groups `A22` (cell runs) and `ANALYSIS`. Secrets come from `<CAIR_HOME>/code/PlanBench/.env` (`--env-file`). Directories under `~/ro27_a22/` are container scratch only; results are the W&B runs and artifacts.
5. **SSH and detached runs.** One-shot `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "…"`; monitors add `-o ServerAliveInterval=30`, and exit code 255 means the network dropped, not the job. Long jobs run with `nohup`, with the log in `~`.
- **Code:** commit and push locally (branch `agent2027/v2`; pre-freeze tip tagged `agent2027-pre.N`), then fetch over SSH with the exact command of 01 §5.2 step 0 (never `git pull`; the server's HTTPS origin has no credentials). Record the `HEAD: …` line it prints in `docs/agent2027/a22_notes.md`.
- **Path:** `<P>` = `<ROGUEONE_CAIR_PATH>` = `<CAIR_HOME>/code/RougeOne` (**inferred; verify** with `ls -d ~/code/RougeOne`).
- **Scripts:** analysis containers mount the checked-out `<P>/scripts/agent2027/analysis` read-only at `/a22`, so the script version is the pushed commit and no image rebuild is needed before the freeze. The `rogueone` package comes from the image.
- **Output ownership:** analysis containers run as `--user $(id -u):$(id -g) -e HOME=/tmp`, so that `~/ro27_a22/` is not root-owned.

- [ ] **A22-T9** [type: run] (0.15 d; target Wed 14 Oct, evening; latest launch Sun 25 Oct) — **Core-10 OpenFE re-run.**
  0. **Manifest:** `make_manifest.py --ablation A22` writes `manifests/agent2027/A22_M3.csv` (R7): rows A22-001…020, `arm_id=openfe`, `replicate=0`, Core-10 × `fold_seed` 42/4, `llm_seed`, `session_id`, `rung_*` and `port_*` empty, `image_tag=classicfe:4`, `env_overrides={}`, `cap_hours=6`, `expected_u=0`, `notes=kfold+folds-file` (to verify that `make_manifest.py` accepts A22). Commit, push, fetch on CAIR.
  1. **CPU pre-flight:**
     `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "nproc; cat /proc/loadavg; df -h ~ | tail -1; docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"`

     Proceed only if:
     - 1-min load + 16 ≤ nproc − 16 (our 16 CPUs leave 16 free);
     - no `classicfe_*` or `ro27-a22-*` container is running;
     - ≥ 20 GB disk is free.

     Otherwise STOP and report.
  2. **Folds:** `ssh … "mkdir -p ~/ro27_a22/folds && docker run --rm --name ro27-a22-folds --network none --cpus 4 --user \$(id -u):\$(id -g) -e HOME=/tmp --entrypoint python3 -v <P>/tasks:/tasks:ro -v <P>/scripts/agent2027/analysis:/a22:ro -v ~/ro27_a22/folds:/out rogue_one:agent2027-pre.N /a22/evaluator_transfer.py folds --tasks /tasks --datasets core10 --seeds 42 4 --stratified-classification 0 --out /out"`
  3. **Install** in a directory of its own, so that T0.4's `~/classicfe_v3/` (its `datasets.txt`, `classicfe_arm.py` and outputs) is never touched (A08-T7 made `BASE` overridable): `ssh … "mkdir -p ~/classicfe_a22 && cp <P>/paper/scripts/classicfe_arm.py ~/classicfe_a22/ && printf '%s\n' 'junglechess|classification|<P>/tasks' 'housing|regression|<P>/tasks' 'myocardial|classification|<P>/tasks' 'wine|regression|<P>/tasks' 'crab|regression|<P>/tasks' 'insurance|regression|<P>/tasks' 'credit-g|classification|<P>/tasks' 'heart|classification|<P>/tasks' 'vehicle|classification|<P>/tasks' 'balance-scale|classification|<P>/tasks' > ~/classicfe_a22/a22_datasets.txt"` (largest-first).
  4. **Launch:** `ssh … "cd ~/classicfe_a22 && BASE=\$HOME/classicfe_a22 DATASETS_FILE=\$HOME/classicfe_a22/a22_datasets.txt FOLD_SEEDS='42 4' ARMS=openfe DUMP_FEATURES=1 FOLDS_DIR=\$HOME/ro27_a22/folds TAG=a22_core10 OUT_SUB=out PARALLEL=4 CELL_TIMEOUT=21600 nohup bash <P>/paper/scripts/classicfe_launch.sh kfold > ~/ro27_a22_openfe.job.log 2>&1 & echo launched"` (20 jobs expected; check the first lines of the log).
  5. **Monitor** (as a background command; exit code 255 means reconnect): `ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 CAIR-gpu14 "while ! grep -q ALL_CELLS_DONE ~/classicfe_a22/run_a22_core10.log; do sleep 300; done; grep CELL_MISSING ~/classicfe_a22/run_a22_core10.log || echo none"`
  6. **Rerun** each `CELL_MISSING` cell alone with the same `BASE`, `DATASETS_FILE`, `ARMS`, `DUMP_FEATURES`, `FOLDS_DIR`, `OUT_SUB`, plus `ONLY=<id> FOLD_SEEDS=<s> CELL_TIMEOUT=43200 TAG=a22_retry`.
  7. **Contingency (only if A22-T4 queued cells):** write `folds` files for the queued SYN-P tasks with `--datasets <ids> --stratified-classification 0`, append their lines to a second datasets file, and run steps 1, 4, 5 and 6 for them (`TAG=a22_contingency`, same `BASE`, `OUT_SUB=out`). Latest launch Sun 25 Oct.

  *Done when:*
  - 20 cell JSONs (`~/classicfe_a22/out/f{42,4}/arm_<ds>_openfe_kfold.json`, `folds_done` = 5) and 200 `.csv.gz` dump files (20 cells × 5 folds × {train, test}) exist, or the missing cells are named;
  - for datasets whose partition equals classicfe's KFold (T7 test), the seed-42 `acc_plain` (C) / `nrmse` (R) of the new JSON is compared with the `value` of the row (dataset, `openfe`, `kfold`) in `paper/results_matched_harness_baselines.csv`; the comparison table is in `docs/agent2027/a22_notes.md`, and every dataset that differs by more than 0.002 is listed as "OpenFE not bit-reproducible" (OpenFE's own seed is unchanged; a miss goes to the threats paragraph and does not block);
  - the 10 Core-10 `data.csv` / `labels.csv` pairs under `<P>/tasks` have the same sha256 as the copies T0.4 used (`~/rogueone-runs-seed2/tasks`, `~/rogueone-runs-wave3/tasks`; checked in a container).
- [ ] **A22-T12** [type: run] (0.1 d; preliminary parity Tue 27 Oct; final launch Mon 2 Nov, 09:00, expected done by 13:00) — Use the latest `rogue_one:agent2027-v2.N`, with the analysis scripts mounted as above.
  0. **Version check:** in a container from `rogue_one:agent2027-v2.0` and from `rogue_one:agent2027-v2.N`, print `xgboost`, `pandas`, `numpy` and `scikit-learn` versions. They must be identical (the replay and the parity depend on them). If not, score in the v2.0 image (scripts mounted) instead. Record the versions in the notes and pass them to `log`.
  1. **Pre-flight** as in T9. T12 starts only when T9 and its retries are finished or killed.
  2. **Fetch** (network on):
     `ssh … "mkdir -p ~/ro27_a22/in && docker run --rm --name ro27-a22-fetch --cpus 4 --user \$(id -u):\$(id -g) -e HOME=/tmp -e WANDB_CACHE_DIR=/in/.wandb_cache --env-file <CAIR_HOME>/code/PlanBench/.env --entrypoint python3 -v <P>/scripts/agent2027/analysis:/a22:ro -v ~/ro27_a22/in:/in rogue_one:agent2027-v2.N /a22/evaluator_transfer.py fetch --group A05 --out /in"`
  3. **Score**, detached, with **`--network none`**:
     `ssh … "mkdir -p ~/ro27_a22/out && nohup docker run --rm --name ro27-a22-score --network none --cpus 16 --user \$(id -u):\$(id -g) -e HOME=/tmp --entrypoint python3 -v <P>/scripts/agent2027/analysis:/a22:ro -v ~/ro27_a22/in:/in:ro -v ~/ro27_a22/folds:/folds:ro -v ~/classicfe_a22/out:/ofe_core:ro -v ~/classicfe_a08/out:/ofe_synp:ro -v ~/rogueone-synp-meta:/meta:ro -v <P>/tasks:/tasks:ro -v ~/ro27_a22/out:/out rogue_one:agent2027-v2.N /a22/evaluator_transfer.py score --in /in --out /out > ~/ro27_a22_score.job.log 2>&1 & echo launched"` (`/ofe_core` holds the Core-10 cells and any SYN-P contingency cells, which take precedence over `/ofe_synp`, A08's `~/classicfe_a08/out/f<seed>/features/`; the preliminary run adds `--parity-only`).
  4. **Monitor:** `ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 CAIR-gpu14 "while ! grep -q A22_SCORE_DONE ~/ro27_a22_score.job.log; do sleep 120; done; tail -30 ~/ro27_a22_score.job.log"`
  5. **Log** (network on, `--env-file` as in step 2, scripts mounted, `~/ro27_a22/out` mounted `:ro` at `/out`): run `evaluator_transfer.py log --in /out --git-sha <HEAD from the code step> --image-tag agent2027-v2.N` in a container named `ro27-a22-log`.
  6. **Tear-down check:** `ssh … "docker ps --format '{{.Names}}' | grep -E '^(ro27-a22-|classicfe_)' || echo none"`.

  *Done when:* the run `ANALYSIS__evaluator_transfer__<YYYYMMDD>` (group `ANALYSIS`, project `RogueOne-AGENT2027`) holds the long table and the parity report, and the tear-down check prints `none`.

### 5.4 Analysis

- [ ] **A22-T13** [type: decision] (0.05 d, Mon 2 Nov, 18:00) — Read the **parity gate**.
  *Done when:* ≥ 90% of episodes pass and the excluded datasets are named; otherwise **ED5** is applied and recorded in the pre-registration log.
- [ ] **A22-T14** [type: analysis] (0.25 d, due Tue 3 Nov) — Compute:
  - per-dataset means and min–max over the 3 runs (f42 r0, f4 r0, f42 r1; 1 run for P0);
  - pooled medians of the per-dataset (Core-10 headroom) or per-task (SYN-P) means with 90% percentile bootstrap CIs, resampling datasets or discriminating tasks, 2,000 resamples, seed 2027 (descriptive; with ≤ 6 units the CIs are wide, so min–max is shown too);
  - SYN-P gains split by `rec_sel` (1 vs 0);
  - `et_gap_closed_<e>`;
  - for `xgb400` only, |gain| / σ_d as a noise yardstick;
  - the two sensitivity analyses.

  No p-values and no Holm. Assign the §6 label from the primary analysis (winsorised, fallback features kept).
  *Done when:* the table and label are logged to the ANALYSIS run.
- [ ] **A22-T15** [type: analysis] (0.25 d; optional; only if T14 is done by Tue 3 Nov, 12:00) — Exploratory: the same pipeline on A16 `set_phased` and A12 `fx_pajM` runs. These use the same datasets and seeds, so no new OpenFE cells are needed.
  *Done when:* it is logged as a separate ANALYSIS run marked exploratory.

### 5.5 Paper outputs

- [ ] **A22-T16** [type: writing] (0.25 d, Wed 4 – Fri 6 Nov) — Write the appendix table "Evaluator transfer":
  - rows: SYN-P discriminating tasks and Core-10 by stratum;
  - columns: gain per evaluator for `raw_ro` and `raw_ofe_k`, the SYN-P ceiling share, and the parity pass rate;
  - balanced accuracy and AUROC side by side.

  Add one main-text validity sentence (the §3 wording for the label) and one threats line (the evaluator-selection coupling; the replay exclusions).
  *Done when:* it is in the AGENT draft, or in the short paper under ED5.
- [ ] **A22-T17** [type: writing] (0.05 d, Thu 12 Nov) — Put `evaluator_transfer.py`, the folds files and the W&B references of the OpenFE dumps into the anonymised replication package.
  *Done when:* it is listed in the package manifest.

## 6. Decision rules and gates (pre-registered Fri 16 Oct; descriptive, outside Holm)

- **Parity gate:** see §4. If it is not met by **Mon 2 Nov, 18:00**, apply **ED5**. Only the parity result moves to the short paper.
- **ED5 trigger (proposed here; confirm in the pre-registration).** A22 moves to the short paper if any of these holds:
  - (a) ED1–ED4 have all fired and E11b is late at G4 (Tue 27 Oct);
  - (b) A22-T1's snapshot (with `Scored`) is missing from `agent2027-v2.0` at the Fri 16 Oct, 16:00 re-check;
  - (c) the parity gate fails.

  ED5 frees about 1 developer-day (the breakdown's figure) and costs 0u. By this document's count it frees ≈ 1.2 d if it fires at G4 (T11–T14, T16, T17; T15 excluded) and ≈ 0.6 d if it fires at the parity gate (T14, T16, T17).
- **Read-out label.** Definitions:
  - H = the 6 headroom datasets; S = SYN-P discriminating tasks with ≥ 3 parity-passing `rec_sel=1` episodes (`synp_p0` is in neither).
  - The **unit mean** of a dataset in H = the mean of the per-episode score over its parity-passing fold-episodes in all runs. The unit mean of a task in S = the same over its parity-passing `rec_sel=1` fold-episodes.
  - A dataset with "control unavailable" does not satisfy any B condition. If fewer than 4 of the 6 H datasets are usable (parity-passing, and control available for B), the label is "not evaluable" and the paper says so. If S is empty, T cannot be assigned.
  - **T (transfers):** for both `linear` and `rf`, the unit mean `et_gain` is > 0 on every task in S **and** ≥ 0 on ≥ 4 of the 6 in H.
  - **X (XGBoost-specific):** assigned only if T does not hold. Let U = the units in S ∪ H whose `xgb400` unit mean gain is > 0. X holds if |U| ≥ 3 and, for each of `linear` and `rf` separately, at least half of U have unit mean gain ≤ 0.
  - **B (no better than brute force):** for each of the three evaluators, the unit mean `et_gain_vs_ofe` is ≤ 0 on ≥ 4 of the 6 in H, **and** the median over S of the per-task unit means is ≤ 0.
  - **N:** none of T, X, B.
  - T or X may co-occur with B. The paper reports every label that holds, and the label comes from the primary analysis only; the sensitivity analyses are reported next to it.
- **Kill dates:** latest OpenFE launch Sun 25 Oct; parity Mon 2 Nov, 18:00; final table Tue 3 Nov.

## 7. Risks and cut levers

| Risk | Mitigation |
|---|---|
| Developer time (binding); about 0.65 d falls in Mon 12 – Mon 19 Oct, which 01 §6.2 does not schedule | ED5 is the pre-declared drop. Partial cuts in order: T15 → `raw_ofe_all` → `rf`. T7–T9 may slide to Sat 24 – Sun 25 Oct. |
| The snapshot is not in v2.0 or lacks `Scored` | Checked at G0a and again Fri 16 Oct, 16:00 (T1). Reconstructing from LLM traces is possible but costs about 1 d, so apply ED5 instead. |
| Replay defects (F39, NaN-filled commands) | The parity gate and E6's check exclude them. Exclusions are reported per dataset. Dead features are left out of `raw_ro` and of k. |
| Library drift between the v2.0 image and the scoring image | Version check in T12 step 0; fall back to scoring in the v2.0 image. |
| Partition mismatch (duplicate rows; classicfe's `random_state=42`; A08's stratified screen vs a non-stratified A05) | `--folds-file` built from Rogue One's own functions (T7, T8). The SYN-P dumps are cross-checked against the folds files (T4); mismatched cells are re-run (T9 step 7). |
| OpenFE timeouts (junglechess, housing) | 6 h cap, then a 12 h solo retry, then "control unavailable". |
| OpenFE not bit-reproducible | The T9 comparison with T0.4 lists the datasets; reported in the threats line, not blocking. |
| Launcher collisions (container names and JSON names carry no seed at `07cbf93`) | A08-T7 adds the seed to both; A22 uses its own `BASE` and never touches `~/classicfe_v3/`. |
| Linear model hurt by extreme values or ordinal codes | Winsorisation, one-hot raw block, and a no-winsorisation sensitivity analysis. |
| SYN-P nRMSE ill-conditioned | ΔR² is primary on SYN-P R. |
| Executing LLM-authored code | `--network none`, read-only mounts, non-root user. |
| CPU contention with GPU sessions | `--cpus ≤ 16`, CPU pre-flight, low-load windows. |
| Fishing | No p-values. The label rule and all hyperparameters are pre-registered. |

## 8. Definition of done

- [ ] T1–T6 are verified, with notes in `docs/agent2027/a22_notes.md`.
- [ ] The manifest `manifests/agent2027/A22_M3.csv` (20 CPU rows, plus up to 8 contingency rows; `image_tag` `classicfe:4`, `expected_u` 0) and the 20 OpenFE cells are in W&B group `A22`.
- [ ] `evaluator_transfer.py` (`folds`, `fetch`, `score`, `log`) and the `--folds-file` / `FOLDS_DIR` patch are merged, with tests.
- [ ] The parity report, the long table, the label and the sensitivity analyses are in the ANALYSIS run.
- [ ] The appendix table, validity sentence and threats line are in the draft (or in the short paper under ED5).
- [ ] The replication-package entries are complete.

---

**New names introduced** (not in the canonical registry):
- **Subcommands:** `evaluator_transfer.py folds | fetch | score [--parity-only] | log`, with the options `--datasets`, `--seeds`, `--stratified-classification`, `--git-sha`, `--image-tag`; the end marker `=== A22_SCORE_DONE rc=<rc> ===`.
- **Flags and environment:** the option `classicfe_arm.py --folds-file` and the launcher variable `FOLDS_DIR`.
- **Fold files:** `folds_<dataset_id>_f<seed>.json`.
- **Dump file:** `<dataset_id>_openfe_fold<k>_test.csv.gz` (next to A08-T2's `<dataset_id>_openfe_fold<k>.csv.gz`, in `<OUT>/f<S>/features/`).
- **E3 snapshot:** `intermediate_steps/df_attributes_explanations_trail_<index>.csv` (named first as A10's fallback, required here in v2.0) and its `Scored` column (plus `Seeded` if E6 sets no marker).
- **Evaluator IDs:** `xgb400`, `linear`, `rf`.
- **Feature-set IDs:** `raw`, `ro`, `raw_ro`, `raw_ofe_k`, `raw_ofe_all`, `raw_phi`.
- **Analysis columns:** `et_gain_<e>`, `et_gain_vs_ofe_<e>`, `et_gap_closed_<e>`, `et_parity_delta`, `et_parity_ok`, `et_k`, `et_k_dead`.
- **Labels:** T, X, B, N.
- **Manifest and W&B:** rows A22-001…028 with arm `openfe` (reused from A08; R5 has no A22 entry), and W&B runs `A22__openfe__<dataset_id>__f<seed>__r0`.
- **CAIR:** `~/ro27_a22/{folds,in,out}`, `~/classicfe_a22/` (`a22_datasets.txt`, `out/`), the job logs `~/ro27_a22_openfe.job.log` and `~/ro27_a22_score.job.log`, and the containers `ro27-a22-{folds,fetch,score,log}`.
- **Repo notes file:** `docs/agent2027/a22_notes.md`.
- **ED5 trigger:** the conditions in §6, proposed here.
