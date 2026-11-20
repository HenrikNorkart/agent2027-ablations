# A05 — M0 · Full-v2 reference and A/A noise floor

| Field | Value |
|---|---|
| **ID** | A05 · M0 · arm `full` |
| **Claim / Holm contrast** | None of its own; outside Holm. It is the Full-v2 control (same partition) for **C1a** (A16), **C1b** (A08), **C3** (A10) and **C5b** (A19, the all-O120 cell), and it supplies **C4b**'s data (A15). It is also: the Full anchor plotted beside A12's fraction (never in the C2a–c model) and Full's recovery in the TOST margin (with A09's oracle); the P7 web reference shown beside **C4a** (A14; not part of the contrast); the O120 anchor and RV1 reference in A17 (C5a itself is fitted on the Qwen rungs only); and the control for A07, A13 and A16's Core-10 no-harm check. |
| **Tier** | T1-minimum. **Never cut**: neither the core runs nor the SYN-P A/A. |
| **Owner** | Henrik |
| **Start / finish** | Freeze-day steps Fri 16 Oct 12:00–17:58 (timeline in §5.3) → launch **Fri 16 Oct 2026, 18:00** (G1). GPU done **Sun 18 Oct** (≈ 12:30–20:00, A07 included). **G2 read-out Mon 19 Oct, by 10:00.** Final statistics Sun 1 – Tue 3 Nov; paper outputs Wed 4 – Fri 6 Nov. |
| **Budget** | **55 runs · 47.5–55u** · 34–39 h of the 4-H100 slot (136–157 H100-h). A07 shares the session (+12 runs, 12–15u), so the session is 59.5–70u, ≈ 42.5–50 h. The freeze smoke (2 one-fold rows, ≈ 0.4u) belongs to the G1 freeze procedure (01 §2.2), not to A05. Developer time ≈ 3.3 d listed in §5 (A05-T2 sits inside X10 and A05-T14 inside E11a). |
| **Models** | O120 `openai/gpt-oss-120b` for S, E, T and W. Server `vllm/vllm-openai:v0.22.0`, port 8010 (8011 for the second server under D5 = TP2). Client image `rogue_one:agent2027-v2.0`. |
| **Status** | Not started |
| **Depends on** | E1–E6, E8a, E13 (or ED1), X1 (`~/ro27_vllm_session.sh` with `MAX_MODEL_LEN`), X3, X10. The G1 decisions: R from A02, the pool mode from A03, and D2, D4, D5, D7. For G2: E11a, A01 (σ_samp pair table, artifact `a01-results`) and the A08 CPU screen (SYN-P `base_no_fe`, brute-force labels). |
| **Feeds into** | G2, and A06–A23 (§4.5) |

*All "hoped" results are hypotheses, not findings. Shared details: `docs/ablations/01_shared_infrastructure.md` ("01 §n"). If this document and 01 disagree on a shared item (flags, fixed settings, metrics, CAIR rules), 01 wins; record the difference as a pre-registration amendment.*

## 1. What we test and why it matters

We run frozen Full-v2 once per instrument at fold seeds 42 and 4. At seed 42 we add an A/A replicate that differs only in the LLM sampling seed. This tells us three things:
- how well v2 does;
- how much identical runs move;
- which planted-feature (SYN-P) tasks sit strictly between 0% and 100% recovery.

Almost every later arm is compared with these runs. They also fix the G2 constants: σ_d, the recovery ICC, the no-harm (NI) margin, and B_d. G2 (Mon 19 Oct) is the last point at which a broken Full-v2 can still be rerun.

## 2. Background you need

**Units.** u = one real-data gpt-oss-120b run (10 iterations × 5 folds). A SYN-P run costs 0.7–1.0u. Throughput ≈ 2 u/h × 0.7 utilisation = 1.4 u/h on ≤ 4 H100.

**Code references.** RougeOne `nested-cv-rebuttal` @ `07cbf93` (`C:/code/PHD/RougeOne`). Short paths:
- `main.py` = `rogue_one_main.py`;
- `sci/agent.py`, `ext/agent.py`, `tst/agent.py` = `rogueone/llm/agents/core/{scientist,extractor,tester}/agent.py`;
- `kn/web_search.py` = `rogueone/llm/agents/core/knowledge/web_search.py`;
- `config.py` = `rogueone/utils/config.py`; `clients.py` = `rogueone/llm/clients.py`. There is no `cfg.py`.

**One run** is 5 outer folds × 10 iterations (`main.py:1420-1424`). Each trial is one `forward_pass` (`main.py:536`):
1. **Reset at fold start.** `_reset_fold_state` (`main.py:417-457`, called at `:543-544`) clears the pool, the focus history and the notebook, but not `df_test_pool`.
2. **Scientist (S) plans.** `determine_focus` (`main.py:548-555`) receives the global trial `index` (`:553`). With an empty focus history (iteration 0 of every fold) the prompt is "Round 1: …" (`sci/agent.py:475`); otherwise it is "Round {index}/{N}" (`:472-473`), so fold 2 shows "Round 11/10 … 19/10" until E2 fixes the counter. S retries up to 100 times (`sci/agent.py:519`), then falls back to a heuristic focus (`:501-517`).
3. **Extractor (E) builds features.** It gets the same message R times in one session (`ext/agent.py:965-971`). Today that is `range(10)` (`:967`); E2 replaces it with `ROGUE_ONE_EXTRACTOR_ROUNDS`. If nothing is accepted, the baseline fallback runs (`:996-997`).
4. **Tester (T) evaluates.**
   - It fits XGBoost with 400 trees (`tst/agent.py:1673` for R, `:1815` for C) and computes an inner-CV score (`_inner_cv_score`, `:194`).
   - It runs an assessment stage (`:1484`) and a pruning stage (`:1580`), then calls `prune_attributes_in_df` (`:2142`).
   - Both stages log tokens under one client name, `"TesterAgent"` (`:1341`). Stage shares therefore need E3's attribution.
5. **Feedback.** S sees only `train_*` metrics and the Tester report (`main.py:384-415`).
6. **Web search (W).** All roles call one `WebSearchAgent` (`sci/agent.py:149`, `ext/agent.py:261`, `tst/agent.py:1058`). Its tokens are logged under its own name (`kn/web_search.py:191`). E3/E4 charge them to the caller.

**Selection and success.**
- `compute_fold_aggregation` (`main.py:1074`) needs all 5 folds (`:1101-1102`). It selects the best inner-CV iteration per fold (`:1109-1154`).
- If the inner-CV column is missing, it **falls back to the held-out metric** (message at `:1125-1130`, text "falling back to the held-out metric"). In M0 that is a defect (§6).
- `run_experiments` swallows exceptions (`main.py:1589-1598`). **A run therefore counts as finished only if rc = 0 and `fold_aggregation_summary.csv` exists** (written at `:1350-1351`).
- **The opposite trap.** The image's default command is `python3 -u /workspace/rogue_one_main.py || tail -f /dev/null` (`Dockerfile:29`; the same string is the no-argument branch of `docker-entrypoint.sh:20-21`; any other command runs through `:24`). A `docker run` without an explicit command, as `paper/scripts/clients_t16_rerun.sh:91-110` does, turns a crash into a hang until the 6 h cap, which looks like a cap hit. `ro27_clients_manifest.sh` must pass an explicit command (checked in A05-T3).
- SYN-P tasks are not in the legacy list (`main.py:1536-1566`) and need `ROGUE_ONE_TASK_REGISTRY` (E1). Without it, `select_paths` (`:1568-1580`) selects nothing and the run exits 0 with no summary.

**Seeds, metric and environment files.**
- `ROGUE_ONE_FOLD_SEED` (`config.py:310`) fixes the partition, via `deterministic_kfold_indices` (`rogueone/utils/train_test_splitter.py:199`). It is stratified only if `ROGUE_ONE_STRATIFIED_FOLDS` (`config.py:323`) is set.
- `ROGUE_ONE_LLM_SEED` (`config.py:330`) is sent as `seed` on every request (`clients.py:211-213`). Runs are not reproducible under `PARALLEL=6` (Finding 36), so each replicate gets its own seed.
- `ROGUE_ONE_DO_BALANCING=True` (`config.py:312`) gives a balanced-weight fit (`tst/agent.py:1861-1870`) and balanced-weight scoring (`:1874-1878`, `:1915-1919`), i.e. balanced accuracy.
- t17 ran with `False`, because CAIR's `.env` sets it (`paper/scripts/classicfe_arm.py:34-38`): plain accuracy from an unweighted fit. So v2 classification is not on t17's scale. The matching `base_no_fe` field is `accw_balanced` (`classicfe_arm.py:374-377`).
- `config.py:23-25` loads `.env`, then `.dev.env` with `override=True`, then `.secret.env`. `.env` therefore loses to `docker -e` (so `v2_defaults.env` must set `ROGUE_ONE_DO_BALANCING=True` explicitly), while `.dev.env` beats `docker -e`.
- **At `07cbf93`, `.dev.env` is tracked in git and not excluded by `.dockerignore`.** It sets `LLM_PORT=8000`, `LLM_API_KEY` and `ROGUE_ONE_K_FOLDS=5`. E1 must delete it on branch `agent2027/v2` (or exclude it in `.dockerignore`); otherwise the build check in A05-T7 fails, and an image built anyway would point every client at port 8000. A05-T3 checks `flags/LLM_PORT` (OQ4).

**Instruments.**

| Set | dataset_id | Role in M0 |
|---|---|---|
| Core-10 | `balance-scale`, `junglechess`, `vehicle`, `heart`, `credit-g`, `myocardial` (C); `housing`, `insurance`, `crab`, `wine` (R) | No-harm reference; source of σ_d. t17 seed SDs, in that order: 0.000, 0.070, 0.029, 0.019, 0.0055, 0.0076, 0.021, 0.063, 0.024, 0.0010 (`paper/results_seed_sweep_t17.csv`, column `sd_across_seeds`, seeds 42/4/5). |
| SYN-P 8 | `synp_p1` x_a/x_b (R) · `synp_p2` (x_a/x_b)·x_c (R) · `synp_p3` 1[x_a/x_b ≥ t] ∧ cat=k (C) · `synp_p4` cyclic hour (C) · `synp_p5` sign(LW·LD² − RW·RD²) (C) · `synp_p6a` named composite (R) · `synp_p6b` its anonymised twin (R) · `synp_p7` rule found only in a local document (C) | Recovery reference, ICC, informative tasks. P6a/b supply C4b. P7 = **web reference and leak canary** (Full is expected at 0). |
| P0 | `synp_p0` log x_a (R), seed 42 only | Held-out negative control; `rec` = 0 by construction |

- SYN-P tasks have n = 2,000 and p ≈ 20–30. Their types are proposed; E8a confirms them in `synp_meta/registry.yaml`. Folders: `tasks/<task_type>/synp_<id>/`.
- φ lives off-mount in `~/rogueone-synp-meta/` and must never be inside the client image (A05-T7 check).

**Metrics** (W&B keys, 01 §4.6):
- **`rec_sel`** (primary): an active, non-fallback feature in the **inner-CV-selected** iteration has |Spearman| ≥ 0.95 with φ on training rows (NMI ≥ 0.9 for P3, P5 and P7). Raw columns and single-column monotone transforms never count. The unit is the fold-episode. Also logged: `rec_any`, `iter_first_rec`.
- **`success`**: the run finishes within 6 h and scores ≥ `base_no_fe` − 2 × s_{d,s} (for nRMSE: ≤ `base_no_fe` + 2 × s_{d,s}). On `junglechess` the score must also be ≥ 0.95, meaning the rule was found (OQ1).
  - `collapse` = not `success`.
  - `func_success` = `success` plus ≥ 1 accepted non-fallback feature in the selected iteration.
- **`heldout_score`**: the mean of the outer-fold scores at the selected iteration (balanced accuracy or nRMSE).
- **Cost:** `tok_compl_scientist`, `tok_compl_extractor`, `tok_compl_tester`, `tok_compl_total`, `llm_calls_total`, `wall_s`, `serper_calls`, `serper_ok_false_rate`, `energy_j`.
- **Funnel:** `fun_*`.

**Derived G2 quantities.** These formulas are *proposed* here and frozen in X10 (A05-T1, A05-T2). For each dataset d, let a = (f42, r0), b = (f42, r1), c = (f4, r0) be `heldout_score`.
- **s_{d,s} and s_d.** s_{d,s} = the fold SD of `base_no_fe` at fold seed s, from A05-T6: `accw_balanced_sd` (C) or `nrmse_sd` (R), the population SD over the 5 outer folds (`classicfe_arm.py:376-381`). `success` and the P0 check use the seed-matched s_{d,s}; the formulas below use s_d = mean(s_{d,42}, s_{d,4}).
- **Sign.** Δ = arm − Full on `heldout_score`, sign-flipped for nRMSE, so Δ > 0 always means the arm is better.
- **`sigma_AA`.** σ̂²_AA = mean_d[((a − b)/s_d)²] / 2 over Core-10 (10 df).
  - The partition variance is σ̂²_p = max(0, (mean_d[(((a + b)/2 − c)/s_d)²] − 1.5 σ̂²_AA) / 2).
  - The sampling share is Z = σ̂²_AA / (σ̂²_AA + σ̂²_p).
- **`sigma_d`.** The SD of {a, b, c} in raw units (2 df), floored at 0.1 × s_d.
  - **Homogeneity rule** (`housing`, `insurance`, `crab` only; classification datasets are never pooled because balancing differs from t17). F = σ̂²_AA,R / σ̂²_samp,R, where σ̂²_AA,R = mean over the 3 datasets of ((a − b)/s_d)² / 2 (3 df), and σ̂²_samp,R = mean over the 6 W0–t17 pairs on the same 3 datasets of (δ/s_d)² / 2 (6 df), from A01's per-pair δ table with A05's s_d in both. Pool iff F lies in the central 95% of F(3, 6) = [0.068, 6.60]. With k < 6 usable W0 pairs use F(3, k); with k = 0 do not pool. Also do not pool if A01-T18 reports a drift flag (mean shift and/or variance inflation, with its trigger) for the t17 baseline on these datasets: the t17 SD is then not a valid second estimator.
  - **If pooled:** σ²_d = (σ̂²_d + SD²_t17,d) / 2 (2 + 2 df; t17 SD from the CSV above), then the same floor.
- **`ni_margin`.** Δ_std = Δ / σ_d (`delta_std`). The proposed margin is M = 1.0, so M_d = σ_d (OQ2). The formula is frozen Fri 16; the value is computed at G2, before any arm's held-out score is read.
- **`icc_recovery`.** A logistic GLMM `rec_sel ~ 1 + (1|task) + (1|task:seed) + (1|run)`; ICC = σ²_task / (σ²_task + σ²_task:seed + σ²_run + π²/3).
  - Primary population: **SYN-P 7** (7 tasks × 3 runs × 5 folds = 105 episodes), because every contrast except C4a uses SYN-P 7 and P7 sits at 0 by design. SYN-P 8 (120 episodes) is reported as a sensitivity analysis.
  - If the fit is singular, use an ANOVA ICC on run-level proportions.
- **Informative task.** Full's `rec_sel` lies strictly between 0/15 and 15/15 (15 = 3 runs × 5 folds). This is for power accounting per contrast only, never for exclusion.
- **`B_d`.** The median `tok_compl_total` per iteration over all iterations of the dataset's 3 A05 runs (150 iterations). Also reported: role shares and Tester-stage shares, which are inputs to A12's token rule and to A19's cost threshold τ (A19 owns τ's definition; `g2_stats.py` computes it).
- **Sentinel task** (A06 §4.3 owns this rule; stated here verbatim in effect). Candidates are the SYN-P 7 tasks. Pick the task whose seed-42 `rec_sel` over r0 + r1 (10 episodes) is closest to 5/10. Ties: the 15-episode rate closest to 0.5, then the lower task index in the order `synp_p1`, `p2`, `p3`, `p4`, `p5`, `p6a`, `p6b`.
- **u_SYN-P.** The median SYN-P `wall_s` ÷ the median Core-10 `wall_s` in this session.

## 3. Hypotheses and interpretation (all hypotheses)

| Topic | Hoped | Null | Opposite |
|---|---|---|---|
| Run health | `success` ≥ 90% (≥ 50/55), 0 cap hits | `success` < 90% or cap hits present, but no G2 trigger with a fixable cause (§6): GO; failure and cap-hit rates are printed next to every number, and low `success` is a finding about v2 | A G2 trigger with a fixable cause: fix, re-freeze, rerun A05 + A07 (≈ 60–70u). Possible **only if caught at G2** |
| Informativeness | ≥ 4 of 8 tasks informative (in practice of the 7 non-P7 tasks); P7 at 0; P1 near 15/15 | Most tasks at the floor or ceiling: recompute power per contrast; "only large effects are detectable". Boundary tasks stay for contrasts that can move them (P7 for C4a; RT3 oracle, L0) | P7 `rec_any` > 0 (leak canary): investigate before A14 (Wed 21 Oct). P0 `rec_sel` > 0: detector bug, fixed in analysis, never a rerun |
| Clustering | ICC ≤ 0.5 | Fit singular or not estimable: ANOVA ICC on run proportions, reported as "clustering not estimable" | ICC > 0.5: effective n ≈ informative tasks × partitions; "inconclusive" expected and stated |
| Noise | Z < 0.5, so pairing by fold seed pays | Z ≥ 0.85 (≈ 1): pairing buys little; the paired analysis and the margin formula stay unchanged | F > 6.60 in the homogeneity rule: v2 is noisier than v1 on the R datasets, so nothing is pooled and σ_d comes from v2 alone (wider margins) |

Sentence the hoped result would allow: *"The frozen v2 recovers X of 8 planted features (Y% of fold-episodes); LLM sampling explains Z% of run-to-run variance, fixing the non-inferiority margin at M."*

## 4. Design

**4.1 Arm and fixed settings.** There is one arm, `full`: Full-v2 as frozen at G1. `env_overrides = {}` on every row except the SYN-P classification rows (`synp_p3`, `synp_p4`, `synp_p5`, `synp_p7`), which get `{"ROGUE_ONE_STRATIFIED_FOLDS":"True"}` if D4 is adopted. That override is part of the task definition. `replicate=0` is the primary run and `replicate=1` the A/A. There is no sentinel: M0 is the reference.

The settings below come from `scripts/agent2027/v2_defaults.env` (01 §4.3, R3/R4).

| Group | Values |
|---|---|
| Loop | `ROGUE_ONE_NUM_ITERATIONS=10` (code default 3, `config.py:305`), `ROGUE_ONE_K_FOLDS=5`, `ROGUE_ONE_SELECTION_SIGNAL=inner_cv`, `ROGUE_ONE_ENABLE_ROLLBACK=False`, `ROGUE_ONE_FOLD_RETRY_LIMIT=1`, `ROGUE_ONE_DO_BALANCING=True`, `ROGUE_ONE_STRATIFIED_FOLDS=False` (`True` on the D4 rows) |
| Run control | `ROGUE_ONE_START_VLLM=False` (default `True`, `main.py:1600`), `ROGUE_ONE_DO_LOGGING=True`, `ROGUE_ONE_DISABLE_WANDB=0`, `ROGUE_ONE_VERBOSE_AGENT_ERRORS=0`, `ROGUE_ONE_RETRIEVAL_TRANSCRIPT` unset, `CONSOLE_DISABLE=False`, `ROGUE_ONE_TASK_REGISTRY` (E1), `ROGUE_ONE_SMOKE_FOLDS` unset |
| Per row (launcher) | `ROGUE_ONE_DATASETS=<dataset_id>`, `ROGUE_ONE_FOLD_SEED`, `ROGUE_ONE_LLM_SEED` (§4.2), `LLM_MODEL=openai/gpt-oss-120b`, `LLM_PORT=8010` (8011 in `S1016b`), `LLM_API_KEY` = the server's `API_KEY` (default `rogueone-nestedcv-key`, `paper/scripts/vllm_session.sh:41`), `LLM_TEMPERATURE=0.7`; `ROGUE_ONE_SCIENTIST_MODEL` / `_PORT`, `ROGUE_ONE_EXTRACTOR_MODEL` / `_PORT`, `ROGUE_ONE_TESTER_MODEL` / `_PORT` empty; run-identity keys (R4) with `ROGUE_ONE_CODE_PHASE=v2` |
| Sampling | `SCIENTIST_AGENT_TEMPERATURE`, `EXTRACTOR_AGENT_TEMPERATURE`, `TESTER_AGENT_TEMPERATURE`, `MEDICAL_KNOWLEDGE_AGENT_TEMPERATURE` = 0.7; `EXTRACTOR_AGENT_MAX_TOKENS=32768`; `TESTER_AGENT_MAX_TOKENS=8192`; `SCIENTIST_AGENT_MAX_TOKENS` pinned at G1; `EXTRACTOR_AGENT_MAX_ITERATIONS` / `TESTER_…` / `SCIENTIST_…` = 100 / 100 / 200; `*_AGENT_REASONING_EFFORT` unset |
| R4 flags | `ROGUE_ONE_TOOL_ERROR_FEEDBACK=message`; `ROGUE_ONE_EXTRACTOR_ROUNDS`, `ROGUE_ONE_TOOL_OUTPUT_MAX_CHARS` and `ROGUE_ONE_FEATURE_POOL_MODE` as frozen at G1; `ROGUE_ONE_REASONING_EFFORT_PASSTHROUGH=0`; `ROGUE_ONE_LLM_TRACE=1`; `ROGUE_ONE_PROMPT_DUMP=1`; `ROGUE_ONE_REPLAY_VALIDITY_CHECK=1`; `ROGUE_ONE_SEARCH_CACHE=shared`; `ROGUE_ONE_SEARCH_CACHE_PATH=/workspace/cache/serper_cache.sqlite`; `ROGUE_ONE_SANDBOX=on` (`off` under ED1); `ROGUE_ONE_SCIENTIST_SIGNAL` only if D2 = L1 |
| Not in v2.0 | The switch flags of E7/E9 (`ROGUE_ONE_WEBSEARCH_ROUTING`, `ROGUE_ONE_KNOWLEDGE_BACKEND`, `ROGUE_ONE_TOPOLOGY`, `ROGUE_ONE_PLANNER_MODE`, `ROGUE_ONE_MEMORY_MODE`, `ROGUE_ONE_ASSESSMENT_MODE`, `ROGUE_ONE_PRUNING_MODE`, `ROGUE_ONE_PRUNING_SHADOW`, `ROGUE_ONE_RESTART_MODE`, `ROGUE_ONE_FEEDBACK_LEVEL`) arrive in v2.1–v2.4 as default-off switches. A05's code path is their Full-v2 default, guarded by the byte-identical prompt diff (01 §2.2). With uniform O120, search already uses the global model, so `caller` routing is a no-op for A05 |
| Infrastructure | D5 = TP4: one O120 server on 4 GPUs. D5 = TP2: two servers × 2 GPUs (`S1016a` on 8010, `S1016b` on 8011). `PARALLEL=6` per server, `CPUS=12`, `RUN_CAP_HOURS=6`, `RUNROOT=~/rogueone-runs-agent2027` |
| W&B | Entity `PlanBench`, project `RogueOne-AGENT2027`; group `A05`; job_type `full`; tags [`S1016a` (or `S1016b`), `v2`, `agent2027-v2.0`]; `WANDB_API_KEY` from `<CAIR_HOME>/code/PlanBench/.env`; `SERPER_API_KEY` from the image's `.env` / `.secret.env` |

**4.2 Seeds.** `fold_seed` ∈ {42, 4}. `llm_seed` = `fold_seed` for r0 and `fold_seed + 1000` = 1042 for the A/A.

**4.3 Run matrix.** Run name: `A05__full__<dataset_id>__f<fold_seed>__r<replicate>`. Manifest: `manifests/agent2027/A05_M0.csv`. Within each block, rows follow the dataset order of §2 (Core-10: `balance-scale` … `wine`; SYN-P 8: `synp_p1` … `synp_p7`), so for example A05-001 = `balance-scale` f42 r0 and A05-038 = `synp_p7` f42 r0.

| Rows | Instrument | fold_seed | replicate | llm_seed | Runs | u |
|---|---|---|---|---|---|---|
| A05-001…010 | Core-10 | 42 | 0 | 42 | 10 | 10 |
| A05-011…020 | Core-10 | 4 | 0 | 4 | 10 | 10 |
| A05-021…030 | Core-10 (A/A) | 42 | 1 | 1042 | 10 | 10 |
| A05-031…038 | SYN-P 8 | 42 | 0 | 42 | 8 | 5.6–8 |
| A05-039…046 | SYN-P 8 | 4 | 0 | 4 | 8 | 5.6–8 |
| A05-047…054 | SYN-P 8 (A/A) | 42 | 1 | 1042 | 8 | 5.6–8 |
| A05-055 | `synp_p0` | 42 | 0 | 42 | 1 | 0.7–1.0 |
| **Total** | | | | | **55** | **47.5–55** |
| A05-901, A05-902 (freeze smoke, outside the total) | `synp_p3`, `heart` | 42 | 0 | 42 | 2 × 1 fold | ≈ 0.4 |

The smoke rows carry `env_overrides = {"ROGUE_ONE_SMOKE_FOLDS":"1"}` (plus the D4 override on `synp_p3`), `notes=smoke`, and run with `ROGUE_ONE_CODE_PHASE=smoke` in session `S1016s`. They share run names with A05-004 (`heart`) and A05-033 (`synp_p3`) and are excluded from every analysis by `code_phase`.

**4.4 Budget.**
- Units: 30 Core-10 runs × 1u + 25 SYN-P runs × 0.7–1.0u = 30 + 17.5–25 = **47.5–55u**. This equals the T1-minimum breakdown (§2 table and the M0 card) and 01 R1.
- The master plan's M0 had 3 seeds (≈ 62–72u). T1-minimum drops seed 5 (breakdown §5, "Extra seeds"), so σ_d has 2 df per dataset.
- Wall clock: ÷ 1.4 u/h = 34–39 h, × 4 GPUs = 136–157 H100-h. Under TP2 the 1.4 u/h figure is an assumption until D5's throughput measurement replaces it.
- With A07 the session is 59.5–70u = 42.5–50 h, so a Fri 18:00 launch ends Sun 12:30–20:00 (the breakdown's GPU plan rounds this to 43–50 h).
- A run takes about 3 h at `PARALLEL=6`, so the 6 h cap is about twice that.
- Manifest `expected_u`: 1.0 (Core-10) and 0.85 (SYN-P) give 51.25u planned.

**4.5 Pairing, order, reuse.**
- Later arms pair with A05 by fold seed, so they share its partition.
- A05 and A07 rows are interleaved in random `order_key` order (RNG seeded by the session ID). They share the server, the load and the Serper cache.
- The A/A shares the cache, so σ_AA is a lower bound for live search (disclosed).
- **Consumers** (they never re-run these runs):

| Consumer | Uses from A05 |
|---|---|
| A06 | sentinel task and band inputs (SYN-P A/A pairs A05-031…038 vs A05-047…054) |
| A07 | HC1-6 × {42, 4} r0 |
| A08 | SYN-P 7, f42 r0 + r1 and f4 r0 |
| A09 | control and f42 focus logs |
| A10, A11 | SYN-P 7 × {42, 4} |
| A12 | Full anchor, f42 r0 + r1 (plots only) |
| A13 | Core-10 f42 mean of r0 + r1 |
| A14 | P7 |
| A15 | P6a/P6b × {f42 r0, f42 r1, f4 r0}, 15 pairs |
| A16 | SYN-P 7 × {42, 4} and Core-10 × 42 |
| A17, A19, A20 | O120 cells |
| A18 | 800 trace prompts (S 200, E 300, T 200, W 100) |
| A21–A23 | Full reference |

**4.6 Failure handling (ITT).**
- **Crash** (rc ≠ 0 or no summary): retried once, alone, at the end of the session.
- **Cap hit:** `docker stop`, logged as `CAP_HIT`, never retried.
- **Persistent failures and cap hits are ITT failures** (`itt_imputed`).
  - `heldout_score` is imputed at `base_no_fe` if augment is frozen. Otherwise it is imputed at a constant predictor on the same outer folds: the majority class (balanced accuracy = 1/K for K classes) or the training-fold mean (its nRMSE); `g2_stats.py` computes these.
  - All 5 episodes get `rec_sel = rec_any = 0`.
  - They are never dropped.
- **Infrastructure faults are not ITT** (proposed for X10). These are vLLM death, a host reboot, or a W&B or Serper outage, visible in the job or vLLM log. They are re-queued in session `S1018a` with `notes=infra-rerun`.
  - **Late rows.** If re-queued rows have not finished by Mon 19 Oct, 06:00, G2 is computed on the finished rows. The constants are frozen at that computation; late rows enter descriptive tables only.
- **Sensitivity analysis:** exclude runs whose selected features are all fallback features.

## 5. Tasks

Every task: owner Henrik. Dates are 2026. Effort in developer-days (d).

### 5.1 Prerequisites (verify before launch)

- [ ] **A05-T1** [type: decision] (0.1 d, due Thu 15 Oct) — Confirm `ROGUE_ONE_DO_BALANCING=True` and its consequences (OQ1, including the `junglechess` 0.95 threshold on balanced accuracy). Also confirm the margin constant (OQ2), the infra-fault and late-row rules (§4.6), the G2 triggers (§6) and the §2 formulas. *Done when:* each item is written in the draft `docs/agent2027/preregistration.md`.
- [ ] **A05-T2** [type: writing] (0.25 d, due Fri 16 Oct 12:00, inside X10) — Commit to the pre-registration the A05 formulas, the homogeneity rule, the sentinel rule (A06 §4.3 wording), the G2 criteria (§6), the ITT, infra and late-row rules, and u_SYN-P. *Done when:* tag `agent2027-prereg` is pushed before A05-001 starts.
- [ ] **A05-T3** [type: config] (0.25 d, due Fri 16 Oct 17:00) — Check the freeze smoke (session `S1016s`, rows A05-901/902, launched in A05-T7):
  - **E1:** R12 run name and config keys present; `flags/*` = `v2_defaults.env`; `flags/ROGUE_ONE_DO_BALANCING=True`, `flags/ROGUE_ONE_NUM_ITERATIONS=10` and `flags/LLM_PORT=8010` (a stray `.dev.env` would show 8000); a SYN-P row resolves through `ROGUE_ONE_TASK_REGISTRY`.
  - **E2:** fold 1's Scientist prompts read "Round 1" … "Round 10/10" (the smoke has one fold; fold 2 is re-checked in A05-T10); exactly R `Runner.run` calls per iteration; bounded tool output.
  - **E3:** non-zero `tok_compl_*` for S, E and T, with search charged to the caller; Tester-stage tokens separable (else A12 uses the joint A−J− token rule); trace count = `llm_calls_total`; focus JSONL; per-inner-fold scores; energy run `S1016s__energy`; `run-output`.
  - **E4:** a cache hit; Extractor searches attributed to E.
  - **E5:** the smoke's prompt dump is byte-identical (`scripts/agent2027/prompt_diff.py`) to `prompt-baseline-v2.0`; if the baseline was taken on a pre-tag image, re-take it from this smoke (OQ5). **E6:** pool mode as frozen; replay check logged. **E13:** on, or ED1 recorded.
  - **Launcher:** `ro27_clients_manifest.sh` passes an explicit container command (§2), so a crash exits instead of hanging.

  *Done when:* all are ticked in the session plan. Any failure blocks the launch.
- [ ] **A05-T4** [type: data] (0.1 d, due Fri 16 Oct 16:00) — Check data, disk and quotas. Host commands are queries only; anything that reads files runs in a container:
  - `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "ls -d ~/code/RougeOne ~/rogueone-serper-cache ~/rogueone-synp-meta && df -h ~ && ls <ROGUEONE_CAIR_PATH>/tasks/classification <ROGUEONE_CAIR_PATH>/tasks/regression"` — the path exists (OQ3), and `tasks/` holds Core-10 plus the 9 `synp_*` folders;
  - `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker run --rm -v <ROGUEONE_CAIR_PATH>/tasks:/t:ro -v ~/rogueone-serper-cache:/c --entrypoint sh rogue_one:agent2027-v2.0 -c 'ls /t/*/synp_*/ | grep -vE \"^(config.yml|data.csv|labels.csv)?$|:$\" | wc -l; sha256sum /t/regression/synp_p6a/labels.csv /t/regression/synp_p6b/labels.csv; touch /c/.w && rm /c/.w && echo cache-writable'"` — no file other than `config.yml`, `data.csv`, `labels.csv` in any `synp_*` folder (count 0), P6a/P6b label hashes equal, cache writable;
  - the E8a detector unit tests pass inside `rogue_one:agent2027-v2.0` (test path per E8a; to verify);
  - Henrik reads the Serper dashboard: credits ≥ 70,000 queries (67 runs + retries × ≤ 1,000 queries per run, 01 E4);
  - free disk on `~` ≥ 67 × 5 × (one smoke row's `output/` size) × 1.5 (a full run has 5 folds).

  *Done when:* every item passes and the output is pasted into the session plan.

### 5.2 Implementation (no code change)

- [ ] **A05-T5** [type: config] (0.1 d, due Fri 16 Oct 12:00, so the freeze tag contains the manifests) — Run `scripts/agent2027/make_manifest.py --ablation A05` locally to write the 55 rows of §4.3 plus A05-901/902. Settings: `image_tag=agent2027-v2.0`, `rung_*=O120`, `port_*=8010`, `cap_hours=6`, `expected_u` 1.0 (real) or 0.85 (SYN-P).
  - Then run `make_manifest.py --plan-session S1016a --rows <A05-001 … A05-055, A07-001 … A07-012>` to write `manifests/agent2027/sessions/S1016a.csv` (67 rows; 63 if CL3 cuts A07), and `--plan-session S1016s --rows A05-901 A05-902`.
  - **Under D5 = TP2**, split pair-aware into `S1016a` (port 8010) and `S1016b` (port 8011, `port_*=8011`): each A07 row goes into the session of its A05 control (same dataset and fold seed, r0; A07-T10 checks this); the remaining rows alternate so the two files differ by ≤ 1 row and hold similar Core-10/SYN-P mixes.

  *Done when:* all files are committed and pushed; the run names of the 55 `v2` rows are unique; the `llm_seed` rule holds; `order_key` is a permutation in each session file.
- [ ] **A05-T6** [type: data] (0.25 d, due Sat 17 Oct, 18:00; CPU, containers only) — Build the `base_no_fe` reference for `success`, ITT and s_{d,s}.
  - **Core-10 × {42, 4}** (20 cells, run one at a time so the CPU cap stays ≤ 16): for each (dataset, task type, seed), `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker run --rm --name classicfe_a05_<ds>_f<s> --network none --cpus=4 -e NJ=4 -e OPENFE_NJ=1 -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e NUMEXPR_NUM_THREADS=4 -v <ROGUEONE_CAIR_PATH>/tasks:/tasks:ro -v <ROGUEONE_CAIR_PATH>/paper/scripts/classicfe_arm.py:/arm.py:ro -v ~/a05_base_no_fe/f<s>:/out classicfe:4 python /arm.py <ds> <task> raw /tasks /out --folds kfold --fold-seed <s>"`. Wrap the 20 cells in one `nohup` loop with its log in `~` (e.g. `~/a05_base_no_fe.job.log`).
  - Why not `classicfe_launch.sh`: it runs `PARALLEL=16` cells × `--cpus=4` (64 CPUs), reads the arm from `~/classicfe_v3/`, and writes `arm_<ds>_<arm>_<folds>.json` without the seed, so seeds 42 and 4 would overwrite each other. `--fold-seed` comes from X3 at the freeze commit; `classicfe:4` must exist (`docker image inspect classicfe:4`).
  - Read `accw_balanced` / `accw_balanced_sd` (C) or `nrmse` / `nrmse_sd` (R).
  - **SYN-P × {42, 4}:** take the raw-arm cells from the A08 CPU screen, after checking that the screen used the current task hashes and the same fold scheme (stratified on C tasks under D4). Otherwise re-run them with the same command (`--folds stratified` on C tasks under D4).
  - Log the JSONs as W&B artifact `base-no-fe-ref` from a container (run `ANALYSIS__base_no_fe__20261017`, entity `PlanBench`, project `RogueOne-AGENT2027`, group `ANALYSIS`).

  *Done when:* `base-no-fe-ref` holds the mean and fold SD per (dataset, seed) for Core-10 and SYN-P 8 + P0.

### 5.3 Execution on CAIR

**Standing rules** (01 §5.1):
- **Containers only.** Every run, analysis, check and W&B query runs inside Docker. The host issues only `docker` / `docker compose` commands (any `docker compose run` over SSH needs `-T`), the `~` launchers, and queries (`nvidia-smi`, `docker ps`, `tail`, `grep`, `ls`, `df`). Never `python` on the host.
- **GPU pre-flight** (01 §5.2 step 1) before every GPU container. If not enough GPUs are free, **stop and report** the per-GPU memory and utilisation, the compute apps and the running containers, and let Henrik decide. Launch with explicit `GPUS=<ids>` and `WAIT_MAX_MIN=0`.
- **≤ 4 of the 8 H100s** without Henrik's explicit approval in chat. Under TP2 the two servers together use 4.
- **W&B is the record.** Runs are stateless and upload `run-output`; secrets come from `.env` files, never from the command line or a manifest.
- **SSH:** `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "<one-shot command>"`. Long jobs are detached with `nohup`, and their logs go in `~`, never in the root-owned run folders. Code reaches CAIR only by commit + push locally, then `git fetch` on CAIR (01 §5.2 step 0).
- **No `git fetch` / checkout in `<ROGUEONE_CAIR_PATH>` while `S1016a`/`S1016b`/`S1018a` run**, because the launcher reads `v2_defaults.env` and the task folders from it.

**Freeze-day timeline (Fri 16 Oct).**

| Time | Task | Step |
|---|---|---|
| by 12:00 | A05-T2 (in X10), A05-T5 | pre-registration and manifests committed; tags `agent2027-v2.0-freeze` and `agent2027-prereg` pushed |
| 12:30–14:00 | A05-T7 | sync, build `rogue_one:agent2027-v2.0`, image checks, install launchers |
| 14:00–15:30 | A05-T7 | pre-flight, then freeze smoke `S1016s` |
| by 16:00 | A05-T4 | data, disk and cache checks |
| by 17:00 | A05-T3 | smoke checklist ticked |
| 17:50–17:58 | A05-T8 | GPU pre-flight, immediately before launch |
| 18:00 | A05-T9 | launch `S1016a` (and `S1016b` under TP2) |
| ≈ 23:00 | A05-T10 | first sanity check |

- [ ] **A05-T7** [type: run] (0.25 d, due Fri 16 Oct 15:30) — Sync, build, install and smoke.
  1. Sync: `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15' git fetch --tags git@github.com:HenrikNorkart/RougeOne.git agent2027/v2 && git checkout -B agent2027/v2 FETCH_HEAD && git log -1 --format='HEAD: %h %s' && git describe --tags"`. If checkout refuses because of local changes (for example a locally edited `.dev.env`), stop and ask Henrik; never discard them.
  2. Build: `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && test ! -e .dev.env && docker build -t rogue_one:agent2027-v2.0 . && docker image inspect rogue_one:agent2027-v2.0 --format '{{.Id}}'"`. Record the image ID.
  3. Image checks: `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker run --rm --entrypoint sh rogue_one:agent2027-v2.0 -c 'ls -d /workspace/.dev.env /workspace/synp_meta /workspace/tasks/*/synp_*/*.json 2>/dev/null | wc -l'"` must print 0 (no `.dev.env`, no φ metadata in the image).
  4. Install: 01 §5.2 step 3 (copies `ro27_session_launch.sh`, `ro27_clients_manifest.sh`, `ladder_session.sh` and `ro27_vllm_session.sh` to `~`, and the manifests to `~/ro27_manifests/`).
  5. Smoke: pre-flight (A05-T8 command), then `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "SESSION_ID=S1016s SESSION_FILE=~/ro27_manifests/sessions/S1016s.csv GPUS=<ids> MODEL=openai/gpt-oss-120b PORT=8010 SERVER_NAME=rogueone-vllm-o120-s1016s LOG=~/vllm_session_S1016s.log IMAGE_RO=rogue_one:agent2027-v2.0 PARALLEL=6 CPUS=12 RUN_CAP_HOURS=6 RUNROOT=~/rogueone-runs-agent2027 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_S1016s.job.log 2>&1 & echo launched"`, with 4 GPU ids (TP4) or 2 (TP2).
  - `<ROGUEONE_CAIR_PATH>` = `<CAIR_HOME>/code/RougeOne` is **inferred; verify it** (OQ3).

  *Done when:* HEAD is `agent2027-v2.0-freeze` (or its diff from the tag touches only `manifests/`), the image ID is logged, the image check prints 0, and `~/ro27_S1016s.job.log` ends with `RO27_SESSION_END S1016s rc=0`.
- [ ] **A05-T8** [type: run] (0.02 d, Fri 16 Oct 17:50–17:58) — GPU pre-flight, no more than about 5 minutes before A05-T9, because explicit `GPUS` skips `vllm_session.sh`'s own free-GPU wait (`:132-154`): `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader && echo '--- compute apps ---' && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader && echo '--- containers ---' && docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'"`.
  - Proceed only if 4 GPUs show < 2,000 MiB used and none of our `ro27-*`, `rogueone-vllm-*`, `llmfe-*` or `caafe-*` containers is running.
  - **Otherwise STOP and report** the per-GPU memory and utilisation, the apps and the containers. Launch at the first clean pre-flight (§6, late start).

  *Done when:* the GPU ids are recorded in the session plan.
- [ ] **A05-T9** [type: run] (0.05 d, Fri 16 Oct 18:00) — Launch detached: `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "SESSION_ID=S1016a SESSION_FILE=~/ro27_manifests/sessions/S1016a.csv GPUS=<ids> MODEL=openai/gpt-oss-120b PORT=8010 SERVER_NAME=rogueone-vllm-o120-s1016a LOG=~/vllm_session_S1016a.log IMAGE_RO=rogue_one:agent2027-v2.0 PARALLEL=6 CPUS=12 RUN_CAP_HOURS=6 RUNROOT=~/rogueone-runs-agent2027 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_S1016a.job.log 2>&1 & echo launched; sleep 5; tail -5 ~/ro27_S1016a.job.log"`.
  - **Under TP2**, run this with 2 GPU ids, then a second job with the other 2 ids and `SESSION_ID=S1016b`, `SESSION_FILE=~/ro27_manifests/sessions/S1016b.csv`, `PORT=8011`, `SERVER_NAME=rogueone-vllm-o120-s1016b`, `LOG=~/vllm_session_S1016b.log`, `ROGUEONE_ALLOW_SECOND_SERVER=1` and job log `~/ro27_S1016b.job.log`. The total stays at 4 GPUs.

  *Done when:* the job log shows vLLM 0.22.0 asserted and both image IDs; the vLLM log (`~/vllm_session_S1016a.log`) shows "vLLM healthy"; `docker ps` shows `ro27-s1016a-energy` and the first `ro27-s1016a-a05-*` / `ro27-s1016a-a07-*` containers.
- [ ] **A05-T10** [type: run] (0.1 d, Fri 16 Oct ≈ 23:00, once ≥ 6 rows have finished) — Sanity check with a W&B query in a container: `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker run --rm --env-file <CAIR_HOME>/code/PlanBench/.env --entrypoint python3 rogue_one:agent2027-v2.0 -c \"import wandb; api=wandb.Api(); [print(r.name, r.state, r.config.get('flags/LLM_PORT'), r.summary.get('tok_compl_scientist'), r.summary.get('tok_compl_extractor'), r.summary.get('tok_compl_tester'), r.summary.get('llm_calls_total')) for r in api.runs('PlanBench/RogueOne-AGENT2027', filters={'group': 'A05'})]\""`. Look for:
  - `OK` lines (rc 0 + summary) for the finished rows;
  - non-zero `tok_compl_*` for every role, and `flags/LLM_PORT` = the row's port;
  - no "falling back to the held-out metric" line in the run's console log (W&B `output.log` or the launcher's per-row log under `~/rogueone-runs-agent2027/S1016a/<row_id>/`);
  - fold 2's Scientist prompts in `run-output` read "Round 1" … "Round 10/10", never above 10;
  - the replay check logged; `run-output` present.

  *Done when:* everything is clean; otherwise apply early abort (§6).
- [ ] **A05-T11** [type: run] (0.05 d per check; Sat 17 – Sun 18 Oct, twice a day) — Monitor as a background command: `ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 CAIR-gpu14 "while ! grep -q 'RO27_SESSION_END' ~/ro27_S1016a.job.log 2>/dev/null; do sleep 120; done; tail -30 ~/ro27_S1016a.job.log"`. Exit 255 means the network dropped, not the job: reconnect.
  - Per-row `OK`/`FAILED`/`CAP_HIT` lines come from `ro27_clients_manifest.sh`, which runs under `vllm_session.sh`'s `exec >> "$LOG"` (`paper/scripts/vllm_session.sh:75`). Unless E1 tees them into the job log, read them with `grep -E 'OK|FAILED|CAP_HIT' ~/vllm_session_S1016a.log | tail -20`.
  - Cross-check W&B states; a `running` run with no live container is a zombie.

  *Done when:* each check is logged in the session plan.
- [ ] **A05-T12** [type: run] (0.1 d, Sun 18 Oct, evening) — After `RO27_SESSION_END`:
  - tear-down check: `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker ps --format '{{.Names}}' | grep -E '^(ro27-|rogueone-vllm-)' || echo none; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"` shows `none` and free GPUs;
  - confirm 55 A05 runs with `run-output`;
  - list the ITT failures and cap hits;
  - re-queue infra faults as `S1018a` after a fresh pre-flight. `S1018a` runs before A17's first small-rung night (`S1018b`–`e`), which starts only after it ends (one job at a time, ≤ 4 GPUs). The late-row rule (§4.6) applies.

  *Done when:* the completeness table is logged to W&B (run `ANALYSIS__a05_completeness__20261018`, group `ANALYSIS`).

### 5.4 Analysis (containers only)

Analysis runs on CAIR after `RO27_SESSION_END` (and after `S1018a`, if any). First sync `<ROGUEONE_CAIR_PATH>` to the latest `agent2027/v2` (01 §5.2 step 0), then mount the analysis code read-only into the frozen image, for example: `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker run --rm --cpus 16 --env-file <CAIR_HOME>/code/PlanBench/.env -v <ROGUEONE_CAIR_PATH>/scripts/agent2027:/workspace/scripts/agent2027:ro -v ~/rogueone-synp-meta:/meta:ro -v ~/ro27_analysis/A05:/out --entrypoint python3 rogue_one:agent2027-v2.0 scripts/agent2027/analysis/<script>.py <args>"`. Script arguments are defined by E8a/E11a/E11b (to verify). Every analysis run logs to W&B as `ANALYSIS__<script>__<YYYYMMDD>`, with the git SHA in its config.

- [ ] **A05-T13** [type: analysis] (0.2 d, due Mon 19 Oct, 08:00) — Fetch and validate:
  - fetch group `A05` with `scripts/agent2027/analysis/fetch_wandb.py`. That script belongs to E11b (due Tue 27 Oct), so its fetch step must be pulled forward into E11a; otherwise use the 01 §5.2 step 7 query and download `run-output` in the same container;
  - run `scripts/agent2027/analysis/recovery_detector.py` with `~/rogueone-synp-meta/` mounted read-only, giving 125 episode rows (SYN-P 8: 120; P0: 5);
  - check inner-CV selection on 55/55 runs;
  - count the replay flags;
  - P0: `rec_sel` = 0, and `heldout_score` within 2 × s_{d,42} of `base_no_fe`;
  - **P7 canary**: any P7 `rec_any` > 0 is investigated before A14 (Wed 21 Oct);
  - `paper/scripts/scan_leakage.py` counts;
  - `serper_ok_false_rate` < 5%.

  *Done when:* the table is in the G2 memo.
- [ ] **A05-T14** [type: analysis] (0.1 d to run; the build is E11a's 0.5 d; due Mon 19 Oct, 08:00) — Run `scripts/agent2027/analysis/g2_stats.py` to compute every §2 derived quantity, the role and Tester-stage token shares (inputs to A12 and A19), A19's τ, A06's band inputs and the ITT constant-predictor values. *Done when:* `ANALYSIS__g2_stats__20261019` exists before any arm launches and before any A13 score is read.
- [ ] **A05-T15** [type: decision] (0.1 d, Mon 19 Oct, by 10:00) — Apply G2 (§6). *Done when:* the decision is in the memo and A09 is released or held.
- [ ] **A05-T16** [type: analysis] (0.1 d, Mon 19 Oct) — Post artifact `a05-handoff`: the episode table (A06, A15), the f42 SYN-P focus logs (A09), and the trace index (A18). *Done when:* the consuming tasks can fetch it.
- [ ] **A05-T17** [type: analysis] (0.25 d; preliminary Tue 27 Oct, final Sun 1 – Tue 3 Nov) — Build the descriptive profile with `scripts/agent2027/analysis/funnel.py` and `scripts/agent2027/analysis/process_metrics.py`: the funnel, errors per accepted feature, fallback share, cost per role, energy, ITT and cap rates, and `delta_std` of `heldout_score` vs `base_no_fe` per dataset (no test). *Done when:* the tables are logged to W&B group `ANALYSIS`.

### 5.5 Paper outputs

- [ ] **A05-T18** [type: writing] (0.25 d; draft Fri 16 Oct, numbers Sun 1 – Tue 3 Nov) — Write the protocol paragraph: the Full-v2 definition, σ, ICC and margin, and the §3 sentence hedged to the outcome. *Done when:* it is in the AGENT draft by Fri 6 Nov.
- [ ] **A05-T19** [type: writing] (0.5 d, Wed 4 – Fri 6 Nov) — Produce:
  - Table 1: Full `rec_sel` per SYN-P task;
  - Table 2: the Full reference row;
  - Fig. 2: Full anytime and Pareto points (A21, A23);
  - Fig. 3: the O120 point;
  - an appendix with the σ_d table, Z and ICC, the A/A scatter, B_d, the P0/P7 controls, the failure rates and the disclosures (cache, 2 df, balancing vs t17, M1/M4 seen before pre-registration).

  *Done when:* each item cites A05 run names.

## 6. Decision rules and gates

- **Launch (G1, Fri 16 Oct, 18:00).** Launch only when A05-T1 to T5 and T7 to T8 are done, `agent2027-prereg` is pushed, and the smoke is clean.
  - **Late start.** If the box is busy, launch at the first clean pre-flight; G2 and A09–A11 then slip by the same amount. If `S1016a` has not launched by **Sat 17 Oct, 06:00**, A07-T14 removes the A07 rows so A05 alone (34–39 h) still ends Sun 16:00–21:00.
- **Early abort** (checked at A05-T10 and A05-T11). Stop the session at once if any of these holds:
  - a role with zero `tok_compl_*` in a finished run;
  - missing traces, per-inner-fold scores or `run-output` in a run that did not crash;
  - any inner-CV fallback line;
  - the same exception (same type and same innermost Rogue One code location in the traceback) in ≥ 2 rows.

  **Procedure:** `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "pkill -TERM -f ro27_clients_manifest.sh; docker ps --format '{{.Names}}' | grep -E '^ro27-s1016[ab]-' | xargs -r docker stop; sleep 60; docker ps --format '{{.Names}}' | grep -E '^(ro27-|rogueone-vllm-)' || echo none"`. This stops only our own launcher and containers, and the clients script must be stopped first so it does not start its retry pass. If a `rogueone-vllm-o120-s1016*` server remains, `docker rm -f` it. Then fix the defect, re-freeze (image `rogue_one:agent2027-v2.0r1`, tag `agent2027-v2.0-freeze-r1`, dated pre-registration amendment) and restart all A05/A07 rows in a new session `S<MMDD><letter>`. The aborted runs stay in W&B under image tag `agent2027-v2.0` and are excluded from analysis.
- **G2, Mon 19 Oct (hard, pre-registered).** Counts are over A05's 55 runs; A07's rows are reported beside them.
  - **NO-GO, unconditional** (instrumentation or selection defects):
    - any inner-CV fallback;
    - any role with zero `tok_compl_*`;
    - missing traces, per-inner-fold scores or `run-output` in any run that did not crash.
  - **NO-GO, conditional:** ≥ 3 cap hits, ≥ 6 ITT failures, or the same exception in ≥ 2 runs, **and** the cause is a fixable defect in Full-v2. "Fixable defect" means a cause in Rogue One code, config or launcher that is identified from logs or traces by 10:00 and has a fix of ≤ 0.5 d that leaves every G1 decision (R, pool mode, D2) unchanged. If no such cause is found by 10:00, the trigger is a finding: GO.
  - **On NO-GO:** fix, re-freeze as `agent2027-v2.0r1`, and rerun A05 + A07 (≈ 60–70u from the reserve) while every arm waits. After G2, such a fix no longer fits and becomes a limitation.
  - **GO** otherwise. A design-caused low `success` is a finding. A detector bug (P0 `rec` > 0) is fixed in analysis and never triggers a rerun.
- **Wording-only outcomes:** < 4 informative tasks, ICC > 0.5, Z ≥ 0.85. **SYN-P is never retuned after M0.**
- **Frozen at G2:** M_d, σ_d, the homogeneity verdict, the ICC, the informative-task list, B_d, τ and the sentinel task.
- **Kill date:** none.

## 7. Risks and cut levers

| Risk | Response |
|---|---|
| A post-freeze defect | Early abort and G2. The ≈ 86u reserve covers one rerun, before G2 only. |
| A crash disguised as a cap hit (default command, `Dockerfile:29`) | A05-T3. If it is seen, re-classify the run as a crash and retry it. |
| `.dev.env` baked into the image (tracked at `07cbf93`; `LLM_PORT=8000` overrides `-e`) | E1 removes it; A05-T7 build check and image check; A05-T3 and A05-T10 check `flags/LLM_PORT` |
| φ metadata baked into the image (`COPY . .` in `Dockerfile`) | A05-T7 image check; E8a keeps `synp_meta` out of the repo tree on CAIR |
| SYN-P at the high end of the cost range | The session ends ≈ Sun 20:00; the first small-rung night (A06/A17) starts later. |
| TP2 throughput unknown (D5) | The end-time estimate uses 1.4 u/h until D5's W1 measurement replaces it |
| Serper outage or credits run out | A05-T4. Re-queue as infra, not ITT. |
| σ_d has 2 df; pooling covers 3 R datasets only | The 0.1 × s_d floor; disclosed. |
| E13 or E4 slips | ED1 (`ROGUE_ONE_SANDBOX=off`). For E4, wait (the cache is part of Full). |

**Cut levers.** CL1–CL7 never touch A05. The core runs and the SYN-P A/A, which is Full's second seed-42 replicate for C1b and C4b, are on the never-cut list.

**Open questions.**
- **OQ1.** The registry freezes `ROGUE_ONE_DO_BALANCING=True`, but t17/W0 ran with `False`. Henrik should confirm, accepting that classification σ is not pooled. The `junglechess` "rule found" threshold (0.95) was set on plain accuracy; confirm it on balanced accuracy in X10.
- **OQ2.** The margin constant (M = 1.0 Δ_std is proposed).
- **OQ3.** `<ROGUEONE_CAIR_PATH>` is unverified.
- **OQ4.** With `.dev.env` tracked and `LLM_PORT=8000` in it, how did t17's clients reach port 8010? Possibly the CAIR copy is edited locally. Check `git status --short .dev.env` on CAIR before E1's deletion lands, because a local edit blocks `git checkout -B`.
- **OQ5.** 01 §2.2 takes the rendered-prompt baseline on Fri morning, before `rogue_one:agent2027-v2.0` is built in the afternoon. A05-T3 compares the v2.0 smoke's dump with it and re-takes the baseline from the v2.0 smoke if they differ.

## 8. Definition of done

- [ ] 55 A05 runs exist in `PlanBench/RogueOne-AGENT2027` (group `A05`), each with `run-output` and full config keys. The ITT and cap-hit list is published.
- [ ] `base-no-fe-ref` is posted. The 125-row episode table exists, P0 = 0, and the P7 canary has been checked.
- [ ] The G2 memo (§2 quantities plus the GO / NO-GO decision) was logged before any arm launched. `a05-handoff` is posted.
- [ ] A05-T17 to T19 are done by Fri 6 Nov.

---

**New names introduced** (not in the registry):
- W&B artifacts `base-no-fe-ref` and `a05-handoff`; analysis runs `ANALYSIS__base_no_fe__20261017` and `ANALYSIS__a05_completeness__20261018`;
- `notes=infra-rerun`, `notes=smoke`;
- smoke rows `A05-901`, `A05-902`;
- sessions `S1016s` (freeze smoke), `S1016b` (TP2) and `S1018a` (infra re-queue), all following the R8 pattern;
- re-freeze image `rogue_one:agent2027-v2.0r1` and tag `agent2027-v2.0-freeze-r1`;
- host folders `~/a05_base_no_fe/` and `~/ro27_analysis/A05/`; container name `classicfe_a05_<ds>_f<s>`;
- the notation s_{d,s}, s_d, Z, F and u_SYN-P;
- OQ1–OQ5.
- **Deviation from R7:** `env_overrides` carries the R3 flag `ROGUE_ONE_STRATIFIED_FOLDS` on the D4 rows.

**Verification notes** (all read-only at `07cbf93` unless stated):
- Branch HEAD `1bdaf42` differs from `07cbf93` only in `paper/` (scripts and data). In `classicfe_arm.py`, lines after 85 shift by +8, and `results_seed_sweep_t17.csv` first appears at `1bdaf42`. The t17 seed SDs in §2 match its `sd_across_seeds` column.
- Confirmed: every `main.py`, `sci/`, `ext/`, `tst/`, `kn/web_search.py`, `config.py`, `clients.py` and `train_test_splitter.py` line cited in §2. Corrections made here:
  - the balanced-weight fit spans `tst/agent.py:1861-1870`;
  - the hang comes from the `Dockerfile:29` default command as well as `docker-entrypoint.sh:20-21`;
  - `sci/agent.py:475` renders "Round 1:" at each fold's first iteration;
  - `classicfe_arm.py:34-38` says the t17 setting came from CAIR's `.env`.
- New findings for E1 and E3:
  - `.dev.env` is tracked and not in `.dockerignore`;
  - the default-command fallback (`Dockerfile:29`, `docker-entrypoint.sh:21`);
  - the shared Tester token bucket (`tst/agent.py:1341`);
  - the caller-streaming branch does not record tokens (`clients.py:527-530`);
  - client output lands in the vLLM log (`vllm_session.sh:75`);
  - `classicfe_launch.sh` uses 64 CPUs and writes seed-less filenames.
- **Cross-document mismatches (not editable here):**
  - A17-T4 cites A05's early sanity check as A05-T12; it is A05-T10.
  - A06 (Depends on, A06-T5) cites `a05-handoff` as A05-T19; it is A05-T16.
  - Resolved in the cross-document pass: A01-T18 now hands `results_w0_t17rep_pairs.csv` (per-pair raw δ), the regression-only pooled SD with df, and the drift verdict to A05-T2 / A05-T14. A05's s_d (balanced-accuracy / raw-unit fold SDs from A05-T6) is applied to the raw δ; A01's own pooled values use the `value_sd` column of `paper/results_matched_harness_baselines.csv`. Both are reported; A05's is the one used in the rule.
