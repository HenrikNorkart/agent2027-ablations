# A01 · W0 — Old-image t17 replicate (LLM-sampling variance estimate)

| Field | Value |
|---|---|
| **ID** | A01 · W0 · arm `t17rep` |
| **Claim / Holm contrast** | None. This is a foundation measurement. It supplies `sigma_samp` (v1) for the M1 no-harm rule (A03), and the v1 split of run-to-run variance into sampling and partition parts (A05 pooling rule, protocol section). It is descriptive and outside Holm. |
| **Tier** | T1-minimum. It runs outside the Oct 13–31 window and outside the CL1–CL7 cut order, but it is **not** on the breakdown's "Never cut" list. Its own §6 kill dates and fallback (A03 uses the t17 SD) apply. |
| **Owner** | Henrik |
| **Start / finish** | Pre-flight and D9 **Wed 7 Oct** → seed-42 session `S1007a` (Wed 7, evening) → seed-4 session `S1008a` (Thu 8) → reruns Fri 9 – Sun 11 → **GPU hard stop Mon 12 Oct, 08:00** → σ̂ table to A03 **Mon 12 Oct, 18:00** → pre-registration entry Fri 16 Oct → pooling inputs to A05 / E11a before G2 (Mon 19 Oct, 08:00) → paper text Nov 4–6 |
| **Budget** | 16 runs (8 datasets × fold seeds {42, 4}) = **16u**, or 20u with the optional `wine` and `balance-scale` rows. All of it is **outside** the Oct 13–31 window. At 1.4 u/h that is ≈ 11.4 h (≈ 46 H100-hours at TP4); 20u is ≈ 14.3 h (≈ 57 H100-hours). Plan for 12–16 h of wall clock (48–64 H100-hours), because each 8-dataset wave has a tail (§4). Developer time ≈ 6.5–7.5 h before Mon 12 Oct: T1–T8 ≈ 1.4 h, T11–T15 ≈ 1.3–2.3 h (the upper figure includes one rerun session), and T9, T10, T16–T18 ≈ 3.8 h (manifest, analysis code, collection and analysis). Writing later (T19, T20) ≈ 2.8 h. The breakdown budgets only ≈ 2 h for W0, so the difference (≈ 4.5–5.5 h) is this document's analysis work; see R11. About 8,000–16,000 Serper queries (*estimate*: 500–1,000 per run, from the master design's risk table; not measured). |
| **Models** | O120 = `openai/gpt-oss-120b` for S, E, T and W (uniform), TP4 on 4 H100, port 8010. Server: the **vLLM image ID that t17's job log recorded** (`vllm/vllm-openai`, 0.22.0). Client image: `rogue_one:nestedcv9`. |
| **Status** | Not started |
| **Depends on** | No engineering task (E1–E13 are not needed). It does need: D9 (Wed 7 Oct); `rogue_one:nestedcv9` and t17's vLLM image still present on CAIR; 4 free H100s; unchanged gpt-oss-120b weights in the HF cache **and** on the Hub `main`; Serper credits; the t17 run dirs; commits `07cbf93` and `1bdaf42` of `nested-cv-rebuttal` pushed to GitHub (**not yet the case on 2026-10-07**: `git ls-remote` shows the remote branch at `2cd6d0a`, and no `agent2027/v2` branch; A01-T4 pushes `nested-cv-rebuttal`, A01-T9 creates `agent2027/v2`). |
| **Feeds into** | A03-T3 σ̂ table (Mon 12 Oct; the M1 rule is committed at G-E6, Tue 13 Oct); A05-T2 (homogeneity rule in the pre-registration, Fri 16 Oct) and A05-T14 / E11a `analysis/g2_stats.py` (G2), which both need the per-pair δ table (A01-T18); A02 `W_ref` fallback (median `wall_s`); protocol and threats-to-validity text; appendix variance table |

*Every "hoped" result below is a hypothesis, not a finding.*

---

## 1. What we test and why it matters

We re-run the exact system that produced t17 on 8 Core-10 datasets at fold seeds 42 and 4. Everything is the same as in t17: image, launchers, settings, data and fold seeds. Each new run has a t17 partner on the **identical outer-fold partition**, so the difference between the two comes only from LLM sampling, plus whatever live web search has drifted since mid-September.

From the 16 pairs we get two quantities:
- **σ_samp**, the v1 sampling-only run-to-run SD;
- **the sampling share** of t17's run-to-run variance. The t17 seed SD mixes partition variance and sampling variance.

The paper needs both:
- the M1 gate (A03) flags harm on a dataset when Δ/(√2·σ̂_samp) < −2.5, and it is committed on Tue 13 Oct;
- pairing every arm with M0 by fold seed is worth something only if partition variance is a large share of the total.

## 2. Background you need

**Code version.** References are to RougeOne `nested-cv-rebuttal@07cbf93` (`C:/code/PHD/RougeOne`). Short paths follow 01 §7 R0: `main.py` = `rogue_one_main.py`, and so on.
- The t17 result CSVs exist only from `nested-cv-rebuttal@1bdaf42`, the next commit (currently the branch tip), which touches only `paper/`.
- `1bdaf42` also adds one column (`partial_folds`, per-dataset CSV only) to `collect_seed_sweep_t17.py`. A01 uses the `07cbf93` collector; the selection rule and the long CSV are identical in both versions.
- Commands below are written for a local **bash** shell (Git Bash on Windows). Multi-line remote commands use `ssh … CAIR-gpu14 bash -s <<'EOF'`, so nothing is expanded locally; replace the `<…>` placeholders before sending. The fenced blocks are indented by the list nesting in this file: strip that indent before sending (the Python in a `-c '…'` string is indentation-sensitive).
- **Session plan** = `docs/agent2027/sessions/a01_session_plan.md` on `agent2027/v2` (new file). Every task below that says "record" appends one dated line there. It is created with A01-T9 and final at A01-T18. Nothing in it is a result; it is the audit trail for the replicate.

**What t17 is.**
- **Launch.** Seed 42 started on 2026-09-15 and seed 4 on 2026-09-21. The chain was `paper/scripts/t17_rerun_launch.sh` → `vllm_session.sh` → `clients_t16_rerun.sh` (t17 used the copies `~/vllm_session.sh` and `~/clients_t16_rerun.sh`, `t17_rerun_launch.sh:50-51`).
- **Images.** Client image `rogue_one:nestedcv9` (`t17_rerun_launch.sh:45`). Server image `vllm/vllm-openai:latest` (`:38`), whose ID was logged and whose version (0.22.0) was asserted in-container (`:58-61`).
- **Where things are.**
  - W&B: `brainflux/RogueOne_NestedCV_T17_Seed<seed>`.
  - Run dirs on CAIR: `~/rogueone-runs-t17s<seed>`.
  - Job logs: `~/t17_rerun_job_s<seed>.log`; session logs `~/vllm_session_t17_rerun_s<seed>.log`.
- **Results.** `paper/results_seed_sweep_t17.csv` (per dataset) and `paper/results_seed_sweep_t17_long.csv` (per dataset × seed), both from `paper/scripts/collect_seed_sweep_t17.py`.
- **Timing** (camera-ready plan log, `paper/reviewer_feedback/CAMERA_READY_EXPERIMENT_PLAN.md`). The 26-dataset concurrent phase at `PARALLEL=6` took about 13 h for seed 42 (done 2026-09-15T20:25Z) and about 9.5 h for seed 4 (launched 2026-09-21T05:15Z, done 14:47Z), i.e. about 2.2–3 h per run. `covtype` then ran solo; it is not part of W0.

**The v1 system as A01 runs it** (unchanged, which is the point):
- **Loop.**
  - The Scientist (S) sets a focus, the Extractor (E) writes features, and the Tester (T) assesses and prunes.
  - All three call one web-search summariser (W), which uses live Serper (`kn/web_search.py:69`). v1 has no search cache.
  - 10 iterations × 5 folds. The reported iteration is chosen by inner CV, and rollback is off.
- **v1 specifics.**
  - E runs 10 `Runner.run` rounds per iteration with no break on success (`ext/agent.py:967-992`).
  - The pool works in replace mode.
  - The Scientist sees the global trial counter (`main.py:553-554`: `index` = global trial 0–49, `max_index` = 10).
  - Tool errors return `None`: `CONSOLE_DISABLE=True` (`clients_t16_rerun.sh:108`, read at `config.py:300`) makes the `disabler()` wrapper (`console.py:31-42`) return `None` for every one-argument `console_error_print(...)` call, which is what the Extractor's tools return on error (e.g. `ext/agent.py:223`, `:501`).
  - Each run ends by posting "RUN FINISHED" to the hard-coded webhook (`main.py:1606`). This is baked into the old image; X8 removes it for v2 only.
- **Settings set by the launcher** (`clients_t16_rerun.sh:93-109`):
  - `ROGUE_ONE_START_VLLM=False`, `ROGUE_ONE_DATASETS=<one dataset>`;
  - `ROGUE_ONE_SELECTION_SIGNAL=inner_cv`, `ROGUE_ONE_ENABLE_ROLLBACK=False`, `ROGUE_ONE_STRATIFIED_FOLDS=False`;
  - `ROGUE_ONE_NUM_ITERATIONS=10`, `ROGUE_ONE_K_FOLDS=5`;
  - `ROGUE_ONE_FOLD_SEED=ROGUE_ONE_LLM_SEED=$SEED`;
  - `ROGUE_ONE_DO_LOGGING=True`, `LLM_PORT=${PORT:-8010}`, `LLM_API_KEY=${API_KEY:-rogueone-nestedcv-key}`;
  - `WANDB_PROJECT_NAME=$PROJECT`, `WANDB_ENTITY=brainflux` (hard-coded, `:106`), `WANDB_API_KEY` read from `<CAIR_HOME>/code/PlanBench/.env` (`:59`), `CONSOLE_DISABLE=True`;
  - the task folder mounted at `/workspace/tasks` (`:109`); `--cpus $CPUS`, `--network host` (`:91-92`).
- **Settings from the image.** The image copies the build context (`COPY . .`, `Dockerfile:13`) with the env-file excludes commented out (`.dockerignore:29-32`). `config.py:23-25` loads `.env`, then **`.dev.env` with `override=True`**, then `.secret.env`.
  - **`.dev.env` is tracked in git at `07cbf93`** and sets `LLM_PORT=8000`, an `LLM_API_KEY` and `ROGUE_ONE_K_FOLDS=5`. If the image contains such a file, it silently overrides the launcher's `-e` flags. t17 worked on port 8010, so the image's effective values must be the launcher's; A01-T6 proves this before any GPU is booked.
  - Classification is **plain accuracy of an unweighted fit**: the server `.env` sets `ROGUE_ONE_DO_BALANCING=False` (`paper/scripts/classicfe_arm.py:34-37`). The code default is `True` (`config.py:312`). A01-T6 confirms the image's effective value.
  - Full-v2 instead uses balanced accuracy (`ROGUE_ONE_DO_BALANCING=True`). A05 owns that mismatch (A05 OQ1).
  - `LLM_MODEL` is not set by the launcher; it comes from the image `.env` and must equal the served name `openai/gpt-oss-120b` (A01-T6).
- **Same splits.** `ROGUE_ONE_FOLD_SEED` (`config.py:310`) feeds `deterministic_kfold_indices` (`rogueone/utils/train_test_splitter.py:199`, called at `main.py:305-308`), unstratified. Same data, same seed and same code give the same partition as t17.
- **Sampling.**
  - `ROGUE_ONE_LLM_SEED` (`config.py:330`) is sent as the OpenAI `seed` on every request (`clients.py:211-213`).
  - A01 keeps t17's `LLM_SEED = FOLD_SEED` (registry R5).
  - Under `PARALLEL=6` batching, vLLM is not byte-reproducible at a fixed seed (Finding 36, noted at `clients_t16_rerun.sh:129-133`), so the partners diverge. The independence check (§2 Definitions, computed by A01-T10) measures how much.
- **W&B logging.**
  - The legacy run name is the task's `experiment_name` (`rogueone/utils/wandb_utils.py:113`), not the R12 pattern, and the R12 config keys (`ablation_id`, `flags/*`, …) are absent. Identity is carried by the W&B project name, the manifest (A01-T9) and the analysis run.
  - Legacy runs upload `df_test_pool_final` and `df_attributes_explanations_final` (`main.py:1400-1416`), `fold_aggregation_csv` (`:1375-1381`, completed runs only) and the tool-error log (`:1513-1521`). They do **not** upload the rest of `output/` (no R12 `run-output` artifact: `intermediate_steps/`, `tester_reports/`, aggregation CSVs as files). A01-T10 therefore uploads the 16 W0 `output/` folders as `a01-run-outputs`.
- **Success** = rc 0 **and** `fold_aggregation_summary.csv` exists (`main.py:1350-1351`; written only when all folds completed, `:1483-1495`). `run_experiments` swallows exceptions (`main.py:1590-1598`). The launcher enforces this rule (`clients_t16_rerun.sh:113-119`). An aborted run still writes a partial `df_test_pool_final.csv` (`save_data`, `main.py:1477`), so a failed run's directory must never be collected as a result (§4 Failure handling).
- **Hang hazard.** The image entrypoint runs `python3 -u /workspace/rogue_one_main.py || tail -f /dev/null` (`docker-entrypoint.sh:21`). A hard crash therefore leaves the container "Up" forever.

**Launcher facts** (`paper/scripts/clients_t16_rerun.sh` and `vllm_session.sh` @07cbf93):
- **Defaults that are wrong for A01**, so set each one explicitly:
  - `IMAGE_RO=rogue_one:nestedcv8` (`:54`);
  - `PROJECT` (`:55`) and `RUNDIR` (`:56`);
  - `SOLO=covtype` (`:64`). The expansion `${SOLO:-covtype}` treats an empty value as unset, so pass `SOLO=' '`;
    *Superseded 2026-10-16:* covtype is never run in any study (Henrik). `SOLO` now defaults to empty (no solo phase) and the script refuses covtype in `DATASETS` or `SOLO`; `SOLO=' '` is still harmless;
  - `DATASETS` = 26 datasets (`:65`), with `SOLO` = `covtype` as the 27th (historically; covtype is banned now);
  - `REPO_TASKS=$HOME/code/RougeOne/tasks` (`:135`);
  - `vllm_session.sh`: `IMAGE=vllm/vllm-openai:v0.22.0` (`:51`), `SERVER_NAME=rogueone-vllm-session` (`:54`), `GPUS=auto` (`:39`).
- **Clients script.**
  - `xargs -P $PARALLEL` (`:161-162`).
  - One solo retry pass (`:167-177`), ending with `=== still failing after solo retry: <list or none> ===` (`:174`).
  - Start and done lines carry UTC timestamps (`:90`, `:115`, `:118`), which give the wall clock per run.
  - End marker `T16_RERUN_ALL_DONE` (`:179`).
  - Container name `rogueone-<basename(RUNDIR) minus "rogueone-runs-">-<dataset>` (`:75-76`), e.g. `rogueone-S1007a-heart`.
  - Per-dataset container log `$RUNDIR/logs/<dataset>.log` (`:110`).
  - **No wall-clock cap.**
- **`vllm_session.sh`.**
  - TP = number of `GPUS`; more than 4 is refused (`:61-69`).
  - A second `rogueone-vllm-*` server is refused (`:121-130`).
  - Explicit `GPUS` skips the auto-wait (`:132-154`).
  - The server runs without `--max-model-len` and without `HF_HUB_OFFLINE`, as in t17 (`:161-168`).
  - All output, including the clients', goes to `$LOG` (`:75`).
  - Health check: up to 30 min, `vLLM healthy after …s` (`:171-182`).
  - The EXIT trap removes the server and writes `=== session end rc=` (`:79-90`).
- **Changes since t17.** `git diff 4b0da38 1ae0771` (no later change up to `07cbf93`) touches only the self-cap guard, the default image and the client container name (plus its ownership guard). None of them changes the run env or the server flags.
- **`t17_rerun_launch.sh` is NOT reused.** It hard-codes t17's own log files (`:48`, `:52`, appended at `:54`) and the moving tag `:latest` (`:38`). A01 calls the two scripts directly and repeats its ID logging and version assertion (`:58-61`) in pre-flight (A01-T2, T3) and at launch (A01-T12).

**Instruments (W0-8 = Core-10 minus `balance-scale` and `wine`).** The t17 values below come from `paper/results_seed_sweep_t17*.csv` @`1bdaf42`. The `base_no_fe` fold SD is `value_sd` in `paper/results_matched_harness_baselines.csv`.

| dataset_id | Type | Metric | t17 f42 | t17 f4 | t17 seed SD (seeds 42, 4, 5) | `base_no_fe` fold SD s_d |
|---|---|---|---|---|---|---|
| `credit-g` | C | accuracy | 0.7310 | 0.7250 | 0.0055 | 0.0498 |
| `heart` | C | accuracy | 0.8659 | 0.8301 | 0.0189 | 0.0315 |
| `junglechess` | C | accuracy | 0.9711 | 0.8484 | 0.0695 | 0.0021 |
| `myocardial` | C | accuracy | 0.7275 | 0.7274 | 0.0076 | 0.0331 |
| `vehicle` | C | accuracy | 0.7731 | 0.7577 | 0.0294 | 0.0375 |
| `crab` | R | n_RMSE | 0.2614 | 0.2169 | 0.0242 | 0.0096 |
| `housing` | R | n_RMSE | 0.2354 | 0.2655 | 0.0213 | 0.0070 |
| `insurance` | R | n_RMSE | 0.3742 | 0.4798 | 0.0630 | 0.0193 |
| *opt.* `wine` | R | n_RMSE | 0.1196 | 0.1215 | 0.0010 | 0.0011 |
| *opt.* `balance-scale` | C | accuracy | 1.000 | 1.000 | 0.000 | 0.0215 |

`balance-scale` scores 1.000 in every t17 run, so its t17 SD is 0 and ρ_d is undefined; it is never pooled. `junglechess` is bimodal: the rule is either found (≥ 0.95) or missed. The optional rows are descriptive only: A03 uses the t17 SD for `balance-scale` and `wine` whether or not they run (A03 §6).

**Definitions** (computed by the A01-T10 script, run in A01-T17; d = dataset, s = fold seed):
- **Complete run** = rc 0, `fold_aggregation_summary.csv`, 5/5 outer folds and the inner metric on every completed trial (collector checks `collect_seed_sweep_t17.py:129-136`). **Complete pair** = the W0 run and its t17 partner are both complete. k_d = number of complete pairs of dataset d (0, 1 or 2). Rc comes from the dataset's **last** `--- done rogueone <dataset> rc=…` line in the session log (a solo retry adds a second start/done pair). For a t17 partner no rc is needed: the summary file under `--t17-roots` and the committed long-CSV columns `folds`, `completed_trials` and `inner_populated` stand in.
- **Population.** The pooled quantities, both drift tests, the verdict and every §6 rule use the **W0-8 pairs only**. Pairs of the optional `balance-scale` and `wine` rows go into a separate descriptive table and never enter them.
- **s_d** = `value_sd` of the row with `baseline=base_no_fe` in `baselines.csv` (= `paper/results_matched_harness_baselines.csv`; exactly one such row per dataset, `fold_scheme=kfold`; the script asserts this). It is the fold SD of another arm, so it is external to both runs. It is in the units of the dataset's metric.
- **Run score** = within each outer fold, take the trial with the best inner-CV metric and record its held-out metric; then average over the 5 folds (`collect_seed_sweep_t17.py:88-97`). The metric is `accuracy` for classification and `n_RMSE` for regression (`:79-80`).
- **δ_{d,s}** = W0 − t17 for classification and t17 − W0 for regression, so positive always means W0 did better.
- **`sigma_samp`** (per dataset, raw units, k_d df): σ²_samp,d = Σ_s δ²_{d,s} / (2·k_d); with two pairs, σ_samp,d = √((δ²_{d,42} + δ²_{d,4}) / 4). The difference of two independent draws has variance 2σ².
- **ρ_d** = σ_samp,d / t17 seed SD_d (t17 SD from the committed `paper/results_seed_sweep_t17.csv`, seeds 42, 4, 5, 2 df). Z_d = ρ_d² (capped at 1) is the per-dataset sampling share. Descriptive only: with 2 df on each side, ρ_d² is F(2, 2)-distributed when sampling is all the variance, so a single ρ_d can be off by a factor of 6 either way.
- **Pooled share (primary)** over the W0-8 datasets with k_d ≥ 1. With v_d = σ²_samp,d and w_d = SD²_t17,d (raw units): **Z_v1 = Σ_d v_d / Σ_d w_d** (not capped; R_raw = √Z_v1; values > 1 are the variance-inflation signal).
  - **Effective df (Satterthwaite).** The dataset variances differ by a factor of about 160 (`junglechess` and `insurance` carry about 79% of Σ w_d), so the nominal 16/16 df overstate the information. Use ν1 = (Σ w_d)² / Σ (w_d² / k_d) and ν2 = (Σ w_d)² / Σ (w_d² / 2), with the **committed t17 variances as weights**. The thresholds below are then fixed by the committed CSV and the pair pattern before any W0 number exists. With two pairs on all 8 datasets ν1 = ν2 ≈ 6.1 (≈ 4.6 without `junglechess`).
  - **CI.** [Z_v1 / F_{0.975}(ν1, ν2), Z_v1 × F_{0.975}(ν2, ν1)] with `scipy.stats.f.ppf` at non-integer df. At the full design the factor is **≈ 5.7** (≈ 7.8 without `junglechess`).
  - **Why not 16/16.** The nominal version (ν1 = Σ k_d = 16, ν2 = 2 × number of datasets = 16, factor 2.76) is written to the summary JSON as `*_nominal` and is **not used for any decision**. A Monte Carlo over these t17 SDs (normal errors, true sampling share 1, no drift, 60,000 draws; the A01-T10 selftest repeats it) gives the nominal "95%" CI a coverage of about 79% and a false variance-inflation flag in about 10.5% of cases, against about 97% and 1.6% for the Satterthwaite version.
  - The interval is still approximate: the t17 runs at seeds 42 and 4 enter both Z_v1's numerator and its denominator. Z_v1 and its CI are also reported without `junglechess`.
- **`sigma_samp_pooled_sd`** (s_d scale): √(Σ over complete pairs of (δ/s_d)² / (2·n_pairs)), n_pairs df. Three versions: all W0-8 pairs (16 df); without `junglechess` (s_d = 0.0021 dominates; 14 df); **`sigma_samp_pooled_sd_reg3`** over `crab`, `housing`, `insurance` only (6 df), which is the value A05's homogeneity rule uses (σ̂²_AA / `sigma_samp`² in the central 95% F(3, 6) interval [0.068, 6.60]; classification is never pooled there).
- **Drift tests.**
  - *Mean shift:* two-sided Wilcoxon signed-rank test on z = δ/s_d over all complete W0-8 pairs, with the exact null distribution by enumerating all 2^n sign flips (mid-ranks for ties; zero differences dropped; p := 1 if no non-zero pair). s_d comes from outside both runs, so the test is valid under exchangeable, symmetric pair differences. Output `drift_p`.
  - *Variance inflation* (`var_inflation`): Z_v1 > F_{0.975}(ν1, ν2) (≈ 5.7 at the full design), i.e. the lower 95% bound of the sampling share exceeds 1: W0 differs from t17 on identical splits by more than t17's seeds differ on different splits. The threshold is written to the summary JSON as `var_inflation_threshold`.
- **Independence.**
  - `pair_identical`: whether the `Attribute` + `Pandas Command` list (columns defined at `main.py:233-243`; file `output/df_attributes_explanations_final.csv`) is byte-identical to t17's, plus the Jaccard of the two command sets. Fold-scoped state is reset at every new fold (`main.py:417-457`), so this file holds the **last outer fold only**.
  - `tester_identical_share`: over the 50 global trial indices i = 0–49, the share for which both partners have `output/tester_reports/tester_report_iteration_<i>.md` (`main.py:849-857`; a report is not written when it is `None`) and the two files are byte-identical. An index missing in either partner counts as not identical. This covers all folds.
- **`wall_s`** per run = done time − start time of its **last** `--- start` / `--- done rogueone` line pair in the session log (a solo retry adds a second pair; flag such runs `retried`).

## 3. Hypotheses & interpretation

The verdict is read on the pooled share Z_v1 (nominally 16 vs 16 df, effectively about 6 vs 6; §2), not on per-dataset counts. A "ρ_d ≤ 0.6 on ≥ 5 of 8" rule would reach "hoped" only 36% of the time even if the true share were exactly 36%, and a "ρ_d > 1.2 on ≥ 4 of 8" drift rule would fire about 43% of the time with no drift at all if sampling were all the variance (F(2, 2) arithmetic). Per-dataset ρ_d and those counts are reported, descriptively.

**How well the pooled verdict can discriminate** (verifier's Monte Carlo; normal errors, σ_d = the t17 SD_d, two pairs on each of the 8 datasets, 40,000 draws; the A01-T10 selftest repeats it). With a true sampling share of 0.36 the point estimate reads "hoped" 50% of the time, "partial" 30% and "null" 20%. At a true share of 0.72 it reads "null" 50%, "partial" 30% and "hoped" 20%. At a share of 1.0 it reads "null" 66%, "partial" 24% and "hoped" 10.5%. A point estimate of 0.36 has a 95% CI of about [0.06, 2.0]. The verdict is therefore a coarse classification, and §6 rule 3 limits the wording to what the CI supports.

- **Hoped (hypothesis H-A01).**
  - Z_v1 ≤ 0.36 (R_raw ≤ 0.6, the breakdown's "σ_samp ≤ 0.6 × the t17 seed SD"): sampling is ≤ 36% of t17's run-to-run variance;
  - no drift (Wilcoxon p ≥ 0.05 and no variance inflation).
  - **Sentence:** *"Re-running the identical system on identical splits moves held-out scores by σ_samp ≈ X (median over datasets), about Y% [CI] of run-to-run variance; the rest comes from the fold partition, so every arm is paired with M0 by fold seed."*
- **Null (hypothesis).** Z_v1 ≥ 0.72 (R_raw ≥ 0.85): sampling dominates, and pairing by fold seed buys little power.
  - The paired analysis is still valid and costs nothing, so it stays. The NI-margin formula is unchanged; only the MDE fails to shrink.
  - More weight moves to the SYN-P instruments.
  - **Reported as** *"most run-to-run variance is LLM sampling."*
- **Partial.** 0.36 < Z_v1 < 0.72: reported as "sampling and partition both contribute", with the CI.
- **Opposite (hypothesis).** Drift: a mean shift (Wilcoxon p < 0.05) **or** variance inflation (Z_v1 > `var_inflation_threshold`, ≈ 5.7 at the full design). Either means the environment has drifted since t17: Serper results, server state or weights.
  - t17 and W0 are **not** pooled into σ_d (A05).
  - The M1 rule falls back to the t17 seed SD, with a caveat (A03).
  - M0's A/A becomes the only v2 sampling estimate.
  - Live-search drift is disclosed as a threat to validity, and v2 relies on the shared Serper cache.
- **Secondary (descriptive).** Is junglechess's bimodality sampling-driven? Compare the number of folds (of 5) with a selected score ≥ 0.95 in t17 and in W0, per seed; the per-fold selected scores come from each run's `df_test_pool_final.csv` with the §2 selection rule. For `insurance`, whose seed-4 run is an outlier, list the five per-fold selected n_RMSE values of both partners per seed and the per-fold δ (there is no natural threshold, so no count).

## 4. Design

**Arms.** One arm, `t17rep`. The control is t17 itself; nothing in t17 is re-run. There are no overrides: every setting is the legacy one (§2).

**Fixed settings.**

| Setting | Value |
|---|---|
| Client image | `rogue_one:nestedcv9` (ID must equal t17's, A01-T2) |
| Server | `<VLLM_REF>` = t17's recorded vLLM image ID (0.22.0 asserted), TP4, `EAGER=0`, port 8010, `API_KEY` default (`rogueone-nestedcv-key`), `PARALLEL=6`, `CPUS=12` (72 of 224 cores) |
| Launchers | `vllm_session.sh` and `clients_t16_rerun.sh` extracted from `07cbf93`, unchanged |
| Seeds | `fold_seed` ∈ {42, 4}; `llm_seed` = `fold_seed` |
| Dataset order | t17's relative order (`clients_t16_rerun.sh:65`): `credit-g heart junglechess myocardial vehicle crab housing insurance`. With the add-on, `balance-scale` goes first and `wine` last. |
| W&B | `brainflux/RogueOne_W0_T17Rep_Seed<seed>` (D9 default) |
| Search | live Serper, no cache (as t17) |

**Run matrix.**

| Group | Arm | Datasets | fold = LLM seed | Runs | Session | Manifest rows | W&B project | Run name (legacy) / alias | t17 partner |
|---|---|---|---|---|---|---|---|---|---|
| G42 | `t17rep` | W0-8 | 42 | 8 | `S1007a` | A01-001…008 | `brainflux/RogueOne_W0_T17Rep_Seed42` | `experiment_name` / `A01__t17rep__<dataset_id>__f42__r0` | `brainflux/RogueOne_NestedCV_T17_Seed42`, `~/rogueone-runs-t17s42` |
| G4 | `t17rep` | W0-8 | 4 | 8 | `S1008a` | A01-009…016 | `…_Seed4` | `…__f4__r0` | `…_T17_Seed4`, `~/rogueone-runs-t17s4` |
| opt. | `t17rep` | `balance-scale`, `wine` | 42, 4 | 2 + 2 | in G42 / G4 | A01-017…020 | as above | as above | as above |

**Budget arithmetic.**
- Runs: 8 datasets × 2 seeds × 1u = 16u. The add-on gives 10 × 2 × 1u = 20u. This matches the T1-minimum breakdown and registry R1 (16 runs, 16–20u, outside the window).
- Planning wall clock: 16u ÷ (2 u/h × 0.7) = 11.4 h, about 46 H100-hours at TP4.
- Expected per wave: with 6 clients a run took about 2.2–3 h in t17 (§2 Timing: 26 runs in 13 h for seed 42, in 9.5 h for seed 4), so an 8-dataset wave takes two rounds, about 5–8 h with the vLLM load (its tail runs only 2 clients, which is where the 0.7 goes). Two waves take 10–16 h; the plan keeps **12–16 h, i.e. 48–64 H100-hours**, as the allowance for retries.
- Serper (*estimate*, 500–1,000 queries per run): 8,000–16,000 (10,000–20,000 with the add-on). Measured afterwards from the Serper dashboard and recorded in the session plan.

**Pairing and randomisation.** Each run is paired by (dataset, fold seed) with its t17 run. There is no randomisation: order and concurrency copy t17. **One deviation:** an 8-dataset wave has a tail with fewer than 6 clients, while t17's 26-dataset waves did not. This is disclosed.

**Failure handling.**
- A run counts as complete only under the §2 definition (rc 0, `fold_aggregation_summary.csv`, 5/5 outer folds, inner metric on every completed trial).
- The launcher retries failures once, alone, at the end of the wave. A run that still fails gets one relaunch in a follow-up session launched by Sun 11 Oct, 18:00 (only if the box is free; §6 rule 5), one session per seed, one after the other.
- For a (dataset, seed) that was rerun in a follow-up session, **only the rerun's directory is collected**; the failed directory is never used (it may hold a partial `df_test_pool_final.csv`).
- **No ITT imputation**: imputing a score would distort a variance estimate. Incomplete pairs are dropped and counted.
- **No 6 h cap**: t17 had none, and adding one would change the replicate. Runs longer than 6 h are flagged, not killed. A dead "Up" container (hang hazard, §2) is stopped by hand and handled as a failure.

## 5. Tasks

### 5.1 Prerequisites (all on Wed 7 Oct, before booking the GPUs)

- [ ] **A01-T1** [type: decision] (5 min, due Wed 7 Oct, 16:00) — **D9.** W&B entity `brainflux` (default; the launcher hard-codes it, `clients_t16_rerun.sh:106`) or `PlanBench`. Choosing `PlanBench` would mean editing the legacy launcher, so the replicate would no longer be "unchanged". If Henrik records nothing by the due time, `brainflux` applies. *Done when:* D9 is recorded in the 01 §6.4 decision log and in the session plan.
- [ ] **A01-T2** [type: config] (10 min, due Wed 7 Oct) — **Image check.**
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "grep -HE '(vllm|rogueone) image=' ~/t17_rerun_job_s42.log ~/t17_rerun_job_s4.log; docker image inspect rogue_one:nestedcv9 --format 'nestedcv9 {{.Id}} {{.Created}}'; docker images -a --no-trunc --format '{{.Repository}}:{{.Tag}} {{.ID}}' | grep -E 'vllm|<none>'"
  ```
  The job logs hold lines of the form `[<UTC>] vllm image=<tag> id=sha256:…` and `[<UTC>] rogueone image=rogue_one:nestedcv9 id=sha256:…` (`t17_rerun_launch.sh:58-59`). The job logs are appended, so a seed may show several launches; use the last one of each. The last command lists untagged images too, because the `:latest` tag t17 used may have moved since.
  *Done when:* the `rogue_one:nestedcv9` ID equals the `rogueone image=` ID in both t17 logs, **and** the `vllm image=` ID is the same in both logs. **If the image is gone, its ID differs, or the two vLLM IDs differ (`:latest` moved between 15 and 21 Sep), STOP**: a rebuild is not a replicate, and one server cannot replicate both seeds. Henrik then decides whether to skip A01 (A03 then uses the t17 SD) or to accept and disclose the difference.
- [ ] **A01-T3** [type: config] (10 min, due Wed 7 Oct) — **vLLM image.**
  - Check that t17's image still exists, by its ID from A01-T2: `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker image inspect <t17 vllm id> --format '{{.Id}} {{.RepoTags}} {{.Created}}'"`. If it prints, take `<VLLM_REF>` = that bare `sha256:` ID (or a tag listed beside it, if one is).
  - Assert the version in a container without GPUs: `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker run --rm --entrypoint python3 <VLLM_REF> -c 'import vllm; print(vllm.__version__)'"`.

  *Done when:* `<VLLM_REF>` and the printed `0.22.0` are written in the session plan. If the image is gone, STOP and report; Henrik decides whether to accept `vllm/vllm-openai:v0.22.0` (the pinned tag) and disclose it.
- [ ] **A01-T4** [type: config] (20 min, due Wed 7 Oct) — **Push, then extract the unchanged scripts and reference data.** No checkout on the server, so its working tree is untouched.
  - **Step 0 (local, Henrik; code delivery is push then fetch).** On 2026-10-07 `git ls-remote origin nested-cv-rebuttal` printed `2cd6d0a…` (the local branch is 73 commits ahead), and neither `07cbf93` nor `1bdaf42` is on GitHub; the launcher scripts are not on GitHub at all. Push from `C:/code/PHD/RougeOne`: `git push origin nested-cv-rebuttal`. The push carries both commits (`07cbf93` is the parent of `1bdaf42`). Check that `git ls-remote origin nested-cv-rebuttal` prints `1bdaf42e6d3b2132c6380af876a6ab56196feb69`. The webhook URL at `main.py:1606` is already on GitHub at `2cd6d0a`, so the push adds no new exposure of it (X8 handles it before release).
  - **Step 1 (CAIR).**
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 bash -s <<'EOF'
  set -e
  cd <ROGUEONE_CAIR_PATH>
  GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15' git fetch git@github.com:HenrikNorkart/RougeOne.git nested-cv-rebuttal
  mkdir -p ~/a01_collect/empty ~/a01_collect/w0 ~/a01_collect/t17 ~/a01_collect/logs ~/a01_collect/analysis
  git show 07cbf93:paper/scripts/vllm_session.sh > ~/a01_vllm_session.sh
  git show 07cbf93:paper/scripts/clients_t16_rerun.sh > ~/a01_clients_t16_rerun.sh
  git show 07cbf93:paper/scripts/collect_seed_sweep_t17.py > ~/a01_collect/collect_seed_sweep_t17.py
  git show 07cbf93:paper/results_matched_harness_baselines.csv > ~/a01_collect/baselines.csv
  git show 1bdaf42:paper/results_seed_sweep_t17.csv > ~/a01_collect/t17_committed.csv
  git show 1bdaf42:paper/results_seed_sweep_t17_long.csv > ~/a01_collect/t17_committed_long.csv
  md5sum ~/a01_vllm_session.sh ~/a01_clients_t16_rerun.sh ~/a01_collect/collect_seed_sweep_t17.py ~/a01_collect/baselines.csv ~/a01_collect/t17_committed.csv ~/a01_collect/t17_committed_long.csv
  md5sum ~/vllm_session.sh ~/clients_t16_rerun.sh 2>/dev/null || true
  EOF
  ```
  *Done when* step 0's `git ls-remote` printed the full `1bdaf42…` SHA, and the md5 values printed by step 1 are (all six recomputed locally from `07cbf93` / `1bdaf42` on 2026-10-07; a failed fetch with a permission error means the server key has no GitHub access, so STOP and report):
  - `8e05c871e769dfd568fe1da33e43ca9b` (vllm_session);
  - `98ffd9a9b620cd4b266722cce851a7c3` (clients);
  - `c68d9ba2b2a725d415ec8dc74ecd2bf6` (collector);
  - `96edb453a12239f75be581ac1e7d1acb` (baselines);
  - `b9ec91f17da0853bd63de3b97510570d` (t17 per-dataset CSV) and `5abbf6299c4a9179b7d617dbccacd635` (t17 long CSV).

  The last line records what t17's own home copies are now; at t17 time they should have been the `4b0da38` versions (`7d7b92c72e73be4705f8414da7b5c85b` vllm_session, `d7cc77416949b24aafe5282fe99ee972` clients). Record, do not fix. `<ROGUEONE_CAIR_PATH>` = `<CAIR_HOME>/code/RougeOne` and the SSH remote URL are **inferred** (registry R6, R8); confirm the path with `ls -d ~/code/RougeOne`.
- [ ] **A01-T5** [type: data] (10 min, due Wed 7 Oct) — **Weights unchanged, locally and on the Hub.** `vllm_session.sh` starts the server online (no `HF_HUB_OFFLINE`), so a moved Hub `main` could be downloaded at launch.
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 bash -s <<'EOF'
  H=~/.cache/huggingface/hub/models--openai--gpt-oss-120b
  echo "local refs/main = $(cat $H/refs/main)"; ls -l --time-style=long-iso $H/snapshots/
  docker run --rm --entrypoint python3 <VLLM_REF> -c 'from huggingface_hub import HfApi; print("hub main =", HfApi().model_info("openai/gpt-oss-120b").sha)'
  EOF
  ```
  *Done when:* exactly one snapshot exists, `refs/main` points to it, it predates 2026-09-15, and `hub main` equals the local `refs/main`. Otherwise STOP and report; Henrik decides (accept and disclose, or a one-off offline server, which is then a disclosed deviation). A04's downloads (Thu 8 – Sun 11 Oct) must not touch `openai/gpt-oss-120b` in the HF cache until A01 ends; A04's own list (Qwen ids, `gpt-oss-20b`) does not include it, but record the constraint in the session plan.
- [ ] **A01-T6** [type: config] (15 min, due Wed 7 Oct) — **Image env, effective settings and W&B access**, printing no secret values.
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 bash -s <<'EOF'
  docker run --rm --entrypoint sh rogue_one:nestedcv9 -c 'for f in /workspace/.env /workspace/.dev.env /workspace/.secret.env; do echo "== $f"; [ -f "$f" ] && sed -nE "s/^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=.*/\2/p" "$f"; done; true'
  docker run --rm -e ROGUE_ONE_START_VLLM=False -e ROGUE_ONE_DATASETS=heart -e ROGUE_ONE_SELECTION_SIGNAL=inner_cv -e ROGUE_ONE_ENABLE_ROLLBACK=False -e ROGUE_ONE_STRATIFIED_FOLDS=False -e ROGUE_ONE_NUM_ITERATIONS=10 -e ROGUE_ONE_K_FOLDS=5 -e ROGUE_ONE_FOLD_SEED=42 -e ROGUE_ONE_LLM_SEED=42 -e ROGUE_ONE_DO_LOGGING=True -e LLM_PORT=8010 -e LLM_API_KEY=rogueone-nestedcv-key -e WANDB_PROJECT_NAME=probe -e WANDB_ENTITY=brainflux -e CONSOLE_DISABLE=True --entrypoint python3 rogue_one:nestedcv9 -c '
  import os, numpy, pandas, scipy, wandb, rogueone.utils.config
  for k in ["LLM_PORT", "LLM_MODEL", "LLM_TEMPERATURE", "ROGUE_ONE_DO_BALANCING", "ROGUE_ONE_NUM_ITERATIONS", "ROGUE_ONE_K_FOLDS", "ROGUE_ONE_FOLD_SEED", "ROGUE_ONE_LLM_SEED", "ROGUE_ONE_STRATIFIED_FOLDS", "ROGUE_ONE_SELECTION_SIGNAL", "ROGUE_ONE_ENABLE_ROLLBACK", "CONSOLE_DISABLE", "WANDB_ENTITY", "WANDB_PROJECT_NAME", "ROGUE_ONE_DATASETS"]:
      print(k, "=", os.environ.get(k))
  print("LLM_API_KEY unchanged:", os.environ.get("LLM_API_KEY") == "rogueone-nestedcv-key")
  print("SERPER_API_KEY set:", bool(os.environ.get("SERPER_API_KEY")))
  '
  docker run --rm --env-file <CAIR_HOME>/code/PlanBench/.env --entrypoint python3 rogue_one:nestedcv9 -c 'import wandb; print("t17 seed-42 runs:", len(wandb.Api().runs("brainflux/RogueOne_NestedCV_T17_Seed42")))'
  EOF
  ```
  *Done when* the session plan records:
  - the key **names** in each env file, and whether `/workspace/.dev.env` exists;
  - effective values `LLM_PORT=8010`, `ROGUE_ONE_FOLD_SEED=ROGUE_ONE_LLM_SEED=42`, `K_FOLDS=5`, `NUM_ITERATIONS=10`, `STRATIFIED_FOLDS=False`, `CONSOLE_DISABLE=True`, `WANDB_ENTITY=brainflux`, `LLM_API_KEY unchanged: True`, `SERPER_API_KEY set: True`, `LLM_MODEL=openai/gpt-oss-120b`, and the `DO_BALANCING` value (expected `False`; the code default is `True`, and `apply_optional_defaults` fills `1` when nothing sets it, `config_keys.py:33`. If it prints `True` or `1`, record it and tell A05, because the "plain accuracy, unweighted fit" statement in §2 and A05 OQ1 then change);
  - the imports succeed (the A01-T10 script relies on numpy, pandas, scipy and wandb in this image). Importing `rogueone.utils.config` also runs `validate_required_env_keys` (`config.py:33`), so a `RuntimeError: Missing required environment variables` (`LLM_MODEL`, `LLM_PORT`, `LLM_API_KEY`, `EMBEDD_MODEL`, `EMBEDD_PORT`, `EMBEDD_API_KEY`) means the image `.env` is incomplete: STOP, the image could not have run t17;
  - the W&B count is ≥ 27 (t17 had 27 datasets; retries can add runs).

  **If any launcher-set key is overridden** (a `.dev.env` in the image), STOP and report: the image would not honour the launcher. If the W&B call fails, `--env-file` may be passing quoted values literally; use `-e WANDB_API_KEY=...` extracted as in `clients_t16_rerun.sh:59` for every A01 `docker run` that needs W&B, and record that.
- [ ] **A01-T7** [type: data] (5 min, due Wed 7 Oct) — **Serper credits.** *Done when:* the Serper dashboard shows ≥ 16,000 queries left (≥ 20,000 with the add-on).
- [ ] **A01-T8** [type: data] (10 min, due Wed 7 Oct) — **Same task data as t17**, checked in a container:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 bash -s <<'EOF'
  docker run --rm --network none -v <ROGUEONE_CAIR_PATH>/tasks:/repo:ro -v ~/rogueone-runs-t17s42/tasks:/t42:ro -v ~/rogueone-runs-t17s4/tasks:/t4:ro --entrypoint bash rogue_one:nestedcv9 -c '
  n=0
  for d in classification/credit-g classification/heart classification/junglechess classification/myocardial classification/vehicle regression/crab regression/housing regression/insurance; do
    for t in t42 t4; do for f in config.yml data.csv labels.csv; do
      a=$(cat /repo/$d/$f 2>/dev/null | md5sum); b=$(cat /$t/$d/$f 2>/dev/null | md5sum)
      [ "$a" = "$b" ] || { echo "DIFF $t $d/$f"; n=$((n+1)); }
    done; done
  done
  echo "differences: $n"'
  EOF
  ```
  With the add-on, also list `classification/balance-scale` and `regression/wine`. *Done when:* it prints `differences: 0`. A `labels.csv` missing on both sides counts as equal.

### 5.2 Implementation (no change to the Rogue One package)

- [ ] **A01-T9** [type: config] (0.1 d, written Wed 7 Oct, committed by Fri 9 Oct on `agent2027/v2`) — Write `manifests/agent2027/A01_W0.csv` (16 rows, plus 4 optional) as the record, using the R7 header. Field values:
  - `row_id` = `A01-001`…`A01-016` in the run-matrix order (`A01-017`…`020` optional), `ablation_id=A01`, `arm_id=t17rep`;
  - `dataset_id`, `task_type` ∈ {`classification`, `regression`};
  - `fold_seed=llm_seed` (42 or 4), `replicate=0`, `is_sentinel=0`;
  - `session_id` = `S1007a` / `S1008a`, and `order_key` = position in `DATASETS`;
  - `image_tag=nestedcv9`, `rung_S`/`rung_E`/`rung_T` = `O120`, `port_S`/`port_E`/`port_T` = `8010`;
  - `env_overrides={}`, `cap_hours=6` (monitor-only), `expected_u=1`;
  - `notes` = `legacy clients_t16_rerun.sh@07cbf93; v2_defaults.env not applied; pair brainflux/RogueOne_NestedCV_T17_Seed<seed>`.

  The rows are **not** run by `ro27_clients_manifest.sh`, and analysis code must never read them as Full-v2 (filter on `ablation_id=A01`). Also create `docs/agent2027/sessions/a01_session_plan.md` (the session plan, §2) with the A01-T1 to A01-T8 records so far.
  - **Branch.** `agent2027/v2` does not exist locally or on GitHub (checked 2026-10-07). If E1 has not created it yet, create it from the pushed code and push it: `git switch -c agent2027/v2 07cbf93 && git push -u origin agent2027/v2` (R8: the branch starts at `07cbf93`).
  - **Ignore rules.** In `C:/code/PHD/RougeOne`, `.gitignore:17` (`*.csv`) ignores `manifests/agent2027/A01_W0.csv`, and `.gitignore:27` (`paper/`) ignores the `paper/results_w0_t17rep*` files of A01-T16/T17. Add them with `git add -f` (the t17 CSVs in `paper/` were force-added the same way), unless E1 has already added `!manifests/agent2027/*.csv` to `.gitignore`.

  *Done when:* the manifest and the session plan are committed and pushed on `agent2027/v2` (`git ls-files manifests/agent2027/A01_W0.csv` prints the path), and the manifest lists 16 (or 20) unique `row_id`s.
- [ ] **A01-T10** [type: code] (0.25 d, due Sat 10 Oct) — Write `scripts/agent2027/analysis/w0_sigma_samp.py` (numpy, pandas, scipy, wandb only, so it runs in `rogue_one:nestedcv9`).
  - **Inputs (arguments):** `--w0-long` (W0 collector long CSV), `--t17-long` and `--t17-sd` (the committed t17 CSVs), `--baselines` (`baselines.csv`, filtered as in the §2 s_d definition), `--logs` (W0 session logs), `--w0-roots` and `--t17-roots` (read-only run roots), `--rerun <dataset>:<seed>:<collector long CSV>:<root>` (repeatable, for A01-T15 reruns; overrides the first-session result for that dataset and seed), `--out`, `--wandb`, `--selftest`. A run is complete only under the §2 definition: the collector's checks plus rc 0 from the log and `fold_aggregation_summary.csv` under `<root>/tasks/<task_type>/<dataset>/output/`.
  - **Outputs:** everything in §2 Definitions (including the three `sigma_samp_pooled_sd` versions, Z_v1 with its CI, ν1, ν2 and `var_inflation_threshold`, the nominal-df versions, `drift_p`, `var_inflation`), the §6 verdicts, per-run `wall_s`, the independence table, the junglechess and insurance fold tables (§3 Secondary), the pair/df counts, and the descriptive table for the optional rows. It first prints `du -sh` of every collected `output/` folder; if the total exceeds 20 GB it warns, uploads everything except files above 500 MB, and lists the excluded files in the summary JSON.
  - **Files:** `results_w0_t17rep.csv` (one row per dataset), `results_w0_t17rep_pairs.csv` (one row per pair: `dataset_id`, `task_type`, `fold_seed`, W0 score, t17 score, δ in raw units, `s_d`, z = δ/s_d, `pair_identical`, Jaccard, `tester_identical_share`, `complete`) and `results_w0_t17rep_summary.json`. The pair table is what A05's homogeneity rule reads (A05 re-scales δ with its own s_d).
  - **W&B:** logs run `ANALYSIS__w0_sigma_samp__<YYYYMMDD>` (entity `PlanBench`, project `RogueOne-AGENT2027`, group `ANALYSIS`) with artifacts `a01-results` (the three result files above plus the W0 long CSV) and `a01-run-outputs` (the W0 `output/` folders of every collected run).

  Commit and push it on `agent2027/v2`. *Done when:* `--selftest` passes (locally, and again in `rogue_one:nestedcv9` as the first step of A01-T17) on synthetic data:
  - W0 = t17 gives σ_samp = 0 and p = 1;
  - a +3·s_d shift on all 16 pairs (distinct small jitter) gives p < 0.001 (the exact two-sided minimum is 2/2^16 ≈ 3·10⁻⁵);
  - draws with a known σ reproduce the σ_samp formula within 1e-9;
  - a constructed case with Z_v1 above `var_inflation_threshold` raises the flag, and one just below does not;
  - ν1 = ν2 within 0.1 of 6.1 for the committed t17 SDs with two pairs on each of the 8 datasets, and `scipy.stats.f.ppf(0.975, 16, 16)` = 2.761 ± 0.001 (the nominal factor);
  - a Monte Carlo (seed 1, 20,000 draws, σ_d = the t17 SD_d, sampling share 1, two pairs per dataset) gives a variance-inflation flag rate in [0.5%, 3%] and a coverage of the true share by the CI in [0.94, 0.99].

### 5.3 Execution on CAIR

**Hard rules (CAIR).**
- **Containers only.** Every execution of code happens inside Docker. The host only issues `docker`, the two legacy bash orchestration scripts in `~` (which themselves only call `docker`, `nvidia-smi` and `curl`), `git fetch` / `git show` (code delivery, below), and read-only queries (`nvidia-smi`, `docker ps/inspect/top/images`, `grep`/`sed -n`/`tail`/`cat`/`ls`/`md5sum` on logs and extracted files), plus `mkdir`/`cp` inside `~/a01_collect`. All Python, including the collector and the analysis, runs in a container.
- **GPU pre-flight before every GPU launch** (each session, including reruns): `nvidia-smi` memory and utilisation per GPU, compute apps, and running containers (A01-T11). If 4 GPUs are not free, **STOP and report** that output to Henrik. Do not queue onto busy GPUs.
- **Never more than 4 of the 8 H100s** without Henrik's explicit approval. A01 uses exactly 4 (TP4). While an A01 session is up, no other RogueOne GPU job may run (A04 downloads use no GPU); `vllm_session.sh:121-130` also refuses a second `rogueone-vllm-*` server.
- **W&B is the system of record**, with secrets from `.env`: the run W&B key comes from the PlanBench `.env` (`clients_t16_rerun.sh:59`), Serper from the image `.env`. A01's runs log to entity `brainflux` (D9, the one registry exception to `PlanBench`); the analysis and the uploaded `output/` folders go to `PlanBench/RogueOne-AGENT2027`. Local run dirs are evidence for collection, not the record.
- **SSH:** one-shot, non-interactive, `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 …`. Exit 255 is a network drop, not a job failure: reconnect and re-read the log. (A01 uses no `docker compose`; if it did, `docker compose run` would need `-T`.)
- **Long jobs:** detached `nohup` launches; launchers and logs in `~`, never in root-owned run folders.
- **Code delivery:** commit and push locally, then `git fetch` over SSH on the server (`GIT_SSH_COMMAND='ssh -o BatchMode=yes …'`). A01 only fetches and `git show`s files into `~`; it never checks out on the server.

- [ ] **A01-T11** [type: run] (10 min, due Wed 7 Oct, before launch) — **GPU pre-flight.**
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader && echo '--- compute apps ---' && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader && echo '--- containers ---' && docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'"
  ```
  *Done when:* 4 GPUs show < 2,000 MiB used and no compute app, and no `ro27-*`, `rogueone-*`, `llmfe-*` or `caafe-*` container is running. Write down their ids as `<ids>` (comma-separated, no spaces, e.g. `0,1,2,3`). **Otherwise STOP and report** the per-GPU memory and utilisation, the compute apps and the containers. Do not queue.
- [ ] **A01-T12** [type: run] (15 min, due Wed 7 Oct, evening) — **Launch `S1007a`** (seed 42), with t17's ID logging at the top of the job log:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 bash -s <<'EOF'
  SID=S1007a; SEED=42; GPU_IDS=<ids>; VREF=<VLLM_REF>
  JOB=$HOME/ro27_${SID}.job.log
  { echo "[$(date -u +%FT%TZ)] A01 $SID seed=$SEED gpus=$GPU_IDS"
    echo "vllm image=$VREF id=$(docker image inspect "$VREF" --format '{{.Id}}')"
    echo "rogueone image=rogue_one:nestedcv9 id=$(docker image inspect rogue_one:nestedcv9 --format '{{.Id}}')"
    echo "gpt-oss-120b refs/main=$(cat $HOME/.cache/huggingface/hub/models--openai--gpt-oss-120b/refs/main)"; } > "$JOB"
  export GPUS="$GPU_IDS" NEED=4 EAGER=0 WAIT_MAX_MIN=0 PORT=8010 MODEL=openai/gpt-oss-120b IMAGE="$VREF" \
    SERVER_NAME="rogueone-vllm-o120-$(echo $SID | tr A-Z a-z)" LOG="$HOME/vllm_session_${SID}.log" \
    SEED PARALLEL=6 CPUS=12 IMAGE_RO=rogue_one:nestedcv9 PROJECT="RogueOne_W0_T17Rep_Seed${SEED}" \
    RUNDIR="$HOME/rogueone-runs-agent2027/${SID}" REPO_TASKS="<ROGUEONE_CAIR_PATH>/tasks" SOLO=' ' \
    DATASETS='credit-g heart junglechess myocardial vehicle crab housing insurance'
  nohup bash ~/a01_vllm_session.sh ~/a01_clients_t16_rerun.sh >> "$JOB" 2>&1 < /dev/null &
  sleep 20; cat "$JOB"; tail -5 "$LOG"
  EOF
  ```
  For the add-on, put `balance-scale` first and `wine` last in `DATASETS`. *Done when:* the job log shows both image IDs equal to A01-T2/T3, and within 30 min the session log shows, in order, `session start … tp=4`, `vLLM healthy`, `seeded task` for all 8 datasets and 6 `--- start rogueone` lines. A `FATAL` line means the EXIT trap has torn the server down: re-run A01-T11 and report.
- [ ] **A01-T13** [type: run] (≈ 15 min spread over the wave; due when `S1007a` ends, expected Thu 8 Oct, early morning) — **Monitor.**
  - **Waiter**, as a background command (exit 255 means the network dropped, not the job; reconnect):
    ```
    ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 CAIR-gpu14 "while ! grep -q '=== session end rc=' ~/vllm_session_S1007a.log 2>/dev/null; do sleep 120; done; grep -E '^--- (start|done)|still failing|no first-pass|T16_RERUN_ALL_DONE|session end' ~/vllm_session_S1007a.log | tail -40"
    ```
  - **Every few hours:**
    ```
    ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 bash -s <<'EOF'
    docker ps --filter name=rogueone-S1007a- --format '{{.Names}} {{.RunningFor}}'
    docker run --rm --env-file <CAIR_HOME>/code/PlanBench/.env --entrypoint python3 rogue_one:nestedcv9 -c 'import wandb; [print(r.name, r.state) for r in wandb.Api().runs("brainflux/RogueOne_W0_T17Rep_Seed42")]'
    EOF
    ```
    For any container older than 4 h, run `docker top <name> -o pid,etime,args` (same ssh wrapper). If it shows no `rogue_one_main.py` process, only `tail -f /dev/null`, run `docker stop <name>`; the launcher then records it as failed and retries it solo. A W&B run in state `running` with no live container is a zombie: note it.

  *Done when:* the session log shows `T16_RERUN_ALL_DONE seed=42` and `=== session end rc=0`, and the line `=== still failing after solo retry: … ===` (or `no first-pass failures`) is copied into the session plan. `rc=0` alone does not mean every dataset succeeded: the session's exit code is that of the clients script, which ends with an `echo`. Each dataset listed as still failing goes to A01-T15.
- [ ] **A01-T14** [type: run] (40 min; launch due Thu 8 Oct, right after `S1007a` ends; the wave is expected to end Thu evening, and no later than Fri 9 Oct, early morning) — **Tear-down check, then `S1008a`.**
  - Run `docker ps --format '{{.Names}}' | grep -E '^(ro27-|rogueone-)' || echo none`, the `nvidia-smi` memory query, and `cat ~/.cache/huggingface/hub/models--openai--gpt-oss-120b/refs/main` (must equal the value in `~/ro27_S1007a.job.log`).
  - Then repeat A01-T11 and A01-T12 with `SID=S1008a`, `SEED=4` (this gives `SERVER_NAME=rogueone-vllm-o120-s1008a`, `LOG=~/vllm_session_S1008a.log`, job log `~/ro27_S1008a.job.log`, `PROJECT=RogueOne_W0_T17Rep_Seed4`, `RUNDIR=~/rogueone-runs-agent2027/S1008a`).
  - Monitor as in A01-T13, with `S1008a` and `…_Seed4`.

  Session IDs follow the launch date (`S<MMDD><letter>`); re-derive them if dates shift. *Done when:* before launch, `none` is printed with all GPUs < 2,000 MiB and `refs/main` unchanged; after the wave, `S1008a` ends with `T16_RERUN_ALL_DONE seed=4` and `rc=0`, and its `still failing …` line is recorded as in A01-T13.
- [ ] **A01-T15** [type: run] (≤ 1 h of developer time; Fri 9 Oct – launched by Sun 11 Oct, 18:00, and only if the box is free by then) — **Reruns.** For the datasets listed under `still failing after solo retry`, launch one follow-up session **per seed**, as A01-T11 + A01-T12 with:
  - `SID` = the next free `S<MMDD><letter>` for the launch day (e.g. `S1009a`, or `S1008b` if it is still Thu 8);
  - the same `SEED` and `PROJECT` as the failed run, `DATASETS='<failed datasets>'`, its own `RUNDIR=~/rogueone-runs-agent2027/<SID>`.

  The two seeds' reruns run **one after the other**, never together (`vllm_session.sh:121-130` refuses a second server, and the cap is 4 GPUs). A rerun runs with fewer than 6 clients, which is the same tail deviation as in §4 and is disclosed with it. *Done when:* every pair is complete, or the pair is recorded as missing in the session plan (a rerun that cannot start by Sun 11 Oct, 18:00, or cannot finish by the Mon 12 Oct, 08:00 hard stop, is recorded as missing).

### 5.4 Analysis

- [ ] **A01-T16** [type: analysis] (20 min, due Mon 12 Oct, 10:00) — **Collect** with t17's own selection rule, CPU-only, in containers:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 bash -s <<'EOF'
  set -e
  C=$HOME/a01_collect
  docker run --rm --network none --cpus 4 -v $HOME/rogueone-runs-agent2027/S1007a:/s42:ro -v $HOME/rogueone-runs-agent2027/S1008a:/s4:ro -v $C/empty:/s5:ro -v $C/w0:/out -v $C:/a01:ro --entrypoint python3 rogue_one:nestedcv9 /a01/collect_seed_sweep_t17.py > $C/w0/collect_stdout.txt
  docker run --rm --network none --cpus 4 -v $HOME/rogueone-runs-t17s42:/s42:ro -v $HOME/rogueone-runs-t17s4:/s4:ro -v $C/empty:/s5:ro -v $C/t17:/out -v $C:/a01:ro --entrypoint python3 rogue_one:nestedcv9 /a01/collect_seed_sweep_t17.py > $C/t17/collect_stdout.txt
  W='credit-g|heart|junglechess|myocardial|vehicle|crab|housing|insurance'   # add |balance-scale|wine for the add-on
  echo "--- unexpected W0 problems:"; sed -n '/PROBLEMS/,$p' $C/w0/collect_stdout.txt | grep -E "^  - ($W)[ :]" | grep -v ' seed 5: no result found' || true
  docker run --rm --network none -v $C:/a01:ro --entrypoint python3 rogue_one:nestedcv9 -c "import pandas as pd; a=pd.read_csv('/a01/t17/results_seed_sweep_t17_long.csv'); b=pd.read_csv('/a01/t17_committed_long.csv'); W=['credit-g','heart','junglechess','myocardial','vehicle','crab','housing','insurance']; m=a[a.dataset.isin(W)&a.seed.isin([42,4])].merge(b,on=['dataset','seed'],suffixes=('_new','_old')); print('t17 cells', len(m), 'max abs diff', float((m.value_new-m.value_old).abs().max()))"
  EOF
  ```
  - The collector reads all 27 datasets and seeds 42, 4, 5, so its own `PROBLEMS` list always holds `no result found` for the non-W0 datasets and for seed 5, and `only 0 seed(s)` for the non-W0 datasets. The filter above shows only lines about W0 datasets and drops only the seed-5 line, so an incomplete W0 run is never hidden. Expected output: none. Any line (`seed 42: no result found`, `only k/5 outer folds`, `inner metric missing`, `only 1 seed(s)`) marks an incomplete run.
  - The collector does not check rc or `fold_aggregation_summary.csv`; A01-T10 does (§2 complete run).
  - If A01-T15 ran, collect each rerun `RUNDIR` into `~/a01_collect/w0_<SID>/` with the same collector, one container per rerun session (`<SEED>` = the rerun's seed, `<OTHER>` = the other of 42 and 4):
    ```
    mkdir -p $C/w0_<SID>; docker run --rm --network none --cpus 4 -v $HOME/rogueone-runs-agent2027/<SID>:/s<SEED>:ro -v $C/empty:/s<OTHER>:ro -v $C/empty:/s5:ro -v $C/w0_<SID>:/out -v $C:/a01:ro --entrypoint python3 rogue_one:nestedcv9 /a01/collect_seed_sweep_t17.py > $C/w0_<SID>/collect_stdout.txt
    ```
    A01-T17 then gets it through `--rerun`.
  - Copy the W0 long CSV into the local RougeOne checkout on `agent2027/v2`: `scp -o BatchMode=yes CAIR-gpu14:a01_collect/w0/results_seed_sweep_t17_long.csv paper/results_w0_t17rep_long.csv`.

  *Done when:* the check prints `t17 cells 16 max abs diff` ≤ 1e-12, and the unexpected-problem list is empty (or every entry is accounted for as an incomplete run in the session plan).
- [ ] **A01-T17** [type: analysis] (30 min, due Mon 12 Oct, 12:00) — **Run A01-T10.**
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 bash -s <<'EOF'
  set -e
  cd <ROGUEONE_CAIR_PATH>
  GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15' git fetch git@github.com:HenrikNorkart/RougeOne.git agent2027/v2
  C=$HOME/a01_collect
  git show FETCH_HEAD:scripts/agent2027/analysis/w0_sigma_samp.py > $C/w0_sigma_samp.py
  cp $HOME/vllm_session_S1007a.log $HOME/vllm_session_S1008a.log $C/logs/
  docker run --rm --network none --entrypoint python3 -v $C:/a01 rogue_one:nestedcv9 /a01/w0_sigma_samp.py --selftest
  docker run --rm --cpus 4 --env-file <CAIR_HOME>/code/PlanBench/.env -v $C:/a01 \
    -v $HOME/rogueone-runs-agent2027/S1007a:/w0/s42:ro -v $HOME/rogueone-runs-agent2027/S1008a:/w0/s4:ro \
    -v $HOME/rogueone-runs-t17s42:/t17/s42:ro -v $HOME/rogueone-runs-t17s4:/t17/s4:ro \
    --entrypoint python3 rogue_one:nestedcv9 /a01/w0_sigma_samp.py \
    --w0-long /a01/w0/results_seed_sweep_t17_long.csv --t17-long /a01/t17_committed_long.csv \
    --t17-sd /a01/t17_committed.csv --baselines /a01/baselines.csv --logs /a01/logs \
    --w0-roots /w0/s42,/w0/s4 --t17-roots /t17/s42,/t17/s4 --out /a01/analysis --wandb
  EOF
  ```
  `set -e` stops the script if the selftest fails, so the real run never starts on a broken script. If the seed-4 session was skipped (§6 rule 5), drop `S1008a` from the `cp`, the mounts and `--w0-roots`. If A01-T15 ran, copy each rerun session log into `$C/logs/`, mount each rerun `RUNDIR` as `/w0/<SID>:ro`, and add `--rerun <dataset>:<seed>:/a01/w0_<SID>/results_seed_sweep_t17_long.csv:/w0/<SID>` for every rerun dataset. Then `scp` `a01_collect/analysis/results_w0_t17rep.csv`, `results_w0_t17rep_pairs.csv` and `results_w0_t17rep_summary.json` into `paper/` of the local checkout and commit them with `paper/results_w0_t17rep_long.csv` on `agent2027/v2` (`git add -f`, because `paper/` is ignored).

  *Done when:* the selftest passed in the image, run `ANALYSIS__w0_sigma_samp__20261012` exists in `PlanBench/RogueOne-AGENT2027` with both artifacts, and the four files are committed.
- [ ] **A01-T18** [type: analysis] (10 min, due **Mon 12 Oct, 18:00**) — **Hand-offs.**
  - **To A03-T3** (σ̂ table): σ_samp,d for each dataset with 2 complete pairs, the drift verdict (with its trigger: mean shift and/or variance inflation) and the independence flag. Datasets with fewer than 2 pairs, `balance-scale` and `wine` use the t17 SD (A03 §6).
  - **To A05-T2 / A05-T14 (E11a `g2_stats.py`):** `results_w0_t17rep_pairs.csv` (A05's rule re-scales the per-pair raw δ on `housing`, `insurance` and `crab` with A05's own s_d, so the pair table is the input), `sigma_samp_pooled_sd_reg3` with its df (the A01-scale cross-check of A05's F(3, 6) homogeneity rule), the 16-df and no-junglechess `sigma_samp_pooled_sd`, Z_v1 with CI, and the drift verdict with its trigger.
  - **To A02:** the median `wall_s` over the W0-8 runs (the `W_ref` fallback, "planning only"; the wave tail runs fewer than 6 clients, so it is a lower bound for a 6-client run).

  *Done when:* the σ̂ inputs are committed next to the M1 rule (A03-T3), the pair table and the A05 values are in `a01-results` and the summary JSON, and the session plan is final.

### 5.5 Paper outputs

- [ ] **A01-T19** [type: writing] (0.1 d, due Fri 16 Oct, 12:00, when A05 expects the `agent2027-prereg` tag to be pushed) — In X10 (`docs/agent2027/preregistration.md`), record A01's results as **known pre-freeze inputs**: σ_samp, Z_v1 with CI, the drift verdict, and how they enter the M1 rule and the A05 pooling rule. *Done when:* this is in the dated `agent2027-prereg` tag.
- [ ] **A01-T20** [type: writing] (0.25 d, numbers Nov 1–3, text due Fri 6 Nov) — Write:
  - one protocol sentence, the §3 hoped, null, partial or opposite sentence, filled with the CI and worded as §6 rule 3 allows;
  - a threats-to-validity line on live-search drift and the v1-to-v2 transfer;
  - an appendix table with t17 SD, σ_samp,d, ρ_d and the δ's for each dataset.

  *Done when:* all three are in the AGENT draft, and their numbers trace to `ANALYSIS__w0_sigma_samp__*`.

## 6. Decision rules & gates

These rules are pre-declared here on Wed 7 Oct, before any W0 result exists, and are copied into X10. They are applied in this order.

1. **Usable data** (complete pairs over W0-8).
   - ≥ 12 complete pairs: full analysis.
   - 8–11 pairs: analysis, with the df stated.
   - Fewer than 8 pairs: A01 is "failed", and A03 uses the t17 SD.
2. **Drift** (checked before the verdict, on W0-8 pairs only). Wilcoxon p < 0.05 (mean shift) **or** Z_v1 > F_{0.975}(ν1, ν2) (variance inflation; `var_inflation_threshold`, ≈ 5.7 at the full design, computed from the committed t17 SDs and the pair pattern) → the opposite branch of §3: no pooling in A05, and A03 falls back to the t17 SD. The Z_v1 verdict is then reported but not used. The nominal 16/16-df threshold (2.76) is reported only as a sensitivity line.
3. **Verdict** (descriptive, no Holm), on the pooled share: Z_v1 ≤ 0.36 → hoped; Z_v1 ≥ 0.72 → null; otherwise partial. Always reported with the CI, and alongside the per-dataset ρ_d and the counts of datasets with ρ_d ≤ 0.6 and ≥ 0.85.
   - **Wording rule.** The category is the point-estimate category. The paper states it without hedge only if the CI excludes the opposite category (upper bound < 0.72 for "hoped", lower bound > 0.36 for "null"). Otherwise it says the data are "consistent with" the category and names the thresholds the CI spans. With the §3 operating characteristics this hedge will usually be needed.
4. **Independence.** If ≥ 3 of the complete pairs have `pair_identical` = true, σ_samp is reported as a **lower bound**. A03 then uses max(A01 σ_samp,d, t17 SD, 0.1 × s_d), its recommended rule anyway.
5. **Kill dates.**
   - `S1007a` must start by **Fri 9 Oct, 18:00**; otherwise A01 is skipped and A03 uses the t17 SD.
   - `S1008a` must start by **Sun 11 Oct, 08:00**; otherwise only seed 42 is used (8 pairs, 8 df).
   - Reruns must be launched by **Sun 11 Oct, 18:00**, which is possible only if the box is free by then. A rerun that cannot start by then, or the second seed's rerun that cannot finish by the hard stop, is recorded as a missing pair.
   - **Mon 12 Oct, 08:00 is the GPU hard stop**, so the box is free for the G0a smoke. `docker stop` anything of A01's still running and analyse what is complete.
   - **σ̂ table by Mon 12 Oct, 18:00**; otherwise A03 uses the t17 SD.

## 7. Risks & cut levers

| # | Risk | Mitigation |
|---|---|---|
| R1 | `nestedcv9` or t17's vLLM image is gone | A01-T2/T3 STOP. A rebuild is not a replicate, so A03 uses the t17 SD. |
| R2 | Live Serper drift: W0 measures sampling plus 2.5–3 weeks of search drift (t17 seed 42 ran 15–16 Sep, seed 4 ran 21–22 Sep) | Treat σ_samp as an upper bound on pure sampling; both drift tests; disclose |
| R3 | CAIR GPUs held by other tenants | Pre-flight STOP; launch at night; kill dates in §6 |
| R4 | HF cache updated, the Hub `main` moved (the server starts online), or A04 downloads touch gpt-oss-120b | A01-T5 STOP; `refs/main` logged at each launch and re-checked in A01-T14; A04's downloads must leave `openai/gpt-oss-120b` alone until A01 ends |
| R5 | Same LLM seed as t17, so some calls are byte-identical | Independence check; lower-bound rule |
| R6 | Lighter load in the wave tail than in t17 | Disclosed; t17's dataset order is kept |
| R7 | 2 df per dataset | A03 uses max(A01, t17, floor); the verdict is pooled (nominal 16 df, but only ≈ 6 effective df on each side because two datasets carry 79% of the variance), per-dataset ρ_d is descriptive only |
| R8 | Plain accuracy (v1) vs balanced accuracy (v2) | Disclosed; A05 pools regression datasets only (A05 OQ1) |
| R9 | junglechess dominates the s_d-scaled pooled σ | A05 uses the regression-only value (6 df), so junglechess does not enter its F-test; the 16-df value is also reported without junglechess; raw-scale Z_v1 is the primary share, but it is dominated by `junglechess` and `insurance` (79% of the t17 variance), so Z_v1 without `junglechess` and the per-dataset ρ_d are reported next to it |
| R10 | Dead container hangs a wave (`docker-entrypoint.sh:21`) | `docker top` check in A01-T13 |
| R11 | Developer time: ≈ 6.5–7.5 h before Mon 12 (T1–T18), against ≈ 2 h in the breakdown and in 01 §6.2, in a week where E-work is already over-committed | T10 is written Sat 10 (weekend, D1); the G0a work comes first; ops tasks are short and batched around launches |
| R12 | A `.dev.env` baked into the image overrides the launcher (`config.py:24`; tracked at `07cbf93` with `LLM_PORT=8000`) | A01-T6 effective-value probe; STOP if any launcher key is overridden |
| R13 | Each run posts to the hard-coded webhook (`main.py:1606`) | Unchanged old-image behaviour; accepted for a byte-identical replicate; X8 fixes v2 |
| R14 | The launcher scripts and commits `07cbf93` / `1bdaf42` are not on GitHub (checked 2026-10-07), and `agent2027/v2` does not exist yet | A01-T4 step 0 pushes `nested-cv-rebuttal` before A01-T12; A01-T9 creates `agent2027/v2` if E1 has not; STOP if the server key has no GitHub access |
| R15 | `.gitignore` ignores `*.csv` (the manifest) and `paper/` (the result files), so evidence can end up uncommitted without a warning | `git add -f`; the Done-when checks use `git ls-files` |

**Cut levers.** A01 is outside the window and outside the CL1–CL7 cut order; it is not on the breakdown's "Never cut" list, and its §6 kill dates already say what each loss costs. Its own levers, in order:
1. drop the optional `balance-scale` and `wine` rows (−4u);
2. seed 42 only (−8u; 8 pairs, 8 df).

## 8. Definition of done

- [ ] `nested-cv-rebuttal` pushed (`1bdaf42…` on GitHub). D9 recorded. Image IDs, `<VLLM_REF>`, the weights snapshot and Hub sha, the image env and effective values, W&B access and the task md5s recorded in the session plan (A01-T1 to T8).
- [ ] `manifests/agent2027/A01_W0.csv` and `docs/agent2027/sessions/a01_session_plan.md` committed on `agent2027/v2` (`git ls-files` lists both).
- [ ] `S1007a` and `S1008a` ended with `rc=0`, their `still failing …` lines are recorded, and the tear-down was clean. Reruns done or missing pairs recorded.
- [ ] `paper/results_w0_t17rep_long.csv`, `paper/results_w0_t17rep.csv`, `paper/results_w0_t17rep_pairs.csv` and `paper/results_w0_t17rep_summary.json` committed (`git add -f`). The t17 re-collection matches the committed CSV.
- [ ] W&B run `ANALYSIS__w0_sigma_samp__<date>` exists with `a01-results` and `a01-run-outputs`.
- [ ] σ̂ inputs handed to A03-T3 by Mon 12 Oct, 18:00. Pooled values handed to A05 / E11a before G2.
- [ ] Pre-registration entry (Fri 16 Oct). Paper sentence, threats line and appendix table (Nov 4–6).

---

**New names introduced** (not in the 01 §7 registry):
- **CAIR files and run dirs:**
  - run dirs `~/rogueone-runs-agent2027/S1007a` and `S1008a` (and rerun `<SID>`s), in the legacy layout `tasks/<task_type>/<dataset_id>/` instead of `<row_id>/`;
  - legacy client container names `rogueone-S1007a-<dataset_id>`, derived at `clients_t16_rerun.sh:75-76`;
  - script extracts `~/a01_vllm_session.sh` and `~/a01_clients_t16_rerun.sh`;
  - the work dir `~/a01_collect/` (`empty/`, `w0/`, `w0_<SID>/`, `t17/`, `logs/`, `analysis/`, `baselines.csv`, `t17_committed.csv`, `t17_committed_long.csv`, `collect_seed_sweep_t17.py`, `w0_sigma_samp.py`).
- **Repo files:**
  - `scripts/agent2027/analysis/w0_sigma_samp.py`;
  - `docs/agent2027/sessions/a01_session_plan.md` (the session plan);
  - `paper/results_w0_t17rep_long.csv`, `paper/results_w0_t17rep.csv`, `paper/results_w0_t17rep_pairs.csv`, `paper/results_w0_t17rep_summary.json`.
- **W&B:** artifacts `a01-results` and `a01-run-outputs`.
- **Placeholders:** `<VLLM_REF>`, `<ids>`.
- **Analysis columns:**
  - `sigma_samp_pooled_sd` (pooled σ_samp on the s_d scale; 16-df, no-junglechess) and `sigma_samp_pooled_sd_reg3` (regression only, 6 df, for A05);
  - `rho_d` and `samp_share_v1` (Z_v1, with `z_v1_ci_low`, `z_v1_ci_high`, the effective df `z_v1_nu1` and `z_v1_nu2`, and the nominal-df versions `z_v1_ci_low_nominal`, `z_v1_ci_high_nominal`);
  - `drift_p`, `var_inflation` and `var_inflation_threshold`;
  - `pair_identical` and `tester_identical_share`.
