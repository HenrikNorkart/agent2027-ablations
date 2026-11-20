# A08 · M5 — Brute-force CPU screen (OpenFE / AutoFeat / base) and LLM-FE / CAAFE comparators (C1b)

*Rogue One v2 design study, ICSE AGENT 2027 (submit Fri 13 Nov 2026), scope T1-minimum. Written Wed 7 Oct 2026. Code refs: RougeOne `nested-cv-rebuttal` @ `07cbf93` (`C:/code/PHD/RougeOne`), read line by line (re-verified Wed 7 Oct). The branch tip is now `1bdaf42` (Finding 43). Of the files A08 cites, it changed only `paper/scripts/classicfe_arm.py` (+8 lines after line 87, +12 net in `generate_autofeat`, +5 in the result record: lines 88–300 shift +8, 304–361 shift +20, 362+ shift +25); both line numbers are given for that file. It also added `paper/results_seed_sweep_t17.csv` and inserted 233 lines into `paper/reviewer_feedback/CAMERA_READY_EXPERIMENT_PLAN.md` before its line 4600 (both numbers given where cited). Its `classicfe_arm.py` changes are inert at default env (`AUTOFEAT_NJ` defaults to `NJ`; `AUTOFEAT_VERBOSE=0`). R8 branches `agent2027/v2` off `07cbf93`, so X3 patches the `07cbf93` file unless the branch is rebased onto `1bdaf42`; then use the HEAD numbers. Short paths: `main.py` = `rogue_one_main.py`; `tst/agent.py` = `rogueone/llm/agents/core/tester/agent.py`; `config.py` = `rogueone/utils/config.py`. Shared doc = `docs/ablations/01_shared_infrastructure.md` ("01 §n"). Every "hoped" result is a hypothesis, not a finding.*

| Field | Value |
|---|---|
| ID | **A08 · M5** · arms `base_no_fe`, `openfe`, `autofeat` (CPU); `llmfe`, `caafe` (GPU) |
| Claim / Holm contrast | (1) **Brute-force gate** (gate `M5S`): an instrument-validity check for every recovery contrast. (2) **C1b** = LLM-FE − Full on `rec_sel`, CAAFE − Full secondary. Two-sided, in the Holm family of 10 (smallest threshold 0.005). System-level only. |
| Tier | T1-minimum. The CPU screen is on the never-cut list; CAAFE is cut lever **CL4**. |
| Owner | Henrik |
| Start / finish | **CPU screen:** launch Mon 12 Oct, ~19:00 (after X3, due 18:00); read Tue 13 Oct, 08:00–10:00 (`M5S`), before A02 launches. Labels final by Thu 15; in X10 on Fri 16. **GPU:** one O120 session, Tue 20 – Sat 24 Oct. Retries by `DF` (Sat 31 Oct, 24:00). C1b final Nov 1–3. |
| Budget | **CPU:** 81 cells, 0u, about 6–12 h overnight at 64 of the box's 224 cores (*est.*). **GPU:** 22 runs budgeted (20 at the proposed task types) × 0.45–0.7u (*est.*, unmeasured) = **9.9–15.4u** ≈ 7–11 h at 1.4u/h ≈ 28–44 H100-h at TP4 (14–22 at TP2, if D5's TP2 measurement shows the same per-server throughput). No new Full runs. Same numbers as the T1-minimum card and R1. |
| Models | CPU: none. GPU: O120 `openai/gpt-oss-120b` on `vllm/vllm-openai:v0.22.0`, port `8010`, X5 token proxy on `8090`; TP4, or TP2 per D5; ≤ 6 clients. |
| Status | Not started |
| Depends on | E8a (+ detector), X3, E1 (`make_manifest.py`, `ro27_clients_manifest.sh`), G0a; D4 (Thu 15); X10; A05 SYN-P runs and G2 (Mon 19); D5; X1 (`MAX_MODEL_LEN`); X4, X5 (Mon 19); E3 energy sidecar; A06 (sentinel row for the GPU session); X6 (Sat 24); E11b (Tue 27) |
| Feeds into | `M5S` → A02 launch; the exclusion list of every confirmatory recovery population; `base_no_fe` held-out score and fold SD per SYN-P task × seed (A02 and A05 `success`; A06 s_task at seed 42); A15 (C4b dropped if P6 is solved); A14 (P7 validity); X10; C1b in Holm (R14) and Table 2; A21 / A23 points; A16 (C1b reading); Fig. 2 (right) · A09, A10, A11, A12, A17, A19, A20, A22 (CPU-screen brute-force labels and the exclusion list) |

## 1. What we test and why it matters

**Screen.** On each planted-feature SYN-P task, do brute-force tools with no LLM (OpenFE, AutoFeat) already find the planted feature φ? A task they solve is labelled *non-discriminating* and leaves every confirmatory recovery population. Without this, a recovery contrast could rest on features a mechanical search finds anyway.

**Comparators.** On the same gpt-oss-120b backbone, does Full-v2 recover-and-select more planted features than two published single-LLM loops: LLM-FE (primary) and CAAFE (secondary)? With E10 cut, this is C1's only external comparison. The comparators also differ in web search, selection rules, budgets and temperature, so C1b is a **system-level** claim; the mechanism half of C1 rests on C1a (A16).

## 2. Background you need

**Rogue One parts touched.** A08 runs no Rogue One code; its control is A05's `full` runs. What matters is the outer partition and the evaluator, because comparators must share both:
- **Folds.** `main.py:305-311` calls `deterministic_kfold_indices` (`rogueone/utils/train_test_splitter.py:199`). This is `StratifiedKFold(5, shuffle, fold_seed).split(zeros, y)` or `KFold(5, shuffle, fold_seed).split(range(n))` (`:296-303`), over `data.csv` row order.
  - Stratification applies only to classification and only when `ROGUE_ONE_STRATIFIED_FOLDS=True` (`main.py:255-281`; flag at `config.py:323`, code default `False`). Labels are aligned to `data.csv` row order through `id`; a class with < 5 members falls back to KFold, as in classicfe (`train_test_splitter.py:257-264`). Under D4 that flag is `True` for SYN-P classification tasks, so M0's seed-42 partition equals LLM-FE's.
  - If raw rows contain exact duplicates, a group-aware splitter is used instead (`main.py:299-303`; splitter `:279-295`), which would break partition equality.
  - Fold state resets at each fold (`main.py:417-457`), so the unit is the **fold-episode**.
- **Evaluator.** XGBoost, 400 trees (`tst/agent.py:1673` regression, `:1815` classification), `random_state = fold_seed` (`:1671`, `:1813`). With `ROGUE_ONE_DO_BALANCING=True` (`config.py:312`) the fit uses balanced weights and the score is balanced accuracy (`tst/agent.py:1861-1878`). The comparator scorers pin XGBoost `random_state=42` at every fold seed (classicfe `XGB_PARAMS`, `classicfe_arm.py:91` / HEAD `:99`; LLM-FE scorer `:81`, `:100`). This touches held-out scores at seeds 4 and 5 only, never recovery; disclose it.

**Comparator scripts** (existing, R6; all CPU containers except LLM calls):

| Arm | Script / image | Facts that matter |
|---|---|---|
| `base_no_fe`, `openfe`, `autofeat` | `paper/scripts/classicfe_arm.py` (CLI arm names `raw`, `openfe`, `autofeat`, `:312` / HEAD `:332`); `classicfe_launch.sh`; image `classicfe:4` (`docker/Dockerfile.classicfe:29-30`: OpenFE 0.0.12, AutoFeat 2.1.3) | Folds hard-code `random_state=42` (`:119-135` / HEAD `:127-143`). Only scores are written, not features (`:352-387` / HEAD `:372-412`); per-cell JSON keeps fold mean and SD (`<key>_sd`, ddof 0), not per-fold scores (`:372-381` / HEAD `:397-406`); output name has no seed (`:383-385` / HEAD `:408-410`). OpenFE keeps **every** feature it returns (`:221-229` / HEAD `:229-237`), the most generous reading. AutoFeat `feateng_steps=2` (`:301` / HEAD `:317`); **AutoFeat 2.1.3 has no seed** (Finding 43, commit `1bdaf42`, `paper/reviewer_feedback/CAMERA_READY_EXPERIMENT_PLAN.md` HEAD `:1474`), so each AutoFeat cell is one unseeded draw. `evaluate()` fits unweighted (`acc_*`) and balanced-weight (`accw_*`) (`:144-180` / HEAD `:152-188`); `accw_balanced` matches Full-v2. Launcher: `BASE=$HOME/classicfe_v3` hard-coded (`:23`, `:97`), dataset heredoc (`:45-73`), container name without seed (`:104`), `.err` file without seed (`:110`), output path `$OUT/arm_<ds>_<arm>_<FOLDS>.json` (`:117`, `:131`), `--cpus=4 --network none`, `NJ=4 OPENFE_NJ=1` (`:118-119`), defaults `CELL_TIMEOUT=10800`, `PARALLEL=16` (`:36-37`), log `$BASE/run_<TAG>.log` ending `ALL_CELLS_DONE` (`:28`, `:147`), `ONLY`/`ARMS`/`TAG`/`OUT_SUB` filters (`:18`, `:22`, `:27`, `:86`). |
| `caafe` | `paper/scripts/caafe_arm.py`; image `caafe:oss120` (`docker/Dockerfile.caafe`) | Classification only (`:117-122`, `:326-335`). CAAFE's prompt, temperature 0.5 and 10 iterations untouched (`:25-29`, `:277`). Gets the same `context_and_goal` string (`:125-141`). Imports `build_folds`/`evaluate` from classicfe (`:105-111`; call `:337`). Saves surviving code per fold, `record['generated_code']` (`:398`); flags `EMPTY_GENERATION` (`:412-422`). Budget and stop mode come from env (`:312-319`; upstream defaults 500 tokens and `server`, applied in `caafe_upstream.patch`); the T0.4 sweep measured **2048 tokens, stop mode `client`** (`paper/reviewer_feedback/CAMERA_READY_EXPERIMENT_PLAN.md:4600` / HEAD `:4833`). Output `<dest>/arm_<ds>_caafe_<folds>.json`, no seed in the name (`:292`). `clients_caafe.sh` hard-codes `--folds kfold` (`:247`) and its dataset list (`:83-103`), and gates on stop-mode and budget probes (stages 0–2), so A08 runs its own cell loop with the T0.4 values pinned. Its cell runner sets `--cpus 6` and `OMP/OPENBLAS/MKL_NUM_THREADS=4` (`:70`, `:239-241`); A08 copies these. |
| `llmfe` | `paper/scripts/llmfe_launch.sh` (image `llmfe:oss120`; checkout `~/llmfe` on CAIR); scorer `llmfe_score_rogueone.py` (image `llmfe:eval`) | **LLM-FE's code is never changed** (decision of 2026-08-28). Generation splits with its own `StratifiedKFold(5, shuffle, 42)` (classification; regression split *to verify*), so it runs only at seed 42. Launcher: `THREADS=8` (`:26`), `LLMFE_MAX_TOKENS=2048` (`:61`), ~7% crash rate at 6 clients (`:19-22`), `/health` check (`:45-49`), no timeout (`:51-77`), `--one <ds>` mode (`:79-82`), which **always exits 0**: the real rc is only in its stdout line `--- done <ds> rc=<rc>` (`:72`), so A08 parses that line; reads `~/llmfe/specs/specification_<ds>.txt` and logs to `~/llmfe/logs/<ds>_<SUFFIX>_split_<i>/` (`:69-70`). Scorer: data `./data/<ds>.csv`, target last (`:114-115`); `REGRESSION` set hard-coded, no SYN-P IDs (`:72-73`); `notebook` mode reproduces an indexing defect that makes its "top 3" near-arbitrary (`:160-186`, `:197-211`); `best3` takes the true top 3 (`:187-196`); records `candidates_used` per split (`:329-332`); fits **unweighted** (`:267`) and reports balanced accuracy (`:321-324`); output name omits the LLM suffix (`:434-436`); it imports LLM-FE's `optimization_utils` and reads `./data` and `./logs/<ds>_<llm>_split_<i>/samples` relative to the working directory (`:64`, `:158`), so it runs in `llmfe:eval` with `~/llmfe` as working directory. Images: `docker/Dockerfile.llmfe_eval` is `FROM llmfe:oss120`, so build `llmfe:oss120` first. |

**Instruments** (built by E8a; n = 2,000, p ≈ 20–30; types are proposals, `synp_meta/registry.yaml` is binding):

| Task | φ | Type | Role here |
|---|---|---|---|
| `synp_p0` | log x_a | R | screen only; detector control (recovery 0 by construction) |
| `synp_p1` | x_a / x_b | R | expected non-discriminating |
| `synp_p2` | (x_a / x_b) · x_c | R | |
| `synp_p3` | 1[x_a / x_b ≥ t] ∧ cat = k | C | multi-step |
| `synp_p4` | sin/cos of an hour-like column | C | |
| `synp_p5` | sign(LW·LD² − RW·RD²) | C | multi-step |
| `synp_p6a` / `synp_p6b` | named composite / exact anonymised twin | R | C4b pair |
| `synp_p7` | composite documented only in a local corpus | C | screen only (P7 policy) |

- Screen = **SYN-P 8 + p0** (9 tasks). Comparators = **SYN-P 7** (p1–p6b). CAAFE covers only SYN-P 7's classification tasks (p3, p4, p5 as proposed).

**Metrics** (R13). `rec_any` / `rec_sel` per fold-episode, from `scripts/agent2027/analysis/recovery_detector.py` (E8a): a feature counts if |Spearman| ≥ 0.95 with φ on the fold's training rows, or NMI ≥ 0.9 for P3, P5, P7; raw columns and single-column monotone transforms never count. "Selected" per system:
- **Full:** the inner-CV-selected iteration.
- **OpenFE / AutoFeat:** every feature passed to XGBoost, so `rec_sel = rec_any`.
- **LLM-FE:** the split's `candidates_used` under `notebook` (primary) or `best3`; **any-program** = every program with a non-null score (selection-free, compared with Full's `rec_any`).
- **CAAFE:** the surviving code; `rec_any` is not observable.

Cost: `tok_compl_total`, `tok_prompt_total`, `llm_calls_total` (X5 proxy), `wall_s`, `energy_j`. Secondary: `heldout_score`.

**Brute-force label** (R10): non-discriminating = OpenFE **or** AutoFeat, judged separately, recovers φ in **≥ 14/15** episodes (5 folds × seeds {42, 4, 5}).

## 3. Hypotheses and interpretation (all hypotheses)

| | Screen | C1b |
|---|---|---|
| **Hoped** | Only P1 is non-discriminating. | Full recovers more depth-2/3 and P5 features than LLM-FE and CAAFE, or the same with fewer tokens. *"On the same gpt-oss-120b backbone, Rogue One recovers X of N discriminating planted features, against Y for LLM-FE and Z for CAAFE; brute force recovers only the depth-1 ratio."* |
| **Null** | Nothing solved: no exclusions; report that P1 was harder than expected. | Inconclusive: "no detectable recovery advantage over single-LLM loops at 120B" with the CI. C1 rests on C1a plus cost (tokens per recovered feature); the pre-written fallback narrative applies. |
| **Opposite** | ≥ 4 of 7 families solved: recalibrate SYN-P before A02 (§6). | A comparator recovers more. Two-sided, so it is published, and the paper argues per-role allocation (C5), cost and bounded context. Selection modes disagreeing in sign → "inconclusive (selection-dependent)". |

**The opposite is at least as likely.** Matched-harness plain accuracy (`paper/results_matched_harness_baselines.csv`, seed-42 kfold partition): CAAFE 0.927 on junglechess against t17's 3-seed mean 0.891 (but t17 seed 42 = 0.971, `paper/results_seed_sweep_t17.csv`, present from `1bdaf42`); LLM-FE 0.800 on myocardial against 0.732. These are plain accuracy on real data, not recovery, so they set no expectation for C1b's size, only its sign uncertainty. *To verify before citing:* those LLM-FE rows were scored with `--folds kfold` on programs generated on stratified splits.

## 4. Design

**Control** (no new runs): A05 `full` on SYN-P 7 at seed 42 `replicate` 0 and 1 (the A/A, never cut) and seed 4 `replicate` 0.

**Fixed settings.**
- Comparators run as published: LLM-FE at its native sample budget (the default in `~/llmfe/main.py`; A08-T4 records the value) and `LLMFE_MAX_TOKENS=2048`; CAAFE at temperature 0.5, 10 iterations, `n_splits=10`, `n_repeats=2`, internal evaluator `logistic` (`caafe_arm.py:277-284`), budget 2048 and stop mode `client` (pinned to T0.4's measured values). OpenFE `OPENFE_NJ=1`; AutoFeat `feateng_steps=2`, `n_jobs=4` (`NJ=4`; `AUTOFEAT_NJ`, `AUTOFEAT_VERBOSE` unset).
- Folds `stratified` for classification (D4). For regression, classicfe and the LLM-FE scorer fall back to `KFold(5, shuffle, seed)` (`classicfe_arm.py:135` / HEAD `:143`, scorer `:139-142`); LLM-FE's own generation split for regression is *to verify* (A08-T4). CAAFE has no regression path.
- Same task text: CAAFE reads `context_and_goal`; LLM-FE gets it through the X4 spec.
- One O120 server, ≤ 6 concurrent clients, same vLLM image and serve flags as A05 (`vllm/vllm-openai:v0.22.0`, `MAX_MODEL_LEN` at its X1 default 131072); 6 h cap per run or cell.
- **Session.** The GPU part is its own session `<S>` = `S<MMDD><x>` (R8), planned with `make_manifest.py --plan-session <S> --rows A08-082…A08-10x [A06 row]`. It is not co-hosted with A10/A11, because comparator rows cannot run under `ro27_clients_manifest.sh`. It qualifies for one A06 `sentinel` row (A06 §4.5), which runs in the same session through `ro27_clients_manifest.sh` (A08-T12).

**Run matrix.**

| Arm | Instruments | Seeds / replicates | Runs | u | W&B run name |
|---|---|---|---|---|---|
| `base_no_fe` | SYN-P 8 + p0 | fold seeds {42, 4, 5} | 27 cells | 0 | `A08__base_no_fe__<dataset_id>__f<seed>__r0` |
| `openfe` | same | same | 27 cells | 0 | `A08__openfe__<dataset_id>__f<seed>__r0` |
| `autofeat` | same | same | 27 cells | 0 | `A08__autofeat__<dataset_id>__f<seed>__r0` |
| `llmfe` | SYN-P 7 | seed 42; `replicate` 0, 1 | 14 | 6.3–9.8 | `A08__llmfe__<dataset_id>__f42__r<0/1>` |
| `caafe` | SYN-P 7, classification (3 proposed, ≤ 4 budgeted) | fold seeds {42, 4} | 6 (≤ 8) | 2.7–5.6 | `A08__caafe__<dataset_id>__f<seed>__r0` |
| **Total** | | | **81 CPU + 20–22 GPU** | **9.9–15.4** (9.0–14.0 at 20) | |

**Budget arithmetic.** CPU 3 arms × 9 tasks × 3 seeds = 81 cells, 0u. GPU 7 × 2 + 2 × #C (#C = classification tasks in SYN-P 7; 3 as proposed). The T1-minimum card and 01 §4.9 budget 14 + 8 = 22 runs, i.e. #C = 4, which covers a fourth classification task if `registry.yaml` makes one. LLM-FE: 14 × 0.45–0.7 = 6.3–9.8u. CAAFE: 6 × 0.45 = 2.7u to 8 × 0.7 = 5.6u. Total 22 × 0.45 = 9.9u to 22 × 0.7 = 15.4u, identical to the breakdown; 9.0–14.0u at 20 runs. Wall clock 9.9–15.4u ÷ 1.4u/h (2 u/h × 0.7) ≈ 7–11 h of the 4-GPU box. The per-run 0.45–0.7u is an estimate. LLM-FE wrote about 20–25 programs per split in T0.4 (*est.*; e.g. ≥ 22 sample files on crab split 5) at ≤ 2,048 tokens, so its real u may be below 0.45: measure it on the first task and record it in the session plan (the GPU part runs after G1, so the G1 re-tier uses the budgeted range).

**C1b populations** (fixed in X10).
- **Primary:** discriminating SYN-P 7 tasks (≤ 6) on the seed-42 partition: Full `r0`+`r1` vs LLM-FE `r0`+`r1`, paired by fold (10 episodes per task per arm).
- **CAAFE (secondary, outside the claim rule):** discriminating classification tasks × {42, 4}; Full contributes seed 42 `r0`, `r1` and seed 4 `r0`.

**Randomisation.** Within the GPU session, run order is the session file's `order_key`, which `make_manifest.py --plan-session` draws with an RNG seeded from the session ID. The only exception: the first CAAFE cell runs first, as the `EMPTY_GENERATION` gate. The two LLM-FE replicates of a task are interleaved, never adjacent.

**Failure handling.**
- **CPU cell missing** (no JSON, or `folds_done` < 5): rerun once alone with `CELL_TIMEOUT=21600`. If still missing, that method labels the task non-discriminating only if it misses ≤ 1 of ≥ 10 completed episodes.
- **GPU run crash** (LLM-FE: `--- done <ds> rc≠0`, or fewer than 5 `split_<i>/samples` dirs; CAAFE: no JSON or `folds_done` < 5): retry once, alone, at session end. Persistent failure or cap hit (`timeout` rc 124) = **ITT**: all its episodes get `rec_sel = rec_any = 0`.
- **LLM-FE split where every selected program fails** (NaN in the scorer): `rec_sel = 0` for that episode under that mode.
- **CAAFE `EMPTY_GENERATION`** means the backbone was not heard (setup fault): rerun; if persistent, write an abstention row, exclude it from the CAAFE estimate and disclose.

## 5. Tasks

### 5.1 Prerequisites (verify before launch)

- [ ] **A08-T1** [type: data] (0.1 d, due Mon 12 Oct, 18:00) — Verify E8a for the screen. *Done when:* on CAIR the 9 folders `<ROGUEONE_CAIR_PATH>/tasks/<task_type>/synp_<id>/` exist with `data.csv` hashes equal to the local build; `synp_meta/registry.yaml` fixes the types; φ metadata sits only in `~/rogueone-synp-meta/`; no `data.csv` has exact-duplicate rows (`id` excluded); the detector's unit tests pass.
- [ ] **A08-T2** [type: code] (X3, 0.25 d, due Mon 12 Oct, 18:00) — `classicfe_arm.py` gains `--fold-seed S` (passed to `build_folds`, default 42) and `--dump-features DIR`: per fold, `<dataset_id>_<arm>_fold<k>.csv.gz` with `id` plus every generated feature on **training rows**, and a JSON of the feature expressions. Defaults must reproduce T0.4 cells. *Done when:* for every SYN-P task at seeds 42, 4, 5 the test-index sets equal `deterministic_kfold_indices` (stratified for C, KFold for R). Each side is computed inside its own image, because the scikit-learn pins differ (classicfe:4 for `build_folds`, `rogue_one:agent2027-pre.N` for `deterministic_kfold_indices`), and the sets are compared in a container. The dumps exist for one OpenFE and one AutoFeat cell. A `raw` cell at seed 42 reproduces its T0.4 `accw_balanced` on one Core-10 task.
- [ ] **A08-T3** [type: decision] (0.05 d, due Thu 15 Oct) — Henrik decides **D4**: notebook primary with best3 and any-program as sensitivity analyses; `ROGUE_ONE_STRATIFIED_FOLDS=True` for SYN-P classification in A05; authoring LLM-FE specs. *Done when:* written into the draft pre-registration. If stratification is rejected, C1b on classification tasks loses fold pairing (`(1|task)` only) and says so.
- [ ] **A08-T4** [type: code] (X4 + X5, 0.5 d, due Mon 19 Oct) — **X4:** specs for `synp_p1`…`synp_p6b` under `scripts/agent2027/llmfe_specs/synp_<id>/`, each derived from an upstream spec of the same task type with only the description replaced by `context_and_goal`; data `~/llmfe/data/synp_<id>.csv` in `data.csv` row order, `id` dropped, target last, written by a container (`--network none`, image `rogue_one:agent2027-v2.N`; the host only copies files). Specs go to `~/llmfe/specs/specification_synp_<id>.txt`. Read `~/llmfe` (read-only) to confirm: the regression split; that the `API_KEY` env reaches the server as the bearer token; and LLM-FE's default sample budget per split (record it in the X10 comparator settings). **X5:** proxy forwards `/health` and attributes tokens by the client's bearer key, forwarding with the server key (`$API_KEY` exported by `ro27_vllm_session.sh`). *Done when:* a 1-task LLM-FE smoke writes `samples_*.json` under `~/llmfe/logs/<ds>_<SUFFIX>_split_<i>/samples/`; its split equals A05's seed-42 partition (scorer split in `llmfe:eval` vs `deterministic_kfold_indices` in the rogue_one image); proxy totals match vLLM's request log within 1% on a 10-call test (A23-T5's threshold).
- [ ] **A08-T5** [type: data] (0.05 d, due Mon 19 Oct, G2) — *Done when:* W&B group `A05` holds the 21 SYN-P 7 runs (seed 42 `r0`, `r1`; seed 4 `r0`; rows in A05-031…054) as `finished` or ITT, with per-episode `rec_sel`/`rec_any` in the `a05-handoff` episode table and non-zero E3 tokens, and G2 reads "go".
- [ ] **A08-T6** [type: config] (0.05 d; `classicfe:4` by Mon 12, 18:00; the rest by Mon 19) — `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "for i in classicfe:4 caafe:oss120 llmfe:oss120 llmfe:eval; do docker image inspect \$i --format \"\$i {{.Id}}\" || echo MISSING \$i; done"`. Rebuild any missing one from `<P>`: `docker build -t classicfe:4 -f docker/Dockerfile.classicfe .`, `docker build -t caafe:oss120 -f docker/Dockerfile.caafe .`, `docker build -t llmfe:oss120 -f docker/Dockerfile.llmfe .`, then `docker build -t llmfe:eval -f docker/Dockerfile.llmfe_eval .` (it is `FROM llmfe:oss120`). Never push these images. *Done when:* four image IDs are in the session plan.

### 5.2 Implementation (all patches keep existing defaults)

- [ ] **A08-T7** [type: code] (0.1 d, due Mon 12 Oct, 18:00) — `classicfe_launch.sh`: `BASE` overridable and exported to the cell script (`:23`, `:97`), so A08 never touches `~/classicfe_v3` (T0.4 outputs, the Finding 43 probe); `DATASETS_FILE` replaces the heredoc (`:45-73`); `FOLD_SEEDS` expands jobs to dataset × arm × seed, passing `--fold-seed`, adding the seed to the container name (`:104`) and `.err` file (`:110`), and writing to `$OUT/f<seed>/` (the missing-cell check `:117`, `:131` follows); `DUMP_FEATURES=1` passes `--dump-features /out/features` (host `$OUT/f<seed>/features/`); the new variables are exported to the cell script next to `:144`. Unset, every new variable reproduces today's behaviour. *Done when:* `bash -n` passes and a dry run lists 81 jobs.
- [ ] **A08-T8** [type: code] (0.1 d, due Tue 13 Oct, 06:00; needed by T15, not by the 19:00 launch) — Detector `--features-table` mode: reads per-fold training-row tables, excludes any column with |Spearman| ≥ 0.999 to a single raw column (value-based form of the single-column-monotone rule), then applies the φ criteria. *Done when:* injected φ gives `rec=1`, `log x_a` gives `rec=0`, and table mode equals formula mode on every episode of the G0a smoke run.
- [ ] **A08-T9** [type: config] (0.05 d, due Mon 12 Oct, 18:00) — `make_manifest.py --ablation A08` writes `manifests/agent2027/A08_M5.csv` (R7). Rows: A08-001…081 CPU (`image_tag=classicfe:4`, rungs and ports empty, `expected_u=0`, `replicate=0`); A08-082…095 `llmfe` (`fold_seed=42`, `replicate` 0/1, `image_tag=llmfe:oss120`); A08-096… `caafe` (2·#C rows, `fold_seed` 42/4, `image_tag=caafe:oss120`). GPU rows have `rung_*=O120`, `port_*=8090`, `expected_u=0.6` (midpoint of 0.45–0.7) and `llm_seed` empty (comparators pass no seed). All rows: `is_sentinel=0`, `session_id` empty, `env_overrides={}`, `cap_hours=6`, comparator settings in `notes`. Rows are executed by the A08 launchers, not `ro27_clients_manifest.sh`. *Done when:* committed with 101–103 rows and unique R12 run names.
- [ ] **A08-T10** [type: code] (0.1 d, due Tue 13 Oct, 06:00; M5S may read the detector table directly if it slips, but W&B logging must then be done by Thu 15) — `scripts/agent2027/a08_log_wandb.py`: one W&B run per manifest row in `PlanBench/RogueOne-AGENT2027`, group `A08`, R12 name, `job_type` = arm, tags [`<SESSION_ID>` or `cpu_screen`, `prefreeze`/`v2`, image tag], R12 identity/version keys (`llm_seed` and `rung_*` empty for CPU rows), comparator settings as `comparator/<NAME>`. Summary: `rec_sel`, `rec_any`, `heldout_score` (`accw_balanced` or nRMSE) and its fold SD (`<key>_sd`), `n_features` per fold, tokens (`tok_compl_total`, `tok_prompt_total`, `llm_calls_total`; GPU rows), `wall_s`, `energy_j`. Episode table and outputs go to artifact `run-output`. It runs in `rogue_one:agent2027-pre.N` (CPU screen) or the current `rogue_one:agent2027-v2.N` (GPU rows), with `--env-file <CAIR_HOME>/code/PlanBench/.env`. *Done when:* one screen cell shows all keys.
- [ ] **A08-T11** [type: code] (0.1 d, due Mon 19 Oct) — `caafe_arm.py --fold-seed` passed to `build_folds` (`:337`); `llmfe_score_rogueone.py` `REGRESSION` (`:72-73`) extended with the SYN-P regression IDs from `registry.yaml`. *Done when:* a `--model skip` CAAFE cell runs at seed 4 and defaults reproduce old outputs.
- [ ] **A08-T12** [type: code] (0.15 d, due Mon 19 Oct) — `scripts/agent2027/a08_comparators_clients.sh`, run by `ro27_vllm_session.sh` (which exports `PORT`, `API_KEY`, `MODEL`, `vllm_session.sh:185-186`):
  1. start the X5 proxy (8090 → `$PORT`) and the E3 energy sidecar `ro27-<session_id>-energy`; log the vLLM image ID (`docker inspect rogueone-vllm-o120-<s> --format '{{.Image}}'`) and the four comparator image IDs;
  2. split `$SESSION_FILE` by `ablation_id`: A08 rows → the comparator queue in `order_key` order; any other row (the A06 sentinel) → a filtered session file run in the background by `SESSION_FILE=<filtered> IMAGE_RO=rogue_one:<current v2.N> PARALLEL=1 CPUS=12 RUN_CAP_HOURS=6 RUNROOT=~/rogueone-runs-agent2027 bash ~/ro27_clients_manifest.sh` on port 8010 directly (Rogue One logs its own tokens through E3), first wave if A06 marks it guard 3;
  3. run CAAFE's first cell; stop CAAFE (not LLM-FE) on `EMPTY_GENERATION`;
  4. LLM-FE per run: `API_KEY=a08-llmfe-<ds>-r<r> SUFFIX=oss120_a08r<r> PORT=8090 CLIENT_LOGDIR=$HOME/ro27_a08/llmfe_logs_r<r> timeout 21600 bash ~/llmfe_launch.sh --one <ds>`, then `docker rm -f llmfe-oss120_a08r<r>-<ds>`; take the rc from the `--- done <ds> rc=` line, not from the exit status (always 0);
  5. CAAFE per cell: `timeout 21600 docker run --rm --name caafe_<ds>_f<s>_a08 --network host --cpus 6 -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e OPENAI_BASE_URL=http://localhost:8090/v1 -e OPENAI_API_KEY=a08-caafe-<ds>-f<s> -e CAAFE_MAX_COMPLETION_TOKENS=2048 -e CAAFE_STOP_MODE=client -v <P>/tasks:/tasks:ro -v <P>/paper/scripts/caafe_arm.py:/app/caafe_arm.py:ro -v <P>/paper/scripts/classicfe_arm.py:/app/classicfe_arm.py:ro -v $HOME/ro27_a08/caafe/f<s>:/out caafe:oss120 python /app/caafe_arm.py <ds> classification /tasks /out --folds stratified --fold-seed <s> --model openai/gpt-oss-120b`, then `docker rm -f caafe_<ds>_f<s>_a08` (`<P>` = `<ROGUEONE_CAIR_PATH>`);
  6. keep ≤ 6 clients in total (5 comparator slots while the sentinel runs); retry each crash once, alone; `wait` for every background job (the script must block, `vllm_session.sh:20-22`); stop the proxy and the sidecar.

  If E1's `ro27_session_launch.sh` accepts a clients-script override, launch through it instead and drop the sidecar and image-ID logging from step 1. *Done when:* a dry run with one task per arm (and a 1-fold smoke sentinel row, `ROGUE_ONE_SMOKE_FOLDS=1`) finishes against a live server, and `bash -n` passes.
- [ ] **A08-T13** [type: writing] (0.25 d, Thu 15 → X10 Fri 16 Oct) — Pre-registration text: screen result and exclusion list; the 7-family count; the incomplete-cell rule; the C1b population, model, claim rule and selection modes; the CAAFE estimate; comparator ITT/abstention rules; the drop rule (§6). *Done when:* in the commit tagged `agent2027-prereg`.

### 5.3 Execution on CAIR

**CAIR rules (binding for every task below):**
1. **Containers only.** Every execution runs inside Docker. The host only issues `docker` / `docker compose` (plus `nvidia-smi`, `docker ps`, `tail`, `uptime`, `grep` checks and the launchers' shell plumbing). A `docker compose run` needs `-T`. CPU containers carry a `--cpus` cap ≤ 16 (01 §5.2).
2. **GPU pre-flight** before any GPU container: run `nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader`, `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader` and `docker ps`. If not enough GPUs are free, **stop and report**.
3. **≤ 4 of the 8 H100s** in total for RogueOne without Henrik's explicit approval.
4. **W&B is the system of record.** Runs are stateless. Secrets come from `.env`: `WANDB_API_KEY` from `<CAIR_HOME>/code/PlanBench/.env` via `--env-file`. Outputs in `~` are staging for A08-T10's `run-output` artifacts only.

**Mechanics.** Every command is one non-interactive shot: `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "<cmd>"`. Code reaches CAIR by commit + push of `agent2027/v2`, then `git fetch` on CAIR (01 §5.2 step 0). Long jobs run under `nohup` and log to `~`, never into root-owned run folders. `<ROGUEONE_CAIR_PATH>` = `<CAIR_HOME>/code/RougeOne` is **inferred; verify** with `ssh … "ls -d ~/code/RougeOne"`. Below, `<P>` = `<ROGUEONE_CAIR_PATH>` and `<S>` / `<s>` = the GPU session ID in upper / lower case.

- [ ] **A08-T14** [type: run] (0.1 d, Mon 12 Oct, ~19:00) — **CPU screen** (no GPU, so no GPU pre-flight; CPU check instead). Prerequisites: T1, T2, T6 (`classicfe:4`), T7 done; X3 + T7 pushed and fetched on CAIR. Check `uptime; docker ps` (A01's clients, 72 cores, should have ended; Finding 43 probe cells `afdet_*` (`paper/scripts/autofeat_determinism_probe.sh:72`, HEAD only) and other `classicfe_*` cells may still run). ≥ 64 free cores of 224 are needed; otherwise lower `PARALLEL` and accept the provisional-label rule. Then:
  `mkdir -p ~/classicfe_a08 && cp <P>/paper/scripts/classicfe_arm.py ~/classicfe_a08/ && printf '%s\n' 'synp_p0|regression|<P>/tasks' 'synp_p1|regression|<P>/tasks' 'synp_p2|regression|<P>/tasks' 'synp_p3|classification|<P>/tasks' 'synp_p4|classification|<P>/tasks' 'synp_p5|classification|<P>/tasks' 'synp_p6a|regression|<P>/tasks' 'synp_p6b|regression|<P>/tasks' 'synp_p7|classification|<P>/tasks' > ~/classicfe_a08/a08_datasets.txt && cd ~/classicfe_a08 && BASE=$HOME/classicfe_a08 DATASETS_FILE=$HOME/classicfe_a08/a08_datasets.txt FOLD_SEEDS='42 4 5' DUMP_FEATURES=1 ARMS='autofeat openfe raw' TAG=a08_screen OUT_SUB=out PARALLEL=16 CELL_TIMEOUT=21600 nohup bash <P>/paper/scripts/classicfe_launch.sh stratified > ~/ro27_a08_screen.job.log 2>&1 &`
  The task types shown are the proposals: take them from `synp_meta/registry.yaml` (binding). Long poles go first; 16 × `--cpus=4` = 64 cores. *Done when:* `ALL_CELLS_DONE` in `~/classicfe_a08/run_a08_screen.log`.
- [ ] **A08-T15** [type: run] (0.1 d, Tue 13 Oct, 07:00–09:00) — Rerun each `CELL_MISSING` alone (`ONLY=<id> ARMS=<arm> FOLD_SEEDS=<s> TAG=a08_retry CELL_TIMEOUT=21600`, same `BASE`/`OUT_SUB`/`DATASETS_FILE`). Score in the image built after T8 merged: `docker run --rm --network none --cpus 8 --entrypoint python3 -v ~/classicfe_a08/out:/in:ro -v ~/rogueone-synp-meta:/meta:ro -v <P>/tasks:/tasks:ro -v ~/ro27_a08/screen:/out rogue_one:agent2027-pre.N /workspace/scripts/agent2027/analysis/recovery_detector.py --features-table /in …` (exact CLI set by E8a; *to verify*); log with A08-T10. *Done when:* 81 cells in W&B and a 405-row episode table (81 cells × 5 folds; the `raw` rows carry `rec = 0` by construction).
- [ ] **A08-T16** [type: run] (0.25 d of operations; one session inside Tue 20 – Sat 24 Oct, slot fixed at planning) — **GPU comparators.** Prerequisites: T4, T5 (G2 "go"), T6, T9–T12 done; session file `manifests/agent2027/sessions/<S>.csv` committed.
  1. **Pre-flight** (rule 2): `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader && docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'"`. Proceed only if 4 GPUs (2 if D5 = TP2) each use < 2,000 MiB, RogueOne's total stays ≤ 4, and no `ro27-*`, `rogueone-vllm-*`, `llmfe-*` or `caafe*` container of ours runs. The one exception is D5 = TP2: a single other TP2 O120 server of ours may run if its GPUs plus ours ≤ 4. Otherwise **stop and report**.
  2. Install: `cp <P>/scripts/agent2027/a08_comparators_clients.sh <P>/paper/scripts/llmfe_launch.sh ~/` plus the 01 §5.2 step-3 launchers and manifests; copy the X4 specs to `~/llmfe/specs/` and the T4 data files to `~/llmfe/data/`.
  3. Launch detached: `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "SESSION_ID=<S> SESSION_FILE=\$HOME/ro27_manifests/sessions/<S>.csv GPUS=<ids> NEED=<4|2> MODEL=openai/gpt-oss-120b PORT=8010 SERVER_NAME=rogueone-vllm-o120-<s> LOG=\$HOME/vllm_session_<S>.log WAIT_MAX_MIN=0 nohup bash -c 'bash ~/ro27_vllm_session.sh ~/a08_comparators_clients.sh; echo \"=== RO27_SESSION_END <S> rc=\$? ===\"' > ~/ro27_<S>.job.log 2>&1 & echo launched"`. With D5 = TP2 and another O120 server live, use port 8011 and `ROGUEONE_ALLOW_SECOND_SERVER=1`, only after a fresh pre-flight.
  4. Monitor per 01 §5.2 step 5 (wait for `RO27_SESSION_END`). Tear-down check per step 6, widened to `grep -E '^(ro27-|rogueone-vllm-|llmfe-|caafe)'`: nothing of ours may be left.

  *Done when:* 14 `llmfe` and 2·#C `caafe` rows are finished or ITT, the sentinel row (if any) is terminal, and proxy token runs are in W&B.
- [ ] **A08-T17** [type: run] (0.05 d, CPU, due Sun 25 Oct and within 24 h of T16's end) — For r ∈ {0, 1} and mode ∈ {notebook, best3}: `mkdir -p ~/ro27_a08/llmfe_r<r>_<mode> && docker run --rm --network none --cpus 8 -e OMP_NUM_THREADS=8 -e PYTHONPATH=/app -v ~/llmfe:/app -w /app -v <P>/paper/scripts/llmfe_score_rogueone.py:/scorer/llmfe_score_rogueone.py:ro -v ~/ro27_a08/llmfe_r<r>_<mode>:/dest llmfe:eval python /scorer/llmfe_score_rogueone.py --problem_name synp_p1 synp_p2 synp_p3 synp_p4 synp_p5 synp_p6a synp_p6b --llm oss120_a08r<r> --seed 42 --folds stratified --selection <mode> --dest /dest --verbose`. The scorer is mounted outside `/app` so that no root-owned mount point is created in `~/llmfe`; `PYTHONPATH=/app` resolves `optimization_utils`. *Done when:* 28 JSONs exist (2 replicates × 2 modes × 7 tasks), none with `folds_done` = 0 unexplained.
- [ ] **A08-T18** [type: run] (0.05 d, by Sat 31 Oct, 24:00 = `DF`; retries are reruns, so they may use Fri 30 – Sat 31; a first launch of a never-started row counts as new and must start by `L-LAST`, Thu 29 Oct) — remaining solo retries, each through the T16 procedure (pre-flight first). *Done when:* every manifest row is terminal (finished or ITT) in W&B.

### 5.4 Analysis

- [ ] **A08-T19** [type: analysis] (0.15 d, Tue 13 Oct, 08:00–10:00; gate `M5S`) — Per task × method: recovered episodes / 15 → R10 label. Checks: p0 has 0 recoveries (else detector bug, fix before A02); `openfe`/`autofeat` − `base_no_fe` on p0 ≈ 0; `base_no_fe` agrees with E8a's headroom calibration; report the rank of OpenFE's first recovering feature (a descriptive top-k reading). Memo run `ANALYSIS__a08_screen__20261013`. *Done when:* the exclusion list is posted and §6 actions are taken.
- [ ] **A08-T20** [type: analysis] (X6 0.25 d + 0.1 d A08 wiring; Sat 24 – due Tue 27 Oct) — `analysis/comparator_parser.py`, each comparator in its own image, `--network none`: LLM-FE programs from each split's `candidates_used` (notebook, best3) and every non-null-score program, applied to the split's training rows as the scorer does (`:114-123`, `:222-237`); CAAFE `generated_code` applied with CAAFE's `run_llm_code` and `make_dataset_numeric` (`caafe_arm.py:210`, `:242-258`). Run the detector in table mode in the current `rogue_one:agent2027-v2.N` image (the detector code must equal the version that scored A05; record `git_sha`); join with A05 episodes (`fetch_wandb.py`). *Done when:* the episode table holds every arm × mode (LLM-FE: 7 tasks × 5 splits × 2 replicates × 3 modes = 210 rows; CAAFE: #C × 2 seeds × 5 folds).
- [ ] **A08-T21** [type: analysis] (0.25 d; preliminary Tue 27 Oct; final by Tue 3 Nov) — `analysis/glmm_holm.py`: **C1b** = logistic GLMM `rec_sel ~ arm + (1|task) + (1|task:fold)` on the primary population, two-sided; exact stratified test (named in X10) under separation; Holm over the surviving family. Sensitivity: best3 vs Full `rec_sel`; any-program vs Full `rec_any`. **CAAFE − Full:** `rec_sel ~ arm + (1|task) + (1|task:seed)`, estimate + 95% CI, outside the claim. Also per-task tables, `tok_compl_total`, `wall_s`, held-out scores. Deliver iso-cost points to A21 and Pareto points to A23. *Done when:* the ANALYSIS run is logged and the C1b row is filled.

### 5.5 Paper outputs

- [ ] **A08-T22** [type: writing] (0.25 d; Table 2 row by Tue 3 Nov, the rest by Fri 6 Nov) — Table 2: C1b row (raw p, Holm verdict, three-mode agreement) and CAAFE row. Fig. 2 (right): LLM-FE and CAAFE as external points on recovery vs completion tokens. Protocol: one sentence on screen exclusions. Appendix: screen table (x/15 per task × method); comparator settings (temperatures, budgets, selection, no web search); disclosures (we authored the LLM-FE specs; notebook defect; OpenFE all-features reading; AutoFeat unseeded draws; CAAFE 2048/`client` deviations; LLM-FE scorer fits unweighted; prior-evidence caveat). *Done when:* drafted in the AGENT paper by Fri 6 Nov.

## 6. Decision rules and gates (pre-registered in X10)

**`M5S`, Tue 13 Oct morning.** A task is non-discriminating if OpenFE or AutoFeat, each judged separately, recovers φ in ≥ 14/15 episodes (R10). Count non-discriminating families out of **7** (P1, P2, P3, P4, P5, P6, P7; p0 is a control). The P6 family counts as solved if P6a **or** P6b is solved.
- **0–3 solved:** exclude those tasks from every confirmatory recovery population (C1a, C1b, C2a–c, C3, C4b, C5b, and the RT3 reading of G3); they stay in the runs as sanity checks and are reported separately in exploratory read-outs (A02, A03, A11, A17, A20). A02 launches as planned.
- **≥ 4 solved:** recalibrate SYN-P (E8a: deeper φ, more decoys or near-collinear pairs), re-screen the changed tasks Tue 13 night; A02 slips ≤ 1 day; the freeze holds.
- **P6a or P6b non-discriminating:** C4b is dropped in X10; Holm then has 9 contrasts (smallest threshold 0.0056).
- **Any recovery on P7 (≥ 1/15 for either method):** P7 is not knowledge-only. E8a redesigns it and re-screens it (9 cells: 3 arms × seeds {42, 4, 5}, overnight) before X10 on Fri 16, and A14 is told (A14-T3 (b)).
- **AutoFeat cells incomplete at 08:00:** label provisionally from OpenFE plus completed episodes; final labels by Thu 15. A label that changes between the provisional and the final read is applied from X10 on and noted in A02's read-out.

**C1b claim rule** (primary population, §4). The test is the two-sided Wald (or, under separation, the X10-named exact stratified) p for the `arm` coefficient, LLM-FE − Full, in the notebook-mode GLMM. A direction is claimed only if (a) that p passes Holm at its rank, **and** (b) the point estimates of LLM-FE − Full have the same strict sign in all three modes: notebook `rec_sel`, best3 `rec_sel`, and any-program `rec_any` against Full's `rec_any`. Otherwise the verdict is "inconclusive", and "inconclusive (selection-dependent)" if (a) holds but (b) fails. A zero estimate in any mode counts as disagreement. The CAAFE − Full estimate never changes the C1b verdict. Planning MDE is about 0.23–0.30, recomputed at G2 from M0's ICC.

**Drop and kill rules.** If no LLM-FE run has started by `L-LAST` (Thu 29 Oct), C1b has no data: drop it, recompute Holm, record a pre-registration amendment. If fewer than 2 discriminating SYN-P 7 tasks remain after `M5S` and any recalibration, C1b is reported descriptively only and dropped from Holm in X10 (rule proposed here; confirm in X10). At the G1 re-tier, **CL4** drops CAAFE: −8 runs, −3.6 to −5.6u as budgeted (−6 runs, −2.7 to −4.2u at three classification tasks). LLM-FE and the CPU screen always stay.

## 7. Risks and cut levers

| Risk | Mitigation |
|---|---|
| Screen solves too much | Recalibration delays A02 ≤ 1 day; the depth ladder (P2, P3) is the guard |
| AutoFeat runtime (communities, p = 102, took ~44 h) or overrun past Tue 08:00 | Long poles first, 6 h cell cap, provisional-label rule |
| AutoFeat unseeded (Finding 43): labels not reproducible | Each episode is one draw of the method; disclose; record `n_features` per fold |
| Notebook selection understates LLM-FE (biases toward Full) | Three-mode agreement rule |
| Confounds: no web search, temperature (0.5 vs 0.7), budgets, selection | System-level wording; C1a carries the mechanism |
| Partition mismatch (row order, duplicates, D4, LLM-FE regression split, scikit-learn version skew between images) | A08-T1, T2, T4 checks, each side computed in its own image; if broken, drop fold pairing for the affected tasks (task random intercept only) and disclose; recovery is still valid on training rows |
| Comparator evaluator seed fixed at 42 (classicfe, LLM-FE scorer) while Full uses `fold_seed` | Affects held-out scores at seeds 4/5 only; recovery unaffected; disclosed |
| A06 placed "A10 + A11 + A08 GPU rows" in one session (`S1020a`) | Resolved: A06's hosting table now lists A10 + A11 only; A08 runs its own session (§4) |
| LLM-FE crashes (~7%) | Solo retry |
| Proxy attribution fails | vLLM request logs, then `llm_calls_total` |
| Images pruned before | Rebuild from Dockerfiles; record IDs |
| Developer time: 0.75 d of A08-local engineering (T1 0.1, T6 0.05, T7–T12 0.6) is not itemised in R2 (T2 = X3, T4 = X4 + X5, T20 includes X6; T13 is A08's share of X10). Operations and analysis add about 1.2 d (T14 0.1, T15 0.1, T16 0.25, T17 0.05, T18 0.05, T19 0.15, T21 0.25, T22 0.25). Mon 12 already carries G0a, X3 and the E6 start, so T8–T10 are due Tue 13, 06:00 | If short, drop T10's energy logging first, then CAAFE (CL4) |

## 8. Definition of done

- [ ] Screen labels are in X10, and every recovery ablation applies the exclusion list.
- [ ] W&B group `A08` holds 81 CPU cells, 14 `llmfe` and 2·#C `caafe` runs, each with `run-output`.
- [ ] The A08 episode table (every arm × selection mode) is joined with A05.
- [ ] C1b (Holm verdict, three-mode agreement) and the CAAFE CI are logged as ANALYSIS runs.
- [ ] Token and wall-clock points are delivered to A21 and A23.
- [ ] Table 2 row, Fig. 2 points and appendix tables are drafted.

---

**New names introduced** (not in the canonical registry):
- Launcher env: `DATASETS_FILE`, `FOLD_SEEDS`, `DUMP_FEATURES`, overridable `BASE` (in `classicfe_launch.sh`).
- CLI options: `classicfe_arm.py --fold-seed / --dump-features` (X3 behaviour, named here), `caafe_arm.py --fold-seed`, detector `--features-table`.
- Scripts: `scripts/agent2027/a08_comparators_clients.sh`, `scripts/agent2027/a08_log_wandb.py`.
- W&B config prefix `comparator/<NAME>`. The CPU screen is logged to `RogueOne-AGENT2027` (R12 does not place it; chosen here) with code-phase tag `prefreeze`.
- CAIR paths and names: `~/classicfe_a08/` (+ `a08_datasets.txt`, `out/f<seed>/`, `out/f<seed>/features/`), `~/ro27_a08/`, `~/ro27_a08_screen.job.log`, LLM-FE suffixes `oss120_a08r0` / `oss120_a08r1`, client keys `a08-<arm>-<dataset_id>-…`, container names `caafe_<ds>_f<s>_a08` and `llmfe-oss120_a08r<r>-<ds>`, proxy container (X5 sets it; suggested `ro27-<session_id>-proxy`).
- Manifest row ranges A08-001…081 (CPU), A08-082…095 (`llmfe`), A08-096… (`caafe`); W&B tag `cpu_screen` for the CPU rows (no session ID).
