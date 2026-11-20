# A02 · M4 — Extractor rounds R ∈ {1, 3, 10}: pilot and freeze decision

*Rogue One v2 design study for ICSE AGENT 2027 (T1-minimum; submit Fri 13 Nov 2026). Written Tue 6 Oct 2026; code references re-verified Wed 7 Oct 2026.*

- **Code refs:** `nested-cv-rebuttal` @ `07cbf93` (`C:/code/PHD/RougeOne`), read line by line. The current tip `1bdaf42` changes only files under `paper/` (result CSVs, the camera-ready plan and `paper/scripts/`). `rogueone/` and `rogue_one_main.py` are identical to `07cbf93`, so every ref below still holds.
- **Short paths (01 §7 R0):** `main.py` = `rogue_one_main.py`; `ext/agent.py` = `rogueone/llm/agents/core/extractor/agent.py`; `config.py` = `rogueone/utils/config.py`.
- **"01 §n"** = `docs/ablations/01_shared_infrastructure.md`.
- **Units (01 §7 R0):** u = one real-data O120 run (10 iterations × 5 folds). Box throughput = 2 u/h × 0.7 utilisation = **1.4 u/h** on ≤ 4 H100.
- **Every "hoped" result is a hypothesis.** Nothing below is a result.

| Field | Value |
|---|---|
| ID | **A02 · M4**. Arms `R10` (control), `R3`, `R1` |
| Claim / Holm contrast | C6 (harness channels). **No Holm contrast.** u per run is a cost outcome outside Holm (01 §7 R14). A02 sets freeze item 13 (R) for all later arms. |
| Tier | T1-minimum; "R=3 vs R=10" is never cut |
| Owner | Henrik |
| Start / finish | Smoke rows **Mon 12 Oct** (G0a smoke session). Launch **Tue 13 Oct ~12:00** (after `M5S`). Runs done **Wed 14, 08:00**. Read-out **Thu 15, 12:00**. R frozen at **G1, Fri 16 Oct**. Paper outputs **Tue 3 Nov**. |
| Budget | **24 runs, 12–20u** (as in the T1-minimum breakdown and 01 §4.9). At 1.4 u/h that is 8.6–14.3 h on 4 H100 (34–57 GPU-h), plus ≈ 2–3 h per solo retry. The 2 smoke rows (≈ 0.3u) run on Mon 12, outside the Tue 13 – Sat 31 Oct window, and are not in the 12–20u. |
| Models | O120 (`openai/gpt-oss-120b`) for S, E, T and W. Served by `vllm/vllm-openai:v0.22.0` at TP4 (pre-D5, as t17), port 8010, with 6 clients. |
| Status | Not started |
| Depends on | E1, E2 (+ R knob, built as A02-T5), E3, E5 and E8a (+ detector), all at G0a Mon 12. `.dev.env` untracked on `agent2027/v2` (E1). `M5S`, i.e. the A08 CPU screen read (Tue 13 morning). The rule commit (A02-T3) on Tue 13. Soft: A03 `replace` rows (for `W_ref` only); D4 (default assumed, §4). A01 (median `wall_s`, the `W_ref` fallback; hand-off A01-T18). |
| Feeds into | G1 `v2_defaults.env` (so every v2 arm A05–A20, indirectly via G1; those documents depend on G1, not on A02 directly) and the G1 re-tier (CL1–CL4). A03 add-on (paired with `R10`). X9, A14-T3 and A14-T4 (P7 queries). A21/A23 (separate pre-freeze panel). The C6 paragraph. |

## 1. What we test and why

In every iteration the Extractor receives the same instruction 10 times in one conversation, and it does not stop after succeeding. We test whether 3 rounds (or 1) recover planted SYN-P features as often as 10, at lower cost per run.

R is frozen for every later arm, so this is the study's largest budget lever. If u per run falls by about 30%, roughly 90–125u is freed (T1-minimum §3). The result is also a C6 design statement: Extractor persistence is part of the architecture.

## 2. Background

**Code path** (all verified at `07cbf93`):
- **Per iteration** (`main.py:536`). The fold state is reset at `iteration_in_fold == 0` (`:543-544` → `_reset_fold_state`, `:417-457`). In replace mode the feature table restarts as `id` only (`:433-435`). Then:
  1. Scientist focus (`:548-555`);
  2. `extractor_agent.generate_patient_attributes(..., focus, step=index)` (`:563-571`);
  3. Tester.

  One run = 5 folds × 10 iterations.
- **Rounds.**
  - `generate_patient_attributes` (`ext/agent.py:172`) opens one `SQLiteSession(f"extractor_agent_session_step_{step or 0}")` per call, i.e. per iteration (`:965`).
  - It then loops `for _ in range(10):` (`:967`, the only such loop in `rogueone/` and `main.py`), calling `Runner().run(agent, f"System now wants to focus on: {focus}.", max_turns=extractor_agent_cfg.max_iterations, session=session, run_config=_run_config)` (`:969-975`). There is no break on success.
  - A **round** is one `Runner.run`, of up to `EXTRACTOR_AGENT_MAX_ITERATIONS` = 100 model turns (`max_turns`; `config.py:373`). Each round re-sends the same message into the same session, so the context grows.
- **Failures.**
  - An exception in a round (max turns exceeded, context length, transport) increments `_tool_call_counter_failed["runner"]`, and the next round starts (`ext/agent.py:977-992`).
  - The counters are reset at the start of every call (`:198-200`), so the logged W&B key `Extractor_tool_usage_failed/runner` (`:1021-1027`) is a **per-iteration** count. The key is absent when the count is 0.
  - If no attribute was added in any round, the deterministic fallback runs (`:996-1002`, defined at `:599-844`). Fallback features never count as recovery.
  - Zero productive features skip the iteration (`main.py:639-663`), or abort the dataset at a fold's first iteration (`:665-717`).
  - An abort writes no `fold_aggregation_summary.csv` (`main.py:1483-1505`). It raises at `:1527-1531`, but `run_experiments` catches the exception (`:1589-1598`), so the container can still exit with rc 0. The launcher therefore detects an abort by the missing summary file (01 §5.3 step 5), not by rc.
- **Prompt and search.**
  - The prompt says "Repeat until you have added a sufficient number of relevant features" (`ext/agent.py:889`). It never mentions R, so the arms differ only in the loop count.
  - Every round can call web search (`search_in_literature_tool`, `:246-280`; shared `WebSearchAgent` at `:166`). Every query goes to `retrieval_transcript.jsonl` (`rogueone/utils/retrieval_audit.py`, written from `kn/web_search.py`).
  - Extractor `max_tokens` = 32768 (`config.py:378`). `ROGUE_ONE_NUM_ITERATIONS` defaults to 3 (`config.py:305`), so always set it to 10.
- **Task registration.** The legacy task list (`main.py:1536-1566`) has no `synp_*` entry. SYN-P rows resolve only through `ROGUE_ONE_TASK_REGISTRY` (E1).

**SYN-P 8** (built by E8a; 01 §4.1):
- n = 2,000, p ≈ 20–30, 3 relevant columns plus correlated decoys and near-collinear pairs, headroom ≥ 0.06 balanced accuracy or ≥ 15% nRMSE.
- Folders `tasks/<task_type>/synp_<id>/`. φ is kept off-mount in `~/rogueone-synp-meta/`.
- Tasks: `synp_p1` x_a/x_b (R) · `synp_p2` (x_a/x_b)·x_c (R) · `synp_p3` threshold-AND of a ratio (C) · `synp_p4` cyclic hour (C) · `synp_p5` balance-scale torque rule (C) · `synp_p6a` named composite (R) · `synp_p6b` its anonymised twin (R) · `synp_p7` a rule documented only in a local corpus (C; at the floor by design, because A02 has live web search only).
- Task types are proposed; E8a confirms them in `synp_meta/registry.yaml`. A02 runs P7 so that its queries feed X9 and A14, and as a pre-freeze leak canary.

**Metrics** (01 §7 R13 names; definitions 01 §4.6):
- **`rec_sel`** (primary) and `rec_any`, from `scripts/agent2027/analysis/recovery_detector.py`.
  - **Recovered:** an active non-fallback feature has |Spearman| ≥ 0.95 with φ on training rows, or NMI ≥ 0.9 for P3/P5/P7 (no credit for P7's ratio alone). Raw columns and single-column monotone transforms never count.
  - `rec_any` = at any iteration of the fold; `rec_sel` = in the iteration that inner CV selects.
  - The unit is the fold-episode: 8 tasks × 5 folds = 40 per arm, paired across arms through shared partitions (same `fold_seed`, same stratification per task).
- **`success`:** the run finishes within the 6 h cap **and** its held-out score (mean over the 5 outer folds at the inner-CV-selected iteration; balanced accuracy for C, nRMSE for R) is no worse than `base_no_fe` by more than 2 × `base_no_fe`'s fold SD. Direction-aware: for nRMSE, "worse" means higher. The `base_no_fe` mean and fold SD come from the A08 CPU screen at seed 42, on the same partition (stratified for C, KFold for R).
- **`u_run`** (analysis quantity, not a W&B key) = Σ wall clock over the row's attempts ÷ `W_ref`.
  - Wall clock per attempt = W&B `wall_s`; for an attempt with no W&B summary, the wall clock the launcher writes to the job log (01 §5.3 step 6). A cap hit counts as 6 h.
  - `W_ref` = the median `wall_s` of a real-data O120 run at R=10 with 6 clients, so that a real-data run = 1u. Ratios between arms do not depend on `W_ref`.
- **Reported:** `tok_compl_<role>`, `tok_prompt_<role>`, `max_prompt_tokens`, `wall_s_extractor`, `context_errors`, `serper_calls`, `energy_j`, `fallback_share`, `fun_*`, `cap_hit`, `itt_imputed`.

## 3. Hypotheses and interpretation (all hypotheses)

| Outcome | Action | Paper sentence (template) |
|---|---|---|
| **Hoped:** R=3 passes all four §6 criteria | Freeze R=3 as a declared design decision; about 90–125u freed | *"R=3 matched R=10 on recovery (x/40 vs y/40 fold-episodes) at z% of the wall clock."* |
| **Null:** recovery equal, but u falls by < 25% | Keep R=10; CL1–CL4 at G1 (R=3 not adopted) | *"Extra Extractor rounds are cheap and harmless."* |
| **Opposite:** R=3 loses recovery (criterion 1 or 2 fails) | Keep R=10; CL1–CL4 at G1 | *"Extractor persistence buys recovery"* (C6) |
| **Reverse:** R=3 recovers *more* than R=10 (e.g. late R=10 rounds hit the context limit) | R=3 is adopted if criteria 2, 3 and 4 also hold (criterion 1 passes by construction) | *"Extra rounds hurt: the growing shared context costs recovery."* Report `context_errors` and runner failures by round (A02-T14). |
| **Mixed:** recovery passes, but `success` falls (criterion 3) | Keep R=10; CL1–CL4 at G1; report which criterion failed | *"Fewer rounds kept recovery but lost runs."* |

- **R=1** is a dose-response point only and is never adopted. With one round, a single runner failure (max turns, context) sends the iteration straight to the fallback, which is a hazard for small rungs.
- **Cost:** shared-session rounds mostly re-prefill a growing context. Completion tokens may barely fall while wall clock falls, so the rule decides on u, not tokens.
- **Replace mode biases the pilot against R=3** (raw columns must be rebuilt): a pass is expected to carry over to augment; a fail may not.
- **Disclose:** CAP-1 and CAP-2 run small Extractors at the frozen R (a C5 confound).

## 4. Design

**Arms.** `R10` / `R3` / `R1` (01 §7 R5). Overrides relative to the draft `scripts/agent2027/v2_defaults.env` at tag `agent2027-pre.N`:
- `ROGUE_ONE_EXTRACTOR_ROUNDS` = `10` / `3` / `1`;
- `ROGUE_ONE_FEATURE_POOL_MODE=replace` (a no-op on a pre-E6 image).

**Fixed (all arms):**
- **Image:** one `rogue_one:agent2027-pre.N` (the G0a tag) for all 24 rows.
- **Loop:** `ROGUE_ONE_NUM_ITERATIONS=10`, `ROGUE_ONE_K_FOLDS=5`, `ROGUE_ONE_SELECTION_SIGNAL=inner_cv`, `ROGUE_ONE_ENABLE_ROLLBACK=False`, `ROGUE_ONE_FOLD_RETRY_LIMIT=1`.
- **Sampling and limits:** temperature 0.7 for every role and W (`LLM_TEMPERATURE`, `*_AGENT_TEMPERATURE`); max tokens E 32768, T 8192, S at the draft value (proposed 16384); turn caps E 100, T 100, S 200; `ROGUE_ONE_REASONING_EFFORT_PASSTHROUGH=0`.
- **Harness flags:** `ROGUE_ONE_TOOL_ERROR_FEEDBACK=message`, `ROGUE_ONE_TOOL_OUTPUT_MAX_CHARS` at the draft value (proposed 8000), `ROGUE_ONE_DO_BALANCING=True`, `ROGUE_ONE_LLM_TRACE=1`, `ROGUE_ONE_PROMPT_DUMP=1`, `ROGUE_ONE_SANDBOX` = `on` if E13 landed at G0a, else `off`, `ROGUE_ONE_START_VLLM=False`, `CONSOLE_DISABLE=False`.
- **Folds:** `ROGUE_ONE_STRATIFIED_FOLDS=True` on `synp_p3`, `p4`, `p5` and `p7` (classification); `False` on the regression tasks. This is the recommended D4 default (D4 is decided only on Thu 15); A03's add-on copies it, and A08's CPU screen uses the same partition.
- **Search:** live Serper. E4 (Serper cache) lands Wed 14 (due Fri 16), so it is not on this image, and `ROGUE_ONE_SEARCH_CACHE` does not exist yet.
- **Run control:** `PARALLEL=6`, `CPUS=12`, `RUN_CAP_HOURS=6`.
- **Invariant:** every other R3/R4 flag takes its draft `v2_defaults.env` value. The effective `flags/<NAME>` values must be identical across the 24 rows except `ROGUE_ONE_EXTRACTOR_ROUNDS` (by arm) and `ROGUE_ONE_STRATIFIED_FOLDS` (by task type). A02-T12 checks this.

**Seeds and pairing.**
- `fold_seed` = `llm_seed` = 42, `replicate=0` (01 §7 R11). No sentinel, because no Full-v2 exists yet.
- All 24 rows run in one session, `S1013a`, in a random `order_key` seeded by the session ID.

**Run matrix** (W&B `PlanBench/RogueOne-AGENT2027-prefreeze`, group `A02`; within each arm, rows are numbered in the order `synp_p1`, `p2`, `p3`, `p4`, `p5`, `p6a`, `p6b`, `p7`):

| `arm_id` | Data × seed | Runs | u/run | u | `row_id` | Run name |
|---|---|---|---|---|---|---|
| `R10` | SYN-P 8 × 42 | 8 | 0.7–1.0 | 5.6–8 | A02-001…008 | `A02__R10__<dataset_id>__f42__r0` |
| `R3` | SYN-P 8 × 42 | 8 | 0.4–0.75 | 3.2–6 | A02-009…016 | `A02__R3__<dataset_id>__f42__r0` |
| `R1` | SYN-P 8 × 42 | 8 | 0.4–0.75 | 3.2–6 | A02-017…024 | `A02__R1__<dataset_id>__f42__r0` |
| **Total** | | **24** | | **12–20** | | |
| smoke (Mon 12, not analysed) | `synp_p1`, 1 fold | 2 | ≈ 0.15 | ≈ 0.3 | A02-901 (`R1`), A02-902 (`R3`) | same pattern; `notes=smoke`, tag `smoke` |

**Manifest values** (01 §7 R7; one line per column):
- `row_id` as above; `ablation_id=A02`; `arm_id` as above; `dataset_id` = `synp_p1` … `synp_p7`; `task_type` = `classification` for p3, p4, p5, p7, else `regression`.
- `fold_seed=42`; `llm_seed=42`; `replicate=0`; `is_sentinel=0`.
- `session_id` empty in the ablation manifest; `S1013a` in the session file (smoke rows: the G0a smoke session, `S1012a`).
- `order_key` assigned by `make_manifest.py --plan-session`.
- `image_tag=agent2027-pre.N`; `rung_S=rung_E=rung_T=O120`; `port_S=port_E=port_T=8010`.
- `env_overrides` = `{"ROGUE_ONE_EXTRACTOR_ROUNDS":"<R>","ROGUE_ONE_FEATURE_POOL_MODE":"replace"}`, plus `"ROGUE_ONE_STRATIFIED_FOLDS":"True"` on the classification rows and `"ROGUE_ONE_SMOKE_FOLDS":"1"` on the smoke rows.
  - **Deviation from R7:** `env_overrides` carries the R3 flag `ROGUE_ONE_STRATIFIED_FOLDS` on the classification rows (as in A05). That override is part of the task definition.
- `cap_hours=6`.
- `expected_u` = 0.85 (`R10`), 0.575 (`R3`, `R1`), 0.15 (smoke). Midpoint check: 8 × 0.85 + 16 × 0.575 = 16.0u, the midpoint of 12–20u.
- `notes` = empty, or `smoke`.

**Budget.** 8 × (0.7–1.0) + 16 × (0.4–0.75) = 5.6–8 + 6.4–12 = **12–20u**, as in the T1-minimum card. Divided by 1.4 u/h this is 8.6–14.3 h, on pre-freeze GPU time that is otherwise idle (T1-minimum §2: the pre-freeze slice holds 126u gross, of which M4 + M1 + CAP-0 A/B use 37–49u).

**Failures (ITT; 01 §4.5).**
- A crash (rc ≠ 0, or no `fold_aggregation_summary.csv`, which includes a dataset abort) is retried once, alone, at the end of the session.
- A persistent failure or a cap hit (never retried) gives `rec_sel = rec_any = 0` for all 5 episodes and `success = 0`. Its time still counts in `u_run`.

## 5. Tasks

All tasks are owned by Henrik. Every date is 2026.

### 5.1 Prerequisites

- [ ] **A02-T1** [type: decision] (0.1 d, due Mon 12 Oct) — Check G0a: E1, E2 (incl. A02-T5), E3, E5 and E8a are merged and tagged `agent2027-pre.N`, and the draft `v2_defaults.env` exists. *Done when:*
  - the G0a smoke run is in W&B under its R12 name, with non-zero `tok_compl_scientist`, `tok_compl_extractor` and `tok_compl_tester`;
  - `llm_traces.jsonl.gz` is in its `run-output` artifact, and `flags/ROGUE_ONE_EXTRACTOR_ROUNDS` is in its config;
  - the 8 `synp_*` folders load via `ExperimentConfig.from_yaml` (`config.py:179`) with no metadata, and resolve through `ROGUE_ONE_TASK_REGISTRY`;
  - the detector unit tests pass (φ → `rec=1`; monotone transform of one column → 0; fallback feature ignored);
  - `git ls-files .dev.env` on `agent2027/v2` prints nothing. The file is tracked at `07cbf93` (it sets `LLM_PORT`, `LLM_API_KEY` and `ROGUE_ONE_K_FOLDS`), and `config.py:24` loads it with `override=True`.
- [ ] **A02-T2** [type: decision] (0.1 d, due **Tue 13 Oct, 10:00**; gate `M5S`) — Read A08's CPU screen (OpenFE, AutoFeat and `base_no_fe` × seeds {42, 4, 5}; run `ANALYSIS__a08_screen__20261013`). A family is non-discriminating if OpenFE or AutoFeat, each judged separately, recovers φ in ≥ 14 of 15 fold-episodes. Families are counted out of 7 (P1, P2, P3, P4, P5, P6 with P6a/P6b counted once, P7). *Done when* the session plan records:
  - the number of non-discriminating families and their names;
  - that `synp_p0` has 0 recoveries (otherwise the detector has a bug: fix it before launch);
  - that `base_no_fe` mean and fold SD at seed 42 exist for all 8 SYN-P 8 tasks.

  Launch proceeds only if the count is < 4 and the p0 check passes. Otherwise the §6 `M5S` gate applies.
- [ ] **A02-T3** [type: writing] (0.1 d, due **Tue 13 Oct, 10:00**) — Commit and push the §6 rules (R rule, error rates, gates, re-tier rule) into the draft `docs/agent2027/preregistration.md`. *Done when:* the commit is pushed before the first A02 run's W&B `created_at`, and its SHA and time are in the session plan.
- [ ] **A02-T4** [type: config] (0.05 d, due Tue 13 Oct, 10:00) — Check quotas. *Done when:*
  - the Serper dashboard shows credit ≥ 2 × 24 × the G0a smoke's query count (lines with `"event": "query"` in its `retrieval_transcript.jsonl`, scaled from 10 to 50 iterations);
  - the W&B key works from a container (01 §5.2 step 7).

### 5.2 Implementation

- [ ] **A02-T5** [type: code] (0.1 d, inside E2, due Mon 12 Oct) — Build the R knob `ROGUE_ONE_EXTRACTOR_ROUNDS`:
  1. In `rogueoneConfig` (`config.py:303-337`), add `self.extractor_rounds = int(os.getenv("ROGUE_ONE_EXTRACTOR_ROUNDS", 10))`, and raise `ValueError` for values < 1.
  2. Add `"ROGUE_ONE_EXTRACTOR_ROUNDS": "10"` to `OPTIONAL_ENV_DEFAULTS` (`rogueone/utils/config_keys.py:22-57`).
  3. Add `rouge_one_cfg` to the import at `ext/agent.py:25` (it is created at `rogueone/utils/__init__.py:15`).
  4. Replace `range(10)` at `ext/agent.py:967` with `range(rouge_one_cfg.extractor_rounds)`.
  5. Register the key with E1's flag registry.

  Leave `:965`, `:971`, `:972` and `:996-1002` untouched. In `tests/unit/agents/test_extractor.py`, monkeypatch `rogueone.llm.agents.core.extractor.agent.Runner` (imported at `ext/agent.py:14`) and count the `run` calls. `rogueoneConfig` is a `Singleton` (`rogueone/utils/singelton.py:22-24`), so set `rouge_one_cfg.extractor_rounds` directly in the count tests, and pop `Singleton._instances[rogueoneConfig]` before testing env parsing. *Done when:*
  - the counts are 1, 3 and 10, and 10 when the env var is unset;
  - `ROGUE_ONE_EXTRACTOR_ROUNDS=0` raises at config construction;
  - `prompt_diff.py` (E5) shows the Full prompt dump is byte-identical to the dump before the change.
- [ ] **A02-T6** [type: run] (0.05 d, due Mon 12 Oct, in the G0a smoke session `S1012a`) — Run the smoke rows A02-901 and A02-902 on `agent2027-pre.N`. This needs A01's session to have ended (one Rogue One job at a time) and the 01 §5.2 GPU pre-flight. *Done when:*
  - both rows end `OK` (rc 0 and `fold_aggregation_summary.csv`), with `flags/ROGUE_ONE_EXTRACTOR_ROUNDS` = 1 and 3 in W&B;
  - in every iteration without a runner failure, the last Extractor request in the trace holds exactly R user messages starting "System now wants to focus on:";
  - both rows' Extractor system prompts are byte-identical.
- [ ] **A02-T7** [type: config] (0.1 d; manifest due **Mon 12 Oct**, before the smoke session; session file due **Tue 13 Oct, 10:00**, after A02-T2) — Generate, commit and push:
  - `scripts/agent2027/make_manifest.py --ablation A02` writes `manifests/agent2027/A02_M4.csv` (26 rows);
  - `make_manifest.py --plan-session S1013a --rows A02-001 … A02-024` writes `manifests/agent2027/sessions/S1013a.csv`. If `M5S` forced a recalibration, regenerate it after the re-screen.

  If A07-T6 asks for it, A07's smoke rows A07-901/902 may be appended to `S1013a`; they are not analysed here. *Done when:*
  - each arm has 8 rows, one per `synp_*`, and the column values match §4;
  - the container names `ro27-s1013a-a02-0nn` are unique;
  - both files are pushed, and the push SHA is in the session plan.

### 5.3 Execution on CAIR

**Hard rules (01 §5.1), for every command below:**
- **Containers only.** Every execution happens inside Docker. The host issues only `docker` / `docker compose` commands (`docker compose run` over SSH needs `-T`), the `~` launcher scripts, `git`, and read-only queries (`nvidia-smi`, `docker ps`, `tail`). Never run `python` on the host.
- **GPU pre-flight** (A02-T9) before every GPU container. If there are not enough free GPUs, **stop and report**; do not queue.
- **≤ 4 of the 8 H100s**, ever, without Henrik's explicit approval in chat. One Rogue One job at a time.
- **W&B is the system of record.** Runs are stateless and upload the `run-output` artifact. Secrets come from `.env` files: `WANDB_API_KEY` from `<CAIR_HOME>/code/PlanBench/.env`; `SERPER_API_KEY` from the image's `.env` / `.secret.env`.
- **SSH** calls are one-shot and non-interactive, with `-o BatchMode=yes -o ConnectTimeout=30`.
- **Code reaches the server** only via commit + push locally, then `git fetch` over SSH on the server.
- **Long jobs** run detached (`nohup`), with the log in the home dir (`~`), never in the root-owned run folders.

`<ROGUEONE_CAIR_PATH>` = `<CAIR_HOME>/code/RougeOne`. **This is inferred; verify it** with `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "ls -d ~/code/RougeOne"`.

- [ ] **A02-T8** [type: run] (0.2 d, due Tue 13 Oct, 11:00) — Locally, push `agent2027/v2` and the tag `agent2027-pre.N`, and note `git rev-parse agent2027-pre.N^{commit}`. Then on CAIR (the build is skipped if the G0a image with this tag already exists):
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15' git fetch git@github.com:HenrikNorkart/RougeOne.git agent2027/v2 && git checkout -B agent2027/v2 FETCH_HEAD && git log -1 --format='HEAD: %H %s' && test ! -e .dev.env && (docker image inspect rogue_one:agent2027-pre.N --format '{{.Id}}' 2>/dev/null || docker build -t rogue_one:agent2027-pre.N .) && docker image inspect rogue_one:agent2027-pre.N --format '{{.Id}}' && cp scripts/agent2027/ro27_session_launch.sh scripts/agent2027/ro27_clients_manifest.sh ~/ && cp paper/scripts/vllm_session.sh ~/ro27_vllm_session.sh && mkdir -p ~/ro27_manifests && cp -r manifests/agent2027/. ~/ro27_manifests/ && ls ~/ro27_manifests/sessions/S1013a.csv"
  ```
  *Done when:*
  - the printed HEAD equals the tag's commit SHA;
  - the image ID is recorded in the session plan (it must equal the smoke rows' image ID if the build was skipped);
  - `test ! -e .dev.env` passed, and the session file is installed.
- [ ] **A02-T9** [type: run] (5 min, due Tue 13 Oct, within 10 min before launch) — Run the pre-flight:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader && echo '--- compute apps ---' && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader && echo '--- containers ---' && docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}' && echo '--- ours ---' && (docker ps --format '{{.Names}}' | grep -E '^(ro27-|rogueone-|llmfe-|caafe_|classicfe_)' || echo none)"
  ```
  *Done when:*
  - at least 4 GPUs show < 2,000 MiB used (`vllm_session.sh` `FREE_MIB`) and ≈ 0% utilisation; their ids are written down;
  - the "ours" block prints `none`. That pattern covers A01's `rogueone-vllm-o120-<sid>` server (the script default is `rogueone-vllm-session`; both match `rogueone-vllm-*`) and its `rogueone-<runtag>-<ds>` clients, and our comparator containers.

  **Otherwise STOP** and report the per-GPU memory and utilisation, the compute apps and the containers to Henrik. Do not queue. If more than 10 min pass before A02-T10, re-run the pre-flight.
- [ ] **A02-T10** [type: run] (0.1 d, due Tue 13 Oct, 12:00) — Launch detached, with the 4 GPU ids from A02-T9:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "SESSION_ID=S1013a SESSION_FILE=~/ro27_manifests/sessions/S1013a.csv GPUS=<4 ids> NEED=4 PORT=8010 MODEL=openai/gpt-oss-120b IMAGE=vllm/vllm-openai:v0.22.0 SERVER_NAME=rogueone-vllm-o120-s1013a LOG=~/vllm_session_S1013a.log WAIT_MAX_MIN=0 IMAGE_RO=rogue_one:agent2027-pre.N RUNROOT=~/rogueone-runs-agent2027 PARALLEL=6 CPUS=12 RUN_CAP_HOURS=6 WANDB_PROJECT_NAME=RogueOne-AGENT2027-prefreeze nohup bash ~/ro27_session_launch.sh > ~/ro27_S1013a.job.log 2>&1 & echo launched; sleep 5; tail -5 ~/ro27_S1013a.job.log"
  ```
  *Done when* (check after ≈ 30 min; vLLM may take up to 30 min to become healthy) the job log shows vLLM 0.22.0, both image IDs, the sidecar `ro27-s1013a-energy` and 6 `ro27-s1013a-a02-*` containers.
- [ ] **A02-T11** [type: run] (0.3 d, Tue 13 – Wed 14 Oct, 08:00) — Monitor and rerun.
  - Watch in the background: `ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 CAIR-gpu14 "while ! grep -q 'RO27_SESSION_END' ~/ro27_S1013a.job.log 2>/dev/null; do sleep 120; done; tail -30 ~/ro27_S1013a.job.log"`. Exit code 255 = the network dropped, not the job; reconnect.
  - **After about 1 h,** check one `R3` and one `R1` run in W&B (from the laptop, or via 01 §5.2 step 7 in a container): the `flags/ROGUE_ONE_EXTRACTOR_ROUNDS` value, the tags [`S1013a`, `prefreeze`, `agent2027-pre.N`], and a non-zero `tok_compl_extractor`.
  - A W&B `running` run with no live container is a zombie.
  - Crashes are retried once, alone, by the launcher. If the server died, re-plan the unfinished rows (same `row_id`) into the contingency session after A03's `S1014b`: `S1014a` if it starts on Wed 14, else `S1015h`. Run a new pre-flight first.
  - A03-T10 waits for `S1013a` to end.

  *Done when:*
  - every row is `OK`, `CAP_HIT` or `FAILED`-after-retry, and is matched in W&B;
  - the tear-down check `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker ps --format '{{.Names}}' | grep -E '^(ro27-|rogueone-vllm-)' || echo none; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"` prints `none`, and the 4 GPUs are back below 2,000 MiB.

### 5.4 Analysis

- [ ] **A02-T12** [type: analysis] (0.5 d, due **Thu 15 Oct, 12:00**) — Write `scripts/agent2027/analysis/a02_readout.py`, commit and push it, and run it on CAIR in a CPU container. The φ metadata is mounted only into this analysis container, never into an agent container. The working tree is not checked out, so a running A03 session is not touched:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "mkdir -p ~/ro27_a02_analysis && cd <ROGUEONE_CAIR_PATH> && GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15' git fetch git@github.com:HenrikNorkart/RougeOne.git agent2027/v2 && git archive FETCH_HEAD scripts/agent2027 | tar -x -C ~/ro27_a02_analysis && docker run --rm --cpus 16 --env-file <CAIR_HOME>/code/PlanBench/.env -e PYTHONPATH=/opt/ro27 -w /opt/ro27 -v ~/ro27_a02_analysis/scripts/agent2027:/opt/ro27:ro -v <ROGUEONE_CAIR_PATH>/tasks:/workspace/tasks:ro -v ~/rogueone-synp-meta:/meta:ro --entrypoint python3 rogue_one:agent2027-pre.N analysis/a02_readout.py --meta /meta --tasks /workspace/tasks"
  ```
  The script:
  1. pulls group `A02` from `PlanBench/RogueOne-AGENT2027-prefreeze` with the `run-output` artifacts, keeping only `config.code_phase == "prefreeze"` (this drops the smoke rows A02-901/902);
  2. checks the §4 flag invariant;
  3. runs the detector over 3 × 40 episodes;
  4. joins A08's `base_no_fe` (seed 42) to compute `success`;
  5. sets `W_ref` = the median `wall_s` of A03's `replace` rows (real data, R=10, 6 clients). If those rows are not done, it uses A01's median, labelled "planning only";
  6. computes the four §6 criteria, a task × arm table, and the cost table: `u_run`, `wall_s_extractor`, `tok_compl_*`, `tok_prompt_*`, `max_prompt_tokens`, `context_errors`, `Extractor_tool_usage_failed/runner` (summed over iterations), `serper_calls`;
  7. reports the criteria again without the `M5S` non-discriminating tasks, descriptively (the rule itself uses all 40 episodes).

  *Done when* `ANALYSIS__a02_readout__20261015` (project `RogueOne-AGENT2027`, group `ANALYSIS`) shows PASS/FAIL per criterion with its numbers, `W_ref` with its source, the R3/R10 and R1/R10 u ratios, each arm's mean u per SYN-P run, and the flag-invariant check.
- [ ] **A02-T13** [type: decision] (0.1 d, due Thu 15 Oct; applied Fri 16 Oct) — Hand off and decide.
  - By Thu 15, 09:00: export the `synp_p7` queries from `retrieval_transcript.jsonl` (`R10` rows first) to X9, A14-T3 and A14-T4.
  - Check the P7 leak canary: `rec_any` should be 0 of 15 episodes.
  - Apply §6, and write `ROGUE_ONE_EXTRACTOR_ROUNDS=<3|10>` into `v2_defaults.env` in the freeze merge.

  *Done when* the G1 entry of the draft `docs/agent2027/preregistration.md` records the four criteria, the adopted R, the mean u per SYN-P run of `R10` and `R3`, the canary result, and whether CL1–CL4 apply.
- [ ] **A02-T14** [type: analysis] (0.5 d, due Tue 3 Nov; needs E11b, due Tue 27 Oct) — Exploratory, outside Holm, raw p only:
  - exact McNemar on the 40 paired episodes (R3 vs R10, R1 vs R10);
  - the `R10` round profile: the share of accepted features first appended in round 1, rounds 2–3 and rounds 4–10 (round index = the number of "System now wants to focus on:" user messages in the traced request), plus runner failures and `context_errors` by round;
  - `fun_*` and `fallback_share` per arm.

  *Done when:* the tables are logged under `ANALYSIS__a02_readout__<YYYYMMDD>`.

### 5.5 Paper outputs

- [ ] **A02-T15** [type: writing] (0.1 d, due Fri 16 Oct) — Write the pre-registration paragraph: the rule as applied and the result. Disclose one seed (with the error rates), replace mode, live search, and that pre-freeze code was seen before the pre-registration. *Done when:* it is in the `agent2027-prereg` commit (X10).
- [ ] **A02-T16** [type: writing] (0.25 d, due Tue 3 Nov) — Write three outputs:
  - **C6 design sentence:** *"R was frozen at R=… after a pre-freeze pilot: R=3 recovered x/40 vs y/40 fold-episodes at z× the wall clock"* (a template, not a result);
  - **appendix table:** task × arm, with `rec_sel`/5, `success`, u, tokens per role and `context_errors`;
  - **A21/A23 inputs:** the three (R, cost, recovery) points for A23's **separate pre-freeze panel**, and the `R10` within-pilot truncation for A21. These points never go on the frozen-v2 front and never truncate frozen Full.

  *Done when:* all three are in the paper draft.

## 6. Decision rules and gates (committed and pushed Tue 13, before launch; A02-T3)

**R rule.** Outcome `rec_sel`; reference `R10`. The rule uses all 8 tasks (40 episodes), including any task `M5S` labels non-discriminating. Adopt **R=3** if and only if all four conditions hold:
1. Σ`rec_sel`(R10) − Σ`rec_sel`(R3) ≤ 4 (of 40). This is the net paired loss; concordant tasks such as P7 at 0/5 do not move it.
2. No task falls from ≥ 3/5 under `R10` to 0/5 under `R3`.
3. The `success` count of `R3` is ≥ that of `R10` (out of 8).
4. Mean `u_run`(R3) ≤ 0.75 × mean `u_run`(R10), means over the 8 rows of each arm (ITT; a cap hit = 6 h).

Otherwise **R=10**. **R=1 is never adopted**, whatever its numbers.

**Error rates** (pre-registered). Criterion 1 wrongly rejects a truly equivalent R=3 about 10–17% of the time (the upper figure assumes within-run correlation). It adopts an R=3 that truly loses 4 episodes about half the time. No equivalence claim is made.

**Gates:**
- **`M5S` (Tue 13, morning).** If ≥ 4 of 7 families are brute-force-solvable, recalibrate SYN-P and re-screen Tue 13 night. A02 then slips by ≤ 1 day, and the late-launch rule applies.
- **Late launch.** If A02 cannot start before Wed 14, 08:00, it runs after A03's Core-10 session (`S1014b`), as session `S1014a` (or `S1015h` if it starts on Thu 15). `R1` is then dropped by a dated amendment (−8 runs, −3.2 to −6u), and the rule is unchanged.
- **Detector.** If the detector unit tests (A02-T1) have not passed by Thu 15, 12:00: R=10, counted as "R=3 not adopted".
- **No read-out by Fri 16, 09:00:** R=10, counted as "R=3 not adopted".
- **G1 (Fri 16).** R goes into `v2_defaults.env`. **Re-tier rule:** apply CL1–CL4 at once (CAP-2 SYN-P only; CAP-1 small-rung real data on 2 datasets; HC1 on 4 high-error datasets; drop CAAFE) if R=3 is not adopted, **or** if the mean u per SYN-P run of `R10` is > 0.85u.
  - This is the canonical wording of the T1-minimum breakdown §2, 01 §4.9 and A07-T9.
  - The audited breakdown proposed using the adopted arm's u instead. That variant is strictly weaker: since criterion 4 forces u(R3) ≤ 0.75 u(R10), it triggers only in a subset of the canonical cases. It is not adopted unless Henrik switches it before A02-T3 (open question 5).
  - Add-backs AB1–AB5 are judged at G1 from the same measured u (01 §4.9); no numeric trigger is defined.
- **Leak canary.** Any P7 recovery → investigate with X9 before the pre-registration. It does not change the R rule.
- **After the freeze,** R is never re-opened, even if A03 adopts augment.

## 7. Risks and cut levers

| Risk | Mitigation |
|---|---|
| G0a slips or `M5S` forces recalibration | Late-launch rule; default R=10 |
| Detector not validated by Thu 15 | R=10 by default (detector gate); disclose |
| `.dev.env` is tracked at `07cbf93` and would override `-e` flags | A02-T1 checks it is untracked; A02-T8's build step refuses it |
| Wall-clock noise on a shared server | One interleaved session; also report tokens, `wall_s_extractor` and per-task ratios |
| One seed; replace mode biases against R=3 | Error rates stated; a pass is expected to carry over to augment; disclosed |
| `R1` raises fallback/abort rates; late `R10` rounds hit the context limit (`ext/agent.py:977-992`) | ITT; fallback never counts; failures by round (A02-T14) |
| A dataset abort exits rc 0 (`main.py:1589-1598`) | The launcher treats a missing `fold_aggregation_summary.csv` as a failure |
| A03's add-on pairs with `R10` on a possibly different pre-freeze image | Valid only if E6's replace-mode byte-identity check passes (A03 verifies) |
| AutoFeat is unseeded (Finding 43, commit `1bdaf42`) | A08's issue; A02 uses only the `M5S` verdict |
| Developer time: A02-T12 (0.5 d) + A03-T13 (0.5 d) exceed the 0.5 d the T1-minimum calendar books for both read-outs (Tue 13 – Fri 16) | Write `a02_readout.py` as a thin wrapper over E8a's detector and E11's fetch code; the readout deadline gates hold either way |

**Cut levers.** A02 is not among CL1–CL7. Its only optional part is `R1`, dropped only under the late-launch rule. A02's result **triggers** CL1–CL4 at G1.

## 8. Definition of done

- [ ] The rule was committed and pushed before launch (SHA and time in the session plan).
- [ ] All 24 rows are in group `A02`, finished or ITT-scored. Each has the R12 config keys, `flags/ROGUE_ONE_EXTRACTOR_ROUNDS` equal to its arm, and a `run-output` artifact. `S1013a__energy` is logged, and the tear-down is verified.
- [ ] `ANALYSIS__a02_readout__20261015` shows the criteria, `W_ref`, the flag-invariant check and the cost table.
- [ ] R is in `v2_defaults.env` at G1; the re-tier, P7 hand-off and canary are recorded in the draft pre-registration.
- [ ] The pre-registration paragraph, C6 sentence, appendix table and A21/A23 inputs are written.

---

**New names introduced** (not in the registry):
- `scripts/agent2027/analysis/a02_readout.py` (A02-T12);
- `u_run` and `W_ref` (analysis quantities, not W&B keys);
- `rogueoneConfig.extractor_rounds`;
- `Extractor_tool_usage_failed/runner` (an existing code key, `ext/agent.py:1021-1027`);
- `~/ro27_a02_analysis/` on CAIR (an analysis scratch copy of `scripts/agent2027/`);
- **session plan**: a free-text checklist kept per session at `docs/agent2027/sessions/<SESSION_ID>.md` (01 R6; A01 uses `docs/agent2027/sessions/a01_session_plan.md`), committed with the session file.
- sessions `S1012a` (the G0a smoke session, which A07-T6 also names), `S1013a`, and the contingencies `S1014a` / `S1015h`; rows `A02-001…024` and `A02-901/902`. All follow the R7/R8 patterns.

**Open questions for Henrik:**
1. Confirm `<ROGUEONE_CAIR_PATH>`.
2. D4 is undecided at launch; A02 assumes stratified folds on the SYN-P classification tasks.
3. Does `agent2027-pre.N` contain E6 at launch? (Assumed no: E6 is due Tue 13, end of day.)
4. "Loses ≤ 4 of 40" is read as the difference in totals (= net paired loss).
5. Re-tier trigger: keep the canonical "R10 mean u per SYN-P run > 0.85u, or R=3 not adopted", or switch to the audited "adopted arm's u"? Decide before A02-T3 (Tue 13, 10:00); the default is canonical.
6. Cross-document mismatches (A03 smoke-row IDs `A07-901/902` and A07-T6's host-session list) were reconciled in the cross-document pass; A07-T6 now names `S1012a`, `S1013a` or `S1014b`.
