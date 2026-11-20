# A09 — RT3: Oracle and misdirected planner (positive control, gate G3)

*Rogue One v2 design study · ICSE AGENT 2027 · submission Fri 13 Nov 2026 · scope T1-minimum · written Wed 7 Oct 2026. Every "hoped" result below is a hypothesis, not a finding. Code references: RougeOne repo, branch `nested-cv-rebuttal` @ `07cbf93` (`C:/code/PHD/RougeOne`); each line was re-read for this document. The branch tip is now `1bdaf42`, which changes only files under `paper/` (scripts, result CSVs and notes), none of which is cited here. Short paths: `main.py` = `rogue_one_main.py`; `sci/agent.py`, `ext/agent.py`, `tst/agent.py` = `rogueone/llm/agents/core/{scientist,extractor,tester}/agent.py`; `config.py`, `console.py` = `rogueone/utils/…`. Script paths `analysis/…`, `rt3_focus.py`, `make_manifest.py` and `prompt_diff.py` live under `scripts/agent2027/` (01 §7 R6). Shared definitions: `docs/ablations/01_shared_infrastructure.md` ("01 §n"). This document restates everything A09 needs. Items marked ‡ are proposals that Henrik freezes in the pre-registration (X10).*

| Field | Value |
|---|---|
| **ID** | A09 · RT3 · arms `oracle`, `misdirected` |
| **Claim / Holm contrast** | Planner half of gate **G3** (Mon 26 Oct). Supplies the oracle ceiling for the FX TOST margin δ (C2a, C2b) and the P7 implementation ceiling for KT3. **Outside Holm** (positive control, 01 §7 R14). |
| **Tier** | T1-minimum, on the **never-cut** list |
| **Owner** | Henrik |
| **Start / finish** | Tooling Sat 17 – Sun 18 Oct · hook and image `v2.1` Mon 19 Oct, morning · **runs Mon 19 Oct** (session `S1019a`, launch ≈ 13:00) · reruns by Wed 21 · δ committed **Fri 23 Oct** (A09-T15) · G3 read-out **Mon 26 Oct** |
| **Budget** | **15 new runs, 10.5–15u** (= 01 §7 R1 and the T1-minimum breakdown). That is 5.25–7.5 h of the 4-H100 slot at 2 u/h (21–30 H100-h), or 7.5–10.7 calendar hours at 0.7 utilisation (30–43 H100-h held at TP4). **Not in the T1-minimum count:** smoke row `A09-000` (1 fold of Full-v2 for the v2.1 prompt guard, ≈ 0.14–0.2u, from the reserve; §4.5). Control: A05 runs, 0 new. Options: AB3 +1 run (+1u); misdirected on `synp_p7` +1 run (+0.7–1u). |
| **Models** | O120 = `openai/gpt-oss-120b` (vLLM `v0.22.0`, port `8010`) for E, T and W. **S makes no LLM calls** in either arm. |
| **Status** | Not started |
| **Depends on** | E1, E2, E3 (focus log), E4, E5, E6, E8a (oracle strings, detector), **E9c**, E11a, X10 · A05 (control, misdirected sources, `a05-handoff`), A08 CPU screen (brute-force gate), the GLMM/exact part of `analysis/glmm_holm.py` (E11b, pulled forward to Sun 25) · D4, D5 · G1, G2 |
| **Feeds into** | G3 · A12 (TOST margin δ, committed by A09-T15; A12 cites it as "A09-T15") · A14 (P7 ceiling) · A19 (the audited breakdown also sets the CAP-2 1B-cell margin from the Full-to-oracle gap; **to verify** once A19 is written) · how A10, A11 and A12 word planner nulls · A21, A23 (reference points) · Table 2 assay row · Fig. 2 · A16 (TOST margin δ); A17 (RV1 margin δ, A09-T15) |

## 1. What we test and why it matters

Each iteration, the Scientist (S) writes a short "focus", and the Extractor (E) is told to follow it. A09 replaces S with a fixed focus read from a file. In `oracle`, the focus states the planted feature φ exactly. In `misdirected`, it is the focus that S wrote on a *different* SYN-P task, with that task's columns renamed to decoys. If recovery rises under the oracle or falls under misdirection, the recovery instrument can see a planner change. The planner nulls in FX (C2c), LF1 (C3) and LF2-L0 can then be read as real nulls. The oracle also separates planning bottlenecks from execution bottlenecks, and it sets the ceiling used for FX's equivalence margin.

## 2. Background you need

**2.1 The planner channel**
- **Loop.** `forward_pass` (`main.py:536`) derives `fold_run_id` and `iteration_in_fold` from the global `index` (`:539-540`). At iteration 0, `_reset_fold_state` (`:543-544`, defined `:417-457`) empties `focus_history` and S's notebook.
- **S writes the focus** in `determine_focus` (`main.py:548-555`; signature `sci/agent.py:75-83`).
  - Prompt: "Round 1" when the focus history is empty, otherwise "Round {index}/{N}" with the focus history (`sci/agent.py:472-475`; E2 makes the round number per fold).
  - Tools: pandas, search, attribute lookup, list and notebook (`:461-469`); "Always perform at least one web search" (`:431`).
  - Failure path: 100 retries (`:519`), then a no-tool fallback (`:549-577`), then a heuristic focus (`:501-517`, `:592`).
- **Recording.**
  - The focus is appended to `focus_history` (`main.py:556`).
  - It is printed (`:557`), but the print is lost when the console is disabled (`console.py:31-41`), and container stdout is not part of `run-output`.
  - It is logged to W&B with step demotion (`sci/agent.py:600-601`).
  - `_log_extractor_iteration_activity` receives it but does not log it (call `main.py:594-604`, body `:905-980`). Only the zero-output abort path logs it (`:629-637`, `:1036`, `:1068`).
  - A09 therefore reads the **E3 focus log** (§2.6) inside each `run-output` artifact.
- **How E uses the focus.**
  - `generate_patient_attributes` gets it (`main.py:563-571`) and sends "System now wants to focus on: {focus}." (`ext/agent.py:971`) in each of the R rounds of one session (`:965-967`).
  - E's prompt says S defines the focus (`:875`) and "Follow the provided focus strictly" (`:892`).
  - E also sees every raw column (`:914-915`) and has its own search tool (`:939`, `:166`).
  - E must append at least once (`:894-899`). Duplicate names are rejected (`:486-489`), a pruned name is restored instead (`:465-485`), and a baseline fallback runs if nothing is appended (`:996-1003`).
  - Zero productive features at iteration 0 abort the dataset; at a later iteration only that iteration is skipped (`main.py:606-663`).
  - The Tester never sees the focus (`tst/agent.py:1437`, `:1442` only mention it).
  - **E can therefore find φ without a useful plan.**

**2.2 What the E9c hook changes.**
- Under `ROGUE_ONE_PLANNER_MODE=oracle|misdirected`, `main.py:548-555` is skipped. The focus for (`fold_run_id`, `iteration_in_fold`) comes from `ROGUE_ONE_PLANNER_FOCUS_FILE`. E, T, pruning, selection and E/T search are unchanged.
- **Side effects of skipping S.** S makes no tokens and no searches, never reads the feedback, and the plan is fixed in advance (open loop). The generic-focus arm is cut, so "wrong content" cannot be separated from "fixed plan".
- `llm` (Full, the default) must stay byte-identical. `none` belongs to E9d and is not used here.

**2.3 Instruments.** The SYN-P tasks are built by E8a: n = 2,000, p ≈ 20–30, 3 relevant columns plus correlated decoys. Folders are `tasks/<task_type>/synp_<id>/`. Types are proposed; E8a confirms them in `synp_meta/registry.yaml`.

| `dataset_id` | φ | Family / depth | Type | A09 arms |
|---|---|---|---|---|
| `synp_p1` | x_a/x_b | ratio, depth 1 | R | both (probably non-discriminating) |
| `synp_p2` | (x_a/x_b)·x_c | ratio × product, depth 2 | R | both |
| `synp_p3` | 1[x_a/x_b ≥ t] ∧ cat = k | threshold-AND of a ratio, depth 3 | C | both (multi-step) |
| `synp_p4` | sin/cos of an hour-like column | cyclic | C | both |
| `synp_p5` | sign(LW·LD² − RW·RD²), real balance-scale rows and names | counterfactual rule | C | both (multi-step) |
| `synp_p6a` / `synp_p6b` | named BMI-like composite / its exact anonymised twin (f1..fp, opaque experiment name) | knowledge vs search | R | both |
| `synp_p7` | ratio × threshold composite documented only in the local corpus, with an unguessable constant | knowledge-only | C | `oracle` only (KT3's ceiling) |

- **Sets.** SYN-P 8 = p1, p2, p3, p4, p5, p6a, p6b, p7; SYN-P 7 = SYN-P 8 without p7. `synp_p0` is not run.
- **Brute-force gate** (A08 CPU screen, read Tue 13 Oct). A task is non-discriminating if OpenFE or AutoFeat, each judged separately, recovers φ in ≥ 14 of 15 fold-episodes (5 folds × seeds 42, 4, 5). Such tasks still run but leave the G3 population.
- **Metadata stays off the mount.** φ and the oracle strings live in `~/rogueone-synp-meta/`, which is never mounted into an agent container. The image copies the whole build context (`Dockerfile:12-13`, `COPY . .`), and `.dockerignore` excludes neither `tasks/` nor `scripts/`. Focus files must therefore **never** be in the repo or the build context.

**2.4 Metrics** (W&B summary keys, 01 §4.6)
- **`rec_sel` (primary).** An active, non-fallback feature has |Spearman| ≥ 0.95 with φ on training rows (NMI ≥ 0.9 for p3, p5 and p7; no credit for p7's ratio alone), and it is present in the iteration that inner CV selects. The unit is the fold-episode. Raw columns and single-column monotone transforms never count.
- **Other outcomes:**
  - recovery: `rec_any`, `iter_first_rec`;
  - success: `success`, `func_success`, `cap_hit`;
  - funnel: `fun_proposed` → `fun_executes` → `fun_valid` → `fun_replay_ok` → `fun_survives_prune` → `fun_selected`;
  - process: `reproposal_ratio`, `restore_pruned_count`, `fallback_share`, `focus_adherence` (new, §9);
  - cost: `tok_compl_*`, `llm_calls_*`, `wall_s`. Manipulation check: `llm_calls_scientist` = `tok_compl_scientist` = 0 in every A09 arm run.

**2.5 Full-v2.** The frozen reference from G1 (Fri 16 Oct, image `rogue_one:agent2027-v2.0`). A09 runs on `rogue_one:agent2027-v2.1`, which is v2.0 plus E9c, with Full byte-identical.

**2.6 Focus log (E3).** One JSON row per (fold, iteration), written next to `main.py:548-557` into `output/focus_log.jsonl` (file name: E3 fixes it; **to verify** against E3 and use E3's name everywhere below). Keys: `fold_run_id`, `iteration_in_fold`, `focus`, `source` (`llm` under Full; the planner mode under A09). A complete run has 50 rows. The run's `output/` folder sits inside its task folder (`/workspace/tasks/<task_type>/<dataset_id>/output/`, from `output_dir` in `config.yml`).

## 3. Hypotheses and interpretation (all hypotheses)

| Outcome | Meaning | Paper sentence / action |
|---|---|---|
| **Hoped (route 1):** oracle ≥ Full > misdirected | Recovery sees planner changes; the plan is a causal channel | *"Recovery responds to the plan: an exact plan raises it and a fixed off-task plan lowers it."* Never claim that wrong *content* alone does this. |
| … oracle > Full mainly on p3/p5 | On multi-step features the bottleneck is planning | *"On multi-step features the bottleneck is planning, not execution."* |
| … oracle ≈ Full, misdirected < Full | G3 passes (route 1). S already finds what an oracle would say on these tasks | δ falls to the 0.10 floor (§6) |
| **Alternative pass (route 2):** misdirected ≈ Full, oracle > Full | Robustness: bad advice is ignored, good advice is used | G3 passes (pre-registered). FX's P ≈ 0 then reads "E compensates". |
| **Null:** oracle ≈ Full ≈ misdirected | The planner channel is inert at 120B; E finds φ by itself | The planner half of G3 fails. C2c, C3 and L0 nulls are worded "planner channel inert", never "planning is unnecessary". The assay row stays in Table 2. |
| **Opposite (b):** oracle < Full, or oracle fails on a task | Execution bottleneck (code, replay or pruning) | The funnel-loss table shows the stage |
| **Opposite (c):** misdirected > Full | S's own plans lead E away from φ | Evidence against the LLM planner; consistent with a negative FX P effect. **Does not pass G3 on its own** (§6). |

## 4. Design

**4.1 Arms** (overrides relative to `scripts/agent2027/v2_defaults.env`). On `synp_p3`, `p4`, `p5` and `p7`, rows also carry `"ROGUE_ONE_STRATIFIED_FOLDS":"True"` if D4 is adopted, exactly as A05 does.

| Arm | `env_overrides` | Focus at every (f, i) |
|---|---|---|
| Control: A05 `full` | `{}` | S (LLM) |
| `oracle` | `{"ROGUE_ONE_PLANNER_MODE":"oracle","ROGUE_ONE_PLANNER_FOCUS_FILE":"/workspace/focus/focus.json"}` | one fixed string stating φ_T |
| `misdirected` | `{"ROGUE_ONE_PLANNER_MODE":"misdirected","ROGUE_ONE_PLANNER_FOCUS_FILE":"/workspace/focus/focus.json"}` | A05's f42 r0 focus on σ(T) at (f, i), after renaming |
| Smoke `A09-000` (guard only, never analysed) | `{"ROGUE_ONE_SMOKE_FOLDS":"1"}`, arm_id `full`, `notes=smoke` | S (LLM); Full-v2 on v2.1, 1 fold |

**4.2 Focus construction** (frozen in X10)
- **Oracle template ‡.** *"Build exactly this feature with `append_new_attribute`: {name} = {expression in T's raw column names}."* For p3, p5 and p7 it adds *"Build it in steps: …"*.
  - {name} is the neutral `oracle_feature` on every task ‡, so neither P6 twin gets a formula name and `synp_p6b` sees only f-names.
  - All constants are given (t and k for p3; the hidden constant for p7).
  - The same string is used at all 50 (f, i). From iteration 1 on, re-appending `oracle_feature` is rejected as a duplicate or restores it if pruned (`ext/agent.py:465-489`). This is intended: the arm measures whether φ is built and kept.
  - E8a writes the string into `synp_meta/<id>.json` (key `oracle_focus`). X10 commits its SHA-256 hash, not the text.
- **σ map ‡.** σ never maps a task to itself or to a task of the same operator family. The ratio-like targets (p1, p2, p3, p6a, p6b) draw from {p4, p5}; p4 and p5 draw from the ratio-like group. p7 is never a source (it is ratio-like and Full ≈ 0 on it). Both twins share one source, so they stay comparable.

  | T | p1 | p2 | p3 | p4 | p5 | p6a | p6b | p7 (option only) |
  |---|---|---|---|---|---|---|---|---|
  | σ(T) | p4 | p4 | p4 | p2 | p1 | p5 | p5 | p4 |
  | fallback source | p5 | p5 | p5 | p1 | p2 | p4 | p4 | p5 |

  - A source fold is **missing** if A05's f42 r0 run on σ(T) has fewer than 10 focus rows for it (crash, cap or iteration-0 abort). Then use σ(T)'s f42 r1 (A/A) for that fold, then the fallback source's f42 r0, then its f42 r1. `rt3_sigma_map.yaml` holds both rows.
  - S's heuristic-fallback strings are kept verbatim and counted.
- **Renaming rule ‡.**
  1. Let I_T be φ_T's input columns. Let D_T be T's other raw columns, sorted by ascending max |Spearman| with I_T (categoricals as ordinal codes; ties broken by column order).
  2. Let C_S be σ(T)'s raw-column names that occur in any source string, with φ_S's inputs first and the rest in order of first occurrence.
  3. Map C_S[k] to the k-th unused column of D_T of the same type (numeric or categorical), falling back to any type. Reuse columns cyclically if D_T runs out.
  4. Replace in a single pass, longest name first, case-sensitively. A match must not touch a letter or digit, except at an `_` boundary, so `LW_x_LD` is rewritten too. Numbers and level labels stay unchanged.
  5. **Assert:** no I_T name remains, and every C_S name was replaced.
- **AB3 (junglechess) ‡.** Source: A05's `housing` f42 r0 focus. junglechess has no planted φ, so I_T is empty and D_T is all raw columns in column order; steps 2–5 apply unchanged.
- **Focus file** (mounted, minimal): `{"dataset_id": "...", "arm_id": "...", "focus": {"<f>:<i>": "<string>"}}`, where f = `fold_run_id` (0–4) and i = `iteration_in_fold` (0–9), with all 50 keys. The rename map and provenance go in a sidecar that is never mounted, because the map reveals which columns are decoys.

**4.3 Fixed settings and seeds.**
- **As Full-v2 (01 §4.3):** 10 iterations × 5 folds; `inner_cv` selection; rollback off; temperature 0.7; R and pool mode as frozen at G1; XGBoost with 400 trees; `PARALLEL=6`; `CPUS=12`; 6 h cap (`RUN_CAP_HOURS=6`); shared Serper cache; `ROGUE_ONE_LLM_TRACE=1`; `ROGUE_ONE_PROMPT_DUMP=1`; the same topology as A05 (TP4, or TP2 per D5).
- **Seeds:** `fold_seed` 42, `llm_seed` 42, `replicate` 0, so every arm shares A05's seed-42 partition.

**4.4 Run matrix** (manifest `manifests/agent2027/A09_RT3.csv`)

| Rows | Group | `arm_id` | Datasets | Seeds | Runs | u | Run name |
|---|---|---|---|---|---|---|---|
| A09-001…008 | Oracle | `oracle` | SYN-P 8 in the order p1, p2, p3, p4, p5, p6a, p6b, p7 | f42 r0 | 8 | 5.6–8.0 | `A09__oracle__synp_<id>__f42__r0` |
| A09-009…015 | Misdirected | `misdirected` | SYN-P 7 in the order p1 … p6b | f42 r0 | 7 | 4.9–7.0 | `A09__misdirected__synp_<id>__f42__r0` |
| — | Control (reused) | A05 `full` | SYN-P 8 | f42 r0, f42 r1, f4 r0 | 0 | 0 | `A05__full__synp_<id>__f<seed>__r<rep>` |
| A09-000 | Smoke (guard) | `full` | the SYN-P task of `prompt-baseline-v2.0` | f42 r0, 1 fold | (1) | ≈ 0.14–0.2 | `A09__full__<task>__f42__r0` |
| A09-016 | Option AB3 | `misdirected` | `junglechess` | f42 r0 | +1 | +1.0 | `A09__misdirected__junglechess__f42__r0` |
| A09-017 | Option | `misdirected` | `synp_p7` | f42 r0 | +1 | +0.7–1.0 | `A09__misdirected__synp_p7__f42__r0` |

**4.5 Budget.**
- 8 × 0.7–1.0u + 7 × 0.7–1.0u = **10.5–15u**, or 7.5–10.7 h at 1.4u per calendar hour. This matches R1 and the T1-minimum breakdown exactly. Runs should cost slightly less than Full, because S is never called.
- **Difference from the breakdown:** smoke row `A09-000` adds ≈ 0.14–0.2u (1 of 5 folds of a SYN-P run). It is needed for merge guard 1 on v2.1 (01 §2.2) and is paid from the reserve, like a sentinel.
- **Other rows that may share the session** (paid by their own documents): A06's `sentinel` (0.7–1.0u, if A06 Q1 = yes); A07's 12 rows (12–15u) if A07-T14 fired. With A07 the session grows to ≈ 17–21 h, ending Tue 20 morning, and `S1020a` (A10/A11) slips by the same amount.
- AB3 is +1u, from the add-back budget (G1) or the reserve (G2, §6).

**4.6 Pairing and randomisation.** One O120 session, `S1019a`. Rows are interleaved by `order_key` (RNG seeded by the session ID), except that `A09-000` and the A06 sentinel get the lowest `order_key` values (first wave). Each row is paired with A05's seed-42 runs on the same task.

**4.7 Failures** (01 §4.5).
- A crash (rc ≠ 0, or no `fold_aggregation_summary.csv`) is retried once, alone, at the end of the session.
- A persistent failure or a cap hit is an ITT failure: all 5 episodes get `rec_sel = rec_any = 0`, and the run is never dropped.
- Infra faults are re-queued with `notes=infra-rerun`, keeping `image_tag=agent2027-v2.1`.
- A row that stops at focus-file validation, before any LLM call, is a setup fault: fix the file and relaunch the row.

## 5. Tasks

### 5.1 Prerequisites (verify before launch)
- [ ] **A09-T1** [type: decision] (0.25 d, due Thu 15 Oct; commit Fri 16) — Write into `docs/agent2027/preregistration.md`:
  - the ‡ items of §4.2 (template and `oracle_feature`, σ map with fallbacks, renaming rule, AB3 source) and the oracle-string hashes;
  - the §6 rules: population, test of record and fitting routine, exact fallback, small-population rule, routes and α, the headroom flag, ITT sensitivity, the incomplete-data rule;
  - the δ formula, gap definition and floor (shared with A12-T1).

  *Done when:* they are in the commit tagged `agent2027-prereg`, before row A05-001 starts.
- [ ] **A09-T2** [type: config] (0.1 d, due Mon 19 Oct, 10:00) — Tick the shared prerequisites against their 01 §3 "done when" checks:
  - E1, E2, E4, E5 (W&B artifact `prompt-baseline-v2.0` exists), E6;
  - E3, with a focus log that has 50 rows per A05 SYN-P run;
  - E8a: 9 SYN-P folders, `oracle_focus` strings in `~/rogueone-synp-meta/`, detector tests green;
  - the A08 gate result (`ANALYSIS__a08_screen__20261013`);
  - `.dev.env` is absent from branch `agent2027/v2` (it is tracked at `07cbf93`; E1 removes it), otherwise the build step in A09-T8 refuses.

  *Done when:* each check is ticked, with a link, in the session plan `docs/agent2027/sessions/S1019a.md`.
- [ ] **A09-T3** [type: data] (0.1 d, due Mon 19 Oct, 10:00) — Confirm three things:
  - G2 is a go (A05-T15);
  - the G2 memo exists (`ANALYSIS__g2_stats__20261019`), with the headroom flag (§6);
  - A05's f42 r0 and r1 runs on the sources `synp_p1`, `p2`, `p4` and `p5` (plus `housing` f42 r0 if AB3 is funded) are `finished`, with focus logs (artifact `a05-handoff`, A05-T16).

  *Done when:* `analysis/fetch_wandb.py` lists these runs and their `run-output` artifacts.

### 5.2 Implementation (E9c = A09-T4 + T5, 0.5–0.75 d as in R2; A09-T6, T7 and T9 are A09's own checks, 0.3 d; image v2.1)
- [ ] **A09-T4** [type: code] (0.25–0.5 d, due Sun 18 Oct) — Write `scripts/agent2027/rt3_focus.py` (subcommands `extract`, `rename`, `build`, `verify`) and `scripts/agent2027/rt3_sigma_map.yaml`, implementing §4.2. Test them on the G0a smoke's focus log and on any A05 SYN-P run already finished. *Done when:* `verify` checks that:
  - every file has 50 keys `"<f>:<i>"`;
  - each oracle string equals its `oracle_focus` metadata string;
  - each misdirected (T, f, i) equals A05's renamed σ(T) focus at (f, i), with any fallback recorded in the sidecar;
  - no I_T name appears in any misdirected string;
  - no file contains another task's φ.
- [ ] **A09-T5** [type: code] (0.25 d, due Mon 19 Oct, 11:00) — Build the hook.
  - In `rogueoneConfig` (`config.py:303-337`), read `ROGUE_ONE_PLANNER_MODE` (default `llm`; any other value than `llm`/`oracle`/`misdirected`/`none` raises) and `ROGUE_ONE_PLANNER_FOCUS_FILE`, and register both in E1's flag registry.
  - In `forward_pass`, branch around `main.py:548-555`. Load the file once. Check that its `dataset_id` equals the single entry of `ROGUE_ONE_DATASETS`.
    - Do **not** compare with `experiment_name`: `synp_p6b`'s is opaque. `select_paths` matches the folder name (`main.py:1573`).
    - Check that the file has `k_folds × num_iterations` keys. Fail before any LLM call otherwise.
  - Keep `:556`. Write `source` = mode into the E3 focus log.

  *Done when:* `tests/unit/test_planner_hook.py` passes:
  - under `llm`, `determine_focus` is called;
  - under `oracle` and `misdirected`, it never is, and E receives the file string for each (f, i);
  - a missing key or a dataset mismatch raises an error.
- [ ] **A09-T6** [type: code] (0.1 d, due Mon 19 Oct, 11:00) — Verify that E1's `ro27_clients_manifest.sh` mounts `~/rogueone-rt3-focus/<row_id>.json` at `/workspace/focus/focus.json:ro` only when `env_overrides` sets an `oracle` or `misdirected` mode, and that it refuses the row if the file is missing. Add this if it is absent. *Done when:* a dry run with one A09 row and one A05 row shows the mount only on the A09 row.
- [ ] **A09-T7** [type: config] (0.1 d, due Mon 19 Oct, 11:30) — Run `make_manifest.py --ablation A09` and commit `manifests/agent2027/A09_RT3.csv` (R7 header). Then plan the session.
  - Rows `A09-000`…`A09-015`, plus the options `A09-016` (AB3) and `A09-017` (p7), as in §4.4.
  - Every row: `rung_S/E/T` = O120, `port_*` = 8010, `image_tag` = `agent2027-v2.1`, `cap_hours` = 6, `notes` = "pair A05 f42" (`smoke` on A09-000). `expected_u` = 0.85 on SYN-P rows, 0.17 on A09-000, 1.0 on A09-016.
  - `make_manifest.py --plan-session S1019a --rows A09-000…A09-015`, plus A06's `sentinel` row if A06 Q1 = yes, A07-001…012 if A07-T14 fired, and A09-016 if AB3 is funded. A09-000 and the sentinel take the lowest `order_key`. Under D5 = TP2, split the rows alternately into `S1019a` (port 8010) and `S1019b` (port 8011).
  - Commit and push before A09-T8, so the image and the server checkout carry both files.

  *Done when:* the header validates, run names are unique, the stratified-fold flag matches A05 per task, and `order_key` is a permutation.

### 5.3 Execution on CAIR
**Hard rules (01 §5.1), for every step below:**
- **Containers only:** every execution (runs, focus building, analysis, W&B queries) happens inside Docker. The host only issues `docker` / `docker compose` commands and the bash launchers in `~`; `nvidia-smi`, `docker ps`, `tail`, `grep`, `mkdir`, `cp` are for pre-flight, monitoring and installation only. Never run `python` on the host. Any `docker compose run` over SSH needs `-T`.
- **GPU pre-flight** before any GPU container (A09-T11). If the GPUs are not free, **stop and report** the per-GPU memory and utilisation, compute apps and running containers to Henrik.
- **≤ 4 H100s** without Henrik's explicit approval in chat; one Rogue One job at a time.
- **W&B is the system of record:** runs are stateless and upload `output/` as artifact `run-output`; entity `PlanBench`, project `RogueOne-AGENT2027`, group `A09`, job_type = `arm_id`, tags [`S1019a`, `v2`, `agent2027-v2.1`].
- **Secrets** come from `.env`: `WANDB_API_KEY` from `<CAIR_HOME>/code/PlanBench/.env`; Serper and LLM settings from `<P>/.env` / `.secret.env`, baked into the image (never push it).
- **Long jobs** run detached with `nohup`; logs go to `~` (`~/ro27_S1019a.job.log`, `~/vllm_session_S1019a.log`), never into the root-owned run folders.
- **Code reaches CAIR** only by local commit and push, then `git fetch` over SSH on the server.

Below, `$SSH` = `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14`, `$SSH_MON` = `ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 CAIR-gpu14`, and `<P>` = `<ROGUEONE_CAIR_PATH>` = `<CAIR_HOME>/code/RougeOne` (**inferred; verify with `$SSH "ls -d ~/code/RougeOne"`**). The git remote `git@github.com:HenrikNorkart/RougeOne.git` is also inferred.

- [ ] **A09-T8** [type: run] (0.1 d, due Mon 19 Oct, 12:00) — Tag `agent2027-v2.1` locally and push `agent2027/v2` with the tag, then on CAIR:
  - **Sync:** `$SSH "cd <P> && GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15' git fetch git@github.com:HenrikNorkart/RougeOne.git agent2027/v2 && git checkout -B agent2027/v2 FETCH_HEAD && git log -1 --format=%h"`
  - **Build:** `$SSH "cd <P> && test ! -e .dev.env && docker build -t rogue_one:agent2027-v2.1 . && docker image inspect rogue_one:agent2027-v2.1 --format '{{.Id}}'"`
  - **Install the launchers and manifests:** `$SSH "cd <P> && cp scripts/agent2027/ro27_session_launch.sh scripts/agent2027/ro27_clients_manifest.sh ~/ && cp paper/scripts/vllm_session.sh ~/ro27_vllm_session.sh && mkdir -p ~/ro27_manifests && cp -r manifests/agent2027/. ~/ro27_manifests/"`

  *Done when:* HEAD equals the `agent2027-v2.1` tag commit, and HEAD and the image ID are in the session plan.
- [ ] **A09-T9** [type: config] (0.1 d, due Mon 19 Oct, 12:30) — Merge guard 2 and the leak check on the image built in A09-T8. Guard 1 (the prompt diff) needs a live server and is done in A09-T12; guard 3 is A06's sentinel (01 §2.2).
  - Unit tests are green (including A09-T5's).
  - `$SSH "docker run --rm --entrypoint find rogue_one:agent2027-v2.1 / \( -path /proc -o -path /sys \) -prune -o \( -name 'focus.json' -o -name 'synp_meta*' -o -name 'rogueone-synp-meta*' -o -name 'rogueone-rt3-focus*' \) -print"` prints nothing.

  *Done when:* both pass and the image ID is recorded in the session plan.
- [ ] **A09-T10** [type: data] (0.1 d, due Mon 19 Oct, 12:45) — Build the focus files in a CPU container:
  ```
  $SSH "mkdir -p ~/rogueone-rt3-focus && docker run --rm --cpus 4 --user \$(id -u):\$(id -g) -e HOME=/tmp --env-file <CAIR_HOME>/code/PlanBench/.env -v ~/rogueone-synp-meta:/meta:ro -v <P>/tasks:/tasks:ro -v ~/rogueone-rt3-focus:/out --entrypoint python3 rogue_one:agent2027-v2.1 /workspace/scripts/agent2027/rt3_focus.py build --verify --manifest /workspace/manifests/agent2027/A09_RT3.csv --sigma /workspace/scripts/agent2027/rt3_sigma_map.yaml --meta /meta --tasks /tasks --source-group A05 --out /out"
  ```
  (`HOME=/tmp` gives W&B a writable cache under `--user`; the script uses entity `PlanBench`, project `RogueOne-AGENT2027`.) *Done when:* 15 files `~/rogueone-rt3-focus/A09-001…015.json` (plus any funded option file) exist, `verify` passes, and the SHA-256 hashes and the `verify` report are logged as W&B run `ANALYSIS__rt3_focus__20261019` (group `ANALYSIS`). The file contents are uploaded later (A09-T15), after every A09 row has ended.
- [ ] **A09-T11** [type: run] (0.1 d, Mon 19 Oct, ≈ 13:00) — Pre-flight and launch.
  - **Pre-flight:** `$SSH "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader && echo '--- compute apps ---' && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader && echo '--- containers ---' && docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'"`
    - Proceed only if the chosen GPUs each use < 2,000 MiB, the total is ≤ 4, and no `ro27-*`, `rogueone-vllm-*`, `llmfe-*` or `caafe-*` container is running (for example the Sun 18 small-rung night session).
    - Otherwise **stop and report** the output to Henrik.
  - **Launch:** `$SSH "SESSION_ID=S1019a SESSION_FILE=~/ro27_manifests/sessions/S1019a.csv GPUS=<ids> MODEL=openai/gpt-oss-120b PORT=8010 SERVER_NAME=rogueone-vllm-o120-s1019a LOG=~/vllm_session_S1019a.log IMAGE_RO=rogue_one:agent2027-v2.1 PARALLEL=6 CPUS=12 RUN_CAP_HOURS=6 RUNROOT=~/rogueone-runs-agent2027 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_S1019a.job.log 2>&1 & echo launched; sleep 5; tail -5 ~/ro27_S1019a.job.log"`
  - **If D5 = TP2:** run a second job on 2 other GPUs with `SESSION_ID=S1019b`, `SESSION_FILE=~/ro27_manifests/sessions/S1019b.csv`, `PORT=8011`, `SERVER_NAME=rogueone-vllm-o120-s1019b`, `LOG=~/vllm_session_S1019b.log`, `ROGUEONE_ALLOW_SECOND_SERVER=1` and job log `~/ro27_S1019b.job.log`, with ΣGPU ≤ 4. A17's night session then takes the next free letter.

  *Done when:* the server is up and 6 `ro27-s1019a-*` client containers per server are running; the pre-flight output is in the session plan.
- [ ] **A09-T12** [type: run] (0.1 d, Mon 19 Oct, within 90 min of launch) — **First-wave check** (this replaces a separate smoke session). For two first-wave A09 arm containers (one per arm if possible):
  - **Focus applied:** `$SSH "docker exec ro27-s1019a-<row> python3 -c \"import json,glob,pathlib; f=json.load(open('/workspace/focus/focus.json'))['focus']['0:0']; o=glob.glob('/workspace/tasks/*/*/output')[0]; r=json.loads(open(o+'/focus_log.jsonl').readline()); hit=any(('System now wants to focus on: '+f) in p.read_text(errors='ignore') for p in pathlib.Path(o,'prompts').rglob('*') if p.is_file()); print(r['source'], r['focus']==f, hit)\""` must print `oracle True True` (or `misdirected True True`).
  - **S silent:** `$SSH "docker exec ro27-s1019a-<row> sh -c 'zcat /workspace/tasks/*/*/output/llm_traces.jsonl.gz 2>/dev/null | wc -l; zcat /workspace/tasks/*/*/output/llm_traces.jsonl.gz 2>/dev/null | grep -c ScientistAgent'"` must print a positive count, then 0 (`ScientistAgent` is S's client name, `sci/agent.py:400`). If the first count is 0, the trace is not flushed yet: repeat later.
  - **Guard 1 (prompt diff)**, once `A09-000` logs `OK` in the job log: `$SSH "docker run --rm --cpus 4 -e HOME=/tmp --env-file <CAIR_HOME>/code/PlanBench/.env --entrypoint bash rogue_one:agent2027-v2.1 -c 'cd /tmp && python3 /workspace/scripts/agent2027/analysis/fetch_wandb.py --artifact prompt-baseline-v2.0 --out base && python3 /workspace/scripts/agent2027/analysis/fetch_wandb.py --run A09__full__<task>__f42__r0 --artifact run-output --out smoke && python3 /workspace/scripts/agent2027/prompt_diff.py base smoke'"` reports byte-identical prompts for the SYN-P task (CLI flags as built in E11b and E5; **to verify**).

  If any check fails, stop the session (`docker stop` its `ro27-s1019a-*` containers; the launcher then tears down the server), fix, and relaunch. *Done when:* all three checks pass and are logged in the session plan.
- [ ] **A09-T13** [type: run] (0.15 d, Mon 19 – Wed 21 Oct) — Monitor, retry and tear down.
  - **Monitor** (background): `$SSH_MON "while ! grep -q RO27_SESSION_END ~/ro27_S1019a.job.log 2>/dev/null; do sleep 120; done; tail -30 ~/ro27_S1019a.job.log"`. Exit code 255 means the network dropped; reconnect.
  - **W&B states:** `$SSH "docker run --rm -e HOME=/tmp --env-file <CAIR_HOME>/code/PlanBench/.env --entrypoint python3 rogue_one:agent2027-v2.1 -c \"import wandb; api=wandb.Api(); [print(r.name, r.state) for r in api.runs('PlanBench/RogueOne-AGENT2027', filters={'group': 'A09'})]\""`. A `running` run with no container is a zombie.
  - **Tear-down check:** `$SSH "docker ps --format '{{.Names}}' | grep -E '^(ro27-|rogueone-vllm-)' || echo none; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"`.
  - Infra reruns go into the Tue 20 session (`S1020a`) as extra rows that keep `image_tag=agent2027-v2.1`.

  *Done when:* `=== RO27_SESSION_END S1019a rc=… ===` is logged, the tear-down check prints `none`, and all 15 arm runs are `finished`, or retried once and marked ITT, by **Wed 21 Oct, 12:00** (before FX launches).
- [ ] **A09-T14** [type: run] (0.05 d, conditional; due with the session that carries the row, at the latest Tue 20 Oct in `S1020a`) — Run `A09-016` (AB3) only if G1 funds the add-back or Henrik approves it at G2 from the reserve (§6). Its σ source is pre-registered ‡ (§4.2). Its outcomes are `success` (junglechess rule found, ≥ 0.95) and `heldout_score` against A05 `junglechess` f42 r0, f42 r1 and f4 r0, descriptive only and outside G3. Run `A09-017` only if Henrik asks. *Done when:* each funded row is `finished` or ITT in W&B, or the session plan records "not funded".

### 5.4 Analysis
All analysis runs in CPU containers on CAIR (`--cpus` ≤ 16, image `rogue_one:agent2027-v2.N`, `/meta` mounted read-only, `-e HOME=/tmp`), using `analysis/fetch_wandb.py`, `recovery_detector.py`, `glmm_holm.py`, `funnel.py` and `process_metrics.py`.
- [ ] **A09-T15** [type: analysis] (0.35 d, due **Fri 23 Oct, 18:00**) — Fetch A09 and the A05 SYN-P controls, run the detector, then compute and commit δ.
  - Detector run `ANALYSIS__recovery_detector__20261023`: every fold-episode gets `rec_sel`, `rec_any` and `iter_first_rec`, with ITT rows filled; every A09 arm run shows `llm_calls_scientist` = 0.
  - Compute the oracle gap and δ exactly as in §6 and append them as an amendment to `docs/agent2027/preregistration.md`; commit and push.
  - Post W&B artifact `rt3-oracle-gap` (group `A09`): per gate-passing SYN-P 7 task, `rec_sel` for `oracle`, Full f42 r0 + r1, and `misdirected`; the gap; δ; the commit SHA.
  - Upload the focus files and sidecars as artifact `rt3-focus-files`.

  *Done when:* the δ commit and the artifact exist by Fri 23 Oct, 18:00, before the first A12 detector log (A12-T14).
- [ ] **A09-T16** [type: analysis] (0.5 d, due Sun 25 Oct) — Run the §6 tests. Also produce:
  - the per-task `rec_sel` table by arm;
  - the funnel-loss table (the stage where φ is lost, for each oracle miss);
  - `focus_adherence`, `reproposal_ratio`, `restore_pruned_count`, `fallback_share`, skipped iterations and tokens per role;
  - the P7 oracle cell for A14.

  *Done when:* the results are logged in the G3 memo run `ANALYSIS__glmm_holm__<YYYYMMDD>` (shared with A14-T14, dated the day it is produced, by Sun 25 Oct). The GLMM/exact part of `glmm_holm.py` must exist by Sun 25, ahead of the rest of E11b (Tue 27).
- [ ] **A09-T17** [type: decision] (0.05 d, Mon 26 Oct, G3) — Record the planner-half verdict (route 1, route 2, pass (fragile), fail, or provisional) and the wording rules of §3. *Done when:* the verdict sits in the G3 memo next to KT3's half.

### 5.5 Paper outputs
- [ ] **A09-T18** [type: writing] (0.1 d, Mon 26 Oct) — Annotate the focus channel in Fig. 1 with the `ROGUE_ONE_PLANNER_MODE` knob, and set the planner wording for C2c and C3. *Done when:* both are in the draft.
- [ ] **A09-T19** [type: writing] (0.5 d; statistics Sun 1 – Tue 3 Nov; due Fri 6 Nov) — Produce:
  - the **Table 2 assay row**: Δ`rec_sel` vs Full for the oracle and misdirected arms (with CIs), Δ`success`, Δ`tok_compl_total`, the funnel stage changed, and the verdict;
  - the **Fig. 2** reference points;
  - the **appendix**: σ map, renaming rule, oracle template, per-task and funnel-loss tables, and the AB3 cell if run.

  *Done when:* all three are in the AGENT draft by Fri 6 Nov.

## 6. Decision rules and gates (pre-registered in X10)
- **Population.** The SYN-P 7 tasks that the A08 gate does not flag (P6a and P6b count as two tasks). P7 under the oracle is reported separately, as KT3's ceiling. AB3 never enters the G3 test.
- **Test of record.** Logistic GLMM `rec_sel ~ arm + (1|task) + (1|task:seed)` (seed = `fold_seed`) at the fold-episode level, arm ∈ {Full, oracle, misdirected}, with Full (A05 f42 r0, f42 r1, f4 r0; 15 episodes per task) as the reference. Each route uses the one-sided Wald z test of its arm coefficient. The fitting routine is named in X10.
  - **Exact fallback.** If the GLMM fails to converge or is singular for the arm effect, or any arm has `rec_sel` = 0 on all or = 1 on all of its population episodes (separation), each route instead uses a one-sided exact conditional test of a common odds ratio stratified by task (exact CMH, the same family as A14-T14), comparing only the two arms involved, with episodes as units.
  - **Small population.** With fewer than 3 population tasks, the exact test is the test of record.
- **Planner half of G3 passes** if either route holds. Each route is tested at one-sided α = 0.025 ‡, so the false-pass rate stays ≤ 0.05 (Bonferroni):
  - **route 1:** misdirected < Full, one-sided p < 0.025;
  - **route 2** (pre-registered alternative; it deviates from master plan §4.1 RT3 and §6.3): oracle > Full, one-sided p < 0.025, with route 1 not significant.
  - Misdirected > Full (two-sided, exploratory) never passes G3 on its own; it is reported as Opposite (c).
- **ITT sensitivity.** Repeat the passing route with failed runs excluded. If the pass depends on ITT zeros (p ≥ 0.025 without them), the verdict reads "pass (fragile)".
- **Power.** The MDE is about 0.26 at one-sided α = 0.05 and about 0.29 at 0.025 (80% power, p ≈ 0.5, ≈ 6 informative tasks: 30 arm episodes vs 90 Full episodes, unclustered).
- **FX margin δ** (formula shared with A12-T1; value computed by A09-T15). gap = pooled `rec_sel` of `oracle` minus pooled `rec_sel` of Full f42 r0 + r1, over all episodes of the population tasks (ITT zeros included). δ = max(0.25 × gap, 0.10), rounded to 3 decimals; if gap ≤ 0, δ = 0.10. If any population oracle row is not terminal by Fri 23 Oct, 18:00, δ = 0.10 and A12 is told.
- **Headroom flag ‡ (G2, Mon 19).** Count the gate-passing tasks whose pooled Full `rec_sel` (A05 f42 r0, f42 r1, f4 r0) is ≥ 0.4. If fewer than 3, route 1 can hardly pass: the G2 memo says so, the G3 rule stays unchanged, and Henrik decides at G2 whether to fund AB3 from the reserve (if G1 did not).
- **Dates.** RT3 has no kill date. Runs finish by Wed 21; δ is committed on Fri 23; G3 is Mon 26. If an A09 row is still not terminal on Sun 25, G3 is read on the terminal rows, the missing rows are listed, and the verdict is marked "provisional" until the final statistics (Nov 1–3).

## 7. Risks and cut levers
| Risk | Mitigation |
|---|---|
| Too little headroom: if Full rarely recovers φ, wrong advice cannot lower recovery | Headroom flag; route 2; AB3 (descriptive only) |
| Degenerate margin (oracle ≈ Full) | Absolute floor 0.10 (A12-T1) |
| Invalid misdirection: strings that name missing columns turn the arm into "no plan" | Frozen renaming rule; `verify` asserts |
| Misdirection makes E produce nothing at iteration 0, which aborts the dataset (`main.py:606-663`) | ITT zeros; the ITT sensitivity rule labels such a pass "fragile" |
| Open-loop and no-S-search confound | Word the claim as "a fixed off-task plan" |
| Oracle strings leaking into the repo, the image or another row | A09-T6, A09-T9; focus files go to W&B only after all A09 rows end (A09-T15). **Cross-cutting:** the image also contains the SYN-P generator source, and agent containers hold a W&B key; E13 must block file reads (flagged to E8a/E13) |
| Session collisions: the Sun 18 small-rung night or A07's fallback rows (A07-T14) | Pre-flight stop-and-report; §4.5 timing; `S1020a` slips if A07 rows join |
| `.dev.env` is tracked at `07cbf93`, so the build guard trips | A09-T2 check; E1 removes it from `agent2027/v2` |
| Timing: A05 sources must exist on Mon 19; the GLMM code by Sun 25 | A09-T3, A09-T16 |

**Cut levers:** none, because RT3 is never cut. Misdirected on p7 is already dropped by default. AB3 (+1u) is the only add-back.

## 8. Definition of done
- [ ] X10 holds the σ map with fallbacks, the renaming rule, the oracle template and hashes, the test of record and its fallbacks, the routes, α, the headroom flag, the ITT sensitivity rule and the δ formula.
- [ ] Image v2.1 passed guards 1 and 2, and the first-wave check passed.
- [ ] 15 runs are in W&B group `A09`, each with a `run-output` artifact; ITT is applied.
- [ ] δ was committed and `rt3-oracle-gap` posted by Fri 23 Oct, 18:00, before any A12 detector run.
- [ ] The G3 memo (verdict, GLMM/exact results, per-task and funnel-loss tables) exists by Mon 26 Oct.
- [ ] The Table 2 row, the Fig. 1 and Fig. 2 items and the appendix are drafted by Fri 6 Nov.

## 9. New names introduced (not in the 01 §7 registry)
- **Scripts and test:** `scripts/agent2027/rt3_focus.py`, `scripts/agent2027/rt3_sigma_map.yaml`, `tests/unit/test_planner_hook.py`.
- **Focus files:**
  - CAIR files `~/rogueone-rt3-focus/<row_id>.json`, with never-mounted sidecars under `~/rogueone-rt3-focus/meta/`;
  - container path `/workspace/focus/focus.json`.
- **Code and log fields:**
  - focus log `output/focus_log.jsonl` (name **to verify** against E3) and its field `source`;
  - config attributes `planner_mode` and `planner_focus_file`;
  - metadata key `oracle_focus` in `synp_meta/<id>.json` (E8a confirms the name);
  - oracle feature name `oracle_feature`.
- **Metric** `focus_adherence`: the share of accepted features whose command reads a raw column named in the focus.
- **Manifest rows** `A09-000` (smoke, arm_id `full`), `A09-016` and `A09-017`.
- **Session plan** `docs/agent2027/sessions/S1019a.md` (pre-flight output, HEAD, image IDs, check results).
- **W&B:** runs `ANALYSIS__rt3_focus__20261019` and `ANALYSIS__recovery_detector__20261023`; artifacts `rt3-oracle-gap` and `rt3-focus-files`. The G3 memo `ANALYSIS__glmm_holm__<YYYYMMDD>` follows the R12 pattern and is shared with A14.
