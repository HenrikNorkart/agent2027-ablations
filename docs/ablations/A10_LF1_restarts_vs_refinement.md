# A10 — LF1: Independent restarts vs sequential refinement (C3)

*Rogue One v2 design study for ICSE AGENT 2027 (8 pp IEEE), T1-minimum, submission Fri 13 Nov 2026. Every "hoped" result here is a hypothesis, not a finding.*

*Code refs: RougeOne `nested-cv-rebuttal` @ `07cbf93` (`C:/code/PHD/RougeOne`). The current tip `1bdaf42` leaves `rogue_one_main.py` and `rogueone/` unchanged. Every line cited was re-verified against `07cbf93` on Wed 7 Oct 2026. Short paths (01 §7 R0): `main.py` = `rogue_one_main.py`; `sci/agent.py`, `ext/agent.py`, `tst/agent.py` = `rogueone/llm/agents/core/{scientist,extractor,tester}/agent.py`; `kn/web_search.py` = `rogueone/llm/agents/core/knowledge/web_search.py`; `config.py` = `rogueone/utils/config.py`; `clients.py` = `rogueone/llm/clients.py`. There is no `cfg.py`. Shared detail lives in `docs/ablations/01_shared_infrastructure.md` ("01 §n").*

| Field | Value |
|---|---|
| **ID** | A10 · LF1 · arm `restarts` (`ROGUE_ONE_RESTART_MODE=every_iteration`) |
| **Claim / Holm contrast** | C3: "the loop builds on earlier rounds". **C3 = Restarts − Full** on `rec_sel`, two-sided. It is one of 10 Holm contrasts (smallest threshold 0.005) and the only confirmatory C3 contrast. |
| **Tier** | T1-minimum, never cut (01 §4.9) |
| **Owner** | Henrik |
| **Start / finish** | Build Sat 17 – Mon 19 Oct · smoke (session `S1020s`) and main launch (session `S1020a`) **Tue 20 Oct** on `rogue_one:agent2027-v2.2` · runs done by about Wed 21 Oct, morning · infra reruns by Tue 27 Oct (Thu 29 Oct at the latest, L-LAST) · statistics Sun 1 – Tue 3 Nov |
| **Budget** | **14 runs, 9.8–14u**, identical to the T1-minimum breakdown and 01 §7 R1. That is 7–10 h of box time at 1.4 u/h (2 u/h × 0.7), or 28–40 H100-hours on TP4. **Outside the 14:** the smoke (1 fold-run, ≈ 0.14–0.2u) and ≤ 2 infra reruns (1.4–2u), both paid from the 15% rerun reserve (01 §4.9). The control costs 0 new runs (A05). **Developer time:** ≈ 3.85 d over T1–T18 (some shared with A11). Only E9a's 0.25 d is in the 01 §6.2 engineering budget; the +0.1 d for the seed rule is not. |
| **Models** | O120 (`openai/gpt-oss-120b`) for S, E, T and W; port `8010` (`8011` for the second server if D5 = TP2); vLLM `vllm/vllm-openai:v0.22.0`; `MAX_MODEL_LEN=131072` |
| **Status** | Not started |
| **Depends on** | E1, E2, E3, E4, E5, E6, E8a, **E9a**, E11a, E11b, X10 · A05 (control), A06 (sentinel), A08 CPU screen (brute-force gate), A09 (G3), A11 (shared session `S1020a`; the shared v2.2 Full-v2 guard is A11-T8) · D2, D3, D4, D5, D7 · G1, G2 A03 (pool mode and re-seeding rule on reset); |
| **Feeds into** | C3 row of Table 2 · the "Restarts" point in Fig. 2 · the Restarts → L0 → Full ladder with A11 · A21 (iso-cost) · A23 (Pareto) · A17 (A10-T18 active-feature log check) |

## 1. What we test and why it matters

Inside each outer fold, Rogue One runs 10 iterations that carry state forward: the feature pool, the attribute table, the Tester's feedback, the focus history and the Scientist's notes. A10 wipes that state before **every** iteration. Each fold then gets 10 independent first-round attempts, and the same inner-CV selection picks the best one.

- If Full recovers planted features more often, the loop is more than best-of-10 sampling plus selection.
- If it does not, the iterations could run in parallel.

This is the paper's only confirmatory evidence about the loop.

## 2. Background

**Loop and carried state**
- `main.py:1440-1442` calls `forward_pass(i)` for i = 0…49, where `fold_run_id = i // 10` and `iteration_in_fold = i % 10` (`:539-540`).
- One iteration runs five steps:
  1. Scientist focus (`:548-556`);
  2. Extractor, R rounds in one session (`:563-571`, `ext/agent.py:965-971`);
  3. fold-safe recomputation of every feature (`main.py:752-758`);
  4. Tester: inner-CV score, report and pruning (`:789-794`);
  5. the trial row is appended to `df_test_pool` (`:813-819`), and a feedback row (`train_*` metrics plus the report, built at `:384-415`) to `df_scientist_feedback_pool` (`:823-831`).
- **Carried within a fold, and wiped by `_reset_fold_state` (`main.py:417-457`):**
  - the pool `df_entities_attributes`;
  - `df_attributes_explanations` (command, `Status`, `Pruned Iteration`, `Prune Rationale`);
  - the feedback pool;
  - `focus_history`;
  - the Scientist notebook (written at `sci/agent.py:358`, read at `:387`);
  - the rollback snapshot (inert, because rollback is off).

  Today the reset runs only when `iteration_in_fold == 0` (`main.py:543-544`).
- **Not carried:** conversation history. Every call opens a fresh, in-memory `SQLiteSession` (`sci/agent.py:499`, `ext/agent.py:965`, `tst/agent.py:1371`, `kn/web_search.py:253`). No agent object holds other cross-iteration state.
- **Never reset:** `df_test_pool`. Selection (`main.py:1074-1205`) picks, per fold, the `completed` trial with the best `inner_cv_accuracy` or `inner_cv_RMSE` (`:1109-1154`). Under restarts it therefore picks the best of 10 independent trials.

**What a restart sees (verified)**
- **Scientist prompt.** With an empty `focus_history` the first user message is the fixed "Round 1 … first round" text (`sci/agent.py:472-475`). In `replace` mode, `list_all_attributes` answers "first round" (`:285-286`); in `augment` mode it lists E6's seed rows, exactly as at Full's iteration 0.
- **Round-dependent rules** apply at every restart: "Prefer first-round baseline on raw attributes" (`:437`) and "exploratory … early round" (`:440`). The heuristic fallback uses its first-round branch (`:502-506`).
- **Round index.** It is in neither the system prompt (`:409-456`) nor the Extractor message (`ext/agent.py:971`). The global index does appear in two places, both wiped by every reset, so a restart only ever sees its own trial number:
  - the note header `Trail {index}` (`sci/agent.py:358`);
  - the `Added = "Trial {step}"` column (`ext/agent.py:475`, `:546`).

  E2 changes the non-empty-history prompt to "Round {i+1}/{N}" (01 §3.3, E2 item 2). Restarts never reach that branch.
- **Starting pool.** The id column in `replace` mode. In `augment` mode, the raw and ordinal seed rows, which E6 re-seeds inside the reset.
- **Empty restarts.** At `iteration_in_fold > 0` they take `iteration_skipped` (`main.py:639-663`). At 0 the dataset aborts (`:665-717`), as it does in Full.
- **Seed hazard (found in verification).**
  - `ROGUE_ONE_LLM_SEED` (`config.py:330-331`) is attached to every request (`clients.py:211-213`, called at `:529` and `:532`; every request path goes through this function).
  - Restart prompts are byte-identical, so with one seed the restarts would be near-copies, and the shared Serper cache would return identical results.
  - A10-T5 fixes this with the restart seed rule.
  - **The same issue affects Full at iteration 0.** In every fold of a run, Full's iteration 0 has the same prompt and seed, and the Extractor sees the same full `df` (`main.py:564-570`). Full's five iteration-0 draws per run may therefore be near-copies. Byte-reproducibility breaks only under `PARALLEL=6` load (Finding 36). This lowers the effective sample size of `q0_iter0` (H3).
- **Logging pitfall.** `df_attributes_explanations_final.csv` (`main.py:1390-1398`) and `fold_aggregation_active_features.csv` (`:1177-1187`, `:1359`) hold only the **last** iteration's table.
  - Per-iteration feature values exist (`intermediate_steps/df_entities_attributes_trail_<index>.csv`, `:839-847`). Per-iteration commands and `Status` do not.
  - `rec_sel` must therefore join E3's per-iteration active-feature log to `fold_aggregation_best_iterations.csv` (`:1353`).
  - The A05 control needs the same log, so it must be in **v2.0** (A10-T18).

**Instruments: SYN-P 7** (E8a; n = 2,000, p ≈ 20–30; folders `tasks/<task_type>/synp_<id>/`; task types proposed, confirmed by E8a in `synp_meta/registry.yaml`)

| dataset_id | φ | Type | A10 role |
|---|---|---|---|
| `synp_p1` | x_a/x_b (D1) | R | single-formula; expected to be non-discriminating (sanity check) |
| `synp_p2` | (x_a/x_b)·x_c (D2) | R | placed by **D3** |
| `synp_p3` | 1[x_a/x_b ≥ t] ∧ cat = k | C | **multi-step** |
| `synp_p4` | sin/cos of an hour-like column | C | single-formula |
| `synp_p5` | sign(LW·LD² − RW·RD²) | C | **multi-step** |
| `synp_p6a`, `synp_p6b` | named composite and its anonymised twin | R | single-formula |

- No `synp_p7` (P7 policy) and no `synp_p0`.
- **Brute-force gate (A08):** a family is non-discriminating if OpenFE or AutoFeat recovers φ in ≥ 14 of 15 fold-episodes (seeds {42, 4, 5}). Such a family stays in the runs but is left out of every confirmatory population. The labels are final on Thu 15 Oct, before D3.

**Metrics** (unit = fold-episode; definitions in 01 §4.6)
- **`rec_sel`** (primary): an active non-fallback feature **in the inner-CV-selected iteration** has |Spearman| ≥ 0.95 with φ on training rows, or NMI ≥ 0.9 for P3 and P5. Raw columns and single-column monotone transforms never count.
- **`rec_any`:** the same test at any iteration. Also `iter_first_rec` and `tok_to_first_rec`.
- **Run outcomes and cost:** `success`, `collapse`, `cap_hit`; `tok_compl_*` (search charged to the caller), `wall_s`, `energy_j`.
- **Process:**
  - `lineage_share`: the share of φ-recoveries whose command or focus builds on an earlier engineered feature or focus;
  - `focus_novelty`, `prune_rate`, `fun_*`;
  - `reproposal_ratio`, in an A10 variant: the reference set is every feature proposed earlier in the same fold (from the per-iteration log), applied identically to both arms. Against the current pool it would be 0 for restarts by construction.

## 3. Hypotheses and interpretation (all hypotheses)

- **H1 (C3):** Full ≥ Restarts on `rec_sel`, with the gap concentrated on P3 and P5 (and P2 if D3 places it there).
- **H2:** on single-formula tasks, a small or zero gap on `rec_any`. Full keeps a built-in `rec_sel` edge even without learning: once found, φ persists in later trials, while a restart's φ must be the one selected.
- **H3 (validity, descriptive, outside Holm):** restarts are independent iteration-0 draws.
  - `q0_iter0` = the share of A05 fold-episodes (r0 + A/A) whose iteration-0 trial holds φ as an active non-fallback feature, per task.
  - Prediction per episode: `rec_any_pred` = 1 − (1 − `q0_iter0`)^10. Pred = Σ `rec_any_pred`; SE = √Σ p(1 − p) over the restart episodes of the C3 population.
  - **Read-out:** "independent draws" if the observed Σ restart `rec_any` ≥ Pred − 1.96·SE; otherwise "correlated draws".
  - This is coarse: `q0_iter0` rests on ≤ 15 episodes per task, which may be correlated (§2, seed hazard). Report `q0_iter0` with its binomial CI.

**Hoped result.** *"Under the same 10-iteration budget and inner-CV selection, sequential refinement recovered the depth-3 and rule features in X% of fold-episodes versus Y% for ten independent restarts."* Write "learns from feedback" only if all three hold:
- C3 passes Holm with Full > Restarts;
- A11's rule R1 holds (L0 − Full < 0, 95% CI excludes 0);
- G3 passes.

**Null (CI includes 0).** The loop behaves like best-of-10 plus selection. Report the CI and the MDE, and say "inconclusive" if the CI is wide. First check A05: if Full's P3/P5 recovery is about 0, no gap is possible there.

**Opposite (Restarts > Full).** Carried state hurts: anchoring, pool pollution, or φ pruned before the selected iteration. Diagnose with φ's `Pruned Iteration`, pool size per iteration and `focus_novelty`. Report it as a negative C3 result; Full-v2 stays frozen.

**Power** (unclustered, 80%; recomputed at G2 from A05's ICC):
- 20 multi-step episodes per arm: MDE ≈ 40–45 pp at α = 0.05, ≈ 55 pp at 0.005.
- With P2 (30 episodes): ≈ 33–35 pp and ≈ 45 pp.
- A pooled test dilutes a multi-step-only effect about 3×.

## 4. Design

**Arms**
- **Control:** A05 `full`, `replicate` 0, SYN-P 7 × {42, 4}: 14 runs, already paid.
- **Treatment:** Full-v2 + `{"ROGUE_ONE_RESTART_MODE":"every_iteration"}`, with the restart seed rule (A10-T5).
- **Disclosed:** the arm removes three things at once:
  1. carried information (focus history, feedback, notes);
  2. the accumulated pool (building blocks, and φ's persistence);
  3. the round regime (every draw gets the first-round rules).

**Fixed** (from `scripts/agent2027/v2_defaults.env`; every other R3/R4 flag stays at its Full-v2 value; the only override is `ROGUE_ONE_RESTART_MODE`):
- `ROGUE_ONE_NUM_ITERATIONS=10` (the code default is 3, `config.py:305`), `ROGUE_ONE_K_FOLDS=5`;
- `ROGUE_ONE_SELECTION_SIGNAL=inner_cv`, `ROGUE_ONE_ENABLE_ROLLBACK=False`, `ROGUE_ONE_DO_BALANCING=True`;
- `ROGUE_ONE_EXTRACTOR_ROUNDS` and `ROGUE_ONE_FEATURE_POOL_MODE` as frozen at G1;
- all temperatures 0.7;
- `ROGUE_ONE_TOOL_ERROR_FEEDBACK=message`, `ROGUE_ONE_SEARCH_CACHE=shared`, `ROGUE_ONE_LLM_TRACE=1`, `ROGUE_ONE_PROMPT_DUMP=1`, `ROGUE_ONE_REPLAY_VALIDITY_CHECK=1`;
- `PARALLEL=6`, `CPUS=12`, `RUN_CAP_HOURS=6`, `MAX_MODEL_LEN=131072`;
- `ROGUE_ONE_STRATIFIED_FOLDS` copied from the paired A05 row's effective value (`True` on p3, p4 and p5 if D4).

**Run matrix** (entity `PlanBench`, project `RogueOne-AGENT2027`, group `A10`, job_type `restarts`, tags [`S1020a`, `v2`, `agent2027-v2.2`]; manifest `manifests/agent2027/A10_LF1.csv`)

| Rows | arm_id | dataset_id | fold_seed / llm_seed / replicate | Runs | u | Run name |
|---|---|---|---|---|---|---|
| A10-001…007 (p1, p2, p3, p4, p5, p6a, p6b) | `restarts` | SYN-P 7 | 42 / 42 / 0 | 7 | 4.9–7 | `A10__restarts__<dataset_id>__f42__r0` |
| A10-008…014 (same order) | `restarts` | SYN-P 7 | 4 / 4 / 0 | 7 | 4.9–7 | `A10__restarts__<dataset_id>__f4__r0` |
| A10-000 (smoke, not analysed) | `restarts` + `ROGUE_ONE_SMOKE_FOLDS=1` | SYN-P task of `prompt-baseline-v2.0` (proposed `synp_p3`) | 42 / 42 / 0 | 1 fold-run | ≈ 0.14–0.2 (reserve) | same pattern; `code_phase=smoke`, session `S1020s`, `notes=smoke` |
| A05 control (reused) | `full` | SYN-P 7 | 42 and 4 / = fold_seed / 0 | 0 new (14) | 0 | `A05__full__<dataset_id>__f<seed>__r0` |
| A05 A/A (sensitivity only) | `full` | SYN-P 7 | 42 / 1042 / 1 | 0 new (7) | 0 | `A05__full__<dataset_id>__f42__r1` |

- **Row order** follows A11: seed-42 block first, then the seed-4 block, each in the order p1…p6b.
- **Budget arithmetic:** 7 tasks × 2 seeds × 1 arm = 14 runs × 0.7–1.0u = **9.8–14u**. That gives 70 fold-episodes per arm, about 60 after the gate removes P1.
- **Session arithmetic:** `S1020a` holds A10 14 + A11 14 + 1 A06 sentinel = 29 runs, 20.3–29u, or 14.5–20.7 h at 1.4 u/h. A launch by 14:00 Tue 20 Oct ends about 04:30–10:45 Wed 21 Oct.
- **Smoke run name.** The smoke's W&B name equals that of the `synp_p3` f42 row by construction. `fetch_wandb.py` keeps only `code_phase=v2` and drops `manifest_row=A10-000`.
- **Pairing:** each row pairs with the A05 row of the same `dataset_id` and `fold_seed`, so both use the same outer partition (`notes=pair with A05-<nnn>`).
  - The pairs are the SYN-P 8 rows without p7: 7 of A05-031…038 (seed 42) and 7 of A05-039…046 (seed 4).
  - If A05 orders SYN-P 8 as p1…p7, these are A05-031…037 and A05-039…045, with A/A A05-047…053. Read the numbers from the committed A05 manifest.
- **Randomisation and its limits.** A10 rows are interleaved with A11's in a random `order_key` (RNG seeded from the session ID), with one A06 `sentinel` row in the first wave.
  - **The C3 contrast is not concurrently randomised.** The control ran in `S1016a` on `agent2027-v2.0` (Fri 16 – Sun 18 Oct); the treatment runs in `S1020a` on `agent2027-v2.2`.
  - Comparability rests on the byte-identical Full-v2 guard (01 §2.2), the unit tests of the switch-off path and the A06 sentinel. A session covariate is aliased with arm and cannot be fitted.
- **Failures** (01 §4.5):
  - A crash (rc ≠ 0, or no `fold_aggregation_summary.csv`) is retried once, alone, at the end of the session.
  - Persistent failures and 6 h cap hits are ITT failures in both arms: every fold-episode scores `rec_sel = rec_any = 0`. They are never dropped, and their rates are reported.
  - Infrastructure faults (vLLM death, or a W&B or Serper outage) are re-queued with `notes=infra-rerun` and are not ITT failures.

## 5. Tasks

**Execution order:** T1–T3, T18 (by Fri 16) → T4, T5, T6, T8 (Mon 19) → T7 (smoke, Tue 20 morning) → T9 (main launch) → T10, T11 → T12–T17.

### 5.1 Prerequisites
- [ ] **A10-T1** [type: decision] (0.25 d, due Thu 15 Oct) — Decide **D3**: either (a) pooled C3 over SYN-P 7 minus the gate, or (b) the multi-step subset P3 + P5, with or without P2. Fix P2's moderator level. Take the A08 gate labels (final Thu 15) into account. If the gate removes every task of the chosen subset, (a) applies. *Done when:* the choice and the fallback are in the draft pre-registration.
- [ ] **A10-T2** [type: decision] (0.1 d, due Thu 15 Oct) — Approve the **restart seed rule** (A10-T5): `seed = llm_seed + index`, where `index` is the global trial index.
  - The alternative considered is `llm_seed + iteration_in_fold`. It mirrors Full's cross-fold seed sharing, but gives correlated restarts across folds.
  - Pre-declare the `replace`-mode rule: if M1 keeps `replace`, report `rec_any` beside `rec_sel` with equal prominence, fitted with the same model but not as an extra Holm test.

  *Done when:* both are written in the draft pre-registration.
- [ ] **A10-T3** [type: writing] (0.25 d, due Fri 16 Oct, 12:00) — Commit the following in `docs/agent2027/preregistration.md`:
  - the C3 population and fallback (T1) and the test of record, including the separation trigger and the sensitivities (§6);
  - the secondaries (`rec_any`, arm × multi-step, `lineage_share`, iso-cost);
  - the H3 formula and read-out rule (§3);
  - the seed rule and the `replace`-mode rule (T2);
  - the G3 reading rule and the L-LAST drop rule for C3 (§6).

  *Done when:* they are inside tag `agent2027-prereg`.
- [ ] **A10-T18** [type: decision] (0.1 d, due Fri 16 Oct, 16:00, before A05 launches at 18:00) — Check on the v2.0 smoke run (01 §2.2) that E3's per-iteration active-feature log has, for every (fold, iteration): the name, `Pandas Command`, `Status` and a fallback flag. A22-T1 additionally needs E6's `Seeded` marker and a `Scored` column in the same snapshot, and A18-T2 needs the E3 trace fields (01 E3); check them in the same pass. The A05 control needs this log for `rec_sel`.
  - If it is missing, it must go into v2.0 before 18:00. It is a log-only E3 change and touches no prompt.
  - If that is impossible, decide at G2 between two options:
    - rerun A05's SYN-P rows (the G2 rerun rule, 01 §2.2);
    - score `rec_sel` from the `df_entities_attributes_trail_<index>.csv` values with every column counted as active, as a disclosed upper bound in both arms.

  *Done when:* the log is found in the smoke's `run-output` artifact, or the G2 option is written in the pre-registration.
- [ ] **A10-T4** [type: decision] (0.25 d, due Mon 19 Oct, G2) — Check before launch:
  1. E1–E6 and E8a are in `agent2027-v2.0`.
  2. The A08 gate labels are read.
  3. The 14 A05 control runs and the 7 SYN-P 7 A/A runs are `finished`, with `run-output` artifacts and non-zero `tok_compl_*` for every role.
  4. A10-T18 passed: the per-iteration active-feature log exists in the A05 SYN-P `run-output` artifacts.
  5. The G2 memo has the ICC, the informative tasks and the sentinel task.
  6. The image contains no `.dev.env`. That file is **tracked in git** at `07cbf93`, and `config.py:24` loads it with `override=True`. It sets:
     - `LLM_PORT=8000`, which would point clients away from port 8010;
     - `LLM_API_KEY=rogueone-dev`;
     - `ROGUE_ONE_K_FOLDS=5`.

     `.dockerignore` does not exclude it, so E1 must untrack it or add it to `.dockerignore`.

  *Done when:* all six are ticked, or the launch is held and the failing item is logged in the session plan.

### 5.2 Implementation (E9a; default-off; Full-v2 stays byte-identical)
- [ ] **A10-T5** [type: code] (0.35 d = E9a's 0.25 d + 0.1 d for the seed rule; due Mon 19 Oct) — Make four changes:
  1. Read `ROGUE_ONE_RESTART_MODE=off|every_iteration` in `config.py` next to `enable_rollback` (`:313`). Any other value raises. Add the flag to E1's flag registry, and confirm that `v2_defaults.env` has `ROGUE_ONE_RESTART_MODE=off`.
  2. Change `main.py:543` to `if iteration_in_fold == 0 or <restart_mode> == "every_iteration":`.
  3. Confirm that E6's augment seeding lives inside `_reset_fold_state`.
  4. **Seed rule**, under `every_iteration` only:
     - add a module-level `contextvars.ContextVar` seed offset (default 0) in `clients.py`;
     - set it to `index` in `forward_pass` after `:540`;
     - make `clients.py:211-213` send `seed = llm_seed + offset`.

     All 50 draws then get distinct seeds. Only trial 0 (fold 0, iteration 0) shares Full's seed; fold f's restart k uses `llm_seed + 10f + k`. `off` leaves the seed at `llm_seed`.

  If A10-T18 had to use the G2 fallback, also write `intermediate_steps/df_attributes_explanations_trail_<index>.csv` in `dump_data` (`main.py:839-847`) for the A10 rows. *Done when:* the A10-T6 tests pass.
- [ ] **A10-T6** [type: code] (0 d extra, inside T5's 0.35 d; due Mon 19 Oct) — Write `tests/unit/test_restart_mode.py`, with `Runner.run` stubbed and modelled on `tests/unit/test_fold_lifecycle.py`. It checks:
  1. `df_test_pool` gains one row per iteration and survives every reset;
  2. the attribute table, feedback pool, `focus_history` and notebook are empty at each iteration start;
  3. augment rows are re-seeded;
  4. the iteration-5 Scientist instructions and first user message equal iteration 0's byte for byte;
  5. requests carry `llm_seed + index` under `every_iteration`, and `llm_seed` under `off` and when unset;
  6. under `off`, `_reset_fold_state` runs only at `iteration_in_fold == 0`, so the switch-off path is unchanged (01 §2.2, guard 2);
  7. an invalid value raises;
  8. no `SQLiteSession` passes a `db_path`.

  *Done when:* `python -m pytest tests/unit -q` passes, including the existing `test_fold_lifecycle.py`. Run it locally, or on CAIR inside the image only: `docker run --rm --entrypoint python3 rogue_one:agent2027-v2.2 -m pytest tests/unit -q`.

### 5.3 Execution on CAIR

**Hard rules** (01 §5.1):
- **Containers only.** Every execution runs inside Docker. The CAIR host only issues `docker` / `docker compose` commands and the `~` launcher scripts, plus `nvidia-smi`, `docker ps` and `tail` checks. Never run `python` on the host. Any `docker compose run` over SSH needs `-T`.
- **GPU pre-flight before every GPU container**, including the smoke. If not enough GPUs are free, **stop and report** the output and let Henrik decide.
- **≤ 4 of the 8 H100s**, ever, without Henrik's explicit approval in chat. One Rogue One job at a time.
- **W&B is the system of record.** Runs are stateless and upload `run-output`.
  - `WANDB_API_KEY` is passed at run time from `<CAIR_HOME>/code/PlanBench/.env`.
  - `SERPER_API_KEY` comes from `.env` / `.secret.env` in `<ROGUEONE_CAIR_PATH>`, baked in at build. Never push these images.
- **Connection.** One-shot `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "…"`. Code reaches CAIR by commit + push of `agent2027/v2`, then `git fetch` on CAIR.
- **Long jobs.** Detached `nohup`, with the log in `~`, never in the root-owned run folders.

`<ROGUEONE_CAIR_PATH>` = `<CAIR_HOME>/code/RougeOne` is **inferred, not verified**. It comes from `paper/scripts/clients_t16_rerun.sh:135` (`REPO_TASKS=$HOME/code/RougeOne/tasks`) and `:59` (home `<CAIR_HOME>`). Verify it once with `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "ls -d ~/code/RougeOne"`.

- [ ] **A10-T8** [type: config] (0.1 d, due Mon 19 Oct) — Generate the manifest and the session plans, then commit and push to `agent2027/v2`. Do this jointly with A11-T9, as one command for both docs.
  - `make_manifest.py --ablation A10` writes `manifests/agent2027/A10_LF1.csv`, rows A10-000…014, with:
    - the R7 columns;
    - `task_type` per E8a;
    - `llm_seed = fold_seed`, `replicate=0`, `is_sentinel=0`;
    - `image_tag=agent2027-v2.2`, `rung_S/E/T=O120`, `port_S/E/T=8010`;
    - `env_overrides={"ROGUE_ONE_RESTART_MODE":"every_iteration"}`, with `ROGUE_ONE_STRATIFIED_FOLDS` copied from the paired A05 row, and `"ROGUE_ONE_SMOKE_FOLDS":"1"` added on A10-000;
    - `cap_hours=6`, `expected_u=0.85`;
    - `notes=pair with A05-<nnn>` (A10-000: `smoke`).
  - `make_manifest.py --plan-session S1020s --rows A10-000 A11-000`.
  - `make_manifest.py --plan-session S1020a --rows A10-001…014 A11-001…014 <next unused A06 sentinel row>`, with the sentinel in the first wave (A06 §4.5).
  - **If D5 = TP2,** split the S1020a rows alternately into `S1020a` (port 8010) and `S1020b` (`port_S/E/T=8011`). Each gets its own A06 sentinel.

  *Done when:*
  - the files are committed and pushed;
  - the dry run gives unique container names `ro27-s1020a-a10-0nn`;
  - `task_type` and the fold scheme match the paired A05 rows.
- [ ] **A10-T7** [type: run] (0.25 d, due Tue 20 Oct, 11:00) — Bring up v2.2 and run the switch smoke in session `S1020s`, **once**, shared with A11-T10 and A11-T11. If A11 already did steps 1–4, verify the HEAD and the image ID instead.
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15' git fetch git@github.com:HenrikNorkart/RougeOne.git agent2027/v2 && git checkout -B agent2027/v2 FETCH_HEAD && git log -1 --format='HEAD: %h %s'"
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader && echo '--- compute apps ---' && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader && echo '--- containers ---' && docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'"
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && (test ! -e .dev.env || grep -qx '.dev.env' .dockerignore) && docker build -t rogue_one:agent2027-v2.2 . && docker image inspect rogue_one:agent2027-v2.2 --format '{{.Id}}' && docker run --rm --entrypoint sh rogue_one:agent2027-v2.2 -c 'test ! -e /workspace/.dev.env && echo no-dev-env'"
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && cp scripts/agent2027/ro27_session_launch.sh scripts/agent2027/ro27_clients_manifest.sh scripts/agent2027/ladder_session.sh ~/ && cp paper/scripts/vllm_session.sh ~/ro27_vllm_session.sh && mkdir -p ~/ro27_manifests && cp -r manifests/agent2027/. ~/ro27_manifests/"
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "SESSION_ID=S1020s SESSION_FILE=~/ro27_manifests/sessions/S1020s.csv GPUS=<ids> MODEL=openai/gpt-oss-120b PORT=8010 SERVER_NAME=rogueone-vllm-o120-s1020s LOG=~/vllm_session_S1020s.log IMAGE_RO=rogue_one:agent2027-v2.2 PARALLEL=6 CPUS=12 RUN_CAP_HOURS=6 RUNROOT=~/rogueone-runs-agent2027 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_S1020s.job.log 2>&1 & echo launched; sleep 5; tail -5 ~/ro27_S1020s.job.log"
  ```
  - **Proceed past the pre-flight only if:**
    - 4 GPUs (2 if TP2) each have < 2,000 MiB used;
    - Rogue One's total stays ≤ 4;
    - no `ro27-*`, `rogueone-vllm-*`, `llmfe-*` or `caafe-*` container of ours is running.

    Otherwise **stop and report** the output.
  - Monitor until `RO27_SESSION_END` appears (as in T10), then run the T10 tear-down check.

  *Done when:*
  - the image ID is recorded in the session plan, and the in-image `.dev.env` check printed `no-dev-env`;
  - A11-T8's shared Full-v2 guard (`scripts/agent2027/prompt_diff.py` against the `prompt-baseline-v2.0` artifact) is byte-identical, and its log is attached to the v2.2 merge;
  - A10-000 has 10 rows in `df_test_pool_final.csv` (`completed` or `iteration_skipped`);
  - the first user message of all 10 Scientist dumps is the "Round 1: …" text;
  - the E3 focus log has 10 pairwise non-identical focus strings;
  - the traces show seeds 42…51 for trials 0–9;
  - W&B shows `flags/ROGUE_ONE_RESTART_MODE=every_iteration` and `code_phase=smoke`.
- [ ] **A10-T9** [type: run] (0.25 d, due Tue 20 Oct, by 14:00) — After T7 passes, launch the main session `S1020a`, **once** for A10 and A11 (this is the same step as A11-T12).
  1. Run a **fresh pre-flight** with the T7 command and the same proceed rule; otherwise stop and report.
  2. Launch:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "SESSION_ID=S1020a SESSION_FILE=~/ro27_manifests/sessions/S1020a.csv GPUS=<ids> MODEL=openai/gpt-oss-120b PORT=8010 SERVER_NAME=rogueone-vllm-o120-s1020a LOG=~/vllm_session_S1020a.log IMAGE_RO=rogue_one:agent2027-v2.2 PARALLEL=6 CPUS=12 RUN_CAP_HOURS=6 RUNROOT=~/rogueone-runs-agent2027 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_S1020a.job.log 2>&1 & echo launched; sleep 5; tail -5 ~/ro27_S1020a.job.log"
  ```
  - **If D5 = TP2:** run two such jobs on 2 GPUs each, with explicit `GPUS`.
    - The second job uses `SESSION_ID=S1020b`, `PORT=8011`, `SERVER_NAME=rogueone-vllm-o120-s1020b`, `LOG=~/vllm_session_S1020b.log` and `ROGUEONE_ALLOW_SECOND_SERVER=1`. Its log goes to `~/ro27_S1020b.job.log`.
    - Start it only after a pre-flight shows that the first server's GPUs are the only ones Rogue One holds. ΣGPU ≤ 4.
  - **Priority:** S1020a runs before A08's GPU comparators and before FX (Wed 21, evening). One job at a time. A08's comparators use their own launcher (A08-T16), so they run in their own session after S1020a ends.

  *Done when:* the job log shows the vLLM version assert, both image IDs and `ro27-s1020a-*` containers, and the W&B runs are `running`.
- [ ] **A10-T10** [type: run] (0.1 d, Tue 20 – due Wed 21 Oct) — Monitor in the background (per job log under TP2):
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 CAIR-gpu14 "while ! grep -q 'RO27_SESSION_END' ~/ro27_S1020a.job.log 2>/dev/null; do sleep 120; done; tail -30 ~/ro27_S1020a.job.log"
  ```
  - Exit code 255 means the network dropped, not the job: reconnect.
  - Check W&B states in a container (01 §5.2 step 7): `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker run --rm --env-file <CAIR_HOME>/code/PlanBench/.env --entrypoint python3 rogue_one:agent2027-v2.2 -c \"import wandb; api=wandb.Api(); [print(r.name, r.state) for r in api.runs('PlanBench/RogueOne-AGENT2027', filters={'group': 'A10'})]\""`. A `running` run with no live container is a zombie: flag it.
  - Tear-down check: `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker ps --format '{{.Names}}' | grep -E '^(ro27-|rogueone-vllm-)' || echo none; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"`.

  *Done when:*
  - the job log has 14 A10 lines, each `OK`, `FAILED` or `CAP_HIT`;
  - the tear-down check prints `none`, and our GPUs are back below 2,000 MiB.
- [ ] **A10-T11** [type: run] (0.1 d, due Tue 27 Oct; Thu 29 Oct at the latest, L-LAST) — Re-queue infrastructure-fault rows (`notes=infra-rerun`) in the next O120 session, which has its own sentinel. Use the T9 pre-flight and launch pattern with that session's ID. Planning assumes ≤ 2 reruns (1.4–2u); more are paid from the 01 §4.9 reserve. *Done when:* every A10 row has a final W&B state (`finished`, or an ITT failure).

### 5.4 Analysis (CPU, in a container, logged as `ANALYSIS__<script>__<YYYYMMDD>`)

**Container pattern** on CAIR (the CLI arguments are set by E11b). The scripts are mounted read-only from the synced repo, because E11b lands after v2.4.
```
ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "mkdir -p ~/ro27_analysis/A10 && docker run --rm --cpus 16 --env-file <CAIR_HOME>/code/PlanBench/.env -v <ROGUEONE_CAIR_PATH>/scripts/agent2027:/workspace/scripts/agent2027:ro -v ~/rogueone-synp-meta:/meta:ro -v ~/ro27_analysis/A10:/out --entrypoint python3 rogue_one:agent2027-v2.4 /workspace/scripts/agent2027/analysis/<script>.py <args>"
```

- [ ] **A10-T12** [type: analysis] (0.25 d, preliminary due Wed 28 Oct; final Sun 1 Nov, after DF) — Run three scripts:
  - `analysis/fetch_wandb.py` for groups A10 and A05 (it keeps `code_phase=v2` and drops A10-000);
  - `analysis/recovery_detector.py` on the per-iteration active-feature log, joined to the selected trials;
  - `analysis/sentinel_drift.py --session S1020a` (and `S1020b` under TP2).

  Then compute H3 (`q0_iter0`, `rec_any_pred`, and the read-out of §3). *Done when:*
  - there is a table of 140 fold-episodes (70 A10 + 70 A05 r0), plus 35 A/A episodes;
  - the H3 verdict is logged;
  - A06's verdict for S1020a is recorded. On FLAG, apply A06 §6 and carry `drift_flag` into T13–T15.
- [ ] **A10-T13** [type: analysis] (0.25 d, due Tue 3 Nov) — Run the C3 test of record (§6) and its sensitivities with `analysis/glmm_holm.py`. *Done when:* the following are logged in an `ANALYSIS` run:
  - the p of record (raw);
  - the odds ratio with its 95% CI;
  - Δ`rec_sel` with its Newcombe 95% CI;
  - which model was used (GLMM or the exact fallback);
  - the Holm decision over the surviving family.
- [ ] **A10-T14** [type: analysis] (0.5 d, due Tue 3 Nov) — Run the secondaries:
  - `rec_any` and arm × multi-step;
  - `lineage_share`, `focus_novelty`, `prune_rate`, `fun_*` (`analysis/process_metrics.py`, `analysis/funnel.py`);
  - the A10 variant of `reproposal_ratio` (§2 Metrics), in **both** arms;
  - `tok_compl_*`, `wall_s`, `energy_j`;
  - the iso-cost view: Full truncated at whole iterations to the Restarts' tokens (A21, `paper/scripts/diag_selection_policy.py` extended; 01 §4.8).

  *Done when:* there is one CSV per item, attached to the `ANALYSIS` run.

### 5.5 Paper outputs
- [ ] **A10-T15** [type: writing] (0.25 d, due Tue 3 Nov) — Write the Table 2 row "C3 Restarts − Full": Δ`rec_sel` (CI), OR (CI), raw p, Holm verdict, Δ`success`, Δ`tok_compl_total`, and the drift caveat if any. *Done when:* the row is in the Table 2 source.
- [ ] **A10-T16** [type: writing] (0.25 d, due Fri 6 Nov) — Write three things:
  - the Restarts point in Fig. 2 (right panel, next to Full's anytime curve);
  - the §3 sentence matching the verdict (§6 wording rules);
  - a threats paragraph covering the three mechanisms, the round regime, the seed rule, the non-concurrent control and the Serper cache.

  *Done when:* all three are in the AGENT draft.
- [ ] **A10-T17** [type: writing] (0.25 d, due Thu 12 Nov) — Build the appendix: per-task `rec_sel` and `rec_any` per arm, H3, `lineage_share`, the rendered restart prompts and the E9a diff. *Done when:* all are in the anonymised replication package.

## 6. Decision rules and gates (pre-registered)

**Population** (D3, A10-T1): (a) pooled = SYN-P 7 minus non-discriminating families; or (b) the multi-step subset {P3, P5} (+ P2 if D3 places it there), minus non-discriminating families. If the gate removes every task of (b), (a) applies. Units: fold-episodes of A10 `restarts` and A05 `full` r0. ITT zeros are included.

**Test of record**
- **Model.**
  - (a): logistic GLMM `rec_sel ~ arm + (1|task) + (1|task:seed)`;
  - (b): `rec_sel ~ arm + task + (1|task:seed)`, with task fixed (2–3 levels).
- **p of record:** the likelihood-ratio test of `arm` (1 df, two-sided). Effect: the odds ratio Restarts/Full with its 95% profile CI. Δ`rec_sel` (Restarts − Full) with a Newcombe 95% CI is reported beside it, descriptive only.
- **Separation fallback.** It is triggered if either arm recovers 0% or 100% of the population's episodes, |log-OR| > 10, or the fitter reports non-convergence.
  - The p of record is then the exact conditional test of a common odds ratio stratified by task × seed (exact Cochran–Mantel–Haenszel, e.g. R `mantelhaen.test(exact = TRUE)`), with episodes as units.
  - Its effect is the conditional MLE odds ratio with an exact CI.
- **Sensitivities** (never enter Holm):
  1. an exact sign-flip permutation over the (task, seed) run pairs of Σ(recovered episodes, Restarts − Full), using all 2^n flips, two-sided;
  2. A05's A/A added as a third Full replicate;
  3. runs whose selected features come only from the fallback excluded (01 §4.5).
- **`replace` mode** (A10-T2): `rec_any` is fitted with the same model and reported with equal prominence. It is a secondary, outside Holm.

**Verdict**
- Holm runs over all surviving contrasts of the family of 10 (01 §4.7); C3's threshold depends on its rank.
- **Full > Restarts** when the Holm-adjusted p < 0.05 and the estimate < 0 (C3 = Restarts − Full).
- **Restarts > Full** when the Holm-adjusted p < 0.05 and the estimate > 0. This is the "Opposite" reading of §3.
- Otherwise **"inconclusive"**, reported with the CI and the MDE.
- Write the raw p and "Holm-adjusted p < 0.05". Never write "Holm-adjusted p < 0.005".

**Gates**
- **G1 (Fri 16 Oct):** pool mode, R and D2 frozen; tag `agent2027-prereg`; A10-T18 passed before A05 launches at 18:00.
- **G2 (Mon 19 Oct):** power and informative tasks are recomputed. The test is never dropped for low power.
- **E9a slip:** if E9a misses v2.2, ship it in v2.3 (G0b-1, Wed 21 Oct) and launch A10 ahead of FX.
- **Sentinel:** if A06 FLAGs `S1020a`, apply A06 §6. If the cause is fixed, rerun from the reserve before L-LAST. Otherwise keep the rows with `drift_flag = 1`. C3 still enters Holm, since a contrast is dropped only before data exist; the Table 2 row carries the drift caveat.
- **G3 (Mon 26 Oct):** it governs only how a null is read. The pass rule is misdirected < Full, or oracle > Full with misdirected ≈ Full (A09). If G3 fails, an A10 null is reported as possible insensitivity, with process metrics only. A significant A10 result stands either way.
- **L-LAST (Thu 29 Oct):** if A10 has not launched, C3 has no data. It is dropped and Holm is recomputed over the rest. This drop rule is pre-registered in A10-T3.
- **DF:** Sat 31 Oct, 24:00.

## 7. Risks and cut levers

- **No lever (CL1–CL7) cuts A10.** CL7 or ED6 (L0 moves to T2) removes the ladder step from A11, and then "learns from feedback" cannot be written.
- **Seed collapse.** Mitigated by the seed rule, the T7 distinct-focus check and H3. If H3 reads "correlated draws", call the restarts correlated draws.
- **Three confounded mechanisms; handicapped comparator.** Read A10 with A11, `rec_any`, `lineage_share` and focus diversity.
- **`replace` mode.** Tiny restart pools favour Full at selection. The A10-T2 `rec_any` rule applies.
- **Missing per-iteration log.** A10-T18 at G1, with the G2 fallback.
- **Non-concurrent control** (v2.0, `S1016a`). Covered by the byte-identical guard, the switch-off unit test and the A06 sentinel. Disclosed.
- **Full's iteration-0 draws are shared across folds** (same prompt, seed and data). Restarts get distinct seeds, which makes the arms' episode correlation structures differ. Disclosed; `(1|task:seed)` absorbs run-level correlation.
- **Low power; P1 at the ceiling.** "Inconclusive" is pre-declared.
- **Token mismatch.** Report both the iteration-matched and the iso-cost views.
- **Serper cache.** It lowers restart diversity; disclose it.
- **Developer time.** The seed rule adds 0.1 d to E9a in an over-full week (01 §6.2). If it slips, E9a still ships without it only after Henrik approves, and H3 becomes mandatory reading.

## 8. Definition of done

- [ ] D3, the seed rule, the H3 rule, the L-LAST drop rule and the C3 text are in `agent2027-prereg`.
- [ ] The per-iteration active-feature log is in v2.0 (A10-T18).
- [ ] E9a is in `agent2027-v2.2`; the Full-v2 diff is byte-identical; the T6 tests pass; the smoke passed.
- [ ] All 14 runs have final W&B states and `run-output` artifacts; A06's verdict for S1020a is recorded.
- [ ] The C3 p of record, the OR and Δ with their CIs, and the Holm verdict are logged; the secondaries and H3 are done.
- [ ] The Table 2 row, Fig. 2 point, results sentence, threats paragraph and appendix are written.

---

**New names introduced** (not in the 01 §7 registry):
- the **restart seed rule** `seed = llm_seed + index`. It is behaviour of `ROGUE_ONE_RESTART_MODE=every_iteration`, implemented through an internal `contextvars.ContextVar` in `clients.py`, not a flag;
- the test file `tests/unit/test_restart_mode.py`;
- the fallback snapshot `intermediate_steps/df_attributes_explanations_trail_<index>.csv` (only if A10-T18 falls back);
- the analysis columns `q0_iter0` and `rec_any_pred` (= 1 − (1 − `q0_iter0`)^10), and the A10 variant of `reproposal_ratio`;
- task A10-T18, added at the end of the numbering so that existing IDs keep their numbers;
- the CAIR folder `~/ro27_analysis/A10/` (analysis outputs; W&B stays the record);
- the `notes` values `smoke`, `infra-rerun` and `pair with A05-<nnn>`;
- sessions `S1020s` (smoke, shared with A11), `S1020a` (main, as in A06, A09 and A11) and `S1020b` (second server, only if D5 = TP2).
