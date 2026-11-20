# A11 — LF2-L0: Open-loop planner (no feedback)

*Rogue One v2 design study · ICSE AGENT 2027 · submit Fri 13 Nov 2026 · scope T1-minimum · written Wed 7 Oct 2026. Every "hoped" result is a hypothesis, not a finding. Code refs: RougeOne repo `nested-cv-rebuttal` @ `07cbf93` (`C:/code/PHD/RougeOne`; no code diff to the branch tip `1bdaf42`). Short paths: `main.py` = `rogue_one_main.py`; `sci/agent.py`, `ext/agent.py`, `tst/agent.py` = `rogueone/llm/agents/core/{scientist,extractor,tester}/agent.py`; `config.py` = `rogueone/utils/config.py`; `agent_roles.py` = `rogueone/llm/prompts/agent_roles.py`. All file:line references were re-verified against `07cbf93` on Wed 7 Oct. E2 and E5 shift line numbers before the freeze, so locate every edit by its text. Shared doc = `docs/ablations/01_shared_infrastructure.md` ("01 §n"). This document restates everything A11 needs.*

| Field | Value |
|---|---|
| **ID** | A11 · LF2-L0 · arm `l0` (`ROGUE_ONE_FEEDBACK_LEVEL=l0`) |
| **Claim / Holm contrast** | Exploratory support for C3 ("the loop learns from feedback"). **Outside Holm** (R14). The confirmatory C3 contrast is A10 (Restarts − Full). |
| **Tier** | T1-minimum. It is the last cut lever (CL7) and the last engineering drop (ED6): L0 → T2. |
| **Owner** | Henrik |
| **Start / finish** | E9b built Mon 19 Oct · image `rogue_one:agent2027-v2.2` built and smoke session `S1020s` run **Tue 20 Oct** by 11:00 · main session `S1020a` (shared with A10) launched **Tue 20 Oct** by 14:00 · runs done ≈ Wed 21 Oct, morning · infra/leak reruns by Tue 27 Oct (Thu 29 Oct at the latest, L-LAST) · preliminary scoring Wed 28 Oct, final scoring Sun 1 Nov (after DF) · verdict Tue 3 Nov · text in draft Fri 6 Nov |
| **Budget** | **14 new runs, 9.8–14u** (14 × 0.7–1.0u, SYN-P), identical to the T1-minimum LF2-L0 card and 01 §7 R1. That is 7–10 box-hours at 1.4 u/h (2 u/h × 0.7) = 28–40 H100-hours at TP4 (14–20 at TP2 if D5). Control: 14 A05 runs (0 new). **Outside the 14**, from the shared 15% rerun reserve (01 §4.9): two 1-fold smoke rows (A11-000 `l0`, A11-015 Full-v2 guard; ≈ 0.14–0.2u each, ≈ 0.28–0.4u) and ≤ 2 reruns (1.4–2u). **Developer time:** ≈ 2.95 d over A11-T1…T20; only E9b's 0.5 d (A11-T5…T8) is in the 01 §6.2 engineering budget; A11-T10…T13 are shared with A10. |
| **Models** | O120 = `openai/gpt-oss-120b` for S, E, T and W (uniform). Served by `vllm/vllm-openai:v0.22.0`, TP4 with 6 clients (or TP2 per D5), port `8010`. |
| **Status** | Not started |
| **Depends on** | E1, E2, E3, E4, E5, E6, E8a, **E9b**, E11a, E11b, X10 · W&B artifact `prompt-baseline-v2.0` (01 §2.2) · A05 (control), A06 (session sentinel), A08 CPU screen (brute-force gate), A09 (G3 planner positive control), A10 (shares the v2.2 release, smoke session `S1020s` and main session `S1020a`) · D2, D3, D4, D5, D7 · G1, G2 |
| **Feeds into** | the exploratory C3 ladder with A10 (Restarts → L0 → Full) · the L0 point on Fig. 2 (right panel) · an exploratory row in Table 2 · A21 (iso-cost point) · A23 (Pareto front) |

## 1. What we test and why it matters

In arm `l0` the Scientist (the planner) never learns how earlier trials went. It sees no `train_*` scores, no Tester report, and no "pruned" labels or prune rationales. It keeps everything else:
- the focus history and notebook;
- web search and raw-data access;
- the names and descriptions of every feature added so far.

The Extractor and the Tester run exactly as in Full-v2. We ask two things: does the planner then recover fewer planted features (`rec_sel`), and does it re-propose features the Tester already removed?

Read A11 together with A10 as a ladder, Restarts → L0 → Full:
- **L0 − Restarts** is the value of the carried-over history and pool;
- **Full − L0** is the value of measured feedback.

A11 is the only arm that tests the "learns from **feedback**" half of C3. Without it, the paper can claim only "builds on earlier rounds".

## 2. Background you need

**The loop.** A run has 5 outer folds × 10 iterations: `main.py:1420-1442` calls `forward_pass(i)` for i = 0…`k_folds × num_iterations − 1`, and `forward_pass` derives `fold_run_id` and `iteration_in_fold` at `main.py:539-540`. Fold state is reset only when `iteration_in_fold == 0` (`main.py:543-544` → `_reset_fold_state`, `main.py:417-457`). The reset empties the attribute table (columns at `:439-449`), the feedback pool (`:450`), `focus_history` (`:451`) and the Scientist's notebook (`:457`). Each iteration has three steps:
1. **Scientist.** `determine_focus` (`main.py:548-555`) receives `df_scientist_feedback_pool`, `df_attributes_explanations` and `focus_history`.
2. **Extractor.** It builds features (`main.py:564-570`).
3. **Tester.** It scores and prunes them (`main.py:789-798`).

One feedback row per trial is then appended to the pool: `main.py:823-831` for completed trials, `:690-713` for aborted ones.

**The verifier-to-planner channel (what L0 cuts).**

| Element | Where (Full-v2) | L0 |
|---|---|---|
| `df_test_results`, the Scientist's copy of the pool (`sci/agent.py:101-103`), read through `generic_pandas_tool` (`:179-271`) | The rows come from `_build_scientist_feedback_row` (`main.py:384-415`): `id`, `status`, `dataset_id`, `fold_seed`, `aborted_fold_id`, `skip_reason`, `feature_assessment` (the Tester's markdown report) and mean **in-sample** `train_*` metrics. These are computed by predicting on `X_train` after fitting on it (`tst/agent.py:1708-1714` R, `:1866-1892` C). Held-out metrics are never shown (docstring `sci/agent.py:190`). | Only the `id` column |
| `attribute_lookup_tool` (`sci/agent.py:305-335`, filter at `:319-321`) | Returns every column: Attribute, Description, Pandas Command, **Status**, Added, **Pruned Iteration**, **Prune Rationale**. Pruned rows stay in the table with `Status="Pruned"` (`tst/agent.py:1302-1310`). | The 3 bold columns are dropped |
| `list_all_attributes` (`sci/agent.py:273-303`) | Names only, pruned ones included (`:289`) | Unchanged |
| Scientist instructions (`sci/agent.py:409-456`) and tool description | They refer to results at the 7 sites in A11-T6 | Neutralised locally |

**What L0 keeps (residual channels; disclosed).**
- The round prompt (`sci/agent.py:472-475`).
- Both Scientist fallbacks: the no-tool fallback (`:549-577`) and the heuristic focus (`:501-517`). The heuristic text mentions "strongest active attributes" but carries no numbers.
- `GLOBAL_GOAL` (`agent_roles.py:7-11`; the Tester sentence is `:10`), which says that a Tester evaluates features "using classification/ regression metrics". It is shared by all three roles, so it stays.
- The variable name `df_test_results` inside `generic_pandas_tool` (kept so the code path is identical); edit 7 changes only its description.
- The Extractor's own view of pruned status: auto-unprune (`ext/agent.py:465-485`), the "restore a pruned feature" instruction (`ext/agent.py:895`), and `Status` in `list_known_attributes_tool` (`ext/agent.py:570-596`) if D7 = expose. At `07cbf93` that tool returns column names only (`:584`); E5 adds `Status` only if D7 = expose. The Extractor has no tool that reads `Prune Rationale`.
- The Tester's pruning floor (`tst/agent.py:119-163`).
- Extractor-written Descriptions, which may argue that a feature is useful but contain no test results.

**L0 therefore cuts only the verifier-to-planner channel.**

**D2 (freeze item 12).** The recommended default keeps the in-sample training fit, and the L1 code is CUT. Under that default, L0 removes "the verifier-to-planner channel as frozen", not "an informative scalar". The Tester report still carries per-feature evidence (MI, SHAP, PFI, VIF), so L0 removes real information even if `train_*` is saturated. If D2 = L1, L0 also drops the L1 column, because it keeps only `id`.

**Instruments.** **SYN-P 7** = `synp_p1`, `synp_p2`, `synp_p3`, `synp_p4`, `synp_p5`, `synp_p6a`, `synp_p6b`. These are generated tasks (n = 2,000, p ≈ 20–30), each with one planted feature φ. P7 and P0 are excluded under the P7 policy.

| Task | φ | Type (proposed; E8a confirms) | Role here |
|---|---|---|---|
| `synp_p1` | x_a / x_b | R | expected non-discriminating (A08 gate), so a sanity check only |
| `synp_p2` | (x_a / x_b) · x_c | R | moderator level set by D3 |
| `synp_p3` | 1[x_a / x_b ≥ t] ∧ (cat = k) | C | multi-step |
| `synp_p4` | sin/cos of an hour-like column | C | single-formula |
| `synp_p5` | sign(LW·LD² − RW·RD²) | C | multi-step |
| `synp_p6a` / `synp_p6b` | named composite / exact anonymised twin | R | single-formula |

**Brute-force gate.** A task is non-discriminating if OpenFE or AutoFeat recovers φ in ≥ 14 of 15 fold-episodes in A08's CPU screen. It still runs, but it is excluded from the analysed population.

**Metrics** (R13 names; unit = the fold-episode):
- **`rec_sel` (primary).** In the iteration that inner CV selects, an active, non-fallback feature has |Spearman| ≥ 0.95 with φ on training rows (NMI ≥ 0.9 for P3 and P5). Raw columns and single-column monotone transforms never count.
- **Other outcomes:** `rec_any`, `iter_first_rec`, `tok_to_first_rec`, `success`, `collapse`, `cap_hit`.
- **Process metrics:**
  - `reproposal_ratio`: a new feature with |ρ| ≥ 0.99 against the active or pruned pool, or the same normalised AST;
  - `restore_pruned_count`;
  - `focus_novelty`: embedding cosine to earlier focuses (a repeat at ≥ 0.9);
  - `report_read_rate`: Scientist tool calls that touch `feature_assessment` (under L0, these are failed attempts);
  - `fallback_share`, `prune_rate`;
  - cost: `tok_compl_scientist`, `llm_calls_scientist`, `tok_compl_total`, `wall_s`.

## 3. Hypotheses and interpretation (all hypotheses)

- **H1, hoped.**
  - L0 lowers `rec_sel` on the multi-step tasks (P3, P5, and P2 if D3 places it there). A smaller change on the single-formula tasks is expected but is not a decision criterion: under `rec_sel`, losing feedback can also change which iteration is selected, so a tie there is not assumed.
  - L0 raises `restore_pruned_count` and `reproposal_ratio`, because it cannot see `Status`. This is the expected signature (descriptive; its reading rule is in §6).
  - The master plan's "focus repetition ≥ 1.5×" is weakly motivated, because L0 keeps the focus history and the anti-repetition rule. It is reported, but it is not a decision criterion.
  - **Sentence:** *"Cutting the verifier-to-planner channel lowered multi-step recovery from X% to Y% and raised restorations of pruned features Z-fold (exploratory)."*
- **H0, null (L0 ≈ Full).** The frozen feedback channel does not drive recovery. Two free checks classify the null (§6 R2):
  - **unused:** Full's Scientist rarely reads the channel;
  - **uninformative:** the `train_*` scalar is saturated.

  **Sentence:** *"The planner's choices were driven by focus history and search rather than by measured feedback (Δ = …, 95% CI …)."* State the limitation that the L1 contrast is cut.
- **H−, opposite (L0 > Full).** The feedback as delivered misleads the planner, for example by pulling it into exploiting the wrong features. Diagnose this with `focus_explore_share`, `feedback_read_share` and the re-proposal metrics. Feedback design (signal; push vs pull) becomes the first T2 item, and Full-v2 stays frozen.

## 4. Design

**Control.** A05 `full`, `replicate` 0, on SYN-P 7 × {42, 4}: 14 existing runs from rows A05-031…046 with the two P7 rows excluded. If A05's SYN-P 8 rows are in the order p1…p7 (P7 last), the control rows are **A05-031…037** (seed 42) and **A05-039…045** (seed 4); check this against the committed `manifests/agent2027/A05_M0.csv` in A11-T9. **Sensitivity reference:** A05's seed-42 A/A (`replicate=1`, `llm_seed` 1042), **A05-047…053** under the same ordering (7 runs).

**Arm.** `l0`, with `env_overrides = {"ROGUE_ONE_FEEDBACK_LEVEL":"l0"}`, plus the paired A05 row's `ROGUE_ONE_STRATIFIED_FOLDS` override, which is part of the task definition (`"True"` on `synp_p3`, `synp_p4`, `synp_p5` if D4 is adopted). Nothing else differs.

**Fixed settings** (every other R3/R4 flag at its `scripts/agent2027/v2_defaults.env` value, exactly as A05 §4.1; the effective values are logged as `flags/<NAME>` per R12):
- **Loop:** `ROGUE_ONE_NUM_ITERATIONS=10` (the code default is 3, `config.py:305`, so it is always set), `ROGUE_ONE_K_FOLDS=5`, `ROGUE_ONE_SELECTION_SIGNAL=inner_cv`, `ROGUE_ONE_ENABLE_ROLLBACK=False`, `ROGUE_ONE_FOLD_RETRY_LIMIT=1`, `ROGUE_ONE_DO_BALANCING=True`;
- **Run control:** `ROGUE_ONE_START_VLLM=False`, `ROGUE_ONE_DO_LOGGING=True`, `ROGUE_ONE_DISABLE_WANDB=0`, `CONSOLE_DISABLE=False`, `ROGUE_ONE_TASK_REGISTRY` (E1), `ROGUE_ONE_SMOKE_FOLDS` unset (smoke rows only: `1`);
- **Sampling:** temperature 0.7 for all roles and the summariser (`LLM_TEMPERATURE`, `SCIENTIST_/EXTRACTOR_/TESTER_/MEDICAL_KNOWLEDGE_AGENT_TEMPERATURE`); `EXTRACTOR_AGENT_MAX_TOKENS=32768`, `TESTER_AGENT_MAX_TOKENS=8192`, `SCIENTIST_AGENT_MAX_TOKENS` as pinned at G1; max iterations E/T/S 100/100/200; `*_AGENT_REASONING_EFFORT` unset;
- **R4 flags:** `ROGUE_ONE_TOOL_ERROR_FEEDBACK=message`; `ROGUE_ONE_EXTRACTOR_ROUNDS`, `ROGUE_ONE_TOOL_OUTPUT_MAX_CHARS` and `ROGUE_ONE_FEATURE_POOL_MODE` as frozen at G1; `ROGUE_ONE_REASONING_EFFORT_PASSTHROUGH=0`; `ROGUE_ONE_LLM_TRACE=1`; `ROGUE_ONE_PROMPT_DUMP=1`; `ROGUE_ONE_REPLAY_VALIDITY_CHECK=1`; `ROGUE_ONE_SEARCH_CACHE=shared`; `ROGUE_ONE_SEARCH_CACHE_PATH=/workspace/cache/serper_cache.sqlite`; `ROGUE_ONE_PLANNER_MODE=llm` (E9c, in v2.1+); `ROGUE_ONE_RESTART_MODE=off` (E9a, same image); `ROGUE_ONE_SANDBOX` as frozen (`on`, or `off` under ED1);
- **Evaluator:** XGBoost 400 trees (`tst/agent.py:1673` R, `:1815` C), unchanged code;
- **Infrastructure:** one O120 server, TP4 (TP2 if D5), `vllm/vllm-openai:v0.22.0`, `MAX_MODEL_LEN=131072`, `PARALLEL=6`, `CPUS=12`, `RUN_CAP_HOURS=6`, `RUNROOT=~/rogueone-runs-agent2027`.

**Seeds.** `fold_seed` ∈ {42, 4}, with `llm_seed = fold_seed` (`replicate=0`).

**Run matrix** (W&B entity `PlanBench`, project `RogueOne-AGENT2027`, group `A11`, job_type = `arm_id` (`l0`; `full` for A11-015 only), tags [`<SESSION_ID>`, `<code_phase>`, `agent2027-v2.2`]; manifest `manifests/agent2027/A11_LF2-L0.csv`)

| Rows | arm_id | dataset_id | fold_seed / llm_seed / replicate | New runs | u | Run name |
|---|---|---|---|---|---|---|
| A11-001…007 (p1, p2, p3, p4, p5, p6a, p6b) | `l0` | SYN-P 7 | 42 / 42 / 0 | 7 | 4.9–7 | `A11__l0__<dataset_id>__f42__r0` |
| A11-008…014 (same order) | `l0` | SYN-P 7 | 4 / 4 / 0 | 7 | 4.9–7 | `A11__l0__<dataset_id>__f4__r0` |
| Control (A05-031…037, A05-039…045) | `full` | SYN-P 7 | 42, 4 / = fold seed / 0 | 0 (14 reused) | 0 | `A05__full__<dataset_id>__f<seed>__r0` |
| A/A reference (A05-047…053) | `full` | SYN-P 7 | 42 / 1042 / 1 | 0 (7 reused) | 0 | `A05__full__<dataset_id>__f42__r1` |
| A11-000 · switch smoke (not analysed) | `l0` + `ROGUE_ONE_SMOKE_FOLDS=1` | SYN-P task of `prompt-baseline-v2.0` (proposed `synp_p3`) | 42 / 42 / 0 | 1 fold-run | ≈ 0.14–0.2 (reserve) | `A11__l0__<dataset_id>__f42__r0` (same name as A11-003; separated by `code_phase=smoke`); session `S1020s`, `notes=smoke` |
| A11-015 · Full-v2 guard smoke for the shared v2.2 release (not analysed) | `full` + `ROGUE_ONE_SMOKE_FOLDS=1` | same task as A11-000 | 42 / 42 / 0 | 1 fold-run | ≈ 0.14–0.2 (reserve) | `A11__full__<dataset_id>__f42__r0`; session `S1020s`, `notes=guard-smoke`, `code_phase=smoke` |

**Budget arithmetic.**
- 7 tasks × 2 seeds × 1 arm = 14 runs × 0.7–1.0u = **9.8–14u** (matches the T1-minimum card and R1).
- At 1.4 u/h (2 u/h × 0.7) on one 4-GPU box, that is 7–10 h of box time, or 28–40 H100-hours.
- Smoke session `S1020s` (A10-000, A11-000, A11-015): 3 fold-runs in parallel, ≈ 0.42–0.6u, ≈ 1 h wall clock including O120 server start-up, so it starts by ≈ 10:00 to meet the 11:00 due time; from the reserve.
- Main session `S1020a` = A10 14 + A11 14 + 1 A06 sentinel = 29 runs, 20.3–29u, or **14.5–20.7 h** at 1.4 u/h. A launch by 14:00 Tue 20 Oct ends ≈ 04:30–10:45 Wed 21 Oct. A08's GPU comparators are **not** in this session: they use their own launcher (A08-T16) and run after `S1020a` ends.

**Units and power.**
- 70 fold-episodes per arm, or about 60 after the gate if P1 drops out.
- Multi-step subset: 20 episodes per arm (30 with P2).
- Planning MDE (unclustered, 01 §4.7): about 40–45 pp at α = 0.05 on 20 episodes per arm; ≈ 33–37 pp on 30 (× √(20/30)); ≈ 23–26 pp on the pooled 60, but a multi-step-only effect is diluted about 3× when pooled. G2 recomputes these from A05's ICC. "Inconclusive" is a pre-declared outcome.
- L0 can also lower ceiling tasks, so for power accounting every task where Full recovers > 0 counts (01 §4.7). Exclusion from the analysed population comes only from the brute-force gate.

**Pairing and randomisation.**
- Each A11 row pairs with the A05 row that has the same `dataset_id` and `fold_seed`, so both use the same outer partition and `llm_seed` (`notes = pair with A05-<nnn>`).
- In `S1020a`, A11 rows are interleaved with A10's in a random `order_key` order (RNG seeded from the session ID).
- The session carries one A06 `sentinel` row in its first wave (`order_key` = session minimum). `S1020a` is the guard-3 session for v2.2 (A06 §4.5), so A11 rows are provisional until A06's verdict for it is logged.
- **The contrast is not concurrently randomised.** The control ran on `agent2027-v2.0` in `S1016a` (`S1016a`/`S1016b` under TP2). Comparability rests on the byte-identical Full-v2 prompt guard (A11-T8), the switch-off unit tests (A11-T7) and the sentinel. A session covariate is aliased with arm and cannot be fitted.

**Failure handling (ITT).**
- A crash (non-zero rc, or no `fold_aggregation_summary.csv`) is retried once, alone, at the end of the session.
- Persistent failures and 6 h cap hits are ITT failures in **both** arms: every fold-episode gets `rec_sel = rec_any = 0`. They are never dropped, and failure and cap-hit rates are reported.
- Infrastructure faults (vLLM dies, W&B or Serper is unreachable) are not agent failures: re-queue the row with `notes=infra-rerun` in the next O120 session that carries its own sentinel.
- A run that fails the leak audit (A11-T16) is an implementation fault, not ITT: fix, re-release and re-run it.
- Sensitivity analysis: exclude runs whose selected features come only from the fallback.

## 5. Tasks

### 5.1 Prerequisites (verify before launch)
- [ ] **A11-T1** [type: decision] (0.1 d, due Thu 15 Oct) — Get Henrik's decisions on:
  - D2: the feedback signal (default: training fit, with a written justification);
  - D3: the C3 population and P2's placement; A11 uses the same population as A10;
  - D4: stratified SYN-P classification folds;
  - D7: expose `Status` to the Extractor.

  Also read D5 (O120 TP2 vs TP4, due Fri 16 Oct; it only changes the session split in A11-T9/T12), and confirm the §6 thresholds marked *proposed*. *Done when:* all are written into the draft `docs/agent2027/preregistration.md`.
- [ ] **A11-T2** [type: writing] (0.1 d, due Fri 16 Oct, before tag `agent2027-prereg`) — Pre-register the A11 rules: §6 R1–R3, the null classification, the process outcomes and the leak audit. *Done when:* they are in the tagged commit (X10), and W&B artifact `prompt-baseline-v2.0` exists.
- [ ] **A11-T3** [type: data] (0.1 d, due Mon 19 Oct, G2) — Check the inputs:
  - the 14 control and 7 A/A A05 rows are `finished`, each with a `run-output` artifact;
  - the E11a G2 memo exists (ICC, informative tasks, sentinel task);
  - the A08 gate labels are recorded.

  *Done when:* a W&B query (01 §5.2 step 7) lists all 21 rows as `finished`.
- [ ] **A11-T4** [type: data] (0.1 d, due Mon 19 Oct) — On one A05 SYN-P `run-output`, confirm the E3 outputs:
  - `llm_traces.jsonl.gz` has a `tools` field and the tool-result messages;
  - the focus JSONL has 50 rows;
  - `report_read_rate`, the fallback counters and `tok_compl_scientist` are present;
  - `tester_reports/` exists (written at `main.py:849-857`).

  *Done when:* all four are found.

### 5.2 Implementation (E9b, 0.5 d, branch `agent2027/v2`, default `full`; Full-v2 must stay byte-identical)
- [ ] **A11-T5** [type: code] (0.1 d, due Mon 19 Oct) — Add the flag:
  - read `ROGUE_ONE_FEEDBACK_LEVEL` (`full` | `l0`) into `rogueoneConfig` next to `selection_signal` (`config.py:317-319`), and raise on any other value;
  - add `"ROGUE_ONE_FEEDBACK_LEVEL": "full"` to `OPTIONAL_ENV_DEFAULTS` (`rogueone/utils/config_keys.py:22-57`) and to the E1 flag registry;
  - add `ROGUE_ONE_FEEDBACK_LEVEL=full` to `v2_defaults.env`;
  - import `rouge_one_cfg` (created at `rogueone/utils/__init__.py:15`) in the `sci/agent.py` import block (`:26-29`).

  *Done when:* a unit test shows unset → `full`, `l0` → `l0`, `L0 ` → `l0` (stripped, lower-cased like `selection_signal`), and any other value → `ValueError`; `v2_defaults.env` and the E1 flag registry contain the flag. (Its appearance as `flags/ROGUE_ONE_FEEDBACK_LEVEL` in `wandb.config` is checked in A11-T11.)
- [ ] **A11-T6** [type: code] (0.25 d, due Mon 19 Oct) — Scientist-only changes, active only when `l0`. Never edit `agent_roles.py`: `AgentRoles.TESTER` is also rendered into the Extractor prompt (`ext/agent.py:873`).
  - **Data views** (module-level helpers, so they can be unit-tested):
    - `_scientist_view_test_results(df, level)`. Under `full` it is exactly `sci/agent.py:101-103`. Under `l0` it returns `df[["id"]].copy()`, or `pd.DataFrame(columns=["id"])` when the pool is empty.
    - `_scientist_view_attributes(df, level)`. Applied to `res` in `attribute_lookup_tool` (`:319-321`). Under `l0` it drops `Status`, `Pruned Iteration` and `Prune Rationale` (`errors="ignore"`).
    - Leave `list_all_attributes`, `main.py`'s pool and all logging unchanged.
  - **Prompt edits.** `_apply_l0_prompt_edits(instructions)` applies replacements 1–6 to the rendered instructions and asserts that each `old` string occurs exactly once. Edit 7 swaps the tool description.

  | # | Site (text @07cbf93; match post-E5 text) | L0 version |
  |---|---|---|
  | 1 | `sci/agent.py:413` "…based on the test results provided by the tester agent and the explanations…" | "…based on the explanations of the attributes extracted by the extractor agent." |
  | 2 | `:414` "Your task is to analyze the test results, the attribute explanations, …" | "Your task is to analyze the attribute explanations, the focus history, and compare…" |
  | 3 | `:423` embeds `AgentRoles.TESTER` (`agent_roles.py:24-29`) | the same text without `:27-28` (the "detailed feedback to the Scientist Agent" sentence and the "This feedback is crucial…" sentence) |
  | 4 | `:430` "2. Identify strongest and weakest tested attributes." | "2. Review the attributes extracted so far." |
  | 5 | `:440` "Focus on exploratory strategies if current performance is poor or an early round, and exploitative refinements if performance is strong." | "Choose between exploratory and exploitative focuses using the focus history and the attributes extracted so far." |
  | 6 | `:443` ", or restoration of features the Tester has previously pruned with concrete arguments for why they should come back" (E5 rewrites the CAR part of this line before the freeze) | clause deleted |
  | 7 | `generic_pandas_tool` description (`:182-200`): "scientist feedback dataframe", the column list (`:186-190`), "insights into the test results" (`:194`), and the `train_accuracy` example (`:199-200`) | "trial-index dataframe with one row per earlier trial and only the column `id`"; "explore the data"; the example `result = df_raw_data.describe()` |

  For edit 7, use `@function_tool(name_override="generic_pandas_tool", description_override=…)`; the name must stay identical for the E3 counters. Check that openai-agents 0.9.1 (`requirements.txt:1`) supports `description_override` (**to verify**: the package is not installed locally, so check it inside the client image). If it does not, define two decorated variants under the same tool name.

  *Done when:* the A11-T7 tests pass.
- [ ] **A11-T7** [type: code] (0.1 d, due Mon 19 Oct) — Write `tests/unit/agents/test_scientist_feedback_level.py`. It must show:
  - **under `full`:** the helpers return their input unchanged, and the rendered instructions and tool description equal those at tag `agent2027-v2.0-freeze` (i.e. `07cbf93` plus the freeze changes, E2/E5 included) — this is guard 2 of 01 §2.2;
  - **under `l0`:**
    - the test-results view has columns `['id']`;
    - `result = df_test_results['feature_assessment']` and `['train_accuracy']` return tool errors;
    - the lookup output has none of the 3 columns;
    - the list still returns pruned names;
    - each edit applies exactly once;
    - none of "test results", "tested attributes", "previously pruned", "feedback to the Scientist" or `train_` remains in the rendered instructions or the tool description;
  - **an invalid flag value** raises.

  *Done when:* CI is green.
- [ ] **A11-T8** [type: code] (0.05 d; merge Mon 19 Oct, guard and tag due Tue 20 Oct, 11:00) — The shared v2.2 release guard (A10 relies on it).
  1. **Mon 19 Oct:** merge E9b with E9a on `agent2027/v2` as the v2.2 release candidate (default `full`/`off`), with the A11-T7 and A10 unit tests green; push. Do **not** tag yet.
  2. **Tue 20 Oct, after A11-T10/T11:** download the `output/prompts/` dumps and traces of A11-015 (Full-v2) and A11-000 (`l0`) from their `run-output` artifacts (in a container, 01 §5.2 step 7 pattern) and run `scripts/agent2027/prompt_diff.py`:
     - A11-015 vs `prompt-baseline-v2.0` (same task and seed): **byte-identical** for every role; tool descriptions are compared via the trace `tools` field (the dump holds only the system prompt and the first user message);
     - A11-000 vs A11-015: the Scientist system prompt and tool description differ by exactly edits 1–7 and nothing else; the Extractor and Tester prompts and tool lists are byte-identical (only iteration 0 of the fold is compared, because later user messages carry run-specific focus history).
  3. Tag the commit that A11-T10 built as `agent2027-v2.2` and push the tag.

  *Done when:* both diffs pass (guard 1 of 01 §2.2), their logs are attached to the v2.2 release record (release PR or annotated tag message), and the tag `agent2027-v2.2` points at the HEAD recorded in A11-T10. If a diff fails: fix, rebuild, re-run S1020s; `S1020a` does not launch until it passes.

### 5.3 Execution on CAIR

**Hard rules (01 §5.1; apply to every step below):**
- **Containers only.** Every execution — runs, analysis, W&B queries, file inspection with Python — happens inside Docker. The CAIR host only issues `docker` / `docker compose` commands and the bash launchers in `~` that issue them; `nvidia-smi`, `docker ps`, `grep` and `tail` are allowed for pre-flight and monitoring. Never run `python …` on the host. Any `docker compose run` over SSH needs `-T`.
- **GPU pre-flight before every GPU container** (the command in A11-T11). If not enough GPUs are free, **stop and report** the per-GPU memory and utilisation, the compute apps and the running containers, and let Henrik decide. Launch with explicit `GPUS=<ids>` from the pre-flight and `WAIT_MAX_MIN=0`, so a session never queues silently.
- **≤ 4 of the 8 H100s**, ever, without Henrik's explicit approval in chat. One Rogue One job at a time.
- **W&B is the system of record.** Runs are stateless and upload `output/` as artifact `run-output`. `WANDB_API_KEY` comes from `<CAIR_HOME>/code/PlanBench/.env` at run time; `SERPER_API_KEY` and LLM settings come from `.env` / `.secret.env` in `<ROGUEONE_CAIR_PATH>`, baked in at build. **Never push these images.**
- **Connection:** one-shot `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "…"` commands only (non-interactive; every command below is written out in full). Code reaches CAIR only by local commit + push of `agent2027/v2`, then `git fetch` over SSH on CAIR.
- **Long jobs:** detached `nohup` launcher, with the job log in `~` (`~/ro27_<SESSION_ID>.job.log`, vLLM log `~/vllm_session_<SESSION_ID>.log`), never in the root-owned run folders.

`<ROGUEONE_CAIR_PATH>` = `<CAIR_HOME>/code/RougeOne`. This is **inferred, not verified**: `REPO_TASKS="$HOME/code/RougeOne/tasks"` at `paper/scripts/clients_t16_rerun.sh:135` and home `<CAIR_HOME>` at `:59`. Verify it once with `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "ls -d ~/code/RougeOne"`.

**Execution order (Mon 19 – Tue 20 Oct):** A11-T9 and A11-T8 step 1 (Mon 19) → A11-T10 (Tue 20, morning) → A11-T11 (smoke `S1020s`, by 11:00) → A11-T8 steps 2–3 (guard and tag) → A11-T12 (main `S1020a`, by 14:00) → A11-T13. A10-T7 and A10-T9 are the same steps; whoever runs first does them once for both documents.

- [ ] **A11-T9** [type: config] (0.1 d, due Mon 19 Oct) — Locally, jointly with A10-T8, run `scripts/agent2027/make_manifest.py --ablation A11` to write `manifests/agent2027/A11_LF2-L0.csv`. It holds rows A11-000…015 with:
  - the R7 columns; `task_type` per E8a (`synp_meta/registry.yaml`); `llm_seed = fold_seed`, `replicate=0`, `is_sentinel=0`;
  - `image_tag=agent2027-v2.2`, `rung_S/E/T=O120`, `port_S/E/T=8010`;
  - `env_overrides={"ROGUE_ONE_FEEDBACK_LEVEL":"l0"}` on A11-000…014, plus `"ROGUE_ONE_STRATIFIED_FOLDS":"True"` copied from the paired A05 row (p3, p4, p5 if D4); A11-000 adds `"ROGUE_ONE_SMOKE_FOLDS":"1"`; A11-015 is `arm_id=full` with `{"ROGUE_ONE_SMOKE_FOLDS":"1"}` (plus the stratified override if its task is classification and D4);
  - `cap_hours=6`, `expected_u=0.85` (A11-000 and A11-015: 0.17);
  - `notes=pair with A05-<nnn>` (A11-000: `smoke`; A11-015: `guard-smoke`). The launcher must set `ROGUE_ONE_CODE_PHASE=smoke` on `smoke`/`guard-smoke` rows and `v2` otherwise (E1 behaviour, **to verify** in the G0a smoke).

  Then plan the sessions:
  - `make_manifest.py --plan-session S1020s --rows A10-000 A11-000 A11-015` (no sentinel: smoke session);
  - `make_manifest.py --plan-session S1020a --rows A10-001…014 A11-001…014 <next unused A06 sentinel row>`, with the sentinel in the first wave (A06 §4.5, A06-T12);
  - **if D5 = TP2:** split the `S1020a` rows alternately into `S1020a` (port 8010) and `S1020b` (`port_S/E/T=8011`), each with its own A06 sentinel.

  Commit and push. *Done when:*
  - the manifest and session files are committed and pushed to `agent2027/v2`;
  - the control-row mapping (A05-031…037, A05-039…045) is checked against the committed A05 manifest, and `task_type` and `ROGUE_ONE_STRATIFIED_FOLDS` match each paired A05 row;
  - a `make_manifest.py` dry run yields unique container names `ro27-s1020a-a11-0nn` and `ro27-s1020s-a11-0nn`.
- [ ] **A11-T10** [type: run] (0.2 d, due Tue 20 Oct, 10:00) — Bring up the code and image on CAIR (= A10-T7 steps 1, 3, 4; if A10 already did them, verify the HEAD and the image ID instead):
  1. **Sync:** `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15' git fetch git@github.com:HenrikNorkart/RougeOne.git agent2027/v2 && git checkout -B agent2027/v2 FETCH_HEAD && git log -1 --format='HEAD: %h %s'"`
  2. **Build** (`config.py:24` loads `.dev.env` with `override=True`, so it must not reach the image): `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && (test ! -e .dev.env || grep -qx '.dev.env' .dockerignore) && docker build -t rogue_one:agent2027-v2.2 . && docker image inspect rogue_one:agent2027-v2.2 --format '{{.Id}}' && docker run --rm --entrypoint sh rogue_one:agent2027-v2.2 -c 'test ! -e /workspace/.dev.env && echo no-dev-env'"`
  3. **Install** the launchers and manifests in `~` (01 §5.2 step 3): `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && cp scripts/agent2027/ro27_session_launch.sh scripts/agent2027/ro27_clients_manifest.sh scripts/agent2027/ladder_session.sh ~/ && cp paper/scripts/vllm_session.sh ~/ro27_vllm_session.sh && mkdir -p ~/ro27_manifests && cp -r manifests/agent2027/. ~/ro27_manifests/"`

  *Done when:* the HEAD hash and the image ID are recorded in the session plan, and the build printed `no-dev-env`.
- [ ] **A11-T11** [type: run] (0.1 d, due Tue 20 Oct, 11:00) — Run the smoke session `S1020s` (A10-000, A11-000, A11-015; = A10-T7's pre-flight and launch commands, steps 2 and 5).
  1. **Pre-flight:** `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader && echo '--- compute apps ---' && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader && echo '--- containers ---' && docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'"`. Proceed only if all three hold:
     - 4 GPUs (2 if TP2) each have < 2,000 MiB used (`FREE_MIB`);
     - Rogue One's total stays ≤ 4 GPUs;
     - no `ro27-*`, `rogueone-vllm-*`, `llmfe-*` or `caafe-*` container of ours is running.

     Otherwise **STOP and report** the output to Henrik.
  2. **Launch detached:** `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "SESSION_ID=S1020s SESSION_FILE=~/ro27_manifests/sessions/S1020s.csv GPUS=<ids from pre-flight> MODEL=openai/gpt-oss-120b PORT=8010 SERVER_NAME=rogueone-vllm-o120-s1020s LOG=~/vllm_session_S1020s.log IMAGE_RO=rogue_one:agent2027-v2.2 PARALLEL=6 CPUS=12 RUN_CAP_HOURS=6 RUNROOT=~/rogueone-runs-agent2027 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_S1020s.job.log 2>&1 & echo launched; sleep 5; tail -5 ~/ro27_S1020s.job.log"`
  3. Monitor until `RO27_SESSION_END S1020s` (A11-T13 loop with `S1020s`), then run the tear-down check (A11-T13).

  *Done when:*
  - W&B shows `flags/ROGUE_ONE_FEEDBACK_LEVEL=l0` and `code_phase=smoke` on A11-000, and `flags/ROGUE_ONE_FEEDBACK_LEVEL=full` on A11-015;
  - each of A11-000 and A11-015 has 10 trial rows in `df_test_pool_final.csv` and a `run-output` artifact with `prompts/` and `llm_traces.jsonl.gz`;
  - the leak audit (A11-T16 rule) finds 0 hits on A11-000;
  - the tear-down check prints `none`.
- [ ] **A11-T12** [type: run] (0.1 d, due Tue 20 Oct, 14:00) — Launch the main session `S1020a`, once for A10 and A11 (= A10-T9), only after A11-T8 has passed and been tagged. If A10 is not ready (E9a slipped), launch it with the A11 rows and the sentinel row only (re-plan the session file with A11-T9's command minus the A10 rows). LF1 has queue priority over A11 in any later session.
  1. Run a **fresh pre-flight** (A11-T11 step 1; same proceed rule; otherwise stop and report).
  2. Launch detached:

  `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "SESSION_ID=S1020a SESSION_FILE=~/ro27_manifests/sessions/S1020a.csv GPUS=<ids from pre-flight> MODEL=openai/gpt-oss-120b PORT=8010 SERVER_NAME=rogueone-vllm-o120-s1020a LOG=~/vllm_session_S1020a.log IMAGE_RO=rogue_one:agent2027-v2.2 PARALLEL=6 CPUS=12 RUN_CAP_HOURS=6 RUNROOT=~/rogueone-runs-agent2027 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_S1020a.job.log 2>&1 & echo launched; sleep 5; tail -5 ~/ro27_S1020a.job.log"`

  - **If D5 = TP2:** run two such jobs on 2 GPUs each. The second uses `SESSION_ID=S1020b`, `PORT=8011`, `SERVER_NAME=rogueone-vllm-o120-s1020b`, `LOG=~/vllm_session_S1020b.log`, `ROGUEONE_ALLOW_SECOND_SERVER=1` and job log `~/ro27_S1020b.job.log`; start it only after a fresh pre-flight shows the first server's GPUs are the only ones Rogue One holds (ΣGPU ≤ 4).

  *Done when:*
  - the job log shows the vLLM version assert and both image IDs;
  - `ro27-s1020a-a11-*` containers are up;
  - the W&B runs are `running`.
- [ ] **A11-T13** [type: run] (0.1 d, Tue 20 – due Wed 21 Oct) — Monitor (per job log under TP2):
  - background loop until the end marker: `ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 CAIR-gpu14 "while ! grep -q 'RO27_SESSION_END' ~/ro27_S1020a.job.log 2>/dev/null; do sleep 120; done; tail -30 ~/ro27_S1020a.job.log"` (exit 255 is a network drop, not the job: reconnect);
  - W&B states, in a container (01 §5.2 step 7): `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker run --rm --env-file <CAIR_HOME>/code/PlanBench/.env --entrypoint python3 rogue_one:agent2027-v2.2 -c \"import wandb; api=wandb.Api(); [print(r.name, r.state) for r in api.runs('PlanBench/RogueOne-AGENT2027', filters={'group': 'A11'})]\""`. A `running` run with no live container is a zombie: flag it;
  - tear-down check (01 §5.2 step 6): `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker ps --format '{{.Names}}' | grep -E '^(ro27-|rogueone-vllm-)' || echo none; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"`.

  *Done when:* 14 A11 rows have an `OK`/`FAILED`/`CAP_HIT` line in the job log, the tear-down check prints `none`, and our GPUs are back below 2,000 MiB.
- [ ] **A11-T14** [type: run] (0.1 d, due Tue 27 Oct; Thu 29 Oct at the latest, L-LAST) — Re-queue rows lost to infrastructure faults (`notes=infra-rerun`), or leak-audit failures (on the fixed, re-guarded image; §6), in a later O120 session that has its own sentinel, using the A11-T12 pre-flight and launch pattern with that session's ID. Fri 30 – Sat 31 Oct are for sentinel-triggered reruns only (A06 §6), and only if they end before DF (Sat 31 Oct, 24:00). *Done when:* every A11 row has a final W&B state (`finished`, or an ITT failure).

### 5.4 Analysis

All analysis runs in containers on CAIR (CPU only, `--cpus 16`), never on the host. The E11b scripts land Tue 27 Oct, after v2.4, so they are mounted read-only from the synced repo (sync first with the A11-T10 step 1 command). Results are logged as `ANALYSIS__<script>__<YYYYMMDD>` in group `ANALYSIS`. Container pattern (CLI arguments are set by E11b):
```
ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "mkdir -p ~/ro27_analysis/A11 && docker run --rm --cpus 16 --env-file <CAIR_HOME>/code/PlanBench/.env -v <ROGUEONE_CAIR_PATH>/scripts/agent2027:/workspace/scripts/agent2027:ro -v ~/rogueone-synp-meta:/meta:ro -v ~/ro27_analysis/A11:/out --entrypoint python3 rogue_one:agent2027-v2.4 /workspace/scripts/agent2027/analysis/<script>.py <args>"
```
(Use the newest v2.N image present on CAIR if v2.4 was not built under ED4. `~/rogueone-synp-meta` holds φ for the recovery detector; it is mounted only into analysis containers, never into run containers, 01 §4.4.)

- [ ] **A11-T15** [type: analysis] (0.25 d, due Wed 28 Oct) — Run the free null-diagnosis checks on A05's 14 SYN-P 7 control runs (and, descriptively, on the 14 L0 runs):
  - `train_metric_saturation`: the share of completed trials with `train_accuracy` ≥ 0.99 (C) or `train_R2` ≥ 0.99 (R). Read it from `df_test_pool_final.csv` in each `run-output` (written at `main.py:1388-1394`; the columns come from `as_dict`, `tst/agent.py:400-424` (R, `train_R2` at `:422`) and `:530-554` (C, `train_accuracy` at `:552`)).
  - `feedback_read_share`: from the traces (definition at the end of this document).

  *Done when:* both are logged per task and pooled over the D3 population, in one `ANALYSIS` run.
- [ ] **A11-T16** [type: analysis] (0.25 d; preliminary due Wed 28 Oct, so leak-audit failures can still be re-queued by L-LAST; final due Sun 1 Nov, after DF) — Score recovery and audit for leaks.
  - Run `scripts/agent2027/analysis/fetch_wandb.py` (groups A11, A05; keep `code_phase=v2` only, which drops A11-000 and A11-015), then `analysis/recovery_detector.py` (`rec_sel`, `rec_any`, `iter_first_rec`, `tok_to_first_rec`).
  - **Leak audit**, on all 14 L0 traces (and A11-000):
    - the Scientist's system prompt and tool descriptions contain none of the A11-T7 phrases;
    - no Scientist tool result contains a `Status`, `Pruned Iteration` or `Prune Rationale` column, a `train_*` name followed by a number, or any 200-character substring of that run's `tester_reports/*.md`. A substring is **not** a hit if it also occurs in text the L0 Scientist may legitimately see: the Extractor's `append_new_attribute` arguments in the same trace (Descriptions and commands), the focus strings, the Scientist's own notes, the task's `context_and_goal`, or raw-data column names and values;
    - error strings that only name a missing column are allowed, and are counted as `l0_removed_column_errors`;
    - any Scientist `generic_pandas_tool` command that uses introspection (`gc`, `inspect`, `sys.modules`, `globals(`, `__`) is listed for manual review; it is a hit only if its result contains feedback content as defined above.
  - Read A06's sentinel verdict for `S1020a` (and `S1020b` under TP2).

  *Done when:*
  - a fold-episode table exists, with 140 rows (70 A11 + 70 A05 r0) plus 35 A/A rows;
  - the audit log lists 0 hits (or the hit runs are handled per §6);
  - the sentinel verdict (PASS/WARN/FLAG) is noted.
- [ ] **A11-T17** [type: analysis] (0.25 d, due Tue 3 Nov) — Fit the estimate of record (§6) with `analysis/glmm_holm.py`, labelled **exploratory**, with no Holm adjustment:
  - population fixed by D3, after the brute-force gate, ITT zeros included; units are fold-episodes of A11 `l0` and A05 `full` r0;
  - model: pooled population `rec_sel ~ arm + (1|task) + (1|task:seed)`; multi-step subset `rec_sel ~ arm + task + (1|task:seed)` (task fixed, 2–3 levels);
  - log: the raw likelihood-ratio p of `arm` (1 df, two-sided), the odds ratio L0/Full with its 95% profile CI, Δ`rec_sel` (L0 − Full) with a Newcombe 95% CI (descriptive), and which model was used;
  - separation fallback (either arm at 0% or 100% of the population's episodes, |log-OR| > 10, or non-convergence): exact conditional test stratified by task × seed (exact Cochran–Mantel–Haenszel) with the conditional-MLE odds ratio and its exact CI;
  - secondaries: `arm × multistep`, `rec_any`, `iter_first_rec`;
  - sensitivities (never change the verdict): an exact sign-flip permutation over the 14 (task, seed) run pairs of Σ(folds recovered, 0–5); fallback-only runs excluded; the per-task L0 − Full gap against the seed-42 A/A gap; on an A06 `drift_flag`, the model with and without the flagged session.

  *Done when:* everything is logged in one `ANALYSIS` run.
- [ ] **A11-T18** [type: analysis] (0.25 d, due Tue 3 Nov) — Run `analysis/process_metrics.py` and `analysis/funnel.py` per arm, with paired run-level differences and bootstrap CIs (descriptive). Cover:
  - `restore_pruned_count`, `reproposal_ratio`;
  - `focus_novelty` (repeat share at cosine ≥ 0.9), `focus_explore_share`;
  - `report_read_rate`, `feedback_read_share`, `l0_removed_column_errors`;
  - `fallback_share`, `prune_rate`, `fun_*`;
  - `tok_compl_scientist`, `llm_calls_scientist`, `tok_compl_total`, `wall_s`, `success`, `cap_hit`.

  Bootstrap: 2,000 resamples of the 14 (task, seed) run pairs, percentile 95% CI. *Done when:* the process table (per arm, paired difference, CI) is logged in an `ANALYSIS` run.
- [ ] **A11-T19** [type: decision] (0.1 d, due Tue 3 Nov) — Apply §6, and combine the result with A10's C3 verdict and G3. *Done when:* a one-paragraph verdict naming exactly one of R1 / R2 (with its null class) / R3, the signature reading, the sentinel status and the G3 status is in the analysis memo.

### 5.5 Paper outputs
- [ ] **A11-T20** [type: writing] (0.25 d, due Fri 6 Nov) — Produce:
  - the L0 point on Fig. 2, right panel (`rec_sel` against `tok_compl_total`);
  - an exploratory L0 row in Table 2: Δ`rec_sel` (Newcombe CI), OR (CI) and raw p marked "exploratory, not in Holm", Δ success, Δ tokens, the funnel stage changed, no-harm "n/a (SYN-P only)", the drift caveat if any, and the §6 verdict;
  - one C3 sentence on the Restarts → L0 → Full ladder;
  - an appendix with the 7-edit diff, the process table, the saturation and read-share checks, the residual channels, and the limitation that L1 is cut.

  Add the manifest, prompt dumps and traces to the replication package (Thu 12 Nov). *Done when:* the text is in the draft by Fri 6 Nov.

## 6. Decision rules and gates (pre-registered Fri 16 Oct; exploratory)

**Estimate of record** (A11-T17): the odds ratio L0/Full on `rec_sel` with its 95% profile CI and the raw two-sided LR p of `arm` (or, under the separation fallback, the exact conditional OR, its exact CI and exact p), on the D3 population after the brute-force gate, ITT zeros included. "CI excludes 0" below means this CI excludes OR = 1 (equivalently Δ`rec_sel` ≠ 0); Δ`rec_sel` with its Newcombe CI is reported beside it. No Holm adjustment (A11 is outside Holm, R14); every statement is labelled **exploratory**. Exactly one of R1–R3 applies.

- **R1, feedback used.** Estimate < 0 (OR < 1) and its 95% CI excludes 0 (raw p < 0.05). The H1 sentence may then be written, labelled exploratory.
  - "The loop learns from feedback" is written only if A10's C3 also passes Holm (Full > Restarts) **and** G3 passes.
- **R2, null.** The CI includes 0 (raw p ≥ 0.05). Report the CI and the G2 MDE, then classify the null using Full's (A05) checks from A11-T15, pooled over the D3 population:
  - **"unused":** `feedback_read_share` < 0.25 (*proposed*);
  - **"uninformative scalar":** `train_metric_saturation` ≥ 0.8 (*proposed*);
  - both labels can apply together; if neither applies, the null is **"inconclusive"**; if the traces needed for `feedback_read_share` are missing, that label is "not assessable".
- **R3, opposite.** Estimate > 0 (OR > 1) and its CI excludes 0. Report "feedback as delivered misleads the planner". It becomes the first T2 item, and Full-v2 is unchanged.
- **Signature (descriptive, never changes R1–R3).** "Re-proposal signature present" if the paired L0 − Full difference in `restore_pruned_count` **or** `reproposal_ratio` is > 0 with its bootstrap 95% CI (A11-T18) excluding 0; otherwise "absent". Focus repetition (`focus_novelty`) is reported only.
- **Sentinel (A06 §6).** If A06 reports FLAG for `S1020a`: the queue is stopped and diagnosed per A06 §6. Cause found and fixed → rerun the A11 rows from the reserve (planned by L-LAST Thu 29 Oct, launched no later than Fri 30 Oct, ending before DF). No cause and the repeat sentinel passes → keep the rows with `drift_flag = 1` and report the estimate with and without the flag. The repeat fails again (real drift) → report A11 as process metrics only. A11 rows are provisional until the v2.2 guard-3 verdict is logged.
- **G3 (Mon 26 Oct).** If A09's planner positive control fails, an R2 null is reported as "planner channel inert on these tasks at 120B", with process metrics only. R1 and R3 results stand either way.
- **Leak audit.** Any hit invalidates that run (an implementation fault, not ITT; see A11-T16). If ≤ 2 runs are hit, fix E9b, re-release as the next v2.N (Full byte-identical, guard A11-T8 repeated) and re-run those rows from the reserve by L-LAST. If more than 2 runs are hit, re-run all 14 rows from the reserve on the fixed image if they can launch by L-LAST (Thu 29 Oct); otherwise report A11 as process metrics only.
- **Kill and cut dates.**
  - If E9b misses v2.2, ship it in v2.3 (Wed 21 Oct, G0b-1) and launch after LF1 (A10 has queue priority).
  - **If A11 has not launched by Mon 26 Oct (*proposed*), apply ED6 (L0 → T2).** Launching later than that leaves no time for the leak-audit and sentinel rerun rules before L-LAST (Thu 29 Oct).
  - CL7 is applied at the G1 re-tier only if CL1–CL6 do not free enough budget.

## 7. Risks and cut levers

1. **An ambiguous null.** D2 keeps the in-sample fit. Mitigation: A11-T15, plus the disclosure that the L1 contrast is cut.
2. **Wording confound.** The edits change the prompt's length and wording, not only its information. Mitigation: local edits only, the E5 diff and the token logs.
3. **Leaks into the other agents** through the shared `AgentRoles`. Mitigation: the byte-identity check on the E and T prompts (A11-T8).
4. **Residual channels** (§2): Extractor auto-unprune and `Status`, Descriptions, fallbacks. Mitigation: report `restore_pruned_count` and `fallback_share` separately. L0 also hides `status`, `dataset_id`, `fold_seed`, `aborted_fold_id` and `skip_reason`, including abort reasons; this is disclosed.
5. **The control comes from another image and session** (v2.0, `S1016a`/`S1016b`); the contrast is not concurrently randomised and a session effect is aliased with arm. Mitigation: the byte-identical guard (A11-T8), the switch-off unit tests (A11-T7) and the A06 sentinel; disclosed.
6. **Low power.** P1 probably sits at the ceiling, which leaves about 6 informative tasks, so "inconclusive" is likely.
7. **Cut levers.**
   - CL7 / ED6 (−14 runs, −9.8 to −14u) remove the "from feedback" step of C3.
   - Pre-declared order inside the C3/C6 group: P7 already dropped → HC1 on 4 datasets (CL3) → L0 to T2.
   - LF1 is never cut.

## 8. Definition of done
- [ ] E9b merged and tagged `agent2027-v2.2`; Full-v2 guard (A11-015 vs `prompt-baseline-v2.0`) byte-identical; unit tests green; 7-edit diff archived; smoke rows A11-000 and A11-015 passed.
- [ ] 14 A11 rows with a final W&B state and a `run-output` artifact; leak audit 0 hits; failure and cap-hit rates recorded; A06 verdict for `S1020a` recorded.
- [ ] Saturation and read-share checks, the GLMM, the process table and the §6 verdict logged in W&B `ANALYSIS`.
- [ ] Fig. 2 point, Table 2 row, C3 sentence and appendix in the draft by Fri 6 Nov; artefacts in the replication package by Thu 12 Nov.

---
**New names introduced** (not in the canonical registry):
- **Analysis columns:**
  - `feedback_read_share`: the share of Full focus decisions at `iteration_in_fold` ≥ 1 with ≥ 1 Scientist tool call that touches `feature_assessment` or a `train_*` column, or calls `attribute_lookup_tool`;
  - `train_metric_saturation`: the share of completed trials with `train_accuracy` ≥ 0.99 (C) or `train_R2` ≥ 0.99 (R) in `df_test_pool_final.csv`;
  - `l0_removed_column_errors`: Scientist tool errors under `l0` whose message only names a removed column;
  - `focus_explore_share`: the share of focuses the Scientist labels exploratory (the label required by `sci/agent.py:432`).
- **Code:**
  - helpers `_scientist_view_test_results`, `_scientist_view_attributes` and `_apply_l0_prompt_edits`;
  - test file `tests/unit/agents/test_scientist_feedback_level.py`.
- **Run plumbing:**
  - smoke row `A11-000` (`l0`) and guard-smoke row `A11-015` (`full`, `notes=guard-smoke`; the shared v2.2 Full-v2 guard for A10 and A11);
  - smoke session `S1020s` (shared with A10-000; no sentinel); main session `S1020a` (as in A06 and A10), plus `S1020b` only if D5 = TP2;
  - manifest `manifests/agent2027/A11_LF2-L0.csv` (the R6 pattern applied to the short ID);
  - analysis output folder `~/ro27_analysis/A11` on CAIR.
