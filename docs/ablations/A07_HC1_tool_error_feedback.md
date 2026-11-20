# A07 · HC1 — Tool-error text shown vs hidden

| Field | Value |
|---|---|
| **ID** | A07 · HC1 · arm `errors_hidden` (control: A05 `full`) |
| **Claim / Holm contrast** | C6, "harness channels are part of the architecture". **Outside Holm** (R14): supporting evidence, raw p only. |
| **Tier** | T1-minimum |
| **Owner** | Henrik |
| **Start / finish** | Checks and smoke Mon 12 – Thu 15 Oct → runs **Fri 16 Oct, 18:00**, interleaved with A05 → data complete Sun 18 Oct (fallback Tue 20 Oct) → final statistics Tue 3 Nov → AGENT draft Fri 6 Nov |
| **Budget** | **12 runs, 12–15u**. At 1.4 u/h this is 8.6–10.7 h of the 4-GPU box, ≈ 34–43 H100-hours. The pre-freeze smoke (2 one-fold rows) costs ≈ 0.4–0.5u of otherwise idle time. |
| **Models** | O120 = `openai/gpt-oss-120b` for S, E, T and W (uniform), served by `vllm/vllm-openai:v0.22.0`. TP4 on port 8010, or two TP2 servers on 8010/8011 (decision D5, O120 TP2 vs TP4, due Fri 16 Oct). |
| **Status** | Not started |
| **Depends on** | E1, E2 (flag), E3 (counters, traces, tokens), E4, E5, X10; G0a; G1 freeze and image `rogue_one:agent2027-v2.0`; A05 (control rows in the same session); E11a (σ_d), E11b (analysis scripts) |
| **Feeds into** | The C6 paragraph, appendix table "HC1" and one row of Table 2 in the AGENT paper; the short paper cites it; A21 (descriptive iso-cost point) · A23 (cost point) |

*Every "hoped" result below is a hypothesis, not a finding.*

## 1. What we test and why it matters

When a tool call fails (bad pandas code, a missing column, a rejected feature), does it matter whether the agent is told why, or whether it just gets `None` back? Our earlier t17 runs hid this text by accident. A07 runs frozen Full-v2 with the text hidden on 6 real datasets × 2 fold seeds, and pairs each run with the matching A05 run where the text is shown. It is the direct test of the execution-feedback channel behind C6. This paper owns HC1; the short paper cites it and keeps the bug story.

## 2. Background you need

**Units and code.** u = one real-data O120 run (10 iterations × 5 folds); throughput is 2 u/h × 0.7 = 1.4 u/h on ≤ 4 H100. Code references are to RougeOne `nested-cv-rebuttal` @ `07cbf93` (all lines verified). Short paths: `sci/`, `ext/`, `tst/agent.py` = `rogueone/llm/agents/core/{scientist,extractor,tester}/agent.py`; `kn/web_search.py` = `rogueone/llm/agents/core/knowledge/web_search.py`; `console.py`, `config.py`, `pandas_exec.py` are in `rogueone/utils/`. Shared doc = `docs/ablations/01_shared_infrastructure.md` ("01 §n"); its §7 is the canonical registry.

**Task and gate IDs used here** (details in 01 §3 and §6):
- E1 = flag registry → `wandb.config`, manifest launcher (`make_manifest.py`, `ro27_session_launch.sh`, `ro27_clients_manifest.sh`); E2 = v2 fixes, item 1 of which is the HC1 flag; E3 = instrumentation (tokens per role, LLM traces, counters, `run-output` artifact); E4 = Serper cache; E5 = rendered-prompt dump and `prompt_diff.py`; E11a = G2 statistics (σ_d); E11b = analysis package; X10 = pre-registration commit (tag `agent2027-prereg`).
- G0a = Mon 12 Oct (engineering merged, smoke passes); G1 = Fri 16 Oct (freeze, pre-registration, M0 + HC1 launch 18:00); G2 = Mon 19 Oct (hard go/no-go); L-LAST = Thu 29 Oct; DF = Sat 31 Oct, 24:00.
- `v2_defaults.env` = `scripts/agent2027/v2_defaults.env`, the Full-v2 flag values (draft at tags `agent2027-pre.N`, final at G1). `env_overrides` in a manifest row are applied on top of it.

**Roles.** The Scientist (S) plans a focus. The Extractor (E) writes features with `generic_pandas_tool` and `append_new_attribute`, in R rounds per iteration (`ext/agent.py:967`, `range(10)` today). The Tester (T) scores, assesses and prunes. All three call the web-search summariser (W). A tool's return value is what the LLM reads on its next turn.

**How t17 hid errors.** t17 set `CONSOLE_DISABLE=True` (`paper/scripts/clients_t16_rerun.sh:108`; read at `config.py:300`). With that set, the `disabler()` wrapper (`console.py:31-42`) skips the function and returns `args[1]` if there are two positional arguments, otherwise `None` (`:37`). Every error site calls `ConsoleManager.console_error_print(message)` or `console_print(message)` on the class, with one positional argument, so it returned `None`. We expect openai-agents 0.9.1 to show this to the model as the text `None`; A07-T6 confirms it in a trace.

**Sites that return `None` under `none`, exactly as in t17** (E2 must route each through the flag):

| Site | Code | What fails |
|---|---|---|
| h1 | `sci/agent.py:115-131` `_report_tool_error` (callers `:170, :266, :299, :331, :363, :389`) | exception in any S tool |
| h2 | `sci/agent.py:153-156` | empty search result |
| h3 | `sci/agent.py:255-258` | S pandas command set no `result` |
| h4 | `ext/agent.py:207-223` `_report_tool_error` (callers `:276, :381, :562, :593`); includes the KeyError hint raised at `pandas_exec.py:144-155` | exception in any E tool |
| h5 | `ext/agent.py:265-268` | empty search result |
| h6 | `ext/agent.py:333-340`, returns `console_print`, not `console_error_print` | ILLEGAL COMMAND |
| h7 | `ext/agent.py:360-363` | E pandas command set no `result` |
| h8 | `ext/agent.py:500-504`, `:511-519`, `:522-536` | append rejected: no `result`, NaN/Inf, non-numeric |
| h9 | `tst/agent.py:1023-1039` `_report_tool_error` (callers `:1069, :1100, :1127, :1141, :1228, :1262, :1323`) | exception in any T tool |

This list is complete: at `07cbf93` these are the only `return ConsoleManager.…` statements in `rogueone/` and `main.py`, and no other `ConsoleManager` return value reaches an agent. h1, h4, h8 (first site) and h9 pass `code=` as a keyword, so `disabler()` saw one positional argument and returned `None`.

**Reachability of h2 and h5.** `WebSearchAgent.explain_query` (`kn/web_search.py:292-309`) returns the v4 fallback strings instead of an empty string, so h2/h5 fire only when the summariser returns a whitespace-only `final_output`. Expect zero or near-zero counts; A07-T6 accepts "confirmed zero".

**Visible in both modes** (plain strings; visible in t17 too):
- v1: the duplicate-name rejection, `ext/agent.py:486-489`;
- v2: the auto-unprune message, `ext/agent.py:485` (not an error);
- v3: the Tester "Pruning is blocked" guard, `tst/agent.py:1283-1289`. It is normally unreachable, because only the pruning stage gets the pruning tool (`tst/agent.py:1582-1588`), after the report flag is set (`:1506`);
- v4: the summariser's fallback strings (`kn/web_search.py:106, :181, :186, :265, :271`) and the Tester executor default "No output variable defined." (`tst/agent.py:1206`).

Runner exceptions (`sci/agent.py:530-545`, `ext/agent.py:977-992`, `tst/agent.py:1398-1415`) never reach an agent in either mode. They are logged as `tool_name="runner"`.

**What is recorded in both arms.** h1, h4 and h9 write a record to `output/agent_toolcall_errors.jsonl` (`ToolErrorReporter`, `rogueone/llm/agents/core/tool_error_reporting.py:47-118`) *before* they return, and increment `_tool_call_counter_failed`. h2, h3, h5–h8, v1 and v3 bypass the reporter, so E3 must count them (A07-T5). (v3 increments `_tool_call_counter_failed` at `tst/agent.py:1284` but writes no JSONL record.) The existing W&B keys `Extractor_tool_usage_failed/*` and `Extractor_tool_errors/total` (`ext/agent.py:1021-1038`) miss them. Console logging stays on in both arms (`CONSOLE_DISABLE=False`).

**Instruments (HC1-6).** Datasets are split by t17 `other_tool_errors` (`paper/t17_extractor_corruption_audit.csv`; the plan attributes it to seed 42, and the CSV's seed column is blank). These are **Extractor-only** error lines over one whole t17 run (5 folds × 10 iterations; the CSV's `tester_iterations` = 50); the CSV has no Scientist or Tester counts. t17 is v1, so this is descriptive only.

| dataset_id | Task | n | p | t17 Extractor errors (final attributes) | Load |
|---|---|---|---|---|---|
| `myocardial` | C | 686 | 91 | 595 (72) | high |
| `vehicle` | C | 846 | 18 | 274 (115) | high |
| `housing` | R | 20,640 | 9 | 196 (151) | high |
| `crab` | R | 3,893 | 8 | 168 (158) | high |
| `heart` | C | 918 | 11 | 66 (158) | low |
| `insurance` | R | 1,338 | 6 | 85 (167) | low |

**Metrics** (canonical R13 names; one value per run):
- **Tool error:** one event at h1–h9, v1 or v3, in any role. v2 (a restoration) and v4 (fallback strings that reach the agent as ordinary tool output), runner exceptions and context errors (HTTP 400, `finish_reason=length`; `context_errors`) are not tool errors.
- **Accepted feature:** a success return of `append_new_attribute` (`ext/agent.py:560`). The success counter also counts auto-unprune restorations (`:484` vs `:559`), so those are subtracted. Deterministic fallback features (`ext/agent.py:599-844`, triggered at `:996-1002`) never count.
- **`err_per_accepted` (primary)** = `n_tool_errors` ÷ `n_accepted`. Report both counts, all roles and Extractor only, and split by site.
- **`err_repeat_rate`:** share of errors whose role, tool, exception type and normalised message (digits and quoted names stripped) equal the previous error by the same role in the same fold-iteration.
- **`recover_within3`:** share of Extractor errors followed by an accepted append within the next 3 Extractor tool calls (from the trace).
- **`silent_rejections`:** h8 + v1 events. **`append_success`:** accepted ÷ append attempts.
- **Extractor tokens per accepted feature** = `tok_compl_extractor` ÷ `n_accepted`.
- **Context and cost:** `max_prompt_tokens`, `context_errors`, `tok_compl_total`, `tok_prompt_total`, `llm_calls_total`, `wall_s`, `fallback_share`, `success`, `cap_hit`.
- **No-harm:** `heldout_score` (balanced accuracy for C, nRMSE for R) and `delta_std` = Δ / σ_d, where σ_d is the dataset's Full-v2 run-to-run SD from A05.

## 3. Hypotheses and interpretation (all hypotheses)

- **Hoped.** Showing the text cuts errors per accepted feature by ≥ 30%: the ratio ρ = shown ÷ hidden is ≤ 0.70. The effect is larger on the high-load datasets, with fewer repeats, more recoveries within 3 calls and fewer Extractor tokens per accepted feature. → *"Hiding tool-error text, as our earlier runs did by accident, raised errors per accepted feature by X% (95% CI …; 12 paired runs on 6 datasets)."*
- **Null.** Errors are rare after the v2 fixes, or the CI for ρ includes 1. → *"After the v2 fixes, the execution-feedback channel was rarely exercised / had no detectable effect at 120B."* Report the counts. C6 then rests on A02 (M4). The channel may matter more at small rungs, but that is not tested here and the paper must not claim it.
- **Opposite.** ρ > 1, or the text inflates context (more `context_errors`, higher `max_prompt_tokens`): the agents chase tracebacks. → Report it as a finding. Context inflation is descriptive only and does not enter the §6 verdict. **The frozen default stays `message` in every case.** HC1 never reopens the freeze.
- **Confound to state in every outcome.** Hiding the text can change how many features are accepted, so ρ is a ratio of ratios. The numerator and the denominator are always reported per arm, so a "fewer errors" result that comes only from fewer attempts is visible.

## 4. Design

**Control.** A05 arm `full`, `replicate=0`, same dataset and fold seed, same session and image. Full-v2 has `ROGUE_ONE_TOOL_ERROR_FEEDBACK=message`.
**Treatment.** Arm `errors_hidden`: `env_overrides = {"ROGUE_ONE_TOOL_ERROR_FEEDBACK":"none"}`, nothing else changed.

**Fixed settings** (`scripts/agent2027/v2_defaults.env` at tag `agent2027-v2.0-freeze`, identical in both arms; every other 01 §7 R3/R4 flag at its Full-v2 value):
- `ROGUE_ONE_NUM_ITERATIONS=10` (code default 3, `config.py:305`, so always set) and `ROGUE_ONE_K_FOLDS=5`;
- `ROGUE_ONE_SELECTION_SIGNAL=inner_cv`, `ROGUE_ONE_ENABLE_ROLLBACK=False`, `ROGUE_ONE_DO_BALANCING=True`, `ROGUE_ONE_STRATIFIED_FOLDS=False` (real data), `ROGUE_ONE_FOLD_RETRY_LIMIT=1`;
- temperature 0.7 for all roles and W; max tokens E 32768, T 8192, S as pinned at G1;
- `ROGUE_ONE_EXTRACTOR_ROUNDS`, `ROGUE_ONE_FEATURE_POOL_MODE` and `ROGUE_ONE_TOOL_OUTPUT_MAX_CHARS` as frozen at G1;
- `ROGUE_ONE_SEARCH_CACHE=shared`, `ROGUE_ONE_LLM_TRACE=1`, `ROGUE_ONE_PROMPT_DUMP=1`, `ROGUE_ONE_REPLAY_VALIDITY_CHECK=1`, `CONSOLE_DISABLE=False`;
- XGBoost 400 trees (code, 01 §4.3).

**Launcher-level keys** (not in `v2_defaults.env`; set on the launch command): `PARALLEL=6`, `CPUS=12`, `RUN_CAP_HOURS=6`, `RUNROOT=~/rogueone-runs-agent2027`.

**Seeds.** Fold seeds {42, 4}; `llm_seed = fold_seed` (`replicate=0`), the same as the paired A05 row.

**Run matrix.** Rows are numbered by dataset in HC1-6 order, seed 42 then 4 (A07-001 = `myocardial` f42, A07-002 = `myocardial` f4, …, A07-012 = `insurance` f4). Run name: `A07__errors_hidden__<dataset_id>__f<fold_seed>__r0`. W&B: `PlanBench/RogueOne-AGENT2027`, group `A07`, job_type `errors_hidden`, tags [`S1016a`, `v2`, `agent2027-v2.0`] (`S1016b` for rows placed there under TP2).

| Group | Rows | Datasets | fold_seed = llm_seed | Runs | u | Control run |
|---|---|---|---|---|---|---|
| high, f42 | 001, 003, 005, 007 | myocardial, vehicle, housing, crab | 42 | 4 | 4–5 | `A05__full__<dataset_id>__f42__r0` |
| high, f4 | 002, 004, 006, 008 | same | 4 | 4 | 4–5 | `A05__full__<dataset_id>__f4__r0` |
| low, f42 | 009, 011 | heart, insurance | 42 | 2 | 2–2.5 | `…__f42__r0` |
| low, f4 | 010, 012 | heart, insurance | 4 | 2 | 2–2.5 | `…__f4__r0` |
| **Total** | | | | **12** | **12–15** | 12 A05 rows (0 new) |
| smoke (not analysed) | 901 (`errors_hidden`), 902 (`full`) | myocardial | 42 | 2 one-fold | ≈ 0.4–0.5 | each other |

Manifest columns for rows 001–012 (R7 header order: `row_id, ablation_id, arm_id, dataset_id, task_type, fold_seed, llm_seed, replicate, is_sentinel, session_id, order_key, image_tag, rung_S, rung_E, rung_T, port_S, port_E, port_T, env_overrides, cap_hours, expected_u, notes`):
- `row_id=A07-001…A07-012`, `ablation_id=A07`, `arm_id=errors_hidden`;
- `dataset_id` and `fold_seed` as in the table; `llm_seed=fold_seed`; `task_type=classification` for `myocardial`, `vehicle`, `heart` and `regression` for `housing`, `crab`, `insurance`;
- `replicate=0`, `is_sentinel=0`;
- `session_id` and `order_key` **empty** in `A07_HC1.csv`; `make_manifest.py --plan-session` fills them (`S1016a`, or `S1016a`/`S1016b` under TP2) in the session file;
- `image_tag=agent2027-v2.0`, `rung_S/E/T=O120`, `port_S/E/T=8010` (8011 for rows placed in `S1016b`);
- `env_overrides={"ROGUE_ONE_TOOL_ERROR_FEEDBACK":"none"}`, `cap_hours=6`, `expected_u=1.0`;
- `notes="pair=A05__full__<dataset_id>__f<fold_seed>__r0; load=<high|low>"`.

Smoke rows (not analysed): `A07-901` (`arm_id=errors_hidden`, `env_overrides={"ROGUE_ONE_TOOL_ERROR_FEEDBACK":"none","ROGUE_ONE_SMOKE_FOLDS":"1"}`) and `A07-902` (`arm_id=full`, `env_overrides={"ROGUE_ONE_SMOKE_FOLDS":"1"}`). Both: `myocardial`, `task_type=classification`, `fold_seed=llm_seed=42`, `replicate=0`, image `agent2027-pre.N` (overrides relative to the draft `v2_defaults.env` at that tag), O120 on port 8010, `cap_hours=6`, `expected_u=0.2`, `notes=smoke`, code phase `smoke`, project `RogueOne-AGENT2027-prefreeze`. Their R12 run names (`A07__errors_hidden__myocardial__f42__r0`, `A07__full__myocardial__f42__r0`) live only in the pre-freeze project, so they never collide with row 001.

**Budget arithmetic.**
- 12 runs × 1.0–1.25u = **12–15u**, as in the T1-minimum card (`docs/AGENT2027_T1min_breakdown.md`, HC1). The master plan prices HC1 at 1u per run (3 seeds, 18u); the +25% high end is the audited group estimate (more turns in the hidden arm on wide, error-heavy tables), and T1-minimum drops the third seed.
- 12–15u ÷ 1.4 u/h = 8.6–10.7 h; × 4 GPUs ≈ 34–43 H100-hours.
- Shared session: A05 47.5–55u + A07 12–15u = 59.5–70u; ÷ 1.4 u/h = 42.5–50 h, so a Fri 18:00 launch ends Sun 12:30–20:00 (the T1-minimum GPU plan rounds this to 43–50 h).
- CL3: 8 runs, 8–10u (−4 runs, −4 to −5u); session 55.5–65u, 39.6–46.4 h.
- Smoke 901/902: 2 × 1 fold = 2 × 0.2u = 0.4u (0.5u at the 1.25 factor), paid from pre-freeze capacity that is idle by construction (T1-minimum: 70–90u idle), so it is outside the 12–15u.
- Crash retries come out of the shared 15% rerun reserve.

**Pairing and randomisation.** A07 rows join A05's session `S1016a`. `make_manifest.py` draws `order_key` from an RNG seeded by the session ID, so both arms share server, load and Serper cache. If D5 = TP2 (sessions `S1016a` on 8010 and `S1016b` on 8011), each A07 row goes into the session that holds its A05 pair. (A05-T5 says the TP2 split is "alternate"; for the 12 A05 control rows and their A07 partners the split must be pair-aware. A07-T10 checks this.) The analysis unit is the 12 (dataset, seed) pairs. A05's seed-42 A/A runs (`replicate=1`) on the same 6 datasets give the noise reference for the paired log-ratio.

**Failures (ITT).**
- A crash (rc ≠ 0, or no `fold_aggregation_summary.csv`) is retried once, alone, at the end of the session. A 6 h cap hit is stopped, logged `CAP_HIT` and never retried. Persistent failures and cap hits are ITT failures, never dropped; held-out score is imputed at `base_no_fe` if augment is frozen, otherwise at the majority-class or mean predictor.
- Infrastructure faults (vLLM death, host reboot, W&B or Serper outage, visible in the logs) are not ITT failures. Those rows are re-queued with `notes=infra-rerun` in A05's infra re-queue session (`S1018a`, after a fresh pre-flight), and the cross-session pair is flagged.
- Process metrics need complete counters. A pair where either run lacks a complete `run-output` artifact (typically a persistent crash or a cap hit, which is stopped before upload) is excluded from the log-ratio model and counted per arm; those runs still enter the §6 harm flag. If fewer than 8 complete pairs remain (6 under CL3), ρ is reported descriptively only (proposal, confirmed in X10). Failure and cap-hit rates per arm are reported as outcomes.

## 5. Tasks

### 5.1 Prerequisites (verify before launch)

- [ ] **A07-T1** [type: code] (0.1 d, due Mon 12 Oct, G0a) — Confirm E2 item 1 is merged on `agent2027/v2`: `ROGUE_ONE_TOOL_ERROR_FEEDBACK=message|none` is registered (E1), `disabler()` no longer touches the return values of `console_error_print` **or** `console_print`, and h1–h9 go through the flag. *Done when:* A07-T4 passes, and the G0a smoke shows `flags/ROGUE_ONE_TOOL_ERROR_FEEDBACK=message` in `wandb.config`.
- [ ] **A07-T2** [type: code] (0.1 d, due Mon 12 Oct, G0a) — Confirm E3: tokens per role, `output/llm_traces.jsonl.gz`, counters for every bypass site, and the `run-output` artifact. *Done when:* in the G0a smoke, `tok_compl_scientist/_extractor/_tester` are non-zero, trace records = `llm_calls_total`, every counter exists (zero allowed), and `run-output` downloads.
- [ ] **A07-T3** [type: config] (0.05 d, due Fri 16 Oct, 15:00) — Confirm E1 (launcher applies `env_overrides`, retries once alone, enforces the cap), E4 (cache in v2.0), E5 (prompt dump) and X10 (contains the A07-T7 block). Also confirm `.dev.env` is untracked on `agent2027/v2`: it is tracked at `07cbf93`, `config.py:24` loads it with `override=True`, and the build step's `test ! -e .dev.env` stops the build if it is present. *Done when:* the tag `agent2027-prereg` is pushed before 18:00; `git ls-files .dev.env` on `agent2027/v2` prints nothing; and each item is ticked in the session plan.

### 5.2 Implementation (inside E2/E3; no new arm code)

- [ ] **A07-T4** [type: code] (0.25 d, inside E2, due Mon 12 Oct) — Write `tests/unit/agents/test_tool_error_feedback.py`. Trigger every site in §2 in both modes (monkeypatch the flag; the existing `tests/unit/agents/test_{scientist,extractor,tester}.py` show how to build the agents). *Done when:* under `none` exactly h1–h9 return `None`; v1–v4 return the same string in both modes; under `message` h1–h9 return the message text; and flipping `CONSOLE_DISABLE` changes no return value.
- [ ] **A07-T5** [type: code] (0.25 d, inside E3, due Mon 12 Oct) — Make the bypass sites countable. At h2, h3, h5–h8, v1 and v3, call `self._tool_error_reporter.report(...)` (`tool_error_reporting.py:81-118`) with a `ValueError(<message>)` and `context={"site": "<h2…v3>"}`; tag h1/h4/h9 records with their site too. The agent-facing return must not change. *Done when:* a unit test shows exactly one JSONL record, with the right `context.site`, for each forced event: ILLEGAL (h6), S and E no-`result` (h3, h7), NaN append, non-numeric append and append without `result` (h8), duplicate name (v1), empty search (h2, h5) and pruning-blocked (v3); and the A07-T4 return-value assertions still pass unchanged after this change.
- [ ] **A07-T6** [type: run] (0.1 d dev + ≈ 0.4–0.5u; due Thu 15 Oct) — Add rows A07-901/902 to the first pre-freeze O120 session on image `agent2027-pre.N`: the G0a smoke session `S1012a` on Mon 12 Oct (the ID A02-T6 fixes), else A02's `S1013a` (Tue 13), else the end of A03's Core-10 session `S1014b` (Wed 14; A03-T8 and A02-T7 allow the append). `S1014a` is reserved for A02's contingency and `S1014e` is A03's E6 smoke session; do not use either. Plan them with `make_manifest.py --plan-session <SESSION_ID> --rows … A07-901 A07-902`, commit and push, and launch with that session's launch task after its GPU pre-flight (§5.3 rules; `WANDB_PROJECT_NAME=RogueOne-AGENT2027-prefreeze`). *Done when:*
  - in the 901 trace, the tool output at every h-site event is the text `None` (this confirms the SDK rendering); in 902 it is the message. If no h-site event occurs in 901, this is recorded as "rendering unconfirmed" and checked instead on the first A07 v2.0 trace (A07-T12);
  - v1 events, if any, show the same text in both;
  - `scripts/agent2027/prompt_diff.py` finds the iteration-1 rendered prompts of all roles byte-identical;
  - each bypass counter fires at least once or is confirmed zero by a grep of the trace.
- [ ] **A07-T7** [type: writing] (0.1 d, draft Thu 15 Oct, committed in X10 Fri 16 Oct) — Write the HC1 block of `docs/agent2027/preregistration.md`: the §2 site list and metric definitions, the §4 complete-pair rule, the A07-T16 model, CI rule and permutation test, the §6 table with its order and thresholds, and the CL3 rule. *Done when:* the block is inside the `agent2027-prereg` commit.
- [ ] **A07-T8** [type: config] (0.05 d; rows 901/902 due Mon 12 Oct, before A07-T6; full file due Thu 15 Oct) — Run `scripts/agent2027/make_manifest.py --ablation A07`, which writes `manifests/agent2027/A07_HC1.csv` (rows 001–012, 901, 902; §4). Commit and push. *Done when:* the CSV has the R7 header in R7 order; `env_overrides` parses as JSON; `session_id` and `order_key` are empty; and every `pair=` run name resolves (from `arm_id`, `dataset_id`, `fold_seed`, `replicate`) to exactly one row of `manifests/agent2027/A05_M0.csv`.

### 5.3 Execution on CAIR (shared session with A05)

Hard rules (01 §5.1; they apply to every command below):
1. **Containers only.** Every execution runs in Docker, including W&B queries and analysis. The CAIR host only issues `docker` / `docker compose` (with `-T` on every `docker compose run` over SSH), plus the bash launchers in `~` that issue them; `nvidia-smi`, `docker ps`, `tail` and `grep` are allowed for pre-flight and monitoring. Never `python …` on the host.
2. **GPU pre-flight** (memory and utilisation per GPU, compute apps, running containers) before launching any GPU container. If not enough GPUs are free, STOP and report the output; do not queue. Launch with explicit `GPUS=<ids>` and `WAIT_MAX_MIN=0`.
3. **≤ 4 of the 8 H100s**, ever, without Henrik's explicit approval in chat. One Rogue One job at a time.
4. **W&B is the system of record** (entity `PlanBench`): runs are stateless, each uploads `run-output`; `WANDB_API_KEY` comes from `<CAIR_HOME>/code/PlanBench/.env`, `SERPER_API_KEY` from the image `.env` / `.secret.env`. Never push these images.
5. **Code path:** commit and push `agent2027/v2` (and tags) locally, then `git fetch` over SSH on the server. SSH is one-shot and non-interactive: `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "…"`.
6. **Long jobs** run detached under `nohup`, with the job log in `~` (`~/ro27_<SESSION_ID>.job.log`), never in the root-owned run folders.

`<ROGUEONE_CAIR_PATH>` = `<CAIR_HOME>/code/RougeOne` (**INFERRED, verify** with `ssh CAIR-gpu14 "ls -d ~/code/RougeOne"`). The server git remote `git@github.com:HenrikNorkart/RougeOne.git` is also inferred (01 R8).

- [ ] **A07-T9** [type: decision] (0.05 d, Fri 16 Oct, G1) — Apply the re-tier. If the mean u per SYN-P run of M4's `R10` arm is > 0.85u, or R=3 is not adopted, CL1–CL4 apply at once (canonical re-tier rule, 01 §4.9); for A07, CL3 drops rows 009–012. *Done when:* the decision (CL3 yes/no), the measured u that triggered it, and the A07 row list are in the session plan.
- [ ] **A07-T10** [type: config] (0.05 d, Fri 16 Oct, by 15:00) — The session file is written by A05-T5: `make_manifest.py --plan-session S1016a --rows A05-001 … A05-055 A07-001 … A07-012` (67 rows; 63 under CL3; split into `S1016a`/`S1016b` if D5 = TP2, keeping each A07 row with its A05 pair). Check its A07 rows, then commit and push. *Done when:* the session file(s) hold all 12 (or 8) A07 rows with `env_overrides={"ROGUE_ONE_TOOL_ERROR_FEEDBACK":"none"}` and `session_id` filled; `order_key` is a permutation within each session; and under TP2 each A07 row sits in the same session file as the A05 row named in its `pair=` note.
- [ ] **A07-T11** [type: run] (0.1 d, Fri 16 Oct, 16:00–18:00; run once for the shared session, normally by A05's launch task A05-T9) — Sync, build, install, pre-flight and launch:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15' git fetch git@github.com:HenrikNorkart/RougeOne.git agent2027/v2 && git checkout -B agent2027/v2 FETCH_HEAD && test ! -e .dev.env && docker build -t rogue_one:agent2027-v2.0 . && docker image inspect rogue_one:agent2027-v2.0 --format '{{.Id}}' && cp scripts/agent2027/ro27_session_launch.sh scripts/agent2027/ro27_clients_manifest.sh ~/ && cp paper/scripts/vllm_session.sh ~/ro27_vllm_session.sh && mkdir -p ~/ro27_manifests && cp -r manifests/agent2027/. ~/ro27_manifests/"
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader && echo '--- compute apps ---' && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader && echo '--- containers ---' && docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'"
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "SESSION_ID=S1016a SESSION_FILE=~/ro27_manifests/sessions/S1016a.csv GPUS=<ids> MODEL=openai/gpt-oss-120b PORT=8010 SERVER_NAME=rogueone-vllm-o120-s1016a LOG=~/vllm_session_S1016a.log IMAGE_RO=rogue_one:agent2027-v2.0 PARALLEL=6 CPUS=12 RUN_CAP_HOURS=6 RUNROOT=~/rogueone-runs-agent2027 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_S1016a.job.log 2>&1 & echo launched; sleep 5; tail -5 ~/ro27_S1016a.job.log"
  ```
  Launch only if the chosen GPUs each use < 2,000 MiB, the total is ≤ 4, and no `ro27-*`, `rogueone-vllm-*`, `llmfe-*` or `caafe-*` container is running; otherwise STOP and report the output. For TP2, set `GPUS=<2 ids>` on the first job and start a second job (`S1016b`, `SESSION_FILE=~/ro27_manifests/sessions/S1016b.csv`, `PORT=8011`, `SERVER_NAME=rogueone-vllm-o120-s1016b`, `LOG=~/vllm_session_S1016b.log`, job log `~/ro27_S1016b.job.log`, `ROGUEONE_ALLOW_SECOND_SERVER=1`) on 2 other free GPUs, so ΣGPU = 4. *Done when:* the image ID is recorded in the session plan, and the job log shows the vLLM version assert and the first rows starting.
- [ ] **A07-T12** [type: run] (0.05 d, Fri 16 Oct, once the first A07 row is live) — Check arm integrity:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker run --rm --env-file <CAIR_HOME>/code/PlanBench/.env --entrypoint python3 rogue_one:agent2027-v2.0 -c \"import wandb; api=wandb.Api(); [print(r.name, r.state, r.config.get('flags/ROGUE_ONE_TOOL_ERROR_FEEDBACK'), r.config.get('image_tag')) for g in ('A05','A07') for r in api.runs('PlanBench/RogueOne-AGENT2027', filters={'group': g})]\""
  ```
  *Done when:* A07 runs show `none` and A05 runs `message`, all with `agent2027-v2.0`. If not, `docker stop` the A07 containers, fix the manifest, and re-queue those rows. If A07-T6 left the SDK rendering unconfirmed, also confirm here from the first A07 `run-output` trace that an h-site tool output reads `None`.
- [ ] **A07-T13** [type: run] (0.1 d in total, Sat 17 – Sun 18 Oct, twice a day) — Monitor in the background: `ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 CAIR-gpu14 "while ! grep -q 'RO27_SESSION_END' ~/ro27_S1016a.job.log 2>/dev/null; do sleep 120; done; tail -30 ~/ro27_S1016a.job.log"` (exit 255 = network drop; reconnect). Under TP2, run the same loop for `~/ro27_S1016b.job.log`. A W&B `running` run with no container is a zombie. *Done when:* `=== RO27_SESSION_END S1016a rc=… ===` (and the `S1016b` marker under TP2) is logged, solo retries are done, and `docker ps --format '{{.Names}}' | grep -E '^(ro27-|rogueone-vllm-)' || echo none` prints `none`.
- [ ] **A07-T14** [type: decision] (0.05 d to decide, + 0.1 d if it fires; decide **Sat 17 Oct, 06:00**) — Fallback. If `S1016a` has not launched by then, remove the A07 rows so A05 still finishes before G2 (A05 alone is 34–39 h, so a Sat 06:00 start ends Sun 16:00–21:00; with A07 it would end Mon 00:30–08:00, too close to the G2 read-out). Run them in the first O120 session after G2 (planned `S1019a`, image `agent2027-v2.1`, Full byte-identical by the G0b-style prompt guard), which carries its A06 `sentinel` row; disclose the cross-session pairing. *Done when:* all 12 (or 8) rows are `OK`, `CAP_HIT` or ITT-classified by Tue 20 Oct, and the A06 sentinel for that session is logged.

### 5.4 Analysis

- [ ] **A07-T15** [type: analysis] (0.1 d; prelim. Tue 27 Oct; final Sun 1 – Tue 3 Nov) — Run `scripts/agent2027/analysis/fetch_wandb.py` and `process_metrics.py` (E11b) on group A07 and the 12 paired A05 rows. *Done when:* the per-run table holds every §2 metric with the per-site split, it is logged as `ANALYSIS__process_metrics__<YYYYMMDD>`, and a hand count of one pair's JSONL matches.
- [ ] **A07-T16** [type: analysis] (0.25 d inside E11b; final Tue 3 Nov) — Fit the HC1 model in `scripts/agent2027/analysis/glmm_holm.py`, flagged **outside Holm**:
  - per pair (d, s): y = log[(e_hid + 0.5)/(a_hid + 0.5)] − log[(e_shw + 0.5)/(a_shw + 0.5)], with e = `n_tool_errors`, a = `n_accepted`;
  - y_ds = μ + u_d + ε_ds, random dataset intercept, REML; report ρ̂ = exp(−μ̂) and the hidden-arm excess exp(μ̂) − 1;
  - **CI rule (pre-registered, unambiguous):** 95% CI for μ = ȳ ± t(0.975; D−1) · SD(ȳ_d)/√D, where ȳ_d is the mean of y within dataset d and D = number of datasets with ≥ 1 complete pair (6, t(5); 4, t(3), under CL3). With 2 complete pairs per dataset this equals the REML random-intercept CI whenever σ̂_u² > 0; when REML truncates σ̂_u² at 0, or pairs are missing, the two differ and **this rule is the one used**. The CI for ρ is [exp(−μ_upper), exp(−μ_lower)]. A dataset with only one complete pair enters with that pair's y;
  - robustness: exact sign-flip permutation over the n complete pairs (2^n patterns: 4,096 at n = 12, 256 at n = 8), statistic = mean y, two-sided p = share of patterns with |mean| ≥ the observed |mean|;
  - the same model, descriptive, for `err_repeat_rate`, `recover_within3`, `append_success`, Extractor tokens per accepted feature, `max_prompt_tokens`, `context_errors`;
  - descriptive only: per-dataset ratios (forest plot), high vs low load, the Δ_std CI (no test), and the A/A log-ratio SD from A05 f42 r0 vs r1.

  *Done when:* logged as `ANALYSIS__glmm_holm__<YYYYMMDD>`, and, on a fixed 12-value toy vector of y (6 datasets × 2 seeds) committed with the script, its μ̂, CI and permutation p equal an independent hand calculation to 1e-6.
- [ ] **A07-T17** [type: decision] (0.05 d, Tue 3 Nov) — Apply §6 in its stated order. *Done when:* the analysis memo records the §6 row that applied, the values that decided it (median control errors, ρ̂, CI, permutation p, number of complete pairs, ITT/cap-hit counts per arm) and the paper sentence.

### 5.5 Paper outputs

- [ ] **A07-T18** [type: writing] (0.25 d, Sun 1 – Tue 3 Nov) — Appendix table "HC1": per pair, errors, accepted and `err_per_accepted` for both arms; pooled ρ with CI; permutation p; secondary metrics; cost; ITT counts; Δ_std CI. Add the HC1 row of Table 2. *Done when:* both are generated by a script from the logged ANALYSIS runs (no hand-typed numbers) and are in the AGENT draft.
- [ ] **A07-T19** [type: writing] (0.1 d, Wed 4 – Fri 6 Nov) — Write 1–2 sentences in the C6 paragraph (§3 wording), labelled "outside the confirmatory family". t17 is historical context only. *Done when:* the AGENT draft contains the sentence of the §6 row recorded in A07-T17, with ρ̂, its 95% CI, the number of pairs and the "outside the confirmatory family" label.
- [ ] **A07-T20** [type: writing] (0.05 d, Sat 7 – Mon 9 Nov) — Check that the short paper cites HC1 and does not present its numbers as primary. *Done when:* every HC1 mention in the short-paper source (search "HC1", "error text", "tool-error") cites the AGENT paper, and none of them appears in its abstract, contribution list or main results table.

## 6. Decision rules and gates (pre-registered in X10)

ρ̂, its 95% CI and the permutation p come from A07-T16 (CI rule there). Rows 1–6 are checked **in order and the first match applies**; row 7 is always evaluated and reported alongside.

| # | Outcome | Rule | Report |
|---|---|---|---|
| 1 | **Too few pairs** | fewer than 8 complete pairs (6 under CL3), §4 | ρ̂ and per-pair values descriptive only; no verdict |
| 2 | **Errors rare** | median all-role `n_tool_errors` over the A05 control runs of the complete pairs < 20 per run. Reference only: t17's lowest Extractor-only count on HC1-6 was 66 (heart) | "channel rarely exercised in v2"; ρ descriptive only |
| 3 | **Supported** | ρ̂ ≤ 0.70, CI upper bound < 1, and permutation p < 0.05 | the hoped sentence |
| 4 | **Smaller or less robust effect** | CI upper bound < 1, but row 3 fails (ρ̂ > 0.70, or permutation p ≥ 0.05) | "shown text reduces errors" + "by less than 30%" if ρ̂ > 0.70 and/or "not robust to the permutation test" if p ≥ 0.05 |
| 5 | **Opposite** | CI lower bound > 1 | "error text increased errors"; default unchanged |
| 6 | **Inconclusive** | otherwise (CI contains 1) | "no detectable effect at N pairs" |
| 7 | **Harm flag** (descriptive) | (ITT failures + cap hits in `errors_hidden`) − (same in the paired A05 controls) ≥ 2 | stated next to ρ |

Gates and dates:
- **G0a (Mon 12 Oct):** A07-T1, T2, T4, T5 done; smoke rows 901/902 in the manifest (A07-T8).
- **Thu 15 Oct:** A07-T6 smoke passed; A07-T8 manifest committed; A07-T7 drafted.
- **G1 (Fri 16 Oct):** A07-T7 inside the `agent2027-prereg` commit; CL3 re-tier; launch at 18:00.
- **Sat 17 Oct, 06:00:** fallback trigger (A07-T14).
- **G2 (Mon 19 Oct):** if a Full-v2 defect is found, fix it and rerun A05 + A07 (≈ 60–70u). This fits only at G2.
- **L-LAST (Thu 29 Oct):** if HC1 has still not run, drop it; C6 rests on A02.
- **DF (Sat 31 Oct, 24:00):** data freeze.

## 7. Risks and cut levers

- **CL3** (third lever in the pre-declared order): only the 4 high-load datasets, 8 runs, 8–10u (−4 runs, −4 to −5u). The load moderator is lost.
- **Fidelity drift from t17** (v1 hidden by mistake, or ILLEGAL's `console_print` path missed). Guarded by A07-T4 and A07-T6.
- **Undercounted numerator** (bypass sites missed) biases ρ toward 1. Guarded by A07-T2 and A07-T5.
- **Floor effect.** After the v2 fixes, errors may be rare at 120B, especially on heart and insurance. The "errors rare" rule handles this.
- **Unstable ratio** when few features are accepted. Numerator and denominator are reported; +0.5 avoids zeros.
- **Cap hits** in the hidden arm on myocardial (91 columns). They are ITT failures and add wall clock; `fallback_share` and `cap_hit` are reported.
- **Low power.** 12 pairs; with 6 datasets, the smallest dataset-level sign-test p is 0.031, so dataset-level claims are descriptive.
- **Session slip.** A07-T14 moves HC1 to Mon 19 – Tue 20 Oct. HC1 must never delay G2. Knock-on: the 12–15u (8.6–10.7 h) join A09's `S1019a`, which is not planned for them in A09; that session then ends about 9–11 h later and A10's Tue 20 session (`S1020a`) starts later by the same amount.

## 8. Definition of done

- [ ] A07-T4 and A07-T5 tests are green; the A07-T6 smoke shows `None` vs text with byte-identical prompts.
- [ ] The HC1 block is in the `agent2027-prereg` commit; `manifests/agent2027/A07_HC1.csv` is committed.
- [ ] 12 rows (8 under CL3) are `finished` or ITT-classified in `PlanBench/RogueOne-AGENT2027`, group `A07`, each with a `run-output` artifact.
- [ ] The `process_metrics` and `glmm_holm` ANALYSIS runs are logged and the §6 verdict is recorded.
- [ ] The appendix table, the Table 2 row and the C6 sentence are in the AGENT draft; the short-paper citation is checked.

---

**New names introduced** (not in the 01 §7 registry):
- Analysis columns `n_tool_errors` and `n_accepted` (numerator and denominator of `err_per_accepted`), and "Extractor tokens per accepted feature" (`tok_compl_extractor` ÷ `n_accepted`).
- Site labels h1–h9 and v1–v4, stored as `context.site` in `agent_toolcall_errors.jsonl` (A07-T5).
- Test file `tests/unit/agents/test_tool_error_feedback.py`.
- Smoke rows A07-901/902; arm `full` appears under `ablation_id=A07` only in row 902.
- `notes` key `pair=<A05 run name>`.
- Threshold "errors rare" = median < 20 all-role tool errors per control run, and the "too few pairs" floor (< 8 complete pairs; < 6 under CL3): proposals to confirm in X10.
- Smoke host session for A07-T6: `S1012a` (fixed by A02-T6), else `S1013a` (A02-T7), else `S1014b` (A03-T8). Smoke-row IDs are `A07-901/902` in every document.
