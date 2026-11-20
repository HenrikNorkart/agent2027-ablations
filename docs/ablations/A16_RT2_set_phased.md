# A16 — RT2: SET-phased single shared context vs three agents (C1a)

| Field | Value |
|---|---|
| **ID** | A16 · RT2 · arm `set_phased` |
| **Claim / Holm contrast** | C1 "decomposition is not just more compute". Supplies **C1a = SET-phased − Full** on `rec_sel`. It is two-sided and sits in the 10-contrast Holm family (smallest threshold 0.005). C1b comes from A08. |
| **Tier** | T1-minimum |
| **Owner** | Henrik |
| **Start / finish** | Spec text into the pre-registration Fri 16 Oct · prompt-module text (A16-T6) may start Sat 17 · build, release guard and image v2.4 Thu 22 Oct · smoke + pilot session Thu 22, night (launch by Fri 23, 12:00 to keep a re-pilot possible; 18:00 absolute) · **K-SET Fri 23 Oct** · SYN-P runs Fri 23 – Sun 25 (session `S1023a`, with A12 block 2) · Core-10 runs Mon 26 – Thu 29 (L-LAST) · reruns until DF (Sat 31, 24:00) · analysis Sun 1 – Tue 3 Nov · text Wed 4 – Fri 6 Nov |
| **Budget** | 26 new runs · **21.8–26u** (the T1-minimum figure). At 2 u/h that is 10.9–13 h of the 4-H100 slot (43.6–52 H100-h when the session holds 4 GPUs), or 15.6–18.6 calendar hours at 0.7 utilisation (1.4 u/h). **Outside the 26 runs**, paid from the reserve: three smoke rows (≈ 0.5–0.6u) and, only if needed, one re-pilot (+2u). |
| **Models** | O120 (`openai/gpt-oss-120b`, MXFP4) for S, E, T and W; port 8010; TP4 with 6 clients, or TP2 per D5 (same topology as A05) |
| **Status** | Not started |
| **Depends on** | E1, E2, E3, E4, E5, E6, E8a, E9f, E11a, E11b, X10; E13 (at risk, only for the label claim); decisions D4 (fold scheme) and D5 (TP); gates G1, G2, G0b-1, K-SET; data from A05 (Full controls), A08 (brute-force gate, `ANALYSIS__a08_screen__20261013`), A06 (sentinels), A09 + A12 (TOST margin δ, committed by A09-T15 on Fri 23 Oct, 18:00 (A12 reuses it)) and A12 (`fx_pAJM` = SE\|T, descriptive point only) |
| **Feeds into** | C1a (Holm); the C1 sentence (together with A08's C1b); a Table 2 row; the right panel of Fig. 2; A21 (iso-cost point); A23 (Pareto point) · A12 (shared session `S1023a`) |

*Code refs: RougeOne `nested-cv-rebuttal` @ `07cbf93` (`C:/code/PHD/RougeOne`). In this branch, `rogue_one_main.py` and `rogueone/` are identical to the current HEAD `1bdaf42`. Short paths follow 01 §7 R0. Every line reference below was re-read at `07cbf93` on Wed 7 Oct. Lines will shift after the E2/E3/E5/E9 merges, so re-anchor on the quoted text. The pinned SDK is `openai-agents==0.9.1` (`requirements.txt:1`); the SDK behaviours cited below were checked only in a local 0.3.3 install and are re-verified on 0.9.1 in A16-T5/T7. Every "hoped" result below is a hypothesis, not a result.*

---

## 1. What we test and why it matters

Suppose one model plans, writes features and judges them, one phase after another, in a **single shared conversation**. It gets the same phase instructions, tools and settings as the three agents. Does it recover planted features as well as Rogue One's three separate agents?

E10 is cut, so Solo-auto and S|ET will not exist. That leaves SET-phased as the **only internal test** that splitting the work into roles is more than spending more inference. It isolates one thing, context and identity partitioning; the compute, the control flow and the tools stay the same. A null result does not sink the paper. Decomposition is then justified by bounded context, per-role model choice (A19) and cost.

## 2. Background you need

**Full-v2, one iteration.** `forward_pass` (`main.py:536-837`) runs S → E → T. Each role builds its own `Agent` with its own system prompt, its own client and its own `SQLiteSession`. In openai-agents the default `SQLiteSession(session_id)` uses `db_path=":memory:"` with one connection per object, so every new session object starts empty (checked in 0.3.3; verify on 0.9.1 in A16-T5).

| Role | Call site | System prompt | Tools | Session |
|---|---|---|---|---|
| **S**, Scientist | `determine_focus` (`main.py:548-555`, def `sci/agent.py:75`) | `sci/agent.py:409-456`. Identity at `:412` ("You are the Scientist Agent…"), team context at `:421-423`; the round line (first user message) at `:472-475` | `:461-469` | `:499`; up to 100 retries (`:519`), each re-sending the round line into the same session; no-tool fallback on its own session (`:549-577`, session `:569-571`); heuristic focus `:501-517` |
| **E**, Extractor | `generate_patient_attributes` (`main.py:563-571`, def `ext/agent.py:172`) | `ext/agent.py:866-926`. Team context at `:871-875`, raw columns at `:914-915` (built at `:859-861`) | `:935-940` | `:965`. The R rounds (`:967`, `range(10)` until E2's knob) re-send `"System now wants to focus on: {focus}."` (`:971`); an exception skips to the next round (`:977-992`). Deterministic fallback at `:996-1002` (no LLM) |
| **T-A**, Tester assessment | `test_hypotheses` (`main.py:789-794`, def `tst/agent.py:776`) → `_run_agent_with_retries` (`:1329`) | `tst/agent.py:1426-1475`. Team block at `:1431-1437`, identity at `:1440`; input at `:1477-1481` | `:1486-1493` | `:1371` (name `:1495`); 20 attempts (`:1373`); fallback string at `:1498-1503`. Runs only if `include_report` (`tst:2113`), which `main.py:223` sets to `True` |
| **T-B**, Tester pruning | same function, called at `:2134` after T-A at `:2115` | `tst/agent.py:1534-1570`. Identity at `:1536`; "assessment as primary context" at `:1539`; input at `:1572-1577` | `:1582-1588` | `:1371` (name `:1590`); fallback string `:1593-1595` |
| **W**, search summariser | behind each role's search tool (`sci:136` `search_tool`, `ext:247` and `tst:1042` `search_in_literature_tool`) | `_run_search_agent` (`kn/web_search.py:57`); instructions `:200-236` | Serper | its own session (`:253`), unchanged in both arms |

- **Labelled executor.** Only the Tester holds `generic_python_executor_tool` (`tst/agent.py:1147-1148`, `exec` at `:1201`), whose `df_attributes` carries `target` (`:873`). S and E never hold it.
- **Shared text.** `AgentRoles.GLOBAL_GOAL` (`rogueone/llm/prompts/agent_roles.py:7-11`). Line 8 reads "You are part of a team of agents…"; line 9 "The overall goal of the team is…"; line 10 "…evaluated by a Tester Agent…".
- **Token attribution.** Tokens are recorded per client name (`clients.py:499-514`): `ScientistAgent` (`sci/agent.py:400`), `ExtractorAgent` (`ext/agent.py:851`), `TesterAgent` (`tst/agent.py:1341`).
- **Cross-iteration state (within a fold).** Only the feature pool (`df_entities_attributes`, `df_attributes_explanations`), the Scientist feedback pool (rows built by `_build_scientist_feedback_row`, `main.py:384-415`), the focus history and the Scientist notebook (`_note_book`) carry over between iterations. All are reset at the start of each fold by `_reset_fold_state` (`main.py:417-457`, called at `:543-544`). No LLM conversation carries over in either arm.
- **Unknown tools.** When a model calls a tool its current `Agent` does not hold, the SDK raises `ModelBehaviorError("Tool <name> not found in agent <agent>")` (0.3.3 `agents/_run_impl.py:558-559`; verify on 0.9.1). The exception aborts that `Runner.run` and is caught by the phase's retry loop: S and T retry, E loses that round.

**SET-phased** (`ROGUE_ONE_TOPOLOGY=set_phased`, task E9f) makes the same calls in the same order, with three changes:
1. **One shared session.** A new `SQLiteSession` object is created at the start of every iteration and used by all phases of that iteration. Nothing carries over to the next iteration.
2. **One neutral system prompt**, identical in every phase.
3. **Role rules in the phase messages.** Each role's rules move verbatim into its phase message, behind a fixed phase header.

Each phase keeps its own tools, `ModelSettings` and client. T always runs last, and the session is replaced every iteration. S and E therefore never see Tester tool outputs, so the label barrier is the same as in Full.

**Known side effects of the construction** (declared, not hidden; listed in the pre-registration):
- the raw-column list, `context_and_goal` and the output-style line, which Full gives to one or some roles, are visible to all phases through the shared system prompt;
- the T phases use the S/E input filter (A16-T7);
- the T-B input repeats the assessment that T-A already wrote into the shared history (counted as cost);
- earlier phases' rules stay visible in the history of later phases (Risk 3, §7).

**Instruments**
- **SYN-P 7:** `synp_p1`, `synp_p2`, `synp_p3`, `synp_p4`, `synp_p5`, `synp_p6a`, `synp_p6b` (E8a; n = 2,000, p ≈ 20–30, planted φ). P7 is excluded under the P7 policy. Proposed task types: C = p3, p4, p5; R = the rest (E8a confirms in `synp_meta/registry.yaml`).
- **Core-10:** `balance-scale`, `junglechess`, `vehicle`, `heart`, `credit-g`, `myocardial` (C); `housing`, `insurance`, `crab`, `wine` (R). Used for the no-harm check only; `myocardial` (p = 91) is the only wide table.
- **Pilot:** `blood` (C, p = 4), `communities` (C, p = 102). Non-core and crash-only. Both are already in the legacy list (`communities` at `main.py:1542`, `blood` at `:1544`).

**Metrics** (01 §4.6)
- **`rec_sel` (primary):** per fold-episode, an active, non-fallback feature with |Spearman| ≥ 0.95 to φ on training rows (NMI ≥ 0.9 for P3/P5) that is **in the iteration that inner CV selects**. Raw columns and single-column monotone transforms never count.
- **Secondary:** `rec_any`, `iter_first_rec`, `success`, `collapse`, `cap_hit`, `tok_compl_*`, `tok_prompt_*`, `llm_calls_*`, `wall_s`, `max_prompt_tokens` (mediator), `context_errors`, `prune_rate`, `fallback_share`, `err_per_accepted`, `fun_*`, `label_barrier_violations`; new: `offphase_tool_calls` (A16-T7).
- **Real data:** `heldout_score` and `delta_std` = Δ/σ_d, with σ_d from A05 (G2 memo `ANALYSIS__g2_stats__20261019`).

**Brute-force gate.** A family that OpenFE or AutoFeat recovers in ≥ 14/15 episodes of A08's CPU screen (`ANALYSIS__a08_screen__20261013`) is excluded from the confirmatory population (P6a and P6b go together). P1 is expected to fall there. Excluded tasks are still run and reported descriptively.

## 3. Hypotheses and interpretation (all hypotheses; none is a result)

The verdict labels are defined in §6 (C1a rule). Each paper sentence may be used only under its verdict.

| Outcome (verdict) | Meaning | Paper sentence / action |
|---|---|---|
| **Hoped** (the plan's own prior, because SYN-P tables are narrow; verdict "equivalent") | SET ≈ Full on `rec_sel`, with more prompt tokens and more context and tool errors; the token and error gap is expected to be largest on the wide tables (`communities` in the pilot, `myocardial` in Core-10; descriptive only) | "Running the same phases in one shared context recovered planted features at a rate equivalent within ±δ (Δ = x, 90% CI [a, b]) but used X× more prompt tokens and Y context/tool errors per 100 calls; separate contexts buy bounded context, not recovery." |
| **Upside** (verdict "Full better") | Full > SET on pooled C1a. Detectable only if the pooled gap is ≥ 23–30 pp (planning MDE 0.23–0.30). Whether the gap sits on P3/P5 is read descriptively, per task | "Separate role contexts, not extra inference, carried a recovery gain (Δ = x, 95% CI [a, b]); per task it was largest on …" (the multi-step wording only if the per-task table shows it) |
| **Null** (verdict "inconclusive"; a priori the most likely, given the MDE) | The CI includes both 0 and ±δ | Report the CI and say the test could not separate the arms. Use the fallback narrative: bounded context, per-role allocation (A19), least privilege if E13 lands, cost per recovered feature. Never write "similar" or "no difference". |
| **Opposite** (verdict "SET better") | SET > Full: the hand-offs lose information (for example, E would gain from seeing S's search results) | Report C1a as a negative result; the hand-off channel becomes a named limitation (the handover-protocol study, master-plan LF4, a T2 item not run). |
| **Engineering collapse** (overlay on any verdict; trigger in §6) | SET fails through overflow, off-phase calls or the 6 h cap, not through its design | Score under ITT, report sensitivity S2 (§4) next to the primary, and word every SET sentence as "SET as built", never as a design verdict. |

## 4. Design

**Control.** A05 `full` (Full-v2), with 0 new runs:
- SYN-P 7 at f42 r0, f42 r1 (A/A, `llm_seed` 1042) and f4 r0: the 21 non-P7 rows among A05-031…054 (Full contributes 3 runs per task, as in A19);
- Core-10 at f42 r0 (A05-001…010) and f42 r1 (A05-021…030). The Core-10 Full reference per dataset is the **mean of r0 and r1**, as in A13; this reuses A05 data at no cost and lowers the noise of the one-seed comparison.

**Arm.** `set_phased`, `env_overrides = {"ROGUE_ONE_TOPOLOGY":"set_phased"}`, image `agent2027-v2.4`.

**Fixed settings** (01 §4.3), all identical to the paired A05 rows:
- 10 iterations × 5 folds (`ROGUE_ONE_NUM_ITERATIONS=10`, `ROGUE_ONE_K_FOLDS=5`); inner-CV selection; rollback off;
- temperature 0.7 for every role and W; reasoning effort not passed; turn and retry caps unchanged (S 200 turns and 100 retries, E 100 turns, T 100 turns and 20 attempts);
- R (`ROGUE_ONE_EXTRACTOR_ROUNDS`) and pool mode (`ROGUE_ONE_FEATURE_POOL_MODE`) as frozen at G1;
- `ROGUE_ONE_TOOL_OUTPUT_MAX_CHARS` and `SCIENTIST_AGENT_MAX_TOKENS` as pinned at G1;
- shared Serper cache; `ROGUE_ONE_LLM_TRACE=1`, `ROGUE_ONE_PROMPT_DUMP=1`;
- `PARALLEL=6`; vLLM `v0.22.0`; `MAX_MODEL_LEN=131072`; 6 h cap (`RUN_CAP_HOURS=6`);
- `ROGUE_ONE_STRATIFIED_FOLDS` copied from the paired A05 row (`True` on p3, p4, p5 if D4 is adopted; `False` otherwise).

**Seeds.** `llm_seed = fold_seed` (`replicate=0`). Pairing is by fold seed, so both arms see the same outer partition.

**Run matrix**

| Arm | Instrument set | fold / llm seed, rep | Runs | u | W&B run name |
|---|---|---|---|---|---|
| `set_phased` (`notes=pilot`) | `blood`, `communities` | 42 / 42, r0 | 2 | 2.0 | `A16__set_phased__<ds>__f42__r0` |
| `set_phased` | SYN-P 7 | 42 / 42, r0 | 7 | 4.9–7 | `A16__set_phased__synp_<id>__f42__r0` |
| `set_phased` | SYN-P 7 | 4 / 4, r0 | 7 | 4.9–7 | `A16__set_phased__synp_<id>__f4__r0` |
| `set_phased` | Core-10 | 42 / 42, r0 | 10 | 10 | `A16__set_phased__<ds>__f42__r0` |
| `full` (A05, reused) | SYN-P 7 | 42 r0, 42 r1, 4 r0 | 21 (0 new) | 0 | `A05__full__synp_<id>__f<s>__r<r>` |
| `full` (A05, reused) | Core-10 | 42 r0, 42 r1 | 20 (0 new) | 0 | `A05__full__<ds>__f42__r<r>` |
| **New total** | | | **26** | **21.8–26** | |
| *Smoke rows, never analysed (reserve)* | `set_phased` on `synp_p3`; `full` on the two `prompt-baseline-v2.0` tasks | 42 / 42, r0; `ROGUE_ONE_SMOKE_FOLDS=1` | 3 × 1 fold | ≈ 0.5–0.6 | `A16__<arm>__<ds>__f42__r0`, `code_phase=smoke` |

**Budget arithmetic.** 14 × 0.7–1.0u + 10 × 1.0u + 2 × 1.0u = 9.8–14 + 10 + 2 = **21.8–26u**, as in the T1-minimum breakdown. The smoke rows (each 1 fold of 10 iterations ≈ 0.15–0.2u) and a possible re-pilot (+2u) come from the reserve and are not part of the 26. If shared contexts slow runs by about 20%, the high end is about 31u. The session sentinels are booked under A06. If K-SET fails, the 24 rows A16-003…026 (19.8–24u) are freed.

**Sessions and randomisation.**
- **Pilot session `S1022b`** (A19 uses `S1022a`, A20 uses `S1022c`, `S1022d` and `S1022e`, A18 uses `S1022p`–`s`): the three smoke rows and the two pilot rows. No sentinel: none of its rows is analysed (A06 §4.5).
- **SYN-P session `S1023a`:** A16-003…016 interleaved with A12 block 2 (A12-033…056) and one A06 sentinel in the first wave (the first analysed session on v2.4, A06 §4.5 guard 3).
- **Core-10 rows** share an O120 session (`S1026<x>` or later) with any remaining O120 rows, plus that session's A06 sentinel.
- `make_manifest.py --plan-session` sets the order (`order_key`, RNG seeded by `SESSION_ID`).

**Failure handling** (01 §4.5)
- A crash (rc ≠ 0 or no `fold_aggregation_summary.csv`) is retried once, alone, at the end of the session.
- A persistent failure or a 6 h cap hit is an ITT failure: every fold-episode gets `rec_sel = rec_any = 0` and `success = 0`, and the held-out score is imputed as in 01 §4.5. ITT runs are never dropped.
- Infrastructure faults (vLLM death, host reboot, W&B or Serper outage, visible in the job or vLLM log) are not ITT if X10 adopts A05's rule: the row is re-queued with `notes=infra-rerun`.
- **Pre-declared sensitivity analyses** (reported next to the primary; they never change the verdict):
  - **S1** (01 §4.5): exclude runs whose selected features come only from the fallback;
  - **S2** (A16): exclude runs in which any phase was **exhausted by context errors** in any iteration, i.e. the phase ended in its fallback (S no-tool or heuristic focus, E deterministic fallback, T-A fallback report, T-B "No pruning summary") **and** at least one context error (HTTP 400 or `finish_reason=length`) was logged for that phase in that iteration.

## 5. Tasks

### 5.1 Prerequisites

- [ ] **A16-T1** [type: writing] (0.25 d, due Fri 16 Oct) — Write the SET construction spec from §5.2 into `docs/agent2027/preregistration.md` (X10). It covers the neutral system prompt, the deletion list and the full before→after substitution table (23 rows, A16-T6), the phase-header text, the E-round and retry rules, the K-SET pass rule (§6), the overflow and tool-line contingencies, the C1a population, model, separation fallback and verdict rule (§6), sensitivity analyses S1/S2, and the engineering-collapse trigger. *Done when:* the section is in the commit tagged `agent2027-prereg`.
- [ ] **A16-T2** [type: config] (0.25 d, due Fri 16 Oct) — Verify the "done when" checks in 01 §3.3 for E1, E2, E3, E4, E5, E6, E8a and E13, and the task data on CAIR. *Done when:*
  - the G0a smoke run shows non-zero `tok_compl_scientist`, `tok_compl_extractor` and `tok_compl_tester`, and `max_prompt_tokens` and `context_errors` in the W&B summary;
  - the E2 100 KB tool-output cut test passes;
  - the W&B artifact `prompt-baseline-v2.0` exists, and its two task names (one SYN-P, one Core-10) are noted for A16-T9;
  - `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "ls -d <ROGUEONE_CAIR_PATH>/tasks/classification/blood <ROGUEONE_CAIR_PATH>/tasks/classification/communities <ROGUEONE_CAIR_PATH>/tasks/*/synp_p{1,2,3,4,5,6a,6b}"` lists all 9 folders, and none of the SYN-P folders contains φ or other metadata (01 §4.4 rule 3).
- [ ] **A16-T3** [type: data] (0.1 d, due Mon 19 Oct, G2) — Confirm the inputs. *Done when:*
  - every A05 `full` control row in §4 is finished or ITT-scored, and `recovery_detector.py` outputs exist for them (A05 `a05-handoff` artifact);
  - A08's non-discriminating list is posted (`ANALYSIS__a08_screen__20261013`);
  - the G2 memo `ANALYSIS__g2_stats__20261019` gives the margin M value, σ_d per dataset, the informative tasks and the sentinel task.
- [ ] **A16-T4** [type: decision] (0.05 d, due Wed 21 Oct, G0b-1) — Confirm that v2.3 is merged and that Full's rendered prompts on v2.3 are byte-identical to `prompt-baseline-v2.0`; E9f branches from that commit. *Done when:* the G0b-1 record (merge log or W&B `ANALYSIS` memo) shows both, and the E9f branch base equals the v2.3 tag.

### 5.2 Implementation (E9f, 1–1.25 d; default-off; due Thu 22 Oct)

Effort of T5–T8 is 1.1 d, more than one working day. **A16-T6 is pure prompt text built from the frozen v2.0 role blocks, so it may start Sat 17 Oct.** If T5–T8 are not done by Thu 22, 20:00, the pilot moves to Fri 23 morning and the K-SET timing rule (§6) applies.

- [ ] **A16-T5** [type: code] (0.25 d, due Thu 22 Oct) — Plumb one session per iteration.
  - Add `ROGUE_ONE_TOPOLOGY` (default `three_agent`) to `rogueoneConfig` in `config.py` (next to `selection_signal`, `config.py:317`), to `OPTIONAL_ENV_DEFAULTS` in `rogueone/utils/config_keys.py`, and to the E1 flag registry, so it is logged as `flags/ROGUE_ONE_TOPOLOGY`.
  - In `forward_pass` (`main.py:536-837`), when the flag is `set_phased`, create one `SQLiteSession(f"set_phased_trial_{index}")` after the fold reset (`:543-544`) and before `determine_focus` (`:548`).
  - Pass it as a keyword `shared_session=` (default `None`) to `determine_focus`, `generate_patient_attributes` and `test_hypotheses`; `test_hypotheses` hands it to its inner `_run_agent_with_retries` (`tst:1329`) for T-A and T-B.
  - Only when it is given does it replace the sessions at `sci/agent.py:499`, `ext/agent.py:965` and `tst/agent.py:1371`. The S no-tool fallback keeps its own fresh session (`sci:569-571`).
  - **Retries and rounds.** The phase header and role block are sent only on the first attempt of a phase. Retries (`sci:519`, `tst:1373`) and E rounds 2..R send only the phase's trailing input (S: the round line; E: the `:971` line; T-A, T-B: their input text), which is exactly what Full re-sends into its own session. (The SDK stores a run's input at run start, so the header is already in the history after a failed attempt; 0.3.3 `agents/run.py:506`, verify on 0.9.1.)
  - The error-reporter context (`sci:538-540`, `ext:985-987`, `tst:1406-1409`) gains a `phase` field (`S`, `E`, `T-A`, `T-B`), because the shared session name no longer identifies the role.
  - *Done when:* unit tests show (a) the same session object for S, every E round, T-A and T-B within one iteration, a new object at the next iteration, and an empty history for each new object; (b) the header and role block appear exactly once per phase in the stored history, also after an injected retry; (c) with `three_agent`, the session names and calls are unchanged (guard 2).
- [ ] **A16-T6** [type: code] (0.5 d, may start Sat 17, due Thu 22 Oct) — Build the prompts in `rogueone/llm/prompts/set_phased.py`. Line numbers are at `07cbf93`; build from the **v2.0-freeze text** of the same blocks, so the E2/E5 edits (round counter `sci:473`, hint removal `sci:441`, `:443`, `ext:901`, freeze item 14 `ext:895`) apply identically in both arms. Role blocks are copied byte-for-byte, indentation included.
  - **Neutral system prompt**, identical for all phases, in this order:
    1. `GLOBAL_GOAL` (`agent_roles.py:7-11`) without line 8, with "The overall goal of the team is" → "The overall goal is" (`:9`) and "by a Tester Agent" → "in the ASSESS/PRUNE phases" (`:10`);
    2. one fixed line: "You will work through four phases (PLAN, BUILD, ASSESS, PRUNE); each phase message states its instructions and tools.";
    3. `# Task context and predictive goal:` followed by `context_and_goal`;
    4. the raw-column block (`ext:914-915`, list built as at `ext:859-861`);
    5. "Do not provide long chain-of-thought narratives." (`sci:455`).
  - **Phase messages:** the frozen header `### PHASE k/4 — <NAME>. From now on only the instructions below and the tools of this phase apply.`, followed by the role block and the trailing input:

    | Phase (k, NAME) | Role block | Trailing input |
    |---|---|---|
    | S (1, PLAN) | `sci:413-415` + `:425-447` + `:452-454` | the round line `:472-475` |
    | E (2, BUILD) | `ext:878-912` + `:917-921` | the `:971` line. **Rounds 2..R send the `:971` line alone**, as in Full. |
    | T-A (3, ASSESS) | `tst:1441-1471` | the input `:1477-1481` |
    | T-B (4, PRUNE) | `tst:1537-1566` | the input `:1572-1577`, verbatim (the duplicate assessment is counted as cost) |

  - **Deleted** (outside the role blocks, so never sent): the identity lines `sci:412`, `tst:1440` (including "Your current stage is FEATURE ASSESSMENT ONLY.") and `tst:1536`, with their `# Role` headers (`tst:1439`, `:1535`); the team blocks `sci:421-423`, `ext:871-875` and `tst:1431-1437`; the per-role copies of the global goal and task context (moved once into the system prompt).
  - **Substituted:** every role mention in the moved text becomes its phase: Scientist → "the PLAN phase", Extractor → "the BUILD phase", Tester → "the ASSESS/PRUNE phases" (any case, with or without "Agent"; only the article and possessive are adjusted). Occurrences at `07cbf93`: `sci:413` (×2), `:432`, `:436`, `:441` (×4), `:442` (×3), `:443`, `:445`, `:447` (×2); `ext:879`, `:899`; `tst:1441`, `:1442`; plus `agent_roles.py:9-10` (above) and "the raw columns listed below" → "the raw columns listed in the system prompt" (`ext:911`, `:912`). That is 23 rows; the exact before→after strings are frozen in A16-T1.
  - **Not changed:** tool names, tool docstrings and schemas (the "same tools" condition). Two docstrings still name roles (`sci:182` "scientist feedback dataframe", `sci:189` "produced by the Tester"); they are identical in both arms and declared.
  - *Done when:*
    - a unit test (no LLM) applies the frozen table to the v2.0 role blocks and reproduces every SET phase message and the system prompt byte-for-byte;
    - in the `set_phased` dumps of A16-000 and the pilot (E5), the system prompt is byte-identical across the 4 phases of every iteration, and `grep -iE "you are the|team of agents|scientist|extractor|tester"` over the system prompts and phase messages returns nothing outside `context_and_goal`.
- [ ] **A16-T7** [type: code] (0.25 d, due Thu 22 Oct) — Keep the per-phase settings, and add counters.
  - Unchanged per phase: the clients (`sci:397-401`, `ext:848-852`, `tst:1338-1345`), `ModelSettings` (`sci:404-407`, `ext:864`, `tst:1348`), and the tool lists (`sci:461-469`, `ext:935-940`, `tst:1486-1493`, `:1582-1588`).
  - Under `set_phased`, every phase uses the S/E input filter (`sci:477-495`, which strips reasoning items and `<|…|>` tokens; the T filter at `tst:1358-1367` strips only reasoning items), because the T phases now see S and E content.
  - The S fallback keeps its own session (`sci:566-574`).
  - Add the counter `offphase_tool_calls`: the number of caught exceptions whose message matches "Tool … not found in agent …", logged per phase. Check that `context_errors` counts HTTP 400 prompt-too-long and `finish_reason=length`, and that each error record carries `phase` (A16-T5).
  - *Done when:* unit tests show that the S and E tool lists exclude `generic_python_executor_tool`, that an injected call to an off-phase tool increments `offphase_tool_calls` for the right phase, and that an injected HTTP 400 increments `context_errors` with the right phase.
- [ ] **A16-T8** [type: code] (0.1 d, due Thu 22 Oct) — Release guard (01 §2.2) and image.
  - Locally: unit tests green (guard 2), including A16-T5 to T7.
  - Push branch `agent2027/v2`; on CAIR fetch and build `rogue_one:agent2027-v2.4` from that commit (commands in §5.3) and record the image ID.
  - Smoke rows in the pilot session (A16-T10): A16-000 (`set_phased`, `synp_p3`) and A16-027/028 (`full` on the two `prompt-baseline-v2.0` tasks, with the same smoke settings as the baseline: `ROGUE_ONE_SMOKE_FOLDS=1`, `ROGUE_ONE_PROMPT_DUMP=1`).
  - Guard 1: `prompt_diff.py` of the A16-027/028 dumps against `prompt-baseline-v2.0` is byte-identical (as E5 defines the comparison; tool descriptions via the trace `tools` field, as in A11-T8).
  - When guards 1 and 2 pass, push the git tag `agent2027-v2.4` on the commit the image was built from (its `git_sha` in W&B). If they fail, fix, rebuild the image under the same docker tag (no analysed row has used it yet) and repeat; the K-SET timing rule applies.
  - *Done when:* A16-000 is in W&B with non-zero tokens for all three roles and the dumped prompts; the guard-1 diff log is attached to the merge; the tag is pushed.

### 5.3 Execution on CAIR

**Rules** (01 §5.1; they apply to every command below):
- **Containers only.** Every run, analysis, W&B query and import check runs inside Docker. The CAIR host only issues `docker` / `docker compose` commands and the launcher scripts in `~` that issue them, plus `nvidia-smi`, `docker ps`, `ls` and `tail` for checks. Never `python …` on the host. Any `docker compose run` over SSH needs `-T`.
- **GPU pre-flight before every GPU container** (command in A16-T10). If not enough GPUs are free, **stop and report** the per-GPU memory and utilisation, the compute apps and the running containers, and let Henrik decide. Launch with explicit `GPUS=<ids>` from the pre-flight and `WAIT_MAX_MIN=0`.
- **≤ 4 of the 8 H100s, ever, without Henrik's explicit approval in chat**; one Rogue One job at a time.
- **W&B (`PlanBench/RogueOne-AGENT2027`) is the system of record.** Runs are stateless and upload `output/` as artifact `run-output`. Secrets come from `.env` files: `WANDB_API_KEY` from `<CAIR_HOME>/code/PlanBench/.env`; `SERPER_API_KEY` and LLM settings from `.env` / `.secret.env` in `<ROGUEONE_CAIR_PATH>`, baked in at build. Never push these images.
- **Code reaches CAIR only by commit + push locally, then `git fetch` over SSH on the server.** Non-interactive SSH only: `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "<one-shot command>"`.
- **Long jobs run detached** with `nohup`; the job log goes in `~` (`~/ro27_<SESSION_ID>.job.log`), never in the root-owned run folders.

`<ROGUEONE_CAIR_PATH>` = `<CAIR_HOME>/code/RougeOne` (**inferred from `paper/scripts/clients_t16_rerun.sh:135` and `:59`; verify once with `ssh CAIR-gpu14 "ls -d ~/code/RougeOne"`**).

**Command reference** (01 §5.2, filled in for A16; `<SID>` = session ID):
```
# Step 0 · code to the server
ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15' git fetch git@github.com:HenrikNorkart/RougeOne.git agent2027/v2 && git checkout -B agent2027/v2 FETCH_HEAD && git log -1 --format='HEAD: %h %s'"
# Step 2 · build v2.4 (only when code changed)
ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && test ! -e .dev.env && docker build -t rogue_one:agent2027-v2.4 . && docker image inspect rogue_one:agent2027-v2.4 --format '{{.Id}}'"
# Step 3 · install launchers and manifests in ~
ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && cp scripts/agent2027/ro27_session_launch.sh scripts/agent2027/ro27_clients_manifest.sh scripts/agent2027/ladder_session.sh ~/ && cp paper/scripts/vllm_session.sh ~/ro27_vllm_session.sh && mkdir -p ~/ro27_manifests && cp -r manifests/agent2027/. ~/ro27_manifests/"
# Step 5 · monitor (run as a background command; exit code 255 = network drop, reconnect)
ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 CAIR-gpu14 "while ! grep -q 'RO27_SESSION_END' ~/ro27_<SID>.job.log 2>/dev/null; do sleep 120; done; tail -30 ~/ro27_<SID>.job.log"
# Step 6 · tear-down check
ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker ps --format '{{.Names}}' | grep -E '^(ro27-|rogueone-vllm-)' || echo none; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"
# Step 7 · W&B states, inside a container
ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker run --rm --env-file <CAIR_HOME>/code/PlanBench/.env --entrypoint python3 rogue_one:agent2027-v2.4 -c \"import wandb; api=wandb.Api(); [print(r.name, r.state) for r in api.runs('PlanBench/RogueOne-AGENT2027', filters={'group': 'A16'})]\""
```

- [ ] **A16-T9** [type: config] (0.1 d, due Thu 22 Oct) — Run `scripts/agent2027/make_manifest.py --ablation A16` to write `manifests/agent2027/A16_RT2.csv` (R7 columns), then commit and push it.

  Columns common to every row:
  - `ablation_id=A16`, `is_sentinel=0`, `replicate=0`, `session_id` and `order_key` empty;
  - `image_tag=agent2027-v2.4`;
  - `rung_S/E/T=O120`, `port_S/E/T=8010`;
  - `cap_hours=6`;
  - `task_type`: SYN-P from `synp_meta/registry.yaml`; Core-10 from 01 §7 R10; `blood` and `communities` = `classification`.

  | row_id | arm_id | dataset_id | fold_seed / llm_seed | env_overrides | expected_u | notes |
  |---|---|---|---|---|---|---|
  | A16-000 | `set_phased` | `synp_p3` | 42 / 42 | TOPOLOGY=`set_phased`, SMOKE_FOLDS=`1` | 0.2 | `smoke` |
  | A16-001–002 | `set_phased` | `blood`, `communities` | 42 / 42 | TOPOLOGY=`set_phased` | 1.0 | `pilot` |
  | A16-003–009 | `set_phased` | SYN-P 7 | 42 / 42 | TOPOLOGY=`set_phased` | 0.85 | pair A05 f42 r0/r1 |
  | A16-010–016 | `set_phased` | SYN-P 7 | 4 / 4 | TOPOLOGY=`set_phased` | 0.85 | pair A05 f4 r0 |
  | A16-017–026 | `set_phased` | Core-10 | 42 / 42 | TOPOLOGY=`set_phased` | 1.0 | pair A05 f42 r0/r1 |
  | A16-027–028 | `full` | the two `prompt-baseline-v2.0` tasks | 42 / 42 | SMOKE_FOLDS=`1` | 0.2 | `smoke-guard` |

  Flag names in `env_overrides` are the full R3/R4 names (`ROGUE_ONE_TOPOLOGY`, `ROGUE_ONE_SMOKE_FOLDS`). If D4 is adopted, rows on `synp_p3`, `synp_p4` and `synp_p5` add `"ROGUE_ONE_STRATIFIED_FOLDS":"True"`, as A05 and A12 do. Smoke rows run with `ROGUE_ONE_CODE_PHASE=smoke` and are never analysed. Re-pilot rows, if needed, are added later as A16-029–030 (`notes=repilot`) in a dated manifest commit. *Done when:* the manifest has 29 rows (26 analysed or pilot + 3 smoke), and the effective fold flags (`ROGUE_ONE_FOLD_SEED`, `ROGUE_ONE_STRATIFIED_FOLDS`, `ROGUE_ONE_K_FOLDS`) of every SYN-P and Core-10 row equal those of its paired A05 rows.
- [ ] **A16-T10** [type: run] (0.15 d, due Thu 22 Oct, night; latest launch Fri 23, 12:00) — Run the smoke + pilot session. Plan it first: `make_manifest.py --plan-session S1022b --rows A16-000 A16-027 A16-028 A16-001 A16-002`, commit and push. Only one Rogue One job runs at a time: this session starts when the box is free and must not be appended to a session that is already running (see Risk 4 for the Thu 22 conflict). Then:
  - **Sync and build:** steps 0, 2 and 3 above.
  - **Pre-flight:**
    ```
    ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader && echo '--- compute apps ---' && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader && echo '--- containers ---' && docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'"
    ```
    Go only if the GPUs you need (4 for TP4, 2 for TP2 per D5) show < 2,000 MiB used, the total you will hold is ≤ 4, and no `ro27-*`, `rogueone-vllm-*`, `llmfe-*` or `caafe-*` container is running. Otherwise **stop and report** the output to Henrik.
  - **Launch** (detached):
    ```
    ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "SESSION_ID=S1022b SESSION_FILE=~/ro27_manifests/sessions/S1022b.csv GPUS=<ids> MODEL=openai/gpt-oss-120b PORT=8010 SERVER_NAME=rogueone-vllm-o120-s1022b LOG=~/vllm_session_S1022b.log IMAGE_RO=rogue_one:agent2027-v2.4 PARALLEL=6 CPUS=12 RUN_CAP_HOURS=6 RUNROOT=~/rogueone-runs-agent2027 MAX_MODEL_LEN=131072 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_S1022b.job.log 2>&1 & echo launched; sleep 5; tail -5 ~/ro27_S1022b.job.log"
    ```
  - **Monitor:** step 5 in the background until `RO27_SESSION_END S1022b` appears; then step 6.
  - *Done when:* the job log shows `RO27_SESSION_END S1022b`, both pilot runs are `finished` in W&B or logged as `FAILED`/`CAP_HIT`, the three smoke rows have a final state, and step 6 prints `none`.
- [ ] **A16-T11** [type: decision] (0.1 d, due Fri 23 Oct, morning; final by 24:00) — Apply the K-SET pass rule (§6). One re-pilot is allowed on Fri 23, after a frozen fix and a dated pre-registration amendment, at +2u from the reserve. *Done when:* the K-SET verdict (go, or ED4) and the per-criterion values are logged as W&B run `ANALYSIS__k_set__20261023`.
- [ ] **A16-T12** [type: run] (0.1 d, due Fri 23 – Sun 25 Oct; latest launch Thu 29, L-LAST) — After a K-SET go, plan `make_manifest.py --plan-session S1023a --rows A16-003…016 A12-033…056 <A06 sentinel row>` with the sentinel's `order_key` set to the session minimum, commit and push, then pre-flight and launch as in T10 (`SESSION_ID=S1023a`). Rows carry their own `image_tag`. *Done when:* all 14 A16 runs are `finished` or ITT-classified, and each has a `run-output` artifact.
- [ ] **A16-T13** [type: run] (0.1 d, due Mon 26 – Thu 29 Oct, L-LAST) — Plan rows A16-017–026 into `S1026<x>` or a later O120 session, with that session's A06 sentinel; pre-flight and launch as in T10. The last launch is **Thu 29** (L-LAST). *Done when:* all 10 runs are finished or ITT-classified, each with a `run-output` artifact.
- [ ] **A16-T14** [type: run] (0.25 d spread, through Sat 31 Oct, DF) — Monitor and clean up.
  - Check W&B states with step 7. A `running` run without a container (`docker ps`) is a zombie: record it in the session notes and re-queue the row (`notes=zombie-rerun`).
  - The launcher retries crashes once automatically; cap hits are not retried; infra faults are re-queued (`notes=infra-rerun`).
  - A rerun must launch by Sat 31, 18:00, so that its 6 h cap ends before DF.
  - Run the tear-down check (step 6) after each session.
  - *Done when:* step 6 prints `none` after the last A16 session, and every A16 row has a final state (`finished`, `ITT`, or `cancelled (K-SET)`).

### 5.4 Analysis (CPU containers on CAIR, `rogue_one:agent2027-v2.4`)

All analysis runs in a container, never on the host, with `--cpus 16` at most:
```
ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "mkdir -p ~/ro27_analysis/A16 && docker run --rm --cpus 16 --env-file <CAIR_HOME>/code/PlanBench/.env -v ~/ro27_analysis/A16:/work -v ~/rogueone-synp-meta:/synp_meta:ro --entrypoint python3 rogue_one:agent2027-v2.4 /workspace/scripts/agent2027/analysis/<script>.py <args>"
```
Results are logged to W&B (`ANALYSIS__<script>__<YYYYMMDD>`, group `ANALYSIS`); `~/ro27_analysis/` is scratch only.

- [ ] **A16-T15** [type: analysis] (0.25 d, preliminary Sun 25 Oct; final Sun 1 Nov) — Run `analysis/fetch_wandb.py` for group A16 (plus the paired A05 rows), then `analysis/recovery_detector.py` on every `run-output`. Recovery is computed only after the TOST margin δ is committed (A09-T15, Fri 23 Oct, 18:00). The preliminary pass is a completeness check; no SET code or prompt may change after K-SET, whatever it shows. Select runs by `manifest_row` and `code_phase=v2`: the smoke rows share the R12 run-name pattern (A16-000 has the same name as the analysed `synp_p3` f42 row) and must never enter an analysis, nor may pilot or re-pilot rows. *Done when:* `rec_sel`, `rec_any` and `iter_first_rec` exist for all 70 SET SYN-P fold-episodes (ITT included) and the 105 paired Full fold-episodes.
- [ ] **A16-T16** [type: analysis] (0.25 d, due Nov 1–3) — Run the C1a test with `analysis/glmm_holm.py`.
  - **Population:** SYN-P 7 minus the brute-force-gated tasks (A08), all runs ITT. Full contributes 3 runs per task (f42 r0, f42 r1, f4 r0), SET 2 (f42, f4).
  - **Model:** logistic GLMM `rec_sel ~ arm + (1|task) + (1|task:seed)` on fold-episodes.
  - **Report:** the raw two-sided p, the Holm-adjusted verdict, and the marginal risk difference SET − Full with 95% and 90% CIs.
  - **Separation fallback (test of record if the GLMM fails to converge, reports a singular fit for the arm term, or either arm's pooled recovery is 0% or 100%):** an exact two-sided test stratified by task with fold-episodes as units (exact conditional logistic regression, as named in the pre-registration).
  - Holm is applied across the surviving family.
  - P3/P5 versus the single-formula tasks is reported descriptively, per task.
  - Sensitivity analyses S1 and S2 (§4) are rerun with the same model.
  - *Done when:* the W&B run `ANALYSIS__glmm_holm__<date>` holds the C1a row (raw p, Holm verdict, risk difference, both CIs, test used, S1, S2).
- [ ] **A16-T17** [type: analysis] (0.25 d, due Nov 1–3) — Compute the mechanism and cost metrics with `analysis/process_metrics.py` and `analysis/funnel.py`:
  - prompt-token and completion-token ratios SET/Full per (task, seed), with Full taken as the mean over its replicates at that seed: geometric mean with a bootstrap CI over tasks;
  - the `max_prompt_tokens` distribution per arm and dataset;
  - `context_errors` and `offphase_tool_calls` per 100 `llm_calls_total`, per phase;
  - `prune_rate`, `fallback_share`, `err_per_accepted` and `fun_*`;
  - descriptive context point: A12 `fx_pAJM` (SE|T, SYN-P 7 × 42) on the same axes, outside Holm.

  *Done when:* there is one table per metric in a W&B `ANALYSIS` run.
- [ ] **A16-T18** [type: analysis] (0.25 d, due Nov 1–3) — Run the Core-10 no-harm check and the leakage scan.
  - Mixed model on `delta_std` (Δ = SET − mean of Full f42 r0 and r1, divided by σ_d from G2), one-sided at margin M (01 §4.7).
  - Run `paper/scripts/scan_leakage.py` on the Core-10 SET outputs: lay out the downloaded `run-output` artifacts as `<root>/tasks/<task_type>/<dataset_id>/output/` (the script reads `df_test_pool_final.csv` there) and mount `<root>` at `/w1`, the seed-42 path that `SEED_ROOTS` (`:36-40`) expects, and an output folder at `/out`: `docker run --rm --cpus 16 -v ~/ro27_analysis/A16/scan_root:/w1:ro -v ~/ro27_analysis/A16/scan_out:/out --entrypoint python3 rogue_one:agent2027-v2.4 /workspace/paper/scripts/scan_leakage.py`.
  - The script screens held-out scores for implausible folds (within-run outliers and above-published-ceiling folds); it does not inspect features directly. It only covers the dataset names it lists (`:54-64`), so SYN-P relies on E13 and the tool-gating test (A16-T7).
  - `label_barrier_violations` = the number of rows in `/out/leakage_scan.csv` for SET runs.
  - *Done when:* the Δ_std CI against M and the `label_barrier_violations` count are logged.
- [ ] **A16-T19** [type: analysis] (0.1 d, due Nov 1–3) — Run `analysis/sentinel_drift.py --session` on `S1023a` and the Core-10 session, and hand the SET token totals to A21 (iso-cost) and A23 (`analysis/pareto.py`). *Done when:* the sentinel verdicts and the iso-cost point (or "no iso-cost point") are recorded in W&B `ANALYSIS`.

### 5.5 Paper outputs

- [ ] **A16-T20** [type: writing] (0.25 d, due Nov 4–6) — Write the Table 2 row "SET-phased (C1a)". Its columns: Δ `rec_sel` (CI), Δ success, Δ prompt and completion tokens, the funnel stage that changed, no-harm Δ_std against M, and the verdict from §6 (Full better / equivalent / inconclusive / SET better, plus "as built" if the collapse trigger fired). Then write the C1 sentence that §3 allows for that verdict, worded narrowly: "SET-phased isolates context and identity partitioning, not free control flow or budget". *Done when:* the row and the sentence are in the draft.
- [ ] **A16-T21** [type: writing] (0.25 d, due Nov 4–6; package Thu 12 Nov) — Add the SET point to the right panel of Fig. 2 (recovery against completion tokens). In the appendix, add: the SET construction rule and the 23-row substitution table, the declared side effects (§2), the pilot table, the prompt-length and context-error table, and `offphase_tool_calls`. Add the rendered SET prompts to the replication package (Thu 12 Nov). *Done when:* the figure point and the appendix items exist, and the prompts are in the package folder.

## 6. Decision rules and gates (pre-registered)

**K-SET, Fri 23 Oct.** The pilot passes only if all of these hold in **each** of the two pilot runs (rates per 100 `llm_calls_total`):
1. the run finishes within 6 h with `fold_aggregation_summary.csv`;
2. no phase is exhausted by context errors (definition in §4, S2), and `context_errors` ≤ 2 per 100 calls;
3. `offphase_tool_calls` ≤ 5 per 100 calls;
4. all three roles show non-zero completion tokens;
5. the A16-T6 dump checks pass on the pilot dumps;
6. the Full byte-identity guard (A16-T8) holds;
7. `scan_leakage.py` flags no pilot fold (01 §3.3, E9f done-when). If it flags one, the flagged fold's features are read; the pilot fails only if a selected feature references the target, otherwise the flag is logged as a false positive.

An infrastructure fault (vLLM death, host reboot, W&B outage) is not a fail: the row is re-queued.

**What a failure triggers:**
- If criterion 2 fails, apply the overflow contingency: tool-output bodies of *completed earlier phases* in the shared history are replaced by `[tool output elided]`; the call and response structure stays. Then re-pilot.
- If criterion 3 fails, add one frozen line to each phase header, "Tools available in this phase: …" (the phase's tool names), then re-pilot.
- Any other failure: one frozen fix, then re-pilot.
- **Re-pilot:** rows A16-029–030 (`notes=repilot`), dated pre-registration amendment first, run as their own session (pre-flight and launch as in A16-T10). The re-pilot must launch by **Fri 23, 18:00** so that its 6 h cap ends before 24:00. Because only one Rogue One job runs at a time, `S1023a` is planned and launched only after the K-SET verdict; under ED4, A12 block 2 runs alone with its own sentinel (A12 §4).
- **Timing:** a first pilot launched after Fri 23, 12:00 leaves no time for a re-pilot.
- If there is no pass by **Fri 23, 24:00**: apply **ED4**. C1a is dropped before any SYN-P data exist, Holm is recomputed, rows A16-003–026 are cancelled, and 19.8–24u are freed. The Table 2 row reads "not run (K-SET)". Moving K-SET is Henrik's decision only, by a dated amendment before 24:00.

**C1a verdict** (δ = the TOST margin committed by A09-T15 on Fri 23 Oct, 18:00 (A12 reuses it): max(25% of the Full–oracle recovery gap from A09's `rt3-oracle-gap` and A05, the pre-registered absolute floor, proposed 0.10); 01 §4.7):
- **Full better:** Holm-adjusted p < 0.05 and risk difference SET − Full < 0.
- **SET better:** Holm-adjusted p < 0.05 and risk difference > 0.
- **Equivalent:** not significant, and the 90% CI of the risk difference lies entirely within ±δ. This is a secondary reading outside the Holm family, stated as such.
- **Inconclusive:** every other case.
- Mechanism claims (tokens, errors, funnel) are descriptive only.
- **Engineering-collapse overlay** (proposed threshold, frozen in A16-T1): if ≥ 3 of the 14 SET SYN-P runs are ITT failures or meet the S2 exhaustion criterion, every SET sentence says "as built" and S2 is reported next to the primary.

**Core-10:** write "no harm beyond M" only if the one-sided CI excludes M; otherwise "inconclusive", which is expected at 1 seed.

**Sentinel:** if `sentinel_drift.py` flags `S1023a` or the Core-10 session, its A16 rows are rerun from the reserve if the rerun session can launch by Sat 31, 18:00; otherwise the analysis adds a session covariate where estimable and the caveat is printed under Table 2.

## 7. Risks and cut levers

1. **Context overflow at 131,072 tokens.** One iteration's history holds S, E × R rounds and T together. Mitigations: E2's bounded tool output, the frozen R, header-once retries (A16-T5), the pilot on `communities`, and the overflow contingency.
2. **Off-phase tool calls.** The SDK raises when the model calls a tool absent from the current phase; each such call costs S or T a retry and costs E a whole round. Whether vLLM 0.22.0's gpt-oss template tolerates a history full of other phases' tool calls is unverified. It is checked in the pilot (criterion 3).
3. **Strawman.** Earlier phases' rules stay visible later in the conversation. Mitigations: the phase header, the verbatim role blocks, the declared side effects, and a construction rule frozen before any core run and published.
4. **Schedule (unresolved across documents).** E7 and E9 occupy Oct 19–22, so the build lands Thu 22, and T5–T8 (1.1 d) exceed one day (T6 may start Sat 17). Thu 22 night is also claimed by A19 (`S1022a`, whole 4-GPU slot for 20–39 h) and A20 (`S1022c` smoke, `S1022d` 6–15 h), and only one Rogue One job may run at a time. The pilot session needs ≤ 6 h of an O120 slot that starts by Fri 23, 12:00; `S1023a` needs about 20–28 h from Fri 23 (39 rows ≈ 27–39u). **Henrik sets the Thu 22 – Fri 23 order**; the proposal here is: A20 smoke (afternoon) → A16 pilot session (evening) → the rest, because K-SET is a hard pre-registered kill date. ED4 is fourth in the engineering drop order. No CL1–CL7 lever touches A16. Cutting the Core-10 part would need a pre-registration amendment.
5. **Power.** MDE is about 0.23–0.30. Core-10 has 1 seed. Width is descriptive only (one wide core table).
6. **Session confound.** The Full runs come from A05 sessions (Oct 16–18) on v2.0. The byte-identity guard, the switch-off unit tests and the A06 sentinels guard against drift.
7. **Search variance.** The shared Serper cache lowers search variance; this is disclosed.
8. **SDK version.** The session and unknown-tool behaviours were read in openai-agents 0.3.3; the image pins 0.9.1. A16-T5 and T7 unit tests run against the pinned version.

## 8. Definition of done

- [ ] SET spec, substitution table and decision rules in the tagged pre-registration (A16-T1).
- [ ] `agent2027-v2.4` released with the Full byte-identity guard, and the K-SET verdict logged (`ANALYSIS__k_set__20261023`).
- [ ] 26 rows in W&B, each finished or ITT-classified (or cancelled at K-SET), each run with a `run-output` artifact; smoke rows tagged `code_phase=smoke`.
- [ ] C1a GLMM and Holm row (with S1, S2), mechanism tables, Core-10 no-harm result and scan count logged in `ANALYSIS` runs.
- [ ] Table 2 row, Fig. 2 point, C1 sentence and appendix items in the draft.

**New names introduced** (not in the 01 §7 registry):
- metric `offphase_tool_calls`;
- module `rogueone/llm/prompts/set_phased.py`;
- session name pattern `set_phased_trial_{index}`; the `phase` field in error records;
- W&B run `ANALYSIS__k_set__20261023`;
- rows A16-000 (`smoke`), A16-027–028 (`full`, `smoke-guard`), A16-029–030 (`repilot`, only if needed);
- the "overflow contingency" and "header-once retry" rules, sensitivity labels S1/S2 and the engineering-collapse overlay (§4, §6).
