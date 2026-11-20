# A18 — RP1 · Role-wise prompt replay across rungs

| Field | Value |
|---|---|
| **ID** | A18 (short ID RP1) |
| **Claim / Holm contrast** | Role-wise half of C5: a capability floor per role. Exploratory and **outside Holm** (R14) |
| **Tier** | T1-minimum ("near-free"). On no cut lever; shrinks only under ED2 |
| **Owner** | Henrik |
| **Start / finish** | Prep Tue 13 – Wed 21 Oct (X2 check Tue 13; pre-registration and E3 add-ons Fri 16; code Tue 20 – Wed 21; `sample` run and CAIR install Wed 21) · GPU, **planned slots** (placed in the session plan by G2, Mon 19; see §4 *Calendar vs the breakdown*): **Thu 22 Oct** (small rungs + O20, packed) → Sat 24 (Q122) → Tue 27 (O120; latest launch Thu 29, L-LAST) · CPU scoring Sun 1 – Tue 3 Nov · paper items by Fri 6 Nov |
| **Budget** | 0 full runs · **5,600 single-turn replays** (800 prompts × 7 targets) · **3.6–7.0u ≈ the card's 4–7u** unpacked (7.2–13.9 GPU-hours incl. 10% retry margin, nominal 2 GPU-h per u); **2.6–4.8u** if the five TP1 targets are packed onto GPUs idle during the O20 session (card: 2–4u; §4 explains the difference) · CPU scoring ≤ 16 cores, typically ≤ 1 h (worst case ≈ 4 h) · developer effort ≈ 2.1–2.6 d (§5, recomputed; the breakdown books only X2 and E11c for RP1, see §4) |
| **Models** | Targets Q0.8, Q2, Q4, Q9, Q122 and O20 (`replay_<rung>`), plus the O120 self-replay (`replay_O120`, the reference). All prompts come from O120 (A05) |
| **Status** | Not started |
| **Depends on** | A05 seed-42 traces (rows A05-001…010 and A05-031…038; GPU done Sun 18 Oct; `a05-handoff` Mon 19, A05-T16) and G2 (Mon 19) · E3 traces plus the A18 add-ons (A18-T2), in `agent2027-v2.0` (Fri 16) · E4 caller attribution (W's `caller` field; Fri 16) · E1 (`make_manifest.py`, `ro27_session_launch.sh`) · X1 and X2 (Tue 13; X1's acceptance check is A04-T3's) · X10 (Fri 16) · A04: `CLIENTS_SCRIPT` passthrough (A04-T7, Wed 14), verdicts and `scripts/agent2027/rungs.csv` (A04-T10/T18, Fri 16) · E8a detector (Mon 12) · E9d J−a rules (v2.3, Wed 21) · E11b `process_metrics.py` (Tue 27) · E11c scorer (A18-T9; registry window Nov 1–6, needed here by Sun 1 Nov) · E13 restricted builtins (optional; ED1) · A17 (shares the O20 slot and the Q122 session timing) · A20 (O20 slot, `S1022d`/`S1022e`) · A19 (informing only: A18 reads which role needs N B) |
| **Feeds into** | A19 CAP-2 (which role needs N B), A17 CAP-1 (separates transport failures from loop failures at small rungs), the Fig. 3 inset, the "1B vs 100B" answer, the replication package |

*Every "hoped" outcome below is a hypothesis, not a finding. Code refs: RougeOne `nested-cv-rebuttal` @ `07cbf93` (local HEAD `1bdaf42` has identical code for every cited file; re-verified Wed 7 Oct with `git diff 07cbf93 1bdaf42 -- rogue_one_main.py rogueone/ paper/scripts/vllm_session.sh paper/scripts/clients_t16_rerun.sh` = empty), read-only. Short paths: `main.py` = `rogue_one_main.py`; `sci/agent.py`, `ext/agent.py`, `tst/agent.py` = `rogueone/llm/agents/core/{scientist,extractor,tester}/agent.py`; `kn/web_search.py` = `rogueone/llm/agents/core/knowledge/web_search.py`; `clients.py` = `rogueone/llm/clients.py`; `config.py` = `rogueone/utils/config.py`; `pandas_exec.py` = `rogueone/utils/pandas_exec.py`. There is no `cfg.py`. u = one real-data O120 run (10 iterations × 5 folds); for replay GPU time we use the nominal conversion 1u = 2 GPU-hours (2 u/h on the 4-GPU box, before the 0.7 utilisation factor). Roles: S = Scientist, E = Extractor, T = Tester, W = web-search summariser. Shared doc = `docs/ablations/01_shared_infrastructure.md` ("01 §n"); it wins on shared items (flags, metrics, CAIR rules).*

## 1. What we test and why it matters

We take the exact situations O120 faced in M0, one LLM turn at a time. Each turn is sent unchanged to every smaller model, which writes only the next message. We then score, separately for S, E, T and W, whether that message is (1) a well-formed tool call or answer, (2) code that runs, and (3) a usable move.

The result is a per-role capability floor at every rung. It separates **transport** failures (tool-call format) from **code and planning** failures. CAP-1's full-loop runs mix the two, so this is the only place they are separated. It also tells CAP-2 *which* role needs a large model. It costs only serving time, because the prompts already exist in M0's traces.

## 2. Background you need

**How a role turn is produced.** Each role is an Agents-SDK `Agent`. One `Runner` run makes several LLM calls ("turns"). Each turn returns either tool calls (an **action**) or final text (an **answer**). Every call goes through `VLLMCompatibleAsyncOpenAI` (`clients.py:471`).

| Role | Client · model | Tools | Accepts an answer when | Fallback |
|---|---|---|---|---|
| S | `sci/agent.py:397-401` · `:458` | `generic_pandas_tool`, `search_tool`, `attribute_lookup_tool`, `list_all_attributes`, `take_a_note`, `read_notes` (`:461-469`) | text > 10 chars (`:528`); up to 100 retries (`:519`) | no-tool agent (`:547-590`), then heuristic focus (`:501-517`) |
| E | `ext/agent.py:848-852` · `:932` | `generic_pandas_tool`, `append_new_attribute`, `list_known_attributes_tool`, `search_in_literature_tool` (`:935-940`) | R rounds per iteration (`:967`; hard-coded 10 today, R frozen at G1 via E2's `ROGUE_ONE_EXTRACTOR_ROUNDS`). E's final text is never read by the loop (`res` is unused, `:969-976`) | baseline features (`:996-1002`; builder `:599-844`) |
| T | `tst/agent.py:1338-1345` · `:1352` | assessment `:1486-1493`; pruning `:1582-1588` (incl. `attribute_pruning_tool`, `:1269`) | non-empty (`:1387`); 20 attempts (`:1373`) | fallback report (`:1498-1503`); pruning floor (`:1617-1632`) |
| W | global endpoint `kn/web_search.py:188-192`, model `:237-241` | `_web_search` (`:74-75`) | non-empty final output (`:264-265`) | give-up string "Unable to provide an explanation at this time." (prompt `:235`, code `:265`); "Internal error." (`:271`) |

**Where code runs.** E's two code tools call `exec_user_command` (`pandas_exec.py:104-156`, exec at `:139-143`). Its scope holds `df` = all of `data.csv` without labels (`main.py:197`) and `df_features` = the entity table. E's `generic_pandas_tool` first refuses commands that start with `df =` / `df_features =` or do not mention `df` (ILLEGAL guard, `ext/agent.py:333-340`), and accepts a None `result` if the command printed something (`:360-363`). An append is accepted only if all of these hold:
- the name is not already Active (`ext/agent.py:465-489`). A name that exists but is Pruned is **auto-unpruned** without running any code (`:466-485`);
- `result` is set (`:500-504`);
- there are no NaN or Inf values (`:510-519`);
- the result is numeric (`:521-536`);
- the assignment `df_features[args.name] = result` (`:538`) raises nothing. A result whose length differs from `df_features` raises, and the `except` at `:561-567` returns an error string.

Accepted appends are written at `:538`. S's pandas tool (`sci/agent.py:179-270`; command normalisation `:233-240`) runs against `df_test_results` (train metrics plus the Tester report; columns `main.py:399-415`, built at `:384-415`, pooled at `:823-831`) and `df_raw_data` (= `data.csv`; scope `sci/agent.py:245-250`, exec `:252`, `result` required `:254-258`). T's executor needs a fitted XGBoost `assessment_model` (`tst/agent.py:1190-1204`) and returns `output`, not `result`.

**Wrapper (transport).** The wrapper:
- pops `reasoning_effort` (`clients.py:525`);
- forces `enable_thinking=False` (`:204`);
- seeds every request from `ROGUE_ONE_LLM_SEED` (`:211-213`; read at `config.py:330-331`), **but only if the request has no `seed` yet**: an explicit caller-supplied seed wins (`:207-213`). The recorded M0 kwargs already carry `seed` (42 or 1042), so the replay must overwrite the field, not merely set the env var (§4).

With tools present it streams (`:539-542`) through `_streaming_create_with_fallback` (`:564-689`). That path does three things:
- up to 3 empty-response retries and 2 splice re-rolls (`:571-572`, `:618-640`);
- name and argument normalisation (`:655-673`);
- a **JSON-only** text rescue (`:675-687`; parser `:37-108`). Qwen's XML-style calls are never rescued.

Each attempt is one call of the module-level `_stream_to_chat_completion` (`:310`, called at `:586`). It sets `finish_reason="tool_calls"` whenever any tool call was assembled (`:402-404`), whatever the stream reported, and otherwise keeps the stream's reason (default "stop", `:422`). A call cut off by `max_tokens` is therefore caught by the argument-parse test of the scores below, not by `finish_reason`.

Reasoning items are stripped before every call (`sci/agent.py:477-495`, `ext/agent.py:943-961`, `tst/agent.py:1358-1367`, `kn/web_search.py:244-251`). The recorded messages therefore contain no reasoning.

**State for reconstruction.**
- `_reset_fold_state` clears everything fold-scoped (`main.py:417-457`).
- `index` gives `fold_run_id = index // ROGUE_ONE_NUM_ITERATIONS` and `iteration_in_fold = index % ROGUE_ONE_NUM_ITERATIONS` (`:539-540`; = 10 in Full-v2).
- After each trial, `dump_data` writes `intermediate_steps/df_entities_attributes_trail_{index}.csv` (`:839-847`; called at `:715` (dataset abort), `:782` (dataset skip) and `:833` (completed trial)). **The iteration-skipped path** (no productive feature at `iteration_in_fold` > 0, `:639-663`) calls neither `dump_data` nor `save_report` and adds no feedback row (`:646-663`), so such a trial has no trail files and its state equals the latest earlier trail of the same fold. At `iteration_in_fold` 0 the attribute table is the empty canonical table (`:439-449`). Tester reports go to `tester_reports/tester_report_iteration_{index}.md` (`:849-857`), and `df_test_pool_final.csv` is written by `save_data` (`:1385-1394`).
- `df_test_pool_final.csv` does **not** hold everything S saw: the Tester's `as_dict` keeps only `train_accuracy` (classification, `tst/agent.py:530-554`) or `train_R2` (regression, `:400-424`), while S's feedback row averages `train_precision`, `train_recall`, `train_f1`, `train_AUROC`, `train_MSE`, `train_MAE` too from `fold_metrics` (`main.py:388-415`). Hence the per-trial dump of S's feedback pool in A18-T2.
- S and E sessions are keyed by trial (`scientist_agent_session_step_{index}`, `sci/agent.py:499`; `extractor_agent_session_step_{index}`, `ext/agent.py:965`). Tester sessions are keyed by fold (`step=fold_run_id`, `main.py:789-794`; session names `tst/agent.py:1495`, `:1590`), so T's history grows across a fold. The trial index must therefore come from the trace record.

**Traces (E3).** With `ROGUE_ONE_LLM_TRACE=1`, one record per LLM call is written in `_wrap_completions` (`clients.py:516-562`) to `output/llm_traces.jsonl.gz`, and uploaded in the `run-output` artifact. Today `clients.py` has no trace logging (verified: no `trace`/`jsonl` in the file). The record holds the wrapper's **final** response, not attempt 1, so the recorded O120 moves can be scored for "rescued" but not for "raw" validity.

**Serving.** `paper/scripts/vllm_session.sh` owns the server, tears it down on exit (`trap cleanup`, `:79-90`) and calls the clients script once the server is healthy (`:184-189`; it exports only `PORT`, `API_KEY` and `MODEL`, `:185`). It refuses TP > 4 (`:66-69`) and refuses a second `rogueone-vllm-*` server unless `ROGUEONE_ALLOW_SECOND_SERVER=1` (`:121-130`). The command is at `:157-168` (`--served-model-name "$MODEL"`, `:166`, so the served name is the HF id). The default `API_KEY` is `rogueone-nestedcv-key` (`:41`). X1 adds `EXTRA_ARGS` and `--max-model-len`; installed on CAIR as `~/ro27_vllm_session.sh` (01 §3 X1).

**Scores** (R13 names; one row per replayed turn, judged on the move the target actually made):

| Score | Definition |
|---|---|
| `rp1_valid_call_raw` | The attempt-1 streamed completion, before re-rolls, normalisation and rescue, is a well-formed move. **Action:** `finish_reason="tool_calls"`, at least one structured call, and **every** structured call names a tool in that stage's recorded `tools` and has arguments that parse as a JSON object with the schema's `required` keys (the same definition as A04's, so the two pre-registered definitions agree). **Answer:** `finish_reason="stop"`, no tool call, and text longer than the role's own acceptance threshold (S > 10 chars; E, T and W > 0). A turn with several calls passes only if every call passes; the share of multi-call turns is reported. |
| `rp1_valid_call_rescued` | The same test on the wrapper's final response, i.e. what the agent would have seen. This is the **transport score**. |
| `rp1_executes` | Scored only for rescued code-bearing actions: E `append_new_attribute` or `generic_pandas_tool`, and S `generic_pandas_tool`. The command must pass the role's own pre-checks (E `generic_pandas_tool`: the ILLEGAL guard, `ext/agent.py:333-340`; S: the normalisation at `sci/agent.py:233-240`), run on the reconstructed state without exception within 60 s, and set a non-None `result` (E `generic_pandas_tool` also passes if it printed non-empty stdout, `:360-363`). An append whose name auto-unprunes (`:466-485`) runs no code: it is flagged `auto_unprune` and left out of the `rp1_executes` denominator. T executor code is not executed (n/a). With several code-bearing calls in one turn, all must pass. |
| `rp1_usable` | **E, append stratum:** an *accepted* append. It passes the code's own checks above (including the assignment at `:538`), is non-constant, and has \|r\| < 0.999 with every Active feature (= `fun_valid`). An `auto_unprune` append counts as usable iff the name is Pruned in the reconstructed state. **E answer (any stratum):** valid if text > 0, but **never usable**: the loop never reads E's final text (`:969-976`), and an answer ends the round with no feature work. Reported separately as the `answer_instead_of_action` share. **Every other action:** valid, and executes if it carries code (T `generic_python_executor_tool`: valid only, n/a for execution). Lookup and prune names must exist in the attribute table. A W query must be unchanged by `_sanitize_query` (`kn/web_search.py:32-54`) apart from whitespace, and have ≤ 12 words. **S answer:** names at least one raw column or existing attribute; is not a repeat (cosine < 0.9 to that fold's focus history, as `focus_novelty`); and is additive (asks for at least one new feature, per the hard rule at `sci/agent.py:441`). **T assessment answer:** has at least 3 of the 4 required sections (`tst/agent.py:1466-1471`) and names ≥ 50% of the Active features. **T prune call:** every name exists and is Active, and on SYN-P it does not prune an attribute the E8a detector marks as recovering φ. **T pruning-stage answer:** valid only (the summary is stored as the `Prune Rationale`, `tst/agent.py:1604-1606`; no content rule is applied to it here). **W answer:** non-empty and not one of the give-up strings (`kn/web_search.py:235`, `:265`, `:271`). |

**Operational definitions** (frozen in A18-T1, before any score is computed; the lexicons are verifier defaults for Henrik to confirm, and A18-T17 validates the S rule):
- *Names a column or attribute:* a case-sensitive substring match of any raw column name or Active attribute name (from the reconstructed state) in the answer text. *Names ≥ 50% of the Active features:* at least half of the Active names occur as substrings.
- *Additive (S):* the answer contains at least one whole word, case-insensitive, from {add, build, create, introduce, construct, derive, engineer, compute, encode, combine, interact, transform, extract} and is not pruning-only. Pruning-only = it contains a word from {remove, drop, prune, retire, delete} or the phrase "keep only", and no build word.
- *Required section present (T assessment):* a line that starts with `#`, `**` or a list marker and contains, case-insensitively, one of "summary", "feature-by-feature", "risks", "recommended candidates" (one match per section).

**Denominators** (fixed here so the rates are unambiguous):
- `rp1_valid_call_raw`, `rp1_valid_call_rescued` and `rp1_usable`: all common-set turns of that role (or stage) at that target.
- `rp1_executes`: the rescued, code-bearing, non-`auto_unprune` actions of that role at that target (a conditional rate).
- **Role rates pool the role's strata unweighted**, i.e. over the §4 allocation as drawn; this allocation is the estimand. Per-stratum rates are also reported.
- `recorded_O120` (the logged M0 responses) is scored on `rp1_valid_call_rescued`, `rp1_executes` and `rp1_usable` only, because the trace holds no attempt-1 response.

**Descriptive outputs:**
- T prune-list Jaccard against O120's recorded list and against J−a (E9d rules). J−a is applied to the metric table parsed from the recorded pruning request (`initial_input`, `tst/agent.py:1572-1577`; columns from `_metric_profiles_to_markdown`, `:692`: `mutual_information`, `vif`, `pfi_mean`, `pfi_std`, `noise_sensitivity`, `mean_abs_shap`);
- the `answer_instead_of_action` share per role and target, and the share of multi-call turns;
- `context_errors`, `max_prompt_tokens`, `rerolls`;
- prompt and completion tokens.

## 3. Hypotheses and interpretation (all hypotheses)

No result exists yet; X and Y below are placeholders filled only from the RP1-D1…D3 read-out. Verdicts are read with the pre-registered rules in §6.

- **Hoped** (RP1-D2 "Hoped"). Floors differ in the order E ≫ S > T. T and S are usable from about Q2; E needs Q4–Q9. The pre-registered rule D2 is weaker than this prose: it needs floor(E) ≥ floor(S) ≥ floor(T) with E at least 2 rungs above T, not "E ≫ S" or "S > T".
  - Paper sentence: *"Code synthesis, not planning or critique, sets the capability floor: a 0.8B model makes valid Tester and Scientist moves in X% of replayed turns but writes accepted feature code in only Y%."*
- **Null** (RP1-D2 "Null"). All roles share one floor.
  - If that floor is ≤ Q0.8, CAP-2's 1B cell is predicted to pass. If it is above Q0.8, the 1B cell is predicted to fail (RP1-D4).
  - Either way, per-role allocation saves nothing beyond choosing the cheapest viable rung. Report exactly that.
- **Opposite.** Two forms:
  - (a) (RP1-D2 "Opposite") E's floor is below S's or T's. "Small planner, big coder" is then backwards, and the paper reports it.
  - (b) (RP1-D3b, defined in §6) Transport fails at every small Qwen rung, but the parsed calls execute well. The bottleneck is then the tool-call format or parser, not the model. Report it as a harness finding (C6-type), per family, with the JSON-only rescue disclosed.
- **Uninformative outcomes** are reported as such, not reinterpreted: a replay-path defect (RP1-D5), a "Partial" order verdict, or unstable floors (bootstrap floor-stability share < 0.6, RP1-D2; the share is reported next to every floor).

## 4. Design

> **SR4 fallback (2026-10-19).** O20 failed Stage 0 and Part VIII.1 (item 15) drops `replay_O20`:
> A18-005 is not run (its number stays a gap; port 8022 is not used by A18). The packed Thu 22
> slot next to an O20 session no longer exists. No rule of §6 changes. Plan:
> - **Manifest:** `manifests/agent2027/A18_RP1.csv` (`scripts/agent2027/manifest_specs/A18.yaml`),
>   rows A18-001…004, 006, 007; `dataset_id=rp1set`, `task_type=mixed` (admitted by
>   `make_manifest.py` for `rp1set` only; `ro27_clients_manifest.sh` refuses such rows),
>   `llm_seed=7042`, `image_tag=agent2027-v2.0` (its `clients.py` is the frozen wrapper the
>   source prompts went through; v2.3 changed `clients.py`, which would need A18-T10's review).
> - **Single-GPU targets as clients of a running Q-rung server.** A replay is one CPU container
>   (`replay_tool.py`, concurrency 8) against an OpenAI endpoint; it needs no GPU of its own.
>   Under D12 the four TP1 targets ride on servers that already run for other rows: A18-003
>   `replay_Q4` on the GPU 7 Q4 server (`S1019g`, port 8034) or A19's Q4 holder (`S1022d`, 8032);
>   A18-001 `replay_Q0.8` on `S1019h` (8035) or A19's Q0.8 holder (`S1022f`, 8030); A18-002
>   `replay_Q2` on `S1019i` (8036); A18-004 `replay_Q9` on `S1019j` (8037). A18-006
>   `replay_Q122` rides on a Q122 TP2 server (A19 `S1022a`, 8040, or A17 `S1024a`, 8041), and
>   A18-007 `replay_O120` on any O120 TP2 server of the A12/A16 sessions. Each target is one
>   `docker run` of `replay_tool.py` with `--port <host port> --model <HF id> --seed 7042
>   --concurrency 8 --clip-max-tokens 131072`, started while that server's session is up
>   (the call fails over to `infra_error` if the server is torn down mid-replay; A18's
>   re-queue rule then applies). Disclosed: the replay shares the server with live rows (load
>   affects wall time only, not the request). If no host session is up, the target runs as its
>   own one-GPU session through `rp1_clients.sh` (A18-T11's unpacked form).
> - **Not runnable yet:** `rp1_score.py sample` (A18-T6), `rp1_clients.sh` (A18-T7) and the
>   `rp1-prompts:v0` artifact (A18-T10) do not exist, so no replay can start; nor can the
>   piggy-back form, which needs `rp1-prompts:v0` too. Slot: from Thu 22 (Henrik's order).
> - **Update (2026-10-19, later):** built on branch `agent2027/a18`: `rp1_score.py` (`sample`,
>   `fetch`, `score`, `attach-log`), `rp1_clients.sh` (CLIENTS_SCRIPT form and stand-alone
>   piggy-back with `HOST_SID`), `replay_tool.py --validity a18`. `rp1-prompts:v0` is logged
>   (800 records, every cell at its n, no fill, no ITT substitution, A18-T5 heavy QC passed on all
>   18 runs; drawn in a CPU container from a `git archive` of commit `30ed810`, since the branch
>   was not pushed). Launch lines: `docs/agent2027/sessions/LAUNCH.md` §6. Still open: `floors`
>   and `publish` (A18-T18), E11b's novelty embedder (`score` uses TF-IDF cosine meanwhile,
>   flagged `novelty_method=tfidf`).

**Control and arms.** The control is the O120 self-replay (`replay_O120`). It gives fresh samples, because each recorded response is only one draw. The arms are `replay_Q0.8`, `replay_Q2`, `replay_Q4`, `replay_Q9`, `replay_Q122` and `replay_O20`. Q27 and Q35-A3 are cut.

**Prompt source.** Every A05 row at `fold_seed=42`, `replicate=0`, `llm_seed=42`: SYN-P 8 (`synp_p1`, `synp_p2`, `synp_p3`, `synp_p4`, `synp_p5`, `synp_p6a`, `synp_p6b`, `synp_p7`; rows A05-031…038) and Core-10 (rows A05-001…010): 18 runs, 90 fold-episodes. `synp_p0` (A05-055) is not used.

Excluded:
- calls of S's no-tool fallback agent (`sci/agent.py:547-590`). It shares S's client (`agent_name="ScientistAgent"`), so it is identified by an empty `tools` list in the recorded request (`:564`);
- calls with no recorded response (HTTP error);
- E answer turns (final text): the loop never reads them (§2), so they are not sampled.

If a source run is an ITT failure, use its A/A twin (`replicate=1`) and record the substitution.

**Allocation** (by the recorded O120 move; within each cell, round-robin over the 18 runs, then over the iteration buckets {0–2, 3–6, 7–9}; RNG seed 42; W also balanced by caller):

| Role / stage | Strata | n |
|---|---|---|
| S | answer (focus) 80 · action 120 | 200 |
| E | `append_new_attribute` 200 · other action 100 | 300 |
| T assessment | answer (report) 40 · action 60 | 100 |
| T pruning | `attribute_pruning_tool` 60 · answer 40 | 100 |
| W | `_web_search` 60 · answer 40 | 100 |
| **Total** | | **800** |

If a cell runs short, fill it from the same role's other cells and log the actual counts.

**Fixed settings.**
- The recorded request is re-sent unchanged (`messages`, `tools`, `tool_choice`, `extra_body`, `temperature` 0.7, and the recorded per-role `max_tokens`: E 32768 (`EXTRACTOR_AGENT_MAX_TOKENS`), T 8192 (`TESTER_AGENT_MAX_TOKENS`), S as pinned at G1 (`SCIENTIST_AGENT_MAX_TOKENS`, applied by E2; today S passes none, `sci/agent.py:404-407`), W none (`kn/web_search.py:195-198`)). Only two fields change:
  - `model` becomes the target HF id (from `rungs.csv`; after SR1, the Qwen3 fallback id, while the arm keeps its rung name);
  - `seed` is **overwritten** with 7042 in the request (the recorded `seed` is dropped first, because a caller seed wins over `ROGUE_ONE_LLM_SEED`, `clients.py:207-213`). 7042 differs from M0's 42 and 1042, so the O120 self-replay is a fresh sample.
- If prompt + `max_tokens` exceeds 131,072 target tokens, X2 clips `max_tokens` to fit (minimum 1,024) and flags the turn `clipped`. Prompt tokens are counted with the target's own chat template (e.g. the server's `/tokenize` endpoint with `messages` and `tools`; to verify in X2). If the prompt alone exceeds the limit, the turn is a `context_error`.
- One sample per prompt per target, at concurrency 8 (`RP1_CONCURRENCY`), in a fixed shuffled order (seed 42). Concurrency changes only batching, not the request; it may be raised to 32 if A18-T4's projection (from A04's per-rung 50-prompt replay times) gives > 6 h for a target, and the value is logged.
- Per-call timeout 600 s. W has no recorded `max_tokens` (and S has none if G1 pins none), so a degenerate small model could otherwise generate to the context limit. A call that times out while the server is healthy (`/health` = 200 straight afterwards) is an **invalid move** (`timeout`, counted per target, like `finish_reason=length`). Only HTTP 5xx, connection resets and an unhealthy server are infra errors (§4 failure handling).
- **No tool is executed during replay:** no Serper calls and no `exec`.
- Serving: `vllm/vllm-openai:v0.22.0` with `MAX_MODEL_LEN=131072`, and per-rung flags and revisions from `rungs.csv` (A04). Qwen set per R9; Qwen thinking is off (`clients.py:204`).
- Client image `rogue_one:agent2027-v2.N` (v2.3 or v2.4 on Thu 22; current tag later). The wrapper code it replays through must equal v2.0's (A18-T10 check).

**Run matrix** (one row = one target; run name `A18__replay_<rung>__rp1set__f42__r0`; W&B project `RogueOne-AGENT2027`, group `A18`):

| Row | arm_id | HF id | TP · port | Session (planned) | When | Replays | GPU-h | u |
|---|---|---|---|---|---|---|---|---|
| A18-001 | `replay_Q0.8` | `Qwen/Qwen3.5-0.8B` | 1 · 8030 | S1022p | Thu 22, packed | 800 | 0.35–0.77 | 0.2–0.4 |
| A18-002 | `replay_Q2` | `Qwen/Qwen3.5-2B` | 1 · 8031 | S1022q | Thu 22, packed | 800 | 0.35–0.77 | 0.2–0.4 |
| A18-003 | `replay_Q4` | `Qwen/Qwen3.5-4B` | 1 · 8032 | S1022r | Thu 22 – Fri 23, packed | 800 | 0.35–0.77 | 0.2–0.4 |
| A18-004 | `replay_Q9` | `Qwen/Qwen3.5-9B` | 1 · 8033 | S1022s | Thu 22, packed | 800 | 0.35–0.77 | 0.2–0.4 |
| A18-005 | `replay_O20` | `openai/gpt-oss-20b` | 1 · 8022 (8020 if run after the O20 session ends) | S1023p | Thu 22 – Fri 23, packed | 800 | 0.35–0.77 | 0.2–0.4 |
| A18-006 | `replay_Q122` | `Qwen/Qwen3.5-122B-A10B-FP8` | 2 (4 if SR3 moved Q122 to TP4) · 8040 | S1024p | Sat 24, before A17's Q122 session (moves with it; latest Wed 28) | 800 | 2.4–4.4 | 1.2–2.2 |
| A18-007 | `replay_O120` (control) | `openai/gpt-oss-120b` | 2 · 8011 (8010 if no other O120 server is up) | S1027p | Tue 27 (latest launch Thu 29) | 800 | 2.4–4.4 | 1.2–2.2 |
| **Total** | | | | | | **5,600** | **6.6–12.7** | **3.3–6.3** |

Per-row u values are rounded; the total is computed from the GPU-hours (6.55–12.65 GPU-h ÷ 2). Under ED2 only A18-001, A18-003, A18-004 and A18-007 are run (§6). If A04 drops a rung (SR2 boot failure, SR3) or SR1 switches the family, the row keeps its number and arm ID and is dropped or re-pointed through `rungs.csv` (A18-T4). Session letters and dates in the table follow the actual launch (new names, §8).

**Budget arithmetic.**
- A TP1 target takes 15–40 min of replay plus about 6 min of load: 21–46 min = 0.35–0.77 GPU-h.
- Q122 and O120 each take 1–2 h plus about 10 min of load on 2 GPUs: 2.3–4.3, rounded up to 2.4–4.4 GPU-h.
- Unpacked: 5 × (0.35–0.77) + 2 × (2.4–4.4) = 6.55–12.65 GPU-h, plus a 10% retry margin = **7.2–13.9 GPU-h = 3.6–7.0u** at 2 GPU-h per u. This is the card's 4–7u.
- Packed: the five TP1 targets ride on GPUs that the CPU-bound O20 session leaves idle, so only Q122 and O120 count. That is 4.8–8.8 GPU-h, plus 10% = 5.3–9.7 GPU-h = **2.6–4.8u**. The card says 2–4u; the difference is this document's 10% retry margin and the rounded-up load time.
- Charged against the 0.7-utilisation capacity of 01 §4.9 (638u gross for 1,824 GPU-h, i.e. ≈ 2.86 GPU-h per u), the same GPU time is 2.5–4.9u unpacked. We keep the nominal (higher) figure for planning.
- **Timing risk.** The 15–40 min per TP1 target assumes most replies end well before `max_tokens`. If a small model often runs to E's 32,768-token limit, a target can take several hours. A18-T4 projects each target's wall time from the 50-prompt replay that A04 logs per rung (`A04__stage0_<rung>__replay50__f42__r0`, concurrency 1; O120 from the X2 check in A18-T3): projected wall time = 4 × the 50-prompt time (800 prompts = 16 × 50, at concurrency 8 with an assumed 4× batching gain). The session cap (6 h), the 600 s per-call timeout and the concurrency rule above bound it.
- **Developer time vs the breakdown.** The task efforts in §5 sum to 2.06 d, or 2.1–2.6 d with E11c's 0–0.5 d. The breakdown books only X2 (0.25–0.5 d, shared with A04) and E11c (0–0.5 d) for RP1. A18's own code and install work (T2 0.1 + T6 0.15 + T7 0.2 + T8 0.05 + T10 0.1 = 0.6 d, falling Fri 16 – Wed 21 Oct) is not in the breakdown's 18–20.5 developer-days to Oct 27, and Tue 20 – Wed 21 already carry E9a, E9b, E7 and E9d. This is an unresolved load issue, not a saving.
- Q122 at TP4 (SR3) doubles its GPU-hours (+1.2–2.2u, from the reserve). ED2 (4 targets): 3 × (0.35–0.77) + (2.4–4.4) = 3.45–6.71 GPU-h, + 10% = 3.8–7.4 GPU-h = **1.9–3.7u**.
- CPU: at most 300 E and 200 S prompts × 8 scored targets (7 + `recorded_O120`) = 2,400 E and 1,600 S candidate commands. Typically < 1 s each plus state rebuild, so ≤ 1 h on 16 cores; the worst case (every command hits the 60 s timeout) is ≈ 4,000 × 60 s / 16 ≈ 4.2 h.
- O120 runs at TP2 even if decision D5 (01 §6.4) picks TP4. A04 session C does the same; it is disclosed.
- **Calendar vs the breakdown.** The T1-minimum calendar puts the RP1 small rungs in W2 (packed) and Q122/O120 in W3. Here Q122 is Sat 24 (end of W2) to share A17's Q122 slot (A17 Q122 Sat 24 – Wed 28); if A17's Q122 session starts later, A18-006 moves with it.
- **Box contention (unresolved across documents).** A12's `S1021a` is planned to run until Thu 22 evening – Fri 23 afternoon, and `S1023a` (A12 block 2 + A16) from Fri 23 until Sat 24 midday – Sun 25 (A12 §4 and §7, A19 §7). `S1021a` holds the 4-GPU O120 slot, or 2 GPUs if decision D5 picks TP2; `S1023a` presumably likewise (to verify in A12). The Thu 22 packed slot assumes that the O20 session (A17/A20) runs then, and the Sat 24 morning Q122 slot assumes `S1023a` has ended. The T11/T12 pre-flight decides on the day. If there is no O20 session or no free GPU, run the small rungs as an unpacked block of up to 4 parallel TP1 sessions (+1–2u, still inside 4–7u) in the first free window before L-LAST (Thu 29), and Q122 right before A17's Q122 session. Henrik places the slots in the session plan by G2 (Mon 19).

**Pairing and randomisation.** Every target sees the same 800 prompts, so all comparisons are paired by prompt. Only the server differs between targets. Sessions are replay-only and get no sentinel (A06).

**Failure handling.**
- **Infra error** (HTTP 5xx, connection reset, or a call that exceeds 600 s while the server is unhealthy): retried once at the end of the same session with the same seed. If it still fails, mark it `infra_error`. If more than 5% of a target's prompts fail this way, rerun that target's session once, from the reserve. A 600 s call on a healthy server is a `timeout`, an invalid move (§4 fixed settings), not an infra error.
- **Context error** (HTTP 400, context length): excluded from the **common set**, the prompts with no context error and no infra error at any target. Counted per target, and the excluded share is reported per role. Sensitivity analysis: count each target's own context errors as invalid moves on all of its prompts (no common set), because the longest, latest-iteration prompts are the ones dropped.
- **`finish_reason=length`:** an invalid move, not an exclusion. A sensitivity analysis drops `clipped` turns.
- **Server fails to boot:** rerun once. If it fails again, drop the target and apply A04's SR rules.
- **Session cap:** `RUN_CAP_HOURS=6`. Unfinished prompts are re-queued once in a new session.
- There are no Rogue One runs, so there is no run-level ITT.

## 5. Tasks

`$CAIR` = `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14`.

**CAIR rules for every step** (01 §5.1; the shared doc wins on these):
- **Containers only.** Every execution (replay, scoring, W&B queries) runs inside Docker. The host issues only `docker` commands, the `~` launchers that issue them, `git fetch`/`checkout` and `cp`/`mkdir` for installs (01 §5.2 steps 0 and 3), and `nvidia-smi` / `docker ps` / `tail` / `grep` for pre-flight and monitoring. Never `python …` on the host. `docker compose` is not used here; if it ever is, `docker compose run` needs `-T` over SSH.
- **GPU pre-flight before every GPU container** (command in A18-T11). If GPUs are short, **stop and report** per-GPU memory and utilisation, the compute apps and the running containers, and let Henrik decide. Launch with explicit `GPUS=<ids>` and `WAIT_MAX_MIN=0`.
- **≤ 4 of the 8 H100s**, counting every Rogue One server that is up, never more without Henrik's explicit approval in chat. Packing A18 servers next to the Thu 22 O20 session is the one sanctioned exception to "one Rogue One job at a time" (01 §6.2 Thu 22; A17 §4; A20 §5), and only while ΣGPU ≤ 4. A18 servers never take ports 8020/8021, which the O20 session uses.
- **W&B is the system of record.** Runs are stateless: every reply set, score table and server log is uploaded as a W&B artifact; `~/ro27_a18_work/` and `~/ro27_a18_logs/` are scratch copies. The key is read from `<CAIR_HOME>/code/PlanBench/.env` as `paper/scripts/clients_t16_rerun.sh:59` does; it is never written to a file or a URL.
- **Long jobs are detached** (`nohup … &`), with the job log in `~` (`~/ro27_<SID>.job.log`), never in a root-owned run folder.
- **Code reaches CAIR** only by committing and pushing `agent2027/v2` locally, then running `git fetch` on the server over SSH (A18-T10).

### 5.1 Prerequisites (verify before launch)

- [ ] **A18-T1** [type: decision] (0.05 d, due Fri 16 Oct, inside X10) — Pre-register RP1:
  - the targets, source runs and allocation table (§4);
  - `llm_seed` 7042;
  - the score table, the operational definitions and lexicons (§2), and the E-answer rule;
  - the common-set rule, the timeout rule and the context-error sensitivity analysis (§4);
  - rules RP1-D1…D6, the floor-stability threshold 0.6 and the D3b minimum denominator 30 (§6);
  - the teacher-forcing disclosure.

  *Done when:* all of it is in the commit tagged `agent2027-prereg`.
- [ ] **A18-T2** [type: code] (0.1 d, up from 0.05 d; due Fri 16 Oct, merge-ready Thu 15 so that the Fri 16 morning merge carries it; must be in `agent2027-v2.0`) — Make sure E3 carries what A18 needs (E3's own due date is Mon 12; the 01 §3 spec of E3 lists a trace per LLM call but not these fields, so confirm them with E3 before G0a):
  - every trace record has `agent_name`, `caller` (for W; from E4's caller attribution), `stage` (T assessment or pruning), `index`, `fold_run_id`, `iteration_in_fold`, `session_name`, `call_seq`, the full request kwargs as sent after `_merge_extra_body_for_vllm` (`messages`, `tools`, `tool_choice`, `extra_body`, `temperature`, `max_tokens`, `model`, `seed`) and the wrapper's final response (`message`, `finish_reason`, `usage`). Implementation hint: `index`, `fold_run_id` and `iteration_in_fold` from a contextvar set in `forward_pass` (`main.py:536-540`); `stage` and `session_name` from the Tester's `_run_agent_with_retries` arguments (`tst/agent.py:1329-1336`; session names `:1495`, `:1590`); `call_seq` = a per-`session_name` counter;
  - `dump_data` (`main.py:839-847`) also writes `df_attributes_explanations_trail_{index}.csv` (the attribute table with `Status`) and `df_scientist_feedback_trail_{index}.csv` (`df_scientist_feedback_pool`, i.e. S's `df_test_results`, `main.py:828-831`). `dump_data` is not called on the iteration-skipped path (§2); the scorer handles that (A18-T9), so T2 does not change that path.

  *Done when:* the v2.0 freeze smoke run (01 §2.2, Fri 16 afternoon) has these fields in 100% of trace records, and both per-trial tables exist for every trial that called `dump_data`. If a field or table is missing at the freeze: the scorer rebuilds `Status` from the trace's pruning calls and flags floor restorations as unknown; S's table is rebuilt from `df_test_pool_final.csv` (`id`, `status`, `train_accuracy` or `train_R2`) plus `tester_reports/` (`feature_assessment`), and an S command that reads any other column is flagged `reconstruction_partial`.
- [ ] **A18-T3** [type: decision] (0.02 d, due Tue 13 Oct) — Check X1 and X2 (`scripts/agent2027/replay_tool.py`). *Done when* all of these hold:
  - X1: A04-T3's X1 check holds (Q4 serves with the Qwen set; the server command line carries `--max-model-len 131072`);
  - X2 replays the 50 A04 prompts against O120 through the `clients.py` wrapper (`VLLMCompatibleAsyncOpenAI`, `clients.py:471`);
  - it records the attempt-1 (raw) response, the wrapper's final response, `attempts`, `finish_reason`, usage and context errors. Attempt 1 can be captured without editing `clients.py` by wrapping the module-level `_stream_to_chat_completion` (`clients.py:310`, called at `:586`) in the replay process (to verify);
  - it overwrites the request `seed` (a recorded `seed` must not survive);
  - it accepts `--model`, `--port`, `--seed`, `--concurrency` and `--clip-max-tokens`, reads prompts from a W&B artifact and writes the replies as a W&B artifact (01 §3 X2 lists none of these options, so check each one);
  - the 50-prompt wall time against O120 is logged (it feeds A18-T4's projection).

  If X2 cannot replay the 50 prompts by Tue 13, apply **ED2** (§6). If it works but lacks some of the options above, A18-T7 adds them to `replay_tool.py` (≤ 0.1 d); ED2 is then not triggered.
- [ ] **A18-T4** [type: decision] (0.04 d, up from 0.02 d; due Fri 16 Oct) — Read A04's outputs: the verdicts SR1–SR5, `rungs.csv` and the `CLIENTS_SCRIPT` passthrough (A04-T7). Project the replay times. *Done when:*
  - `scripts/agent2027/rungs.csv` is committed with revisions and `extra_args`;
  - A04-T7's done-when holds on CAIR (the installed `ro27_session_launch.sh` honours `CLIENTS_SCRIPT`, and behaves as before when it is unset);
  - the A18 target list is recorded in the pre-registration;
  - each target has a projected wall time (§4 *Timing risk*: 4 × the 50-prompt time from its `A04__stage0_<rung>__replay50__f42__r0` run in project `RogueOne-AGENT2027-prefreeze`; O120 from A18-T3). A target that projects > 6 h gets `RP1_CONCURRENCY=32`, and the choice is recorded.

  If SR1 fired, the targets switch to the Qwen3 ids (Q0.8 → `Qwen/Qwen3-0.6B` and so on, R9 fallback ladder) and keep their rung arm IDs. If SR3 dropped Q122, A18-006 is dropped and the Qwen ladder ends at Q9. If SR4 failed O20, A18-005 is still run unless O20 failed to boot. In that case A17 and A20 no longer run an O20 session, so the packed slot does not exist and the small rungs run as the unpacked block (A18-T11, last bullet). A rung that failed to boot is dropped and the drop is recorded.
- [ ] **A18-T5** [type: data] (0.05 d, Mon 19 Oct, after G2 and A05-T16) — Light QC of the source traces. The scorer and E11b's `fetch_wandb.py` (due Tue 27) do not exist yet, so use a `wandb.Api()` query in a container (01 §5.2 step 7 form) and fetch artifact `a05-handoff` (the trace index, A05-T16). *Done when*, for all 18 source runs (rows A05-001…010 and A05-031…038): the run is `finished`, artifact `run-output` exists, and the summary has `llm_calls_total` > 0. The heavy checks run as assertions at the start of `rp1_score.py sample` (A18-T6, run in A18-T10): the trace record count equals `llm_calls_total`, the T2 fields are in 100% of records, the T2 tables exist for every trial that called `dump_data`, and each stratum of the §4 allocation has at least its target n available (else the shortfall is logged and the §4 fill rule applies).

  If a source run is an ITT failure, substitute its A/A twin (`replicate=1`, rows A05-021…030 or A05-047…054) and record it. If M0 is rerun after a G2 defect, A18 waits for the new traces.

### 5.2 Implementation

- [ ] **A18-T6** [type: code] (0.15 d, due Tue 20 Oct) — Add `sample` to `scripts/agent2027/analysis/rp1_score.py`. It first runs the heavy QC assertions of A18-T5, then draws the §4 allocation from the 18 traces (seed 42). Each record stores:
  - the request kwargs and the recorded O120 response;
  - the role, stage and stratum;
  - the run, fold, `index` and `call_seq`;
  - pointers to the state files.

  *Done when:* unit tests on a fixture trace pass: the allocation counts match §4 exactly, two runs with seed 42 give identical output, the fill rule triggers and is logged when a cell is short, and a record with a missing T2 field fails the QC assertion. The production run (artifact `rp1-prompts:v0`) is part of A18-T10, because the code reaches CAIR only then.
- [ ] **A18-T7** [type: code] (0.2 d = 0.1 d script + up to 0.1 d for any option X2 lacks, see A18-T3; due Wed 21 Oct) — Write `scripts/agent2027/rp1_clients.sh`, installed as `~/ro27_rp1_clients.sh` and called through A04's `CLIENTS_SCRIPT` passthrough in `ro27_session_launch.sh` (A04-T7). It runs in the foreground of `~/ro27_vllm_session.sh` (which tears the server down after it returns) and reads `PORT`, `API_KEY`, `MODEL` (exported by `vllm_session.sh`, `:185`) and, inherited from the launch command's environment, `SESSION_ID`, `SESSION_FILE`, `SERVER_NAME`, `IMAGE_RO`, `RUN_CAP_HOURS` and `RP1_ROW` (check against A04-T7's launcher that none of them is scrubbed). `<rung>` and the image tag come from row `$RP1_ROW` of `$SESSION_FILE` (`arm_id` = `replay_<rung>`, `image_tag`). The script:
  1. reads the W&B key as `clients_t16_rerun.sh:59` does and runs `docker run --rm --name ro27-<sid>-<row> --network host --cpus 4 -e WANDB_API_KEY -e WANDB_ENTITY=PlanBench -e WANDB_PROJECT_NAME=RogueOne-AGENT2027 -e LLM_API_KEY=$API_KEY -e ROGUE_ONE_LLM_SEED=7042` (names lower-case) with image `$IMAGE_RO`, `--entrypoint python3`, and `<ROGUEONE_CAIR_PATH>/scripts/agent2027` mounted read-only at `/workspace/scripts/agent2027`. The cap is a watchdog that issues `docker stop ro27-<sid>-<row>` after `RUN_CAP_HOURS` hours and is killed on normal exit, as E1c does. `timeout` alone is not enough: `python3` is PID 1 in the container and ignores SIGTERM, unless `--init` is also given. Inside, it calls `replay_tool.py --prompts rp1-prompts:v0 --model $MODEL --port $PORT --seed 7042 --concurrency ${RP1_CONCURRENCY:-8} --clip-max-tokens 131072`. The run is named `A18__replay_<rung>__rp1set__f42__r0` (group `A18`, job_type `replay_<rung>`, tags [`<SESSION_ID>`, `v2`, `<image_tag>`]); its config holds the R12 identity, version and model keys (`task_type=mixed`, `rung_S/E/T` = the rung, `model_S/E/T/W` = the target HF id, `parallel` = concurrency) and `flags/ROGUE_ONE_LLM_SEED=7042` (the other R3/R4 flags do not apply: no Rogue One process runs); its summary holds `rp1_valid_call_raw__<role>`, `rp1_valid_call_rescued__<role>`, `context_errors`, `max_prompt_tokens`, `rerolls` and the `infra_error`/`clipped`/`timeout` counts (validity needs only the recorded `tools`, so it is scored online); it uploads artifact `rp1-replies-<rung>`;
  2. re-queues infra errors once, at the end, with the same seed;
  3. before returning, writes `docker logs $SERVER_NAME > ~/ro27_a18_logs/vllm_server_<SID>.log 2>&1` and uploads it to the same run.

  *Done when:* `bash -n` passes, `DRY_RUN=1` prints the three steps, and a 5-prompt local dry run against a stub OpenAI-compatible server logs a run (`WANDB_MODE=offline` is acceptable) with the keys above.
- [ ] **A18-T8** [type: config] (0.05 d, due Wed 21 Oct) — Run `make_manifest.py --ablation A18` to write `manifests/agent2027/A18_RP1.csv`, rows A18-001…007, with these values:
  - `ablation_id=A18`, `arm_id=replay_<rung>`, `dataset_id=rp1set`, `task_type=mixed`, `fold_seed=42`, `llm_seed=7042`, `replicate=0`, `is_sentinel=0`, `session_id` empty, `image_tag=agent2027-v2.<N>` (the tag current at launch);
  - `rung_S=rung_E=rung_T=<rung>`, `port_S=port_E=port_T=<port per §4>`, `env_overrides={}`, `cap_hours=6`;
  - `expected_u` 0.3 (TP1 targets) or 1.7 (Q122, O120);
  - `notes=replay-only`.

  Also create one session file per row (`make_manifest.py --plan-session <SID> --rows A18-00n`). Replay-only sessions carry no sentinel row (A06 §4; if `--plan-session` adds one by default, use its no-sentinel option, name to verify in E1). Session letters and dates follow the actual launch. Under ED2 plan only A18-001, -003, -004 and -007. *Done when:* the files are committed and the R7 header validates.
- [ ] **A18-T9** [type: code] (E11c, 0–0.5 d; may start Tue 27 Oct, due Sun 1 Nov) — Add `fetch`, `score`, `floors` and `publish` to `rp1_score.py`.
  - **Fold-state reconstruction** (trial `index`, fold from `fold_run_id`). "Latest trail" below means the trail file of the highest trial `k` < `index` in the same fold that exists. That is `k = index−1` unless that trial was iteration-skipped, which writes no trail files (§2).
    - **E:** `df` is `data.csv`. `df_features` is `df_entities_attributes_trail_{k}.csv` of the latest trail (the id-only table if there is none, i.e. at `iteration_in_fold` 0 or after only skipped trials, `main.py:433-437`), plus this session's accepted appends (and auto-unprunes) re-executed in `call_seq` order up to the replayed call. The Active set comes from `df_attributes_explanations_trail_{k}.csv` of the latest trail (the empty canonical table if none, `main.py:439-449`) plus those appends.
    - **S:** `df_test_results` is `df_scientist_feedback_trail_{k}.csv` of the latest trail (empty at `iteration_in_fold` 0); fallback per A18-T2. `df_raw_data` is `data.csv`.
    - **T:** the Active set is E's end state for trial `index`, minus names pruned by earlier `attribute_pruning_tool` calls in the same pruning session (`call_seq` order). The pruning floor (`tst/agent.py:1617-1632`) acts only after the pruning run ends, so it does not alter the Active set seen by any call.
    - The actual outcome of a recorded call is read from its tool-result message in the next call of the same session (`call_seq + 1`).
  - **Prerequisite checks** (before scoring): E9d's rules function and the E8a detector import in the image; the detector exposes a per-feature `rec` flag (01 §3 E8a lists only per-episode outputs; if so, add a per-feature mode here); the embedder of `process_metrics.py` loads offline (see A18-T16); on 5 turns J−a runs on a metric table parsed from a recorded pruning request.
  - **Scoring:** the §2 rules and denominators. Generated code runs in a subprocess with a 60 s timeout and a 4 GB memory cap, using E13's restricted builtins if E13 landed. A command that mentions `labels`, `open(`, `import os` or `subprocess` is still executed but flagged `suspicious`, and `rp1_executes` is also reported without those commands (the score container mounts `labels.csv`, see §7). J−a comes from E9d; the focus embedding from `process_metrics.py`; φ from `recovery_detector.py`.

  *Done when:* unit tests pass on 10 hand-labelled turns that cover every role and stage; the recorded O120 responses, scored as pseudo-target `recorded_O120`, reach `rp1_valid_call_rescued` ≥ 0.95; and their E append acceptance matches the logged outcome of those same calls (next tool-result message) within 2 pp.

### 5.3 Execution on CAIR

> **SR4 fallback (2026-10-19).** A18-T11's packing around the O20 session (`S1022p`–`s`,
> `S1023p`, port 8022) is void; A18-005 is dropped. The TP1 targets run as client containers of
> the small-rung servers listed in §4 (GPU 7 chain `S1019g`–`S1019j`, or A19's holders
> `S1022d`/`S1022f`); A18-T12 rides on A19's or A17's Q122 server instead of a separate
> `S1024p`; A18-T13 rides on an O120 TP2 server. All wait for A18-T6/T7/T10. Command template:
> `docs/agent2027/sessions/LAUNCH.md` (section A18).

- [ ] **A18-T10** [type: run] (0.1 d, up from 0.05 d; Wed 21 Oct) — Commit and push `agent2027/v2` locally (T6–T8 files), then sync and install on CAIR (01 §5.2 steps 0 and 3), and draw the prompt set:
  ```
  $CAIR "cd <ROGUEONE_CAIR_PATH> && export GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15' && git fetch git@github.com:HenrikNorkart/RougeOne.git tag agent2027-v2.0-freeze && git fetch git@github.com:HenrikNorkart/RougeOne.git agent2027/v2 && git checkout -B agent2027/v2 FETCH_HEAD && git log -1 --format='HEAD: %h %s' && git diff --stat agent2027-v2.0-freeze HEAD -- rogueone/llm/clients.py && cp scripts/agent2027/ro27_session_launch.sh scripts/agent2027/ro27_clients_manifest.sh scripts/agent2027/ladder_session.sh ~/ && cp paper/scripts/vllm_session.sh ~/ro27_vllm_session.sh && cp scripts/agent2027/rp1_clients.sh ~/ro27_rp1_clients.sh && mkdir -p ~/ro27_a18_logs ~/ro27_a18_work ~/ro27_manifests && cp -r manifests/agent2027/. ~/ro27_manifests/"
  $CAIR "docker image inspect rogue_one:agent2027-v2.<N> --format '{{.Id}}'"
  ```
  (The tag is fetched first so that `FETCH_HEAD` is the branch.) If the image tag does not exist yet, build it as in 01 §5.2 step 2 (`test ! -e .dev.env && docker build -t rogue_one:agent2027-v2.<N> .`). `<ROGUEONE_CAIR_PATH>` = `<CAIR_HOME>/code/RougeOne` is **inferred; verify it** with `$CAIR "ls -d ~/code/RougeOne"` (A04-T1 does this check by Wed 7 Oct; reuse its result if it printed the path).

  Then draw the prompt set in a container, detached because it downloads the 18 `run-output` artifacts (flag names are fixed in A18-T6):
  ```
  $CAIR "mkdir -p ~/ro27_a18_work && nohup docker run --rm --name ro27-a18-sample --cpus 8 --env-file <CAIR_HOME>/code/PlanBench/.env -v \$HOME/ro27_a18_work:/work -v <ROGUEONE_CAIR_PATH>/scripts/agent2027:/workspace/scripts/agent2027:ro --entrypoint python3 rogue_one:agent2027-v2.<N> /workspace/scripts/agent2027/analysis/rp1_score.py sample --out /work --seed 42 > ~/ro27_a18_sample.log 2>&1 & echo launched"
  ```
  The sample run logs W&B artifact `rp1-prompts:v0` (project `RogueOne-AGENT2027`, run `ANALYSIS__rp1_score__20261021`, group `ANALYSIS`). It is never re-drawn after the first replay.

  *Done when:* HEAD is logged; the `clients.py` diff against the freeze tag is empty, or (e.g. E7 in v2.3) it touches none of `_merge_extra_body_for_vllm`, `_wrap_completions`, `_streaming_create_with_fallback`, `_stream_to_chat_completion`, `_resolve_text_tool_call`, `_normalize_tool_name` or `_normalize_tool_arguments`, and Henrik has seen the diff; the image ID is recorded in the session notes; the A18-T5 assertions passed; and `rp1-prompts:v0` holds 800 records plus the realised allocation table.
- [ ] **A18-T11** [type: run] (0.1 d, Thu 22 – Fri 23 Oct, during the O20 session of A17/A20) — **Packed small rungs and O20.** Before **each** launch, run the pre-flight:
  ```
  $CAIR "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader && echo '--- compute apps ---' && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader && echo '--- containers ---' && docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'"
  ```
  Proceed only if all three hold:
  - the GPU you pin shows < 2,000 MiB used;
  - the GPUs held by our `rogueone-vllm-*` servers, plus the new one, total ≤ 4;
  - no `llmfe-*` or `caafe-*` container of ours is up, and the only `ro27-*` / `rogueone-vllm-*` containers up are the O20 session's (A17/A20) and earlier A18 links.

  Otherwise **STOP and report** the per-GPU memory and utilisation, the compute apps and the containers. Then launch, for example A18-001:
  ```
  $CAIR "SESSION_ID=S1022p SESSION_FILE=\$HOME/ro27_manifests/sessions/S1022p.csv GPUS=<id> NEED=1 PORT=8030 MODEL=Qwen/Qwen3.5-0.8B SERVER_NAME=rogueone-vllm-q0.8-s1022p EXTRA_ARGS='<rungs.csv extra_args>' MAX_MODEL_LEN=131072 EAGER=0 WAIT_MAX_MIN=0 ROGUEONE_ALLOW_SECOND_SERVER=1 LOG=\$HOME/vllm_session_S1022p.log IMAGE_RO=rogue_one:agent2027-v2.<N> RUN_CAP_HOURS=6 CLIENTS_SCRIPT=\$HOME/ro27_rp1_clients.sh RP1_ROW=A18-001 nohup bash ~/ro27_session_launch.sh > ~/ro27_S1022p.job.log 2>&1 & echo launched; sleep 5; tail -5 ~/ro27_S1022p.job.log"
  ```
  - Chain the targets on the GPUs the O20 session leaves free (2 if it holds 2 GPUs on 8020/8021; up to 3 if it holds 1): GPU a runs Q0.8 → Q2 → Q4; GPU b runs Q9 → O20 (`replay_O20` on port **8022**, `SERVER_NAME=rogueone-vllm-o20-s1023p`). Each link is its own session, launched after the previous one's `RO27_SESSION_END` and a fresh pre-flight. Add `RP1_CONCURRENCY=32` to the launch for a target that A18-T4 flagged (> 6 h projected).
  - No replay traffic ever goes to 8020/8021. If port 8022 is not accepted, A18-005 runs on 8020 after `RO27_SESSION_END` of the session that holds 8020 (A20: `S1022d`). A20-T10 and A20 §7 now also use 8022 for `replay_O20` (port registered in 01 R8).
  - If no O20 session is running, run them as a short block of up to 4 parallel TP1 sessions (ΣGPU ≤ 4, 8022 not needed: `replay_O20` then takes 8020).

  *Done when:* five job logs end with `RO27_SESSION_END … rc=0`, the five `rp1-replies-*` artifacts exist, and each run's summary has its `rp1_valid_call_raw__<role>` and `rp1_valid_call_rescued__<role>` keys.
- [ ] **A18-T12** [type: run] (0.05 d, Sat 24 Oct, morning; moves with A17's Q122 session, latest launch Wed 28) — **Q122.** Run the same pre-flight (2 GPUs; 4 if SR3 moved Q122 to TP4), then launch S1024p in the A18-T11 form (`GPUS=<id1>,<id2>`, `RP1_ROW=A18-006`, TP2, port 8040, `MODEL` and `EXTRA_ARGS` from `rungs.csv`, so `Qwen/Qwen3-32B` and its TP if SR1 fired; `SERVER_NAME=rogueone-vllm-q122-s1024p`) on the GPUs that A17's Q122 session will use. A17 starts after the tear-down. *Done when:* `rp1-replies-Q122` exists and its online `rp1_valid_call_*__<role>` summary keys are logged.
- [ ] **A18-T13** [type: run] (0.05 d, Tue 27 Oct; latest launch Thu 29, L-LAST) — **O120 self-replay.** Same pre-flight (2 GPUs). S1027p runs in the A18-T11 form (`GPUS=<id1>,<id2>`, `RP1_ROW=A18-007`) at TP2 (`MODEL=openai/gpt-oss-120b`, `EXTRA_ARGS` empty, `SERVER_NAME=rogueone-vllm-o120-s1027p`). Coordinate with A17-T13's O120 F0-pool session, planned for the same day.
  - Use port 8011 with `ROGUEONE_ALLOW_SECOND_SERVER=1` if a TP2 O120 session is already on 8010 and the total stays ≤ 4.
  - Otherwise use 8010 between sessions.

  *Done when:* `rp1-replies-O120` exists and RP1-D5(a) is checked and logged right after the session: the online `rp1_valid_call_rescued__<role>` of `replay_O120` is ≥ 0.9 for every role. If not, find the replay-path defect and relaunch by Thu 29 (§6).
- [ ] **A18-T14** [type: run] (0.02 d per session, ≈ 0.15 d in all; at the end of each session, the last by Fri 30 Oct) — **Monitor and tear down.** Monitor in the background (exit 255 means the network dropped; reconnect):
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 CAIR-gpu14 "while ! grep -q RO27_SESSION_END ~/ro27_<SID>.job.log; do sleep 120; done; tail -30 ~/ro27_<SID>.job.log"
  ```
  Then run the tear-down check (01 §5.2 step 6) and confirm the W&B run state is `finished` (a `running` run with no live container is a zombie):
  ```
  $CAIR "docker ps --format '{{.Names}}' | grep -E '^(ro27-|rogueone-vllm-)' || echo none; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"
  ```
  In packed mode the O20 session's own containers may still be listed; only the A18 names (`ro27-<sid>-a18-*`, the A18 `SERVER_NAME`) must be gone. *Done when:* each session is torn down, and the infra-error share is ≤ 5% per target, or the target has been rerun once.

### 5.4 Analysis

- [ ] **A18-T15** [type: analysis] (0.05 d, Fri 30 Oct, G6) — Make a preliminary transport table from the W&B summaries `rp1_valid_call_raw__<role>` and `rp1_valid_call_rescued__<role>` of the targets finished by then, read with a `wandb.Api()` query in a container (01 §5.2 step 7 form) and logged as a W&B table. It is descriptive only, for the knee read-out, and applies no decision rule. *Done when:* run `ANALYSIS__rp1_score__20261030` (group `ANALYSIS`) holds the table.
- [ ] **A18-T16** [type: analysis] (0.25 d, Sun 1 – Mon 2 Nov) — Score the replies inside containers. `fetch` downloads `rp1-prompts:v0`, the seven `rp1-replies-<rung>` artifacts and the 18 source `run-output` artifacts, so it needs the network. `score` runs with **`--network none`** and is detached because its worst case is ≈ 4 h (§4). Because `score` is offline, the embedder of `process_metrics.py` must already be on disk: if it is an HF model, add `-v $HOME/.cache/huggingface:/root/.cache/huggingface:ro -e HF_HUB_OFFLINE=1` to the `score` command (embedder not fixed in 01; verify in A18-T9):
  ```
  $CAIR "docker run --rm --name ro27-a18-fetch --cpus 4 --env-file <CAIR_HOME>/code/PlanBench/.env -v \$HOME/ro27_a18_work:/work -v <ROGUEONE_CAIR_PATH>/scripts/agent2027:/workspace/scripts/agent2027:ro --entrypoint python3 rogue_one:agent2027-v2.<N> /workspace/scripts/agent2027/analysis/rp1_score.py fetch --out /work"
  $CAIR "nohup docker run --rm --name ro27-a18-score --network none --cpus 16 --memory 64g -v \$HOME/ro27_a18_work:/work -v <ROGUEONE_CAIR_PATH>/tasks:/workspace/tasks:ro -v \$HOME/rogueone-synp-meta:/synp_meta:ro -v <ROGUEONE_CAIR_PATH>/scripts/agent2027:/workspace/scripts/agent2027:ro --entrypoint python3 rogue_one:agent2027-v2.<N> /workspace/scripts/agent2027/analysis/rp1_score.py score --work /work --synp-meta /synp_meta --exec-timeout 60 > ~/ro27_a18_score.log 2>&1 & echo launched"
  ```
  Monitor `~/ro27_a18_score.log` as in T14. Then run `publish` (same form as `fetch`), which uploads `rp1-scored` from run `ANALYSIS__rp1_score__<YYYYMMDD>`. The φ metadata is mounted into this analysis container only, never into an agent. The `tasks` mount holds `labels.csv` (see the §7 risk on untrusted generated code). *Done when:* artifact `rp1-scored` has 5,600 + 800 (`recorded_O120`) rows (3,200 + 800 under ED2), with every score and exclusion flag.
- [ ] **A18-T17** [type: analysis] (0.25 d, Mon 2 Nov) — Validate implementability with κ. Draw 60 S answer turns, stratified across targets (artifact `rp1-kappa`). Two raters label them "implementable and additive" independently; disagreements are settled by discussion to give the consensus label. "The rule" is the §2 S-answer rule (column match, novelty, additive lexicon) as frozen in A18-T1. *Done when:* Cohen's κ is logged, both between the raters and between the rule and the rater consensus.
- [ ] **A18-T18** [type: analysis] (0.1 d, Tue 3 Nov) — Run `rp1_score.py floors` on the common set, in a container with `--network none` (same form as `score`, reading `/work`). A second `publish` call, with network, uploads the results:
  - Wilson 95% CIs per role × target × score;
  - the ratio to O120, with a cluster bootstrap over the 90 source fold-episodes (2,000 resamples, seed 42) and the floor-stability share (the share of resamples whose floor equals the point-estimate floor);
  - exact McNemar tests between adjacent rungs (exploratory, unadjusted);
  - the T Jaccards;
  - the sensitivity analyses of §4 (no common set; `clipped` dropped; `suspicious` and `reconstruction_partial` turns dropped);
  - the RP1-D1…D5 verdicts (D5(a) was checked on Tue 27, D5(b) is checked here).

  *Done when:* `ANALYSIS__rp1_score__20261103` and artifact `rp1-floors` exist.

### 5.5 Paper outputs

- [ ] **A18-T19** [type: writing] (0.1 d, Wed 4 Nov) — Draw the Fig. 3 inset: a heatmap of role × target, each cell showing the `rp1_usable` ratio to O120. Hatch the cells that are transport-limited, and mark each role's floor. *Done when:* it is in the AGENT draft.
- [ ] **A18-T20** [type: writing] (0.15 d, Thu 5 Nov; package Thu 12 Nov) — Write the appendix table (rates with CIs, common-set size, context and infra errors, `clipped` share) and the disclosures:
  - teacher forcing;
  - cross-family template rendering;
  - Qwen thinking off (`clients.py:204`);
  - the JSON-only rescue (`clients.py:675-687`);
  - O120 at TP2;
  - partial reconstruction.

  *Done when:* both are in the draft, and `rp1-prompts` and `rp1-scored` are in the replication package.
- [ ] **A18-T21** [type: writing] (0.05 d, Fri 6 Nov) — Write the §3 sentence or its null/opposite form, and the RP1-D4 agreement with A19. *Done when:* it is in the draft.

## 6. Decision rules and gates (pre-registered in X10)

In this document a bare "D1"…"D6" means RP1-D1…D6. The registry's decisions are written "decision D<n> (01 §6.4)".

- **RP1-D1 · Floor.**
  - A target passes when its rate is ≥ 0.8 × O120's self-replay rate, for the same role and score, on the common set. The rule uses the point estimates; the CIs and the floor-stability share are reported, not used in the rule.
  - The floor is the smallest rung r on the Qwen ladder Q0.8 < Q2 < Q4 < Q9 < Q122 such that r and every larger Qwen rung pass. At the ends it reads "≤ Q0.8" or "> Q122" (no rung passes, or Q122 fails).
  - For the order rules the floor is coded by position: "≤ Q0.8" and Q0.8 = 1, Q2 = 2, Q4 = 3, Q9 = 4, Q122 = 5, "> Q122" = 6. Under ED2, or after an A04 drop, the ladder holds only the measured Qwen rungs (ED2: Q0.8 < Q4 < Q9, ends "≤ Q0.8" and "> Q9") and "rungs apart" counts positions on that ladder.
  - O20 is reported as pass/fail (it is not on the Qwen ladder).
  - The primary score is `rp1_usable`. The floor on `rp1_valid_call_rescued` is the transport floor. S, E and T enter the order verdict. W's floor is computed the same way and is descriptive only.
- **RP1-D2 · Order verdict** (S, E and T floors, as positions):

  | Verdict | Condition |
  |---|---|
  | **Hoped** | floor(E) ≥ floor(S) ≥ floor(T), and floor(E) is at least 2 Qwen rungs above floor(T) |
  | **Null** | all three floors are on one rung |
  | **Opposite** | floor(E) < max(floor(S), floor(T)) |
  | **Partial** | anything else, reported as such |

  The four cases are disjoint and cover every outcome. If the floor-stability share (A18-T18) of any of the three floors is < 0.6, the verdict is reported as "uninformative (unstable floors)", with the point-estimate verdict in brackets.
- **RP1-D3 · Failure type.**
  - (a) At a failing rung, a role is **transport-limited** if it fails D1 on `rp1_valid_call_rescued`. It is **code/planning-limited** if transport passes but `rp1_usable` fails.
  - (b) **RP1-D3b · Harness finding.** E is **transport-bottlenecked** if, at every Qwen rung in {Q0.8, Q2, Q4} that was served, `rp1_valid_call_rescued` fails D1 for E while `rp1_executes` for E passes D1. Both are conditional on parsed calls, and the `rp1_executes` comparison is evaluable only where its denominator is ≥ 30 at that rung and at O120 (else "not evaluable"). Report it per family (Qwen vs gpt-oss), with the JSON-only rescue disclosed.
- **RP1-D4 · A19 prediction.** CAP-2's 1B cell is predicted to pass if and only if floor(S) ≤ Q0.8 and floor(T) ≤ Q0.8 on `rp1_usable` (for S under D6's κ rule, on the substitute score). Agreement with A19 is reported descriptively: RP1 is teacher-forced, A19 is on-policy.
- **RP1-D5 · Replay-path check.** Investigate before reading any floor if either holds:
  - (a) *Online, right after the O120 session (A18-T13, Tue 27):* O120's self-replay `rp1_valid_call_rescued__<role>` is below 0.9 for any role. If a replay-path defect is found, fix X2 and rerun the affected targets by Thu 29 (L-LAST).
  - (b) *At scoring (A18-T18, Sun 1 – Tue 3 Nov):* `recorded_O120` and the O120 self-replay differ by more than 10 pp on any of `rp1_valid_call_rescued`, `rp1_executes` or `rp1_usable`, for any role. No GPU rerun is possible after DF (Sat 31 Oct, 24:00): report the defect with a caveat, and show the floors with and without the affected role.
- **RP1-D6 · Late scoring.** E11c (A18-T9) is due Sun 1 Nov; D6 is the last-moment rule.
  - If E11c is not done by Tue 3 Nov, the paper reports transport floors only (`rp1_valid_call_raw`, `rp1_valid_call_rescued`, scored online). D2 and D4 are then not evaluated, and the §3 sentence becomes a transport-only sentence. Executes and usable go to the replication package.
  - If κ(rule vs consensus) < 0.6, S's `rp1_usable` is replaced for S by the joint pass of `rp1_valid_call_rescued` and `rp1_executes` (S passes a rung only if both pass D1).
- **ED2 · X2 late.** If X2 cannot replay the 50 prompts by Tue 13 Oct (A18-T3), the targets shrink to Q0.8, Q4, Q9 and O120 (A18-001, -003, -004, -007; 3,200 replays, 1.9–3.7u, rounded to the breakdown's "about 2–4u"), launched Tue 27 – Thu 29 Oct. Q2, O20 and Q122 are not measured: the D1 ladder is the reduced one (D1, third bullet) and no O20 pass/fail is reported.
- **Gates:**

  | Gate | What must hold for A18 |
  |---|---|
  | G1 (Fri 16) | T1, T2 (merge-ready) and T4 |
  | G2 (Mon 19) | M0 traces valid; T5 light QC passes |
  | L-LAST (Thu 29) | last replay launch |
  | G6 (Fri 30) | preliminary transport table (T15), descriptive only |
  | DF (Sat 31, 24:00) | no GPU work after this |

  A18 has no kill date. It is never cut beyond ED2.

## 7. Risks and cut levers

- **Teacher forcing.** Every history was written by O120, so the rates are an upper bound on small models' own behaviour. CAP-1 (A17) gives the on-policy view.
- **Cross-family rendering.** Qwen's template re-tokenises O120's history, which can push prompts past 131k tokens. These become context errors, handled by the common-set rule. The dropped prompts are the longest, latest-iteration ones, so the common-set floors describe shorter prompts. The §4 sensitivity analysis (own context errors counted as invalid) shows the size of this bias.
- **Selection.** The strata condition on O120's move. A target that answers where O120 acted is still judged on its own move. For E an answer is never usable (§2). That can raise E's floor and so favours the "hoped" order; the `answer_instead_of_action` share is reported to show how much.
- **Reconstruction.** It may be partial in late iterations (floor restorations, auto-unprune, iteration-skipped trials), especially without T2's add-on. Flagged turns are dropped in a sensitivity analysis.
- **Single turn.** Replay misses multi-turn error recovery. HC1 (A07) covers that.
- **Stack.** The Qwen3.5 ids, the `qwen3_coder` parser and Q122's KV room are unverified until A04 passes.
- **No idle GPUs on Thu 22.** Unpacked replay then costs +1–2u, still inside 4–7u. The Thu 22 and Sat 24 slots also collide with `S1021a` and `S1023a` in the other documents (§4 *Box contention*).
- **Untrusted generated code.** It runs offline (`--network none`), with timeouts and a memory cap. The score container still mounts `tasks/` (needed for `data.csv`), which holds `labels.csv`. With E13 off, a generated command could read it, as it could in the live loop. Such commands are flagged `suspicious` and reported separately (A18-T9). Mounting only each task's `data.csv` and `config.yml` is the stricter option.
- **Developer time.** E11c has no slack, so D6 applies. A18's own code (0.6 d, Fri 16 – Wed 21 Oct) is not in the breakdown's engineering total (§4).
- **Cut levers.** CL1–CL7 do not touch A18. Packing is the only cost lever, saving about 2u. ED2 is the only drop.

## 8. Definition of done

- [ ] The pre-registration items (T1) are in `agent2027-prereg`, and the E3 add-ons (T2) are in v2.0 or the fallback is recorded.
- [ ] `rp1-prompts:v0` holds 800 frozen records.
- [ ] Seven `A18__replay_<rung>__rp1set__f42__r0` runs are `finished` (four under ED2; fewer if A04 dropped a rung), each with an `rp1-replies-<rung>` artifact and a server log in `~/ro27_a18_logs/`. All sessions are torn down cleanly.
- [ ] `rp1-scored` and `rp1-floors` exist, plus `ANALYSIS__rp1_score__<date>` with the D1–D5 verdicts and κ.
- [ ] The Fig. 3 inset, the appendix table, the disclosures and the paper sentence are drafted by Fri 6 Nov.

---

**New names introduced** (not in the canonical registry; to be added to 01 §7 if accepted):
- **Seed:** `llm_seed` = 7042 for every A18 replay (R11 has no rule for replays). It keeps the O120 self-replay distinct from M0's 42 and 1042 and from A17's F0 seeds (42000–42639, 4000–4639), the sentinel seeds (5000 + k, 6000 + k) and the A/A seeds.
- **Run and manifest:**
  - dataset slot `rp1set`, giving run name `A18__replay_<rung>__rp1set__f42__r0`;
  - `task_type=mixed` in `A18_RP1.csv` (R7 allows only classification or regression).
- **Port** 8022 (registered in 01 R8 in the cross-document pass) for a third O20 server, used only for the packed A18 replay, because the O20 session uses 8020 and 8021. If not accepted, run A18-005 on 8020 after the session that holds 8020 ends (+0.2–0.4u).
- **Session IDs** S1022p–s, S1023p, S1024p and S1027p. They are placeholders: use the actual date and the next free letter.
- **Files and folders:**
  - `scripts/agent2027/rp1_clients.sh` (installed as `~/ro27_rp1_clients.sh`);
  - the `rp1_score.py` subcommands `sample`, `fetch`, `score`, `floors` and `publish`;
  - `~/ro27_a18_logs/` and `~/ro27_a18_work/`;
  - `intermediate_steps/df_attributes_explanations_trail_{index}.csv` and `intermediate_steps/df_scientist_feedback_trail_{index}.csv` (E3 add-ons, A18-T2).
- **Containers:** `ro27-a18-sample`, `ro27-a18-fetch`, `ro27-a18-score` (analysis; no session).
- **Launcher env:** `RP1_ROW` and `RP1_CONCURRENCY`.
- **W&B:**
  - artifacts `rp1-prompts`, `rp1-replies-<rung>`, `rp1-scored`, `rp1-floors` and `rp1-kappa`;
  - summary keys `rp1_<score>__<role>` (role ∈ S, E, T, W);
  - row flags `infra_error`, `context_error`, `clipped`, `timeout`, `suspicious`, `reconstruction_partial`, `auto_unprune` and `attempts`; share `answer_instead_of_action`;
  - pseudo-target `recorded_O120`.
- **Reused from A04** (also not in the registry): the `CLIENTS_SCRIPT` passthrough and `scripts/agent2027/rungs.csv`.
- **Rule IDs** RP1-D1…D6 (with D3b and the sub-checks D5(a), D5(b)). In text a bare D<n> means these, see §6.
- **Verifier defaults to confirm by Fri 16 Oct** (they enter the pre-registration through A18-T1): the E-answer rule, the S-additive, section and column lexicons (§2), the floor-stability threshold 0.6 and the D3b minimum denominator of 30 (§6), the 600 s timeout rule and the 4× batching gain used in A18-T4's projection (§4).
- **Open questions for Henrik:**
  - Who is the second κ rater?
  - Is O120 at TP2 acceptable for the self-replay if decision D5 (01 §6.4) = TP4?
  - Will E3 accept the T2 trace fields (A18-T2) before the freeze? If not, the §5 fallbacks apply and the reconstruction is partial.
  - Confirm `<ROGUEONE_CAIR_PATH>` (A04-T1 checks it on Wed 7 Oct).
  - Port for `replay_O20`: resolved to 8022 in 01 R8, A18 and A20 (8021 stays the second O20 server).
  - Where do the A18 slots go, given `S1021a` / `S1023a` (§4 *Box contention*)? Henrik places them by G2.
