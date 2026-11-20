# A04 — CAP-0 · Stage-0 capability smoke tests, model downloads, parser checks

| Field | Value |
|---|---|
| **ID** | A04 (short ID CAP-0). Executes engineering task **E12** |
| **Claim / Holm contrast** | None. Validity prerequisite for C5a (A17) and C5b (A19) |
| **Tier** | T1-minimum (prerequisite; on no cut list) |
| **Owner** | Henrik |
| **Start / finish** | Downloads Thu 8 – Sun 11 Oct · sessions A/B planned **Thu 15 Oct** (earliest Tue 13; fallback Fri 16, ended by 16:00) · session C **Wed 21 Oct**, in the box gap between the G0b-1 smoke `S1021b` and the evening launch of `S1021a` (fallback: the first box gap after `S1021a` ends, before A19 launches) · final table before A19 launches |
| **Budget** | 8 single-fold runs + 350 streamed calls + 1 KV probe · **7–13u** (A/B 5.25–8.75u pre-freeze, C 2–4u post-freeze) · 14.5–25.5 GPU-hours · ≈ 203 GB of downloads (no GPU): ≈ 173 GB for the six new rungs (the breakdown's figure) + ≈ 30 GB of Qwen3 insurance |
| **Developer effort** | 1.08 d over 21 tasks; 0.84 d is A04's own work, and the breakdown's E12 = 0.5 d covers only the Tue 13 – Fri 16 part (reconciled after §5.5) |
| **Models** | Q0.8, Q2, Q4, Q9, Q122, O20 (new); O120 (session C only). Insurance downloads: Qwen3-0.6B, -1.7B, -4B, -8B |
| **Status** | Not started |
| **Depends on** | E1 (manifest launcher, `ROGUE_ONE_SMOKE_FOLDS`, `flags/*` logging), E2, E3 (traces, tokens per role, counters), the draft `scripts/agent2027/v2_defaults.env` at tag `agent2027-pre.N`, the G0a smoke traces (Mon 12) · X1, X2 (Tue 13) · session C: E4 (caller role on W calls), E5 (prompt dump), E7 in `rogue_one:agent2027-v2.3` (G0b-1, Wed 21) |
| **Feeds into** | A17, A18, A19, A20, A06 (Q4 small-rung sentinel), X10 pre-registration, replication package |

*Every "hoped" outcome is a hypothesis. Code refs: RougeOne `nested-cv-rebuttal` @ `07cbf93`, read-only; every ref below was re-checked against that commit. `main.py` = `rogue_one_main.py`; `config.py` = `rogueone/utils/config.py`; `clients.py` = `rogueone/llm/clients.py`; `sci/`, `ext/`, `tst/agent.py` = `rogueone/llm/agents/core/{scientist,extractor,tester}/agent.py`; `kn/web_search.py` = `rogueone/llm/agents/core/knowledge/web_search.py`. u = one real-data O120 run (10 iterations × 5 folds) = 2 GPU-hours at the plan rate (2 u/h on the 4-GPU box).*

## 1. What we test and why

Before 88–165u goes into the capability ablations (A17 53.0–97.1u, A18 4–7u, A19 27.6–54u, A20 3.5–7u), A04 checks four things for every rung:
- (a) it downloads and boots on the unmodified `vllm/vllm-openai:v0.22.0` image;
- (b) it makes streamed tool calls that Rogue One can parse;
- (c) it finishes one short end-to-end fold.
- After E7, check (d) confirms that a mixed-model run sends each role, and that role's web-search helper W, to the intended server.

The C5 claims can be read only if a small rung fails because the model is weak, not because the serving stack is broken. A04 keeps the two apart: a broken stack switches the model family, while a weak model is data. A04 also pins the HF revisions and per-family serving flags that the paper and the replication package must state.

## 2. Background you need

**Routing.**
- Roles: `AGENT_ROLES` = scientist, extractor, tester (`config.py:221`).
- `ROGUE_ONE_<ROLE>_MODEL` / `_PORT` fall back to `LLM_MODEL` / `LLM_PORT` (`config.py:246-260`). The global endpoint is `http://localhost:<port>/v1` (`:251`), and the per-role endpoint is `endpoint_for` (`:272-274`).
- `config.py:246` types `LLM_MODEL` as a gpt-oss `Literal`. This is only a type hint: nothing in `rogueone/` or `main.py` branches on the model name (checked by grep at `07cbf93`).
- Role clients:
  - S: client at `sci/agent.py:397-400`, model at `:458`;
  - E: client at `ext/agent.py:848-852`, model at `:932`;
  - T: client at `tst/agent.py:1338-1341`, model at `:1352`.
- W (`WebSearchAgent`) always uses the global endpoint and model (`kn/web_search.py:188-191`, `:238`). This is by design (`config.py:217-220`). E7 adds `ROGUE_ONE_WEBSEARCH_ROUTING=caller`.
- **All three roles call W:**
  - S through `search_tool` (`sci/agent.py:149`);
  - E through `search_in_literature_tool` (`ext/agent.py:261`);
  - T through `search_in_literature_tool` (`tst/agent.py:1058`).
- The Scientist's prompt says to "always perform at least one web search" (`sci/agent.py:431`), so nearly every fold has a Scientist W call. E and T search only when they choose to.
- W's client is created with `agent_name="WebSearchAgent"` (`kn/web_search.py:191`). The calling role is not recorded today. E4 passes it through `explain_query` (`kn/web_search.py:292`), and E3 writes the trace.
- The server's model name is the HF id (`--served-model-name "$MODEL"`, `paper/scripts/vllm_session.sh:166`). The `model` field of a traced response therefore names the server that answered.

**Image hazard.**
- `.dev.env` is tracked at `07cbf93` and sets `LLM_PORT=8000` and `LLM_API_KEY`.
- `config.py:24` loads it with `override=True`, and `Dockerfile:13` (`COPY . .`) copies it into the image.
- An image built from that context sends **every** role to port 8000, whatever the launcher passes.
- The build step therefore refuses a context that contains `.dev.env` (A04-T12), and A04-T2 checks the effective `flags/LLM_PORT`.

**Client wrapper** (`VLLMCompatibleAsyncOpenAI`, `clients.py:471`).
- It drops `reasoning_effort` (`:525`). It sets `enable_thinking=False` in the chat template for every model (`:204`).
- With tools present, it streams (`:539-542`) through `_streaming_create_with_fallback` (`:564-689`):
  - up to 3 empty-response retries and 2 splice re-rolls (`:571-572`);
  - name and argument normalisation (`:655-673`);
  - a **JSON-only** text rescue (`:675-687`; the parser is `_resolve_text_tool_call`, `:37-108`).
- Qwen's XML-style calls cannot be rescued, so rescue rates differ by family by construction.

**Extractor tools** (`ext/agent.py:935-940`): `generic_pandas_tool` (`:282-283`), `append_new_attribute` (`:388-389`), `list_known_attributes_tool` (`:569-570`), `search_in_literature_tool` (`:246-247`).

**Run loop.**
- `total_trials = k_folds × num_iterations` (`main.py:1420-1424`). There is no single-fold option today; E1 adds `ROGUE_ONE_SMOKE_FOLDS` (assumed to run the first n folds of the k-fold partition; A04-T2 confirms this).
- Two code defaults must be overridden:
  - `ROGUE_ONE_NUM_ITERATIONS` defaults to 3 (`config.py:305`), so set 10;
  - `ROGUE_ONE_START_VLLM` defaults to true (`main.py:1600`), so set `False`.
- balance-scale is in the legacy task list (`main.py:1537`), at `tasks/classification/balance-scale/`.
- The image `CMD` ends in `|| tail -f /dev/null` (`Dockerfile:29`), so a crashed client hangs until the 6 h launcher watchdog (`cap_hours`).

**`paper/scripts/vllm_session.sh`.**
- It refuses TP > 4 (`:66-69`). It refuses a second `rogueone-vllm-*` server unless `ROGUEONE_ALLOW_SECOND_SERVER=1` (`:121-130`).
- It mounts `~/.cache/huggingface` at `/root/.cache/huggingface` (`:164`).
- The server command is built at `:157-168`. Its only option today is `--enforce-eager`. X1 appends `$EXTRA_ARGS` and `--max-model-len $MAX_MODEL_LEN`.
- It waits up to 30 min (120 × 15 s) for health (`:171-182`).
- It writes to `$LOG`, which defaults to `~/vllm_session.log`. Concurrent sessions must set distinct `LOG` values.
- On exit it removes the server (`:79-90`). This **deletes the server log** unless the log was dumped first.

**Rungs** (R8, R9):

| Rung | HF id | TP | Port | `EXTRA_ARGS` (X1 appends `--max-model-len 131072`) |
|---|---|---|---|---|
| Q0.8 / Q2 / Q4 / Q9 | `Qwen/Qwen3.5-0.8B` / `-2B` / `-4B` / `-9B` | 1 | 8030 / 8031 / 8032 / 8033 | Qwen set + `--revision <sha>` |
| Q122 | `Qwen/Qwen3.5-122B-A10B-FP8` | 2 (4 if the KV cache is short, SR3) | 8040 | Qwen set + `--revision <sha>` |
| O20 | `openai/gpt-oss-20b` | 1 | 8020 | `--revision <sha>` |
| O120 | `openai/gpt-oss-120b` | 2 (session C) | 8010 | empty (t17's command; the cached sha is recorded) |

- **Qwen set:**
  - `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --language-model-only`.
  - This is R9's Qwen set without `--max-model-len`, because X1 appends that flag. Do not put it in `extra_args` a second time.
- **Fallback ladder:** Q0.8→`Qwen/Qwen3-0.6B`, Q2→`Qwen/Qwen3-1.7B`, Q4→`Qwen/Qwen3-4B`, Q9→`Qwen/Qwen3-8B`, Q122→`Qwen/Qwen3-32B`.

**Definitions** (pre-registered in X10).
- **Valid call, raw.** The first streamed completion, before any retry or rescue, meets all of these:
  - `finish_reason="tool_calls"`;
  - at least one structured call;
  - every structured call names one of the 4 Extractor tools;
  - every call's arguments parse as a JSON object with the schema's required keys.
- **Valid call, rescued.** The same test, applied to the wrapper's final response (what the agent sees).
- **Rate.** Valid calls out of 50 frozen prompts. A request that errors counts as invalid.
- **Parser exception.** A server-log `Traceback` or `ERROR` line from a tool-parser or reasoning-parser module.
- **Stack signature.** A parser exception, an HTTP 5xx, or a boot failure ((a) not healthy within 30 min).
- **W call.** A trace record of an LLM call made by a `WebSearchAgent` client. Its **caller** is the role that E4 attaches.
- **Misrouted call** (`misrouted_calls`, same definition as A19-T9). A trace record whose response `model` differs from the intended model. For S, E and T records the intended model is that role's model; for W records it is the caller's model.
- **Metrics (R13):** `tok_compl_scientist`, `tok_compl_extractor`, `tok_compl_tester`, `wall_s`, `cap_hit`, `fallback_share`, `rescues_per100_<family>`, `context_errors`, `max_prompt_tokens`.

## 3. Hypotheses and interpretation (all are hypotheses; no result exists yet)

| Outcome | Meaning and action |
|---|---|
| **Hoped** | Every rung boots. Q9 and O20 reach ≥ 80% rescued valid calls. Q122 holds 6 clients on TP2. Check (d) shows `misrouted_calls` = 0 for every role and for W. → *"All rungs ran on one unmodified serving image (vLLM 0.22.0), with identical prompts and serving flags fixed per family."* |
| **Null / partial** | Q0.8, Q2 (or Q4) make few valid calls, or finish only through fallbacks. This is the first capability data point for A17/A18, reported per family. It is not a reason to swap models (SR2). |
| **Opposite** | **Qwen3.5 fails at Q9 (SR1):** switch to the Qwen3 ladder (≈ +1 day). 0.6B–8B are pre-downloaded; Qwen3-32B (± 14B) adds ≈ 65–92 GB. "1B" becomes Qwen3-0.6B. **O20 fails (SR4):** A19 switches to its Q122/Q4 fallback, and A20 moves to Q4. **Q122 fails (SR3):** O120 is the only top anchor, and C5a is fitted on 0.8–9B. **O20 and Q122 both fail:** A19 has no pre-registered pair, so Henrik decides at A04-T18. **Routing fails (SR6):** A19 is blocked until E7 is fixed. |

## 4. Design

**Checks.**
- **(a) Boot.** Pass = healthy within 30 min. Log the load time, the `GPU KV cache size` and the `Maximum concurrency for 131,072 tokens`.
  - Q122 also gets the KV probe: the 6 longest frozen prompts, sent concurrently.
- **(b) Streamed calls.** The 50 frozen Extractor prompts (A04-T5) go through X2 and the `clients.py` wrapper: single turn, temperature 0.7, concurrency 1, `ROGUE_ONE_LLM_SEED=42`. Record:
  - the raw and rescued valid-call rates;
  - parser exceptions;
  - the share with `finish_reason=length`.
- **(c) End to end.** balance-scale, fold 0 of the seed-42 5-fold partition, 10 iterations (`ROGUE_ONE_SMOKE_FOLDS=1`).
  - **Pass** requires all of: `fold_aggregation_summary.csv` written; `tok_compl_<role>` > 0 for S, E and T; a non-empty trace; the required R12 `wandb.config` keys.
  - A fallback-only completion still passes, with `fallback_share` recorded.
- **(d) Mixed routing** (session C).
  - `misrouted_calls` = 0 over every S, E, T and W trace record.
  - `wandb.config` agrees: `model_S/E/T`, `flags/ROGUE_ONE_<ROLE>_MODEL` / `_PORT`, and `flags/ROGUE_ONE_WEBSEARCH_ROUTING=caller`.
  - In each row the global `LLM_MODEL` is the session's **third** model, the one no role uses. A W call that wrongly follows the global setting therefore lands on a server that no role should reach, for every caller.

**Fixed settings.**
- **Seeds and identity:** `ROGUE_ONE_FOLD_SEED=42`, `ROGUE_ONE_LLM_SEED=42` (`replicate=0`), `is_sentinel=0`, `ROGUE_ONE_CODE_PHASE=smoke`, `ROGUE_ONE_ABLATION_ID=A04`.
- **Run:**
  - `ROGUE_ONE_DATASETS=balance-scale`;
  - `ROGUE_ONE_NUM_ITERATIONS=10`;
  - `ROGUE_ONE_K_FOLDS=5` with `ROGUE_ONE_SMOKE_FOLDS=1`;
  - `ROGUE_ONE_SELECTION_SIGNAL=inner_cv`, `ROGUE_ONE_ENABLE_ROLLBACK=False`;
  - `ROGUE_ONE_STRATIFIED_FOLDS=False` (real dataset), `ROGUE_ONE_DO_BALANCING=True`, `ROGUE_ONE_FOLD_RETRY_LIMIT=1`;
  - `LLM_TEMPERATURE=0.7` and every `*_AGENT_TEMPERATURE=0.7`;
  - `EXTRACTOR_AGENT_MAX_TOKENS=32768`, `TESTER_AGENT_MAX_TOKENS=8192`, `SCIENTIST_AGENT_MAX_TOKENS` at its draft (A/B) or frozen (C) value;
  - `ROGUE_ONE_LLM_TRACE=1`, `ROGUE_ONE_PROMPT_DUMP=1`, `ROGUE_ONE_START_VLLM=False`, `ROGUE_ONE_DISABLE_WANDB=0`.
  - Every other R3/R4 flag is taken from `v2_defaults.env` and logged as `flags/*`.
- **Model and endpoint** (rows 001–006, uniform):
  - `rung_S = rung_E = rung_T` = the rung, and `port_S/E/T` = its port;
  - the launcher sets `LLM_MODEL` = the rung's HF id, `LLM_PORT` = its port, and `LLM_API_KEY` = the server's `API_KEY` (R3);
  - `ROGUE_ONE_<ROLE>_MODEL` / `_PORT` stay empty.
- **Images:**
  - Sessions A/B use `rogue_one:agent2027-pre.N` with the draft `v2_defaults.env` at that tag. Pool mode and R are irrelevant to a transport test; their values are logged.
  - Session C uses `rogue_one:agent2027-v2.3` with the frozen `v2_defaults.env`.
- **Server:** `vllm/vllm-openai:v0.22.0`, `EAGER=0`, `MAX_MODEL_LEN=131072`, `WAIT_MAX_MIN=0`, explicit `GPUS`.
- **Launcher:** `cap_hours=6`, `CPUS=12` per client, `RUNROOT=~/rogueone-runs-agent2027`.
- **No sentinel** (A06 rule for smoke sessions). A04 runs are never pooled with analysis runs.

**Run matrix.** Fold-run names follow R12, e.g. `A04__stage0_Q0.8__balance-scale__f42__r0`. Replay names are `A04__<arm_id>__replay50__f42__r0`.

| Group | Session IDs | Servers (rung·TP·port) | Rows → `arm_id` | Checks | Fold-runs | Calls |
|---|---|---|---|---|---|---|
| A | S1015a–d (one per server) | Q0.8·1·8030, Q2·1·8031, Q4·1·8032, Q9·1·8033 | A04-001…004 → `stage0_Q0.8`, `stage0_Q2`, `stage0_Q4`, `stage0_Q9` | a b c | 4 | 200 |
| B | S1015e, S1015f (S1015g = Q122 TP4 retry) | Q122·2·8040; O20·1·8020 | A04-005 → `stage0_Q122`; A04-006 → `stage0_O20` | a b c + KV probe | 2 | 100 (+6 probe) |
| C | S1021m (`ladder_session.sh`) | O120·2·8010 + O20·1·8020 + Q0.8·1·8030 | A04-007 → `mixcheck_esmall`; A04-008 → `mixcheck_1b`; replay `stage0_O120` | d + c; b on O120 | 2 | 50 |
| **Total** | 7 (+1 retry) | ΣTP ≤ 4 per group | 8 rows | | **8** | **350** |

> **SR4 fallback (2026-10-19).** O20 failed SR4, so session C is re-planned as Q122·2·8040 +
> Q4·1·8032 + Q0.8·1·8030 (ΣTP = 4, image `agent2027-v2.3`), as SR4 states; the O120/O20 rows of
> the two tables here are superseded (pre-registration Part V.4, Part VII.3). SR6 and its pass
> rule are unchanged. Rows (`manifests/agent2027/A04_CAP-0.csv`, session file `S1021m.csv`):
>
> | Row | S | E | T | Global `LLM_MODEL` / `LLM_PORT` (third model) |
> |---|---|---|---|---|
> | A04-007 `mixcheck_esmall` | Q122 / 8040 | Q4 / 8032 | Q122 / 8040 | `Qwen/Qwen3.5-0.8B` / 8030 |
> | A04-008 `mixcheck_1b` | Q0.8 / 8030 | Q122 / 8040 | Q0.8 / 8030 | `Qwen/Qwen3.5-4B` / 8032 |
>
> - The launcher now keeps a global given in `env_overrides` on a mixed row (A04-T8's
>   requirement; `ro27_clients_manifest.sh`, test `test_mixed_rows_global_override_and_routing`),
>   and `make_manifest.py` admits `LLM_MODEL`/`LLM_PORT` there only as a pair on mixed rows.
> - The `stage0_O120` replay (check (b) on O120) has no O120 server in this layout and is not run
>   in session C; O120's TP2 valid-call rate comes from A18's `replay_O120` control instead.
> - `ladder_session.sh` was never built: session C = main `S1021m` (owns Q122 TP2; clients wait
>   for 8032 and 8030) + holders `S1021n` (Q4) and `S1021o` (Q0.8) through
>   `scripts/agent2027/ro27_hold_clients.sh`; lines in `docs/agent2027/sessions/LAUNCH.md`.
>   Gate: G0b-1 (v2.3 switch smokes, `S1021b`). Needs 4 GPUs for 1–2 h; A19 cannot launch
>   before its SR6 verdict (A04-T19).

**Session-C rows.** The global model is the session model that no role in the row uses.

| Row | S | E | T | Global `LLM_MODEL` / `LLM_PORT` |
|---|---|---|---|---|
| `mixcheck_esmall` | O120 | O20 | O120 | `Qwen/Qwen3.5-0.8B` / 8030 |
| `mixcheck_1b` | Q0.8 | O120 | Q0.8 | `openai/gpt-oss-20b` / 8020 |

- **Difference from A19.** A19's production rows set the global to O120 / 8010 in both arms (A19 §4). Under `caller` the global is unused, so this does not change A19's validity. There, a W call that wrongly went global would land on O120, which equals its caller's model for S/T in `mix_esmall` and for E in `mix_1b`. Such a misroute is harmless there but invisible. A04 uses the third model so that this misroute is visible for every caller.
- If SR1 switched the family, `Qwen/Qwen3.5-0.8B` becomes `Qwen/Qwen3-0.6B`. HF ids are always read from `rungs.csv`.

**Manifest** (R7; `make_manifest.py --ablation A04` → `manifests/agent2027/A04_CAP-0.csv`).
- Every row has `dataset_id=balance-scale`, `task_type=classification`, `fold_seed=42`, `llm_seed=42`, `replicate=0`, `is_sentinel=0`, `cap_hours=6`, `expected_u=0.2` and `notes=smoke`. `image_tag` is the actual `agent2027-pre.<N>` for rows 001–006 and `agent2027-v2.3` for rows 007–008.
- `env_overrides`:
  - rows 001–006: `{"ROGUE_ONE_SMOKE_FOLDS":"1"}`;
  - rows 007–008 also carry `ROGUE_ONE_WEBSEARCH_ROUTING` and the global `LLM_MODEL` / `LLM_PORT` from the table above.

Example rows:
```
A04-001,A04,stage0_Q0.8,balance-scale,classification,42,42,0,0,,,agent2027-pre.<N>,Q0.8,Q0.8,Q0.8,8030,8030,8030,"{""ROGUE_ONE_SMOKE_FOLDS"": ""1""}",6,0.2,smoke
A04-007,A04,mixcheck_esmall,balance-scale,classification,42,42,0,0,,,agent2027-v2.3,O120,O20,O120,8010,8020,8010,"{""ROGUE_ONE_SMOKE_FOLDS"": ""1"", ""ROGUE_ONE_WEBSEARCH_ROUTING"": ""caller"", ""LLM_MODEL"": ""Qwen/Qwen3.5-0.8B"", ""LLM_PORT"": ""8030""}",6,0.2,smoke
A04-008,A04,mixcheck_1b,balance-scale,classification,42,42,0,0,,,agent2027-v2.3,Q0.8,O120,Q0.8,8030,8010,8030,"{""ROGUE_ONE_SMOKE_FOLDS"": ""1"", ""ROGUE_ONE_WEBSEARCH_ROUTING"": ""caller"", ""LLM_MODEL"": ""openai/gpt-oss-20b"", ""LLM_PORT"": ""8020""}",6,0.2,smoke
```
- The launcher applies `env_overrides` before it adds the launcher-derived keys (01 §5.3 step 2). It must not overwrite an `LLM_MODEL` / `LLM_PORT` given in `env_overrides`. A04-T8 checks this.

**W&B.**
- Entity `PlanBench`, group `A04`, job_type = `arm_id`, tags [`<SESSION_ID>`, `smoke`, `<image_tag>`].
- Projects: `RogueOne-AGENT2027-prefreeze` for A/B; `RogueOne-AGENT2027` for C.

Session IDs assume the planned dates. On another date, use the actual date and the next letter that no document in `docs/ablations/` claims for that date (grep first). Wed 21 uses `a`–`d` (A12, A13, A14), so A04 takes `m`.

**Budget** (u = 2 GPU-hours).

| Session | GPUs × wall | GPU-h | u |
|---|---|---|---|
| A | 4 × 1.5–2.5 h | 6–10 | 3–5 |
| B | 3 × 1.5–2.5 h | 4.5–7.5 | 2.25–3.75 |
| C | 4 × 1–2 h | 4–8 | 2–4 |
| **Total** | | **14.5–25.5** | **7.25–12.75 ≈ 7–13** |

- **Why u exceeds the rows' `expected_u`** (8 × 0.2 = 1.6u): each server serves one client plus one replay, so the GPUs are mostly idle. The u above is GPU-time held, including loads (1–3 min small, 5–10 min Q122/O120) and the 50-call replays.
- **Pre-freeze vs post-freeze.** A/B = 5.25–8.75u. This is the breakdown's CAP-0 share of the 37–49u pre-freeze demand (M4 12–20u + M1 20u + CAP-0 A/B). That slice is ≈ 70–90u idle by construction, so A/B displaces nothing. C (2–4u) is post-freeze demand.
- **If session B is charged as box-time** (its 4th GPU sits idle under the one-job rule), B costs 3–5u and the total is 8–14u. This is immaterial, for the same pre-freeze-idle reason.
- **From the 15% rerun reserve, outside 7–13u:**
  - Q122 TP4 retry ≈ 2u (4 GPUs × ≈ 1 h);
  - Qwen3-ladder rerun under SR1 ≈ 4.5–7.5u (session A again, plus Qwen3-32B on 2 GPUs × 1.5–2.5 h);
  - session-C rerun 2–4u;
  - a small-rung cap hit ≤ 3u (1 GPU × 6 h).

**Failure handling** (launcher, 01 §5.3).
- Success = rc 0 **and** `fold_aggregation_summary.csv`.
- A crash or a missing summary is retried once, solo.
- A cap hit gets `docker stop`, is logged as `CAP_HIT` (`cap_hit`), and is not retried.
- What a failure means is decided only by §6: SR2 for Q0.8–Q4, SR1 for Q9 and for Q122 parser exceptions, SR3 for Q122, SR4 for O20, SR6 for session C.

## 5. Tasks

`$CAIR` = `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14` (non-interactive one-shot commands only).

**CAIR rules for every step** (01 §5.1):
- **Containers only.** Every execution runs in Docker: runs, replays, downloads, analysis and W&B queries. The host only issues:
  - `docker` commands and the `~` launchers that issue them;
  - `git fetch` / `checkout` (step 0) and copies into `~`;
  - read-only queries (`nvidia-smi`, `docker ps`, `tail`, `grep`, `ls`).
  Never run `python` on the host. Any `docker compose run` over SSH needs `-T`.
- **GPU pre-flight before every GPU container** (A04-T13 step 1). If GPUs are short, **stop and report** the per-GPU memory and utilisation, the compute apps and the running containers. Henrik decides.
- **≤ 4 of the 8 H100s**, never more without Henrik's explicit approval in chat. One Rogue One job at a time: never alongside A02, A03, A05/A07 or the Wed 21 sessions `S1021a`/`S1021b`.
- **W&B is the system of record.** Runs are stateless, and each uploads `run-output`. The key is read from `<CAIR_HOME>/code/PlanBench/.env` (as `paper/scripts/clients_t16_rerun.sh:59` does).
- **Code reaches CAIR** only by commit + push of `agent2027/v2` locally, then `git fetch` on the server (step 0 in A04-T12).
- **Long jobs** use a detached `nohup` launcher, with the log in `~` and never in the root-owned run folders.

### 5.1 Prerequisites

- [ ] **A04-T1** [type: config] (0.05 d, due Wed 7 Oct) — Verify `<ROGUEONE_CAIR_PATH>` = `<CAIR_HOME>/code/RougeOne` (**inferred**), and check disk:
  `$CAIR "ls -d ~/code/RougeOne/tasks/classification/balance-scale && docker run --rm -v \$HOME/.cache/huggingface:/hf --entrypoint df vllm/vllm-openai:v0.22.0 -h /hf"`.
  *Done when:* the path prints and ≥ 250 GB is free (203 GB now, plus margin). Otherwise stop and report. Note that ≈ 95 GB more is needed if SR1 fires.
- [ ] **A04-T2** [type: decision] (0.05 d, due Mon 12 Oct) — Check G0a. *Done when* the G0a smoke run (prefreeze project) shows all of:
  - exit 0 under `ROGUE_ONE_SMOKE_FOLDS=1`, with exactly fold 0 run (this confirms E1's semantics);
  - `fold_aggregation_summary.csv`;
  - `tok_compl_<role>` > 0 for S, E and T;
  - a trace with full messages, tool schemas, `agent_name` and the response `model`;
  - `rung_*`, `model_*` and `flags/*` in `wandb.config`, with `flags/LLM_PORT` equal to the row's port (no `.dev.env` override);
  - the draft `scripts/agent2027/v2_defaults.env` present at tag `agent2027-pre.N`.
- [ ] **A04-T3** [type: decision] (0.02 d, due Tue 13 Oct) — Check X1 and X2. *Done when* all three hold:
  - `~/ro27_vllm_session.sh` serves Q4 with the Qwen set;
  - with `EXTRA_ARGS` empty, the O120 command equals t17's plus `--max-model-len 131072`;
  - X2 turns 50 G0a Extractor prompts against O120 into a raw/rescued table.

  If X2 is late, apply **ED2**: compute (b) from the fold-run's E3 counters (the rescue counters plus the trace's first-attempt `finish_reason`), and A18 shrinks.
- [ ] **A04-T4** [type: decision] (0.02 d, due Wed 21 Oct, before A04-T15) — Check v2.3. *Done when* all of these hold:
  - the E7 no-op test passed (G0b-1: prompts byte-identical to `prompt-baseline-v2.0`);
  - the A19-T7 unit test (W client built on the caller's endpoint) is green;
  - the `S1021b` smoke traces carry the caller role on W records (E4);
  - `~/ladder_session.sh` refuses ΣTP > 4, and its launch syntax is copied into A04-T15;
  - the v2.3 image ID is recorded.

### 5.2 Implementation

- [ ] **A04-T5** [type: data] (0.05 d, due Mon 12 Oct, evening) — Freeze the prompt set.
  - Sample 50 Extractor turns (RNG seed 42) from the G0a traces in which O120 emitted a tool call.
  - If fewer than 50 such turns exist, take all of them and top up from the first A02 traces (Tue 13). Record this in the artifact metadata.
  - Store messages, tool schemas and temperature as W&B artifact `stage0-prompts` (prefreeze project).
  - Flag the 6 longest as the KV probe.

  *Done when:* artifact `v0` holds 50 records, each with its `max_prompt_tokens`.
- [ ] **A04-T6** [type: code] (0.05 d, due Thu 8 Oct) — `scripts/agent2027/stage0_download.py`. It must never download, refresh or delete `openai/gpt-oss-120b` (or any non-listed repo) in the shared HF cache: A01 requires the O120 weights in `~/.cache/huggingface` to stay unchanged (single snapshot, `refs/main` unchanged) until A01 ends on Mon 12 Oct, and A01 compares them with the Hub `main`.
  - For each id, call `huggingface_hub.snapshot_download(id, ignore_patterns=["original/*","metal/*"])`, catching errors per model.
  - Print the TSV `id, revision_sha, size_GB, OK|FAILED`.
  - `--inspect-only` uses `local_files_only=True`.

  *Done when:* `Qwen/Qwen3-0.6B` prints OK locally, and a bogus id prints FAILED without aborting the run.
- [ ] **A04-T7** [type: code] (0.1 d, due Wed 14 Oct) — `scripts/agent2027/stage0_clients.sh`, installed as `~/ro27_stage0_clients.sh`. Also add a `CLIENTS_SCRIPT` passthrough to `ro27_session_launch.sh` and `ladder_session.sh` (default `~/ro27_clients_manifest.sh`). The script's env is `STAGE0_REPLAY_RUNGS` (default: the session's rung; `O120` in session C) and `STAGE0_KV_PROBE=1` (Q122 only). For each replay rung, the script:
  1. runs X2 on `stage0-prompts` in container `ro27-<session_id>-replay50`:
     - `docker run --rm --network host --cpus 4 --env-file <CAIR_HOME>/code/PlanBench/.env -e LLM_MODEL=<hf_id> -e LLM_PORT=<port> -e LLM_API_KEY=$API_KEY -e ROGUE_ONE_LLM_SEED=42 --entrypoint python3 rogue_one:<tag> /workspace/scripts/agent2027/replay_tool.py …`;
     - it logs the replay run `A04__<arm_id>__replay50__f42__r0`, plus the KV probe on Q122;
  2. runs `~/ro27_clients_manifest.sh` on `$SESSION_FILE`;
  3. before returning, dumps each server's log with `docker logs <server_name> > ~/ro27_a04_logs/vllm_server_<SESSION_ID>_<rung>.log 2>&1`.

  *Done when:*
  - `bash -n` passes;
  - `DRY_RUN=1` prints the three steps with resolved names;
  - with `CLIENTS_SCRIPT` unset, both launchers behave exactly as before.
- [ ] **A04-T8** [type: config] (0.05 d; rows 001–006 due Wed 14 Oct, rows 007–008 due Wed 21) — Run `make_manifest.py --ablation A04` → `manifests/agent2027/A04_CAP-0.csv`. Create the session files with `--plan-session`: S1015a–f (and S1015g, holding row 005, for the TP4 retry) and S1021m. *Done when:*
  - the files are committed and validate against R7;
  - a `DRY_RUN` env print of rows 007–008 shows the `env_overrides` values of `LLM_MODEL` / `LLM_PORT` surviving the launcher.
- [ ] **A04-T9** [type: code] (0.1 d, due Wed 14 Oct) — `scripts/agent2027/analysis/stage0_table.py`.
  - Pull the group-`A04` runs (`wandb.Api()`).
  - Parse the server logs: load time, KV tokens, maximum concurrency, parser exceptions, preemption.
  - For session C, compute `misrouted_calls` per role and for W, using the §2 definition.
  - Apply §6.
  - Log `ANALYSIS__stage0_table__<YYYYMMDD>` with one row per rung, and the logs as artifact `stage0-server-logs`.

  *Done when:* the verdicts are correct on a toy case (fake logs + the G0a run + a fake trace with one misrouted W record).
- [ ] **A04-T10** [type: config] (0.05 d; draft due Wed 14 Oct; verdicts Fri 16 Oct) — `scripts/agent2027/rungs.csv` (`rung, hf_id, revision, tp, port, extra_args, stage0_verdict`). A17–A20 read their serving flags from it.
  - The draft (verdict column empty) is committed with A04-T8 and installed on CAIR before A04-T13.
  - A04-T18 fills in the verdicts.

  *Done when:* the verdict version is committed before the tag `agent2027-v2.0-freeze` (or as the dated addendum of A04-T18's fallback).

### 5.3 Execution on CAIR

- [ ] **A04-T11** [type: run] (0.1 d, Thu 8 – Sun 11 Oct; no GPU, so no pre-flight) — Fetch the branch (step 0 in A04-T12), then:
  ```
  $CAIR "mkdir -p ~/ro27_a04_logs && nohup docker run --rm --name ro27-a04-download -v \$HOME/.cache/huggingface:/root/.cache/huggingface -v <ROGUEONE_CAIR_PATH>/scripts/agent2027:/s:ro --entrypoint python3 vllm/vllm-openai:v0.22.0 /s/stage0_download.py Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B Qwen/Qwen3.5-122B-A10B-FP8 openai/gpt-oss-20b Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B > ~/ro27_a04_logs/download.log 2>&1 &"
  ```
  - Then run the same container with `--inspect-only openai/gpt-oss-120b`.
  - Fetch Qwen3-14B/32B only if SR1 fires.

  *Done when:*
  - all 10 print OK;
  - their shas and the O120 sha are in the draft `rungs.csv`;
  - any failed Qwen3.5 id was reported to Henrik the same day.
- [ ] **A04-T12** [type: run] (0.05 d, Thu 15 Oct, 08:00) — Build or confirm `rogue_one:agent2027-pre.N`, then install the launchers.
  ```
  $CAIR "cd <ROGUEONE_CAIR_PATH> && GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15' git fetch git@github.com:HenrikNorkart/RougeOne.git agent2027/v2 && git checkout -B agent2027/v2 FETCH_HEAD && git log -1 --format='HEAD: %h %s'"
  $CAIR "cd <ROGUEONE_CAIR_PATH> && test ! -e .dev.env && docker build -t rogue_one:agent2027-pre.<N> . && docker image inspect rogue_one:agent2027-pre.<N> --format '{{.Id}}'"
  $CAIR "cd <ROGUEONE_CAIR_PATH> && cp scripts/agent2027/ro27_session_launch.sh scripts/agent2027/ro27_clients_manifest.sh scripts/agent2027/ladder_session.sh ~/ && cp scripts/agent2027/stage0_clients.sh ~/ro27_stage0_clients.sh && cp paper/scripts/vllm_session.sh ~/ro27_vllm_session.sh && mkdir -p ~/ro27_manifests ~/ro27_a04_logs && cp -r manifests/agent2027/. ~/ro27_manifests/"
  ```
  If `test ! -e .dev.env` fails, stop: E1/E2 must untrack `.dev.env` or add it to `.dockerignore`.

  *Done when:*
  - the image ID is recorded;
  - `docker run --rm --entrypoint test rogue_one:agent2027-pre.<N> ! -e /workspace/.dev.env` exits 0;
  - the 6 `S1015*` session files are in `~/ro27_manifests/sessions/`.
- [ ] **A04-T13** [type: run] (0.03 d hands-on; Thu 15 Oct, after A03's Core-10 rows end; 1.5–2.5 h wall) — **Session A.**
  1. Pre-flight:
     ```
     $CAIR "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader && echo '--- compute apps ---' && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader && echo '--- containers ---' && docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'"
     ```
     Proceed only if 4 GPUs show < 2,000 MiB used and no `ro27-*`, `rogueone-vllm-*`, `llmfe-*` or `caafe-*` container is running. Otherwise stop and report.
  2. Launch S1015a–d, one pinned GPU each (`<rung>` lower-case, e.g. `q0.8`):
     ```
     $CAIR "SESSION_ID=<SID> SESSION_FILE=\$HOME/ro27_manifests/sessions/<SID>.csv GPUS=<id> PORT=<port> MODEL=<hf_id> SERVER_NAME=rogueone-vllm-<rung>-<sid> EXTRA_ARGS='<rungs.csv extra_args>' MAX_MODEL_LEN=131072 ROGUEONE_ALLOW_SECOND_SERVER=1 WAIT_MAX_MIN=0 EAGER=0 LOG=\$HOME/vllm_session_<SID>.log IMAGE_RO=rogue_one:agent2027-pre.<N> PARALLEL=1 CPUS=12 RUN_CAP_HOURS=6 RUNROOT=\$HOME/rogueone-runs-agent2027 CLIENTS_SCRIPT=\$HOME/ro27_stage0_clients.sh nohup bash ~/ro27_session_launch.sh > ~/ro27_<SID>.job.log 2>&1 & echo launched; sleep 5; tail -5 ~/ro27_<SID>.job.log"
     ```

  *Done when:* all 4 job logs end with `=== RO27_SESSION_END <SID> rc=0 ===` and the A04-T16 tear-down check prints `none`.
- [ ] **A04-T14** [type: run] (0.03 d hands-on; Thu 15 Oct, afternoon; 1.5–2.5 h wall) — **Session B.**
  - Run the same pre-flight (3 GPUs needed) and use the same launch form for S1015e (Q122, `GPUS=<id1>,<id2>`, port 8040) and S1015f (O20, port 8020).
  - If the KV probe fails SR3 at TP2, relaunch Q122 alone at TP4 as S1015g once S1015f ends, after a fresh 4-GPU pre-flight.

  *Done when:* both rows (and S1015g if needed) have finished and been torn down.
- [ ] **A04-T15** [type: run] (0.03 d hands-on; Wed 21 Oct, 1–2 h wall) — **Session C.**
  > **SR4 fallback (2026-10-19).** Launch form replaced: holders `S1021n` (Q4, 8032) and
  > `S1021o` (Q0.8, 8030) first, then main `S1021m` (Q122 TP2, 8040, `WAIT_PORTS="8032 8030"`,
  > `PARALLEL=2`), all via `ro27_session_launch.sh` with `CLIENTS_SCRIPT=~/ro27_hold_clients.sh`
  > (`docs/agent2027/sessions/LAUNCH.md`). No `STAGE0_REPLAY_RUNGS=O120` (no O120 server).
  - **Slot.** After the v2.3 build (A12-T10) and the G0b-1 smoke `S1021b`, before `S1021a` launches in the evening. Both are sequenced, never concurrent (A13 §9).
  - **Fallback.** If no slot is free on Wed 21, use the first box gap after `S1021a` ends (Thu 22 evening at the earliest), as `S<MMDD>m`. A19 cannot launch before SR6 passes.
  - Install the v2.3 launchers, `~/ro27_stage0_clients.sh` and the manifests (A04-T12, third command), then run the pre-flight for 4 GPUs.
  - Launch through E7's `ladder_session.sh`. The `SERVERS` syntax below is a placeholder for E7's interface, as in A19-T13; replace it with the form recorded in A04-T4.
    ```
    $CAIR "SESSION_ID=S1021m SESSION_FILE=\$HOME/ro27_manifests/sessions/S1021m.csv SERVERS='O120:<g1>,<g2>:8010 O20:<g3>:8020 Q0.8:<g4>:8030' IMAGE_RO=rogue_one:agent2027-v2.3 PARALLEL=2 CPUS=12 RUN_CAP_HOURS=6 RUNROOT=\$HOME/rogueone-runs-agent2027 MAX_MODEL_LEN=131072 WAIT_MAX_MIN=0 EAGER=0 LOG=\$HOME/vllm_session_S1021m.log CLIENTS_SCRIPT=\$HOME/ro27_stage0_clients.sh STAGE0_REPLAY_RUNGS=O120 nohup bash ~/ladder_session.sh > ~/ro27_S1021m.job.log 2>&1 & echo launched; sleep 5; tail -5 ~/ro27_S1021m.job.log"
    ```
  - Per-server `EXTRA_ARGS` come from `rungs.csv`.
  - O120 runs at TP2 whatever D5 decides, because A19 serves it at TP2.

  *Done when:* rows 007–008 and the `stage0_O120` replay have finished and been torn down.
- [ ] **A04-T16** [type: run] (0.02 d in total; Thu 15 Oct for A/B, plus S1015g if needed; Wed 21 Oct for C) — Monitor in the background:
  `ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 CAIR-gpu14 "while ! grep -q RO27_SESSION_END ~/ro27_<SID>.job.log 2>/dev/null; do sleep 120; done; tail -30 ~/ro27_<SID>.job.log"`.
  - Exit 255 means the network dropped, not the job; reconnect.
  - Check W&B states in parallel. A `running` run with no live container is a zombie.
  - Tear-down check: `$CAIR "docker ps --format '{{.Names}}' | grep -E '^(ro27-|rogueone-vllm-)' || echo none; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"`.

  *Done when:* the check prints `none`, and every server log of the session is in `~/ro27_a04_logs/`.

### 5.4 Analysis

- [ ] **A04-T17** [type: analysis] (0.05 d, Thu 15 Oct, evening) — Build the A/B table:
  `$CAIR "docker run --rm --env-file <CAIR_HOME>/code/PlanBench/.env -v \$HOME/ro27_a04_logs:/logs:ro --entrypoint python3 rogue_one:agent2027-pre.<N> /workspace/scripts/agent2027/analysis/stage0_table.py --logs /logs --project RogueOne-AGENT2027-prefreeze --group A04"`.
  *Done when:* `ANALYSIS__stage0_table__20261015` (prefreeze project, group `A04`) shows (a)–(c) and a verdict per rung. This is E12's done-when.
- [ ] **A04-T18** [type: decision] (0.05 d; Fri 16 Oct, 09:00 at the latest) — Apply SR1–SR5. Write the verdicts, the stack decision, the Q122/O20 topology and the flags into `rungs.csv` and the draft pre-registration.
  - **If A/B ran on Fri 16** (fallback): decide by 16:30.
  - **If a rung is still unfinished at 16:00:** stop it, because the box goes to A05/A07 at 18:00. Run it in the first small-rung night (Sun 18) before any A17 row on that rung. Its verdict is then a dated pre-registration addendum.

  *Done when:* both files are committed before the freeze tag, or the addendum is committed before that rung's first A17/A18 row.
- [ ] **A04-T19** [type: analysis] (0.03 d; due Wed 21 Oct, right after A04-T15; in the fallback, the day session C runs, and always before A19 launches) — Rerun `stage0_table.py` (image v2.3) with session C included. Log it to `RogueOne-AGENT2027`, group `ANALYSIS`. *Done when:* it shows:
  - `misrouted_calls` and an SR6 verdict per role and for W, in both mixed rows;
  - the number of W calls per caller;
  - O120's TP2 valid-call rate.

### 5.5 Paper outputs

- [ ] **A04-T20** [type: writing] (0.05 d, due Fri 16 Oct, inside X10) — The pre-registration lists:
  - the rung ids and revisions;
  - the per-family flags;
  - the §2 definitions;
  - SR1–SR6;
  - the fallback ladder.

  *Done when:* they are in the commit tagged `agent2027-prereg`.
- [ ] **A04-T21** [type: writing] (0.1 d; Nov 1–3, package Thu 12 Nov) — Write:
  - an appendix table (rung, HF id, revision, TP, flags, raw/rescued valid calls, KV capacity, verdict);
  - the §3 methods sentence, only if the results support it;
  - the disclosures: Qwen thinking is off (`clients.py:204`); the rescue is JSON-only (`clients.py:37-108`, `:675-687`), so rescues are reported per family.

  *Done when:* all three are in the draft, and `rungs.csv` is in the replication package.

**Effort reconciliation.** The tasks total 1.08 d.
- **Counted elsewhere: 0.24 d.** A04-T2 (G0a acceptance), A04-T3 (X1/X2), A04-T4 (E7/G0b-1), A04-T20 (X10) and A04-T21 (writing block).
- **A04's own work: 0.84 d:**
  - Wed 7 – Sun 11: 0.25 d (A04-T1, T5, T6, T11). This falls on the Sat 10 / Sun 11 weekend (01 §6.2; decision D1);
  - Tue 13 – Fri 16: 0.52 d (A04-T7–T10, T12–T14, T17, T18 and part of T16). This matches the breakdown's "Stage-0 0.5";
  - Wed 21 – Thu 22: 0.07 d (A04-T15, T19 and part of T16).
- E12's registry effort of 0.5 d therefore under-counts by ≈ 0.3 d.

## 6. Decision rules and gates (pre-registered in X10)

- **SR1 · Qwen3.5 stack failure.**
  - **Trigger** (any one):
    - Q9's rescued valid-call rate is < 80% (< 40/50);
    - any parser exception occurs in Q9's replay or fold-run;
    - Q9 fails (a).
  - **Action:** move the whole Qwen family to the Qwen3 ladder (Q122 → Qwen3-32B; 14B only if 32B does not fit). Rerun A/B from the reserve: on Fri 16 before 16:00, or otherwise after A05/A07 and before A17's first small-rung night.
  - **A parser exception at Q122 only** (Q9 clean): the default is SR3's drop path (keep the Qwen3.5 small ladder; O120 is the only top anchor). Henrik may choose the family switch instead at A04-T18.
  - If the Qwen3 ladder also triggers SR1, escalate to Henrik.
- **SR2 · Small rungs Q0.8, Q2 and Q4.** These outcomes are data, never a reason to swap:
  - low valid-call rates;
  - parser exceptions (the parser choking on malformed output);
  - fallback-only completions;
  - failed or capped fold-runs.
  They are reported per rung.
  - **The one exception is a boot failure (a):** retry once with the SR5 cudagraph flag.
    - If it still fails, the default is to drop that rung from A17/A18.
    - If fewer than 3 Qwen3.5 rungs at ≤ 9B would remain, SR1's action applies instead.
- **SR3 · Q122.**
  - **Pass** requires both:
    - all 6 concurrent probe requests complete without error or preemption warning;
    - KV capacity ≥ 6 × (longest probe prompt + 32,768) tokens. Capacity is read from the `GPU KV cache size` line; if that line is absent (hybrid attention), use `Maximum concurrency for 131,072 tokens` × 131,072.
  - Q122's valid-call rate is recorded as data.
  - **Fails at TP2:** retry at TP4 (S1015g). A17's Q122 session then uses either TP4 × 6 clients (same unit price) or fewer clients (disclosed; Henrik chooses).
  - **No load, or fails at TP4:** drop Q122. O120 becomes the only top anchor, and C5a is fitted on 0.8–9B.
- **SR4 · O20.**
  - **O20 fails if:** (a) fails, (c) fails after the retry, or its rescued valid-call rate is < 80% (< 40/50). Then:
    - A19 applies its §6 fallback: big = Q122 (TP2, 8040), small = Q4 (TP1, 8032). `mix_esmall` becomes S, T on Q122 and E on Q4; `mix_1b` becomes S, T on Q0.8 and E on Q122. The reference is A17 `f2_Q122`, and the `fx_pajM` comparison is dropped;
    - A20 moves to Q4 (`fx_pajM_Q4`);
    - A17 drops `f2_O20` (A17-059…069, −5.5 to −11u) and its F0 pool (A17-143…149);
    - session C is re-planned as Q122·2 + Q4·1 + Q0.8·1 (ΣTP = 4), with the same third-model rule.
  - **If Q122 has also failed SR3,** Henrik decides A19's pair at A04-T18.
- **SR5 · The image never changes.**
  - The only flag addition allowed without Henrik is a per-family `--max-cudagraph-capture-size <n>`, recorded in `rungs.csv`.
  - Henrik decides any other change before G1, and it is pre-registered.
- **SR6 · Routing (d).**
  - **Pass** requires all of:
    - `misrouted_calls` = 0 over every S, E, T and W record in both mixed rows;
    - ≥ 1 Scientist W call in each row;
    - `wandb.config` agrees (§4 check (d)).
  - E and T W calls are checked whenever they occur. A role with none is reported as "W untested for this caller".
  - **Fail:** A19 is blocked. Fix E7 and rerun session C (2–4u, reserve).
  - **A row with no Scientist W call:** rerun it once. If there is still none, rely on the A19-T7 unit test and disclose this.
  - **Still failing on Tue 27 Oct:** escalate, because A19's latest full-matrix launch is Tue 27 and C5b can no longer finish before L-LAST (Thu 29).

**Gates:**
- **G1 (Fri 16):**
  - the A/B verdicts and `rungs.csv` are frozen;
  - A/B must end by 16:00, so that the 4 GPUs are free for A05/A07 at 18:00;
  - unfinished rungs follow the A04-T18 fallback.
- **Before A19 launches:** SR6 has passed (A19-T5).

## 7. Risks and cut levers

**Unverified until A04 runs:**
- the `qwen3_coder` streaming parser and `--language-model-only` on 0.22.0;
- the Qwen3.5 hybrid-attention cudagraph assertion (the SR5 flag);
- whether each Qwen3.5 model accepts a 131,072-token window (a refusal is a boot failure; escalate);
- O20's harmony path on 0.22.0;
- Q122 FP8 (≈ 125 GB on 160 GB at TP2) leaving enough KV room;
- whether `--revision <sha>` also pins the tokenizer on 0.22.0 (if not, add `--tokenizer-revision <sha>` under SR5, decided by Henrik);
- free disk on CAIR;
- the Qwen3.5 HF ids themselves (registry R9: unverified until A04 passes).

**Other risks:**
- **Wide tables.** The probe prompts come from the G0a dataset, so A17's first Q122 myocardial run must watch `context_errors` and preemption.
- **Wall clock.** A weak Scientist can make 100 retries (`sci/agent.py:519`) × 200 turns (`config.py:411`). A cap hit costs ≤ 3u and is recorded.
- **`.dev.env`.** If it reaches an image, every role goes to port 8000 (§2). It is guarded by A04-T2 and A04-T12.
- **Box contention.** A/B depend on A03's Core-10 rows ending on Thu 15. C needs a 1–2 h gap on Wed 21, when `S1021b` and `S1021a` also run.
- **Developer time.** Freeze week also holds E4, E6 and E8b. If X2 slips, apply ED2.

**Cut levers:** none cuts A04. Contingencies are paid from the rerun reserve.

## 8. Definition of done

- [ ] 10 downloads OK with shas, and the O120 sha recorded.
- [ ] `rungs.csv` committed before `agent2027-v2.0-freeze` (or with the A04-T18 addendum). The SR1–SR4 decisions are in the pre-registration.
- [ ] W&B group `A04` holds:
  - 6 fold-runs and 6 replays in prefreeze;
  - 2 mixcheck fold-runs and the O120 replay in `RogueOne-AGENT2027`;
  - the artifacts `stage0-prompts` and `stage0-server-logs`.
- [ ] Verdict tables exist after A/B (Thu 15 or Fri 16) and after C (before A19 launches), with `misrouted_calls` = 0.
- [ ] Every session ended with `rc=0` and a clean tear-down.
- [ ] The A04-T21 items are drafted.

---

**New names introduced** (not in the registry; propose for 01 §7):
- **Arm** `stage0_O120`: the O120 replay in session C. R5 lists `stage0_<rung>` for A/B only.
- **Run name** `A04__<arm_id>__replay50__f42__r0` (`replay50` sits in the `dataset_id` slot).
- **Session IDs:** S1015a–g and S1021m (fallback `S<MMDD>m`). This replaces the earlier `S1021a`, which A12/A13/A14 claim.
- **Containers:** `ro27-a04-download` (downloads; no session) and `ro27-<session_id>-replay50` (X2 replays).
- **Files:**
  - `scripts/agent2027/stage0_download.py`;
  - `scripts/agent2027/stage0_clients.sh` (installed as `~/ro27_stage0_clients.sh`);
  - `scripts/agent2027/analysis/stage0_table.py`;
  - `scripts/agent2027/rungs.csv`;
  - the manifest `manifests/agent2027/A04_CAP-0.csv` (R6 pattern `<AID>_<short>` with the short ID verbatim; A17/A20 drop the hyphen in theirs).
- **Env and paths:** `CLIENTS_SCRIPT`, `STAGE0_REPLAY_RUNGS`, `STAGE0_KV_PROBE`; `~/ro27_a04_logs/` with `vllm_server_<SESSION_ID>_<rung>.log`.
- **Artifacts:** `stage0-prompts` and `stage0-server-logs`.
- **Summary keys:** `stage0_valid_call_raw`, `stage0_valid_call_rescued`, `stage0_finish_length_share`, `stage0_parser_exceptions`, `stage0_kv_tokens`, `stage0_kv_max_concurrency`, `stage0_load_s`. Also `misrouted_calls`, shared with A19-T9.
- **Rules:** SR1–SR6, prefixed so they do not clash with the registry's R-sections.
- **Deviations:**
  - rows 007–008 put the R3 keys `LLM_MODEL` / `LLM_PORT` in `env_overrides`, where R7 expects R4 flags only;
  - rows 007–008 use the third session model as the global, while A19 uses O120 (§4, "Difference from A19");
  - `--revision <sha>` is added to every new rung's `EXTRA_ARGS`, while R9 lists flags without it; O120 keeps t17's command.
