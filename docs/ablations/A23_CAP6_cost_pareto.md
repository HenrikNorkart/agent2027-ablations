# A23 — CAP-6: Cost Pareto front (post-hoc)

| Field | Value |
|---|---|
| **ID** | A23 · CAP-6 |
| **Claim / Holm contrast** | The cost side of **C5** (per-role model allocation, capability floor) and the "equal at lower cost" half of **C1**. It also displays the token co-primary behind the C2a/C2b "reducible" verdicts. **No Holm contrast**: everything here is descriptive, under rules pre-registered in X10 |
| **Tier** | T1-minimum ("free": no GPU) |
| **Owner** | Henrik |
| **Start / finish** | Prerequisite checks Mon 12 Oct (G0a) · rules in the pre-registration Fri 16 Oct (G1) · daily coverage checks Fri 16 – Sat 31 Oct · δ read from A09-T15 Fri 23 Oct · `pareto.py` done with E11b Tue 27 Oct (G4) · **draft Tue 27 Oct** · **final Sun 1 – Tue 3 Nov** · figures Wed 4 – Fri 6 Nov · replication package Thu 12 Nov. (The breakdown table lists CAP-6's start as "Nov 1": that is the final pass. R1's "draft Tue 27 Oct" uses only data already paid for, so both hold.) |
| **Budget** | **0 new runs · 0u · 0 GPU-hours.** It reuses 411–420 full runs (paid by A02, A03, A05, A07–A14, A16, A17, A19, A20; reconciled with R1 in §4.4), 14 A08 CPU cells and 77 A17 F0 pool jobs (49,280 one-shot proposals). CPU: under 1 h on ≤ 16 CPUs. Developer time ≈ 2.4 d (§4.4); 0.5 d of it is inside E11b, the other ≈ 1.9 d is A23-local and not in R2 |
| **Models** | No model is served. The points span O120, O20, Q0.8, Q2, Q4, Q9 and Q122 (§7 R9), plus the A19 role mixes |
| **Status** | Not started |
| **Depends on** | E1, E3, E4, E7 (A19 points only), X5, X7 + E11d (F0 curves), E11a (G2 memo), E11b (`fetch_wandb.py`, `recovery_detector.py`, `pareto.py`), X10; A09-T15 (δ); data from A02, A03, A05, A07–A14, A16, A17, A19, A20; A06 drift verdicts; A21 anytime curves; DF Sat 31 Oct |
| **Feeds into** | Fig. 2 (right panel); a Fig. 3 cost panel or inset; the Table 2 "Δ tokens" column (consistency); the C1 and C5 cost sentences; the appendix energy accounting; the replication package; A21 OQ1 (the `flop_proxy` token basis, decided in §2.3) |

*Every "hoped" result is a hypothesis. Code references: RougeOne `nested-cv-rebuttal` @ `07cbf93` (`C:/code/PHD/RougeOne`). The checked-out tip `1bdaf42` differs only under `paper/`, and every line cited below was re-read at that tip. Never cite the worktree copy of the code (01 Appendix A item 16). Short paths: `main.py` = `rogue_one_main.py`; `sci/agent.py`, `ext/agent.py`, `tst/agent.py` = `rogueone/llm/agents/core/{scientist,extractor,tester}/agent.py`; `kn/web_search.py` = `rogueone/llm/agents/core/knowledge/web_search.py`; `config.py` = `rogueone/utils/config.py`; `clients.py` = `rogueone/llm/clients.py`. There is no `cfg.py`. "01 §n" = `docs/ablations/01_shared_infrastructure.md`.*

---

## 1. What we test and why it matters

We put every post-freeze configuration from T1-minimum on one chart. The vertical axis is how much planted structure it finds (`rec_sel` on SYN-P). The horizontal axis is what it cost: completion tokens, a FLOP proxy and GPU energy. We then ask which configurations are not beaten on both axes at once. No new runs are needed.

The figure matters for two reasons. It answers the reviewer question "is the multi-agent design just 3× the compute?", because SET-phased, SE|T, Restarts, L0 and the minimal harness sit on the same token axis as Full's anytime curve. It also gives C5 its cost reading: does a smaller backbone, or a big-Extractor / small-S+T mix, reach most of O120's recovery for a fraction of its FLOPs? If Full-v2 is dominated, this figure is the fallback narrative, and it is published as such.

*Not in T1-minimum:* the master plan's "effort levels" (the reasoning-effort ladder is T2) and the S|ET and Solo-auto points (E10 is cut). They are not plotted.

## 2. Background you need

**2.1 How cost is recorded in the code today** (verified at `07cbf93`):
- **Per-call tokens.** Every role talks to vLLM through `VLLMCompatibleAsyncOpenAI` (`clients.py:471`).
  - `_wrap_completions` (`clients.py:516-562`) records usage once per returned response (`:559`).
  - `_record_token_usage` (`:499-514`) adds it to the module-level accumulators `_AGENT_INPUT_TOKENS` / `_AGENT_OUTPUT_TOKENS` / `_AGENT_TOTAL_TOKENS` (`:32-34`). Nothing resets them, so they accumulate over the whole container run, across folds.
  - It then calls `log_llm_token_usage`, which logs W&B history keys `LLM_tokens/<agent>/{input,output,total}` and `LLM_tokens_cumulative/<agent>/…` (`rogueone/utils/wandb_utils.py:136-167`). Each call is a separate `wandb.log` with `step=None` (`:167`), so every row carries W&B's `_timestamp` (the time the call **ended**). A23 needs E3 to keep one history row per call with `_timestamp`; it uses that per-call time series for energy apportionment.
- **Agent labels** (the `agent_name` passed to `create_openai_client_for_vllm`):
  - `ScientistAgent` (`sci/agent.py:400`);
  - `ExtractorAgent` (`ext/agent.py:851`);
  - `TesterAgent` (`tst/agent.py:1341`);
  - `WebSearchAgent` (`kn/web_search.py:191`). W is built on the **global** `llm_cfg.endpoint` (`kn/web_search.py:188-191`; defined at `config.py:251`), not on its caller's endpoint. `config.py:217-220` documents this as deliberate.
  - A fifth label, `KnowledgeAgent` (`rogueone/llm/agents/core/knowledge/agent.py:56-59`), exists but is never called in a run: the Scientist only constructs it (`sci/agent.py:64`), and the Extractor's and Tester's "knowledge agents" are `WebSearchAgent` (`ext/agent.py:166`, `tst/agent.py:661`). A23-T1 checks that it logs 0 tokens.
- **Why old token exports were zero.** Without `include_usage` the stream never yields a usage chunk, so every record was (0, 0, 0). The fix is at `clients.py:324-335`. E3 must still prove non-zero tokens at G0a.
- **Uncounted re-rolls (a gap that E3 must close).** `_streaming_create_with_fallback` (`clients.py:564`) loops up to 6 attempts (`:584`; 3 empty-response retries + 2 splice re-rolls + 1). It discards splice-damaged completions (`continue` at `:631`) and retries empty ones; a retryable exception inside an attempt triggers a second request in the same attempt (`:589-602`). Only the final response reaches `:559`, so discarded attempts cost GPU time but no tokens are logged. Re-rolls are most frequent at small rungs, so leaving them out would flatter exactly the cheap points.
- **Per-role routing.** `AGENT_ROLES = ("scientist", "extractor", "tester")` (`config.py:221`). The per-role values are built from `ROGUE_ONE_<ROLE>_MODEL` / `_PORT` at `config.py:253-260` and returned by `model_for` / `endpoint_for` (`config.py:262-274`), falling back to `LLM_MODEL` / `LLM_PORT`. E7 adds `ROGUE_ONE_WEBSEARCH_ROUTING=caller`, so W follows its caller. A FLOP proxy for the A19 mixes is correct only if W's tokens are charged at the caller's model.
- **Qwen thinking is off** (`clients.py:204`, `enable_thinking=False`). Qwen completion tokens therefore contain no hidden reasoning. gpt-oss reasoning is *expected* to be included in `completion_tokens`; E3 verifies this against the trace on vLLM 0.22.0.
- **What is scored.**
  - Per-fold state resets at each fold (`main.py:543-544` → `_reset_fold_state`, `:417-457`), so fold-episodes are the unit.
  - The reported iteration is chosen by inner CV (`compute_fold_aggregation`, `main.py:1074`; selection metric `:1109-1134`).
  - The outputs are written at `main.py:1350-1365` (`fold_aggregation_summary.csv`, `fold_aggregation_best_iterations.csv`, `fold_aggregation_active_features.csv`).
- **No wall-clock timing exists in `main.py`** (it does not import `time`). `wall_s` comes from E3, with W&B `_runtime` as the fallback.

**2.2 What the shared tasks add** (01 §3):
- **E3:** `tok_compl_<role>` including reasoning and re-rolls; `tok_search_<role>` charged to the caller; `llm_calls_<role>`, `rerolls`, `wall_s`; tokens per role **per iteration**; traces (`ROGUE_ONE_LLM_TRACE=1`, `output/llm_traces.jsonl.gz`); `run-output` artifacts. It also adds the NVML energy sidecar `scripts/agent2027/energy_sampler.py` (1 Hz on the session's GPUs, logged to W&B run `<SESSION_ID>__energy`; container `ro27-<session_id>-energy`).
- **E1:** config keys `rung_S/E/T`, `model_S/E/T/W`, `server_topology`, `parallel`, `session_id`, `manifest_row`, `code_phase`, `image_tag`, and every flag as `flags/<NAME>` (R12).
- **E4:** search tokens charged to the caller.
- **X5:** the token proxy on port 8090 for LLM-FE and CAAFE.
- **X7 / E11d:** F0 pools (`f0_generate.py`, artifact `f0-pool`) and their scoring (`f0_score.py`).
- **A21:** Full truncated at whole iterations (the anytime curve), in `ANALYSIS__diag_selection_policy__<YYYYMMDD>` (tables `m2_anytime`, `m2_isocost`; file `isocost_points.csv`).
- **A06:** per-session drift verdicts (`ANALYSIS__sentinel_drift__<YYYYMMDD>`, column `drift_flag`).

**2.3 Definitions**
- **Units.** u = one real-data O120 run (10 iterations × 5 folds). A SYN-P run costs 0.7–1.0u and a small-model run 0.5–1.0u. Throughput is 2 u/h × 0.7 on ≤ 4 H100.
- **Instruments.**
  - SYN-P 7 = `synp_p1`, `synp_p2`, `synp_p3`, `synp_p4`, `synp_p5`, `synp_p6a`, `synp_p6b`. Proposed types: C = p3, p4, p5; R = the rest (E8a confirms them in `synp_meta/registry.yaml`).
  - **Primary population** = the discriminating SYN-P 7 tasks. A task is non-discriminating if OpenFE or AutoFeat recovers φ in ≥ 14/15 episodes (seeds {42, 4, 5}) in the A08 CPU screen; P1 is expected to be excluded. The list is fixed at M5S (Tue 13 Oct) and written into X10.
  - Core-10 = `balance-scale`, `junglechess`, `vehicle`, `heart`, `credit-g`, `myocardial`, `housing`, `insurance`, `crab`, `wine`. HC1-6 and CAP real-4 are subsets of it.
  - `synp_p7` is used in Table E only.
- **Recovery `rec_sel`** (primary 1). In the inner-CV-selected iteration, an active non-fallback feature has \|Spearman\| ≥ 0.95 with φ on the training rows, or NMI ≥ 0.9 for the rule families (P3, P5, P7). Raw columns and single-column monotone transforms never count. It is read per fold-episode from `analysis/recovery_detector.py`, the same table that `glmm_holm.py` uses.
- **Success.** `func_success` = the run finishes within the cap, scores ≥ `base_no_fe` − 2 × fold SD, **and** the selected iteration holds ≥ 1 accepted non-fallback LLM feature.
- **Cost axes.** Tokens of different models are not comparable cost, so the primary axis depends on the panel:
  - **Panel A** (one backbone, O120): primary `tok_compl_total` per SYN-P run; secondary `flop_proxy`, `energy_j`.
  - **Panel B** (different backbones and role mixes): primary `flop_proxy`; secondary `energy_j`, and `tok_compl_total` for reference.
  - **Appendix only:** `wall_s`, because it depends on server layout.
- **FLOP proxy.** `flop_proxy` = Σ_roles 2 · N_active(`model_<role>`) · (`tok_prompt_<role>` + `tok_compl_<role>`), where both token counts include the W calls charged to that role (E3/E4), so W is charged at its caller's model.
  - The sensitivity version, `flop_proxy_compl`, uses completion tokens only. This closes A21's OQ1: prompt + completion is primary, and A21 follows.
  - N_active values: Q0.8 0.8B · Q2 2B · Q4 4B · Q9 9B · Q122 10B · O20 3.6B · O120 5.1B. If A04 triggers the Qwen3 fallback ladder, use that ladder's parameter counts (Qwen3-0.6B 0.6B, -1.7B 1.7B, -4B 4B, -8B 8B, -32B 32B).
  - The proxy ignores attention and KV-cache cost, prefix caching and MoE routing. This is disclosed.
- **Energy `energy_j`** (01 §4.6: "apportioned to runs by token share per server"). Inputs: the session's energy run `<SESSION_ID>__energy` and the per-call token history of every run in the session.
  1. Map each GPU id to its server from the session's launch line (`GPUS`, or `SERVERS` for `ladder_session.sh`) in `~/ro27_<SESSION_ID>.job.log` and the runs' `server_topology` key. Integrate each server's power (sum over its GPUs, trapezoid rule) into 60 s bins aligned to the first sample.
  2. Assign each call's completion tokens to the server it was sent to (the role's `port_<role>`; W calls to their caller's server under E7) and to time: spread uniformly over the call's [start, end] if E3's trace records both, otherwise put all of them in the bin holding the call's `_timestamp`. Split each bin's energy across runs in proportion to their tokens on that server in that bin.
  3. A bin with no tokens on a server is split equally among the runs alive in it. A run is alive from its W&B start time to start + `_runtime`.
  4. A bin with no run alive is session overhead (server start-up, model load, tear-down). It is reported, not charged.
  5. A mixed run (A19) sums its share on each server.
  6. **Sensitivity:** every bin split equally among the runs alive, ignoring tokens.
  7. **Coverage** of a run = the share of its alive seconds in which every GPU of its server(s) has a power sample within 2 s. Energy is valid for that run only at ≥ 95% coverage (R-A23-5).
  8. This is GPU energy only. Because it uses token shares, it is not independent of tokens. Its added information is hardware efficiency across layouts (O120 on TP4 or TP2 per D5, vs 4 × TP1 small rungs).
- **Dominance and front.** Configuration *a* dominates *b* on a cost axis if rec(*a*) ≥ rec(*b*) and cost(*a*) ≤ cost(*b*), with at least one strict. The front is the non-dominated set, computed **separately per panel and per cost axis** (panel A: its configurations including Full; panel B: its configurations including Full and `fx_pajM`). Reference points (§4.1) never enter a front.
- **Front membership** `p_front` = the share of paired bootstrap resamples (§4.5) in which a configuration is non-dominated within its panel on that axis.

## 3. Hypotheses and interpretation (all are hypotheses)

Panels A and B are classified separately. Within a panel, the **first** matching row in table order applies. The conditions use the rules in §6 and are fixed in X10.

| Outcome | Panel | Pre-registered condition | Meaning / paper wording / action |
|---|---|---|---|
| **Measurement failure** | A or B | R-A23-5 invalidates the panel's primary axis (token validation fails for a contributing session) | Tokens: fall back to `llm_calls_total` and `wall_s` (within-session comparisons only, 01 §4.8 rule 1). Panel B then has no FLOP axis: it is shown as a descriptive scatter on `wall_s` without a front, and disclosed. (Energy coverage below 95% does not classify a panel: R-A23-5 drops the energy axis for the affected points only, and FLOPs are reported for them.) |
| **Opposite** | A | `llmfe`, `caafe` (against its matched Full marker), `fx_pajM` or `set_phased` meets R-A23-3 against Full on tokens; or Full reads "dominated" (R-A23-1) | The pre-written fallback narrative: "decomposition is not justified by cost at 120B". Published as such, with least privilege (if E13 lands) and per-role allocation as the remaining arguments. |
| **Hoped (A)** | A | Full reads "on the front" on tokens, and none of the four configurations above meets R-A23-3 | *"On one backbone, no single-context, single-LLM or minimal-harness alternative reached Full's recovery at lower token cost."* `fx_*` cells with a− or j− that meet R-A23-3 are C2's "reducible" context, not a C1 opposite. |
| **Null (A)** | A | Otherwise (for example Full reads "near the front") | Report the front as it is; the C1 cost sentence says "not resolved on cost". |
| **Hoped (B)** | B | `mix_1b` (S and T on Q0.8, E on O120) meets R-A23-3 against Full on `flop_proxy` and reads "on the front" | *"Per-role model allocation moves the system onto the cost frontier: with S and T at 0.8B, recovery stays within δ at X% of Full's FLOPs."* Disclose that `mix_1b` crosses families. If R-A23-4 marks the point for fallbacks, use A19's "deterministic parts suffice" reading instead. |
| **Hoped variant (B)** | B | `mix_1b` does not, but some `f2_<rung>` does | *"A smaller backbone reaches Full's recovery within δ at X% of its FLOPs."* No per-role allocation claim. |
| **Null B** | B | No configuration meets R-A23-3, and no configuration's paired `rec_sel` difference from Full has a 95% CI that excludes 0 | Recovery cannot be told apart at this sample size. The figure becomes a cost ranking of designs that are not separated on recovery. C2a/C2b's token co-primary still comes from A12, not from this figure. |
| **Null A** | B | Otherwise: every point cheaper than Full loses recovery beyond what R-A23-2 allows | No cheap configuration is competitive; the front is a pure trade-off with Full at its high end. Report it as it is. C5 rests on A17/A19 alone. |

## 4. Design

**4.1 Control, points and panels.** The reference is A05 `full` (Full-v2, O120). Each configuration is one point: mean `rec_sel` against mean cost per SYN-P run.
- **Panel A (Fig. 2, right):** O120 architecture and loop arms against the A21 Full anytime curve. The front configurations are:
  - `full`;
  - `set_phased`;
  - all 8 `fx_*` cells, with `fx_pAJM` (= SE|T) and `fx_pajM` (minimal harness) highlighted;
  - `restarts` and `l0`;
  - external: `llmfe`.
  - **Reference points (never in a front):** `caafe`, drawn on the classification subset next to a matched Full marker (§4.5); A09 `oracle` and `misdirected` as hollow markers (the oracle uses privileged information); A08 `openfe` and `autofeat` (seed 42) as horizontal dashed lines at their screen recovery on the primary population, labelled "brute force, 0 LLM tokens", because 0 cannot be placed on a log axis (A21 hands these "no iso-cost point" arms to A23).
- **Panel B (Fig. 3 panel or inset; appendix if space runs out):** capability on the FLOP axis, with `rec_sel` and `func_success` rows. The points are:
  - `f2_<rung>` for every rung, with O120 = A05 `full`;
  - `mix_esmall` and `mix_1b`;
  - `fx_pajM_O20` and `fx_pajM`;
  - the `f0pool_<rung>` best-of-k curves (k = 1..128, scored by E11d) drawn as lines, never in the front (01 §4.8 rule 3).
- **Panel C (appendix):** real data, as cost ratio vs Full against `delta_std`, with margin M. It covers A07, A13, A16 Core-10, A17 real-4 and A19 real-4. Descriptive only.
- **Panel D (appendix, pre-freeze code):** the A02 R curve (`R1`, `R3`, `R10`) and A03 `replace` / `augment`. These points never appear on the frozen front (01 §4.8 rule 6).
- **Table E (appendix):** `synp_p7` cost and recovery for the A14 arms, A05 `full` and the A09 `oracle`.

**4.2 Fixed settings.**
- The primary population is the discriminating SYN-P 7 tasks at **fold seed 42**. Every configuration has that partition (r0, plus r1 where it exists). LLM-FE uses its native seed-42 partition; A08-T4 checks that it equals A05's seed-42 partition, in which case it is paired by fold, otherwise by task only.
- Sensitivity analyses (A23-T17) use all seeds and all 7 tasks.
- Costs are reported per SYN-P run of 10 iterations × 5 folds; all arms hold 10 iterations fixed.
- **Run selection.** A run enters only if its `manifest_row` is a row of the owner's manifest (`manifests/agent2027/<AID>_<short>.csv`) whose `notes` do not contain `smoke` or `pilot`, and its `code_phase` is `v2` (`prefreeze` for panel D). Smoke rows can share a run name with a real row (A10, A13), so runs are never selected by name alone. If several W&B runs share a `manifest_row` (crash + retry), the last finished one is used.

**4.3 Run matrix (all runs reused; W&B entity `PlanBench`, project `RogueOne-AGENT2027` unless stated otherwise)**

| Panel | Source | `arm_id` | Instrument × seeds | Runs | Run-name pattern |
|---|---|---|---|---|---|
| A, B anchor | A05 | `full` | SYN-P 7 × f42 r0, f42 r1, f4 r0 | 21 | `A05__full__synp_<id>__f<42/4>__r<0/1>` |
| A curve | A21 | Full truncated at k = 1..10 | from the A05 runs above | 0 | `ANALYSIS__diag_selection_policy__<YYYYMMDD>` (`m2_anytime`, `m2_isocost`) |
| A | A08 | `llmfe` | SYN-P 7, native seed-42 partition, r0/r1 | 14 | `A08__llmfe__synp_<id>__f42__r<0/1>` |
| A (reference) | A08 | `caafe` | SYN-P 7 classification tasks × {42, 4} | 6 (≤ 8 budgeted) | `A08__caafe__synp_<id>__f<s>__r0` |
| A (reference lines) | A08 | `openfe`, `autofeat` | SYN-P 7 × 42 (CPU cells) | 14 cells | `A08__<arm_id>__synp_<id>__f42__r0` |
| A | A10 / A11 | `restarts` / `l0` | SYN-P 7 × {42, 4} | 14 + 14 | `A10__restarts__synp_<id>__f<s>__r0`, `A11__l0__synp_<id>__f<s>__r0` |
| A | A12 | 8 `fx_*` cells | SYN-P 7 × 42 | 56 | `A12__<arm_id>__synp_<id>__f42__r0` |
| A | A16 | `set_phased` | SYN-P 7 × {42, 4} (the 2 pilot rows are excluded) | 14 | `A16__set_phased__synp_<id>__f<s>__r0` |
| A (reference) | A09 | `oracle`, `misdirected` | SYN-P 7 × 42 | 14 | `A09__<arm_id>__synp_<id>__f42__r0` |
| B | A17 | `f2_Q0.8`, `f2_Q4`, `f2_Q122` | SYN-P 7 × {42, 4} | 42 | `A17__f2_<rung>__synp_<id>__f<s>__r0` |
| B | A17 | `f2_Q2`, `f2_Q9`, `f2_O20` | SYN-P 7 × 42 | 21 | same |
| B curves | A17 | `f0pool_<rung>` | 128 proposals × 5 folds × 7 tasks, per rung-seed (11 rung-seeds) | 77 jobs, 49,280 proposals | `A17__f0pool_<rung>__synp_<id>__f<s>__r0` (artifact `f0-pool`); scores from E11d |
| B | A19 | `mix_esmall`, `mix_1b` | SYN-P 7 × {42, 4} | 28 | `A19__<arm_id>__synp_<id>__f<s>__r0` |
| B | A20 | `fx_pajM_O20` | SYN-P 7 × 42 | 7 | `A20__fx_pajM_O20__synp_<id>__f42__r0` |
| C | A05 | `full` | Core-10 × f42 r0, f42 r1, f4 r0 | 30 | `A05__full__<dataset_id>__f<s>__r<r>` |
| C | A07, A13, A16 | `errors_hidden`, `search_off`, `set_phased` | HC1-6 × {42, 4}; Core-10 × 42; Core-10 × 42 | 12 + 10 + 10 | R12 pattern |
| C | A17, A19 | `f2_<rung>`, `mix_*` | CAP real-4 × 42 (per manifest) | 24 + 8 | R12 pattern |
| D | A02, A03 | `R10`/`R3`/`R1`; `replace`/`augment` | SYN-P 8 × 42; Core-10 × 42 (+ SYN-P 7 add-on) | 24 + 20 (+7) | project `RogueOne-AGENT2027-prefreeze` |
| E | A14, A05, A09 | `kb_*`, `full`, `oracle` | `synp_p7` | 18 + 3 + 1 | R12 pattern |
| flags | A06 | `sentinel`, `sentinel_Q4` | per session | 9–17 | drift verdicts only (`drift_flag`) |
| conditional | RV1 / RV2 / RV3 | CAP-5 cells; `search_off` f4; foldover `fx_*` | if triggered | RV2 10, RV3 28, RV1 as triggered | added to B / C / A |

**4.4 Budget arithmetic.**
- **New:** 0 runs, so 0u and 0 GPU-hours. This matches the breakdown (CAP-6: 0 runs, 0u) and R1.
- **Reused full runs** (all paid by their owners):
  - panels A + B: 21 + 14 + (6–8) + 28 + 56 + 14 + 14 + 63 + 28 + 7 = **251–253**;
  - panel C: 30 + 12 + 10 + 10 + 24 + 8 = **94**;
  - panel D: 24 + 20 (+7) = **44–51**;
  - panel E: 18 + 3 + 1 = **22**;
  - **total 411–420**.
- **Reconciliation with R1.** R1's A02–A20 runs, excluding A06's sentinels and the A03 add-on, are 416 (R1's 425–433 adds A06's 9–17). A23 leaves out A05's `synp_p0` run (1), A16's 2 pilot runs and 0–2 budgeted-but-unrun CAAFE rows (411–413), and adds the A03 add-on (0–7): 411–420. A06's sentinels are read for drift flags only. A01 (outside the window, entity `brainflux`), A04 (fold-runs), A15 (no new runs), and A18 (replays) supply no points.
- **Not runs:** 14 A08 CPU cells and 77 A17 F0 pool jobs.
- **CPU:** a W&B scan of ≈ 420 runs' per-call history (`scan_history`) and the session energy runs, plus a bootstrap of B = 2,000 resamples × ≈ 25 front configurations (panels A and B) × 3 axes. All 47 or so points over the five panels are summarised once. Estimated under 1 h on ≤ 16 CPUs; A23-T8 measures it.
- **Developer time ≈ 2.4 d** (sum of §5 efforts):
  - checks and pre-registration (T1–T7, T13): 0.30 + 0.10 + ≈ 0.17 (16 days × 5 min) ≈ 0.57 d;
  - code (T8–T12): 0.75 d, of which 0.5 d is A23's share of E11b (1.75–2 d) and 0.25 d (energy apportionment) is on top of R2;
  - analysis (T15–T18): 0.65 d, inside the Nov 1–3 statistics window;
  - paper outputs (T19–T22): ≈ 0.47 d, inside the Nov 4–12 writing calendar;
  - optional CAIR run (T14): 0.05 d.

**4.5 Aggregation, pairing and uncertainty.**
- **Point estimate:** y = mean `rec_sel` over the fold-episodes of the population; x = mean cost per run over the configuration's runs in the population.
- **Uncertainty:** a paired two-stage cluster bootstrap with `seed=20261016` and B = 2,000.
  - **Stage 1** resamples the population's tasks with replacement, with the **same task draw for every configuration** in the panel. `caafe` and its matched Full marker use their own draw from the discriminating classification tasks.
  - **Stage 2** resamples the 5 outer-fold indices with replacement within each drawn task, again with the same draw for every configuration, since all share the seed-42 partition. (LLM-FE joins this draw only if A08-T4 confirmed the partition; otherwise its folds are drawn independently.) Each drawn fold contributes, for every run of the configuration on that task (r0, and r1 where it exists), that fold-episode's `rec_sel` and that fold's cost. Fold cost = the run's per-iteration cost keys (E3) summed over the fold's 10 iterations. Where per-fold cost does not exist (the X5 external arms, or A21's G1 fallback), use the run's cost ÷ 5.
  - Cost per run in a resample = 5 × mean fold cost.
- **Outputs per configuration:**
  - 95% percentile CIs on both axes;
  - `p_front` per axis;
  - the cost ratio vs Full, computed within each resample, with its 95% CI;
  - the 90% and 95% CIs of the paired `rec_sel` difference vs Full (for R-A23-2 and §3);
  - tokens (and FLOPs) per recovered episode, reported as "no recovery" if the count is 0.
- **Matched Full marker for `caafe`:** Full's `rec_sel` and cost on the same discriminating classification tasks, using A05 f42 r0, f42 r1 and f4 r0 (the A08 CAAFE population).
- **Iso-cost overlay:** A21's Full curve, truncated at whole iterations only. If an arm spends less than Full's first iteration, A21 reports "no iso-cost point", and so does this figure.
- There is no randomisation: the analysis is post hoc. The external arms are paired by task (LLM-FE also by fold if A08-T4 passes).

**4.6 Failure handling.**
- **ITT:** a failed or capped run contributes all 5 episodes at `rec_sel = 0` (01 §4.5). Its cost counts as spent (tokens up to the failure or the cap) and is never imputed.
- **Retries:** a retried crash is costed as the scored retry. The crashed attempt's tokens are listed in the appendix.
- **No reruns:** A23 orders no reruns, and in particular none for energy alone.
- **Missing data:**
  - A source that is cut or killed simply leaves no point, and the cut is noted.
  - Points from sessions that A06 marks `drift_flag = 1` are annotated, and a sensitivity analysis excludes them.
- **The 6 h cap** comes from the owners' manifests (`cap_hours=6`, launcher `RUN_CAP_HOURS=6`).

## 5. Tasks

### 5.0 How checks are recorded and run

- **Check log.** Every prerequisite and coverage check (T1–T6, T13) writes one row to W&B table `a23_checks` in run `ANALYSIS__pareto_checks__<YYYYMMDD>` (entity `PlanBench`, project `RogueOne-AGENT2027`, group `ANALYSIS`). Columns: `check_id`, `session_id`, `run_name`, `value`, `pass`, `note`. One run per day on which a check is made. After Tue 27 Oct, `pareto.py` also writes the coverage table into `ANALYSIS__pareto__<YYYYMMDD>`.
- **Where to run.** On the local workstation (it only reads W&B; `WANDB_API_KEY` from the PlanBench `.env`), or on CAIR inside a container as in A23-T14 / 01 §5.2 step 7. Never with host Python on CAIR.
- **Query pattern** (key names other than R12/R13 are whatever E3 logs; adjust once E3 lands):
  ```python
  import wandb
  api = wandb.Api()
  P = "PlanBench/RogueOne-AGENT2027"   # "PlanBench/RogueOne-AGENT2027-prefreeze" for the G0a smoke
  for r in api.runs(P, filters={"tags": {"$in": ["<SESSION_ID>"]}}):
      s = r.summary
      print(r.name, r.config.get("manifest_row"), r.state, s.get("_runtime"),
            [s.get(k) for k in ("tok_compl_scientist", "tok_compl_extractor", "tok_compl_tester",
                                "llm_calls_total", "rerolls")])
      # per-call rows: always scan_history; history() down-samples to 500 rows
      n_calls = sum(1 for _ in r.scan_history(keys=["_timestamp", "<E3 per-call token key>"]))
  ```

### 5.1 Prerequisites (verify before the draft; record each result in `a23_checks`)

- [ ] **A23-T1** [type: data] (0.1 d, due Mon 12 Oct, G0a) — Check E3 on the G0a smoke run (`ROGUE_ONE_SMOKE_FOLDS=1`, project `RogueOne-AGENT2027-prefreeze`, tag `smoke`):
  - `tok_compl_scientist`, `tok_compl_extractor` and `tok_compl_tester` are > 0;
  - `tok_search_<role>` is > 0 for every role that searched (per the retrieval transcript);
  - the trace record count equals `llm_calls_total`;
  - `rerolls` exists and equals the number of discarded-attempt records in the trace (`clients.py:584-631`); 0 is acceptable only if the trace also has none;
  - per role, the trace's summed completion tokens equal `tok_compl_<role>` within 1% (so re-rolls and search tokens are inside the role totals);
  - per-call history rows carry `_timestamp`, and no tokens are logged under `KnowledgeAgent`.

  *Done when:* all six are recorded as pass. A fail is filed against E3 the same day and rechecked on the Fri 16 Oct freeze smoke (01 §2.2); if it still fails there, R-A23-5's token fallback is recorded for every v2.0 session.
- [ ] **A23-T2** [type: data] (0.05 d, due Mon 12 Oct) — The smoke session has a `<SESSION_ID>__energy` run (look in both projects). *Done when:* the run exists, its median sample interval per GPU is 0.9–1.1 s, and its GPU id set equals the session's `GPUS`.
- [ ] **A23-T3** [type: data] (0.05 d, due Sat 17 Oct) — The first v2.0 runs (A05, A07; session `S1016a`) carry `rung_S/E/T`, `model_S/E/T/W`, `server_topology`, `parallel`, `session_id`, `manifest_row`, `code_phase`, `image_tag` and every `flags/<NAME>`. *Done when:* a query over the session (§5.0) finds no missing key on any finished run.
- [ ] **A23-T4** [type: data] (0.05 d, due Wed 21 Oct, G0b-1; fallback Thu 22 Oct morning, before A19 launches) — In an A04 session-C mixed run (`mixcheck_esmall` or `mixcheck_1b`):
  - `model_W` equals its caller's model (E7, `ROGUE_ONE_WEBSEARCH_ROUTING=caller`);
  - Σ_role `tok_search_<role>` equals the summed completion tokens of the W calls in the trace (E4);
  - every per-call history row can be mapped to a server (its role, and for W calls the caller role).

  *Done when:* all three hold; otherwise A19's FLOP and energy points are flagged.
- [ ] **A23-T5** [type: data] (0.05 d, due Wed 21 Oct, after the first A08 GPU session, which starts Tue 20 Oct) — X5's per-client prompt and completion totals match the vLLM request log within 1%, and X5 logs one row per call with `_timestamp`. *Done when:* both are recorded. If the first fails, `llmfe`/`caafe` appear on the `wall_s` axis only. If only the second fails, their energy uses the equal-split method.
- [ ] **A23-T6** [type: data] (0 d, due Fri 23 Oct; G2 items Mon 19 Oct) — Copy the inputs into `a23_checks`:
  - from the G2 memo `ANALYSIS__g2_stats__20261019` (E11a): margin M, the informative-task list, and A19's cost threshold τ (used only to annotate the A19 points with A19's own 1B verdict);
  - from X10: the discriminating-task list (A08 screen, M5S Tue 13 Oct);
  - from A09-T15 (Fri 23 Oct, 18:00): the value of δ.

  *Done when:* all five values are in `a23_checks`.

### 5.2 Implementation (A23's share of E11b; code under `scripts/agent2027/analysis/`)

- [ ] **A23-T7** [type: writing] (0.1 d, due Thu 15 Oct) — Write the A23 paragraph for the X10 pre-registration: the axes per panel, population, run selection, FLOP and energy definitions, bootstrap, `p_front` thresholds, equality band, the §3 classification and the decision rules (§6). *Done when:* it is in the `agent2027-prereg` commit (Fri 16 Oct).
- [ ] **A23-T8** [type: code] (0.2 d, due Tue 27 Oct) — `pareto.py build-points`. Through `fetch_wandb.py`, pull every run in the §4.3 matrix by group and `job_type`, applying the §4.2 run selection: identity keys, R13 cost keys (summary and per-iteration), `success`, `func_success`, `cap_hit`, `itt_imputed`, `fallback_share`, and per-episode `rec_sel` from the detector table. Apply ITT and add the `ACTIVE_PARAMS` table (§2.3). *Done when:* `pareto_points` (one row per run) logs to `ANALYSIS__pareto__<YYYYMMDD>`, and on every A05 run its `tok_compl_total` equals the W&B summary value exactly.
- [ ] **A23-T9** [type: code] (0.25 d, due Tue 27 Oct) — `pareto.py energy`: the §2.3 apportionment, from `<SESSION_ID>__energy` plus per-call `_timestamp` history, with the equal-split sensitivity and per-run coverage. Expose `apportion_energy()` so that `process_metrics.py` (A05, A13, A19) imports the same `energy_j`. *Done when:* on one session, Σ run energy + overhead = session energy within 0.5%.
- [ ] **A23-T10** [type: code] (0.05 d, due Tue 27 Oct) — Recompute `flop_proxy` and `flop_proxy_compl` per run from `model_<role>`. *Done when:* the result agrees with E3's `flop_proxy` within 1% where E3 logs it; any difference in token basis is resolved in favour of the X10 definition.
- [ ] **A23-T11** [type: code] (0.15 d, due Tue 27 Oct) — `pareto.py front`: dominance per panel and axis, the §4.5 bootstrap, `p_front`, cost ratios vs Full, paired `rec_sel` difference CIs and tokens per recovered episode. *Done when:* a toy set of 4 configurations with a known front gives `p_front` = 1 for the front and 0 for the dominated points.
- [ ] **A23-T12** [type: code] (0.1 d, due Tue 27 Oct) — `pareto.py plot`: panels A–D and Table E. Use log-x, 95% CI whiskers, hollow reference markers, the brute-force reference lines and the A21 curve; panel D is labelled "pre-freeze code". *Done when:* the figures render from `pareto_points` alone.

### 5.3 Execution on CAIR (no GPU job)

**Rules** (01 §5.1; they apply to every command below):
- **Containers only.** Every execution happens inside Docker, including analysis and W&B queries. The host issues `docker` commands only, plus `nvidia-smi`, `docker ps`, `tail`, `mkdir` and `grep` for pre-flight and monitoring. Never run `python …` on the host. Any `docker compose run` over SSH needs `-T`.
- **GPU pre-flight before any GPU container.** A23 launches none, and its container must never get `--gpus`. If a GPU step is ever added, first run 01 §5.2 step 1 and **stop and report** unless the needed GPUs are free:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader && echo '--- compute apps ---' && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader && echo '--- containers ---' && docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'"
  ```
- **≤ 4 of the 8 H100s**, ever, without Henrik's explicit approval in chat.
- **W&B is the system of record.** Runs are stateless; results live in `ANALYSIS__*` runs. The key comes from `<CAIR_HOME>/code/PlanBench/.env`.
- **SSH:** non-interactive one-shot commands with `-o BatchMode=yes -o ConnectTimeout=30`. Long jobs run detached with `nohup`, and the log goes in `~`, never in root-owned run folders.
- `<ROGUEONE_CAIR_PATH>` = `<CAIR_HOME>/code/RougeOne` is **inferred, not verified** (01 §5.2). Confirm once with `ssh CAIR-gpu14 "ls -d ~/code/RougeOne"`.

The primary place to run the analysis is the local workstation, since it only reads W&B.

- [ ] **A23-T13** [type: data] (5 min/day, Fri 16 – Sat 31 Oct; due Sat 31 Oct) — After each session ends, check it (§5.0): an energy run exists; every run's energy coverage is ≥ 95% (§2.3 step 7); tokens are non-zero for every role; `_timestamp` rows exist. *Done when:* `a23_checks` has one row per session for every session in the window. A session that fails is flagged at once to the session's owner, while reruns are still possible (the rerun window ends with launches on Fri 30 Oct).
- [ ] **A23-T14** [type: run] (0.05 d, optional, CPU only; as needed, latest Tue 3 Nov) — Run on CAIR when local resources fall short.
  1. **Code to the server** (01 §5.2 step 0): commit and push `agent2027/v2` locally, then
     ```
     ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15' git fetch git@github.com:HenrikNorkart/RougeOne.git agent2027/v2 && git checkout -B agent2027/v2 FETCH_HEAD && git log -1 --format='HEAD: %h %s'"
     ```
  2. **Check what is running** (sets the CPU cap) and **launch detached:**
     ```
     ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker ps --format '{{.Names}}' | grep -E '^(ro27-|rogueone-vllm-)' || echo 'no session running'; mkdir -p ~/ro27_analysis && nohup sh -c 'docker run --rm --name ro27-analysis-pareto --cpus 4 --user \$(id -u):\$(id -g) -e HOME=/tmp -e WANDB_DIR=/workspace/out -e PYTHONPATH=/workspace/repo --env-file <CAIR_HOME>/code/PlanBench/.env -v <ROGUEONE_CAIR_PATH>:/workspace/repo:ro -v \$HOME/ro27_analysis:/workspace/out -w /workspace/repo --entrypoint python3 rogue_one:agent2027-v2.<N> scripts/agent2027/analysis/pareto.py all --out /workspace/out; echo \"=== RO27_ANALYSIS_END pareto rc=\$? ===\"' > ~/ro27_analysis_pareto.job.log 2>&1 & echo launched"
     ```
     Use `--cpus 4` while a session runs and ≤ 16 after DF. If the image lacks `matplotlib`, run `plot` locally.
  3. **Monitor** (as a background command; exit code 255 means the network dropped, not the job):
     ```
     ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 CAIR-gpu14 "while ! grep -q 'RO27_ANALYSIS_END' ~/ro27_analysis_pareto.job.log 2>/dev/null; do sleep 60; done; tail -20 ~/ro27_analysis_pareto.job.log"
     ```

  *Done when:* the log ends with `rc=0` and the W&B analysis run exists.

### 5.4 Analysis

- [ ] **A23-T15** [type: analysis] (0.25 d, due Tue 27 Oct) — Draft front on the data so far (A05, A07, A08 GPU, A09–A11, A13, A16 SYN-P, early A12/A17/A19/A20), labelled preliminary, using A21's draft `isocost_points.csv` (A21-T12) and δ from A23-T6. *Done when:* `ANALYSIS__pareto__20261027` holds the points, `p_front` and draft panels, plus an issues list (missing axes, flagged sessions).
- [ ] **A23-T16** [type: analysis] (0.25 d, due Tue 3 Nov; starts Sun 1 Nov, after DF) — Final front with all panels. The A21 curve (A21-T13) and the E11d best-of-k tables (A17-T18) must be final first; both are due by Tue 3 Nov. *Done when:* `ANALYSIS__pareto__<YYYYMMDD>` (≥ 20261101) holds the `pareto_points` and `pareto_front` tables and the figures.
- [ ] **A23-T17** [type: analysis] (0.1 d, due Tue 3 Nov) — Sensitivities:
  - (a) all 7 tasks;
  - (b) all seeds;
  - (c) fallback-only runs excluded;
  - (d) `flop_proxy_compl`;
  - (e) equal-split energy;
  - (f) A06-flagged sessions (`drift_flag = 1`) excluded;
  - (g) `func_success` as the y-axis;
  - (h) one joint front over panels A and B on `flop_proxy`.

  *Done when:* a table in `ANALYSIS__pareto__<YYYYMMDD>` lists, per sensitivity, every point whose R-A23-1 wording or §3 outcome changes (empty if none), and every change is in the appendix text.
- [ ] **A23-T18** [type: decision] (0.05 d, due Tue 3 Nov) — Apply R-A23-1…7 (§6) and fix the wording of each point; apply the §3 classification to panels A and B. *Done when:* the verdict column of `pareto_front` is filled and the §3 outcome rows for panels A and B are logged in the same run.

### 5.5 Paper outputs

- [ ] **A23-T19** [type: writing] (0.15 d, due Fri 6 Nov; Wed 4 – Fri 6 Nov) — **Fig. 2, right panel** = panel A. Choose the sentence from §3 according to the logged outcome. *Done when:* the panel and sentence are in the AGENT draft.
- [ ] **A23-T20** [type: decision] (0.02 d, due Wed 4 Nov) — Put panel B into Fig. 3 (as a panel or inset) or into the appendix; main text holds ≤ 3 figures. *Done when:* the choice is written in the draft's figure list.
- [ ] **A23-T21** [type: writing] (0.25 d, due Fri 6 Nov) — Write the appendix:
  - the energy accounting (method, coverage, overhead share, GPU-only scope);
  - the FLOP-proxy caveats;
  - panels C and D and Table E;
  - the full points table;
  - the pre-freeze disclosure.

  Check that Table 2's "Δ tokens" values equal `pareto_points`. *Done when:* the appendix is in the draft and there are no mismatches.
- [ ] **A23-T22** [type: writing] (0.05 d, due Thu 12 Nov) — The replication package contains `pareto.py`, `pareto_points.csv` and the bootstrap seed (X8, the webhook-key removal, is done before it). *Done when:* the package rebuilds the figures offline from `pareto_points.csv`.

## 6. Decision rules and gates (pre-registered in X10, Fri 16 Oct)

| Rule | Statement |
|---|---|
| **R-A23-1 Front wording** | `p_front` ≥ 0.8 → "on the front"; 0.5 ≤ `p_front` < 0.8 → "near the front"; `p_front` < 0.5 → "dominated". Computed per panel and per axis. The panel's primary axis (A: `tok_compl_total`; B: `flop_proxy`) sets the main-text wording; the other axes are reported in the appendix. |
| **R-A23-2 Equality band** | Two configurations are "equal on recovery" only if the 90% bootstrap CI of their paired `rec_sel` difference lies within ±δ. δ = max(0.25 × (oracle − Full) `rec_sel` gap, 0.10), the value committed by A09-T15 (Fri 23 Oct, 18:00; δ = 0.10 if the gap ≤ 0 or oracle rows are late). It is the same δ as A12's TOST and A19's 1B verdict. Descriptive; outside Holm. |
| **R-A23-3 "Cheaper at equal recovery"** | Requires R-A23-2 against Full, a point estimate of the cost ratio vs Full ≤ 0.70 on the panel's primary axis (matching the C2 30% rule), and a 95% upper bound of that ratio < 1.0. |
| **R-A23-4 Caveat markers** | A configuration with > 20% of its population runs `itt_imputed`, or mean `fallback_share` > 10%, is marked. "On the front" then carries the caveat, and sensitivity (c) is mandatory. |
| **R-A23-5 Axis validity** | Tokens are valid only if A23-T1 and A23-T13 pass for every contributing session (otherwise use `llm_calls_total` / `wall_s`). Energy is valid for a run only at ≥ 95% coverage. External points go on the token axis only if A23-T5 passes. |
| **R-A23-6 No mixing** | Pre-freeze points (D), `synp_p7` (E) and real data (C) never enter the SYN-P 7 front. A09, `caafe` (classification subset; shown with its matched Full marker), the A08 brute-force lines and the F0 curves are never in a front. |
| **R-A23-7 No confirmatory reading** | No contrast is tested here. Superiority claims come only from the R14 Holm family. If an R14 verdict and this figure disagree in direction, the paper uses the R14 verdict and reports the A23 reading as descriptive. |

**Gates.**
- **G1 (Fri 16 Oct):** the rules are committed.
- **G2 (Mon 19 Oct):** M, the informative tasks and τ are available; the discriminating tasks have been fixed since M5S (Tue 13 Oct).
- **Fri 23 Oct, 18:00:** δ committed by A09-T15.
- **G4 (Tue 27 Oct):** E11b is due. If `pareto.py` is late, the draft is a plain scatter from `fetch_wandb.py` without the bootstrap. The final front must have the bootstrap by Tue 3 Nov, or it reports point estimates labelled "no uncertainty".
- **DF (Sat 31 Oct, 24:00):** the data are final.
- **Kill dates:** none, since there are no GPU runs.

## 7. Risks and cut levers

| Risk | Mitigation |
|---|---|
| Tokens are zero or miss re-rolls (`clients.py:559`, `:584-631`), which biases cost in favour of small rungs | A23-T1 at G0a; R-A23-5 fallback |
| Per-iteration token attribution is missing (A21's top risk), so per-fold cost is unavailable | Bootstrap uses run cost ÷ 5 per fold (§4.5); disclosed |
| W&B `history()` down-samples to 500 rows, which would corrupt per-call energy apportionment | Always `scan_history` (§5.0) |
| W runs on the global endpoint (`kn/web_search.py:188-191`) if E7 slips | A23-T4; A19 FLOPs flagged |
| Energy depends on layout (TP2 vs TP4 per D5; 4 × TP1) and on token-share apportionment; `_timestamp` marks a call's end, so per-bin attribution is approximate | Tokens and FLOPs are primary; equal-split sensitivity; session totals are exact; method disclosed |
| ~6 informative tasks at 1 seed give wide CIs, and most points read "near the front" | Pre-declared wording and §3 classification; no confirmatory claims |
| The external arms differ in harness and partition (CAAFE is classification only) | Pairing by task (by fold for LLM-FE if A08-T4 passes); a matched Full marker; R-A23-5, R-A23-6 |
| Developer time (E11b is 1.75–2 d by Tue 27 Oct; A23 adds ≈ 1.9 d outside R2) | Draft fallback (§6); A23 is not in the engineering drop order |

**Cut levers that remove points (the figure survives every one):**
- **CL1:** no A19 real points.
- **CL2:** fewer A17 real points (10 fewer rows).
- **CL3:** fewer A07 points in panel C.
- **CL4:** no `caafe` reference point.
- **CL5:** fewer sentinels, so fewer sessions with a drift verdict.
- **CL6 and ED3:** fewer or no F0 curves.
- **CL7 and ED6:** no `l0`.
- **ED4 (K-SET Fri 23 Oct):** no `set_phased`.
- **ED1** (E13) removes only the least-privilege argument from the Opposite narrative; **ED2** and **ED5** do not touch A23.
- **Reserve-paid points (RV1–RV3)** are added only if triggered.

## 8. Definition of done

- [ ] The X10 paragraph is committed (Fri 16 Oct).
- [ ] A23-T1…T6 are recorded in `a23_checks` as passed, or the fallbacks are applied; δ is recorded (Fri 23 Oct).
- [ ] `pareto.py` passes its toy test and the A05 dry run (Tue 27 Oct).
- [ ] The draft front is logged on Tue 27 Oct; the final front, sensitivities, verdict column and §3 outcomes are logged by Tue 3 Nov.
- [ ] The Fig. 2 right panel, the Fig. 3 / appendix decision, the appendix and the Table 2 consistency check are done (Fri 6 Nov).
- [ ] The replication-package items are in place (Thu 12 Nov).

---

**New names introduced** (not in the §7 registry):
- **Metric:** `flop_proxy_compl` (completion-only FLOP proxy) and `p_front` (front membership).
- **`pareto.py` interface:**
  - subcommands `build-points`, `energy`, `front`, `plot`, `all`; option `--out`;
  - constant `ACTIVE_PARAMS`;
  - function `apportion_energy()`.
- **W&B:** tables `pareto_points`, `pareto_front` and the coverage table inside `ANALYSIS__pareto__<YYYYMMDD>`; check run `ANALYSIS__pareto_checks__<YYYYMMDD>` with table `a23_checks`.
- **Labels and CAIR names:**
  - panel labels A–E;
  - bootstrap seed `20261016`;
  - container `ro27-analysis-pareto`;
  - log `~/ro27_analysis_pareto.job.log` with end marker `=== RO27_ANALYSIS_END pareto rc=<rc> ===`;
  - folder `~/ro27_analysis/`.
- **Rule IDs** R-A23-1…7.
