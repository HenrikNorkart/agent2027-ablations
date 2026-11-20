# A13 · KT1-Off — Web search removed on Core-10 (cost + no-harm)

*Rogue One v2 design study · ICSE AGENT 2027 (8 pp IEEE) · submit **Fri 13 Nov 2026** · scope T1-minimum · written Tue 6 Oct 2026, verified Wed 7 Oct 2026. Every "hoped" result is a hypothesis, not a finding.*

*Code references are to RougeOne `nested-cv-rebuttal` @ `07cbf93` (`C:/code/PHD/RougeOne`). Every cited line was re-read on Wed 7 Oct with `git show 07cbf93:<path>`. The branch HEAD `1bdaf42` differs from `07cbf93` only in nine `paper/` files (two seed-sweep CSVs, two notes, five `paper/scripts/` files); none of them is cited here. **Line numbers will move on `agent2027/v2`** once E2–E5 land (they add code to the same files), so E9e locates every edit by the quoted string, not by line number. Short paths (01 §7 R0): `main.py` = `rogue_one_main.py`; `sci/agent.py`, `ext/agent.py`, `tst/agent.py` = `rogueone/llm/agents/core/{scientist,extractor,tester}/agent.py`; `kn/web_search.py` = `rogueone/llm/agents/core/knowledge/web_search.py`; `config.py` = `rogueone/utils/config.py`; `clients.py` = `rogueone/llm/clients.py`. There is no `cfg.py`. "01 §n" = `docs/ablations/01_shared_infrastructure.md`; if this document and 01 disagree on a shared item, 01 wins (01 §1).*

| Field | Value |
|---|---|
| ID | **A13 · KT1-Off** (arm `search_off`) |
| Claim / Holm contrast | C4: the cost side of the claim, and "pluggable". **A cost outcome outside the Holm family** (01 §7 R14). There is no confirmatory accuracy test; no-harm is reported only as a CI. |
| Tier | T1-minimum |
| Owner | Henrik |
| Start / finish | Smoke Wed 21 Oct afternoon (G0b-1). Launch **Wed 21 Oct, evening**, in the shared v2.3 O120 session `S1021a` (§4, A13-T10). Runs done by **Fri 23 Oct**. RV2 decision **Mon 26 Oct** (G3). Latest start **Thu 29 Oct** (L-LAST). Final statistics **Sun 1 – Tue 3 Nov**. |
| Budget | **10 new runs, 9–11u** (= T1-minimum breakdown and 01 §7 R1). That is 6.4–7.9 h of box time at 1.4u/h, about 26–31 H100-hours at TP4. Control: 0 new runs (A05 pays). Sentinel: 0 (A06 pays). Smoke row A13-901: 1 fold ≈ 0.14–0.2u, from the rerun reserve, as for the other G0b-1 smoke rows. Reserve RV2: 10 runs (10u, 01 §4.9), only if triggered. |
| Models | O120 `openai/gpt-oss-120b` for S, E and T on `vllm/vllm-openai:v0.22.0`. One server on port `8010`, TP4 (or TP2 per D5), `PARALLEL=6`. W is absent in this arm. |
| Status | Not started |
| Depends on | A05 (control runs, `sigma_d`), A06 (session sentinel), A12 (shares session `S1021a` and the smoke session `S1021b`); E1, E2, E3, E4, E5, E9e, E11a, E11b, X10; gates G1, G2, G0b-1; decision D5 |
| Feeds into | the KT1-Off row of Table 2; the C4 paragraph; Fig. 1 (W knob); A21 (iso-cost point); A23 (Pareto point); A14 (`kb_off` reuses this edit list and switch) · A12 (shared session `S1021a`/`S1021b`) |

## 1. What we test and why

We unplug the web-search channel from all three agents: the tool itself and every prompt line that asks for a search. We then run frozen v2 on the 10 Core-10 real datasets at fold seed 42, and ask two questions:
- How much cheaper is a run, in completion tokens, wall clock, LLM calls and Serper calls?
- Do held-out scores stay inside the pre-set no-harm margin M?

**Why it matters.** C4 says knowledge is a *pluggable, costed* channel. A13 is the only test that the system runs end to end with the channel removed, and the only measurement of what the channel costs. All old token logs are zero or empty (§2), so no estimate exists yet. A13 also delivers the ablation the NeurIPS version admitted it lacked: its No-Search arm removed only the Scientist's search.

## 2. Background you need

**The loop** (`main.py:536-571`, `:789-794`). Each iteration runs in this order:
1. The Scientist sets a focus (`main.py:548-555`).
2. The Extractor makes R rounds in one session (`main.py:563-571` → `ext/agent.py:965-975`; R is the loop at `:967`, made a knob by E2).
3. The Tester runs Stage A, assessment (`tst/agent.py:1483-1496`), then Stage B, pruning (`:1579-1591`).

Fold state is reset at `main.py:543-544` (`_reset_fold_state`, `:417-457`). All Core-10 datasets are in the legacy task list (`main.py:1536-1566`).

**The knowledge channel W.** Each role constructs its own `WebSearchAgent` (`sci/agent.py:67`, `ext/agent.py:166`, `tst/agent.py:661`; imported through the shim `rogueone/llm/agents/web_search.py`).
- Every call goes through `explain_query` (`kn/web_search.py:292-309`) into `_run_search_agent` (`:57-271`). That is an LLM summariser (`:194-242`) whose only tool is `_web_search` (`:74-186`).
- Serper is called at `search.run(variant)` (`:128`), with the key from `SERPER_API_KEY` (`:69`). Each call is recorded in the retrieval transcript `output/retrieval_transcript.jsonl` with `channel="web_search"` (`:96-167`, `:301-306`; `rogueone/utils/retrieval_audit.py`).
- The summariser always uses the global `LLM_MODEL` (`:238`; see the comment at `config.py:217-221`). Its settings come from `MEDICAL_KNOWLEDGE_AGENT_*` (`config.py:382-393`): temperature 0.7, at most 100 turns (`kn/web_search.py:259`).
- Its tokens are logged as `agent_name="WebSearchAgent"` (`kn/web_search.py:188-192`, `clients.py:499-514`). E3/E4 re-charge them to the calling role.
- Per-role call counts already reach W&B as `Scientist/tool_usage/search_tool` (`sci/agent.py:605`), `Extractor_tool_usage/search_in_literature_tool` (`ext/agent.py:1009`) and `Tester_tool_usage/search_in_literature_tool` (`tst/agent.py:2359`).

| Role | Tool (definition) | Tool list | Prompt lines that ask for search |
|---|---|---|---|
| S | `search_tool`, `sci/agent.py:135-174` | `:464` | `:414` ("…and compare with existing information from the web…"); `:431` ("(search, lookup, …) Always perform at least one web search."); `:438` ("Use search_tool actively when uncertain.") |
| E | `search_in_literature_tool`, `ext/agent.py:246-280`. Its docstring also offers pandas documentation lookup (`:252`). | `:939` | `:921` (the "Available tools" list) |
| T | `search_in_literature_tool`, `tst/agent.py:1041-1073` | Stage A `:1492` only; Stage B has none | none (Stage A instructions `:1426-1475` never mention search) |

**Not affected:**
- The `KnowledgeAgent` (`sci/agent.py:64-66`) is inactive: its tool is commented out at `:176-177` and `:463`.
- The Scientist's no-tool fallback already has `tools=[]` (`:564`).
- The team-context texts (`rogueone/llm/prompts/agent_roles.py:7-29`) never mention search.
- Class and method docstrings that mention search (`sci/agent.py:48`, `:87`; `ext/agent.py:150`, `:156`) are never sent to the model.

A call to a tool that is no longer listed would surface as a runner error at `sci/agent.py:530-545`, `ext/agent.py:977-992` or `tst/agent.py:1398-1415`.

**Historical prior (descriptive only).**
- The old No-Search arm removed only the Scientist's `search_tool` (`paper/69d60462bc7e9be26c88be13/appendix/reproducibility.tex:40`; `sections/limitations.tex:124-126`). Extractor and Tester kept the same `WebSearchAgent`; the paper's "dense-vector retrieval" description at `reproducibility.tex:32` does not match the code.
- Its mean Δ (No-Search − Full) is **+0.0007** over the 18 rows of `tables/ablation_classification.tex` (recomputed on Wed 7 Oct from the rounded 3-decimal table values; the master plan's "about 0.001"). It ran under the old protocol and vLLM 0.11.0, so it serves as a prior only.
- There is no token prior. In `paper/analysis_RogueTwo_Full_GPTOSS_120B_no_search.csv` (31 rows) every `tok_*` value is 0 or empty. In `paper/analysis_full120b.csv` (89 rows) only 5 rows have a non-zero `tok_scientist`, all with the same value (2,222), and every other `tok_*` is 0.

**Instruments: Core-10** (01 §4.1; 01 §7 R10). These are real data, used for the no-harm check only. The H1/H2 slots are empty ("not screened"). Knowledge type follows master plan §2.2 (K = knowledge, S = statistical, R = rule).

| Row | `dataset_id` | Type | n | p | t17 seed SD | Knowledge type |
|---|---|---|---|---|---|---|
| A13-001 | `balance-scale` | C | 625 | 4 | 0.000 | R+K |
| A13-002 | `junglechess` | C | 44,819 | 6 | 0.070 | R+K |
| A13-003 | `vehicle` | C | 846 | 18 | 0.029 | S |
| A13-004 | `heart` | C | 918 | 11 | 0.019 | K |
| A13-005 | `credit-g` | C | 1,000 | 20 | 0.0055 | K+S |
| A13-006 | `myocardial` | C | 686 | 91 | 0.0076 | K |
| A13-007 | `housing` | R | 20,640 | 9 | 0.021 | K |
| A13-008 | `insurance` | R | 1,338 | 6 | 0.063 | K |
| A13-009 | `crab` | R | 3,893 | 8 | 0.024 | K |
| A13-010 | `wine` | R | 6,497 | 12 | 0.0010 | K+S |

**Metrics** (01 §7 R13 names):
- **Cost:**
  - `tok_compl_total` and `tok_compl_<role>`, including reasoning and re-rolls, with search charged to the caller;
  - `tok_search_<role>`;
  - `tok_prompt_total`;
  - `llm_calls_total`;
  - `wall_s` and `wall_s_<role>`;
  - `serper_calls`;
  - `energy_j` (apportioned by token share, so it is not independent of tokens);
  - `flop_proxy` = 2 · 5.1B · tokens.
- **Score:**
  - `heldout_score` is the outer-fold score at the inner-CV-selected iteration, averaged over 5 folds: balanced accuracy (C) or nRMSE (R).
  - Δ is oriented so that **positive means Off is better**: Δ = Off − control for balanced accuracy; Δ = control − Off for nRMSE.
  - Δ_std (`delta_std`) = Δ / σ_d, where σ_d (`sigma_d`) is the dataset's Full-v2 run-to-run SD from the G2 memo (01 §4.7), floored at 0.1 × the fold SD of `base_no_fe`.
- **Secondary:** `success`, `collapse`, `cap_hit`; the funnel `fun_*`; `raw_coverage` (as a mediator); `err_per_accepted`; `fallback_share`.

**Units.** u = one real-data O120 run (10 iterations × 5 folds). Throughput is 2 u/h × 0.7 = 1.4 u/h on ≤ 4 H100.

## 3. Hypotheses and interpretation (all are hypotheses)

The labels in the last column are the pre-registered verdicts of §6.

| Outcome | Meaning | Paper wording / action |
|---|---|---|
| **Hoped** | Cost verdict *clearly cheaper* (§6 K2): the 95% CI of the paired `tok_compl_total` ratio lies below 1 and the point saving is ≥ 5% (planning guess 10–30%, with no prior). No-harm verdict *no harm beyond M*. | *"Unplugging the knowledge channel from all three roles cut completion tokens by X% (95% CI a–b; 10 paired datasets) and wall clock by Y%; held-out score stayed within margin M (1 seed)."* If the no-harm verdict is *inconclusive*, the score clause becomes *"its effect on real-data score is inconclusive at one seed"*. |
| **Null** | Cost verdict *small cost centre* (K3): the point saving is < 5%. | *"Search is a small cost centre: Z% of Full's completion tokens."* Z is read free from A05's caller-attributed logs (A13-T16). |
| **Inconclusive** | Cost verdict *inconclusive* (K4): point saving ≥ 5% but the CI includes 1. This is a pre-registered outcome. | *"The point saving was X%, but the CI (a–b) includes no saving at one seed."* |
| **Opposite (a)** | No-harm verdict *harm* (pooled), or the `junglechess` parts of the RV2 trigger fire (§6). | At one seed this is not evidence. Check the RV2 trigger (§6). A seed-4 replication stays descriptive and outside Holm. |
| **Opposite (b)** | Cost verdict *more expensive* (K1): the agents add turns to make up for the missing tool. A plausible route is Extractor code errors, since E loses its pandas-documentation lookup. | Report it as a harness finding: *"removing the tool did not save cost."* Quantify it with the compensation term (§4) and `err_per_accepted`. |
| **Opposite (c)** | Off scores better: the lower bound of pooled Δ_std (§4) is > 0. | Descriptive only, at one seed: *"removing search did not lower, and may have raised, real-data score."* Never claim search is harmful. |

Never claim "public recipes are already in parametric memory". No real-data Parametric arm exists in T1-minimum.

## 4. Design

**Control.** A05 `full` on Core-10 at fold seed 42, using both replicates: `replicate` 0 (`llm_seed` 42) and `replicate` 1 (A/A, `llm_seed` 1042). **The control value per dataset is the mean of the two runs.** A05 runs on `agent2027-v2.0`. Comparing it with a later image is valid only because every v2.N keeps Full's rendered prompts byte-identical to `prompt-baseline-v2.0` (E5 diff, checked at G0b-1) and the session passes the A06 sentinel.

**Arm `search_off`.**
- `env_overrides = {"ROGUE_ONE_KNOWLEDGE_BACKEND": "off"}`. Every other flag comes from `scripts/agent2027/v2_defaults.env`.
- **Fixed settings** (01 §4.3):
  - 10 iterations × 5 folds (`ROGUE_ONE_NUM_ITERATIONS=10`, `ROGUE_ONE_K_FOLDS=5`);
  - `ROGUE_ONE_SELECTION_SIGNAL=inner_cv`, `ROGUE_ONE_ENABLE_ROLLBACK=False`;
  - temperature 0.7 (`LLM_TEMPERATURE`, `*_AGENT_TEMPERATURE`);
  - `ROGUE_ONE_EXTRACTOR_ROUNDS` and `ROGUE_ONE_FEATURE_POOL_MODE` as frozen at G1;
  - `ROGUE_ONE_LLM_TRACE=1`, `ROGUE_ONE_PROMPT_DUMP=1`;
  - the 6 h cap (`RUN_CAP_HOURS=6`, `cap_hours=6`);
  - `PARALLEL=6`, `CPUS=12`;
  - the same O120 topology as A05 (D5).
- `ROGUE_ONE_SEARCH_CACHE` keeps its v2 value (`shared`); with no search calls it has no effect.

**Off edit list** (pre-registered; E9e implements it; the `web` branch stays byte-identical). Line numbers are at `07cbf93`; E9e finds each edit by its quoted string.

| # | File:line | Edit under `off` | Verified by |
|---|---|---|---|
| 1 | `sci/agent.py:464` | drop `search_tool` from `tools=[…]` | trace `tools` field; unit test |
| 2 | `sci/agent.py:414` | replace ", the focus history, and compare with existing information from the web to determine" with ", and the focus history to determine". **Not in 01 E9e's list** (found during verification); confirm in A13-T1. | prompt dump |
| 3 | `sci/agent.py:431` | replace "(search, lookup, quick dataframe checks). Always perform at least one web search." with "(lookup, quick dataframe checks)." | prompt dump |
| 4 | `sci/agent.py:438` | delete the line "- Use search_tool actively when uncertain." | prompt dump |
| 5 | `ext/agent.py:921`, `:939` | drop `search_in_literature_tool` from the prompt's "Available tools" list (`:921`) and from `tools=[…]` (`:939`) | `:921` prompt dump; `:939` trace `tools` field |
| 6 | `tst/agent.py:1492` | drop `search_in_literature_tool` from the Stage A `tools=[…]` | trace `tools` field; unit test |

The E5 prompt dump holds only the rendered system prompt and the first user message of each agent call, so tool-list edits (1, 5 at `:939`, 6) are checked in the trace's `tools` field and by unit tests, not by `prompt_diff.py`.

Nothing else changes. The `WebSearchAgent` objects are still constructed, which is harmless: `WebSearchAgent.__init__` only stores the config (`kn/web_search.py:284-290`), and the Serper wrapper is created only inside a call (`:72`).

**Run matrix**

| Group | Rows | Arm | Datasets | `fold_seed` | `llm_seed` | `replicate` | Runs | u | W&B run name |
|---|---|---|---|---|---|---|---|---|---|
| New | A13-001…010 | `search_off` | Core-10 | 42 | 42 | 0 | **10** | 9–11 | `A13__search_off__<dataset_id>__f42__r0` |
| Control (reused) | A05 rows | `full` (A05) | Core-10 | 42 | 42 / 1042 | 0 / 1 | 0 new (20 reused) | 0 | `A05__full__<dataset_id>__f42__r{0,1}` |
| Sentinel | A06 row | `sentinel` (A06) | task chosen at G2 | per A06 | 5000+k | k | 0 new (A06 pays) | 0 | A06 naming |
| Smoke (never analysed) | A13-901 | `search_off` | `heart` | 42 | 42 | 0 | 1 fold | 0.14–0.2 (reserve) | `A13__search_off__heart__f42__r0`, tag `smoke` |
| RV2 (conditional) | A13-011…020 | `search_off` | Core-10 | 4 | 4 | 0 | 10 | 10 (reserve) | `A13__search_off__<dataset_id>__f4__r0`, against `A05__full__<dataset_id>__f4__r0` |

The smoke row shares its run name with A13-004. Analysis therefore always selects runs by the config key `manifest_row` (A13-001…010, plus A13-011…020 under RV2), never by name. For a row retried after a crash, the analysed run is the retry.

**Budget arithmetic.**
- 10 runs × 0.9–1.1u gives 9–11u. The low end assumes a modest saving; the high end allows for compensation (opposite b). This equals the T1-minimum breakdown ("10 runs × 1u = 9–11u") and 01 §7 R1.
- Box time is 9–11u ÷ 1.4u/h = 6.4–7.9 h; with the sentinel (≈ 1u) about 7–9 h. At TP4 that is 26–31 H100-hours (× 4 GPUs); at TP2, half the GPUs for the same session share.
- Smoke A13-901: 1 of 5 folds ≈ 0.2u at most, from the rerun reserve (not in the 9–11u).
- RV2: 10 runs = 10u nominal (01 §4.9; 9–11u at the same per-run range), from the reserve.
- Serper calls: 0. A Full run makes an estimated 500–1,000 before cache hits (01 E4).

**Session plan** (default matches A12-T9 and A06's session table):
- **Smoke:** A13-901 runs in the G0b-1 smoke session `S1021b` (Wed 21 afternoon) together with A12-057…064 and A14-901…904.
- **Main:** A13-001…010 run in `S1021a` (Wed 21 evening) together with A12 block 1 (A12-001…032), one A06 sentinel row and, if A14 joins as its §4 prefers, A14-001…018. `S1021a` totals 32.4–44u (≈ 23–31 h) without A14, or ≈ 45–62u (≈ 32–44 h) with it, so it ends between Thu 22 evening and Fri 23 afternoon.
- **Fallbacks:** if E9d (A12) misses G0b-1 but E9e passes, A13 runs alone: smoke in `S1021c` (A13-901 only), then `S1021d` (A13-001…010 + one A06 sentinel row, ≈ 7–9 h). If E9e misses G0b-1, A13 waits; its rows go into the next v2.N O120 session after E9e passes, with latest start Thu 29 Oct.

**Pairing and randomisation.**
- Off is paired with the control by dataset and outer partition (fold seed 42).
- The `llm_seed` matches A05 r0, but sampling is not byte-reproducible under load (01 §4.2), so the pairing is by partition only.
- Rows run in random `order_key` order (RNG seeded by the session ID, `make_manifest.py --plan-session`), interleaved with the session's other rows and its A06 sentinel.

**Cost analysis** (`analysis/glmm_holm.py`, paired log-ratio routine; outside Holm; raw p values, unadjusted):
- For each dataset d, r_d = ln(Off ÷ mean of the two Full runs), on `tok_compl_total`.
- Report the geometric-mean ratio exp(mean r_d), with a t-based 95% CI over the 10 values (t₉) and the exact sign-flip permutation p (2¹⁰ flips). Saving = 1 − ratio.
- Do the same for `wall_s`, `llm_calls_total`, `tok_prompt_total`, each `tok_compl_<role>`, `energy_j` and `flop_proxy`.
- **Calibration:** report the A/A log-ratio ln(r1/r0) per dataset alongside.
- **Compensation term** (per dataset) = Off `tok_compl_total` − (Full `tok_compl_total` − Σ_role `tok_search_<role>`), with Full = mean of r0 and r1. Positive = the agents spent extra non-search tokens without search.
- **Populations:** primary = the pairs where all three runs completed; sensitivity = all pairs at the tokens actually consumed.

**No-harm analysis** (01 §4.7 conventions; scores sealed until the G2 memo exists, A13-T4).
- Δ_std(d) is computed per dataset against the mean of the two Full runs.
- With one Off run per dataset, the dataset intercept and residual of the shared no-harm model (01 §4.7) are confounded, so the model reduces to the unweighted mean of the 10 Δ_std values with fixed variance Var(Δ_std) ≈ 1 + ½ = 1.5 (Off minus a mean of two runs, in σ_d units). The task-type term is reported as the C-minus-R difference of means, descriptive only. Confirm this reduction in X10 (A13-T1).
- **Bounds:** lower bound = mean − 1.645 · √(1.5/10) = mean − 0.637; upper bound = mean + 0.637 (one-sided 95% each). Sensitivity: the same with the empirical SD of the 10 values and t₉ (1.833).
- Test against −M. The margin M is in Δ_std units; its formula is fixed in X10 (Fri 16) and its value in the G2 memo (Mon 19).
- *Planning arithmetic, not a result:* if the true Δ is 0, the chance of a "no harm beyond M" verdict is about 50% at M = 0.64 and about 83% at M = 1.0. "Inconclusive" is therefore likely unless M ≳ 0.65.
- **Descriptive extras:**
  - per-task-type exact Wilcoxon signed-rank (C n = 6, smallest two-sided p 0.031; R n = 4, smallest p 0.125);
  - `junglechess` rule discovery (held-out ≥ 0.95) in the 5 Off folds against the mean count of the two Full runs;
  - knowledge type K-only {`heart`, `myocardial`, `housing`, `insurance`, `crab`} against mixed/non-K {`balance-scale`, `junglechess`, `vehicle`, `credit-g`, `wine`}.

**Failures (ITT; 01 §4.5).**
- A crash (non-zero rc, or no `fold_aggregation_summary.csv`) is retried **once, alone**, at the end of the session.
- A persistent failure or a 6 h cap hit is an ITT failure. It is never dropped, and a cap hit is never retried.
- Its held-out score is imputed at `base_no_fe` if augment is canonical, otherwise at the majority-class (C) or mean (R) predictor.
- Failure and cap-hit rates go next to every number.
- If an A05 control run failed, the cost control uses the surviving run (flagged), and the no-harm control uses its ITT value.

## 5. Tasks

**A13 analysis notes** = `docs/agent2027/analysis_notes/A13.md` on branch `agent2027/v2`, committed after each analysis task. **Decision log** = 01 §6.4. Total effort ≈ 2.2 developer-days, of which 0.15 d (A13-T6) sits inside E9e's 0.25 d.

### 5.1 Prerequisites (verify before launch)
- [ ] **A13-T1** [type: decision] (0.1 d, due Fri 16 Oct) — Write the A13 section of `docs/agent2027/preregistration.md` (X10). It must contain:
  - the Off edit list, including rows 2–3 and their exact replacement texts;
  - the cost verdicts K1–K4 (§6);
  - the no-harm rule, including the reduction to a mean with Var(Δ_std) = 1.5 and the bound formula;
  - the RV2 trigger (i)–(iii);
  - the K-only/mixed split;
  - the label "cost outcome, outside Holm".

  Ask for 01 E9e to be updated with edit 2. *Done when:* the section is in the commit tagged `agent2027-prereg`.
- [ ] **A13-T2** [type: data] (0.05 d, due Mon 19 Oct) — Check E3/E4 on the 20 A05 Core-10 seed-42 runs:
  - all three `tok_compl_<role>` > 0;
  - `tok_search_<role>` > 0 for at least one role, with Extractor searches landing in `tok_search_extractor`;
  - `serper_calls` > 0;
  - on one run, the trace record count = `llm_calls_total`.

  *Done when:* a W&B query (project `PlanBench/RogueOne-AGENT2027`, group `A05`) shows all of this, and the result is in the A13 analysis notes. If token logging fails validation, the primary cost outcome falls back to `llm_calls_total` + `wall_s` (01 §4.8 rule 1), and that is written in the notes.
- [ ] **A13-T3** [type: data] (0.05 d, due Mon 19 Oct) — Confirm the control set: 20 runs `A05__full__<dataset_id>__f42__r{0,1}`, each `finished` or recorded as ITT. *Done when:* the list of 20 run IDs with their states is in the A13 analysis notes.
- [ ] **A13-T4** [type: data] (0.02 d, due Mon 19 Oct) — Confirm the G2 memo. *Done when:* W&B run `ANALYSIS__g2_stats__20261019` (group `ANALYSIS`, E11a) exists and holds `sigma_d` for all 10 Core-10 datasets, `sigma_AA` and `ni_margin` (M). No A13 `heldout_score` may be read before this holds.
- [ ] **A13-T5** [type: config] (0.02 d, due Fri 16 Oct) — Record the D5 topology used by A05 (TP4 or TP2). *Done when:* the topology is in the A13 analysis notes and A13-T12 uses the same `GPUS` count.

### 5.2 Implementation (E9e owns the code; A13 verifies it)
- [ ] **A13-T6** [type: code] (0.15 d, inside E9e, due Wed 21 Oct, before the v2.3 build) — In E9e:
  - read `ROGUE_ONE_KNOWLEDGE_BACKEND` once (default `web`; values `web|off|parametric|local`) and register it in the E1 flag registry;
  - apply edits 1–6 behind `== "off"`, locating each by its quoted string, and keep the `web` strings byte-identical.

  *Done when:* for the `off` branch, the E9e diff touches only edits 1–6 plus the flag plumbing (the `parametric`/`local` wiring is checked by A14-T8).
- [ ] **A13-T7** [type: code] (0.1 d, due Wed 21 Oct) — Unit tests in `tests/unit/agents/test_knowledge_backend.py`:
  - under `off`, the S tool list lacks `search_tool`, and the E list and T Stage A list lack `search_in_literature_tool`;
  - the rendered S/E system prompts contain none of "web search", "search_tool", "search_in_literature_tool" or "information from the web";
  - under `web`, the lists and prompts equal those at `07cbf93` plus the E5 neutralisation.

  The repo has no CI, so run the tests in the image (CPU only, no GPU pre-flight needed):
  `docker run --rm --cpus 4 --entrypoint bash rogue_one:agent2027-v2.3 -c "python3 -m pip install -q pytest pytest-timeout && python3 -m pytest tests/unit -q -m 'not requires_llm'"`
  (on CAIR, wrap it in `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "…"`). *Done when:* the new tests and the existing `tests/unit` suite pass in that container, and the output is pasted into the A13 analysis notes.
- [ ] **A13-T8** [type: run] (0.1 d, due Wed 21 Oct; part of the G0b-1 smoke in `S1021b`) — Run smoke row A13-901 (`search_off`, `heart`, fold seed 42, `env_overrides` adds `"ROGUE_ONE_SMOKE_FOLDS": "1"`, `notes=smoke`). Reference dump = the `prompts/` folder in the `run-output` artifact of `A05__full__heart__f42__r0` (Full-v2 on v2.0; G0b-1 guarantees v2.3 renders Full identically). Check:
  - `scripts/agent2027/prompt_diff.py <A05 heart prompts> <A13-901 prompts>`, restricted to the system prompts of fold 0, iteration 0, shows exactly edits 2, 3, 4 and the `:921` line of edit 5, and nothing else (user messages and later iterations differ by design and are ignored);
  - in `llm_traces.jsonl.gz`, no request's `tools` field contains `search_tool` or `search_in_literature_tool` (edits 1, 5, 6), and there are 0 tool calls with those names;
  - `serper_calls` = 0 and every `tok_search_<role>` = 0;
  - `output/retrieval_transcript.jsonl` has no `channel="web_search"` rows;
  - there are no runner errors naming either tool.

  *Done when:* all five hold and are recorded in the A13 analysis notes. A call to a missing tool would confound A07, so any such call blocks the launch.
- [ ] **A13-T9** [type: config] (0.1 d, due Tue 20 Oct) — Run `scripts/agent2027/make_manifest.py --ablation A13` to write `manifests/agent2027/A13_KT1-Off.csv`:
  - rows A13-001…010 (§2 order, fold seed 42);
  - rows A13-011…020 (fold seed 4, `llm_seed` 4, `notes=RV2-conditional`; never planned into a session unless RV2 fires);
  - row A13-901 (smoke, as in A13-T8).

  Header and first row (01 §7 R7):
  ```
  row_id,ablation_id,arm_id,dataset_id,task_type,fold_seed,llm_seed,replicate,is_sentinel,session_id,order_key,image_tag,rung_S,rung_E,rung_T,port_S,port_E,port_T,env_overrides,cap_hours,expected_u,notes
  A13-001,A13,search_off,balance-scale,classification,42,42,0,0,,,agent2027-v2.3,O120,O120,O120,8010,8010,8010,"{""ROGUE_ONE_KNOWLEDGE_BACKEND"": ""off""}",6,1.0,pair A05 full f42 r0+r1
  ```
  *Done when:* the file is committed and pushed, and a dry run gives unique container names `ro27-<session_id>-a13-0nn` / `ro27-<session_id>-a13-901`.

### 5.3 Execution on CAIR

**Rules for every step (01 §5.1):**
- **Containers only.** Every execution runs inside Docker. The host only issues `docker` / `docker compose` commands (any `docker compose run` over SSH needs `-T`) and runs the launch scripts in `~`, plus `nvidia-smi`, `docker ps` and `tail` for checks. Never run `python …` on the host.
- **GPU pre-flight before any GPU container** (A13-T11). If there are not enough free GPUs, **stop and report** per-GPU memory and utilisation, compute apps and running containers to Henrik.
- **Never more than 4 of the 8 H100s** without Henrik's explicit approval in chat. One Rogue One job at a time. Launch with explicit `GPUS=<ids>` from the pre-flight and `WAIT_MAX_MIN=0`, so a session never queues silently.
- **W&B is the system of record.** Runs are stateless and upload `output/` as artifact `run-output`. Project `PlanBench/RogueOne-AGENT2027`; `WANDB_API_KEY` from `<CAIR_HOME>/code/PlanBench/.env`. `SERPER_API_KEY` and LLM settings come from `.env` / `.secret.env` baked into the image; never push these images.
- **SSH:** one-shot, non-interactive commands only: `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "…"`.
- **Code to the server:** commit and push locally, then `git fetch` over SSH on the server (A13-T10 step 3).
- **Long jobs:** detached `nohup` launcher, log in `~` (never in the root-owned run folders).
- `<ROGUEONE_CAIR_PATH>` = `<CAIR_HOME>/code/RougeOne` is **INFERRED** (from `paper/scripts/clients_t16_rerun.sh:135`, `REPO_TASKS="${REPO_TASKS:-$HOME/code/RougeOne/tasks}"`, and the home `<CAIR_HOME>` at `:59`). Verify it once: `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "ls -d ~/code/RougeOne"`.

- [ ] **A13-T10** [type: run] (0.1 d, due Wed 21 Oct, after G0b-1) — Plan and stage the session.
  1. Plan (default, §4): add A13-901 to the `S1021b` smoke call and A13-001…010 to the `S1021a` call that A12-T9 makes:
     `make_manifest.py --plan-session S1021a --rows A12-001…A12-032 A13-001…A13-010 <A06 sentinel row> [A14-001…A14-018]`.
     Fallback (§4): `--plan-session S1021c --rows A13-901`, then `--plan-session S1021d --rows A13-001…A13-010 <A06 sentinel row>`.
  2. Commit and push branch `agent2027/v2` (tag `agent2027-v2.3`).
  3. Sync the server:
     `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15' git fetch git@github.com:HenrikNorkart/RougeOne.git agent2027/v2 && git checkout -B agent2027/v2 FETCH_HEAD && git log -1 --format='HEAD: %h %s'"`
     (the remote URL is inferred; verify that the server key has access).
  4. If `rogue_one:agent2027-v2.3` is absent, build it:
     `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && test ! -e .dev.env && docker build -t rogue_one:agent2027-v2.3 . && docker image inspect rogue_one:agent2027-v2.3 --format '{{.Id}}'"`
  5. Install launchers and manifests in `~`:
     `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && cp scripts/agent2027/ro27_session_launch.sh scripts/agent2027/ro27_clients_manifest.sh scripts/agent2027/ladder_session.sh ~/ && cp paper/scripts/vllm_session.sh ~/ro27_vllm_session.sh && mkdir -p ~/ro27_manifests && cp -r manifests/agent2027/. ~/ro27_manifests/"`

  *Done when:* `~/ro27_manifests/sessions/<SESSION_ID>.csv` exists and lists every A13 row planned for it, the server HEAD equals the pushed `agent2027-v2.3` tag, and the image ID is in the A13 analysis notes.
- [ ] **A13-T11** [type: run] (0.01 d / 5 min, Wed 21 Oct, immediately before each launch) — GPU pre-flight:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader && echo '--- compute apps ---' && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader && echo '--- containers ---' && docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'"
  ```
  *Done when:* 4 GPUs (2 if TP2) show < 2,000 MiB used, the total requested is ≤ 4, and no `ro27-*`, `rogueone-vllm-*`, `llmfe-*` or `caafe-*` container is running (A04 session C and the A08 GPU part run the same days). **Otherwise STOP** and report memory, utilisation, apps and containers to Henrik. If `S1021a` was already launched by A12-T12 after its own pre-flight, this task and A13-T12 are not repeated; record the session ID instead.
- [ ] **A13-T12** [type: run] (0.01 d / 5 min, Wed 21 Oct, evening) — Launch detached, with the log in `~` (`<SESSION_ID>` = `S1021a` by default; `S1021c`/`S1021d` in the fallback):
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "SESSION_ID=<SESSION_ID> SESSION_FILE=~/ro27_manifests/sessions/<SESSION_ID>.csv GPUS=<ids> MODEL=openai/gpt-oss-120b PORT=8010 IMAGE_RO=rogue_one:agent2027-v2.3 PARALLEL=6 CPUS=12 RUN_CAP_HOURS=6 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_<SESSION_ID>.job.log 2>&1 & echo launched; sleep 5; tail -5 ~/ro27_<SESSION_ID>.job.log"
  ```
  Names (lower-case): server `rogueone-vllm-o120-<session_id>`, clients `ro27-<session_id>-<row_id>`, energy sidecar `ro27-<session_id>-energy`. *Done when:* the job log shows the vLLM version assert, both image IDs and the first rows started.
- [ ] **A13-T13** [type: run] (0.1 d spread over Wed 21 – Fri 23 Oct) — Monitor in the background:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 CAIR-gpu14 "while ! grep -q 'RO27_SESSION_END' ~/ro27_<SESSION_ID>.job.log 2>/dev/null; do sleep 120; done; tail -30 ~/ro27_<SESSION_ID>.job.log"
  ```
  Exit code 255 means the network dropped, not the job, so reconnect. In parallel, query the A13 run states inside a container:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker run --rm --env-file <CAIR_HOME>/code/PlanBench/.env --entrypoint python3 rogue_one:agent2027-v2.3 -c \"import wandb; api=wandb.Api(); [print(r.name, r.config.get('manifest_row'), r.state) for r in api.runs('PlanBench/RogueOne-AGENT2027', filters={'group': 'A13'})]\""
  ```
  A `running` run with no live container is a zombie. *Done when:* the job log shows `=== RO27_SESSION_END <SESSION_ID> rc=0 ===`.
- [ ] **A13-T14** [type: run] (0.05 d, due Fri 23 Oct) — Close out:
  - each crashed A13 row got exactly one solo retry;
  - `CAP_HIT` rows were not retried;
  - the tear-down check shows no containers left:
    `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker ps --format '{{.Names}}' | grep -E '^(ro27-|rogueone-vllm-)' || echo none; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"`

  *Done when:* 10 A13 runs (A13-001…010) are in W&B, each `finished` or ITT, each with every R12 config key, `flags/ROGUE_ONE_KNOWLEDGE_BACKEND=off` and a `run-output` artifact.

### 5.4 Analysis
All analysis runs inside a `rogue_one:agent2027-v2.N` container (on CAIR with `--cpus` ≤ 16), never on the CAIR host.
- [ ] **A13-T15** [type: analysis] (0.1 d, due Fri 23 Oct) — Process check:
  - `serper_calls` = 0 on all 10 runs, and the trace search-tool count = 0;
  - run `analysis/sentinel_drift.py` on the session's A06 sentinel.

  *Done when:* both results are in the A13 analysis notes. If the sentinel drifts, A13 is flagged and Henrik decides whether to re-queue the 10 rows (10u from the reserve; must start by Thu 29 Oct).
- [ ] **A13-T16** [type: analysis] (0.25 d, preliminary Sat 24 Oct, due Tue 27 Oct with E11b) — Cost read (token outcomes are not sealed):
  - `analysis/fetch_wandb.py`, then the §4 cost analysis and the cost verdict K1–K4 (§6);
  - the compensation term;
  - Full's search share Σ`tok_search_<role>` ÷ `tok_compl_total`, per role, from the 20 A05 Core-10 seed-42 runs.

  *Done when:* W&B run `ANALYSIS__glmm_holm__<YYYYMMDD>` holds the per-dataset table and the verdict, and the verdict is in the A13 analysis notes.
- [ ] **A13-T17** [type: analysis] (0.25 d, Sat 24 – Mon 26 Oct; due Mon 26 Oct) — No-harm read (only after A13-T4):
  - pooled Δ_std and its bounds against −M, by the closed-form rule of §4 (no E11b dependency, so it is ready for G3);
  - the descriptive extras;
  - `analysis/funnel.py` and `analysis/process_metrics.py` (`fun_*`, `raw_coverage`, `err_per_accepted`) once E11b is done (Tue 27).

  *Done when:* the verdict (no harm beyond M / harm / inconclusive) and the three RV2 inputs (§6) are in the A13 analysis notes.
- [ ] **A13-T18** [type: decision] (0.05 d, due **Mon 26 Oct**, G3) — Evaluate the RV2 trigger (§6). If it fires and ≥ 10u of unspent reserve remains (01 §4.9), plan rows A13-011…020 into the next O120 v2.N session that starts by Thu 29 Oct (for example with A16's Core-10 rows), set their `image_tag` to that session's image, and run A13-T11–T14 for it. *Done when:* the decision (fire / not fire, with the three inputs) is in the decision log (01 §6.4) and the A13 analysis notes.
- [ ] **A13-T19** [type: analysis] (0.25 d, due Tue 3 Nov) — Final scripted rerun of T16 and T17 (plus RV2 descriptively, if run). Hand Off's tokens and score to A21 (Full truncated at whole iterations to Off's tokens, via the extended `paper/scripts/diag_selection_policy.py`; "no iso-cost point" if Off spends less than Full's first iteration) and to A23 (`analysis/pareto.py`). *Done when:* the Table 2 numbers regenerate from W&B artifacts by script.

### 5.5 Paper outputs
- [ ] **A13-T20** [type: writing] (0.1 d, due Fri 6 Nov) — **Table 2, KT1-Off row:**
  - Δ `tok_compl_total` (%) with its CI;
  - Δ `wall_s` (%);
  - `serper_calls` (Full vs 0);
  - the funnel stage changed;
  - the no-harm Δ_std CI and verdict;
  - the tag "cost outcome, outside Holm; 1 seed".

  *Done when:* the row is in the AGENT draft.
- [ ] **A13-T21** [type: writing] (0.1 d, due Fri 6 Nov) — Write the C4 sentence from §3 that matches the verdicts, and annotate Fig. 1's W knob as "web | off (A13) | parametric, local (A14)". *Done when:* both are in the draft, with no parametric-memory claim.
- [ ] **A13-T22** [type: writing] (0.15 d, due Thu 12 Nov) — Appendix and replication package:
  - the edit list and prompt diff;
  - the per-dataset cost and score table;
  - Full's per-role search share;
  - **disclosures:** the Serper cache lowers Full's search variance and wall clock; the control ran in other sessions (Fri 16 – Mon 19 Oct) on image v2.0, so `wall_s` and `energy_j` carry session effects; the old No-Search arm was Scientist-only.

  *Done when:* all of this is in the anonymised replication package.

## 6. Decision rules and gates (pre-registered in X10)
- **Cost verdict** on `tok_compl_total` (outside Holm; raw p values, unadjusted). Evaluate in this order; exactly one applies:
  - **K1 more expensive** (opposite b): the 95% CI of the ratio lies wholly above 1;
  - **K2 clearly cheaper:** the CI lies wholly below 1 **and** the point ratio ≤ 0.95 (saving ≥ 5%);
  - **K3 small cost centre:** the point ratio > 0.95 (saving < 5%), and K1 does not hold;
  - **K4 inconclusive:** the point ratio ≤ 0.95 but the CI includes 1.
- **No-harm verdict** (one-sided, α = 0.05; bounds as defined in §4):
  - *no harm beyond M* if the lower bound of pooled Δ_std > −M;
  - *harm* if the upper bound < −M;
  - *inconclusive* otherwise.

  01 §4.7 has only the first and last; *harm* is A13-specific and feeds RV2.
- **RV2 trigger** (checked Mon 26 Oct). It fires if any of the following holds:
  - (i) the pooled no-harm verdict is *harm*;
  - (ii) `junglechess` Δ_std < −M (point estimate);
  - (iii) Off's `junglechess` rule-discovery count (folds with held-out ≥ 0.95, out of 5) ≤ the mean count of the two Full runs − 2.

  01 §4.9 names only an adverse `junglechess` result; (i) widens it and must be confirmed in X10. If it fires: 10 runs at fold seed 4 against A05 `full` f4 r0 (one control run, so Var(Δ_std) ≈ 2), paid from the reserve. It is a replication, descriptive, outside Holm.
- **Gates.**
  - Launch only after G0b-1 (Wed 21) passes with Full's prompts byte-identical to `prompt-baseline-v2.0` and A13-T7/T8 passed. If E9e misses it, A13 waits.
  - If A13 has not started by **Thu 29 Oct** (L-LAST), drop it and report only Full's search share from A05.
  - Fri 30 – Sat 31 are for reruns only; data freeze is Sat 31 Oct, 24:00.

## 7. Risks and cut levers
- **One seed.** `junglechess` (SD 0.070) and `insurance` (SD 0.063) can swing Δ, hence the cost-first framing. The only add-back is RV2.
- **A leftover search instruction** would cause calls to a missing tool, confounded with A07. Guarded by A13-T7, T8 and T15.
- **Token logging invalid.** The token outcome is void; fall back to calls and wall clock, and say so.
- **Session mismatch with the control.** Tokens are robust to it; wall clock and energy are not. The A06 sentinel and constant `PARALLEL=6` mitigate it.
- **Schedule.** A13 is not among cut levers CL1–CL7. Under pressure, the L-LAST drop rule (§6) applies.

## 8. Definition of done
- [ ] Pre-registration section committed (T1); edit 2 confirmed.
- [ ] Switch verified: unit tests, prompt diff, trace `tools` check, smoke with 0 Serper calls and 0 missing-tool calls (T6–T8).
- [ ] 10 `A13__search_off__*__f42__r0` runs (A13-001…010) in W&B, finished or ITT, each with `run-output`.
- [ ] Cost verdict, no-harm verdict, sentinel check and RV2 decision recorded (A13 analysis notes, `ANALYSIS` run, decision log).
- [ ] Table 2 row, C4 sentence, Fig. 1 knob and appendix done; A21/A23 points delivered.

## 9. Open issues (not resolvable inside this document)
- **Session ID collision (resolved):** A04 session C is now `S1021m`; `S1021a` stays the shared v2.3 O120 session.
- **GPU overlap on Wed 21 – Thu 22:** A04 session C (O120 TP2 + O20 + Q0.8 = 4 GPUs) cannot run at the same time as `S1021a` (TP4 or TP2) under the ≤ 4-GPU, one-job rule. The 01 §6.2 calendar lists both on Wed 21; they must be sequenced.
- **A12-T9 adds A13-901 (and A14-901…904) to the `S1021b` smoke call** (A12 §4 lists A12-057…065, A13-901 and A14-901…904).
- **01 E9e** lacks edit 2 (`sci/agent.py:414`), and **01 §4.9 RV2** names only `junglechess`; both are raised in A13-T1.

---
**New names introduced (confirm in X10):**
- cost verdicts K1–K4 and the 5% "clearly cheaper" threshold;
- the A13-specific no-harm verdict *harm* and the closed-form bound (Var(Δ_std) = 1.5);
- the RV2 trigger (i)–(iii);
- the K-only vs mixed/non-K split of Core-10;
- the exact replacement texts for edits 2–3;
- smoke row A13-901; fallback session IDs `S1021c`, `S1021d`;
- `tests/unit/agents/test_knowledge_backend.py`; `docs/agent2027/analysis_notes/A13.md`.
