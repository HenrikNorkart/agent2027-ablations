# 01 · Shared infrastructure for the AGENT 2027 ablations

*Rogue One v2 design study · ICSE AGENT 2027 workshop (8 pp IEEE) · submission **Fri 13 Nov 2026** · scope **T1-minimum** · one developer (owner of every task: **Henrik**) · written Tue 6 Oct 2026.*

*Sources: `docs/AGENT2027_T1min_breakdown.md` (budgets, cards, calendar, gates, decisions), `docs/AGENT2027_ablation_directions.md` (designs, protocol §3, metrics §3.4, instruments §2, engineering §6.2), and the audited per-group breakdown (workflow `wf_d0e803a6-c9f`). Every file:line reference below was re-checked against the RougeOne repo, branch `nested-cv-rebuttal` at commit `07cbf93` (`C:/code/PHD/RougeOne`). The code copy inside this worktree belongs to a different, anonymised lineage and has different line numbers. Do not use it.*

*Every "hoped" result in any ablation document is a hypothesis, not a finding.*

---

## Contents

1. Purpose and how to use the ablation documents
2. The frozen "Rogue One v2" and the freeze procedure
3. Engineering tasks (checklist)
4. Common protocol
5. CAIR run procedure, launcher and W&B naming
6. Gate calendar, developer calendar and writing calendar
7. Canonical registry (names every document must use verbatim)
- Appendix A. Code-verification notes and discrepancies found

---

## 1. Purpose and how to use the ablation documents

**What this document is.** It is the one place that defines what every ablation shares:
- the frozen reference system (Full-v2);
- the engineering tasks that must land before any arm runs;
- the instruments, seeds, fixed settings, metrics and statistics;
- the procedure for running anything on CAIR;
- the gate calendar;
- the canonical names (§7).

**What the ablation documents are.** There are 23 of them (`A01_…` to `A23_…`; full list in §7 R1). Each one is **self-contained**: a reader can run and analyse that ablation from its own document. Each ablation document:
- restates its question, arms, instruments, run list, budget, decision rule and analysis;
- lists its **prerequisites as task IDs from §3 of this document** (for example "E2, E9a, X10"). It does not re-specify those tasks;
- uses the names in §7 exactly: flags, arm IDs, dataset IDs, script paths, W&B names, metric names and contrast IDs.

**How to use them.**
1. Before launching an ablation, open its document and tick its prerequisite tasks against §3. If one is not "done when"-verified, do not launch.
2. Check the gate it depends on (§6).
3. Generate its manifest rows (§5.3), plan the session, run the CAIR procedure (§5.1–5.2), then analyse with the scripts named in §7 R6.

**Rule of precedence.** If an ablation document and this document disagree, **this document wins** for shared items: flags, fixed settings, metrics, statistics conventions and CAIR rules. The ablation document wins for its own arm definitions and decision rules. Record any disagreement as a pre-registration amendment, never as a silent edit.

**Units.**
- **u** = one real-data gpt-oss-120b run (10 iterations × 5 folds).
- A SYN-P run costs 0.7–1.0u; a small-model run costs 0.5–1.0u.
- Throughput is about 2 u/h × 0.7 utilisation on ≤ 4 H100s, which is about 1.4u per wall-clock hour.

---

## 2. The frozen "Rogue One v2" and the freeze procedure

### 2.1 Definition of Full-v2

Full-v2 is the reference system. It is the control for almost every arm, and it is fixed at **G1 on Fri 16 Oct 2026**. It is Rogue One (Scientist → Extractor → Tester, plus a shared web-search summariser) with the freeze items below. Items from master plan §3.1 that are not in T1-minimum are marked.

| # | Freeze item (master plan §3.1) | Full-v2 setting | Implemented by | T1-minimum status |
|---|---|---|---|---|
| 1 | Tool-error feedback becomes its own flag | `ROGUE_ONE_TOOL_ERROR_FEEDBACK=message`, decoupled from `disabler()` | E2 | kept |
| 2 | Bookkeeping: per-fold round counter; `Pruned Iteration` records the trial | fixed | E2 | kept |
| 3 | Score after pruning (P4) | fixed | E2 | kept |
| 4 | Bounded tool output | `ROGUE_ONE_TOOL_OUTPUT_MAX_CHARS` pinned at G1 | E2 | kept |
| 5 | Augment vs replace, plus replay semantics | `ROGUE_ONE_FEATURE_POOL_MODE` = **M1 (A03) decision at G1**. **G1: `augment`, frozen by design under the E6 pre-declared fallback (2026-10-14); A03 runs as replace × Core-10 × 42 inside A05's sessions.** No fit-on-train replay fix, so a per-run replay-validity check is mandatory (`ROGUE_ONE_REPLAY_VALIDITY_CHECK=1`). Tier B datasets are excluded. | E6 | kept |
| 6 | Neutralise dataset-specific prompt hints; dump and diff rendered prompts | done | E5 | kept (two more hint sites found, Appendix A) |
| 7 | Web-search helper follows its caller's model | `ROGUE_ONE_WEBSEARCH_ROUTING=caller` | E7 | kept, but lands **Wed 21 Oct, after the freeze**: must be proven a no-op for uniform-model configs |
| 8 | Instrumentation (tokens per role, traces, energy, counters) | `ROGUE_ONE_LLM_TRACE=1`, all E3 counters | E3 | kept |
| 9 | Shared Serper query→result cache | ~~`ROGUE_ONE_SEARCH_CACHE=shared`~~ **G1: `off`** (Henrik, 2026-10-14): live SearXNG (`ROGUE_ONE_WEB_SEARCH_PROVIDER=searxng`), no cache; URLs, engines and caller still logged | E4 | kept (cache not used) |
| 10 | Pin remaining settings | every flag in `scripts/agent2027/v2_defaults.env` and in `wandb.config`; `ROGUE_ONE_DO_BALANCING=True` explicit; `SCIENTIST_AGENT_MAX_TOKENS` applied; **6 h wall-clock cap** | E1, E2, launcher | kept |
| 11 | Sandbox for generation-side code | `ROGUE_ONE_SANDBOX=on` | E13 | **G1: landed, `on`** (import allow-list, Henrik 2026-10-14) |
| 12 | Scientist feedback signal | **Decision D2.** ~~Recommended: keep the in-sample training fit, with a written justification. The L1 code is **CUT**.~~ **D2 = L1 (Henrik, 2026-10-09): `ROGUE_ONE_SCIENTIST_SIGNAL=inner_cv_l1`** | E14 | built (2026-10-14) |
| 13 | Extractor rounds R | `ROGUE_ONE_EXTRACTOR_ROUNDS` = **M4 (A02) decision at G1** (R=3 or R=10). **G1: R=10** (A02 read-out 2026-10-14) | E2 (knob) | kept |
| 14 | Prompt and tool must agree on pruned status (`ext/agent.py:895` vs `list_known_attributes_tool` `:570-596`) | **Decision D7.** Recommended: expose `Status` in the tool output. **D7 = expose (Henrik, 2026-10-09); not implemented at `agent2027-v2.0`, open in the pre-registration (Part VIII)** | E5 | decision (not in the breakdown; added here because it changes Full) |
| — | SYN-P classification fold scheme | `ROGUE_ONE_STRATIFIED_FOLDS=True` on SYN-P classification tasks if **D4** is adopted (so that M0's seed-42 partition equals LLM-FE's). **D4 adopted** | manifest | decision D4 |
| — | O120 serving topology | TP4 with 6 clients (t17), or TP2 for all post-freeze O120 jobs: **decision D5**, before M0. **D5 = TP2** (measured 2026-10-12: 1.45× TP4 throughput, 0 cap hits) | launcher | decision D5 |

**Fixed in every Full-v2 run and every arm** (unless that parameter is the factor under study):
- 10 iterations × 5 folds;
- inner-CV selection, rollback off;
- temperature 0.7;
- vLLM `vllm/vllm-openai:v0.22.0`;
- XGBoost 400 trees;
- R as frozen;
- `PARALLEL=6` per O120 server;
- the 6 h cap;
- ITT failure scoring.

The full table of fixed settings is in §4.3.

### 2.2 Freeze procedure (Thu 15 – Fri 16 Oct)

| When | Step | Done when |
|---|---|---|
| Thu 15 Oct | Read out M4 (A02) and M1 (A03) against their decision rules, which were committed Tue 13 before launch. Fix R and the pool mode. Run X9 (P7 guessability and retrieval checks). Decide D2, D3, D4, D7, and confirm the defaults (§6.4). | Decisions written into the draft pre-registration |
| Fri 16, morning | Merge every freeze-scope task into branch `agent2027/v2`: E1–E6, E8a, E13 (if ready), E12 outputs (serving flags per family) and the E8b corpus text. Write `scripts/agent2027/v2_defaults.env` with every §7 R3/R4 flag at its Full-v2 value. | `v2_defaults.env` committed; CI unit tests green |
| Fri 16, morning | **Rendered-prompt baseline.** Run a Full-v2 smoke (`ROGUE_ONE_SMOKE_FOLDS=1`, `ROGUE_ONE_PROMPT_DUMP=1`) on one SYN-P task and one Core-10 task. Store the dumps as the byte-identity reference for every later merge. | Dumps uploaded to W&B (artifact `prompt-baseline-v2.0`) |
| Fri 16, midday | **Pre-registration commit (X10).** Commit `docs/agent2027/preregistration.md` and tag `agent2027-prereg`, dated, before any Full-v2 result exists. Commit the freeze and tag `agent2027-v2.0-freeze`. | Tags pushed |
| Fri 16, afternoon | On CAIR: sync code (§5.2 step 2), build `rogue_one:agent2027-v2.0`, record the image ID, run one smoke run through the manifest launcher. | Smoke run in W&B with non-zero tokens for all roles, a trace file, and all required config keys (§7 R12) |
| Fri 16, 18:00 | Launch M0 (A05) interleaved with HC1 (A07) on `agent2027-v2.0`. | Session running; pre-flight logged |

**What the pre-registration commit must contain:**
- the Holm family of 10 contrasts, and the drop rules that shrink it;
- the margin formulas (NI and TOST) with their floors;
- the σ-pooling homogeneity rule;
- the M5 exclusion rule and the screen result;
- the LLM-FE selection handling;
- the SYN-P fold scheme;
- the P7 policy;
- the FX fraction, the restated falsifier and the alias-reading rules;
- the RT3 σ map, the column-renaming rule and the G3 pass rule;
- the KT3 corpus (correct, wrong and distractor documents), φ_wrong, and the seed counts;
- the C3 definition (D3);
- the C5a outcome and its separation-safe test;
- the RP1 floor threshold;
- the F0 prompt;
- the CAP-2 token metric;
- the token co-primary rule;
- the sentinel rule;
- the M1 and M4 rules as applied;
- an honest disclosure that M1 and M4 pilots ran on pre-freeze code before the commit.

**Rule after the freeze.** Any change that alters Full-v2 means re-running M0 and every arm. That fits only if the change is caught at **G2 (Mon 19 Oct)**, before any arm launches: one M0 + HC1 rerun costs about 60–70u. Post-freeze merges are allowed only if they are **default-off switches** (E7, E8b backend, E9a–f) and each one passes all three of these guards:
1. the E5 rendered-prompt diff of Full-v2 against `prompt-baseline-v2.0` is **byte-identical**;
2. unit tests show the switch-off path is unchanged;
3. the next session's sentinel (A06) sits within M0's band.

Each such merge produces a new image `rogue_one:agent2027-v2.N`. The image tag is logged in `wandb.config`.

**Planned post-freeze releases:**

| Image | Date | Contents |
|---|---|---|
| v2.1 | Mon 19 Oct | E9c planner hook |
| v2.2 | Tue 20 Oct | E9a restarts, E9b L0 |
| v2.3 | Wed 21 Oct (G0b-1) | E9d FX switches, E9e knowledge switch, E8b backend, E7 routing |
| v2.4 | Thu 22 Oct | E9f SET-phased |

---

## 3. Engineering tasks (checklist)

### 3.1 Summary

Notation: `main.py` = `rogue_one_main.py`; `sci/`, `ext/`, `tst/` = `rogueone/llm/agents/core/{scientist,extractor,tester}/agent.py`; `kn/web_search.py` = `rogueone/llm/agents/core/knowledge/web_search.py`; `config.py`, `console.py`, `pandas_exec.py` are under `rogueone/utils/`; `clients.py` = `rogueone/llm/clients.py`.

| ID | Task | Effort (d) | Due | Gate | Needed by (ablation docs) |
|---|---|---|---|---|---|
| E1 | Flag registry → `wandb.config`; manifest-driven task registration; manifest launcher | 0.75 | Mon 12 Oct | G0a | A02–A20 (and A21–A23 read its config keys) |
| E2 | v2 fixes, including the R knob | 1.1 | Mon 12 Oct | G0a | all v2 runs; A02 (R knob); A07 (flag) |
| E3 | Instrumentation: tokens, LLM traces, energy, counters, focus log, artifacts | 1.25 | Mon 12 Oct | G0a | all; essential for A07, A11, A13, A18, A19, A21, A23 |
| E4 | Serper cache, URL logging, caller attribution | 0.5 | Fri 16 Oct | G1 | A05–A20 (search-on arms), A13; A04 session C (caller role on W); A23 (cache/caller attribution for cost) |
| E5 | Prompt-hint neutralisation; rendered-prompt dump and diff | 0.5 | Mon 12 Oct | G0a | A02, A03, A04 (session C), A05, A07, A09–A17, A19, A20; every post-freeze merge |
| E6 | Augment mode and replay-validity check | 1–1.5 | **Tue 13 Oct, end of day** | G-E6 | A03, A05, A10, all v2 |
| E7 | Web-search routing follows caller; `ladder_session.sh` | 1 | Wed 21 Oct | G0b-1 | A19, A04 (session C); no-op check for A17, A18, A20; A23 (A19 points only) |
| E8a | SYN-P generator, context template, headroom calibration, minimal recovery detector, P7/P6 CPU screens, P6b twin | 1.35–1.75 | Mon 12 Oct | G0a | A02, A03, A05, A06, A08–A12, A14–A22 |
| E8b | **Reduced:** KT3 probe corpus and TF-IDF local backend | 1 | corpus Thu 15 Oct; backend Wed 21 Oct | G1, G0b-1 | A14 |
| E9a | Restart switch | 0.25 | Mon 19 Oct | (v2.2) | A10 |
| E9b | L0 switch | 0.5 | Tue 20 Oct | (v2.2) | A11 |
| E9c | Planner hook (oracle/misdirected) and RT3 tooling | 0.5–0.75 | Mon 19 Oct | (v2.1) | A09; A12 (introduces the `ROGUE_ONE_PLANNER_MODE` reader) |
| E9d | FX switches: SE\|T digest, template report, J−a with shadow mode, amnesic memory | 1.0 | Wed 21 Oct | G0b-1 | A12, A18 (J−a rules), A20 |
| E9e | Knowledge-backend switch (off / parametric / local wiring) | 0.25 | Wed 21 Oct | G0b-1 | A13, A14 |
| E9f | SET-phased build (phase-gated prompts) | 1–1.25 | Thu 22 Oct | K-SET (Fri 23) | A16 |
| E11a | G2 statistics: σ_d, σ_AA, ICC, margin value, informative tasks, token shares | 0.5 | Mon 19 Oct, 08:00 | G2 | A05–A07, A09–A17, A19, A20, A22, A23 |
| E11b | Analysis package (rest) | 1.75–2 | Tue 27 Oct | G4 | A06 and A07 (pulled forward as A06-T8), A08–A23 |
| E11c | RP1 scorer and fold-state reconstruction | 0–0.5 | Nov 1–6 (CPU; the scorer is needed by Sun 1 Nov for A18-T9) | — | A18 |
| E11d | F0 scorer | 0.5 | Nov 1–3 (CPU) | — | A17, A23 |
| E12 | Stage-0 capability smoke tests (executed as A04) | 0.5 | Fri 16 Oct (A/B); Thu 22 Oct (C) | G1 | A04, A17–A20 |
| E13 | Sandbox: restricted builtins; labels in memory; metadata off the mount (**at risk**) | 0.5 | Mon 12 Oct | G0a | A03 (A03-T2: merged or ED1), A05 (Full), A16, A18 (optional restricted builtins); paper claim |
| X1 | `EXTRA_ARGS` + `MAX_MODEL_LEN` hook in `vllm_session.sh` | 0.1 | Tue 13 Oct | — | A04, A05, A08, A17–A20 |
| X2 | Replay tool | 0.25–0.5 | Tue 13 Oct | — | A04, A18 |
| X3 | classicfe feature dump and `--fold-seed` | 0.25 | **Mon 12 Oct, 18:00** | — | A05, A08, A15 (CPU screen), A22 |
| X4 | LLM-FE spec files for SYN-P 7 | 0.25 | Mon 19 Oct | — | A08 |
| X5 | Token-logging proxy for the external arms | 0.25 | Mon 19 Oct | — | A08, A21, A23 |
| X6 | Comparator-output parser | 0.25 | Sat 24 Oct | — | A08, A16 (C1b reading) |
| X7 | F0 generator | 0.25 | Fri 23 Oct | — | A17, A23 |
| X8 | Remove and rotate the hard-coded webhook key | 0.1 | before any release (by Thu 12 Nov) | — | replication package |
| X9 | P7 guessability and retrieval checks | 0.25 | Thu 15 Oct | G1 | A14 |
| X10 | Pre-registration commit | 0.5 | Fri 16 Oct | G1 | all |

**Load.** The total to Oct 27 is **18–20.5 developer-days against 15.5 working days.** It holds only if the weekends of 10–11, 17–18 and 24–25 Oct are worked (decision D1). Run operations come on top.

**Engineering drop order** (pre-declared; apply in this order if the schedule slips):

| Step | Trigger | Consequence |
|---|---|---|
| ED1 | E13 slips | least-privilege claim dropped |
| ED2 | X2 replay tool slips past Tue 13 | Stage 0 uses only the end-to-end fold's counters; RP1 shrinks to Q0.8, Q4, Q9 and O120 |
| ED3 | X7 F0 generator not running by Wed 28 Oct | F0-best-of-k dropped (C5a unaffected) |
| ED4 | E9f SET not working by Fri 23 Oct (K-SET) | C1a dropped; Holm family shrinks |
| ED5 | — | M3 (A22) moves to the short paper |
| ED6 | — | L0 (A11) moves to T2 |

### 3.2 CUT (not built in T1-minimum)

| Item | What it was | Consequence |
|---|---|---|
| **E10** | Tool-factory refactor (`sci/agent.py:135-393`, `ext/agent.py:246-567`, `tst/agent.py:1041-1327`), `core/solo/agent.py`, Solo-auto-matched, S\|ET, Solo-native | C1 rests on C1a (SET-phased, built through phase-gated prompts in E9f) and C1b (external comparators, A08). SE\|T comes only from FX. |
| **SSP-3 generator** (part of the full E8b) | Semi-synthetic heart/housing/credit-g tables, named and anonymised twins, width distractors | C4b moves to the P6a/P6b twins (A15). The SSP-3 targets are pairwise ops that the brute-force screen would rule out anyway. |
| **Headroom screen** (part of the full E8b) | CPU screen to fill H1/H2 | H1/H2 stay empty. The paper says "not screened", not "the screen found nothing". |
| **L1 feedback-signal code** | Inner-CV scalar on a disjoint split for the Scientist | Freeze item 12 is decided by D2. `ROGUE_ONE_SCIENTIST_SIGNAL` is built only if D2 = L1. |
| Fallbacks-off switch | CAP-5 | Built only if RV1 triggers at G4 (Tue 27 Oct) |
| Q27, Q35-A3 rungs | CAP ladder and RP1 targets | Not downloaded |

### 3.3 Task details

Each block lists: the change (file:line @`07cbf93`), effort, due date, owner, **done when** (acceptance test), and which ablation documents depend on it.

---

#### E1 · Flag registry, manifest-driven task registration, manifest launcher
- [ ] **Owner:** Henrik · **Effort:** 0.75 d (tight; E1c is the largest part) · **Due:** Mon 12 Oct (G0a) · **In freeze:** yes (item 10)
- **E1a, flag registry.**
  - New module `rogueone/utils/flag_registry.py`. It lists every key in §7 R3/R4 plus the run-identity keys. It reads their **effective** values *after* dotenv loading.
  - `config.py:23-25` loads `.env`, then `.dev.env` with `override=True`, then `.secret.env`. A baked-in `.dev.env` can therefore silently override `docker -e` flags.
  - In `init_wandb_run` (`rogueone/utils/wandb_utils.py:83-121`), merge `flags/<KEY>` and the identity keys into `config=`.
  - Set `name=` (currently `cfg.experiment_name`, `:110-113`) to the run-name pattern in §7 R12.
  - Entity `PlanBench`.
- **E1b, task registration.**
  - New flag `ROGUE_ONE_TASK_REGISTRY` (a path to a YAML list of `config.yml` paths) replaces the hard-coded list at `main.py:1536-1566` when set.
  - `select_paths` (`main.py:1568-1580`) still filters by `ROGUE_ONE_DATASETS`.
  - Unset means the legacy 28-config list, byte-identical.
  - Also add `ROGUE_ONE_SMOKE_FOLDS=n`, which truncates `total_trials` at `main.py:1420-1424` to n × iterations. It is for smoke and Stage-0 runs only, and is never used in analysis.
- **E1c, manifest launcher.** Three files, used as described in §5.3:
  - `scripts/agent2027/make_manifest.py`
  - `scripts/agent2027/ro27_session_launch.sh`
  - `scripts/agent2027/ro27_clients_manifest.sh`

  It must provide:
  - a per-row task-directory copy (concurrent runs of the same dataset otherwise collide on `output/`);
  - container names unique per session and row;
  - interleaving in randomised order;
  - a sentinel slot;
  - retry-once-solo for crashes;
  - success = `fold_aggregation_summary.csv` exists (lesson 3 of `clients_t16_rerun.sh:38-45`);
  - the 6 h cap via `docker stop`;
  - the identity env vars.

  Repository hygiene (found in the A01/A22 checks): `.gitignore` ignores `*.csv` and `paper/`, so `manifests/agent2027/**/*.csv` would be silently ignored. E1 adds `!manifests/agent2027/**` (and `!manifests/agent2027/**/*.csv`) to `.gitignore`; files under `paper/` need `git add -f`. Branch `nested-cv-rebuttal` (`07cbf93`, `1bdaf42`) is **not on GitHub** as of 7 Oct (the remote tip is `2cd6d0a`); it must be pushed before any server fetch (A01-T4), because the launchers in R6 "Existing" exist only locally.
- **Done when:**
  - [ ] the G0a smoke run appears in W&B under the §7 R12 name, with every required config key and with effective flag values that match `v2_defaults.env`;
  - [ ] the build context on CAIR contains no `.dev.env`, which the image build step checks;
  - [ ] a 3-row dry-run manifest gives correct, unique container names; a deliberately failing row is retried once, alone, at the end; a row with `cap_hours=0.01` is stopped and logged as `cap_hit`, not retried;
  - [ ] unsetting `ROGUE_ONE_TASK_REGISTRY` selects the legacy list (unit test).
- **Needed by:** every GPU ablation (A02–A20). A21–A23 query its config keys.

#### E2 · v2 fixes (+ R knob)
- [ ] **Owner:** Henrik · **Effort:** 1.1 d · **Due:** Mon 12 Oct (G0a) · **In freeze:** items 1–4, 10, 13
- **Change:**
  1. **Tool-error feedback flag** `ROGUE_ONE_TOOL_ERROR_FEEDBACK=message|none`. Decouple `disabler()` (`console.py:31-41`, which returns `args[1]` or `None` when the console is disabled) from tool return values, for **both** `console_error_print` and `console_print`. Route every return site through the flag:
     - `_report_tool_error`: `sci/agent.py:115-131`, `ext/agent.py:207-223`, `tst/agent.py:1023-1039` (this includes the `pandas_exec.py:146-155` KeyError hints raised into them);
     - direct `console_error_print` returns: `sci/agent.py:154`, `:256`; `ext/agent.py:266`, `:361`, `:501-534`;
     - ILLEGAL COMMAND, which returns `console_print` (`ext/agent.py:338-340`).

     The duplicate-name rejection (`ext/agent.py:486-489`) returns a plain string and stays **visible in both modes**, as it was in t17.
  2. **Per-fold round counter.** `main.py:553` passes the global `index`, so `sci/agent.py:473` renders "Round 12/10". Pass `iteration_in_fold` and render "Round {i+1}/{N}".
  3. **`Pruned Iteration` records the trial.** `tst/agent.py:1309-1310` and `:1607-1608` write `f"Trial {step}"`, where `step` is `fold_run_id` (`main.py:792-793`). Write `trial_index` instead.
  4. **P4, score after pruning.** After `self.prune_attributes_in_df(...)` at `tst/agent.py:2142`, rebuild `X_train`/`X_test` from the post-pruning active set before any metric is computed. This removes the one-iteration pruning lag.
  5. **Bounded tool output** via `ROGUE_ONE_TOOL_OUTPUT_MAX_CHARS` (pinned at G1; proposed 8000):
     - Extractor `generic_pandas_tool` result: `ext/agent.py:366-375`;
     - Tester `generic_python_executor_tool`: `tst/agent.py:1208`. The `:1210` bound is applied only to the log line, so the returned output is unbounded today.
  6. **Scientist `max_tokens`.** The Scientist's `ModelSettings` (`sci/agent.py:403-407`) passes no `max_tokens`. Apply `SCIENTIST_AGENT_MAX_TOKENS` (`config.py:412`); its value is pinned at G1 (proposed 16384).
  7. **Reasoning-effort passthrough flag** `ROGUE_ONE_REASONING_EFFORT_PASSTHROUGH=0|1`. `clients.py:522-525` pops `reasoning_effort`. The default is 0, which is the current behaviour; 1 is for T2 only.
  8. **R knob** `ROGUE_ONE_EXTRACTOR_ROUNDS`, which replaces `for _ in range(10)` at `ext/agent.py:967`. The code default stays 10; v2 takes the A02 decision.
- **Done when:**
  - [ ] unit tests per return site in both modes. With `none`, exactly the t17 paths return `None` and the duplicate message stays visible.
  - [ ] a 2-fold smoke shows "Round 1/10 … 10/10" for fold 2 in the dumped Scientist prompts.
  - [ ] `Pruned Iteration` values equal `Trial <trial_index>`.
  - [ ] a test where the Tester prunes a feature shows the scored feature count equal to the active count.
  - [ ] a 100 KB tool result is cut at the cap, with a truncation marker.
  - [ ] `ROGUE_ONE_EXTRACTOR_ROUNDS=3` gives exactly 3 `Runner.run` calls per iteration (E3 counter).
- **Needed by:** all v2 runs; A02 (R knob); A07 (flag).

#### E3 · Instrumentation (tokens, traces, energy, counters, focus log, artifacts)
- [ ] **Owner:** Henrik · **Effort:** 1.25 d · **Due:** Mon 12 Oct (G0a) · **In freeze:** item 8
- **Change:**
  - **Tokens per role per iteration**, including reasoning and discarded re-rolls (the splice re-roll at `clients.py:572-640`):
    - today they are recorded per `agent_name` in `_record_token_usage` (`clients.py:499-514`);
    - web-search calls are logged as `WebSearchAgent` (`kn/web_search.py:191`), so they must be **charged to the calling role** (contextvar set by the caller; see E4);
    - verify on 0.22.0 that `completion_tokens` includes reasoning tokens (compare with the trace).
  - **Full LLM request/response traces** under `ROGUE_ONE_LLM_TRACE=1`, written in `VLLMCompatibleAsyncOpenAI._wrap_completions` (`clients.py:516-562`) to `output/llm_traces.jsonl.gz`. `clients.py` has no trace logging today.
  - **Rescue counters per model family:** the text-format tool-call rescue (`_resolve_text_tool_call`, `clients.py:37-80`; JSON only, so Qwen's XML-style calls are never rescued) and splice re-rolls.
  - **Fallback counters:**
    - Extractor baseline fallback (`ext/agent.py:599-844`, triggered at `:993-1003`);
    - Scientist heuristic focus (`sci/agent.py:501-517`);
    - Tester fallback report (`tst/agent.py:1499-1505`);
    - pruning floor (`tst/agent.py:1617-1630`).
  - **Error counters for every path that bypasses `_report_tool_error`:**
    - ILLEGAL (`ext:338`);
    - duplicate name (`ext:486-489`);
    - append rejections (`ext:501-534`; these are also the silent-rejection counter);
    - empty-search and no-`result` returns (`sci:154`, `sci:256`, `ext:266`, `ext:361`);
    - plus HTTP 400 / `finish_reason=length` context errors.
  - **Report reads:** the Scientist's `generic_pandas_tool` (`sci/agent.py:180-270`) touching `feature_assessment`.
  - **Active features per iteration; per-inner-fold scores** (`_inner_cv_score`, `tst/agent.py:194`); **wall clock per agent**.
  - **Focus JSONL** per (fold, iteration), written next to `main.py:548-557`. The console print of the focus is lost when the console is disabled.
  - **GPU energy:** sidecar `scripts/agent2027/energy_sampler.py`, NVML power at 1 Hz on the session's GPUs, logged to W&B run `<SESSION_ID>__energy`.
  - **Trace fields (A18-T2):** every `llm_traces.jsonl.gz` record carries `agent_name`, `caller` (for W), `stage` (Tester assessment or pruning), `index`, `fold_run_id`, `iteration_in_fold`, `session_name`, `call_seq`, the full request kwargs as sent (`messages`, `tools`, `tool_choice`, `extra_body`, `temperature`, `max_tokens`, `model`, `seed`) and the wrapper's final response (`message`, `finish_reason`, `usage`).
  - **Per-trial tables (A18-T2, A22-T1, A10-T18):** `dump_data` (`main.py:839-847`) also writes `intermediate_steps/df_attributes_explanations_trail_<index>.csv` (every attribute row including `Pruned`, with `Attribute`, `Pandas Command`, `Status` after pruning, the fallback flag, E6's `Seeded` marker and `Scored` = 1 iff the row was not `Pruned` before `test_hypotheses`, copied before that call) and `df_scientist_feedback_trail_<index>.csv` (the Scientist's feedback pool).
  - **Stateless rule:** at run end, upload the run's `output/` folder (traces, prompts, focus log, `df_test_pool_final.csv`, attribute tables, error JSONL, retrieval transcript, `fold_aggregation_summary.csv`) as W&B artifact `run-output`.
- **Done when:**
  - [ ] the G0a smoke shows **non-zero** completion tokens for all three roles in W&B, with search tokens charged to the caller. Historically every `tok_*` export was zero, so this check is mandatory.
  - [ ] the trace has exactly one record per LLM call (count = `llm_calls_total`).
  - [ ] the focus JSONL has one row per iteration.
  - [ ] 100% of trace records carry the A18 fields above, and both per-trial tables exist for every trial that called `dump_data` (checked on the v2.0 freeze smoke, Fri 16 Oct, by A10-T18, A18-T2 and A22-T1).
  - [ ] every counter listed above exists in the summary (zero allowed).
  - [ ] the energy run is logged for the smoke session.
  - [ ] the `run-output` artifact downloads and the analysis fetch script reads it.
- **Needed by:** practically every ablation. Critical for A05→A18 (traces), A09 (focus log), A07 (error counters), A11 (report reads), A13 and A19 (tokens per role), A21 and A23 (cost axes).

#### E4 · Serper cache, URL logging, caller attribution
- [ ] **Owner:** Henrik · **Effort:** 0.5 d · **Due:** Fri 16 Oct (must be in `agent2027-v2.0`) · **In freeze:** item 9
- **Change:**
  - **Cache.** Add a query→result SQLite cache inside `_web_search` (`kn/web_search.py:75-187`; the Serper call is `search.run(variant)` at `:128`). It is keyed by the sanitised query (`_sanitize_query`, `:32`), controlled by `ROGUE_ONE_SEARCH_CACHE=shared|off`, and lives at `ROGUE_ONE_SEARCH_CACHE_PATH`, mounted from `~/rogueone-serper-cache/`.
  - **URLs.** Switch `.run()` to `.results()` and record URLs in the existing retrieval transcript (`rogueone/utils/retrieval_audit.py`; calls at `kn/web_search.py:96-156`).
  - **Caller attribution.**
    - `ext/agent.py:270-272` logs Extractor searches as `"ScientistAgent"`. Fix this.
    - Pass the caller role through `WebSearchAgent.explain_query` (`kn/web_search.py:292`), so that tokens and Serper calls are charged to the caller.
    - Callers: `sci/agent.py:149`, `ext/agent.py:261`, `tst/agent.py:1058`.
  - Log the `serper_ok_false_rate`.
- **Done when:**
  - [ ] the same query twice in one run hits Serper once (cache hit logged);
  - [ ] transcript rows carry the caller role and URLs;
  - [ ] Extractor searches are attributed to the Extractor;
  - [ ] `serper_calls` and `serper_ok_false_rate` appear in W&B;
  - [ ] Serper credits were checked before M0 (about 500–1,000 queries per run before cache hits).
- **Needed by:** A05–A20 (every search-on arm); A13 (the per-role search share comes free from Full's logs). Disclose that the cache lowers live-search variance.

#### E5 · Prompt-hint neutralisation, rendered-prompt dump and diff
- [ ] **Owner:** Henrik · **Effort:** 0.5 d · **Due:** Mon 12 Oct (G0a) · **In freeze:** items 6 and 14
- **Change:**
  - **Neutralise these dataset-specific hints:**
    - balance-scale torque examples: `kn/web_search.py:82-83` (tool docstring) and `:225`, `:228-229` (summariser instructions);
    - CAR advice: `sci/agent.py:441` (run wi5r0oki; the "buying/maint/safety" example) and `:443` ("CAR with 6 raw categorical columns");
    - **also found during verification:** game-specific helper names in the Extractor prompt, `ext/agent.py:454` and `:901` (`def winner(b)`, `minimax`, `solve`, `lines = [...]`). Replace them with neutral names.
  - **Freeze item 14 (decision D7):** make `list_known_attributes_tool` (`ext/agent.py:570-596`, names only) agree with the instruction at `ext/agent.py:895` (look up `Status="Pruned"`). Recommended: expose `Status`.
  - **Prompt dump:** with `ROGUE_ONE_PROMPT_DUMP=1`, write the rendered system prompt and first user message of every agent call to `output/prompts/`.
  - **Diff tool:** `scripts/agent2027/prompt_diff.py <dumpA> <dumpB>` reports byte-level differences per role, phase and iteration.
- **Done when:**
  - [ ] on a non-balance-scale, non-CAR task, rendered prompts contain none of {torque, CAR, buying, maint, safety, winner, minimax}, except inside that task's own `context_and_goal`;
  - [ ] `prompt_diff.py` of the same config across two commits with no intended prompt change reports byte-identical prompts;
  - [ ] the leak check for A15 (P6b names, experiment name, `dataset_id` in the feedback pool) runs on the dump.
- **Needed by:** A05 (defines Full), A09–A16, A19, A20; it is the **regression guard for every post-freeze merge** (§2.2).

#### E6 · Augment mode and replay-validity check
- [ ] **Owner:** Henrik · **Effort:** 1–1.5 d · **Due:** **merged by end of Tue 13 Oct** (gate G-E6; M1 launches Wed 14). The G0a week is full, so in practice E6 must start on the weekend of 10–11 Oct or on Mon 12. · **In freeze:** item 5
- **Change:**
  - `ROGUE_ONE_FEATURE_POOL_MODE=replace|augment`. Augment seeds raw and ordinal features as replayable, prune-protected rows inside `_reset_fold_state` (`main.py:417-457`).
    - It reuses `ext/agent.py:599-844` (baseline features) and `:42-82` (`_ordinal_encode_categorical`).
    - It exempts these rows from pruning in `attribute_pruning_tool` (`tst/agent.py:1269-1327`) and from the floor logic (`tst/agent.py:119-160`).
  - Per-run replay-validity check with `paper/scripts/diag_finding39_authoring_frame.py` (`ROGUE_ONE_REPLAY_VALIDITY_CHECK=1`); it logs `replay_nonrepro_folds` and `replay_nan_fills` in the W&B summary. There is no fit-on-train fix, so Tier B datasets stay out.
  - Seeded rows carry a boolean attribute-table column `Seeded` (used by A22, A17 `func_success` and A03).
  - With the flag `replace`, behaviour is byte-identical to before E6. M4's runs and M1's replace arm share this commit.
- **Done when:**
  - [ ] under augment, the iteration-0 pool holds every raw column (coverage 1.0); a pruning test cannot remove them; the seeding is re-applied by any later call of `_reset_fold_state` (needed by E9a);
  - [ ] under replace, the prompt dump and pool are identical to the pre-E6 commit;
  - [ ] the replay check result is logged on every run.
- **Contingency (pre-declared):** if E6 is not merged by the end of Tue 13 Oct, freeze augment by design (E6 by Fri 16), and A03 instead runs frozen replace × Core-10 × 42 (10u) against M0's seed-42 runs.
- **Needed by:** A03; A05 (if augment); A10 (reset seeding); every v2 run.

#### E7 · Web-search routing follows caller; multi-server ladder launcher
- [ ] **Owner:** Henrik · **Effort:** 1 d · **Due:** Wed 21 Oct (G0b-1) · **In freeze:** item 7, but lands after the freeze, so the no-op guard is mandatory
- **Change:**
  - Add `"websearch"` to `AGENT_ROLES` (`config.py:221`; there is no `cfg.py`).
  - Under `ROGUE_ONE_WEBSEARCH_ROUTING=caller`, `_run_search_agent` uses the **calling role's** endpoint and model instead of the global `llm_cfg` (`kn/web_search.py:189` endpoint, `:238` model).
  - `scripts/agent2027/ladder_session.sh`:
    - N vLLM servers in one session;
    - refuses a session if ΣTP > 4;
    - per-family `EXTRA_ARGS`;
    - CPU-aware client cap (≤ 12–14 clients in total, 12 CPUs each);
    - sets `ROGUEONE_ALLOW_SECOND_SERVER=1` internally (`paper/scripts/vllm_session.sh:121-130`);
    - interface (as used by A04 session C and A19, and needed by A17's Q122 slot): `SERVERS='<RUNG>:<gpu ids>:<port> …'`, e.g. `SERVERS='O120:<g1>,<g2>:8010 O20:<g3>:8020 Q0.8:<g4>:8030'`, with the other env keys of `ro27_session_launch.sh`; it must also accept two servers of one rung on different ports (R8) and honour `CLIENTS_SCRIPT` (A04-T7).
- **Done when:**
  - [ ] for a uniform-model config, the E5 prompt diff and the endpoints in the traces are identical to `prompt-baseline-v2.0` (unit test plus diff), so M0 and the CAP-1 runs before Oct 21 stay valid;
  - [ ] the A04 session-C check shows each role and its search helper reaching the intended server, in the traces and in `wandb.config`.
- **Needed by:** A19 (required); A04 (session C); A17, A18, A20 (no-op check only, because uniform rungs already send search to the global model = the rung).

#### E8a · SYN-P generator, context template, headroom calibration, minimal recovery detector, CPU screens, P6b twin
- [ ] **Owner:** Henrik · **Effort:** 1.35–1.75 d (1.0 generator, 0.25–0.5 detector, 0.1–0.25 twin and screens) · **Due:** Mon 12 Oct (G0a), so that the M5 CPU screen runs Monday night
- **Change:**
  - **Generator** `scripts/agent2027/synp/generate_synp.py`:
    - writes 9 task folders `tasks/<task_type>/synp_<id>/` (`config.yml`, `data.csv`, `labels.csv`) in the existing task format;
    - n = 2,000; p ≈ 20–30: 3 relevant columns, correlated decoys, near-collinear pairs;
    - one shared `context_and_goal` template with column descriptions;
    - **standardised regression targets**, so that J−a's SHAP 0.1 threshold means the same on every task;
    - headroom calibration: XGB(raw + φ) − XGB(raw) ≥ 0.06 balanced accuracy, or ≥ 15% nRMSE;
    - stratified-fold compatible for classification tasks.
  - **Metadata stays off the mount.** φ, seeds, φ_wrong, oracle focus strings and P5/P7 detector definitions go to `synp_meta/<id>.json`, which on CAIR lives at `~/rogueone-synp-meta/` and is **never** mounted into an agent container. The only exception is the RT3 oracle focus file, derived by E9c.
  - **Registration** of the 9 folders in the E1 task registry. Task types are fixed in `synp_meta/registry.yaml` (proposal in §7 R10).
  - **Minimal recovery detector** `scripts/agent2027/analysis/recovery_detector.py`:
    - replays every active non-fallback feature's pandas command on the fold's training rows (`rogueone/utils/fold_safe_features.py`);
    - flags `rec` when |Spearman| ≥ 0.95 with φ, or normalised MI ≥ 0.9 for rule and threshold families (P3, P5, P7);
    - excludes raw columns, single-column monotone transforms, and features from the fallback;
    - outputs `rec_any`, `rec_sel` and `iter_first_rec` per fold-episode.

    It is needed by **G1** (M4/M1 rules) and **G2** (ICC, informative tasks), not only at E11.
  - **CPU screens:**
    - pairwise-op screen `scripts/agent2027/synp/screen_pairwise.py`: no unary or pairwise op of the raw columns may reach |ρ| ≥ 0.95 (NMI ≥ 0.9) with φ on P6a and P7;
    - P7's ratio component alone must not meet the detector criterion (build an unguessable constant into φ).
  - **P6b exact twin of P6a** (decision D8, by Mon 12):
    - same rows, same φ, same generator seed;
    - columns renamed f1..fp and levels c1..ck (order kept);
    - a neutral context of equal length and an opaque experiment name;
    - if φ is a monomial, each P6b column rescaled by a hidden positive constant.

    P6a's φ must be a named composite that passes the pairwise screen. A BMI-like w/h² with realistic ranges does **not** pass, because it is nearly rank-identical to w/h.
- **Done when:**
  - [ ] the 9 folders load through `ExperimentConfig.from_yaml` and contain no metadata;
  - [ ] headroom passes on P1–P7 (P0 has none by design);
  - [ ] detector unit tests pass: φ injected as a feature gives `rec=1`; a single-column monotone transform gives `rec=0`; a fallback feature is ignored;
  - [ ] the P7 and P6a screens pass;
  - [ ] the P6b/P6a row and φ hashes match, and the E5 grep finds no name leakage.
- **Needed by:** A02, A03 (add-on), A05, A06, A08, A09 (oracle strings), A10, A11, A12, A14 (P7), A15 (twin), A16, A17, A19, A20, A21, A22.

#### E8b · Reduced: KT3 probe corpus and TF-IDF local backend
- [ ] **Owner:** Henrik · **Effort:** 1 d · **Due:** corpus text and an offline TF-IDF index by **Thu 15 Oct** (needed by X9 and the Fri 16 pre-registration); backend integrated as a dormant switch by **Wed 21 Oct** (G0b-1). The breakdown placed the whole day after Oct 17, but the corpus must be frozen on Oct 16. Write it together with P7 in E8a.
- **Change:**
  - **Corpora** under `scripts/agent2027/corpora/`, author-written, frozen in the pre-registration and disclosed:
    - `kt3_correct/`: the P7 document plus 5–10 distractor documents;
    - `kt3_wrong/`: the same corpus with the P7 document replaced by φ_wrong. φ_wrong must satisfy |ρ(φ_wrong, φ)| < 0.5 and have about zero headroom on CPU.
  - **Local backend** `rogueone/llm/agents/core/knowledge/local_corpus.py`, using sklearn TF-IDF (`rank_bm25` is not in `requirements.txt`). It is selected by `ROGUE_ONE_KNOWLEDGE_BACKEND=local` with `ROGUE_ONE_LOCAL_CORPUS=kt3_correct|kt3_wrong`, and swaps the `_web_search` tool (`kn/web_search.py:241`) inside the same summariser (`_run_search_agent`, `:57-241`). Tool names, docstrings and the summariser prompt stay **byte-identical** across the parametric and local arms.
- **Done when:**
  - [ ] in X9, the offline replay of M4's P7 search queries ranks the correct document near the top;
  - [ ] a 1-fold smoke on P7 under `local` completes. **Kill rule:** if this has not passed by **Tue 27 Oct**, KT3 (A14) is cut;
  - [ ] the Full-v2 prompt diff is unchanged.
- **CUT from the full E8b:** SSP-3, real anonymised twins, width distractors, CPU headroom screen.
- **Needed by:** A14.

#### E9 · Switch arms (all default-off; Full-v2 byte-identical)
All E9 merges follow the post-freeze guard in §2.2. Restarts and L0 are built **first**, because LF1 (A10) is the only Holm contrast for C3.

**E9a · Restart switch**
- [ ] **Effort:** 0.25 d · **Due:** Mon 19 Oct (image v2.2, launch Tue 20)
- **Change:** `ROGUE_ONE_RESTART_MODE=off|every_iteration`. Call `_reset_fold_state()` at **every** iteration; today it runs only when `iteration_in_fold == 0` (`main.py:543-544`).
  - `df_test_pool` is never touched by the reset (verified), so inner-CV selection still works.
  - Augment seeding happens inside the reset (E6).
  - "Round 1" is selected automatically when the focus history is empty (`sci/agent.py:472-475`).
- **Done when:**
  - [ ] unit tests: `df_test_pool` survives every reset; augment rows are re-seeded;
  - [ ] the Scientist prompt at iteration 5 renders the same template as at iteration 0;
  - [ ] the Full prompt diff is unchanged.
- **Needed by:** A10.

**E9b · L0 switch (open-loop planner)**
- [ ] **Effort:** 0.5 d · **Due:** Tue 20 Oct (image v2.2)
- **Change:** `ROGUE_ONE_FEEDBACK_LEVEL=full|l0`.
  - `df_test_results` is cut to the trial `id`. The rows are built at `main.py:384-415` and read through `sci/agent.py:180-270`.
  - `attribute_lookup_tool` (`sci/agent.py:306-335`) drops `Status`, `Pruned Iteration` and `Prune Rationale`.
  - **The Scientist's rendered prompt only** neutralises:
    - `sci/agent.py:413-414`;
    - `:430`;
    - `:440`;
    - the restoration clause of the anti-repetition rule (`:443`);
    - the `generic_pandas_tool` docstring (`:180-200`);
    - the TESTER team-context text (`rogueone/llm/prompts/agent_roles.py:24-29`), edited in a **local copy**, never in the shared enum.
- **Done when:**
  - [ ] the E5 diff shows exactly these 7 edits;
  - [ ] a test confirms the L0 Scientist cannot read any `train_*` or `feature_assessment` value;
  - [ ] the Full diff is unchanged.
- **Needed by:** A11.

**E9c · Planner hook and RT3 tooling**
- [ ] **Effort:** 0.25 d hook + 0.25–0.5 d tooling · **Due:** Mon 19 Oct, morning (image v2.1)
- **Change:**
  - `ROGUE_ONE_PLANNER_MODE=oracle|misdirected` with `ROGUE_ONE_PLANNER_FOCUS_FILE=<path>`. This skips `determine_focus` (`main.py:548-555`) and uses the focus string from the file for (fold, iteration).
  - **Tooling:**
    - the pre-registered σ map;
    - the column-renaming rule (σ(T)'s relevant columns → T's decoys least correlated with φ's inputs);
    - extraction of M0's seed-42 focus strings from the E3 focus JSONL.
  - Oracle strings come from off-mount metadata. The focus file is mounted **only** for the RT3 arms.
- **Done when:**
  - [ ] oracle runs show the fixed string in every iteration;
  - [ ] the misdirected focus for (T, f, i) equals M0's σ(T) focus at (f, i) after renaming.
- **Needed by:** A09.

**E9d · FX switches**
- [ ] **Effort:** 1.0 d · **Due:** Wed 21 Oct (G0b-1, image v2.3)
- **Change:**
  - **P− (SE|T):** `ROGUE_ONE_PLANNER_MODE=none`.
    - The Scientist is not called (`main.py:548` skipped).
    - In **round 1 only**, the Extractor's message (`ext/agent.py:971`) gets a pre-specified digest. It is injected once because the R rounds share one session (`:965-971`). The digest holds: the last Tester report (LLM or template), the train-metric row, the Status/Prune Rationale table, and the focus history.
    - The Extractor must start with a `FOCUS:` line, which is logged as the focus history.
  - **A−:** `ROGUE_ONE_ASSESSMENT_MODE=template`. A template report with the same numbers: the metric table via `_metric_profiles_to_markdown` (`tst/agent.py:692`), the prunes, and the top-5 SHAP. Under J+, its prune field lists the LLM's prunes.
  - **J−a:** `ROGUE_ONE_PRUNING_MODE=rules`. The Stage-B rules (`tst/agent.py:1545-1558`) are executed as code:
    - in a VIF ≥ 1e10 cluster, prune the member with the lowest mean|SHAP| (ties broken by MI);
    - never prune for VIF when SHAP > 0.1;
    - for VIF in [10, 1e10), prune when |PFI| ≤ 1 SD of its permutation spread and SHAP < 0.1;
    - keep the same floor (`tst/agent.py:119`, `:1617`).

    `ROGUE_ONE_PRUNING_SHADOW=1` (FX J+ cells) logs J−a's decisions without applying them; they must never reach a prompt.
  - **M−:** `ROGUE_ONE_MEMORY_MODE=amnesic`. The focus history is truncated to the last focus (`main.py:552`, `sci/agent.py:472-475`), and the notebook tools (`take_a_note`, `read_notes` in the tool list at `sci/agent.py:461-468`) are off.
- **Done when:**
  - [ ] the **G0b-1 smoke** runs all 8 FX cells for one fold on one SYN-P task;
  - [ ] the prune-rate manipulation check differs between J+ and J−a;
  - [ ] shadow decisions are logged;
  - [ ] the Full prompt diff is unchanged.
- **Needed by:** A12, A20.

**E9e · Knowledge-backend switch**
- [ ] **Effort:** 0.25 d · **Due:** Wed 21 Oct (G0b-1, image v2.3)
- **Change:** `ROGUE_ONE_KNOWLEDGE_BACKEND=web|off|parametric|local`.
  - **off** removes the search tool from all three roles: `search_tool` at `sci/agent.py:464`; `search_in_literature_tool` at `ext/agent.py:939` and in the prompt's tool list at `:921`; `tst/agent.py:1492`. It also removes the search instructions `sci/agent.py:431` ("Always perform at least one web search") and `:438`. The Scientist's no-tool fallback already has `tools=[]` (`sci/agent.py:564`).
  - **parametric** keeps the same summariser with no tool and the instruction "answer from your own knowledge".
  - **local** is the E8b backend.
- **Done when:**
  - [ ] the E5 diff for `off` shows only the search lines removed;
  - [ ] a 1-fold smoke makes no calls to missing tools (such calls would confound A07);
  - [ ] the Full diff is unchanged.
- **Needed by:** A13, A14.

**E9f · SET-phased build (phase-gated prompts; E10 is cut)**
- [ ] **Effort:** 1–1.25 d · **Due:** build lands **Thu 22 Oct** (image v2.4); crash-only pilot that night; **kill date Fri 23 Oct (K-SET)**
- **Change:** `ROGUE_ONE_TOPOLOGY=three_agent|set_phased`.
  - **One session per iteration.** One `SQLiteSession` is created per iteration in `forward_pass` (`main.py:536-548`) and passed to `determine_focus` (replacing `sci/agent.py:499`), to the Extractor's R rounds (replacing `ext/agent.py:965`), and to Tester Stage A and Stage B (replacing `tst/agent.py:1371`; session names at `:1496` and `:1590`).
  - **One neutral system prompt** holding only shared content: the global goal, `context_and_goal`, the raw column list and the output style. Each role's workflow and hard-rule block goes **verbatim** into its phase message. The "You are the X Agent" and "current stage" lines are removed by a fixed rule frozen before any core run.
  - Per-phase tool lists (the generating phases never hold the Tester's labelled executor) and per-phase `ModelSettings`.
  - The fallback paths keep their own fresh sessions, and they are counted.
- **Done when:**
  - [ ] the crash-only pilot on `blood` and `communities` (102 columns) completes;
  - [ ] the Full diff is unchanged;
  - [ ] `paper/scripts/scan_leakage.py` is clean on the SET outputs.
- **Needed by:** A16.

#### E11 · Analysis package
- [ ] **E11a, G2 statistics.** 0.5 d · **Due:** Mon 19 Oct 08:00 · script `scripts/agent2027/analysis/g2_stats.py`.
  - **Outputs:**
    - σ_d per dataset; pooled σ_AA; σ_samp from A01, and the homogeneity check that decides whether t17/W0 are pooled;
    - the recovery ICC (task vs partition vs sampling);
    - the NI-margin **value**;
    - the informative-task list per contrast;
    - Tester Stage A/B token shares (FX token rule);
    - the CAP-2 big-model token threshold;
    - B_d;
    - the sentinel task.
  - **Done when:** it reproduces a hand-computed toy example, and the G2 memo (logged to W&B, group `ANALYSIS`) exists before any arm launches.
- [ ] **E11b, rest of the package.** 1.75–2 d · **Due:** Tue 27 Oct.
  - **Scripts** under `scripts/agent2027/analysis/`:
    - `fetch_wandb.py`;
    - the full `recovery_detector.py` (adds the any-iteration and selected variants, the φ_wrong detector and the P5 LW·LD detector);
    - `glmm_holm.py` (GLMMs, exact tests, TOST, Holm; the implementation, e.g. glmmTMB in an R container for beta-binomial, is named in the pre-registration);
    - `funnel.py`;
    - `process_metrics.py` (errors, re-proposal and restoration, focus novelty, implementability sheets, report reads, lineage, LLM-vs-rule Jaccard, κ sample sheets);
    - `sentinel_drift.py`;
    - `evaluator_transfer.py` (A22);
    - `pareto.py` (A23);
    - the anytime/iso-cost extension of `paper/scripts/diag_selection_policy.py` (A21).
  - **Done when:** each script reproduces a toy example, and runs end to end on M0's artifacts.
- [ ] **E11c, RP1 scorer.** 0–0.5 d for fold-state reconstruction, plus CPU scoring Nov 1–6 · `scripts/agent2027/analysis/rp1_score.py` · needed by A18.
- [ ] **E11d, F0 scorer.** 0.5 d, CPU from Nov 1 · `scripts/agent2027/f0_score.py`. It reuses `pandas_exec.py`, `fold_safe_features.py` and Stage C. Needed by A17 and A23.
- **Owner:** Henrik. **Needed by:** A05–A23. The M0 Jaccard (LLM vs rules) is computed post hoc from the logged metric tables, because the shadow mode lands after M0.

#### E12 · Stage-0 capability smoke tests (executed as A04)
- [ ] **Owner:** Henrik · **Effort:** 0.5 d · **Due:** sessions A/B Tue 13 – Fri 16 Oct (before the freeze); session C Wed 21 – Thu 22 Oct (after E7)
- **Change:**
  - Downloads inside a container (pinned revisions, disk checked first).
  - Per-family serving flags recorded.
  - 50 streamed Extractor calls per model, taken from the **G0a smoke traces**, through X2.
  - One balance-scale fold × 10 iterations per new model (`ROGUE_ONE_SMOKE_FOLDS=1`).
  - Mixed-routing check (d) after E7.
- **Pass rules** (pre-declared):
  - if Q9 makes < 80% valid calls, or vLLM throws parser exceptions, switch to the Qwen3 dense ladder (§7 R9);
  - if Q122 alone fails, keep the small ladder, use O120 as the only top anchor, and fit C5a on 0.8–9B;
  - if O20 fails, CAP-2 uses Q122/Q4 and CAP-3 moves to Q4;
  - **the image is never changed.**
- **Done when:** a pass/fail table per rung exists in W&B (project `RogueOne-AGENT2027-prefreeze`, group `A04`).
- **Needed by:** A04, A17, A18, A19, A20.

#### E13 · Sandbox (at risk)
- [ ] **Owner:** Henrik · **Effort:** 0.5 d · **Due:** Mon 12 Oct (G0a); must be in `agent2027-v2.0` · **In freeze:** item 11
- **Change:** `ROGUE_ONE_SANDBOX=on|off`.
  - Pass a restricted `__builtins__` (no `open`, `__import__`, `exec`, `eval`) to the generation-side exec scopes: `pandas_exec.py:139-143` (Extractor), `sci/agent.py:245-252` (Scientist), and the Tester's free-code scope `tst/agent.py:1195-1204`.
  - Serve labels to the Tester through an in-memory handle rather than relying on the mounted `labels.csv`. The mount is at `/workspace/tasks` (`paper/scripts/clients_t16_rerun.sh:109`; labels copied at `:145`).
  - Generator metadata is off the mount (E8a).
  - `scan_leakage.py` runs on SET outputs.
- **Done when:**
  - [ ] `open('/workspace/tasks/…/labels.csv')` fails in all three scopes;
  - [ ] a normal pandas command and the documented `groupby('id')` pattern still work;
  - [ ] the G0a smoke passes with the sandbox on.
- **If not merged by Fri 16:** `ROGUE_ONE_SANDBOX=off` is frozen and the least-privilege claim is dropped (ED1). Report `scan_leakage.py` counts only.
- **Needed by:** A05 (defines Full), A16; the fallback narrative.

#### X1 · `EXTRA_ARGS` and `MAX_MODEL_LEN` hook in `vllm_session.sh`
- [ ] **Owner:** Henrik · **Effort:** 0.1 d · **Due:** Tue 13 Oct
- **Change:**
  - `paper/scripts/vllm_session.sh:157-168` adds only `--enforce-eager` today (`:157-158`). Append `$EXTRA_ARGS` and `--max-model-len $MAX_MODEL_LEN` (default 131072).
  - `ROGUEONE_ALLOW_SECOND_SERVER` (`:123`), `SERVER_NAME`, `PORT` and `GPUS` already exist.
  - Install as `~/ro27_vllm_session.sh` on CAIR, so other routines that use `~/vllm_session.sh` are untouched.
- **Done when:**
  - [ ] Q4 serves with `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --language-model-only`;
  - [ ] with `EXTRA_ARGS` empty, the O120 command line equals t17's except for `--max-model-len 131072` (the model's default).
- **Needed by:** A04, A17, A18, A19, A20.

#### X2 · Replay tool
- [ ] **Owner:** Henrik · **Effort:** 0.25–0.5 d · **Due:** Tue 13 Oct (ED2 if late)
- **Change:** `scripts/agent2027/replay_tool.py`.
  - It reads E3 traces and sends the recorded prompts **single-turn**, with the same tool schemas and temperature 0.7, through the `clients.py` wrapper (`VLLMCompatibleAsyncOpenAI`, `:471`).
  - It records raw vs rescued tool calls (attempt 1 raw and the wrapper's final response), `attempts`, parse errors, `finish_reason`, usage and context errors.
  - Options (A18-T3): `--model`, `--port`, `--seed` (overwrites any recorded request `seed`), `--concurrency`, `--clip-max-tokens` (clips `max_tokens` to fit 131,072 and flags the turn `clipped`); reads prompts from, and writes replies to, a W&B artifact.
- **Done when:** 50 recorded Extractor prompts from the G0a smoke replay against O120 and produce a valid-call table.
- **Needed by:** A04 (check b), A18.

#### X3 · classicfe feature dump and fold seed
- [ ] **Owner:** Henrik · **Effort:** 0.25 d · **Due:** **Mon 12 Oct, 18:00** (the M5 CPU screen runs that night; the breakdown listed it in the Tue–Fri window)
- **Change:** in `paper/scripts/classicfe_arm.py`:
  - add `--fold-seed` (`build_folds`, `:119-135`, hard-codes `random_state=42`);
  - add `--dump-features DIR`, which writes each fold's generated OpenFE/AutoFeat features to CSV (today only scores are written, `:376-387`).
- **Done when:** the seed-42 kfold partition equals Rogue One's `deterministic_kfold_indices` (`rogueone/utils/train_test_splitter.py:199`) on a test task, for both kfold and stratified, and the per-fold feature CSVs exist.
- **Needed by:** A08 (screen), A22 (mechanical-FE control, including a Core-10 × {42, 4} OpenFE re-run).

#### X4 · LLM-FE spec files for SYN-P 7
- [ ] **Owner:** Henrik · **Effort:** 0.25 d · **Due:** Mon 19 Oct (GPU part from Tue 20)
- **Change:**
  - Write specs for `synp_p1`…`synp_p6b` from the same `context_and_goal` template, under `scripts/agent2027/llmfe_specs/synp_<id>/`.
  - Copy them into `~/llmfe/` on CAIR in the layout `paper/scripts/llmfe_launch.sh` expects; verify that layout first.
  - LLM-FE's code is **not** changed (Henrik's 2026-08-28 decision). Writing specs for a rival system is decision D4.
- **Done when:** a 1-task LLM-FE run completes and writes `samples_*.json`.
- **Needed by:** A08.

#### X5 · Token-logging proxy
- [ ] **Owner:** Henrik · **Effort:** 0.25 d · **Due:** Mon 19 Oct
- **Change:** `scripts/agent2027/token_proxy/`.
  - A containerised OpenAI-compatible pass-through on port 8090, in front of the vLLM server.
  - It logs prompt and completion tokens per client to W&B (group `A08`).
  - There are no edits to LLM-FE or CAAFE.
- **Done when:** token totals per client match vLLM's own request log on a 10-call test.
- **Needed by:** A08, A21 (iso-cost points), A23.

#### X6 · Comparator-output parser
- [ ] **Owner:** Henrik · **Effort:** 0.25 d · **Due:** Sat 24 Oct
- **Change:** `scripts/agent2027/analysis/comparator_parser.py`.
  - Turns LLM-FE `samples_*.json` and CAAFE `record['generated_code']` (`paper/scripts/caafe_arm.py:398`) into per-fold feature tables for the recovery detector.
  - LLM-FE is parsed under three selections: `notebook` (primary), `best3`, and "any scored program" (`paper/scripts/llmfe_score_rogueone.py --selection`, `:384`).
- **Done when:** the detector scores one LLM-FE run and one CAAFE run under every selection.
- **Needed by:** A08 (C1b), A16 (C1b reading).

#### X7 · F0 generator
- [ ] **Owner:** Henrik · **Effort:** 0.25 d · **Due:** Fri 23 Oct (ED3: drop F0-best-of-k if it is not running by Wed 28 Oct)
- **Change:** `scripts/agent2027/f0_generate.py`.
  - A single-shot JSON list of ≤ 10 `{name, pandas_command, argumentation}` with no tools.
  - A pool of 128 proposals per fold, with batched decoding.
  - The prompt text is frozen in the pre-registration.
  - It logs tokens per proposal.
- **Done when:** one rung × one task × 5 folds × 128 pool is generated and logged as a W&B artifact.
- **Needed by:** A17, A23.

#### X8 · Remove and rotate the hard-coded webhook key
- [ ] **Owner:** Henrik · **Effort:** 0.1 d · **Due:** before any release; scheduled Nov 1–12, before the anonymised replication package on Thu 12 Nov
- **Change:** `main.py:1606` holds a Google Chat webhook URL with an embedded key and token. Remove it, read it from the environment if it is still needed, and rotate the key.
- **Done when:** `grep -n "chat.googleapis" -r .` is empty on the release branch and the old key is revoked.
- **Needed by:** the replication package (all documents).

#### X9 · P7 guessability and retrieval checks
- [ ] **Owner:** Henrik · **Effort:** 0.25 d · **Due:** Thu 15 Oct
- **Change:**
  - About 20 single-shot gpt-oss-120b calls with P7's `context_and_goal` and column descriptions, run as a container on the M4 server. **Zero** hits on φ are required.
  - Replay the search queries logged in M4's P7 runs offline through the TF-IDF index over `kt3_correct`. The correct document must rank near the top. If it does not, adjust the corpus vocabulary and disclose that.
- **Done when:** both results are in the pre-registration. Any guessability hit means redesigning P7 before the freeze.
- **Needed by:** A14.

#### X10 · Pre-registration commit
- [ ] **Owner:** Henrik · **Effort:** 0.5 d · **Due:** Fri 16 Oct (G1)
- **Change:** `docs/agent2027/preregistration.md` on branch `agent2027/v2`, tagged `agent2027-prereg`. Contents are listed in §2.2. It doubles as the draft protocol section.
- **Done when:** the tag is pushed before the first M0 run starts.
- **Needed by:** all documents.

---

## 4. Common protocol

### 4.1 Instruments

**Core-10 real panel.** These are no-harm and external-validity checks only; no causal claim is made on real data. The H1/H2 headroom slots are empty: "not screened". Values are from master plan §2.2 (*measured*):

| dataset_id | Task | n | p | Stratum | t17 seed SD | Used for |
|---|---|---|---|---|---|---|
| `balance-scale` | C | 625 | 4 | headroom (at ceiling) | 0.000 | M0, M1, KT1-Off, RT2 Core-10; not in W0 (always 1.000) |
| `junglechess` | C | 44,819 | 6 | headroom | 0.070 | + CAP real-4; rule found ≥ 0.95 |
| `vehicle` | C | 846 | 18 | headroom | 0.029 | + CAP real-4; HC1 high-error (274) |
| `heart` | C | 918 | 11 | no-harm | 0.019 | + CAP real-4; HC1 low-error (66) |
| `credit-g` | C | 1,000 | 20 | no-harm | 0.0055 | — |
| `myocardial` | C | 686 | 91 | no-harm | 0.0076 | + CAP real-4; HC1 high-error (595); only wide table |
| `housing` | R | 20,640 | 9 | headroom | 0.021 | HC1 high-error (196) |
| `insurance` | R | 1,338 | 6 | no-harm | 0.063 | HC1 low-error (85) |
| `crab` | R | 3,893 | 8 | headroom | 0.024 | HC1 high-error (168) |
| `wine` | R | 6,497 | 12 | headroom | 0.0010 | optional in W0 |

- HC1 error counts are t17 seed-42 `other_tool_errors` (`paper/t17_extractor_corruption_audit.csv`).
- **CAP real-4** = junglechess, vehicle, heart, myocardial.
- **Pilot only (non-core):** `blood` (C, p = 4) and `communities` (C, p = 102), for the RT2 crash-only pilot.

**SYN-P planted-feature tasks.** Built by E8a: n = 2,000, p ≈ 20–30, headroom ≥ 0.06 balanced accuracy or ≥ 15% nRMSE.

| dataset_id | Planted φ | Family / depth | Task type (proposed; E8a confirms by Mon 12) | Notes |
|---|---|---|---|---|
| `synp_p0` | log x_a | none (trees are invariant to it) | R | Negative control on held-out score only. Recovery = 0 by construction, because the detector excludes monotone transforms of one column. Runs once: M0, seed 42; also in the M5 CPU screen. |
| `synp_p1` | x_a / x_b | algebraic ratio, D1 | R | Expected to be flagged non-discriminating by M5 (run, reported as a sanity check) |
| `synp_p2` | (x_a / x_b) · x_c | ratio × product, D2 | R | Middle rung of the depth ladder; its moderator level is fixed by D3 |
| `synp_p3` | 1[x_a / x_b ≥ t] ∧ (cat = k) | threshold-AND of a ratio, D3 | C | multi-step |
| `synp_p4` | sin/cos of an hour-like column | cyclic / temporal | C | operator breadth |
| `synp_p5` | sign(LW·LD² − RW·RD²) on real balance-scale rows with real names | counterfactual rule | C | multi-step; the textbook prior LW·LD is wrong |
| `synp_p6a` | well-known named composite (BMI-like but passing the pairwise screen) | knowledge vs search | R | named twin |
| `synp_p6b` | the same φ on the same rows; f1..fp names, c1..ck levels, neutral context, opaque name, optional hidden rescaling | knowledge vs search | R (= P6a) | exact anonymised twin |
| `synp_p7` | fictional documented composite (ratio × threshold with an unguessable constant), documented only in the local corpus | knowledge-only | C | positive control for the knowledge channel |

- **SYN-P 8** = P1, P2, P3, P4, P5, P6a, P6b, P7.
- **SYN-P 7** = P1–P6b (no P7).
- **Multi-step moderator:** P3, P5. **Single-formula:** P1, P4, P6a, P6b. P2 is placed by D3.
- Under this proposal, about 4 of the 9 tasks are classification. CAAFE covers only the classification tasks of SYN-P 7: P3, P4, P5.

**P7 policy** (pre-registered default):

| Instrument set | Used by |
|---|---|
| **SYN-P 8** (with P7) | A05 M0 (web reference and leak canary), A02 M4 (all three R arms), A08 M5 **CPU screen** (plus P0), A09 RT3 **oracle** arm (KT3's ceiling), A14 KT3 (P7 only) |
| **SYN-P 7** (P1–P6b) | every other arm: A03 add-on, A08 GPU comparators, A09 misdirected, A10, A11, A12, A16, A17, A19, A20 |
| `synp_p0` | A05 (seed 42, one run) and the A08 CPU screen only |

If M0 recovers P7 in any fold-episode, that is the **leak canary**: investigate before A14 runs. P7 is not re-added to other arms. The policy saves about 31 runs (20–34u).

**Brute-force gate (from A08's CPU screen).** A family is **non-discriminating** if OpenFE or AutoFeat, each judged separately, recovers φ in ≥ 14 of 15 fold-episodes (5 folds × seeds {42, 4, 5}). It stays in the runs as a sanity check but is excluded from every confirmatory recovery population. The screen result is read on **Tue 13 Oct morning, before M4 launches**. If half or more of the families are solved, recalibrate SYN-P before any GPU run uses it.

### 4.2 Seeds

- **Fold seeds:** {42, 4} by default.
  - Seed 5 is used for the M5 CPU screen, the KT3 seed sets, and the optional P6 seed-5 pair (AB2).
  - **KT3 6-seed set** (Off, Local-correct) = {42, 4, 5, 1, 2, 3}; **3-seed set** (Parametric, Local-wrong) = {42, 4, 5}.
- **LLM-seed rule** (`ROGUE_ONE_LLM_SEED`):
  - primary replicate (`replicate=0`): `llm_seed = fold_seed`, as in t17;
  - A/A replicate (`replicate=1`): `fold_seed + 1000`;
  - O120 sentinel number k: `5000 + k`;
  - small-rung sentinel number k: `6000 + k`.

  Finding 36 showed byte-reproducibility breaks under `PARALLEL=6` load, so independence never depends on server load.
- **Pairing:** every arm is paired with M0 by fold seed (same outer partition). The SYN-P A/A at seed 42 is Full's second seed-42 replicate for C1b and C4b, so it must never be cut.
- **LLM-FE** runs 2 replicates at its native seed-42 partition. Its generation step uses `StratifiedKFold(5, shuffle, 42)` and its code is not changed.

### 4.3 Fixed settings (Full-v2 and every arm unless it is the factor under study)

| Setting | Value | Flag / source |
|---|---|---|
| Iterations × folds | 10 × 5 | `ROGUE_ONE_NUM_ITERATIONS=10` (code default 3, `config.py:305`, so always set it), `ROGUE_ONE_K_FOLDS=5` |
| Selection | inner CV | `ROGUE_ONE_SELECTION_SIGNAL=inner_cv` |
| Rollback | off | `ROGUE_ONE_ENABLE_ROLLBACK=False` |
| Temperature | 0.7, all roles and the summariser | `LLM_TEMPERATURE`, `*_AGENT_TEMPERATURE` |
| Max tokens | E 32768, T 8192, S pinned at G1 | `EXTRACTOR_/TESTER_/SCIENTIST_AGENT_MAX_TOKENS` |
| Reasoning effort | not passed (popped at `clients.py:525`), as in t17 | `ROGUE_ONE_REASONING_EFFORT_PASSTHROUGH=0` |
| Turn and retry caps | E 100 turns, T 100, S 200 turns; S up to 100 retries (`sci/agent.py:519`); T 20 attempts (`tst/agent.py:1373`) | unchanged |
| Extractor rounds R | as frozen at G1 (A02) | `ROGUE_ONE_EXTRACTOR_ROUNDS` |
| Feature pool | as frozen at G1 (A03) | `ROGUE_ONE_FEATURE_POOL_MODE` |
| Evaluator | XGBoost 400 trees (`tst/agent.py:1673` R, `:1815` C); the assessment/SHAP model uses 300 (`:927`, `:956`) | code |
| Balancing | on, explicit | `ROGUE_ONE_DO_BALANCING=True` |
| Folds | KFold by default; stratified on SYN-P classification tasks if D4 | `ROGUE_ONE_STRATIFIED_FOLDS` |
| Serving | `vllm/vllm-openai:v0.22.0`; in-container version asserted and image ID logged; `MAX_MODEL_LEN=131072` | launcher |
| Client image | `rogue_one:agent2027-v2.N`; its base stays `vllm/vllm-openai:v0.11.0` for the client Python environment (`Dockerfile:3`), as in t17 | build |
| Topology | O120 TP4 with 6 clients, or TP2 per D5; one Rogue One job at a time; ≤ 4 GPUs | launcher |
| Concurrency | `PARALLEL=6` per O120 server; `CPUS=12` per client; ≤ 12–14 clients in total on small-rung sessions | launcher |
| Web search | shared Serper cache, caller-attributed | E4 |
| Cap | **6 h wall clock per run**; a cap hit is an ITT failure and is not retried | launcher `RUN_CAP_HOURS=6` |
| Error feedback | `message` | E2 |
| Console | `CONSOLE_DISABLE=False` (after E2 it no longer changes tool returns); A01 alone keeps t17's `True` | env |
| Traces and prompts | `ROGUE_ONE_LLM_TRACE=1`, `ROGUE_ONE_PROMPT_DUMP=1` | E3, E5 |
| Logging | W&B entity `PlanBench` (§7 R12) | E1 |

### 4.4 No-leakage rule

1. **The held-out fold never reaches an agent.** The Scientist sees only `train_*` metrics and the Tester report (`main.py:384-415`; tool docstring `sci/agent.py:189-190`). Rollback stays off.
2. **Inner-CV scores and agents.** Any arm that feeds inner-CV scores to an agent computes them on an inner split disjoint from the one used for selection. In T1-minimum this applies only if D2 = L1.
3. **SYN-P metadata never reaches an agent.** φ, seeds, φ_wrong and oracle strings are never in a mounted folder. Only the RT3 oracle and misdirected arms get a focus file. The KT3 corpora are mounted only in `kb_local_*` arms.
4. **Labels.** The label barrier is enforced by E13 when it lands. Label-barrier violations are reported only as aggregate `paper/scripts/scan_leakage.py` counts; the detail belongs to the short paper.
5. **Inner-CV optimism** is computed per arm as a validity check only, and cited to the short paper.
6. **Pre-freeze pilots.** M4 and M1 results are seen before the pre-registration. This is disclosed. SYN-P is never retuned after M0 is seen.

### 4.5 Failures and ITT scoring

- A crash (non-zero rc, or no `fold_aggregation_summary.csv`) is retried **once, alone**, at the end of its session.
- **Persistent failures and 6 h cap hits are ITT failures.** They are never dropped, and the failure and cap-hit rates are reported next to every number.
  - **Held-out score:** imputed at `base_no_fe` if augment is canonical, otherwise at the majority-class or mean predictor. **Inside M1 both arms use the same imputation** (or failures are reported separately).
  - **Recovery** (made explicit here): a failed or capped run contributes all its fold-episodes with `rec_sel = rec_any = 0`.
- A pre-declared sensitivity analysis excludes runs whose selected features come only from the fallback.

### 4.6 Metric definitions (canonical names in §7 R13)

| Metric | Definition |
|---|---|
| **Recovery** `rec_any` / **recovered-and-selected** `rec_sel` (primary 1) | An active, non-fallback feature has \|Spearman\| ≥ 0.95 with φ on training rows, or NMI ≥ 0.9 for the rule and threshold families (P3, P5, P7; no credit for P7's ratio alone). Raw columns and single-column monotone transforms never count. `rec_any` = at any iteration of the fold; **`rec_sel` = present in the iteration that inner CV selects (primary)**. Unit: the fold-episode. Also `iter_first_rec` and `tok_to_first_rec`. Validated by two raters on a stratified sample of 60 detector hits and misses (Cohen's κ). |
| **Success / collapse** (primary 2) | `success` = the run finishes within the cap and scores ≥ `base_no_fe` − 2 × fold SD; on junglechess the rule is found (≥ 0.95). `collapse` = not success. **`func_success`** (C5a's outcome) = success **and** the inner-CV-selected iteration contains ≥ 1 accepted non-fallback LLM feature. It is needed because under augment a model that adds nothing still passes `success`. |
| **Cost** (primary 3) | Completion tokens per role, including reasoning and re-rolls, with search charged to the calling role (`tok_compl_*`); prompt tokens; LLM calls; wall clock (total and per agent); Serper calls; GPU energy (NVML, apportioned to runs by token share per server); FLOP proxy = 2 · N_active · tokens. |
| **Feature funnel** (primary 4) | proposed → executes → numeric, non-constant, non-duplicate (\|r\| < 0.999) → replay OK → survives pruning → in the selected iteration (`fun_*`). |
| Held-out score (no-harm column) | Outer-fold score at the inner-CV-selected iteration, averaged over 5 folds: balanced accuracy (C) or nRMSE (R); AUROC and log-loss secondary. Δ_std = Δ / σ_d. |
| Anytime AUC | Area under the held-out score of the inner-CV argmax over prefix 0..t, against iterations and against cumulative completion tokens |
| Append success / silent rejections | accepted ÷ attempts; NaN, non-numeric or duplicate rejections |
| Errors per accepted feature; repeat rate; recovery within 3 | tool errors ÷ accepted features (numerator and denominator also reported); the same exception repeated; a successful append within 3 calls after an error |
| Duplicate / re-proposal ratio | a new feature with \|ρ\| ≥ 0.99 on training rows, or an identical normalised AST, against the active and pruned pool; restorations of pruned features counted separately |
| Survival, prune rate, floor activations | from the per-iteration active lists |
| LLM-vs-rule pruning agreement | Jaccard between the LLM's prunes and J−a's (shadow mode or post hoc); descriptive only |
| Focus novelty / implementability | embedding cosine to earlier focuses (repeat if ≥ 0.9); share of focuses that lead to ≥ 1 accepted feature (κ-validated on 60 labels) |
| Report-read rate | Scientist tool calls that touch `feature_assessment` |
| Raw-column coverage; operator mix; AST complexity | coverage audit; operator shares |
| Fallback share; rescues per 100 calls (per family); context errors; cap hits | deterministic-fallback features and heuristic focuses; transport rescues; HTTP 400 and `finish_reason=length`; 6 h cap hits |
| Role-replay validity (RP1) | per role and rung: valid tool call (raw and rescued), executes, usable |
| Label-barrier violations | aggregate count from `scan_leakage.py` only |

### 4.7 Statistics conventions

**Confirmatory family.** Ten two-sided contrasts, Holm across all ten at family-wise α = 0.05, so the smallest threshold is 0.005.
- Write the raw p and "Holm-adjusted p < 0.05". Never write "Holm-adjusted p < 0.005".
- Every contrast is tested on SYN-P or on P7 only, never on real data.

| ID | Contrast | Supplied by | Instrument | Primary outcome and test |
|---|---|---|---|---|
| C1a | SET-phased − Full | A16 RT2 | SYN-P 7 × {42, 4} | `rec_sel`; GLMM `rec ~ arm + (1\|task) + (1\|task:seed)` |
| C1b | LLM-FE − Full (CAAFE − Full secondary) | A08 M5 | SYN-P 7, ≤ 6 discriminating | `rec_sel`; notebook selection primary; a direction is claimed only if notebook, best3 and any-program agree |
| C2a | Assessment (A) main effect | A12 FX | SYN-P 7 × 42 | `rec_sel`, **TOST** |
| C2b | Pruning (J) main effect | A12 FX | SYN-P 7 × 42 | `rec_sel`, **TOST** |
| C2c | Planner (P) main effect | A12 FX | SYN-P 7 × 42 | `rec_sel`, two-sided |
| C3 | Restarts − Full | A10 LF1 | SYN-P 7 × {42, 4} | `rec_sel`; population fixed by D3 (pooled, or the multi-step subset) |
| C4a | Local-correct − Off on P7 | A14 KT3 | `synp_p7`, 6 seeds each | `rec_sel`; exact two-sided test stratified by fold seed, episodes as units; run-level permutation as sensitivity |
| C4b | Anonymised (P6b) − named (P6a), search on | A15 KT2-min (data from A05) | `synp_p6a/p6b` × {42, 4, A/A} | `rec_sel`; exact McNemar |
| C5a | Trend in `func_success` over log active parameters (Qwen rungs only) | A17 CAP-1 | SYN-P 7 | Firth-penalised logistic or exact Cochran–Armitage stratified by task (test of record named in the pre-registration) |
| C5b | E-small − all-big | A19 CAP-2 | SYN-P 7 × {42, 4} | `rec_sel`; GLMM as C1a |

- **Drop rules.** A contrast is dropped only before any data on it exist.
  - If P6 fails the brute-force or pairwise screen (about Oct 14), C4b is dropped in the pre-registration, leaving 9 contrasts (smallest threshold 0.0056).
  - If SET is killed at K-SET (Fri 23 Oct, before its SYN-P runs), C1a is dropped.
  - If KT3 is killed at G4 (Tue 27), C4a has no data and G3 fails by default.
  - Holm is recomputed over the remaining contrasts.
- **SYN-P models.**
  - Main model: logistic GLMM at fold-episode level, `rec ~ arm + (1|task)`, plus `(1|task:seed)` where there are ≥ 2 seeds. There is no `family` term (aliased with task).
  - **FX:** run-level binomial counts with beta-binomial overdispersion, fitted on the **8 fraction cells only**: `rec ~ P + A + J + M + PA(=−JM) + PJ(=−AM) + PM(=−AJ) + (1|task)`. The Full anchor is not in the model.
  - **Why fold-episodes are the unit.** Per-fold state is reset by `_reset_fold_state` (`main.py:417-457`), so on one task the fold-episodes are conditionally independent agent episodes.
  - Where separation is likely (Off = 0/30), use the pre-registered exact tests.
- **Informative tasks.** A task is informative when Full's recovery is strictly between 0 and 1. This is used **for power accounting only, per contrast**, never as an exclusion rule: P7 is C4a's only task, the RT3 oracle can lift floor tasks, and misdirected and L0 can lower ceiling tasks. Exclusion from confirmatory populations comes only from the brute-force gate.
- **Power.** It is recomputed at G2 from M0's ICC and the number of informative tasks. Planning MDEs (unclustered, about 6 informative tasks):
  - FX main effect: 90% CI about ±0.10–0.12;
  - C1a: about 0.23–0.30;
  - RT3 misdirected vs Full: about 0.26;
  - C3 on 20 multi-step episodes: about 40–45 pp at α = 0.05, about 55 pp at 0.005;
  - C4a at 6 seeds: a gap of about 30–40 pp (18/30 vs 6/30, p = 0.0033);
  - C4b on 15 pairs: about 60 pp (≥ 9 discordant pairs, all in one direction).

  "Inconclusive" is a pre-declared outcome.
- **Real-data no-harm check** (pre-set non-inferiority):
  - Δ_std = Δ / σ_d, where σ_d is the dataset's Full-v2 run-to-run SD from M0. It is pooled with t17/W0 **only if the pre-registered homogeneity check passes**, and floored at 0.1 × the fold SD of `base_no_fe`.
  - Mixed model pooled across task types: dataset random intercept, task type as a fixed effect, dataset-specific residual variances.
  - One-sided at margin M. **M's formula is fixed in the pre-registration (Fri 16); its value is computed at G2 (Mon 19) before any arm's held-out score is read.** KT1-Off's held-out scores stay sealed until then.
  - "No harm beyond margin M" only if the CI excludes the margin; otherwise "inconclusive".
  - Per-task-type Wilcoxon is descriptive only.
- **Equivalence (TOST).** On SYN-P only, with 90% CIs. Margin = max(25% of the Full–oracle recovery gap from A09 + A05, a pre-registered absolute floor), committed **before FX recovery is unblinded**.
- **"Reducible" verdicts (C2a, C2b)** also need **≥ 30% fewer completion tokens** as a co-primary, reported with its CI. The rule is fixed at G2: if a Tester stage is < 30% of total tokens, measure against the replaced stage's own tokens, or use the joint A−J− vs A+J+ contrast.
- **Cost outcomes outside Holm** (direction near-certain by construction): KT1-Off tokens (A13), M4 u per run (A02), CAP-2 big-model token ratio (A19).
- **Other conventions:**
  - Friedman only with ≤ 5 arms; no Nemenyi; no MRR.
  - HC1: paired log-ratio mixed model with a dataset intercept, and a sign-flip permutation over the 12 pairs (a bootstrap over 6 datasets is too unstable).
  - Everything not listed above is exploratory.

**Decision rules owned by gates** (stated in full in the ablation documents):
- **M4 (A02):** adopt R=3 if it loses ≤ 4 of 40 episodes against R=10, no task falls from ≥ 3/5 to 0/5, the success rate does not fall, and u per run falls by ≥ 25%.
- **M1 (A03):** adopt augment if mean Δ_std ≥ −0.75, no dataset has Δ/(√2·σ̂_samp) < −2.5 (σ̂_samp from A01, or t17 seed SD as fallback), and SYN-P recovery does not fall (testable only with the add-on).

### 4.8 Budget-matching rules

1. **Primary cost axis:** completion tokens including reasoning, summed over roles, with search charged to the caller and re-rolls included. If token logging fails validation, use LLM-call count and wall clock, which are reliable only because arms are interleaved in one session.
2. **Iso-cost comparison for every arm** (M2, A21): Full is truncated to the tokens that arm actually spent, at **whole iterations only** (k ≥ 1). If an arm spends less than Full's first iteration, report "no iso-cost point". Pareto plots, not single points.
3. **F0-best-of-k** (A17): `k_matched` = F2's completion tokens per fold at that rung ÷ mean tokens per proposal. It is a true token-matched contrast only where `k_matched` ≤ 128. Otherwise plot the best-of-k curve (k = 1..128) and F2's anytime curve on one token axis and report the token ratio. Only "best-of-128 ≥ F2" is conclusive there.
4. **Hold fixed:** R, iterations (10), `max_tokens` per model, and the turn and retry caps, except where that parameter is the factor under study.
5. **Run hygiene:**
   - arms interleaved in random order within each session;
   - **one sentinel per session** (A06): Full-v2 on O120 sessions, Q4 × `synp_p2` × 42 on small-rung sessions; all Q122 cells run in one session, which then needs none;
   - constant concurrency;
   - the shared Serper cache.
6. **Pre-freeze points never mix with the frozen front.** M4 and M1 (pre-freeze code) get their own panel in A21 and A23 and are never used for iso-cost truncation of frozen Full.

### 4.9 Budget, capacity and levers

| AID | Short | New runs | u (low–high) | Earliest start |
|---|---|---|---|---|
| A01 | W0 | 16 | 16–20 (*outside window*) | Wed 7 Oct |
| A02 | M4 | 24 | 12–20 | Tue 13 Oct |
| A03 | M1 | 20 (+7 opt.) | 20 (+4.9–7) | Wed 14 Oct |
| A04 | CAP-0 | 8 fold-runs + 350 calls | 7–13 | Tue 13 Oct |
| A05 | M0 | 55 | 47.5–55 | Fri 16 Oct, 18:00 |
| A06 | M0-S | 9–17 | 5.7–17 | Sun 18 Oct (small rung) / Tue 20 Oct (O120) |
| A07 | HC1 | 12 | 12–15 | Fri 16 Oct |
| A08 | M5 | 22 GPU + 81 CPU cells | 9.9–15.4 | CPU Mon 12 / GPU Tue 20 Oct |
| A09 | RT3 | 15 | 10.5–15 | Mon 19 Oct |
| A10 | LF1 | 14 | 9.8–14 | Tue 20 Oct |
| A11 | LF2-L0 | 14 | 9.8–14 | Tue 20 Oct |
| A12 | FX | 56 | 39.2–56 | Wed 21 Oct, evening |
| A13 | KT1-Off | 10 | 9–11 | Wed 21 Oct |
| A14 | KT3-P7 | 18 | 12.6–18 | Wed 21 Oct, night |
| A15 | KT2-min | 0 (+2 opt.) | 0 (+1.4–2) | data from Fri 16 Oct |
| A16 | RT2 | 26 | 21.8–26 | Fri 23 Oct (SYN-P); Mon 26 Oct (Core-10) |
| A17 | CAP-1 | 87 + 49,280 one-shot | 53.0–97.1 | Sun 18 Oct, night |
| A18 | RP1 | 5,600 replays | 4–7 | Thu 22 Oct |
| A19 | CAP-2 | 36 | 27.6–54 | Thu 22 Oct |
| A20 | CAP-3 | 7 | 3.5–7 | Thu 22 Oct |
| A21 | M2 | 0 (CPU) | 0 | preliminary Mon 19 Oct; final Nov 1–3 |
| A22 | M3 | 0 (CPU) | 0 | after Mon 19 Oct; final Nov 1–3 |
| A23 | CAP-6 | 0 (CPU) | 0 | draft Tue 27 Oct; final Nov 1–3 |
| **Total in window** | | **≈ 425–433 full runs** | **314.9–474.5** | |

A18's own recomputation is 3.6–7.0u unpacked (2.6–4.8u packed onto idle GPUs), which lies inside the card's 4–7u used here; A04, A05, A06, A07 and the other rows match their documents' budget lines exactly. The recomputed in-window run count is 416 (A02–A20 excluding A06's sentinels and the A03 add-on) plus A06's 9–17 = 425–433 (A23 §4.4).

The A03 add-on is listed here at SYN-P 7 (7 runs, 4.9–7u), following the P7 policy. The breakdown card says SYN-P 8 (+5.6–8u); see Appendix A.

**Capacity.**
- The window runs Tue 13 Oct 00:00 – Sat 31 Oct 24:00: 456 h × 0.7 × 2 = 638u gross.
- Pre-freeze time is mostly idle by construction (only M4, M1 and CAP-0 A/B run on pre-freeze code: 37–49u).
- Post-freeze capacity is Fri 16 18:00 → Sat 31: **512u**. Usable ≈ **550–570u**.
- Post-freeze demand is 278–426u, or 319–490u with a 15% rerun reserve. **It fits at every point.**
- **The binding constraint is one developer's time, not GPU.** Re-tier at G1 from measured u: if the mean u per SYN-P run of M4's `R10` arm is > 0.85u, or R=3 is not adopted, apply CL1–CL4 at once (the re-tier rule; same wording in A02 §6, A07-T9 and §6.1).

**Cut levers** (pre-declared order; together they free about 37–59u):

| Lever | Cut | Saving |
|---|---|---|
| CL1 | CAP-2 on SYN-P only | −8 runs, −8 to −12u |
| CL2 | CAP-1 small-rung real data on junglechess and myocardial only | −10 runs, −5 to −10u |
| CL3 | HC1 on the 4 high-error datasets only | −4 runs, −4 to −5u |
| CL4 | Drop CAAFE, keep LLM-FE | −8 runs, −3.6 to −5.6u |
| CL5 | Sentinel in every second O120 session (floor of 6) | −3 to −5u |
| CL6 | Drop the F0 pools at Q122 and O120 | −3.5 to −7u |
| CL7 | Move LF2-L0 to T2 | −14 runs, −9.8 to −14u |

**Never cut:**
- M0's core runs and its SYN-P A/A;
- M1 on Core-10;
- the M5 CPU screen;
- M4's R=3 vs R=10;
- LF1 and RT3;
- KT3's Off and Local-correct arms;
- the 2-seed rungs Q0.8, Q4 and Q122.

**Add-backs if G1 measures u at the low end** (in order):

| ID | Add-back | Cost |
|---|---|---|
| AB1 | M1 SYN-P add-on | — |
| AB2 | P6 seed-5 pair | +1.4–2u |
| AB3 | RT3 misdirected on junglechess | +1u |
| AB4 | Q2 at 2 seeds | +3.7–7.5u |
| AB5 | VP1 paraphrased prompts (paraphrase set frozen by Oct 16) | +4.9–7u |

**Paid from the reserve, only if triggered:**

| ID | Item | Trigger | Cost |
|---|---|---|---|
| RV1 | CAP-5 fallbacks-off | a rung below 9B lands within margin of O120 on `rec_sel`; checked at G4 | 10–16u |
| RV2 | KT1-Off at seed 4 | adverse result on junglechess by about Oct 26 | 10u |
| RV3 | FX semi-foldover | at G4 | 28 runs, 19.6–28u |

---

## 5. CAIR run procedure, launcher and W&B naming

### 5.1 Hard rules (apply to every launch)

1. **Containers only.**
   - Every execution happens inside Docker: runs, analysis, downloads, import checks, W&B queries.
   - The CAIR host only issues `docker` / `docker compose` commands, plus the bash orchestration scripts in `~` that issue them. Queries like `nvidia-smi`, `docker ps` and `tail` are allowed for pre-flight and monitoring.
   - Never run `python …` on the host.
   - Any `docker compose run` over SSH needs `-T`.
2. **GPU pre-flight before every GPU container** (command in §5.2 step 1).
   - If there are not enough free GPUs, **stop and report** the per-GPU memory and utilisation, the compute apps and the running containers. Let Henrik decide.
   - Launch with **explicit `GPUS=<ids>`** taken from the pre-flight, and `WAIT_MAX_MIN=0`, so a session never queues silently. The legacy auto-wait in `vllm_session.sh:132-154` is not used.
3. **≤ 4 of the 8 H100s, ever, without Henrik's explicit approval in chat.**
   - One Rogue One job at a time.
   - Multi-server sessions go through `ladder_session.sh` (E7), which refuses ΣTP > 4.
   - Before E7 lands, sessions with several servers (CAP-0 A/B, CAP-1 small rungs) start one `ro27_vllm_session.sh` per rung with pinned `GPUS`, distinct `SERVER_NAME` and `PORT`, and `ROGUEONE_ALLOW_SECOND_SERVER=1`. This is done only after a pre-flight shows those GPUs are the only ones Rogue One holds.
4. **W&B is the system of record.**
   - Runs are stateless. Every run uploads its `output/` folder as artifact `run-output` (E3).
   - Local run folders on CAIR are secondary and root-owned.
   - Secrets:
     - `WANDB_API_KEY` is passed at run time from `<CAIR_HOME>/code/PlanBench/.env`, as `clients_t16_rerun.sh:59` does;
     - `SERPER_API_KEY` and LLM settings come from `.env` / `.secret.env` in `<ROGUEONE_CAIR_PATH>`, loaded at `config.py:23-25` and baked in at image build. **Never push these images.**
   - A run that cannot reach W&B is a setup fault to fix, not a reason to log locally.

### 5.2 Step-by-step procedure

**Paths and identities:**
- `<ROGUEONE_CAIR_PATH>` = `<CAIR_HOME>/code/RougeOne`. **This is inferred, not verified.** It comes from `REPO_TASKS="$HOME/code/RougeOne/tasks"` (`paper/scripts/clients_t16_rerun.sh:135`) and home `<CAIR_HOME>` (`:59`; `paper/scripts/cap_guard.sh:27`). Henrik should confirm it once with `ssh CAIR-gpu14 "ls -d ~/code/RougeOne"`.
- Task data live in `<ROGUEONE_CAIR_PATH>/tasks/` (untracked data).
- The existing launchers run from copies in `~` (`paper/scripts/t17_rerun_launch.sh:50-51`).
- Git remote for server fetch: `git@github.com:HenrikNorkart/RougeOne.git`. This is inferred from the local HTTPS origin; verify that the server key has access.

**Step 0 · Code to the server.**
- Implement locally, then commit and push branch `agent2027/v2`. During the pre-freeze phase the branch tip is tagged `agent2027-pre.N`.
- On CAIR (`<SESSION_ID>`, `<TAG>` etc. are placeholders):
```
ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15' git fetch git@github.com:HenrikNorkart/RougeOne.git agent2027/v2 && git checkout -B agent2027/v2 FETCH_HEAD && git log -1 --format='HEAD: %h %s'"
```

**Step 1 · GPU pre-flight** (read the output before doing anything else):
```
ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader && echo '--- compute apps ---' && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader && echo '--- containers ---' && docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'"
```
- **Proceed only if:**
  - the GPUs you need are free (memory used < 2,000 MiB, `vllm_session.sh` `FREE_MIB`);
  - the total is ≤ 4;
  - no `ro27-*`, `rogueone-vllm-*`, `llmfe-*` or `caafe-*` container of ours is running.
- Otherwise STOP and report.

**Step 2 · Build the image** (only when code changed; `<TAG>` = `agent2027-pre.N` or `agent2027-v2.N`):
```
ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && test ! -e .dev.env && docker build -t rogue_one:<TAG> . && docker image inspect rogue_one:<TAG> --format '{{.Id}}'"
```
Record the image ID in the session plan (`docs/agent2027/sessions/<SESSION_ID>.md`, R6). The `.dev.env` check exists because `config.py:24` loads it with `override=True`.

**Step 3 · Install launchers and manifests in `~`** (`paper/scripts/vllm_session.sh` is only on the server once `nested-cv-rebuttal` has been pushed and fetched, or is carried by `agent2027/v2`; see E1 and A01-T4):
```
ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && cp scripts/agent2027/ro27_session_launch.sh scripts/agent2027/ro27_clients_manifest.sh scripts/agent2027/ladder_session.sh ~/ && cp paper/scripts/vllm_session.sh ~/ro27_vllm_session.sh && mkdir -p ~/ro27_manifests && cp -r manifests/agent2027/. ~/ro27_manifests/"
```

**Step 4 · Launch detached** (long jobs use nohup; the log goes in `~`, never in the root-owned run folders):
```
ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "SESSION_ID=<SESSION_ID> GPUS=<ids> MODEL=<hf_id> IMAGE_RO=rogue_one:<TAG> PARALLEL=6 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_<SESSION_ID>.job.log 2>&1 & echo launched; sleep 5; tail -5 ~/ro27_<SESSION_ID>.job.log"
```
`ro27_session_launch.sh`:
- asserts the vLLM version in-container and logs both image IDs (as `t17_rerun_launch.sh:58-61` does);
- starts the energy sidecar;
- calls `~/ro27_vllm_session.sh ~/ro27_clients_manifest.sh`, which owns the server lifecycle and tears it down on exit;
- ends with the marker line `=== RO27_SESSION_END <SESSION_ID> rc=<rc> ===`.

**Step 5 · Monitor** (run as a background command; exit code 255 means the network dropped, not the job, so reconnect):
```
ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 CAIR-gpu14 "while ! grep -q 'RO27_SESSION_END' ~/ro27_<SESSION_ID>.job.log 2>/dev/null; do sleep 120; done; tail -30 ~/ro27_<SESSION_ID>.job.log"
```
Check W&B states (`running` / `finished` / `crashed`) in parallel. A `running` run with no live container is a zombie.

**Step 6 · Tear-down check:**
```
ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker ps --format '{{.Names}}' | grep -E '^(ro27-|rogueone-vllm-)' || echo none; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"
```

**Step 7 · W&B queries inside a container** (no host Python):
```
ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker run --rm --env-file <CAIR_HOME>/code/PlanBench/.env --entrypoint python3 rogue_one:<TAG> -c \"import wandb; api=wandb.Api(); [print(r.name, r.state) for r in api.runs('PlanBench/RogueOne-AGENT2027', filters={'group': '<AID>'})]\""
```

**Model downloads** (CAP-0) run inside a container into `~/.cache/huggingface` with pinned revisions. Check free disk first.

**CPU-only work on CAIR** also runs in containers, with a `--cpus` cap of ≤ 16 so the box keeps headroom:
- the M5 screen uses the classicfe image built from `docker/Dockerfile.classicfe` via `paper/scripts/classicfe_launch.sh`;
- analysis and scoring use `rogue_one:agent2027-v2.N`.

### 5.3 Manifest-driven launcher

**Files:**
- **Ablation manifests:** `manifests/agent2027/<AID>_<short>.csv`, generated by `scripts/agent2027/make_manifest.py --ablation <AID>` and committed. A manifest lists every planned run of that ablation.
- **Session files:** `manifests/agent2027/sessions/<SESSION_ID>.csv`, a subset of rows from one or more ablation manifests plus the sentinel row where A06 §4.5 requires one (analysed Full-v2 O120 and small-rung sessions; none for smoke, pilot, A18 replay-only, F0-only, A05's own and Q122-only sessions). They are created by `make_manifest.py --plan-session <SESSION_ID> --rows <row_ids…>`, which assigns `session_id` and a random `order_key` (RNG seeded from the session ID).

**Columns (CSV, header required):**

| Column | Meaning |
|---|---|
| `row_id` | `<AID>-<nnn>`, unique for the whole study |
| `ablation_id` | A01–A23 |
| `arm_id` | from §7 R5 |
| `dataset_id` | from §7 R10 |
| `task_type` | `classification` / `regression` |
| `fold_seed` | 42, 4, 5, 1, 2, 3 |
| `llm_seed` | per the §4.2 rule |
| `replicate` | 0 primary, 1 A/A, k sentinel number |
| `is_sentinel` | 0/1 |
| `session_id` | `S<MMDD><a-z>` (empty in ablation manifests) |
| `order_key` | integer; run order within the session |
| `image_tag` | e.g. `agent2027-v2.0` |
| `rung_S`, `rung_E`, `rung_T` | rung IDs from §7 R9 (all equal for uniform runs) |
| `port_S`, `port_E`, `port_T` | server ports from §7 R8 |
| `env_overrides` | JSON object of flag overrides relative to `v2_defaults.env`; `{}` for Full-v2 |
| `cap_hours` | 6 |
| `expected_u` | planning cost |
| `notes` | free text (for example "pilot", "smoke", "pair with A05-014") |

Run status is **not** stored in the manifest; it comes from W&B.

**What `ro27_clients_manifest.sh` does for each row**, in `order_key` order, with at most `PARALLEL` concurrent rows:
1. Create `~/rogueone-runs-agent2027/<SESSION_ID>/<row_id>/tasks/<task_type>/<dataset_id>/`, copying `config.yml`, `data.csv` and `labels.csv`. Every row gets its own copy, because outputs are written inside the task folder.
2. Build the env from `v2_defaults.env`, then `env_overrides`, then the identity keys (`ROGUE_ONE_ABLATION_ID`, `ROGUE_ONE_ARM_ID`, `ROGUE_ONE_REPLICATE`, `ROGUE_ONE_SESSION_ID`, `ROGUE_ONE_MANIFEST_ROW`, `ROGUE_ONE_IS_SENTINEL`, `ROGUE_ONE_CODE_PHASE`, `ROGUE_ONE_IMAGE_TAG`, `ROGUE_ONE_GIT_SHA`, `ROGUE_ONE_SERVER_TOPOLOGY`, `ROGUE_ONE_RUNG_S/E/T`), plus the dataset, seeds, per-role models and ports, the W&B project and the key.
3. `docker run -d --name ro27-<session_id>-<row_id> --network host --ipc=host --cpus 12` with the row's task mount at `/workspace/tasks`, the Serper cache at `/workspace/cache`, and, only for the arms that need them, the RT3 focus file or the KT3 corpus. Image `rogue_one:<image_tag>`.
4. Wait with a watchdog of `cap_hours`. On timeout, `docker stop`, log `CAP_HIT`, and do not retry.
5. Success = rc 0 **and** `fold_aggregation_summary.csv` exists. Otherwise log `FAILED` and queue the row for one solo retry at the end of the session.
6. Write one line per row to the job log: `row_id`, `rc`, wall clock, `CAP_HIT`/`FAILED`/`OK`.

### 5.4 W&B naming convention

- **Entity:** `PlanBench`.
- **Projects:**
  - `RogueOne-AGENT2027`: every run on a `agent2027-v2.N` image (A05–A20), the A08 comparator token logs, the session energy runs, and the analysis runs (group `ANALYSIS`).
  - `RogueOne-AGENT2027-prefreeze`: the G0a smoke, A02, A03, A04 sessions A/B, and the X9 checks.
  - **A01 only:** entity `brainflux`, project `RogueOne_W0_T17Rep_Seed<seed>`. It uses the unchanged t17 launcher, which hard-codes `WANDB_ENTITY=brainflux` (`clients_t16_rerun.sh:106`). The pair partners are in `brainflux/RogueOne_NestedCV_T17_Seed<seed>`. This is decision D9.
- **Run name:** `<AID>__<arm_id>__<dataset_id>__f<fold_seed>__r<replicate>`. Examples:
  - `A05__full__synp_p3__f42__r1` (M0 A/A);
  - `A12__fx_pajM__synp_p5__f42__r0`;
  - `A17__f2_Q4__vehicle__f42__r0`.
- **Group** = AID; **job_type** = `arm_id`; **tags** = [`<SESSION_ID>`, `<code_phase>`, `<image_tag>`].
- **Energy runs:** `<SESSION_ID>__energy` (job_type `energy`).
- **Analysis runs:** `ANALYSIS__<script>__<YYYYMMDD>`.
- **Required config keys** (written by E1; the full list is in §7 R12): the run identity, data and seeds, the full flag dump, models and topology, and version and phase.

---

## 6. Gate calendar, developer calendar and writing calendar

### 6.1 Gates

| Gate | Date | What must hold | Consequence if not |
|---|---|---|---|
| **G0a** | **Mon 12 Oct** | E1, E2 (+ R knob), E3, E5, E8a (+ detector, screens, P6b), E13 merged. One cheap smoke run (`ROGUE_ONE_SMOKE_FOLDS=1`) passes with non-zero token logs for all roles, traces and a complete `wandb.config`. X3 is done by 18:00, and the M5 CPU screen launches that night. | Pre-freeze pilots slip. If E13 is the only miss, apply ED1. |
| M5S (screen read) | Tue 13 Oct, morning | M5 CPU screen read: non-discriminating families labelled; < half of the families solved | Recalibrate SYN-P before M4 launches |
| **G-E6** | **Tue 13 Oct, end of day** | E6 merged; M1 and M4 decision rules committed before launch | M1 fallback: freeze augment by design; frozen replace × Core-10 × 42 vs M0 |
| (readout) | Thu 15 Oct | M4 and M1 read-outs; X9 done; decisions D2, D3, D4, D7 | — |
| **G1 / freeze** | **Fri 16 Oct** | R, pool mode and feedback signal frozen. `v2_defaults.env`, tags `agent2027-v2.0-freeze` and `agent2027-prereg` (dated) pushed. Re-tier from measured u (re-tier rule: apply CL1–CL4 at once if the mean u per SYN-P run of the `R10` arm is > 0.85u, or R=3 is not adopted). M0 + HC1 launch at 18:00. | No arm may launch |
| **G2** | **Mon 19 Oct** | **Hard go/no-go** (E11a): noise floor, ICC, NI-margin value, sentinel task, informative-task list, Tester token shares, CAP-2 cost threshold, B_d. Per-inner-fold scores are confirmed logged. | A defect caught here: fix, then rerun M0 + HC1 (about 60–70u; fits only now) |
| **G0b-1** | **Wed 21 Oct** | E7 and E9d/E9e (with E8b backend) merged as v2.3. All switches run in a 1-fold smoke. Full-v2's rendered prompts are byte-identical to `prompt-baseline-v2.0`. | The affected arms wait; FX after Oct 22 loses its G4 actions |
| **K-SET** | **Fri 23 Oct** | SET-phased build works (pilot passed) | ED4: C1a dropped; 19.8–24u freed |
| **G3** | **Mon 26 Oct** | Positive controls: RT3 (misdirected < Full, **or** oracle > Full with misdirected ≈ Full, the pre-registered alternative) and KT3 (Local-correct > Off on P7). **Title decided:** "Knowledge-Informed" stays only if Local-correct clearly beats both Off and Parametric. | Planner nulls are reported as "planner channel inert"; knowledge becomes process metrics only; the title word is dropped |
| **G4** | **Tue 27 Oct** | FX aliases reported as unattributable unless the reserve allows a semi-foldover (RV3). **KT3 kill date** (backend smoke must have passed). **RV1 (CAP-5) trigger check.** E11b done. | — |
| (latest starts) | Wed 28 Oct | KT3 latest start (night); F0 generator must be running (ED3) | — |
| **L-LAST** | **Thu 29 Oct** | Last new launch (KT1-Off latest start is that night) | — |
| **G6** | **Fri 30 Oct** | Knee read-out, descriptive only (the CAP-3 rung was already fixed at O20) | Fri 30 – Sat 31 are for reruns only |
| **DF** | **Sat 31 Oct, 24:00** | Data freeze: no GPU runs after this | — |

### 6.2 Developer calendar (one developer; weekends needed, decision D1)

| Day | Development | GPU / CPU |
|---|---|---|
| Tue 6 Oct | E2 (+ R knob, error flag) | — |
| Wed 7 | Launch W0 (A01; its own developer time is 6.5–7.5 h spread over Wed 7 – Mon 12, not 2 h, an over-commit against the E-task days, see 6.4 open question); E1a/E1b | **A01 starts** (old image, PARALLEL=6) |
| Thu 8 | E1c launcher; E3; **D1** | A01 |
| Fri 9 | E3; E5 | A01 |
| Sat 10 | E8a generator, detector, P7 corpus text (E8b part) | A01; downloads (A04) |
| Sun 11 | E8a screens and P6b twin; E13 | A01; downloads |
| Mon 12 | G0a smoke; X3 by 18:00; start E6; **D6, D8** | A01 ends; **M5 CPU screen at night** |
| Tue 13 | E6 merge (G-E6); X1; X2; commit M1/M4 rules | read the screen → **A02 M4** |
| Wed 14 | E4 | **A03 M1** (morning); A02 tail |
| Thu 15 | M4/M1 read-outs; X9; E8b corpus frozen; **D2, D3, D4, D7** | A04 sessions A/B when the box is free |
| Fri 16 | X10 pre-registration; **G1 freeze**; build v2.0; smoke | **A05 M0 + A07 HC1 from 18:00** |
| Sat 17 – Sun 18 | E11a; E9c tooling (σ map, focus extraction); **D5** before M0 | A05/A07; first small-rung night after M0 ends (A17, A06 small) |
| Mon 19 | **G2**; E9c hook → v2.1; E9a, E9b; X4, X5 | **A09 RT3** oracle (misdirected once M0 seed-42 SYN-P exists); A17 small rungs (night) |
| Tue 20 | v2.2 (E9a, E9b); E7 | **A10, A11**; A08 GPU part (Tue 20 – Sat 24); O120 sentinels start |
| Wed 21 | **G0b-1**: E9d, E9e, E8b backend, E7 → v2.3 | **A12 FX** (evening); **A13 KT1-Off**; **A14 KT3** (night); A04 session C |
| Thu 22 | E9f SET → v2.4; pilot at night | **A19 CAP-2**, **A20 CAP-3** + A17 O20, **A18 RP1** small rungs (packed onto idle GPUs) |
| Fri 23 | **K-SET**; X7 | **A16 RT2** SYN-P |
| Sat 24 – Sun 25 | X6; E11b | A17 Q122 session (Sat 24 – Wed 28); W2 overflow |
| Mon 26 | **G3**; Fig. 1 and title | A16 Core-10 (Oct 26–29) |
| Tue 27 | **G4**; E11b done | RV1/RV3 if triggered; A18 O120 self-replay; A17 O120 F0 pools |
| Wed 28 – Thu 29 | — | last launches (L-LAST Thu 29) |
| Fri 30 – Sat 31 | — | **G6**; reruns only; **DF Sat 31, 24:00** |

**GPU plan by week:**

| Week | Demand | Capacity at 0.7 |
|---|---|---|
| W0 (Tue 6 – Mon 12): W0 replicate, downloads, M5 CPU screen | 16–20u, outside the window | — |
| W1 (Tue 13 – Sun 18) | 97–127u | 202u, 126u of it pre-freeze and mostly unusable |
| W2 (Mon 19 – Sun 25) | about 155–235u | 235u |
| W3 (Mon 26 – Sat 31) | about 60–120u | 202u |

### 6.3 Writing calendar

| Date | Writing |
|---|---|
| Fri 16 Oct | The pre-registration doubles as the draft protocol section |
| Mon 26 Oct | Fig. 1 (the architecture as it actually runs, annotated with ablation knobs) and the title (G3) |
| Sun 1 – Tue 3 Nov | Final statistics; M2 (A21), CAP-6 (A23), RP1/F0 scoring (E11c, E11d); Table 2 |
| Wed 4 – Fri 6 Nov | Full draft of the AGENT paper (≤ 3 main-text figures) |
| Sat 7 – Mon 9 Nov | Short-paper draft |
| Tue 10 – Wed 11 Nov | Co-author review |
| Thu 12 Nov | Revisions; anonymised replication package (X8 done before it) |
| **Fri 13 Nov** | **Submit both** |

### 6.4 Decisions still needed from Henrik

| ID | Decision | Due | Recommended default |
|---|---|---|---|
| D1 | Work the weekends of Oct 10–11, 17–18 and 24–25, or pre-approve the engineering drop order (ED1 first) | Thu 8 Oct | — |
| D2 | Freeze item 12: the Scientist's feedback signal | Thu 15 Oct | Keep the in-sample training fit with a written justification; L1 code stays cut |
| D3 | C3 definition: pooled over about 6 tasks, or the arm effect within the multi-step subset (P3, P5); where P2 goes | Thu 15 Oct | — |
| D4 | C1b set-up: LLM-FE notebook primary with best3 and any-program as sensitivity analyses; `ROGUE_ONE_STRATIFIED_FOLDS=True` for SYN-P classification; authoring LLM-FE spec files | Thu 15 Oct | Yes to all three |
| D5 | O120 topology for all post-freeze O120 jobs: TP2 (matches CAP-2) or TP4 (disclose the mismatch) | Fri 16 Oct, before M0 | Measure TP2 throughput in W1 |
| D6 | M1 SYN-P add-on (AB1); accept the E6 fallback | Mon 12 Oct | Run the add-on on idle pre-freeze GPU time |
| D7 | Freeze item 14: expose `Status` in `list_known_attributes_tool`, or remove the instruction at `ext/agent.py:895` | Thu 15 Oct | Expose |
| D8 | KT2-min: the P6b exact-twin specification (option A); P6a's φ passes the pairwise screen | Mon 12 Oct (before E8a closes) | Option A, with hidden rescaling |
| D9 | W0's W&B entity (`brainflux`, as the launcher hard-codes, or `PlanBench`); confirm that `rogue_one:nestedcv9` still exists on CAIR | Wed 7 Oct | `brainflux` |

**Open question (box order Thu 22 – Wed 28 Oct; found by the cross-document check, raised in A12, A13, A16, A18, A19, A20).** §6.2 puts A19, A20 + A17 O20 slot, A18 small rungs and the A16 pilot on Thu 22, but the one-Rogue-One-job-at-a-time rule and ≤ 4 H100 do not allow it. Claims on the box, with the session IDs now fixed across documents:
- `S1021a` (A12 block 1 + A13 ± A14): Wed 21 ≈ 20:00 until Thu 22 evening – Fri 23 afternoon; A04 session C is `S1021m` (before the A12 smoke or after `S1021a`);
- A19 `S1022a` (whole box, 20–39 h), A20 `S1022c` smoke and `S1022d`/`S1022e` (O20 sessions, ports 8020/8021; A17's 11 `f2_O20` rows ride in them), A16 pilot `S1022b` (launch by Fri 23, 12:00), A18 packed replays `S1022p`–`s`/`S1023p` (ports 8030–8033 and 8022);
- `S1023a` (A12 block 2 + A16 SYN-P): Fri 23 until Sat 24 midday – Sun 25; A17 Q122 session (and A18 `S1024p`): Sat 24 – Wed 28; A16 Core-10 Mon 26 – Thu 29; A18 O120 `S1027p` and A17 O120 F0 pools Tue 27.
Not all of this fits; the order is Henrik's call at G2 (Mon 19 Oct). Proposal in A16 §7: A20 smoke → A16 pilot → the rest. Apply CL1/CL2/CL6 and ED3/ED4 in the order of §4.9 if the slots overrun L-LAST (Thu 29 Oct).

**Open question (developer time outside R2).** Per-document task sums include checks, CAIR operation and writing that the breakdown books elsewhere, but several documents also carry engineering that R2 and the 18–20.5 d total do not: A18's own code and install ≈ 0.6 d (Fri 16 – Wed 21 Oct, when Tue 20 – Wed 21 are already overloaded); A22's ≈ 0.65 d in Mon 12 – Mon 19 Oct (01 §6.2 schedules no A22 work before Nov 1, and its T10 0.4 d is assumed to sit inside E11b's 1.75–2 d); A23's 0.25 d energy apportionment on top of E11b; A10's extra 0.1 d. Also X7's 0.25 d may be too small for `f0_generate.py`, the F0 clients script and the unit test (if it overruns, ED3 applies), and E11d is booked only Nov 1–3 while the 100–150 core-hour F0 scoring estimate is unmeasured. Decide with D1 (Thu 8 Oct) which of these to absorb or cut (ED2, ED3, ED5).

**Open question (A01 developer time).** The breakdown and §6.2 book about 2 h of developer time for W0, A01 itself needs 6.5–7.5 h (T10 lands on Sat 10 Oct). Either accept the over-commit or drop the optional A01 reruns (A01-T15); decide with D1 (Thu 8 Oct).

**Open question (re-tier trigger, A02 §8 item 5).** The canonical rule above tests the `R10` arm's mean u even if R=3 is adopted. A02 asks whether to test the adopted arm's u instead. Default: keep the canonical rule; decide before A02-T3 (Tue 13 Oct, 10:00).

**Defaults adopted unless Henrik objects by Thu 15 Oct** (they go into the pre-registration):
- the P7 policy;
- FX as I = −PAJM, with the falsifier restated on the minimal harness;
- KT2-min (P6 twins) replacing SSP-3;
- KT3 at 6 seeds for Off and Local-correct;
- CAP-3 at O20;
- the sentinel rule;
- the corrected M1 and M4 rules;
- W0 on 8 datasets;
- the G3 title rule.

---

## 7. Canonical registry (names every ablation document must use verbatim)

### R0 · Conventions
- **Code references:** RougeOne repo `nested-cv-rebuttal` @ `07cbf93` (`C:/code/PHD/RougeOne`).
  - Short paths: `main.py` = `rogue_one_main.py`; `sci/agent.py`, `ext/agent.py`, `tst/agent.py` = `rogueone/llm/agents/core/{scientist,extractor,tester}/agent.py`; `kn/web_search.py` = `rogueone/llm/agents/core/knowledge/web_search.py`; `config.py` = `rogueone/utils/config.py`; `clients.py` = `rogueone/llm/clients.py`; `console.py` = `rogueone/utils/console.py`; `pandas_exec.py` = `rogueone/utils/pandas_exec.py`.
  - There is no `cfg.py`.
- **Units:** u = one real-data O120 run (10 iterations × 5 folds); SYN-P run 0.7–1.0u; small-model run 0.5–1.0u; throughput 2 u/h × 0.7 on ≤ 4 H100.
- **Owner** of every task: Henrik. Dates are 2026, weekday-correct (Tue 6 Oct is a Tuesday).
- **Role letters:** S = Scientist, E = Extractor, T = Tester, W = web-search summariser. **Full-v2** = the frozen reference (§2).
- Shared doc = `docs/ablations/01_shared_infrastructure.md`, cited as "01 §n".

### R1 · Ablations and document filenames

| AID | Short ID | File | Supplies | New runs | u | Earliest start |
|---|---|---|---|---|---|---|
| A01 | W0 | `A01_W0_old_image_replicate.md` | σ_samp (v1) for the M1 rule; variance split | 16 | 16–20 (outside window) | Wed 7 Oct |
| A02 | M4 | `A02_M4_extractor_rounds.md` | freezes R; cost outcome (outside Holm); C6 | 24 | 12–20 | Tue 13 Oct |
| A03 | M1 | `A03_M1_augment_vs_replace.md` | freezes the pool mode (gate) | 20 (+7 add-on) | 20 (+4.9–7) | Wed 14 Oct |
| A04 | CAP-0 | `A04_CAP0_stage0_smoke_tests.md` | rung pass/fail (= E12) | 8 fold-runs + 350 calls | 7–13 | Tue 13 Oct |
| A05 | M0 | `A05_M0_full_v2_reference.md` | Full-v2 reference for all; σ_d, ICC, margin, B_d; C4b data; RP1 traces | 55 | 47.5–55 | Fri 16 Oct, 18:00 |
| A06 | M0-S | `A06_M0S_session_sentinel.md` | session-drift guard | 9–17 | 5.7–17 | Sun 18 (small rung) / Tue 20 Oct (O120) |
| A07 | HC1 | `A07_HC1_tool_error_feedback.md` | C6 (outside Holm) | 12 | 12–15 | Fri 16 Oct |
| A08 | M5 | `A08_M5_bruteforce_screen_and_comparators.md` | brute-force gate; **C1b** | 22 GPU + 81 CPU cells | 9.9–15.4 | CPU Mon 12 / GPU Tue 20 Oct |
| A09 | RT3 | `A09_RT3_planner_oracle_misdirected.md` | planner half of G3; TOST ceiling | 15 | 10.5–15 | Mon 19 Oct |
| A10 | LF1 | `A10_LF1_restarts_vs_refinement.md` | **C3** | 14 | 9.8–14 | Tue 20 Oct |
| A11 | LF2-L0 | `A11_LF2L0_open_loop_planner.md` | exploratory C3 | 14 | 9.8–14 | Tue 20 Oct |
| A12 | FX | `A12_FX_component_half_fraction.md` | **C2a, C2b, C2c**; minimal harness at 120B | 56 | 39.2–56 | Wed 21 Oct, evening |
| A13 | KT1-Off | `A13_KT1_search_off.md` | C4 cost outcome (outside Holm) | 10 | 9–11 | Wed 21 Oct |
| A14 | KT3-P7 | `A14_KT3_knowledge_probe_P7.md` | **C4a**; knowledge half of G3 | 18 | 12.6–18 | Wed 21 Oct, night |
| A15 | KT2-min | `A15_KT2min_named_vs_anonymised.md` | **C4b** (from A05 data) | 0 (+2 opt.) | 0 (+1.4–2) | data from Fri 16 Oct |
| A16 | RT2 | `A16_RT2_set_phased.md` | **C1a** | 26 | 21.8–26 | Fri 23 Oct (SYN-P); Mon 26 Oct (Core-10) |
| A17 | CAP-1 | `A17_CAP1_scale_ladder.md` | **C5a**; F0-best-of-k | 87 + 49,280 one-shot | 53.0–97.1 | Sun 18 Oct, night |
| A18 | RP1 | `A18_RP1_rolewise_replay.md` | per-role floors (exploratory) | 5,600 replays | 4–7 | Thu 22 Oct |
| A19 | CAP-2 | `A19_CAP2_role_mixing.md` | **C5b**; 1B cell | 36 | 27.6–54 | Thu 22 Oct |
| A20 | CAP-3 | `A20_CAP3_harness_at_O20.md` | architecture × capability (exploratory) | 7 | 3.5–7 | Thu 22 Oct |
| A21 | M2 | `A21_M2_anytime_curve.md` | iso-cost points; anytime curves | 0 (CPU) | 0 | preliminary Mon 19; final Nov 1–3 |
| A22 | M3 | `A22_M3_evaluator_transfer.md` | evaluator transfer (validity) | 0 (CPU) | 0 | after Mon 19; final Nov 1–3 |
| A23 | CAP-6 | `A23_CAP6_cost_pareto.md` | cost Pareto front | 0 (CPU) | 0 | draft Tue 27 Oct; final Nov 1–3 |

Total in window ≈ 425–433 full runs, 314.9–474.5u. Usable capacity ≈ 550–570u; post-freeze 512u. Details, cut levers and reserve: 01 §4.9.


### R2 · Engineering and support tasks (details in 01 §3)

| ID | Title | Effort (d) | Due |
|---|---|---|---|
| E1 | Flag registry → wandb.config; manifest-driven task registration; manifest launcher | 0.75 | Mon 12 Oct (G0a) |
| E2 | v2 fixes (+ R knob) | 1.1 | Mon 12 Oct (G0a) |
| E3 | Instrumentation: tokens per role, LLM traces, energy, counters, focus log, run-output artifact | 1.25 | Mon 12 Oct (G0a) |
| E4 | Serper cache, URL logging, caller attribution | 0.5 | Fri 16 Oct (G1) |
| E5 | Prompt-hint neutralisation; rendered-prompt dump and diff | 0.5 | Mon 12 Oct (G0a) |
| E6 | Augment mode + replay-validity check | 1–1.5 | Tue 13 Oct, end of day (G-E6) |
| E7 | Web-search routing follows caller; `ladder_session.sh` | 1 | Wed 21 Oct (G0b-1) |
| E8a | SYN-P generator, context template, headroom calibration, minimal recovery detector, P7/P6 CPU screens, P6b twin | 1.35–1.75 | Mon 12 Oct (G0a) |
| E8b | KT3 probe corpus + TF-IDF local backend (reduced) | 1 | corpus Thu 15 Oct; backend Wed 21 Oct |
| E9a | Restart switch | 0.25 | Mon 19 Oct |
| E9b | L0 switch | 0.5 | Tue 20 Oct |
| E9c | Planner hook (oracle/misdirected) + RT3 tooling | 0.5–0.75 | Mon 19 Oct |
| E9d | FX switches (SE\|T digest, template report, J−a + shadow, amnesic) | 1.0 | Wed 21 Oct (G0b-1) |
| E9e | Knowledge-backend switch | 0.25 | Wed 21 Oct (G0b-1) |
| E9f | SET-phased build | 1–1.25 | Thu 22 Oct (kill K-SET Fri 23) |
| E11a | G2 statistics | 0.5 | Mon 19 Oct, 08:00 |
| E11b | Analysis package (rest) | 1.75–2 | Tue 27 Oct |
| E11c | RP1 scorer + fold-state reconstruction | 0–0.5 | Nov 1–6 (scorer needed by Sun 1 Nov) |
| E11d | F0 scorer | 0.5 | Nov 1–3 |
| E12 | Stage-0 smoke tests (run as A04) | 0.5 | A/B Thu 15 – Fri 16 Oct; C Wed 21 Oct (fallback: first box gap before A19 launches on Thu 22) |
| E13 | Sandbox (at risk) | 0.5 | Mon 12 Oct (G0a) |
| X1 | `EXTRA_ARGS` + `MAX_MODEL_LEN` hook in `vllm_session.sh` | 0.1 | Tue 13 Oct |
| X2 | Replay tool | 0.25–0.5 | Tue 13 Oct |
| X3 | classicfe feature dump + `--fold-seed` | 0.25 | Mon 12 Oct, 18:00 |
| X4 | LLM-FE spec files for SYN-P 7 | 0.25 | Mon 19 Oct |
| X5 | Token-logging proxy | 0.25 | Mon 19 Oct |
| X6 | Comparator-output parser | 0.25 | Sat 24 Oct |
| X7 | F0 generator | 0.25 | Fri 23 Oct (drop if not running by Wed 28) |
| X8 | Webhook-key removal + rotation (`main.py:1606`) | 0.1 | before release (Thu 12 Nov) |
| X9 | P7 guessability + retrieval checks | 0.25 | Thu 15 Oct |
| X10 | Pre-registration commit | 0.5 | Fri 16 Oct |

**CUT:**
- **E10:** tool-factory refactor, Solo-auto, S\|ET, Solo-native;
- **SSP-3 generator** and twins;
- **headroom screen** (H1/H2 "not screened");
- **L1 feedback-signal code**;
- fallbacks-off switch, unless RV1 fires;
- Q27 and Q35-A3.

**Engineering drop order:** ED1 E13 → ED2 X2 (RP1 shrinks) → ED3 X7/F0 → ED4 SET (C1a) → ED5 M3 → ED6 L0.

### R3 · Existing flags (verified) and their Full-v2 values (`scripts/agent2027/v2_defaults.env`)

| Flag | Read at | Full-v2 value |
|---|---|---|
| `ROGUE_ONE_NUM_ITERATIONS` | `config.py:305` | `10` (code default 3, so always set) |
| `ROGUE_ONE_K_FOLDS` | `config.py:309` | `5` |
| `ROGUE_ONE_FOLD_SEED` | `config.py:310` | per row |
| `ROGUE_ONE_LLM_SEED` | `config.py:330` | per row (R11 rule) |
| `ROGUE_ONE_SELECTION_SIGNAL` | `config.py:317` | `inner_cv` |
| `ROGUE_ONE_ENABLE_ROLLBACK` | `config.py:313` | `False` |
| `ROGUE_ONE_STRATIFIED_FOLDS` | `config.py:323` | `True` on SYN-P classification tasks if D4; else `False` |
| `ROGUE_ONE_DO_BALANCING` | `config.py:312` | `True` (explicit) |
| `ROGUE_ONE_FOLD_RETRY_LIMIT` | `config.py:311` | `1` |
| `ROGUE_ONE_DATASETS` | `config.py:335` | one `dataset_id` per container |
| `ROGUE_ONE_DO_LOGGING` | `config.py:308` | `True` |
| `ROGUE_ONE_START_VLLM` | `main.py:1600` | `False` |
| `ROGUE_ONE_DISABLE_WANDB` | `main.py:162` | `0` |
| `ROGUE_ONE_VERBOSE_AGENT_ERRORS` | `main.py:65` | `0` |
| `ROGUE_ONE_RETRIEVAL_TRANSCRIPT` | `rogueone/utils/retrieval_audit.py` | unset (default `<output>/retrieval_transcript.jsonl`) |
| `CONSOLE_DISABLE` | `config.py:300` | `False` (A01 only: `True`) |
| `LLM_MODEL`, `LLM_PORT`, `LLM_API_KEY` | `config.py:246-250` | rung HF id, server port, server key |
| `LLM_TEMPERATURE` | `config.py:247` | `0.7` |
| `ROGUE_ONE_SCIENTIST_MODEL` / `_PORT`, `ROGUE_ONE_EXTRACTOR_MODEL` / `_PORT`, `ROGUE_ONE_TESTER_MODEL` / `_PORT` | `config.py:224-260` | empty (uniform); set only in A19 and A04 session C |
| `SCIENTIST_AGENT_TEMPERATURE`, `EXTRACTOR_AGENT_TEMPERATURE`, `TESTER_AGENT_TEMPERATURE`, `MEDICAL_KNOWLEDGE_AGENT_TEMPERATURE` (summariser) | `config.py:371-409` | `0.7` |
| `EXTRACTOR_AGENT_MAX_TOKENS` | `config.py:378` | `32768` |
| `TESTER_AGENT_MAX_TOKENS` | `config.py:402` | `8192` |
| `SCIENTIST_AGENT_MAX_TOKENS` | `config.py:412` | pinned at G1 (applied by E2; proposed 16384; `v2_defaults.env` sets 16384) |
| `EXTRACTOR_AGENT_MAX_ITERATIONS` / `TESTER_…` / `SCIENTIST_…` | `config.py:373,401,411` | `100` / `100` / `200` |
| `*_AGENT_REASONING_EFFORT` | `config.py:372,400,410` | unset (popped at `clients.py:525`) |
| `WANDB_ENTITY` / `WANDB_PROJECT_NAME` / `WANDB_API_KEY` | `wandb_utils.py:110-115` | `PlanBench` / per R12 / from the PlanBench `.env` |
| `SERPER_API_KEY` | `kn/web_search.py:69` | from the image `.env` / `.secret.env` |

**`paper/scripts/vllm_session.sh` env:**
- `GPUS` (explicit ids), `NEED`, `PORT`, `API_KEY`, `MODEL`, `EAGER=0`, `LOG`;
- `IMAGE=vllm/vllm-openai:v0.22.0`;
- `SERVER_NAME`, `ROGUEONE_ALLOW_SECOND_SERVER` (multi-server only);
- `FREE_MIB=2000`, `WAIT_MAX_MIN=0`, `WAIT_POLL_S`;
- new from X1: `EXTRA_ARGS`, `MAX_MODEL_LEN=131072`.

**Legacy client launcher env** (A01 only, `clients_t16_rerun.sh`): `PARALLEL=6`, `CPUS=12`, `SEED`, `IMAGE_RO`, `PROJECT`, `RUNDIR`, `DATASETS`, `SOLO`, `REPO_TASKS`.

### R4 · New flags (one canonical name each; Full-v2 value in **bold**)

| Flag | Values | Task | Used by |
|---|---|---|---|
| `ROGUE_ONE_TOOL_ERROR_FEEDBACK` | **`message`** / `none` | E2 | A07 |
| `ROGUE_ONE_EXTRACTOR_ROUNDS` | `10` (code default) / `3` / `1`; **v2 = R frozen at G1 = `10`** | E2 | A02 |
| `ROGUE_ONE_TOOL_OUTPUT_MAX_CHARS` | **integer pinned at G1** (proposed 8000; `v2_defaults.env` sets 8000) | E2 | all |
| `ROGUE_ONE_REASONING_EFFORT_PASSTHROUGH` | **`0`** / `1` | E2 | T2 only |
| `ROGUE_ONE_LLM_TRACE` | **`1`** / `0` | E3 | A05→A18, all |
| `ROGUE_ONE_PROMPT_DUMP` | **`1`** / `0` | E5 | merge guards; A09–A16 diffs |
| `ROGUE_ONE_TASK_REGISTRY` | path to a YAML list of `config.yml` (unset = legacy list `main.py:1536-1566`) | E1 | all |
| `ROGUE_ONE_SMOKE_FOLDS` | **unset** / n (smoke and Stage-0 only) | E1 | G0a, A04 |
| `ROGUE_ONE_FEATURE_POOL_MODE` | `replace` / `augment`; **v2 = `augment`** (E6 pre-declared fallback, not the M1 rule) | E6 | A03 |
| `ROGUE_ONE_REPLAY_VALIDITY_CHECK` | **`1`** / `0` | E6 | all |
| `ROGUE_ONE_SEARCH_CACHE` | ~~`shared`~~ / **`off`** (code default; **v2 value since 2026-10-14**, live search) / `run` (in-process dedupe only) / `replay` (read-only, a miss never goes upstream) / `record` (always upstream, overwrite) | E4 | all search-on arms |
| `ROGUE_ONE_SEARCH_CACHE_PATH` | **`/workspace/cache/serper_cache.sqlite`** (host `~/rogueone-serper-cache/`) | E4 | all |
| `ROGUE_ONE_WEB_SEARCH_PROVIDER` | `serper` (code default) / **`searxng`** (HQ #266, 2026-10-13) | — | all |
| `ROGUE_ONE_SEARXNG_URL` | **`http://127.0.0.1:8888`** (container `ro27-searxng`, image `searxng:ro27-2026.10.11`) | — | all |
| `ROGUE_ONE_WEBSEARCH_ROUTING` | **`caller`** / `global` | E7 | A19, A04 session C |
| `ROGUE_ONE_KNOWLEDGE_BACKEND` | **`web`** / `off` / `parametric` / `local` | E9e (+ E8b) | A13, A14 |
| `ROGUE_ONE_LOCAL_CORPUS` | `kt3_correct` / `kt3_wrong` (only with `local`) | E8b | A14 |
| `ROGUE_ONE_TOPOLOGY` | **`three_agent`** / `set_phased` | E9f | A16 |
| `ROGUE_ONE_PLANNER_MODE` | **`llm`** / `none` (P−, SE\|T) / `oracle` / `misdirected` | E9c, E9d | A09, A12, A20 |
| `ROGUE_ONE_PLANNER_FOCUS_FILE` | path (oracle/misdirected only; mounted only for A09) | E9c | A09 |
| `ROGUE_ONE_MEMORY_MODE` | **`full`** / `amnesic` | E9d | A12 |
| `ROGUE_ONE_ASSESSMENT_MODE` | **`llm`** / `template` | E9d | A12, A20 |
| `ROGUE_ONE_PRUNING_MODE` | **`llm`** / `rules` (J−a) | E9d | A12, A20 |
| `ROGUE_ONE_PRUNING_SHADOW` | **`0`** / `1` (FX J+ cells) | E9d | A12 |
| `ROGUE_ONE_RESTART_MODE` | **`off`** / `every_iteration` | E9a | A10 |
| `ROGUE_ONE_FEEDBACK_LEVEL` | **`full`** / `l0` | E9b | A11 |
| `ROGUE_ONE_SANDBOX` | **`on`** if E13 lands (else `off`, frozen) | E13 | all |
| `ROGUE_ONE_FALLBACKS` | **`on`** / `off`; reserved, built only if RV1 fires | — | CAP-5 |
| `ROGUE_ONE_SCIENTIST_SIGNAL` | `train_fit` (code default) / **`inner_cv_l1`**; built because D2 = L1 | E14 | all (A11 `l0` removes it) |

**Per-role model routing** = the existing `ROGUE_ONE_<ROLE>_MODEL` / `_PORT` plus `ROGUE_ONE_WEBSEARCH_ROUTING`. There is no other routing flag.

**Run-identity keys** (set by the launcher, logged by E1):
- `ROGUE_ONE_ABLATION_ID`, `ROGUE_ONE_ARM_ID`, `ROGUE_ONE_REPLICATE`, `ROGUE_ONE_SESSION_ID`, `ROGUE_ONE_MANIFEST_ROW`, `ROGUE_ONE_IS_SENTINEL`;
- `ROGUE_ONE_CODE_PHASE` (`smoke` / `prefreeze` / `v2`);
- `ROGUE_ONE_IMAGE_TAG`, `ROGUE_ONE_GIT_SHA`, `ROGUE_ONE_SERVER_TOPOLOGY`;
- `ROGUE_ONE_RUNG_S`, `ROGUE_ONE_RUNG_E`, `ROGUE_ONE_RUNG_T`.

**Launcher-level keys:** `SESSION_ID`, `SESSION_FILE`, `IMAGE_RO`, `RUNROOT=~/rogueone-runs-agent2027`, `PARALLEL`, `CPUS=12`, `RUN_CAP_HOURS=6`.

**Do not use:** `ROGUE_ONE_SEARCH_ROLES`, `cfg.py`, or `ROGUE_ONE_EXTRA_TASKS` (it does not exist).

### R5 · Arm IDs (overrides relative to Full-v2)

- **A01:** `t17rep`. Image `rogue_one:nestedcv9`; legacy `paper/scripts/clients_t16_rerun.sh` and `vllm_session.sh` unchanged (`CONSOLE_DISABLE=True`, `STRATIFIED_FOLDS=False`, `LLM_SEED=FOLD_SEED`, `PARALLEL=6`, entity `brainflux`).
- **A02:** `R10`, `R3`, `R1` (`ROGUE_ONE_EXTRACTOR_ROUNDS=10/3/1`, `FEATURE_POOL_MODE=replace`, image `agent2027-pre.N`).
- **A03:** `replace`, `augment` (`FEATURE_POOL_MODE`; R=10; image `agent2027-pre.N`). The add-on is `augment` on SYN-P 7, paired with A02 `R10`.
- **A04:** `stage0_<rung>` (sessions A/B), `mixcheck_esmall`, `mixcheck_1b` (session C).
- **A05:** `full` (`replicate` 0 = primary, 1 = A/A).
- **A06:** `sentinel` (O120) and `sentinel_Q4` (small-rung sessions).
- **A07:** `errors_hidden` (`TOOL_ERROR_FEEDBACK=none`). Its control is A05 `full`.
- **A08:** `base_no_fe`, `openfe`, `autofeat` (CPU, classicfe image); `llmfe` (`replicate` 0/1); `caafe`.
- **A09:** `oracle` (`PLANNER_MODE=oracle` + focus file), `misdirected` (`PLANNER_MODE=misdirected` + focus file).
- **A10:** `restarts` (`RESTART_MODE=every_iteration`).
- **A11:** `l0` (`FEEDBACK_LEVEL=l0`).
- **A12** (I = −PAJM): `fx_pAJM`, `fx_PaJM`, `fx_PAjM`, `fx_PAJm`, `fx_pajM` (minimal harness), `fx_paJm`, `fx_pAjm`, `fx_Pajm`.
  - Uppercase = + (as in Full); lowercase = − (reduced).
  - P/p = `PLANNER_MODE` `llm`/`none`; A/a = `ASSESSMENT_MODE` `llm`/`template`; J/j = `PRUNING_MODE` `llm`/`rules`; M/m = `MEMORY_MODE` `full`/`amnesic`.
  - J+ cells set `PRUNING_SHADOW=1`.
  - The Full anchor is A05 `full` (`fx_PAJM` is never run).
- **A13:** `search_off` (`KNOWLEDGE_BACKEND=off`).
- **A14:** `kb_off` (`off`), `kb_parametric` (`parametric`), `kb_local_correct` (`local` + `LOCAL_CORPUS=kt3_correct`), `kb_local_wrong` (`local` + `kt3_wrong`).
- **A15:** no new arm. The optional AB2 rows are `full` on `synp_p6a`/`synp_p6b` with `fold_seed=5`.
- **A16:** `set_phased` (`TOPOLOGY=set_phased`); pilot rows use the same arm with `notes=pilot`.
- **A17:** `f2_<rung>` (all roles and W on the rung, e.g. `f2_Q0.8`, `f2_O20`); `f0pool_<rung>` (X7 pools, including `f0pool_O120`).
- **A18:** `replay_<rung>` (including `replay_O120`, the self-replay).
- **A19:** `mix_esmall` (S = O120, E = O20, T = O120); `mix_1b` (S = Q0.8, E = O120, T = Q0.8). W follows its caller.
- **A22:** `openfe` (CPU, classicfe image `classicfe:4`; rows A22-001…020 under `ablation_id=A22`, `replicate=0`; reuses A08's arm name, no new arm ID). A21 and A23 add no arms.
- **A20:** `fx_pajM_O20` (minimal harness at O20). The comparison cells are A17 `f2_O20`, A05 `full` and A12 `fx_pajM`.

### R6 · Scripts and paths

**Existing (reuse; do not rename):**
- `paper/scripts/vllm_session.sh`, `clients_t16_rerun.sh`, `t17_rerun_launch.sh`;
- `paper/scripts/classicfe_arm.py`, `classicfe_launch.sh`, `caafe_arm.py`, `clients_caafe.sh`, `llmfe_launch.sh`, `llmfe_score_rogueone.py`;
- `paper/scripts/scan_leakage.py`, `diag_selection_policy.py`, `diag_finding39_authoring_frame.py`;
- `docker/Dockerfile.{classicfe,caafe,llmfe,llmfe_eval}`;
- data: `paper/results_seed_sweep_t17.csv`, `paper/results_matched_harness_baselines.csv`, `paper/t17_extractor_corruption_audit.csv`, `paper/selection_policy_audit.csv`.

**New.** All of these live under `scripts/agent2027/` in the RougeOne repo unless a path says otherwise.

| Path | Task | Purpose |
|---|---|---|
| `make_manifest.py` | E1 | ablation manifests and session files |
| `ro27_session_launch.sh` | E1 | detached job wrapper: version asserts, energy sidecar → `ro27_vllm_session.sh` |
| `ro27_clients_manifest.sh` | E1 | clients script that runs one session file |
| `v2_defaults.env` | G1 | Full-v2 flag values |
| `ladder_session.sh` | E7 | multi-server sessions (ΣTP ≤ 4) |
| `energy_sampler.py` | E3 | NVML sidecar → W&B `<SESSION_ID>__energy` |
| `prompt_diff.py` | E5 | rendered-prompt diff |
| `synp/generate_synp.py` | E8a | SYN-P P0–P7 (+ P6b twin) |
| `synp/screen_pairwise.py` | E8a | pairwise-op screen |
| `corpora/kt3_correct/`, `corpora/kt3_wrong/` | E8b | KT3 corpora |
| `rogueone/llm/agents/core/knowledge/local_corpus.py` | E8b | TF-IDF backend (in the package, not under `scripts/agent2027/`) |
| `replay_tool.py` | X2 | single-turn replay |
| `token_proxy/` | X5 | token-logging proxy, port 8090 |
| `llmfe_specs/synp_<id>/` | X4 | LLM-FE specs |
| `f0_generate.py` | X7 | F0 pools |
| `f0_score.py` | E11d | F0 scoring |
| `analysis/recovery_detector.py` | E8a / E11b | recovery detector |
| `analysis/fetch_wandb.py` | E11b | pulls runs and artifacts |
| `analysis/g2_stats.py` | E11a | G2 statistics |
| `analysis/glmm_holm.py` | E11b | GLMM, exact tests, TOST, Holm |
| `analysis/funnel.py` | E11b | feature funnel |
| `analysis/process_metrics.py` | E11b | process metrics |
| `analysis/sentinel_drift.py` | E11b | sentinel drift test |
| `analysis/comparator_parser.py` | X6 | comparator outputs → detector input |
| `analysis/rp1_score.py` | E11c | RP1 scoring |
| `analysis/evaluator_transfer.py` | E11b | A22 |
| `analysis/pareto.py` | E11b | A23 |
| `paper/scripts/diag_selection_policy.py` (extended) | E11b | A21 anytime / iso-cost |

**Doc-local scripts (created by the named ablation document, registered here so that other documents cite them verbatim):**

| Path (under `scripts/agent2027/`) | Owner | Purpose |
|---|---|---|
| `rungs.csv` | A04 | per-rung `rung, hf_id, revision, tp, port, extra_args, stage0_verdict`; read by A17, A18, A19, A20 |
| `stage0_download.py`, `stage0_clients.sh` (installed `~/ro27_stage0_clients.sh`), `analysis/stage0_table.py` | A04 | Stage-0 downloads, clients, pass/fail table; `ro27_session_launch.sh` and `ladder_session.sh` gain a `CLIENTS_SCRIPT` passthrough (A04-T7) |
| `analysis/w0_sigma_samp.py` | A01 | W0 pair table and σ_samp |
| `analysis/a02_readout.py`, `analysis/a03_readout.py`, `e6_seed_check.py` | A02, A03 | pre-freeze read-outs; E6 seed check |
| `a08_comparators_clients.sh`, `a08_log_wandb.py`, `llmfe_specs/synp_<id>/` (X4) | A08 | comparator launch and token logging |
| `rt3_focus.py`, `rt3_sigma_map.yaml` | A09 | oracle/misdirected focus strings |
| `analysis/a15_leak_audit.py` | A15 | KT2-min leak audit |
| `f0_clients.sh` | A17 | F0 clients (with X7 `f0_generate.py`) |
| `rp1_clients.sh` (installed `~/ro27_rp1_clients.sh`) | A18 | replay clients (with X2 `replay_tool.py`) |

**Repo files:**
- manifests: `manifests/agent2027/<AID>_<short>.csv`, with `<short>` copied verbatim from R1 including hyphens (`A04_CAP-0.csv`, `A06_M0-S.csv`, `A11_LF2-L0.csv`, `A17_CAP-1.csv`, `A19_CAP-2.csv`, `A20_CAP-3.csv`);
- session files: `manifests/agent2027/sessions/<SESSION_ID>.csv`;
- pre-registration: `docs/agent2027/preregistration.md`.
- session plans (free-text pre-flight and check logs, one per session): `docs/agent2027/sessions/<SESSION_ID>.md`; A01's cross-session log is `docs/agent2027/sessions/a01_session_plan.md`.

**CAIR paths:**
- `<ROGUEONE_CAIR_PATH>` = `<CAIR_HOME>/code/RougeOne` (**INFERRED, verify**);
- tasks: `<ROGUEONE_CAIR_PATH>/tasks/`;
- SYN-P metadata: `~/rogueone-synp-meta/` (never mounted);
- run root: `~/rogueone-runs-agent2027/<SESSION_ID>/<row_id>/`;
- Serper cache: `~/rogueone-serper-cache/`;
- job log: `~/ro27_<SESSION_ID>.job.log`;
- vLLM log: `~/vllm_session_<SESSION_ID>.log`;
- installed launchers: `~/ro27_session_launch.sh`, `~/ro27_clients_manifest.sh`, `~/ladder_session.sh`, `~/ro27_vllm_session.sh`;
- manifests: `~/ro27_manifests/`;
- HF cache: `~/.cache/huggingface`;
- W&B key: `<CAIR_HOME>/code/PlanBench/.env`.

### R7 · Manifest format (CSV, header required)

`row_id, ablation_id, arm_id, dataset_id, task_type, fold_seed, llm_seed, replicate, is_sentinel, session_id, order_key, image_tag, rung_S, rung_E, rung_T, port_S, port_E, port_T, env_overrides, cap_hours, expected_u, notes`

- `row_id` = `<AID>-<nnn>`.
- `env_overrides` = a JSON object of R4 flags relative to `v2_defaults.env` (`{}` = Full-v2).
- `cap_hours` = 6.
- Status is never stored; it comes from W&B.

### R8 · Images, git, sessions, ports

- **Client images:**
  - `rogue_one:nestedcv9` (A01 only; verify it exists);
  - `rogue_one:agent2027-pre.N` (G0a smoke, A02, A03, A04 sessions A/B);
  - `rogue_one:agent2027-v2.0` (freeze, Fri 16; A05, A07);
  - `rogue_one:agent2027-v2.N`, N ≥ 1 (switch releases with Full byte-identical): v2.1 Mon 19 (E9c), v2.2 Tue 20 (E9a, E9b), v2.3 Wed 21 (E9d, E9e, E8b backend, E7), v2.4 Thu 22 (E9f).
- **Server image:** `vllm/vllm-openai:v0.22.0`. A01 uses the vLLM image ID that t17's job log recorded.
- **Git:**
  - branch `agent2027/v2` off `nested-cv-rebuttal@07cbf93`;
  - tags `agent2027-pre.N`, `agent2027-v2.0-freeze`, `agent2027-prereg`, `agent2027-v2.N`;
  - server fetch from `git@github.com:HenrikNorkart/RougeOne.git` (inferred).
- **Session ID:** `S<MMDD><a-z>`, e.g. `S1016a`.
  - Container: `ro27-<session_id>-<row_id>`, lower-case.
  - Server: `rogueone-vllm-<rung>-<session_id>`, lower-case, e.g. `rogueone-vllm-o120-s1016a`.
  - Energy sidecar: `ro27-<session_id>-energy`.
**Session-ID map** (fixed in the cross-document pass; letters not listed are free; `<x>` = assigned at planning, grep `docs/ablations/` first):

| Date | Session ID | Holder |
|---|---|---|
| Wed 7 / Thu 8 Oct | `S1007a` / `S1008a`; reruns `S1008b`, `S1009a` | A01 seeds 42 / 4; A01-T15 reruns |
| Mon 12 | `S1012a` | G0a smoke: A02-901/902 (and A07-901/902 if A07-T6 uses it) |
| Tue 13 | `S1013a` | A02 main (A07-901/902 optional) |
| Wed 14 | `S1014e` / `S1014b`; `S1014f`… | A03 E6 smoke / A03 Core-10 (A07-901/902 optional); A03 re-smokes |
| Wed 14 – Thu 15 | `S1014a`, `S1015h` | A02 contingency (late launch) |
| Thu 15 – Fri 16 | `S1015a`–`d`, `S1015e`–`f`, `S1015g` | A04 session A (4 servers), session B, TP4 retry |
| Fri 16 | `S1016s`; `S1016a`; `S1016b` | A05 freeze smoke; A05 + A07; TP2 sibling |
| Sun 18 | `S1018a`; `S1018b`–`e` | A05 infra re-queue; A17 N1 (Q4, Q0.8, Q2, Q9) |
| Mon 19 | `S1019a`; `S1019b`; `S1019c`–`f` | A09; A09 TP2 sibling; A17 N2 |
| Tue 20 | `S1020s`; `S1020a`; `S1020b` | A10/A11 smoke; A10 + A11 main; TP2 sibling |
| Wed 21 | `S1021b`; `S1021a`; `S1021c`/`d`; `S1021m` | A12 smoke (+ A13-901, A14-901…904); A12 block 1 + A13 + A14; A13 fallback; A04 session C |
| Thu 22 | `S1022a`; `S1022b`; `S1022c`; `S1022d`/`e`; `S1022p`–`s` | A19; A16 pilot; A20 smoke; A20 + A17 O20 (8020 / 8021); A18 packed replays |
| Fri 23 – Sun 25 | `S1023a`; `S1023p` | A12 block 2 + A16 SYN-P; A18 `replay_O20` (8022) |
| Sat 24 / Tue 27 | `S1024p`; `S1027p` | A18 `replay_Q122`; A18 `replay_O120` |
| later | `<x>` | A08 GPU session (Tue 20 – Sat 24), A17 Q122 and F0 sessions, A16 Core-10, A06-only reruns |

- **Ports:** O120 `8010` (second O120 server `8011`), O20 `8020`, Q0.8 `8030`, Q2 `8031`, Q4 `8032`, Q9 `8033`, Q122 `8040` (second Q122 server `8041`), second O20 server `8021` (A17/A20 `S1022e`), third O20 server `8022` (A18 packed `replay_O20`), token proxy `8090`.

### R9 · Model rungs

| Rung | HF id | Active / total | Precision | GPUs (TP) | Serve flags |
|---|---|---|---|---|---|
| `Q0.8` | `Qwen/Qwen3.5-0.8B` | 0.8B dense | BF16 | 1 (TP1) | Qwen set¹ |
| `Q2` | `Qwen/Qwen3.5-2B` | 2B | BF16 | 1 (TP1) | Qwen set |
| `Q4` | `Qwen/Qwen3.5-4B` | 4B | BF16 | 1 (TP1) | Qwen set |
| `Q9` | `Qwen/Qwen3.5-9B` | 9B | BF16 | 1 (TP1) | Qwen set |
| `Q122` | `Qwen/Qwen3.5-122B-A10B-FP8` | 10B / 122B (MoE) | FP8 | 2 (TP2); TP4 if KV is short | Qwen set |
| `O20` | `openai/gpt-oss-20b` | 3.6B / 21B (MoE) | MXFP4 | 1 (TP1) | `MAX_MODEL_LEN` only. **Failed SR4 at A04 (2026-10-15): CAP-2 uses Q122/Q4, CAP-3 moves to Q4, A17 drops `f2_O20`** |
| `O120` | `openai/gpt-oss-120b` | 5.1B / 117B (MoE) | MXFP4 | 4 (TP4, t17), or 2 (TP2) per D5; **D5 = TP2** | `MAX_MODEL_LEN` only (as t17) |

¹ Qwen set = `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --language-model-only --max-model-len 131072`. A per-family `--max-cudagraph-capture-size` may be added and recorded; the image is never changed. Qwen thinking is off (`clients.py:204`). HF revisions are pinned at A04. All Qwen3.5 ids are unverified until A04 passes.

- **Fallback ladder** (if Q9 fails Stage 0): `Q0.8`→`Qwen/Qwen3-0.6B`, `Q2`→`Qwen/Qwen3-1.7B`, `Q4`→`Qwen/Qwen3-4B`, `Q9`→`Qwen/Qwen3-8B`, `Q122`→`Qwen/Qwen3-32B`. Download 0.6B–8B now; 14B and 32B only if needed.
- **Cut:** `Q27` (`Qwen/Qwen3.5-27B`), `Q35-A3` (`Qwen/Qwen3.5-35B-A3B`).
- **Clients:** O120 runs 6 clients per server. Small-rung sessions run ≤ 12–14 clients in total (CAP-1 layout: 4 × TP1 with 3 clients each). 12 CPUs per client.
- **Unit costs:** small rungs and O20 0.5–1.0u; Q122 0.7–1.0u on SYN-P and 1u on real data.

### R10 · Dataset IDs

- **Banned:** `covtype`. **Hard rule: covtype is never run, in any study** (Henrik, 2026-10-16: "COVTYPE should never be run!"). This applies to every Rogue One study, not only AGENT 2027: no launcher, dataset list, manifest, session or retry may include it. The launchers in `paper/scripts/` and `scripts/agent2027/` refuse it and exit non-zero. Existing covtype results stay as historical records, but nothing may re-run it.
- **Core-10:** `balance-scale`, `junglechess`, `vehicle`, `heart`, `credit-g`, `myocardial` (C); `housing`, `insurance`, `crab`, `wine` (R).
- **CAP real-4:** `junglechess`, `vehicle`, `heart`, `myocardial`.
- **HC1-6:** `myocardial`, `vehicle`, `housing`, `crab` (high error load); `heart`, `insurance` (low).
- **W0-8:** Core-10 minus `balance-scale` and `wine`.
- **Pilot:** `blood`, `communities` (RT2 only).
- **SYN-P:** `synp_p0`, `synp_p1`, `synp_p2`, `synp_p3`, `synp_p4`, `synp_p5`, `synp_p6a`, `synp_p6b`, `synp_p7`.
  - Proposed types: C = p3, p4, p5, p7; R = p0, p1, p2, p6a, p6b. E8a confirms these in `synp_meta/registry.yaml`.
  - Folders: `tasks/<task_type>/synp_<id>/`.
- **Sets:** SYN-P 8 = p1, p2, p3, p4, p5, p6a, p6b, p7. SYN-P 7 = SYN-P 8 minus p7. Multi-step = p3, p5. Single-formula = p1, p4, p6a, p6b. p2 is placed by D3.
- **P7 policy:**
  - SYN-P 8 in A05, A02, the A08 CPU screen (plus p0) and the A09 oracle;
  - p7 only in A14;
  - SYN-P 7 everywhere else;
  - p0 only in A05 (seed 42) and the A08 CPU screen.
- **Brute-force label:** non-discriminating = OpenFE or AutoFeat recovers φ in ≥ 14/15 episodes (seeds {42, 4, 5}).

### R11 · Seeds

- **Fold seeds:** {42, 4}.
  - {42, 4, 5} for the A08 CPU screen and the KT3 3-seed arms.
  - KT3 6-seed set = {42, 4, 5, 1, 2, 3}.
  - AB2 uses 5.
- **`llm_seed`:**
  - `replicate=0` → `fold_seed`;
  - A/A (`replicate=1`) → `fold_seed + 1000`;
  - O120 sentinel k → `5000 + k`;
  - small-rung sentinel k → `6000 + k`.
- **LLM-FE:** 2 replicates at its native seed-42 partition.

### R12 · W&B

- **Entity:** `PlanBench`.
- **Projects:**
  - `RogueOne-AGENT2027` (all `agent2027-v2.N` runs, comparator token logs, energy runs, `ANALYSIS` group);
  - `RogueOne-AGENT2027-prefreeze` (smoke, A02, A03, A04 sessions A/B, X9);
  - A01 only: `brainflux/RogueOne_W0_T17Rep_Seed<seed>` (D9).
- **Run name:** `<AID>__<arm_id>__<dataset_id>__f<fold_seed>__r<replicate>`.
  - group = `<AID>`; job_type = `<arm_id>`; tags = [`<SESSION_ID>`, `<code_phase>`, `<image_tag>`].
  - Energy runs: `<SESSION_ID>__energy`.
  - Analysis runs: `ANALYSIS__<script>__<YYYYMMDD>`.
- **Required config keys:**
  - identity: `ablation_id`, `arm_id`, `dataset_id`, `task_type`, `fold_seed`, `llm_seed`, `replicate`, `is_sentinel`, `session_id`, `manifest_row`;
  - versions and phase: `code_phase`, `git_sha`, `image_tag`, `image_id`, `vllm_image_id`, `prereg_tag`;
  - models and topology: `rung_S`, `rung_E`, `rung_T`, `model_S`, `model_E`, `model_T`, `model_W`, `server_topology`, `parallel`;
  - **every** R3/R4 flag as `flags/<NAME>` with its effective (post-dotenv) value.
- **Artifact:** `run-output` (the whole `output/` folder) at run end.

### R13 · Metric names (W&B summary keys and analysis columns)

- **Recovery:** `rec_sel` (primary), `rec_any`, `iter_first_rec`, `tok_to_first_rec`.
- **Success:** `success`, `func_success`, `collapse`, `cap_hit`, `itt_imputed`.
- **Cost:**
  - `tok_compl_scientist`, `tok_compl_extractor`, `tok_compl_tester`, `tok_compl_total` (search included, charged to the caller);
  - `tok_search_<role>`, `tok_prompt_<role>`, `tok_prompt_total`;
  - `llm_calls_<role>`, `llm_calls_total`, `rerolls`;
  - `wall_s`, `wall_s_<role>`;
  - `serper_calls`, `serper_ok_false_rate`;
  - `energy_j`, `flop_proxy`, `max_prompt_tokens`, `context_errors`.
- **Funnel:** `fun_proposed`, `fun_executes`, `fun_valid`, `fun_replay_ok`, `fun_survives_prune`, `fun_selected`.
- **Score:** `heldout_score` (balanced accuracy or nRMSE), `heldout_auroc`, `heldout_logloss`, `delta_std`, `anytime_auc`.
- **Process:**
  - errors: `err_per_accepted`, `err_repeat_rate`, `recover_within3`, `silent_rejections`, `append_success`;
  - pool: `reproposal_ratio`, `restore_pruned_count`, `prune_rate`, `floor_activations`, `prune_jaccard`;
  - planner: `focus_novelty`, `focus_implementable`, `report_read_rate`, `lineage_share`;
  - coverage and harness: `raw_coverage`, `fallback_share`, `rescues_per100_<family>`, `label_barrier_violations`.
- **RP1:** `rp1_valid_call_raw`, `rp1_valid_call_rescued`, `rp1_executes`, `rp1_usable`.
- **Derived (G2):** `sigma_d`, `sigma_AA`, `sigma_samp`, `icc_recovery`, `ni_margin`, `B_d`.
- **Documented extensions** (introduced by one ablation document, valid analysis columns, not W&B summary keys unless stated): `rec_iter` (detector, per fold-iteration; A21), `flop_proxy_compl` (completion-only sensitivity of `flop_proxy`; A21, A23), per-role splits written `<metric>_<role>` (e.g. `fallback_share_<role>`, A19) and `rp1_<score>__<role>` with a double underscore (A18), `tok_compl_tester_assess` / `tok_compl_tester_prune` (A12), `replay_nonrepro_folds` and `replay_nan_fills` (E6, A03).

### R14 · Holm contrasts

| ID | Contrast | Supplied by | Outcome |
|---|---|---|---|
| C1a | SET-phased − Full | A16 | `rec_sel` |
| C1b | LLM-FE − Full (CAAFE secondary) | A08 | `rec_sel` |
| C2a | A main effect (TOST) | A12 | `rec_sel` |
| C2b | J main effect (TOST) | A12 | `rec_sel` |
| C2c | P main effect | A12 | `rec_sel` |
| C3 | Restarts − Full | A10 | `rec_sel` |
| C4a | Local-correct − Off on P7 | A14 | `rec_sel` |
| C4b | P6b − P6a (anonymised − named) | A15 (A05 data) | `rec_sel` |
| C5a | `func_success` trend over log active parameters (Qwen) | A17 | `func_success` |
| C5b | E-small − all-big | A19 | `rec_sel` |

- Family 10 → smallest threshold 0.005 (9 → 0.0056; 8 → 0.00625).
- Drop rules: C4b if P6 fails the screen (before the pre-registration); C1a at K-SET; C4a if KT3 is killed at G4.
- **Outside Holm:** cost outcomes for A02, A13 and A19 (big-model token ratio); A07; A11; A18; A20; A09 (positive control).

### R15 · Gates and dates

| Gate | Date |
|---|---|
| `G0a` | Mon 12 Oct |
| `M5S` (CPU screen read before M4) | Tue 13 Oct, morning |
| `G-E6` | Tue 13 Oct, end of day |
| `G1` (freeze + pre-registration; M0 + HC1 launch 18:00) | Fri 16 Oct |
| `G2` (hard go/no-go) | Mon 19 Oct |
| `G0b-1` | Wed 21 Oct |
| `K-SET` | Fri 23 Oct |
| `G3` (positive controls + title) | Mon 26 Oct |
| `G4` (FX aliases; KT3 kill; RV1 check) | Tue 27 Oct |
| `L-LAST` (last new launch) | Thu 29 Oct |
| `G6` (knee read-out) | Fri 30 Oct |
| `DF` (data freeze) | Sat 31 Oct, 24:00 |

- **Internal deadlines:** X3 Mon 12 at 18:00; X9 Thu 15; KT3 latest start Wed 28 (night); F0 drop rule Wed 28.
- **Writing:**
  - Fri 16 Oct: protocol draft;
  - Mon 26 Oct: Fig. 1 and title;
  - Nov 1–3: final statistics and Table 2;
  - Nov 4–6: AGENT draft;
  - Nov 7–9: short paper;
  - Nov 10–11: co-author review;
  - Nov 12: revisions and replication package;
  - **Fri 13 Nov: submit.**

### R16 · Levers and decisions

- **Cut levers:** CL1 CAP-2 SYN-P only · CL2 CAP-1 small-rung real data on 2 datasets · CL3 HC1 on 4 high-error datasets · CL4 drop CAAFE · CL5 sentinel every second O120 session (floor 6) · CL6 drop F0 pools at Q122/O120 · CL7 L0 → T2.
- **Add-backs:** AB1 M1 SYN-P add-on · AB2 P6 seed-5 pair · AB3 RT3 misdirected on junglechess · AB4 Q2 at 2 seeds · AB5 VP1.
- **Reserve-paid:** RV1 CAP-5 fallbacks-off · RV2 KT1-Off seed 4 · RV3 FX semi-foldover.
- **Engineering drops:** ED1 E13 · ED2 X2 · ED3 X7 · ED4 SET · ED5 M3 · ED6 L0.
- **Decisions:**

| ID | Decision | Due |
|---|---|---|
| D1 | Developer weekends / drop order | Thu 8 Oct |
| D2 | Scientist feedback signal (freeze item 12) | Thu 15 Oct |
| D3 | C3 definition and P2 placement | Thu 15 Oct |
| D4 | C1b set-up (notebook primary, stratified SYN-P C folds, LLM-FE specs) | Thu 15 Oct |
| D5 | O120 TP2 vs TP4 | Fri 16 Oct |
| D6 | M1 add-on and E6 fallback | Mon 12 Oct |
| D7 | Freeze item 14, pruned status | Thu 15 Oct |
| D8 | P6b twin spec | Mon 12 Oct |
| D9 | W0 W&B entity and `nestedcv9` image check | Wed 7 Oct |

**G1 status (2026-10-15).** D1–D12 are answered (HQ #244–#255): D2 = L1; D3 = multi-step subset {P3, P5, P2}; D4 adopted; D5 = TP2; D6 = E6 fallback accepted; D7 = expose `Status` (not yet implemented); D8 = Reynolds (option B); D9 = `brainflux`; D10 = adopted arm's u; D11 = Satterthwaite rule; D12 = parallel jobs on one server. Re-tier at G1: CL1–CL4 applied (R=3 not adopted; R10 mean u 1.285u). Binding text: `docs/agent2027/preregistration.md`.

---

## Appendix A. Code-verification notes and discrepancies found

All checks were read-only at `nested-cv-rebuttal@07cbf93` (the local `HEAD` of `C:/code/PHD/RougeOne`).

**Confirmed as stated in the master plan:**
- `main.py:384-415` (Scientist feedback = `train_*` metrics);
- `:417-457` (`_reset_fold_state`, which leaves `df_test_pool` untouched);
- `:543-544` (reset only at `iteration_in_fold == 0`);
- `:548-555` (`determine_focus`; `:553` passes the global `index`);
- `:1536-1566` (hard-coded 28 configs);
- `:1568-1580` (`select_paths`);
- `:1606` (webhook URL with an embedded key);
- `ext/agent.py:965` / `:967` / `:971` (one session; `range(10)` with no break; the same message each round);
- `ext/agent.py:895` vs `:570-596` (status instruction vs a names-only tool);
- `ext/agent.py:338` (ILLEGAL via `console_print`) and `:486-489` (duplicate rejection as a plain string);
- `sci/agent.py:64` (`KnowledgeAgent` created; its tool commented out at `:177`);
- `sci/agent.py:413-414`, `:430`, `:431`, `:438`, `:440`, `:443`, `:464`, `:472-475`, `:499`, `:519`;
- `tst/agent.py:119`, `:692`, `:1201`, `:1371`, `:1492`, `:1539`, `:1545-1558`, `:1617`, `:2142`;
- `console.py:31-41`;
- `pandas_exec.py:139-143`;
- `config.py:221` (`AGENT_ROLES`);
- `clients.py:37-80`, `:204`, `:499-514`, `:522-525`;
- `kn/web_search.py:82-83`, `:189`, `:225-229`, `:238`.

**Corrections and additions:**
1. **`Pruned Iteration`** is written at `tst/agent.py:1309-1310` and `:1607-1608` as `Trial {step}`, where `step = fold_run_id` (`main.py:792`). It records the **fold**; the fix is to write `trial_index`.
2. **`_report_tool_error`** is defined at `sci:115` (returns at `:131`), `ext:207` (`:223`) and `tst:1023` (`:1039`). The master plan's single-line refs point at the return lines.
3. **The Tester's output bound** at `tst/agent.py:1210` applies only to the log line; the returned output is unbounded (E2.5).
4. **The Scientist's `ModelSettings`** (`sci/agent.py:403-407`) never receives `max_tokens` (E2.6).
5. **Two more dataset-specific prompt hints:** `ext/agent.py:454` and `:901` (game helper names). Added to E5.
6. **`config.py:24` loads `.dev.env` with `override=True`.** A `.dev.env` baked into the image would silently override every `docker -e` flag. E1 logs effective values, and the build step refuses a context that contains `.dev.env`.
7. **`ROGUE_ONE_NUM_ITERATIONS` defaults to 3** (`config.py:305`, `config_keys.py:27`). It must always be set to 10.
8. **Search-token attribution.** Search tokens are logged under `agent_name="WebSearchAgent"` (`kn/web_search.py:191`), and Extractor searches are logged as `"ScientistAgent"` in the retrieval log (`ext/agent.py:270-272`). E3 and E4 charge both to the caller.
9. **Shared search helper.** The Extractor and Tester "knowledge agents" are both `WebSearchAgent` (`ext/agent.py:166`, `tst/agent.py:661`). There is no Chroma RAG in any active path, as the master plan states.
10. **`classicfe_arm.py` hard-codes `random_state=42`** in `build_folds` (`:119-135`) and writes only scores (`:376-387`). This confirms X3.
11. **No single-fold option exists** (`main.py:1420-1424`). Stage-0 and the G0a smoke need `ROGUE_ONE_SMOKE_FOLDS` (E1b).
12. **Image defaults.** `clients_t16_rerun.sh` defaults to `rogue_one:nestedcv8` (`:54`); t17 used `nestedcv9` through `t17_rerun_launch.sh:45`. A01 must set `IMAGE_RO=rogue_one:nestedcv9` explicitly. `WANDB_ENTITY=brainflux` is hard-coded at `clients_t16_rerun.sh:106`.
13. **Line-number offset.** `sci/agent.py:413-414` is correct for the L0 edits; the loop-harness audit's `:412-413` is off by one.
14. **Schedule conflicts in the breakdown:**
    - X3 (classicfe dump) is budgeted in the Tue 13–Fri 16 window but needed on Mon 12 evening.
    - E6 is due at the end of Tue 13 but budgeted in the same window, so it must start on the weekend or Mon 12.
    - The E8b corpus must be frozen in the Fri 16 pre-registration (and X9 on Thu 15 needs it), but all of E8b was budgeted after Oct 17.
15. **P7 policy vs the M1 card.** The M1 add-on card says "SYN-P 8"; the pre-registered P7 policy does not list M1 among the P7 arms. Canonical here: **SYN-P 7** (7 runs, 4.9–7u). Henrik may override.
16. **Worktree code is a different lineage.** The code inside this worktree (`.claude/worktrees/conference-search-b-c-03347a/`) has different line numbers (for example, Tester `include_report` at `:520` vs `:642`). Never cite it.
