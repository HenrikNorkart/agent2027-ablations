# A15 — KT2-min: Named vs anonymised twins P6a/P6b (C4b, from M0 data)

| Field | Value |
|---|---|
| **ID** | A15 · KT2-min |
| **Claim / Holm contrast** | C4, clause "knowledge helps only when semantics exist". Confirmatory **C4b** = P6b − P6a (anonymised − named), search on, outcome `rec_sel`, exact two-sided McNemar, Holm family of 10 (01 §7 R14) |
| **Tier** | T1-minimum (replaces the full KT2; SSP-3 is cut) |
| **Owner** | Henrik |
| **Start / finish** | D8 candidates screened and twin code Sun 11 Oct · D8 fixed and P6 folders generated Mon 12 Oct (G0a) · screen read Tue 13 Oct (M5S) · pre-freeze prompt-leak check Thu 15 Oct · pre-registration Fri 16 Oct (G1) · data from A05, Fri 16 18:00 – Sun 18 Oct · integrity at G2 Mon 19 Oct · preliminary at G3 Mon 26 Oct · final Tue 3 Nov · disclosures Thu 12 Nov |
| **Budget** | **0 new runs, 0u, 0 GPU-hours.** Reuses 6 A05 runs (4.2–6u, paid by A05). Optional AB2: +2 runs, +1.4–2u ≈ 1.0–1.4 h of the ≤ 4-H100 box (≈ 4–6 H100-hours). Contingency only if A02's P6b runs fail: one 1-fold smoke (`A15-901`, ≈ 0.15–0.2u of idle pre-freeze capacity). Developer time over A15-T1…T22: 1.95–2.1 d with AB2. Of this, 0.55–0.7 d is already budgeted inside E8a, X10 and E11b, and 0.3 d sits in the writing calendar, so ≈ 1.1 d is A15-specific (≈ 0.9 d without AB2) |
| **Models** | O120 (`openai/gpt-oss-120b`) for S, E, T and W, served by `vllm/vllm-openai:v0.22.0` (TP4, or TP2 per D5), 6 clients per server |
| **Status** | Not started |
| **Depends on** | D8; E1 (manifest launcher and `ROGUE_ONE_TASK_REGISTRY`); E3; E4; E5; E8a (generator, P6b twin, pairwise screen, detector); X3 (`--fold-seed` for the CPU screen); A08 CPU screen (M5S); X10; A05 (including its SYN-P A/A); E11a; E11b; A14-T2 (same `.dockerignore` edit) |
| **Feeds into** | C4b (Holm); C4 wording; Table 2 row "KT2-min"; G3 title discussion (supporting only) |

*Every "hoped" result is a hypothesis. Code references: RougeOne `nested-cv-rebuttal` @ `07cbf93` (`C:/code/PHD/RougeOne`), re-verified line by line on Wed 7 Oct 2026. The branch head is now `1bdaf42`, which changes only files under `paper/`; of the files cited here, only `paper/scripts/classicfe_arm.py` moves (its head line numbers are given in brackets). Short paths (01 §7 R0): `main.py` = `rogue_one_main.py`; `sci/agent.py`, `ext/agent.py`, `tst/agent.py` = `rogueone/llm/agents/core/{scientist,extractor,tester}/agent.py`; `kn/web_search.py` = `rogueone/llm/agents/core/knowledge/web_search.py`; `config.py` = `rogueone/utils/config.py`; `pandas_exec.py` = `rogueone/utils/pandas_exec.py`. "01 §n" = `docs/ablations/01_shared_infrastructure.md`.*

---

## 1. What we test and why it matters

P6a and P6b have the same 2,000 rows, the same planted formula φ and byte-identical labels. P6a has realistic column names, category levels and a domain task description. P6b has opaque names (`f01…`, `c01…`) and a neutral description. We ask whether Full-v2, with search on, recovers-and-selects a well-known named formula less often on P6b. Both twins already run inside M0 (A05), so C4b costs no GPU time. In T1-minimum it is the only test of "helps only when semantics exist"; without it, C4 rests on C4a (A14) alone.

## 2. Background you need

**Where task semantics reach the agents** (verified):

| Channel | Code | P6b treatment |
|---|---|---|
| `context_and_goal` (required key, `config.py:170`) | S prompt `sci/agent.py:449-450`; E prompt `ext/agent.py:923-924`; both T prompts `tst/agent.py:1473-1474`, `:1568-1569` | neutral text from the same template; character count within ±2% of P6a's |
| Column names | The E prompt lists every raw column, **sorted alphabetically** (`sorted(...)` at `ext/agent.py:859-861`, rendered at `:914-915`). S browses `df_raw_data` through its own `generic_pandas_tool` (`sci/agent.py:179-271`; scope `:245-250`, `exec` at `:252`). E browses `df` through `pandas_exec.py` | `fNN` names, numbered so that the sorted list has the same order in both twins (see "Instrument") |
| Level strings and value ranges | the same pandas tools | `cNN` levels (see "Instrument"); hidden positive per-column constants if φ is a monomial |
| `experiment_name` | Becomes `dataset_id` in Tester results (`tst/agent.py:1775`, `:1790`, `:2050`, `:2104`) and in skip/abort rows (`main.py:648`, `:673`, `:693`, `:767`). It reaches S through the feedback row (`main.py:384-415`, field `:402`) as a column of `df_test_results` (`sci/agent.py:188`) | `experiment_name` = `dataset_id` in **both** twins (`synp_p6a`, `synp_p6b`) |
| **Twin-invariant template text** (identical in both twins) | Agent-facing clinical framing that E5 does not neutralise: "such as a disease or a market trend" and "patient records" (`rogueone/llm/prompts/agent_roles.py:9`, `:15`); "patient attributes" and "clinical relevance" in the E tool schemas (`ext/agent.py:114`, `:126`, `:143`) and the tool docstring "Explain a patient attribute" (`:391`) | Unchanged. It primes **both** twins toward a clinical domain. Consequences: D8 prefers a non-clinical φ (A15-T1), and template-constant term hits are not leaks (A15-T7, T22) |
| Not agent-facing | `experiment_description`; `knowledge_db_collection_name` (S's Chroma `KnowledgeAgent` is built with a hard-coded collection at `sci/agent.py:64-66` and never queried; its tool is commented out at `:176-177`); the W&B config and run name (`rogueone/utils/wandb_utils.py:107-114`). `labels.csv` is read at `rogueone/utils/train_test_splitter.py:349` and joined on `id,target` (`main.py:269`) | set neutral anyway; `labels.csv` byte-identical |

**Knowledge channel (on in Full-v2).**
- The S prompt says "Always perform at least one web search" (`sci/agent.py:431`).
- All three roles have a search tool that calls `WebSearchAgent.explain_query` (`kn/web_search.py:292-309`):
  - S `search_tool` (`sci/agent.py:135-174`, call at `:149`);
  - E `search_in_literature_tool` (`ext/agent.py:246-261`, agent created at `:166`);
  - T `search_in_literature_tool` (`tst/agent.py:1041-1058`, agent created at `:661`).
- The summariser W (`_run_search_agent`, `kn/web_search.py:57`) has **no task context** in its prompt (`:200-236`). It sees only the caller's question and sends queries to Serper (`:128`).
- Transcript `<output>/retrieval_transcript.jsonl` (`rogueone/utils/retrieval_audit.py:49`):
  - one `event="query"` row per Serper attempt (`kn/web_search.py:96-167`). Fallback variants of the same tool call get their own rows with `variant_index` 1–3 (built at `:108-122`, logged in the loop at `:125-167`);
  - one `event="answer"` row per caller question (`:301-308`).
- E4 adds the caller role, URLs and a logged cache hit; E3 charges search tokens to the caller (`tok_search_<role>`).

**Why the twins pair exactly.**
- Outer folds come from `compute_entity_duplicate_groups` on the raw table plus `deterministic_kfold_indices` (`main.py:299-311`; `rogueone/utils/train_test_splitter.py:111`, `:199`).
- If no two rows are identical (`id` excluded), every row is its own group. The grouping is then dropped (`train_test_splitter.py:274-277`) and plain shuffled `KFold` runs (`:302`), so the partition depends only on n, k and the seed. Renaming and positive rescaling keep row equality, so the twins also share any duplicate structure. P6 is regression, so stratification does not apply.
- Fold state is reset at every fold (`_reset_fold_state`, `main.py:417-457`, called at `:543-544`). Fold-episodes are therefore conditionally independent and are the unit of analysis.

**Brute-force screen is name-blind, but not level-blind.**
- `classicfe_arm.py` renames columns to `f{i}` by position before OpenFE (`paper/scripts/classicfe_arm.py:206` [head `:214`]).
- It encodes string columns as ordinal codes in **alphabetical** order of the level strings (`:113-115`). This is why P6b's level labels must sort in the same order as P6a's (see "Instrument").
- With that rule, the P6a and P6b screens can differ only through rescaling.
- AutoFeat is unseeded (`:301` [head `:317`]; Finding 43, commit `1bdaf42`), so each AutoFeat cell is a single random draw.

**What a P6b container can reach** (this defines the leak checks):
- **Its own task folder only.** The launcher mounts a per-row copy at `/workspace/tasks` (01 §5.3 steps 1 and 3), so P6a's folder is not mounted. Any `tasks/` tree baked into the image is shadowed by that mount.
- **Everything baked into the image.** `Dockerfile:13` is `COPY . .`, and `.dockerignore` excludes neither `scripts/agent2027/`, `synp_meta/` nor `docs/`. The generator, the P6a LLM-FE spec (X4) and the pre-registration (which names φ) would all be baked in. A15-T9 closes this.
- **The shared Serper cache** at `/workspace/cache` (read-write). It holds P6a's queries and results.
- **The network** (`--network host`) and the container's `WANDB_API_KEY`.
- **How agents could get there.** Agent code runs through `exec` with full builtins in all three roles: E `pandas_exec.py:140-143`, S `sci/agent.py:245-252`, T `tst/agent.py:1190-1204`. No `__builtins__` key is set, so Python inserts the full builtins. Until E13 lands, `open()`, `sqlite3`, `os`, `wandb` and `requests` are reachable. A15-T17 and T22 audit for this.

**Instrument** (built by E8a; task type R is proposed and confirmed in `synp_meta/registry.yaml`; folders `tasks/regression/synp_p6a/` and `tasks/regression/synp_p6b/`):

| dataset_id | Definition |
|---|---|
| `synp_p6a` | n = 2,000, p ≈ 20–30. φ is a **well-known named composite** of 3 relevant columns, plus correlated decoys. It must pass `screen_pairwise.py`: no unary or pairwise op of the raw columns may reach \|ρ\| ≥ 0.95 with φ. A BMI-like w/h² with realistic ranges **fails**, because it is nearly rank-identical to w/h. It has realistic names and a domain `context_and_goal` that does **not** name φ or its synonyms (proposed; D8 confirms). It has a neutral integer `id`, and no two rows are identical (`id` excluded). |
| `synp_p6b` | The **same** rows, row order, `id` column, generator seed and byte-identical `labels.csv`. `data.csv` keeps P6a's column order. **Column names** are `f01…fNN`, numbered by the rank of the P6a name under Python `sorted()` and zero-padded to two digits. The sorted column list in the E prompt (`ext/agent.py:860`) then has every column at the same position in both twins. **Level strings** are `c01…cKK` per column, numbered by the rank of the P6a level string under `sorted()`. The alphabetical ordinal codes of the CPU screen (`classicfe_arm.py:113-115`) are then identical. P6b has a neutral `context_and_goal` from the same template. If φ is a monomial, **every** numeric column is multiplied by a hidden positive constant (no shifts). This hides value ranges and keeps every rank and every scale-equivariant feature. |

- **`config.yml` of both twins.** It holds `experiment_name` (= `dataset_id`), `experiment_description`, `knowledge_db_path`, `knowledge_db_collection_name`, `feature_file_path`, `label_file_path`, `context_and_goal` and `task_type: regression`.
  - The five keys from `knowledge_db_path` to `context_and_goal` are required (`config.py:160-170`).
  - `task_type` defaults to `classification` if it is missing (`config.py:175`), so it must be set.
  - Every value except `context_and_goal` is neutral and is built the same way in both twins.
- **The manipulation** is names + levels + description jointly. These cannot be separated, and the paper says so.
- **Off-mount metadata.** φ, the name map and the constants live only in `synp_meta/` (CAIR: `~/rogueone-synp-meta/`). They are never mounted into an agent container.
- **Naming rule versus 01.** The zero-padded, sorted-rank rule refines 01 E8a's "f1..fp / c1..ck (order kept)". It is needed because of the two `sorted()` sites above. Recorded as a pre-registration detail (A15-T11).

**Metrics** (unit: fold-episode).
- **`rec_sel`** (primary): an active, non-fallback feature in the inner-CV-selected iteration has |Spearman| ≥ 0.95 with φ on training rows. Raw columns and single-column monotone transforms never count.
  - The detector (`scripts/agent2027/analysis/recovery_detector.py`, E8a) replays features through `rogueone/utils/fold_safe_features.py:37`.
  - φ is computed from the off-mount metadata on the P6a values, aligned by `id`. On P6b, rescaling multiplies φ by a positive constant, so Spearman is unchanged.
- `rec_any` (any iteration) and `iter_first_rec`.
- **Paired difference.** Over the n pairs, n10 = pairs with P6a `rec_sel` = 1 and P6b = 0; n01 = pairs with P6a = 0 and P6b = 1. Δ = rate(P6b) − rate(P6a) = (n01 − n10)/n. Δ < 0 means names help.
- **Brute-force label:** a task is non-discriminating if OpenFE or AutoFeat recovers φ in ≥ 14/15 fold-episodes (seeds {42, 4, 5}).

## 3. Hypotheses and interpretation (all are hypotheses)

Thresholds are given for 15 pairs; with AB2 (20 pairs) they are in brackets.

| Outcome | Pattern | Meaning and reporting |
|---|---|---|
| **Hoped** | P6a ≥ 12/15 [16/20] and P6b ≤ 3/15 [4/20] on `rec_sel`; P6a recovers earlier; P6b's queries are generic (lower `query_domain_share`) | *"Hiding column names and the task description lowered recovery of a named formula from a% to b% on otherwise identical data (exact McNemar p = …, Holm-adjusted p < 0.05)."* The clause stays. |
| **Null** | No Holm-significant difference | *"At 120B, names did not change recovery of this formula."* Drop the clause; C4 rests on C4a. A true gap below about 60 pp [45 pp] reads as "inconclusive", never "no effect". No equivalence claim. If D8 chose a clinical φ, add that the clinical template (§2) may have let P6b guess the domain. |
| **Ceiling** | Both ≥ 14/15 [19/20] | Statistical search finds φ without names. Reported as the null. |
| **Floor** | Both ≤ 1/15 [1/20] | Inconclusive. Check A09's oracle on P6a/P6b. If it recovers φ, the bottleneck is planning or knowledge; otherwise it is execution. |
| **Opposite** | P6b > P6a, Holm-significant | Names mislead, for example by anchoring on a textbook variant. Diagnose from P6a near-misses (detector \|ρ\| 0.80–0.95) and the queries. Report "semantics can be a liability"; this weakens the title (G3). |

## 4. Design

**Control and arm.** No new arm. Both twins run Full-v2 (arm `full`, `env_overrides={}`) with search on.

**Fixed settings** (all from `scripts/agent2027/v2_defaults.env`, 01 §7 R3/R4; the ones that matter here):
- `ROGUE_ONE_NUM_ITERATIONS=10`, `ROGUE_ONE_K_FOLDS=5`, `ROGUE_ONE_SELECTION_SIGNAL=inner_cv`, `ROGUE_ONE_ENABLE_ROLLBACK=False`, `ROGUE_ONE_DO_BALANCING=True`, `ROGUE_ONE_FOLD_RETRY_LIMIT=1`;
- `ROGUE_ONE_STRATIFIED_FOLDS=False` (regression);
- `LLM_TEMPERATURE=0.7`, and 0.7 for `SCIENTIST_AGENT_TEMPERATURE`, `EXTRACTOR_AGENT_TEMPERATURE`, `TESTER_AGENT_TEMPERATURE` and `MEDICAL_KNOWLEDGE_AGENT_TEMPERATURE` (W);
- `ROGUE_ONE_EXTRACTOR_ROUNDS` and `ROGUE_ONE_FEATURE_POOL_MODE` as frozen at G1; `ROGUE_ONE_TOOL_OUTPUT_MAX_CHARS` as pinned at G1; `ROGUE_ONE_TOOL_ERROR_FEEDBACK=message`; `ROGUE_ONE_REPLAY_VALIDITY_CHECK=1`; `ROGUE_ONE_SANDBOX` as frozen (on only if E13 lands);
- search: `ROGUE_ONE_SEARCH_CACHE=shared` at `ROGUE_ONE_SEARCH_CACHE_PATH=/workspace/cache/serper_cache.sqlite`. On v2.3+ images (AB2 only) also `ROGUE_ONE_KNOWLEDGE_BACKEND=web` and `ROGUE_ONE_WEBSEARCH_ROUTING=caller`, which are no-ops for a uniform O120 run;
- `ROGUE_ONE_LLM_TRACE=1`, `ROGUE_ONE_PROMPT_DUMP=1`;
- launcher: `PARALLEL=6`, `CPUS=12`, `RUN_CAP_HOURS=6`;
- W&B entity `PlanBench`, project `RogueOne-AGENT2027`.

**Run matrix.**

| Group | Source | dataset_id | fold_seed / replicate / llm_seed | Runs | Episodes | W&B run names (row IDs) |
|---|---|---|---|---|---|---|
| Named | A05 | `synp_p6a` | 42/0/42 · 4/0/4 · 42/1/1042 | 3 | 15 | `A05__full__synp_p6a__f42__r0`, `…__f4__r0`, `…__f42__r1` (expected A05-036, -044, -052) |
| Anonymised | A05 | `synp_p6b` | the same | 3 | 15 | `A05__full__synp_p6b__f42__r0`, `…__f4__r0`, `…__f42__r1` (expected A05-037, -045, -053) |
| AB2 (optional) | A15 | `synp_p6a`, `synp_p6b` | 5/0/5 | 2 | 10 | `A15__full__synp_p6a__f5__r0` (A15-001), `A15__full__synp_p6b__f5__r0` (A15-002) |

The expected A05 row IDs assume that `make_manifest.py` keeps the SYN-P 8 order p1, p2, p3, p4, p5, p6a, p6b, p7 inside each A05 block, as A02 does. Confirm them in `manifests/agent2027/A05_M0.csv` (A15-T5); the run names are authoritative.

**Budget arithmetic.**
- Core: 6 runs × 0.7–1.0u = 4.2–6u, charged to A05, so **0u for A15**. This gives 15 pairs.
- AB2: 2 × 0.7–1.0u = **1.4–2u**. At 2 u/h × 0.7 = 1.4 u/h this is ≈ 1.0–1.4 h (× 4 GPUs ≈ 4–6 H100-hours), and gives 20 pairs.
- This matches the T1-minimum breakdown ("0 new runs; an optional seed-5 pair adds 1.4–2u") and the registry (01 §7 R1: 0 (+2 opt.), 0 (+1.4–2)u). The contingency smoke `A15-901` (≈ 0.15–0.2u) is extra. It runs only if A02's P6b runs fail, and only on idle pre-freeze capacity.

**Pairing.**
- A pair is (fold_seed, replicate, fold index). Both members share the partition, `llm_seed`, image and session.
- A05 randomises its own order. Under D5 = TP2, both twins of a pair must sit in the same session file (A15-T5). The two AB2 rows go in **one** session, with the random `order_key` from `make_manifest.py`.
- AB2 pairs run on a later `agent2027-v2.N`. Full-v2 is byte-identical across v2.N (the G0b-1 prompt-diff rule, 01 §6.1), and each pair lies within one image.

**Power** (exact two-sided McNemar; worst-case Holm threshold 0.005):
- 9:0 → p = 0.0039 (passes); 8:0 → 0.0078 (fails).
- 12:1 → 0.0034 (passes); 10:1 → 0.0117 and 11:1 → 0.0063 (fail).
- So on 15 pairs the gap must be ≥ 60 pp (9:0), or ≥ 73 pp with one reverse pair (12:1). On 20 pairs it must be ≥ 45 pp (9:0), or ≥ 55 pp with one reverse pair. On 10 pairs (no A/A) it must be ≥ 90 pp.

**Failures.**
- A crash is retried once, alone, at the end of the session.
- A persistent failure or a 6 h cap hit counts under ITT: all 5 episodes get `rec_sel` = `rec_any` = 0. Cap hits are not retried.
- Failure and cap-hit rates are reported per twin.

## 5. Tasks

### 5.1 Prerequisites (verify before analysis or the AB2 launch)
- [ ] **A15-T1** [type: decision] (0.1 d; candidates screened Sun 11 Oct; **due Mon 12 Oct, 12:00**) — **D8:** fix P6a's φ, its column ranges, and whether to rescale. Proposals, in order of preference. Each must pass `screen_pairwise.py` with comparable log-variance across its 3 inputs:
  - (a) Reynolds number v·D/ν. A monomial, so rescaling is exact.
  - (b) Ideal-gas pressure n·T/V. A monomial, but a realistic temperature range may fail the screen.
  - (c) Cockcroft–Gault (140 − age)·weight/(72·creatinine). It is not a monomial, so there is no rescaling and the value-range channel must be disclosed. It is also clinical, and the agent templates already frame every task as clinical (§2). That lets P6b guess the domain without names and biases C4b toward the null. Use (c) only if (a) and (b) fail, and disclose it.

  *Done when:* D8 (φ, ranges, rescale yes/no, the chosen candidate and why the others were rejected) is in the pre-registration draft, and `scripts/agent2027/synp/screen_pairwise.py` passes on P6a.
- [ ] **A15-T2** [type: data] (0.1 d, inside E8a; due Mon 12 Oct, 18:00, G0a, before the M5 CPU screen launches) — E8a acceptance for P6. *Done when:*
  - both folders load via `ExperimentConfig.from_yaml` (`config.py:179`), carry `task_type: regression`, and contain no metadata;
  - both resolve through the `ROGUE_ONE_TASK_REGISTRY` YAML (E1; the legacy list `main.py:1536-1566` has no SYN-P entry), and the folder names are exactly `synp_p6a` / `synp_p6b` (`select_paths` matches on the parent folder name, `main.py:1573`);
  - `labels.csv` SHA-256 values are equal and the `id` columns are identical. After the name and level maps are inverted, every P6b numeric column equals c_j × the P6a column (c_j = 1 without rescaling) to a relative tolerance of 1e-9, and every categorical column is equal. φ hashes match;
  - `data.csv` has no exact-duplicate rows (`id` excluded);
  - headroom passes on both (E8a: XGB(raw + φ) − XGB(raw) ≥ 15% nRMSE);
  - `deterministic_kfold_indices` gives identical test sets on both for seeds {42, 4, 5};
  - the `context_and_goal` character counts agree within ±2%;
  - the folders exist on CAIR at `<ROGUEONE_CAIR_PATH>/tasks/regression/synp_p6a/` and `…/synp_p6b/`, with hashes equal to the local build (A08-T1 checks the same).
- [ ] **A15-T3** [type: decision] (0.05 d; due Tue 13 Oct morning, M5S, before A02 launches) — read the A08 CPU screen for `synp_p6a` and `synp_p6b`. *Done when:*
  - both are < 14/15 for OpenFE and for AutoFeat;
  - the OpenFE counts differ by ≤ 1 episode. AutoFeat differences are reported but not gated, because AutoFeat is unseeded.

  A larger OpenFE gap means the rescaling or the level coding changed name-blind discoverability. Redesign or drop (§6).
- [ ] **A15-T4** [type: data] (0.05 d; due Fri 16 Oct, before the 18:00 launch) — E4 lands only on Wed 14 Oct, so check E3 and E5 on the G0a smoke (Mon 12) and all three on the `agent2027-v2.0` freeze smoke. *Done when* the v2.0 smoke shows:
  - transcript `event="query"` rows with the caller role and URLs, and a logged cache hit for a repeated query that still writes a `query` row (E4);
  - non-zero `tok_search_scientist` (E3; S must search at least once);
  - a prompt dump under `output/prompts/` (E5).
- [ ] **A15-T5** [type: config] (0.05 d; due Fri 16 Oct, 15:00, with A05-T5) — *Done when:*
  - `manifests/agent2027/A05_M0.csv` contains the 6 P6 runs in §4, including the seed-42 SYN-P A/A (never cut), and their row IDs are noted in §4;
  - if D5 = TP2 (two sessions `S1016a`/`S1016b`), both twins of each pair are in the **same** session file. This overrides A05's alternating split for these 6 rows.

### 5.2 Implementation (A15's share of E8a, E5, X10 and E11b, plus one audit script)
- [ ] **A15-T6** [type: code] (0.1–0.25 d, inside E8a; code due Sun 11 Oct; final folders generated Mon 12 Oct after A15-T1) — implement the twin in `scripts/agent2027/synp/generate_synp.py`, as specified in §2. Both rescale branches must be unit-tested, so D8 only sets a parameter. The function must:
  - copy the rows, the row order, `id` and `labels.csv`, and keep P6a's column order in `data.csv`;
  - rename columns to `f01…fNN` and levels to `c01…cKK` by sorted rank (§2);
  - apply the hidden constants only if D8 says the formula is a monomial;
  - pad the neutral context to equal length (±2% of characters);
  - write `config.yml` as in §2, with `experiment_name` = `dataset_id`, `task_type: regression` and neutral values for `experiment_description`, `knowledge_db_collection_name` and `knowledge_db_path`;
  - write the name map, the level map and the constants to `synp_meta/` only.

  *Done when:* the unit tests pass for both branches and the A15-T2 checks pass on the generated folders.
- [ ] **A15-T7** [type: data] (0.05 d, inside E8a; due Mon 12 Oct) — write `synp_meta/p6_domain_terms.txt`. Content:
  - the **strong** terms, marked as such: the formula's name and synonyms;
  - P6a's column names and their tokens, level strings, unit words, and the domain nouns of P6a's context;
  - generic words such as "ratio" are excluded.

  Then grep the list over the twin-invariant template text (`rogueone/llm/prompts/agent_roles.py` and the four agent files `sci/agent.py`, `ext/agent.py`, `tst/agent.py`, `kn/web_search.py`). Record every term found there (for example "patient" or "clinical") as **template-constant** in the same file. *Done when:* its SHA-256 is in the pre-registration.
- [ ] **A15-T8** [type: code] (0.05 d, inside E8a; due Mon 12 Oct) — static leak check: a case-insensitive, word-boundary grep of the term list over P6b's `config.yml`, the `data.csv` header and categorical values, and the `labels.csv` header. Also grep P6a's `context_and_goal` for the strong terms. *Done when:* both greps return 0 hits.
- [ ] **A15-T9** [type: config] (0.05 d; due Mon 12 Oct, before the first `agent2027-pre.N` build; recheck at every image that runs P6 rows) — keep P6 material out of client images. Add these to `.dockerignore`:
  - `scripts/agent2027/synp/` (generator);
  - `synp_meta/`;
  - `scripts/agent2027/llmfe_specs/` (P6a's LLM-FE spec carries P6a's description; LLM-FE mounts `~/llmfe` and does not need the rogue_one image);
  - `docs/agent2027/` (the pre-registration names φ; A14-T2 adds this entry too).

  Generator and screen work then runs locally, or with that folder mounted explicitly. Then, for each of `agent2027-pre.N` (Tue 13), `agent2027-v2.0` (Fri 16) and every later `agent2027-v2.N` that runs P6 rows (AB2 and the exploratory arms), run this on CAIR. It is a non-GPU container, so no pre-flight is needed. Each strong term is a separate `-e`:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker run --rm --network none --entrypoint sh rogue_one:<TAG> -c 'for p in /workspace/scripts/agent2027/synp /workspace/scripts/agent2027/llmfe_specs /workspace/synp_meta /workspace/docs/agent2027; do test -e \$p && echo LEAK \$p; done; grep -rIilF -e \"<strong term 1>\" -e \"<strong term 2>\" /workspace --exclude-dir=tasks; echo checked'"
  ```
  `/workspace/tasks` is excluded because the per-row mount shadows it at run time (§2). *Done when:* the output is only `checked` for every listed image. Any other hit is a file to add to `.dockerignore` before that image runs P6 rows.
- [ ] **A15-T22** [type: code] (0.1 d; due Wed 14 Oct, so that A15-T10 can use it) — write `scripts/agent2027/analysis/a15_leak_audit.py`. Inputs: W&B project, group, twin dataset IDs, the term list and the task root. It downloads the `run-output` artifacts of the matching runs and reports four checks per P6b run:
  - (a) **Prompt hits.** It greps the term list (case-insensitive, word boundary) over the P6b prompt dumps (`output/prompts/`). A hit is a **leak** only if its line does not occur verbatim in the P6a dump of the same role and call. Lines that do occur there are template text (for example `ext/agent.py:114`). Template-constant hits are listed, not counted.
  - (b) **Feedback pool.** The `dataset_id` values of the feedback pool must equal `synp_p6b`.
  - (c) **Agent-written code.** It greps the tool-call arguments in the E3 LLM traces (S, E and T code, including every `Pandas Command`) for `open(`, `read_csv`, `read_json`, `read_table`, `read_parquet`, `sqlite`, `serper_cache`, `/workspace`, `os.listdir`, `os.walk`, `glob`, `os.environ`, `wandb`, `subprocess`, `requests`, `urllib` and `socket`. A hit counts only if the code reads something other than its in-memory dataframes. Each hit is listed with its call for manual review.
  - (d) **Fold test sets.** It recomputes them with `deterministic_kfold_indices` from both task folders, and checks that the data hashes in the runs' task folders equal A15-T2's.

  It logs `ANALYSIS__a15_leak_audit__<YYYYMMDD>` (group `ANALYSIS`). *Done when:* on a toy pair of dumps it flags a planted P6a name in a P6b context block, ignores the same planted word in a line shared by both dumps, and flags a planted `open('/workspace/cache/…')` call.
- [ ] **A15-T10** [type: data] (0.05 d; due Thu 15 Oct, before X10) — run A15-T22 on A02's P6b runs (`A02__R10__synp_p6b__f42__r0`, `A02__R3__…`, `A02__R1__…` = A02-007, -015, -023; project `RogueOne-AGENT2027-prefreeze`; image `agent2027-pre.N`; command as in §5.4). If those runs failed, run a 1-fold P6b smoke instead:
  - row `A15-901`, `arm_id=full`, `ROGUE_ONE_SMOKE_FOLDS=1`, `notes=smoke`, image `agent2027-pre.N`, project `RogueOne-AGENT2027-prefreeze`;
  - launched on idle pre-freeze capacity with the full 01 §5.2 procedure (pre-flight, ≤ 4 GPUs).

  *Done when:* checks (a) and (b) report 0 leaks in the S, E, T and W system prompts, tool docstrings, first user messages and feedback-pool `dataset_id`. Hits in agent-written text are logged as "guessed domain", not as a leak. Check (c) is reported for information.
- [ ] **A15-T11** [type: writing] (0.1 d, inside X10; due Fri 16 Oct) — write the C4b block of `docs/agent2027/preregistration.md`. It covers:
  - the switch from SSP-3;
  - D8;
  - the twin naming rule (§2);
  - the pair definition, Δ, the exact McNemar test, the Newcombe CI and ITT;
  - the term-list hash, the template-constant rule and the `query_domain_share` definition;
  - the leak definition and the drop and leak rules (§6);
  - the AB2 status;
  - the §5.4 analyses;
  - a **blinding note**: A02/A03 pre-freeze P6 runs exist before this commit. They were read only through their own decision rules and the A15-T10 leak check, never as a P6a-vs-P6b table. They are excluded from C4b and reported in a separate pre-freeze panel.

  *Done when:* the block is in the `agent2027-prereg` tag.
- [ ] **A15-T12** [type: code] (0.15 d, inside E11b; due Sun 25 Oct) — two additions:
  - a C4b block in `scripts/agent2027/analysis/glmm_holm.py`. It builds the pairs, runs an exact two-sided McNemar on (n10, n01), and computes Δ with a Newcombe (1998, method 10) 95% CI for the paired difference;
  - `query_domain_share` in `analysis/process_metrics.py`. The unit is one W tool call: `event="query"` rows with `variant_index` 0 or missing. The share is the fraction of those rows whose `query` contains ≥ 1 listed, non-template-constant term (case-insensitive, word boundary), split by caller role. Field names are as at `kn/web_search.py:96-167`; re-check them after E4.

  *Done when:*
  - toy inputs give p = 0.0039 (9:0) and p = 0.0034 (12:1);
  - for 9:0 of 15 the CI contains Δ = −0.6 and excludes 0;
  - a toy transcript gives the hand-counted share.

### 5.3 Execution on CAIR (core: none; AB2 only)

**Rules for any launch** (01 §5.1):
- **Containers only.** The host issues only `docker` and `docker compose` (with `-T` over SSH), the `~/ro27_*` launchers that issue them, and the monitoring queries `nvidia-smi`, `docker ps` and `tail`. Never `python` on the host.
- **GPU pre-flight before every GPU container.** If GPUs are short, stop and report the per-GPU memory and utilisation, the compute apps and the containers.
- **≤ 4 of the 8 H100s**, ever, without Henrik's explicit approval in chat. Run one Rogue One job at a time, with explicit `GPUS=<ids>` and `WAIT_MAX_MIN=0`.
- **W&B is the system of record.** Runs are stateless and upload `run-output`. `WANDB_API_KEY` comes from `<CAIR_HOME>/code/PlanBench/.env`; `SERPER_API_KEY` and LLM settings come from the image `.env` / `.secret.env`.
- **Long jobs** use a detached `nohup` launcher, with the log in `~` and never in the root-owned run folders. Code reaches CAIR only by commit and push locally, then `git fetch` over SSH on the server.
- **Server path.** `<ROGUEONE_CAIR_PATH>` = `<CAIR_HOME>/code/RougeOne` is **inferred, not verified**. Confirm it once with `ssh CAIR-gpu14 "ls -d ~/code/RougeOne"`.

- [ ] **A15-T13** [type: decision] (0.05 d; due Fri 16 Oct, G1) — fund AB2 only if G1 measures u at the low end and AB1 is funded (add-back order: AB1, then AB2). Decide on measured u only, never on P6 recovery. *Done when:* the decision is recorded in the pre-registration. Afterwards, AB2 may be dropped only for capacity, and the drop is disclosed.
- [ ] **A15-T14** [type: config] (0.05 d; due Tue 20 Oct, if AB2) — `scripts/agent2027/make_manifest.py --ablation A15` writes `manifests/agent2027/A15_KT2-min.csv` with two rows:
  - `A15-001` (`synp_p6a`) and `A15-002` (`synp_p6b`);
  - `ablation_id=A15`, `arm_id=full`, `task_type=regression`, `fold_seed=5`, `llm_seed=5`, `replicate=0`, `is_sentinel=0`;
  - `rung_S/E/T=O120`, `port_S/E/T=8010`, `env_overrides={}`, `cap_hours=6`, `expected_u=0.85`;
  - `image_tag` = the host session's `agent2027-v2.N` (one that passed A15-T9).

  Add both rows to **one** O120 session between Wed 21 and Thu 29 Oct: normally a host session such as A12's or A13's, which carries its own A06 sentinel. Use `make_manifest.py --plan-session <SESSION_ID> --rows A15-001 A15-002 …`. Then commit and push `agent2027/v2`. *Done when:* the session file holds both rows and the push is on `origin`.
- [ ] **A15-T15** [type: run] (0.1 d developer time, ≈ 1.0–1.4 h of box time; due Thu 29 Oct, L-LAST) — in the normal case the host ablation's launch runs steps 0–4 below, and A15 only checks the "Done when". If A15 launches its own session (its own `<SESSION_ID>` plus an A06 sentinel row), follow 01 §5.2:
  0. Sync code and install the launchers and manifests:
     ```
     ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15' git fetch git@github.com:HenrikNorkart/RougeOne.git agent2027/v2 && git checkout -B agent2027/v2 FETCH_HEAD && git log -1 --format='HEAD: %h %s' && cp scripts/agent2027/ro27_session_launch.sh scripts/agent2027/ro27_clients_manifest.sh scripts/agent2027/ladder_session.sh ~/ && cp paper/scripts/vllm_session.sh ~/ro27_vllm_session.sh && mkdir -p ~/ro27_manifests && cp -r manifests/agent2027/. ~/ro27_manifests/ && docker image inspect rogue_one:agent2027-v2.N --format '{{.Id}}'"
     ```
     The image must already exist; build it only if code changed (01 §5.2 step 2).
  1. GPU pre-flight:
     ```
     ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader && echo '--- compute apps ---' && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader && echo '--- containers ---' && docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'"
     ```
     Continue only if all of these hold. Otherwise STOP and report:
     - the chosen GPUs each use < 2,000 MiB;
     - there are ≤ 4 of them (4 for TP4, 2 for TP2);
     - no `ro27-*`, `rogueone-vllm-*`, `llmfe-*` or `caafe-*` container of ours is running.
  2. Launch detached:
     ```
     ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "SESSION_ID=<SESSION_ID> SESSION_FILE=~/ro27_manifests/sessions/<SESSION_ID>.csv GPUS=<ids> MODEL=openai/gpt-oss-120b PORT=8010 SERVER_NAME=rogueone-vllm-o120-<session_id> LOG=~/vllm_session_<SESSION_ID>.log IMAGE_RO=rogue_one:agent2027-v2.N PARALLEL=6 CPUS=12 RUN_CAP_HOURS=6 RUNROOT=~/rogueone-runs-agent2027 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_<SESSION_ID>.job.log 2>&1 & echo launched; sleep 5; tail -5 ~/ro27_<SESSION_ID>.job.log"
     ```

  *Done when:* the job log ends with `=== RO27_SESSION_END <SESSION_ID> rc=0 ===` and has an `OK` line for A15-001 and A15-002, and both W&B runs (group `A15`) are `finished` with a `run-output` artifact.
- [ ] **A15-T16** [type: run] (0.05 d; during the session, at latest Sat 31 Oct, DF) — monitor in the background:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 CAIR-gpu14 "while ! grep -q 'RO27_SESSION_END' ~/ro27_<SESSION_ID>.job.log 2>/dev/null; do sleep 120; done; tail -30 ~/ro27_<SESSION_ID>.job.log"
  ```
  - Exit 255 means the network dropped: reconnect.
  - Cross-check the W&B states. A `running` run with no live container is a zombie.
  - Then run the teardown check: `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker ps --format '{{.Names}}' | grep -E '^(ro27-|rogueone-vllm-)' || echo none"`.

  *Done when:* the teardown check prints `none`. A twin that fails twice is ITT, and the pair is kept.

### 5.4 Analysis

Analysis runs on CAIR in a CPU container (`--cpus` ≤ 16), or locally. It uses the `scripts/agent2027` tree of the current branch, mounted read-only, as in A02. `~/rogueone-synp-meta/` is mounted read-only into **analysis** containers only. Template (fill `<IMAGE>`, `<script>` and its arguments; the CLI of each script is set by the task that writes it):
```
ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "mkdir -p ~/ro27_a15/out && cd <ROGUEONE_CAIR_PATH> && GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15' git fetch git@github.com:HenrikNorkart/RougeOne.git agent2027/v2 && rm -rf ~/ro27_a15/scripts && git archive FETCH_HEAD scripts/agent2027 | tar -x -C ~/ro27_a15 && docker run --rm --cpus 16 --env-file <CAIR_HOME>/code/PlanBench/.env -e PYTHONPATH=/opt/ro27 -w /opt/ro27 -v ~/ro27_a15/scripts/agent2027:/opt/ro27:ro -v <ROGUEONE_CAIR_PATH>/tasks:/workspace/tasks:ro -v ~/rogueone-synp-meta:/meta:ro -v ~/ro27_a15/out:/out --entrypoint python3 rogue_one:<IMAGE> analysis/<script>.py <arguments>"
```
Files in `~/ro27_a15/out` are root-owned, so read them only through containers or W&B.

- [ ] **A15-T17** [type: analysis] (0.15 d; due Mon 19 Oct, G2) — integrity and leak audit of the 6 A05 P6 runs. Run `analysis/a15_leak_audit.py` (A15-T22) with project `RogueOne-AGENT2027`, group `A05` and image `agent2027-v2.0`. Run `recovery_detector.py` on the same runs. Check that:
  - all 6 runs are `finished`, with failure and cap status noted;
  - checks (a) and (b) report 0 leaks;
  - check (c) shows no read of anything other than in-memory dataframes (including the Serper cache, `/workspace/scripts`, `/workspace/docs`, W&B and the network outside the search tool);
  - check (d) shows identical fold test sets across twins.

  *Done when:* the audit run and its result are in the G2 memo. Any leak follows the leak rule (§6).
- [ ] **A15-T18** [type: analysis] (0.05 d; due Mon 26 Oct, G3) — preliminary C4b on 15 pairs: the 2×2 table, n10 and n01, p, and Δ with its CI. *Done when:* it is in the G3 memo, labelled preliminary.
- [ ] **A15-T19** [type: analysis] (0.2 d; due Tue 3 Nov) — final analysis on 15 pairs (20 with AB2). AB2 runs first pass the A15-T17 audit (group `A15`).
  - **Primary:** exact McNemar on `rec_sel`. The raw p enters Holm in `glmm_holm.py`.
  - **Secondary (descriptive, per twin):** `rec_any`, `iter_first_rec` (censored at 11), `query_domain_share`, `tok_search_<role>`, `serper_calls`, `fallback_share`, `success`, `collapse`, `cap_hit`, `heldout_score`.
  - **Sensitivity:** drop fallback-only runs; drop pairs with an ITT member; replicate 0 only; A05 pairs only (if AB2 ran).
  - **Exploratory, outside Holm:** P6b − P6a inside the other arms that run both twins (A08 `llmfe`, A09 `misdirected`, A10–A12, A16, A17, A19, A20), fitted as `rec_sel ~ twin + arm + (1|seed)`. The A09 `oracle` recovery on P6a/P6b is the implementability ceiling. The pre-freeze A02/A03 P6 runs go in a separate panel.

  *Done when:* W&B run `ANALYSIS__glmm_holm__<YYYYMMDD>` (group `ANALYSIS`) holds the table `c4b_pairs`, p, the Holm-adjusted p, and Δ with its CI.

### 5.5 Paper outputs
- [ ] **A15-T20** [type: writing] (0.2 d; due Fri 6 Nov) — three items:
  - the Table 2 row "KT2-min (C4b)": Δ `rec_sel` (CI), Δ `success`, Δ `tok_compl_total`, funnel stage changed, no-harm "n/a (synthetic)", verdict (necessary / reducible / equivalent / inconclusive);
  - one Results sentence, worded as in §3 (write "Holm-adjusted p < 0.05", never "< 0.005");
  - an appendix paragraph on the twin spec, the leak checks and `query_domain_share`.

  *Done when:* all three are in the AGENT draft.
- [ ] **A15-T21** [type: writing] (0.1 d; due Thu 12 Nov) — disclosures:
  - the instrument switched from SSP-3 before any data existed;
  - names, levels and description are manipulated jointly;
  - the value-range channel, if there is no rescaling;
  - the agent templates' clinical framing is identical in both twins;
  - the pre-freeze P6 pilots were visible before the pre-registration;
  - the authors built the task;
  - there is no real-named table and no search-off cell, so semantics × search is untested;
  - the Serper cache is shared;
  - agent code had full builtins if E13 did not land.

  *Done when:* these are in the paper, and the term list and twin generator are in the replication package.

## 6. Decision rules and gates

**Definition (used below).** An **input leak** in P6b is either of these:
- a term-list hit (template-constant terms excluded) in a twin-specific line of any P6b system prompt, tool docstring, first user message or feedback-pool `dataset_id` (A15-T22 checks (a) and (b));
- any P6b agent code that reads P6a material or a P6a-derived store: a baked P6 file, the Serper cache file, W&B, or another run's output (check (c)).

Domain words that the agents write themselves are "guessed domain", not a leak.

| When | Rule |
|---|---|
| Mon 12 Oct (G0a, D8) | If P6a fails `screen_pairwise.py`, take the next D8 candidate the same day. If none passes by Mon 12 Oct, 24:00, drop C4b. |
| Tue 13 Oct (M5S) | If P6a **or** P6b is non-discriminating (≥ 14/15 for OpenFE or AutoFeat), or the OpenFE parity check fails and is not fixed by **Wed 14 Oct, 18:00** (fixed = regenerated P6b whose re-run OpenFE cells, seeds {42, 4, 5}, pass parity), drop C4b **before any post-freeze GPU P6 data exist**. Holm then has 9 contrasts (smallest threshold 0.0056), and A15 becomes descriptive. There is no fallback instrument. |
| Thu 15 Oct | An input leak in A15-T10 must be fixed (template, generator or `.dockerignore`) and re-checked before X10. Otherwise C4b is pre-registered as dropped. |
| Fri 16 Oct (G1, X10) | C4b is pre-registered or recorded as dropped; AB2 is decided. P6a/P6b are then frozen: a change would mean re-running M0. |
| Mon 19 Oct (G2) | **Leak rule:** an input leak in an A05 P6b run makes C4b invalid. It is a declared deviation, it enters Holm with p = 1 (the family stays at 10), and the clause is dropped. The same rule applies to AB2 runs at A15-T19. |
| Thu 29 Oct (L-LAST) | Last AB2 launch. Fri 30 – Sat 31 are for reruns only; DF is Sat 31, 24:00. |
| Tue 3 Nov | Holm-adjusted p < 0.05 with Δ < 0 → "names help". Holm-adjusted p < 0.05 with Δ > 0 → "names hurt". Otherwise "inconclusive", with the CI. Floor and ceiling are read as in §3. |

## 7. Risks and cut levers
- **The screen kills P6.** Mitigated by the three D8 candidates and by screening inside E8a. If it happens, C4b is dropped and C4 rests on C4a.
- **Low power.** At least 60 pp is needed on 15 pairs; AB2 lowers this to at least 45 pp. Without the A05 SYN-P A/A only 10 pairs remain (≥ 90 pp needed), which is why that run is on the never-cut list.
- **The twin is not exact.** Rescale monomials only. Use the sorted-rank naming rule. Run the A15-T2 and A15-T3 parity checks. AutoFeat noise is not gated.
- **Template priming.** The agent templates frame every task as clinical (§2). With a clinical φ, P6b can guess the domain, which biases C4b toward the null. This is mitigated by the D8 preference order and disclosed. Neutralising that text is outside A15 (it changes Full; see "Open issues").
- **Leaks** through `experiment_name` → `dataset_id`, the context, level strings, value ranges, files baked into the image, the shared Serper cache, or out-of-band access from agent code. Covered by A15-T6 and T8–T10, T17 and T22.
- **Pre-freeze peeking.** A02/A03 P6 data precede X10. Covered by the blinding note (A15-T11) and the separate panel.
- **Dependence on A05.** A G2 rerun of A05 delays A15's data by about 2 days.
- **External validity.** One synthetic table with real-sounding names; disclosed.
- **Cut levers.** None of CL1–CL7 touches A15 (0 runs). AB2 is its only optional spend.

## 8. Definition of done
- [ ] D8 recorded; P6b generated; A15-T2 and A15-T3 pass, or C4b is formally dropped before X10.
- [ ] Term list hashed in the pre-registration; the static (T8), image (T9), pre-freeze prompt (T10) and G2 (T17) leak checks find 0 input leaks.
- [ ] The 6 A05 P6 runs (+2 AB2 runs if funded) are `finished` with `run-output` artifacts.
- [ ] `c4b_pairs`, p, the Holm-adjusted p, and Δ with its CI are in an `ANALYSIS` run, with the secondary, sensitivity and exploratory outputs.
- [ ] The Table 2 row, Results sentence, appendix and disclosures are in the draft.

---

**New names introduced** (not in the canonical registry):
- `synp_meta/p6_domain_terms.txt`: the frozen domain-term list, with "strong" and "template-constant" markers. It is off-mount and excluded from the image; on CAIR it lives in `~/rogueone-synp-meta/`.
- `query_domain_share`: the per-run share of W tool calls whose Serper query contains ≥ 1 listed, non-template-constant term.
- `c4b_pairs`: the W&B analysis table.
- `scripts/agent2027/analysis/a15_leak_audit.py` (A15-T22) and its W&B run `ANALYSIS__a15_leak_audit__<YYYYMMDD>`.
- `.dockerignore` entries `scripts/agent2027/synp/`, `synp_meta/`, `scripts/agent2027/llmfe_specs/` and `docs/agent2027/` (A15-T9; the last is shared with A14-T2).
- The P6b naming rule `f01…fNN` / `c01…cKK` by sorted rank (refines 01 E8a's "f1..fp / c1..ck").
- `A15-001`, `A15-002`, the smoke row `A15-901` and `manifests/agent2027/A15_KT2-min.csv` follow the R6/R7 patterns. The working folder on CAIR is `~/ro27_a15/`.

**Open issues (for Henrik / 01):**
- The clinical template text (`agent_roles.py:9`, `:15`; `ext/agent.py:114`, `:126`, `:143`, `:391`) is not on E5's neutralisation list. Adding it would change Full-v2 before the freeze. This is a decision for E5/G1, not for A15.
- `<ROGUEONE_CAIR_PATH>` and the server git remote are inferred (01 §5.2).
- The paths of the E3 trace files and the E4 transcript fields are set by those tasks. A15-T22 must be pointed at them once they land.
