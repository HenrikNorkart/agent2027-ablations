# A19 · CAP-2 — Role mixing: E-small cell and 1B (S+T at Q0.8) cell (C5b)

*Rogue One v2 design study · ICSE AGENT 2027 (8 pp IEEE) · submit **Fri 13 Nov 2026** · scope T1-minimum · written Wed 7 Oct 2026 (plan date Tue 6 Oct).*

*Code refs: RougeOne `nested-cv-rebuttal` @ `07cbf93` (`C:/code/PHD/RougeOne`). Every cited line was re-read at HEAD `1bdaf42` (verifier pass, Wed 7 Oct); `rogue_one_main.py`, `rogueone/`, `tests/`, `vllm_session.sh` and `clients_t16_rerun.sh` are unchanged between the two (`git diff 07cbf93 1bdaf42` touches only other `paper/scripts/` files). Short paths: `main.py` = `rogue_one_main.py`; `sci/`, `ext/`, `tst/agent.py` = `rogueone/llm/agents/core/{scientist,extractor,tester}/agent.py`; `kn/web_search.py` = `rogueone/llm/agents/core/knowledge/web_search.py`; `config.py` = `rogueone/utils/config.py`; `clients.py` = `rogueone/llm/clients.py`. Lines will shift after the E2/E3/E5/E7 merges, so re-anchor on the quoted code. "01 §n" = `docs/ablations/01_shared_infrastructure.md`. Every "hoped" result is a hypothesis, not a finding.*

| Field | Value |
|---|---|
| **ID** | A19 · CAP-2 · arms `mix_esmall`, `mix_1b` |
| **Claim / Holm contrast** | C5, "roles can use different models". Supplies **C5b = E-small − all-big** on `rec_sel`, on SYN-P 7 × {42, 4}. C5b is two-sided and one of the 10 Holm contrasts (smallest threshold 0.005). The **1B cell** is a pre-registered hypothesis **outside Holm**. The big-model token ratio is a **cost outcome outside Holm**. |
| **Tier** | T1-minimum |
| **Owner** | Henrik |
| **Start / finish** | Pre-registration text Fri 16 Oct · token threshold τ fixed at G2, Mon 19 Oct · E7 in image v2.3 and A04 session C, Wed 21 Oct · δ committed by A09-T15, Fri 23 Oct 18:00 · **earliest launch Thu 22 Oct** (R1), in the first whole-box slot after `S1021a` (A12/A13/A14) ends, which A13/A14 estimate at Thu 22 evening – Fri 23 afternoon; the breakdown's GPU plan books CAP-2 in W3 (Mon 26 – Sat 31), so the planned window is **Thu 22 – Tue 27** (box exclusive for 20–39 h) · latest full-matrix launch **Tue 27 Oct** · CL1 launch by Wed 28 · L-LAST Thu 29 · reruns until DF (Sat 31, 24:00) · statistics Sun 1 – Tue 3 Nov · text Wed 4 – Fri 6 Nov |
| **Budget** | **36 new runs, 27.6–54u.** That is 13.8–27 box-hours at 2 u/h, or 19.7–38.6 wall-hours at 0.7 utilisation: 79–154 H100-hours on 4 GPUs. The control costs 0 new runs (A05 pays); the session sentinel is paid by A06. |
| **Models** | O120 `openai/gpt-oss-120b` (MXFP4, TP2, port 8010) · O20 `openai/gpt-oss-20b` (MXFP4, TP1, port 8020) · Q0.8 `Qwen/Qwen3.5-0.8B` (BF16, TP1, port 8030, Qwen set, thinking off). One session, ΣTP = 4, server image `vllm/vllm-openai:v0.22.0`, client image `rogue_one:agent2027-v2.3`. |
| **Status** | Not started |
| **Depends on** | E1, E3 (incl. the A19-T0 focus-source and trace fields), E4, E5, E7, E8a, E11a, E11b, X1, X10; the rest of Full-v2 via image v2.0+ (E2, E6). A04 (SR4 O20 verdict, SR6 routing pass, `rungs.csv`), A05 (control runs, per-role token shares), A06 (sentinel), A08 (brute-force gate), A09 (oracle gap → margin δ, A09-T15). A12 `fx_pajM` (descriptive only). Gates G0a, G1, G2, G0b-1. Decisions D4 (stratified SYN-P C folds), D5. A17 (context rows `f2_Q0.8`, `f2_O20`, optional all-big `f2_Q122`) · A18 (informing only) · A20 (O20 slot timing). |
| **Feeds into** | The Table 2 row for C5b and the 1B row; the CAP-2 role-grid inset of Fig. 3; the C5 paragraph and the "1B vs 100B" answer; A23 (Pareto points for the role mixes); A21 (iso-cost points); the A18 read-out; the replication package |

> **SR4 fallback (2026-10-19).** O20 failed Stage 0, so the §6 "SR4 fallback" paragraph applies
> (pre-registration Part II, Part V.19, Part VII.1 item 14; Part VII.3 supersedes the O20/O120
> text of §1–§5). Applied here, with no threshold changed:
> - **Rungs:** big = Q122 (`Qwen/Qwen3.5-122B-A10B-FP8`, TP2, `--max-num-seqs 256`, port 8040),
>   small = Q4 (`Qwen/Qwen3.5-4B`, TP1, 8032), Q0.8 (TP1, 8030); ΣTP = 4. `mix_esmall` = S, T on
>   Q122 and E on Q4; `mix_1b` = S, T on Q0.8 and E on Q122. Global `LLM_MODEL`/`LLM_PORT` =
>   Q122/8040 in both arms. W follows its caller (E7, `ROGUE_ONE_WEBSEARCH_ROUTING=caller`, image
>   `agent2027-v2.3`). `mix_1b` is all-Qwen, so the cross-family disclosure lapses.
> - **Reference** = A17 `f2_Q122` (A17-070…083, SYN-P 7 × {42, 4}; no A/A, 2 runs per task;
>   `big_tok_ratio` denominator = the single paired run). **δ** = the O120 δ of A09-T15,
>   used unchanged with the Q122 reference (Part VIII.1, disclosed). **τ = 0.60** (Amendment 2,
>   G2: median s_E = 0.425 ≤ 0.50; from O120 shares, disclosed). `fx_pajM` comparison dropped
>   (re-run at Q122 only if Henrik approves at G4).
> - **CL1** (SYN-P only): 28 rows, A19-001…014 (`mix_esmall`) and A19-019…032 (`mix_1b`);
>   A19-015…018 and A19-033…036 are not generated. Manifest `manifests/agent2027/A19_CAP-2.csv`
>   (`scripts/agent2027/manifest_specs/A19.yaml`).
> - **Session:** `ladder_session.sh` was never built. A19 runs as one main launch `S1022a` (owns
>   the Q122 TP2 server, 8040; its clients wait until 8032 and 8030 are healthy) plus two holder
>   launches that own the other servers and stay up until `S1022a` ends: `S1022d` (Q4, 8032; it
>   also runs A20's 7 `fx_pajM_Q4` rows and `sentinel_Q4` A06-103 as its own clients, so the
>   session group carries its sentinel) and `S1022f` (Q0.8, 8030; no rows). Script
>   `scripts/agent2027/ro27_hold_clients.sh`; launch lines in `docs/agent2027/sessions/LAUNCH.md`.
>   Gates unchanged: G0b-1 (v2.3), **SR6** from A04 session C (`S1021m`, SR4 layout), the τ
>   addendum (done, Amendment 2), A19-T9a; slot from Thu 22 (Henrik's box order).

---

## 1. What we test and why it matters

We run two **role-mixing cells** and compare each with the all-O120 Full-v2 reference (A05):
- **E-small** (`mix_esmall`): only the code-writing Extractor moves to gpt-oss-20b. Its search summariser moves with it. Does the Extractor *need* the big model? This is the confirmatory contrast C5b.
- **1B** (`mix_1b`): the Scientist and the Tester, with their search summarisers, run on Qwen3.5-0.8B, and the Extractor stays on O120. Can roles that judge and summarise run at ~1B, given a big code writer?

**Why it matters.** C5 claims that roles can be given different models. Per-role model allocation is also one of the pre-written fallback claims if C1 fails (master plan §1, "Fallback narrative"). A19 is the paper's most direct answer, per role, to "can about 1B do as well as about 100B?". A17 (whole system per rung) and A18 (one-turn replay) give the context around it.

## 2. Background you need

**The loop.** `forward_pass` (`main.py:536-837`) runs:
1. the Scientist's focus (`:548-555`);
2. the Extractor's R rounds (`:563-571` → `ext/agent.py:965-975`; `range(10)` at `:967` until E2's R knob);
3. the Tester's Stage A, assessment (`tst/agent.py:1483-1496`), and Stage B, pruning (`:1579-1591`), called at `main.py:789-794`.

Fold state resets at `main.py:543-544`.

**Per-role routing already exists for S, E and T.**
- `AGENT_ROLES = ("scientist", "extractor", "tester")` (`config.py:221`). The comment at `:217-220` says the retrieval helpers "deliberately stay on `LLM_MODEL`".
- `ROGUE_ONE_<ROLE>_MODEL` / `_PORT` (names built by `_role_env`, `config.py:224-225`) fall back to `LLM_MODEL` / `LLM_PORT` when unset or empty (`:253-260`). They are resolved by `model_for` / `endpoint_for` (`:262-274`; `model_for` raises `KeyError` for an unknown role), exposed as `scientist_model` / `extractor_model` / `tester_model` (`:276-286`), and registered at `rogueone/utils/config_keys.py:51-56`. Unit tests: `tests/unit/utils/test_per_agent_model.py` (7 test functions, marked `unit`; the repo has no CI workflow).
- The endpoint is always `http://localhost:<port>/v1` (`config.py:251`, `:274`). Clients run with `--network host` (01 §5.3), so all three servers must be on CAIR-gpu14 itself.
- Call sites:
  - S: `sci/agent.py:397-401` (endpoint) and `:457-460` (model); the no-tool fallback also uses it at `:560-563`.
  - E: `ext/agent.py:848-852` and `:931-934`.
  - T: `tst/agent.py:1338-1356`.
- **One API key for every server.** `LLM_API_KEY` is global (`config.py:249`) and every client passes `llm_cfg.api_key` (`sci:399`, `ext:850`, `tst:1340`, `kn/web_search.py:190`). All three servers must therefore run with the same `API_KEY`. The `vllm_session.sh` default is `rogueone-nestedcv-key` (`paper/scripts/vllm_session.sh:41`).
- **The model string must equal the served name.** `vllm_session.sh:166` serves `--served-model-name "$MODEL"`, the HF id, so the per-role `_MODEL` values are HF ids.

**The web-search summariser W does not follow its caller today. E7 fixes this.**
- Each role builds its own `WebSearchAgent`: `sci/agent.py:67`, `ext/agent.py:166` and `tst/agent.py:661`.
- The tools that use it are S `search_tool` (`sci:135-174`, call at `:149`), E `search_in_literature_tool` (`ext:246-280`, call at `:261`) and T `search_in_literature_tool` (`tst:1041-1073`, call at `:1058`).
- All three go through `explain_query` (`kn/web_search.py:292-309`) to `_run_search_agent` (`:57-271`). That function builds its client on the **global** `llm_cfg.endpoint` (`:189`) and `llm_cfg.model` (`:238`), and logs as `agent_name="WebSearchAgent"` (`:191`).
- So in a mixed run today, every W call goes to the global model. E7 adds `"websearch"` to `AGENT_ROLES` and, under `ROGUE_ONE_WEBSEARCH_ROUTING=caller`, uses the caller's endpoint and model. E3/E4 charge W tokens to the caller.
- The `KnowledgeAgent` is constructed (`sci:64-66`) and would also use the global model (`rogueone/llm/agents/core/knowledge/agent.py:56-60`, `:125-128`), but its tool is commented out (`sci:176-177`, `:463`). So nothing active uses the global model once E7 lands (`rogueone/utils/run_vllm.py:162` reads `LLM_MODEL` only on the in-container vLLM start path, `main.py:1600`, whose code default is `True`; `v2_defaults.env` sets `ROGUE_ONE_START_VLLM=False`, R3).

**Fallbacks that can hide a weak role.** These matter for reading the 1B cell.
- **S:**
  - up to 100 attempts (`sci:519`), each with up to `SCIENTIST_AGENT_MAX_ITERATIONS` = 200 turns (`config.py:411`);
  - an attempt succeeds if its output is longer than 10 characters (`sci:528`);
  - next, a no-tool fallback agent at temperature ≤ 0.3 with 4 turns (`:547-577`);
  - last, a fixed heuristic focus (`:501-517`, returned at `:592`).
- **T:**
  - each stage gets up to 20 attempts (`tst:1373`);
  - if assessment fails, a fixed "fallback report" is used (`:1498-1503`);
  - if pruning fails, nothing is pruned (`:1593-1595`);
  - a pruning floor restores features up to max(3, ⌈0.2 · pre_active⌉) (`_enforce_pruning_floor`, `:119-163`; `(pre_active + 4) // 5` at `:138`; applied at `:1613-1635`);
  - if every feature would be pruned, pruning is skipped (`:1637-1644`);
  - both stage outcomes are already logged to W&B as `tester_stage/assessment_status` and `tester_stage/pruning_status` (`:2215-2216`; values `completed` / `failed` / `floor_enforced` / `skipped`).
- **E:** if nothing was appended, deterministic baseline features are added (`_apply_baseline_fallback_attributes`, `ext:599`, triggered at `:996-1002`; today only a console line records it).
- **S fallback source is not logged today:** nothing distinguishes a focus from the tool agent (`sci:528-529`), the no-tool fallback (`:576-577`) or the heuristic (`:592`). A19-T0 adds this to E3.
- **Transport:**
  - gpt-oss text-format tool calls are rescued (`_resolve_text_tool_call`, `clients.py:37-108`; applied at `:675-687`). The rescue handles JSON only (`_extract_first_json_object`, `:145`), so Qwen XML calls are not rescued;
  - Qwen thinking is off (`clients.py:204`);
  - `ROGUE_ONE_LLM_SEED` goes on every request (`:211-213`);
  - `reasoning_effort` is popped (`:525`).

**Rungs** (R9; ports R8; HF revisions from A04's `rungs.csv`):

| Rung | HF id | Active / total | GPUs | Port | Serve flags |
|---|---|---|---|---|---|
| O120 | `openai/gpt-oss-120b` | 5.1B / 117B MoE | 2 (TP2) | 8010 | `MAX_MODEL_LEN` only |
| O20 | `openai/gpt-oss-20b` | 3.6B / 21B MoE | 1 (TP1) | 8020 | `MAX_MODEL_LEN` only |
| Q0.8 | `Qwen/Qwen3.5-0.8B` | 0.8B dense | 1 (TP1) | 8030 | `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --language-model-only --max-model-len 131072` |

O20 was chosen as the small Extractor for continuity with M0 and FX: it is the same family and the same transport path as O120, and its 3.6B active parameters are close to Q4's 4B.

**Instruments** (R10).
- **SYN-P 7** = `synp_p1` (x_a/x_b, R), `synp_p2` ((x_a/x_b)·x_c, R), `synp_p3` (threshold-AND, C), `synp_p4` (cyclic, C), `synp_p5` (balance-CF, C), `synp_p6a` / `synp_p6b` (named / anonymised formula, R). Task types are as proposed; E8a confirms them. P7 is excluded by the P7 policy.
- **CAP real-4** = `junglechess`, `vehicle`, `heart`, `myocardial` (all C). They are used only for success and the no-harm check.
- **Brute-force gate:** a family that OpenFE or AutoFeat recovers in ≥ 14/15 fold-episodes of A08's CPU screen is excluded from the confirmatory population. P1 is expected to fall there.

**Metrics** (R13 names).
- **`rec_sel`** (primary) and **`rec_any`**:
  - an active, **non-fallback** feature has |Spearman| ≥ 0.95 with φ on training rows, or NMI ≥ 0.9 for the rule families (P3, P5);
  - raw columns and single-column monotone transforms never count;
  - `rec_sel` requires the feature to be in the iteration that inner CV selects;
  - the unit is the fold-episode.
- **Success:**
  - `success` = finished within the 6 h cap and scored ≥ `base_no_fe` − 2 × fold SD (on junglechess: the rule is found, ≥ 0.95);
  - `func_success` = `success` **and** the selected iteration holds ≥ 1 accepted non-fallback LLM feature;
  - also `collapse` and `cap_hit`.
- **Cost:**
  - `tok_compl_<role>` (including reasoning and re-rolls; search charged to the caller), `tok_search_<role>`, `llm_calls_<role>`, `wall_s_<role>`;
  - `energy_j`;
  - `flop_proxy` = Σ_role 2 · N_active(rung of that role) · tokens.
- **Process:** `fallback_share`, `rescues_per100_<family>`, `context_errors`, `focus_implementable`, `floor_activations`, `err_per_accepted`, `fun_*`.
- **Big-model token ratio** (`big_tok_ratio`, new) = the cell run's completion tokens generated **on O120** (search charged to the caller) ÷ the paired A05 run's `tok_compl_total`.
  - For `mix_1b` the numerator is `tok_compl_extractor`; for `mix_esmall` it is `tok_compl_scientist` + `tok_compl_tester`.
  - The denominator is the all-big run, not the cell's own total, because a 0.8B Scientist spinning through retries would inflate a share-of-own-tokens metric.
  - Pairing: an f42 cell run is divided by the mean of A05 f42 r0 and r1; an f4 cell run by A05 f4 r0.
- **Role token shares** s_S, s_E, s_T (inputs to τ, §6) = the median, over the 21 A05 SYN-P 7 control runs (f42 r0, f42 r1, f4 r0), of `tok_compl_<role>` ÷ `tok_compl_total` (search charged to the caller). E11a computes them for the G2 memo.
- **No-harm column** (real-4 only): `heldout_score` and `delta_std` (= Δ_std = Δ / `sigma_d`).

**Prior (descriptive only).** The only earlier evidence is the old O20 vs O120 rebuttal arm, which used one model for all roles: 5 real datasets at one seed, held-out score (not recovery), signed mean Δ = −0.004 (`paper/reviewer_feedback/REBUTTAL_FINAL.md:55`), measured with corrupted vLLM 0.11.0 tool calls. It points to E-small ≈ all-big, so the null for C5b is plausible.

## 3. Hypotheses and interpretation (all are hypotheses)

| Outcome | Meaning | Paper wording / action |
|---|---|---|
| **Hoped** | E-small clearly loses `rec_sel`. The 1B cell stays within margin δ on `rec_sel`, has no `func_success` drop larger than δ, keeps `big_tok_ratio` ≤ τ, and neither 0.8B role is "mostly fallback" (§6). | *"Only the code-writing role needs a large model: with Scientist and Tester on a 0.8B model and the Extractor on gpt-oss-120b, planted-feature recovery stayed within δ of the all-120B system while the big model generated X% of the all-big run's completion tokens."* |
| **Honest prior** | E-small ≈ all-big, and the 1B cell is inconclusive (about 6 informative tasks × 2 seeds). | Report CIs and process metrics. Claim nothing beyond "no large drop". |
| **Null (C5b)** | E-small ≈ all-big. | *"gpt-oss-20b (3.6B active), with its own 20B summariser, already suffices as Extractor on these tasks; the Extractor's floor lies below O20, where only A17/A18 can locate it."* Claim equivalence only if the 90% CI lies inside ±δ. |
| **Opposite (a)** | The 1B cell clearly loses. | S and/or T need more capacity. The A18 per-role floors and A17's rungs say roughly how much: *"S and T need at least N B."* The per-role process metrics (generic or repeated focuses; a Tester that prunes the planted feature) say which role. |
| **Opposite (b)** | The 1B cell passes, but a 0.8B role is mostly on its fallbacks. | The pass then means *"a deterministic planner/Tester suffices"*, which supports C2, not *"a 0.8B model works"*. Read it against A12's `fx_pajM` (S and T replaced by rules, E on O120). |
| **Opposite (c)** | E-small *beats* all-big (C5b is two-sided). | Check `rescues_per100_gpt-oss`, `fallback_share` and `cap_hit` per cell before interpreting. Report it; no design change follows. |

## 4. Design

> **SR4 fallback (2026-10-19).** The control below (A05 O120) is replaced by A17 `f2_Q122`; the
> arm table becomes `mix_esmall` = Q122 / 8040 · **Q4 / 8032** · Q122 / 8040 and `mix_1b` =
> **Q0.8 / 8030** · Q122 / 8040 · **Q0.8 / 8030**, with global Q122 / 8040 in `env_overrides`
> (the launcher no longer overwrites a global given there on a mixed row; `make_manifest.py`
> admits `LLM_MODEL`/`LLM_PORT` only as a pair and only on mixed rows). The real-4 rows are cut
> (CL1). `PARALLEL=6` on the main session (the Q122 server carries every row); `search cache
> shared` is superseded by live search (Part VII.1 item 4). Pairing (by partition with the
> f2_Q122 run of the same task and seed), randomisation and failure handling are unchanged.

**Control (0 new runs).** A05 `full`, all roles and W on O120:
- SYN-P 7 at f42 r0 (`llm_seed` 42), f42 r1 (the A/A, `llm_seed` 1042) and f4 r0, i.e. 21 runs, 3 per task, exactly as C1a (A16) uses them;
- real-4 at f42 r0 and r1, i.e. 8 runs; the real-data control value is the mean of the two runs.

**Arms** (R5). `env_overrides` = `{"ROGUE_ONE_WEBSEARCH_ROUTING": "caller", "LLM_MODEL": "openai/gpt-oss-120b", "LLM_PORT": "8010"}` for both; rows on `synp_p3`, `synp_p4`, `synp_p5` also add `"ROGUE_ONE_STRATIFIED_FOLDS": "True"` exactly when the paired A05 row has it (D4).
- `caller` is already the v2.3 default; it is restated as a guard.
- The global model is O120 / 8010 in both arms. This equals A04 row 008 (`mixcheck_1b`); A04 row 007 (`mixcheck_esmall`) deliberately uses O20 / 8020 as a routing probe, which A19 does not need because SR6 has already passed. Under `caller` nothing active uses the global model, so a W call of a small-model role (E in `mix_esmall`; S, T in `mix_1b`) that reaches O120 is a routing fault, visible in the traces (`misrouted_calls`). A fault on a big-model role's W call is invisible but harmless (same model).
- **Deviation from R7:** `LLM_MODEL` / `LLM_PORT` are R3 keys, not R4 flags. They go in `env_overrides` because a mixed row has no single rung for the launcher to derive them from (same deviation as A04 rows 007–008).
- Leave `ROGUE_ONE_WEBSEARCH_MODEL` / `_PORT` **unset**. They become readable once E7 adds `"websearch"` to `AGENT_ROLES`, but they are not registry flags.

| Arm | S (`rung_S`/`port_S`) | E | T | W | Big-model roles |
|---|---|---|---|---|---|
| `mix_esmall` | O120 / 8010 | **O20 / 8020** | O120 / 8010 | follows caller | S, T (+ their W) |
| `mix_1b` | **Q0.8 / 8030** | O120 / 8010 | **Q0.8 / 8030** | follows caller | E (+ its W) |

The launcher maps `rung_*` to the R9 HF id and sets `ROGUE_ONE_SCIENTIST_MODEL`/`_PORT`, `ROGUE_ONE_EXTRACTOR_MODEL`/`_PORT` and `ROGUE_ONE_TESTER_MODEL`/`_PORT`, plus `ROGUE_ONE_RUNG_S/E/T`.

**Fixed settings** (01 §4.3; all from `scripts/agent2027/v2_defaults.env`):
- run: 10 iterations × 5 folds; `inner_cv`; rollback off;
- generation: temperature 0.7 for all roles and W; E 32768 / T 8192 / S pinned max tokens; turn and retry caps unchanged;
- R and pool mode as frozen at G1; `ROGUE_ONE_STRATIFIED_FOLDS` identical to the paired A05 row;
- logging: `ROGUE_ONE_LLM_TRACE=1`, `ROGUE_ONE_PROMPT_DUMP=1`;
- caching: `ROGUE_ONE_SEARCH_CACHE=shared`;
- launcher: `RUN_CAP_HOURS=6`, `PARALLEL=6` in total (each row uses O120, and O120 is held to 6 clients as in t17), `CPUS=12` per client (72 cores).

**Run matrix.** W&B entity `PlanBench`, project `RogueOne-AGENT2027`, group `A19`, job_type = arm, tags [`<SESSION_ID>`, `v2`, `agent2027-v2.3`].

| Group | Arm | Datasets | fold_seed (`llm_seed`) | rep | Rows | Runs | u | Run name |
|---|---|---|---|---|---|---|---|---|
| New | `mix_esmall` | SYN-P 7 | 42 (42), 4 (4) | 0 | A19-001…014 | 14 | 9.8–21 | `A19__mix_esmall__synp_<id>__f<42\|4>__r0` |
| New | `mix_esmall` | real-4 | 42 (42) | 0 | A19-015…018 | 4 | 4–6 | `A19__mix_esmall__<dataset_id>__f42__r0` |
| New | `mix_1b` | SYN-P 7 | 42 (42), 4 (4) | 0 | A19-019…032 | 14 | 9.8–21 | `A19__mix_1b__synp_<id>__f<42\|4>__r0` |
| New | `mix_1b` | real-4 | 42 (42) | 0 | A19-033…036 | 4 | 4–6 | `A19__mix_1b__<dataset_id>__f42__r0` |
| Control (reused) | A05 `full` | SYN-P 7; real-4 | 42 r0/r1, 4 r0; 42 r0/r1 | 0/1 | — | 0 new (29) | 0 | `A05__full__<dataset_id>__f<s>__r<r>` |
| Descriptive (reused) | A12 `fx_pajM` | SYN-P 7 | 42 | 0 | — | 0 new (7) | 0 | `A12__fx_pajM__synp_<id>__f42__r0` |
| Context (reused) | A17 `f2_Q0.8`, `f2_O20`; A18 `replay_Q0.8`, `replay_O20` | — | — | — | — | 0 new | 0 | A17/A18 naming |
| Sentinel | A06 `sentinel` (all O120) | G2 task | 42 (5000+k) | k | A06 row | 1 (A06 pays) | (0.7–1.0) | `A06__sentinel__<task>__f42__r<k>` |
| **Total new** | | | | | | **36** | **27.6–54** | |

Row order inside each block: SYN-P in the order p1, p2, p3, p4, p5, p6a, p6b, first at f42 and then at f4; real-4 in the order junglechess, vehicle, heart, myocardial.

**Budget arithmetic** (recomputed; identical to the breakdown card and 01 §4.9: 36 runs, 27.6–54u; CL1 −8 runs, −8 to −12u).
- **Pricing.** u is priced as box occupancy. A19 holds all 4 GPUs but runs only 6 clients, because every row uses the one O120 server. At the R0 rate of 2 u/h for 6 clients, a run costs (its wall-hours) / 3 in u, and a run that hits the 6 h cap costs 2u. O120 at TP2 decodes more slowly than t17's TP4, which pushes toward the high end.
- **Low end:** 28 SYN-P × 0.7u + 8 real × 1.0u = 19.6 + 8 = **27.6u**.
- **High end:** 36 × 1.5u = **54u**, an average run of 4.5 h. This is the case where the 0.8B Scientist (100 attempts × 200 turns) pushes many `mix_1b` runs toward the cap.
- **Box time:** 27.6–54u ÷ 2 u/h = 13.8–27 h; at 0.7 utilisation, 19.7–38.6 wall-hours (the breakdown's "1–2 days of the box").
- **CL1** (drop the 8 real rows): 28 runs, 27.6 − 8 = 19.6u to 54 − 12 = 42u; 14–30 wall-hours.
- **Not counted here:** the session sentinel (A06 pays, 0.7–1.0u), the 29 control runs (A05 pays), any SR6 rerun of A04 session C (reserve, 2–4u), and the optional SR4 `fx_pajM` re-run at Q122 (reserve, 4.9–7u; §6).

**Pairing and randomisation.**
- Each cell row is paired with A05 by (`dataset_id`, `fold_seed`): the same outer partition.
- `llm_seed` = `fold_seed` matches A05 r0, but sampling is not byte-reproducible under load, so the pairing is by partition only.
- Both cells and the sentinel are interleaved in one session in a random `order_key` (RNG seeded by the session ID).
- **Cross-family rule:** each cell is compared **only with all-big**, never with the other cell.

**Failure handling (ITT; 01 §4.5).**
- **Crash** (non-zero rc, or no `fold_aggregation_summary.csv`): retried **once, alone**, at the end of the session.
- **Persistent failure or 6 h cap hit:** an ITT failure. It is never dropped or retried. It contributes `rec_sel = rec_any = 0` for all 5 fold-episodes and `success = func_success = 0`. Its held-out score is imputed at `base_no_fe` under augment, otherwise at the majority-class or mean predictor.
- **Infrastructure fault** (a server dies, W&B is unreachable): not ITT. The row is re-queued with the same `row_id` and seeds in a rerun session, by Sat 31 at the latest, and the requeue is written in `notes`.
- Failure and cap-hit rates are shown next to every number.

## 5. Tasks

### 5.1 Prerequisites (verify before launch)

- [ ] **A19-T0** [type: code] (0.1 d, due Mon 12 Oct, G0a, inside E3; freeze item 8) — Make sure E3 logs what A19's fallback and routing metrics need, because E3 freezes before A19 is built:
  - a per-iteration `focus_source` ∈ {`agent`, `no_tool_fallback`, `heuristic`} in the focus JSONL (`sci:528-529`, `:576-577`, `:592`); E3's list names only the heuristic counter;
  - the Extractor baseline-fallback count per iteration (`ext:996-1002`);
  - per trace record: the calling role (W calls carry the caller via E4's contextvar), `base_url`, the requested `model` and the response `model`.

  *Done when:* the G0a smoke's `run-output` has `focus_source` on every focus row, the fallback count in the summary (zero allowed) and the four trace fields on every trace record.
- [ ] **A19-T1** [type: decision] (0.15 d, due Fri 16 Oct) — Write the A19 section of `docs/agent2027/preregistration.md` (X10). It must contain:
  - C5b and its GLMM, and the C5b drop rule (§6);
  - the 1B verdict rules, the margin δ formula (A09/A12), the `big_tok_ratio` formula, the s_E definition and the τ rule (§2, §6);
  - the fallback-interpretation rule and the `mix_1b` stop rule with its default;
  - the cross-family rule; real data as CIs only;
  - the SR4 fallback rungs and the `fx_pajM` default under SR4;
  - the disclosure that R was frozen on O120 (A02), so small models might prefer another R.

  *Done when:* the section is in the commit tagged `agent2027-prereg`.
- [ ] **A19-T2** [type: decision] (0.02 d, due Fri 16 Oct) — Record D5. A19 serves O120 at TP2 whatever D5 says. If M0 runs at TP4, draft the disclosure sentence. *Done when:* D5 and the disclosure (if needed) are in the pre-registration.
- [ ] **A19-T3** [type: data] (0.05 d, due Mon 19 Oct, G2) — Check that the G2 memo `ANALYSIS__g2_stats__20261019` (E11a) gives:
  - M0's median per-role completion-token shares s_S, s_E, s_T over the 21 A05 SYN-P 7 control runs (§2; search charged to the caller);
  - τ (§6);
  - the informative SYN-P 7 tasks;
  - `sigma_d` for real-4;
  - the sentinel task.

  *Done when:* τ is in a dated pre-registration addendum before A19 launches.
- [ ] **A19-T4** [type: data] (0.05 d, due Tue 20 Oct) — Confirm the 29 A05 control runs. Each must be `finished` or ITT, have `tok_compl_<role>` > 0 for all roles and have a `run-output` artifact. *Done when:* the list is in the A19 analysis notes.
- [ ] **A19-T5** [type: decision] (0.02 d, Wed 21 / Thu 22 Oct, before launch) — Read A04 (A04-T19).
  - **SR4:** O20 passed, so use the O120/O20 pair (otherwise §6, fallback).
  - **SR1:** the Q0.8 id; `Qwen/Qwen3-0.6B` if the family switched.
  - **SR6:** routing passed on v2.3 for `mixcheck_esmall` and `mixcheck_1b`, for S, E, T **and** W.
  - The O120, O20 and Q0.8 revisions are in `rungs.csv`.
  - **Launcher contract:** A04 session C's job log (`~/ro27_S1021a.job.log` or its renamed ID) shows that E7's `ladder_session.sh` does what `ro27_session_launch.sh` does (01 §5.2 step 4): in-container vLLM version assert, client and server image IDs, energy sidecar, per-row `ROGUE_ONE_<ROLE>_MODEL` / `_PORT` exported from `rung_*` / `port_*`, and the closing `=== RO27_SESSION_END <SESSION_ID> rc=<rc> ===` line.

  *Done when:* all five are confirmed and noted in the A19 analysis notes. **No launch without SR6.**
- [ ] **A19-T6** [type: data] (0.02 d; A08 part Wed 21 Oct, δ part Fri 23 Oct after 18:00) — Confirm the A08 brute-force-gate result (the excluded SYN-P families; read on Tue 13 Oct, M5S), and that A09-T15 has committed δ to the pre-registration (due Fri 23 Oct, 18:00). *Done when:* both are in the analysis notes. δ is needed before the 1B cell is unblinded (A19-T18), not before launch.

### 5.2 Implementation (E7 owns the code; A19 verifies it)

- [ ] **A19-T7** [type: code] (0.15 d, due Wed 21 Oct, inside E7) — Extend `tests/unit/utils/test_per_agent_model.py`:
  - (i) under `caller` with S = Q0.8/8030, E = O120/8010 and T = Q0.8/8030, the `WebSearchAgent` owned by each role builds its client on that role's endpoint and model (today `kn/web_search.py:189`, `:238`);
  - (ii) under a uniform config, endpoint and model equal the global ones;
  - (iii) `global` reproduces today's behaviour;
  - (iv) `model_for` still raises for an unknown role.

  *Done when:* `pytest -m unit tests/unit/utils/test_per_agent_model.py` passes on the commit tagged `agent2027-v2.3` (the repo has no CI workflow, so paste the pytest summary line into the E7 commit message), and the E5 diff of Full-v2 against `prompt-baseline-v2.0` is byte-identical.
- [ ] **A19-T8** [type: config] (0.05 d, due Wed 21 Oct; Thu 22 morning if A04 session C slips) — Check that E1 writes, for a mixed row:
  - `model_S/E/T/W`, `rung_S/E/T` and `server_topology`;
  - `flags/ROGUE_ONE_WEBSEARCH_ROUTING=caller`;
  - `flags/ROGUE_ONE_<ROLE>_MODEL` and `_PORT` for each role.

  *Done when:* all of them appear on A04's `mixcheck_*` runs in W&B.
- [ ] **A19-T9a** [type: analysis] (0.1 d, due Wed 21 Oct, before launch; reuse A04's `stage0_table.py` SR6 logic) — Add `misrouted_calls` to `analysis/process_metrics.py`: trace records whose response `model` (or `base_url`) ≠ the intended model (port) of the calling role, W included, counted per role. It must run alone on one `run-output` artifact (`--misrouted-only`), because T15 and T17 use it during and right after the session. *Done when:* a toy trace with one planted misroute gives 1, and the script gives `misrouted_calls` = 0 on both A04 `mixcheck_*` artifacts.
- [ ] **A19-T9b** [type: analysis] (0.2 d, due Tue 27 Oct, inside E11b) — Add to `analysis/process_metrics.py`:
  - `fallback_share_<role>`: S = share of iterations whose `focus_source` ≠ `agent` (`sci:547-592`; A19-T0); T = share of Tester stages with `tester_stage/assessment_status` or `tester_stage/pruning_status` = `failed` (`tst:1498-1503`, `:1593-1595`, logged at `:2215-2216`); E = share of iterations with a baseline fallback (`ext:996-1002`);
  - `planted_pruned`: fold-episodes in which a feature that recovers φ (the `rec_any` detector rule) was pruned by T;
  - `big_tok_ratio` (§2).

  *Done when:* a toy example passes and the script runs end to end on A04's `mixcheck_*` and A05's artifacts.
- [ ] **A19-T10** [type: config] (0.1 d, due Wed 21 Oct) — Run `scripts/agent2027/make_manifest.py --ablation A19` to write `manifests/agent2027/A19_CAP-2.csv` with rows A19-001…036 (§4). Rows 015–018 and 033–036 get `notes=CL1-cuttable`.
  ```
  row_id,ablation_id,arm_id,dataset_id,task_type,fold_seed,llm_seed,replicate,is_sentinel,session_id,order_key,image_tag,rung_S,rung_E,rung_T,port_S,port_E,port_T,env_overrides,cap_hours,expected_u,notes
  A19-001,A19,mix_esmall,synp_p1,regression,42,42,0,0,,,agent2027-v2.3,O120,O20,O120,8010,8020,8010,"{""ROGUE_ONE_WEBSEARCH_ROUTING"": ""caller"", ""LLM_MODEL"": ""openai/gpt-oss-120b"", ""LLM_PORT"": ""8010""}",6,0.85,pair A05 full synp_p1 f42 r0+r1
  A19-019,A19,mix_1b,synp_p1,regression,42,42,0,0,,,agent2027-v2.3,Q0.8,O120,Q0.8,8030,8010,8030,"{""ROGUE_ONE_WEBSEARCH_ROUTING"": ""caller"", ""LLM_MODEL"": ""openai/gpt-oss-120b"", ""LLM_PORT"": ""8010""}",6,0.85,pair A05 full synp_p1 f42 r0+r1
  ```
  `expected_u` is 0.85 for SYN-P rows and 1.0 for real rows (planning midpoints, as in A05; the A19 range is 0.7–1.5u). `task_type` and the fold scheme must equal the paired A05 row: under D4 the `synp_p3`/`p4`/`p5` rows (A19-003…005, 010…012, 021…023, 028…030) carry `"ROGUE_ONE_STRATIFIED_FOLDS": "True"` in `env_overrides`. *Done when:* the file is committed and pushed, it validates against R7 (22 columns, 36 unique `row_id`s, `cap_hours`=6), and a dry run gives unique names `ro27-<session_id>-a19-0nn`.

### 5.3 Execution on CAIR

> **SR4 fallback (2026-10-19).** A19-T11 step 1 is done (`manifests/agent2027/sessions/S1022a.csv`,
> 28 rows; `S1022d.csv`, 8 rows = A20-001…007 + A06-103 first wave). A19-T13 no longer uses
> `ladder_session.sh` or the `SERVERS=` placeholder: launch `S1022d` (Q4) and `S1022f` (Q0.8) as
> holders first, then `S1022a` (Q122 TP2) with `WAIT_PORTS="8032 8030"`; every launch goes
> through `ro27_session_launch.sh` with `CLIENTS_SCRIPT=~/ro27_hold_clients.sh` (exact lines:
> `docs/agent2027/sessions/LAUNCH.md`). Server names: `rogueone-vllm-q122-s1022a`,
> `rogueone-vllm-q4-s1022d`, `rogueone-vllm-q0.8-s1022f`; `server_topology` per row is
> `mixed:S=…;E=…;T=…;G=Qwen/Qwen3.5-122B-A10B-FP8@8040`. A19-T14 loops over these three server
> names. The "one Rogue One job at a time" and "≤ 4 GPUs" sentences are superseded (D12; Henrik's
> Oct 2026 GPU allowance); the trio needs 4 GPUs.

**Hard rules** (01 §5.1):
- **Containers only.** The host only issues `docker` / `docker compose` (`-T` over SSH) and runs the `~` launchers. `nvidia-smi`, `docker ps`, `docker logs` and `tail` are allowed for checks. Never run host Python.
- **GPU pre-flight before every GPU container.** If there are not enough free GPUs, **stop and report**.
- **Never more than 4 of the 8 H100s** without Henrik's explicit approval.
- **W&B is the system of record.** Runs are stateless. `WANDB_API_KEY` comes from `<CAIR_HOME>/code/PlanBench/.env`; Serper and LLM settings come from the image `.env` / `.secret.env`.
- **Long jobs:** detached `nohup`, with the log in `~`.
- `<ROGUEONE_CAIR_PATH>` = `<CAIR_HOME>/code/RougeOne` is **INFERRED** from `paper/scripts/clients_t16_rerun.sh:59,135`. Verify it once with `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "ls -d ~/code/RougeOne"`. The git remote `git@github.com:HenrikNorkart/RougeOne.git` is also inferred (01 §5.2).
- **Session ID.** `S1022a` below is the worked example for a Thu 22 launch. Use `S<MMDD><next free letter>` for the actual launch day (A20 uses `S1022c`/`S1022d`/`S1022e`, A16's pilot is `S1022b`) and substitute it everywhere, lower-cased in container names.
- **One Rogue One job at a time.** A19 needs the **whole 4-GPU slot**, so it starts only after the preceding session (`S1021a`, or A20/A17's O20 session, or A17's Q122 session) has ended and been torn down, never alongside it.

- [ ] **A19-T11** [type: run] (0.1 d, launch day, Thu 22 – Tue 27 Oct) — Plan and stage the session.
  1. Locally: `python scripts/agent2027/make_manifest.py --plan-session S1022a --rows A19-001…A19-036 <next A06 sentinel row>`; commit and push `agent2027/v2`.
  2. Sync the server:
     `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15' git fetch git@github.com:HenrikNorkart/RougeOne.git agent2027/v2 && git checkout -B agent2027/v2 FETCH_HEAD && git log -1 --format='HEAD: %h %s'"`
  3. Check the image (it should exist from A04 session C and `S1021a`):
     `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker image inspect rogue_one:agent2027-v2.3 --format '{{.Id}}'"`
     A19 stays on v2.3 even if v2.4 exists, because SR6 was verified on v2.3. If the image is absent, rebuild it from the tag, not from the branch tip:
     `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=15' git fetch git@github.com:HenrikNorkart/RougeOne.git tag agent2027-v2.3 && git checkout agent2027-v2.3 && test ! -e .dev.env && docker build -t rogue_one:agent2027-v2.3 . && git checkout agent2027/v2 && docker image inspect rogue_one:agent2027-v2.3 --format '{{.Id}}'"`
  4. Install the launchers and manifests in `~` (01 §5.2 step 3):
     `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && cp scripts/agent2027/ro27_session_launch.sh scripts/agent2027/ro27_clients_manifest.sh scripts/agent2027/ladder_session.sh ~/ && cp paper/scripts/vllm_session.sh ~/ro27_vllm_session.sh && mkdir -p ~/ro27_manifests && cp -r manifests/agent2027/. ~/ro27_manifests/"`

  *Done when:* `~/ro27_manifests/sessions/S1022a.csv` exists with 37 rows (36 A19 + 1 sentinel), the server HEAD equals the pushed commit, and the v2.3 image ID is recorded in the session notes.
- [ ] **A19-T12** [type: run] (0.01 d, launch day, right before T13) — GPU pre-flight:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader && echo '--- compute apps ---' && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader && echo '--- containers ---' && docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'"
  ```
  *Done when:* 4 GPUs show < 2,000 MiB used, and no `ro27-*`, `rogueone-vllm-*`, `llmfe-*` or `caafe-*` container of ours is running. **Otherwise STOP** and report per-GPU memory and utilisation, the compute apps and the containers to Henrik.
- [ ] **A19-T13** [type: run] (0.01 d, launch day, right after T12) — Launch the three-server session through E7's `ladder_session.sh` (ΣTP = 4; it refuses more). Use one `API_KEY` for all servers (the `vllm_session.sh:41` default, which the clients' `LLM_API_KEY` must equal), `EXTRA_ARGS` from `rungs.csv`, and `MAX_MODEL_LEN=131072`:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "SESSION_ID=S1022a SESSION_FILE=\$HOME/ro27_manifests/sessions/S1022a.csv SERVERS='O120:<g1>,<g2>:8010 O20:<g3>:8020 Q0.8:<g4>:8030' API_KEY=rogueone-nestedcv-key IMAGE_RO=rogue_one:agent2027-v2.3 PARALLEL=6 CPUS=12 RUN_CAP_HOURS=6 RUNROOT=\$HOME/rogueone-runs-agent2027 MAX_MODEL_LEN=131072 WAIT_MAX_MIN=0 EAGER=0 LOG=\$HOME/vllm_session_S1022a.log nohup bash ~/ladder_session.sh > ~/ro27_S1022a.job.log 2>&1 & echo launched; sleep 5; tail -5 ~/ro27_S1022a.job.log"
  ```
  The `SERVERS` syntax is a placeholder: use the usage line of the merged E7 script (A04 session C used the same script). GPU ids come from T12; the four ids must be exactly the four free GPUs that T12 showed.
  - Names: servers `rogueone-vllm-o120-s1022a`, `rogueone-vllm-o20-s1022a` and `rogueone-vllm-q0.8-s1022a`; clients `ro27-s1022a-<row_id>`; energy sidecar `ro27-s1022a-energy`.
  - `server_topology` is logged as `O120-TP2:8010|O20-TP1:8020|Q0.8-TP1:8030`.

  *Done when:* the job log shows the vLLM version assert, both image IDs, three healthy servers and the first 6 rows started.
- [ ] **A19-T14** [type: run] (0.05 d, launch day, about 45 min after T13) — Early transport check (vLLM's default access log prints one `POST /v1/chat/completions` line per request):
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "for s in o120 o20 q0.8; do echo \$s; docker logs --since 40m rogueone-vllm-\$s-s1022a 2>&1 | grep -c 'POST /v1/chat/completions'; docker logs --since 40m rogueone-vllm-\$s-s1022a 2>&1 | grep -c -E ' 404 |does not exist'; done"
  ```
  Also query W&B (01 §5.2 step 7, inside a container) for the `model_S/E/T/W`, `rung_S/E/T` and `flags/*` config of the live runs. Server logs cannot attribute a call to a role, so this only catches gross faults; per-role routing is checked from traces in T15 and T17. *Done when:* each server whose arm has a live row shows > 0 requests (if the first wave holds only one arm, repeat once the other arm's first row has run 40 min), no server shows 404 / "does not exist", and the W&B config of every live run matches its manifest row. If not, **stop the session** (stop the nohup launcher, `docker stop` the `ro27-s1022a-*` clients; the launcher tears down the servers) and fix it before any further run.
- [ ] **A19-T15** [type: run] (0.1 d spread over the session, launch day + up to 2 days; ends by Sat 31 Oct) — Monitor in the background:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 CAIR-gpu14 "while ! grep -q 'RO27_SESSION_END' ~/ro27_S1022a.job.log 2>/dev/null; do sleep 120; done; tail -30 ~/ro27_S1022a.job.log"
  ```
  - Exit code 255 means the network dropped, not the job; reconnect.
  - A W&B `running` run with no container is a zombie.
  - **First-wave routing check:** as soon as the first `mix_esmall` row and the first `mix_1b` row have finished, run `analysis/process_metrics.py --misrouted-only` (A19-T9a) on their `run-output` artifacts, inside a `rogue_one:agent2027-v2.3` container with `--env-file <CAIR_HOME>/code/PlanBench/.env` and `-v <ROGUEONE_CAIR_PATH>/scripts/agent2027:/workspace/scripts/agent2027:ro` (the A17-T13 pattern; T9a may postdate the v2.3 build). Never on the host. Any `misrouted_calls` > 0: stop the session as in T14.
  - **Stop-rule checkpoint:** when 6 `mix_1b` rows have completed (expected about 12 h in), apply the `mix_1b` stop rule (§6).

  *Done when:* `=== RO27_SESSION_END S1022a rc=0 ===` appears, and the first-wave check and the stop-rule outcome are written in the session notes.
- [ ] **A19-T16** [type: run] (0.05 d, the day the session ends) — Close out:
  - each crashed row got exactly one solo retry; `CAP_HIT` rows were not retried;
  - the tear-down check shows no `ro27-` / `rogueone-vllm-` containers and the GPUs are free (01 §5.2 step 6);
  - infrastructure faults are planned into a rerun session (Fri 30 – Sat 31 at the latest), with the same `row_id`s and seeds, a new sentinel row, and the same pre-flight, launch and tear-down steps (T11–T16).

  *Done when:* 36 A19 runs (or 36 minus the rows the stop rule removed) are in W&B, each `finished` or ITT, with every R12 config key and a `run-output` artifact.

### 5.4 Analysis

- [ ] **A19-T17** [type: analysis] (0.1 d, within 1 day of session end; by Sat 31 Oct) — Validity checks, in containers (01 §5.2 step 7 pattern):
  - `misrouted_calls` = 0 on all 36 runs (A19-T9a);
  - `analysis/sentinel_drift.py --session S1022a`. A FLAG that coincides with TP2-vs-TP4 is reported as a topology effect (A06 rule).

  *Done when:* both are logged as `ANALYSIS` runs. **Any misrouted run is void**: rerun it after the E7 fix (before DF, Sat 31 Oct 24:00) and record it in `notes`.
- [ ] **A19-T18** [type: analysis] (0.25 d; preliminary within 2 days of session end and not before δ is committed (Fri 23, 18:00), final Sun 1 Nov) — Run `analysis/fetch_wandb.py --group A19` (plus the paired A05 and A12 rows), then `analysis/recovery_detector.py` on every `run-output`. *Done when:* `rec_sel`, `rec_any` and `iter_first_rec` exist for 140 cell fold-episodes (14 SYN-P runs × 5 folds × 2 cells) and 105 control fold-episodes (21 × 5), minus gated tasks.
- [ ] **A19-T19** [type: analysis] (0.25 d, Nov 1–3) — Test **C5b** with `analysis/glmm_holm.py`.
  - Logistic GLMM `rec_sel ~ arm + (1|task) + (1|task:seed)` on fold-episodes, arm ∈ {`full`, `mix_esmall`}. The population is SYN-P 7 minus the gated tasks; Full contributes 3 runs per task.
  - Report the two-sided p, the marginal risk difference with its 95% CI and, descriptively, the 90% CI against ±δ.
  - Separation fallback: the pre-registered exact test stratified by task.
  - Holm is applied across the surviving family.
  - Sensitivity analyses: without the A/A; excluding fallback-only selected runs.

  *Done when:* `ANALYSIS__glmm_holm__<YYYYMMDD>` holds the C5b row.
- [ ] **A19-T20** [type: analysis] (0.25 d, Nov 1–3) — Read the **1B cell** (outside Holm). Fit the same two-arm GLMM with `mix_1b` and compute:
  - the `rec_sel` risk difference with 90% and 95% CIs against δ;
  - the `func_success` difference (14 vs 21 SYN-P runs) with an exact CI;
  - `big_tok_ratio`: the geometric mean over the paired SYN-P (task, seed) cells (f42 against the mean of r0 and r1), with a 95% bootstrap CI that resamples tasks. The primary population is pairs where both runs completed; the sensitivity analysis uses all pairs at the tokens actually spent;
  - `fallback_share_<role>`, `planted_pruned`, `focus_implementable`, `floor_activations` and `rescues_per100_<family>` per cell.

  Apply the §6 verdict rules. *Done when:* the verdict (pass / pass-via-fallbacks / loses / inconclusive) is logged.
- [ ] **A19-T21** [type: analysis] (0.15 d, Nov 1–3) — Descriptive context. The S+T ladder at f42: 0B (`fx_pajM`, A12), 0.8B (`mix_1b`) and 120B (`full`). Next to it: A17 `f2_Q0.8` and `f2_O20` (all roles small), and A18 per-role floors for Q0.8 (S, T) and O20 (E). Also report `mix_esmall`'s `big_tok_ratio` (≈ 1 − s_E by construction). *Done when:* the role-grid table is logged as an `ANALYSIS` run (group `ANALYSIS`, tag `A19`).
- [ ] **A19-T22** [type: analysis] (0.1 d, Nov 1–3) — Real-4:
  - `success`, `func_success`, `collapse` and `cap_hit` per cell;
  - `delta_std` (Δ_std = Δ `heldout_score` / `sigma_d`) against the mean of A05 f42 r0 and r1, with a CI only;
  - the junglechess rule found (≥ 0.95) per fold.

  *Done when:* the table is logged; no inferential claim is made.
- [ ] **A19-T23** [type: analysis] (0.1 d, Nov 1–3) — Hand the per-cell points to A23 (`analysis/pareto.py`; `flop_proxy` with each role's N_active; per-server energy apportioned by the tokens that server produced) and to A21 (iso-cost truncation of Full). *Done when:* both scripts read the A19 rows.

### 5.5 Paper outputs

- [ ] **A19-T24** [type: writing] (0.1 d, Nov 4–6) — Table 2:
  - **C5b row:** RD with 95% CI, raw p, Holm-adjusted verdict, `cap_hit` and `fallback_share`;
  - **1B row:** labelled "outside Holm": `rec_sel` RD against δ, `func_success`, and `big_tok_ratio` against τ.

  *Done when:* both rows are in the draft.
- [ ] **A19-T25** [type: writing] (0.1 d, Nov 4–6) — Fig. 3 inset: a role grid (S/E/T rung per cell) × {`rec_sel`, `func_success`, `big_tok_ratio`}, with all-big and `fx_pajM` as anchors. *Done when:* the figure is generated by a script from W&B.
- [ ] **A19-T26** [type: writing] (0.1 d, Nov 4–6) — Write the C5 sentence from §3 that matches the verdict, and the "1B vs 100B" paragraph together with A17 and A18. Always state active parameters (5.1B / 3.6B / 0.8B), the fallback shares and the cap-hit rates. *Done when:* the text is in the draft.
- [ ] **A19-T27** [type: writing] (0.15 d, by Thu 12 Nov) — Appendix and replication package:
  - the per-task table, the routing audit (`misrouted_calls`) and the manifest;
  - **disclosures:** cross-family mixing (Qwen S/T with a gpt-oss E); TP2 vs M0's topology (D5); server load differs from M0 (wall clock and energy not comparable, tokens are); R was frozen on O120; the Serper cache lowers search variance; Qwen XML calls are not rescued.

  *Done when:* all of this is in the package.

## 6. Decision rules and gates (pre-registered in X10)

> **SR4 fallback (2026-10-19).** The "SR4 fallback" paragraph below is now in force; τ = 0.60 is
> fixed (Amendment 2). Every rule, threshold and drop date in this section is unchanged. The
> paragraph's schedule sentence ("right after A17's Q122 session … within Sat 24 – Wed 28") is
> replaced by Henrik's slot from Thu 22 with its own Q122 server (`S1022a`); A17's Q122 session
> `S1024a` runs separately, so A19 is not the "single Q122 session" and its sentinel is the
> `sentinel_Q4` in the Q4 holder `S1022d`. The `mix_1b` stop rule's re-plan reads "Q122 TP2 + Q4
> TP1 (3 GPUs)" instead of "O120 TP2 + O20 TP1".

**C5b (Holm):**
- **"E-small loses"** if the Holm-adjusted p < 0.05 and RD < 0;
- **"E-small better"** if the Holm-adjusted p < 0.05 and RD > 0;
- otherwise **"no detectable difference"**. It becomes "equivalent within δ (descriptive)" only if the 90% CI lies inside ±δ.

**Margin δ** = max(0.25 × gap, 0.10), rounded to 3 decimals, where gap = pooled `rec_sel` of A09 `oracle` minus pooled `rec_sel` of A05 Full f42 r0 + r1 over the population tasks (ITT zeros included); δ = 0.10 if gap ≤ 0 (A09 §6, shared with A12-T1). A09-T15 commits it by Fri 23 Oct, 18:00, before FX or A19 recovery is unblinded. A19 uses the same δ on the `rec_sel` and `func_success` scales.

**τ (cost threshold, fixed at G2 from the §2 definition of s_E):**
- τ = 0.60 if M0's median s_E ≤ 0.50;
- otherwise τ = s_E + 0.10. The paper then states that the planned 60% target (master plan CAP-2) left less than 0.10 headroom over the Extractor's own share, or was unattainable by construction if s_E ≥ 0.60.

**1B verdict** (outside Holm). The four labels are mutually exclusive:
- **loses:** the `rec_sel` 95% CI lies wholly below 0 **and** the `rec_sel` 90% CI is not inside ±δ;
- **pass:** the `rec_sel` 90% CI lies inside ±δ, **and** the `func_success` point Δ ≥ −δ, **and** the 95% upper bound of `big_tok_ratio` is ≤ τ, **and** neither fallback share exceeds 0.5. If the 95% CI also lies wholly below 0, the label stays "pass" and the text reports "a detectable drop of RD, within δ";
- **pass via fallbacks:** all pass conditions hold except that `fallback_share_scientist` > 0.5 (share of iterations whose `focus_source` ≠ `agent`) or `fallback_share_tester` > 0.5 (share of Tester stages with status `failed`), each pooled over all `mix_1b` SYN-P iterations. It is reported with the "deterministic parts suffice" reading (§3, Opposite b);
- **inconclusive:** anything else; the failed condition(s) are logged with the label.

**`mix_1b` stop rule** (budget guard, outside Holm):
- Checked once, when 6 `mix_1b` rows have completed (expected about 12 h in). If ≥ 5 of those 6 are `CAP_HIT`, the **default** is to stop the remaining `mix_1b` rows; Henrik may override within 2 h, in writing in the session notes. To stop: stop the nohup launcher after the in-flight rows finish, tear down, and re-plan the outstanding `mix_esmall` rows plus a new sentinel row in a new session (O120 TP2 + O20 TP1; 3 GPUs), with the same pre-flight.
- The 1B result is then reported as "S+T at 0.8B cannot complete the loop within 6 h (k/n cap hits)"; rows not run are reported as not run (not ITT).
- C5b is unaffected.

**SR4 fallback** (O20 failed Stage 0; A04 SR4):
- big = Q122 (`Qwen/Qwen3.5-122B-A10B-FP8`, TP2, 8040), small = Q4 (`Qwen/Qwen3.5-4B`, TP1, 8032), Q0.8 unchanged (TP1, 8030); ΣTP = 4:
  - `mix_esmall` = S, T on Q122 and E on Q4;
  - `mix_1b` = S, T on Q0.8 and E on Q122;
  - global `LLM_MODEL` / `LLM_PORT` = Q122 / 8040.
- The reference becomes A17 `f2_Q122` (the same SYN-P 7 × {42, 4} + real-4 × 42), at no extra cost. It has no A/A, so Full contributes 2 runs per task, not 3, and the `big_tok_ratio` denominator for f42 is the single f42 run.
- τ stays as fixed at G2 from M0 (O120 shares); the paper discloses this.
- The session runs right after A17's Q122 session, or shares its Q122 server in place of A17's second Q122 server (A17 then runs 1 × Q122 TP2); either way within Sat 24 – Wed 28. If it shares the Q122 session, no sentinel is needed (01 §4.8, "all Q122 cells run in one session").
- The arm IDs stay the same; the `rung_*` columns carry the substitution.
- The `fx_pajM` comparison is at O120, so it no longer anchors the S+T ladder. The master plan (§5.2 CAP-2) and A04 SR4 say the 0B cells are re-run at Q122, but T1-minimum budgets no such runs. Default: the comparison is dropped. Re-running `fx_pajM` at Q122 on SYN-P 7 × 42 (7 runs, 4.9–7u) is paid from the reserve only if Henrik approves at G4 (Tue 27).

**C5b drop rule (new; pre-register in X10):** C5b is dropped only if it has no data, i.e. SR6 still fails on Tue 27 or no whole-box slot exists by L-LAST (Thu 29). Holm is then recomputed over the remaining contrasts (01 §4.7 drop-rule convention).

**Gates and kill dates:**
- Launch only after **G0b-1** (v2.3, Full byte-identical), the A04 **SR6** pass (A19-T5), the **G2** τ addendum (A19-T3), A19-T9a, and the tear-down of the preceding session.
- Unblind recovery (A19-T18) only after δ is committed (A09-T15, Fri 23 Oct, 18:00).
- **CL1** is applied at G1 (Fri 16) if the re-tier fires (CL1–CL4 together).
- If A19 has not launched by **Tue 27 Oct**, apply CL1 and launch by **Wed 28** at the latest (28 runs, 14–30 wall-hours, so it ends before DF). **L-LAST Thu 29** is the absolute limit.
- If SR6 still fails on Tue 27, or the box cannot be had by Thu 29: the C5b drop rule above applies.
- **DF:** Sat 31 Oct, 24:00. Rows not finished by DF are "not run" (not ITT) and reported as such.

## 7. Risks and cut levers

- **Low power.** The effective n is about ≤ 6 informative tasks × 2 seeds, so a Holm threshold as low as 0.005 needs a large within-task effect, and "inconclusive" is likely. The framing therefore leans on the token ratio and the process metrics.
- **Cost may sit at the 54u top** if the 0.8B Scientist loops through 100 retries. Mitigations: the stop rule and the 6 h cap. A CL1 cut saves 8–12u.
- **Routing faults.** E7 lands after the freeze. Guards: A19-T7, SR6 (A19-T5), T9a, T14, the first-wave check in T15, and T17.
- **Topology.** A19 runs O120 at TP2. If D5 keeps M0 at TP4, disclose it. The session sentinel bounds the effect.
- **Cross-family mixing.** Qwen and gpt-oss use different chat templates and transport rescues. Each cell is compared only with all-big.
- **The cost hypothesis depends on s_E.** The "E+T ≈ 95% of tokens" figure is an unmeasured RogueTwo proxy; τ is set from M0 at G2.
- **Small-model failure modes:** Qwen XML calls are not rescued; `context_errors` on myocardial (91 columns); runs kept alive by fallbacks. These are reported per role.
- **Schedule (unresolved across documents).** 01 §6.2 lists A19 on Thu 22, but the box is then still busy: `S1021a` (A12 block 1 + A13 ± A14) runs until Thu 22 evening – Fri 23 afternoon; A20/A17 O20 (`S1022c`/`S1022d`), A18 small rungs and A16's pilot are also planned for Thu 22; `S1023a` (A12 block 2 + A16 SYN-P) starts Fri 23; A17's Q122 session holds 4 GPUs Sat 24 – Wed 28; A16 Core-10 runs Mon 26 – Thu 29. The breakdown's GPU plan books CAP-2 in W3 (Mon 26 – Sat 31). A19 needs the whole 4-GPU slot for 20–39 h, and only one Rogue One job may run at a time, so Henrik must place it explicitly in the session plan (by G2, Mon 19). Planned window: Thu 22 – Tue 27; Wed 28 with CL1.
- **Cut levers:** **CL1** (CAP-2 on SYN-P only) is the **first** pre-declared cut. A19 is not in the never-cut list.

## 8. Definition of done

- [ ] E3 focus-source and trace fields present in the G0a smoke (T0).
- [ ] Pre-registration section (T1) and the G2 τ addendum (T3) committed before launch; δ committed before unblinding (T6).
- [ ] E7 tests pass; SR6 passed; `misrouted_calls` script ready (T9a); `A19_CAP-2.csv` committed (T7–T10).
- [ ] 36 `A19__mix_*` runs (28 under CL1) in W&B, each finished or ITT, with `run-output`, the full R12 config and `misrouted_calls` = 0.
- [ ] Sentinel check, C5b GLMM/Holm row, 1B verdict, `big_tok_ratio`, per-role fallback table and real-4 table logged as `ANALYSIS` runs.
- [ ] Table 2 rows, Fig. 3 inset, C5 text, appendix and disclosures done; A21/A23 hand-offs delivered.

---

**New names introduced (confirm in X10 / at E7 merge):**
- metrics `big_tok_ratio`, `misrouted_calls`, `planted_pruned` and `fallback_share_<role>`; the focus-log field `focus_source` (A19-T0, E3);
- the s_E definition (median over the 21 A05 SYN-P 7 runs) and the τ rule (0.60, or s_E + 0.10);
- the 1B verdict rules (mutually exclusive labels), including the `func_success` point-Δ ≥ −δ clause and the 0.5 fallback threshold;
- the `mix_1b` stop rule (≥ 5 of the first 6 completed `mix_1b` rows `CAP_HIT`; default stop, Henrik may override);
- the C5b drop rule (no data by L-LAST); R14 lists drop rules only for C4b, C1a and C4a;
- the SR4 default of dropping the `fx_pajM` anchor (the master plan and A04 SR4 say "re-run at Q122");
- the analysis flag `process_metrics.py --misrouted-only`;
- the manifest file `A19_CAP-2.csv` (`<AID>_<short>` with R1's short ID verbatim; A04, A17 and A20 drop the hyphen: `A04_CAP-0.csv`, `A17_CAP-1.csv`, `A20_CAP-3.csv`; align when E1's `make_manifest.py` fixes the rule);
- the example session `S1022a`;
- the `SERVERS` launch syntax of `ladder_session.sh` (a placeholder for E7's interface);
- the `server_topology` string format;
- global `LLM_MODEL` / `LLM_PORT` = O120 / 8010 in `env_overrides` (R3 keys where R7 expects R4 flags; the same deviation as A04 rows 007–008, though A04 row 007 uses O20 / 8020).
