# A20 — CAP-3: Full vs minimal harness at O20 (architecture × capability)

| Field | Value |
|---|---|
| **ID** | A20 · CAP-3 · new arm `fx_pajM_O20` |
| **Claim / Holm contrast** | C5 (capability), as the direct answer to meta-review concern 1 (backbone vs framework). **Exploratory, outside Holm** (R14). No Holm contrast. |
| **Tier** | T1-minimum |
| **Owner** | Henrik |
| **Start / finish** | Pre-registration text Thu 15 Oct (committed Fri 16 in X10) · O20 smoke **Thu 22 Oct, afternoon** · main sessions **Thu 22 Oct, night** as planned in 01 §6.2, or the first slot Henrik assigns after `S1021a` ends (§7 risk 6) · runs done ≈ 4–10 h after launch · latest launch Thu 29 Oct (L-LAST) · interaction read-out after A12's G4 (Tue 27 Oct) · final statistics Sun 1 – Tue 3 Nov · text Wed 4 – Fri 6 Nov · appendix Thu 12 Nov |
| **Budget** | **7 new runs, 3.5–7u** (= R1 and the T1-minimum card; arithmetic in §4). The smoke row (≈ 0.05–0.1u including the server boot) and any retries come from the 15% rerun reserve, not from these 3.5–7u. The 21 comparison runs (+7 A05 A/A) are paid by A05, A12 and A17. |
| **Models** | O20 = `openai/gpt-oss-20b` (3.6B active / 21B, MXFP4) for S, E, T and W. Served on `vllm/vllm-openai:v0.22.0`, TP1, ports 8020 (and 8021 for the second O20 server, a name A17 introduced), `MAX_MODEL_LEN=131072`, revision and `extra_args` from `scripts/agent2027/rungs.csv` (A04-T10). Comparison rung O120 = `openai/gpt-oss-120b` (from A05 and A12). |
| **Status** | Not started |
| **Depends on** | E1, E2, E3, E4 (shared Serper cache), E5, E7 (no-op proof for uniform rungs), E8a (+ recovery detector), **E9d** (image v2.3, G0b-1), E11a, E11b, E12 / A04 (O20 passes Stage 0), X1, X10. A05 `full`, A12 `fx_pajM`, A17 `f2_O20` (same sessions). A08 CPU screen (brute-force gate). FX TOST margin δ (for reading R4 only). D4. Default "CAP-3 at O20" (01 §6.4; adopted unless Henrik objects by Thu 15 Oct). |
| **Feeds into** | Fig. 3 inset (architecture × capability). Exploratory Table 2 row. The C5 discussion. A23 (Pareto points), A21 (anytime curves). Read together with A18 (RP1 floors) and A19 `mix_esmall`. · A17 (O20 slot sessions `S1022d`/`S1022e`) |

*Every "hoped" result below is a hypothesis, not a finding. No A20 data exist yet.*

> **SR4 fallback (2026-10-19).** O20 failed Stage 0 (A04 SR4), so the §6 Q4 switch is in force
> (pre-registration Part V.20, Part VII.1 item 14; Part VII.3 supersedes the O20 text of this
> document). Read "O20" as "Q4" throughout, with these specifics; no reading rule or threshold
> changes:
> - **Arm** `fx_pajM_Q4`: S, E, T and W on `Qwen/Qwen3.5-4B` @ `851bf6e` (TP1), image
>   `agent2027-v2.3` (E9d switches; overrides identical to A12 `fx_pajM`). Rows A20-001…007
>   (p3, p5, p2, p6a, p6b, p4, p1; f42 r0) and the smoke A20-008 (`synp_p3`, 1 fold × 3
>   iterations); manifest `manifests/agent2027/A20_CAP-3.csv`.
> - **Cells:** Full at the small rung = A17 `f2_Q4` f42 (A17-030…036, image v2.0, GPU 7 chain
>   session `S1019g`); minimal at O120 = A12 `fx_pajM`; Full at O120 = A05 `full` f42 r0. The
>   within-session pairing at the small rung is lost (cross-session, and v2.0 vs v2.3 images
>   whose Full-v2 prompts are byte-identical by the G0b-1 guard), and I crosses families
>   (Qwen vs gpt-oss); both disclosed. Estimands are renamed D_Q4 and I = D_Q4 − D_O120; R1–R5
>   apply as written with "O20" read as "Q4" (`flop_proxy` N_active = 4B).
> - **Sessions:** smoke `S1022c` (one Q4 server, port 8038, gated by G0b-1); main rows in
>   `S1022d`, the Q4 server that also serves A19's Extractor (port 8032), with `sentinel_Q4`
>   A06-103 in the first wave (first Q4 session on v2.3). If A19 cannot start (SR6 fails), launch
>   `S1022d` alone as a plain one-server session (it does not wait for A19). The O20 sessions
>   `S1022d`/`S1022e` (8020/8021) of §4/§5 no longer exist.

## 1. What we test and why

With the weaker backbone gpt-oss-20b, does splitting the work across three LLM agents help more than it does with gpt-oss-120b? We run the **minimal harness** at O20: one LLM role (the Extractor and its search helper) plus deterministic code for planning, assessment and pruning. We compare it with Full at O20 (A17) and with the same pair at O120 (A12 minimal harness, A05 Full). The 2 × 2 interaction (harness × rung) answers the meta-review's "is it the backbone or the framework?" with a measured number. The literature points both ways on whether role splits help small models or whether hand-offs hurt them.

## 2. Background

**Code:** `C:/code/PHD/RougeOne`, `nested-cv-rebuttal` @ `07cbf93`. The cited files are identical at local HEAD `1bdaf42` (`git diff 07cbf93 1bdaf42 -- rogue_one_main.py rogueone/` is empty), and every line cited here was re-read at `07cbf93` on Wed 7 Oct. Short paths (R0): `main.py` = `rogue_one_main.py`; `sci/agent.py`, `ext/agent.py`, `tst/agent.py` = `rogueone/llm/agents/core/{scientist,extractor,tester}/agent.py`; `kn/web_search.py` = `rogueone/llm/agents/core/knowledge/web_search.py`; `config.py` = `rogueone/utils/config.py`; `clients.py` = `rogueone/llm/clients.py`.

**One iteration** is `forward_pass` (`main.py:536`). Per-fold state is reset at `:543-544` (`_reset_fold_state`, `:417-457`; the Scientist notebook is cleared at `:457`).
- **S (planner).** `determine_focus` (`main.py:548-555`) receives `focus_history` (`:552`), and the focus is appended at `:556`. Tools are at `sci/agent.py:461-469`, including `search_tool` at `:464`. The history prompt is at `:472-475`. Each call gets a session named per step (`:499`). A heuristic fallback focus is defined at `:501-517`, and there are up to 100 retries (`for _ in range(100)`, `:519`).
- **E (coder).** `generate_patient_attributes` (`ext/agent.py:172`) is called at `main.py:563-571`. It runs `for _ in range(10)` (`ext/agent.py:967`; the R knob of E2 replaces this) in one session (`:965`), sending "System now wants to focus on: {focus}." (`:971`). Its tools include `search_in_literature_tool` (`:939`). A deterministic baseline fallback fires if nothing is added (`:996-1002`). The system prompt names the Scientist (`:872`, `:875`, `:879`) and says to follow the focus strictly (`:892`).
- **T (critic).** `test_hypotheses` is called at `main.py:789-794`.
  - Stage A, the LLM assessment, is called at `tst/agent.py:2115`. It has a search tool (`:1492`) and a fallback report (`:1498-1503`).
  - Stage B, the LLM pruning, is called at `:2134`; it has no search tool (`:1582-1588`). Its prompt rules are at `:1545-1559`: the VIF/SHAP/PFI rules at `:1545-1556` and a squares-before-bases derivative rule at `:1557-1559`. The floor is `_enforce_pruning_floor` (`:119`, applied at `:1617`).
  - The planner receives the Stage-A text (`:2159-2160`). The metric table is rendered by `_metric_profiles_to_markdown` (`:692`).
- **W (search summariser).** At `07cbf93` it always uses the **global** model and endpoint: `llm_cfg.endpoint` (`kn/web_search.py:189`) and `llm_cfg.model` (`:238`); its tokens are logged as `WebSearchAgent` (`:191`). Per-role overrides fall back to the global `LLM_MODEL` / `LLM_PORT` (`config.py:224-260`; `:246`, `:250`). From v2.3, E7's `ROGUE_ONE_WEBSEARCH_ROUTING=caller` (Full-v2 value) sends W to the caller's endpoint. In a uniform O20 row both rules give the same server (the row's port), so every role and W hit O20.
- **gpt-oss-20b specifics.**
  - It often emits tool calls as JSON text. These are rescued by `_resolve_text_tool_call` (`clients.py:37-108`, called at `:681`), so rescue rates are reported, not treated as failures.
  - `reasoning_effort` is popped (`clients.py:525`).
  - The `LLM_MODEL` type hint at `config.py:246` (a `Literal`) is not enforced.

**Minimal harness = FX cell P−A−J−aM+** (E9d switches; specified in A12-T5…T8):
- **P−:** `ROGUE_ONE_PLANNER_MODE=none`. `main.py:548-555` is skipped. In round 1 only, a digest (last `df_scientist_feedback_pool` row with report and `train_*`, the `Attribute | Status | Pruned Iteration | Prune Rationale` table, own earlier `FOCUS:` lines) is pushed into E's message at `ext/agent.py:971` and logged to `output/fx_digest.jsonl`; rounds 2…R send "System now wants to focus on: {own_focus}.". E must open with a `FOCUS:` line; if it does not, the first 300 characters are used and `focus_line_missing` is counted (A12-T5). The Extractor system prompt is unchanged (it still names the Scientist; declared in A12).
- **A−:** `ROGUE_ONE_ASSESSMENT_MODE=template`. A template report with the same numbers (counts, the `:692` table, top-5 \|SHAP\|, prunes) replaces Stage A (`tst/agent.py:2115`; A12-T6).
- **J−a:** `ROGUE_ONE_PRUNING_MODE=rules`. The Stage-B rules at `:1545-1556` run as code, keeping the same floor; the derivative rule `:1557-1559` is **not** executed (declared in A12-T7).
- **M+:** full memory (`ROGUE_ONE_MEMORY_MODE=full`, the Full-v2 value).
- **Effect:** the only LLM callers left are E and W (through E's search tool). S's and Stage A's search calls disappear with them, so `serper_calls` is reported per cell.

**Instruments.** **SYN-P 7** = `synp_p1`, `synp_p2`, `synp_p3`, `synp_p4`, `synp_p5`, `synp_p6a`, `synp_p6b` (R10).
- Each task has n = 2,000, p ≈ 20–30 and one planted φ. φ lives off-mount in `~/rogueone-synp-meta/` (never mounted into an agent container).
- Proposed types: C = p3, p4, p5; R = p1, p2, p6a, p6b. E8a confirms these in `synp_meta/registry.yaml`.
- P7 and P0 are not run (P7 policy).
- **Analysis population** = SYN-P 7 minus the families that the A08 CPU screen labels non-discriminating (OpenFE or AutoFeat recovers φ in ≥ 14/15 episodes, seeds {42, 4, 5}; p1 expected). Those tasks are run but excluded from the primary model (01 §4.7: exclusion comes only from the brute-force gate). n_T = the number of tasks in the analysis population (expected 6).
- **Informative tasks** (Full's recovery strictly between 0 and 1; E11a G2 memo) are used for power accounting only, never to exclude tasks (01 §4.7).

**Metrics** (R13; 01 §4.6):
- **`rec_sel` (primary).** In a fold-episode, an active, non-fallback feature has \|Spearman\| ≥ 0.95 with φ on training rows (NMI ≥ 0.9 for p3 and p5), **and** it is in the inner-CV-selected iteration. Raw columns and single-column monotone transforms never count.
- **Secondary:** `rec_any`, `func_success` (success **and** ≥ 1 accepted non-fallback LLM feature in the selected iteration), `success`, `collapse`, `cap_hit`.
- **Cost:** `tok_compl_total` and `tok_compl_<role>` (reasoning, re-rolls and search charged to the caller), `llm_calls_<role>`, `serper_calls`, `wall_s`, `energy_j`, `flop_proxy` (= 2 · N_active · tokens, with N_active = 3.6B for O20 and 5.1B for O120).
- **Process:** `fun_*`, `fallback_share`, `rescues_per100_<family>` (family `gpt-oss`), `err_per_accepted`, `append_success`, `context_errors`, `focus_line_missing` (A12 name).

**Units:** u = one real-data O120 run (10 iterations × 5 folds). An O20 run costs 0.5–1.0u. Throughput is 2 u/h × 0.7 = 1.4 u/h on ≤ 4 H100.

## 3. Hypotheses and interpretation (all hypotheses)

Let D_r = `rec_sel`(Full at rung r) − `rec_sel`(minimal at rung r), on the probability scale, where a positive value means decomposition helps. Define **I = D_O20 − D_O120**. Estimates and CIs come from A20-T13; the verdict comes from the pre-registered rules R1–R5 in §6.

| Outcome | Rule (§6) | Reading | Paper sentence / consequence |
|---|---|---|---|
| **Hoped:** I > 0 (decomposition helps more at O20) | R1 (or R2, suggestive) | Splitting planning and critique off the coder compensates for a weak backbone | *"Role decomposition pays off when the backbone is weak: at gpt-oss-20b, Full recovered X pp more planted features than the minimal harness, against Y pp at 120B."* (R2 only: "directionally consistent, not conclusive") |
| **Null:** D ≈ 0 at both rungs | R4 | A deterministic harness suffices at both scales. This supports C2 and weakens C1. | *"At both scales, one LLM role plus a deterministic harness matched the three-agent loop on recovery within ±δ, at Z% of its tokens."* |
| **Opposite:** D_O20 < 0 | R3 | Hand-offs hurt weak models (extra failure surface, fallbacks, cap hits) | *"At gpt-oss-20b the minimal harness recovered more than Full; decomposition should be used only with capable backbones."* Diagnose with `fallback_share`, `cap_hit`, `focus_line_missing` and the funnel stage. |
| **Floor:** both O20 cells at `rec_sel` = 0 | R5 | O20 cannot implement φ in either architecture (an execution floor; cross-check A18) | Recovery is uninformative. Report `func_success`, success, funnel and tokens only. |
| **Inconclusive** (expected) | none fires | The data cannot separate the readings | *"CAP-3 was inconclusive: I = X pp (95% CI a to b)."* plus the descriptive cost ratio |

**Honest prior:** "inconclusive". With 30 fold-episodes per cell (n_T = 6; 35 with all 7 tasks), the planning MDE for I is about 50 pp: SE(I) ≈ √(4 · 0.25 / 30) ≈ 0.18 at p ≈ 0.5, and 2.8 · 0.18 ≈ 0.51 (α = 0.05 two-sided, 80% power, unclustered; task clustering only widens it). The token ratio (minimal ≪ Full) is near-certain by construction and is reported descriptively.

## 4. Design

> **SR4 fallback (2026-10-19).** Cells: `fx_pajM_Q4` (A20-001…007, new) · A17 `f2_Q4` f42
> (A17-030…036, reused, other session) · A12 `fx_pajM` · A05 `full` r0 (+ r1 sensitivity).
> Manifest rows carry `rung_S/E/T=Q4`, `port_S/E/T=8032`, `image_tag=agent2027-v2.3`,
> `notes=pair=A17__f2_Q4__<d>__f42__r0; o120=…`; the smoke row is A20-008. The session layout
> below (two O20 servers, task blocks with A17 `f2_O20`) is void: `S1022c` (smoke) and `S1022d`
> (7 rows + A06-103) replace it, so the "no sentinel in the O20 sessions" rule no longer applies
> (A06 §4.5: a Q4 session carries `sentinel_Q4`). Budget unchanged (7 runs, 3.5–7u).

**Cells** (all at fold seed 42, `replicate=0`, SYN-P 7; uniform rung for S, E, T and W):

| Cell | AID · arm_id | Rung | Rows | Runs | Status | Run-name pattern |
|---|---|---|---|---|---|---|
| Minimal harness at O20 | **A20 · `fx_pajM_O20`** | O20 | A20-001…007 | **7** | **new** | `A20__fx_pajM_O20__<dataset_id>__f42__r0` |
| Full at O20 | A17 · `f2_O20` | O20 | A17-059…065 (p1, p2, p3, p4, p5, p6a, p6b) | 7 | reused; same sessions | `A17__f2_O20__<dataset_id>__f42__r0` |
| Minimal harness at O120 | A12 · `fx_pajM` | O120 | A12 blocks 1–2 (`S1021a`, `S1023a`) | 7 | reused | `A12__fx_pajM__<dataset_id>__f42__r0` |
| Full at O120 | A05 · `full` | O120 | A05 SYN-P f42 rows (`S1016a`/`b`) | 7 (+7 A/A `r1`, sensitivity only) | reused | `A05__full__<dataset_id>__f42__r0` |
| O20 smoke (never analysed) | A20 · `fx_pajM_O20` | O20 | A20-008 | 1 (`synp_p3`, 1 fold × 3 iterations) | new (reserve) | same name as A20-001; separated by `code_phase=smoke` |

**Manifest rows** (`manifests/agent2027/A20_CAP-3.csv`, R7 header, generated by `make_manifest.py --ablation A20`):
- **Rows A20-001…007:** `synp_p3` (classification), `synp_p5` (classification), `synp_p2` (regression), `synp_p6a` (regression), `synp_p6b` (regression), `synp_p4` (classification), `synp_p1` (regression). `task_type` follows `synp_meta/registry.yaml` if E8a changes a type.
- **Fixed fields for each row:**
  - identity and seeds: `ablation_id=A20`, `arm_id=fx_pajM_O20`, `fold_seed=42`, `llm_seed=42` (R11: `replicate=0` → `fold_seed`), `replicate=0`, `is_sentinel=0`; `session_id` and `order_key` empty (set by `--plan-session`);
  - image: `image_tag` = the same `agent2027-v2.N` (N ≥ 3) as A17's O20 rows in the session;
  - rungs and ports: `rung_S/E/T=O20`, `port_S/E/T=8020` (rows planned into `S1022e` get 8021 in the session file, §5.3). Uniform, so the launcher sets `LLM_MODEL=openai/gpt-oss-20b` and `LLM_PORT` = the row's port, and leaves `ROGUE_ONE_<ROLE>_MODEL/_PORT` empty;
  - `env_overrides={"ROGUE_ONE_PLANNER_MODE":"none","ROGUE_ONE_ASSESSMENT_MODE":"template","ROGUE_ONE_PRUNING_MODE":"rules"}` (identical to A12's `fx_pajM`);
  - `cap_hours=6`, `expected_u=0.75`;
  - `notes="pair=A17__f2_O20__<d>__f42__r0; o120=A12__fx_pajM__<d>__f42__r0|A05__full__<d>__f42__r0"`.
- **D4.** If D4 is adopted, the p3, p4 and p5 rows add `"ROGUE_ONE_STRATIFIED_FOLDS":"True"`, exactly as A05, A12 and A17 do.
- **Smoke row A20-008:** `synp_p3`; the same overrides plus `"ROGUE_ONE_SMOKE_FOLDS":"1"` and `"ROGUE_ONE_NUM_ITERATIONS":"3"`; `expected_u=0.05`; `notes=smoke`; `ROGUE_ONE_CODE_PHASE=smoke` (as in A03/A10). `fetch_wandb.py` keeps only `code_phase=v2` and drops `manifest_row=A20-008`.
- **Example row** (CSV, A19 style):
  ```
  A20-001,A20,fx_pajM_O20,synp_p3,classification,42,42,0,0,,,agent2027-v2.<N>,O20,O20,O20,8020,8020,8020,"{""ROGUE_ONE_PLANNER_MODE"":""none"",""ROGUE_ONE_ASSESSMENT_MODE"":""template"",""ROGUE_ONE_PRUNING_MODE"":""rules""}",6,0.75,"pair=A17__f2_O20__synp_p3__f42__r0; o120=A12__fx_pajM__synp_p3__f42__r0|A05__full__synp_p3__f42__r0"
  ```

**Fixed settings** (`scripts/agent2027/v2_defaults.env`, Full-v2; 01 §4.3):
- `ROGUE_ONE_NUM_ITERATIONS=10`, `ROGUE_ONE_K_FOLDS=5`;
- `ROGUE_ONE_SELECTION_SIGNAL=inner_cv`, `ROGUE_ONE_ENABLE_ROLLBACK=False`;
- temperature 0.7 for all roles and W;
- max tokens: `EXTRACTOR_AGENT_MAX_TOKENS=32768`, `TESTER_AGENT_MAX_TOKENS=8192`, `SCIENTIST_AGENT_MAX_TOKENS` as pinned at G1;
- `ROGUE_ONE_EXTRACTOR_ROUNDS` and `ROGUE_ONE_FEATURE_POOL_MODE` as frozen at G1;
- `ROGUE_ONE_TOOL_ERROR_FEEDBACK=message`;
- `ROGUE_ONE_LLM_TRACE=1`, `ROGUE_ONE_PROMPT_DUMP=1`;
- `ROGUE_ONE_SEARCH_CACHE=shared` at `ROGUE_ONE_SEARCH_CACHE_PATH`;
- `ROGUE_ONE_WEBSEARCH_ROUTING=caller` (no-op for uniform rows);
- `CPUS=12` per client; `RUN_CAP_HOURS=6`.

The image is **never** changed to suit O20 (A04 SR5).

**Budget arithmetic** (recomputed):
- **A20 (this document):** 7 runs × 0.5–1.0u = **3.5–7u**; planning value 7 × 0.75 = 5.25u. At u-pricing that is 3.5–7u ÷ 2 u/h = 1.75–3.5 h of the 4-H100 slot (7–14 H100-hours equivalent), or 3.5–7u ÷ 1.4 u/h = 2.5–5 h at 0.7 utilisation. Matches R1 (7 runs, 3.5–7u) and the T1-minimum card.
- **Smoke A20-008:** 1 fold × 3 iterations = 3/50 of a run × 0.5–1.0u = 0.03–0.06u, plus about 6 min of O20 load ⇒ ≈ 0.05–0.1u, from the reserve.
- **Shared O20 slot** (A20 + A17's 11 `f2_O20` rows, which A17 pays: 5.5–11u): 18 rows, 9–18u = 6.4–12.9 h of whole-box time at 1.4 u/h. u-pricing books box time; the O20 runs are CPU-bound, so the slot physically uses only 2 H100 and leaves 2 for A18's packed replays. Under CL2 (A17 drops `vehicle` and `heart` at O20) the slot has 16 rows, 8–16u.
- **Wall clock** (planning assumption 2–5 h per O20 run with 6 clients per server; replace it with 5 × the A04 `stage0_O20` fold time at A20-T2): two sessions of 9 rows each at `PARALLEL=6` ⇒ 2 waves ⇒ **≈ 4–10 h**; worst case with 6 h caps and one solo retry ≤ 18 h. Fallback layout (one server, 18 rows): 3 waves ≈ 6–15 h, worst case ≤ 24 h.

**Session layout** (realises A17 §4's O20 slot: 2 × O20 TP1 on 8020 and 8021, 6 clients each):
- Two sibling single-server sessions, launched like A05's TP2 split (A05-T9), **task-blocked** so that each task's two O20 cells share a session and a server:
  - **`S1022d`** (port 8020): A20-001…004 (p3, p5, p2, p6a) + A17-061, -063, -060, -064 (the same four tasks) + A17-066 (`junglechess`) = 9 rows;
  - **`S1022e`** (port 8021): A20-005…007 (p6b, p4, p1) + A17-065, -062, -059 (the same three tasks) + A17-067…069 (`vehicle`, `heart`, `myocardial`) = 9 rows.
- **Fallback** (only one GPU free, or Henrik prefers one server): a single session `S1022d` with all 18 rows on 8020.
- A17-T12 names `ladder_session.sh` for this slot. Its behaviour with two servers of the same rung (server names, binding rows to a port) is not specified in E7, so this document uses the sibling-session form, which needs only E1 and `vllm_session.sh` (verified: `SERVER_NAME`, `ROGUEONE_ALLOW_SECOND_SERVER`, `paper/scripts/vllm_session.sh:121-130`). If Henrik prefers `ladder_session.sh`, keep the same task blocks, one per server.

**Pairing and randomisation.**
- Both O20 cells of every task run in **one session on one server**, interleaved by random `order_key` (RNG seeded by the session ID). They share the server, load and Serper cache. Never split a task's O20 pair across sessions; if a session dies on an infrastructure fault, re-run the unfinished rows of the affected tasks together in one new session (`notes=infra-rerun`) and disclose any split.
- All cells share A05's seed-42 outer partition.
- The O120 pair comes from other sessions (A05: `S1016a`/`S1016b` on v2.0; A12: `S1021a`/`S1023a` on v2.3). This is a disclosed session confound in D_O120, guarded by A06's sentinels.
- No sentinel is used in the O20 sessions (A06 §4 hosting rule and its open question Q2; if Henrik adds a `sentinel_Q4` row there, A06 owns it).

**Failures (ITT; 01 §4.5, A17 §4).**
- A crash (rc ≠ 0, or no `fold_aggregation_summary.csv`) is retried once, alone, at the end of the session. Because the client image's `CMD` ends in `|| tail -f /dev/null` (`Dockerfile:29`), a crashed run can hang until the cap: a `CAP_HIT` whose container log shows a Python traceback before the hang is reclassified as a crash.
- Persistent failures and genuine 6 h cap hits (never retried) are ITT failures: all 5 fold-episodes get `rec_sel = rec_any = 0`, and `success = func_success = 0`.
- Infrastructure faults (vLLM death, host reboot, W&B or Serper outage) are re-queued with `notes=infra-rerun`; they are not ITT.
- Cap hits are likely **asymmetric**: Full at O20 can loop through up to 100 Scientist retries (`sci/agent.py:519`), while the minimal harness makes no S or T calls. `cap_hit` is reported per cell.

## 5. Tasks

### 5.1 Prerequisites (verify each before launch)
- [ ] **A20-T1** [type: writing] (0.1 d, due Thu 15 Oct; committed Fri 16 in X10) — Write the CAP-3 section of `docs/agent2027/preregistration.md`: the four cells, the estimands D_r and I, the rung O20 (Q4 fallback, §6), the analysis population rule, reading rules R1–R5 with their order and the near-floor guard, exploratory status outside Holm, the A05 `r0`-only primary, the GLMM of A20-T13, and the session plan. *Done when:* the text is in the commit tagged `agent2027-prereg`.
- [ ] **A20-T2** [type: decision] (0.02 d; Fri 16 Oct, 09:00, at the A04 verdict A04-T18) — O20 passed Stage 0 (A04 SR4: boots, the fold run completes, ≥ 80% rescued valid calls). `scripts/agent2027/rungs.csv` holds O20's revision sha, TP 1, port 8020 and its `extra_args`. Copy the `stage0_O20` fold wall time into this document's wall-clock estimate. *Done when:* the O20 row reads `stage0_verdict=pass` in the committed `rungs.csv`; if it reads `fail`, the Q4 switch (§6) is written into the pre-registration before the freeze tag.
- [ ] **A20-T3** [type: config] (0.05 d; check Mon 19 Oct, G2) — A `fetch_wandb.py` dry pull of one finished A05 run on `agent2027-v2.0` shows: every R3/R4 flag as `flags/<NAME>`; `rung_S/E/T` and `model_W`; non-zero `tok_compl_scientist`, `tok_compl_extractor` and `tok_compl_tester`; `fallback_share` and `rescues_per100_<family>`. *Done when:* the pull reports no missing key and no zero token count, and the E11a G2 memo (W&B group `ANALYSIS`) exists.
- [ ] **A20-T4** [type: config] (0.05 d; check Wed 21 Oct, G0b-1) — Confirm four things: (1) E9d is in `agent2027-v2.3`; (2) A12's G0b-1 smoke `S1021b` passed (A12-T11); (3) the E5 prompt diff shows Full byte-identical to `prompt-baseline-v2.0`; (4) E7's unit test shows that uniform rungs send W to the row's own endpoint (A17-T9). *Done when:* all four have a W&B or CI link in the G0b-1 note. If not, A20 waits.
- [ ] **A20-T5** [type: data] (0.02 d; check Tue 13 Oct, after M5S; recorded in the pre-registration draft by Thu 15 Oct) — A08-T1 is ticked (the SYN-P folders exist at `<ROGUEONE_CAIR_PATH>/tasks/<task_type>/synp_<id>/` with hashes equal to the local build), and the A08 screen labels are recorded. *Done when:* the pre-registration draft lists the analysis population (SYN-P 7 minus the non-discriminating families) by name.

### 5.2 Implementation (no Rogue One code change: A20 reuses the E9d switches unchanged)
- [ ] **A20-T6** [type: config] (0.05 d, due Wed 21 Oct) — Run `scripts/agent2027/make_manifest.py --ablation A20` to generate `manifests/agent2027/A20_CAP-3.csv` (rows A20-001…008, as in §4). *Done when:* the file is committed, has the R7 header, and has 8 unique `row_id`s and 7 unique analysis run names.
- [ ] **A20-T7** [type: code] (0.1 d, due Tue 27 Oct; part of E11b) — Add `--mode cap3` to `scripts/agent2027/analysis/glmm_holm.py`. It fits the 2 × 2 harness × rung model of A20-T13, uses the Firth fallback, reports marginal-mean risk differences and the R1–R5 verdict, and **never** adds its result to the Holm family. *Done when:* it reproduces a hand-computed 2 × 2 toy table (D_O20, D_O120, I and the verdict) to 3 decimals.

### 5.3 Execution on CAIR

> **SR4 fallback (2026-10-19).** A20-T8's session files exist: `S1022c` (A20-008) and `S1022d`
> (A20-001…007 + A06-103, `--sentinel-first`). A20-T9 launches the smoke on one free GPU as a
> Q4 server (`MODEL=Qwen/Qwen3.5-4B`, Q4 `EXTRA_ARGS` from `rungs.csv`, port 8038,
> `PARALLEL=1`); its manipulation checks read "model_S/E/T/W = `Qwen/Qwen3.5-4B`, E and W
> endpoints on the session port". A20-T10 launches `S1022d` as A19's Q4 holder (port 8032,
> `CLIENTS_SCRIPT=~/ro27_hold_clients.sh HOLD_FOR=S1022a`, `PARALLEL=4`), or alone if A19 is not
> launching. Exact lines: `docs/agent2027/sessions/LAUNCH.md`. The ports 8020/8021/8022 and the
> A18 packing around the O20 session are void.

**Hard rules** (01 §5.1):
- **Containers only.** Every execution (runs, analysis, W&B queries) happens inside Docker. The host only issues `docker` / `docker compose` commands and the `~` launcher scripts that issue them, plus `git` sync, `cp`, `ls`, `nvidia-smi`, `docker ps` and `tail`. Never run host Python. `docker compose run` over SSH needs `-T`.
- **GPU pre-flight** (A20-T9 command) before every GPU container. If the GPUs are not free, **stop and report** the per-GPU memory and utilisation, the compute apps and the running containers, and let Henrik decide. Launch with explicit `GPUS=<ids>` and `WAIT_MAX_MIN=0`.
- **≤ 4 of the 8 H100s** without Henrik's explicit approval in chat. One Rogue One job at a time; the planned co-tenant of this slot is A18's packed replays (A17 §4, A18-T11), and the GPUs of all our `rogueone-vllm-*` servers together stay ≤ 4.
- **W&B is the system of record.** Runs are stateless: each uploads its `output/` folder as artifact `run-output`; the root-owned local run folders are secondary. Project `PlanBench/RogueOne-AGENT2027`, group `A20`. Secrets: `WANDB_API_KEY` from `<CAIR_HOME>/code/PlanBench/.env`; `SERPER_API_KEY` and LLM settings from `.env` / `.secret.env` in `<ROGUEONE_CAIR_PATH>`, baked into the image (never push it).
- **SSH:** non-interactive one-shot commands only: `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "<cmd>"`, written `ssh …` below.
- **Long jobs:** detached `nohup` launcher; the job log goes in `~` (`~/ro27_<SESSION_ID>.job.log`), never in the root-owned run folders.
- **Code path:** commit and push `agent2027/v2` locally, then `git fetch` over SSH on the server.
- `<ROGUEONE_CAIR_PATH>` = `<CAIR_HOME>/code/RougeOne` (**INFERRED, verify** once with `ssh … "ls -d ~/code/RougeOne"`).

- [ ] **A20-T8** [type: config] (0.05 d, Thu 22 Oct, morning; the same action as A17-T10 for the O20 slot, done once) — Plan the sessions locally:
  - `make_manifest.py --plan-session S1022c --rows A20-008`;
  - `make_manifest.py --plan-session S1022d --rows A20-001 A20-002 A20-003 A20-004 A17-060 A17-061 A17-063 A17-064 A17-066`;
  - `make_manifest.py --plan-session S1022e --rows A20-005 A20-006 A20-007 A17-059 A17-062 A17-065 A17-067 A17-068 A17-069`, with `port_S/E/T=8021` in the `S1022e` file (as A05-T5 does for `S1016b`). Under CL2, drop A17-067 and A17-068.
  - Fallback layout: one `S1022d` holding all 18 rows on 8020.

  Commit and push `agent2027/v2`. Then sync, build if the tag is new, and install (01 §5.2 steps 0, 2, 3):
  ```
  ssh … "cd <ROGUEONE_CAIR_PATH> && GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15' git fetch git@github.com:HenrikNorkart/RougeOne.git agent2027/v2 && git checkout -B agent2027/v2 FETCH_HEAD && git log -1 --format='HEAD: %h %s'"
  ssh … "cd <ROGUEONE_CAIR_PATH> && test ! -e .dev.env && docker build -t rogue_one:agent2027-v2.<N> . && docker image inspect rogue_one:agent2027-v2.<N> --format '{{.Id}}'"
  ssh … "cd <ROGUEONE_CAIR_PATH> && cp scripts/agent2027/ro27_session_launch.sh scripts/agent2027/ro27_clients_manifest.sh scripts/agent2027/ladder_session.sh ~/ && cp paper/scripts/vllm_session.sh ~/ro27_vllm_session.sh && mkdir -p ~/ro27_manifests && cp -r manifests/agent2027/. ~/ro27_manifests/"
  ```
  Never rebuild an existing tag: end-of-session retries resolve `rogue_one:<image_tag>` when they start. *Done when:* the CAIR HEAD equals the pushed tag, the image ID is recorded, and `~/ro27_manifests/sessions/S1022{c,d,e}.csv` exist, with each task's A20 row and A17 row in the same file.
- [ ] **A20-T9** [type: run] (0.05 d; Thu 22 Oct, afternoon, ≈ 0.5 h) — Run the pre-flight:
  ```
  ssh … "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader && echo '--- compute apps ---' && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader && echo '--- containers ---' && docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'"
  ```
  Proceed only if all of these hold:
  - the GPU you pin shows < 2,000 MiB used (`FREE_MIB`);
  - the GPUs held by our `rogueone-vllm-*` servers, plus this one, total ≤ 4;
  - no `ro27-*`, `rogueone-vllm-*`, `llmfe-*` or `caafe-*` container of ours is running, apart from A18's packed replay servers.

  Otherwise STOP and report. Then launch the smoke:
  ```
  ssh … "SESSION_ID=S1022c SESSION_FILE=~/ro27_manifests/sessions/S1022c.csv GPUS=<1 id> NEED=1 MODEL=openai/gpt-oss-20b PORT=8020 SERVER_NAME=rogueone-vllm-o20-s1022c EXTRA_ARGS='<rungs.csv O20 extra_args>' MAX_MODEL_LEN=131072 EAGER=0 LOG=~/vllm_session_S1022c.log IMAGE_RO=rogue_one:agent2027-v2.<N> RUNROOT=~/rogueone-runs-agent2027 PARALLEL=1 CPUS=12 RUN_CAP_HOURS=6 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_S1022c.job.log 2>&1 & echo launched; sleep 5; tail -5 ~/ro27_S1022c.job.log"
  ```
  Add `ROGUEONE_ALLOW_SECOND_SERVER=1` only if another of our servers is up and the pre-flight showed the total stays ≤ 4 (`vllm_session.sh:121-130` refuses otherwise). *Done when:* the job log ends with `RO27_SESSION_END S1022c rc=0`, the smoke row has `fold_aggregation_summary.csv`, and in W&B:
  - `llm_calls_scientist = 0` and `llm_calls_tester = 0` (no Stage A or Stage B LLM call);
  - `model_S/E/T/W` = `openai/gpt-oss-20b`, and the trace endpoints of E and W are port 8020;
  - `output/fx_digest.jsonl` exists, and the prompt dump shows the digest in round 1 only of each iteration;
  - `focus_line_missing` is logged. A high rate is data, not a failure.
- [ ] **A20-T10** [type: run] (0.05 d; Thu 22 Oct, night, or the slot Henrik assigns; latest launch Thu 29 Oct) — Repeat the pre-flight for 2 GPUs (1 in the fallback), then launch both sessions detached:
  ```
  ssh … "SESSION_ID=S1022d SESSION_FILE=~/ro27_manifests/sessions/S1022d.csv GPUS=<id 1> NEED=1 MODEL=openai/gpt-oss-20b PORT=8020 SERVER_NAME=rogueone-vllm-o20-s1022d EXTRA_ARGS='<rungs.csv O20 extra_args>' MAX_MODEL_LEN=131072 EAGER=0 LOG=~/vllm_session_S1022d.log IMAGE_RO=rogue_one:agent2027-v2.<N> RUNROOT=~/rogueone-runs-agent2027 PARALLEL=6 CPUS=12 RUN_CAP_HOURS=6 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_S1022d.job.log 2>&1 & echo launched; sleep 5; tail -5 ~/ro27_S1022d.job.log"
  ssh … "SESSION_ID=S1022e SESSION_FILE=~/ro27_manifests/sessions/S1022e.csv GPUS=<id 2> NEED=1 MODEL=openai/gpt-oss-20b PORT=8021 SERVER_NAME=rogueone-vllm-o20-s1022e ROGUEONE_ALLOW_SECOND_SERVER=1 EXTRA_ARGS='<rungs.csv O20 extra_args>' MAX_MODEL_LEN=131072 EAGER=0 LOG=~/vllm_session_S1022e.log IMAGE_RO=rogue_one:agent2027-v2.<N> RUNROOT=~/rogueone-runs-agent2027 PARALLEL=6 CPUS=12 RUN_CAP_HOURS=6 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_S1022e.job.log 2>&1 & echo launched; sleep 5; tail -5 ~/ro27_S1022e.job.log"
  ```
  - The first command also needs `ROGUEONE_ALLOW_SECOND_SERVER=1` if A18 servers are already up (only after the pre-flight shows the total stays ≤ 4).
  - A18 may pack small-rung replays onto the other ≤ 2 GPUs; A18 starts those servers itself after its own pre-flight (A18-T11).
  - **No replay traffic may hit ports 8020 or 8021 while S1022d/e rows run.** A18-005 (`replay_O20`) runs on its own port 8022 (01 R8) and may co-run; if A18 falls back to 8020 it starts only after `RO27_SESSION_END S1022d`.

  *Done when:* each job log shows the vLLM version assert, both image IDs, the energy sidecar (`S1022d__energy`, `S1022e__energy`), and its first 6 rows started.
- [ ] **A20-T11** [type: run] (0.1 d; Fri 23 Oct, or within 1 day of the launch; latest Fri 30 Oct) — Monitor each session as a background command:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 CAIR-gpu14 "while ! grep -q 'RO27_SESSION_END' ~/ro27_S1022d.job.log 2>/dev/null; do sleep 120; done; tail -30 ~/ro27_S1022d.job.log"
  ```
  (and the same for `S1022e`).
  - Exit code 255 means the network dropped, not the job; reconnect.
  - Check run states in a container: `ssh … "docker run --rm --env-file <CAIR_HOME>/code/PlanBench/.env --entrypoint python3 rogue_one:agent2027-v2.<N> -c \"import wandb; api=wandb.Api(); [print(r.name, r.state) for r in api.runs('PlanBench/RogueOne-AGENT2027', filters={'group': 'A20'})]\""`. A `running` run with no live container is a zombie.
  - Tear-down check: `ssh … "docker ps --format '{{.Names}}' | grep -E '^(ro27-|rogueone-vllm-)' || echo none; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"`.

  *Done when:* both job logs end with `RO27_SESSION_END … rc=0`; all 7 A20 rows (and A17's rows) are `finished` or ITT-logged, each with a `run-output` artifact and tags [`S1022d` or `S1022e`, `v2`, `agent2027-v2.<N>`]; and the tear-down check prints `none` for our containers.

### 5.4 Analysis (CPU containers on CAIR, `--cpus` ≤ 16; `~/rogueone-synp-meta/` mounted read-only into analysis containers only)

Command template (exact script CLIs are set by E11b):
```
ssh … "mkdir -p ~/ro27_a20_analysis && docker run --rm --cpus 16 --env-file <CAIR_HOME>/code/PlanBench/.env -v <ROGUEONE_CAIR_PATH>/scripts/agent2027:/workspace/scripts/agent2027:ro -v ~/rogueone-synp-meta:/meta:ro -v ~/ro27_a20_analysis:/out --entrypoint python3 rogue_one:agent2027-v2.<N> /workspace/scripts/agent2027/analysis/<script>.py <args>"
```

- [ ] **A20-T12** [type: analysis] (0.1 d; Tue 27 Oct, after A12 is scored, or within 1 day of S1022d/e ending; latest Sun 1 Nov) — Run `analysis/fetch_wandb.py` → `analysis/recovery_detector.py` on the four cells (A20 r0; A17 `f2_O20` SYN-P 7 r0; A12 `fx_pajM` r0; A05 `full` r0 + r1), keeping `code_phase=v2` and filling ITT rows. *Done when:* 140 (+ 35 A/A) fold-episodes (4 cells × 7 tasks × 5 folds) are scored and logged to group `ANALYSIS`.
- [ ] **A20-T13** [type: analysis] (0.25 d, due Tue 3 Nov) — Run `glmm_holm.py --mode cap3`.
  - **Model:** fold-episode logistic GLMM `rec_sel ~ H + R + H:R + (1|task)`, effects-coded (H: Full +1 / minimal −1; R: O120 +1 / O20 −1), on the analysis population only. If n_T < 3, skip the model and report the per-task table only.
  - **Estimands:** D_O20, D_O120 and I, as model-predicted marginal means on the probability scale (tasks equally weighted), with 95% CIs (90% CIs for R4).
  - **Separation fallback:** Firth-penalised logistic regression with task fixed effects, plus a per-task 2 × 2 listing.
  - **Sensitivities:** Full at O120 = A05 r0 + r1; fallback-only runs excluded; cap hits excluded; all 7 tasks.
  - **Secondary:** `func_success` (run level, Firth); log `tok_compl_total ~ H*R + (1|task)`, giving the minimal/Full ratio per rung with a 90% CI; `flop_proxy`.

  *Done when:* `ANALYSIS__glmm_holm__<YYYYMMDD>` holds the table with all three estimands and the sensitivity rows.
- [ ] **A20-T14** [type: analysis] (0.2 d, due Tue 3 Nov) — Run `analysis/process_metrics.py` and `analysis/funnel.py` per cell:
  - the funnel stage where φ is lost;
  - `fallback_share` (Scientist heuristic focus, Tester fallback report, Extractor baseline);
  - `focus_line_missing`, `rescues_per100_<family>`, `err_per_accepted`, `append_success`;
  - `cap_hit`, `context_errors`, `serper_calls`, `wall_s`.

  Add a descriptive three-point row at E = O20, comparing S/T deterministic (A20), S/T at O20 (A17) and S/T at O120 (A19 `mix_esmall`, f42 rows). *Done when:* the tables are in the W&B group `ANALYSIS`.
- [ ] **A20-T15** [type: decision] (0.05 d, due Tue 3 Nov) — Apply R1–R5 (§6) in the stated order and write the verdict and sentence into the analysis memo. Hand the points to `analysis/pareto.py` (A23). *Done when:* the memo is logged and A23 shows the 4 cells.

### 5.5 Paper outputs
- [ ] **A20-T16** [type: writing] (0.15 d, due Fri 6 Nov) — Fig. 3 inset: `rec_sel` by rung × harness with CIs, annotated with the token ratio. Add an exploratory Table 2 row "CAP-3 architecture × capability" (D_O20, D_O120, I, Δ success, token ratio, verdict) and one §3 sentence that the verdict allows. In threats to validity: one seed, the O120 session confound, rescue rates differing by model, and cross-family if Q4 is used. *Done when:* all are in the AGENT draft.
- [ ] **A20-T17** [type: writing] (0.1 d, due Thu 12 Nov) — Appendix and replication package: the per-task 2 × 2 tables, rendered prompts of `fx_pajM_O20`, O20 serving flags and revision, the rescue disclosure, and the fallback sensitivity. *Done when:* these are in the anonymised package.

## 6. Decision rules and gates (pre-registered; exploratory, outside Holm)

> **SR4 fallback (2026-10-19).** The "A04 verdict" gate below fired: A20 runs on Q4. R1–R5, their
> order, the near-floor guard and δ are unchanged, with O20 read as Q4. Remaining gates: G0b-1
> (E9d smoke `S1021b` passed, Full byte-identical; A20-T4) before the smoke `S1022c`; the smoke's
> checks before `S1022d`; launch from Thu 22 (Henrik's slot), kill date L-LAST Thu 29.

**Reading rules** (CIs from A20-T13 on the analysis population; δ = A12's FX TOST margin, max(0.25 × the A09/A05 oracle − Full gap, 0.10); if A12 has not fixed δ by Tue 3 Nov, δ = 0.10):
- **R5, floor:** both O20 cells have `rec_sel` = 0 on every task of the analysis population. The recovery reading is then "not estimable"; use `func_success` and cost.
- **R1, "weak backbone benefits more":** the 95% CI of I lies entirely above 0.
- **R3, "minimal harness wins at O20":** the 95% CI of D_O20 lies entirely below 0.
- **R4, "harness suffices at both scales":** the 90% CIs of D_O20 and D_O120 both lie within ±δ, **and** the pooled O20 `rec_sel` (both cells) is ≥ 0.10. If the pooled O20 rate is < 0.10, R4 is replaced by "near floor: not estimable".
- **R2, "directionally consistent" (suggestive only):** the point estimate I ≥ 0.15, and the per-task D_O20 > D_O120 (strictly) on at least ⌈2 n_T / 3⌉ tasks of the analysis population (4 of 6).
- **Otherwise:** "inconclusive". This is the expected outcome, since the planning MDE for I is about 50 pp (§3).
- **Order:** evaluate R5, R1, R3, R4, R2. The verdict is the first rule that fires; any other rule that also fires is reported as a secondary label.

Never write "significant" or a Holm-adjusted p for A20.

**Gates and dates:**
- **Thu 15 Oct:** "CAP-3 at O20" is a default adopted unless Henrik objects (01 §6.4). The rung is fixed now because G6 comes too late.
- **A04 verdict (Fri 16 Oct, 09:00):** if O20 fails SR4, A20 moves to **Q4**. The arm becomes `fx_pajM_Q4` (port 8032), compared against A17 `f2_Q4` at f42 (A17-030…036). Those A17 rows run in the small-rung nights, so the within-session pairing is lost; this and the cross-family interaction (Qwen vs gpt-oss) are disclosed. The `fx_pajM_Q4` rows are planned into the next session that serves Q4 (A17 N3), or into their own one-server session.
- **G2 (Mon 19 Oct):** A20-T3 passes, or A20 waits.
- **G0b-1 (Wed 21 Oct):** E9d and the FX smoke must pass (A20-T4), or A20 waits.
- **Launch:** Thu 22 Oct night, or the first slot Henrik assigns when the box is free of other Rogue One jobs. **Kill date: L-LAST, Thu 29 Oct.** If A20 is not launched by then it is dropped; no Holm family is affected. A17's O20 rows run without it.
- **If A12 `fx_pajM` is missing** (FX not launched by Wed 28): report D_O20 only, with no interaction. If only some A12 tasks exist, fit I on the tasks with all four cells and report D_O20 on all tasks.
- **G6 (Fri 30 Oct):** descriptive only; the rung was fixed at O20 in advance.
- **DF:** Sat 31 Oct, 24:00. No A20 GPU run after this.

## 7. Risks and cut levers

1. **O20 recovery floor** (an execution floor, as RP1 predicts) makes CAP-3 uninformative on recovery (R5, or R4's near-floor guard). Read it with A18.
2. **Low power:** one seed and about 6 analysis tasks (MDE ≈ 50 pp).
3. **Fallbacks mask failures:** Full at O20 may run on heuristic focuses and fallback reports, which makes it a partly deterministic harness. Report `fallback_share`; run the fallback-excluded sensitivity.
4. **Asymmetric cap hits:** ITT is primary; the cap-excluded result is a sensitivity analysis. Crashes that hang behind `tail -f /dev/null` are reclassified (§4).
5. **P− at O20** may often lack the `FOCUS:` line, and P− also turns pulled feedback into pushed feedback (as in A12).
6. **Box contention Thu 22 – Fri 23.** `S1021a` (A12 block 1 + A13 + A14, O120) is estimated to hold the box until Fri 23, 04:00–16:00 (A14 §4); A19 (`S1022a`) needs the whole 4-GPU slot for 20–39 h; A16's pilot must run before K-SET (Fri 23). With one Rogue One job at a time, the Thu 22 night slot in 01 §6.2 is not feasible as written. **Henrik sets the order** (A16 §7 proposes: A20 smoke → A16 pilot → the rest). A20 slips at most to L-LAST (Thu 29).
7. **Session confound at O120** only: disclosed; A06 sentinels.
8. **Cross-document conflicts:** A17-T12 now uses the same two sibling sessions (`S1022d`/`S1022e`) as this document (resolved). A18's `replay_O20` uses port 8022 (01 R8), not 8021, so it need not wait for `S1022e` (resolved). Still open: A16 §7 quotes the old one-server duration (6–15 h) for this slot.

**Cut levers.** A20 is not in CL1–CL7. CL2 (CAP-1 small-rung real data on 2 datasets) may shrink A17's O20 real rows from 4 to 2 (A17-067, -068 dropped from `S1022e`), but it never touches A20. The CAP-3 SET cell, a second seed and the CAP-3 half-fraction were already given up in T1-minimum.

## 8. Definition of done
- [ ] The CAP-3 pre-registration section is committed (Fri 16 Oct).
- [ ] The O20 Stage-0 pass is recorded in `rungs.csv` (or the Q4 switch is applied and documented).
- [ ] The smoke `S1022c` passed its manipulation checks.
- [ ] 7 runs are in W&B group `A20`, each with `run-output`, task-blocked with A17 `f2_O20` in `S1022d`/`S1022e`; failures are ITT-scored.
- [ ] The 2 × 2 GLMM, cost and process tables are logged in `ANALYSIS`; the R1–R5 verdict is recorded.
- [ ] The Fig. 3 inset, the exploratory Table 2 row, the sentence and the appendix items are written.

---

**New names introduced** (not in the canonical registry):
- **Session IDs** `S1022c` (smoke), `S1022d` (port 8020) and `S1022e` (port 8021). `S1022a` is A19's and `S1022b` is A16's pilot session (cross-document map); A18's packed replays use `S1022p`–`s`.
- **Server names** `rogueone-vllm-o20-s1022c`, `rogueone-vllm-o20-s1022d`, `rogueone-vllm-o20-s1022e` (R8 pattern).
- **Arm** `fx_pajM_Q4` (only if O20 fails Stage 0).
- **Option** `glmm_holm.py --mode cap3`.
- **Manifest file** `manifests/agent2027/A20_CAP-3.csv` (follows the R6 pattern); smoke row `A20-008`.
- **Estimand labels** D_O20, D_O120, I; **analysis population** and n_T.
- **Analysis folder** `~/ro27_a20_analysis/` on CAIR.
- **Names reused from A04, A12 and A17:** `scripts/agent2027/rungs.csv`, `focus_line_missing`, `output/fx_digest.jsonl`, port `8021` (second O20 server).
