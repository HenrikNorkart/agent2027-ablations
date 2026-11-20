# A12 — FX: 2^(4−1) half-fraction of planner × assessment × pruning × memory

| Field | Value |
|---|---|
| **ID** | A12 · FX |
| **Claim / Holm** | C2 "which roles need an LLM". **C2a** = A main effect (TOST), **C2b** = J main effect (TOST), **C2c** = P main effect (two-sided); outcome `rec_sel` (01 §7 R14) |
| **Tier · Owner · Status** | T1-minimum · Henrik · Not started |
| **Dates** | Build Mon 19 – Wed 21 Oct, 12:00 (E9d) · smoke `S1021b` Wed 21 Oct, afternoon (G0b-1) · **launch `S1021a` Wed 21 Oct, ≈ 20:00** · block 2 `S1023a` Fri 23 Oct · runs done Sat 24 – Sun 25 (latest Mon 26, 08:00) · δ committed Fri 23, 18:00 (A09-T15) · G4 Tue 27 · statistics Sun 1 – Tue 3 Nov · text Wed 4 – Fri 6 Nov |
| **Budget** | **56 runs, 39.2–56u** (8 cells × 7 tasks × 1 seed × 0.7–1.0u; identical to the T1-minimum FX card and 01 §7 R1) = 19.6–28 h of the 4-H100 slot at 2 u/h (78–112 H100-hours at TP4), or 28–40 h at 0.7 utilisation. **Outside the 56:** smoke + Full guard ≈ 0.5–0.7u (from the 15% rerun reserve, 01 §4.9); session sentinels (paid by A06); reserve **RV3**: 28 runs, 19.6–28u, only if triggered at G4. **Developer time:** ≈ 3.8 d over A12-T1…T23; only E9d's 1.0 d (T4–T8) is in the 01 §3 engineering budget. |
| **Models** | O120 `openai/gpt-oss-120b` for S, E, T and W; server `vllm/vllm-openai:v0.22.0`, port 8010, `MAX_MODEL_LEN=131072`, TP4 (TP2 if D5) |
| **Depends on** | E1–E5, E8a (+ detector), E9c (introduces the `ROGUE_ONE_PLANNER_MODE` reader, v2.1), **E9d**, E11a, E11b, X10; A08 CPU screen (brute-force gate); A09-T15 (δ, Fri 23); A06 (sentinels); A05 (Full anchor, analysis only); A13 and A14 (share `S1021a`/`S1021b`); A16 (shares `S1023a`); D2, D3, D4, D5; gates G1, G2, G0b-1, K-SET, G4 |
| **Feeds into** | Table 2 rows C2a–c; Fig. 2 left; A16 (SE\|T = `fx_pAJM`); A19 and A20 (minimal harness = `fx_pajM`); A21–A23 · A13 (shared session `S1021a`/`S1021b`) |

*Every "hoped" result is a hypothesis, not a finding.*

## 1. What we test and why

Each of four LLM components is swapped for a version with no LLM call:
- **P:** no Scientist; the Extractor sets its own focus (SE|T);
- **A:** a template instead of the Tester's written assessment;
- **J:** the Tester's pruning rules run as code (J−a);
- **M:** a planner that forgets its history.

A balanced half of the 16 combinations runs on planted-feature tasks, and we measure recovery and tokens. It is the paper's only C2 test (measurement vs judgement). It also yields the minimal harness (one LLM role plus code) that A19 and A20 reuse.

## 2. Background

Code: `C:/code/PHD/RougeOne`, `nested-cv-rebuttal` @ `07cbf93`. The files below are unchanged at local HEAD `1bdaf42` (`git diff 07cbf93 1bdaf42 -- rogue_one_main.py rogueone/` is empty), and every line below was re-read on Wed 7 Oct. Short paths per 01 §7 R0: `main.py` = `rogue_one_main.py`; `sci/agent.py`, `ext/agent.py`, `tst/agent.py` = `rogueone/llm/agents/core/{scientist,extractor,tester}/agent.py`; `tst/metrics.py` = `rogueone/llm/agents/core/tester/metrics.py`; `config.py` = `rogueone/utils/config.py`.

| Part | Code | FX lever |
|---|---|---|
| Fold reset | `main.py:543-544` → `_reset_fold_state` `:417-457` (notebook cleared `:457`) | — |
| Scientist call | `determine_focus` `main.py:548-555`; history passed `:552`, appended `:556`, rendered `sci/agent.py:472-475`; fresh session `:499`; 100 retries `:519-545`, no-tool fallback `:549-577`, heuristic fallback `:501-517` | P skips it; M truncates |
| Pulled feedback | `generic_pandas_tool` over `df_test_results.feature_assessment` (`sci/agent.py:180-190`; rows `main.py:384-415`, appended `:821-831`); `attribute_lookup_tool` `sci/agent.py:306` returns the attribute rows incl. `Status`, `Prune Rationale` | P− pushes it instead |
| Notebook | `take_a_note` `sci/agent.py:338`, `read_notes` `:370-387`, tool list `:461-469`; prompt lines `:429` ("Review prior notes"), `:433` ("Save key reasoning in notes") | M− |
| Extractor | `generate_patient_attributes` `ext/agent.py:172` (called `main.py:563-571`): one session `:965`, `range(10)` rounds `:967` (R knob from E2), each sends `"System now wants to focus on: {focus}."` `:971`; Scientist named `:872`, `:875`, `:879`; "Follow the provided focus strictly" `:892`; baseline fallback `:996-1002` | P− digest in round 1 |
| Metrics | 300-tree model `tst/agent.py:927`, `:956`; MI, VIF, PFI, \|SHAP\| via `compute_feature_metrics` `:967-974` (VIF `_compute_vif` `tst/metrics.py:75`, ε = 1e-12 so a perfect collinearity gives ≈ 1e12; PFI 5 repeats `tst/metrics.py:119-143`); table `_metric_profiles_to_markdown` `tst/agent.py:692` (columns `feature_name, mutual_information, vif, pfi_mean, pfi_std, noise_sensitivity, mean_abs_shap, warnings`) | A− reuses |
| Stage A | `generate_assessment` `:1423-1507` (call `:2115`; search tool `:1492`); writes the report `:1505` and sets `_assessment_report_produced` `:1506`, which gates pruning `:1283-1289`; Stage-A notes reach Stage B via `get_notes_tool` `:1107` | A− |
| Stage B | `generate_pruning` `:1509-1656` (call `:2134`); report is "primary context" `:1539` (passed `:1572-1577`); rules `:1545-1556`; derivative rule `:1557-1559`; rationale = whole summary `:1597-1609`; floor max(3, ⌈0.2·pre-active⌉) `_enforce_pruning_floor` `:119-163`, applied `:1617`; all-pruned guard `:1637-1644` | J−a |
| Report to planner | Stage-A text `:2159-2160`; `include_report=True` `main.py:223` (False blanks the report, so A− keeps True) | A |
| Tokens | both stages use client `agent_name="TesterAgent"` `tst/agent.py:1341`; stage agent names `:1484`, `:1580`; stage system prompts say "FEATURE ASSESSMENT ONLY" `:1440` and "PRUNING ONLY" `:1536` | stage split |
| Config | `rogueoneConfig` `config.py:303-337` | new flags |

**Instruments.** SYN-P 7 = `synp_p1`, `synp_p2`, `synp_p3`, `synp_p4`, `synp_p5`, `synp_p6a`, `synp_p6b` (01 §7 R10): n = 2,000, p ≈ 20–30, one planted φ each; E8a standardises regression targets so the SHAP 0.1 threshold is comparable across tasks. φ lives off-mount in `~/rogueone-synp-meta/`.
- Task types (proposed; E8a confirms them in `synp_meta/registry.yaml`): C = p3, p4, p5; R = p1, p2, p6a, p6b.
- Multi-step = p3, p5; single-formula = p1, p4, p6a, p6b; p2 per D3.
- No P7 (P7 policy, 01 §7 R10).
- Families the A08 screen flags as non-discriminating (OpenFE or AutoFeat recovers φ in ≥ 14/15 episodes, seeds {42, 4, 5}; p1 expected) are run but are not confirmatory.

**Metrics** (01 §4.6, names per 01 §7 R13). `rec_sel` (primary): within a fold-episode, an active non-fallback feature has \|Spearman\| ≥ 0.95 with φ on training rows (NMI ≥ 0.9 for p3, p5) **and** is in the iteration that inner CV selects. Raw columns and single-column monotone transforms never count. Also: `rec_any`, `success`, `collapse`, `cap_hit`, `itt_imputed`, `tok_compl_<role>`, `tok_compl_total` (search charged to the caller), `llm_calls_<role>`, `fun_*`, `prune_rate`, `prune_jaccard`, `floor_activations`, `reproposal_ratio`, `report_read_rate`, `fallback_share`.

**Full-v2** (`scripts/agent2027/v2_defaults.env`): 10 iterations × 5 folds, inner-CV selection, rollback off, temperature 0.7, R and pool mode as frozen at G1, Scientist feedback signal per D2, `PARALLEL=6`, 6 h cap. All four factors at "+".

## 3. Hypotheses and interpretation

Δ_X = mean `rec_sel`(X+) − mean `rec_sel`(X−), so positive means the LLM version is better.

| Effect | Hoped | Null | Opposite |
|---|---|---|---|
| A (C2a) | Δ_A inside ±δ at ≥ 30% fewer tokens: *"A template assessment lost no detectable recovery (CI …) at X% fewer tokens."* | Expected: 90% CI ±0.10–0.12 crosses δ → "not resolvable at 1 seed"; the story moves to tokens, funnel, prune rate | Δ_A > 0: the assessment is necessary. Δ_A < 0: the LLM report hurts (e.g. through Stage B, AJ coupling) |
| J (C2b) | As A, for rule pruning | As A | Δ_J > 0: LLM pruning needed. Δ_J < 0: a pruner flaw only if φ is lost at `fun_survives_prune` under J+; else "less pruning helps" (J−a never prunes VIF < 10) |
| P (C2c) | Δ_P > 0, mostly p3/p5: *"A dedicated planner mattered on multi-step features."* | A real null only if A09 passes G3; otherwise "planner channel inert" | Δ_P < 0: pushed feedback without a planner beats pulled feedback with one; C1 moves to the fallback narrative |
| M (exploratory) | Δ_M > 0 on p3/p5; `reproposal_ratio` ≥ 1.5× under M− | Pool and Status table carry the memory | Forgetting helps |
| Falsifier (outside Holm) | Neither outcome is hoped for: it is a design check | CI of `fx_pajM` − Full crosses ±δ: inconclusive | CI inside ±δ: *"one LLM role plus a deterministic harness suffices"*. CI below −δ: the S and T LLM components jointly matter |

- PA ≈ 0 is not a safe prior, because P− pushes the report every iteration.
- **Power honesty.** At δ = 0.10 (the floor) and a planned 90% CI half-width of 0.10–0.12, TOST cannot pass even at Δ = 0. TOST can pass only if δ is clearly above the CI half-width, i.e. an oracle–Full gap of roughly ≥ 0.45. "Inconclusive" for C2a/C2b is therefore the expected outcome, and it is pre-declared.

## 4. Design

**Fraction I = −PAJM** (an odd number of "−"). It is resolution IV: main effects are aliased with three-factor interactions (P = −AJM, …), and two-factor interactions are aliased in pairs **PA = −JM, PJ = −AM, PM = −AJ**. PM and AJ are both non-zero by construction: memory under P− has no notebook, and A feeds an LLM pruner only under J+. Full and all-minus are not in this half, so the falsifier moves to the minimal harness (a declared deviation, in the pre-registration).

**Levels.** Flags are `ROGUE_ONE_<NAME>` (01 §7 R4); this section omits the prefix. "−" means the following; "+" is Full-v2.
- **P−** (`PLANNER_MODE=none`): Scientist off; the round-1 digest goes to the Extractor (A12-T5).
- **A−** (`ASSESSMENT_MODE=template`) (A12-T6).
- **J−** (`PRUNING_MODE=rules`, J−a) (A12-T7). J+ cells set `PRUNING_SHADOW=1`.
- **M−** (`MEMORY_MODE=amnesic`) (A12-T8). Under P−, M− only shortens the digest's focus history.

**Run matrix.** Manifest `manifests/agent2027/A12_FX.csv` (01 §7 R7). Every main row: SYN-P 7 task, `task_type` per the registry, `fold_seed=42`, `llm_seed=42`, `replicate=0`, `is_sentinel=0`, `rung_S/E/T=O120`, `port_S/E/T=8010`, `image_tag=agent2027-v2.3`, `cap_hours=6`, `expected_u=0.85`. Run name `A12__<arm_id>__<dataset_id>__f42__r0`; W&B `PlanBench/RogueOne-AGENT2027`, group `A12`, job_type `<arm_id>`, tags [`<SESSION_ID>`, `v2`, `agent2027-v2.3`]; `ROGUE_ONE_CODE_PHASE=v2`. `env_overrides` = the listed flags, with the prefix, as JSON (e.g. `{"ROGUE_ONE_PLANNER_MODE":"none","ROGUE_ONE_PRUNING_SHADOW":"1"}`).

| arm_id | P A J M | Overrides | Runs |
|---|---|---|---|
| `fx_pAJM` (SE\|T) | − + + + | PLANNER_MODE=none, PRUNING_SHADOW=1 | 7 |
| `fx_PaJM` | + − + + | ASSESSMENT_MODE=template, PRUNING_SHADOW=1 | 7 |
| `fx_PAjM` | + + − + | PRUNING_MODE=rules | 7 |
| `fx_PAJm` | + + + − | MEMORY_MODE=amnesic, PRUNING_SHADOW=1 | 7 |
| `fx_pajM` (minimal) | − − − + | PLANNER_MODE=none, ASSESSMENT_MODE=template, PRUNING_MODE=rules | 7 |
| `fx_paJm` | − − + − | PLANNER_MODE=none, ASSESSMENT_MODE=template, MEMORY_MODE=amnesic, PRUNING_SHADOW=1 | 7 |
| `fx_pAjm` | − + − − | PLANNER_MODE=none, PRUNING_MODE=rules, MEMORY_MODE=amnesic | 7 |
| `fx_Pajm` | + − − − | ASSESSMENT_MODE=template, PRUNING_MODE=rules, MEMORY_MODE=amnesic | 7 |
| Full anchor | + + + + | A05 `full` f42 r0 + r1 (knockout plots and the falsifier; never in the factorial model) | 0 |
| *RV3 (G4, reserve)* | P− semi-foldover | `fx_paJM` (none, template, SHADOW=1), `fx_pAjM` (none, rules), `fx_pAJm` (none, amnesic, SHADOW=1), `fx_pajm` (none, template, rules, amnesic) × SYN-P 7 | *28* |

- If D4 is adopted, rows on p3, p4 and p5 add `"ROGUE_ONE_STRATIFIED_FOLDS":"True"` (an R3 flag), exactly as A05 does.
- Rows are numbered by task: A12-001…008 = p3 (cells in table order), then p5, p2, p6a, p6b, p4, p1 (…056).
- Smoke rows A12-057…064: all 8 cells on `synp_p3` with `SMOKE_FOLDS=1`, `NUM_ITERATIONS=3` (an R3 override, smoke only), `notes=smoke`, `code_phase=smoke`. Never analysed.
- **Full guard row A12-065** (needed for guard 1 of 01 §2 on v2.3): arm `full`, `env_overrides={"ROGUE_ONE_SMOKE_FOLDS":"1"}` (10 iterations, so the Scientist prompt matches the baseline), on the SYN-P task used for `prompt-baseline-v2.0` (proposed `synp_p3`; **to verify** against the baseline artifact), `notes=smoke-guard`. Never analysed. If A13 or A14 already planned a v2.3 Full guard row, drop A12-065 and cite theirs.
- RV3 rows (only if triggered): A12-066…093, same numbering by task, cells in the order listed above.
- **Budget:** 8 × 7 × 1 = 56 runs × 0.7–1.0u = 39.2–56u. Smoke ≈ 8 × (1/5 folds) × (3/10 iterations) = 0.48 run-equivalents (0.34–0.48u); guard ≈ 0.2 run-equivalents (0.14–0.2u).

**Sessions (task-blocked).** All 8 cells of a task share one session, in random `order_key`, so cell contrasts are within-session. One A06 `sentinel` row per main session, in the first wave of the first v2.3 session (A06 guard 3).
- Smoke `S1021b` (no sentinel, A06 §4.5): A12-057…065 plus A13-901 and A14-901…904 (the A13 and A14 smoke rows, per A13 §4 and A14-T10).
- Block 1 = p3, p5, p2, p6a (A12-001…032) in `S1021a`, with A13-001…010, A14-001…018 (A14's default) and one A06 sentinel: 61 runs, 44.7–62u ≈ 32–44 h at 1.4 u/h. Launched Wed 21 ≈ 20:00, it ends Fri 23 ≈ 04:00–16:00. Without A14: 43 runs, 32.1–44u, ends Thu 22 ≈ 19:00 – Fri 23 ≈ 03:30.
- Block 2 = p6b, p4, p1 (A12-033…056) in `S1023a`, with A16-003…016 (A16-T12) and one A06 sentinel: 39 runs, 27.3–39u ≈ 19.5–28 h. Started after `S1021a` ends and after K-SET (Fri 23, ≈ 16:00–20:00), it ends Sat 24 ≈ 12:00 – Sun 25 ≈ 00:00. Block-2 rows keep `image_tag=agent2027-v2.3` (the launcher starts each row from its own `image_tag`, 01 §5.3); A16 rows use v2.4. If SET is killed at K-SET, block 2 runs alone: 25 runs, 17.5–25u ≈ 12.5–18 h.
- **If D5 = TP2:** the session uses 2 GPU ids. A second O120 server is allowed only by task (all 8 cells of a task on the same server): second session `S1021<x>`, `PORT=8011`, `SERVER_NAME=rogueone-vllm-o120-s1021<x>`, `ROGUEONE_ALLOW_SECOND_SERVER=1`, its own sentinel (A10-T8 pattern).
- **ID collision resolved; GPU sequencing still open (before A12-T9):** A04 session C also names its ladder session `S1021a`, and it uses 4 GPUs on Wed 21. A12, A13 and A14 keep `S1021a`; A04 session C was renamed `S1021m` (A04 §10) and runs before the A12 smoke or after `S1021a` ends. The A12-T11 pre-flight stops the launch if A04's containers are up.

**Failures** (01 §4.5).
- A crash (rc ≠ 0 or no `fold_aggregation_summary.csv`) is retried once, alone, at the end of the session.
- A persistent failure or a 6 h cap hit is an ITT failure: all 5 fold-episodes count `rec_sel = rec_any = 0` and `success = 0`, `itt_imputed = 1`; the run is never dropped, and the rate is reported per cell.
- Infrastructure faults (vLLM death, host reboot, W&B or Serper outage, visible in the job or vLLM log) are not ITT: they are re-queued with `notes=infra-rerun` into the next O120 session that has a sentinel, before L-LAST (Thu 29) (A05 §4 rule, proposed for X10).
- Sensitivity analysis: exclude runs whose selected features come only from the fallback.

## 5. Tasks

### 5.1 Prerequisites
- [ ] **A12-T1** [type: writing] (0.25 d, due Thu 15 Oct; committed Fri 16 in X10) — Write the FX pre-registration section: the fraction and its aliases; the restated falsifier and its test (§6); the alias-reading rules (§6, G4); the model (T16); the δ formula (A09 §6: gap = pooled `rec_sel` of A09 `oracle` minus pooled `rec_sel` of A05 Full f42 r0 + r1 over the population tasks, ITT zeros included; δ = max(0.25 × gap, 0.10), rounded to 3 decimals; δ = 0.10 if gap ≤ 0 or if any population oracle row is not terminal by Fri 23, 18:00); the token co-primary rule and its G2 branches (T3); the T5–T8 specifications; task-blocked sessions; the ITT and infra-rerun rules; the blinding rule (T14); the RV3 trigger. *Done when:* the section is in the commit tagged `agent2027-prereg`.
- [ ] **A12-T22** [type: config] (0.05 d, due Mon 12 Oct, G0a) — Hand E3 the stage-split requirement: Tester completion tokens are logged per stage as `tok_compl_tester_assess` and `tok_compl_tester_prune` (logging only; no prompt changes; e.g. a per-stage key passed through `_run_agent_with_retries`, `tst/agent.py:1329-1345`). *Done when:* the G0a smoke's W&B summary has both keys, non-zero, and their sum equals `tok_compl_tester`. Fallback if E3 lands without it: the shares are derived from `output/llm_traces.jsonl.gz` by the stage line in the system prompt (`:1440` / `:1536`), and A05's E3 check records "stage tokens not separable" (then T3 uses the joint rule).
- [ ] **A12-T2** [type: config] (0.05 d, due Mon 19 Oct, 08:00) — Check that E1–E5 are in `agent2027-v2.0`. *Done when:* an A05 run shows every R3/R4 flag as `flags/<NAME>`, non-zero `tok_compl_tester_assess` and `tok_compl_tester_prune` (or the T22 trace fallback reproduces them), and the `prompt-baseline-v2.0` artifact exists.
- [ ] **A12-T3** [type: decision] (0.05 d, due Mon 19 Oct, G2) — The E11a memo fixes the token rule (01 §4.7). Default: ≥ 30% fewer `tok_compl_total`. If a Tester stage is < 30% of `tok_compl_total` in A05, the pre-registered options are (i) the replaced stage's own tokens or (ii) the joint A−J− vs A+J+ contrast. Option (i) is degenerate for FX (the replaced stage's tokens are 0 by construction), so the recommendation is to judge A and J on `tok_compl_tester`, with (ii) if the stages are not separable. *Done when:* the chosen rule is in the G2 memo (`ANALYSIS__g2_stats__20261019`), and the A08 non-discriminating families are listed in the pre-registration.

### 5.2 Implementation (= E9d, 1.0 d; default-off; Full byte-identical)
- [ ] **A12-T4** [type: code] (0.1 d, due Mon 19 Oct, after the E9c hook) — Read `ROGUE_ONE_ASSESSMENT_MODE`, `ROGUE_ONE_PRUNING_MODE`, `ROGUE_ONE_PRUNING_SHADOW` and `ROGUE_ONE_MEMORY_MODE` in `rogueoneConfig` (`config.py:303-337`), and extend E9c's `ROGUE_ONE_PLANNER_MODE` reader with `none`. Register all five in the E1 flag registry (`rogueone/utils/flag_registry.py`). Confirm `v2_defaults.env` holds the Full values `llm`/`llm`/`llm`/`0`/`full` (01 §2 writes every R4 flag at G1; add any that are missing; the values equal the code defaults, so Full is unchanged). *Done when:* a smoke run logs all five as `flags/<NAME>` with those values.
- [ ] **A12-T6** [type: code] (0.15 d, due Tue 20 Oct) — **A−.** Replace the call at `tst/agent.py:2115` with a template: (1) active and new feature counts; (2) the `:692` metric table; (3) the top-5 features by `mean_abs_shap`; (4) "Pruned this iteration" (feature and source: `llm`, `ja_vif_cluster`, `ja_vif_pfi`, or `floor-restored`), appended after Stage B and before `:2159`.
  - Stage B gets sections 1–3; the planner gets 1–4. Under J+, section 4 lists the LLM's prunes.
  - Set `_assessment_report_produced = True`, set `_assessment_stage_status = "template"`, and write the report with `_write_assessment_report` after section 4 is appended.
  - Declared: Stage B's `get_notes_tool` (`:1107`) returns an empty notebook under A−, because Stage A's `take_note_tool` never runs. Section 4 has no A+ counterpart in the report text; the same information is in the Status/Prune Rationale table, which the planner sees at both A levels.

  *Done when:* unit tests show 0 Stage-A LLM calls and that J+ prunes without gate errors (`:1283-1289`).
- [ ] **A12-T8** [type: code] (0.1 d, due Tue 20 Oct) — **M−.** Pass `self.focus_history[-1:]` at `main.py:552`, and set `self.scientist_agent._note_book = ""` at the start of each `determine_focus` call. The prompt and tool list stay byte-identical, so `take_a_note`/`read_notes` still work within one call but nothing persists across iterations. **Declared deviation from 01 §3 E9d** ("notebook tools off"): removing the tools would change the tool list and leave `sci/agent.py:429` and `:433` pointing at missing tools; 01 E9d should be aligned to this text. *Done when:* a unit test shows a one-element history and an empty `read_notes` at the start of the second and later calls, and the trace `tools` field is identical to Full's.
- [ ] **A12-T5** [type: code] (0.35 d, due Wed 21 Oct, 12:00) — **P−.**
  - Skip `main.py:548-555`.
  - Round 1 (`ext/agent.py:971`) sends "The Scientist Agent is not available; choose the focus yourself. Start with one line `FOCUS: <one sentence>`, then build features for it." plus a digest:
    1. the last `df_scientist_feedback_pool` row (report, the frozen Full-v2 feedback columns per D2, i.e. `train_*` under the default, and `status`);
    2. the `Attribute | Status | Pruned Iteration | Prune Rationale` table;
    3. the earlier own `FOCUS:` lines (the last one only under M−).

    Each section is capped at `ROGUE_ONE_TOOL_OUTPUT_MAX_CHARS`, and the digest is logged to `output/fx_digest.jsonl`; in iteration 0 the digest reads "No previous trial in this fold."
  - Rounds 2..R send `"System now wants to focus on: {own_focus}."`.
  - If there is no `FOCUS:` line, use the first 300 characters of the round-1 output and count `focus_line_missing`. Return the focus to `main.py`, append it at `main.py:556`, and write it to the E3 focus JSONL.
  - The system prompt is unchanged (declared: `ext/agent.py:872`, `:875`, `:879`, `:892` still name the Scientist).

  *Done when:* a unit test shows 0 Scientist calls, the digest in round 1 only, and one focus-JSONL row per iteration.
- [ ] **A12-T7** [type: code] (0.3 d, due Wed 21 Oct, 12:00) — **J−a.** Replace the call at `tst/agent.py:2134`. "SHAP" = `mean_abs_shap`, "MI" = `mutual_information` from the metric profiles. Augment-protected rows (E6, if pool mode = augment) are never pruned.
  - `ja_vif_cluster`: while any active feature has VIF ≥ 1e10, prune the feature with the lowest SHAP (ties: lowest MI) among those with VIF ≥ 1e10 **and** SHAP ≤ 0.1, then recompute VIF on the remaining active training columns with `_compute_vif` (`tst/metrics.py:75`). Stop when no feature has VIF ≥ 1e10 or none of them has SHAP ≤ 0.1.
  - `ja_vif_pfi`: then prune every feature with table VIF in [10, 1e10), \|pfi_mean\| ≤ pfi_std and SHAP < 0.1.
  - VIF < 10 or missing values: never pruned. The derivative rule (`:1557-1559`) is not executed (declared).
  - The rationale is the rule label; `Pruned Iteration` is set as at `:1607-1609`; then apply the same floor (`:1617`) and all-pruned guard (`:1637-1644`), and record `_pruning_decisions` with the rule labels.
  - **Shadow** (J+ cells, `PRUNING_SHADOW=1`): run J−a on a copy of the table after Stage B's input is fixed and log its decisions to `output/pruning_shadow.jsonl`; it never enters a prompt.

  *Done when:* toy tests pass: a dummy trap prunes exactly one low-SHAP member; a high-SHAP (> 0.1) collinear pair is not pruned; a finite-VIF zero-PFI low-SHAP feature is pruned; a VIF < 10 feature is never pruned.

### 5.3 Execution on CAIR
**Rules** (01 §5.1):
- **Containers only.** Every run, analysis, unit test and W&B query runs inside Docker. The host only does git sync, copies files, runs the `~` launchers, issues `docker` commands (`docker compose run` needs `-T` over SSH) and monitors (`nvidia-smi`, `docker ps`, `tail`). Never run `python …` on the host.
- **GPU pre-flight before every GPU container.** If GPUs are busy, **stop and report** the per-GPU memory and utilisation, the compute apps and the containers to Henrik. Launch with explicit `GPUS=<ids>` and `WAIT_MAX_MIN=0`.
- **≤ 4 of the 8 H100s** without Henrik's explicit approval in chat; one Rogue One job at a time.
- **W&B `PlanBench/RogueOne-AGENT2027` is the system of record**; runs are stateless and upload `run-output`; secrets come from `.env` (`WANDB_API_KEY` from `<CAIR_HOME>/code/PlanBench/.env`; `SERPER_API_KEY` from the image `.env`/`.secret.env`). Never push the images.
- Long jobs: detached `nohup` launcher, job log in `~` (never in the root-owned run folders).

All commands are one-shot: `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "…"`. `<ROGUEONE_CAIR_PATH>` = `<CAIR_HOME>/code/RougeOne` (**INFERRED, verify** once with `ssh … "ls -d ~/code/RougeOne"`). `<ids>` = the GPU ids from the pre-flight (4 at TP4, 2 at TP2).

- [ ] **A12-T9** [type: config] (0.1 d, due Wed 21 Oct, 13:00) — Locally: `make_manifest.py --ablation A12` → `manifests/agent2027/A12_FX.csv` (rows 001–065; RV3 rows 066–093 are generated only if T23 fires). Then `--plan-session S1021b --rows A12-057…065 A13-901 A14-901…904` and `--plan-session S1021a --rows A12-001…032 A13-001…010 A14-001…018 <next unused A06 sentinel row>` (A13/A14 rows only if their own checks passed; A13 §4, A14 §4). Commit, tag `agent2027-v2.3` (if not already tagged for E9e/E8b/E7), and push branch and tag. *Done when:* both session CSVs are committed, validate against 01 §7 R7, `S1021a.csv` holds no A04 row, and a dry run yields unique container names `ro27-s1021a-a12-0nn`.
- [ ] **A12-T10** [type: run] (0.1 d, due Wed 21 Oct, 14:00) — Sync, build and install (01 §5.2 steps 0, 2, 3). v2.3 is shared with A13, A14 and A20: if `rogue_one:agent2027-v2.3` already exists from the same HEAD, skip the build and record its ID.
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15' git fetch git@github.com:HenrikNorkart/RougeOne.git agent2027/v2 && git checkout -B agent2027/v2 FETCH_HEAD && git log -1 --format='HEAD: %h %s'"
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && test ! -e .dev.env && docker build -t rogue_one:agent2027-v2.3 . && docker image inspect rogue_one:agent2027-v2.3 --format '{{.Id}}'"
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && cp scripts/agent2027/ro27_session_launch.sh scripts/agent2027/ro27_clients_manifest.sh scripts/agent2027/ladder_session.sh ~/ && cp paper/scripts/vllm_session.sh ~/ro27_vllm_session.sh && mkdir -p ~/ro27_manifests && cp -r manifests/agent2027/. ~/ro27_manifests/"
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker run --rm --entrypoint python3 rogue_one:agent2027-v2.3 -m pytest tests/unit -q"
  ```
  *Done when:* HEAD matches the pushed tag, the image ID is recorded in the session plan, `~/ro27_manifests/sessions/S1021b.csv` and `S1021a.csv` exist, and the in-image unit tests (T5–T8) pass.
- [ ] **A12-T11** [type: run] (0.25 d, due Wed 21 Oct, 18:00; G0b-1) — Pre-flight, then run the smoke session `S1021b`.
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader && echo '--- compute apps ---' && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader && echo '--- containers ---' && docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'"
  ```
  Proceed only if the target GPUs show < 2,000 MiB used, the total requested is ≤ 4, and no `ro27-*`, `rogueone-vllm-*`, `llmfe-*` or `caafe-*` container is running (A04 session C and the A08 GPU part run the same days). **Otherwise STOP** and report memory, utilisation, apps and containers to Henrik.
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "SESSION_ID=S1021b SESSION_FILE=~/ro27_manifests/sessions/S1021b.csv GPUS=<ids> MODEL=openai/gpt-oss-120b PORT=8010 SERVER_NAME=rogueone-vllm-o120-s1021b LOG=~/vllm_session_S1021b.log IMAGE_RO=rogue_one:agent2027-v2.3 PARALLEL=6 CPUS=12 RUN_CAP_HOURS=6 RUNROOT=~/rogueone-runs-agent2027 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_S1021b.job.log 2>&1 & echo \$! > ~/ro27_S1021b.pid; sleep 5; tail -5 ~/ro27_S1021b.job.log"
  ```
  Wait for `RO27_SESSION_END` with the T13 loop (session `S1021b`, ≈ 1–1.5 h). Read W&B summaries in a container (T13 query) and the prompt dumps from each row's `run-output` (A09-T12 pattern: `fetch_wandb.py` + `prompt_diff.py` in a `--cpus 4` container; CLI flags per E5/E11b, **to verify**). *Done when* all of the following hold:
  - all 9 A12 smoke rows end `OK` (rc 0 and `fold_aggregation_summary.csv`);
  - `llm_calls_scientist = 0` in P− rows, and `output/fx_digest.jsonl` has one record per iteration;
  - no Stage-A LLM call in A− rows and no Stage-B LLM call in J− rows (by stage key, or by the stage line in the traces);
  - in J− rows every `Prune Rationale` is `ja_vif_cluster` or `ja_vif_pfi`; `prune_rate` differs between the J+ and J− cells, or, if equal, `prune_jaccard` between a J+ row's LLM prunes and its shadow decisions is < 1 in at least one iteration;
  - `output/pruning_shadow.jsonl` exists in J+ rows only, and no prompt dump or trace contains shadow text;
  - in `fx_PAJm` and `fx_Pajm`, the iteration-2 Scientist prompt lists exactly one prior focus;
  - `prompt_diff.py` shows the A12-065 Full row byte-identical to `prompt-baseline-v2.0` (tool descriptions compared via the trace `tools` field).

  If any check fails, FX does not launch that evening (§6, Dates).
- [ ] **A12-T12** [type: run] (0.05 d, Wed 21 Oct, ≈ 20:00; check at ≈ 22:00) — Re-run the T11 pre-flight, then:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "SESSION_ID=S1021a SESSION_FILE=~/ro27_manifests/sessions/S1021a.csv GPUS=<ids> MODEL=openai/gpt-oss-120b PORT=8010 SERVER_NAME=rogueone-vllm-o120-s1021a LOG=~/vllm_session_S1021a.log IMAGE_RO=rogue_one:agent2027-v2.3 PARALLEL=6 CPUS=12 RUN_CAP_HOURS=6 RUNROOT=~/rogueone-runs-agent2027 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_S1021a.job.log 2>&1 & echo \$! > ~/ro27_S1021a.pid; sleep 5; tail -5 ~/ro27_S1021a.job.log"
  ```
  About 2 h in, repeat the T11 switch checks on the live W&B history of the running A12 rows (or on the first finished rows if summaries are written only at run end). If any fails, stop the session:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "pkill -TERM -f 'ro27_(session_launch|clients_manifest|vllm_session)\.sh'; docker ps --format '{{.Names}}' | grep -E '^(ro27-s1021a-|rogueone-vllm-o120-s1021a$)' | xargs -r docker stop"
  ```
  (one Rogue One job at a time, so the `pkill` pattern hits only this session). Merge a default-off fix that passes the 01 §2 guards (new image `agent2027-v2.N`), regenerate the affected rows with that `image_tag`, and rerun them from the reserve before L-LAST. Rows from a defective image are never analysed. *Done when:* the version assert, both image IDs and the first `ro27-s1021a-*` rows appear in the job log.
- [ ] **A12-T13** [type: run] (0.2 d, Thu 22 – Sun 25 Oct; due Mon 26 Oct, 08:00) — Monitor in the background; exit code 255 means the network dropped, not the job, so reconnect:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 CAIR-gpu14 "while ! grep -q 'RO27_SESSION_END' ~/ro27_S1021a.job.log 2>/dev/null; do sleep 120; done; tail -30 ~/ro27_S1021a.job.log"
  ```
  - W&B states inside a container; a `running` run with no live container is a zombie:
    ```
    ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker run --rm -e HOME=/tmp --env-file <CAIR_HOME>/code/PlanBench/.env --entrypoint python3 rogue_one:agent2027-v2.3 -c \"import wandb; api=wandb.Api(); [print(r.name, r.state) for r in api.runs('PlanBench/RogueOne-AGENT2027', filters={'group': 'A12'})]\""
    ```
  - Tear-down check after each session: `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker ps --format '{{.Names}}' | grep -E '^(ro27-|rogueone-vllm-)' || echo none; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"`.
  - Block 2 (Fri 23, after K-SET and after `S1021a`'s `RO27_SESSION_END`): `make_manifest.py --plan-session S1023a --rows A12-033…056 A16-003…016 <next A06 sentinel row>` (A16-T12; without the A16 rows if SET was killed), commit, push, re-sync `~/ro27_manifests/` (T10, third command), fresh T11 pre-flight, then the T12 launch with `SESSION_ID=S1023a`, `SERVER_NAME=rogueone-vllm-o120-s1023a`, `LOG=~/vllm_session_S1023a.log`, `IMAGE_RO=rogue_one:agent2027-v2.4` (`rogue_one:agent2027-v2.3` if SET was killed), and job log `~/ro27_S1023a.job.log`.
  - Infra faults: re-queue per §4 Failures.

  *Done when:* all 56 runs are `finished` or ITT-classified in W&B, each with a `run-output` artifact, and the tear-down check prints `none`.

### 5.4 Analysis (CPU containers, `--cpus` ≤ 16, `~/rogueone-synp-meta/` mounted read-only)
Container pattern (A10 §5.4): `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "mkdir -p ~/ro27_analysis/A12 && docker run --rm --cpus 16 -e HOME=/tmp --env-file <CAIR_HOME>/code/PlanBench/.env -v ~/rogueone-synp-meta:/meta:ro -v ~/ro27_analysis/A12:/out --entrypoint python3 rogue_one:agent2027-v2.3 /workspace/scripts/agent2027/analysis/<script>.py <args>"` (CLI flags as built in E11b; **to verify**). Every analysis run logs to W&B group `ANALYSIS` as `ANALYSIS__<script>__<YYYYMMDD>`.
- [ ] **A12-T14** [type: analysis] (0.02 d, due Fri 23 Oct, 18:00) — **Blinding.** No detector runs on A12 rows until A09-T15 has committed δ and posted `rt3-oracle-gap`. Token, funnel and process metrics may be read earlier. *Done when:* the δ commit time precedes the first A12 detector log.
- [ ] **A12-T15** [type: analysis] (0.1 d, Sat 24 – due Mon 26 Oct) — `analysis/fetch_wandb.py --group A12` → `analysis/recovery_detector.py` (plus A05 `full` f42 r0 + r1 on SYN-P 7 for the anchor). *Done when:* 280 A12 episodes are scored (56 runs × 5 folds, ITT included) and logged as `ANALYSIS__recovery_detector__<YYYYMMDD>`.
- [ ] **A12-T16** [type: analysis] (0.25 d, due Tue 27 Oct, G4) — `analysis/glmm_holm.py` (FX mode), 8 cells only, confirmatory tasks only (SYN-P 7 minus the A08 non-discriminating families):
  - run-level binomial counts (k of 5), beta-binomial, ±1 coding: `rec ~ P + A + J + M + PA + PJ + PM + (1|task)` (PA = −JM, PJ = −AM, PM = −AJ);
  - Δ_X = predicted mean over the X+ cells − over the X− cells, tasks equally weighted;
  - C2a and C2b by TOST (90% CI against ±δ; p = the larger of the two one-sided p, which enters Holm); C2c two-sided;
  - fallback model, used if the beta-binomial fit does not converge or its Hessian is not positive definite: Firth logistic with task fixed effects;
  - tokens: the same fixed effects on log tokens (linear mixed model, outcome per T3), giving the ratio X−/X+ = exp(−2·β_X) with a 90% CI;
  - falsifier: the same beta-binomial model `rec ~ arm + (1|task)` on `fx_pajM` r0 vs A05 `full` f42 r0 + r1, Δ = predicted mean difference with tasks equally weighted, 90% CI.

  *Done when:* `ANALYSIS__glmm_holm__20261027` is logged with Δ, CIs and p for P, A, J, M, the three alias pairs, the token ratios and the falsifier.
- [ ] **A12-T17** [type: analysis] (0.25 d, due Tue 27 Oct) — `process_metrics.py` and `funnel.py`, per cell: `prune_rate`, `prune_jaccard` (J+ LLM vs shadow), `floor_activations`, `reproposal_ratio`, `report_read_rate` (P+ cells), the funnel stage where φ is lost, `fallback_share` plus a fallback-excluded refit, `success`, `cap_hit`, `focus_line_missing`; one-at-a-time knockouts against the A05 anchor (descriptive). *Done when:* the tables are logged in group `ANALYSIS`.
- [ ] **A12-T18** [type: decision] (0.1 d, due Tue 27 Oct, G4) — Alias read-out and the RV3 decision (§6); hand `fx_pajM` to A19 and A20, and `fx_pAJM` to A16. *Done when:* the alias verdicts, the RV3 yes/no with the reserve figure, and the hand-offs are in the G4 memo.
- [ ] **A12-T23** [type: run] (0.15 d; launch by Wed 28 Oct, 20:00; only if T18 triggers RV3) — Generate rows A12-066…093 (§4), plan `S1028<x>` (next free letter) with one A06 sentinel, and launch with the T11 pre-flight and the T12 command. Refit with T16 on the 12 cells: `rec ~ P + A + J + M + PA + PJ + PM + AJ + AM + JM + block + (1|task)` (block = fraction vs foldover session; rank 12, so all two-factor interactions are estimable if three-factor interactions are negligible). *Done when:* 28 runs are terminal by Fri 30 Oct (before DF Sat 31, 24:00) and the refit is logged as `ANALYSIS__glmm_holm__<YYYYMMDD>`.
- [ ] **A12-T19** [type: analysis] (0.1 d, due Tue 3 Nov) — The final C2a–c p-values enter Holm with the rest of the family (smallest threshold 0.005 at 10 contrasts; recomputed if C4b, C1a or C4a are dropped). Write the raw p and "Holm-adjusted p < 0.05", never "< 0.005". *Done when:* the Table 2 rows C2a–c are filled with verdicts per §6.

### 5.5 Paper outputs
- [ ] **A12-T20** [type: writing] (0.5 d, due Fri 6 Nov) — Fig. 2 left: Δ_P, Δ_A, Δ_J, Δ_M and the alias pairs on `rec_sel` and on tokens, with the minimal harness marked. Table 2 rows C2a–c and the §3 sentence. *Done when:* both are in the AGENT draft.
- [ ] **A12-T21** [type: writing] (0.25 d, due Thu 12 Nov) — Appendix: alias table, deviation notes (falsifier on the minimal harness; M− notebook; P− system prompt; J−a derivative rule), the T5–T8 specifications, per-cell funnel, Jaccard, rendered prompts. *Done when:* it is in the anonymised replication package.

## 6. Decision rules and gates
- **C2a / C2b verdicts** (token ratio = X−/X+ on the T3 outcome). Applied in this order; the first match is the verdict, and the CI is always reported:
  1. *reducible* = Holm-significant TOST **and** the upper 90% bound of the token ratio ≤ 0.70;
  2. *equivalent, no saving* = Holm-significant TOST only;
  3. *necessary* (secondary, unadjusted) = the two-sided 95% CI of Δ_X lies above 0;
  4. *LLM version worse* (secondary, unadjusted) = the 95% CI lies below 0 (for J, read with the §3 funnel rule);
  5. otherwise *inconclusive* (expected).
- **C2c:** Holm-significant Δ_P > 0 → "the separate LLM planner (pulled feedback) beat SE|T (pushed last-trial digest)"; Holm-significant Δ_P < 0 → "push without a planner wins"; otherwise inconclusive, worded by G3 (Mon 26): if A09's planner half passes, "no detectable planner effect"; if it fails, "planner channel inert on these tasks at 120B".
- **Falsifier** (outside Holm; T16): 90% CI of `fx_pajM` − Full inside ±δ → the paper says "one LLM role plus a deterministic harness suffices"; CI entirely below −δ → "the S and T LLM components jointly matter"; otherwise inconclusive. The session and image confound (A05 ran in `S1016a/b` on v2.0) is disclosed and guarded by the A06 sentinels.
- **G4 (Tue 27):** alias pairs are reported jointly, never attributed to one member. If the Wald p of any pair coefficient (PA, PJ or PM in T16) is < 0.05 (unadjusted) **and** the G4 memo shows ≥ 28u of unspent reserve (01 §4.9), launch **RV3** (A12-T23) by Wed 28, 20:00. Otherwise the pair is reported as "unattributable".
- **Dates:** the G0b-1 smoke (T11) must pass, or FX waits. A launch after Thu 22 drops the G4 actions (01 §6.1). If FX has not launched by Wed 28, 08:00, C2a–c are dropped before any data exist and Holm is recomputed. L-LAST Thu 29; DF Sat 31, 24:00.

## 7. Risks and cut levers
1. **Power:** a CI of ±0.10–0.12 means TOST is likely inconclusive (cannot pass at δ = 0.10); process metrics carry the story.
2. **Margin:** if oracle ≈ Full, the 0.10 floor applies.
3. **P− is also pull→push, and last-trial only:** the digest carries only the last feedback row; report `report_read_rate` in P+ cells.
4. **J− is also less pruning:** report `prune_rate` per cell and task type.
5. **Token rule may be unreachable for one stage:** fixed at G2 (T3); stage tokens need T22.
6. **Fallbacks and partial M−:** the pool and Status table still carry memory; count fallbacks per cell, and disclose.
7. **One developer:** E9d (1.0 d) is spread over Mon 19 – Wed 21 next to E9c, E9a/E9b, E9e, E8b's backend and E7 (01 §6.2 lists them on the same days). If E9d lands after Wed 21, run the blocks in priority order and lose G4.
8. **Box contention:** `S1021a` holds the 4-GPU O120 slot until Fri 23 ≈ 04:00–16:00, overlapping the Thu 22 plans of A19, A20, A17 O20 and the A16 pilot (01 §6.2). Sequencing is Henrik's call; under D5 = TP2 the other 2 GPUs remain usable.
9. **Session-ID collision** with A04 session C (§4): must be resolved before T9.

FX is not in CL1–CL7. Dropping a task needs a pre-registration amendment. RV3 is paid from the reserve only.

## 8. Definition of done
- [ ] Pre-registration section committed (Fri 16); δ committed before unblinding (Fri 23, 18:00).
- [ ] E9d in v2.3; Full byte-identical (A12-065 or the shared v2.3 guard); smoke passed.
- [ ] 56 runs in W&B group `A12` with `run-output`; ITT applied; infra reruns done before L-LAST.
- [ ] GLMM, TOST, token, falsifier and process tables; G4 memo (and the RV3 refit if triggered).
- [ ] Hand-offs to A16, A19 and A20; Table 2, Fig. 2 and appendix written.

**New names introduced:** `tok_compl_tester_assess`, `tok_compl_tester_prune`, `focus_line_missing`; `output/fx_digest.jsonl`, `output/pruning_shadow.jsonl`; rule labels `ja_vif_cluster`, `ja_vif_pfi`; RV3 arms `fx_paJM`, `fx_pAjM`, `fx_pAJm`, `fx_pajm`; rows A12-065 (Full guard) and A12-066…093 (RV3); sessions `S1021b` (smoke), `S1021a`, `S1023a`, `S1028<x>` (RV3).

**Open items for Henrik / other documents:**
1. ID collision resolved: A04 session C is `S1021m`; `S1021a` stays with A12/A13/A14. Still open: sequence A04 session C against the FX smoke `S1021b` and the `S1021a` launch (01 §6.4, box order).
2. 01 §3 E9d says the M− notebook tools are "off"; this document keeps them and clears the notebook per call (T8). Align 01.
3. 01 §3 E3 does not list the Tester stage split that the FX token rule needs; T22 requests it by G0a.
4. The SYN-P task of `prompt-baseline-v2.0` (assumed `synp_p3`) and the CLI flags of `prompt_diff.py`, `fetch_wandb.py` and the E11b scripts are **to verify**; `<ROGUEONE_CAIR_PATH>` and the server git remote are inferred.
