# A16 · K-SET read-out (A16-T11), 2026-10-22

Rule: `docs/ablations/A16_RT2_set_phased.md` §6 ("K-SET, Fri 23 Oct"), with Henrik's 2026-10-21
decisions (pre-registration Amendment 6: wording as drafted, 29 rows). Every criterion must hold in
**each** of the two pilot runs; rates are per 100 `llm_calls_total`.

Pilot runs: session `S1022b`, image `rogue_one:agent2027-v2.4` (tag `agent2027-v2.4` = `ca92424`, pushed),
O120 TP2, launched 2026-10-19 16:55Z.

| Row | Task | Status | wall_s | `llm_calls_total` |
|---|---|---|---|---|
| A16-001 (`set_phased`, pilot) | `blood` (p = 4) | OK, rc 0 | 5,160 | 2,139 |
| A16-002 (`set_phased`, pilot) | `communities` (p = 102) | OK, rc 0 | 6,470 | 2,309 |

## Verdict: **GO** (all 7 criteria pass in both pilot runs; no re-pilot, no ED4)

| # | Criterion | A16-001 `blood` | A16-002 `communities` | Pass |
|---|---|---|---|---|
| 1 | Finishes within 6 h with `fold_aggregation_summary.csv` | 1.43 h, summary present | 1.80 h, summary present | yes |
| 2 | No phase exhausted by context errors (S2), and `context_errors` ≤ 2 / 100 calls | 0 of 200 phase-iterations exhausted; 0 fallbacks; `context_errors` 0 (0.00 / 100) | 0 of 200 exhausted; 0 fallbacks; 0 (0.00 / 100) | yes |
| 3 | `offphase_tool_calls` ≤ 5 / 100 calls | 1 (T-B) = 0.05 / 100 | 2 (E) = 0.09 / 100 | yes |
| 4 | Non-zero completion tokens for all three roles | S 195,791 · E 370,156 · T 265,064 | S 211,948 · E 420,484 · T 270,456 | yes |
| 5 | A16-T6 dump checks on the pilot dumps | 1 distinct system prompt across all 4 phases × 50 iterations; role-word grep 0 hits | same: 1 distinct system prompt; 0 hits | yes |
| 6 | Full byte-identity guard (A16-T8) | A16-027 (`synp_p3`): 80/80 files identical (`S1019_guards.md`) | A16-028 (`heart`): 81/81 identical (below) | yes |
| 7 | `scan_leakage.py` flags no pilot fold | 0 flagged (5 folds scanned) | 0 flagged (5 folds scanned) | yes |

The smoke row A16-000 (`set_phased`, `synp_p3`, 1 fold) also passes: OK in 1,471 s, `context_errors`
0, `offphase_tool_calls` 0, S/E/T tokens 32,709 / 77,926 / 57,700, 1 distinct system prompt over 10
iterations, 0 role-word hits.

`max_prompt_tokens`: 41,485 (`blood`), 58,920 (`communities`, the wide table), against `MAX_MODEL_LEN`
131,072, so the overflow risk (§7 risk 1) did not materialise at p = 102.

## How each criterion was computed

- **1, 3, 4, 2 (rate):** `output/instrumentation_summary.json` of each run (keys `llm_calls_total`,
  `context_errors`, `offphase_tool_calls{,_s,_e,_ta,_tb}`, `tok_compl_{scientist,extractor,tester}`,
  `wall_s`) and the launcher job log `~/ro27_S1022b.job.log`.
- **2 (S2 exhaustion):** `output/set_phased_log.jsonl` (50 rows = 5 folds × 10 iterations, 4 phases
  each): `"exhausted": false` in all 200 phase-iterations of each run, and every `fallbacks` list empty.
  With 0 context errors, S2's "fallback **and** a context error in the same phase-iteration" cannot
  hold anyway.
- **5 (A16-T6 dump checks):** the `output/prompts/` dumps (A16-001 and A16-002: 600 files each;
  A16-000: 120), copied read-only. (a) Every `*.system.txt` of a run has the same md5: one distinct
  system prompt, so it is byte-identical across the 4 phases of every iteration. (b)
  `grep -iE "you are the|team of agents|scientist|extractor|tester"` over all system and phase
  (`*.user.txt`) messages: 0 hits even *without* removing `context_and_goal`. `tools.json` files were
  not grepped (tool docstrings are unchanged by design and declared, A16-T6 "Not changed").
- **6 (guard 1):** the guard-1 comparison of `S1019_guards.md` (all LLM-independent files present in both
  dumps: every role system prompt and `tools.json`, plus the Scientist's round-1 user message). A16-028
  (`heart`) against the `prompt-baseline-v2.0` source dump `S1016t`/A05-902 (`heart`): both dumps have
  the same 120 file names; 81 files compared (80 system/tools + `scientist__focus__iter000__c01.user.txt`),
  **81 identical**. This closes the open item in `S1019_guards.md` ("compared when the `heart` baseline is
  fetched").
- **7 (scan):** `paper/scripts/scan_leakage.py` unchanged, in a CPU-only `rogue_one:agent2027-v2.4`
  container (`--network none --cpus 8`), with the pilot run folders mounted read-only at its hard-coded
  seed-42 roots (`/w1` = A16-001, `/w2` = A16-002). It read `df_test_pool_final.csv` (50 completed
  trials each) and printed `flagged folds: 0`. Ceilings used: `blood` 0.774, `communities` 0.711.

## Caveats

- **Timing.** The pilot ran Mon 19 Oct (earlier than the planned Thu 22 night), on the frozen v2.4 code.
  The SET wording Henrik approved on Wed 21 is the wording that ran (v2.4 unchanged since). The verdict
  is applied on Thu 22, before the Fri 23 deadline.
- **Dump naming.** In the dumps the Tester phases are indexed per fold (5 iteration indices with 10
  calls each), while S and E are indexed per iteration (50). Check 5(a) holds regardless, because the run
  has a single distinct system prompt.
- **Not done here:** the W&B run `ANALYSIS__k_set__20261023` (A16-T11's done-when) is not logged; this
  file is the read-out.

## Contingency switches (not triggered)

Neither K-SET contingency exists in code (no match for the elision text, the "Tools available in this
phase" line, or any switch in `rogueone/`). Under §6 they are applied **only** if criterion 2 or 3
fails; both pass by a wide margin, so neither is triggered.
- **Overflow contingency** (criterion 2): in `rogueone/utils/topology.py`'s `SharedIterationSession`,
  replace tool-output bodies of completed earlier phases with `[tool output elided]` when a later
  phase's input is built, behind a default-off flag. About 0.3–0.5 d with tests, plus a guard-1 rerun
  and a new image.
- **Tool-line contingency** (criterion 3): append "Tools available in this phase: …" to `PHASE_HEADER`
  in `rogueone/llm/prompts/set_phased.py` behind a default-off flag, and add the line to the frozen
  text. About 0.1–0.2 d, plus the same guard rerun and image.

Building either now would mean a new image (v2.5) for no triggered need. A16's analysed rows should run
on `agent2027-v2.4`, the image the pilot validated.

## Manifest and launch sets

`manifests/agent2027/A16_RT2.csv` has 31 rows: 29 `set_phased` + 2 `full`. Under A16-T9's done-criterion
("29 rows: 26 analysed or pilot + 3 smoke"), the 29 are A16-000 (smoke), A16-001–002 (pilot),
A16-003–026 (24 analysed) and A16-027–028 (`full` smoke-guards). The other two rows, **A16-029–030**
(`repilot`), were added in advance. With K-SET GO they are never launched, which gives the approved
29 rows. They can be left in the manifest as "cancelled (not needed)" or dropped in a dated manifest
commit.

- **A16-T12 (SYN-P, 14 rows):** A16-003–016, on `synp_p1`, `p2`, `p3`, `p4`, `p5`, `p6a` and `p6b`, each
  at fold seed 42 (paired with A05 f42 r0/r1) and fold seed 4 (paired with A05 f4 r0). Stratified on
  p3, p4 and p5. Image `agent2027-v2.4`, O120.
- **A16-T13 (Core-10, 10 rows):** A16-017–026: `balance-scale`, `junglechess`, `vehicle`, `heart`,
  `credit-g`, `myocardial`, `housing`, `insurance`, `crab` and `wine`, at f42 r0. Image
  `agent2027-v2.4`, O120, last launch Thu 29 (L-LAST).
- Every A16 session carries its own A06 sentinel row (§5.3 / A06 §4.5), with the sentinel's `order_key`
  at the session minimum.


## Decision (Henrik, 2026-10-22)

GO stands. The pilot rows ran on Mon 19 Oct, before the Wed 21 sign-off, on the identical approved v2.4
wording; this is disclosed. The contingency switches are not built. A16-029/030 are cancelled.
