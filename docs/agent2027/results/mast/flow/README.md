# flow/ — RQ1 information-flow pipeline (EXPLORATORY, post hoc)

Everything in this folder is exploratory and outside the pre-registered Holm family.

## Files
| file | content |
|---|---|
| `flowlib.py` | trace loading and per-cycle event reconstruction |
| `replay.py` | re-executes every `append_new_attribute` command of a run and scores it against phi (value-based detector) -> `replay/<row_id>.csv` |
| `build_cycles.py` | builds `cycles.csv` (one row per run x fold x iteration) |
| `analyze.py` | funnel, fold-level, first-mention, focus-specificity, effort tables -> `tables.md`, `funnel.csv`, `folds.csv`, `first_mention.csv`, `cycles_analysis.csv` |
| `focus_eval.py` | value-based check of formulas written in the focus / Scientist text (`focus_phi_exact`, `S_phi_exact`), appended to `cycles.csv` and `cycles_analysis.csv` |
| `search_cycles.csv` | per cycle (P2/P4/P6b): number of Scientist web searches and whether the returned text contains the phi-op keywords |
| `make_packets.py` | RQ2 annotation packets -> `../packets/U##.md`, key -> `../packets_key.csv` |
| (REPORT) | findings were returned to the main session as text (the subagent was not allowed to write report files); `tables.md` holds all numbers |

Run order: `python -X utf8 replay.py <row_id>` for each run (`runs.txt`), then `build_cycles.py`, `analyze.py`,
`focus_eval.py`, `make_packets.py`. Examples in REPORT.md were pulled by hand with ad-hoc queries on these tables (ids given).

## Trace format and mapping (verified by hand on A05-034, A16-006, A05-040, A16-004, A05-037, A16-009)

* `llm_traces.jsonl.gz`: one record per LLM call, sorted by `seq`. Every record carries `role`
  (scientist / extractor / tester), `stage` (focus, search, append, assessment, pruning), `agent_name`,
  `fold_run_id` (== `test_fold_id`), `iteration_in_fold` (0-9) and `index` (global trial = 10*fold + iteration).
  **Cycle attribution is read off these fields; nothing is inferred from timestamps.** 5 folds x 10 iterations
  = 50 cycles per run, 57 runs -> 2850 cycles.
* `messages` is the call's input history (list of dicts; assistant `tool_calls` inside the history are
  Python-repr strings), `message` is the call's output (dict with `content` or `tool_calls`; 5 calls in A05-034
  have `message=None` = failed call, counted as `n_nomsg`).
* Event reconstruction: for each call we take its OUTPUT message (tool calls with name + JSON arguments, or text).
  The tool result is looked up by `tool_call_id` in any later call's input history (role `tool`). This is robust to
  the fact that Full keeps one session per role per iteration (`scientist_agent_session_step_N`,
  `extractor_agent_session_step_N`, `tester_agent_assessment_step_N`, ...) whereas the single agent keeps one
  session per iteration (`set_phased_trial_N`) holding all four phases.
* Single agent (`set_phased`): calls carry an extra `phase` field (S, E, T-A, T-B) and the same `role`/`stage`
  values as Full (S=scientist/focus, E=extractor/append, T-A=tester/assessment, T-B=tester/pruning), so role
  attribution is identical in both arms. `set_phased_log.jsonl` confirms one session per trial with four phases
  and reports `offphase_tool_calls` (0 in the runs inspected). A new conversation is started for every
  iteration: the single agent does NOT see earlier iterations' conversation (only notes / test-result tables via
  tools), exactly like the Full agents.
* Web search: the Scientist's `search_tool` spawns a `WebSearchAgent` sub-agent (records with
  `agent_name=WebSearchAgent`, `role=scientist`, `stage=search`; in single-agent runs also 20-ish with
  `role=extractor`). These sub-agent calls are excluded from "Scientist's own reasoning"; the query the Scientist
  sent is kept (`S_search_*` columns) and the returned answer is a tool result (not counted as reasoning).
* Focus = the Scientist's last non-empty text output in the cycle; cross-checked with `focus_log.jsonl`
  (`focus_matches_log` column). The Extractor receives it verbatim as `System now wants to focus on: <focus>`
  (Full: as the only user message, re-sent at each of the ~10 runner turns; single: appended to the shared
  conversation after the PLAN messages).
* Extractor features: every `append_new_attribute` call (args `name`, `command`, `argumentation`/explanation). The
  result string gives the status: `added to df_features` = accepted, `was pruned — restored` = restored,
  `already exists` = exists, anything else = error (mostly `Invalid JSON input ... args.argumentation Field
  required`, i.e. a schema error in the call; the code itself is often fine and is still replayed as an attempt).
* `df_entities_attributes_final.csv` / `df_attributes_explanations_final.csv` cover only the LAST fold
  (Trials 40-49) and cannot map features of folds 0-3; they are used only to rebuild the raw data (seed columns).
* Tester: assessment text = last text output of the `assessment` stage (fallback `tester_reports/tester_report_iteration_<index>.md`),
  pruning = `attribute_pruning_tool` calls (`attribute_names_list`) in the `pruning` stage.
* Tokens per role and cycle: summed `usage` over the cycle's calls by `role` (search sub-agent included in the
  role that called it); `iteration_log.jsonl` per-role tokens are carried along for comparison (`itlog_*`).
* Tool errors per role: tool results matching an error pattern (see `flowlib.ERR_RE`) + failed calls.
* Hidden reasoning (gpt-oss) is NOT logged (`reasoning_chars` only), so "Scientist reasoning" means visible
  tool-call arguments and text only.

## Phi in Extractor code: value-based replay (E_phi_code)
Static/regex matching of code was too unreliable (e.g. `f05/(f07*f13)` and `load/(area*length)` contain all
columns and both operators but are not phi). Instead `replay.py`:
1. rebuilds the raw table (2000 rows) from the seed columns of `df_entities_attributes_final.csv` (categoricals
   decoded through the ordinal maps in the seed commands; missing values were 0 in the seeds; the label is not
   available and is replaced by random noise — phi is label-free);
2. replays every `append_new_attribute` command of a fold in call order (`df`, `df_features`, `pd`, `np` in scope;
   accepted features are added to `df_features` so later code that references them works; commands that import
   anything except numpy/pandas/math/scipy/sklearn or touch files are not executed);
3. scores each result with the repo detector's criterion (`scripts/agent2027/analysis/recovery_detector.py`):
   |Spearman| >= 0.95 with phi (rule family P3/P5: or NMI >= 0.9), excluding raw columns, weakly monotone
   single-raw-column transforms, and Spearman hits that do not beat the best raw column by 0.01.

**Validation:** fold-level "some accepted feature recovers phi" vs the official `rec_any` in
`episodes_final.csv`: 283/285 folds agree (2 folds where the replay finds phi but the official detector does not: A16-009 fold 3, a PCA component at rho 0.965 fitted on all rows; A16-801 fold 1, a product written at iteration 7). Differences: all rows instead of the fold's
training rows; restored/duplicate features are not re-scored.

`E_phi_code` = an ACCEPTED feature written in this cycle recovers phi. `E_phi_attempt` = any attempt (incl.
schema-rejected calls) recovers phi. `E_phi_cols` / `E_phi_op_static` are the static (regex) variants, kept for
reference.

## Text keyword sets (case-insensitive) — S_reason_phi, focus_phi, T_assess_*
Column mention: column name with `-`, `_` or space interchangeable, not inside a longer identifier
(`hours_since_start_sq` does NOT count as a mention of `hours_since_start`; backticked `hours_since_start` does).
`k of n` = number of distinct phi columns mentioned.

Operation mention (`*_op`), all groups must match (AND) unless a "complete" term matches:
* DIV = `ratio|divid|quotient| / |x/y|÷|per (unit|area|cross|section|length|metre|meter)|normali[sz]`
* MUL = `product|multipl| * |x*y|×|times|interaction`
* P2 (a/b*c): DIV AND MUL; complete: `elongation|deformation|deflection|axial (displacement|extension)|PL/A`
* P6a (a*b/c): DIV AND MUL; complete: `reynolds`
* P6b (a*b/c, anonymous f05,f13,f07): DIV AND MUL
* P4 (sin(2*pi*h/24)): `sin|sine|sinus|cos|cosine|fourier|periodic|cyclic|circadian|diurnal|24h|24-hour|hour of (the) day|time of day|daily cycle|% 24|mod 24`
* P3 ((a/b >= thr) & e==lvl): DIV AND `threshold|cutoff|>=|≥| > |exceed|above|flag|indicator|binary|high-` AND `self[_ -]?employ`
* P5 (sign(a*b^2 - c*d^2)): `torque|moment|lever|weight*distance|product|multipl` AND `squared|**2|^2|²|square|pow(` AND `sign|np.sign|differen|minus|compar|heavier|balance|tip`

`S_reason_phi` = Scientist's own tool-call arguments + non-final text in the cycle mention all n phi columns AND
the op. `focus_phi` = the same on the focus text. `focus_phi_cols` = all phi columns only.
`focus_formula` = an explicit operator expression linking two phi columns (`col_a * col_b`, `col_a / col_b`, either
order, up to 4 quote/bracket chars); for P4 `sin(...hours_since_start`, `hours_since_start / 24`, `% 24`,
or `2*pi ... hours_since_start`.
These keyword sets are deliberately coarse: for P2/P6 "all columns + ratio + product words" can be satisfied by a
focus that proposes several unrelated ratios and products. Treat `S_reason_phi`/`focus_phi` as "phi-compatible
mention", not "phi stated".

`T_mentions_phi_feat` = the assessment text names (as a whole identifier) a phi-recovering feature that is active
in the fold at that moment (created this cycle or earlier and not pruned). `T_pruned_phi` = a pruning call in the
cycle names such a feature.

## Additional measures
* `focus_phi_exact` / `S_phi_exact` (`focus_eval.py`): backtick spans and `name = expr` fragments in the focus (resp.
  the Scientist's pre-focus tool arguments, notes and text) are normalised (× · -> *, ÷ -> /, ^ -> **, π -> pi),
  bound sequentially (so `stress = a / b` then `stress * c` resolves) and scored with the same detector. This is a
  strict lower bound on "the focus states phi": prose ("product of axial stress and member length") and formulas
  using intermediates defined in prose are missed.
* `pre_disc` (`cycles_analysis.csv`): the cycle is at or before the first `E_phi_code` cycle of its fold (or the fold
  never codes phi). Later cycles mostly re-use or transform an existing phi feature (log, bins, interactions), so the
  funnel is reported on pre-discovery cycles as well.

## Packets (RQ2) and blinding
`make_packets.py` samples 48 cycles (P2, P4, P6b x arm x 8; 4 from never-recovered folds = official `rec_any`=0 and
no `E_phi_code`, iterations 1-8, spread over runs round-robin; 4 from recovering folds = first `E_phi_code` cycle or
the cycle just before it, chosen at random). Seed 42 (numpy default_rng). Unit order is shuffled.
Each packet shows only what each role PRODUCED (tool calls, their results, text); role inputs (system prompts, phase
instructions, the Full Extractor's re-sent focus message) are not shown. Replacements: Scientist/Extractor/Tester ->
Planner/Coder/Evaluator, PLAN/BUILD/ASSESS/PRUNE -> PLANNER/CODER/EVALUATOR, "phase" -> "step", session/run ids and
tool-call ids removed.
Arm cues that could NOT be removed:
* the single agent's Scientist uses `take_a_note`/`read_notes` far more often (P6b: 105 vs 10 note calls over 7 runs
  each) and searches more (0.7-0.9 vs 0.35-0.43 `search_tool` calls per cycle); the tools exist in both arms;
* the single agent's Tester produces about half as much text (assessment ~5.4-6.1k vs ~9.4-11k chars) and prunes
  fewer features (12-20 vs 24-40 per cycle);
* in the single agent, the coder/evaluator text occasionally refers back to the planner's reasoning in the same
  conversation (e.g. "as noted in the plan").
What each role could SEE differs by arm and is recorded in `packets_key.csv` column `context_visible` (Full: the
Coder sees only the focus; single: the Coder sees the Planner's tool calls, tool results and focus of the same cycle,
and the Evaluator sees both). Neither arm sees earlier iterations' conversations (new session per iteration).
`packets_key.csv` is the unblinding key; annotators must not read it.
