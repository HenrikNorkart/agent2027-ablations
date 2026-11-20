# E5 · Prompt-hint neutralisation, rendered-prompt dump and diff: change record

Spec: `01_shared_infrastructure.md` E5 (§2 freeze item 6). Branch `agent2027/e5` (from `agent2027/v2` at `a84b6ab`).
Paths: `ext/agent.py` = `rogueone/llm/agents/core/extractor/agent.py`, `sci/agent.py` = `.../scientist/agent.py`,
`kn/web_search.py` = `.../knowledge/web_search.py`. Line numbers are in the **new** file (post-E5); the old ones are the
`07cbf93` numbers, shifted by +1 after the one-line import added at the top of each file.

Rule applied: replace each dataset-specific example by a neutral generic one of the same shape, so every prompt keeps its
structure, bullet list and instructions. No instruction was added or removed.

## Changed

| # | Site (new line) | Before | After | Why |
|---|---|---|---|---|
| 1 | `kn/web_search.py:83-84` (tool docstring) | `Good: "torque formula balance scale physics"` / `Bad: "torque = rF sin(θ) equilibrium"` | `Good: "variance formula statistics explained"` / `Bad: "var = sum((x - mean)^2) / n"` | balance-scale torque (spec) |
| 2 | `kn/web_search.py:222` | `(e.g. "= rF sin(θ)", "F = ma")` | `(e.g. "= sum(x) / n", "y = mx + b")` | torque formula (found by grep) |
| 3 | `kn/web_search.py:226` | `say "torque formula" not "T = rF sin theta"` | `say "variance formula" not "s = sum(x - m) / n"` | spec `:225` |
| 4 | `kn/web_search.py:229` | `"balance scale torque moment physics", ..., "weight distance interaction features"` | `"variance formula statistics explained", ..., "ratio interaction features"` | spec `:228`; "weight distance" is the balance-scale feature pair |
| 5 | `kn/web_search.py:230` | `"torque = r × F × sin(θ)"` | `"mean = sum(x) / n"` | spec `:229` |
| 6 | `sci/agent.py:442` | `(CAR run wi5r0oki failed exactly this way)` (deleted) and `"build target-encoded versions of buying/maint/safety; ..."` | `"build target-encoded versions of the categorical columns; ..."` | CAR advice (spec `:441`) |
| 7 | `sci/agent.py:444` | `(e.g. CAR with 6 raw categorical columns)` | `(e.g. a dataset with only a handful of raw categorical columns)` | CAR advice (spec `:443`). The rest of the line, including the "restoration of features the Tester has previously pruned" clause, is unchanged (A11 deletes that clause later, located by its text). |
| 8 | `ext/agent.py:455` (tool docstring) | `- def winner(b): ... ... result = df.apply(winner, ...)` | `- def helper(b): ... ... result = df.apply(helper, ...)` | spec `:454` |
| 9 | `ext/agent.py:902` (system prompt) | `Helper functions (def winner(...), minimax, solve, ...), lookup tables (lines = [...])` | `Helper functions (def helper(...), score, solve, ...), lookup tables (groups = [...])` | spec `:901` |
| 10 | `ext/agent.py:904` | `add count_x, count_o separately ... df_features['count_x']` | `add count_a, count_b separately ... df_features['count_a']` | tic-tac-toe x/o counts; same bullet as #9 |
| 11 | `ext/agent.py:434` (tool docstring) | `df['class'].map({'unacc': 0, 'acc': 1})` | `df['class'].map({'a': 0, 'b': 1})` | CAR target labels, not in the spec list |
| 12 | `ext/agent.py:442-444` | `buying_map = {'low':1,'med':2,'high':3,'vhigh':4}; ...['buying']` | `size_map = {'low':1,'med':2,'high':3}; ...['size']` | CAR `buying` column; the done-when grep contains `buying` |
| 13 | `ext/agent.py:446-449` | tic-tac-toe `lines = [['top-left-square',...]]`, `== 'x'`, `wins` | `groups = [['col_a','col_b','col_c'],['col_d','col_e','col_f']]`, `== 'yes'`, `hits` | tic-tac-toe board squares; same multi-statement shape |
| 14 | `ext/agent.py:455` | `- result = 'top-left-square'  # WRONG ...` | `- result = 'col_a'  # WRONG ...` | tic-tac-toe |
| 15 | `ext/agent.py:906` | `result = 'top-left-square'` is wrong; use `(df['top-left-square'] == 'x')...` | `result = 'col_a'` is wrong; use `(df['col_a'] == 'yes')...` | tic-tac-toe |
| 16 | `ext/agent.py:912` | `df['safety_ord']` ... `df_features['safety_ord']` (3 occurrences) | `size_ord` | CAR `safety` column; the done-when grep contains `safety` |

Items 11-16 are **not in the spec list** (the spec names only `:454` and `:901` for the Extractor). They were found by
grepping every prompt string; the done-when grep (`CAR`, `buying`, `safety`, `winner`) fails without them. Consequence for
A15/A11: the Extractor system prompt, tool docstrings and Scientist prompt change for **every** dataset.

## Left unchanged (dataset-specific-looking, not neutralised)

| Site | Text | Why left |
|---|---|---|
| `ext/agent.py:55` (docstring of `_ordinal_encode_categorical`) | "CAR (and other all-categorical datasets)" | helper docstring, never sent to a model |
| `ext/agent.py:619-622, 803-806, 858` | comments mentioning CAR, `buying/maint/doors/...`, "car run 09vnwi4s" | `#` comments, never rendered |
| `tester/agent.py:2178`, `clients.py:326` | comments mentioning CAR runs | comments |
| `clients.py:579` | "safety net" | generic English in a docstring |
| `ext/agent.py:318, 438` | `lab_value`, `age` in code-example strings | generic clinical-style column names present in the Extractor prompt for every dataset, not tied to one benchmark dataset; see decision in the report |
| `agent_roles.py:9` | "such as a disease or a market trend" | generic framing of the global goal |
| `tester/agent.py:1562` | "Squares (`feat_sq`) and interactions (`feat_a_x_feat_b`)" | generic feature-engineering vocabulary |
| `knowledge/agent.py` (inactive RAG path) | no dataset-specific strings found | nothing to change |
| Task `context_and_goal` text | dataset-specific by design | exempt per E5 done-when 1 |

Not done here (listed under E5 but a decision, not a neutralisation): **freeze item 14 / decision D7**, exposing `Status` in
`list_known_attributes_tool`. It changes tool output and therefore Full-v2, so it waits for the D7 decision.

## Rendered-prompt dump

* Flag: `ROGUE_ONE_PROMPT_DUMP=1` (off by default; `v2_defaults.env` should pin `1`). Optional `ROGUE_ONE_PROMPT_DUMP_DIR`
  overrides the target; default `<output_dir>/prompts/`.
* Code: `rogueone/utils/prompt_dump.py` (`dump_prompt`), called once per agent call just before the retry loop in
  `sci/agent.py` (phases `focus`, `fallback`), `ext/agent.py` (`append`), `tester/agent.py` (`Tester Agent (Assessment)`,
  `Tester Agent (Pruning)`) and `kn/web_search.py` (`search`, first call only; the user message is a placeholder because the
  query is data-dependent). The dump never raises into the caller.
* Per call it writes `<role>__<phase>__iter<NNN>__c<SEQ>.{system.txt,user.txt,tools.json}`. `tools.json` holds each tool's
  name, description and parameter schema, because tool docstrings are part of the rendered prompt and several E5 edits live
  there. Iteration = the `step` / `index` argument of the call.

## Diff and leak scan

    python scripts/agent2027/prompt_diff.py <dumpA> <dumpB>        # exit 0 identical, 1 different, 2 error
    python scripts/agent2027/prompt_diff.py --scan <dump> [--term T ...] [--allow-text context_and_goal.txt]

`<dump>` may be the `prompts/` directory or its parent (e.g. a fetched `run-output` artifact). Default scan terms are the E5
list {torque, CAR, buying, maint, safety, winner, minimax} (case-insensitive whole words); `--term` adds A15 names (P6b
names, experiment name, `dataset_id`); `--allow-text` exempts the task's own context text.

Tests: `tests/unit/test_prompt_dump_diff.py`.