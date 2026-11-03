# X7 / A17-T2 — F0 single-shot prompt — DRAFT for Henrik (not pre-registered)

**Status: DRAFT, 2026-10-19. Not frozen. No F0 row may run until Henrik approves this text and
it is committed as a dated amendment to `docs/agent2027/preregistration.md` (Part VIII.2,
"before first F0 row: F0 prompt text", A17-T2 / X7).** The F0 pool rows (A17-101…177, minus
the dropped O20 pool A17-143…149) are not in `manifests/agent2027/A17_CAP-1.csv` and are in no
session file. `f0_generate.py` / `f0_clients.sh` (A17-T7) are not built.

Fixed by A17-T2 and pre-registration Part V.17 (not open for change here): the contents
(`context_and_goal`; column names and dtypes; 5 training-fold rows without the target; the
Extractor command convention; "return only a JSON list of ≤ 10 {name, pandas_command,
argumentation}"), temperature 0.7, `max_tokens` 32768, no tools, no search, the F0 seed rule
`llm_seed·1000 + 128·f + j`. Open for Henrik: the exact wording below, and the four questions
at the end.

## Request shape

- One chat request per draw, through `create_openai_client_for_vllm` (`rogueone/llm/clients.py`),
  so thinking-off and parameter stripping match F2. `messages` = [system, user] as below;
  no `tools`, no `tool_choice`; `temperature=0.7`; `max_tokens=32768`; `seed` = the F0 seed
  rule value (passed explicitly; it overrides `ROGUE_ONE_LLM_SEED`).
- `{...}` fields are filled per (task, outer fold): `{context_and_goal}` verbatim from the
  task's `config.yml` (the same string the Extractor receives); `{columns}` one line per raw
  column of `data.csv` except `id`, as `- <name> (<pandas dtype>)`; `{rows}` = the first 5
  rows (by `id` order) of the outer-fold **training** rows of `data.csv`, rendered with
  `DataFrame.to_csv(index=False)`, without the target (the target lives in `labels.csv` and is
  never shown).

## System message (DRAFT)

```
You are a feature-engineering assistant for tabular machine learning. You propose new
features as pandas code. You cannot run code, call tools or search the web: answer in one
message.
```

## User message (DRAFT)

```
# Task
{context_and_goal}

# Raw table `df`
Columns (name and dtype):
{columns}

Five example rows (training data; the prediction target is not shown):
{rows}

# How each feature is computed
Each feature is one Python snippet that runs in a fresh scope with these names available:
- `df`: the raw table above (several rows may share one `id`),
- `df_features`: a table with one row per `id` (column `id`) to which features are added,
- `pd` (pandas) and `np` (numpy).
Rules:
- Assign the final value to a variable named `result`.
- `result` must be numeric, without NaN or inf, and have one value per row of
  `df_features`, aligned with `df_features['id']`. Use a `groupby('id')` and map it onto
  `df_features['id']`, for example:
  result = df_features['id'].map(df.groupby('id')['age'].first())
- Define any helper (mapping, mask, function) inside the same snippet; nothing persists
  between snippets.
- Use `&`, `|` and `~` for boolean operations on pandas Series, never `and` / `or` / `not`.
- Do not reassign `df` or `df_features`, and do not read files.

# Answer format
Return ONLY a JSON list of at most 10 objects, with no text before or after it:
[{"name": "<feature name>", "pandas_command": "<snippet>", "argumentation": "<why this feature should help predict the target>"}, ...]
```

## Mapping to the scorer (A17-T8, for reference)

`pandas_command` is run by `exec_user_command` (`rogueone/utils/pandas_exec.py`) with `df` =
the raw table and `df_features` = an id-only frame; it becomes the attribute table's
`Pandas Command` column for `materialize_fold_safe_features`. A draw whose text is not a JSON
list, or whose snippets all fail, is a failed draw (score −∞), as A17 §4 states.

## Questions for Henrik

1. Wording above, or closer to the Extractor's own system prompt (which names the Scientist,
   the tools and the `append_new_attribute` schema)? The draft keeps only the command
   convention, so F0 differs from F2 by scaffold and budget, as A17 §4 labels it.
2. Field name `pandas_command` (A17-T2) vs the Extractor tool's `command`: keep
   `pandas_command` (as registered)?
3. Example rows: the first 5 training rows by `id` (deterministic, as drafted) or 5 rows drawn
   with the fold seed?
4. A JSON list wrapped in a ```json fence: accept (strip the fence before parsing) or count as
   a failed draw? Draft proposal: accept, and log `fenced=1`.
