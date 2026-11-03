# A04 session C · SR6 routing check

## A04-T19 SR6 verdict (2026-10-22)

**Verdict: PASS.** `misrouted_calls` = 0 over every S, E, T and W trace record in both mixed rows; each row has
Scientist W calls (135 and 8); `wandb.config` matches the planned routing. A19's SR6 gate (A04 §6, A19-T5) is met.
E and T made no W calls in either row: **W untested for the Extractor and Tester callers.**

Rule: `docs/ablations/A04_CAP0_stage0_smoke_tests.md` §6 SR6, check (d) in §4. Layout: SR4-fallback session C
(Q122·TP2·8040 + Q4·8032 + Q0.8·8030, image `agent2027-v2.3`). `stage0_table.py` (named in A04-T19) was never
written; this read-out replaces it for SR6 only (the other A04-T19 outputs are not produced here; O120's TP2
valid-call rate comes from A18's `replay_O120`, as the SR4 note says).

### Data and method

- Rows `A04-007` (`mixcheck_esmall`) and `A04-008` (`mixcheck_1b`), session `S1021m`, launched 2026-10-21 05:50Z;
  both **CAP_HIT** at 6 h (rc 137, 11:53Z). W&B `PlanBench/RogueOne-AGENT2027`: `jd64crkz`, `t9cuempe`
  (state `crashed` = killed at the cap).
- Source: each row's `output/llm_traces.jsonl.gz` (`ROGUE_ONE_LLM_TRACE=1`) on CAIR, read in a CPU-only
  `rogue_one:agent2027-v2.3` container with read-only mounts. The gzip streams end without an end-of-stream marker
  (container killed at the cap); every complete line before the cut was read and none was unparsable.
- The trace records carry the requested `model` and the server-reported `response_model`, not the port or base URL.
  In session C each port served a different model (8040 Q122, 8032 Q4, 8030 Q0.8), so `response_model` identifies
  the server that answered. A call is counted as misrouted if `response_model` (or, for a call that returned an
  error and has no response, the requested `model`) differs from the role's planned model, or if requested and
  served model differ. For W calls the planned model is the caller's (`ROGUE_ONE_WEBSEARCH_ROUTING=caller`); the
  row's global `LLM_MODEL` is the third model, so a W call that followed the global would have shown up as that
  model. No record names the third model.
- W calls are the web-search summariser's LLM calls: trace records with `stage=search` and a `caller`. The
  SearXNG queries in `retrieval_transcript.jsonl` (A04-007: 257 queries / 9 answers; A04-008: 4 / 4) are not LLM
  calls and do not name the calling role.

### Planned routing (A04 SR4 note)

| Row | S | E | T | Global `LLM_MODEL` / `LLM_PORT` (third model) |
|---|---|---|---|---|
| A04-007 `mixcheck_esmall` | Q122 / 8040 | Q4 / 8032 | Q122 / 8040 | Q0.8 / 8030 |
| A04-008 `mixcheck_1b` | Q0.8 / 8030 | Q122 / 8040 | Q0.8 / 8030 | Q4 / 8032 |

### Calls per role and server (trace records)

| Row | Role | Records (by stage) | Model / server | Served OK | Error, no response (judged by requested model) | Misrouted |
|---|---|---|---|---|---|---|
| A04-007 | S + W ← S | focus 54, search (W) 135 | Q122 · 8040 | 188 | 1 | 0 |
| A04-007 | E | append 3,524 | Q4 · 8032 | 3,463 | 61 | 0 |
| A04-007 | T | assessment 637, pruning 2,710 | Q122 · 8040 | 3,213 | 134 | 0 |
| A04-008 | S + W ← S | focus 7,335, search (W) 8 | Q0.8 · 8030 | 6,572 | 771 | 0 |
| A04-008 | E | append 2,368 | Q122 · 8040 | 2,341 | 27 | 0 |
| A04-008 | T | assessment 1,731, pruning 7,103 | Q0.8 · 8030 | 8,634 | 200 | 0 |

Totals: A04-007 7,060 records, A04-008 18,545 records. (The S and W calls share one role label in the traces;
the error split between them is not separated here.)
**`misrouted_calls` = 0 in both rows.** No record was served by, or requested, the row's third model.

**W calls per caller:** Scientist 135 (A04-007) and 8 (A04-008); Extractor 0 and 0; Tester 0 and 0.

### `wandb.config` check (§4 check (d))

| Key | A04-007 | A04-008 | Matches plan |
|---|---|---|---|
| `model_S` / `flags/ROGUE_ONE_SCIENTIST_MODEL` / `_PORT` | Q122 / 8040 | Q0.8 / 8030 | yes |
| `model_E` / `flags/ROGUE_ONE_EXTRACTOR_MODEL` / `_PORT` | Q4 / 8032 | Q122 / 8040 | yes |
| `model_T` / `flags/ROGUE_ONE_TESTER_MODEL` / `_PORT` | Q122 / 8040 | Q0.8 / 8030 | yes |
| `flags/ROGUE_ONE_WEBSEARCH_ROUTING`, `model_W` | `caller`, `caller` | `caller`, `caller` | yes |
| `flags/LLM_MODEL` / `flags/LLM_PORT` (global = third model) | Q0.8 / 8030 | Q4 / 8032 | yes |

### Caveats

- **Partial runs.** Both rows were killed at the 6 h cap after fold 0, iteration 8 (A04-007) and iteration 7 / 6
  (A04-008), although the manifest sets `ROGUE_ONE_SMOKE_FOLDS=1`; SR6 is judged on all calls made before the cap.
  The cap hits are recorded under SR2-type completion findings, not under SR6.
- **W coverage.** Only the Scientist issued W calls, so the E- and T-caller paths of `caller` routing are untested
  (A04 §6 allows this; reported as "W untested for this caller"). In A19 a misrouted E/T W call would therefore not
  have been seen here.
- **Port inferred from the served model**, not logged per call (unambiguous in this layout, because the three
  servers served three different models).
- Error calls: 989 records returned an error (mostly in A04-008's Q0.8 Scientist, 771); A04-007 includes 48
  context-length 400s. They are routing-correct by requested model; they bear on SR2-type quality, not SR6.
