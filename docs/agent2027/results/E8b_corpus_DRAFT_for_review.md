# DRAFT: E8b KT3 corpus and TF-IDF replay, for Henrik's review

> **Status: DRAFT, not frozen.** The corpus text below was written by Claude on Mon 19 Oct 2026 and has not
> been reviewed. Nothing here is a pre-registration amendment yet. After Henrik approves (or edits) the corpus,
> the hashes, φ_wrong values and replay counts are recomputed and copied into `preregistration.md`
> Part VIII.3 as the G0b-1 amendment (Part V.14, Part VIII.2), **before A14's first row**.
> Any edit to a corpus file changes its hash and requires a re-run of the replay
> (`scripts/agent2027/analysis/kt3_replay.py`, command in §5).

Branch `agent2027/e8b` (based on `agent2027/e9e-e7` @ `cdd866f`). Sources: A14 §2, §4, A14-T1, A14-T2, A14-T4;
01 §3.3 E8b; pre-registration Part II "X9", Part V.14, Part VIII.2.

**Leak rule (A14-T2).** P7's threshold constant and φ_wrong's constant appear in the repository **only** in the
two `marenko_index.md` files under `scripts/agent2027/corpora/`. This note therefore names them symbolically:

* `THR_P7` = `p7_constants.thr` in `~/rogueone-synp-meta/synp_p7.json` on CAIR, written in the document at full
  precision (9 significant digits, as stored).
* `THR_B` = `phi_wrong.consts.thr_b` in the same file (the median anchor spacing), written at the stored precision.

The unit test `test_p7_constants_appear_only_in_the_corpora` reads both numbers from the corpus files and fails
if either occurs in any other file of the checkout (task folders excluded).

## 1. What the corpora contain

`kt3_correct/` and `kt3_wrong/` hold the same 8 files (the P7 document + 7 distractors). Only the formula
paragraph of `marenko_index.md` differs (unit test `test_kt3_corpora_differ_only_in_the_p7_formula_paragraph`).

| File | Title (first line) | Role | Bytes | Mentions "Marenko"? |
|---|---|---|---|---|
| `marenko_index.md` | Marenko index: ranking stay-cable segments for inspection | **P7 document** | 1,991 (correct) / 1,974 (wrong), −0.9% | yes (domain term) |
| `cable_vibration_damping.md` | Stay-cable vibration and damping: what the vibration readings mean | distractor (vibration RMS, Scruton number, dampers) | 1,897 | no |
| `corrosion_pit_assessment.md` | Corrosion pits on cable strands: measurement and assessment | distractor (how pit depth is measured; humidity/temperature/dust drivers; qualitative grades only) | 1,710 | no |
| `anchorage_zone_inspection.md` | Anchorage zone inspection for stay cables | distractor (seals, guide pipes, drainage; record holds anchor spacing) | 1,551 | no |
| `cable_tension_measurement.md` | Measuring the tension in stay-cable strands | distractor (vibration/lift-off/load-cell methods; no formula) | 1,758 | no |
| `main_cable_dehumidification.md` | Dehumidification of suspension bridge cables | distractor (dry-air injection, RH < 40%) | 1,589 | no |
| `wire_breaks_and_fatigue.md` | Wire breaks and fatigue in stay cables | distractor (fatigue, acoustic monitoring, MFL) | 1,502 | no |
| `inspection_records_and_scheduling.md` | Cable inspection records and inspection scheduling | distractor (what a record holds; scheduling; mentions all the decoy/noise quantities neutrally) | 1,722 | no |

Design choices (all open to review, §6):

* **Fictional setting** that matches P7's `context_and_goal`: an "inspection office" with practice notes
  (IO-4.2 for the index, IO-B.x background notes, IO-P.x procedure notes). No real standard is cited.
* **No distractor defines a numeric rule** on tension, anchor spacing or corrosion depth, and none combines two of
  the three relevant quantities, so a distractor cannot by itself lead to φ, a near-φ (the screen's
  best guess `ratio × 1[c ≥ q40]`, ρ 0.927) or φ_wrong. Corrosion grades are qualitative only.
* **No distractor names "Marenko".** The P7 document is the only document that defines or mentions the term.
* **No worked example** in the P7 document: an example would have to be swapped too, breaking the
  "only the formula paragraph differs" rule.
* The P7 document's last paragraph says that ambient conditions, sensor details, operator shifts and site region
  "do not enter the Marenko index". It is identical in both corpora. It tells the agent which columns are
  irrelevant to the index, which is a realistic property of such a document but goes slightly beyond "the
  formula" (review item R3).

### 1.1 P7 document, formula paragraph (constants redacted here)

Both versions sit at bytes 247–782 (correct) and 247–765 (wrong) of the file, i.e. inside the first
1,000 characters and inside the 1,500-character hit text (unit-tested).

* **kt3_correct (φ):** "The Marenko index of a cable segment is the measured tension in the cable strand, in kN,
  divided by the spacing between the segment's cable anchor points, in metres, counted only when the deepest
  measured corrosion pit on the strand is at least `THR_P7` mm deep. When the deepest corrosion pit is shallower
  than `THR_P7` mm, the Marenko index of the segment is 0. In symbols: Marenko index = (strand tension / anchor
  spacing) x [corrosion pit depth >= `THR_P7` mm], where the bracket is 1 when the condition holds and 0
  otherwise."
  = `(strand_tension_kn / anchor_spacing_m) * (corrosion_depth_mm >= THR_P7)`, the metadata φ exactly.
* **kt3_wrong (φ_wrong, "rotated roles", the first E8a candidate with |ρ| < 0.5):** "… the measured tension in the
  cable strand, in kN, divided by the depth of the deepest measured corrosion pit on the strand, in mm, counted only
  when the spacing between the segment's cable anchor points is at least `THR_B` m. When the anchor spacing is
  shorter than `THR_B` m, the Marenko index of the segment is 0. In symbols: Marenko index = (strand tension /
  corrosion pit depth) x [anchor spacing >= `THR_B` m] …"
  = `(strand_tension_kn / corrosion_depth_mm) * (anchor_spacing_m >= THR_B)`, the metadata φ_wrong exactly.

The rest of the P7 document (title, scope line, "How the office uses the Marenko index", "Data sources") is
byte-identical in both corpora. It says segments with the highest index are flagged for close-up inspection and
that "the threshold and the units are fixed by this practice note and must not be rounded".

## 2. φ_wrong values (A14 §2 criterion; to be copied into the amendment)

Recomputed on Mon 19 Oct on the developer machine from the CAIR task files (`data.csv` sha256 matches
`synp_p7.json`'s `data_sha256`; φ sha256 matches `phi_sha256`):

| Check | Value | Criterion | Pass |
|---|---|---|---|
| \|ρ(φ_wrong, φ)\|, Spearman, all 2,000 rows | 0.193 (ρ = −0.193) | < 0.5 | yes |
| \|ρ(φ_wrong, φ)\| on fold training rows, StratifiedKFold(5, shuffle) × seeds {42, 4, 5, 1, 2, 3} (30 training sets) | max 0.212, min 0.174 | < 0.5 | yes |
| Headroom XGB(raw + φ_wrong) − XGB(raw), balanced accuracy, E8a procedure, seeds {42, 4, 5} | mean +0.00004 (per seed +0.0026, −0.0110, +0.0086) | < 0.01 | yes |
| (E8a report, CAIR container, 13 Oct) | −0.0018 | < 0.01 | yes |
| Reference: headroom of φ itself, same run | mean 0.080, min 0.072 (E8a report: 0.1045 / 0.0953) | ≥ 0.06 | yes |

The developer-machine headroom differs from the E8a container values (φ 0.080 vs 0.1045) by the XGBoost build /
platform; both pass. **For the amendment, quote the E8a container values** (or re-run `calibrate_synp.py` in a
`rogue_one` container) and give the ρ values above.

## 3. Corpus hashes (A14-T1 definition)

| Corpus | `kt3_corpus_sha256` |
|---|---|
| `kt3_correct` | `c91e3df836aaa25aa55e71fab72ba693106d83dd1ae6d0c17794692c58d63059` |
| `kt3_wrong` | `ad56cf91a9ea74dc8b03dd32c58d8ac22283a75cdaa8a124c4b61dfbea49aa4d` |

Per-file sha256 (identical in both folders except `marenko_index.md`):

```
2b84cc50e8e582fdfad4b5d1500eb962820797a93858236db6bfeb1f1fbb4ca2  ./anchorage_zone_inspection.md
3aa3c2f8358dce04a2270fb8f06a0973484eea5618c7d852127252fc75dbabc5  ./cable_tension_measurement.md
1de06c97caa5e359fe074ca2febc9026bdd3fde79e6f5b35599e607eaa33f50c  ./cable_vibration_damping.md
3b9e3cff3a2f3f193889164f900710b995915d6447900aa4fd0089cc443eb2b9  ./corrosion_pit_assessment.md
ece031eac6ee0444474ed74cdb5b5511a5c9dfe79534fefa2703c4848b55e4b4  ./inspection_records_and_scheduling.md
ba14852c191795b7b09987d42f50d9f24e2990839ae20293cbb7c447378f8ae2  ./main_cable_dehumidification.md
8fae3191c97ae44744f67b36fef8bdcb35b6a1ef72819ebdf3e4bd22581d833f  ./marenko_index.md   (kt3_correct)
493f8838d4484b7f4f034a9b1a954dc3f4493849bc2a956ba3366491f700a325  ./marenko_index.md   (kt3_wrong)
faf72de515709b01d3522fcb036fe50b6e5740528f71a1bba47cb08afbd62b8d  ./wire_breaks_and_fatigue.md
```

**Hash caveat (found while testing).** The A14-T1 one-liner
`find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum` is only well-defined with GNU
coreutils' default **text-mode** listing (`<hex>  ./<file>`, two spaces), as on Linux and in every `rogue_one`
container. Git for Windows' `sha256sum` defaults to binary mode (`<hex> *./<file>`) and gives a different hash
(`14950bb3…` for `kt3_correct`). On Windows use `xargs -0 sha256sum --text`. `local_corpus.corpus_sha256` implements
the Linux form, and a unit test checks it against the shell pipeline with `--text`. The amendment should state
"GNU coreutils text mode" next to the definition. The corpora are pinned to LF line endings in
`.gitattributes` so every checkout hashes the same bytes.

## 4. Backend as implemented (A14 §4 / X10 V.14 settings)

`rogueone/llm/agents/core/knowledge/local_corpus.py`: every regular file is one UTF-8 document; `doc_id` = relative
path; title = first non-empty line without a leading `#`; body = the rest. `TfidfVectorizer(lowercase=True,
stop_words="english", ngram_range=(1,2), sublinear_tf=True)` fitted on title + body; score = cosine; top 3 with
score > 0, ties by `doc_id`; hit text `"<title>: <first 1,500 chars of the body>"`. E9e's `_local_search_impl`
runs the unchanged sanitiser and query-variant loop over it (next variant when no positive score, then "No results
found."). `kt3_corpus_sha256` goes to `wandb.config` only under `ROGUE_ONE_KNOWLEDGE_BACKEND=local` (Full-v2's
config is unchanged).

## 5. TF-IDF replay of A02's P7 queries (A14-T4 / X9 retrieval check)

**Input (frozen query set).** The `event="query"` rows of `retrieval_transcript.jsonl` in A02's three `synp_p7`
runs (`S1013d`: A02-008 R10, A02-016 R3, A02-024 R1; image `agent2027-pre.3`, SearXNG), copied read-only from
`~/rogueone-runs-agent2027/S1013d/<row>/tasks/classification/synp_p7/output/` on CAIR (grep only, no host Python).
One replayed query per logged tool call (`variant_index` 0), i.e. 400 + 546 + 375 = **1,321 calls**, 1,082 distinct
strings. The list is committed as `docs/agent2027/results/E8b_a02_p7_queries.jsonl` (row id, seq, query). The A02
readout's "search queries" column (401 / 547 / 377) counts all query rows incl. variant retries.

```
python scripts/agent2027/analysis/kt3_replay.py \
    --queries docs/agent2027/results/E8b_a02_p7_queries.jsonl --json <out.json>
```

**Results** (P7 document = `marenko_index.md`; rank among the ≤ 3 hits served):

| Count | kt3_correct | kt3_wrong |
|---|---|---|
| Calls containing the domain term "Marenko" | 332 (A02-008: 0, A02-016: 190, A02-024: 142) | 332 |
| … P7 document rank 1 | **332 / 332 (100%)** | 332 / 332 |
| … P7 document in top 3 | **332 / 332 (100%)** — criterion ≥ 50%: **pass** | 332 / 332 |
| All calls: P7 document rank 1 / 2 / 3 / absent | 726 / 171 / 222 / 202 | 698 / 164 / 232 / 227 |
| All calls: P7 document in top 3 | 1,119 / 1,321 (84.7%) | 1,094 / 1,321 (82.8%) |
| Calls without the term but P7 document in top 3 | 787 / 989 | 762 / 989 |
| Calls with no hit in any variant ("No results found.") | 19 | 19 |
| Calls where a shorter fallback variant was needed | 0 | 0 |
| **Same P7 rank in both corpora** | **1,233 / 1,321 (93.3%)** — criterion ≥ 90%: **pass** | |
| Same served doc list in both corpora | 1,221 / 1,321 | |

The 88 rank differences are generic corrosion queries ("bridge cable corrosion temperature humidity tension
vibration", "vibration effect on bridge cable corrosion"): the wrong formula says "corrosion pit depth" where the
correct one says "anchor spacing", so it ranks a little differently on corrosion vocabulary. **No vocabulary change
was made** to reach these numbers (this is the first and only version of the text).

**Observation for review (R1).** The P7 document is not only found for its term: it is the top hit for 55% of all
calls and in the top 3 for 85%, because it shares the generic vocabulary (cable, strand, tension, corrosion pit,
inspection, anchor spacing) of almost every P7 query. In a `kb_local_*` run the summariser will therefore see the
P7 document for most searches, including searches that never name the index. That makes the channel test easy to
pass on retrieval; it does not by itself give the formula to the Scientist (the summariser must relay it).

## 6. What Henrik must review before the freeze

* **R1 — Retrieval strength.** Accept that the P7 document dominates retrieval (§5), or make the distractors
  compete more (e.g. more distractor text on corrosion/tension, or one distractor that mentions "Marenko" in
  passing without defining it). Any change re-runs §5 and is disclosed as a vocabulary change.
* **R2 — Formula wording.** Both formula paragraphs: words + an ASCII "In symbols" line, no column names. Should
  the document use the column names (`strand_tension_kn` …)? Proposal: no (a real practice note would not); the
  quantities are described with the column-description wording.
* **R3 — "do not enter the index" sentence** (last paragraph of the P7 document, both arms). Keep, or cut to the
  three "Data sources" bullets only.
* **R4 — Distractor set.** 7 distractors (rule: 5–10). Check that each is technically plausible and that none
  hints at φ or φ_wrong. `inspection_records_and_scheduling.md` names every decoy/noise quantity of the table;
  keep or trim.
* **R5 — "highest index is flagged"** in the P7 document (both arms): it states the direction of the label rule.
  Keep (proposal) or drop.
* **R6 — Hash definition.** Add "GNU coreutils text mode" to the pre-registered definition (§3).
* **R7 — φ_wrong headroom number** to quote: E8a container value (−0.0018) or a fresh container re-run.
* **R8 — Docker context.** `.dockerignore` now excludes `scripts/agent2027/corpora/` **and** `docs/agent2027/`
  (A14-T2). The latter was never excluded before, so `agent2027-pre.N`, `v2.0`, `v2.1` and `v2.2` images carry
  the pre-registration text; the P7 constants were never in the repo before this branch, so the A14-T2 grep for
  them on those images should still print nothing (to be run on CAIR inside containers at the v2.3 build).

After approval: recompute §3 and §5 (`kt3_replay.py`), paste §2, §3, §5 and any vocabulary change into
`preregistration.md` Part VIII.3 as the dated G0b-1 amendment, then build `rogue_one:agent2027-v2.3`.
