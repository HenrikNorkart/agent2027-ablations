# AGENT2027 pre-registration

Pre-registration and draft protocol for the Rogue One v2 design study (ICSE AGENT 2027). It is committed on branch
`agent2027/v2` and tagged `agent2027-prereg` at X10 (Fri 16 Oct 2026), before the first Full-v2 run (A05, launch
Fri 16 Oct, 18:00). Nothing in this file is a post-freeze result. The pre-freeze pilot results it quotes (A01, A02,
A04, the A08 CPU screen, X9) were known before it was written; this is disclosed in Part VII.

**Status of this text:** final (X10). Drafted 2026-10-15; every item that was marked `TODO(Henrik)` was resolved
by Henrik on 2026-10-16 (Part VIII.1).

**How the parts relate.**
- Part I fixes Full-v2 (every frozen flag value).
- Part II keeps every dated pre-freeze section **verbatim** (A02 rules, amendment and erratum; A03 fallback; E13;
  knowledge channel; E14; A04 verdicts; X9). Later parts refer to them; they are not edited.
- Part III records further pre-freeze inputs (A01, the M5 screen, the A02 re-tier, the E6 fix).
- Parts IV–VI state the shared statistics, the per-ablation rules due at G1 and the gates (including G2).
- Part VII lists deviations and disclosures; Part VIII lists open items with dates and holds later amendments.
- **Incorporation by reference.** The decision-rule sections (§6, and the tasks named below) of each ablation
  document `docs/ablations/A01…A23_*.md` and `docs/ablations/01_shared_infrastructure.md` **at the tagged commit**
  are part of this pre-registration. This file restates the rules that are due at G1. Where an ablation document
  and this file disagree, **this file wins**; where an ablation document still assumes a setting that Part I
  overrides (for example a shared search cache, R=3, a replace pool, TP4, O20 as the CAP-2/CAP-3 rung, or CAAFE
  runs), Part I and the decisions in Parts II–III apply (list in Part VII.3).
- Amendments after the tag are added as new dated sections at the end of this file (Part VIII), never by editing
  the text above them.

## Contents

- Part I — Frozen configuration (Full-v2)
- Part II — Pre-freeze read-outs and decisions (dated sections, verbatim)
- Part III — Additional pre-freeze inputs recorded at X10 (A01, M5 screen, A02 re-tier, E6 fix)
- Part IV — Shared statistical rules (Holm family, margins, ITT, conventions)
- Part V — Per-ablation rules due at G1
- Part VI — Gates (G1 to DF), including the G2 rules
- Part VII — Deviations and disclosures
- Part VIII — Open items with dates (`TODO(Henrik)` (resolved 2026-10-16, Part VIII.1)) and amendments

---

## Part I — Frozen configuration (Full-v2)

### I.1 Code, image and serving

- **Code:** branch `agent2027/v2`. The Full-v2 freeze candidate is tagged `agent2027-v2.0` (commit `55862e5`,
  tag message: "Full-v2 freeze candidate (E1-E6, E13, E14; augment, R=10, sandbox on, L1, live SearXNG). Smoke
  S1015l passed on pre.5."). The freeze tag `agent2027-v2.0-freeze` and this file's tag `agent2027-prereg` are
  pushed at X10. `TODO(Henrik)` (resolved 2026-10-16, Part VIII.1): state the commit the two tags point to, and confirm that `agent2027-v2.0` and
  `agent2027-v2.0-freeze` name the same code tree (01 §2.2 expects the freeze tag; the image tag already exists).
- **Client image:** `rogue_one:agent2027-v2.0`, built on CAIR from that commit; its image ID is logged in
  `wandb.config` of every run. `TODO(Henrik)` (resolved 2026-10-16, Part VIII.1): record the image ID here once built (A05-T7). Base of the client
  Python environment `vllm/vllm-openai:v0.11.0` (`Dockerfile:3`), as in t17.
- **Pre-freeze images:** `agent2027-pre.1` … `agent2027-pre.5`. `pre.5` carries the E6 fix of 2026-10-15
  (Part III.4); smoke `S1015l` passed on it.
- **Server:** `vllm/vllm-openai:v0.22.0` (version asserted in-container, image ID logged), `MAX_MODEL_LEN=131072`.
- **Model rungs** (`scripts/agent2027/rungs.csv`, HF revisions pinned):

  | Rung | HF id | Revision | TP | Port | A04 verdict |
  |---|---|---|---|---|---|
  | O120 | `openai/gpt-oss-120b` | `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` | 2 | 8010 (8011 second server) | reference |
  | Q122 | `Qwen/Qwen3.5-122B-A10B-FP8` | `a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9` | 2 (`--max-num-seqs 256`) | 8040 | pass (SR3 at TP2) |
  | Q9 | `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | 1 | 8033 | pass (SR1 not triggered) |
  | Q4 | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 1 | 8032 | pass-data (SR2) |
  | Q2 | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | 1 | 8031 | pass-data (SR2) |
  | Q0.8 | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | 1 | 8030 | pass-data (SR2) |
  | O20 | `openai/gpt-oss-20b` | `6cee5e81ee83917806bbde320786a8fb61efebee` | 1 | 8020 | **fail (SR4)**, fallback applied |

  Qwen rungs serve with `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
  --language-model-only --revision <rev>`. The Q4 id and revision above are the ones used for every Q4 row,
  including the A06 Q4 anchor and sentinels (A06-T4).
- **O120 topology (D5): TP2** for all post-freeze O120 jobs. D5's pre-set rule was applied mechanically on
  2026-10-12 (HQ #258): two TP2 servers gave 2.73 runs/h against 1.88 runs/h for TP4 (1.45×), with 0 cap hits in
  either. This matches CAP-2's topology, so the A19 topology disclosure is not needed.
- **Launcher-level settings:** `PARALLEL=6` per O120 server, `CPUS=12` per client, `RUN_CAP_HOURS=6` (a 6 h cap hit is
  an ITT failure and is never retried), `RUNROOT=~/rogueone-runs-agent2027`, explicit `GPUS`, `WAIT_MAX_MIN=0`.
  Evaluator XGBoost 400 trees (assessment/SHAP model 300), in code.
- **GPU use (D12 and the weekend allowance).** D12 (Henrik, 2026-10-09) lifted the one-Rogue-One-job-at-a-time rule:
  several experiments may run as clients of the same vLLM server. The limit stays ≤ 4 H100 unless Henrik approves
  more in chat. For the A05 weekend (Fri 16 – Sun 18 Oct) Henrik allowed all 8 GPUs, so up to four O120 TP2 servers
  may be used, as sessions `S1016a`–`S1016d` (Part V.5).

### I.2 Frozen flag values (01 §7 R3 and R4)

Source of truth: `scripts/agent2027/v2_defaults.env` at the tagged commit. The launcher reads it from inside the
image, applies the row's `env_overrides`, then the per-row keys and the run-identity keys. The values below are copied
from that file.

**R3 — existing flags**

| Flag | Full-v2 value | Note |
|---|---|---|
| `ROGUE_ONE_NUM_ITERATIONS` | `10` | code default 3, so always set |
| `ROGUE_ONE_K_FOLDS` | `5` | |
| `ROGUE_ONE_SELECTION_SIGNAL` | `inner_cv` | |
| `ROGUE_ONE_ENABLE_ROLLBACK` | `False` | |
| `ROGUE_ONE_STRATIFIED_FOLDS` | `False`; **`True` on SYN-P classification rows** (`synp_p3`, `p4`, `p5`, `p7`) via `env_overrides` | D4 adopted (Henrik, 2026-10-09) |
| `ROGUE_ONE_DO_BALANCING` | `True` (explicit) | balanced accuracy; not on t17's scale |
| `ROGUE_ONE_FOLD_RETRY_LIMIT` | `1` | |
| `ROGUE_ONE_DO_LOGGING` | `True` | |
| `ROGUE_ONE_START_VLLM` | `False` | |
| `ROGUE_ONE_DISABLE_WANDB` | `0` | |
| `ROGUE_ONE_VERBOSE_AGENT_ERRORS` | `0` | |
| `CONSOLE_DISABLE` | `False` | A01 alone kept t17's `True` |
| `LLM_TEMPERATURE` | `0.7` | |
| `SCIENTIST_AGENT_TEMPERATURE`, `EXTRACTOR_AGENT_TEMPERATURE`, `TESTER_AGENT_TEMPERATURE`, `MEDICAL_KNOWLEDGE_AGENT_TEMPERATURE` | `0.7` | summariser included |
| `EXTRACTOR_AGENT_MAX_TOKENS` | `32768` | |
| `TESTER_AGENT_MAX_TOKENS` | `8192` | |
| `SCIENTIST_AGENT_MAX_TOKENS` | `16384` | marked "PIN@G1 (proposed 16384)" in the env file; `TODO(Henrik)` (resolved 2026-10-16, Part VIII.1) confirm |
| `EXTRACTOR_AGENT_MAX_ITERATIONS` / `TESTER_…` / `SCIENTIST_…` | `100` / `100` / `200` | |
| `*_AGENT_REASONING_EFFORT` | unset | popped by the client, as in t17 |
| `ROGUE_ONE_RETRIEVAL_TRANSCRIPT` | unset | default `<output>/retrieval_transcript.jsonl` |
| `WANDB_ENTITY` | `PlanBench` | |

Per-row keys set by the launcher (never in the env file): `ROGUE_ONE_FOLD_SEED`, `ROGUE_ONE_LLM_SEED`,
`ROGUE_ONE_DATASETS`, `LLM_MODEL`, `LLM_PORT`, `LLM_API_KEY`, `ROGUE_ONE_<ROLE>_MODEL` / `_PORT` (empty = uniform;
set only in A19 and A04 session C), `WANDB_PROJECT_NAME`, `WANDB_API_KEY`, `ROGUE_ONE_TASK_REGISTRY`.

**R4 — new flags**

| Flag | Full-v2 value | Decided by |
|---|---|---|
| `ROGUE_ONE_TOOL_ERROR_FEEDBACK` | `message` | freeze item 1 (A07 sets `none`) |
| `ROGUE_ONE_EXTRACTOR_ROUNDS` | `10` | A02: R=10 (R=3 not adopted; Part II) |
| `ROGUE_ONE_TOOL_OUTPUT_MAX_CHARS` | `8000` | marked "PIN@G1 (proposed 8000)" in the env file; `TODO(Henrik)` (resolved 2026-10-16, Part VIII.1) confirm |
| `ROGUE_ONE_REASONING_EFFORT_PASSTHROUGH` | `0` | |
| `ROGUE_ONE_LLM_TRACE` | `1` | |
| `ROGUE_ONE_PROMPT_DUMP` | `1` | |
| `ROGUE_ONE_FEATURE_POOL_MODE` | `augment` | A03: E6 pre-declared fallback, augment frozen by design (Part II) |
| `ROGUE_ONE_REPLAY_VALIDITY_CHECK` | `1` | mandatory: there is no fit-on-train replay fix; Tier B datasets excluded |
| `ROGUE_ONE_SEARCH_CACHE` | `off` | Henrik, 2026-10-14: live search, no cache (Part II) |
| `ROGUE_ONE_SEARCH_CACHE_PATH` | `/workspace/cache/serper_cache.sqlite` | inert while the cache is off |
| `ROGUE_ONE_WEB_SEARCH_PROVIDER` | `searxng` | Amendment 1 to A02 (2026-10-13); HQ #266 |
| `ROGUE_ONE_SEARXNG_URL` | `http://127.0.0.1:8888` | container `ro27-searxng`, image `searxng:ro27-2026.10.11`, settings `scripts/agent2027/searxng/settings.yml` |
| `ROGUE_ONE_WEBSEARCH_ROUTING` | `caller` | E7 lands post-freeze (v2.3); at v2.0 the flag is not read by the code, and with uniform O120 routing is a no-op |
| `ROGUE_ONE_KNOWLEDGE_BACKEND` | `web` | switch arrives in v2.3 (E9e); not read at v2.0 |
| `ROGUE_ONE_TOPOLOGY` | `three_agent` | switch arrives in v2.4 (E9f); not read at v2.0 |
| `ROGUE_ONE_PLANNER_MODE` | `llm` | switch arrives in v2.1/v2.3 (E9c, E9d) |
| `ROGUE_ONE_MEMORY_MODE` | `full` | E9d (v2.3); not read at v2.0 |
| `ROGUE_ONE_ASSESSMENT_MODE` | `llm` | E9d (v2.3); not read at v2.0 |
| `ROGUE_ONE_PRUNING_MODE` | `llm` | E9d (v2.3); not read at v2.0 |
| `ROGUE_ONE_PRUNING_SHADOW` | `0` | E9d (v2.3); not read at v2.0 |
| `ROGUE_ONE_RESTART_MODE` | `off` | E9a (v2.2); not read at v2.0 |
| `ROGUE_ONE_FEEDBACK_LEVEL` | `full` | E9b (v2.2); not read at v2.0 |
| `ROGUE_ONE_SANDBOX` | `on` | E13 landed (smoke `S1015l` passed with sandbox on); import allow-list per Part II "E13" |
| `ROGUE_ONE_FALLBACKS` | `on` | reserved; built only if RV1 fires |
| `ROGUE_ONE_SCIENTIST_SIGNAL` | `inner_cv_l1` | D2 = L1 (Henrik, 2026-10-09); E14, Part II |

Not set in Full-v2: `ROGUE_ONE_SMOKE_FOLDS` (smoke rows only), `ROGUE_ONE_LOCAL_CORPUS`, `ROGUE_ONE_PLANNER_FOCUS_FILE`.
The switch flags marked "not read at v2.0" sit at their Full-v2 value now so that the env file does not change when
the switches land; each post-freeze release must pass the three merge guards of 01 §2.2 (byte-identical rendered
prompts against `prompt-baseline-v2.0`, switch-off unit tests, the next A06 sentinel within band).

### I.3 Other freeze items and G1 decisions

| Item | Decision | Source |
|---|---|---|
| Freeze items 1–4 (error-feedback flag; per-fold round counter and `Pruned Iteration`; score after pruning; bounded tool output) | in v2.0 (E2) | 01 §2.1 |
| Freeze item 6 (prompt-hint neutralisation; rendered-prompt dump and diff) | in v2.0 (E5) | 01 §2.1 |
| Freeze item 8 (instrumentation) | in v2.0 (E3) | 01 §2.1 |
| Freeze item 14 / D7 (pruned status) | decided: expose `Status` in `list_known_attributes_tool` (Henrik, 2026-10-09). **Not implemented at `agent2027-v2.0` (`55862e5`):** the tool still returns a comma-separated list of names only (`extractor/agent.py:587-614`), while the Extractor prompt (`:913`) tells the agent to call it and look for `Status="Pruned"`. `TODO(Henrik)` (resolved 2026-10-16, Part VIII.1): implement before the freeze tag (changes Full-v2; re-take the prompt baseline and smoke), or record D7 as not applied and disclose the prompt/tool mismatch | HQ #250 |
| D3 (C3 population) | Option C: confirmatory population = multi-step subset {P3, P5, P2} minus non-discriminating families; P2 pre-registered as a two-step composition (ratio × column). Pooled SYN-P 7 is a secondary analysis. If the gate removes every subset task, pooled applies | HQ #246 |
| D4 (C1b set-up) | yes to all three: LLM-FE notebook selection primary, best3 and any-program as sensitivity (three-mode agreement rule); stratified folds on SYN-P classification; we author the LLM-FE spec files (X4) and disclose it | HQ #247 |
| D6 (M1 add-on, E6 fallback) | E6 fallback accepted; revert-or-limitation choice at G2 | HQ #249 |
| D8 (P6 twin) | Option B: P6a φ = Reynolds number v·D/ν, with hidden positive rescaling of every numeric P6b column; fallback order (a) Reynolds → (b) ideal-gas n·T/V → (c) Cockcroft–Gault, decided by `screen_pairwise.py`. The code default is `reynolds` (`scripts/agent2027/synp/tasks.py`). `TODO(Henrik)` (resolved 2026-10-16, Part VIII.1): confirm Reynolds passed the pairwise screen and record why (b) and (c) were not needed | HQ #251, A15-T1 |
| D10 (re-tier trigger) | Option B: the trigger tests the **adopted** arm's mean u per SYN-P run. With R=10 adopted, the adopted arm is `R10`, so this coincides with the canonical wording quoted in Part II | HQ #253 |
| D11 (A01 variance rule) | Satterthwaite effective-df rule (Part III.1) | HQ #254 |
| D12 (box order, parallel jobs) | one-job-at-a-time lifted; priority rule for Thu 22 – Wed 28 Oct; slots pinned at G2 | HQ #255 |
| covtype | excluded from every AGENT2027 study (Henrik, 2026-10-12); widened 2026-10-16 to **every Rogue One study**: covtype is never run (Henrik) | HQ #259 |

---

## Part II — Pre-freeze read-outs and decisions (dated sections, verbatim)

The sections below were committed before this X10 text, each before the runs it governs or on the date it states.
They are reproduced unchanged.

## A02 M4 — Extractor rounds (rules committed before launch; A02-T3)

Source: `docs/ablations/A02_M4_extractor_rounds.md` §6. Session `S1013a` (or a late-launch session),
rows `A02-001…024`, W&B `PlanBench/RogueOne-AGENT2027-prefreeze`, group `A02`.

**R rule.** Outcome `rec_sel`; reference `R10`. The rule uses all 8 SYN-P 8 tasks (40 episodes), including
any task `M5S` labels non-discriminating. Adopt **R=3** if and only if all four conditions hold:

1. Σ`rec_sel`(R10) − Σ`rec_sel`(R3) ≤ 4 (of 40), the net paired loss (difference in totals).
2. No task falls from ≥ 3/5 under `R10` to 0/5 under `R3`.
3. The `success` count of `R3` is ≥ that of `R10` (out of 8).
4. Mean `u_run`(R3) ≤ 0.75 × mean `u_run`(R10), means over the 8 rows of each arm (ITT; a cap hit = 6 h).

Otherwise **R=10**. **R=1 is never adopted**, whatever its numbers.

**Error rates.** Criterion 1 wrongly rejects a truly equivalent R=3 about 10–17% of the time (the upper
figure assumes within-run correlation). It adopts an R=3 that truly loses 4 episodes about half the time.
No equivalence claim is made.

**Gates.**
- `M5S`: if ≥ 4 of 7 families are brute-force-solvable in A08's CPU screen, recalibrate SYN-P and
  re-screen; A02 slips by ≤ 1 day and the late-launch rule applies. Launch also requires `synp_p0` to have
  0 recoveries in the screen.
- Late launch: if A02 cannot start before Wed 14 Oct, 08:00, it runs after A03's Core-10 session as
  `S1014a` (or `S1015h` on Thu 15); `R1` is then dropped by a dated amendment; the rule is unchanged.
- Detector: if the detector unit tests have not passed by Thu 15, 12:00, R=10 ("R=3 not adopted").
- No read-out by Fri 16, 09:00: R=10 ("R=3 not adopted").
- G1 (Fri 16): R goes into `v2_defaults.env`. **Re-tier rule (canonical wording; open question 5 not
  switched):** apply CL1–CL4 at once if R=3 is not adopted, **or** if the mean u per SYN-P run of `R10`
  is > 0.85u. Add-backs AB1–AB5 are judged at G1 from the same measured u.
- Leak canary: any `synp_p7` recovery is investigated with X9 before the pre-registration; it does not
  change the R rule.
- After the freeze, R is never re-opened, even if A03 adopts augment.

**Analysis conventions.** ITT: a failed or capped row counts as 0 recoveries and (if capped) 6 h; a
fallback never counts as a recovery. One fold seed (42), one replicate, replace mode. Stratified folds on
the classification tasks (`synp_p3`, `p4`, `p5`, `p7`; the D4 default), KFold on regression.

**Disclosures fixed now.** One seed (with the error rates above); replace mode; pre-freeze code was seen
before this pre-registration.

**Amendment 1 (2026-10-13, before any A02 result was read).** Knowledge channel. The first launch
(`S1013a`/`S1013b`, image `agent2027-pre.2`, 07:58Z) ran with web search dead: the Serper account was out of
credits (Finding 45; HQ #266). Henrik switched the web-search provider to a self-hosted SearXNG metasearch
instance; both sessions were stopped at 09:40Z with **no row finished** and nothing read, and A02 was relaunched
as `S1013c`/`S1013d` on image `agent2027-pre.3` (`ROGUE_ONE_WEB_SEARCH_PROVIDER=searxng`). The aborted runs carry
the W&B tags `S1013a`/`S1013b` and are excluded from every analysis. The R rule, error rates and gates above are
unchanged. Disclosed: search results come from upstream engines scraped by SearXNG (at launch: Google CSE, Yahoo,
Bing), so they vary over time; tool name, docstring and summariser prompt are byte-identical to the Serper
version.

**Erratum to Amendment 1 (2026-10-14).** Amendment 1 states that no row of `S1013a`/`S1013b` finished. That is
wrong: `S1013a`'s job log records `A02-021` (R1, `synp_p5`, image `agent2027-pre.2`, no web search) as
`status=OK` at 2026-10-13 09:30:09Z, ten minutes before the 09:40Z stop (W&B run `c2gjumk3`, 5,337 s). Its outcome
was not read before the relaunch decision, which was taken because search was dead, not on any result. The row is
excluded as Amendment 1 says; `A02-021` was rerun in `S1013c` and only that run enters the read-out. All 11 other
`S1013a`/`S1013b` rows were killed (rc 137) or never finished. Found by the A02-T12 read-out's W&B cross-check.

## A03 M1 — Augment vs replace: pre-declared fallback in force (2026-10-14)

E6 missed gate G-E6 (merged Wed 14 Oct, after the Tue 13 Oct end-of-day deadline). Henrik (2026-10-14) applies the
pre-declared contingency of `docs/ablations/01_shared_infrastructure.md` §3.3 E6 unchanged: **pool mode is frozen to
`augment` by design**, and A03 runs instead as frozen `replace` × Core-10 × fold seed 42 (rows A03-031…040, 10u) inside
A05's sessions on `agent2027-v2.0`, compared against A05 `full` A05-001…010 (A03-T12). No A03 Core-10 session runs
before the freeze.

## E13 — Sandbox import policy (2026-10-14)

Henrik keeps the import allow-list: under `ROGUE_ONE_SANDBOX=on`, generated code may import the numerical stack and
pure-computation stdlib modules; `os`, `sys`, `io`, `pathlib`, `builtins` and direct `__import__` of other modules fail.
This departs from 01 §3.3 E13's "no `__import__`", which would break the Tester's routine `from sklearn … import …`.

## Knowledge channel and A04 call validity (Henrik, 2026-10-14)

- **Web search:** provider SearXNG (self-hosted on CAIR, image `searxng:ro27-2026.10.11`, settings as committed in
  `scripts/agent2027/searxng/settings.yml`), queried live. **No search cache** (`ROGUE_ONE_SEARCH_CACHE=off`): every
  query goes upstream. Every retrieval-audit row logs the returned URLs, answering engines and calling role.
  Disclosed: results depend on which upstream engines answer at query time.
- **A04 valid call:** the **strict** reading. A call is valid only if every required key of the tool's argument
  schema is present at every nesting level (for `append_new_attribute`: `args.name`, `args.command`,
  `args.argumentation`), because the tool's pydantic model rejects a call that lacks any of them. The top-level-only
  reading is reported descriptively (`*_valid_top`). SR1 and SR4 apply to the strict rescued rate.

## E14 — Scientist feedback signal L1 (D2, Henrik 2026-10-09; built 2026-10-14)

Full-v2 sets `ROGUE_ONE_SCIENTIST_SIGNAL=inner_cv_l1` (code default `train_fit`, byte-identical to before).
- **Signal.** The Scientist's per-trial `train_*` columns are replaced by one inner-CV scalar on the outer-train rows R:
  `cv_accuracy` (classification; balanced accuracy under `DO_BALANCING`, like the pool's `accuracy`) or `cv_RMSE`
  (regression, lower is better; replaces train MSE/MAE/R²). Same XGBoost settings as the reported model. The Tester's
  assessment text the Scientist reads is still computed in-sample on R; only the scalar changes.
- **Disjoint means fold-disjoint.** Selection keeps its 4-fold inner split V₁…V₄ (seed `fold_seed`), unchanged. The
  feedback split W₁…W₄ deals each Vₖ round-robin (by class, seed `fold_seed+1000003`) over the four Wⱼ, so no
  feedback fold equals or contains a selection fold (|Wⱼ ∩ Vₖ| ≤ ⌈|Vₖ|/4⌉). Row-disjoint splits are impossible
  without shrinking selection, which validates on every row of R; that would change selection in every arm and confound
  the LF2 contrast. Both scores still use R's labels and stay correlated (measured by the T2 same-vs-disjoint check).
- **No-leakage check (01 §4.4 item 2).** Every iteration asserts that both splits use only R (no test-fold id), that
  each is a partition of R, and the overlap bound. A violation raises and crashes the run (ITT failure). Every scored
  run must log `l1_split_disjoint=1`. With fewer than 8 training rows L1 is unavailable (same fallback as selection).
- **Cost.** 4 extra XGBoost fits per iteration (≈ 0.5–3 s at 800–16,000 rows); negligible against LLM time.

## A04 CAP-0 verdicts, sessions A/B (A04-T18; Henrik 2026-10-15)

Sessions `S1014m`–`S1014r` (image `agent2027-pre.3`, GPUs 4,5, ended 2026-10-15 01:02Z). Strict valid-call reading
as registered on 2026-10-14, before the O20 replay ran. Rescued strict / top-level (raw strict):
Q0.8 49/50 (40), Q2 47/50 (35), Q4 50/50 (50), Q9 50/50 (48), Q122 50/50 (49), O20 39/48 (37).
- **SR1 not triggered** (Henrik confirmed): Q9 50/50; Q9's server errors are context-length 400s and request-history
  JSON decoding in `chat_utils`, not tool- or reasoning-parser modules. The Qwen3.5 ladder stays.
- **SR2:** small-rung fold-runs are data: Q0.8, Q4 and Q9 hit the 6 h cap; Q2 finished. They loop (Q2 iteration 0:
  1,528 Scientist calls, 85M prompt tokens).
- **SR3 passes at TP2:** KV 899,630 tokens ≥ 6 × (24,983 + 32,768) = 346,506; the 6-prompt probe completed with no
  errors or preemption. The KV probe's prompts reach only 25k tokens. Q122 serves with `--max-num-seqs 256`.
- **SR4 fails; fallback applied** (Henrik): O20 39/50 < 40 (9 of 11 invalid calls lack a required inner key).
  A19 big = Q122 (TP2, 8040), small = Q4 (TP1, 8032); `mix_esmall` = S, T on Q122 and E on Q4; `mix_1b` = S, T on
  Q0.8 and E on Q122; reference A17 `f2_Q122`; `fx_pajM` comparison dropped. A20 moves to Q4. A17 drops `f2_O20`
  (A17-059…069) and its F0 pool (A17-143…149). Session C is re-planned as Q122·2 + Q4·1 + Q0.8·1 (ΣTP = 4).

## X9 — P7 guessability and retrieval (2026-10-15)

- **Guessability passes:** 20 single-shot gpt-oss-120b calls (temperature 0.7, seeds 1000–1019) saw P7's
  `context_and_goal` with its column descriptions and gave 5 formula guesses each; 100 guesses scored with the
  recovery detector on all rows: **0 hits**, best |score| 0.73. No P7 redesign.
- **Retrieval check deferred** (Henrik: the path most likely to complete): the offline replay of A02's P7 search
  queries through the TF-IDF index over `kt3_correct` runs when E8b's corpus and backend land (backend due Wed 21 Oct,
  G0b-1), before A14 launches. If the correct document does not rank near the top, the corpus vocabulary is adjusted and
  disclosed, as X9 specifies. The corpus is therefore not frozen in this pre-registration.

---

## Part III — Additional pre-freeze inputs recorded at X10

These results were known before this X10 text was written. They are recorded as inputs; none of them is a Full-v2
result.

### III.1 A01 W0 — known pre-freeze inputs (A01-T19; D11)

Source: `paper/results_w0_t17rep_summary.json` (generated 2026-10-12 06:34Z; committed on `agent2027/v2`) and
`docs/ablations/A01_W0_old_image_replicate.md` §6. A01 ran on the old image `rogue_one:nestedcv9` (v1 code, W&B entity
`brainflux`, D9).

- **Variance rule (D11, Henrik 2026-10-09, before any W0 result).** A01's pooled sampling share Z_v1 and its
  variance-inflation test use the **Satterthwaite effective-df rule** (t17 variances as weights; thresholds fixed from
  the committed t17 CSV before any W0 data). The nominal F(16, 16) version is logged as `*_nominal` and is **never used
  for a decision**. The version without `junglechess` is a sensitivity analysis.
- **Rule 1 (usable data):** 16 of 16 complete pairs (8 datasets × seeds 42, 4): full analysis.
- **Rule 2 (drift):** none. Wilcoxon p = 0.528 (W+ = 81, n = 16). Z_v1 = 0.372 against the Satterthwaite threshold
  F_0.975(6.13, 6.13) = 5.69: no variance inflation. (Nominal, not used: threshold 2.76, also no inflation.)
- **Rule 3 (verdict, descriptive):** Z_v1 = 0.372, 95% CI [0.065, 2.12] (Satterthwaite ν₁ = ν₂ = 6.13): **partial**,
  worded "consistent with partial (CI spans 0.36, 0.72)". Per-dataset ρ_d ≤ 0.6 on 3 datasets, ≥ 0.85 on 2.
  Sensitivity without `junglechess`: Z = 0.56, CI [0.072, 4.40]. (Nominal CI, not used: [0.135, 1.03].)
- **Rule 4 (independence):** 0 of 16 pairs byte-identical; σ_samp is not a lower bound.
- **Pooled values handed on:** `sigma_samp_pooled_sd` 4.344 (16 df) and 1.434 without `junglechess` (14 df);
  regression-only `sigma_samp_pooled_sd_reg3` 2.158 (6 df), the A01-scale cross-check of A05's F(3, 6) homogeneity
  rule. Per-pair raw δ: `paper/results_w0_t17rep_pairs.csv`.
- **How they enter later rules.** (i) A03's σ̂_d table (Part V.3) uses σ_samp,d. (ii) A05's homogeneity rule (Part
  V.5) uses the per-pair δ on `housing`, `insurance` and `crab`, rescaled with A05's own s_d; because A01 reports no
  drift, the t17 SD remains admissible as a second estimator there.
- **Caveat.** A01 ran on 2026-10-07 – 10-09, when the Serper account was probably already out of credits (HQ #266),
  and the legacy outputs cannot show whether t17 (Sept 15–22) had live search. σ_samp may therefore include a
  search-on versus search-off difference. Disclosed (Part VII).

### III.2 M5 CPU screen (`M5S`) and the brute-force exclusion

Source: A08 CPU screen, read 2026-10-13 07:50Z and rescored on all 81 cells on 2026-10-14 (405 episodes, 0 errors;
`paper/results/agent2027/a08_screen_episodes.csv`).

- Recoveries (of 15 episodes, OpenFE | AutoFeat): p1 15 | 15; p2, p3, p4, p5, p6a, p6b, p7 0 | 0; p0 0 | 0.
- **Non-discriminating families: 1 of 7 (P1).** Fewer than 4, so SYN-P was not recalibrated. P6a and P6b are both
  discriminating, so C4b is **not** dropped. P7: 0/15 for both methods (P7 validity check (b) passes).
- **Exclusion rule (A08 §6, applied):** `synp_p1` is excluded from every confirmatory recovery population (C1a, C1b,
  C2a–c, C3, C4b, C5b and the RT3 reading of G3). It stays in the runs as a sanity check and in exploratory read-outs.
  The confirmatory SYN-P population is therefore **{p2, p3, p4, p5, p6a, p6b}** ("SYN-P 7 minus P1"); C4a uses `synp_p7`
  only (P7 policy, 01 R10); C5a is not gated (A17).
- **Detector amendment (disclosed).** The p0 control first failed (near-monotone two-column features of the p0 column
  passed the 0.999 value rule). Before A02 launched, the detector was fixed (commit `a6b64ac`): a Spearman hit must beat
  its best single raw column by 0.01 (`no_gain_over_raw`). After the fix p0 = 0/15 for both methods. The same detector
  is used for every arm.

### III.3 A02 read-out and the G1 re-tier

Source: `docs/agent2027/results/A02_readout.md` (merged `e8ec1a1`). Rules: Part II "A02 M4".

- **R = 10** (R=3 not adopted). Criteria 1–3 pass (net loss 15 − 13 = 2 of 40; no ≥ 3/5 → 0/5 drop; success R3 6/8 vs
  R10 4/8). Criterion 4 fails: mean `u_run` ratio 0.9644 / 1.2852 = **0.7504** > 0.75 (R3 4.3 s per row too slow; the rule
  has no tolerance). Leak canary: `synp_p7` `rec_any` = 0/15 over all arms.
- **Re-tier (applied at G1).** Both triggers hold: R=3 is not adopted, and the mean u per SYN-P run of the `R10` arm
  (the adopted arm, D10) is 1.285u > 0.85u. **CL1–CL4 apply at once:**
  - **CL1:** CAP-2 (A19) on SYN-P only; its real-4 rows (A19-015…018, 033…036) are cut.
  - **CL2:** CAP-1 (A17) small-rung real data on 2 datasets only (`junglechess`, `myocardial`).
  - **CL3:** HC1 (A07) on the 4 high-error datasets only: `myocardial`, `vehicle`, `housing`, `crab`. Rows
    A07-009…012 (`heart`, `insurance`) are cut; 8 rows remain (A07-001…008).
  - **CL4:** CAAFE dropped (A08); LLM-FE and the CPU screen stay.
  - CL5–CL7 are not applied at G1.
- **Add-backs AB1–AB5** are funded only if G1 measures u at the low end. The measured u is at the high end, so **none is
  funded**. (AB1, the M1 SYN-P add-on, could only run before the freeze and did not run; A03's criterion (c) is
  "untested".) `TODO(Henrik)` (resolved 2026-10-16, Part VIII.1): confirm "no add-backs".
- R is never re-opened after the freeze.

### III.4 E6 augment bug found and fixed (2026-10-15)

Smoke `S1015k` (image `agent2027-pre.4`) found a bug in augment mode: the seeded empty columns of the attribute table
were `float64`, and pandas 3 raised a `TypeError` on the prune writes. Fixed in `05dea0a` (seeded rows keep empty columns
as `object`) with a regression test; image `agent2027-pre.5`. Smoke `S1015l` (pre.5) passed 4/4: augment `raw_coverage`
1.0 at every iteration, replace 0.0; the replay check ran on every row (it flagged one non-reproducing `credit-g` fold
in both arms: an agent `pd.cut` feature fails fold-safe replay, 0.536 vs 0.692 — the Finding-39 effect the check exists
to catch); `l1_split_disjoint=1` with 0 failures; sandbox on. No Full-v2 run used the buggy code.

---

## Part IV — Shared statistical rules

Source: 01 §4.2–§4.8 and R14. Restated here; 01 governs details not repeated.

### IV.1 Confirmatory family (Holm)

Ten two-sided contrasts, Holm at family-wise α = 0.05 (smallest threshold 0.005). Write the raw p and "Holm-adjusted
p < 0.05", never "Holm-adjusted p < 0.005". Every contrast is tested on SYN-P or P7, never on real data.

| ID | Contrast | Ablation | Population (after Part III.2) | Test of record |
|---|---|---|---|---|
| C1a | SET-phased − Full | A16 | SYN-P 7 minus P1 × {42, 4} | GLMM `rec_sel ~ arm + (1\|task) + (1\|task:seed)`; exact stratified test under separation |
| C1b | LLM-FE − Full | A08 | SYN-P 7 minus P1, seed-42 partition | GLMM `rec_sel ~ arm + (1\|task) + (1\|task:fold)`; three-mode agreement rule |
| C2a | Assessment (A) main effect, TOST | A12 | SYN-P 7 minus P1 × 42, 8 fraction cells | beta-binomial fraction model |
| C2b | Pruning (J) main effect, TOST | A12 | as C2a | as C2a |
| C2c | Planner (P) main effect | A12 | as C2a | as C2a, two-sided |
| C3 | Restarts − Full | A10 | D3: {P3, P5, P2} (none gated) | `rec_sel ~ arm + task + (1\|task:seed)`, LR test; exact CMH under separation |
| C4a | Local-correct − Off on P7 | A14 | `synp_p7`, 6 seeds per arm | exact conditional test stratified by fold seed |
| C4b | P6b − P6a (search on) | A15 (A05 data) | 15 pairs | exact McNemar |
| C5a | `func_success` trend over log₁₀ active parameters (Qwen rungs) | A17 | SYN-P 7 (not gated) | exact Cochran–Armitage stratified by task × seed |
| C5b | E-small − all-big | A19 | SYN-P 7 minus P1 × {42, 4} | GLMM as C1a |

**Drop rules.** A contrast is dropped only before any data on it exist, and Holm is recomputed over the rest:
- C4b — P6 failed the screen: **not triggered** (Part III.2). C4b is also pre-registered as invalid (enters Holm with
  p = 1, family stays at 10) if an input leak is found in an A05 P6b run at G2 (A15 §6).
- C4a — any P7 validity check fails and P7 cannot be fixed by Fri 16; or KT3 killed at G4 (Tue 27).
- C1a — SET killed at K-SET (Fri 23).
- C1b — no LLM-FE run started by L-LAST (Thu 29), or fewer than 2 discriminating SYN-P 7 tasks (6 remain).
- C2a–c — FX not launched by Wed 28, 08:00.
- C3 — A10 not launched by L-LAST.
- C5b — no data (SR6 still failing on Tue 27, or no slot by L-LAST).

**Outside Holm:** cost outcomes (A02 u per run, A13 KT1-Off tokens, A19 big-model token ratio), A07 HC1, A11, A18, A20,
A09 (positive control), A06, A21, A23. Everything not listed is exploratory.

### IV.2 Margins

- **Real-data no-harm (NI).** Δ_std = Δ / σ_d, σ_d = the dataset's Full-v2 run-to-run SD from A05 (pooled with t17
  only if the homogeneity rule passes; Part V.5), floored at 0.1 × the fold SD of `base_no_fe`. Mixed model pooled across
  task types (dataset random intercept, task type fixed, dataset-specific residual variances), one-sided at margin M.
  **M = 1.0 Δ_std, i.e. M_d = σ_d** (A05 OQ2; `TODO(Henrik)` (resolved 2026-10-16, Part VIII.1) confirm the constant). The formula is fixed here; the value
  is computed at G2 (Mon 19) before any arm's held-out score is read. "No harm beyond M" only if the CI excludes −M;
  otherwise "inconclusive". Per-task-type Wilcoxon is descriptive.
- **Equivalence (TOST) margin δ** (SYN-P only, 90% CIs), shared by A09, A12, A16, A17 (RV1), A19, A20, A23:
  gap = pooled `rec_sel` of A09 `oracle` − pooled `rec_sel` of A05 Full f42 r0 + r1, over all episodes of the
  population tasks, ITT zeros included; **δ = max(0.25 × gap, 0.10)**, rounded to 3 decimals; δ = 0.10 if gap ≤ 0 or if
  any population oracle row is not terminal by Fri 23 Oct, 18:00. The absolute floor is 0.10. The value is committed by
  A09-T15 as a dated amendment on Fri 23 Oct, 18:00, before FX recovery is unblinded.
- **Token co-primary (A12, C2a/C2b "reducible").** ≥ 30% fewer completion tokens (upper 90% bound of the X−/X+ ratio
  ≤ 0.70). If a Tester stage is < 30% of `tok_compl_total` in A05, use (i) the replaced stage's own tokens, or (ii) the
  joint A−J− vs A+J+ contrast if stages are not separable. The branch is chosen at G2 from A05's stage shares and
  written in the G2 memo.

### IV.3 Units, models and failures

- **Seeds.** Fold seeds {42, 4}; seed 5 for the M5 screen and the KT3 seed sets. KT3: Off and Local-correct at
  {42, 4, 5, 1, 2, 3}; Parametric and Local-wrong at {42, 4, 5}. LLM seed = fold seed (r0), fold seed + 1000 (A/A),
  5000 + k (O120 sentinel k), 6000 + k (small-rung sentinel k). Every arm pairs with A05 by fold seed.
- **SYN-P fold scheme.** Stratified folds (`ROGUE_ONE_STRATIFIED_FOLDS=True`) on the SYN-P classification tasks
  (`synp_p3`, `p4`, `p5`, `p7`), so M0's seed-42 partition equals LLM-FE's `StratifiedKFold(5, shuffle, 42)`; KFold on
  regression and on real data (D4).
- **P7 policy** (01 R10). SYN-P 8 (with P7) in A05, A02, the A08 CPU screen (plus p0) and the A09 oracle; P7 alone in
  A14; SYN-P 7 everywhere else; p0 only in A05 (seed 42) and the CPU screen. P7 is A05's web reference and leak canary
  (Full expected at 0).
- **Main SYN-P model.** Logistic GLMM at fold-episode level, `rec ~ arm + (1|task)` plus `(1|task:seed)` where there are
  ≥ 2 seeds; no family term. Pre-registered exact tests where separation is likely. Informative tasks (Full strictly
  between 0 and 1) are used for power accounting only, never to exclude. Power is recomputed at G2 from M0's ICC;
  "inconclusive" is a pre-declared outcome.
- **ITT.** A crash (rc ≠ 0, or no `fold_aggregation_summary.csv`) is retried once, alone, at the end of its session.
  Persistent failures and 6 h cap hits are ITT failures, never dropped: all fold-episodes get `rec_sel = rec_any = 0`;
  the held-out score is imputed at **`base_no_fe`** (augment is frozen). Failure and cap-hit rates are reported next to
  every number. Sensitivity: exclude runs whose selected features are all fallback features.
- **Infrastructure faults are not ITT** (A05 §4.6, adopted here for every ablation): vLLM death, host reboot, a W&B or
  search-backend outage, visible in the job or vLLM log. Such rows are re-queued with `notes=infra-rerun`.
- **Pre-freeze points never mix with the frozen front** (01 §4.8 rule 6).
- **Budget matching** (01 §4.8): completion tokens including reasoning and re-rolls, search charged to the caller;
  iso-cost truncation of Full at whole iterations only.
- **Proposed thresholds.** Every threshold or default marked "proposed" in the rule sections of an ablation document is
  frozen at its stated value by this commit, unless this file states otherwise. `TODO(Henrik)` (resolved 2026-10-16, Part VIII.1): confirm this blanket
  adoption (list in Part VIII).

---

## Part V — Per-ablation rules due at G1

Each subsection names the binding source. Text restated here is binding; the source sections are incorporated by
reference (preamble).

### V.1 A01 · W0

Results and rules: Part III.1. A01 is outside the confirmatory family. A01-T15 (optional reruns) was dropped by D1.

### V.2 A02 · M4

Part II ("A02 M4", Amendment 1, Erratum) and Part III.3. As applied: R = 10.

### V.3 A03 · M1 under the pre-declared fallback

Source: A03 §4, §6, A03-T12, A03-T15; Part II "A03 M1".

- **Pool mode:** `augment`, frozen by design (G-E6 missed). The M1 rule did **not** decide the pool mode; it is applied
  to the fallback rows as a check.
- **Rows:** A03-031…040 = arm `replace` (`env_overrides = {"ROGUE_ONE_FEATURE_POOL_MODE":"replace"}`, everything
  else Full-v2) × Core-10 × fold seed 42 (`llm_seed` 42), image `agent2027-v2.0`, `notes=fallback`. Each is paired with
  A05 `full` f42 r0 on the same dataset (A05-001…010; the A/A rows are not used) and sits in the session of its A05 pair.
- **Read-out (A03-T15, G2 Mon 19 Oct).** Δ_d = A05 `full` − A03 `replace` (score_aug − score_rep for C; nRMSE_rep −
  nRMSE_aug for R; Δ > 0 = augment better); `delta_std` = Δ_d / σ̂_d.
  - **(a)** mean `delta_std` over the complete pairs (at least 8 of 10) ≥ −0.75;
  - **(b)** no complete pair has Δ_d / (√2 · σ̂_d) < −2.5, and no dataset has augment failing or capping where replace
    completed;
  - **(c)** untested (the SYN-P add-on could only run before the freeze); disclosed.
  - Primary = complete pairs only, failures reported separately; sensitivity = both arms ITT-imputed at the
    majority-class or mean predictor.
  - **Outcome.** (a) and (b) hold: augment stands. Either fails: Henrik chooses at G2 between reverting to replace
    (re-freeze and rerun A05 + A07, ≈ 60–70u, which fits only at G2) and reporting it as a limitation (D6).
  - `TODO(Henrik)` (resolved 2026-10-16, Part VIII.1): what applies if fewer than 8 of the 10 fallback pairs are complete at G2 (proposal: read (a) and (b)
    descriptively and treat the check as "not evaluable"; augment stands).
- **σ̂_d (pre-registered default estimator):** σ̂_d = max(A01 σ_samp,d, t17 seed SD_d, 0.1 × `base_no_fe` fold SD_d),
  A01 omitted for `balance-scale` and `wine` (no W0 pairs). The first two terms:

  | dataset | A01 σ_samp,d | t17 SD | max (before floor) |
  |---|---|---|---|
  | `balance-scale` | — | 0.000 | 0.000 (floor decides) |
  | `junglechess` | 0.0245 | 0.070 | 0.070 |
  | `vehicle` | 0.0037 | 0.029 | 0.029 |
  | `heart` | 0.0082 | 0.019 | 0.019 |
  | `credit-g` | 0.0141 | 0.0055 | 0.0141 |
  | `myocardial` | 0.0170 | 0.0076 | 0.0170 |
  | `housing` | 0.0151 | 0.021 | 0.021 |
  | `insurance` | 0.0501 | 0.063 | 0.063 |
  | `crab` | 0.0153 | 0.024 | 0.024 |
  | `wine` | — | 0.0010 | 0.0010 (floor decides) |

  The floor term comes from A05-T6 (`base-no-fe-ref`, due Sat 17 Oct) and is applied mechanically. Classification
  values from t17 and A01 are on plain accuracy (t17/W0 ran with `DO_BALANCING=False`); v2 uses balanced accuracy.
  Disclosed.
- **Manipulation check:** `raw_coverage` = 1.0 at iteration 0 in every augment fold of A05; a failure is an E6 bug, fixed
  and rerun, not evidence.

### V.4 A04 · CAP-0

Part II ("A04 CAP-0 verdicts"), `scripts/agent2027/rungs.csv` (Part I.1), and A04 §2 definitions and §6 SR1–SR6 by
reference. Session C (A04-T15) is re-planned as Q122·2 + Q4·1 + Q0.8·1 (ΣTP = 4), each row's global model the unused
third model; SR6 (routing) gates A19. Any rung verdict that arrives after the tag is a dated addendum committed before
that rung's first A17/A18 row.

### V.5 A05 · M0 (A05-T2)

Source: A05 §2, §4, §6.

- **Rows (55):** Core-10 × {f42 r0, f4 r0, f42 r1} (A05-001…030), SYN-P 8 × the same (A05-031…054), `synp_p0` × f42 r0
  (A05-055). Freeze smoke A05-901/902 (`synp_p3`, `heart`, one fold, session `S1016s`) is excluded from every analysis.
  `env_overrides = {}` except `{"ROGUE_ONE_STRATIFIED_FOLDS":"True"}` on the SYN-P classification rows (D4).
- **Sessions (D5 = TP2; weekend allowance).** The A05 rows, A07-001…008 and A03-031…040 (73 rows) are spread over up
  to four O120 TP2 sessions `S1016a`–`S1016d` (ports 8010, 8011, …), random `order_key` within each session (RNG seeded
  by the session ID). **Each A07 row and each A03 row sits in the session of its A05 pair** (same dataset and fold seed,
  r0). The rest are split so the session files differ by ≤ 1 row and hold similar Core-10/SYN-P mixes. A05's sessions
  carry no A06 sentinel (they are the reference). `TODO(Henrik)` (resolved 2026-10-16, Part VIII.1): fix the number of servers (2–4), their ports and the
  GPU ids before A05-T5 writes the session files.
- **Definitions (frozen formulas; values at G2).** For dataset d let a = (f42, r0), b = (f42, r1), c = (f4, r0) be
  `heldout_score`.
  - s_{d,s} = population SD over the 5 outer folds of `base_no_fe` at fold seed s (A05-T6: `accw_balanced_sd` for C,
    `nrmse_sd` for R); s_d = mean(s_{d,42}, s_{d,4}). `success` and the P0 check use s_{d,s}.
  - Δ = arm − Full on `heldout_score`, sign-flipped for nRMSE, so Δ > 0 always means the arm is better.
  - σ̂²_AA = mean_d[((a − b)/s_d)²] / 2 over Core-10 (10 df); σ̂²_p = max(0, (mean_d[(((a + b)/2 − c)/s_d)²] − 1.5
    σ̂²_AA) / 2); Z = σ̂²_AA / (σ̂²_AA + σ̂²_p).
  - σ_d = SD of {a, b, c} in raw units (2 df), floored at 0.1 × s_d.
  - **Homogeneity rule** (`housing`, `insurance`, `crab` only; classification datasets are never pooled because
    balancing differs from t17): F = σ̂²_AA,R / σ̂²_samp,R, with σ̂²_AA,R = mean over the 3 datasets of ((a − b)/s_d)² / 2
    (3 df) and σ̂²_samp,R = mean over the 6 W0–t17 pairs on the same 3 datasets of (δ/s_d)² / 2 (6 df), A01's per-pair δ
    rescaled with A05's s_d. Pool iff F lies in the central 95% of F(3, 6) = [0.068, 6.60] (F(3, k) with k < 6 usable
    pairs; no pooling with k = 0), and only if A01 reports no drift (it reports none, Part III.1). If pooled:
    σ²_d = (σ̂²_d + SD²_t17,d) / 2, then the same floor.
  - `ni_margin`: M_d = M · σ_d with M = 1.0 (Part IV.2).
  - `icc_recovery`: logistic GLMM `rec_sel ~ 1 + (1|task) + (1|task:seed) + (1|run)`; ICC = σ²_task / (σ²_task +
    σ²_task:seed + σ²_run + π²/3). Primary population SYN-P 7 (105 episodes); SYN-P 8 as sensitivity; ANOVA ICC on run
    proportions if singular.
  - Informative task: Full's `rec_sel` strictly between 0/15 and 15/15 (power accounting only).
  - B_d = median `tok_compl_total` per iteration over the 150 iterations of the dataset's 3 A05 runs; role and
    Tester-stage shares reported (inputs to A12's token rule and A19's τ).
  - u_SYN-P = median SYN-P `wall_s` ÷ median Core-10 `wall_s` in the session(s).
  - **Sentinel task:** A06 rule (Part V.6).
- **`success`:** finishes within 6 h and `heldout_score` ≥ `base_no_fe` − 2 × s_{d,s} (nRMSE: ≤ `base_no_fe` + 2 ×
  s_{d,s}); on `junglechess` also ≥ 0.95 ("rule found"). `TODO(Henrik)` (resolved 2026-10-16, Part VIII.1) (A05 OQ1): confirm the 0.95 threshold on balanced
  accuracy (it was set on plain accuracy). `func_success` = `success` and ≥ 1 accepted non-fallback LLM feature in the
  selected iteration.
- **ITT / infra / late rows.** ITT as Part IV.3 (imputation at `base_no_fe`). Infra faults are re-queued in session
  `S1018a` with `notes=infra-rerun`. If re-queued rows are not finished by Mon 19 Oct, 06:00, G2 is computed on the
  finished rows; the constants are frozen at that computation and late rows enter descriptive tables only.
- **Search.** The A/A replicate samples live SearXNG search (cache off). σ_AA therefore includes the variation of live
  search results; it is **no longer a lower bound** on live-search variance, as A05 §4.5 assumed under the shared cache.
- **Launch and early abort.** Launch only when A05-T1–T5 and T7–T8 are done, `agent2027-prereg` is pushed and the smoke
  is clean. Late start: if `S1016a` has not launched by Sat 17 Oct, 06:00, A07-T14 removes the A07 rows. Early abort
  (A05-T10, T11): a role with zero `tok_compl_*` in a finished run; missing traces, per-inner-fold scores or `run-output`
  in a run that did not crash; any inner-CV fallback line; the same exception (type and innermost Rogue One code
  location) in ≥ 2 rows. Then stop, fix, re-freeze as `agent2027-v2.0r1` / `agent2027-v2.0-freeze-r1` with a dated
  amendment, and restart all A05/A07/A03 rows in a new session; aborted runs are excluded.
- **G2 rules:** Part VI.2.

### V.6 A06 · M0-S session sentinel (A06-T6)

Source: A06 §4.3–4.9, §6. **Sentinels are never pooled into Full for primary contrasts.**

- **Sentinel-task rule** (applied at G2, before any arm result is read): candidates SYN-P 7 (P7 and P0 excluded); pick the
  task whose seed-42 `rec_sel` over A05 r0 + r1 (10 episodes) is closest to 5/10; ties: the 15-episode rate (adding f4
  r0) closest to 0.5, then the lower task index (`synp_p1`, `p2`, `p3`, `p4`, `p5`, `p6a`, `p6b`). A05 §2 states the same
  order. If no candidate is informative, the rule is kept (A06 Q3 default; `TODO(Henrik)` (resolved 2026-10-16, Part VIII.1)).
- **Arms.** `sentinel`: O120, fold seed 42, the G2 task, `llm_seed` 5000 + k; under TP2 alternating ports 8010/8011.
  `sentinel_Q4`: Q4 (revision in Part I.1), `synp_p2`, fold seed 42, `llm_seed` 6000 + k, port 8032.
  `replicate = k`, `is_sentinel = 1`; an llm_seed is never reused.
- **Placement.** One sentinel per qualifying session (one more per further 48 h); the first qualifying session on each
  new image (v2.1 if it carries one, v2.2, v2.3, v2.4) puts it in the first wave (guard-3 session). No sentinel in A05's
  sessions `S1016a–d`, `S1018a`, smoke sessions, A18 replay-only sessions, F0-pool-only sessions, and the single Q122
  session. The first small-rung session also carries A17's `f2_Q4 × synp_p2 × f42 × r0` row (the Q4 anchor). Because
  A20 now runs on Q4 (SR4), the former O20-session exception (A06 Q2) no longer arises; A20's Q4 rows run in a session
  that serves Q4 and that session carries a `sentinel_Q4` under the general rule. A06 Q1 (sentinel in `S1019a` on v2.1):
  `TODO(Henrik)` (resolved 2026-10-16, Part VIII.1); default yes if the G2 memo is out before A09 launches (required anyway if A07-T14 moved the HC1 rows
  there).
- **Per-session check.** Five channels: cost (`tok_compl_total`, `llm_calls_total`, log), time (`wall_s`, log),
  execution (`err_per_accepted` log1p; execute rate `fun_executes / fun_proposed`; `context_errors` log1p), search
  (`serper_ok_false_rate`, the search-failure rate, raw), score (`heldout_score / s_task`). Band for metric m: centre =
  mean of the two reference runs (A05 r0, r1 on the task); half-width = max(3 · √1.5 · σ_AA,m, floor_m), σ_AA,m =
  √(Σ dᵢ² / 2n) over A05's 8 SYN-P A/A pairs at seed 42 (8 df); floors 0.1 (log, log1p), 0.05 (rates), 0.1 (scaled
  score). A channel is outside if any of its metrics is outside. Q4 band: centre = mean of the Q4 anchor and
  `sentinel_Q4` k = 1; σ_Q4,m = max(σ_AA,m, |anchor − k1| / √2).
- **Hard indicators** (any → FLAG): persistent crash; cap hit (Q4: only if neither Q4 reference capped); search failure
  rate ≥ 0.5; inner-CV column missing or all-NaN; a `flags/*` value differing from the reference (other than identity,
  model, port and post-v2.0 flags at their Full value); a `vllm_image_id` differing from the reference.
- **Verdicts.** PASS: no channel outside and no hard indicator. WARN: exactly one channel outside. FLAG: ≥ 2 channels
  outside (≥ 3 for Q4), any hard indicator, or the same channel outside in two consecutive sentinel-bearing sessions of
  the same arm on the same image. `rec_sel` per session is descriptive only.
- **Pooled tests** (Nov 1–3, α = 0.05, outside Holm): two-sided Fisher exact test of pooled O120 sentinel episodes X/Y
  against A05's Z/10; sensitivities: exact test stratified by outer fold, run-level permutation; descriptive exact
  Cochran–Armitage trend over session order and Q4 homogeneity (Fisher–Freeman–Halton). If Fisher p < 0.05: every
  arm-vs-M0 contrast gets a drift caveat and each confirmatory GLMM is refitted with the sentinels as Full observations,
  `rec_sel ~ arm + period + (1|task) + (1|task:seed)`; report whether any Holm verdict changes.
- **FLAG actions** (A06 §6): stop the queue; diagnose within 12 h; repeat the sentinel next session; cause fixed → rerun
  the session's arm rows from the reserve (by L-LAST where possible, launched no later than Fri 30); no cause and the
  repeat passes → keep rows with `drift_flag = 1` and report with and without; repeat fails → no further sessions on that
  image, revert, disclose. Guard 3 needs PASS (or WARN not repeated in the next sentinel session on that image).
- **Failures.** Crash: one solo retry, same row and seed. Cap hit: never retried. Infra fault: new k, new row.
- **CL5** (not applied at G1): sentinel in every second O120 session, floor of 6; guard-3 sessions exempt.

### V.7 A07 · HC1, under CL3 (A07-T7 block)

Source: A07 §2, §4, A07-T16, §6. Outside Holm (raw p only), claim C6.

- **Design.** Arm `errors_hidden` (`env_overrides = {"ROGUE_ONE_TOOL_ERROR_FEEDBACK":"none"}`, nothing else changed)
  against A05 `full` r0 on the same dataset and fold seed, same session and image. **Under CL3: 4 datasets**
  (`myocardial`, `vehicle`, `housing`, `crab`) × fold seeds {42, 4} = **8 pairs**, rows A07-001…008 (`llm_seed` =
  fold seed). Each row sits in the session of its A05 pair.
- **Sites (code at `07cbf93`).** Under `none` exactly these return `None`, as in t17: h1 `sci/agent.py:115-131`
  `_report_tool_error` (any S tool exception); h2 `sci/agent.py:153-156` (empty search result); h3 `sci/agent.py:255-258`
  (S pandas command set no `result`); h4 `ext/agent.py:207-223` `_report_tool_error`, including the KeyError hint of
  `pandas_exec.py:144-155` (any E tool exception); h5 `ext/agent.py:265-268` (empty search result); h6
  `ext/agent.py:333-340` (ILLEGAL COMMAND, via `console_print`); h7 `ext/agent.py:360-363` (E pandas command set no
  `result`); h8 `ext/agent.py:500-504`, `:511-519`, `:522-536` (append rejected: no `result`, NaN/Inf, non-numeric); h9
  `tst/agent.py:1023-1039` `_report_tool_error` (any T tool exception). Visible in both modes: v1 duplicate-name rejection
  (`ext/agent.py:486-489`), v2 auto-unprune message (`:485`, not an error), v3 Tester "Pruning is blocked" guard
  (`tst/agent.py:1283-1289`), v4 summariser fallback strings and the Tester executor default "No output variable
  defined.". Runner exceptions never reach an agent in either mode.
- **Metrics (one value per run).** Tool error = one event at h1–h9, v1 or v3, any role (not v2, v4, runner exceptions or
  context errors). Accepted feature = a success return of `append_new_attribute` minus auto-unprune restorations;
  deterministic fallback features never count. **`err_per_accepted` (primary)** = `n_tool_errors` / `n_accepted` (both
  counts reported, all roles and Extractor only, split by site). Secondary: `err_repeat_rate` (same role, tool, exception
  type and normalised message as the previous error of that role in the same fold-iteration); `recover_within3`
  (Extractor errors followed by an accepted append within the next 3 Extractor tool calls); `silent_rejections` (h8 + v1);
  `append_success`; Extractor tokens per accepted feature; `max_prompt_tokens`, `context_errors`, cost, `fallback_share`,
  `success`, `cap_hit`. No-harm: `delta_std` (descriptive).
- **Complete-pair rule.** A pair where either run lacks a complete `run-output` artifact is excluded from the log-ratio
  model and counted per arm; those runs still enter the harm flag. **If fewer than 6 complete pairs remain (under CL3),
  ρ is descriptive only.**
- **Model.** Per pair (d, s): y = log[(e_hid + 0.5)/(a_hid + 0.5)] − log[(e_shw + 0.5)/(a_shw + 0.5)], e =
  `n_tool_errors`, a = `n_accepted`; y_ds = μ + u_d + ε_ds (random dataset intercept, REML); ρ̂ = exp(−μ̂) (shown ÷
  hidden), hidden-arm excess exp(μ̂) − 1.
- **CI rule (binding).** 95% CI for μ = ȳ ± t(0.975; D − 1) · SD(ȳ_d)/√D, ȳ_d = mean of y within dataset d, D = number
  of datasets with ≥ 1 complete pair (**D = 4, t(3), under CL3**). A dataset with one complete pair enters with that
  pair's y. This rule is used even where it differs from the REML CI. CI for ρ = [exp(−μ_upper), exp(−μ_lower)].
- **Permutation test (robustness).** Exact sign-flip over the n complete pairs (2ⁿ patterns; 256 at n = 8), statistic =
  mean y, two-sided p = share of patterns with |mean| ≥ observed |mean|.
- **§6 verdict table** (rows 1–6 in order, first match applies; row 7 always reported):

  | # | Outcome | Rule | Report |
  |---|---|---|---|
  | 1 | Too few pairs | fewer than 6 complete pairs (CL3) | ρ̂ and per-pair values descriptive; no verdict |
  | 2 | Errors rare | median all-role `n_tool_errors` over the A05 control runs of the complete pairs < 20 per run | "channel rarely exercised in v2"; ρ descriptive |
  | 3 | Supported | ρ̂ ≤ 0.70, CI upper bound < 1, and permutation p < 0.05 | the hoped sentence |
  | 4 | Smaller or less robust effect | CI upper bound < 1, row 3 fails | "shown text reduces errors" + "by less than 30%" and/or "not robust to the permutation test" |
  | 5 | Opposite | CI lower bound > 1 | "error text increased errors"; default unchanged |
  | 6 | Inconclusive | otherwise | "no detectable effect at N pairs" |
  | 7 | Harm flag (descriptive) | (ITT failures + cap hits in `errors_hidden`) − (same in the paired controls) ≥ 2 | stated next to ρ |

- **CL3 rule (applied).** The re-tier fired (Part III.3), so rows A07-009…012 are cut; the load moderator (high vs low)
  is lost and is not reported. The frozen default stays `message` whatever HC1 shows.
- **Fallback.** If `S1016a` has not launched by Sat 17 Oct, 06:00, the A07 rows move to the first O120 session after G2
  (`S1019a`, v2.1) with an A06 sentinel; cross-session pairing is disclosed. If HC1 has not run by L-LAST, it is dropped
  and C6 rests on A02.

### V.8 A08 · M5 (A08-T13)

Source: A08 §2, §4, §6, A08-T21.
- Screen result and exclusion list: Part III.2.
- Incomplete CPU cell: missing if no JSON or `folds_done` < 5; rerun once alone (`CELL_TIMEOUT=21600`); if still
  missing, that method labels the task non-discriminating only if it misses ≤ 1 of ≥ 10 completed episodes.
- **C1b population:** discriminating SYN-P 7 tasks (6: p2, p3, p4, p5, p6a, p6b) on the seed-42 partition; Full r0 + r1
  vs LLM-FE r0 + r1, paired by fold. Selected features: Full = inner-CV-selected iteration; LLM-FE `candidates_used`
  under notebook (primary) or best3; any-program = every program with a non-null score.
- **C1b model and claim rule:** GLMM `rec_sel ~ arm + (1|task) + (1|task:fold)`, two-sided Wald p for `arm`
  (LLM-FE − Full) in the notebook-mode model; under separation an exact stratified test by task with fold-episodes as
  units (exact conditional logistic regression). A direction is claimed only if (a) that p passes Holm at its rank
  **and** (b) the point estimates have the same strict sign in all three modes (notebook `rec_sel`, best3 `rec_sel`,
  any-program `rec_any` against Full's `rec_any`); a zero counts as disagreement. Otherwise "inconclusive", or
  "inconclusive (selection-dependent)" if (a) holds and (b) fails.
- **CL4 applied:** CAAFE is not run; the CAAFE secondary estimate and its rules lapse.
- Comparator ITT: GPU crash retried once alone; persistent failure or cap = ITT (`rec_sel = rec_any = 0`); an LLM-FE
  split where every selected program fails gets `rec_sel = 0` for that episode under that mode.
- Comparator settings: LLM-FE native sample budget, `LLMFE_MAX_TOKENS=2048`; OpenFE `OPENFE_NJ=1`, all features kept;
  AutoFeat `feateng_steps=2`, `n_jobs=4`. LLM-FE's default sample budget is recorded by A08-T4 (Mon 19) as a dated
  addendum. We author the LLM-FE spec files (X4) and disclose it.
- If the partition check fails for a task, fold pairing is dropped for it (task intercept only), disclosed.

### V.9 A09 · RT3 (A09-T1)

Source: A09 §4.2 (‡ items), §6. Positive control for G3, outside Holm.
- Oracle template "Build exactly this feature with `append_new_attribute`: {name} = {expression in T's raw column
  names}." (p3, p5, p7 add "Build it in steps: …"); {name} = `oracle_feature`; all constants given; same string at all
  50 (fold, iteration) slots. The SHA-256 of each `oracle_focus` string is committed, not the text. `TODO(Henrik)` (resolved 2026-10-16, Part VIII.1): add
  the hashes (they are generated off-repo from φ metadata).
- σ map (T → σ(T), fallback): p1→p4 (p5), p2→p4 (p5), p3→p4 (p5), p4→p2 (p1), p5→p1 (p2), p6a→p5 (p4), p6b→p5 (p4),
  p7 (option only) → p4 (p5). Missing source fold (< 10 focus rows): σ(T) f42 r1, then the fallback's f42 r0, then f42
  r1. Renaming rule as A09 §4.2 (rank by max |Spearman| with φ inputs; type-matched; single pass, longest name first;
  assert no φ input name remains).
- Population: SYN-P 7 minus P1. Test of record: GLMM `rec_sel ~ arm + (1|task) + (1|task:seed)`, arms {Full, oracle,
  misdirected}, Full = A05 f42 r0, f42 r1, f4 r0. `TODO(Henrik)` (resolved 2026-10-16, Part VIII.1): name the fitting routine (A09 §6 says "named in X10").
  Exact fallback (non-convergence, singular, or separation): one-sided exact conditional test of a common OR stratified
  by task, two arms at a time. With fewer than 3 population tasks the exact test is the test of record.
- G3 planner half passes if either route holds, each at one-sided α = 0.025: route 1 misdirected < Full; route 2 oracle >
  Full with route 1 not significant. Misdirected > Full never passes. ITT sensitivity: a pass that depends on ITT zeros
  reads "pass (fragile)". Headroom flag at G2 (fewer than 3 gate-passing tasks with Full `rec_sel` ≥ 0.4: noted; AB3
  is not funded under the re-tier). Incomplete-data rule: rows not terminal on Sun 25 → G3 read on terminal rows,
  "provisional".
- δ: Part IV.2; value by A09-T15 on Fri 23, 18:00.

### V.10 A10 · LF1 (A10-T3)

Source: A10 §3, §4, §6. Contrast C3.
- Population (D3): multi-step subset {P3, P5, P2} minus gated families (none gated); pooled SYN-P 7 minus P1 as
  secondary. Units: fold-episodes of `restarts` and A05 `full` r0, ITT zeros included.
- Test of record: `rec_sel ~ arm + task + (1|task:seed)` (task fixed), LR test of `arm` (1 df, two-sided); OR with 95%
  profile CI; Δ`rec_sel` with Newcombe CI beside it. Separation fallback (either arm at 0% or 100%, |log-OR| > 10, or
  non-convergence): exact CMH stratified by task × seed, conditional-MLE OR with exact CI. Sensitivities (never in Holm):
  exact sign-flip permutation over (task, seed) pairs; A05 A/A as third Full replicate; fallback-only runs excluded.
- Secondaries: `rec_any`, arm × multi-step, `lineage_share`, iso-cost via A21. H3 (descriptive): `rec_any_pred` = 1 − (1
  − `q0_iter0`)¹⁰; "independent draws" if observed Σ restart `rec_any` ≥ Pred − 1.96·SE, else "correlated draws".
- Restart seed rule: fold f's restart k uses `llm_seed + 10f + k`. The replace-mode rule lapses (augment frozen).
- Verdicts: Holm-adjusted p < 0.05 with the sign deciding "Full > Restarts" or "Restarts > Full"; else "inconclusive".
  G3 governs only how a null is read. L-LAST drop: C3 dropped if A10 has not launched by Thu 29. A10-T18: if the
  per-iteration active-feature log is missing from v2.0, the G2 choice is (i) rerun A05 SYN-P rows or (ii) score
  `rec_sel` from the trail files with every column counted as active (disclosed upper bound in both arms).

### V.11 A11 · LF2-L0 (A11-T2)

Source: A11 §6. Exploratory, outside Holm. Under D2 = L1 the `l0` arm also drops the L1 scalar; the leak audit's `Status`
column check applies as written.
- Estimate of record: OR L0/Full on `rec_sel` (95% profile CI, raw two-sided LR p), D3 population, ITT zeros included;
  separation fallback as A10. R1 feedback used: OR < 1 and CI excludes 1 ("the loop learns from feedback" only if C3
  passes Holm with Full > Restarts and G3 passes). R2 null, classified by Full's checks: "unused" if
  `feedback_read_share` < 0.25; "uninformative scalar" if `train_metric_saturation` ≥ 0.8. **Under L1 the Scientist's
  scalar is the inner-CV score, not `train_*`; `TODO(Henrik)` (resolved 2026-10-16, Part VIII.1): confirm that `train_metric_saturation` is read on the
  L1 scalar (`cv_accuracy`/`cv_RMSE`) or dropped.** R3 opposite: OR > 1 and CI excludes 1.
- Signature (descriptive), leak audit (any hit invalidates the run; ≤ 2 hits → fix and rerun those rows; > 2 → rerun all
  14 if possible by L-LAST, else process metrics only), sentinel and G3 rules as A11 §6. ED6 if not launched by Mon 26.

### V.12 A12 · FX (A12-T1)

Source: A12 §2, §6, A12-T3, T5–T8, T14, T16.
- Fraction I = −PAJM (resolution IV; PA = −JM, PJ = −AM, PM = −AJ). Cells `fx_pAJM`, `fx_PaJM`, `fx_PAjM`, `fx_PAJm`,
  `fx_pajM` (minimal harness), `fx_paJm`, `fx_pAjm`, `fx_Pajm` × SYN-P 7 × f42 r0; J+ cells set `PRUNING_SHADOW=1`. The
  falsifier moves to the minimal harness (declared deviation).
- Model: 8 cells only, SYN-P 7 minus P1; run-level beta-binomial, ±1 coding, `rec ~ P + A + J + M + PA + PJ + PM +
  (1|task)`; Δ_X = predicted mean X+ − X− with tasks equally weighted; Firth logistic with task fixed effects if it does
  not converge. C2a, C2b by TOST against ±δ (p = larger one-sided p); C2c two-sided. Tokens: LMM on log tokens, ratio
  X−/X+ = exp(−2β_X) with 90% CI.
- C2a/C2b verdict order: reducible (Holm-significant TOST and upper 90% token-ratio bound ≤ 0.70) → equivalent, no
  saving → necessary (95% CI of Δ_X above 0, unadjusted) → LLM version worse → inconclusive. C2c: Holm-significant
  Δ_P > 0 or < 0, else inconclusive worded by G3.
- Falsifier (outside Holm): `fx_pajM` r0 vs A05 Full f42 r0 + r1, 90% CI inside ±δ → "one LLM role plus a
  deterministic harness suffices"; below −δ → "S and T LLM components jointly matter"; else inconclusive.
- Level specifications T5–T8 (P−, A−, J−a, M−) as A12 §2; J−a never prunes augment-seeded rows.
- Task-blocked sessions: block 1 = p3, p5, p2, p6a (`S1021a`); block 2 = p6b, p4, p1 (`S1023a`).
- Blinding: no detector runs on A12 rows until A09-T15 has committed δ (Fri 23, 18:00). G4 alias rule: pairs reported
  jointly; RV3 semi-foldover only if a pair coefficient has p < 0.05 and ≥ 28u reserve remain. Drop: not launched by Wed
  28, 08:00 → C2a–c dropped.

### V.13 A13 · KT1-Off (A13-T1)

Source: A13 §4, §6. Cost outcome, outside Holm.
- Off edit list (`ROGUE_ONE_KNOWLEDGE_BACKEND=off`; `web` byte-identical): the six edits of A13 §6 item 1 (drop
  `search_tool` from the Scientist's tools; the two Scientist prompt rewrites, including edit 2 which is not in 01 E9e;
  delete "Use search_tool actively when uncertain."; drop `search_in_literature_tool` from the Extractor's list and tools
  and from the Tester's Stage A tools). Nothing else changes.
- Cost verdict on `tok_compl_total` (geometric-mean ratio Off ÷ mean of the two Full runs, t-based 95% CI with t₉,
  exact sign-flip p over 2¹⁰ flips), in order: K1 more expensive (CI above 1); K2 clearly cheaper (CI below 1 and ratio
  ≤ 0.95); K3 small cost centre (ratio > 0.95, not K1); K4 inconclusive.
- No-harm: mean of the 10 Δ_std (control = mean of A05 f42 r0 + r1), Var(Δ_std) ≈ 1.5, bounds mean ∓ 0.637; "no harm
  beyond M" if lower bound > −M, "harm" if upper bound < −M, else inconclusive. Scores sealed until the G2 memo.
- RV2 trigger (Mon 26): (i) pooled verdict "harm", (ii) `junglechess` Δ_std < −M, or (iii) Off's `junglechess`
  rule-discovery count ≤ Full's mean − 2. K-only vs mixed split as A13 §6. Under cache off, Full's search calls all go
  upstream; A13's disclosure about a cache lowering Full's search variance lapses.

### V.14 A14 · KT3-P7 (A14-T5), with the corpus deferred

Source: A14 §2–§6, A14-T14. Contrast C4a.
- Arms and seeds: `kb_off` and `kb_local_correct` at {42, 4, 5, 1, 2, 3} (6 runs, 30 episodes each); `kb_parametric`
  and `kb_local_wrong` at {42, 4, 5}. 18 runs.
- Test of record (C4a): exact conditional test stratified by fold seed, two-sided, episodes as units; statistic T = Σ
  Local-correct recoveries; null = convolution over seeds of the per-seed hypergeometric; two-sided p = Σ P(t) over t
  with P(t) ≤ P(T_obs). Unit test: 18/30 vs 6/30 spread 3/5 vs 1/5 per seed gives p = 0.0051. Sensitivity: run-level
  6-vs-6 exact permutation (924 splits). H2 (secondary, outside Holm): same test, Local-correct vs Parametric and vs
  Local-wrong on {42, 4, 5}. Wilson CIs per arm.
- TF-IDF backend: `TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2), sublinear_tf=True)` over
  whole documents; top k = 3; each hit `<title>: <first 1,500 chars>`; all-zero scores → next query variant, then "No
  results found.".
- Parametric text: "*You have no tools and no access to the web or documents. Answer the user's query from your own
  knowledge, as specifically as you can; if you do not know, say so.*" replacing the summariser's tool-referring lines;
  `tools=[]`.
- φ_wrong criterion: |ρ(φ_wrong, φ)| < 0.5 on training rows and XGB(raw + φ_wrong) − XGB(raw) < 0.01 balanced accuracy.
- **Deferred (Henrik, 2026-10-15; Part II "X9"):** the corpus is not frozen here. Corpus contents rule (`kt3_correct` = P7
  document + 5–10 distractors; `kt3_wrong` = the same files with only the formula paragraph swapped to φ_wrong) and hash
  definition (SHA-256 of the sorted `sha256sum` listing) are fixed now. The corpus hashes, the φ_wrong ρ and headroom
  values, and the TF-IDF replay counts (P7 document in the top 3 for ≥ 50% of queries containing the domain term; same
  rank in both corpora for ≥ 90% of queries) are committed as a **dated amendment when E8b lands (G0b-1, Wed 21 Oct),
  before A14's first row**. Any vocabulary change is listed and disclosed. A14-T10's hash check is against that
  amendment.
- P7 validity (A14-T3): (a) pairwise screen — `TODO(Henrik)` (resolved 2026-10-16, Part VIII.1) record the result; (b) M5S 0/15 both methods — pass;
  (c) X9 guessability 0/100 — pass; (d) A02 P7 canary 0/15 — pass.
- §3 interpretation thresholds, G2 leak-canary rule (default if undecided by end of Mon 19: C4a dropped before data),
  G3 knowledge half (pass if raw p ≤ 0.005; provisional 0.005–0.05; fail otherwise) and the title rule as A14 §6. Kill
  date G4 (Tue 27); latest start Wed 28 night.

### V.15 A15 · KT2-min (A15-T11)

Source: A15 §2, §5.4, §6. Contrast C4b on A05 data.
- Twin rule (D8 option B; Part I.3): columns `f01…fNN` by sorted rank of the P6a name; levels `c01…cKK`; neutral
  context ±2% chars; hidden positive rescaling of every numeric column (φ is a monomial); same rows, order, id and
  byte-identical `labels.csv`.
- Pair = (fold_seed, replicate, fold index) over A05 `synp_p6a`/`p6b` × {f42 r0, f4 r0, f42 r1} (15 pairs); Δ = rate(P6b) −
  rate(P6a) = (n01 − n10)/n; exact two-sided McNemar; Newcombe (method 10) 95% CI; ITT. AB2 (seed-5 pair) is not funded.
- Leak definition and rules as A15 §6; G2 leak in an A05 P6b run → C4b invalid, enters Holm with p = 1.
  `query_domain_share` per A15 §2 (unit = one W tool call). `TODO(Henrik)` (resolved 2026-10-16, Part VIII.1): add the SHA-256 of
  `synp_meta/p6_domain_terms.txt` (A15-T7).
- **Blinding note, corrected.** A15 planned to state that the pre-freeze P6 runs were never read as a P6a-vs-P6b table.
  That is not true: the A02 read-out (`docs/agent2027/results/A02_readout.md`, 2026-10-14) tabulates `rec_sel` per task
  and arm, including `synp_p6a` 5/5/5 and `synp_p6b` 1/0/1 (R10/R3/R1), on pre-freeze code with web search dead
  (Amendment 1 applies to `S1013a/b`; the read-out sessions `S1013c/d` had SearXNG). These runs are excluded from C4b and
  reported only in a separate pre-freeze panel; the contrast was seen before this commit and this is disclosed (Part VII).

### V.16 A16 · RT2 (A16-T1)

Source: A16-T6 (SET construction spec: neutral system prompt, deletion list, the 23-row substitution table, phase-header
text "### PHASE k/4 — <NAME>. From now on only the instructions below and the tools of this phase apply.", E-round and
retry rules), §6 (K-SET pass rule with 7 criteria per pilot run, contingencies, C1a model and separation fallback,
verdict rule, sensitivities S1/S2, engineering-collapse overlay at ≥ 3 of 14 SET SYN-P runs). C1a population: SYN-P 7
minus P1; Full 3 runs per task, SET 2. δ absolute floor 0.10 (Part IV.2). K-SET Fri 23; ED4 drops C1a if no pass by Fri
23, 24:00.

### V.17 A17 · CAP-1 (A17-T1, T3)

Source: A17 §2, §4, §6.
- Outcome `func_success` at run level: the condition must hold in ≥ 3 of 5 folds (sensitivities 5/5 and ≥ 1/5).
  Accepted = appended by the Extractor and `Status` ≠ Pruned; fallback features, raw columns and augment-seeded rows never
  count; no summary file → 0.
- Population: Qwen rungs × SYN-P 7 (not gated): Q0.8, Q4, Q122 at seeds {42, 4}; Q2, Q9 at 42; 56 runs. x = log₁₀ active
  B: Q0.8 −0.097, Q2 0.301, Q4 0.602, Q9 0.954, Q122 1.000.
- Test of record (C5a): exact Cochran–Armitage trend test stratified by task × seed (14 strata; convolution; scores =
  x), two-sided. Sensitivity: Firth logistic `func_success ~ log10(active) + task + seed`. Retry rule: the final attempt
  is the datum.
- **Rung decisions (A04, Part II):** SR1 not triggered (Qwen3.5 ladder stays); Q122 at TP2 (SR3 pass); **SR4: `f2_O20`
  (A17-059…069) and its F0 pool (A17-143…149) are dropped.** **CL2 applied:** small-rung real data on `junglechess` and
  `myocardial` only. AB4 (Q2 at 2 seeds) not funded.
- Functional threshold, F0 reporting rule, RV1 trigger (G4; floor guard: a rung cannot trigger if pooled O120 `rec_sel`
  ≤ δ) and kill dates as A17 §6.
- **F0 prompt.** Contents fixed by A17-T2 (context_and_goal; column names and dtypes; 5 training-fold rows without the
  target; the Extractor command convention; "return only a JSON list of ≤ 10 {name, pandas_command, argumentation}";
  temperature 0.7, max_tokens 32768, no tools, no search). `TODO(Henrik)` (resolved 2026-10-16, Part VIII.1): the exact text is not written yet (X7 due Fri
  23); commit it as a dated amendment before the first F0 pool row.

### V.18 A18 · RP1 (A18-T1, T4)

Source: A18 §2, §4, §6. Exploratory, outside Holm.
- Source prompts: A05 f42 r0 SYN-P 8 + Core-10 (18 runs); allocation n = 800 (S 200, E 300, T-assessment 100,
  T-pruning 100, W 100) as A18-T1; `llm_seed` 7042; scores `rp1_valid_call_raw`, `rp1_valid_call_rescued`,
  `rp1_executes`, `rp1_usable` with the per-role rules and lexicons of A18 §2; common-set and 600 s timeout rules.
- **RP1-D1 floor:** a target passes when its rate is ≥ 0.8 × O120's self-replay rate (same role and score, common set);
  floor = the smallest Qwen rung that passes with every larger rung passing. Primary score `rp1_usable`; transport floor
  on `rp1_valid_call_rescued`. Floor-stability threshold 0.6; D3b minimum denominator 30. D2–D6 as A18 §6.
- Targets: `replay_Q0.8`, `Q2`, `Q4`, `Q9`, `Q122` (TP2), control `replay_O120`. `TODO(Henrik)` (resolved 2026-10-16, Part VIII.1): keep `replay_O20` as a
  descriptive pass/fail target (it needs its own O20 server now that no O20 session runs) or drop it.

### V.19 A19 · CAP-2 under SR4 and CL1 (A19-T1, T2)

Source: A19 §2, §6 (SR4 fallback paragraph).
- **Rungs (SR4):** big = Q122 (TP2, 8040), small = Q4 (TP1, 8032), Q0.8 (TP1, 8030). `mix_esmall` = S, T on Q122 and E
  on Q4; `mix_1b` = S, T on Q0.8 and E on Q122; global `LLM_MODEL`/`LLM_PORT` = Q122/8040. W follows its caller.
  **Reference = A17 `f2_Q122`** (SYN-P 7 × {42, 4}); it has no A/A, so the reference contributes 2 runs per task.
- **CL1:** SYN-P only (28 runs).
- **C5b:** E-small − all-big on `rec_sel`, SYN-P 7 minus P1 × {42, 4}; GLMM `rec_sel ~ arm + (1|task) + (1|task:seed)`,
  arms {`f2_Q122`, `mix_esmall`}; separation fallback = exact test stratified by task; two-sided, in Holm. Verdicts:
  Holm-adjusted p < 0.05 with RD < 0 "E-small loses", RD > 0 "E-small better", else "no detectable difference"
  ("equivalent within δ (descriptive)" only if the 90% CI lies in ±δ). Each cell is compared only with the all-big
  reference.
- **`big_tok_ratio`** = the cell run's completion tokens generated on the big model (search charged to the caller) ÷ the
  paired reference run's `tok_compl_total` (E for `mix_1b`; S + T for `mix_esmall`); geometric mean over paired (task,
  seed) cells with a 95% bootstrap CI resampling tasks. **τ** (fixed at G2 from A05's O120 shares): τ = 0.60 if the median
  s_E ≤ 0.50, else s_E + 0.10; disclosed that τ comes from O120 shares while the reference is Q122.
- 1B verdicts, `mix_1b` stop rule (≥ 5 of the first 6 rows capped → stop, Henrik may override within 2 h) and C5b drop
  rule as A19 §6. The `fx_pajM` comparison is dropped by default (re-run at Q122 only if Henrik approves at G4).
  `mix_1b` is all-Qwen under SR4, so the cross-family disclosure lapses. Disclosed: R was frozen on O120 (A02).
- `TODO(Henrik)` (resolved 2026-10-16, Part VIII.1): δ is computed from the A09 oracle vs A05 Full at O120, but C5b's reference is now `f2_Q122`; confirm
  that the O120 δ is used unchanged for the "equivalent within δ" reading and the 1B verdicts.

### V.20 A20 · CAP-3 on Q4 (A20-T1, T2, T5)

Source: A20 §4, §6 (Q4 switch). Exploratory, outside Holm.
- SR4 failed, so A20 moves to Q4: arm `fx_pajM_Q4` (port 8032) against A17 `f2_Q4` at f42 (A17-030…036); O120 cells
  A05 `full` f42 r0 and A12 `fx_pajM`. The within-session pairing at the small rung is lost, and the interaction now
  crosses families (Qwen vs gpt-oss); both disclosed.
- Estimands D_r = `rec_sel`(Full at rung r) − `rec_sel`(minimal at rung r); I = D_Q4 − D_O120. GLMM
  `rec_sel ~ H + R + H:R + (1|task)` effects-coded on the analysis population **SYN-P 7 minus P1: `synp_p2`, `p3`, `p4`,
  `p5`, `p6a`, `p6b`**; Firth with task fixed effects under separation.
- Reading rules in order R5 (floor) → R1 (95% CI of I above 0) → R3 (95% CI of D_Q4 below 0) → R4 (both 90% CIs within
  ±δ and pooled small-rung `rec_sel` ≥ 0.10) → R2 (suggestive) → inconclusive. Never "significant" or a Holm-adjusted p.

### V.21 A21 · M2 (A21-T4)

Source: A21 §4, §6. Descriptive, outside Holm. k grid 1..10; 8 selection rules (`inner_cv`, `iter0`, `last`,
`mean_all`, `oracle`, `patience_2`, `patience_3`, `one_se`); ITT at every k (held-out imputed at `base_no_fe`); iso-cost
rule k*_i = max{k : C_F,i(k) ≤ C_A,i} at whole iterations, "no iso-cost point" below one iteration; bootstrap 2,000
reps, seed 20261016; `flop_proxy` = Σ_roles 2·N_active·(prompt + completion); reading rules R-A to R-E; OQ1–OQ3 closed
as A21 states. There are no pre-freeze A03 points (A03-031…040 are post-freeze rows).

### V.22 A22 · M3

Moved to the short paper by D1 (ED5 applied 2026-10-09). Its §4/§6 rules remain frozen by reference for that paper; it
is not part of the AGENT 2027 analysis.

### V.23 A23 · CAP-6 (A23-T7)

Source: A23 §2.3, §3, §4, §6. Descriptive, outside Holm. Population: SYN-P 7 minus P1 (Part III.2). Axes, energy
apportionment, run selection, paired two-stage cluster bootstrap (seed 20261016, B = 2,000), `p_front` thresholds
(≥ 0.8 on the front; 0.5–0.8 near; < 0.5 dominated) and rules R-A23-1…7 as A23 §6. Panel B uses `fx_pajM_Q4`; `f2_O20`
and the A19 real-4 points are gone; no pre-freeze A03 points.

---

## Part VI — Gates

### VI.1 G1 (Fri 16 Oct)

Before 18:00: this file and the freeze committed and tagged (`agent2027-prereg`, `agent2027-v2.0-freeze`) and pushed;
`v2_defaults.env` as Part I.2; manifests and session files committed; image `rogue_one:agent2027-v2.0` built and its ID
recorded; freeze smoke `S1016s` clean (A05-T3 checklist); rendered-prompt baseline `prompt-baseline-v2.0` stored;
re-tier applied (Part III.3). Then A05 + A07 + A03 launch (`S1016a`–`d`). No arm may launch before G1.

### VI.2 G2 (Mon 19 Oct, hard go/no-go; read-out by 10:00)

Computed by `g2_stats.py` (E11a) before any arm launches and before any A13 score is read. Counts are over A05's 55 runs;
A07 and A03 rows are reported beside them.
- **NO-GO, unconditional:** any inner-CV fallback; any role with zero `tok_compl_*`; missing traces, per-inner-fold
  scores or `run-output` in a run that did not crash.
- **NO-GO, conditional:** ≥ 3 cap hits, ≥ 6 ITT failures, or the same exception in ≥ 2 runs, **and** the cause is a
  fixable defect in Full-v2 (identified from logs or traces by 10:00, fix ≤ 0.5 d, every G1 decision unchanged). If no
  such cause is found by 10:00, the trigger is a finding: GO.
- **On NO-GO:** fix, re-freeze as `agent2027-v2.0r1`, rerun A05 + A07 (+ A03 rows) from the reserve while every arm waits.
  After G2 such a fix becomes a limitation.
- Wording-only outcomes: < 4 informative tasks, ICC > 0.5, Z ≥ 0.85. SYN-P is never retuned after M0. A detector bug (P0
  `rec` > 0) is fixed in analysis, never rerun.
- **Frozen at G2 and written in the G2 memo:** σ_d (and the homogeneity verdict), σ_AA, Z, M_d, the ICC and the
  informative-task list, B_d and role/Tester-stage shares, A12's token-rule branch, A19's τ (dated addendum before A19
  launches), A06's sentinel task, Z/10 and band table, A09's headroom flag, A13's M value, A21's validity checks, the
  A10-T18 option if needed, and recomputed power per contrast.
- **A03 fallback read-out** (Part V.3) and Henrik's revert-or-limitation choice if it fails.
- **Leak checks:** P7 canary (any A05 P7 `rec_any` > 0 → A14 §6 rule); A15 leak rule on A05 P6b runs.
- Box order for Thu 22 – Wed 28 Oct (D12) pinned from M0's measured run lengths.

### VI.3 Later gates

| Gate | Date | Rule |
|---|---|---|
| G0b-1 | Wed 21 Oct | E7, E9d/E9e and the E8b backend merged as v2.3; switch smokes pass; Full-v2 prompts byte-identical to `prompt-baseline-v2.0`. KT3 corpus amendment (Part V.14) committed before A14's first row |
| δ commit | Fri 23 Oct, 18:00 | A09-T15 amendment; before any A12 detector log |
| K-SET | Fri 23 Oct | A16 pilot pass rule; else ED4 (C1a dropped) |
| G3 | Mon 26 Oct | positive controls (A09 planner half; A14 knowledge half) and the title rule |
| G4 | Tue 27 Oct | FX alias rule / RV3; KT3 kill date; RV1 check; A19 SR6 deadline; A06 interim |
| L-LAST | Thu 29 Oct | last new launch; unlaunched contrasts dropped per Part IV.1 |
| G6 | Fri 30 Oct | knee read-out (descriptive) |
| DF | Sat 31 Oct, 24:00 | data freeze |

Post-freeze merges are allowed only as default-off switches that pass the three guards (01 §2.2). Any change that
alters Full-v2 means re-running M0 and every arm.

---

## Part VII — Deviations and disclosures

### VII.1 Disclosures fixed now

1. **Pre-freeze pilots were seen before this commit.** A01 (old image), A02 (R rule), A04 (CAP-0), the A08 CPU screen
   and X9 ran on pre-freeze code and were read before this text. Their results are in Parts II–III. M1 (A03) did not run
   before the freeze; its fallback rows run on v2.0 and are read at G2.
2. **A02 per-task results include a P6a-vs-P6b table** (Part V.15); C4b uses A05 data only.
3. **Knowledge channel.** Web search uses self-hosted SearXNG (Google CSE, Yahoo, Bing at launch), queried live with no
   cache. Results vary over time and with which upstream engines answer. The retrieval audit logs URLs, answering engines
   and calling role. A02's first launch (`S1013a/b`) had no search (Amendment 1).
4. **Change from the plan: search cache off.** 01 §2.1 item 9 and A05/A07/A09–A11/A13/A15–A17/A19/A20 planned
   `ROGUE_ONE_SEARCH_CACHE=shared`. Henrik decided on 2026-10-14 to run with no cache. Consequences: arms do not share
   search results; the A05 A/A samples live search, so σ_AA includes search variation and is no longer a lower bound for
   live search; disclosures in the ablation documents that say a shared cache lowers search variance do not apply; the
   sentinel's search channel reads the search-failure rate (`serper_ok_false_rate` is the existing key name).
5. **Pool mode augment by design**, not by the M1 rule (E6 missed G-E6). M1 criterion (c) untested. E6 bug fixed on
   2026-10-15 before any Full-v2 run (Part III.4).
6. **R frozen at 10 on O120 pilots** (A02, one seed, replace mode, no equivalence claim); small models might prefer
   another R (A17–A20 disclosure).
7. **Sandbox import allow-list** departs from 01 §3.3 E13 "no `__import__`" (Part II "E13"). The sandbox is not a
   security boundary; label-barrier violations are reported as aggregate `scan_leakage.py` counts.
8. **Scientist feedback signal L1** (D2) replaces the in-sample training fit; disjoint means fold-disjoint, not
   row-disjoint (Part II "E14").
9. **Detector amendment** `no_gain_over_raw` (Part III.2), applied before A02 launched.
10. **Balancing.** v2 uses balanced accuracy (`DO_BALANCING=True`); t17/W0 used plain accuracy, so classification σ is
    never pooled with t17, and A03's σ̂ table mixes scales on classification datasets.
11. **Scale of SYN-P runs.** R10 SYN-P runs measured 1.285u (pre-freeze), above the 0.7–1.0u used in the plan.
12. **Topology.** All O120 work at TP2 (D5); A05 uses up to four TP2 servers under the weekend allowance; client load per
    server is constant (`PARALLEL=6`).
13. **LLM-FE spec files** for SYN-P were authored by us (X4).
14. **SR4 fallback.** CAP-2 uses Qwen rungs (big Q122, small Q4) and CAP-3 uses Q4; the A19 τ comes from O120 shares;
    A20's small-rung pairing is cross-session and its interaction crosses model families.

### VII.2 Declared deviations from the master plan

- FX falsifier on the minimal harness (A12); RT3 route 2 (A09); HC1 on 4 datasets (CL3); CAAFE dropped (CL4); CAP-2 on
  SYN-P only (CL1); CAP-1 small-rung real data on 2 datasets (CL2); A22 moved to the short paper (ED5); A01 reruns dropped
  (D1); one-job-at-a-time lifted (D12).

### VII.3 Ablation-document text superseded by this file

The following statements in the ablation documents are out of date at the tag; Part I and the decisions above apply:
- `ROGUE_ONE_SEARCH_CACHE=shared` and shared-cache disclosures: A05 §4.1/§4.5 and T4 (Serper credits check), A07 §4,
  A09 §4.3, A10 §2/§4/§7, A11 §4, A13 §4 and T22, A15 §2/§4/§6 and T4/T21, A16 §4/§7, A17 §4, A19 §4 and T27, A20 §4;
  01 §4.3 and §4.8 rule 5.
- TP4 as M0's topology or "TP4 or TP2 per D5": every document (D5 = TP2).
- "≤ 4 of the 8 H100s" for the A05 weekend (A05 §5.3, A07 §5.3): Henrik's weekend allowance applies.
- A05-T5 "67 rows / 63 under CL3" and a two-session split: 73 rows (55 + 8 + 10) over up to four sessions.
- O20 as CAP-2 small rung, CAP-3 rung, A17 `f2_O20` and its F0 pool, the Thu 22 O20 sessions and `replay_O20` packing:
  A04 session C table, A06 §4.5 and Q2, A17, A18, A19, A20, A21, A23 (SR4).
- CAAFE runs and the CAAFE estimate (A08; CL4). A19 real-4 rows (CL1). A07 rows 009–012 (CL3).
- A03-T17's "M1 ran on pre-freeze code / results seen before the pre-registration"; A03's pre-freeze sessions and
  hand-offs (`W_ref` from A03 replace; A21/A23 pre-freeze A03 panel).
- A10's replace-mode rule; ITT imputation branches "if replace" (augment is frozen).
- A14's corpus hashes, φ_wrong values and replay counts "in the pre-registration" (deferred to the G0b-1 amendment).
- A15's blinding note (Part V.15).
- A05 §2 "`ROGUE_ONE_SCIENTIST_SIGNAL` only if D2 = L1" and 01 freeze item 12 "L1 code CUT" (D2 = L1 adopted).

---

## Part VIII — Open items with dates and amendments

### VIII.1 `TODO(Henrik)` (resolved 2026-10-16, Part VIII.1) before the tag (Fri 16 Oct, before 18:00)

**Resolution (Henrik, 2026-10-16: D7 "A"; everything else "accept all").** Every proposal in the list below is
adopted as written; the inline `TODO(Henrik)` (resolved 2026-10-16, Part VIII.1) markers in this file are resolved by this paragraph.
1. Tags: `agent2027-v2.0` was moved on 2026-10-16 from `55862e5` to `04a2225` to include D7 and the covtype ban (only
   the smoke `S1016s` had used the old tag; its rerun is `S1016t`). `agent2027-v2.0-freeze` points to the same commit as
   `agent2027-v2.0`; `agent2027-prereg` points to the commit that records the freeze (Part VIII.3); relative to `agent2027-v2.0` it changes
   only documents and launcher/comparator scripts (the covtype refusals), with no change under `rogueone/`,
   `rogue_one_main.py`, `requirements.txt`, `setup.py` or the `Dockerfile`.
3. `SCIENTIST_AGENT_MAX_TOKENS=16384` and `ROGUE_ONE_TOOL_OUTPUT_MAX_CHARS=8000` are confirmed as the pins.
4. D7 implemented (`cadb013`): `list_known_attributes_tool` returns every attribute with `(Active)` or `(Pruned)`;
   restores are counted by `append_restored` / `num_restored_attributes`. The tool description changed, so the Full-v2
   prompt baseline is taken from the `S1016t` smoke.
5. D8: Reynolds passed `screen_pairwise.py` (best pairwise |ρ| 0.818 for v/ν, < 0.95), so the fallbacks (b) and (c)
   were never reached; P7 also passed (0.799 for log(corrosion)/anchor_spacing², NMI 0.382; < 0.95 / < 0.9)
   (`docs/ablations/E8a_calibration_report.md`). This is A14's P7 validity check (a).
6. No add-backs (AB1–AB5) are funded.
7. Every "proposed" threshold in the ablation rule sections is adopted as written (blanket adoption).
8. M = 1.0 Δ_std (M_d = σ_d) and the `junglechess` 0.95 threshold on balanced accuracy are adopted.
9. Session layout: up to four gpt-oss-120b TP2 sessions `S1016a`–`S1016d` on ports 8010–8013 over GPU pairs
   {6,7}, {0,1}, {2,3}, {4,5}, as they are free (Henrik allowed all 8 GPUs over the weekend). Rows are split
   pair-aware: every A07 and A03 row sits in the session of its A05 pair; the rest alternate.
10–16. The stated proposals and defaults are adopted (A03 fallback with < 8 pairs: read as proposed; A06 Q1 yes if the
   G2 memo is out, Q3 keep the rule; A09 routine and A15/A09 hashes are committed with their first rows; A11 reads
   `train_metric_saturation` on the L1 scalar as proposed; A18 drops `replay_O20` (O20 failed SR4); A19 uses the O120 δ
   with the Q122 reference as proposed, disclosed).

**Covtype (Henrik, 2026-10-16).** covtype is never run, in any study. Its task folders on CAIR were moved to
`~/QUARANTINE-covtype-NEVER-RUN`; the code refuses it (`rogueone/utils/task_registry.py`, `BANNED_DATASETS`); the
T1.1b control arm's covtype run was stopped on 2026-10-16. No AGENT2027 row has ever included covtype.

Original list:


1. Tags: commit for `agent2027-v2.0-freeze` and `agent2027-prereg`; confirm it is the same tree as `agent2027-v2.0`
   (Part I.1).
2. Client image ID of `rogue_one:agent2027-v2.0` (A05-T7).
3. `SCIENTIST_AGENT_MAX_TOKENS=16384` and `ROGUE_ONE_TOOL_OUTPUT_MAX_CHARS=8000`: confirm the proposed pins (the env file
   still labels them "PIN@G1").
4. D7: `list_known_attributes_tool` does **not** return `Status` at `agent2027-v2.0`; implement before the freeze tag (Full-v2 changes: new prompt baseline and smoke) or record D7 as not applied and disclose (Part I.3).
5. D8: confirm Reynolds passed `screen_pairwise.py`; record why the fallbacks were not needed. A14 P7 validity check (a)
   (pairwise screen) result.
6. "No add-backs funded" (AB1–AB5) under the measured u.
7. Blanket adoption of every "proposed" threshold in the ablation rule sections (Part IV.3), notably: A07 "errors rare"
   < 20 and "too few pairs" < 6 (CL3); A11 thresholds 0.25 / 0.8 and its Mon 26 ED6 date; A13 Off edit 2, the Var(Δ_std)
   = 1.5 reduction and RV2 trigger (i); A14 §3 thresholds; A16 δ floor 0.10 and engineering-collapse threshold; A17 ≥ 3/5
   folds rule and RV1 floor guard; A18 verifier defaults (4× batching gain); A08 "fewer than 2 discriminating tasks" rule;
   A12 infra-rerun rule.
8. M = 1.0 Δ_std (A05 OQ2) and the `junglechess` 0.95 threshold on balanced accuracy (A05 OQ1).
9. A05/A07/A03 session layout: number of TP2 servers (2–4), ports, GPU ids (Part V.5).
10. A03 fallback with fewer than 8 complete pairs at G2 (Part V.3).
11. A06 Q1 (sentinel in `S1019a`) and Q3 (no informative candidate: keep the rule or use `wine` × 42).
12. A09: oracle-string hashes; name of the GLMM fitting routine.
13. A11: `train_metric_saturation` under L1.
14. A15: SHA-256 of the domain term list.
15. A18: keep or drop `replay_O20`.
16. A19: use of the O120 δ with the Q122 reference.

### VIII.2 Scheduled amendments after the tag

| Due | Amendment | Owner task |
|---|---|---|
| Sat 17 Oct | `base-no-fe-ref` floor values for the A03 σ̂ table (applied mechanically) | A05-T6 |
| Mon 19 Oct (G2) | G2 memo constants (Part VI.2); A19 τ; A12 token branch; LLM-FE default sample budget (A08-T4) | A05-T14, A19-T3, A12-T3, A08-T4 |
| Wed 21 Oct (G0b-1) | KT3 corpus hashes, φ_wrong values, TF-IDF replay counts, any vocabulary change | E8b, A14-T1/T4, X9 |
| Fri 23 Oct, 18:00 | δ value (gap and δ) | A09-T15 |
| before first F0 row | F0 prompt text | A17-T2, X7 |
| before a rung's first A17/A18 row | any late A04 verdict | A04-T18 |
| as needed | re-freeze (`agent2027-v2.0r1`), K-SET move, contrast drops, sentinel FLAG actions | per gate |

### VIII.3 Freeze record and amendments

**Freeze record (2026-10-16).** Image `rogue_one:agent2027-v2.0` = `sha256:910b699501ebeb9fb6c35714676be932314fd5cb8336595764d72bc2f00de78f`,
built from `agent2027-v2.0` (`04a2225`) with the same `.env` / `.secret.env` as `pre.1`–`pre.5`. Freeze smoke `S1016t`
(A05-901 `synp_p3`, A05-902 `heart`, one fold each, GPUs 6,7): both OK; `FEATURE_POOL_MODE=augment`,
`EXTRACTOR_ROUNDS=10`, `SANDBOX=on`, `SCIENTIST_SIGNAL=inner_cv_l1`, `SEARCH_CACHE=off`; `raw_coverage` 1.0 every
iteration; `l1_split_disjoint=1` in 10/10 checks; replay check ran, 0 non-reproducing folds; D7 visible
(`list_known_attributes_tool` lists every attribute with its status). The Full-v2 prompt baseline is this smoke's
prompt dump. Earlier smokes: `S1015k` (pre.4, augment rows failed: dtype bug), `S1015l` (pre.5, 4/4 OK),
`S1016s` (v2.0 at `55862e5`, before D7; 2/2 OK).

Amendments:

**Amendment 2 (2026-10-19, G2 decision, Henrik).** G2 is **GO**. Memo: `docs/agent2027/results/G2_memo.md`;
constants: `docs/agent2027/results/g2_constants.json` (commit `d807357`).
- The conditional NO-GO trigger "same exception in ≥ 2 runs" fired: A05-028 (insurance), A05-038 (`synp_p7`) and
  A05-044 (`synp_p6a`) each failed attempt 1 with `XGBoostError: Input data contains inf` (an agent feature that is
  ±inf on one fold partition; the Tester does not map inf to NaN). All three succeeded on the launcher's one retry
  (0 cap hits, 0 ITT failures). Henrik ruled it **a finding, not a fixable defect to re-freeze for**: Full-v2 stays
  frozen at `agent2027-v2.0` with no inf guard; every arm keeps the same one-retry rule, and the crash rate (3 of 76
  A05/A07/A03 attempts) is reported as a limitation. No arm image carries an inf fix.
- Frozen at G2 (values in the memo): σ_d = M_d per dataset (M = 1.0), σ_AA, Z = 0.915 (≥ 0.85: wording only),
  homogeneity pooled with t17, ICC 0.899 (ANOVA fallback; > 0.5: wording only), informative tasks p2, p4, p6b (< 4:
  wording only), **A06 sentinel task `synp_p2`**, A12 branch (i), **A19 τ = 0.60**, A13 M = 1.0, A09 headroom flag not
  raised, A10-T18 option not needed, A21 validity 55/55.
- A03 fallback: 10/10 pairs, mean Δ_std +0.49; **augment stands**. P7 canary 0/15; A15 P6b leak rule: no leak.
- A05-T6 `base-no-fe-ref` floors: memo §6 (late; written 2026-10-19).
- **Detector.** The registered recovery detector did not apply the `no_gain_over_raw` amendment (Part III.2) outside
  the A08 screen mode, so P0 read 4/5. Henrik: amend `recovery_detector.py` so every read-out uses the amended rule,
  and re-check A02 (CPU). With the amendment P0 = 0/5 and every SYN-P count is unchanged.
- Box order Thu 22 – Wed 28 Oct: memo §5 proposal, to be pinned by Henrik.
- Record: `S1019q` (A06-101, Q4 sentinel; no arm) was launched at 07:26Z, before this memo, under the Sun 18 Q4
  sentinel plan; it reads no arm score. Its launcher recorded `GIT_SHA` `b18ac72` (scripts/manifests only beyond
  `04a2225`; `rogueone/`, `rogue_one_main.py` and `v2_defaults.env` identical); image `agent2027-v2.0`.

**Amendment 3 (2026-10-20, A17 small rungs, Henrik: option b).** Every Q4 row run so far hit the 6 h cap:
A04-003 (Stage 0), the sentinels A06-101 (`S1019q`) and A06-102, and the five first-wave A17 rows A17-035, -036,
-040, -044 and -047 (`S1019g`); A04 also saw cap hits on Q0.8 (A04-001) and Q9 (A04-004). Henrik decided to run
**one replicate per small rung** and then free GPU 7: Q4 counts as done with the rows above; Q0.8, Q2 and Q9 each run
one row on `synp_p2` × f42 × r0 (A17-002, A17-020, A17-049; sessions `S1020h`/`i`/`j`), the same cell as the Q4
anchor. The other A17 small-rung F2 rows (S1019h/i/j as planned, and the rest of S1019g) are **not run**; A17 reports
the small rungs as "cannot complete the loop within the 6 h cap" from these replicates, as a disclosed deviation
from the A17 row plan. `S1019g` was stopped at 2026-10-19T22:12Z; its second-wave rows were stopped mid-run and are
not results. The A06 Q4 sentinel (k = 1, 2 both CAP_HIT) cannot read drift; A06's Q4 channel is reported as not
evaluable (limitation). The Q122 rung (A17 `S1024a`) and the A18 replays are unchanged.


**Amendment 4 (2026-10-21, shared servers for A19; Henrik: "set up A19 with shared GPUs").** A19's mixed
rows no longer get a dedicated Q122 TP2 server. Their Q122 roles use A17's Q122 server (`S1024b`,
`rogueone-vllm-q122-s1024b`, port 8041, GPUs 5,6), which holds for `S1022a` after its own 18 rows
(`HOLD_FOR=S1022a`); `S1022a.csv` ports 8040 → 8041. The Q4 (8032) and Q0.8 (8030) servers share one GPU,
each with a `--gpu-memory-utilization` share. Purpose: the mixed sessions left most of their GPUs idle
(one role active at a time). Effect: per-request latency rises while A17 and A19 rows share the Q122 server,
so wall-clock and cost-per-hour channels are not comparable with dedicated-server rows, and 6 h cap hits may
be more frequent; caps stay ITT. Tokens, calls and scores are unaffected. Disclosed as a deviation.

**Note (2026-10-21, A17 Q122 relaunch).** `S1024a` (launched 06:22Z) served `openai/gpt-oss-120b` instead of
Q122: it went through the GPU 5,6 queue script, which then ignored `RUNG=`. It was stopped at 08:26Z; its 6
started rows are not results (W&B tag `INVALID_wrong_model`). The same 18 rows run as `S1024b` on Q122.
`~/ro27_queue.sh` on CAIR is now the RUNG-aware version.

**Amendment 5 (2026-10-21, G0b-1: KT3 corpus freeze, A14; Henrik approved E8b R1–R8 as drafted).**
Source: `docs/agent2027/results/E8b_corpus_DRAFT_for_review.md` (§2, §3, §5). No vocabulary change was made.
- R1: the P7 document dominates TF-IDF retrieval (rank 1 for 726/1,321 = 55% of replayed A02 P7 calls, top 3 for
  1,119/1,321 = 84.7%); accepted and disclosed. A14 therefore tests relay and use of retrieved knowledge, not
  retrieval. A14-T4 criteria: P7 document in top 3 for 332/332 calls naming the domain term (≥ 50%: pass); same
  P7 rank in both corpora for 1,233/1,321 = 93.3% (≥ 90%: pass).
- R2, R3, R5: wording as drafted (quantities in words, no column names; "do not enter the index" sentence kept;
  "highest index is flagged" kept). R4: all 7 distractors kept.
- φ_wrong (A14 §2): |ρ(φ_wrong, φ)| = 0.193 on all rows, max 0.212 over 30 fold training sets (< 0.5);
  headroom XGB(raw + φ_wrong) − XGB(raw) = −0.0018 (E8a container value, R7; < 0.01); φ's own headroom
  0.1045 / 0.0953 (≥ 0.06).
- Corpus hashes (A14-T1 definition, **GNU coreutils text mode**, R6): `kt3_correct`
  `c91e3df836aaa25aa55e71fab72ba693106d83dd1ae6d0c17794692c58d63059`, `kt3_wrong`
  `ad56cf91a9ea74dc8b03dd32c58d8ac22283a75cdaa8a124c4b61dfbea49aa4d`; re-verified 2026-10-21 on the CAIR
  run-time mount (`~/code/RougeOne/scripts/agent2027/corpora/`).
- R8 / A14-T2 (2026-10-21, inside containers): the fixed-string grep for the P7 constant and φ_wrong's constant
  prints nothing in `v2.0`–`v2.4`; `/workspace/scripts/agent2027/corpora` is absent in every image and
  `docs/agent2027` is absent from `v2.3` on. `agent2027-pre.3` has 83 hits of the 6-digit φ_wrong string, all
  inside longer floats in stale `bank`/`adult`/`vehicle` outputs under `/workspace/tasks/` (e.g. `20611.854905…`):
  coincidental digit runs, no P7 material.

**Amendment 6 (2026-10-21, Henrik's decisions on open items).**
- A16 SET-phased: the 23-substitution wording is approved; K-SET contingency switches are built (default off);
  **29 rows** (not 31).
- A18: the strict valid-call reading is kept for every model; the reference bar becomes relative to O120's own
  strict rate (0.89) instead of the absolute 0.95.
- X7 F0 prompt: frozen as drafted (neutral wording; field `pandas_command`; first 5 training rows by `id`; a JSON
  list inside a ```json fence is accepted after stripping the fence).
- A08 LLM-FE: regression tasks run under the alias `crab`; the earlier local change to `~/llmfe/sampler.py` is
  disclosed. The launcher changes (mixed-row global, `G=` topology, `ro27_hold_clients.sh`) are approved.
- A12 token rule: branch (i) as written (the replaced stage's own tokens); for the FX arms it is satisfied by
  construction, which is disclosed. A12-T3's alternative is not adopted.
- A06: the execute-rate channel is dropped (no `fun_*` keys in v2.0); disclosed.
- Power: MDE = planning MDE × √(6/n_inf) × √(1 + 4·ICC), n_inf = 3, ICC = 0.899 (C1a ≈ 0.80; FX ±0.33; RT3 0.79;
  C3 > 1). "Inconclusive" is the expected outcome for effects below these.
- A15: the derived P6 term list used in the G2 audit is frozen as A15-T7: the list produced by
  `scripts/agent2027/analysis/g2_stats.py::p6_terms` (strong terms + P6a column tokens and level strings, generic
  words dropped) at the commit of this amendment; the G2 "no input leak" result stands.
- A02: the absolute 1.285u stays as reported (its W_ref was the planning fallback); the decision is a ratio.

**Amendment 7 (2026-10-22, A19 launch; Henrik: "Run A19").**
- **τ addendum (A19, due before launch).** τ = 0.60, from G2's O120 role shares (median s_E = 0.425 ≤ 0.50; G2 memo
  §2.4). The reference model is Q122, but τ comes from O120 shares (disclosed). τ is also the CAP-2 big-model token
  threshold.
- **Gates.** G0b-1 passed (Amendment 5). SR6 passed (A04-T19, `docs/agent2027/results/A04_session_C_SR6.md`):
  0 misrouted calls in both session-C rows, Scientist W calls in both, W&B config as planned. E and T W routing is
  untested (no such calls occurred), so A19 inherits that gap.
- **Servers (updates Amendment 4).** A17's Q122 server (`S1024b`) was released on 22 Oct 03:50Z, after all 18 A17
  rows hit the cap and before A19 was approved, so A19 gets its own Q122 TP2 server (`S1022a`, port 8041, GPUs 3,4).
  The Q4 (8032, `S1022e`) and Q0.8 (8030, `S1022f`) servers share GPU 7 (`--gpu-memory-utilization` 0.50 / 0.30),
  as in Amendment 4. Both energy sidecars on GPU 7 measure the same device, so per-server energy is not separable.
- **Prior expectation (disclosed).** Every Qwen-served run so far hit the 6 h cap (A04 session C, A17 Q0.8–Q122,
  A20 on Q4). A19's mixed rows are expected to cap as well; caps stay ITT. The A19 §6 `mix_1b` stop rule (stop at
  6 completed `mix_1b` rows) applies as written.
- **Rescheduled (05:50Z).** Henrik needs GPUs 3,4 after `S1021d`. The idle holders `S1022e`/`S1022f` were stopped
  (no A19 row had started); A19 now launches after `S1023a` ends, with Q122 on GPUs 1,2 (port 8041) and the Q4/Q0.8
  holders `S1022k`/`S1022l` on GPU 7.

**Amendment 8 (2026-10-22, Henrik's decisions).**
- **Q122 12 h probe (not pre-registered; descriptive).** No Qwen-served run has finished inside the 6 h cap (A17
  Q0.8–Q122: 21/21 capped incl. all 18 `f2_Q122` rows; A20 8/8; A04 session C 2/2), so C5a/C5b have no completed
  Qwen rows. One extra row, **A17-900** (`f2_Q122_cap12`: A17-071's cell, `synp_p2` f42 r0, Q122, image `v2.0`),
  runs with a **12 h cap** in its own session `S1022m` (Q122 TP2, GPUs 5,6, after `S1022h`). It answers only "does
  Q122 finish the loop at all with twice the budget"; it never enters C5a/C5b or any pre-registered test. C5a/C5b
  are reported as no data, with cap hits, tokens and iterations reached as descriptive evidence.
- **A16.** The K-SET contingency switches are not built (K-SET passed; A16 stays on the validated `v2.4` image).
  The K-SET GO stands although the pilot rows ran on Mon 19 Oct, before the Wed 21 sign-off, on the identical
  approved wording (disclosed). Re-pilot rows A16-029/030 are marked cancelled.
- **O120 extras.** A18's `replay_O120` and A17's X7 F0 pools run as extra clients of A16 Core-10's O120 server
  (`S1022h`); this adds load to that server while they run (disclosed with A16's wall-clock figures).

**Amendment 9 (2026-10-22, X7 / A17 F0 pools; Henrik's decisions).** F0 generator `scripts/agent2027/f0_generate.py`
and launcher `f0_clients.sh` at `d1d7ffb` (70 pool rows in `manifests/agent2027/A17_F0.csv`, kept out of
`A17_CAP-1.csv`).
- **Servers.** O120 pools (A17-164…177) ride A16 Core-10's server (`S1022h`); Q122 (A17-150…163), Q4 (A17-122…135)
  and Q0.8 (A17-101…114) pools ride A19's servers (`S1022a`, `S1022k`, `S1022l`) as extra clients. Q2 (A17-115…121)
  and Q9 (A17-136…142) pools get one short TP1 session each on GPU 7 after A19, before ED3 (Wed 28). Shared servers
  affect wall clock only.
- **A17-T18 metric split.** The generator reports `f0_parse_rate` (a JSON list with `name` and `pandas_command`;
  a ```json fence is stripped); the scorer (A17-T8) reports `f0_exec_rate` and the combined parse-and-execute rate
  that A17-T18 defines. The generator never executes model output.
- **Over-limit answers** (> 10 features) are kept whole and flagged `over_limit`; the scorer uses the **first 10 in
  the model's order** and reports the over-limit rate.
- **Artifacts** are named `f0-pool-<rung>-<ds>-f<seed>` (type `f0-pool`), one per pool. **Fold guard:** folds are
  checked against an independent `deterministic_kfold_indices` call (A05 outputs do not log per-id fold membership).

**Amendment 10 (2026-10-23, why Qwen runs cap; A19 stopped; A04/A20 longer-cap reruns; Henrik's decisions).**
- **Diagnosis (from `iteration_log.jsonl` of the capped runs; descriptive).** A full run is 50 iterations (5 folds ×
  10). Per iteration, the O120 Extractor makes ≈ 28 calls at ≈ 12k prompt tokens (A12-033: 156 s/iteration); the
  Qwen Extractor makes ≈ 300–450 calls at ≈ 50k prompt tokens (≈ 50× the prompt tokens), almost all repeated
  `generic_pandas_tool` exploration calls (finish_reason `tool_calls` in 8,656 of 8,662 sampled A20-001 calls), so
  each round runs to its turn limit. Progress at the cap: A20 (Q4, 4 rows per server) 11–13/50 iterations
  (1,600–1,850 s each; ≈ 23–26 h projected); A20 smoke alone 615 s/iteration (≈ 8.5 h projected); A17-900 (Q122,
  alone, 12 h) 26/50; A04 session C 8/10 and 7/10 (1 fold; ≈ 6.3 h and 8 h projected); A19 4/50 (≈ 70 h projected).
- **A19 stopped early.** Stopped 2026-10-23 07:29Z (`S1022a`, holders `S1022k`/`S1022l`). Result: the 12 completed
  rows (all CAP_HIT at 6 h) plus the projection above; the 6 rows in flight were stopped mid-run and are not results;
  10 rows were never launched. Disclosed deviation from A19's row plan.
- **A20 rerun with a 12 h cap (sensitivity).** Rows A20-901…907 (= A20-001…007, arm `fx_pajM_Q4_cap12`), one row per
  session and per GPU (`S1023e`…`S1023k`, Q4 TP1, PARALLEL=1). The 6 h rows A20-001…007 stay the pre-registered result.
- **A04 session C rerun with a 10 h cap (sensitivity).** Rows A04-907/908 (= A04-007/008, `_cap10`), main `S1023m`
  (Q122 TP2, GPUs 1,2, port 8043) with holders `S1023n` (Q4) and `S1023o` (Q0.8) sharing GPU 7. The remaining Q0.8 F0
  pools resume on `S1023o` (the `S1022l` host was stopped with A19).
- **Re-placed 07:50Z (Henrik: A04 first on GPUs 5+6).** PlanBench took GPUs 1,2 after they were released, so the
  first A04 holders (`S1023n`/`S1023o`) were stopped before the main session started (no A04 row ran). Order on CAIR
  (`~/ro27_plan_1023.sh`): GPU 7 runs the remaining Q0.8 F0 pools (`S1023u`, own server), then A20-903 (`S1023g`);
  when A20-901/902/903 have ended, A04-907/908 run as `S1023m` (Q122 TP2 on GPUs 5,6) with holders `S1023s`/`S1023t`
  on GPU 7; then A20-904…907 (`S1023h`/`i`/`j`/`k`) on GPUs 5–7. **08:30Z:** GPUs 3,4 came free (the routine's
  wave ended), so A04's Q122 moved there (`~/ro27_plan_1023b.sh`): `S1023m` on GPUs 3,4 as soon as `S1023u` ends,
  holders on GPU 7; A20-903…907 queue on GPUs 5,6 (after A20-902/901) and GPU 7 (after A04).

**Amendment 11 (2026-10-23, power replicates; Henrik: option 1). Not pre-registered; sensitivity only.** G2 left 3
informative SYN-P tasks (`synp_p2`, `synp_p4`, `synp_p6b`) and an ICC of 0.899, so the scaled MDEs are large (C1a ≈ 0.80,
FX ±0.33). To narrow the intervals, 45 extra O120 rows run on exactly those three tasks, each paired with a Full row
of the same fold seed and replicate (`llm_seed` = fold seed + 1000·replicate, as in A05):
- Full (A05-801…806): f4 r1 and f42 r2 per task (more control replicates);
- A09 oracle / misdirected (A09-801…806): f4 r0 (pairs with A05 f4 r0);
- A10 restarts, A11 L0, A16 SET (A10/A11/A16-801…803): f42 r1 (pair with A05 f42 r1);
- A12, all 8 FX arms (A12-801…824): f4 r0 (pair with A05 f4 r0).
Rows keep their ablation's image. Sessions (one image each, with sentinels A06-006…011): `S1024c`/`S1024d` (A12, v2.3),
`S1024e` (A05, v2.0), `S1024f` (A09, v2.1), `S1024g` (A10+A11, v2.2), `S1024h` (A16, v2.4), queued on GPU pairs 3,4 /
1,2 / 5,6 after the A04/A20 reruns (about Fri 21:00Z – Sat midday). **Analysis rule:** every pre-registered test is
computed first on the pre-registered rows only; a second, clearly labelled sensitivity version adds these rows. No
pre-registered result is replaced. Recovery on these rows is computed only after δ is committed (Fri 18:00).

**Amendment 12 (2026-10-24 ≈ 20:30Z): FX margin δ committed late (A09-T15).** δ was due Fri 23 Oct, 18:00 Oslo,
and was committed about 26 h late, because CAIR could not be reached from the analysis laptop from Fri afternoon until
Sat ≈ 20:10Z. **Blinding held:** no detector had run on any A12 row before this commit (no A12 analysis output exists
on CAIR; the first A12 detector run follows this commit). All six population oracle rows were terminal long before
the deadline (S1019a, Mon 19), so the "not terminal" fallback (δ = 0.10) does not apply.
- Population: SYN-P 7 minus P1 (`synp_p2`, `p3`, `p4`, `p5`, `p6a`, `p6b`), amended detector, ITT zeros (none needed).
- Oracle (A09-002…007): 30 / 30 episodes, `rec_sel` = 1.000. Full f42 r0 + r1 (A05-032…037, 048…053): 20 / 60,
  `rec_sel` = 0.333 (identical to `G2_recovery_by_task.csv`).
- gap = 0.667; **δ = max(0.25 × 0.667, 0.10) = 0.167.** Per-episode scores: `docs/agent2027/results/A09_delta_episodes.csv`;
  write-up: `docs/agent2027/results/A09_delta.md`.
- Power note (A12 §2): δ = 0.167 lies above the floor but below the planned CI half-width (0.10–0.12 at the floor
  assumption; ±0.33 scaled MDE after G2), so "inconclusive" stays the expected C2a/C2b outcome.
- Run status at the same time (infra only): the six A09 power replicates (A09-801…806, `S1024f`) failed setup because
  their RT3 focus files had not been built; they were built with the pre-registered builder (`rt3_focus.py build
  --verify`, PASS; oracle hashes identical to A09-002/004/007's) and the rows relaunched as `S1024r` (GPUs 5,6,
  2026-10-24 20:14Z). A20-901…907 (12 h cap): 7/7 OK. A04-907/908 (10 h cap): both CAP_HIT again.

**Amendment 13 (2026-10-24 ≈ 20:45Z, more power replicates; Henrik: "We still have 5 GPUs idle, make sure they are
doing something useful over the night"). Not pre-registered; sensitivity only, same analysis rule as Amendment 11.**
70 more O120 rows on the three informative tasks (`synp_p2`, `synp_p4`, `synp_p6b`), each paired with an existing Full
row of the same fold seed and replicate (`llm_seed` = fold seed + 1000·replicate):
- A12, all 8 FX arms: A12-825…848 at f42 r1 (pairs A05-048/050/053) and A12-849…872 at f4 r1 (pairs A05-801/803/805);
- A16 SET, A10 restarts, A11 L0: rows 804…806 at f4 r1 (pairs A05-801/803/805) and 807…809 at f42 r2 (pairs
  A05-802/804/806);
- sentinels A06-013…017 (A06-012 already existed).
Sessions: `S1024s`/`S1024t`/`S1024w`/`S1024x` (A12, v2.3, 12 rows each), `S1024u` (A16, v2.4), `S1024v` (A10 + A11,
v2.2). Queues: GPUs 1,2 `S1024s` → `S1024u` → `S1024x`; GPUs 3,4 `S1024t` → `S1024v`; GPUs 5,6 `S1024w` after `S1024r`.
GPU 7 stays free (one H100 cannot host the O120 server at the session context length). Expected end Sun 25 ≈ 12:00Z.

**Amendment 14 (2026-10-25 ≈ 13:10Z, idle GPUs; Henrik: "GPUs 1-7 are still idle, i would like to see us utilize
them"). Not pre-registered; sensitivity only (Amendment 11 analysis rule).** Amendment 13 finished 12:30Z (70 rows
OK, except A12-827: an ITT failure from the same XGBoost `inf` crash as A12-035). 36 more rows:
- A12, all 8 FX arms, at f42 r2 on `synp_p2`/`p4`/`p6b` (A12-873…896; pairs Full A05-802/804/806), sessions `S1025a`/`S1025b`
  (v2.3, GPUs 1,2 and 3,4, sentinels A06-018/019);
- A05 Full anchors at f4 r2 and f42 r3 on the same tasks (A05-807…812), session `S1025c` (v2.0, GPUs 5,6, sentinel A06-020);
- A20 Q4 minimal harness, 12 h cap, at f4 r0 on the same tasks (A20-908…910; pairs O120 `fx_pajM` f4 r0, A12-805/813/821),
  session `S1025d` (Q4 TP1 on GPU 7, PARALLEL=3, no sentinel, as `S1023e`…`k`).
Expected end: O120 sessions ≈ 18:00–19:00Z; A20 ≈ Mon 01:00Z.
- *Addendum 17:50Z:* GPUs 1,2 freed early (`S1025a` done), so A16 SET-phased runs at f4 r2 and f42 r3 (A16-810…815,
  paired with the new Full anchors A05-807…812) as `S1025e` (v2.4, sentinel A06-021). This tightens C1a (sensitivity only).

**Amendment 15 (2026-10-25 ≈ 21:00Z, RV3 launched before G4; Henrik: "If there are any runs we plan to do, then we
should not wait. That is just a waste of GPU resources").** The A12 RV3 P− semi-foldover (A12 §4, A12-T23) is
generated and launched now instead of after the G4 trigger (Tue 27), because GPUs 3,4 are idle. Rows A12-066…093
exactly as A12 §4 (4 cells `fx_paJM`, `fx_pAjM`, `fx_pAJm`, `fx_pajm` × SYN-P 7, f42 r0, v2.3, numbered by task),
task-blocked as the fraction: `S1025f` = p3, p5, p2, p6a (GPUs 3,4, sentinel A06-022); `S1025g` = p6b, p4, p1 (GPUs
1,2 after `S1025e`, sentinel A06-023). Integrity rule: the G4 trigger (Wald p of PA, PJ or PM < 0.05 in the
pre-registered T16 GLMM on the fraction rows, plus the reserve check) is evaluated exactly as written and **no detector
runs on A12-066…093 until the G4 memo is committed**. If G4 triggers RV3, the 12-cell refit of A12-T23 is the analysis
of record (its only deviation is the earlier launch, disclosed). If not, the pairs stay "unattributable" in the
confirmatory read-out and the 12-cell refit is reported as exploratory.

**Amendment 16 (2026-10-25 ≈ 21:30Z, real-data check; Henrik approved "follow up with 2"). Exploratory, outside Holm;
not pre-registered.** Question: do the SYN-P role findings (planner needed; assessment and pruning replaceable) carry
over to real data? Arms on Core-10 (no covtype), v2.3, 6 h cap, each row cloned from its A05 Full pair (same dataset,
fold seed, `llm_seed`, replicate 0, `env_overrides` = the arm's factors only): `fx_pAJM` (P−), `fx_PaJM` (A−),
`fx_PAjM` (J−) and `fx_PajM` (the cheap configuration: planner kept, assessment and pruning replaced; not a cell of the
fraction). Rows A12-701…740 at f42 r0 (pairs A05 Full f42 r0) and A12-741…780 at f4 r0 (pairs A05 Full f4 r0).
Sessions `S1025h`/`S1025i` (f42) and `S1025j`/`S1025k` (f4), 5 datasets each, sentinels A06-024…027; queued on GPUs 5,6
(`S1025h` → `S1025j`, after `S1025c`) and 3,4 (`S1025i` → `S1025k`, after `S1025f`).
Analysis (fixed now, before any arm row ends): outcome Δ = arm − Full on `heldout_score`, sign-flipped for nRMSE,
standardised by s_d (A05 definitions); per arm, mean standardised Δ over the 20 (dataset, seed) pairs with a t 90% CI
and a dataset-cluster bootstrap; reading "non-inferior on real data" iff the 90% CI lower bound > −M with M = 1.0 (the
A05 `ni_margin` in s_d units); "worse" iff the 95% CI upper bound < 0; else inconclusive. Tokens: ratio of
`tok_compl_total` arm/Full with a 90% CI (log scale, dataset FE). `success` and `func_success` reported descriptively.
Version confound (arms v2.3, Full v2.0) is disclosed, as for the A12 falsifier.

**Amendment 17 (2026-10-26 ≈ 11:45Z, A17 Qwen ladder relaunch; Henrik 2026-10-25: "restart the Qwen ladder but
investigate why they time out first. If we need to do any updates to the tooling or the program, then do this
before").** All 33 first-wave A17 runs ended CAP_HIT/FAILED. Diagnosis (smoke rows A17-951…968, never analysed):
(i) vLLM 0.22's `qwen3_coder` tool parser does not follow `$ref`, so object arguments arrived as JSON strings;
(ii) the Tester/Extractor exec normaliser turned every literal `\n` into a newline, splitting valid string literals,
and the Scientist's pandas tool rejected bare expressions; (iii) the small models repeated one identical tool call
up to 1,048 times, which the harness retry limits (Scientist 100×200) amplified to context-length failures. Three
switches, all off by default so O120 requests and behaviour are unchanged (unit-tested), are turned on for every
Qwen row: `ROGUE_ONE_INLINE_TOOL_SCHEMA_REFS` (inline `$defs`), `ROGUE_ONE_EXEC_COMPAT` (unescape only code that
does not compile as sent; a trailing bare expression sets `result`; printed output counts as the Scientist's result)
and `ROGUE_ONE_LOOP_GUARD=3` (after 3 identical consecutive tool calls in one round, the next request is sent without
tools and with a note to answer). Commits `7a6180a`, `12bad18`, `ab57b7b`, `00b1991`, `d6fea39`; image
`rogue_one:agent2027-v2.0-qfix5` (v2.0 plus 5 patched files). Smoke 5 at Q4 (1 fold × 3 iterations): 26–36 min, at
most one fallback per run. Rows A17-401…468 re-run the 68 first-wave rows (A17-001…) one-to-one (same dataset, fold seed,
`llm_seed`, rung), 12 h cap; sessions `S1027a`…`h`, one rung each (Q4 ×2, Q122 ×2 at TP2, Q0.8 ×2, Q9, Q2). They
replace the first-wave rows in every A17 analysis (C5a; A19's `f2_Q122` reference); the first wave is reported as a
descriptive ITT failure with this diagnosis. Disclosure: Qwen rungs run with a harness that differs from O120's by
these three switches, so the ladder compares models under a Qwen-compatible harness, not under the identical code
path; the loop-guard fire count is reported per rung (`loop_guard` counter).
*Addendum 12:00Z:* A17 Q122 sessions `S1027c`/`d` run with PARALLEL=5 (was 4; 2 waves instead of 3).

**Amendment 18 (2026-10-26 ≈ 12:00Z, A19 and A20 relaunch; Henrik: "I approve A19 and A20").** Same diagnosis and
the same three switches as Amendment 17, image `rogue_one:agent2027-v2.3-qfix5` (A19 and A20 ran on v2.3), 12 h cap.
- **A19:** rows A19-401…428 re-run all 28 A19 rows one-to-one (both arms × SYN-P 7 × seeds {42, 4}; the first wave
  ran 12 of them, all CAP_HIT, and was stopped 23 Oct). Sessions by arm: `S1027i` = `mix_esmall` (S, T on Q122 TP2
  port 8041, GPUs 1,2; E on the Q4 holder `S1027k`, port 8044) after A17 `S1027c`; `S1027j` = `mix_1b` (S, T on the
  Q0.8 holder `S1027l`, port 8045; E on Q122 TP2 port 8042, GPUs 3,4) after A17 `S1027d`. Both holders share GPU 6
  (0.50 / 0.30 memory) after A17 `S1027f`. Reference = the new A17 `f2_Q122` rows (Amendment 17).
- **A20:** rows A20-401…410 re-run all ten 12 h-cap rows (A20-901…910: f42 × 7, f4 × 3), not only the three CAP_HIT
  f4 rows: A20-901…907 finished on the unfixed harness with the Extractor fallback in 49/50 iterations, so they are
  not comparable with the fixed A17 `f2_Q4` reference. They run as inner rows of the Q4 holder `S1027k` before it
  holds. A20-901…910 are reported descriptively as the first wave.
Launcher `~/ro27_launch_A19r.sh` (pattern of the first A19 launch). Expected end: A20 ≈ Wed 28; A19 ≈ Thu 29.

**Decisions of 2026-10-26 ≈ 19:45Z (Henrik, /unblock; no amendment to analyses).**
- *A17 retries:* the runner's one solo retry per failed row is kept for the relaunched A17 rows, including the
  XGBoost `inf` crash (`rc0-no-fold_aggregation_summary`; at 19:31Z 6/6 finished Q4/Q122 rows had it on attempt 1,
  Q0.8 7/7 OK). Henrik accepts that the queued sessions behind them (Q2 `S1027h`, Q9 `S1027g`, A19 `S1027i/j`) slip
  by about a day and may launch after L-LAST (Thu 29); they still end before DF only if the slip stays under ~2 days.
- *RV1 (HQ #314):* evaluated on the relaunched Q0.8/Q2/Q4 rows as each rung ends; if it fires, the 14 RV1 rows
  (fallbacks off) are launched by Thu 29, GPU permitting. `ROGUE_ONE_FALLBACKS=off` is registered but not yet
  implemented; it is built now, flag-gated, default on (Full-v2 unchanged).
- *X6 / C1b:* LLM-FE best-3 recovery is primary; the notebook's own selection is reported as a secondary row.

**Amendment 19 (2026-10-27 ≈ 22:00Z, inf-guard sensitivity re-runs; Henrik: "Fix + rerun as sensitivity").**
Diagnosis: the relaunched Qwen rows that end `rc0-no-fold_aggregation_summary` die on `XGBoostError: Input data
contains inf`. The Tester's scoring fits clip every fold matrix (`nan_to_num`, ±1e6), but the assessment fit and the
pruning tools received the raw matrix, so one infinite replayed feature ended the run. Fix `ROGUE_ONE_INF_GUARD`
(commit `ae1be1a`): the same clipping on the Tester's fold matrices, counted as instrumentation `inf_guard`; off by
default, so Full-v2/O120 behaviour is unchanged. Images `rogue_one:agent2027-v2.{0,3}-qfix6` = qfix5 + this patch
only (`scripts/agent2027/patch_inf_guard.py`; the other qfix5 files are byte-identical).
- **Rows (exploratory sensitivity, outside Holm; the as-run rows stay the rows of record, ITT zeros included):**
  every row whose final attempt had ended on the inf crash at 21:30Z, re-run one-to-one with the same arm, dataset,
  seeds, env plus `ROGUE_ONE_INF_GUARD=1`, 12 h cap: A17-501…505 (Q4: A17-434, 436, 438, 431, 432), A17-506/507
  (Q122: A17-451, 462), A20-501…509 (A20-401…408, 410), A19-501/502 (A19-402, 405). Not re-run: CAP_HIT rows and
  the rc=139 segfault A17-458 (not the inf crash). `scripts/agent2027/gen_amend19.py`.
- **Where:** no new GPU. `S1027m` (14 rows, PARALLEL 5) are clients of the Q4 holder `S1027k` (port 8044, GPU 6);
  `S1027n` (4 rows, PARALLEL 4) of the Q122 server of `S1027j` (port 8042; the A19 rows keep their Q0.8 roles on
  8045). Started 21:56Z. The host servers stay up until their A19 main sessions end; a row cut off by a teardown is
  reported as infra-lost, not as a result.
- **Reading:** per rung, the as-run ITT result and the inf-guard result side by side, with the inf-crash rate.
  Because Full (A05) ran without the guard, the guarded rows are never compared with Full in a confirmatory test
  (RV1 is evaluated on the rows of record).
- *Addendum (2026-10-27 ≈ 22:30Z, Henrik /unblock):* (i) rows that end on the inf crash after 21:30Z are added to
  Amendment 19 in the same way (next free A17-5xx / A19-5xx / A20-5xx id, same host servers) at each check, as long
  as they launch by L-LAST (Thu 29). (ii) A row that has an inf-guard re-run gets no further unfixed retry: it was
  removed from its session's plan snapshot (`_plan/rows.usv`, backup `.bak_amend19`), so its pending retry logs
  `status=FAILED rc=setup reason=row-not-in-plan` without a container. Removed: A20-402…408, 410 (S1027k), A17-432
  (S1027b), A19-402, 405 (S1027j). Retries already running (A20-401, A17-431) were left to finish. The row of record
  is the first attempt (ITT zero).
- *Amendment 19b (2026-10-28 ≈ 21:45Z, Henrik: "GPU 7 + Q9 on GPU 5").* Of the 21:56Z re-runs, A20-501 ended OK
  (its row of record A20-401 had crashed), but all five A17 Q4 re-runs (S1027m) and the three Q122 re-runs (S1027n)
  hit the 12 h cap at 09:58Z: as clients of shared, overloaded servers they were too slow. Changes:
  (i) the five Q4 rows are re-run again on a dedicated Q4 server, `S1028a` (GPU 7, idle since Q2 ended 06:32Z,
  port 8046, PARALLEL 5): A17-508…512 re-run A17-434, 436, 438, 431, 432; (ii) all eight Q9 rows of record that
  crashed on inf (S1027g: A17-442…449; 0/9 Q9 rows OK) get re-runs A17-513…520 on a dedicated Q9 server, `S1028b`
  (GPU 5, port 8047, PARALLEL 8 so every row launches before L-LAST), queued behind S1027g; S1027g's pending unfixed
  retries A17-445…449 were removed from its plan snapshot (the running A17-444 retry was left). Both with an 18 h
  row cap (sensitivity rows only; rows of record keep 12 h). A17-450 (CAP_HIT) is not re-run. (iii) No further
  Q122/A19 sensitivity re-runs: no 2-GPU slot is free before L-LAST; those rungs are reported as-run, plus whatever
  S1027n finishes. `scripts/agent2027/gen_amend19b.py`.
- *Amendment 19c (2026-10-29 ≈ 14:05Z, Henrik /unblock: "Dedicated Q122 on GPU 1–2"; supersedes 19b (iii)).* GPUs 1,2
  had been idle since ≈ 00:20Z. Six A19 rows of record ended on the inf crash after 21:30Z on 10-27 and had no
  re-run (the XGBoost message is line-wrapped in `container.log`, so it is matched as `Input data contains` +
  `` `inf` ``): A19-409, 410, 411, 413, 414 (S1027j, mix_1b) and A19-421 (S1027i, mix_esmall). Re-runs
  A19-503…508, one-to-one, same arm, dataset, seeds and env plus `ROGUE_ONE_INF_GUARD=1`, image
  `agent2027-v2.3-qfix6`, 18 h row cap as 19b. Servers: main `S1029a` Q122 TP2 on GPUs 1,2 (port 8048,
  PARALLEL 6); holders `S1029b` Q4 (port 8049, 0.50) and `S1029c` Q0.8 (port 8050, 0.30) on the first of GPU 7
  (free when S1028a ends ≈ 16:00Z) or GPU 6 to empty, so the mixed rows' small roles get the same per-rung
  servers as in S1027k/S1027l. Nothing launches after 23:30Z (L-LAST). The pending unfixed retry A19-414 was
  removed from S1027j's plan snapshot (backup `rows.usv.bak_amend19c`; A19-409…411 already had both attempts);
  the running A19-413 retry is left to finish, as A20-401 and A17-431 were; the as-run rows stay the rows of
  record. `scripts/agent2027/gen_amend19c.py`, `scripts/agent2027/ro27_launch_amend19c.sh`.

**Decisions of 2026-10-29 ≈ 00:30Z (Henrik), F0 scoring (E11d, `scripts/agent2027/f0_score.py`, commit `7fac6b7`), fixed
before any F0 pool is scored beyond the two end-to-end checks (A17-122 Q4, A17-164 O120, `synp_p1`):**
- *Validation frame of record:* `id_only`, as A17-T8 states (`--validate-frame id_only`, the default). The augment
  frame (id + E6 seed columns, as F2 ran) is a labelled sensitivity (`--validate-frame augment`); the paper notes
  that id-only disadvantages F0 against F2 (Q4 end-to-end check: 34% vs 45–57% of draws execute).
- *Exec compatibility:* Qwen rungs' F0 pools are scored with `ROGUE_ONE_EXEC_COMPAT=1`, as their F2 rows ran
  (Amendment 17); O120 / gpt-oss pools with it off (frozen Full-v2 default). The setting is logged per run in
  `run_info.json`.

**Decisions of 2026-10-29 ≈ 01:00Z (Henrik), A18 and novelty metrics (no amendment to analyses).**
- *A18 / HQ #315:* the pre-registered scoring stays the result of record (floors S = Q4, E = > Q122, T = ≤ Q0.8;
  D2 = Hoped). The JSON-string-args decoding (`rp1_score --decode-json-string-args`, `rp1_verdicts --scored-315`,
  commit `6f81cf2`) is reported side by side as a sensitivity (E executes 0.01–0.02 → 0.52–0.92 across Qwen rungs;
  E floor Q4, stability 0.44; D2 "uninformative"). The text states that the primary E failure is a transport/format
  failure (stringified args, the qwen3_coder issue of Amendment 17), not a code-capability failure.
- *Novelty metrics* (A18 novelty, `process_metrics` focus novelty): no embedding model is pre-registered; both use
  the TF-IDF cosine stand-in, labelled in the paper as a lexical measure.

**Amendments 19d, 19e and 19f (decided 2026-10-30; written up 2026-11-02 from the launcher scripts and manifests on CAIR,
where they were recorded at the time).** All rows are exploratory sensitivity re-runs or retries under the Amendment 19
reading rule: the as-run rows stay the rows of record, ITT zeros included, and no confirmatory test or Holm verdict uses
them.
- *19d (Henrik 2026-10-30: "queue the 7 cap-hit Q9 rows on GPUs 1,2,7").* The S1028b inf-guard re-runs of the Q9 rows
  had all hit the 18 h cap at PARALLEL 8 on one server except A17-518 (15–24 of 50 iterations reached). Re-runs of
  A17-513…517, 519, 520 as A17-521…527, same image (`agent2027-v2.0-qfix6`), env and 18 h cap, but at most 3 rows per
  dedicated Q9 TP1 server: `S1030a` (GPU 1, port 8051, A17-521…523), `S1030b` (GPU 2, 8052, A17-524/525), `S1030c`
  (GPU 7, 8053, A17-526/527). Launcher `~/ro27_launch_amend19d.sh`.
- *19e (Henrik 2026-10-30, chain before the VPN cut).* Retries of rows that had failed and were never re-run:
  A12-781 and A12-782 (retries of A12-035 and A12-827; `S1030d`, O120 TP2, 6 h cap) and A17-528 (retry of A17-458,
  the rc=139 segfault; `S1030e`, Q122 TP2, 12 h cap). Same image and env as the original rows. Launcher
  `~/ro27_chain_1030.sh`.
- *19f (Henrik 2026-10-30).* Retry of A17-521 (rc=139 segfault in `S1030a` after 3 h) as A17-529, `S1031a`, own Q9
  server (port 8056), same image, env and 18 h cap. Launcher `~/ro27_launch_amend19f.sh`.
- *Outcomes (final read-out 2026-11-02, `docs/agent2027/results/final_20261102/`).* OK: A17-523, 526, 528, 529,
  A12-781, 782; CAP_HIT: A17-522, 524, 525, 527; A17-521 FAILED then CAP_HIT. With every re-run replacing its row of
  record (latest re-run wins), Q9 `func_success` is 4/7 and the C5a trend p = 3.9e-6 (rows of record: Q9 0/7,
  p = 0.0013); the C5b sensitivity with the A19 re-runs is 12/60 vs 20/60, exact p = 0.017. The A12 retries recover
  3/5 and 2/5; the RV3 analysis of record is unchanged. The first final read-out (same day) missed these rows because
  it used a manifest copy predating them; the read-out of record includes them.
