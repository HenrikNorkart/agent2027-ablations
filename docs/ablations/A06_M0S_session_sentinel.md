# A06 — M0-S · Session sentinel for drift

| Field | Value |
|---|---|
| **ID** | A06 · M0-S · arms `sentinel` (O120) and `sentinel_Q4` (small-rung sessions) |
| **Claim / Holm contrast** | None. This is a validity guard outside Holm. Every post-M0 contrast against A05 relies on it (C1a, C1b, C3, C4a, C5b, and the A12 Full anchor). It is also **guard 3** for the post-freeze images v2.1–v2.4. |
| **Tier** | T1-minimum. The only allowed reduction is cut lever CL5. |
| **Owner** | Henrik |
| **Start / finish** | First Q4 sentinel **Sun 18 Oct (night)**; first O120 sentinel **Tue 20 Oct** (Mon 19 Oct instead if A07-T14 moves HC1 into `S1019a` or Q1 = yes; both are inside the 6–12 runs). The last one runs in the last session before DF (Sat 31 Oct, 24:00). Interim read at G4 (Tue 27 Oct); final test Sun 1 – Tue 3 Nov. |
| **Budget** | **9–17 runs**: 6–12 O120 at 0.7–1.0u plus 3–5 Q4 at 0.5–1.0u, giving **5.7–17u** (identical to the T1-minimum breakdown and 01 §4.9). At 1.4u per box-hour that is 4.1–12.1 box-hours, or about 16–49 H100-hours. A06 has **no session of its own**: each sentinel takes one of the host session's `PARALLEL` client slots (concurrency is not raised). Infrastructure re-queues and FLAG repeats (§4.9, §6) are paid from the 01 §4.9 rerun reserve, not from these 9–17. |
| **Models** | O120 = `openai/gpt-oss-120b` (MXFP4; TP4, or TP2 under D5; port 8010, and 8011 for a second server). Q4 = `Qwen/Qwen3.5-4B` (BF16, TP1, port 8032, Qwen set; fallback `Qwen/Qwen3-4B` if A04 moves to the Qwen3 ladder). All roles and W are on one model. |
| **Status** | Not started |
| **Depends on** | A05 (seed-42 SYN-P runs r0/r1, the 8 SYN-P A/A pairs, and the `a05-handoff` artifact from A05-T16); A08 CPU screen (`base_no_fe` on SYN-P at seed 42, for s_task); E1, E3, E4, E8a; E11a (G2 names the task and band); E11b (`sentinel_drift.py`, pulled forward as A06-T8); X10; A04 session A (Q4 passes Stage 0); A17 (hosts the small-rung sessions and supplies the Q4 anchor row); D4, D5 |
| **Feeds into** | Guard 3 for v2.1–v2.4. The validity of every cross-session comparison with A05 (A07 fallback, A08 GPU part, A09–A16, A19). Session stability of A17's small-rung nights (C5a). The cost axes of A21 and A23. Paper appendix and threats to validity. |

*Every hoped result below is a hypothesis, not a finding. Names follow 01 §7 (the canonical registry).*

## 1. What we test and why it matters

M0 (A05) runs once, Fri 16 – Sun 18 Oct. Every later arm is compared with those runs, and no arm re-runs Full-v2 in its own session. Anything that changes between sessions would therefore look like an arm effect. Examples: a new image, a vLLM restart with different settings, a Serper outage, a stray `.dev.env`, a different topology or load.

A06 adds one fixed Full-v2 run, the **sentinel**, to every later session and checks it against M0. It makes no claim of its own. It is the only guard against session drift, and it earns the appendix line *"no session drift"*.

## 2. Background you need

**Code paths that can drift.** All refs are on `nested-cv-rebuttal` @ `07cbf93`. Every line below was re-read (re-verified Wed 7 Oct against the checkout at `1bdaf42`); `git diff 07cbf93 1bdaf42` is empty for `rogue_one_main.py`, `rogueone/`, `paper/scripts/vllm_session.sh` and `paper/scripts/clients_t16_rerun.sh`.

| What | Where | Why it matters for A06 |
|---|---|---|
| Fold reset | `_reset_fold_state` `main.py:417-457`, called at `:543-544` when `iteration_in_fold == 0` | The 5 fold-episodes of a run are separate agent episodes, so the unit of recovery is the fold-episode. |
| Outer partition | `ROGUE_ONE_FOLD_SEED` `config.py:310` → `deterministic_kfold_indices` (`rogueone/utils/train_test_splitter.py:199`); stratified only via `config.py:323` | At fold seed 42 the sentinel sees exactly M0's partition. |
| Sampling seed | `ROGUE_ONE_LLM_SEED` `config.py:329-330`, injected in `_merge_extra_body_for_vllm` (`clients.py:200-213`; seed at `:211-213`; `:204` turns Qwen thinking off) | Under `PARALLEL=6` this does not make runs reproducible (Finding 36; `paper/scripts/clients_t16_rerun.sh:129-133`). A distinct seed per sentinel only guarantees that the sentinels are independent. |
| Effective flags | `config.py:23-25`: `.env`, then `.dev.env` with `override=True`, then `.secret.env` | A `.dev.env` baked into an image would silently change Full. E1 logs the effective values as `flags/<NAME>`. |
| Live search | `kn/web_search.py:72` (live `GoogleSerperAPIWrapper`); `:125-182` (query variants, each attempt logged `ok=True/False` to the retrieval audit); `:179-181` (early return on credit or auth failures) | Outages and exhausted credits are session-level shocks. The E4 shared cache softens them but does not remove them. |
| Selection | `compute_fold_aggregation` `main.py:1074`. It returns nothing unless all folds completed (`:1101-1102`), and falls back to the held-out metric if the inner-CV column is missing (`:1114-1130`). | A silent fallback is a drift indicator. |
| Silent success | `run_experiments` `main.py:1590-1598` swallows exceptions. The summary is written at `:1350-1351`. | Success = rc 0 **and** `fold_aggregation_summary.csv` (`clients_t16_rerun.sh:38-45`). |
| Task registration | Hard-coded list `main.py:1536-1566`; `select_paths` `:1568-1580` | `synp_*` tasks run only through `ROGUE_ONE_TASK_REGISTRY` (E1). |
| Serving | `paper/scripts/vllm_session.sh:51` (`vllm/vllm-openai:v0.22.0`); refuses TP > 4 (`:66-69`); auto-wait `:132-154` | The auto-wait is never used: pin `GPUS` and set `WAIT_MAX_MIN=0`. |

**Definitions**
- **Full-v2.** Every flag at its `scripts/agent2027/v2_defaults.env` value: 10 iterations × 5 folds, inner-CV selection, rollback off, temperature 0.7, R and pool mode as frozen at G1 (Fri 16 Oct), `ROGUE_ONE_LLM_TRACE=1`, 6 h cap per run. Later images (v2.N) add only default-off switches whose Full values reproduce v2.0.
- **Session.** One server lifecycle running one session file, `manifests/agent2027/sessions/<SESSION_ID>.csv`. The ID has the form `S<MMDD><a-z>`. For A06, servers launched together count as **one** session even when each has its own session file:
  - under D5 = TP2, the two O120 servers (e.g. `S1016a` on 8010 and `S1016b` on 8011, the A05 pattern); the sentinel row goes into the file of the server whose port it uses (§4.2);
  - a **small-rung night**: the per-rung jobs launched together that night (one `ro27_vllm_session.sh` per rung before E7, or one `ladder_session.sh` after it); the `sentinel_Q4` row goes into the Q4 server's file (port 8032).
- **SYN-P 7.** The tasks `synp_p1`, `synp_p2`, `synp_p3`, `synp_p4`, `synp_p5`, `synp_p6a`, `synp_p6b` (this is also the **task-index order** used for tie-breaks): n = 2,000, φ planted, φ never mounted. `synp_p2`: φ = (x_a/x_b)·x_c (regression; E8a confirms the type in `synp_meta/registry.yaml`).
- **`rec_sel`.** Scored per fold-episode. It is 1 if the inner-CV-selected iteration contains an active, non-fallback feature that matches φ on training rows: \|Spearman\| ≥ 0.95, or NMI ≥ 0.9 for P3 and P5. Raw columns and single-column monotone transforms never count. ITT: a failed or capped run contributes 5 zeros.
- **`success`.** The run finishes within the cap and scores ≥ `base_no_fe` − 2 × fold SD. **`func_success`** = `success` **and** the inner-CV-selected iteration contains ≥ 1 accepted non-fallback LLM feature (01 §4.6).
- **`context_errors`.** HTTP 400 responses plus `finish_reason=length` completions in the run (01 §4.6).
- **s_task.** The fold SD of `base_no_fe` on the sentinel task at seed 42, taken from the A08 CPU screen (classicfe image, Mon 12 Oct). For σ_AA of the score channel (§4.7), each A/A pair is scaled by its own task's s.

## 3. Hypotheses and interpretation (all hypotheses)

- **Hoped: a flat series.**
  - Every session verdict is PASS or an isolated WARN (about 12% of sessions are expected to WARN under no drift, §4.7), no session is FLAGged, and the pooled test gives p ≥ 0.05.
  - Paper: *"No session drift: process metrics stayed within M0's band, and pooled sentinel recovery was X/Y against M0's Z/10."*
- **Null.** This is the same outcome as the hoped one, but power is low (§4.8). Report it honestly as *"no drift ≥ ~35 pp detected"* (the K-dependent shift from §4.8), never as "no drift".
- **Opposite: drift.**
  - A session is flagged: its arm rows are rerun if the reserve allows; otherwise they carry a flag and a caveat.
  - If the pooled test rejects, every arm-vs-M0 contrast gets a drift caveat and a period-term sensitivity analysis.
  - If drift starts with a new image, that release counts as having changed Full-v2 (§6).
  - Drift is two-sided: later sentinels recovering **more** than M0 (for example because the shared Serper cache is warmer, or a vLLM restart changes batching) is drift too and gets the same caveat; it would make arm-vs-M0 contrasts look better than they are.
  - A FLAG that coincides with a known, logged change (a topology change, a different vLLM image) is reported as that change's effect, not as an unexplained defect (§7).

## 4. Design

**4.1 Control.** M0's two seed-42 runs on the sentinel task, `A05__full__<task>__f42__r0` and `…__r1` (llm_seed 42 / 1042). Together they give Z/10 fold-episodes. They are reused; A06 makes no new control runs.

**4.2 Arms.** Both arms use `env_overrides = {}`: pure Full-v2 on the host session's image (`image_tag` = the host's tag, written into the row when it is planned into the session). If a session mixes images (rows carry their own `image_tag`, e.g. A16's v2.4 pilot appended to a v2.3 session per A16-T10), the sentinel takes the **newest** image in the session, and that session is the newest image's guard-3 session if it is the first to run it. The one exception is `{"ROGUE_ONE_STRATIFIED_FOLDS":"True"}`, copied from the A05 row when the task is a classification task and D4 is adopted.
- `sentinel`: O120 for every role and W, fold seed 42, on the **G2 sentinel task**. Under D5 = TP4 it runs on port 8010; under TP2 it alternates 8010 (odd k) and 8011 (even k). In A19's mixed session it uses the O120 server (port 8010).
- `sentinel_Q4`: Q4 for every role and W, `synp_p2`, fold seed 42, port 8032.

**4.3 Sentinel-task rule.** Pre-registered Fri 16 Oct and applied at G2 (Mon 19 Oct), before any arm result is read.
1. Candidates are the SYN-P 7 tasks. P7 is excluded because its recovery is 0 by design (P7 policy), and P0 because its recovery is 0 by construction.
2. Pick the task whose seed-42 `rec_sel` over M0 r0 + r1 (10 episodes) is closest to 5/10.
3. Tie-breaks, in order: the 15-episode rate (adding f4 r0) closest to 0.5, then the lower task index (order as in §2).

**Why seed 42 first.** The sentinel runs at fold seed 42 and is tested only against M0's seed-42 episodes (Z/10, §4.8), so the seed-42 rate is what sets the test's power.

**Conflict with A05 §2 (to resolve before X10, Fri 16 Oct).** A05 §2 currently states the rule the other way round: the 15-episode rate first, the seed-42 rate as tie-break (its candidate set, SYN-P 7, agrees). A05 says this rule is owned by A06, so the version above is binding; A05 §2 and the X10 text must be aligned to it (A06-T6).

**4.4 Seeds and identity.**
- k = 1, 2, … is the sentinel number within each arm.
- `replicate = k`, `is_sentinel = 1`.
- `llm_seed = 5000 + k` (O120) or `6000 + k` (Q4). An llm_seed is never reused: an infrastructure re-queue gets a new k (and a new row, A06-013… or A06-106…).
- `cap_hours = 6`.
- W&B: entity `PlanBench`, project `RogueOne-AGENT2027`, group `A06`, job_type = arm, tags [`<SESSION_ID>`, `v2`, `<image_tag>`]. Run name `A06__<arm_id>__<dataset_id>__f42__r<k>`.

**4.5 Placement** (binding rule).
- **One sentinel per qualifying session**, with `order_key` random (`make_manifest.py --plan-session`).
  - A session that runs past 48 h gets one more sentinel for each further 48 h.
  - **The first qualifying session on each new image** (v2.1 only if it carries a sentinel, see below; v2.2, v2.3, v2.4) puts its sentinel in the **first wave**: set its `order_key` to the session minimum. This is the **guard-3 session** for that image (§6).
- **Qualifying sessions:**
  - every O120 session on a v2.N image with analysed full runs, from Tue 20 Oct to DF: A08 GPU part, A10–A16, A19's O120 server, and rerun sessions;
  - `S1019a` (A09, v2.1) **if** A07-T14 has moved the HC1 rows into it (A07 then requires a sentinel), or if Q1 = yes;
  - an O120 pilot session with full-length rows that is the first session on a new image (A16's crash-only pilot `S1022b` on v2.4, which A16-T10 plans with a sentinel);
  - every small-rung night (A17) that has a Q4 server. A small-rung night without a Q4 server carries no sentinel; record that in the session plan.
- **No sentinel:** A05's own sessions (`S1016a/b`) and A05's infra re-queue `S1018a` (they are the reference); smoke sessions whose rows use `ROGUE_ONE_SMOKE_FOLDS` (e.g. A10/A11's smoke `S1020s`, A12's `S1021b`, A04 sessions A–C incl. `S1021m`); A18 replay-only sessions; F0-pool-only sessions; the single Q122 session; and the Thu 22 O20 session (A20 against A17 `f2_O20` is a within-session comparison; see Q2).
- **Q4 reference.** The first small-rung session also carries A17's `f2_Q4 × synp_p2 × f42 × r0` row, the **Q4 anchor**.
  - The anchor plus `sentinel_Q4` k = 1 form the two-run Q4 reference, analogous to M0's r0 and r1.
  - Sentinels k ≥ 2 are checked against that reference.

**4.6 Run matrix**

| Arm | Rows | Rung | Dataset | Fold seed | llm_seed | k | Runs | u | Run name |
|---|---|---|---|---|---|---|---|---|---|
| `sentinel` | A06-001…012 | O120 | G2 task (SYN-P 7) | 42 | 5000+k | 1…6–12 | 6–12 | 4.2–12 | `A06__sentinel__<task>__f42__r<k>` |
| `sentinel_Q4` | A06-101…105 | Q4 | `synp_p2` | 42 | 6000+k | 1…3–5 | 3–5 | 1.5–5 | `A06__sentinel_Q4__synp_p2__f42__r<k>` |
| reused: M0 | A05-0xx (task's rows in 031…038 and 047…054) | O120 | G2 task | 42 | 42 / 1042 | — | 0 | 0 | `A05__full__<task>__f42__r{0,1}` |
| reused: Q4 anchor | A17's row | Q4 | `synp_p2` | 42 | 42 | — | 0 (A17 pays) | 0 | `A17__f2_Q4__synp_p2__f42__r0` |
| **Total new** | | | | | | | **9–17** | **5.7–17** | |

**Budget arithmetic.** O120: 6 × 0.7 = 4.2u to 12 × 1.0 = 12u. Q4: 3 × 0.5 = 1.5u to 5 × 1.0 = 5u. Total 5.7–17u, the same as the T1-minimum breakdown (M0-S card) and 01 §4.9. Divided by 1.4u per box-hour (2 u/h × 0.7), that is 4.1–12.1 h of the 4-GPU box, or 16–49 H100-hours. The count of 6–12 O120 sentinels follows from the hosting plan below: about 8–10 qualifying O120 sessions Tue 20 – Sat 31, plus up to 2 conditional ones (`S1019a`, extra 48-h sentinels), and the CL5 floor of 6.

**Planned hosting** (dates from 01 §6.2; session IDs as planned in the host documents, `<x>` = letter not yet fixed). The Image column is the plan; the row always takes the host session's actual image.

| When | Host session | Session ID | Image | A06 row |
|---|---|---|---|---|
| Sun 18, night | A17 small rungs, first night (+ Q4 anchor) | `S1018b` (proposed; `S1018a` is A05's re-queue) | v2.0 | `sentinel_Q4` k=1 (reference, no verdict) |
| Mon 19 | A09 RT3, plus the A07 rows if A07-T14 fired | `S1019a` | v2.1 | `sentinel`, first wave, **required** if A07-T14 fired, otherwise only if Q1 = yes (guard 3 for v2.1) |
| Mon 19, night | A17 small rungs | `S1019c` (A17's N2 Q4 server; `S1019b` is left to A09's TP2 sibling) | v2.0 or v2.1 (A17's choice) | `sentinel_Q4` k=2 |
| Tue 20 | A10 + A11 (A08's GPU comparators run after it in their own session, A08 §4; A08 carries its own sentinel if it hosts O120 rows) | `S1020a` (the main session shared by A10 and A11; smoke is `S1020s`, `S1020b` is the TP2 sibling; the sentinel goes into the **main** session, never the smoke one) | v2.2 | `sentinel`, first wave (guard 3 for v2.2, and for v2.1 if `S1019a` had none) |
| Wed 21 | A13 in its own day session by default (A13-T10), then A12 block 1 + A14 from the evening (A13 joins it if G0b-1 passes late) | A12, A13 and A14 share `S1021a` (smoke `S1021b`; A04 session C is now `S1021m`) | v2.3 | one `sentinel` per session; the first v2.3 session puts it in the first wave (guard 3) |
| Thu 22 | A19 (the O120 server of the mix) | per A19 | v2.3 | `sentinel` |
| Thu 22, night | A16 crash-only SET pilot (blood, communities) | `S1022b` (A16-T10) | v2.4 | `sentinel`, first wave (guard 3 for v2.4) if it is the first v2.4 session |
| Fri 23 | A16 SYN-P + A12 block 2 | `S1023a` | v2.4 | `sentinel` (first wave and guard 3 if the pilot session had none) |
| Sat 24 – Thu 29 | A16 Core-10 (`S1026<x>`), A15 AB2, A08 tail, re-queues, RV items | per host | v2.4 | one per session, at least one per 48 h |
| Fri 30 – Sat 31 | rerun sessions | per host | v2.4 | one per session |
| further small-rung nights | remaining A17 small-rung rows | per A17 | v2.x | `sentinel_Q4` k=3…5 |

**4.7 Per-session check** (`sentinel_drift.py --session`). The metrics are grouped into 5 channels, so that correlated metrics are not double-counted.

| Channel | Metric (R13 name) and scale |
|---|---|
| Cost | `tok_compl_total`, `llm_calls_total` (log) |
| Time | `wall_s` (log) |
| Execution | `err_per_accepted` (log1p); execute rate `fun_executes ÷ fun_proposed` (raw); `context_errors` (log1p) |
| Search | `serper_ok_false_rate` (raw) |
| Score | `heldout_score` ÷ s_task (a collapse shows up here) |

- **Band for metric m.**
  - Centre: c_m = the mean of the two reference runs (M0 r0 and r1 on the sentinel task).
  - Half-width: max(3 · √1.5 · σ_AA,m, floor_m).
  - σ_AA,m = √(Σ dᵢ² / 2n) over M0's n = 8 SYN-P A/A pairs at seed 42 (A05-031…038 against A05-047…054, paired by task), with dᵢ = r0 − r1 on the metric's scale (for the score channel, each pair divided by its own task's s). This gives 8 df.
  - Floors: 0.1 on log and log1p scales; 0.05 for the rates; 0.1 for the scaled score. (If every M0 run has `context_errors` = 0, σ_AA,m = 0 and the floor applies: a single context error, log1p(1) = 0.69, is outside.)
  - The factor √1.5 makes this a prediction band for one new run against a two-run mean (Var = σ²(1 + 1/2)).
- **Channel outside.** A channel is outside its band when **any** of its metrics is outside its band.
- **Q4 band.** Centre = the mean of the Q4 reference (Q4 anchor and `sentinel_Q4` k = 1). Half-width = max(3 · √1.5 · σ_Q4,m, floor_m), with σ_Q4,m = max(M0's σ_AA,m, \|anchor − k1\| / √2).
- **Hard indicators.** Any one of these means FLAG:
  1. a crash that persists after the solo retry;
  2. `cap_hit` (for Q4: only if neither Q4 reference run hit the cap; otherwise a Q4 cap hit is reported descriptively);
  3. `serper_ok_false_rate` ≥ 0.5;
  4. the inner-CV column missing or all-NaN in the run's results (the `main.py:1114-1130` fallback);
  5. a `flags/*` value that differs from the reference's (M0 r0, or the Q4 anchor), other than identity, model and port keys and post-v2.0 flags at their Full value;
  6. a `vllm_image_id` that differs from the reference's.
  
  (`context_errors` is a channel metric, not a hard indicator: "above the maximum of M0's 16 runs" would fire in up to 1/17 ≈ 6% of sessions under no drift, by exchangeability, and a FLAG stops the host's queue.)
- **Verdicts.**
  - **PASS:** no channel outside its band and no hard indicator.
  - **WARN:** exactly one channel outside and no hard indicator.
  - **FLAG:** 2 or more channels outside (≥ 3 for Q4, because its scale is borrowed), or any hard indicator, or the same channel outside in two consecutive sentinel-bearing sessions of the same arm on the same image.
  - **Expected false-alarm rates** (computed exactly for this document: P(\|t₈\| > 3) = 0.017 per metric, 8 metrics in 5 channels, independent): under no drift about **12%** of O120 sessions WARN and about **1%** FLAG through the channel rules (0.7% from ≥ 2 channels, 0.5% from a repeated channel); the floors only lower these rates. Hard indicators 1–2 add M0's own persistent-crash and cap-hit rates. For Q4 the scale is borrowed, so these rates are only indicative; if the borrowed σ were exact, the ≥ 3-channel FLAG would fire in under 0.1% of sessions.
- `rec_sel` (k/5) is shown per session for description only; it is never part of the verdict.

**4.8 Pooled tests** (`sentinel_drift.py --final`, α = 0.05, outside Holm).
- **Primary.** A two-sided Fisher exact test on fold-episodes: pooled O120 sentinels X/Y against M0's Z/10.
- **Power.** With Z = 5/10, only an observed rate ≤ 0.13 or ≥ 0.87 rejects at K = 6 sentinels (Y = 30 episodes), and ≤ 0.18 or ≥ 0.82 at K = 12 (Y = 60). That is a shift of about 32–37 pp. If Z ends up further from 5 (e.g. Z = 3/10), only ≤ 0.05 or ≥ 0.67 rejects at K = 12. *Computed exactly for this document (scipy `fisher_exact`, two-sided).*
- **Sensitivity analyses:**
  - an exact test stratified by outer fold (labels permuted within each of the 5 folds; the partition is shared);
  - a run-level permutation test (M0's 2 runs against the K sentinels). Its minimum two-sided p is 2 / C(K + 2, 2) (0.056 at K = 7, 0.044 at K = 8), so it cannot reach p < 0.05 unless K ≥ 8.
- **Descriptive:**
  - an exact Cochran–Armitage trend test over session order, with M0 = 0;
  - the Q4 series, with homogeneity of `rec_sel` and `func_success` across sessions (Fisher–Freeman–Halton, exact or 10⁵ Monte Carlo draws).
- **Pooling rule.** Sentinels are **never** pooled into Full for the primary contrasts.

**4.9 Failure handling.**
- A crash is retried once, alone, at the end of the session, under the same row and seed.
- A cap hit is never retried (ITT).
- Infrastructure faults (server death, a W&B outage) follow A05's rule: the row is re-queued in a new session with a new k (new row) and `notes=infra-rerun`. The session they occurred in is analysed as flagged.
- Recovery detection on sentinel rows uses the same `analysis/recovery_detector.py` (E8a, extended in E11b) and the same φ metadata as M0.

## 5. Tasks

### 5.1 Prerequisites (verify before the first launch)
- [ ] **A06-T1** [type: decision] (0.05 d, due Thu 15 Oct) — Henrik confirms the sentinel rule (§4.3–4.8), a default adopted unless he objects, and answers Q1–Q3. *Done when:* the answers are in the draft pre-registration.
- [ ] **A06-T2** [type: config] (0.05 d, due Mon 12 Oct, G0a; piggybacks on E1's 3-row dry run) — Test E1 with a 3-row dry-run session file that contains one row with `is_sentinel=1` (`arm_id=sentinel`, any `synp_*` task, `replicate=1`, `llm_seed=5001`, `env_overrides={"ROGUE_ONE_SMOKE_FOLDS":"1"}`, `notes=smoke`, image `agent2027-pre.N`, project `RogueOne-AGENT2027-prefreeze`). *Done when:*
  - the container is named `ro27-<session_id>-a06-001`;
  - W&B (prefreeze project) shows `A06__sentinel__…__r1` with `is_sentinel=1`, `llm_seed=5001`, `code_phase=smoke`, and `flags/*` equal to the then-current draft `v2_defaults.env` apart from `ROGUE_ONE_SMOKE_FOLDS`;
  - a `synp_*` dataset resolves through `ROGUE_ONE_TASK_REGISTRY`.
- [ ] **A06-T3** [type: data] (0.05 d, due Mon 12 Oct) — Check that the G0a smoke summary has every §4.7 metric. *Done when:*
  - `tok_compl_total`, `llm_calls_total` and `wall_s` are non-zero;
  - `err_per_accepted`, `fun_proposed`, `fun_executes`, `serper_ok_false_rate`, `context_errors` and `cap_hit` are present;
  - `vllm_image_id` and `server_topology` are logged.
- [ ] **A06-T4** [type: decision] (0.02 d, due Fri 16 Oct) — Confirm that Q4 passed A04 session A and that its HF revision is pinned. *Done when:* the Q4 id and revision are recorded in the session plan and the X10 text. They stay fixed for the whole series (the fallback `Qwen/Qwen3-4B`, if used, is used for every Q4 row, including the anchor).
- [ ] **A06-T5** [type: data] (0.02 d, due Mon 19 Oct, 08:00) — Confirm that A05 rows A05-031…054 are finished and that recovery detection has run on them. *Done when:* the G2 memo `ANALYSIS__g2_stats__20261019` names the sentinel task and Z/10, and the A05 `a05-handoff` artifact (A05-T19, episode table) can be fetched.

### 5.2 Implementation
- [ ] **A06-T6** [type: writing] (0.1 d, due Fri 16 Oct, 12:00, inside X10) — Write §4.3–4.9 and §6 into `docs/agent2027/preregistration.md`. *Done when:* the text is committed before tag `agent2027-prereg`, includes the sentence "sentinels are never pooled into Full for primary contrasts", and A05 §2's sentinel-task wording matches §4.3 (seed-42 rate first).
- [ ] **A06-T7** [type: code] (0.1 d, due Mon 19 Oct, 08:00; part of E11a) — Extend `scripts/agent2027/analysis/g2_stats.py` to output the sentinel task (with the §4.3 ranking table of all 7 candidates), Z/10, s_task, and for each §4.7 metric c_m, σ_AA,m and the band limits. *Done when:* a hand-computed toy example is reproduced and the band table is in the G2 memo.
- [ ] **A06-T8** [type: code] (0.25 d, pulled forward from E11b, due Tue 20 Oct, 08:00) — Write `scripts/agent2027/analysis/sentinel_drift.py --session <SESSION_ID>`. It fetches the sentinel's summary, `flags/*` and `run-output` from W&B, applies §4.7, and logs `ANALYSIS__sentinel_drift__<YYYYMMDD>` (tagged with the session ID). *Done when:*
  - a toy run with 2 perturbed channels gives FLAG;
  - a toy run with only `context_errors` perturbed gives WARN;
  - M0's own r1, scored against r0 alone, gives PASS;
  - a changed `flags/` value gives FLAG.
- [ ] **A06-T9** [type: config] (0.05 d; Q4 block due Sat 17 Oct, O120 block due Mon 19 Oct after G2) — Create `manifests/agent2027/A06_M0-S.csv` with the 01 §7 R7 columns (`make_manifest.py --ablation A06`). Common values: `ablation_id=A06`, `fold_seed=42`, `replicate=k`, `llm_seed` per §4.4, `is_sentinel=1`, `session_id` empty, `image_tag` filled with the host's tag at planning, `cap_hours=6`, `env_overrides={}` (or the D4 stratified override, §4.2). *Done when:* the file is committed and pushed and copied to `~/ro27_manifests/` (§5.3).
  - **By Sat 17 Oct:** rows A06-101…105 (`sentinel_Q4`, `synp_p2`, `regression`, rung Q4 for S/E/T, ports 8032, `expected_u=0.75`).
  - **Mon 19 Oct, after G2:** rows A06-001…012 (`sentinel`, the G2 task and its `task_type`, O120 for S/E/T, port 8010 for all roles under TP4; under TP2 8010 for odd k and 8011 for even k; `expected_u=0.85`).
  - Re-queue rows (A06-013…, A06-106…) are appended only when §4.9 or §6 needs them.
- [ ] **A06-T10** [type: code] (0.25 d, due Tue 27 Oct; part of E11b) — Add `sentinel_drift.py --final`, which runs the §4.8 tests and the §6 period sensitivity. *Done when:* it reproduces a toy example and runs end to end on M0's artifacts.

### 5.3 Execution on CAIR
**Rules for every launch** (CAIR hard rules, 01 §5.1):
1. **Containers only.** Every execution (runs, analysis, W&B queries) happens inside Docker. The host issues only `docker` / `docker compose` commands (`-T` for any `docker compose run` over SSH), the `~` launchers that issue them, and read-only queries (`nvidia-smi`, `docker ps`, `grep`, `tail`). Never `python …` on the host.
2. **GPU pre-flight before any GPU container** (command below). If there are not enough free GPUs, **stop and report** the per-GPU memory and utilisation, the compute apps and the running containers, and let Henrik decide. Launch with explicit `GPUS=<ids>` from the pre-flight and `WAIT_MAX_MIN=0` (the auto-wait at `vllm_session.sh:132-154` is never used; its default is 600 min).
3. **≤ 4 of the 8 H100s**, ever, without Henrik's explicit approval in chat. One Rogue One job at a time.
4. **W&B is the system of record.** Runs are stateless and upload `run-output`; `WANDB_API_KEY` comes from `<CAIR_HOME>/code/PlanBench/.env` (extracted as `clients_t16_rerun.sh:59` does), `SERPER_API_KEY` and LLM settings from `.env` / `.secret.env` in `<ROGUEONE_CAIR_PATH>`. Never push the images.
5. **Long jobs run detached** with `nohup`; the job log goes in `~` (`~/ro27_<SESSION_ID>.job.log`), never in the root-owned run folders.

All SSH calls are non-interactive one-shots: `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "<command>"`.

`<ROGUEONE_CAIR_PATH>` = `<CAIR_HOME>/code/RougeOne` (**INFERRED** from `clients_t16_rerun.sh:135` `REPO_TASKS=$HOME/code/RougeOne/tasks` and `:59`; verify once with `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "ls -d ~/code/RougeOne"`). The fetch remote `git@github.com:HenrikNorkart/RougeOne.git` is also inferred (01 §5.2).

**Code and manifests to CAIR** (01 §5.2 steps 0 and 3). Commit and push `agent2027/v2` locally, then:
```
ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15' git fetch git@github.com:HenrikNorkart/RougeOne.git agent2027/v2 && git checkout -B agent2027/v2 FETCH_HEAD && git log -1 --format='HEAD: %h %s'"
ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "cd <ROGUEONE_CAIR_PATH> && mkdir -p ~/ro27_manifests && cp -r manifests/agent2027/. ~/ro27_manifests/ && ls ~/ro27_manifests/sessions/"
```

A06 never launches a session of its own. The host ablation's launch must include its row, after this pre-flight:
```
ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader && echo '--- compute apps ---' && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader && echo '--- containers ---' && docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'"
```
- Proceed only if the needed GPUs show < 2,000 MiB used, ΣGPU ≤ 4, and none of our `ro27-*`, `rogueone-vllm-*`, `llmfe-*` or `caafe-*` containers is running. Otherwise STOP and report.
- The detached launch of an O120 host session (01 §5.2 step 4; the host's document owns the exact line):
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "SESSION_ID=<SID> SESSION_FILE=~/ro27_manifests/sessions/<SID>.csv GPUS=<ids from pre-flight> NEED=<n> MODEL=openai/gpt-oss-120b PORT=8010 SERVER_NAME=rogueone-vllm-o120-<sid> LOG=~/vllm_session_<SID>.log IMAGE_RO=rogue_one:agent2027-v2.<N> RUNROOT=~/rogueone-runs-agent2027 PARALLEL=6 CPUS=12 RUN_CAP_HOURS=6 WANDB_PROJECT_NAME=RogueOne-AGENT2027 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_<SID>.job.log 2>&1 & echo launched; sleep 5; tail -5 ~/ro27_<SID>.job.log"
  ```
  Under TP2 the second server uses `<SID'>`, `PORT=8011`, `SERVER_NAME=rogueone-vllm-o120-<sid'>` and `ROGUEONE_ALLOW_SECOND_SERVER=1`, on 2 other free GPUs (total ≤ 4).
- The Q4 server of a small-rung night (A17 owns the launch; before E7 one job per rung): `MODEL=Qwen/Qwen3.5-4B NEED=1 GPUS=<1 id> PORT=8032 SERVER_NAME=rogueone-vllm-q4-<sid> EXTRA_ARGS='--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --language-model-only' MAX_MODEL_LEN=131072 ROGUEONE_ALLOW_SECOND_SERVER=1`, the rest as above with `PARALLEL=3`. The pre-flight must show that all GPUs of the night (≤ 4) are free.

**Tasks**
- [ ] **A06-T11** [type: run] (0.05 d, Sun 18 Oct, before the first small-rung launch) — Add A06-101 **and** A17's `f2_Q4 × synp_p2 × f42 × r0` row to the first small-rung session file (`make_manifest.py --plan-session <id> --rows …`, e.g. `S1018b`), with A06-101 in the first wave, both in the Q4 server's session file. Commit, push, fetch on CAIR, then copy the manifests to `~/ro27_manifests/` (commands above). *Done when:* both rows are in the committed session file, and in `~/ro27_manifests/sessions/`, before launch.
- [ ] **A06-T12** [type: run] (0.02 d per session, Mon 19 Oct – Sat 31 Oct) — Before each qualifying session (§4.5) is planned, add the next free `sentinel` or `sentinel_Q4` row (lowest unused k) to the host's `--plan-session` call, with `image_tag` = the host's tag and, under TP2, the port that matches k's parity. In the guard-3 session of a new image, set its `order_key` to the session minimum.
  - Under CL5, use only every second O120 session (floor 6), but **never skip a guard-3 session** (first qualifying session on a new image) or a session whose host document requires a sentinel (A07 fallback in `S1019a`).

  *Done when:* every qualifying session file holds its sentinel, or a CL5 skip is recorded in the session plan.
- [ ] **A06-T13** [type: run] (0.01 d per session; monitoring, every sentinel-bearing session) — Watch the sentinel row's `OK` / `FAILED` / `CAP_HIT` line, or `=== RO27_SESSION_END <id> rc=… ===`:
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "grep -E 'A06-[0-9]{3}|RO27_SESSION_END' ~/ro27_<SESSION_ID>.job.log | tail -5"
  ```
  Exit code 255 means the network dropped, not the job: reconnect. Cross-check the W&B state in a container (01 §5.2 step 7, with `filters={'group': 'A06'}`). *Done when:* the sentinel's final state (`OK` / `FAILED` after the solo retry / `CAP_HIT`) is recorded in the session plan. A `running` W&B state with no live container is a zombie: report it.
- [ ] **A06-T14** [type: analysis] (0.05 d per session; within 12 h of the sentinel finishing, from Tue 20 Oct when A06-T8 lands) — Run the check in a CPU-only container (no GPU, so no GPU pre-flight; `--cpus` ≤ 16):
  ```
  ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14 "docker run --rm --cpus 4 --env-file <CAIR_HOME>/code/PlanBench/.env -v <ROGUEONE_CAIR_PATH>/scripts/agent2027:/workspace/scripts/agent2027:ro -v <CAIR_HOME>/rogueone-synp-meta:/meta:ro --entrypoint python3 rogue_one:agent2027-v2.<newest N> /workspace/scripts/agent2027/analysis/sentinel_drift.py --session <SESSION_ID> --meta /meta"
  ```
  - The image's `WORKDIR` is `/workspace` (`Dockerfile`), so the read-only mount replaces the baked-in `scripts/agent2027` with the fetched version.
  - The SYN-P metadata (φ) is mounted only into this analysis container, never into a run container.
  - If W&B rejects the key because it is quoted in `.env`, pass `-e WANDB_API_KEY=…` extracted as `clients_t16_rerun.sh:59` does.
  - For the first small-rung night (`S1018b`), this step computes the Q4 reference instead of a verdict. It runs as soon as A06-T8 lands (by Tue 20 Oct, 08:00), before the k = 2 verdict.

  *Done when:* the verdict (or the Q4 reference) is logged in W&B as `ANALYSIS__sentinel_drift__<YYYYMMDD>` tagged with the session ID, and in the session plan. A new image is cleared only on PASS, or on WARN under the §6 guard-3 rule.
- [ ] **A06-T15** [type: decision] (0.1–0.25 d per FLAG; same day as any FLAG, diagnosis within 12 h) — Apply the §6 FLAG actions. *Done when:* the actions are recorded in the notes of the `ANALYSIS` run and, if rows are rerun, the rerun rows are in a committed session plan.

### 5.4 Analysis
- [ ] **A06-T16** [type: analysis] (0.1 d, Tue 27 Oct, G4) — Interim pooled read (`sentinel_drift.py --final` on the sessions so far, or by hand from the per-session verdicts if A06-T10 is late). It is operational only and is not a reported test. *Done when:* the read is logged as an `ANALYSIS` run, and any rerun it triggers is in a committed session plan that launches by L-LAST (Thu 29 Oct).
- [ ] **A06-T17** [type: analysis] (0.25 d, Sun 1 – Tue 3 Nov) — Run `sentinel_drift.py --final` on all sessions. *Done when:* an `ANALYSIS` run holds:
  - the primary Fisher test and both sensitivity tests;
  - the trend test;
  - per-channel time series against M0's band;
  - the Q4 homogeneity table;
  - if the primary test rejects, the period sensitivity (§6).

### 5.5 Paper outputs
- [ ] **A06-T18** [type: writing] (0.25 d, Wed 4 – Fri 6 Nov; package Thu 12 Nov) — *Done when* all of these are in the draft and the replication package:
  - an appendix paragraph and a small figure: per-session sentinel channels against M0's band, plus `rec_sel` k/5;
  - the hoped sentence, or the drift caveat, in threats to validity, with the detectable-shift statement;
  - `A06_M0-S.csv` and all session files.

## 6. Decision rules and gates

| When | Rule |
|---|---|
| Fri 16 Oct (G1, X10) | The formulas are pre-registered; the values come at G2. |
| Mon 19 Oct (G2) | The task, Z/10 and the bands are fixed before any arm result is read, and never revised. If G2 cannot compute them, the O120 sentinels wait, and arms launched in the meantime are provisional until a later sentinel passes. |
| Guard-3 session of v2.1 (only if `S1019a` carries a sentinel) / v2.2 / v2.3 / v2.4 (§4.5) | **Guard 3**, together with guard 1 (the byte-identical E5 prompt diff) and guard 2 (switch-off unit tests). It needs PASS, or WARN as long as the next sentinel-bearing session on that image does not have the same channel outside. v2.2 also covers v2.1 when `S1019a` had no sentinel. Arm rows that ran on the new image before its guard-3 verdict are provisional until it is logged. |
| Any **FLAG** | 1. Stop the session's queue if it is still running.<br>2. Diagnose within 12 h: `~/vllm_session_<id>.log`, the job log, both image IDs, the `flags/*` diff, the retrieval transcript and the energy run.<br>3. Repeat the sentinel in the next session. A stand-alone repeat session needs Henrik's go.<br>4. **Cause found and fixed:** rerun the session's arm rows from the rerun reserve, planned by L-LAST (Thu 29 Oct) where possible and launched no later than Fri 30 Oct (rerun window), so that they end before DF.<br>5. **No cause, and the repeat passes** (or no repeat is possible before DF): keep the rows, set `drift_flag = 1` for that session, and report every affected contrast with and without it.<br>6. **The repeat fails again:** real drift. No further sessions on that image; revert to the previous v2.N; disclose. |
| Tue 27 Oct (G4) | Interim read. Rerun decisions are made now, while the reserve and the dates still allow. |
| Thu 29 Oct (L-LAST) | Last new launch. Sentinel-triggered reruns found after this go into the Fri 30 – Sat 31 rerun window only if they end before DF (Sat 31, 24:00); otherwise step 5 applies. |
| Nov 1–3 | Fisher p < 0.05: every arm-vs-M0 contrast gets a drift caveat, and each confirmatory GLMM is refitted with the sentinels added as Full observations, `rec_sel ~ arm + period + (1|task) + (1|task:seed)` (period 0 = M0's sessions, 1 = every later session). The period effect is identified from the sentinel task alone; disclose that. Report whether any Holm verdict changes. Otherwise report the hoped sentence and the detectable shift. |

**Kill date:** none. A06 is never cut beyond CL5.

## 7. Risks and cut levers

| Risk | Response |
|---|---|
| Low power: M0 gives only 10 reference episodes | The detectable shift is about 32–37 pp. The per-session verdicts rest on the process channels, and the wording is pre-set (§3). |
| No informative task (every candidate at 0/10 or 10/10) | Keep the rule (a move off the floor is still detectable), or use Q3's alternative. |
| Topology changes (TP2 vs TP4; A19's mixed servers) | `server_topology` is logged. A FLAG that coincides with a topology change is reported as a topology effect, not as a defect. |
| The Serper cache hides changes in web content | It is shared by all arms; disclosed. |
| Q4 series: borrowed scale; the anchor must exist | The max(σ) rule and the ≥ 3-channel FLAG threshold. A17 must place the anchor in the first small-rung session (A06-T11). |
| Developer time (A06-T7 + T8 = 0.35 d in the over-full Sat 17 – Mon 19 window) | Fallback: query the sentinel's summary in a container (01 §5.2 step 7) and compare it by hand with the G2 band table. |
| Session-ID collisions across host documents (resolved in the cross-document pass: A04 session C is `S1021m`; A10/A11 main session is `S1020a`; A17 N2 uses `S1019c`–`f`) | A06-T12 checks the session file name before adding a row; the sentinel always goes into the main (non-smoke) session file. |
| A small-rung night without a Q4 server | No `sentinel_Q4` that night (§4.5); recorded in the session plan. If fewer than 3 Q4 sentinels result, the Q4 series is reported as descriptive only. |
| **CL5** (pre-declared lever 5) | A sentinel in every second O120 session, with a floor of 6: −3 to −5u (T1-minimum breakdown). Guard-3 sessions are exempt, so the realised saving may be at the low end. |

## 8. Definition of done
- [ ] The sentinel rule is committed under tag `agent2027-prereg`.
- [ ] The G2 memo holds the sentinel task, Z/10 and the band table.
- [ ] `A06_M0-S.csv` is committed. Every qualifying session has its sentinel, or a recorded CL5 skip. 9–17 runs (plus any re-queue rows) are in W&B group `A06`, each with `run-output` and the R12 config keys (`is_sentinel=1`).
- [ ] Every sentinel has a logged verdict, every FLAG has its actions logged, and the guard-3 status of v2.1–v2.4 is recorded.
- [ ] The final `ANALYSIS__sentinel_drift__<YYYYMMDD>` run exists.
- [ ] The appendix paragraph, the figure and the threats-to-validity sentence are in the draft.

**Open questions (for Henrik)**
- **Q1.** Should the Mon 19 A09 session (`S1019a`), the first on v2.1, get an O120 sentinel? Recommended: yes, within the 6–12 runs, if the G2 memo is out before A09 launches. Moot if A07-T14 fires: the A07 rows then require it.
- **Q2.** Should the Thu 22 O20 session get a `sentinel_Q4`? Recommended: no, unless a Q4 server is already up for the A18 replays. (This is a deliberate exception to 01 §4.8's "one sentinel per session"; O20 is not in the C5a trend, which uses the Qwen rungs only.)
- **Q3.** If no SYN-P 7 candidate is informative at seed 42: keep the rule (the default), or use `wine` × 42, which detects pipeline drift only?

**New names introduced** (not in 01 §7):
- the `sentinel_drift.py` options `--session`, `--final` and `--meta`;
- the verdicts PASS / WARN / FLAG;
- the 5 drift channels;
- the symbols σ_AA,m and s_task;
- the analysis column `drift_flag` and the covariate `period`;
- the "Q4 anchor" / Q4 reference;
- the row blocks A06-001…012 and A06-101…105 (re-queues from A06-013 / A06-106);
- `notes=infra-rerun` (taken from A05);
- "guard-3 session" and "small-rung night" (as defined in §2 and §4.5);
- the session IDs `S1018b` and `S1019c` (A17's Q4 servers on its first two small-rung nights; A17 fixes them).
