# G4 memo (Tue 27 Oct gate, written Mon 26 Oct 2026)

Items of G4 (pre-registration Part VI): FX alias rule / RV3; KT3 kill date; RV1 check; A19 SR6 deadline; A06 interim.
Per Amendment 15, this memo is committed **before** any detector runs on the RV3 rows A12-066…093.

## 1. A12 FX alias rule (RV3 trigger) — **fires**

Rule (V.12): pairs PA = −JM, PJ = −AM, PM = −AJ are reported jointly; RV3 semi-foldover only if a pair coefficient
has p < 0.05 in the pre-registered model (and the reserve check).

Data: pre-registered fraction rows A12-001…056, 8 cells × SYN-P 7 minus P1 × f42 r0 = 48 runs, 240 fold-episodes,
1 ITT run (zeros kept). Episodes from `~/ro27_analysis/A12/a12_episodes.csv` (detector run of 24 Oct after the δ commit;
RV3 rows not scored). Run table: `G4_A12_runs.csv`.

Model of record (V.12): run-level beta-binomial, ±1 coding, `k ~ P + A + J + M + PA + PJ + PM + (1|task)`, ML with
the task intercept integrated by 40-node Gauss–Hermite quadrature (`scripts/agent2027/analysis/glmm_holm.py g4`,
toy recovery check passes). Converged; φ = 4.3e3 (no extra-binomial variation), σ_task = 3.94. Full output:
`G4_A12_glmm.json`.

| Term | β | SE | Wald p | Δ_X (X+ − X−) | 90% CI | TOST p (±0.167) |
|---|---|---|---|---|---|---|
| P | 0.895 | 0.278 | 0.0013 | +0.143 | [0.073, 0.212] | 0.28 |
| A | 0.020 | 0.253 | 0.94 | +0.003 | [−0.064, 0.071] | < 0.0001 |
| J | 0.342 | 0.257 | 0.18 | +0.055 | [−0.013, 0.123] | 0.0034 |
| M | −0.813 | 0.274 | 0.0030 | −0.130 | [−0.199, −0.061] | 0.19 |
| **PA (= −JM)** | **−0.558** | 0.263 | **0.034** | −0.090 | [−0.158, −0.022] | — |
| PJ (= −AM) | 0.197 | 0.254 | 0.44 | +0.032 | [−0.036, 0.100] | — |
| PM (= −AJ) | 0.274 | 0.258 | 0.29 | +0.044 | [−0.024, 0.113] | — |

Cross-checks (same 48 runs, `glmm_check.R`, `glmm_firth.R`, R 4.4.1 container `ro27-rstats`):
- glmmTMB beta-binomial: **does not converge** (φ → ∞, non-positive-definite Hessian), i.e. the binomial boundary.
- glmmTMB binomial GLMM (the boundary model): PA p = 0.039, PJ 0.44, PM 0.32; P 0.0015, M 0.0039.
- Pre-registered non-convergence fallback, Firth logistic with task fixed effects: PA p = 0.042, PJ 0.45, PM 0.35;
  P 0.0010, M 0.0031.

**Verdict:** the p condition holds in every fit (PA p 0.034–0.042; PJ and PM > 0.28). Reserve check: the RV3 rows
were already run under Amendment 15 on GPUs that were otherwise idle (S1025f/g, ended Mon 26 01:44Z / 03:16Z), so
no reserve is consumed now; the condition is treated as met. **RV3 is triggered: the 12-cell refit of A12-T23
(rows A12-001…056 + A12-066…093) is the A12 analysis of record**, as Amendment 15 fixed in advance. The detector may
now run on A12-066…093.

Context for the read-out (not a G4 decision): the informative tasks are p2 and p6b (p3, p5 are 0 in every cell; p6a
is 4–5/5 in every cell; p4 is 0–1/5), so the fraction's effects rest on two tasks. The A and J main effects are
TOST-equivalent at ±0.167 on this fit (unadjusted; Holm across the family is at the final read-out); P is positive
(C2c direction); the PA alias is now separated by RV3.

## 2. KT3 kill date — **not killed**

A14 ran to completion on the local-corpus backend (its rows produced retrieval from the corpus), so the backend
was working before the kill date. PRELIM read-out:
`kb_local_correct` 16/35 vs `kb_off` 0/35. C4a stays in the family.

## 3. RV1 (CAP-5) check — **not evaluable at G4; decision needed**

Rule (A17 §6, A17-T15): fires if any of Q0.8, Q2, Q4 has Δ = pooled `rec_sel`(rung) − pooled `rec_sel`(A05 Full
f42 r0) ≥ −δ on SYN-P 7 × 42; floor guard as written.
- On the first-wave A17 rows (all CAP_HIT/FAILED, ITT zeros) every rung has `rec_sel` = 0, so Δ = −(Full) < −δ:
  **does not fire.**
- Those rows are replaced by A17-401…468 (Amendment 17, relaunched Mon 26 with the Qwen tool fixes), which end
  Tue 27 – Wed 28. RV1 is therefore not evaluable on the rows of record by G4.
- Open for Henrik: re-evaluate RV1 when the relaunched Q0.8/Q2/Q4 rows end, and if it fires, run the 14 RV1 rows
  (fallbacks off, needs the `ROGUE_ONE_FALLBACKS=off` build) launched by L-LAST Thu 29? Or record RV1 as evaluated on
  the first wave only and report "small suffices" claims as not testable? (Filed in HQ.)

## 4. A19 SR6 deadline — **passed earlier**

SR6 (routing) passed at A04-T19 (`A04_session_C_SR6.md`). A19 is relaunched under Amendment 18.

## 5. A06 interim (sentinel drift) — **pending**

`sentinel_drift.py` (E11b) is not yet written; the interim drift check follows with the rest of E11b this week.

## Addendum (Mon 26 Oct, after the memo commit `dfa0c63`): RV3 12-cell refit (A12-T23, analysis of record)

Detector run on A12-066…093 only after the memo above was committed (episodes `~/ro27_analysis/all/episodes_1026.csv`,
all 28 RV3 rows OK). Model `rec ~ P + A + J + M + PA + PJ + PM + AJ + AM + JM + block + (1|task)`, fraction (48) +
foldover (24) runs, SYN-P 7 minus P1, 360 fold-episodes, design rank 12/12 (`glmm_holm.py rv3`,
`RV3_A12_glmm.json`, `RV3_A12_runs.csv`). φ = 5.9e3, σ_task = 3.6.

| Term | β | Wald p (beta-binomial) | p glmmTMB binomial | p Firth (task FE) | Δ_X | 90% CI | TOST p |
|---|---|---|---|---|---|---|---|
| P | +0.869 | 0.0010 | 0.0001 | 0.0009 | +0.148 | [0.077, 0.218] | 0.32 |
| A | +0.019 | 0.94 | 0.89 | 0.93 | +0.003 | [−0.068, 0.075] | 0.0001 |
| J | +0.331 | 0.19 | 0.18 | 0.21 | +0.058 | [−0.013, 0.129] | 0.0057 |
| M | −0.787 | 0.0026 | 0.0003 | 0.0031 | −0.134 | [−0.203, −0.065] | 0.22 |
| PA | −0.373 | 0.11 | 0.077 | 0.14 | −0.065 | [−0.131, 0.001] | 0.0056 |
| PJ | +0.266 | 0.25 | 0.21 | 0.29 | +0.046 | [−0.020, 0.113] | 0.0014 |
| PM | −0.150 | 0.52 | 0.48 | 0.54 | −0.026 | [−0.093, 0.041] | 0.0003 |
| AJ | −0.414 | 0.11 | 0.11 | 0.12 | −0.072 | [−0.144, 0.000] | 0.015 |
| AM | +0.074 | 0.77 | 0.73 | 0.75 | +0.013 | [−0.060, 0.086] | 0.0002 |
| JM | +0.168 | 0.51 | 0.48 | 0.51 | +0.029 | [−0.043, 0.102] | 0.0009 |
| block | +0.313 | 0.22 | 0.22 | 0.24 | +0.054 | [−0.019, 0.128] | 0.0057 |

Reading: with the aliases separated, **no two-factor interaction is significant** in any of the three fits (PA
p 0.08–0.14, AJ ≈ 0.11); the fraction's PA signal (p 0.034) does not survive de-aliasing. Main effects are unchanged
from the fraction: planner P positive (Δ +0.15), M negative (Δ −0.13), assessment A and pruning J TOST-equivalent at
±0.167 (unadjusted; Holm at the final read-out). The quadrature optimiser stopped on precision loss (`converged:
false`), but its estimates agree with both R fits to within 0.1 in β and in every significance call. glmmTMB's
intercept SE is NaN (non-positive-definite Hessian), the slopes are unaffected.

## Addendum (Wed 28 Oct ≈ 22:30Z): RV1 evaluated on the relaunched A17 rows — **does not fire**

Rule (A17 §6, Part IV): RV1 fires if a rung r ∈ {Q0.8, Q2, Q4} has Δ_r = pooled `rec_sel`(r) − pooled `rec_sel`(A05
Full f42 r0) ≥ −δ on SYN-P 7 × seed 42, ITT zeros included; floor guard: no trigger if pooled O120 `rec_sel` ≤ δ.
δ = 0.167 (Amendment 12). Rows of record: Amendment 17 relaunch (A17-401…468, all final for Q0.8, Q2, Q4);
detector of record via `recovery_rows.py` (episodes on CAIR `~/ro27_analysis/rv1/rv1_episodes_final.csv`).
Inf-guard re-runs (Amendment 19) are not used, as Amendment 19 fixed in advance.

| Rung (seed 42, SYN-P 7) | Episodes | Pooled `rec_sel` | Δ vs Full | ≥ −0.167? |
|---|---|---|---|---|
| A05 Full f42 r0 (A05-031…037) | 35 | 0.400 (14/35) | — | floor guard passes (0.400 > δ) |
| Q0.8 | 35 | 0.000 | −0.400 | no |
| Q2 | 35 | 0.000 | −0.400 | no |
| Q4 | 35 | 0.143 (5/35: p2 1/5, p6a 4/5) | −0.257 | no |
| Q9 (descriptive) | 35 | 0.000 (8/9 rows inf-crash ITT) | −0.400 | — |
| Q122 (descriptive) | 35 | 0.343 (p2 4/5, p4 1/5, p6a 5/5, p6b 2/5) | −0.057 | — (not an RV1 rung) |

**Verdict: RV1 does not fire; the 14 fallbacks-off rows are not run and `ROGUE_ONE_FALLBACKS=off` is not built.**
"Small suffices" cannot be claimed for any rung ≤ 4B. Reading: Q0.8 and Q2 complete the loop (16/16 and 9/9 OK) but
recover no planted feature in any episode; Q4 recovers only on the shallow p6a (plus one p2 fold); Q122 comes within
δ of Full (descriptive, outside RV1). All seeds pooled: Q0.8 0.000 (70 eps), Q4 0.157 (70), Q122 0.286 (70).

## Addendum (Thu 29 Oct ≈ 00:00Z): A06 interim (sentinel drift) — **no drift**

`scripts/agent2027/analysis/sentinel_drift.py` (commit `f822416`), bands frozen at G2 (`g2_constants.json`), outputs
on CAIR in `~/ro27_analysis/sentinel/`. Per session: O120 sentinels A06-001…027 → 22 PASS, 5 WARN (all on the time
channel; A06-021/025 are solo retries, faster by construction), 0 FLAG; no flag or `vllm_image_id` differences. Q4:
A06-101 is the reference, A06-102/103 not evaluable (anchor A17-031 never ran; every Q4 sentinel capped, as Amendment
3). Interim pooled read (A06-T16): Fisher 89/135 (0.659) vs Z = 5/10, p = 0.322; fold-stratified exact p = 0.498;
run permutation p = 0.54; trend p = 0.86. **No evidence of server drift across sessions.**
