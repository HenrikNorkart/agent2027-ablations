# E8a calibration and screen report (SYN-P, master seed 2027, n = 2,000)

Produced by `scripts/agent2027/synp/calibrate_synp.py` and `screen_pairwise.py` (CPU only).
Headroom model: XGBoost with the Tester's assessment hyper-parameters (300 trees, lr 0.05, depth 6,
subsample/colsample 0.9, hist), 5-fold CV, fold seeds {42, 4, 5}. Classification = balanced-accuracy gain of
`raw + phi` over `raw` (pass >= 0.06). Regression = relative RMSE reduction (pass >= 0.15). The calibration scan
picks the weakest signal whose seed-42 headroom reaches the aim (0.075 / 0.18); the table gives the 3-seed mean (min).

| task | type | signal parameter | target | achieved mean (min) | pass |
|---|---|---|---|---|---|
| synp_p0 | R | noise 0.5 | none by design (\|h\| < 0.02) | 0.0034 (-0.0044) | yes |
| synp_p1 | R | noise 0.1 | >= 0.15 | 0.1969 (0.1691) | yes |
| synp_p2 | R | noise 0.25 | >= 0.15 | 0.2200 (0.2153) | yes |
| synp_p3 | C | beta 1.5 (ratio threshold = 80th percentile) | >= 0.06 | 0.0963 (0.0930) | yes |
| synp_p4 | C | beta 0.5 | >= 0.06 | 0.0746 (0.0713) | yes |
| synp_p5 | C | flip 0.0 (deterministic rule) | >= 0.06 | 0.2970 (0.2854) | yes |
| synp_p6a | R | noise 0.25 | >= 0.15 | 0.1966 (0.1749) | yes |
| synp_p6b | R | same data as P6a (hidden rescaling) | >= 0.15 | 0.2019 (0.1748) | yes |
| synp_p7 | C | beta 1000 (hard rule), prevalence 0.1 | >= 0.06 | 0.1045 (0.0953) | yes |

P7 `phi_wrong` headroom (must be < 0.01): -0.0018 (ok).

## Pairwise screen (P6a and P7 are gated)

| task | max \|rho\| (best op) | max NMI | result |
|---|---|---|---|
| synp_p6a (Reynolds number v*D/nu) | 0.818 (v / nu) | n/a | pass (< 0.95) |
| synp_p7 | 0.799 (log(corrosion) / anchor_spacing^2) | 0.382 | pass (< 0.95 / < 0.9) |

P7 extras: ratio alone rho 0.297, NMI 0.382 (does not meet the detector criterion); worst threshold guess on the
grid of round quantiles and round numbers: `ratio * 1[c >= q40]` with rho 0.927 and NMI 0.697, i.e. below both
cut-offs (margin 0.023 on rho). A BMI-like w / h^2 with realistic ranges fails the screen (unit test).

## What had to be tuned (honest log)

* First calibration pass: P3 and P7 missed (headroom 0.02 and 0.005): XGBoost recovers a ratio threshold of two
  independent log-normal columns almost perfectly. Changes: (i) the two ratio inputs of P3 and P7 are correlated
  (latent rho 0.85), so the ratio varies only along the thin direction orthogonal to the data cloud, which axis-aligned
  trees approximate badly; (ii) the P3 ratio threshold moved from the median to the 80th percentile (phi prevalence 9%);
  (iii) P7 labels are a hard rule on phi with 10% positives. The classification base-rate rule was also changed from
  "force P(y=1) = 0.5" (which made binary-phi labels noisy even at beta = infinity) to a cut half-way between the two phi
  levels (binary phi) or at the (1 - prevalence) quantile (continuous phi).
* P5 first selected 30% label flips because the scan runs weakest-signal-first; for the deterministic rule the flip grid
  now runs cleanest-first, which keeps flip = 0.
* Regression grid refined (0.35, 0.25, 0.15) so that P2 and P6a are not made easier than needed.
