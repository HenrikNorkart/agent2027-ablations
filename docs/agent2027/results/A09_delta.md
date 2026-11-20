# A09-T15: FX margin δ

Computed 2026-10-24 ≈ 20:20Z on CAIR (CPU container, `rogue_one:agent2027-v2.3`,
`scripts/agent2027/analysis/recovery_rows.py`, amended detector). Committed as Amendment 12, about 26 h after the
pre-registered deadline (Fri 23 Oct, 18:00); no A12 detector output existed before the commit.

| Task | Oracle (A09) | Full f42 r0 | Full f42 r1 |
|---|---|---|---|
| synp_p2 | 5/5 | 3/5 | 2/5 |
| synp_p3 | 5/5 | 0/5 | 0/5 |
| synp_p4 | 5/5 | 0/5 | 0/5 |
| synp_p5 | 5/5 | 0/5 | 0/5 |
| synp_p6a | 5/5 | 5/5 | 5/5 |
| synp_p6b | 5/5 | 1/5 | 4/5 |
| **Pooled** | **30/30 = 1.000** | **20/60 = 0.333** | |

gap = 1.000 − 0.333 = 0.667 → **δ = max(0.25 × 0.667, 0.10) = 0.167** (3 decimals).

Rows: A09-002…007 (`S1019a`), A05-032…037 and A05-048…053 (`S1016a/b/c`), all status OK, no ITT zeros.
The Full numbers equal `G2_recovery_by_task.csv` (f42r0, f42r1). Episodes: `A09_delta_episodes.csv`.
The W&B artifact `rt3-oracle-gap` is still to be posted (W&B API was unreachable from the laptop at commit time).
