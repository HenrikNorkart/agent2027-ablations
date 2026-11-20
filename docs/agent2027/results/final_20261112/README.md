# Final read-out with the seed-4 2B/9B extension (2026-11-12)

The 2026-11-10 read-out (`../final_20261110`) plus the 14 seed-4 rows of record of Amendments 19i/19j
(A17-701..703, 705, 708..713 from 19i; A17-715..718, the clean re-runs of the stalled A17-704/706/707/714, from 19j).

- `rows_final.csv`, `episodes_final.csv`, `funnel_episodes_5xx.csv`: the 11-10 tables with `rows_7xx.csv`, `ep_7xx.csv`
  and `funnel_7xx/funnel_episodes.csv` appended (`merge7xx.py`; scored on CAIR 2026-11-12 by `~/ro27_7xx_par.sh`).
- `final_readout.json`, `c5a_func_success.csv`, `holm_table.csv`: `final_tests.py report` (with the `seed4_ladder`
  block) on those tables, manifests `mflat4` (= `mflat3` + `A17_7xx.csv`); `glmm_results.json`, `describe.json` reused from 11-10.
- Changes against 11-10: new `contrasts.C5a.seed4_ladder` (exploratory) and `outside_holm.cost_synp` for Q2/Q9
  (more runs). C5a of record and all confirmatory results are unchanged (C5a raw p=0.0013, Holm 0.0092).
- seed4_ladder (crash re-runs swapped in, 14 runs per rung): functional success 0.8B 0/14, 2B 4/14, 4B 8/14,
  9B 14/14, 122B-A10B 11/14; recovery 0/70, 0/70, 10/70, 22/70, 25/70; exact stratified trend p=1.5e-9.
- Fig. 4: `python scripts/agent2027/analysis/make_ladder_fig.py --in docs/agent2027/results/final_20261112 --paper paper/agent2027`.
