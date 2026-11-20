# Trace analysis (paper Section IV-E, Table IV) — exploratory

Post hoc, outside the confirmatory Holm family. Hypothesis cycles of the complete system (A05/C) and the
single-agent configuration (A16/S) on P2, P4 and P6b were annotated with the MAST taxonomy (Cemri et al., 2025).

| path | content |
|---|---|
| `MAST_CODEBOOK.md` | codebook given to both annotators: MAST modes, their operationalisation for Rogue One, the unit (one hypothesis cycle), and the added `missed_signal` and `handoff_fidelity` notes |
| `packets/U01.md`..`U48.md` | the 48 blinded annotation packets (24 per configuration), one hypothesis cycle each |
| `packets_key.csv` | unblinding key: unit -> configuration, run, task, fold, iteration, sampling stratum |
| `annot_A/`, `annot_B/` | labels and quoted evidence of the two independent annotators (`annot_B_write.py` writes annotator B's labels) |
| `mast_consensus.csv` | labels on which both annotators agree (Table IV) |
| `kappa.py` | per-mode and pooled Cohen's kappa (pooled 0.79) |
| `unblind.py` | joins labels with the key; counts per configuration and Fisher tests (printed) |
| `flow/` | information-flow pipeline (cycle tables, funnel, Planner-pointing rates used in the A7 text); see `flow/README.md` |
| `capacity/` | tool-error and prompt-token profiles per role (A7 prompt tokens, A8 tool errors and replay); `derived/` holds the tables, `cair_a18/final_1102/` the A18 replay verdicts |

Run `kappa.py` and `unblind.py` from this folder. The `flow/` and `capacity/` scripts additionally need the raw run
outputs (`llm_traces.jsonl.gz`, about 300 MB, not released); set `MAST_DATA` (and `SYNP_META` for task metadata) to a local copy.
The derived tables they produced are included.
