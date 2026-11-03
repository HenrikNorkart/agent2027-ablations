# AGENT 2027 replication package

Replication package for **"Which Parts of a Multi-Agent Loop Earn Their Tokens? Pre-registered Component Ablations of an LLM Feature-Engineering System"** (AGENT 2027 workshop, ICSE 2027).

- `docs/agent2027/preregistration.md`: the pre-registration and all dated amendments.
- `docs/agent2027/results/final_20261102/`: per-run rows, fold-episode recovery scores and the final read-out (GLMM/Holm, sentinel, Pareto, single-shot and anytime summaries).
- `manifests/agent2027/`: run manifests.
- `scripts/agent2027/analysis/`: the analysis scripts that generate every number, table and figure in the paper (`final_rows.py`, `final_tests.py`, `final_glmm.R`, `make_agent2027_tex.py`, `make_ieee.py`).
- `rogueone/`, `rogue_one_main.py`, `docker/`: the Rogue One system under test.

This is a history-free snapshot of the development branch `agent2027/v2`. Run-finished notifications read an optional `ROGUE_ONE_WEBHOOK_URL` environment variable.

---

# Rogue One

Reference implementation for the paper **Knowledge-Informed Automatic Feature Extraction via Collaborative Large Language Model Agents** (arXiv:2511.15074).

Rogue One is a multi-agent system that iteratively proposes, justifies, and evaluates new tabular features under a domain-knowledge-grounded loop. Three LLM agents — **Scientist**, **Extractor**, and **Tester** — collaborate over a fixed cross-validation harness; every proposed feature carries a free-text *argumentation* sourced from a Knowledge / Web-Search retrieval arm. Both classification and regression tasks are supported on the same code path.

---

## Architecture

```
                 ┌────────────────────────┐
                 │   Scientist Agent      │  picks next investigation focus
                 │   (focus selection)    │
                 └──────────┬─────────────┘
                            │ focus
                            ▼
   ┌───────────────────────────────────────────────────┐
   │  Extractor Agent                                  │
   │   • proposes new candidate features               │
   │   • computes columns on df_entities_attributes    │
   │   • each feature has an argumentation sourced     │
   │     from Knowledge Agent / Web Search             │
   └──────────┬────────────────────────────────────────┘
              │ feature pool
              ▼
   ┌───────────────────────────────────────────────────┐
   │  Tester Agent                                     │
   │   • trains XGBoost over 5-fold CV                 │
   │   • prunes weak features                          │
   │   • returns metrics + qualitative report          │
   └──────────┬────────────────────────────────────────┘
              │ metrics, report
              └──> back to Scientist for the next iteration
```

Supporting agents:
- **Knowledge Agent** — retrieval-augmented lookup over a local Chroma vector DB.
- **Web Search Agent** — online evidence lookup via the Serper API (optional, controlled by `SERPER_API_KEY`).

The full agent loop is implemented in `rogue_one_main.py:RogueOneAgentNetwork.forward_pass`.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `rogue_one_main.py` | Experiment orchestration entry point |
| `rogueone/` | Python package |
| `rogueone/llm/agents/` | Public agent imports (compatibility wrappers) |
| `rogueone/llm/agents/core/` | Agent implementations (Scientist, Extractor, Tester, Knowledge) |
| `rogueone/utils/config.py` | `ExperimentConfig` + runtime singletons (`LLMConfig`, `WandBConfig`, …) |
| `tasks/` | Per-dataset YAML configs and CSV/JSON fixtures |
| `tests/` | Pytest suite (unit + integration; see `tests/README.md`) |
| `scripts/` | Helper scripts: vLLM launch wrappers, smoke validation |

---

## Installation

```bash
git clone <anonymous-repo-url>
cd RougeOne
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Tested with Python 3.10–3.12.

---

## Environment configuration

```bash
cp .env.example .env
```

The minimum required keys are the LLM and embedding endpoints:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_MODEL` | `openai/gpt-oss-20b` | Generation model id |
| `LLM_PORT` | `8000` | Port of the local OpenAI-compatible server |
| `LLM_API_KEY` | `rogueone-dev` | API key for that endpoint |
| `EMBEDD_MODEL` | `Qwen/Qwen3-Embedding-4B` | Embedding model id |
| `EMBEDD_PORT` | `8011` | Port of the local embedding server |
| `EMBEDD_API_KEY` | `change-me` | API key for the embedding endpoint |
| `WANDB_API_KEY` | `""` | Optional — set to enable W&B logging |
| `SERPER_API_KEY` | `""` | Optional — enables the web-search arm |

All Rogue One runtime knobs are also driven through environment variables; see `.env.example` for the full list with defaults and `rogueone/utils/config_keys.py` for the canonical key registry. Per-dataset settings (target column, label file, class balance, knowledge DB collection) are stored in `tasks/<task_type>/<dataset>/config.yml` and parsed into `ExperimentConfig`.

### Bringing up the local LLM endpoint

The paper's results were produced against `gpt-oss-120b` served by vLLM. Convenience launch scripts:

```bash
bash scripts/run_vllm_oss120.sh          # gpt-oss-120b (used in the paper)
bash scripts/run_vllm_for_dev.sh         # smaller dev model
bash scripts/run_vllm_qwen3_embedd.sh    # Qwen3 embedding model
```

These spin up an OpenAI-compatible server on `http://localhost:8000`. All Rogue One agents talk to that endpoint via a small client wrapper (`rogueone/llm/clients.py`) that fixes a vLLM/`openai-agents` interaction (vLLM returns `logprobs: null`, which trips the agent library's Pydantic validation — the wrapper strips the field on the wire).

---

## Running an experiment

> **Hard rule: covtype is never run, in any study** (Henrik, 2026-10-16: "COVTYPE should never be run!"). This applies to every Rogue One study, not only AGENT 2027: no launcher, dataset list, manifest, session or retry may include it. The launchers in `paper/scripts/` and `scripts/agent2027/` refuse it and exit non-zero. Existing covtype results stay as historical records, but nothing may re-run it.

The default entry point runs the full task list configured at the bottom of `rogue_one_main.py`:

```bash
python rogue_one_main.py
```

Each task spawns one `RogueOneAgentNetwork` and runs `k_folds × num_iterations` trials. Results are streamed to W&B (if enabled) and dumped under `figures_for_testing/` and `tasks/<task>/output/`.

### Quick smoke validation

One classification + one regression dataset, intended to verify a fresh checkout end-to-end:

```bash
bash scripts/smoke_refactor_validation.sh
```

### Disabling W&B

```bash
ROGUE_ONE_DISABLE_WANDB=1 python rogue_one_main.py
```

or pass `enable_wandb=False` to `RogueOneAgentNetwork(...)`.

---

## Reproducing the paper's results

The paper reports results from a 5-fold CV with deterministic per-fold test-id sets (seed `42`). The full set of reproducibility hyperparameters — generation temperature, reasoning effort, max iterations per agent, the XGBoost evaluator settings, embedding model, ablation switches, and per-dataset config files — is documented in **Appendix B (Reproducibility) of the paper**.

Compute scale of the headline runs:
- 8 × NVIDIA H100 80GB cluster (vLLM tensor-parallel hosting `gpt-oss-120b`)
- ~70 GPU-hours per Full-system pass over all benchmark datasets
- ~360 GPU-hours total across all reported configurations

Smaller models (e.g., `gpt-oss-20b` via the dev wrapper) reproduce the qualitative dynamics on a single 80GB GPU at a fraction of the runtime, but do not match the headline numbers.

---

## Testing

Tests are executed with `pytest`:

```bash
pytest tests/                  # full suite (requires localhost:8000 LLM endpoint)
pytest tests/unit/             # unit only — no LLM needed
pytest -m "not requires_llm"   # skip LLM-dependent tests
pytest --cov=rogueone --cov-report=term-missing
```

Markers (declared in `pytest.ini` with `--strict-markers`): `unit`, `integration`, `requires_llm`, `agent`, `config`, `task_io`, `slow`.

Tests that require the live LLM endpoint are explicitly marked `@pytest.mark.requires_llm` and are designed to **fail loudly** when the endpoint is unreachable — that is intentional, not a flake.

See `tests/README.md` for the full testing policy and structure.

---

## Stable public imports

The top-level agent imports are stable; internal modules under `rogueone/llm/agents/core/` may evolve.

```python
from rogueone.llm.agents import ExtractorAgent, ScientistAgent, TesterAgent
```

---

## License

Rogue One is released under the Apache License 2.0 — see [`LICENSE`](LICENSE). The code interacts with `gpt-oss-120b` (Apache 2.0) and `Qwen3-Embedding-4B` (Apache 2.0); benchmark datasets retain their original upstream licenses (UCI / OpenML / Kaggle — listed per dataset in the paper's reproducibility appendix).

---

## Citation

```bibtex
@inproceedings{anonymous_rogueone_2026,
  title  = {Knowledge-Informed Automatic Feature Extraction via Collaborative Large Language Model Agents},
  author = {Anonymous},
  booktitle = {Submitted to NeurIPS 2026 (under double-blind review)},
  year   = {2026}
}
```
