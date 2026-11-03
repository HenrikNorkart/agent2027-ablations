#!/usr/bin/env python3
"""A08-T10 · one W&B run per A08 manifest row (01 §7 R12/R13; A08 §5.2).

Entity ``PlanBench``, project ``RogueOne-AGENT2027``, group ``A08``, run name
``A08__<arm_id>__<dataset_id>__f<fold_seed>__r<replicate>``, job_type = arm, tags
[``<SESSION_ID>`` or ``cpu_screen``, ``prefreeze``/``v2``, image tag]. The run id is derived
from the run name, so re-logging a row UPDATES its run instead of creating a duplicate.

Config: the R12 identity / version keys (``llm_seed`` and ``rung_*`` empty for CPU rows) and the
comparator settings as ``comparator/<NAME>``. Summary (R13 names): ``rec_sel``, ``rec_any``
(fold-episode means; per-fold values in the episode table), ``heldout_score`` (``accw_balanced``
for classicfe classification, ``acc_balanced`` for the LLM-FE scorer, ``nrmse`` for regression)
and ``heldout_score_sd``, ``n_features`` per fold, ``wall_s``, ``status``/``cap_hit``/
``itt_imputed``, and for GPU rows the X5 proxy tokens ``tok_prompt_total``, ``tok_compl_total``,
``llm_calls_total`` (+ ``context_errors`` = HTTP 400 + finish_reason=length). ``energy_j`` is
the session's E3 energy run (``<SID>__energy``); a per-row share needs A23's apportioning, so it
is not set here. Every input that exists for a row goes into artifact ``run-output``.

Kinds:
  ``--kind cpu``    rows A08-001…081. Inputs: ``--cells-dir`` (classicfe out: ``f<seed>/arm_<ds>_<arm>_stratified.json``,
                    arm ``raw`` = ``base_no_fe``) and ``--episodes`` (the detector table,
                    paper/results/agent2027/a08_screen_episodes.csv).
  ``--kind llmfe``  rows A08-082…095 of one session. Inputs: ``--session-file``, ``--status``
                    (a08_comparators_clients.sh status.tsv), ``--proxy-totals`` (X5 totals.json),
                    optional ``--llmfe-logs`` (~/llmfe/logs, sample counts + samples artifact),
                    ``--scores-dir`` (llmfe_score_rogueone.py JSONs, notebook mode) and
                    ``--episodes`` (comparator_parser episode table, once X6 exists).

Runs in a container on CAIR (``rogue_one:agent2027-v2.N``, ``--env-file`` PlanBench ``.env``);
``--dry-run`` prints the payloads and needs no W&B. covtype rows are refused.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

ENTITY = "PlanBench"
PROJECT = "RogueOne-AGENT2027"
CPU_ARM_TO_CLASSICFE = {"base_no_fe": "raw", "openfe": "openfe", "autofeat": "autofeat"}
COMPARATOR_SETTINGS = {
    "base_no_fe": {"impl": "classicfe_arm.py raw", "image": "classicfe:4", "evaluator": "XGBoost 400 trees, random_state 42",
                   "folds": "stratified (C) / KFold (R)"},
    "openfe": {"impl": "classicfe_arm.py openfe", "image": "classicfe:4", "openfe": "0.0.12", "OPENFE_NJ": 1,
               "features_kept": "all returned", "folds": "stratified (C) / KFold (R)"},
    "autofeat": {"impl": "classicfe_arm.py autofeat", "image": "classicfe:4", "autofeat": "2.1.3",
                 "feateng_steps": 2, "n_jobs": 4, "seeded": False, "folds": "stratified (C) / KFold (R)"},
    "llmfe": {"impl": "LLM-FE e9a58ca main.py (code unchanged)", "image": "llmfe:oss120",
              "backbone": "openai/gpt-oss-120b (vLLM v0.22.0, TP2)", "LLMFE_MAX_TOKENS": 2048,
              "budget": "native: max_sample_nums 20x5 -> 20 samples per split, samples_per_prompt 3",
              "temperature": "server default", "spec": "X4 (authored by us; scripts/agent2027/llmfe_specs)",
              "regression_problem_name_alias": "crab", "split": "LLM-FE's own (seed 42) = Rogue One seed-42 partition",
              "selection_primary": "notebook", "scorer": "paper/scripts/llmfe_score_rogueone.py",
              "web_search": False},
}


def run_name(r: dict) -> str:
    return f"A08__{r['arm_id']}__{r['dataset_id']}__f{r['fold_seed']}__r{r['replicate']}"


def run_id(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "-", name).lower()[:128]


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [dict(r) for r in csv.DictReader(fh)]
    for r in rows:
        if "covtype" in r.get("dataset_id", "").lower():
            raise SystemExit("FATAL: covtype is never run or logged (hard rule 2026-10-16)")
    return [r for r in rows if r.get("ablation_id") == "A08"]


def _episodes(path: Path | None, arm_names: tuple[str, ...]) -> dict:
    """(dataset, arm, fold_seed) -> list of episode dicts (detector table)."""
    out: dict = {}
    if not path or not Path(path).is_file():
        return out
    with open(path, newline="", encoding="utf-8") as fh:
        for e in csv.DictReader(fh):
            if e.get("arm") in arm_names:
                key = (e["dataset_id"], e["arm"], str(e.get("fold_seed", "42")), str(e.get("replicate", "")))
                out.setdefault(key, []).append(e)
    return out


def _mean(xs):
    xs = [float(x) for x in xs if x not in ("", None)]
    return sum(xs) / len(xs) if xs else None


def base_config(r: dict, session: str, phase: str) -> dict:
    return {
        "ablation_id": "A08", "arm_id": r["arm_id"], "dataset_id": r["dataset_id"], "task_type": r["task_type"],
        "fold_seed": int(r["fold_seed"]), "llm_seed": r.get("llm_seed", ""), "replicate": int(r["replicate"]),
        "is_sentinel": int(r.get("is_sentinel") or 0), "session_id": session, "manifest_row": r["row_id"],
        "code_phase": phase, "image_tag": r.get("image_tag", ""),
        "git_sha": os.environ.get("ROGUE_ONE_GIT_SHA", ""), "image_id": os.environ.get("ROGUE_ONE_IMAGE_ID", ""),
        "vllm_image_id": os.environ.get("ROGUE_ONE_VLLM_IMAGE_ID", ""), "prereg_tag": os.environ.get("ROGUE_ONE_PREREG_TAG", ""),
        "rung_S": r.get("rung_S", ""), "rung_E": r.get("rung_E", ""), "rung_T": r.get("rung_T", ""),
        "model_S": "openai/gpt-oss-120b" if r.get("rung_S") == "O120" else "",
        "model_E": "openai/gpt-oss-120b" if r.get("rung_E") == "O120" else "",
        "model_T": "openai/gpt-oss-120b" if r.get("rung_T") == "O120" else "",
        "model_W": "", "server_topology": "uniform:O120@proxy" if r.get("rung_E") else "cpu",
        "parallel": "", **{f"comparator/{k}": v for k, v in COMPARATOR_SETTINGS.get(r["arm_id"], {}).items()},
    }


def cpu_payloads(rows, cells_dir: Path | None, episodes_path: Path | None) -> list[dict]:
    eps = _episodes(episodes_path, tuple(CPU_ARM_TO_CLASSICFE.values()))
    out = []
    for r in rows:
        if r["arm_id"] not in CPU_ARM_TO_CLASSICFE:
            continue
        cf_arm = CPU_ARM_TO_CLASSICFE[r["arm_id"]]
        files, summary = [], {}
        cell = (Path(cells_dir) / f"f{r['fold_seed']}" / f"arm_{r['dataset_id']}_{cf_arm}_stratified.json") if cells_dir else None
        if cell and cell.is_file():
            doc = json.loads(cell.read_text(encoding="utf-8"))
            files.append(str(cell))
            key = "accw_balanced" if r["task_type"] == "classification" else "nrmse"
            summary.update(heldout_score=doc.get(key), heldout_score_sd=doc.get(f"{key}_sd"),
                           heldout_metric=key, folds_done=doc.get("folds_done"), wall_s=doc.get("secs"),
                           n_features=doc.get("n_features"), error=doc.get("error"))
            summary["status"] = "OK" if doc.get("folds_done") == 5 else "MISSING"
        else:
            summary["status"] = "MISSING"
        ep = eps.get((r["dataset_id"], cf_arm, str(r["fold_seed"]), ""), [])
        if ep:
            summary.update(rec_sel=_mean(e["rec_sel"] for e in ep), rec_any=_mean(e["rec_any"] for e in ep),
                           n_episodes=len(ep), rec_sel_episodes=sum(int(e["rec_sel"]) for e in ep))
        out.append({"name": run_name(r), "row": r, "tags": ["cpu_screen", "prefreeze", r.get("image_tag", "")],
                    "config": base_config(r, "", "prefreeze"), "summary": summary,
                    "episodes": ep, "files": files})
    return out


def _status_rows(path: Path | None) -> dict:
    last: dict = {}
    if path and Path(path).is_file():
        with open(path, newline="", encoding="utf-8") as fh:
            for s in csv.DictReader(fh, delimiter="\t"):
                last[s["row_id"]] = s  # the last attempt wins
    return last


def llmfe_payloads(rows, session: str, status_path, totals_path, logs_dir, scores_dir, episodes_path) -> list[dict]:
    status = _status_rows(status_path)
    totals = json.loads(Path(totals_path).read_text(encoding="utf-8")) if totals_path and Path(totals_path).is_file() else {}
    eps = _episodes(episodes_path, ("llmfe",))
    out = []
    for r in rows:
        if r["arm_id"] != "llmfe" or (session and r.get("session_id") != session):
            continue
        key = f"a08-llmfe-{r['dataset_id']}-r{r['replicate']}"
        suffix = f"oss120_a08r{r['replicate']}"
        st = status.get(r["row_id"], {})
        summary = {"status": st.get("status", "NOT_RUN"), "attempts": int(st.get("attempt") or 0),
                   "wall_s": int(st["wall_s"]) if st.get("wall_s", "").isdigit() else None,
                   "splits_ok": st.get("splits_ok"), "cap_hit": int(st.get("status") == "CAP_HIT"),
                   "itt_imputed": int(st.get("status") in ("CAP_HIT", "FAILED"))}
        t = totals.get(key)
        if t:
            summary.update(tok_prompt_total=t["prompt_tokens"], tok_compl_total=t["completion_tokens"],
                           llm_calls_total=t["calls"], llm_calls_error=t["error_calls"],
                           context_errors=t["http_400"] + t["finish_length"], tok_usage_missing=t["usage_missing"])
        files = []
        if logs_dir:
            n_samples = []
            for i in range(1, 6):
                d = Path(logs_dir) / f"{r['dataset_id']}_{suffix}_split_{i}" / "samples"
                n_samples.append(len(list(d.glob("samples_*.json"))) if d.is_dir() else 0)
                if d.is_dir():
                    files.append(str(d))
            summary["n_samples_per_split"] = n_samples
        if scores_dir:
            sc = Path(scores_dir) / f"llmfe_{r['dataset_id']}_stratified.json"
            if not sc.is_file():
                sc = Path(scores_dir) / f"llmfe_{r['dataset_id']}_kfold.json"
            if sc.is_file():
                doc = json.loads(sc.read_text(encoding="utf-8"))
                k = "acc_balanced" if r["task_type"] == "classification" else "nrmse"
                summary.update(heldout_score=doc.get(k), heldout_score_sd=doc.get(f"{k}_sd"), heldout_metric=k,
                               heldout_folds_done=doc.get("folds_done"),
                               n_features=[len(p.get("candidates_used") or []) for p in doc.get("per_fold", [])])
                files.append(str(sc))
        ep = eps.get((r["dataset_id"], "llmfe", str(r["fold_seed"]), str(r["replicate"])), []) or \
            eps.get((r["dataset_id"], "llmfe", str(r["fold_seed"]), ""), [])
        if ep:
            summary.update(rec_sel=_mean(e["rec_sel"] for e in ep), rec_any=_mean(e["rec_any"] for e in ep),
                           n_episodes=len(ep))
        cfg = base_config(r, session, "v2")
        cfg.update(client_key=key, llmfe_suffix=suffix)
        out.append({"name": run_name(r), "row": r, "tags": [session, "v2", r.get("image_tag", "")],
                    "config": cfg, "summary": summary, "episodes": ep, "files": files})
    return out


def log_payloads(payloads: list[dict], project: str, entity: str) -> None:
    import wandb

    for p in payloads:
        run = wandb.init(entity=entity, project=project, group="A08", job_type=p["row"]["arm_id"],
                         name=p["name"], id=run_id(p["name"]), resume="allow",
                         tags=[t for t in p["tags"] if t], config=p["config"], reinit=True)
        for k, v in p["summary"].items():
            if v is not None:
                run.summary[k] = v
        if p["episodes"]:
            cols = list(p["episodes"][0].keys())
            run.log({"episodes": wandb.Table(columns=cols, data=[[e.get(c) for c in cols] for e in p["episodes"]])})
        if p["files"]:
            art = wandb.Artifact("run-output-" + run_id(p["name"]), type="run-output")
            for f in p["files"]:
                if Path(f).is_dir():  # an LLM-FE split's samples/ dir -> <ds>_<suffix>_split_<i>/
                    art.add_dir(f, name=Path(f).parent.name)
                else:
                    art.add_file(f)
            run.log_artifact(art)
        run.finish()
        print(f"logged {p['name']} status={p['summary'].get('status')}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kind", choices=["cpu", "llmfe"], required=True)
    ap.add_argument("--manifest", type=Path, help="A08_M5.csv (cpu; default: the repo's)")
    ap.add_argument("--session-file", type=Path, help="sessions/<SID>.csv (llmfe)")
    ap.add_argument("--cells-dir", type=Path)
    ap.add_argument("--episodes", type=Path)
    ap.add_argument("--status", type=Path)
    ap.add_argument("--proxy-totals", type=Path)
    ap.add_argument("--llmfe-logs", type=Path)
    ap.add_argument("--scores-dir", type=Path)
    ap.add_argument("--rows", nargs="*", help="only these row ids")
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--entity", default=ENTITY)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.kind == "cpu":
        rows = read_rows(a.manifest or Path(__file__).resolve().parents[2] / "manifests/agent2027/A08_M5.csv")
        payloads = cpu_payloads(rows, a.cells_dir, a.episodes)
    else:
        if not a.session_file:
            ap.error("--kind llmfe needs --session-file")
        rows = read_rows(a.session_file)
        session = rows[0]["session_id"] if rows else ""
        payloads = llmfe_payloads(rows, session, a.status, a.proxy_totals, a.llmfe_logs, a.scores_dir, a.episodes)
    if a.rows:
        payloads = [p for p in payloads if p["row"]["row_id"] in set(a.rows)]
    if not payloads:
        print("nothing to log", file=sys.stderr)
        return 1
    if a.dry_run:
        for p in payloads:
            print(json.dumps({k: p[k] for k in ("name", "tags", "summary")} | {"n_files": len(p["files"]),
                             "n_episodes": len(p["episodes"])}, default=str))
        return 0
    log_payloads(payloads, a.project, a.entity)
    return 0


if __name__ == "__main__":
    sys.exit(main())
