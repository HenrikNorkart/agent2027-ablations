"""Minimal W&B fetch for AGENT 2027 analyses (E3; E11b extends it).

Fetches runs of ``<entity>/<project>`` (filtered by group / run name / run id), writes
each run's ``config.json`` and ``summary.json`` and downloads its ``run-output``
artifact (the whole ``output/`` folder: ``llm_traces.jsonl.gz``, ``focus_log.jsonl``,
``iteration_log.jsonl``, ``intermediate_steps/``, ``prompts/``, error JSONL, ...).

It also reads a downloaded (or local) ``output/`` folder and checks the E3 contract
(``--check``): trace records vs ``llm_calls_total``, focus rows, iteration rows,
per-trial tables, A18 trace fields.

Usage (inside the analysis container)::

    python fetch_wandb.py --project RogueOne-AGENT2027 --group A05 --out ./fetched
    python fetch_wandb.py --read ./fetched/<run>/run-output --check

``wandb`` is imported only when fetching; ``read_run_output`` is stdlib only. Per-call
histories must be read with ``scan_history`` (``history()`` down-samples to 500 rows).
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_ENTITY = "PlanBench"
DEFAULT_PROJECT = "RogueOne-AGENT2027"
ARTIFACT_TYPE = "run-output"

A18_TRACE_FIELDS = (
    "agent_name caller stage index fold_run_id iteration_in_fold session_name call_seq "
    "messages tools tool_choice extra_body temperature max_tokens model seed "
    "message finish_reason usage"
).split()


# --------------------------------------------------------------------------- reading
def _find(root: Path, name: str) -> Optional[Path]:
    direct = root / name
    if direct.exists():
        return direct
    hits = sorted(root.rglob(name))
    return hits[0] if hits else None


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[arg-type]
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_run_output(path: str | Path) -> dict[str, Any]:
    """Load an ``output/`` folder (or an unpacked ``run-output`` artifact).

    Returns ``{"root", "summary", "traces", "focus", "iterations", "trial_tables",
    "files"}``; missing pieces are ``None`` / empty.
    """
    root = Path(path)
    out: dict[str, Any] = {"root": str(root), "summary": None, "traces": None, "focus": None,
                           "iterations": None, "trial_tables": {}, "files": []}
    out["files"] = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    for key, name in (("summary", "instrumentation_summary.json"),):
        p = _find(root, name)
        if p:
            out[key] = json.loads(p.read_text(encoding="utf-8"))
    for key, name in (("traces", "llm_traces.jsonl.gz"), ("focus", "focus_log.jsonl"),
                      ("iterations", "iteration_log.jsonl")):
        p = _find(root, name)
        if p:
            out[key] = read_jsonl(p)
    steps = _find(root, "intermediate_steps")
    if steps:
        for f in sorted(steps.glob("df_attributes_explanations_trail_*.csv")):
            out["trial_tables"].setdefault(_trial_index(f.name), {})["attributes"] = str(f)
        for f in sorted(steps.glob("df_scientist_feedback_trail_*.csv")):
            out["trial_tables"].setdefault(_trial_index(f.name), {})["feedback"] = str(f)
    return out


def _trial_index(name: str) -> int:
    return int(re.search(r"trail_(\d+)\.csv$", name).group(1))


def check_run_output(data: dict[str, Any]) -> list[str]:
    """E3 done-when checks; returns the list of problems (empty = all good)."""
    problems: list[str] = []
    summary = data.get("summary") or {}
    traces = data.get("traces")
    if not summary:
        problems.append("instrumentation_summary.json missing")
    if traces is None:
        problems.append("llm_traces.jsonl.gz missing")
    else:
        if summary and summary.get("llm_calls_total") != len(traces):
            problems.append(f"trace records ({len(traces)}) != llm_calls_total ({summary.get('llm_calls_total')})")
        bad = [i for i, r in enumerate(traces) if any(f not in r for f in A18_TRACE_FIELDS)]
        if bad:
            problems.append(f"{len(bad)} trace records miss A18 fields (first: {bad[0]})")
        if traces and summary:
            compl = sum(
                int((r.get("usage") or {}).get("completion_tokens") or 0)
                + int((r.get("discarded_usage") or {}).get("completion_tokens") or 0)
                for r in traces
            )
            total = summary.get("tok_compl_total") or 0
            if total and abs(compl - total) > 0.01 * total:
                problems.append(f"trace completion tokens ({compl}) differ from tok_compl_total ({total}) by >1%")
    for role in ("scientist", "extractor", "tester"):
        if summary and not (summary.get(f"tok_compl_{role}") or 0) > 0:
            problems.append(f"tok_compl_{role} is zero")
    iters = data.get("iterations") or []
    focus = data.get("focus")
    if focus is None:
        problems.append("focus_log.jsonl missing")
    elif iters and len(focus) != len(iters):
        problems.append(f"focus rows ({len(focus)}) != iteration rows ({len(iters)})")
    missing = [i for i in range(len(iters)) if i not in data.get("trial_tables", {})]
    if iters and missing:
        problems.append(f"no per-trial tables for {len(missing)} iterations (dump_data not reached on skipped paths is expected)")
    return problems


# --------------------------------------------------------------------------- fetching
def fetch_runs(
    entity: str,
    project: str,
    out_dir: str | Path,
    group: Optional[str] = None,
    names: Optional[list[str]] = None,
    run_ids: Optional[list[str]] = None,
    download: bool = True,
) -> list[Path]:
    """Download config, summary and the ``run-output`` artifact of the matching runs."""
    import wandb  # lazy: only needed on the analysis container

    api = wandb.Api()
    filters: dict[str, Any] = {}
    if group:
        filters["group"] = group
    if names:
        filters["display_name"] = {"$in": names}
    if run_ids:
        filters["name"] = {"$in": run_ids}
    written: list[Path] = []
    for run in api.runs(f"{entity}/{project}", filters=filters or None):
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", run.name)
        d = Path(out_dir) / safe
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps(dict(run.config), indent=2, default=str), encoding="utf-8")
        (d / "summary.json").write_text(json.dumps(dict(run.summary), indent=2, default=str), encoding="utf-8")
        (d / "meta.json").write_text(
            json.dumps({"id": run.id, "name": run.name, "state": run.state, "group": run.group,
                        "job_type": run.job_type, "created_at": str(run.created_at)}, indent=2),
            encoding="utf-8",
        )
        if download:
            for art in run.logged_artifacts():
                if art.type == ARTIFACT_TYPE:
                    art.download(root=str(d / "run-output"))
                    break
        written.append(d)
    return written


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default=DEFAULT_ENTITY)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--group")
    ap.add_argument("--run", action="append", dest="runs", help="run display name (repeatable)")
    ap.add_argument("--run-id", action="append", dest="run_ids")
    ap.add_argument("--out", default="./fetched")
    ap.add_argument("--no-download", action="store_true", help="only config and summary")
    ap.add_argument("--read", help="read a local output/ or run-output folder instead of fetching")
    ap.add_argument("--check", action="store_true", help="with --read: run the E3 contract checks")
    args = ap.parse_args(argv)
    if args.read:
        data = read_run_output(args.read)
        print(json.dumps({k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in data.items()
                          if k != "summary"}, indent=2))
        if args.check:
            problems = check_run_output(data)
            for p in problems:
                print("PROBLEM:", p)
            return 1 if problems else 0
        return 0
    paths = fetch_runs(args.entity, args.project, args.out, args.group, args.runs, args.run_ids,
                       download=not args.no_download)
    print(f"fetched {len(paths)} runs into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
