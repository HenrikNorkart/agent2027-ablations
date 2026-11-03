"""A18 (RP1) role-wise replay: draw the prompt set and score the replies.

Subcommands (A18 §5; ``docs/ablations/A18_RP1_rolewise_replay.md``):

``sample`` (A18-T6, run in A18-T10)
    Fetches the 18 A05 seed-42 source runs (rows A05-001…010, A05-031…038; the A/A twin
    ``replicate=1`` replaces an ITT failure and the substitution is recorded), runs the
    heavy QC assertions of A18-T5, classifies every trace record into the §4 strata and
    draws the §4 allocation (800 prompts, round-robin over runs then iteration buckets
    {0–2, 3–6, 7–9}; W also by caller; string-seeded RNG, seed 42). Writes
    ``rp1-prompts-v0.jsonl.gz`` (replay order: a fixed shuffle, seed 42), the realised
    allocation table and a meta file, and with ``--wandb`` logs artifact ``rp1-prompts``
    (project ``RogueOne-AGENT2027``, run ``ANALYSIS__rp1_score__<YYYYMMDD>``, group
    ``ANALYSIS``). It refuses to draw again once ``rp1-prompts`` exists (A18-T10: never
    re-drawn) unless ``--allow-redraw``.

``fetch`` (A18-T16)
    Downloads ``rp1-prompts:<v>``, the ``rp1-replies-<rung>`` artifacts and the source
    state files the prompt records point at (trail tables, focus log, config).

``score`` (A18-T9 / T16; runs offline, ``--network none``)
    One row per (target, prompt), including the pseudo-target ``recorded_O120``:
    ``rp1_valid_call_raw``/``_rescued`` (A18 §2 move test, shared with ``replay_tool``),
    ``rp1_executes`` (generated code run in a subprocess on the reconstructed fold state,
    60 s timeout, 4 GB address-space cap, E13 restricted builtins when the image has them)
    and ``rp1_usable`` (the §2 per-role rules), plus the exclusion and sensitivity flags
    (``context_error``, ``infra_error``, ``clipped``, ``timeout``, ``suspicious``,
    ``reconstruction_partial``, ``auto_unprune``) and the T prune-list Jaccards against
    O120 and against J−a (E9d ``ja_rules`` on the parsed metric table). It also writes
    descriptive per (target, role, score) rates on the common set. Floors, bootstrap and
    the D1–D5 verdicts are A18-T18: ``rp1_verdicts.py`` (reads ``scored/rp1_scored.csv``).
    ``--decode-json-string-args --out-subdir scored_315`` writes the HQ #315 scoring
    sensitivity (JSON-string object arguments decoded; default off).

``attach-log``
    Resumes a finished replay run by id and uploads a file (the vLLM server log) to it
    (A18-T7 step 3).

Provisional parts (flagged in the output, see A18-T9 prerequisites):
* S focus novelty uses TF-IDF cosine (``novelty_method=tfidf``) because E11b's
  ``process_metrics.py`` embedder does not exist yet; ``--novelty off`` drops the rule.
* The SYN-P "does not prune a φ-recovering attribute" check uses
  ``recovery_detector.evaluate_feature_table`` (``counted``) on the trial's entity table;
  without ``--synp-meta`` it is reported ``unavailable``.

Containers only on CAIR (never ``python`` on the host). Sample (A18-T10)::

    docker run --rm --name ro27-a18-sample --cpus 8 \\
      --env-file /raid/home/henrbr16/code/PlanBench/.env -v $HOME/ro27_a18_work:/work \\
      -v <scripts/agent2027>:/workspace/scripts/agent2027:ro --entrypoint python3 \\
      rogue_one:agent2027-v2.0 /workspace/scripts/agent2027/analysis/rp1_score.py \\
        sample --out /work --seed 42 --wandb
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import gzip
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

_HERE = Path(__file__).resolve().parent
_AGENT2027 = _HERE.parent
_REPO = _AGENT2027.parents[1]
for _p in (str(_HERE), str(_AGENT2027), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fetch_wandb  # noqa: E402  (scripts/agent2027/analysis)
import replay_tool  # noqa: E402  (scripts/agent2027)

ENTITY = "PlanBench"
PROJECT = "RogueOne-AGENT2027"
O120 = "openai/gpt-oss-120b"
LLM_SEED = 7042
PROMPTS_ARTIFACT = "rp1-prompts"
PROMPTS_FILE = "rp1-prompts-v0.jsonl.gz"
ROLES = ("S", "E", "T", "W")

#: (row_id, dataset, task_type, twin row) -- A05 seed-42 r0 rows (A18 §4 prompt source)
SOURCE_ROWS = (
    ("A05-001", "balance-scale", "classification", "A05-021"),
    ("A05-002", "junglechess", "classification", "A05-022"),
    ("A05-003", "vehicle", "classification", "A05-023"),
    ("A05-004", "heart", "classification", "A05-024"),
    ("A05-005", "credit-g", "classification", "A05-025"),
    ("A05-006", "myocardial", "classification", "A05-026"),
    ("A05-007", "housing", "regression", "A05-027"),
    ("A05-008", "insurance", "regression", "A05-028"),
    ("A05-009", "crab", "regression", "A05-029"),
    ("A05-010", "wine", "regression", "A05-030"),
    ("A05-031", "synp_p1", "regression", "A05-047"),
    ("A05-032", "synp_p2", "regression", "A05-048"),
    ("A05-033", "synp_p3", "classification", "A05-049"),
    ("A05-034", "synp_p4", "classification", "A05-050"),
    ("A05-035", "synp_p5", "classification", "A05-051"),
    ("A05-036", "synp_p6a", "regression", "A05-052"),
    ("A05-037", "synp_p6b", "regression", "A05-053"),
    ("A05-038", "synp_p7", "classification", "A05-054"),
)
EXCLUDED_DATASETS = ("covtype",)  # never, any study (registry ban)

#: §4 allocation: cell -> (role, stage, n). Order is the canonical record order.
ALLOCATION: dict[str, tuple[str, str, int]] = {
    "S_answer": ("S", "focus", 80),
    "S_action": ("S", "focus", 120),
    "E_append": ("E", "append", 200),
    "E_other": ("E", "append", 100),
    "Ta_answer": ("T", "assessment", 40),
    "Ta_action": ("T", "assessment", 60),
    "Tp_prune": ("T", "pruning", 60),
    "Tp_answer": ("T", "pruning", 40),
    "W_search": ("W", "search", 60),
    "W_answer": ("W", "search", 40),
}
#: recorded O120 move per cell (for ``answer_instead_of_action``)
CELL_MOVE = {c: ("answer" if c.endswith("_answer") else "action") for c in ALLOCATION}
ITER_BUCKETS = ((0, 2, "it0-2"), (3, 6, "it3-6"), (7, 9, "it7-9"))
REQUEST_FIELDS = ("messages", "tools", "tool_choice", "parallel_tool_calls", "temperature",
                  "max_tokens", "top_p", "extra_body", "model", "seed")


class QCError(RuntimeError):
    """A18-T5 heavy QC assertion failed (A18-T6 done-when)."""


# =========================================================================== helpers
def _is_placeholder(v: Any) -> bool:
    return isinstance(v, str) and (v.startswith("<openai.") or v in ("NOT_GIVEN", "NotGiven", "Omit"))


def tool_names(tools: Any) -> list[str]:
    out = []
    for t in tools or []:
        if isinstance(t, dict) and isinstance(t.get("function"), dict) and t["function"].get("name"):
            out.append(str(t["function"]["name"]))
    return out


def call_names(message: Any) -> list[str]:
    return [str(c.get("name")) for c in replay_tool.calls_from_message(message or {})]


def iter_bucket(iteration_in_fold: Any) -> str:
    try:
        it = int(iteration_in_fold)
    except (TypeError, ValueError):
        return "it?"
    for lo, hi, name in ITER_BUCKETS:
        if lo <= it <= hi:
            return name
    return "it?"


def classify_record(rec: dict) -> tuple[Optional[str], str]:
    """(§4 cell or None, reason) for one E3 trace record, by the recorded O120 move."""
    if rec.get("error"):
        return None, "error"
    msg = rec.get("message")
    if not msg:
        return None, "no_response"
    agent = rec.get("agent_name")
    names = call_names(msg)
    action = bool(names)
    if agent == "ScientistAgent":
        if not tool_names(rec.get("tools")):
            return None, "s_no_tool_fallback"  # sci/agent.py no-tool fallback agent
        return ("S_action" if action else "S_answer"), "ok"
    if agent == "ExtractorAgent":
        if not action:
            return None, "e_answer"  # never read by the loop (§2)
        return ("E_append" if "append_new_attribute" in names else "E_other"), "ok"
    if agent == "TesterAgent":
        stage = str(rec.get("stage") or "")
        if stage.startswith("assess"):
            return ("Ta_action" if action else "Ta_answer"), "ok"
        if stage.startswith("prun"):
            if not action:
                return "Tp_answer", "ok"
            if "attribute_pruning_tool" in names:
                return "Tp_prune", "ok"
            return None, "tp_other_action"  # not a §4 pruning cell
        return None, f"tester_stage:{stage}"
    if agent == "WebSearchAgent":
        return ("W_search" if action else "W_answer"), "ok"
    return None, f"agent:{agent}"


# =========================================================================== source runs
class SourceRun:
    """One A05 source run, as local files: traces, summary, iteration log, file listing."""

    def __init__(self, row_id: str, dataset: str, task_type: str, run_name: str, root: Path,
                 meta: Optional[dict] = None, substituted_for: Optional[str] = None):
        self.row_id = row_id
        self.dataset = dataset
        self.task_type = task_type
        self.run_name = run_name
        self.root = Path(root)
        self.meta = meta or {}
        self.substituted_for = substituted_for
        out = self.root / "run-output"
        self.out = out if out.exists() else self.root
        self._files: Optional[list[str]] = None
        self._trails: dict[str, set[int]] = {}

    @property
    def files(self) -> list[str]:
        if self._files is None:
            listing = self.root / "files.json"
            if listing.exists():
                self._files = json.loads(listing.read_text(encoding="utf-8"))
            else:
                self._files = sorted(str(p.relative_to(self.out)).replace("\\", "/")
                                     for p in self.out.rglob("*") if p.is_file())
        return self._files

    def _trace_path(self) -> Path:
        p = fetch_wandb._find(self.out, "llm_traces.jsonl.gz")
        if p is None:
            raise QCError(f"{self.run_name}: llm_traces.jsonl.gz missing")
        return p

    def iter_traces(self) -> Iterator[tuple[int, dict]]:
        """(line number, record), streamed (a run's trace is ~10^2 MB once parsed)."""
        with gzip.open(self._trace_path(), "rt", encoding="utf-8") as fh:
            i = -1
            for line in fh:
                line = line.strip()
                if line:
                    i += 1
                    yield i, json.loads(line)

    def traces(self) -> list[dict]:
        return [r for _, r in self.iter_traces()]

    def summary(self) -> dict:
        p = fetch_wandb._find(self.out, "instrumentation_summary.json")
        if p is not None:
            return json.loads(p.read_text(encoding="utf-8"))
        return dict(self.meta.get("summary") or {})

    def focus_log(self) -> list[dict]:
        p = fetch_wandb._find(self.out, "focus_log.jsonl")
        return fetch_wandb.read_jsonl(p) if p else []

    def trail_indices(self, kind: str) -> set[int]:
        if kind not in self._trails:
            pat = re.compile(rf"(?:^|/)intermediate_steps/{kind}_trail_(\d+)\.csv$")
            self._trails[kind] = {int(m.group(1)) for f in self.files for m in [pat.search(f)] if m}
        return self._trails[kind]


def scan_source(run: SourceRun, qc: bool = True) -> tuple[dict[str, Any], list[dict], collections.Counter]:
    """One streaming pass over a run's trace: A18-T5 heavy QC (raises :class:`QCError`) and
    slim §4 candidates (the full records are re-read only for the drawn prompts)."""
    n = 0
    bad: list[int] = []
    first_missing: list[str] = []
    cands: list[dict] = []
    reasons: collections.Counter = collections.Counter()
    for i, rec in run.iter_traces():
        n += 1
        miss = [f for f in fetch_wandb.A18_TRACE_FIELDS if f not in rec]
        if miss:
            bad.append(i)
            first_missing = first_missing or miss
        cell, why = classify_record(rec)
        reasons[why] += 1
        if cell is not None:
            cands.append({"_run": run.row_id, "_seq": int(rec.get("seq") or 0), "_line": i, "_cell": cell,
                          "_bucket": iter_bucket(rec.get("iteration_in_fold")), "caller": rec.get("caller"),
                          "_source": run})
    summ = run.summary()
    total = summ.get("llm_calls_total")
    dumped = run.trail_indices("df_entities_attributes")
    if qc:
        if total is None or int(total) != n:
            raise QCError(f"{run.run_name}: trace records ({n}) != llm_calls_total ({total})")
        if bad:
            raise QCError(f"{run.run_name}: {len(bad)} trace records miss A18-T2 fields "
                          f"(first #{bad[0]}: {sorted(first_missing)})")
        attrs = run.trail_indices("df_attributes_explanations")
        feed = run.trail_indices("df_scientist_feedback")
        missing = sorted(i for i in dumped if i not in attrs or i not in feed)
        if missing:
            raise QCError(f"{run.run_name}: T2 per-trial tables missing for trials {missing[:10]} "
                          f"({len(missing)} of {len(dumped)} trials that called dump_data)")
    qc_row = {"run": run.run_name, "row_id": run.row_id, "trace_records": n, "llm_calls_total": total,
              "trials_dumped": len(dumped), "candidates": len(cands)}
    return qc_row, cands, reasons


def qc_source(run: SourceRun) -> dict[str, Any]:
    """A18-T5 heavy QC only; raises :class:`QCError` on a failed assertion."""
    return scan_source(run, qc=True)[0]


def latest_trail(indices: set[int], index: int, fold_run_id: int, n_iter: int = 10) -> Optional[int]:
    """Highest trial k < index of the same fold that has trail files (§5 A18-T9)."""
    lo = int(fold_run_id) * n_iter
    cands = [k for k in indices if lo <= k < int(index)]
    return max(cands) if cands else None


# =========================================================================== allocation
def _rng(seed: int, *parts: Any) -> random.Random:
    return random.Random("rp1|" + "|".join(str(p) for p in (seed,) + parts))


def _tree_insert(tree: dict, keys: list[str], item: dict) -> None:
    node = tree
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node.setdefault(keys[-1], []).append(item)


def _round_robin(node: Any, seed: int, path: tuple) -> Iterator[dict]:
    """Nested round-robin: one item per child per pass, children in a seeded order."""
    if isinstance(node, list):
        items = sorted(node, key=lambda c: (c["_run"], c["_seq"]))
        _rng(seed, "leaf", *path).shuffle(items)
        yield from items
        return
    keys = sorted(node)
    _rng(seed, "order", *path).shuffle(keys)
    gens = [_round_robin(node[k], seed, path + (k,)) for k in keys]
    while gens:
        alive = []
        for g in gens:
            try:
                yield next(g)
            except StopIteration:
                continue
            alive.append(g)
        gens = alive


def cell_generator(cands: list[dict], cell: str, seed: int) -> Iterator[dict]:
    tree: dict = {}
    for c in cands:
        keys = [c["_run"], c["_bucket"]]
        if cell.startswith("W_"):
            keys = [str(c.get("caller") or "?")] + keys
        _tree_insert(tree, keys, c)
    return _round_robin(tree, seed, (cell,))


def draw_allocation(cands_by_cell: dict[str, list[dict]], seed: int = 42,
                    allocation: Optional[dict[str, tuple[str, str, int]]] = None,
                    log: Optional[list[str]] = None) -> tuple[list[dict], list[dict]]:
    """Draw the §4 allocation; short cells are filled from the same stage, then role.

    Returns (chosen records in canonical order, realised allocation rows).
    """
    alloc = allocation or ALLOCATION
    log = log if log is not None else []
    gens = {cell: cell_generator(cands_by_cell.get(cell, []), cell, seed) for cell in alloc}
    chosen: dict[str, list[dict]] = {cell: [] for cell in alloc}

    def take(cell: str, n: int) -> int:
        got = 0
        for _ in range(n):
            try:
                item = next(gens[cell])
            except StopIteration:
                break
            chosen[cell].append(item)
            got += 1
        return got

    short: dict[str, int] = {}
    for cell, (_role, _stage, n) in alloc.items():
        got = take(cell, n)
        if got < n:
            short[cell] = n - got
            log.append(f"cell {cell}: {got}/{n} available (short {n - got})")
    fills: list[dict] = []
    for cell, deficit in short.items():
        role, stage, _ = alloc[cell]
        same_stage = [c for c, (r, s, _) in alloc.items() if c != cell and r == role and s == stage]
        same_role = [c for c, (r, s, _) in alloc.items() if c != cell and r == role and s != stage]
        for donor in same_stage + same_role:
            if deficit <= 0:
                break
            got = take(donor, deficit)
            if got:
                for item in chosen[donor][-got:]:
                    item["_filled_for"] = cell
                fills.append({"short_cell": cell, "donor": donor, "n": got})
                log.append(f"fill: {got} from {donor} for short cell {cell}")
                deficit -= got
        if deficit > 0:
            log.append(f"cell {cell}: still {deficit} short after the fill rule (role {role} exhausted)")
    out: list[dict] = []
    table: list[dict] = []
    for cell, (role, stage, n) in alloc.items():
        recs = chosen[cell]
        out.extend(recs)
        table.append({
            "cell": cell, "role": role, "stage": stage, "target_n": n,
            "drawn": len(recs), "own": sum(1 for r in recs if "_filled_for" not in r),
            "fill_donated": sum(1 for r in recs if "_filled_for" in r),
            "available": len(cands_by_cell.get(cell, [])),
            "runs": len({r["_run"] for r in recs}),
            **{f"bucket_{b}": sum(1 for r in recs if r["_bucket"] == b) for _, _, b in ITER_BUCKETS},
        })
    return out, table


# =========================================================================== records
def make_prompt_record(rec: dict, run: SourceRun, cell: str, focus_by_index: dict[int, str]) -> dict:
    """One ``rp1-prompts`` record (request, recorded O120 response, strata, state pointers)."""
    role, stage, _ = ALLOCATION[cell]
    req = {k: rec.get(k) for k in REQUEST_FIELDS if rec.get(k) is not None and not _is_placeholder(rec.get(k))}
    index = int(rec["index"])
    fold = int(rec["fold_run_id"])
    ent = run.trail_indices("df_entities_attributes")
    k = latest_trail(ent, index, fold)
    files = lambda i: {  # noqa: E731
        "entities": f"intermediate_steps/df_entities_attributes_trail_{i}.csv",
        "attributes": f"intermediate_steps/df_attributes_explanations_trail_{i}.csv",
        "feedback": f"intermediate_steps/df_scientist_feedback_trail_{i}.csv",
    }
    history = [focus_by_index[i] for i in sorted(focus_by_index) if fold * 10 <= i < index]
    usage = rec.get("usage") or {}
    mpt = max([int(usage.get("prompt_tokens") or 0)]
              + [int(a.get("prompt_tokens") or 0) for a in rec.get("attempts") or []])
    return {
        "prompt_id": None,
        "role": role,
        "stage": stage,
        "stratum": cell,
        "recorded_move": CELL_MOVE[cell],
        "agent_name": rec.get("agent_name"),
        "caller": rec.get("caller"),
        "request": req,
        "recorded_response": {
            "message": rec.get("message"), "finish_reason": rec.get("finish_reason"),
            "usage": rec.get("usage"), "rerolls": rec.get("rerolls"), "rescue": rec.get("rescue"),
            "response_model": rec.get("response_model"),
        },
        "max_prompt_tokens": mpt,
        "provenance": {
            "row_id": run.row_id, "run_name": run.run_name, "run_id": run.meta.get("id"),
            "run_output_artifact": run.meta.get("artifact"), "dataset_id": run.dataset,
            "task_type": run.task_type, "substituted_for": run.substituted_for,
            "seq": rec.get("seq"), "call_seq": rec.get("call_seq"), "session_name": rec.get("session_name"),
            "index": index, "fold_run_id": fold, "iteration_in_fold": rec.get("iteration_in_fold"),
            "iter_bucket": iter_bucket(rec.get("iteration_in_fold")), "filled_for": rec.get("_filled_for"),
        },
        "state": {
            "trail_k": k,
            "trail_k_files": files(k) if k is not None else None,
            "trail_index_files": files(index) if index in ent else None,
            "data_csv": f"{run.task_type}/{run.dataset}/data.csv",
            "focus_history": history,
        },
    }


def write_prompt_set(result: dict, out_dir: Path, meta_extra: Optional[dict] = None) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {r["prompt_id"]: r for r in result["records"]}
    paths = {"prompts": out_dir / PROMPTS_FILE, "allocation": out_dir / "rp1-allocation.csv",
             "meta": out_dir / "rp1-prompts-v0.meta.json"}
    # mtime=0 and no file name in the gzip header: the same draw gives byte-identical files
    with open(paths["prompts"], "wb") as raw, \
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        for pid in result["replay_order"]:
            gz.write((json.dumps(by_id[pid], ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    with open(paths["allocation"], "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(result["allocation"][0].keys()))
        w.writeheader()
        w.writerows(result["allocation"])
    sha = hashlib.sha256(paths["prompts"].read_bytes()).hexdigest()
    meta = {
        "n": len(result["records"]), "allocation": result["allocation"], "log": result["log"],
        "qc": result["qc"], "classify_reasons": result.get("classify_reasons"),
        "available": result.get("available"), "prompts_sha256": sha,
        "roles": dict(collections.Counter(r["role"] for r in result["records"])),
        "max_prompt_tokens_max": max((r["max_prompt_tokens"] for r in result["records"]), default=0),
        **(meta_extra or {}),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return paths


# =========================================================================== W&B fetching (sample)
def _valid_runs(api: Any, name: str) -> list[Any]:
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters={"display_name": name}))
    ok = [r for r in runs if r.state == "finished" and (r.summary.get("llm_calls_total") or 0) > 0]
    return sorted(ok, key=lambda r: str(r.created_at), reverse=True)


def _run_output_artifact(run: Any) -> Optional[Any]:
    for art in run.logged_artifacts():
        if art.type == "run-output":
            return art
    return None


SAMPLE_FILES = ("llm_traces.jsonl.gz", "instrumentation_summary.json", "focus_log.jsonl",
                "iteration_log.jsonl")


def fetch_source_run(api: Any, row: tuple, root: Path, log: list[str]) -> SourceRun:
    """Resolve row -> run (twin on ITT failure) and download the files ``sample`` needs."""
    row_id, ds, task_type, twin = row
    if ds in EXCLUDED_DATASETS:
        raise SystemExit(f"{row_id}: dataset {ds} is excluded")
    for rid, rep, subst in ((row_id, 0, None), (twin, 1, row_id)):
        name = f"A05__full__{ds}__f42__r{rep}"
        for run in _valid_runs(api, name):
            art = _run_output_artifact(run)
            if art is None:
                log.append(f"{name} ({run.id}): no run-output artifact; skipped")
                continue
            d = root / name
            (d / "run-output").mkdir(parents=True, exist_ok=True)
            entries = sorted(art.manifest.entries.keys())
            for f in SAMPLE_FILES:
                if f in entries and not (d / "run-output" / f).exists():
                    art.get_entry(f).download(root=str(d / "run-output"))
            (d / "files.json").write_text(json.dumps(entries), encoding="utf-8")
            meta = {"id": run.id, "name": run.name, "state": run.state, "created_at": str(run.created_at),
                    "artifact": f"{ENTITY}/{PROJECT}/{art.name}", "row_id": rid,
                    "config": {k: v for k, v in dict(run.config).items() if k.startswith(("flags/", "image"))}}
            (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
            if subst:
                log.append(f"ITT substitution: {row_id} -> twin {twin} ({name}, {run.id})")
            return SourceRun(rid, ds, task_type, name, d, meta, substituted_for=subst)
        log.append(f"{name}: no finished run with llm_calls_total > 0 and a run-output artifact")
    raise QCError(f"{row_id} ({ds}): neither r0 nor the twin {twin} is usable")


def load_local_runs(src_dir: Path, rows: Iterable[tuple] = SOURCE_ROWS) -> list[SourceRun]:
    """Pre-fetched layout ``<src>/<run name>/{meta.json, files.json, run-output/...}``."""
    runs = []
    for row_id, ds, task_type, twin in rows:
        for rid, rep, subst in ((row_id, 0, None), (twin, 1, row_id)):
            d = src_dir / f"A05__full__{ds}__f42__r{rep}"
            if d.exists():
                meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
                runs.append(SourceRun(rid, ds, task_type, d.name, d, meta, substituted_for=subst))
                break
        else:
            raise QCError(f"{row_id} ({ds}): no local source run under {src_dir}")
    return runs


def cmd_sample(args: argparse.Namespace) -> int:
    out = Path(args.out)
    log: list[str] = []
    api = None
    if args.wandb or not args.from_dir:
        import wandb  # noqa: PLC0415

        api = wandb.Api()
        if args.wandb and not args.allow_redraw:
            try:
                existing = api.artifact(f"{ENTITY}/{args.project}/{PROMPTS_ARTIFACT}:latest")
                raise SystemExit(f"{PROMPTS_ARTIFACT} already exists ({existing.version}); it is never "
                                 "re-drawn after the first replay (A18-T10). Pass --allow-redraw only if no "
                                 "replay has used it.")
            except SystemExit:
                raise
            except Exception:
                pass
    if args.from_dir:
        runs = load_local_runs(Path(args.from_dir))
    else:
        src = out / "sources"
        runs = [fetch_source_run(api, row, src, log) for row in SOURCE_ROWS]
    t0 = time.time()
    result = build_prompt_set(runs, args.seed)
    result["log"] = log + result["log"]
    meta_extra = {
        "seed": args.seed, "llm_seed_for_replay": LLM_SEED, "sampler": "scripts/agent2027/analysis/rp1_score.py",
        "sampler_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_sha": os.getenv("GIT_SHA"), "source_runs": [
            {"row_id": r.row_id, "run": r.run_name, "id": r.meta.get("id"), "artifact": r.meta.get("artifact"),
             "substituted_for": r.substituted_for} for r in runs],
        "build_s": round(time.time() - t0, 1),
    }
    paths = write_prompt_set(result, out, meta_extra)
    for line in result["log"]:
        print("[rp1 sample]", line)
    print(json.dumps({"n": len(result["records"]), "allocation": result["allocation"]}, indent=1))
    if args.wandb:
        import wandb  # noqa: PLC0415

        name = args.run_name or f"ANALYSIS__rp1_score__{time.strftime('%Y%m%d', time.gmtime())}"
        meta = json.loads(paths["meta"].read_text())
        run = wandb.init(entity=ENTITY, project=args.project, group="ANALYSIS", job_type="rp1_sample",
                         name=name, tags=["A18", "rp1", args.image_tag or ""],
                         config={"seed": args.seed, "n": meta["n"], "sampler_sha256": meta["sampler_sha256"],
                                 "git_sha": meta.get("git_sha")})
        for r in runs:
            if r.meta.get("artifact"):
                run.use_artifact(r.meta["artifact"])
        run.log({"rp1_allocation": wandb.Table(columns=list(result["allocation"][0].keys()),
                                               data=[list(x.values()) for x in result["allocation"]])})
        art = wandb.Artifact(PROMPTS_ARTIFACT, type="prompt-set",
                             description="A18-T6 RP1 prompt set (800 single-turn O120 prompts, seed 42)",
                             metadata={k: v for k, v in meta.items() if k not in ("qc",)})
        for p in paths.values():
            art.add_file(str(p))
        logged = run.log_artifact(art)
        logged.wait()
        print(f"[rp1 sample] logged {ENTITY}/{args.project}/{PROMPTS_ARTIFACT}:{logged.version}")
        run.summary.update({"rp1_prompts_n": meta["n"], "rp1_prompts_version": logged.version})
        run.finish()
    return 0


def build_prompt_set(runs: list[SourceRun], seed: int = 42, qc: bool = True) -> dict[str, Any]:
    """QC + classification + allocation + records (pure given the local run files)."""
    cands: dict[str, list[dict]] = collections.defaultdict(list)
    reasons: collections.Counter = collections.Counter()
    qc_rows = []
    for run in runs:
        qc_row, items, why = scan_source(run, qc=qc)
        qc_rows.append(qc_row)
        reasons.update(why)
        for it in items:
            cands[it["_cell"]].append(it)
    log: list[str] = []
    for cell, (_r, _s, n) in ALLOCATION.items():
        if len(cands.get(cell, [])) < n:
            log.append(f"QC: stratum {cell} has {len(cands.get(cell, []))} < {n} available")
    chosen, table = draw_allocation(cands, seed, log=log)
    wanted: dict[str, dict[int, dict]] = collections.defaultdict(dict)
    for item in chosen:
        wanted[item["_run"]][item["_line"]] = item
    full: dict[tuple[str, int], dict] = {}
    for run in runs:
        lines = wanted.get(run.row_id) or {}
        if not lines:
            continue
        for i, rec in run.iter_traces():
            if i in lines:
                rec["_filled_for"] = lines[i].get("_filled_for")
                full[(run.row_id, i)] = rec
    focus = {r.row_id: {int(f["index"]): str(f.get("focus")) for f in r.focus_log() if f.get("index") is not None}
             for r in runs}
    records = []
    for i, item in enumerate(chosen):
        run = item["_source"]
        rec = make_prompt_record(full[(run.row_id, item["_line"])], run, item["_cell"], focus[run.row_id])
        rec["prompt_id"] = f"rp1-{i:03d}"
        records.append(rec)
    order = list(range(len(records)))
    _rng(seed, "replay-order").shuffle(order)
    for pos, i in enumerate(order):
        records[i]["replay_order"] = pos
    return {"records": records, "replay_order": [records[i]["prompt_id"] for i in order],
            "allocation": table, "log": log, "qc": qc_rows, "classify_reasons": dict(reasons),
            "available": {c: len(v) for c, v in cands.items()}}


# =========================================================================== §2 rules
BUILD_WORDS = frozenset("add build create introduce construct derive engineer compute encode combine "
                        "interact transform extract".split())
PRUNE_WORDS = frozenset("remove drop prune retire delete".split())
T_SECTIONS = ("summary", "feature-by-feature", "risks", "recommended candidates")
W_GIVE_UP = ("Unable to provide an explanation at this time.", "Internal error.")
CODE_TOOLS = {"S": frozenset({"generic_pandas_tool"}),
              "E": frozenset({"generic_pandas_tool", "append_new_attribute"})}
SUSPICIOUS_RE = re.compile(r"labels|open\(|import\s+os|subprocess")
_SECTION_LINE_RE = re.compile(r"^\s*(#|\*\*|[-*+]\s|\d+[.)]\s)")
W_MAX_WORDS = 12
NOVELTY_MAX_COS = 0.9
_MAX_QUERY_LENGTH = 256


def sanitize_query(query: str) -> str:
    """Copy of ``web_search._sanitize_query`` (v2.0); a unit test pins it to the original."""
    query = re.sub(r"[=<>≥≤≠≈∝∞±×÷√∑∏∫∂∇∆θφπσμλαβγδεζηικνξρτυψω]+", " ", query)
    query = re.sub(r"[{}()\[\]|\\^~`@#$%&*_]", " ", query)
    query = re.sub(r"\s+", " ", query).strip()
    if len(query) > _MAX_QUERY_LENGTH:
        query = query[:_MAX_QUERY_LENGTH].rsplit(" ", 1)[0]
    return query


def w_query_usable(query: Any) -> bool:
    """§2 W query: unchanged by ``_sanitize_query`` apart from whitespace, at most 12 words."""
    if not isinstance(query, str):
        return False
    return re.sub(r"\s+", " ", query).strip() == sanitize_query(query) and len(query.split()) <= W_MAX_WORDS


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", (text or "").lower()))


def is_additive(text: str) -> bool:
    """§2: at least one build word (whole word, case-insensitive); pruning-only never is."""
    return bool(_words(text) & BUILD_WORDS)


def is_pruning_only(text: str) -> bool:
    w = _words(text)
    return bool((w & PRUNE_WORDS) or "keep only" in (text or "").lower()) and not (w & BUILD_WORDS)


def names_any(text: str, names: Iterable[str]) -> bool:
    return any(n and n in (text or "") for n in names)


def share_named(text: str, names: Iterable[str]) -> Optional[float]:
    names = [n for n in names if n]
    if not names:
        return None
    return sum(1 for n in names if n in (text or "")) / len(names)


def t_sections_present(text: str) -> int:
    """Number of the 4 required T sections present (one match per section)."""
    found = set()
    for line in (text or "").splitlines():
        if not _SECTION_LINE_RE.match(line):
            continue
        low = line.lower()
        for sec in T_SECTIONS:
            if sec in low:
                found.add(sec)
    return len(found)


def focus_novelty(answer: str, history: list[str], method: str = "tfidf") -> tuple[Optional[float], bool]:
    """(max cosine to the fold's focus history, novel?); no history -> (None, True)."""
    hist = [h for h in history if h and h.strip()]
    if method == "off" or not hist:
        return None, True
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415
    from sklearn.metrics.pairwise import cosine_similarity  # noqa: PLC0415

    try:
        m = TfidfVectorizer().fit_transform(hist + [answer or ""])
    except ValueError:  # empty vocabulary
        return None, True
    cos = float(cosine_similarity(m[-1], m[:-1]).max())
    return cos, cos < NOVELTY_MAX_COS


def parse_args_json(raw: Any) -> Optional[dict]:
    try:
        v = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return None
    return v if isinstance(v, dict) else None


def call_args(call: dict, decode_json_strings: bool = False) -> dict:
    """Arguments of a call, unwrapping the ``{"args": {...}}`` pydantic wrapper (E tools).

    ``decode_json_strings`` (HQ #315 sensitivity, default off): a wrapper value sent as a JSON
    string holding an object (vLLM ``qwen3_coder`` ``$ref`` bug) is decoded before unwrapping.
    """
    a = parse_args_json(call.get("arguments")) or {}
    inner = a.get("args")
    if decode_json_strings and isinstance(inner, str):
        inner = parse_args_json(inner)
    return inner if isinstance(inner, dict) else a


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# --------------------------------------------------------------------------- message history
def _content_text(c: Any) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in c)
    return "" if c is None else str(c)


def session_history(messages: list[dict]) -> list[dict]:
    """Earlier tool calls of this session with their tool-result text, in order."""
    results = {}
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == "tool":
            results[m.get("tool_call_id")] = _content_text(m.get("content"))
    out = []
    for m in messages or []:
        if not (isinstance(m, dict) and m.get("role") == "assistant"):
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            out.append({"name": fn.get("name"), "arguments": fn.get("arguments"),
                        "result": results.get(tc.get("id"))})
    return out


def e_history_effects(messages: list[dict]) -> tuple[list[dict], list[str]]:
    """(accepted appends [{name, command}], auto-unpruned names) from the session's earlier calls,
    read from their tool-result messages (A18-T9: the actual outcome of a recorded call)."""
    appends, restored = [], []
    for h in session_history(messages):
        if h["name"] != "append_new_attribute" or not h["result"]:
            continue
        a = call_args(h)
        if "explained, documented, and added to df_features" in h["result"]:
            appends.append({"name": a.get("name"), "command": a.get("command")})
        elif "has been restored to Active status" in h["result"]:
            restored.append(a.get("name"))
    return appends, restored


_PRUNED_LIST_RE = re.compile(r"Attributes pruned successfully:\s*(\[[^\]]*\])")


def t_pruned_in_session(messages: list[dict]) -> list[str]:
    """Names actually pruned by the session's earlier ``attribute_pruning_tool`` calls, read
    from the tool result (E6: seeded raw columns are reported as "Not pruned (protected …)")."""
    out = []
    for h in session_history(messages):
        if h["name"] != "attribute_pruning_tool" or not h["result"]:
            continue
        m = _PRUNED_LIST_RE.search(h["result"])
        if m:
            try:
                out += [str(n) for n in ast.literal_eval(m.group(1))]
            except (ValueError, SyntaxError):
                pass
        elif "Attributes pruned successfully" in h["result"]:
            out += [str(n) for n in (call_args(h).get("attribute_names_list") or [])]
    return out


_TABLE_HDR = re.compile(r"\|\s*feature_name\s*\|\s*mutual_information\s*\|")


def parse_metric_table(messages: list[dict]) -> Optional[list[dict]]:
    """First ``_metric_profiles_to_markdown`` table in the request (tst/agent.py)."""
    for m in messages or []:
        text = _content_text(m.get("content") if isinstance(m, dict) else None)
        hit = _TABLE_HDR.search(text)
        if not hit:
            continue
        lines = text[hit.start():].splitlines()
        header = [h.strip() for h in lines[0].strip().strip("|").split("|")]
        rows = []
        for line in lines[2:]:
            if not line.strip().startswith("|"):
                break
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            row = dict(zip(header, cells))

            def num(k, row=row):
                try:
                    v = float(row.get(k, "N/A"))
                    return None if math.isnan(v) else v
                except ValueError:
                    return None
            rows.append({"feature_name": row.get("feature_name"), "mutual_information": num("mutual_information"),
                         "vif": num("vif"), "permutation_importance_mean": num("pfi_mean"),
                         "permutation_importance_std": num("pfi_std"), "mean_abs_shap": num("mean_abs_shap")})
        return rows
    return None


def ja_prune_set(table: Optional[list[dict]], protected: Iterable[str] = ()) -> Optional[list[str]]:
    """J−a (E9d ``fx_switches.ja_rules``) on a parsed metric table; None if unavailable.

    No training matrix is available, so VIF is not recomputed after a cluster prune (one
    cluster victim per table); A18 §2 applies J−a to the parsed table on purpose."""
    if table is None:
        return None
    try:
        from types import SimpleNamespace  # noqa: PLC0415

        from rogueone.utils.fx_switches import ja_rules  # noqa: PLC0415
    except Exception:
        return None
    profiles = [SimpleNamespace(**r) for r in table]
    names = [r["feature_name"] for r in table]
    dec = ja_rules(profiles, None, names, protected=protected, vif_fn=lambda X, n: [None] * len(n))
    return [d.feature for d in dec]


# =========================================================================== reconstruction
def _read_csv(path: Optional[Path]):
    import pandas as pd  # noqa: PLC0415

    if path is None or not Path(path).exists():
        return None
    return pd.read_csv(path)


class PromptState:
    """Trusted part of a prompt's reconstructed fold state (no generated code runs here)."""

    def __init__(self, rec: dict, sources_root: Path, tasks_root: Path):
        self.rec = rec
        prov, st = rec["provenance"], rec["state"]
        self.run_dir = Path(sources_root) / prov["run_name"] / "run-output"
        self.data_csv = Path(tasks_root) / st["data_csv"]
        self.flags: set[str] = set()
        kf = st.get("trail_k_files") or {}
        self.entities_k = self.run_dir / kf["entities"] if kf else None
        self.attributes_k = self.run_dir / kf["attributes"] if kf else None
        self.feedback_k = self.run_dir / kf["feedback"] if kf else None
        idx = st.get("trail_index_files") or {}
        self.attributes_index = self.run_dir / idx["attributes"] if idx else None
        self.entities_index = self.run_dir / idx["entities"] if idx else None
        for p in (self.entities_k, self.attributes_k, self.feedback_k):
            if p is not None and not p.exists():
                self.flags.add("reconstruction_partial")
        if not self.data_csv.exists():
            self.flags.add("reconstruction_partial")
        self._attr_k = None
        self._cols = None
        self._t_active = None

    def attributes_k_table(self):
        if self._attr_k is None:
            self._attr_k = _read_csv(self.attributes_k)
        return self._attr_k

    @staticmethod
    def _status_sets(table) -> tuple[set[str], set[str]]:
        if table is None or "Attribute" not in table.columns:
            return set(), set()
        names = table["Attribute"].astype(str)
        if "Status" in table.columns:
            pruned = table["Status"].astype(str).str.strip().str.lower() == "pruned"
        else:
            pruned = names != names
        return set(names[~pruned]), set(names[pruned])

    def raw_columns(self) -> list[str]:
        if self._cols is None:
            import pandas as pd  # noqa: PLC0415

            self._cols = ([c for c in pd.read_csv(self.data_csv, nrows=0).columns if c != "id"]
                          if self.data_csv.exists() else [])
        return self._cols

    def seeded_names(self) -> set[str]:
        t = self.attributes_k_table()
        if t is None or "Seeded" not in t.columns:
            return set()
        return set(t.loc[t["Seeded"].fillna(0).astype(int) == 1, "Attribute"].astype(str))

    def e_sets(self) -> dict[str, Any]:
        """Active / Pruned names at the replayed E call (trail k + this session's history)."""
        active, pruned = self._status_sets(self.attributes_k_table())
        appends, restored = e_history_effects(self.rec["request"].get("messages") or [])
        for a in appends:
            if a.get("name"):
                active.add(str(a["name"]))
        for n in restored:
            pruned.discard(str(n))
            active.add(str(n))
        return {"active": sorted(active), "pruned": sorted(pruned), "appends": appends, "restored": restored}

    def s_names(self) -> list[str]:
        active, _ = self._status_sets(self.attributes_k_table())
        return sorted(set(self.raw_columns()) | active)

    def all_attribute_names(self) -> set[str]:
        a, p = self._status_sets(self.attributes_k_table())
        return a | p | set(self.raw_columns()) | set(self.t_active_raw())

    def t_active_raw(self) -> list[str]:
        """The trial's scored set (``Scored`` = not Pruned before the Tester), else the metric table."""
        if self._t_active is None:
            names: set[str] = set()
            t = _read_csv(self.attributes_index)
            if t is not None and "Scored" in t.columns:
                names = set(t.loc[t["Scored"].fillna(0).astype(int) == 1, "Attribute"].astype(str))
            if not names:
                table = parse_metric_table(self.rec["request"].get("messages") or [])
                if table:
                    names = {str(r["feature_name"]) for r in table}
                    self.flags.add("t_active_from_metric_table")
            self._t_active = sorted(names)
        return self._t_active

    def t_active(self) -> list[str]:
        """Active set seen by a T call: the scored set minus earlier prunes of this session."""
        names = set(self.t_active_raw())
        if not names and self.rec["role"] == "T":
            self.flags.add("reconstruction_partial")
        return sorted(names - set(t_pruned_in_session(self.rec["request"].get("messages") or [])))

    def exec_spec(self, role: str) -> dict:
        prov = self.rec["provenance"]
        spec: dict[str, Any] = {"role": role, "data_csv": str(self.data_csv),
                                "iteration_in_fold": prov.get("iteration_in_fold")}
        if role == "E":
            sets = self.e_sets()
            spec.update(entities_csv=str(self.entities_k) if self.entities_k else None,
                        seed_iter0=self.entities_k is None, history_appends=sets["appends"],
                        active=sets["active"], pruned=sets["pruned"])
        else:
            spec.update(feedback_csv=str(self.feedback_k) if self.feedback_k else None)
        return spec


# =========================================================================== code execution (child)
def _child_exec(spec: dict, calls: list[dict], timeout_s: int) -> list[dict]:
    """Runs inside the ``_exec`` subprocess: rebuild the state, run each call on a fresh copy."""
    os.environ.setdefault("ROGUE_ONE_SANDBOX", "on")
    import contextlib  # noqa: PLC0415
    import io  # noqa: PLC0415
    import signal  # noqa: PLC0415
    from textwrap import dedent  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    try:
        from rogueone.utils.sandbox import sandbox_scope  # noqa: PLC0415
    except Exception:  # pragma: no cover - image without E13
        def sandbox_scope(scope, environ=None):
            return scope
    try:
        from rogueone.utils.pandas_exec import exec_user_command  # noqa: PLC0415
    except Exception:  # pragma: no cover
        def exec_user_command(command, df, df_features):
            scope = {"pd": pd, "df": df, "np": np, "df_features": df_features}
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                exec(dedent(command).strip(), sandbox_scope(scope))  # noqa: S102
            return scope.get("result", None), buf.getvalue()

    class _Alarm(Exception):
        pass

    def _on_alarm(signum, frame):
        raise _Alarm()

    has_alarm = hasattr(signal, "SIGALRM")
    if has_alarm:
        signal.signal(signal.SIGALRM, _on_alarm)

    def timed(fn):
        if has_alarm:
            signal.alarm(int(timeout_s))
        try:
            return fn()
        finally:
            if has_alarm:
                signal.alarm(0)

    df = pd.read_csv(spec["data_csv"])
    partial = False
    role = spec["role"]
    base = known = None
    pruned: set[str] = set()
    active: list[str] = []
    feedback = None
    if role == "E":
        if spec.get("entities_csv"):
            base = pd.read_csv(spec["entities_csv"])
        else:
            base = pd.DataFrame(df["id"].unique(), columns=["id"])
            if spec.get("seed_iter0") and int(spec.get("iteration_in_fold") or 0) == 0:
                try:
                    from rogueone.utils.fold_safe_features import seed_raw_feature_rows  # noqa: PLC0415

                    canon = pd.DataFrame(columns=["Attribute", "Description", "Pandas Command", "Status",
                                                  "Added", "Pruned Iteration", "Prune Rationale"])
                    base, _ = seed_raw_feature_rows(df, base, canon)
                except Exception:
                    partial = True
            else:
                partial = True
        for a in spec.get("history_appends") or []:
            try:
                res, _ = timed(lambda a=a: exec_user_command(a["command"], df.copy(), base.copy()))
                if isinstance(res, (pd.Series, pd.DataFrame)):
                    res = res.reset_index(drop=True).values
                base[a["name"]] = res
            except BaseException:  # noqa: BLE001
                partial = True
        known = set(map(str, base.columns))
        pruned = set(spec.get("pruned") or [])
        active = [n for n in spec.get("active") or [] if n in known and n != "id"]
    else:
        fb = spec.get("feedback_csv")
        feedback = pd.read_csv(fb) if fb and Path(fb).exists() else pd.DataFrame()

    out = []
    for call in calls:
        name, a = call["name"], call.get("args") or {}
        r: dict[str, Any] = {"tool": name, "executes": False, "reconstruction_partial": partial}
        cmd = a.get("command")
        r["suspicious"] = bool(isinstance(cmd, str) and SUSPICIOUS_RE.search(cmd))
        try:
            if role == "S":
                if not isinstance(cmd, str):
                    r["error"] = "command_not_str"
                else:
                    norm = re.sub(r";[ \t]*\n[ \t]+", "\n", dedent(cmd).strip())
                    scope = sandbox_scope({"pd": pd, "np": np, "df_test_results": feedback.copy(),
                                           "df_raw_data": df.copy()})

                    def run_s(norm=norm, scope=scope):
                        with contextlib.redirect_stdout(io.StringIO()):
                            exec(norm, scope)  # noqa: S102
                        return scope.get("result", None)
                    r["executes"] = timed(run_s) is not None
            elif name == "generic_pandas_tool":
                if not isinstance(cmd, str):
                    r["error"] = "command_not_str"
                elif (cmd.strip().startswith("df =") or cmd.strip().startswith("df_features =")
                      or "df" not in cmd.strip()):
                    r["precheck"] = "illegal"  # ext/agent.py ILLEGAL guard
                else:
                    res, captured = timed(lambda: exec_user_command(cmd, df.copy(), base.copy()))
                    r["executes"] = res is not None or bool((captured or "").strip())
            elif name == "append_new_attribute":
                attr = str(a.get("name"))
                if attr in known and attr in pruned:
                    r.update(auto_unprune=True, accepted=True, usable_append=True)
                elif attr in known:
                    r.update(precheck="duplicate_active", accepted=False, usable_append=False)
                elif not isinstance(cmd, str):
                    r["error"] = "command_not_str"
                else:
                    feats = base.copy()
                    res, _ = timed(lambda: exec_user_command(cmd, df.copy(), feats))
                    r["executes"] = res is not None
                    r.update(_append_checks(res, attr, feats, active, np, pd))
        except _Alarm:
            r.update(error="timeout", exec_timeout=True)
        except BaseException as exc:  # noqa: BLE001 - generated code can raise anything
            r["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        out.append(r)
    return out


def _append_checks(result, name, feats, active, np, pd) -> dict:
    """``append_new_attribute`` acceptance (ext/agent.py) + fun_valid (non-constant, |r| < 0.999)."""
    if result is None:
        return {"accepted": False, "reject": "no_result", "usable_append": False}
    try:
        if isinstance(result, (pd.Series, pd.DataFrame)):
            result = result.reset_index(drop=True).values
        if isinstance(result, np.ndarray):
            if np.isnan(result).any() or np.isinf(result).any():
                return {"accepted": False, "reject": "nan", "usable_append": False}
            if not np.issubdtype(result.dtype, np.number):
                return {"accepted": False, "reject": "non_numeric", "usable_append": False}
        else:
            if pd.isnull(result) or result in [float("inf"), float("-inf")]:
                return {"accepted": False, "reject": "nan", "usable_append": False}
            pd.Series(result.flatten() if hasattr(result, "flatten") else result).astype(float)
        feats[name] = result
    except BaseException as exc:  # noqa: BLE001
        return {"accepted": False, "reject": f"error:{type(exc).__name__}", "usable_append": False}
    col = pd.to_numeric(feats[name], errors="coerce").to_numpy(dtype=float)
    constant = bool(np.nanstd(col) == 0) if col.size else True
    max_r = 0.0
    for other in active:
        if other == name or other not in feats.columns:
            continue
        o = pd.to_numeric(feats[other], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(o) & np.isfinite(col)
        if ok.sum() < 3 or np.std(o[ok]) == 0 or np.std(col[ok]) == 0:
            continue
        max_r = max(max_r, abs(float(np.corrcoef(col[ok], o[ok])[0, 1])))
    return {"accepted": True, "constant": constant, "max_abs_r": max_r,
            "usable_append": (not constant) and max_r < 0.999}


def run_exec(spec: dict, calls: list[dict], timeout_s: int = 60, mem_gb: float = 4.0) -> list[dict]:
    """Run :func:`_child_exec` in a subprocess (4 GB address-space cap, wall-clock backstop)."""
    payload = json.dumps({"spec": spec, "calls": calls, "timeout": timeout_s})

    def limit():  # pragma: no cover - POSIX only
        import resource  # noqa: PLC0415

        cap = int(mem_gb * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    kw = {"preexec_fn": limit} if os.name == "posix" and mem_gb else {}
    env = dict(os.environ, ROGUE_ONE_SANDBOX=os.getenv("ROGUE_ONE_SANDBOX", "on"),
               PYTHONPATH=os.pathsep.join([str(_REPO), str(_AGENT2027), str(_HERE)]))
    try:
        p = subprocess.run([sys.executable, str(Path(__file__).resolve()), "_exec"], input=payload,
                           capture_output=True, text=True, timeout=timeout_s * max(1, len(calls)) + 120,
                           env=env, **kw)
    except subprocess.TimeoutExpired:
        return [{"tool": c["name"], "executes": False, "error": "timeout", "exec_timeout": True} for c in calls]
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        err = (p.stderr or "")[-300:]
        return [{"tool": c["name"], "executes": False, "error": f"child_failed rc={p.returncode}: {err}"}
                for c in calls]


# =========================================================================== scoring one move
def move_of_reply(row: dict) -> dict:
    """Final and attempt-1 moves of one ``replay_calls.jsonl`` row."""
    fm = row.get("final_message") or {}
    att = (row.get("attempt_details") or [None])[0]
    raw = None
    if att and "error" not in att:
        raw = {"finish_reason": att.get("server_finish_reason") or att.get("finish_reason"),
               "content": att.get("content"), "tool_calls": att.get("tool_calls") or []}
    final = None
    if row.get("final_finish_reason") is not None or fm:
        final = {"finish_reason": row.get("final_finish_reason"), "content": fm.get("content"),
                 "tool_calls": fm.get("tool_calls") or []}
    return {"final": final, "raw": raw, "error_kind": row.get("error_kind"), "clipped": bool(row.get("clipped"))}


def move_of_recorded(rec: dict) -> dict:
    rr = rec.get("recorded_response") or {}
    msg = rr.get("message") or {}
    return {"final": {"finish_reason": rr.get("finish_reason"), "content": msg.get("content"),
                      "tool_calls": replay_tool.calls_from_message(msg)},
            "raw": None, "error_kind": None, "clipped": False}


def _args_for(role: str, call: dict, decode_json_strings: bool = False) -> dict:
    if role == "E":
        return call_args(call, decode_json_strings)
    return parse_args_json(call.get("arguments")) or {}


def score_move(rec: dict, move: dict, state: Optional[PromptState], *, execute: bool = True,
               exec_timeout: int = 60, novelty: str = "tfidf", phi_cache: Optional[dict] = None,
               synp_meta: Optional[Path] = None, decode_json_strings: bool = False) -> dict:
    """All §2 scores for one (prompt, target) move.

    ``decode_json_strings``: HQ #315 scoring sensitivity (default off = pre-registered scoring):
    JSON-string object arguments are decoded before the required-key check
    (``replay_tool.missing_required_keys``) and before E's ``args`` wrapper is unwrapped.
    """
    dj = decode_json_strings
    role, stratum = rec["role"], rec["stratum"]
    tools = rec["request"].get("tools") or []
    thr = replay_tool.ANSWER_MIN_CHARS.get(role, 0)
    row: dict[str, Any] = {"prompt_id": rec["prompt_id"], "role": role, "stage": rec["stage"],
                           "stratum": stratum, "error_kind": move.get("error_kind"),
                           "context_error": move.get("error_kind") == "context",
                           "infra_error": move.get("error_kind") == "infra",
                           "timeout": move.get("error_kind") == "timeout", "clipped": move.get("clipped")}
    raw = move.get("raw")
    if raw is not None:
        row["rp1_valid_call_raw"] = bool(replay_tool.classify_move(
            raw["finish_reason"], raw["tool_calls"], raw["content"], tools, answer_min_chars=thr,
            decode_json_strings=dj)["valid"])
    else:
        row["rp1_valid_call_raw"] = False
    fin = move.get("final")
    if fin is None:
        row.update(rp1_valid_call_rescued=False, rp1_usable=False, rp1_executes=None, move=None)
        return row
    v = replay_tool.classify_move(fin["finish_reason"], fin["tool_calls"], fin["content"], tools,
                                  answer_min_chars=thr, decode_json_strings=dj)
    row["rp1_valid_call_rescued"] = bool(v["valid"])
    row["valid_reason"] = v["reason"]
    calls = fin["tool_calls"]
    row["move"] = "action" if calls else "answer"
    row["n_calls"] = len(calls)
    row["answer_instead_of_action"] = rec.get("recorded_move") == "action" and not calls
    row["rp1_executes"] = None
    if not v["valid"]:
        row["rp1_usable"] = False
        return row
    text = fin.get("content") or ""

    if not calls:  # ------------------------------------------------ answers
        if role == "E":
            usable = False  # never read by the loop (§2)
        elif role == "S":
            names = state.s_names() if state else []
            cos, novel = focus_novelty(text, rec["state"].get("focus_history") or [], novelty)
            row.update(s_names_column=names_any(text, names), s_additive=is_additive(text),
                       s_pruning_only=is_pruning_only(text), s_focus_max_cos=cos, s_novel=novel,
                       novelty_method=novelty)
            usable = row["s_names_column"] and row["s_additive"] and novel
        elif role == "T" and rec["stage"] == "assessment":
            act = state.t_active() if state else []
            sec = t_sections_present(text)
            share = share_named(text, act)
            row.update(t_sections=sec, t_named_share=share)
            usable = sec >= 3 and share is not None and share >= 0.5
        elif role == "W":
            usable = len(text) > 0 and text.strip() not in W_GIVE_UP
        else:  # T pruning-stage answer: valid only
            usable = True
        row["rp1_usable"] = bool(usable)
        row["flags"] = ";".join(sorted(state.flags)) if state else ""
        return row

    # ---------------------------------------------------------------- actions
    code_calls = [c for c in calls if c["name"] in CODE_TOOLS.get(role, ())]
    exec_res: list[dict] = []
    if code_calls and execute and state is not None:
        exec_res = run_exec(state.exec_spec(role), [{"name": c["name"], "args": _args_for(role, c, dj)}
                                                    for c in code_calls], timeout_s=exec_timeout)
    row["auto_unprune"] = any(r.get("auto_unprune") for r in exec_res)
    row["suspicious"] = any(r.get("suspicious") for r in exec_res)
    if exec_res:
        counted = [r for r in exec_res if not r.get("auto_unprune")]
        row["rp1_executes"] = all(r.get("executes") for r in counted) if counted else None
        row["exec_errors"] = ";".join(str(r.get("error") or r.get("precheck") or r.get("reject") or "")
                                      for r in exec_res)
    flags = set(state.flags) if state else set()
    if any(r.get("reconstruction_partial") for r in exec_res):
        flags.add("reconstruction_partial")
    res_iter = iter(exec_res)
    ok: list[Optional[bool]] = []
    prune_names: list[str] = []
    for c in calls:
        a = _args_for(role, c, dj)
        name = c["name"]
        if name in CODE_TOOLS.get(role, ()):
            r = next(res_iter, None)
            if r is None:
                ok.append(None)  # not executed (--no-exec): usable unknown
            elif name == "append_new_attribute":
                ok.append(bool(r.get("usable_append")))
            else:
                ok.append(bool(r.get("executes")))
        elif name == "attribute_lookup_tool":
            names = a.get("attribute_names_list") or []
            known = state.all_attribute_names() if state else set()
            ok.append(bool(names) and all(str(n) in known for n in names))
        elif name == "attribute_pruning_tool":
            names = [str(n) for n in (a.get("attribute_names_list") or [])]
            prune_names += names
            act = set(state.t_active()) if state else set()
            ok.append(bool(names) and all(n in act for n in names))
        elif name == "_web_search":
            ok.append(w_query_usable(a.get("query")))
        else:
            ok.append(True)  # every other action: valid
    if prune_names:
        recorded = [str(n) for c in replay_tool.calls_from_message(
            (rec.get("recorded_response") or {}).get("message") or {})
            if c.get("name") == "attribute_pruning_tool"
            for n in (call_args(c).get("attribute_names_list") or [])]
        row["t_prune_list"] = ";".join(prune_names)
        row["jaccard_O120"] = jaccard(prune_names, recorded) if recorded else None
        ja = ja_prune_set(parse_metric_table(rec["request"].get("messages") or []),
                          protected=state.seeded_names() if state else ())
        row["jaccard_Ja"] = jaccard(prune_names, ja) if ja is not None else None
        row["phi_check"] = "n/a"
        if str(rec["provenance"].get("dataset_id", "")).startswith("synp_"):
            rec_names = phi_recovering(rec, state, synp_meta, phi_cache)
            if rec_names is None:
                row["phi_check"] = "unavailable"
            else:
                hit = sorted(set(prune_names) & rec_names)
                row["phi_check"] = ("pruned_recovering:" + ",".join(hit)) if hit else "ok"
                if hit:
                    ok.append(False)
    row["rp1_usable"] = None if any(o is None for o in ok) else all(ok)
    row["flags"] = ";".join(sorted(flags))
    return row


def phi_recovering(rec: dict, state: Optional[PromptState], synp_meta: Optional[Path],
                   cache: Optional[dict]) -> Optional[set[str]]:
    """Names in the trial's entity table that the E8a detector counts as recovering φ."""
    if state is None or synp_meta is None or state.entities_index is None or not state.entities_index.exists():
        return None
    key = (rec["provenance"]["run_name"], rec["provenance"]["index"])
    if cache is not None and key in cache:
        return cache[key]
    try:
        import pandas as pd  # noqa: PLC0415

        import recovery_detector as rd  # noqa: PLC0415

        df_raw = pd.read_csv(state.data_csv)
        ents = pd.read_csv(state.entities_index)
        phi, det = rd.load_phi(synp_meta, rec["provenance"]["dataset_id"], df_raw)
        mask = rd.train_mask_for_fold(df_raw, None, 42, int(rec["provenance"]["fold_run_id"]))
        feats = df_raw[["id"]].merge(ents, on="id", how="left")
        pf = rd.evaluate_feature_table(feats[mask].reset_index(drop=True), phi[mask],
                                       df_raw[mask].reset_index(drop=True), det)
        out: Optional[set[str]] = set(pf.loc[pf["counted"], "feature"].astype(str))
    except Exception:
        out = None
    if cache is not None:
        cache[key] = out
    return out


# =========================================================================== fetch / score CLIs
def read_prompts(path: Path) -> list[dict]:
    p = Path(path)
    if p.is_dir():
        p = replay_tool._pick_file(p)
    return replay_tool.read_records(p)


def cmd_fetch(args: argparse.Namespace) -> int:
    import wandb  # noqa: PLC0415

    api = wandb.Api()
    work = Path(args.out)
    art = api.artifact(f"{ENTITY}/{args.project}/{args.prompts}")
    pdir = Path(art.download(root=str(work / "prompts")))
    print(f"[rp1 fetch] prompts {art.name} -> {pdir}")
    for rung in args.rungs.split(","):
        try:
            a = api.artifact(f"{ENTITY}/{args.project}/rp1-replies-{rung}:latest")
            a.download(root=str(work / "replies" / rung))
            print(f"[rp1 fetch] replies {rung}: {a.version}")
        except Exception as exc:  # a target not run yet
            print(f"[rp1 fetch] replies {rung}: missing ({type(exc).__name__})")
    need: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for r in read_prompts(pdir):
        prov, st = r["provenance"], r["state"]
        for files in (st.get("trail_k_files"), st.get("trail_index_files")):
            for f in (files or {}).values():
                need[(prov["run_name"], prov["run_output_artifact"])].add(f)
    for (run_name, ref), files in sorted(need.items()):
        a = api.artifact(ref)
        root = work / "sources" / run_name / "run-output"
        for f in sorted(files):
            if not (root / f).exists():
                try:
                    a.get_entry(f).download(root=str(root))
                except Exception as exc:
                    print(f"[rp1 fetch] {run_name}: {f} missing ({type(exc).__name__})")
        print(f"[rp1 fetch] {run_name}: {len(files)} state files")
    return 0


def load_replies(work: Path) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for d in sorted((work / "replies").glob("*")):
        f = next(iter(sorted(d.rglob("replay_calls.jsonl"))), None)
        if f is not None:
            out[d.name] = {r["prompt_id"]: r for r in replay_tool.read_records(f)}
    return out


def score_all(prompts: list[dict], replies: dict[str, dict[str, dict]], *, sources_root: Path,
              tasks_root: Path, execute: bool = True, exec_timeout: int = 60, jobs: int = 8,
              novelty: str = "tfidf", synp_meta: Optional[Path] = None,
              decode_json_strings: bool = False) -> list[dict]:
    """Score every (target, prompt) plus ``recorded_O120``; marks the common set.

    ``decode_json_strings``: HQ #315 sensitivity (see :func:`score_move`); default off."""
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    states = {r["prompt_id"]: PromptState(r, sources_root, tasks_root) for r in prompts}
    phi_cache: dict = {}
    work_items = []
    for rec in prompts:
        work_items.append(("recorded_O120", rec, move_of_recorded(rec)))
        for target, rows in sorted(replies.items()):
            row = rows.get(rec["prompt_id"])
            move = move_of_reply(row) if row else {"final": None, "raw": None, "error_kind": "missing",
                                                   "clipped": False}
            work_items.append((target, rec, move))

    def one(item):
        target, rec, move = item
        out = score_move(rec, move, states[rec["prompt_id"]], execute=execute, exec_timeout=exec_timeout,
                         novelty=novelty, phi_cache=phi_cache, synp_meta=synp_meta,
                         decode_json_strings=decode_json_strings)
        out["target"] = target
        out["run_name"] = rec["provenance"]["run_name"]
        out["fold_episode"] = f"{rec['provenance']['run_name']}#f{rec['provenance']['fold_run_id']}"
        if target == "recorded_O120":
            out["rp1_valid_call_raw"] = None  # the trace holds no attempt-1 response
        return out

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        rows = list(ex.map(one, work_items))
    excluded = {r["prompt_id"] for r in rows if r["target"] != "recorded_O120"
                and (r.get("context_error") or r.get("infra_error") or r.get("error_kind") == "missing")}
    for r in rows:
        r["common_set"] = r["prompt_id"] not in excluded
    return rows


SCORES = ("rp1_valid_call_raw", "rp1_valid_call_rescued", "rp1_executes", "rp1_usable")


def rate_table(rows: list[dict]) -> list[dict]:
    """Descriptive rates per (target, role or stratum, score) on the common set (§2 denominators)."""
    keyed = collections.defaultdict(list)
    for r in rows:
        if r.get("common_set"):
            keyed[(r["target"], r["role"])].append(r)
            keyed[(r["target"], f"stratum:{r['stratum']}")].append(r)
    out = []
    for (target, key), rr in sorted(keyed.items()):
        for s in SCORES:
            den = [r for r in rr if r.get(s) is not None and not (s == "rp1_executes" and r.get("auto_unprune"))]
            num = sum(1 for r in den if r.get(s))
            out.append({"target": target, "role": key, "score": s, "num": num, "den": len(den),
                        "rate": num / len(den) if den else None})
    return out


SCORED_COLUMNS = (
    "target prompt_id role stage stratum run_name fold_episode common_set rp1_valid_call_raw "
    "rp1_valid_call_rescued rp1_executes rp1_usable move n_calls answer_instead_of_action valid_reason "
    "error_kind context_error infra_error timeout clipped auto_unprune suspicious flags exec_errors "
    "s_names_column s_additive s_pruning_only s_focus_max_cos s_novel novelty_method t_sections "
    "t_named_share t_prune_list jaccard_O120 jaccard_Ja phi_check"
).split()


def cmd_score(args: argparse.Namespace) -> int:
    work = Path(args.work)
    prompts = read_prompts(Path(args.prompts) if args.prompts else work / "prompts")
    replies = load_replies(work)
    t0 = time.time()
    rows = score_all(prompts, replies, sources_root=work / "sources", tasks_root=Path(args.tasks_root),
                     execute=not args.no_exec, exec_timeout=args.exec_timeout, jobs=args.jobs,
                     novelty=args.novelty, synp_meta=Path(args.synp_meta) if args.synp_meta else None,
                     decode_json_strings=args.decode_json_string_args)
    out = work / args.out_subdir
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rp1_scored.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(SCORED_COLUMNS), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(out / "rp1_rates.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["target", "role", "score", "num", "den", "rate"])
        w.writeheader()
        w.writerows(rate_table(rows))
    summ = {"n_rows": len(rows), "targets": sorted({r["target"] for r in rows}),
            "common_set": len({r["prompt_id"] for r in rows if r["common_set"]}),
            "score_s": round(time.time() - t0, 1), "novelty_method": args.novelty,
            "exec": not args.no_exec, "exec_timeout": args.exec_timeout}
    if args.decode_json_string_args:  # key absent by default: the primary summary is unchanged
        summ["decode_json_string_args"] = True
    (out / "rp1_score_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=2))
    return 0


def cmd_attach_log(args: argparse.Namespace) -> int:
    import wandb  # noqa: PLC0415

    run = wandb.init(entity=ENTITY, project=args.project, id=args.run_id, resume="must")
    art = wandb.Artifact(args.artifact, type="server-log")
    art.add_file(args.file)
    run.log_artifact(art)
    run.finish()
    print(f"[rp1 attach-log] {args.file} -> run {args.run_id} as {args.artifact}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["_exec"]:  # child of run_exec
        data = json.loads(sys.stdin.read())
        res = _child_exec(data["spec"], data["calls"], int(data.get("timeout") or 60))
        sys.stdout.write("\n" + json.dumps(res, default=str) + "\n")
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="A18-T6: draw rp1-prompts")
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--from-dir", default=None, help="pre-fetched source runs (tests/offline)")
    s.add_argument("--wandb", action="store_true", help="log artifact rp1-prompts")
    s.add_argument("--allow-redraw", action="store_true")
    s.add_argument("--project", default=PROJECT)
    s.add_argument("--run-name", default=None)
    s.add_argument("--image-tag", default=os.getenv("ROGUE_ONE_IMAGE_TAG"))
    f = sub.add_parser("fetch", help="A18-T16: prompts, replies and state files")
    f.add_argument("--out", required=True)
    f.add_argument("--project", default=PROJECT)
    f.add_argument("--prompts", default=f"{PROMPTS_ARTIFACT}:v0")
    f.add_argument("--rungs", default="Q0.8,Q2,Q4,Q9,Q122,O120")
    c = sub.add_parser("score", help="A18-T9/T16: score every (target, prompt), offline")
    c.add_argument("--work", required=True)
    c.add_argument("--prompts", default=None)
    c.add_argument("--tasks-root", default="/workspace/tasks")
    c.add_argument("--synp-meta", default=None)
    c.add_argument("--exec-timeout", type=int, default=60)
    c.add_argument("--jobs", type=int, default=16)
    c.add_argument("--novelty", choices=("tfidf", "off"), default="tfidf")
    c.add_argument("--no-exec", action="store_true")
    c.add_argument("--decode-json-string-args", action="store_true",
                   help="HQ #315 sensitivity: decode JSON-string object args before the required-key "
                        "check and E's args unwrapping (default off = pre-registered scoring)")
    c.add_argument("--out-subdir", default="scored",
                   help="output folder under --work (e.g. scored_315 for the #315 sensitivity)")
    a = sub.add_parser("attach-log", help="upload a file to an existing replay run")
    a.add_argument("--run-id", required=True)
    a.add_argument("--file", required=True)
    a.add_argument("--artifact", required=True)
    a.add_argument("--project", default=PROJECT)
    args = ap.parse_args(argv)
    return {"sample": cmd_sample, "fetch": cmd_fetch, "score": cmd_score,
            "attach-log": cmd_attach_log}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
