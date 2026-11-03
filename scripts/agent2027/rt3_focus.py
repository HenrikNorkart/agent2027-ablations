#!/usr/bin/env python3
"""RT3 focus files for A09 (E9c tooling; A09 §4.2, A09-T4/T10; pre-registration Part V.9).

Builds one focus file per A09 arm row, read by the E9c planner hook
(``ROGUE_ONE_PLANNER_MODE=oracle|misdirected``, ``rogueone/utils/planner_mode.py``)::

    {"dataset_id": "synp_p3", "arm_id": "oracle", "focus": {"0:0": "...", ..., "4:9": "..."}}

* ``oracle``: the ``oracle_focus`` string of the task's off-mount metadata
  (``synp_meta/<id>.json``, written by E8a) at all 50 (fold, iteration) keys.
* ``misdirected``: A05's seed-42 focus strings on σ(T) (``rt3_sigma_map.yaml``), read from
  the E3 focus log (``focus_log.jsonl`` in each run's ``run-output`` W&B artifact), at
  the same (fold, iteration), with σ(T)'s column names renamed to T's decoys.

Renaming rule (A09 §4.2 ‡):
  1. I_T = φ_T's input columns. D_T = T's other raw columns, sorted by ascending
     max |Spearman| with I_T (categoricals as ordinal codes; ties by column order).
  2. C_S = σ(T)'s raw-column names that occur in any source string, φ_S's inputs first,
     the rest in order of first occurrence.
  3. C_S[k] -> the k-th unused D_T column of the same type (numeric / categorical),
     falling back to any type; D_T is reused cyclically when it runs out.
  4. One pass, longest name first, case-sensitive; a match must not touch a letter or
     digit (``_`` is a boundary, so ``LW_x_LD`` is rewritten). Numbers and level labels
     stay unchanged.
  5. Assert: no I_T name remains and every C_S occurrence was replaced.

The focus file is minimal; the rename map and provenance go to a sidecar
``<out>/meta/<row_id>.json`` that is never mounted (it reveals which columns are decoys).

Subcommands::

    rt3_focus.py extract --run A05__full__synp_p4__f42__r0 [--cache DIR]      # dump one focus log
    rt3_focus.py rename  --target synp_p3 --source synp_p4 --focus-log F --meta M --tasks T
    rt3_focus.py build   --manifest A09_RT3.csv --sigma rt3_sigma_map.yaml --meta /meta \\
                         --tasks /tasks --source-group A05 --out /out [--verify]
    rt3_focus.py verify  --manifest A09_RT3.csv --sigma ... --meta /meta --tasks /tasks --out /out

Run on CAIR in a CPU container only (A09-T10). Source focus logs are fetched from W&B
(entity ``PlanBench``, project ``RogueOne-AGENT2027``; ``WANDB_API_KEY`` from the env file)
into ``<out>/meta/source_logs/`` unless ``--cache`` points elsewhere. Exit 0 = ok,
1 = verification failed, 2 = usage / input error.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_ENTITY = "PlanBench"
DEFAULT_PROJECT = "RogueOne-AGENT2027"
ARTIFACT_TYPE = "run-output"
FOCUS_LOG = "focus_log.jsonl"
FILE_ARMS = ("oracle", "misdirected")
SIGMA_DEFAULT = Path(__file__).resolve().parent / "rt3_sigma_map.yaml"


class RT3Error(Exception):
    """Input or rule violation; the CLI exits 2 (input) or 1 (verify)."""


# --------------------------------------------------------------------------- small io
def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def dump_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def load_sigma(path: str | Path) -> dict:
    import yaml

    sigma = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("source_ablation", "source_arm", "source_fold_seed", "source_replicates",
                "source_run_pattern", "min_rows_per_fold", "k_folds", "num_iterations", "map"):
        if key not in sigma:
            raise RT3Error(f"{path}: missing {key!r}")
    for t, entry in sigma["map"].items():
        if entry.get("source") == t or entry.get("fallback") == t:
            raise RT3Error(f"{path}: σ maps {t} to itself")
        if entry.get("source") == "synp_p7" or entry.get("fallback") == "synp_p7":
            raise RT3Error(f"{path}: synp_p7 is never a source")
    return sigma


def read_manifest(path: str | Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8-sig")
    return [dict(r) for r in csv.DictReader(io.StringIO(text))]


def focus_rows_of_manifest(rows: list[dict], only: Optional[Iterable[str]] = None) -> list[dict]:
    """Manifest rows that carry a file planner mode (the rows that get a focus file)."""
    wanted = set(only or [])
    out = []
    for r in rows:
        ov = json.loads(r.get("env_overrides") or "{}")
        mode = str(ov.get("ROGUE_ONE_PLANNER_MODE", "")).strip().lower()
        if mode not in FILE_ARMS:
            continue
        if r["arm_id"] != mode:
            raise RT3Error(f"{r['row_id']}: arm_id {r['arm_id']!r} != ROGUE_ONE_PLANNER_MODE {mode!r}")
        if wanted and r["row_id"] not in wanted:
            continue
        out.append(r)
    return out


def load_meta(meta_dir: str | Path, dataset_id: str) -> Optional[dict]:
    p = Path(meta_dir) / f"{dataset_id}.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def all_metas(meta_dir: str | Path) -> dict[str, dict]:
    out = {}
    for p in sorted(Path(meta_dir).glob("synp_*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        out[m.get("task_id", p.stem)] = m
    return out


def task_dir(tasks_root: str | Path, dataset_id: str) -> Path:
    hits = sorted(p for p in Path(tasks_root).glob(f"*/{dataset_id}") if (p / "data.csv").is_file())
    if len(hits) != 1:
        raise RT3Error(f"expected one task folder {tasks_root}/*/{dataset_id} with data.csv, found {hits}")
    return hits[0]


def raw_frame(tasks_root: str | Path, dataset_id: str):
    import pandas as pd

    df = pd.read_csv(task_dir(tasks_root, dataset_id) / "data.csv")
    return df.drop(columns=[c for c in df.columns if c == "id"])


def phi_inputs(meta: Optional[dict]) -> list[str]:
    """φ's input columns in φ's variable order; [] when there is no planted φ (AB3)."""
    if not meta:
        return []
    return list(dict.fromkeys(meta["phi"]["cols"].values()))


def is_categorical(series) -> bool:
    import pandas as pd

    return not pd.api.types.is_numeric_dtype(series)


# --------------------------------------------------------------------------- focus logs
def parse_focus_log(path: str | Path) -> dict[str, dict]:
    """``focus_log.jsonl`` -> {"f:i": {"focus", "source"}}. A repeated (f, i) (fold retry)
    keeps the last row, i.e. the attempt that counted."""
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = f"{int(r['fold_run_id'])}:{int(r['iteration_in_fold'])}"
            out[key] = {"focus": r["focus"] if isinstance(r["focus"], str) else str(r["focus"]),
                        "source": r.get("source", "")}
    return out


def fold_row_count(log: dict[str, dict], fold: int) -> int:
    return sum(1 for k in log if int(k.split(":")[0]) == fold)


def source_run_name(sigma: dict, dataset_id: str, replicate: int) -> str:
    return sigma["source_run_pattern"].format(
        ablation=sigma["source_ablation"], arm=sigma["source_arm"], dataset=dataset_id,
        fold_seed=sigma["source_fold_seed"], replicate=replicate,
    )


class LogStore:
    """Focus logs by run name: a local cache dir (``<cache>/<run>/focus_log.jsonl``), filled
    from W&B on demand (``fetch=True``)."""

    def __init__(self, cache: str | Path, *, fetch: bool = True, entity: str = DEFAULT_ENTITY,
                 project: str = DEFAULT_PROJECT, group: Optional[str] = None):
        self.cache = Path(cache)
        self.fetch = fetch
        self.entity, self.project, self.group = entity, project, group
        self._api = None
        self.provenance: dict[str, dict] = {}

    def _path(self, run: str) -> Path:
        return self.cache / re.sub(r"[^A-Za-z0-9_.-]+", "_", run) / FOCUS_LOG

    def get(self, run: str) -> Optional[dict[str, dict]]:
        p = self._path(run)
        if not p.is_file() and self.fetch:
            self._download(run, p)
        if not p.is_file():
            return None
        prov = p.parent / "provenance.json"
        if prov.is_file():
            self.provenance[run] = json.loads(prov.read_text(encoding="utf-8"))
        return parse_focus_log(p)

    def _download(self, run: str, dest: Path) -> None:
        import wandb  # lazy: CAIR analysis container only

        if self._api is None:
            self._api = wandb.Api(timeout=120)
        filters: dict[str, Any] = {"display_name": run}
        if self.group:
            filters["group"] = self.group
        runs = list(self._api.runs(f"{self.entity}/{self.project}", filters=filters))
        if not runs:
            return
        # Same name twice (infra rerun): prefer finished, then the newest.
        runs.sort(key=lambda r: (r.state == "finished", str(r.created_at)), reverse=True)
        chosen = runs[0]
        art = next((a for a in chosen.logged_artifacts() if a.type == ARTIFACT_TYPE), None)
        if art is None:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            entry = art.get_entry(FOCUS_LOG)
        except KeyError:
            return
        got = Path(entry.download(root=str(dest.parent)))
        if got.resolve() != dest.resolve():
            got.replace(dest)
        (dest.parent / "provenance.json").write_text(dump_json({
            "run_name": run, "run_id": chosen.id, "state": chosen.state,
            "created_at": str(chosen.created_at), "artifact": f"{art.name}",
            "candidates": [{"run_id": r.id, "state": r.state, "created_at": str(r.created_at)} for r in runs],
        }), encoding="utf-8")


# --------------------------------------------------------------------------- renaming
_WORD = r"[^\W_]"  # a letter or digit (unicode); "_" and punctuation are boundaries


def name_regex(names: Iterable[str]) -> Optional[re.Pattern]:
    names = sorted(set(names), key=lambda s: (-len(s), s))
    if not names:
        return None
    alt = "|".join(re.escape(n) for n in names)
    return re.compile(rf"(?<!{_WORD})(?:{alt})(?!{_WORD})")


def names_in(text: str, names: Iterable[str]) -> list[str]:
    """Names occurring in ``text`` under the rule-4 boundary, in order of first occurrence
    (longest-first at any position, as the rename pass would match them)."""
    rx = name_regex(names)
    if rx is None:
        return []
    return list(dict.fromkeys(m.group(0) for m in rx.finditer(text)))


def rank_decoys(df_t, inputs: list[str]) -> list[str]:
    """D_T: T's raw columns minus I_T, ascending max |Spearman| with I_T; ties by column order."""
    import numpy as np
    import pandas as pd

    def codes(s):
        if is_categorical(s):
            return pd.Series(pd.factorize(s, sort=True)[0], index=s.index, dtype=float)
        return s.astype(float)

    others = [c for c in df_t.columns if c not in inputs]
    if not inputs:
        return others
    ranks = {c: codes(df_t[c]).rank() for c in list(df_t.columns)}
    score = {}
    for c in others:
        vals = []
        for i in inputs:
            r = ranks[c].corr(ranks[i])
            vals.append(0.0 if r is None or np.isnan(r) else abs(float(r)))
        score[c] = max(vals)
    return sorted(others, key=lambda c: (score[c], others.index(c)))


def build_rename_map(c_s: list[str], s_types: dict[str, bool], d_t: list[str],
                     t_types: dict[str, bool]) -> dict[str, str]:
    """Rule 3. ``*_types`` map a column to True if categorical."""
    if c_s and not d_t:
        raise RT3Error("target has no decoy columns to rename into")
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for name in c_s:
        if len(used) == len(d_t):
            used.clear()  # cyclic reuse
        free = [c for c in d_t if c not in used]
        same = [c for c in free if t_types[c] == s_types.get(name, False)]
        pick = (same or free)[0]
        mapping[name] = pick
        used.add(pick)
    return mapping


def apply_rename(text: str, mapping: dict[str, str]) -> str:
    rx = name_regex(mapping)
    if rx is None:
        return text
    return rx.sub(lambda m: mapping[m.group(0)], text)


@dataclass
class RenamePlan:
    target: str
    source: str
    inputs_t: list[str]
    decoys_t: list[str]
    c_s: list[str]
    mapping: dict[str, str]
    s_types: dict[str, bool] = field(repr=False, default_factory=dict)


def plan_rename(target: str, source: str, source_strings: list[str], *, tasks_root, meta_dir) -> RenamePlan:
    df_t = raw_frame(tasks_root, target)
    df_s = raw_frame(tasks_root, source)
    meta_t, meta_s = load_meta(meta_dir, target), load_meta(meta_dir, source)
    if target.startswith("synp_") and meta_t is None:
        raise RT3Error(f"no metadata for {target} in {meta_dir}")
    inputs_t = phi_inputs(meta_t)
    missing = [c for c in inputs_t if c not in df_t.columns]
    if missing:
        raise RT3Error(f"{target}: φ inputs {missing} are not columns of data.csv")
    decoys = rank_decoys(df_t, inputs_t)
    s_cols = list(df_s.columns)
    occurring: list[str] = []
    for s in source_strings:
        occurring.extend(names_in(s, s_cols))
    occurring = list(dict.fromkeys(occurring))
    phi_s = [c for c in phi_inputs(meta_s) if c in occurring]
    c_s = phi_s + [c for c in occurring if c not in phi_s]
    s_types = {c: is_categorical(df_s[c]) for c in s_cols}
    t_types = {c: is_categorical(df_t[c]) for c in df_t.columns}
    mapping = build_rename_map(c_s, s_types, decoys, t_types)
    return RenamePlan(target, source, inputs_t, decoys, c_s, mapping, s_types)


def rename_checked(text: str, plan: RenamePlan) -> str:
    """Rules 4-5 on one string."""
    out = apply_rename(text, plan.mapping)
    left_inputs = names_in(out, plan.inputs_t)
    if left_inputs:
        raise RT3Error(f"{plan.target}: φ input name(s) {left_inputs} remain after renaming")
    targets = set(plan.mapping.values())
    left_src = [n for n in names_in(out, plan.c_s) if n not in targets]
    if left_src:
        raise RT3Error(f"{plan.target}: source name(s) {left_src} were not replaced")
    return out


# --------------------------------------------------------------------------- focus files
def keys(k_folds: int, n_iter: int) -> list[str]:
    return [f"{f}:{i}" for f in range(k_folds) for i in range(n_iter)]


def oracle_file(row: dict, meta: dict, k_folds: int, n_iter: int) -> tuple[dict, dict]:
    s = meta.get("oracle_focus")
    if not isinstance(s, str) or not s.strip():
        raise RT3Error(f"{row['row_id']}: metadata of {row['dataset_id']} has no oracle_focus")
    doc = {"dataset_id": row["dataset_id"], "arm_id": "oracle", "focus": {k: s for k in keys(k_folds, n_iter)}}
    side = {"row_id": row["row_id"], "dataset_id": row["dataset_id"], "arm_id": "oracle",
            "oracle_focus_sha256": sha256_text(s), "meta_task_id": meta.get("task_id")}
    return doc, side


def choose_source_folds(target: str, sigma: dict, store: LogStore) -> tuple[dict[int, dict], dict]:
    """Per fold: the first of σ r0, σ r1, fallback r0, fallback r1 with >= min rows."""
    entry = sigma["map"].get(target)
    if not entry:
        raise RT3Error(f"σ map has no entry for {target}")
    k, minimum = int(sigma["k_folds"]), int(sigma["min_rows_per_fold"])
    chain = []
    for ds in (entry.get("source"), entry.get("fallback")):
        if ds:
            for rep in sigma["source_replicates"]:
                chain.append((ds, int(rep), source_run_name(sigma, ds, int(rep))))
    logs = {name: store.get(name) for _, _, name in chain}
    chosen: dict[int, dict] = {}
    for f in range(k):
        for step, (ds, rep, name) in enumerate(chain):
            log = logs[name]
            if log is not None and fold_row_count(log, f) >= minimum:
                chosen[f] = {"dataset": ds, "replicate": rep, "run": name, "chain_step": step, "log": log}
                break
        else:
            raise RT3Error(f"{target}: no source run in {[c[2] for c in chain]} has fold {f} complete")
    info = {"chain": [c[2] for c in chain],
            "available": {name: (None if lg is None else {str(f): fold_row_count(lg, f) for f in range(k)})
                          for name, lg in logs.items()}}
    return chosen, info


def misdirected_file(row: dict, sigma: dict, store: LogStore, *, tasks_root, meta_dir) -> tuple[dict, dict]:
    target = row["dataset_id"]
    k, n = int(sigma["k_folds"]), int(sigma["num_iterations"])
    chosen, info = choose_source_folds(target, sigma, store)
    focus_src: dict[str, dict] = {}
    for f in range(k):
        log = chosen[f]["log"]
        for i in range(n):
            key = f"{f}:{i}"
            if key not in log:
                raise RT3Error(f"{target}: source {chosen[f]['run']} lacks {key}")
            focus_src[key] = {**log[key], "source_dataset": chosen[f]["dataset"]}
    # One rename plan per source dataset (a fallback fold may come from the fallback task).
    plans: dict[str, RenamePlan] = {}
    for ds in dict.fromkeys(v["source_dataset"] for v in focus_src.values()):
        strings = [v["focus"] for v in focus_src.values() if v["source_dataset"] == ds]
        plans[ds] = plan_rename(target, ds, strings, tasks_root=tasks_root, meta_dir=meta_dir)
    focus = {key: rename_checked(v["focus"], plans[v["source_dataset"]]) for key, v in focus_src.items()}
    doc = {"dataset_id": target, "arm_id": "misdirected", "focus": focus}
    side = {
        "row_id": row["row_id"], "dataset_id": target, "arm_id": "misdirected",
        "sigma": sigma["map"][target],
        "folds": {str(f): {"run": chosen[f]["run"], "dataset": chosen[f]["dataset"],
                           "replicate": chosen[f]["replicate"], "chain_step": chosen[f]["chain_step"],
                           "fallback_used": chosen[f]["chain_step"] > 0,
                           "run_provenance": store.provenance.get(chosen[f]["run"])}
                  for f in range(k)},
        "source_availability": info,
        # S's fallback strings are kept verbatim and counted (A09 §4.2).
        "heuristic_focus_count": sum(1 for v in focus_src.values() if v.get("source") == "heuristic"),
        "source_counts": {src: sum(1 for v in focus_src.values() if str(v.get("source")) == src)
                          for src in sorted({str(v.get("source")) for v in focus_src.values()})},
        "rename": {ds: {"inputs_t": p.inputs_t, "decoys_t_order": p.decoys_t, "c_s": p.c_s,
                        "map": p.mapping} for ds, p in plans.items()},
        "source_focus_sha256": {key: sha256_text(v["focus"]) for key, v in focus_src.items()},
    }
    return doc, side


# --------------------------------------------------------------------------- verify
def verify_files(rows: list[dict], out_dir: Path, sigma: dict, store: LogStore, *, tasks_root,
                 meta_dir) -> list[str]:
    """A09-T4 done-when checks; returns a list of failures (empty = pass)."""
    fails: list[str] = []
    k, n = int(sigma["k_folds"]), int(sigma["num_iterations"])
    want = keys(k, n)
    metas = all_metas(meta_dir)
    for row in rows:
        rid, ds, arm = row["row_id"], row["dataset_id"], row["arm_id"]
        p = out_dir / f"{rid}.json"
        if not p.is_file():
            fails.append(f"{rid}: missing {p}")
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        if doc.get("dataset_id") != ds or doc.get("arm_id") != arm:
            fails.append(f"{rid}: header {doc.get('dataset_id')}/{doc.get('arm_id')} != {ds}/{arm}")
        focus = doc.get("focus") or {}
        if sorted(focus) != sorted(want) or len(focus) != len(want):
            fails.append(f"{rid}: keys are not the {len(want)} '<f>:<i>' keys")
            continue
        if set(doc) != {"dataset_id", "arm_id", "focus"}:
            fails.append(f"{rid}: unexpected top-level keys {sorted(set(doc) - {'dataset_id', 'arm_id', 'focus'})}")
        own = metas.get(ds)
        if arm == "oracle":
            if own is None:
                fails.append(f"{rid}: no metadata for {ds}")
            elif any(focus[key] != own.get("oracle_focus") for key in want):
                fails.append(f"{rid}: oracle strings differ from {ds}'s oracle_focus")
        else:
            try:
                expect, _ = misdirected_file(row, sigma, store, tasks_root=tasks_root, meta_dir=meta_dir)
            except RT3Error as exc:
                fails.append(f"{rid}: cannot recompute ({exc})")
                expect = None
            if expect is not None:
                bad = [key for key in want if focus[key] != expect["focus"][key]]
                if bad:
                    fails.append(f"{rid}: misdirected strings differ from renamed σ(T) focus at {bad[:5]}")
            inputs = phi_inputs(own)
            leaks = sorted({nm for key in want for nm in names_in(focus[key], inputs)})
            if leaks:
                fails.append(f"{rid}: φ input name(s) {leaks} appear in a misdirected string")
        # No file carries another task's φ (a misdirected file carries no φ at all).
        for tid, m in metas.items():
            if arm == "oracle" and tid == ds:
                continue
            needles = [m.get("phi_pandas_expr"), m.get("oracle_focus")]
            hit = [x for x in needles if x and any(x in focus[key] for key in want)]
            if hit:
                fails.append(f"{rid}: contains the φ of {tid}")
    return fails


# --------------------------------------------------------------------------- CLI
def _store(a, out_dir: Optional[Path]) -> LogStore:
    cache = Path(a.cache) if a.cache else (out_dir / "meta" / "source_logs")
    return LogStore(cache, fetch=not a.no_fetch, entity=a.entity, project=a.project, group=a.source_group)


def cmd_extract(a) -> int:
    store = _store(a, Path(a.out) if a.out else Path("."))
    for run in a.run:
        log = store.get(run)
        if log is None:
            print(f"{run}: no focus log", file=sys.stderr)
            return 2
        per_fold = {f: fold_row_count(log, f) for f in range(a.k_folds)}
        srcs: dict[str, int] = {}
        for v in log.values():
            srcs[v["source"]] = srcs.get(v["source"], 0) + 1
        print(json.dumps({"run": run, "rows": len(log), "per_fold": per_fold, "sources": srcs}))
    return 0


def cmd_rename(a) -> int:
    log = parse_focus_log(a.focus_log)
    strings = [v["focus"] for v in log.values()]
    plan = plan_rename(a.target, a.source, strings, tasks_root=a.tasks, meta_dir=a.meta)
    print(dump_json({"inputs_t": plan.inputs_t, "decoys_t_order": plan.decoys_t, "c_s": plan.c_s,
                     "map": plan.mapping}))
    if a.show:
        for key, v in log.items():
            print(f"--- {key}\n{rename_checked(v['focus'], plan)}")
    return 0


def cmd_build(a) -> int:
    sigma = load_sigma(a.sigma)
    out_dir = Path(a.out)
    rows = focus_rows_of_manifest(read_manifest(a.manifest), a.rows)
    if not rows:
        raise RT3Error(f"{a.manifest}: no oracle/misdirected rows selected")
    store = _store(a, out_dir)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    report = {"manifest": str(a.manifest), "sigma_sha256": sha256_text(Path(a.sigma).read_text(encoding="utf-8")),
              "files": {}}
    for row in rows:
        if row["arm_id"] == "oracle":
            meta = load_meta(a.meta, row["dataset_id"])
            if meta is None:
                raise RT3Error(f"{row['row_id']}: no metadata {a.meta}/{row['dataset_id']}.json")
            doc, side = oracle_file(row, meta, int(sigma["k_folds"]), int(sigma["num_iterations"]))
        else:
            doc, side = misdirected_file(row, sigma, store, tasks_root=a.tasks, meta_dir=a.meta)
        text = json.dumps(doc, ensure_ascii=False, indent=1) + "\n"
        (out_dir / f"{row['row_id']}.json").write_text(text, encoding="utf-8")
        side["focus_file_sha256"] = sha256_text(text)
        (out_dir / "meta" / f"{row['row_id']}.json").write_text(dump_json(side), encoding="utf-8")
        report["files"][row["row_id"]] = {"dataset_id": row["dataset_id"], "arm_id": row["arm_id"],
                                          "sha256": side["focus_file_sha256"],
                                          "fallback_folds": [f for f, v in side.get("folds", {}).items()
                                                             if v["fallback_used"]],
                                          "heuristic_focus_count": side.get("heuristic_focus_count", 0),
                                          "source_counts": side.get("source_counts", {})}
        print(f"wrote {row['row_id']} ({row['dataset_id']}, {row['arm_id']}) sha256={side['focus_file_sha256'][:12]}")
    rc = 0
    if a.verify:
        fails = verify_files(rows, out_dir, sigma, store, tasks_root=a.tasks, meta_dir=a.meta)
        report["verify"] = {"ok": not fails, "failures": fails}
        for f in fails:
            print(f"VERIFY FAIL {f}", file=sys.stderr)
        print(f"verify: {'PASS' if not fails else 'FAIL'} ({len(rows)} files)")
        rc = 0 if not fails else 1
    (out_dir / "meta" / "build_report.json").write_text(dump_json(report), encoding="utf-8")
    return rc


def cmd_verify(a) -> int:
    sigma = load_sigma(a.sigma)
    out_dir = Path(a.out)
    rows = focus_rows_of_manifest(read_manifest(a.manifest), a.rows)
    fails = verify_files(rows, out_dir, sigma, _store(a, out_dir), tasks_root=a.tasks, meta_dir=a.meta)
    for f in fails:
        print(f"VERIFY FAIL {f}", file=sys.stderr)
    print(f"verify: {'PASS' if not fails else 'FAIL'} ({len(rows)} files)")
    return 0 if not fails else 1


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, *, needs_out=True):
        p.add_argument("--cache", help="focus-log cache dir (default <out>/meta/source_logs)")
        p.add_argument("--no-fetch", action="store_true", help="never call W&B; use the cache only")
        p.add_argument("--entity", default=DEFAULT_ENTITY)
        p.add_argument("--project", default=DEFAULT_PROJECT)
        p.add_argument("--source-group", default="A05", help="W&B group of the source runs")
        p.add_argument("--out", required=needs_out)

    p = sub.add_parser("extract", help="fetch and summarise focus logs of source runs")
    p.add_argument("--run", action="append", required=True)
    p.add_argument("--k-folds", type=int, default=5)
    common(p, needs_out=False)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("rename", help="show the rename plan of one target/source pair")
    p.add_argument("--target", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--focus-log", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--show", action="store_true", help="also print the renamed strings")
    p.set_defaults(func=cmd_rename)

    for name, fn in (("build", cmd_build), ("verify", cmd_verify)):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True)
        p.add_argument("--sigma", default=str(SIGMA_DEFAULT))
        p.add_argument("--meta", required=True, help="off-mount SYN-P metadata dir (synp_<id>.json)")
        p.add_argument("--tasks", required=True, help="task root (tasks/<type>/<dataset>/data.csv)")
        p.add_argument("--rows", nargs="*", help="restrict to these row ids")
        if name == "build":
            p.add_argument("--verify", action="store_true")
        common(p)
        p.set_defaults(func=fn)

    a = ap.parse_args(argv)
    try:
        return a.func(a)
    except RT3Error as exc:
        print(f"rt3_focus: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
