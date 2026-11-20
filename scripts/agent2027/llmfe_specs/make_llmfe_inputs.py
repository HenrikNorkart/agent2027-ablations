#!/usr/bin/env python3
"""X4 · LLM-FE spec files and data for SYN-P 7 (A08-T4; 01 §3.3 X4; pre-registration V.8, VII.1 item 13).

LLM-FE's code is never changed (Henrik, 2026-08-28). We author one spec per SYN-P task (decision
D4, disclosed): each is an UPSTREAM spec of the same task type (LLM-FE e9a58ca, vendored under
``upstream/``: ``specification_heart.txt`` for classification, ``specification_crab.txt`` for
regression) with exactly two edits:

1. the ``<Task>`` description is replaced by the task's ``context_and_goal`` (the same text
   every Rogue One arm reads from ``config.yml``);
2. the seed ``modify_features`` program is replaced by the identity program. The upstream seeds
   build a dataset-specific feature (heart: RestingBP*MaxHR; crab: Weight counts) whose columns
   do not exist here; any SYN-P-specific seed would be a hint about phi, so the seed adds nothing.

``evaluate()`` (LLM-FE's own 4-fold XGBoost search score) is upstream verbatim. Column
descriptions for LLM-FE's ``[FEATURES]`` slot (``<problem>-metadata.json``) are parsed from the
same ``context_and_goal`` column block.

Two subcommands:

``specs`` (developer machine; reads only ``config.yml``) writes the committed files
    scripts/agent2027/llmfe_specs/synp_<id>/specification_synp_<id>.txt
    scripts/agent2027/llmfe_specs/synp_<id>/synp_<id>-metadata.json

``data`` (on CAIR, in a container: ``--network none``, image ``rogue_one:agent2027-v2.N``; the
host only copies the output) writes, for LLM-FE's ``./data`` and ``./specs`` layout
(``paper/scripts/llmfe_launch.sh``: ``~/llmfe/specs/specification_<ds>.txt``, ``./data/<ds>.csv``,
``./data/<ds>-metadata.json``)::

    <out>/specs/specification_synp_<id>.txt         (copied from the committed spec, checked)
    <out>/data/synp_<id>.csv                        (data.csv row order, ``id`` dropped, target last)
    <out>/data/synp_<id>-metadata.json
    <out>/alias/synp_<id>/crab.csv (+ crab-metadata.json)   regression tasks only, see below
    <out>/MANIFEST.json                             (sha256 + row counts of every file)

The CSV is assembled at TEXT level (csv module), so every value is byte-identical to
``data.csv`` / ``labels.csv``; labels are joined on ``id`` and every id must match.

REGRESSION ALIAS. LLM-FE's ``main.py`` decides regression with a hard-coded name list
(``['forest-fires','housing','insurance','bike','wine','crab']``) and otherwise splits with
``StratifiedKFold`` on the label-encoded target, which raises on a continuous target. Without
editing LLM-FE, a regression SYN-P task is therefore run with ``--problem_name crab`` and a
per-task data directory (``alias/synp_<id>/`` mounted at ``/app/data``) whose ``crab.csv`` IS the
task's data. ``problem_name`` is used by ``main.py`` only for the regression test and the two
``./data`` paths, so the alias changes nothing else; ``--log_path`` and ``--spec_path`` keep the
real name. Its split is then ``KFold(5, shuffle=True, random_state=42)``, which is Rogue One's
regression partition at fold seed 42 (verify with ``check_fold_partition.py llmfe``).

HARD RULE: covtype is never used (it is not a SYN-P task; any covtype name is refused).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "upstream"
SYNP7 = ("synp_p1", "synp_p2", "synp_p3", "synp_p4", "synp_p5", "synp_p6a", "synp_p6b")
TEMPLATES = {"classification": "specification_heart.txt", "regression": "specification_crab.txt"}
#: The name main.py treats as regression and whose upstream spec is our regression template.
REGRESSION_ALIAS = "crab"
UPSTREAM_COMMIT = "e9a58ca3416ae74dcc7549dffdf79b9289604091"
TARGET_COL = "target"

IDENTITY_SEED = '''@equation.evolve
def modify_features(df_input) -> pd.DataFrame:
    """
    Thought 1: No transformation yet; this baseline returns the input columns unchanged.
    New Feature 1: none
    """
    df_output = df_input.copy()

    return df_output
'''

_TASK_RE = re.compile(r"(<Task>\n)(.*?)(\n\n###\n<Features>)", re.S)
_COL_RE = re.compile(r"^\s*-\s+([^:]+?):\s*(.*?)\s*$")


def _refuse_covtype(names) -> None:
    for n in names:
        if "covtype" in str(n).lower():
            raise SystemExit("FATAL: covtype is never used in any Rogue One study (Henrik, 2026-10-16)")


def _find_task_dir(tasks_root: Path, task: str) -> Path:
    hits = [d for d in (tasks_root / "classification" / task, tasks_root / "regression" / task)
            if (d / "config.yml").is_file()]
    if len(hits) != 1:
        raise SystemExit(f"FATAL: expected exactly one tasks/<type>/{task}/config.yml under {tasks_root}, found {hits}")
    return hits[0]


def load_task(tasks_root: Path, task: str) -> dict:
    import yaml

    d = _find_task_dir(tasks_root, task)
    cfg = yaml.safe_load((d / "config.yml").read_text(encoding="utf-8"))
    task_type = cfg.get("task_type") or d.parent.name
    if task_type != d.parent.name:
        raise SystemExit(f"FATAL: {task}: config task_type {task_type} vs folder {d.parent.name}")
    context = str(cfg["context_and_goal"]).strip()
    return {"task": task, "dir": d, "task_type": task_type, "context": context}


def column_descriptions(context: str) -> dict[str, str]:
    """``- name: description`` lines of the context's column block, in order."""
    out: dict[str, str] = {}
    in_block = False
    for line in context.splitlines():
        if line.strip() == "Columns:":
            in_block = True
            continue
        if in_block:
            if line.strip().startswith("Goal:"):
                break
            m = _COL_RE.match(line)
            if m:
                out[m.group(1).strip()] = m.group(2).strip()
    if not out:
        raise SystemExit("FATAL: no column block found in context_and_goal")
    return out


def render_spec(task_type: str, context: str) -> str:
    template = (UPSTREAM / TEMPLATES[task_type]).read_text(encoding="utf-8").replace("\r\n", "\n")
    if '"""' in context:
        raise SystemExit("FATAL: context_and_goal contains a triple quote")
    new, n = _TASK_RE.subn(lambda m: m.group(1) + context + m.group(3), template, count=1)
    if n != 1:
        raise SystemExit(f"FATAL: no <Task> section in {TEMPLATES[task_type]}")
    head, sep, _ = new.partition("@equation.evolve")
    if not sep:
        raise SystemExit(f"FATAL: no @equation.evolve in {TEMPLATES[task_type]}")
    return head + IDENTITY_SEED


def spec_task_text(spec: str) -> str:
    m = _TASK_RE.search(spec)
    return m.group(2) if m else ""


# --------------------------------------------------------------------------- specs
def cmd_specs(args) -> int:
    tasks = args.tasks or list(SYNP7)
    _refuse_covtype(tasks)
    for t in tasks:
        if t not in SYNP7:
            raise SystemExit(f"FATAL: {t} is not a SYN-P 7 task (LLM-FE runs SYN-P 7 only, A08 §2)")
        info = load_task(Path(args.tasks_root), t)
        out = Path(args.out) / t
        out.mkdir(parents=True, exist_ok=True)
        spec = render_spec(info["task_type"], info["context"])
        (out / f"specification_{t}.txt").write_bytes(spec.encode("utf-8"))
        meta = column_descriptions(info["context"])
        (out / f"{t}-metadata.json").write_bytes((json.dumps(meta, indent=4) + "\n").encode("utf-8"))
        print(f"{t}: {info['task_type']}, template {TEMPLATES[info['task_type']]}, {len(meta)} columns -> {out}")
    return 0


# --------------------------------------------------------------------------- data
def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def build_table(task_dir: Path) -> tuple[str, int, list[str]]:
    """data.csv (+ labels.csv joined on id) -> LLM-FE CSV text: id dropped, target last."""
    head, rows = _read_csv(task_dir / "data.csv")
    lhead, lrows = _read_csv(task_dir / "labels.csv")
    if head[0] != "id" or lhead[:2] != ["id", TARGET_COL]:
        raise SystemExit(f"FATAL: {task_dir}: unexpected headers {head[:1]} / {lhead}")
    labels = {}
    for r in lrows:
        if r[0] in labels:
            raise SystemExit(f"FATAL: {task_dir}: duplicate label id {r[0]}")
        labels[r[0]] = r[1]
    ids = [r[0] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"FATAL: {task_dir}: duplicate ids in data.csv")
    if set(ids) != set(labels):
        raise SystemExit(f"FATAL: {task_dir}: data.csv and labels.csv ids differ")
    if TARGET_COL in head:
        raise SystemExit(f"FATAL: {task_dir}: a feature column is called {TARGET_COL!r}")
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(head[1:] + [TARGET_COL])
    for r in rows:
        if len(r) != len(head):
            raise SystemExit(f"FATAL: {task_dir}: ragged row id={r[0]}")
        w.writerow(r[1:] + [labels[r[0]]])
    return buf.getvalue(), len(rows), head[1:]


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def cmd_data(args) -> int:
    tasks = args.tasks or list(SYNP7)
    _refuse_covtype(tasks)
    out = Path(args.out)
    manifest = {"upstream_llmfe_commit": UPSTREAM_COMMIT, "regression_alias": REGRESSION_ALIAS, "files": {}}

    def put(rel: str, data: bytes, **extra) -> None:
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        manifest["files"][rel] = {"sha256": _sha(data), "bytes": len(data), **extra}

    for t in tasks:
        if t not in SYNP7:
            raise SystemExit(f"FATAL: {t} is not a SYN-P 7 task")
        info = load_task(Path(args.tasks_root), t)
        spec_path = Path(args.specs_dir) / t / f"specification_{t}.txt"
        meta_path = Path(args.specs_dir) / t / f"{t}-metadata.json"
        spec = spec_path.read_text(encoding="utf-8").replace("\r\n", "\n")
        # The committed spec must carry exactly this task's context (no drift from config.yml).
        if spec_task_text(spec) != info["context"] or spec != render_spec(info["task_type"], info["context"]):
            raise SystemExit(f"FATAL: {spec_path} does not match {info['dir']}/config.yml; re-run `specs`")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        text, n, cols = build_table(info["dir"])
        if list(meta) != cols:
            raise SystemExit(f"FATAL: {t}: metadata columns {list(meta)} != data.csv columns {cols}")
        data_b = text.encode("utf-8")
        meta_b = meta_path.read_bytes().replace(b"\r\n", b"\n")
        put(f"specs/specification_{t}.txt", spec.encode("utf-8"))
        put(f"data/{t}.csv", data_b, rows=n, task_type=info["task_type"])
        put(f"data/{t}-metadata.json", meta_b)
        if info["task_type"] == "regression":
            put(f"alias/{t}/{REGRESSION_ALIAS}.csv", data_b, rows=n, alias_of=t)
            put(f"alias/{t}/{REGRESSION_ALIAS}-metadata.json", meta_b, alias_of=t)
        print(f"{t}: {n} rows, {len(cols)} features + {TARGET_COL}, {info['task_type']}")
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}/MANIFEST.json ({len(manifest['files'])} files)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("specs", help="render the committed spec + metadata files from config.yml")
    s.add_argument("--tasks-root", required=True, help="tasks/ (with <type>/synp_<id>/config.yml)")
    s.add_argument("--out", default=str(HERE))
    s.add_argument("--tasks", nargs="*")
    d = sub.add_parser("data", help="build LLM-FE data/spec/alias files (run in a container)")
    d.add_argument("--tasks-root", required=True)
    d.add_argument("--specs-dir", default=str(HERE))
    d.add_argument("--out", required=True)
    d.add_argument("--tasks", nargs="*")
    a = ap.parse_args(argv)
    return cmd_specs(a) if a.cmd == "specs" else cmd_data(a)


if __name__ == "__main__":
    sys.exit(main())
