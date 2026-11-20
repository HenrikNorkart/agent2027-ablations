"""SYN-P generator CLI (E8a): writes the 9 task folders and the off-mount metadata.

    python scripts/agent2027/synp/generate_synp.py \
        --tasks-root <dir>/tasks --meta-dir ~/rogueone-synp-meta

Task folders (``<tasks-root>/<task_type>/synp_<id>/``) contain exactly
``config.yml``, ``data.csv`` and ``labels.csv`` -- the existing task format
(``ExperimentConfig.from_yaml``; ``labels.csv`` = ``id,target``).  Everything else
(phi, constants, seeds, phi_wrong, oracle focus strings, detector definitions,
name/level maps, hashes) goes to ``<meta-dir>/<task_id>.json`` plus
``registry.yaml`` and ``task_registry.yml``.  The meta dir must **not** be inside the
tasks root or inside a git checkout (the repo is baked into agent images).

P7's unguessable constant is derived from ``--p7-secret`` (or env
``SYNP_P7_SECRET``) and then *read back from* ``<meta-dir>/synp_p7.json`` on every
later run; it is never hard-coded in the repository.

Registration of the folders in the E1 task registry is E1's job
(``ROGUE_ONE_TASK_REGISTRY``: a YAML list of ``config.yml`` paths); this script
only emits that list as ``<meta-dir>/task_registry.yml`` (``--registry-prefix`` sets
the path prefix, e.g. ``/workspace/tasks``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

if __package__ in (None, ""):  # allow `python scripts/agent2027/synp/generate_synp.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.agent2027.synp import p6_twin  # noqa: E402
from scripts.agent2027.synp import tasks as T  # noqa: E402

EXPERIMENT_DESCRIPTION = "Synthetic tabular benchmark task."
KNOWLEDGE_DB_COLLECTION = "default_collection"
TASK_FILES = ("config.yml", "data.csv", "labels.csv")


def _inside(child: Path, parent: Path) -> bool:
    child, parent = child.resolve(), parent.resolve()
    return child == parent or parent in child.parents


def _inside_git_checkout(p: Path) -> bool:
    p = p.resolve()
    return any((q / ".git").exists() for q in (p, *p.parents))


def check_meta_dir(meta_dir: Path, tasks_root: Path, allow_in_repo: bool = False) -> None:
    if _inside(meta_dir, tasks_root) or _inside(tasks_root, meta_dir):
        raise ValueError("the metadata dir and the tasks root must be disjoint (metadata must stay off the mount)")
    if not allow_in_repo and _inside_git_checkout(meta_dir):
        raise ValueError(f"{meta_dir} is inside a git checkout; metadata must live outside the repository")


def config_dict(g: T.GeneratedTask, path_style: str = "relative") -> dict[str, Any]:
    prefix = {"relative": "", "workspace": f"/workspace/tasks/{g.task_type}/{g.task_id}/"}[path_style]
    return {
        "experiment_name": g.task_id,
        "experiment_description": EXPERIMENT_DESCRIPTION,
        "knowledge_db_path": "./knowledge_db",
        "knowledge_db_collection_name": KNOWLEDGE_DB_COLLECTION,
        "feature_file_path": f"{prefix}data.csv",
        "label_file_path": f"{prefix}labels.csv",
        "context_and_goal": g.context,
        "modality": "tabular",
        "task_type": g.task_type,
    }


def write_task_folder(g: T.GeneratedTask, tasks_root: Path, path_style: str = "relative") -> Path:
    d = tasks_root / g.task_type / g.task_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "data.csv").write_bytes(T.data_csv_bytes(g.data))
    (d / "labels.csv").write_bytes(g.labels.to_csv(None, index=False, lineterminator="\n").encode())
    (d / "config.yml").write_text(
        yaml.safe_dump(config_dict(g, path_style), sort_keys=False, allow_unicode=True, width=10_000),
        encoding="utf-8",
    )
    return d


def write_meta(meta_dir: Path, g: T.GeneratedTask) -> None:
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / f"{g.task_id}.json").write_text(json.dumps(g.meta, indent=2, default=float), encoding="utf-8")


def load_p7_constants(meta_dir: Path) -> dict[str, Any] | None:
    f = meta_dir / "synp_p7.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8")).get("p7_constants")
    return None


def build_all(
    tasks_root: Path, meta_dir: Path, *, master_seed: int = T.MASTER_SEED, n: int = T.N_ROWS,
    p7_secret: str = "", p6a_candidate: str = T.DEFAULT_P6A, path_style: str = "relative",
    registry_prefix: str | None = None, params: dict[str, dict[str, float]] | None = None,
    allow_meta_in_repo: bool = False, keys: tuple[str, ...] = T.TASK_KEYS,
) -> dict[str, T.GeneratedTask]:
    tasks_root, meta_dir = Path(tasks_root), Path(meta_dir)
    check_meta_dir(meta_dir, tasks_root, allow_meta_in_repo)
    p7_secret = p7_secret or os.environ.get("SYNP_P7_SECRET", "")
    p7_const = load_p7_constants(meta_dir)
    out: dict[str, T.GeneratedTask] = {}
    for key in T.GENERATED_KEYS:
        if key not in keys and not (key == "p6a" and "p6b" in keys):
            continue
        out[key] = T.generate_task(
            key, master_seed=master_seed, n=n, params=(params or {}).get(key),
            p7_secret=p7_secret, p7_constants=p7_const if key == "p7" else None,
            p6a_candidate=p6a_candidate,
        )
    if "p6b" in keys:
        out["p6b"] = p6_twin.make_twin(out["p6a"], master_seed=master_seed)
    for key in list(out):
        if key not in keys:
            out_key = out.pop(key)  # p6a built only as p6b's source
            del out_key
    registry: dict[str, Any] = {}
    cfg_paths: list[str] = []
    for key, g in out.items():
        write_task_folder(g, tasks_root, path_style)
        write_meta(meta_dir, g)
        registry[g.task_id] = {"task_type": g.task_type, "family": g.meta["family"]}
        rel = f"{g.task_type}/{g.task_id}/config.yml"
        cfg_paths.append(f"{registry_prefix.rstrip('/')}/{rel}" if registry_prefix else str((tasks_root / rel).resolve()))
    (meta_dir / "registry.yaml").write_text(yaml.safe_dump(registry, sort_keys=True), encoding="utf-8")
    (meta_dir / "task_registry.yml").write_text(yaml.safe_dump(cfg_paths, sort_keys=False), encoding="utf-8")
    return out


def main(argv=None) -> int:  # pragma: no cover - thin CLI
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks-root", required=True, type=Path)
    ap.add_argument("--meta-dir", required=True, type=Path)
    ap.add_argument("--master-seed", type=int, default=T.MASTER_SEED)
    ap.add_argument("--n", type=int, default=T.N_ROWS)
    ap.add_argument("--p7-secret", default="")
    ap.add_argument("--p6a-candidate", default=T.DEFAULT_P6A, choices=sorted(T.P6A_CANDIDATES))
    ap.add_argument("--path-style", default="relative", choices=["relative", "workspace"])
    ap.add_argument("--registry-prefix", default=None)
    ap.add_argument("--params-json", default=None, help="JSON {key: {param: value}} signal overrides")
    ap.add_argument("--keys", nargs="*", default=list(T.TASK_KEYS))
    a = ap.parse_args(argv)
    params = json.loads(Path(a.params_json).read_text()) if a.params_json else None
    out = build_all(a.tasks_root, a.meta_dir, master_seed=a.master_seed, n=a.n, p7_secret=a.p7_secret,
                    p6a_candidate=a.p6a_candidate, path_style=a.path_style, registry_prefix=a.registry_prefix,
                    params=params, keys=tuple(a.keys))
    for key, g in out.items():
        print(f"{g.task_id}: {g.task_type} {g.data.shape} data_sha256={g.meta['data_sha256'][:12]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
