"""Task registration (AGENT 2027 E1b).

``ROGUE_ONE_TASK_REGISTRY`` -- a path to a YAML list of ``config.yml`` paths --
replaces the hard-coded legacy list when it is set. Unset, the legacy config list
below is used (same order as ``rogue_one_main.py`` had inline, minus covtype). Either
way :func:`select_paths` still filters by ``ROGUE_ONE_DATASETS``.

HARD RULE (Henrik, 2026-10-16): covtype is never run, in any study. It is not in the
legacy list, :func:`task_paths` drops it from any registry, and asking for it by name
raises :class:`BannedDatasetError`.

``ROGUE_ONE_SMOKE_FOLDS=n`` runs only the first ``n`` outer folds (the partition
itself stays the ``ROGUE_ONE_K_FOLDS`` split), so ``total_trials`` becomes
``n * num_iterations`` and fold aggregation expects ``n`` folds. Smoke and Stage-0
runs only; never used in analysis.

Standard library + PyYAML only; nothing from ``rogueone`` is imported, so this
module is cheap to unit test.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

REGISTRY_ENV = "ROGUE_ONE_TASK_REGISTRY"
SMOKE_FOLDS_ENV = "ROGUE_ONE_SMOKE_FOLDS"

#: Datasets that must never run (Henrik, 2026-10-16: "we should never ever run COVTYPE").
BANNED_DATASETS: frozenset[str] = frozenset({"covtype"})


class BannedDatasetError(ValueError):
    """A banned dataset (covtype) was requested."""


def is_banned(path_or_name: Path | str) -> bool:
    """True if a config path (by its parent directory) or a dataset name is banned."""
    p = Path(str(path_or_name))
    names = {p.name.lower(), p.parent.name.lower()} if p.suffix else {p.name.lower()}
    return bool(names & BANNED_DATASETS)


#: The legacy list, moved from ``rogue_one_main.py``'s ``__main__`` block; covtype removed.
LEGACY_TASK_PATHS: list[Path] = [
    Path("/workspace/tasks/classification/balance-scale/config.yml"),
    Path("/workspace/tasks/classification/pc1/config.yml"),
    Path("/workspace/tasks/classification/tic-tac-toe/config.yml"),
    Path("/workspace/tasks/classification/junglechess/config.yml"),
    Path("/workspace/tasks/classification/communities/config.yml"),
    Path("/workspace/tasks/classification/eucalyptus/config.yml"),
    Path("/workspace/tasks/classification/blood/config.yml"),
    Path("/workspace/tasks/classification/car/config.yml"),
    Path("/workspace/tasks/classification/arrhythmia/config.yml"),
    Path("/workspace/tasks/classification/bank/config.yml"),
    Path("/workspace/tasks/classification/breast-w/config.yml"),
    Path("/workspace/tasks/classification/diabetes/config.yml"),
    Path("/workspace/tasks/classification/cmc/config.yml"),
    Path("/workspace/tasks/classification/adult/config.yml"),
    Path("/workspace/tasks/classification/heart/config.yml"),
    Path("/workspace/tasks/classification/vehicle/config.yml"),
    Path("/workspace/tasks/classification/credit-g/config.yml"),
    Path("/workspace/tasks/classification/myocardial/config.yml"),
    #
    Path("/workspace/tasks/regression/forest-fires/config.yml"),
    Path("/workspace/tasks/regression/wine/config.yml"),
    Path("/workspace/tasks/regression/plasma_retinol/config.yml"),
    Path("/workspace/tasks/regression/housing/config.yml"),
    Path("/workspace/tasks/regression/insurance/config.yml"),
    Path("/workspace/tasks/regression/crab/config.yml"),
    Path("/workspace/tasks/regression/bike/config.yml"),
    Path("/workspace/tasks/regression/airfoil_self_noice/config.yml"),
    Path("/workspace/tasks/regression/diamonds/config.yml"),
]


def load_task_registry(registry_path: Path | str) -> list[Path]:
    """Read a YAML list of ``config.yml`` paths.

    Accepted shapes: a plain list, or a mapping with a ``tasks`` list. Relative
    entries are resolved against the registry file's directory. Raises
    ``ValueError`` on anything else, and ``FileNotFoundError`` when the registry
    file is missing -- a set-but-broken registry must fail loudly, never fall back
    to the legacy list.
    """
    import yaml

    registry_path = Path(registry_path)
    if not registry_path.is_file():
        raise FileNotFoundError(f"{REGISTRY_ENV}={registry_path} does not exist")
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("tasks")
    if not isinstance(data, list) or not all(isinstance(x, str) and x.strip() for x in data):
        raise ValueError(
            f"{REGISTRY_ENV}={registry_path} must hold a YAML list of config.yml paths"
        )
    out: list[Path] = []
    for entry in data:
        p = Path(entry.strip())
        if not p.is_absolute() and not entry.strip().startswith("/"):
            p = registry_path.parent / p
        out.append(p)
    return out


def task_paths(environ: Optional[Mapping[str, str]] = None) -> list[Path]:
    """The candidate config paths: the registry when set, else the legacy list."""
    env = os.environ if environ is None else environ
    reg = str(env.get(REGISTRY_ENV, "") or "").strip()
    paths = list(LEGACY_TASK_PATHS) if not reg else load_task_registry(reg)
    return [p for p in paths if not is_banned(p)]


def select_paths(
    all_paths: Iterable[Path],
    wanted: list[str],
    on_missing: Optional[Callable[[str], None]] = None,
) -> list[Path]:
    """Restrict the sweep to ``wanted`` dataset names (``ROGUE_ONE_DATASETS``),
    matching on the config's parent dir. Empty ``wanted`` keeps every path."""
    banned = sorted({w.lower() for w in wanted} & BANNED_DATASETS)
    if banned:
        raise BannedDatasetError(
            f"ROGUE_ONE_DATASETS asks for {banned}: covtype is never run (hard rule, 2026-10-16)"
        )
    all_paths = [p for p in all_paths if not is_banned(p)]
    if not wanted:
        return all_paths
    chosen = [p for p in all_paths if p.parent.name.lower() in wanted]
    matched = {p.parent.name.lower() for p in chosen}
    for name in wanted:
        if name not in matched and on_missing is not None:
            on_missing(name)
    return chosen


def parse_smoke_folds(value: Optional[str]) -> Optional[int]:
    """``ROGUE_ONE_SMOKE_FOLDS`` -> int >= 1, or None when unset/empty."""
    if value is None or str(value).strip() == "":
        return None
    n = int(str(value).strip())
    if n < 1:
        raise ValueError(f"{SMOKE_FOLDS_ENV} must be >= 1, got {n}")
    return n


def evaluated_fold_count(k_folds: int, smoke_folds: Optional[int]) -> int:
    """How many outer folds a run evaluates: ``min(smoke_folds, k_folds)`` or all."""
    if smoke_folds is None:
        return int(k_folds)
    return max(1, min(int(smoke_folds), int(k_folds)))


def total_trials(k_folds: int, num_iterations: int, smoke_folds: Optional[int]) -> int:
    """``k_folds * num_iterations``, truncated to ``n * num_iterations`` under smoke."""
    return evaluated_fold_count(k_folds, smoke_folds) * int(num_iterations)
