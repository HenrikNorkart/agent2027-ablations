"""E9c (AGENT 2027): planner hook, ``ROGUE_ONE_PLANNER_MODE`` (A09 RT3 §2.2, §4.2).

Modes
-----
``llm`` (default; unset or empty means ``llm``)
    Full-v2: the Scientist writes the focus (``determine_focus``). Unchanged.
``oracle`` / ``misdirected``
    The Scientist is not called. The focus for (``fold_run_id``, ``iteration_in_fold``)
    is read from ``ROGUE_ONE_PLANNER_FOCUS_FILE`` (mounted at
    ``/workspace/focus/focus.json`` by ``ro27_clients_manifest.sh`` for these rows only).
``none``
    Reserved for E9d (SE|T, FX ``P-``). Recognised so that it can never silently fall
    back to ``llm``, but not implemented in this image: it raises.

Focus file (A09 §4.2, minimal; never holds the rename map)::

    {"dataset_id": "synp_p3", "arm_id": "oracle", "focus": {"0:0": "...", ..., "4:9": "..."}}

Validation happens before any LLM call (``load_focus_plan``): the file's ``dataset_id``
must equal the single entry of ``ROGUE_ONE_DATASETS`` (the task folder name; never the
``experiment_name``, which is opaque for ``synp_p6b``), ``arm_id`` must equal the mode,
and the keys must be exactly ``"<f>:<i>"`` for f < k_folds and i < num_iterations, each
with a non-empty string.

This module is stdlib only, so tests and the CAIR tooling can import it without the
``rogueone`` runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

MODE_ENV = "ROGUE_ONE_PLANNER_MODE"
FILE_ENV = "ROGUE_ONE_PLANNER_FOCUS_FILE"

LLM = "llm"
ORACLE = "oracle"
MISDIRECTED = "misdirected"
NONE = "none"  # E9d

#: Every value the reader accepts (01 §3 E9c / A09-T5). Anything else raises.
KNOWN_MODES: tuple[str, ...] = (LLM, ORACLE, MISDIRECTED, NONE)
#: Modes that read the focus from a file instead of the Scientist.
FILE_MODES: frozenset[str] = frozenset({ORACLE, MISDIRECTED})
#: Modes this image implements (``none`` = E9d P−, handled by ``fx_switches``).
IMPLEMENTED_MODES: frozenset[str] = frozenset({LLM, ORACLE, MISDIRECTED, NONE})


class PlannerFocusError(ValueError):
    """The focus file is missing, malformed, or does not match the run."""


def parse_planner_mode(value: Optional[str]) -> str:
    """``ROGUE_ONE_PLANNER_MODE`` -> canonical mode. Unset/empty -> ``llm``."""
    mode = (value or "").strip().lower() or LLM
    if mode not in KNOWN_MODES:
        raise ValueError(f"{MODE_ENV} must be one of {list(KNOWN_MODES)}, got {value!r}")
    if mode not in IMPLEMENTED_MODES:
        raise NotImplementedError(
            f"{MODE_ENV}={mode!r} is reserved for E9d and not implemented in this image"
        )
    return mode


def focus_key(fold_run_id: int, iteration_in_fold: int) -> str:
    return f"{int(fold_run_id)}:{int(iteration_in_fold)}"


def expected_keys(k_folds: int, num_iterations: int) -> list[str]:
    return [focus_key(f, i) for f in range(int(k_folds)) for i in range(int(num_iterations))]


@dataclass(frozen=True)
class FocusPlan:
    """A validated focus file: one fixed focus string per (fold, iteration)."""

    mode: str
    dataset_id: str
    arm_id: str
    path: str
    focus: Mapping[str, str] = field(repr=False)

    def focus_for(self, fold_run_id: int, iteration_in_fold: int) -> str:
        key = focus_key(fold_run_id, iteration_in_fold)
        try:
            return self.focus[key]
        except KeyError:
            raise PlannerFocusError(f"focus file {self.path} has no key {key!r}") from None


def validate_focus_doc(
    doc: Any,
    *,
    mode: str,
    dataset_id: str,
    k_folds: int,
    num_iterations: int,
    path: str = "<focus file>",
) -> FocusPlan:
    if not isinstance(doc, dict):
        raise PlannerFocusError(f"{path}: top level must be a JSON object")
    missing_top = [k for k in ("dataset_id", "arm_id", "focus") if k not in doc]
    if missing_top:
        raise PlannerFocusError(f"{path}: missing key(s) {missing_top}")
    file_ds = str(doc["dataset_id"]).strip().lower()
    if file_ds != str(dataset_id).strip().lower():
        raise PlannerFocusError(
            f"{path}: dataset_id {doc['dataset_id']!r} does not match the run's dataset {dataset_id!r}"
        )
    if str(doc["arm_id"]).strip().lower() != mode:
        raise PlannerFocusError(f"{path}: arm_id {doc['arm_id']!r} does not match {MODE_ENV}={mode!r}")
    focus = doc["focus"]
    if not isinstance(focus, dict):
        raise PlannerFocusError(f"{path}: 'focus' must be an object of '<fold>:<iteration>' keys")
    want = expected_keys(k_folds, num_iterations)
    missing = [k for k in want if k not in focus]
    extra = sorted(set(focus) - set(want))
    if missing or extra:
        raise PlannerFocusError(
            f"{path}: needs exactly the {len(want)} keys '<f>:<i>' (f < {k_folds}, i < {num_iterations}); "
            f"missing {missing[:5]}{'...' if len(missing) > 5 else ''}, unexpected {extra[:5]}"
        )
    bad = [k for k in want if not isinstance(focus[k], str) or not focus[k].strip()]
    if bad:
        raise PlannerFocusError(f"{path}: empty or non-string focus at {bad[:5]}")
    return FocusPlan(
        mode=mode, dataset_id=file_ds, arm_id=str(doc["arm_id"]), path=path,
        focus={k: focus[k] for k in want},
    )


def run_dataset_id(dataset_filter: Sequence[str]) -> str:
    """The single dataset of a file-mode run (``ROGUE_ONE_DATASETS``)."""
    if len(dataset_filter) != 1:
        raise PlannerFocusError(
            f"{MODE_ENV} oracle/misdirected needs exactly one dataset in ROGUE_ONE_DATASETS, "
            f"got {list(dataset_filter)}"
        )
    return dataset_filter[0]


def load_focus_plan(
    mode: str,
    focus_file: Optional[str],
    *,
    dataset_filter: Sequence[str],
    k_folds: int,
    num_iterations: int,
) -> Optional[FocusPlan]:
    """Return the validated plan for a file mode, ``None`` for ``llm``. Raises on any mismatch."""
    mode = parse_planner_mode(mode)
    if mode not in FILE_MODES:
        return None
    if not focus_file or not str(focus_file).strip():
        raise PlannerFocusError(f"{MODE_ENV}={mode} needs {FILE_ENV}")
    path = Path(str(focus_file).strip())
    if not path.is_file():
        raise PlannerFocusError(f"{FILE_ENV}={path} does not exist")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PlannerFocusError(f"{path}: not readable JSON ({exc})") from exc
    return validate_focus_doc(
        doc, mode=mode, dataset_id=run_dataset_id(dataset_filter),
        k_folds=k_folds, num_iterations=num_iterations, path=str(path),
    )
