#!/usr/bin/env python3
"""E6 seed-equivalence check (AGENT 2027, A03-T6a step 2). CPU only, no LLM.

For each dataset this calls the augment seeding function
(``rogueone.utils.fold_safe_features.seed_raw_feature_rows``, the one
``_reset_fold_state`` calls) on the task's ``data.csv`` and checks:

  (i)  every raw column (except ``id``) is seeded, and every seeded column
       (``<col>`` or ``<col>_ord``) equals the ``base_no_fe`` reference matrix
       (``classicfe_arm.prep()`` then ``to_matrix`` / ``nan_to_num``, written by
       A03-T6a step 1 as ``<ref>/<dataset>_prep.csv``) by ``id``:
       max |diff| <= 1e-6 after float32;
  (ii) every seed command replays with ``materialize_fold_safe_features`` on the
       train rows and on the test rows of outer fold 0 (``deterministic_kfold_indices``,
       seed 42, unstratified, duplicate-grouped as in ``rogue_one_main``) to values
       identical to the authoring frame, with no replay NaN fill.

Usage (inside the Rogue One image)::

    python3 scripts/agent2027/e6_seed_check.py --tasks /workspace/tasks \
        --ref /check --datasets myocardial,credit-g,heart

Prints ``E6_SEED_CHECK PASS`` (exit 0) when every dataset passes, otherwise
``E6_SEED_CHECK FAIL`` (exit 1). ``--emulate-ref`` replaces the reference files
with an in-process emulation of ``prep()`` (for unit tests and local runs); it then
prints ``E6_SEED_CHECK PASS_EMULATED_REF``, never the real PASS line.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# rogueone.utils refuses to import without these; the values are irrelevant here (no LLM).
for _k, _v in {"EMBEDD_API_KEY": "x", "EMBEDD_MODEL": "x", "EMBEDD_PORT": "1", "LLM_MODEL": "x",
               "LLM_API_KEY": "x", "LLM_PORT": "1"}.items():
    os.environ.setdefault(_k, _v)

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rogueone.utils import instrumentation  # noqa: E402
from rogueone.utils.fold_safe_features import (  # noqa: E402
    REPLAY_NAN_FILLS_COUNTER,
    materialize_fold_safe_features,
    seed_raw_feature_rows,
    seeded_attribute_mask,
)
from rogueone.utils.train_test_splitter import (  # noqa: E402
    compute_entity_duplicate_groups,
    deterministic_kfold_indices,
)

ATTR_COLUMNS = [
    "Attribute",
    "Description",
    "Pandas Command",
    "Status",
    "Added",
    "Pruned Iteration",
    "Prune Rationale",
]
TOL = 1e-6


def find_data_csv(tasks_root: Path, dataset: str) -> Path:
    for task_type in ("classification", "regression"):
        p = tasks_root / task_type / dataset / "data.csv"
        if p.is_file():
            return p
    raise FileNotFoundError(f"no data.csv for {dataset!r} under {tasks_root}")


def emulated_reference(df_raw: pd.DataFrame) -> pd.DataFrame:
    """``classicfe_arm.prep()`` + ``to_matrix`` as pandas 2.2.3 computes them.

    pandas 3 reads strings as ``str``, not ``object``, so the emulation detects
    categoricals with ``not is_numeric_dtype`` (what ``dtype == object`` means there).
    """
    X = df_raw.drop(columns=["id"]).copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            categories = sorted(X[col].dropna().astype(str).unique().tolist())
            X[col] = X[col].astype(str).map({v: i for i, v in enumerate(categories)})
    X = X.apply(pd.to_numeric, errors="coerce")
    M = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    out = pd.DataFrame(M, columns=X.columns)
    out["id"] = df_raw["id"].to_numpy()
    return out


def seed_frames(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    entities = pd.DataFrame(df_raw["id"].unique(), columns=["id"])
    return seed_raw_feature_rows(df_raw, entities, pd.DataFrame(columns=ATTR_COLUMNS))


def check_dataset(
    df_raw: pd.DataFrame,
    ref: pd.DataFrame,
    *,
    k: int = 5,
    seed: int = 42,
) -> list[str]:
    """Return a list of failure messages (empty = pass)."""
    failures: list[str] = []
    entities, attrs = seed_frames(df_raw)
    seeded = attrs[seeded_attribute_mask(attrs)]
    raw_cols = [c for c in df_raw.columns if c != "id"]

    # ---- (i) coverage and equality with the base_no_fe matrix
    if len(seeded) != len(raw_cols):
        failures.append(f"(i) seeded {len(seeded)} rows for {len(raw_cols)} raw columns")
    if set(attrs["Added"]) != {"Seed"} or set(attrs["Status"]) != {"Active"}:
        failures.append("(i) seeded rows must be Added='Seed', Status='Active'")
    ref_by_id = ref.set_index("id")
    ent_by_id = entities.set_index("id")
    common = ent_by_id.index.intersection(ref_by_id.index)
    if len(common) != len(ent_by_id):
        failures.append(f"(i) {len(ent_by_id) - len(common)} entity ids missing from the reference")
    for col in raw_cols:
        name = col if col in ent_by_id.columns else f"{col}_ord"
        if name not in ent_by_id.columns:
            failures.append(f"(i) raw column {col!r} not seeded")
            continue
        if col not in ref_by_id.columns:
            failures.append(f"(i) raw column {col!r} missing from the reference")
            continue
        a = ent_by_id.loc[common, name].to_numpy(dtype=np.float32)
        b = ref_by_id.loc[common, col].to_numpy(dtype=np.float32)
        diff = float(np.max(np.abs(a - b))) if len(a) else 0.0
        if not diff <= TOL:
            failures.append(f"(i) {name!r} vs reference {col!r}: max |diff| = {diff:g}")

    # ---- (ii) replay on fold 0's train and test partitions
    ids = entities["id"].tolist()
    groups = compute_entity_duplicate_groups(df_raw, ids, exclude_cols=("id",))
    folds = deterministic_kfold_indices(
        n_rows=len(ids), n_splits=k, random_state=seed, stratify_labels=None, dedup_groups=groups
    )
    test_ids = {ids[i] for i in folds[0][1]}
    is_test = df_raw["id"].isin(test_ids).to_numpy()
    before = instrumentation.get_counter(REPLAY_NAN_FILLS_COUNTER)
    for label, mask in (("train", ~is_test), ("test", is_test)):
        replay = materialize_fold_safe_features(df_raw, seeded, mask).set_index("id")
        for name in seeded["Attribute"]:
            got = replay[name].to_numpy(dtype=float)
            want = ent_by_id.loc[replay.index, name].to_numpy(dtype=float)
            if not np.array_equal(got, want):
                n_bad = int(np.sum(~np.isclose(got, want, equal_nan=True)))
                failures.append(f"(ii) {name!r} replay differs on fold-0 {label} rows ({n_bad})")
    if instrumentation.get_counter(REPLAY_NAN_FILLS_COUNTER) != before:
        failures.append("(ii) replay NaN-filled a seed command")
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", required=True, help="tasks root (holds classification/, regression/)")
    ap.add_argument("--ref", default="", help="folder with <dataset>_prep.csv reference matrices")
    ap.add_argument("--datasets", required=True, help="comma-separated dataset ids")
    ap.add_argument("--emulate-ref", action="store_true", help="emulate prep() instead of --ref")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    if not a.emulate_ref and not a.ref:
        ap.error("--ref is required unless --emulate-ref is given")

    all_ok = True
    for ds in [d.strip() for d in a.datasets.split(",") if d.strip()]:
        df_raw = None
        try:
            df_raw = pd.read_csv(find_data_csv(Path(a.tasks), ds))
            ref = (
                emulated_reference(df_raw)
                if a.emulate_ref
                else pd.read_csv(Path(a.ref) / f"{ds}_prep.csv")
            )
            failures = check_dataset(df_raw, ref, k=a.k, seed=a.seed)
        except Exception as exc:  # noqa: BLE001
            failures = [f"error: {type(exc).__name__}: {exc}"]
        n_raw = None if df_raw is None else len(df_raw.columns) - 1
        if failures:
            all_ok = False
            print(f"[{ds}] FAIL ({n_raw} raw columns)")
            for f in failures[:50]:
                print(f"    {f}")
        else:
            print(f"[{ds}] ok: {n_raw} raw columns seeded; (i) matches prep(); (ii) fold-0 replay identical")

    if all_ok:
        print("E6_SEED_CHECK PASS_EMULATED_REF" if a.emulate_ref else "E6_SEED_CHECK PASS")
        return 0
    print("E6_SEED_CHECK FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
