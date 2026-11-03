"""X3 / A08-T2 partition check: classicfe `build_folds` vs Rogue One's outer folds.

The two sides pin different scikit-learn versions (classicfe:4 vs rogue_one:agent2027-*),
so each side is computed inside its own image and the JSONs are diffed afterwards:

  # classicfe side (classicfe:4; mount the classicfe_arm.py under test at /arm)
  docker run --rm --network none -v <P>/tasks:/tasks:ro -v <dir>:/arm:ro -v <x>:/x \
    classicfe:4 python /x/check_fold_partition.py classicfe /tasks/regression/synp_p1 \
    --seed 4 --classicfe-dir /arm > cf.json
  # Rogue One side (rogue_one:agent2027-pre.N; repo at /workspace)
  docker run --rm --network none --entrypoint python3 -v <P>/tasks:/tasks:ro ... \
    rogue_one:agent2027-pre.2 /x/check_fold_partition.py rogueone \
    /tasks/regression/synp_p1 --seed 4 > ro.json
  # compare (either image)
  python check_fold_partition.py compare cf.json ro.json

`classicfe` mode calls `classicfe_arm.prep` + `build_folds(..., seed=S)` with scheme
`stratified` (classicfe falls back to KFold for regression). `rogueone` mode replays
rogue_one_main.py's fold construction verbatim: entities = data.csv `id` in first-seen
order, stratify labels only for classification (ROGUE_ONE_STRATIFIED_FOLDS=True, as
D4 sets for SYN-P classification; pass --no-stratify for the False default),
`compute_entity_duplicate_groups` over the raw table, `deterministic_kfold_indices`.

`llmfe` mode (A08-T4) replays LLM-FE's own generation split, ~/llmfe/main.py at e9a58ca,
verbatim: read the LLM-FE CSV (target = last column), regression iff --llmfe-problem-name is
in main.py's hard-coded list, else label-encode y and StratifiedKFold(5, shuffle, 42); test
positions are mapped to ids through the task's data.csv row order (the X4 builder keeps it).
Run it in llmfe:oss120 (LLM-FE's scikit-learn pin):

  docker run --rm --network none -v <P>/tasks:/tasks:ro -v ~/llmfe/data:/d:ro -v <x>:/x     llmfe:oss120 python /x/check_fold_partition.py llmfe /tasks/classification/synp_p3     --llmfe-csv /d/synp_p3.csv --llmfe-problem-name synp_p3 > lf.json
  (regression: --llmfe-csv <alias dir>/crab.csv --llmfe-problem-name crab)

Output: JSON with per-fold test row positions and test ids, in fold order. `compare`
reports equality both in fold order and as an unordered set of sets; exit 0 iff the
ordered partitions are identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _task_parts(task_dir: str) -> tuple[str, str, str]:
    path = Path(task_dir).resolve()
    return path.name, path.parent.name, str(path.parent.parent)


def _ids(task_dir: str):
    import pandas as pd

    return pd.read_csv(Path(task_dir) / "data.csv", usecols=["id"])["id"].tolist()


def _doc(mode, task_dir, seed, scheme, n_rows, test_idx, ids, extra=None):
    folds = [sorted(int(i) for i in fold) for fold in test_idx]
    doc = {
        "mode": mode,
        "task": _task_parts(task_dir)[0],
        "task_type": _task_parts(task_dir)[1],
        "seed": seed,
        "scheme": scheme,
        "n_rows": int(n_rows),
        "fold_sizes": [len(f) for f in folds],
        "test_idx": folds,
        "test_ids": [sorted(ids[i] for i in f) for f in folds],
    }
    if extra:
        doc.update(extra)
    return doc


def run_classicfe(args) -> dict:
    sys.path.insert(0, args.classicfe_dir)
    import classicfe_arm  # noqa: E402
    import sklearn

    dataset, task, root = _task_parts(args.task_dir)
    X, y = classicfe_arm.prep(dataset, task, root)
    try:
        splits, scheme = classicfe_arm.build_folds(X, y, task, args.folds, seed=args.seed)
    except TypeError:
        if args.seed != 42:
            raise SystemExit("this classicfe_arm.py has no `seed` parameter (pre-X3)")
        splits, scheme = classicfe_arm.build_folds(X, y, task, args.folds)
    return _doc("classicfe", args.task_dir, args.seed, scheme, len(X),
                [te for _, te in splits], _ids(args.task_dir),
                {"sklearn": sklearn.__version__})


def run_rogueone(args) -> dict:
    import contextlib

    sys.path.insert(0, args.repo)
    import pandas as pd
    import sklearn

    # rogueone's config import prints "Loaded environment variables ..." to stdout;
    # keep stdout pure JSON.
    with contextlib.redirect_stdout(sys.stderr):
        from rogueone.utils.train_test_splitter import (
            compute_entity_duplicate_groups,
            deterministic_kfold_indices,
        )

    dataset, task, _ = _task_parts(args.task_dir)
    df = pd.read_csv(Path(args.task_dir) / "data.csv")
    entities = pd.DataFrame(df["id"].unique(), columns=["id"])
    stratify = None
    if task == "classification" and not args.no_stratify:
        labels = pd.read_csv(Path(args.task_dir) / "labels.csv")
        ordered = entities[["id"]].merge(labels[["id", "target"]], on="id", how="left")
        if ordered["target"].isna().any():
            raise SystemExit("some feature-frame ids have no matching label")
        stratify = ordered["target"].to_numpy()
    groups = compute_entity_duplicate_groups(df, entities["id"].tolist(), exclude_cols=("id",))
    n_groups = int(len(set(groups.tolist())))
    with contextlib.redirect_stdout(sys.stderr):
        folds = deterministic_kfold_indices(
            n_rows=len(entities), n_splits=args.k, random_state=args.seed,
            stratify_labels=stratify, dedup_groups=groups,
        )
    if n_groups < len(entities):
        scheme = "stratified_group" if stratify is not None else "group_kfold"
    else:
        scheme = "stratified" if stratify is not None else "kfold"
    return _doc("rogueone", args.task_dir, args.seed, scheme, len(entities),
                [te for _, te in folds], entities["id"].tolist(),
                {"sklearn": sklearn.__version__, "n_dup_groups": n_groups,
                 "duplicates": n_groups < len(entities)})


#: ~/llmfe/main.py:41 at e9a58ca (LLM-FE decides regression by name only).
LLMFE_REGRESSION_NAMES = ("forest-fires", "housing", "insurance", "bike", "wine", "crab")


def run_llmfe(args) -> dict:
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn import preprocessing
    from sklearn.model_selection import KFold, StratifiedKFold

    if not args.llmfe_csv or not args.llmfe_problem_name:
        raise SystemExit("llmfe mode needs --llmfe-csv and --llmfe-problem-name")
    if "covtype" in (args.llmfe_problem_name + args.llmfe_csv).lower():
        raise SystemExit("covtype is never used (hard rule 2026-10-16)")
    # main.py, verbatim in effect: df -> X (target dropped), y; label-encode y unless regression.
    df = pd.read_csv(args.llmfe_csv)
    target_attr = df.columns[-1]
    X = df.convert_dtypes()
    y = df[target_attr].to_numpy()
    X = X.drop(target_attr, axis=1)
    is_regression = args.llmfe_problem_name in LLMFE_REGRESSION_NAMES
    if not is_regression:
        y = preprocessing.LabelEncoder().fit_transform(y)
    skf = (KFold(n_splits=5, shuffle=True, random_state=42) if is_regression
           else StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    test_idx = [te for _, te in skf.split(X, y)]
    ids = _ids(args.task_dir)
    if len(ids) != len(df):
        raise SystemExit(f"LLM-FE CSV has {len(df)} rows, data.csv has {len(ids)}")
    return _doc("llmfe", args.task_dir, 42, "kfold" if is_regression else "stratified", len(df),
                test_idx, ids, {"sklearn": sklearn.__version__, "numpy": np.__version__,
                                "llmfe_problem_name": args.llmfe_problem_name,
                                "llmfe_is_regression": is_regression})


def compare(a_path: str, b_path: str) -> int:
    a = json.loads(Path(a_path).read_text())
    b = json.loads(Path(b_path).read_text())
    ordered = a["test_ids"] == b["test_ids"]
    unordered = (
        {tuple(f) for f in a["test_ids"]} == {tuple(f) for f in b["test_ids"]}
    )
    out = {
        "task": a.get("task"), "seed": a.get("seed"),
        "a": f"{a['mode']}:{a['scheme']} sklearn={a.get('sklearn')}",
        "b": f"{b['mode']}:{b['scheme']} sklearn={b.get('sklearn')}",
        "equal_ordered": ordered, "equal_as_sets": unordered,
    }
    if not ordered:
        out["fold_overlap"] = [
            len(set(x) & set(y)) for x, y in zip(a["test_ids"], b["test_ids"])
        ]
    print(json.dumps(out))
    return 0 if ordered else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=["classicfe", "rogueone", "llmfe", "compare"])
    parser.add_argument("task_dir", help="tasks/<type>/<id> (or the first JSON for compare)")
    parser.add_argument("other", nargs="?", help="second JSON (compare only)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--folds", default="stratified", choices=["kfold", "stratified"],
                        help="classicfe scheme (regression always uses KFold)")
    parser.add_argument("--no-stratify", action="store_true",
                        help="rogueone: ROGUE_ONE_STRATIFIED_FOLDS=False")
    here = Path(__file__).resolve()
    repo_guess = here.parents[2] if len(here.parents) > 2 else here.parent
    parser.add_argument("--classicfe-dir", default=str(repo_guess / "paper" / "scripts"))
    parser.add_argument("--repo", default="/workspace")
    parser.add_argument("--llmfe-csv", help="llmfe: the CSV LLM-FE reads (./data/<problem_name>.csv)")
    parser.add_argument("--llmfe-problem-name", help="llmfe: the --problem_name LLM-FE runs under")
    args = parser.parse_args()
    if args.mode == "compare":
        raise SystemExit(compare(args.task_dir, args.other))
    runner = {"classicfe": run_classicfe, "rogueone": run_rogueone, "llmfe": run_llmfe}[args.mode]
    doc = runner(args)
    print(json.dumps(doc))


if __name__ == "__main__":
    main()
