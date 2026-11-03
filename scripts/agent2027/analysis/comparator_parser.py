"""X6 · comparator-output parser (AGENT 2027; 01 section 3.3 X6).

Turns the LLM-generated programs of the comparator systems into per-fold
*training-row* feature tables that ``recovery_detector.py --features-table``
scores::

    <out>/f42/<dataset_id>_<arm>_fold<k>.csv.gz      id + one column per generated feature
    <out>/manifest.json                              provenance, checks, per-program outcomes
    <out>/programs.csv                               one row per (run, split, program)

The arm name must not contain ``_`` (the detector's file-name regex is
``<dataset>_<arm>_fold<k>``, arm = ``[^_]+``), so arms are hyphenated:
``llmfe-r0-notebook``, ``llmfe-r0-best3``, ``llmfe-r0-any``, ``caafe``.
Fold ``k`` is 0-based (LLM-FE ``split_<i>`` = fold ``i-1``).

LLM-FE (subcommand ``llmfe``)
-----------------------------
Each ``<logs>/<task>_<suffix>_split_<i>/samples/samples_<n>.json`` holds one program
(``function`` = source of ``modify_features(df_input)``) and its search ``score``.

* **Input frame.** The program is executed on exactly the frame LLM-FE's ``main.py``
  handed the search for that split: the LLM-FE data CSV (``<llmfe-data>/<task>.csv``, or
  the regression alias ``<alias-root>/<task>/crab.csv``), ``convert_dtypes()``, target
  (last column) dropped, ``string`` columns label-encoded, ``fillna(0)``, rows
  ``X.iloc[train_idx]`` of ``KFold(5, shuffle, 42)`` (regression) /
  ``StratifiedKFold(5, shuffle, 42)`` on the label-encoded target (classification).
  The LLM-FE CSV is ``data.csv`` in row order with ``id`` dropped (``make_llmfe_inputs.py``),
  so row ``r`` of the frame is ``data.csv`` row ``r``; this is checked value by value
  before anything is executed, and the split's test indices are checked against
  ``<partition-dir>/llmfe_<task>.json``.
* **Features.** Only columns that are *new* (name not among the input columns) count;
  input columns modified in place or dropped are ignored (counted in ``programs.csv``).
  Non-numeric new columns are label-encoded on the training rows (sorted levels, as
  ``LabelEncoder`` does in LLM-FE's scorer); bool -> float.  Columns are written as
  ``s<sample_order>__<name>``.
* **Selections** (``paper/scripts/llmfe_score_rogueone.py``):
  ``notebook`` -- evaluation.ipynb verbatim, *including* its defects: positions in
  ``os.listdir`` order (with the ``idx == 1`` sentinel inserted) are used as file
  numbers, ``None`` scores become the sentinel, top 3 by ``heapq.nlargest``, position 1
  skipped; a selected file number with no file is recorded as a failure.
  ``best3`` -- the genuine top-3 by score, ``None`` skipped.
  ``any`` -- every program with a non-``None`` score.
  A selection's table is the union of the new columns of its programs that executed.
* **Execution.** Every program of a split runs once, in a forked child process with a
  wall-clock timeout; any exception, a non-DataFrame return, a row-count change, or a
  timeout yields no features for that program (recorded).  Run this only inside a
  container (``--network none``).

CAAFE (subcommand ``caafe``)
----------------------------
``arm_<dataset>_caafe_<scheme>.json`` (``paper/scripts/caafe_arm.py``): fold ``k``'s
``record['generated_code'][k]`` is executed CAAFE-style (``exec(code, {}, {'df': df,
'pd': pd, 'np': np})``) on ``classicfe_arm.prep`` + ``build_folds`` training rows
(re-implemented here, identical), new columns kept as above.  A08 has no CAAFE rows
(cut CL4), so this path has no A08 input.

HARD RULE: covtype is never parsed, executed or scored.
"""

from __future__ import annotations

import argparse
import heapq
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SYNP7 = ("synp_p1", "synp_p2", "synp_p3", "synp_p4", "synp_p5", "synp_p6a", "synp_p6b")
SELECTIONS = ("notebook", "best3", "any")
REGRESSION_ALIAS = "crab"
SEED = 42
N_SPLITS = 5
_SENTINEL = -1000000.0  # evaluation.ipynb's max_score


def _refuse_covtype(*names) -> None:
    for n in names:
        if "covtype" in str(n).lower():
            raise SystemExit("FATAL: covtype is never used in any Rogue One study (Henrik, 2026-10-16)")


# --------------------------------------------------------------------------- task data
def find_task_dir(tasks_root: Path, task: str) -> Path:
    hits = sorted(tasks_root.glob(f"*/{task}/data.csv"))
    if len(hits) != 1:
        raise SystemExit(f"FATAL: expected one tasks/<type>/{task}/data.csv, found {hits}")
    return hits[0].parent


def load_synp(tasks_root: Path, task: str) -> tuple[pd.DataFrame, pd.Series, str]:
    d = find_task_dir(tasks_root, task)
    data = pd.read_csv(d / "data.csv")
    labels = pd.read_csv(d / "labels.csv")
    y = data[["id"]].merge(labels[["id", "target"]], on="id", how="left")["target"]
    if y.isna().any():
        raise SystemExit(f"FATAL: {task}: labels missing for some ids")
    return data, y, d.parent.name


def _same_values(a: pd.Series, b: pd.Series) -> bool:
    if len(a) != len(b):
        return False
    na, nb = a.isna().to_numpy(), b.isna().to_numpy()
    if not np.array_equal(na, nb):
        return False
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        return bool(np.array_equal(a.to_numpy(dtype=float, na_value=np.nan)[~na],
                                   b.to_numpy(dtype=float, na_value=np.nan)[~nb]))
    return bool((a[~na].astype(str).to_numpy() == b[~nb].astype(str).to_numpy()).all())


def verify_alias(llmfe_df: pd.DataFrame, data: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """The LLM-FE table must be data.csv (id dropped) row by row, with target last."""
    feat_cols = [c for c in data.columns if c != "id"]
    out = {"rows_llmfe": len(llmfe_df), "rows_synp": len(data),
           "columns_equal": list(llmfe_df.columns[:-1]) == feat_cols,
           "target_name": str(llmfe_df.columns[-1])}
    bad = [c for c in feat_cols if c not in llmfe_df.columns or not _same_values(llmfe_df[c], data[c])]
    out["columns_value_mismatch"] = bad
    out["target_equal"] = _same_values(llmfe_df.iloc[:, -1].reset_index(drop=True), y.reset_index(drop=True))
    out["ok"] = bool(out["rows_llmfe"] == out["rows_synp"] and out["columns_equal"] and not bad
                     and out["target_equal"])
    return out


def llmfe_frame(df: pd.DataFrame, regression: bool) -> tuple[pd.DataFrame, np.ndarray]:
    """LLM-FE main.py's preprocessing, verbatim in effect."""
    from sklearn import preprocessing

    label_encoder = preprocessing.LabelEncoder()
    target_attr = df.columns[-1]
    X = df.convert_dtypes()
    y = df[target_attr].to_numpy()
    X = X.drop(target_attr, axis=1)
    for col in X.columns:
        if X[col].dtype == "string":
            X[col] = label_encoder.fit_transform(X[col])
    X = X.fillna(0)
    if not regression:
        y = label_encoder.fit_transform(y)
    return X, y


def llmfe_splits(X: pd.DataFrame, y: np.ndarray, regression: bool):
    from sklearn.model_selection import KFold, StratifiedKFold

    skf = (KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED) if regression
           else StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED))
    return list(skf.split(X, y))


# --------------------------------------------------------------------------- execution
def _child(kind: str, source: str, frame_path: str, out_path: str) -> None:
    try:
        import warnings

        warnings.filterwarnings("ignore")
        df = pd.read_pickle(frame_path)
        if kind == "llmfe":
            ns = {"pd": pd, "np": np}
            exec(source, ns)  # noqa: S102 - LLM-generated program, container-only
            res = ns["modify_features"](df.copy())
        else:  # caafe: run_llm_code semantics
            loc = {"df": df.copy(), "pd": pd, "np": np}
            exec(source, {}, loc)  # noqa: S102
            res = loc["df"]
        if not isinstance(res, pd.DataFrame):
            raise TypeError(f"returned {type(res).__name__}, not a DataFrame")
        res.to_pickle(out_path)
    except BaseException as exc:  # noqa: BLE001
        with open(out_path + ".err", "w", encoding="utf-8") as fh:
            fh.write(f"{type(exc).__name__}: {str(exc)[:300]}")
        os._exit(1)
    os._exit(0)


def run_program(kind: str, source: str, frame: pd.DataFrame, tmp: Path, tag: str,
                timeout: float) -> tuple[pd.DataFrame | None, str]:
    """Execute one program in a forked child; returns (output frame | None, status)."""
    fp, op = tmp / f"{tag}.in.pkl", tmp / f"{tag}.out.pkl"
    frame.to_pickle(fp)
    ctx = mp.get_context("fork")
    p = ctx.Process(target=_child, args=(kind, source, str(fp), str(op)))
    t0 = time.time()
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.kill()
        p.join()
        return None, f"timeout>{timeout:g}s"
    err = Path(str(op) + ".err")
    if err.exists():
        return None, "error: " + err.read_text(encoding="utf-8")
    if not op.exists():
        return None, f"error: child exit {p.exitcode} without output"
    res = pd.read_pickle(op)
    for f in (fp, op):
        f.unlink(missing_ok=True)
    return res, f"ok ({time.time() - t0:.1f}s)"


def encode_new_columns(out: pd.DataFrame, inp: pd.DataFrame) -> tuple[pd.DataFrame | None, dict]:
    """New columns of a program's output, row-aligned with ``inp``, all numeric."""
    info = {"n_out_cols": int(out.shape[1]), "n_dropped_input": 0, "n_new": 0, "n_encoded": 0,
            "n_dup_names": 0, "n_modified_input": 0}
    if len(out) != len(inp):
        return None, {**info, "fail": f"row count {len(out)} != {len(inp)}"}
    if out.index.equals(inp.index):
        aligned = out
    elif set(out.index) == set(inp.index) and out.index.is_unique:
        aligned = out.loc[inp.index]
    else:  # positional (e.g. reset_index inside the program)
        aligned = out.set_axis(inp.index, axis=0)
    cols = [str(c) for c in aligned.columns]
    info["n_dup_names"] = len(cols) - len(set(cols))
    aligned = aligned.loc[:, ~pd.Index(cols).duplicated()]
    aligned.columns = [str(c) for c in aligned.columns]
    in_cols = set(map(str, inp.columns))
    info["n_dropped_input"] = len(in_cols - set(aligned.columns))
    for c in in_cols & set(aligned.columns):
        try:
            if not _same_values(aligned[c].reset_index(drop=True), inp[c].reset_index(drop=True)):
                info["n_modified_input"] += 1
        except Exception:  # noqa: BLE001
            info["n_modified_input"] += 1
    new = [c for c in aligned.columns if c not in in_cols]
    feats = {}
    for c in new:
        s = aligned[c]
        try:
            if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
                v = pd.Series(s.to_numpy(dtype=float, na_value=np.nan), index=s.index)
            else:
                raise TypeError
        except Exception:  # noqa: BLE001 - categorical/string/object -> sorted-level codes
            st = s.astype(object)
            mask = pd.isna(st)
            levels = sorted(set(st[~mask].astype(str)))
            codes = {lv: float(i) for i, lv in enumerate(levels)}
            v = st.map(lambda x: np.nan if pd.isna(x) else codes[str(x)]).astype(float)
            info["n_encoded"] += 1
        feats[c] = v.to_numpy()
    info["n_new"] = len(feats)
    return pd.DataFrame(feats, index=inp.index), info


# --------------------------------------------------------------------------- LLM-FE selections
def notebook_selection(listing: list[str], scores: dict[str, Any]) -> tuple[list[int], list[int]]:
    """evaluation.ipynb / llmfe_score_rogueone.py ``notebook``: returns (best_k, file numbers
    actually opened).  ``listing`` is os.listdir order; ``scores`` maps file name -> score."""
    max_score = _SENTINEL
    data = []
    for idx, filename in enumerate(listing):
        if filename.endswith(".json"):
            if idx == 1:
                data.append(max_score)
            data.append(scores[filename])
    data = [max_score if v is None else v for v in data]
    best_k = heapq.nlargest(3, range(len(data)), key=data.__getitem__)
    return best_k, [n for n in best_k if n != 1]


def best3_selection(scores_by_n: dict[int, Any]) -> list[int]:
    scored = [(sc, n) for n, sc in scores_by_n.items() if sc is not None]
    return [n for _, n in sorted(scored, key=lambda t: -t[0])[:3]]


def any_selection(scores_by_n: dict[int, Any]) -> list[int]:
    return sorted(n for n, sc in scores_by_n.items() if sc is not None)


def write_table(path: Path, ids: np.ndarray, parts: list[pd.DataFrame]) -> int:
    tab = pd.concat([pd.DataFrame({"id": ids})] + [p.reset_index(drop=True) for p in parts], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(path, index=False, compression="gzip")
    return tab.shape[1] - 1


def cmd_llmfe(a) -> int:
    tasks = a.tasks or list(SYNP7)
    _refuse_covtype(*tasks)
    out = Path(a.out)
    fdir = out / f"f{SEED}"
    tmp = out / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"subcommand": "llmfe", "argv": sys.argv, "runs": {}}
    prog_rows = []
    for task in tasks:
        data, y, task_type = load_synp(Path(a.tasks_root), task)
        regression = task_type == "regression"
        src = (Path(a.alias_root) / task / f"{REGRESSION_ALIAS}.csv" if regression
               else Path(a.llmfe_data) / f"{task}.csv")
        llmfe_df = pd.read_csv(src)
        check = verify_alias(llmfe_df, data, y)
        if not check["ok"]:
            raise SystemExit(f"FATAL: {task}: {src} is not data.csv row by row: {check}")
        X, yy = llmfe_frame(llmfe_df, regression)
        splits = llmfe_splits(X, yy, regression)
        part = json.loads((Path(a.partition_dir) / f"llmfe_{task}.json").read_text())
        ro_part_path = Path(a.partition_dir) / f"rogueone_{task}.json"
        ro_part = json.loads(ro_part_path.read_text()) if ro_part_path.exists() else None
        split_ok = all(sorted(te.tolist()) == sorted(part["test_idx"][k]) for k, (_, te) in enumerate(splits))
        if not split_ok:
            raise SystemExit(f"FATAL: {task}: recomputed LLM-FE split != {a.partition_dir}/llmfe_{task}.json")
        same_as_ro = (ro_part is not None and all(
            sorted(part["test_idx"][k]) == sorted(ro_part["test_idx"][k]) for k in range(N_SPLITS)))
        ids = data["id"].to_numpy()
        for rep in a.reps:
            prefix = a.suffix_template.format(task=task, rep=rep)
            run_key = f"{task}:r{rep}"
            rinfo: dict[str, Any] = {"source_csv": str(src), "task_type": task_type, "alias_check": check,
                                     "n_nan_input": int(llmfe_df.iloc[:, :-1].isna().sum().sum()),
                                     "split_matches_partition": split_ok,
                                     "llmfe_partition_equals_rogueone": same_as_ro,
                                     "log_prefix": prefix, "splits": {}}
            for i, (tr, _te) in enumerate(splits, start=1):
                k = i - 1
                sdir = Path(a.llmfe_logs) / f"{prefix}_split_{i}" / "samples"
                sinfo: dict[str, Any] = {"dir": str(sdir)}
                rinfo["splits"][str(i)] = sinfo
                if not sdir.is_dir():
                    sinfo["missing"] = True
                    continue
                listing = os.listdir(sdir)
                sinfo["listdir_order"] = listing
                frame = X.iloc[tr]
                scores, srcs, orders = {}, {}, {}
                for fn in listing:
                    if fn.endswith(".json"):
                        rec = json.loads((sdir / fn).read_text(encoding="utf-8"))
                        n = int(fn.split("_")[1].split(".")[0])
                        scores[fn], srcs[n], orders[n] = rec["score"], rec["function"], rec.get("sample_order")
                scores_by_n = {int(fn.split("_")[1].split(".")[0]): sc for fn, sc in scores.items()}
                best_k, nb_open = notebook_selection(listing, scores)
                sel = {"notebook": nb_open, "best3": best3_selection(scores_by_n),
                       "any": any_selection(scores_by_n)}
                sinfo.update(n_programs=len(srcs), n_scored=sum(v is not None for v in scores_by_n.values()),
                             notebook_best_k=best_k, selected=sel)
                feats: dict[int, pd.DataFrame | None] = {}
                need = sorted(set().union(*sel.values()))
                for n in sorted(srcs):
                    row = {"dataset_id": task, "replicate": rep, "split": i, "fold": k, "sample": n,
                           "sample_order": orders.get(n), "score": scores_by_n[n],
                           "in_notebook": n in sel["notebook"], "in_best3": n in sel["best3"],
                           "in_any": n in sel["any"]}
                    if n not in need and not a.execute_all:
                        row["status"] = "not_executed"
                        prog_rows.append(row)
                        continue
                    res, status = run_program("llmfe", srcs[n], frame, tmp, f"{task}_r{rep}_s{i}_n{n}", a.timeout)
                    if res is not None:
                        enc, info = encode_new_columns(res, frame)
                        row.update(info)
                        if enc is None:
                            status = "error: " + info["fail"]
                        else:
                            feats[n] = enc.add_prefix(f"s{n}__")
                    row["status"] = status
                    prog_rows.append(row)
                for n in need:
                    if n not in srcs:  # notebook defect: a position with no such file
                        prog_rows.append({"dataset_id": task, "replicate": rep, "split": i, "fold": k,
                                          "sample": n, "status": "error: no such file samples_%d.json" % n,
                                          "in_notebook": True})
                sinfo["tables"] = {}
                for s in a.selections:
                    parts = [feats[n] for n in sel[s] if n in feats]
                    arm = f"llmfe-r{rep}-{s}"
                    path = fdir / f"{task}_{arm}_fold{k}.csv.gz"
                    ncol = write_table(path, ids[tr], parts)
                    sinfo["tables"][s] = {"path": str(path), "n_programs_selected": len(sel[s]),
                                          "n_programs_with_features": len(parts), "n_feature_cols": ncol}
                print(f"[x6] {task} r{rep} split {i}: {len(srcs)} programs, notebook={sel['notebook']} "
                      f"best3={sel['best3']} any={len(sel['any'])} -> "
                      + ", ".join(f"{s}:{sinfo['tables'][s]['n_feature_cols']}" for s in a.selections),
                      flush=True)
            manifest["runs"][run_key] = rinfo
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prog_rows).to_csv(out / "programs.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    try:
        tmp.rmdir()
    except OSError:
        pass
    return 0


# --------------------------------------------------------------------------- CAAFE
def caafe_prep(data: pd.DataFrame) -> pd.DataFrame:
    """classicfe_arm.prep's feature frame (id dropped, object columns -> sorted codes)."""
    X = data.drop(columns=["id"])
    for col in X.columns:
        if X[col].dtype == object:
            cats = sorted(X[col].dropna().astype(str).unique().tolist())
            X[col] = X[col].astype(str).map({v: i for i, v in enumerate(cats)})
    return X.apply(pd.to_numeric, errors="coerce")


def caafe_folds(n_rows: int, y: np.ndarray, task: str, scheme: str):
    """classicfe_arm.build_folds, seed 42."""
    from sklearn.model_selection import KFold, StratifiedKFold

    if task == "classification" and scheme == "stratified":
        _, counts = np.unique(y, return_counts=True)
        if counts.min() >= 5:
            return list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros((n_rows, 1)), y))
    return list(KFold(5, shuffle=True, random_state=SEED).split(range(n_rows)))


def cmd_caafe(a) -> int:
    out = Path(a.out)
    fdir = out / f"f{SEED}"
    tmp = out / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"subcommand": "caafe", "argv": sys.argv, "cells": {}}
    prog_rows = []
    for jp in a.arm_json:
        rec = json.loads(Path(jp).read_text(encoding="utf-8"))
        ds = rec["dataset"]
        _refuse_covtype(ds, jp)
        cell = {"json": str(jp), "error": rec.get("error"), "folds_done": rec.get("folds_done"),
                "scheme": rec.get("fold_scheme_effective")}
        manifest["cells"][ds] = cell
        codes = rec.get("generated_code") or []
        if rec.get("abstain") or not codes:
            cell["skipped"] = "abstain or no generated_code"
            continue
        data, y, task_type = load_synp(Path(a.tasks_root), ds)
        X = caafe_prep(data)
        scheme = rec.get("fold_scheme_requested") or "kfold"
        folds = caafe_folds(len(X), y.to_numpy(), task_type, scheme)
        arm = a.arm
        for k, code in enumerate(codes):
            tr = folds[k][0]
            frame = X.iloc[tr]
            row = {"dataset_id": ds, "fold": k, "code_chars": len(code.strip())}
            parts = []
            if code.strip():
                res, status = run_program("caafe", code, frame, tmp, f"caafe_{ds}_{k}", a.timeout)
                if res is not None:
                    enc, info = encode_new_columns(res, frame)
                    row.update(info)
                    if enc is None:
                        status = "error: " + info["fail"]
                    else:
                        parts = [enc]
            else:
                status = "empty_code"
            row["status"] = status
            prog_rows.append(row)
            ncol = write_table(fdir / f"{ds}_{arm}_fold{k}.csv.gz", data["id"].to_numpy()[tr], parts)
            print(f"[x6] caafe {ds} fold {k}: {status}, {ncol} feature columns", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prog_rows).to_csv(out / "programs_caafe.csv", index=False)
    (out / "manifest_caafe.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    try:
        tmp.rmdir()
    except OSError:
        pass
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="X6 comparator-output parser -> detector feature tables")
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("llmfe", help="LLM-FE samples_*.json -> per-fold tables (notebook/best3/any)")
    l.add_argument("--llmfe-logs", required=True, help="LLM-FE ./logs (holds <prefix>_split_<i>/samples)")
    l.add_argument("--llmfe-data", required=True, help="LLM-FE ./data (classification inputs)")
    l.add_argument("--alias-root", required=True, help="A08 llmfe_alias/ (regression crab.csv per task)")
    l.add_argument("--partition-dir", required=True, help="A08 partition/ (llmfe_<task>.json)")
    l.add_argument("--tasks-root", required=True)
    l.add_argument("--out", required=True, help="output dir; tables go to <out>/f42/")
    l.add_argument("--tasks", nargs="*", default=None)
    l.add_argument("--reps", nargs="*", type=int, default=[0, 1])
    l.add_argument("--suffix-template", default="{task}_oss120_a08r{rep}",
                   help="log-dir prefix; split dirs are <prefix>_split_<i>")
    l.add_argument("--selections", nargs="*", default=list(SELECTIONS), choices=SELECTIONS)
    l.add_argument("--timeout", type=float, default=120.0, help="seconds per program execution")
    l.add_argument("--execute-all", action="store_true",
                   help="also execute programs no selection uses (score None, not in notebook top-3)")
    c = sub.add_parser("caafe", help="CAAFE arm JSON generated_code -> per-fold tables")
    c.add_argument("--arm-json", nargs="+", required=True)
    c.add_argument("--tasks-root", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--arm", default="caafe")
    c.add_argument("--timeout", type=float, default=120.0)
    a = ap.parse_args(argv)
    return cmd_llmfe(a) if a.cmd == "llmfe" else cmd_caafe(a)


if __name__ == "__main__":
    sys.exit(main())
