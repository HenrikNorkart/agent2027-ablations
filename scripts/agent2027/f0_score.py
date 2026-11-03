"""E11d / A17-T8: score A17 F0 pools (``f0pool_<rung>``) -- per draw, per fold, best-of-k.

Spec: ``docs/ablations/A17_CAP1_scale_ladder.md`` A17-T8 (this scorer), A17-T18 (analysis
columns), §2 (F0 analysis columns, Stage C), §4 (framework levels, F0 seed rule), §5 (failed
draws); pre-registration Amendment 9 (parse/exec metric split, over-limit answers: first 10 in
the model's order). Pools are written by ``scripts/agent2027/f0_generate.py`` (X7, A17-T7):
one folder per A17-1xx row with ``draws.jsonl``, ``folds.json``, ``summary.json``.

Per draw (fold f, draw j)
-------------------------
1. **Parse** the raw answer with the generator's own ``parse_answer`` (a ```json fence is
   stripped; a JSON list of objects with string ``name`` and ``pandas_command``). Over-limit
   answers (> 10 items) are cut to the first 10 in the model's order (Amendment 9).
2. **Validate** each item with ``exec_user_command`` (``rogueone/utils/pandas_exec.py``) with
   ``df`` = the raw table and ``df_features`` = an id-only frame (A17-T8), applying the accept
   checks of the Extractor's ``append_new_attribute`` (``ext/agent.py``): ``result`` set,
   no NaN/inf, numeric, assignable to ``df_features`` (one value per id), name not already
   present. Accepted items are added to the validation frame, as the Extractor adds them.
   A draw **executes** when at least one item is accepted.
3. **Attribute table** in the frozen pool mode (augment, pre-registration "A03 M1"): the E6
   seed rows of every raw column (``seed_raw_feature_rows``) followed by the accepted items,
   with the ``Attribute`` / ``Pandas Command`` columns ``materialize_fold_safe_features``
   expects.
4. **Replay** fold-safely with ``build_fold_safe_entities_folds`` (the orchestrator's own call,
   ``rogue_one_main.py``): train rows and test rows replayed separately.
5. **Stage C**: the Tester's matrices (``test_hypotheses``: drop ``id``, the train folds
   vstacked in fold order), ``_inner_cv_score`` (``tst/agent.py``; 4 inner folds,
   ``seed=fold_seed``, balanced weights under ``ROGUE_ONE_DO_BALANCING``) and the 400-tree
   XGBoost held-out model with the Tester's params (regression: ``n_RMSE`` = RMSE / mean of the
   training target; classification: accuracy with balanced test weights = balanced accuracy).
6. **Recovery**: ``recovery_detector.detect_episode`` (amended detector of record) on the
   fold's training rows, active set = the accepted items, selected iteration = 0.

A draw that fails to parse, has no accepted item, or whose Stage C fails is a **failed draw**
(selection score -inf, A17 §5). Errors never stop the run.

Best-of-k (A17-T8)
------------------
For k = 1..n (n = 128), the mean over 200 seeded random orderings of the draws of the draw
with the best inner-CV score among the first k (ties: first in the ordering, as ``idxmax``):
its ``rec_sel``, its held-out score, and the cumulative completion tokens of the first k
draws. If all first k draws failed, the cell takes the ITT value (01 §4.5, augment is
canonical): held-out = ``base_no_fe`` (the seed-only pool scored by the same Stage C on the
same fold) and ``rec_sel = 0``.

Outputs (``--out``)
-------------------
* ``cache/<row>.draws.csv``: the per-draw table of one pool (reused unless ``--rescore``).
* ``draws.csv``: all per-draw rows; ``curves.csv``: one row per (row, fold, k);
  ``fold_summary.csv``: one row per (row, fold) with the A17-T18 columns
  ``f0_parse_rate`` (generator definition, Amendment 9), ``f0_exec_rate`` (executes | parses),
  ``f0_parse_exec_rate`` (the A17-T18 "parse and execute" share), ``k_matched``,
  ``tok_ratio_f2_f0`` and ``rec_sel_bestk`` (the last three need ``--f2-tokens``);
  ``run_info.json``.

CPU only. Run on CAIR only inside a one-shot ``rogue_one`` container (never host Python),
``--network none`` recommended (it executes model-written code), e.g.::

  docker run --rm --network none --cpus 16 --user $(id -u):$(id -g) -e HOME=/tmp \\
    -v ~/ro27_f0_work:/pools:ro -v ~/code/RougeOne/tasks:/tasks:ro \\
    -v ~/rogueone-synp-meta:/meta:ro -v <code>:/code:ro -v ~/ro27_analysis/f0:/out \\
    -w /code --entrypoint python3 rogue_one:agent2027-v2.3 scripts/agent2027/f0_score.py \\
    --pools /pools --tasks /tasks --meta /meta --out /out --cpus 16
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import multiprocessing
import os
import re
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- frozen design
N_FOLDS = 5
N_DRAWS = 128
MAX_ITEMS = 10                 # Amendment 9: first 10 in the model's order
N_ORDERINGS = 200              # A17-T8: 200 seeded random orderings
ORDER_SEED = 20271101          # not pre-registered: fixed here, logged in run_info.json
ITEM_TIMEOUT_S = 120           # per-item validation exec (POSIX only)
DRAW_TIMEOUT_S = 900           # replay + Stage C of one draw (POSIX only)
RESERVED_NAMES = frozenset({"id", "target"})
#: Validation frame: ``id_only`` (A17-T8 as written, the default) or ``augment`` (id + the E6
#: seed columns, i.e. the Extractor's authoring frame under augment; sensitivity only).
VALIDATE_FRAMES = ("id_only", "augment")

#: Frozen Full-v2 settings the Tester and the replay read from ``rouge_one_cfg``
#: (``scripts/agent2027/v2_defaults.env``). ``setdefault``: an explicit env wins.
V2_ENV = {
    "ROGUE_ONE_DO_BALANCING": "True",
    "ROGUE_ONE_FEATURE_POOL_MODE": "augment",
    "ROGUE_ONE_SANDBOX": "on",
    "ROGUE_ONE_K_FOLDS": "5",
    "ROGUE_ONE_STRATIFIED_FOLDS": "False",
    "CONSOLE_DISABLE": "True",
}
_DUMMY_ENV = {"EMBEDD_API_KEY": "x", "EMBEDD_MODEL": "x", "EMBEDD_PORT": "1", "LLM_MODEL": "x",
              "LLM_API_KEY": "x", "LLM_PORT": "1"}

DRAW_COLUMNS = [
    "row_id", "rung", "dataset_id", "task_type", "fold_seed", "fold", "draw", "seed", "present",
    "error_kind", "finish_reason", "completion_tokens", "prompt_tokens",
    "parse_ok", "parse_error", "fenced", "n_items", "over_limit", "n_items_used",
    "n_items_valid", "item_errors", "exec_ok", "scored", "fail_reason",
    "inner_cv", "heldout", "sel_score", "rec_sel", "hits", "best_rho", "n_features", "wall_s",
]


# --------------------------------------------------------------------------- env / imports
def setup_env() -> None:
    for k, v in {**_DUMMY_ENV, **V2_ENV}.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


def _f0gen():
    setup_env()
    from scripts.agent2027 import f0_generate  # noqa: PLC0415

    return f0_generate


# --------------------------------------------------------------------------- Stage C params
def heldout_params(task_type: str, fold_seed: int, n_jobs: int = 1) -> dict[str, Any]:
    """The Tester's 400-tree held-out params (``tst/agent.py`` get_regression_metrics /
    get_classification_metrics_tabular; ``random_state = self.fold_seed``). ``n_jobs`` is the
    only change (the Tester uses ``_XGB_N_JOBS``; thread count does not change ``hist``)."""
    common = {"tree_method": "hist", "random_state": int(fold_seed), "device": "cpu",
              "n_estimators": 400, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.9,
              "colsample_bytree": 0.9, "n_jobs": int(n_jobs)}
    if task_type == "classification":
        return {"objective": "binary:logistic", "eval_metric": "logloss", **common}
    return {"objective": "reg:squarederror", "eval_metric": "rmse", **common}


def stage_c(x_train, y_train, x_test, y_test, task_type: str, fold_seed: int,
            balancing: bool, n_jobs: int = 1) -> tuple[Optional[float], float]:
    """(inner-CV score, held-out score) exactly as the Tester computes them for one trial."""
    import numpy as np  # noqa: PLC0415
    import xgboost as xgb  # noqa: PLC0415
    from sklearn.metrics import accuracy_score, mean_squared_error  # noqa: PLC0415
    from sklearn.preprocessing import LabelEncoder  # noqa: PLC0415
    from sklearn.utils import class_weight  # noqa: PLC0415

    from rogueone.llm.agents.core.tester.agent import _inner_cv_score  # noqa: PLC0415

    params = heldout_params(task_type, fold_seed, n_jobs)
    xtr = np.nan_to_num(np.asarray(x_train, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    xte = np.nan_to_num(np.asarray(x_test, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    if task_type == "classification":
        le = LabelEncoder()
        le.fit(np.concatenate([y_train, y_test]))
        ytr = np.asarray(le.transform(y_train)).ravel().astype(int)
        yte = np.asarray(le.transform(y_test)).ravel().astype(int)
        w_tr = class_weight.compute_sample_weight("balanced", y=ytr) if balancing else None
        w_te = class_weight.compute_sample_weight("balanced", y=yte) if balancing else None
        model = xgb.XGBClassifier(**params)
        model.fit(xtr, ytr, sample_weight=w_tr)
        heldout = float(accuracy_score(yte, model.predict(xte), sample_weight=w_te))
        inner = _inner_cv_score(X_train=xtr, y_train=ytr, base_params=params,
                                task_type="classification", seed=int(fold_seed),
                                balance_weights=bool(balancing))
    else:
        ytr = np.asarray(y_train, dtype=float)
        yte = np.asarray(y_test, dtype=float)
        model = xgb.XGBRegressor(**params)
        model.fit(xtr, ytr)
        rmse = float(np.sqrt(mean_squared_error(yte, model.predict(xte))))
        heldout = rmse / float(np.mean(ytr))  # TestResultRegression.n_rmses: rmse / mean_val
        inner = _inner_cv_score(X_train=xtr, y_train=ytr, base_params=params,
                                task_type="regression", seed=int(fold_seed))
    return inner, heldout


def selection_score(inner: Optional[float], task_type: str) -> float:
    """Higher is better: inner accuracy, or minus inner RMSE. Failed -> -inf."""
    if inner is None or not math.isfinite(float(inner)):
        return -math.inf
    return float(inner) if task_type == "classification" else -float(inner)


# --------------------------------------------------------------------------- pool reading
def load_pool(pool_dir: Path) -> tuple[dict[str, Any], dict[tuple[int, int], dict]]:
    """Pool metadata and its draws (last record per (fold, draw), as the generator resumes)."""
    pool_dir = Path(pool_dir)
    folds = json.loads((pool_dir / "folds.json").read_text(encoding="utf-8"))
    summ_p = pool_dir / "summary.json"
    summary = json.loads(summ_p.read_text(encoding="utf-8")) if summ_p.exists() else {}
    meta = {
        "row_id": folds.get("row_id") or summary.get("row_id") or pool_dir.name,
        "dataset_id": folds["dataset_id"], "fold_seed": int(folds["fold_seed"]),
        "stratified": bool(folds.get("stratified", False)), "test_ids": folds["test_ids"],
        "run_name": summary.get("run_name"), "model": summary.get("model"),
        "complete": summary.get("complete"), "pool_dir": str(pool_dir),
    }
    m = re.match(r"A17__f0pool_(?P<rung>[^_]+)__", meta["run_name"] or "")
    meta["rung"] = m.group("rung") if m else None
    if "covtype" in json.dumps({k: v for k, v in meta.items() if k != "test_ids"}).lower():
        raise ValueError("covtype is never run (Henrik, 2026-10-16)")
    draws: dict[tuple[int, int], dict] = {}
    p = pool_dir / "draws.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                draws[(int(r["fold"]), int(r["draw"]))] = r
            except Exception:  # noqa: BLE001 - a truncated last line must not stop scoring
                continue
    return meta, draws


def manifest_rows(path: Optional[Path]) -> dict[str, dict[str, str]]:
    if path is None or not Path(path).exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["row_id"]: r for r in csv.DictReader(fh)}


def resolve_task_type(tasks_root: Path, dataset_id: str, man_row: Optional[dict]) -> str:
    if man_row and man_row.get("task_type"):
        return man_row["task_type"]
    hits = [t for t in ("classification", "regression") if (Path(tasks_root) / t / dataset_id).is_dir()]
    if len(hits) != 1:
        raise ValueError(f"cannot resolve task_type of {dataset_id} under {tasks_root}")
    return hits[0]


# --------------------------------------------------------------------------- parse + validate
def parse_draw(rec: Optional[dict]) -> dict[str, Any]:
    """Generator parse (``f0_generate.parse_answer``) of the raw text; items cut to 10."""
    f0 = _f0gen()
    parsed = f0.parse_answer(None if rec is None else rec.get("raw_text"))
    items = list(parsed["items"] or [])
    return {"parse_ok": bool(parsed["ok"]), "parse_error": parsed["error"],
            "fenced": bool(parsed["fenced"]), "n_items": int(parsed["n_items"]),
            "over_limit": bool(parsed["over_limit"]), "items": items[:MAX_ITEMS]}


@contextlib.contextmanager
def _time_limit(seconds: float):
    """SIGALRM time limit (POSIX main thread only; a no-op elsewhere)."""
    usable = hasattr(signal, "SIGALRM") and seconds and seconds > 0
    if usable:
        try:
            def _raise(signum, frame):  # noqa: ARG001
                raise TimeoutError(f"exceeded {seconds}s")
            old = signal.signal(signal.SIGALRM, _raise)
            signal.setitimer(signal.ITIMER_REAL, float(seconds))
        except ValueError:  # not the main thread
            usable = False
    try:
        yield
    finally:
        if usable:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)


def _accept_result(result: Any, n_rows: int):
    """The accept checks of ``append_new_attribute`` (ext/agent.py); returns the stored value
    or raises ``ValueError`` with the Extractor's reason."""
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    if result is None:
        raise ValueError("no_result")
    if isinstance(result, (pd.Series, pd.DataFrame)):
        result = result.reset_index(drop=True).values
    if isinstance(result, np.ndarray):
        if np.isnan(result).any() or np.isinf(result).any():
            raise ValueError("nan_or_inf")
        if not np.issubdtype(result.dtype, np.number):
            raise ValueError("non_numeric")
    else:
        if pd.isnull(result) or result in [float("inf"), float("-inf")]:
            raise ValueError("nan_or_inf")
        pd.Series(result.flatten() if hasattr(result, "flatten") else result).astype(float)
    return result


def validate_items(items: list[dict], df_raw, *, item_timeout: float = ITEM_TIMEOUT_S,
                   df_features=None) -> tuple[list[dict], list[str]]:
    """Validate parsed items in order; returns (accepted items, per-item status strings).

    ``df_features`` defaults to the id-only frame of A17-T8; a frame passed in (the
    ``--validate-frame augment`` sensitivity: id + E6 seed columns) is copied, not mutated."""
    import pandas as pd  # noqa: PLC0415

    from rogueone.utils.pandas_exec import exec_user_command  # noqa: PLC0415

    if df_features is not None:
        df_features = df_features.copy()
    else:
        try:
            df_features = pd.DataFrame(df_raw["id"].unique(), columns=["id"])
        except Exception:  # noqa: BLE001 - same fallback as the orchestrator
            df_features = pd.DataFrame({"id": list(df_raw.index)})
    raw_names = {str(c) for c in df_raw.columns if c != "id"}
    accepted, status = [], []
    for item in items:
        name = str(item.get("name", "")).strip()
        command = item.get("pandas_command")
        if not name or name in RESERVED_NAMES:
            status.append(f"{name or '?'}:reserved_name")
            continue
        if name in raw_names or name in df_features.columns:
            status.append(f"{name}:duplicate")
            continue
        try:
            with _time_limit(item_timeout), contextlib.redirect_stdout(io.StringIO()):
                result, _ = exec_user_command(command, df_raw, df_features)
                value = _accept_result(result, len(df_features))
                df_features[name] = value
        except Exception as exc:  # noqa: BLE001 - any failure rejects the item
            reason = exc.args[0] if isinstance(exc, ValueError) and exc.args and \
                exc.args[0] in ("no_result", "nan_or_inf", "non_numeric") else type(exc).__name__
            status.append(f"{name}:{reason}")
            if name in df_features.columns:
                df_features.drop(columns=[name], inplace=True)
            continue
        accepted.append({"name": name, "pandas_command": command,
                         "argumentation": str(item.get("argumentation", ""))})
        status.append(f"{name}:ok")
    return accepted, status


# --------------------------------------------------------------------------- task context
class TaskContext:
    """Raw table, labels, seed rows, phi and folds of one (dataset, fold_seed) pool."""

    def __init__(self, tasks_root: Path, task_type: str, dataset_id: str, fold_seed: int,
                 stratified: bool, test_ids: list, meta_dir: Optional[Path],
                 validate_frame: str = "id_only"):
        import numpy as np  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415

        from rogueone.utils.fold_safe_features import seed_raw_feature_rows  # noqa: PLC0415

        f0 = _f0gen()
        if "covtype" in str(dataset_id).lower():
            raise ValueError("covtype is never run (Henrik, 2026-10-16)")
        self.task_type, self.dataset_id, self.fold_seed = task_type, dataset_id, int(fold_seed)
        d = Path(tasks_root) / task_type / dataset_id
        self.df_raw = pd.read_csv(d / "data.csv")
        labels = pd.read_csv(d / "labels.csv")
        if "id" not in labels.columns and "patient_id" in labels.columns:
            labels = labels.rename(columns={"patient_id": "id"})
        self.labels = labels
        # fold guard: the pool's folds must equal an independent rebuild (A17-T7)
        rebuilt = f0.outer_folds(self.df_raw, labels, task_type, self.fold_seed, stratified)
        if [sorted(map(str, f["test_ids"])) for f in rebuilt] != [sorted(map(str, t)) for t in test_ids]:
            raise ValueError(f"{dataset_id} f{fold_seed}: folds.json differs from the rebuilt folds")
        self.fixed_test_ids = [set(f["test_ids"]) for f in rebuilt]
        ent = pd.DataFrame(self.df_raw["id"].unique(), columns=["id"])
        empty = pd.DataFrame(columns=["Attribute", "Description", "Pandas Command", "Status",
                                      "Added", "Pruned Iteration", "Prune Rationale"])
        self.seed_entities, self.seed_attrs = seed_raw_feature_rows(self.df_raw, ent, empty)
        if validate_frame not in VALIDATE_FRAMES:
            raise ValueError(f"validate_frame must be one of {VALIDATE_FRAMES}")
        self.validate_frame = validate_frame
        self.phi, self.detector = None, None
        if meta_dir is not None and (Path(meta_dir) / f"{dataset_id}.json").exists():
            from scripts.agent2027.analysis import recovery_detector as rd  # noqa: PLC0415

            self.phi, self.detector = rd.load_phi(meta_dir, dataset_id, self.df_raw)
        self.train_masks = [~self.df_raw["id"].isin(t).to_numpy() for t in self.fixed_test_ids]
        self._np = np

    def attribute_table(self, accepted: list[dict]):
        import pandas as pd  # noqa: PLC0415

        rows = [{"Attribute": it["name"], "Description": it["argumentation"],
                 "Pandas Command": it["pandas_command"], "Status": "Active", "Added": "F0",
                 "Seeded": False} for it in accepted]
        attrs = pd.concat([self.seed_attrs, pd.DataFrame(rows, columns=list(self.seed_attrs.columns))],
                          ignore_index=True) if rows else self.seed_attrs.copy()
        attrs["is_fallback"] = False
        return attrs

    def matrices(self, attrs, fold: int):
        """The Tester's matrices for test fold ``fold`` (``test_hypotheses``)."""
        import numpy as np  # noqa: PLC0415

        from rogueone.utils.fold_safe_features import build_fold_safe_entities_folds  # noqa: PLC0415

        with contextlib.redirect_stdout(io.StringIO()):
            folds = build_fold_safe_entities_folds(
                df_raw=self.df_raw, attribute_explanations=attrs, labels_df=self.labels,
                fixed_test_ids=self.fixed_test_ids, current_test_fold_id=fold)
        folds = [d.drop("id", axis=1) for d in folds]
        names = [c for c in folds[0].columns if c != "target"]
        x = [d[names].to_numpy() for d in folds]
        y = [d["target"].to_numpy() for d in folds]
        tr = [i for i in range(len(folds)) if i != fold]
        return (np.vstack([x[i] for i in tr]), np.hstack([y[i] for i in tr]), x[fold], y[fold],
                len(names))

    def detect(self, attrs, accepted: list[dict], fold: int) -> dict[str, Any]:
        if self.phi is None or not accepted:
            return {"rec_sel": None if self.phi is None else 0, "hits": "", "best_rho": None}
        from scripts.agent2027.analysis import recovery_detector as rd  # noqa: PLC0415

        names = [it["name"] for it in accepted]
        with contextlib.redirect_stdout(io.StringIO()):
            res = rd.detect_episode(self.df_raw, attrs, self.train_masks[fold], self.phi,
                                    self.detector, {0: names}, 0, amended=True)
        pf = res.per_feature.set_index("feature")
        rhos = [float(pf.loc[n, "rho"]) for n in names if n in pf.index]
        return {"rec_sel": int(res.rec_sel), "hits": ";".join(res.hits_by_iter.get(0, [])),
                "best_rho": max(rhos) if rhos else None}


# --------------------------------------------------------------------------- worker
_CTX: dict[tuple, TaskContext] = {}
_CTX_ARGS: dict[tuple, tuple] = {}


def _init_worker(ctx_args: dict[tuple, tuple]) -> None:
    setup_env()
    _CTX_ARGS.clear()
    _CTX_ARGS.update(ctx_args)


def _ctx(key: tuple) -> TaskContext:
    if key not in _CTX:
        _CTX[key] = TaskContext(*_CTX_ARGS[key])
    return _CTX[key]


def score_draw(ctx: TaskContext, fold: int, rec: Optional[dict], *,
               item_timeout: float = ITEM_TIMEOUT_S, draw_timeout: float = DRAW_TIMEOUT_S
               ) -> dict[str, Any]:
    """Score one draw (``rec`` None = missing from the pool, counted as failed)."""
    t0 = time.monotonic()
    out: dict[str, Any] = {"present": rec is not None, "exec_ok": False, "scored": False,
                           "fail_reason": None, "inner_cv": None, "heldout": None,
                           "sel_score": -math.inf, "rec_sel": None, "hits": "", "best_rho": None,
                           "n_items_valid": 0, "item_errors": "", "n_features": None}
    rec = rec or {}
    out.update(seed=rec.get("seed"), error_kind=rec.get("error_kind"),
               finish_reason=rec.get("finish_reason"),
               completion_tokens=int(rec.get("completion_tokens") or 0),
               prompt_tokens=int(rec.get("prompt_tokens") or 0))
    p = parse_draw(rec if out["present"] else None)
    out.update({k: p[k] for k in ("parse_ok", "parse_error", "fenced", "n_items", "over_limit")})
    out["n_items_used"] = len(p["items"])
    if not out["present"]:
        out["fail_reason"] = "missing"
    elif not p["parse_ok"]:
        out["fail_reason"] = "parse"
    else:
        # a copy: model code may mutate ``df`` in place, which must not leak into later draws
        frame = ctx.seed_entities if ctx.validate_frame == "augment" else None
        accepted, status = validate_items(p["items"], ctx.df_raw.copy(), item_timeout=item_timeout,
                                          df_features=frame)
        out["n_items_valid"] = len(accepted)
        out["item_errors"] = ";".join(s for s in status if not s.endswith(":ok"))
        if not accepted:
            out["fail_reason"] = "exec"
        else:
            out["exec_ok"] = True
            attrs = ctx.attribute_table(accepted)
            try:
                with _time_limit(draw_timeout):
                    xtr, ytr, xte, yte, nf = ctx.matrices(attrs, fold)
                    out["n_features"] = nf
                    inner, held = stage_c(xtr, ytr, xte, yte, ctx.task_type, ctx.fold_seed,
                                          _balancing())
                out["inner_cv"], out["heldout"] = inner, held
                out["sel_score"] = selection_score(inner, ctx.task_type)
                out["scored"] = math.isfinite(out["sel_score"]) and held is not None                     and math.isfinite(float(held))
                if not out["scored"]:
                    out["sel_score"] = -math.inf
                    out["fail_reason"] = "stage_c_no_score"
            except Exception as exc:  # noqa: BLE001 - incl. the time limit
                out["fail_reason"] = f"stage_c:{type(exc).__name__}"
            if out["scored"]:
                try:
                    with _time_limit(draw_timeout):
                        out.update(ctx.detect(attrs, accepted, fold))
                except Exception as exc:  # noqa: BLE001
                    out["rec_sel"] = 0
                    out["fail_reason"] = f"detector:{type(exc).__name__}"
    if not out["scored"] and ctx.phi is not None:
        out["rec_sel"] = 0
    out["wall_s"] = round(time.monotonic() - t0, 3)
    return out


def score_base(ctx: TaskContext, fold: int) -> dict[str, Any]:
    """``base_no_fe`` for one fold: the seed-only (augment iteration-0) pool through Stage C."""
    xtr, ytr, xte, yte, nf = ctx.matrices(ctx.attribute_table([]), fold)
    inner, held = stage_c(xtr, ytr, xte, yte, ctx.task_type, ctx.fold_seed, _balancing())
    return {"base_inner_cv": inner, "base_no_fe_heldout": held, "base_n_features": nf}


def _balancing() -> bool:
    from rogueone.utils import rouge_one_cfg  # noqa: PLC0415

    return bool(rouge_one_cfg.do_balancing)


def _job(task: tuple) -> tuple:
    key, kind, fold, draw, rec = task
    ctx = _ctx(key)
    if kind == "base":
        try:
            return (key, kind, fold, draw, score_base(ctx, fold))
        except Exception as exc:  # noqa: BLE001
            return (key, kind, fold, draw, {"base_inner_cv": None, "base_no_fe_heldout": None,
                                            "base_n_features": None,
                                            "base_error": f"{type(exc).__name__}: {exc}"[:300]})
    try:
        return (key, kind, fold, draw, score_draw(ctx, fold, rec))
    except Exception as exc:  # noqa: BLE001 - never crash the run
        return (key, kind, fold, draw, {"present": rec is not None, "scored": False,
                                        "exec_ok": False, "sel_score": -math.inf,
                                        "fail_reason": f"worker:{type(exc).__name__}"})


# --------------------------------------------------------------------------- best-of-k
def orderings(n: int, n_orderings: int = N_ORDERINGS, seed: int = ORDER_SEED):
    """``n_orderings`` seeded random permutations of ``range(n)`` (one row each)."""
    import numpy as np  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(n) for _ in range(n_orderings)])


def best_of_k(sel_score, heldout, rec_sel, tokens, base_heldout, perms):
    """Best-of-k curves for k = 1..n averaged over the orderings ``perms`` (n_ord x n).

    Selection = argmax of ``sel_score`` among the first k draws of an ordering, ties to the
    first in the ordering; a prefix of only failed draws (-inf) takes the ITT value
    (held-out = ``base_heldout``, rec_sel = 0). Returns a list of dicts, one per k.
    """
    import numpy as np  # noqa: PLC0415

    s = np.asarray(sel_score, dtype=float)
    h = np.asarray([np.nan if v is None else float(v) for v in heldout], dtype=float)
    r = np.asarray([0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
                    for v in rec_sel], dtype=float)
    rec_known = not all(v is None for v in rec_sel)
    t = np.asarray(tokens, dtype=float)
    n_ord, n = perms.shape
    best_val = np.full(n_ord, -np.inf)
    best_idx = np.full(n_ord, -1)
    cum_tok = np.zeros(n_ord)
    base = np.nan if base_heldout is None else float(base_heldout)
    rows = []
    for k in range(n):
        j = perms[:, k]
        v = s[j]
        upd = v > best_val
        best_val = np.where(upd, v, best_val)
        best_idx = np.where(upd, j, best_idx)
        cum_tok += t[j]
        itt = best_idx < 0
        safe = np.where(itt, 0, best_idx)
        held = np.where(itt, base, h[safe])
        rs = np.where(itt, 0.0, r[safe])
        rows.append({"k": k + 1, "heldout_bestk": float(np.mean(held)),
                     "rec_sel_bestk": float(np.mean(rs)) if rec_known else None,
                     "tok_compl_cum": float(np.mean(cum_tok)), "itt_share": float(np.mean(itt))})
    return rows


# --------------------------------------------------------------------------- per-fold summary
def fold_summary(draw_rows: list[dict], curve: list[dict], base: dict, f2_tok: Optional[float]
                 ) -> dict[str, Any]:
    n = len(draw_rows)
    parsed = sum(1 for d in draw_rows if d.get("parse_ok"))
    execd = sum(1 for d in draw_rows if d.get("exec_ok"))
    scored = sum(1 for d in draw_rows if d.get("scored"))
    items_used = sum(int(d.get("n_items_used") or 0) for d in draw_rows)
    items_ok = sum(int(d.get("n_items_valid") or 0) for d in draw_rows)
    tok = sum(int(d.get("completion_tokens") or 0) for d in draw_rows)
    rec_vals = [d.get("rec_sel") for d in draw_rows if d.get("rec_sel") is not None]
    first = next((d for d in draw_rows if int(d["draw"]) == 0), None)
    base_h = base.get("base_no_fe_heldout")
    d1_ok = bool(first and first.get("scored"))
    out = {
        "n_draws": n, "n_missing": sum(1 for d in draw_rows if not d.get("present")),
        "f0_parse_rate": parsed / n if n else None,
        "f0_exec_rate": execd / parsed if parsed else None,
        "f0_parse_exec_rate": execd / n if n else None,
        "f0_scored_rate": scored / n if n else None,
        "over_limit_rate": sum(1 for d in draw_rows if d.get("over_limit")) / n if n else None,
        "item_accept_rate": items_ok / items_used if items_used else None,
        "tok_compl_pool": tok, "tok_compl_mean_per_draw": tok / n if n else None,
        **base,
        "f0_heldout": first.get("heldout") if d1_ok else base_h,
        "f0_rec_sel": (int(first.get("rec_sel") or 0) if d1_ok else 0) if rec_vals else None,
        "f0_itt": int(not d1_ok),
        "rec_any_pool": int(any(v == 1 for v in rec_vals)) if rec_vals else None,
        "heldout_best128": curve[-1]["heldout_bestk"] if curve else None,
        "rec_sel_best128": curve[-1]["rec_sel_bestk"] if curve else None,
        "f2_tok_compl_total": f2_tok, "k_matched": None, "k_used": None,
        "k_matched_le_128": None, "tok_ratio_f2_f0": None, "rec_sel_bestk": None,
        "heldout_bestk_matched": None,
    }
    mean_tok = out["tok_compl_mean_per_draw"]
    if f2_tok is not None and mean_tok and n:
        k_matched = float(f2_tok) / float(mean_tok)
        k_used = int(min(max(math.floor(k_matched), 1), n))
        out.update(k_matched=k_matched, k_used=k_used, k_matched_le_128=bool(k_matched <= n),
                   tok_ratio_f2_f0=k_matched / n,
                   rec_sel_bestk=curve[k_used - 1]["rec_sel_bestk"],
                   heldout_bestk_matched=curve[k_used - 1]["heldout_bestk"])
    return out


def read_f2_tokens(path: Optional[Path]) -> dict[tuple, float]:
    """``--f2-tokens`` CSV: rung, dataset_id, fold_seed, fold, tok_compl_total (F2 per fold)."""
    if path is None:
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[(r["rung"], r["dataset_id"], int(r["fold_seed"]), int(r["fold"]))] = float(r["tok_compl_total"])
    return out


# --------------------------------------------------------------------------- driver
def _fmt(v: Any) -> Any:
    if isinstance(v, float) and math.isinf(v):
        return "-inf" if v < 0 else "inf"
    return v


def _read_cache(path: Path) -> list[dict]:
    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(path, keep_default_na=True)
    rows = []
    for r in df.to_dict("records"):
        r = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
        r["sel_score"] = float(r["sel_score"]) if r.get("sel_score") is not None else -math.inf
        for b in ("present", "parse_ok", "exec_ok", "scored", "over_limit", "fenced"):
            if r.get(b) is not None:
                r[b] = str(r[b]).lower() in ("true", "1")
        rows.append(r)
    return rows


def discover_pools(paths: Iterable[str], rows: Optional[set[str]]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        cand = [p] if (p / "folds.json").exists() else sorted(q for q in p.iterdir() if (q / "folds.json").exists())
        out.extend(q for q in cand if rows is None or q.name in rows)
    return out


def run(args: argparse.Namespace) -> int:
    setup_env()
    import pandas as pd  # noqa: PLC0415

    t_start = time.monotonic()
    out = Path(args.out)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    man = manifest_rows(Path(args.manifest) if args.manifest else None)
    f2 = read_f2_tokens(Path(args.f2_tokens) if args.f2_tokens else None)
    pools = discover_pools(args.pools, set(args.rows) if args.rows else None)
    if not pools:
        print("[f0_score] no pools found", file=sys.stderr)
        return 2

    plans, ctx_args, refused = [], {}, []
    base_tasks: dict[tuple, tuple] = {}
    draw_tasks: list[tuple] = []
    for pdir in pools:
        try:
            meta, draws = load_pool(pdir)
            task_type = resolve_task_type(Path(args.tasks), meta["dataset_id"], man.get(meta["row_id"]))
        except Exception as exc:  # noqa: BLE001
            refused.append(f"{pdir.name}: {exc}")
            print(f"[f0_score] REFUSE {pdir}: {exc}", file=sys.stderr, flush=True)
            continue
        n_draws = int(args.n_draws or N_DRAWS)
        n_folds = len(meta["test_ids"])
        key = (meta["dataset_id"], meta["fold_seed"], meta["stratified"], task_type)
        ctx_args[key] = (Path(args.tasks), task_type, meta["dataset_id"], meta["fold_seed"],
                         meta["stratified"], meta["test_ids"], Path(args.meta) if args.meta else None,
                         args.validate_frame)
        cache = out / "cache" / f"{meta['row_id']}.draws.csv"
        extra = sorted(k for k in draws if not (0 <= k[0] < n_folds and 0 <= k[1] < n_draws))
        meta.update(task_type=task_type, n_draws=n_draws, n_folds=n_folds, key=key, cache=cache,
                    n_extra_records=len(extra))
        plans.append(meta)
        for f in range(n_folds):  # base_no_fe is shared by every pool of a (dataset, fold_seed)
            base_tasks.setdefault((key, f), (key, "base", f, -1, None))
        meta["cached"] = cache.exists() and not args.rescore
        if not meta["cached"]:
            for f in range(n_folds):
                for j in range(n_draws):
                    draw_tasks.append((key, "draw", f, j, draws.get((f, j))))
    jobs = list(base_tasks.values()) + draw_tasks
    print(f"[f0_score] {len(plans)} pools, {len(jobs)} jobs, cpus={args.cpus}", flush=True)

    done: list[tuple] = []
    if args.cpus <= 1:
        _init_worker(ctx_args)
        done = [_job(j) for j in jobs]
    else:
        # spawn, not fork: a forked child of a parent that has started OpenMP threads
        # (XGBoost) can deadlock (seen once in the test suite)
        with ProcessPoolExecutor(max_workers=args.cpus, initializer=_init_worker,
                                 initargs=(ctx_args,),
                                 mp_context=multiprocessing.get_context("spawn")) as ex:
            for i, res in enumerate(ex.map(_job, jobs, chunksize=2)):
                done.append(res)
                if (i + 1) % 500 == 0:
                    print(f"[f0_score] {i + 1}/{len(jobs)} jobs, {time.monotonic() - t_start:.0f}s",
                          flush=True)
    results = {(r[0], r[2]): r[4] for r in done[:len(base_tasks)]}
    draw_results = done[len(base_tasks):]

    # draws back to their pools (job order = pool order, folds, draws)
    by_pool: dict[str, list[dict]] = {}
    di = 0
    for meta in plans:
        if meta["cached"]:
            by_pool[meta["row_id"]] = _read_cache(meta["cache"])
            continue
        rows = []
        for f in range(meta["n_folds"]):
            for j in range(meta["n_draws"]):
                key, _, fold, draw, res = draw_results[di]
                di += 1
                if (key, fold, draw) != (meta["key"], f, j):
                    raise RuntimeError("job order mismatch")
                rows.append({"row_id": meta["row_id"], "rung": meta["rung"],
                             "dataset_id": meta["dataset_id"], "task_type": meta["task_type"],
                             "fold_seed": meta["fold_seed"], "fold": f, "draw": j, **res})
        pd.DataFrame([{c: _fmt(r.get(c)) for c in DRAW_COLUMNS} for r in rows],
                     columns=DRAW_COLUMNS).to_csv(meta["cache"], index=False)
        by_pool[meta["row_id"]] = _read_cache(meta["cache"])

    perms_cache: dict[int, Any] = {}
    all_draws, curves, summaries = [], [], []
    for meta in plans:
        rows = by_pool[meta["row_id"]]
        all_draws.extend(rows)
        n = meta["n_draws"]
        perms = perms_cache.setdefault(n, orderings(n, args.n_orderings, args.order_seed))
        for f in range(meta["n_folds"]):
            fr = sorted((r for r in rows if int(r["fold"]) == f), key=lambda r: int(r["draw"]))
            base = results.get((meta["key"], f), {})
            curve = best_of_k([r["sel_score"] for r in fr], [r.get("heldout") for r in fr],
                              [r.get("rec_sel") for r in fr],
                              [int(r.get("completion_tokens") or 0) for r in fr],
                              base.get("base_no_fe_heldout"), perms)
            ident = {"row_id": meta["row_id"], "rung": meta["rung"], "dataset_id": meta["dataset_id"],
                     "task_type": meta["task_type"], "fold_seed": meta["fold_seed"], "fold": f}
            curves.extend({**ident, **c} for c in curve)
            f2_tok = f2.get((meta["rung"], meta["dataset_id"], meta["fold_seed"], f))
            summaries.append({**ident, "pool_complete": meta.get("complete"),
                              **fold_summary(fr, curve, base, f2_tok)})

    pd.DataFrame([{c: _fmt(r.get(c)) for c in DRAW_COLUMNS} for r in all_draws],
                 columns=DRAW_COLUMNS).to_csv(out / "draws.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "curves.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "fold_summary.csv", index=False)
    info = {"pools": [m["row_id"] for m in plans], "refused": refused, "cpus": args.cpus,
            "validate_frame": args.validate_frame,
            "n_orderings": args.n_orderings, "order_seed": args.order_seed,
            "wall_s": round(time.monotonic() - t_start, 1),
            "env": {k: os.environ.get(k) for k in list(V2_ENV) + ["ROGUE_ONE_EXEC_COMPAT",
                                                                  "ROGUE_ONE_INF_GUARD"]},
            "git_sha": os.environ.get("ROGUE_ONE_GIT_SHA"), "scorer": "scripts/agent2027/f0_score.py"}
    try:
        import xgboost  # noqa: PLC0415

        info["versions"] = {"xgboost": xgboost.__version__, "pandas": pd.__version__}
    except Exception:  # noqa: BLE001
        pass
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    print(f"[f0_score] done: {len(plans)} pools in {info['wall_s']}s -> {out}", flush=True)
    return 0 if not refused else 3


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pools", nargs="+", required=True,
                    help="pool folders, or a root holding one folder per A17-1xx row")
    ap.add_argument("--rows", nargs="*", help="only these row ids (folder names)")
    ap.add_argument("--tasks", required=True, help="tasks root (classification/, regression/)")
    ap.add_argument("--meta", help="SYN-P meta folder (synp_*.json) for the recovery detector")
    ap.add_argument("--manifest", default=str(_REPO_ROOT / "manifests" / "agent2027" / "A17_F0.csv"))
    ap.add_argument("--f2-tokens", help="CSV rung,dataset_id,fold_seed,fold,tok_compl_total")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpus", type=int, default=1)
    ap.add_argument("--rescore", action="store_true", help="ignore cache/<row>.draws.csv")
    ap.add_argument("--n-draws", type=int, default=None, help="testing only; frozen at 128")
    ap.add_argument("--validate-frame", choices=VALIDATE_FRAMES, default="id_only",
                    help="df_features at validation: id_only (A17-T8, default) or augment "
                         "(id + seed columns; sensitivity, use a separate --out)")
    ap.add_argument("--n-orderings", type=int, default=N_ORDERINGS)
    ap.add_argument("--order-seed", type=int, default=ORDER_SEED)
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
