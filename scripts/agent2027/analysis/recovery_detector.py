"""Minimal recovery detector for the SYN-P tasks (AGENT 2027, E8a).

Definition (01 section 4.6, directions section 2.3 -- fixed in advance, CPU only):

* A feature is **recovered** when, on the fold's *training rows*, it has
  ``|Spearman| >= 0.95`` with the planted phi, or -- for the rule and threshold
  families (P3, P5, P7) -- a normalised mutual information ``>= 0.9`` with phi.
* **Not counted:** raw columns, monotone transforms of a single raw column, and
  features produced by the deterministic fallback (``Description`` starts with
  ``"Fallback"``, or an explicit ``is_fallback`` flag).
* ``rec_any``: some active, counted feature recovers phi at *any* iteration.
* ``rec_sel``: the same, restricted to the iteration that inner CV selects
  (the primary outcome).
* ``iter_first_rec``: first iteration at which a counted feature recovers phi.

Features are re-computed by replaying every attribute's stored pandas command on
the fold's training rows through
``rogueone.utils.fold_safe_features.materialize_fold_safe_features`` (the same
numbers the evaluator sees).

**Detector amendment (pre-registration Part III.2 / Amendment 2, 2026-10-19).**
Formula mode (agent runs) applies by default the same raw-column exclusions as
``--features-table`` mode: the replayed feature table is scored with
:func:`evaluate_feature_table` (raw-column names, ``monotone_rho`` >= 0.999 with one
raw column, and ``no_gain_over_raw``: a Spearman hit must beat its best single raw
column by ``RAW_GAIN_MARGIN``), fallback features excluded as before.  Before
2026-10-19 the amendment only ran in ``--features-table`` mode, so the P0 null task
read rec_sel 4/5 on A05.  ``amended=False`` (CLI ``--registered-formula``) keeps the
registered pre-amendment formula-mode rule (exact weakly-monotone check on numeric
raw columns only) for reproducing earlier read-outs.  ``--features-table`` mode is
unchanged (the A08 screen reproduces exactly).

The module is self-contained (numpy/pandas/scipy, plus lazy imports of the repo
replay helper) so it can be mounted read-only into the analysis container.  The
CPU pairwise screen imports its criterion functions from here so both use the
identical definition.  The full detector (E11b) adds the phi_wrong detector and
the P5 LW*LD (textbook prior) detector.

``--features-table`` mode (A08-T8, brute-force comparators OpenFE / AutoFeat /
raw): instead of replaying pandas commands, it reads per-fold *training-row*
feature tables dumped by ``paper/scripts/classicfe_arm.py --dump-features``
(``<dataset_id>_<arm>_fold<k>.csv.gz``: an ``id`` column plus one column per
generated feature; any CSV(.gz) of that shape works), joins them to the task's
``data.csv`` by ``id`` and applies the same phi criteria.  Single-column
exclusion is value-based: a column is excluded when its |Spearman| with a single
raw column is ``>= 0.999`` (raw categoricals enter as the sorted-level codes
classicfe feeds to OpenFE/AutoFeat), in addition to raw-column names and the
exact weakly-monotone check of formula mode.  A Spearman-criterion hit is also
excluded (``no_gain_over_raw:<col>``) unless its |Spearman| with phi exceeds its
best single-raw-column |Spearman| by ``RAW_GAIN_MARGIN`` (0.01).  Every generated feature is passed
to XGBoost by the brute-force arms, so ``rec_sel = rec_any`` by definition
(A08 section 2).
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

SPEARMAN_THRESHOLD = 0.95
NMI_THRESHOLD = 0.9
NMI_BINS = 10
_MONOTONE_PREFILTER = 0.8
_MIN_VALID_FRACTION = 0.9
MONOTONE_RHO_THRESHOLD = 0.999  # --features-table: value-based single-column rule
# --features-table: a Spearman-criterion hit must track phi better than its best
# single raw column by this margin (A08 screen amendment 2026-10-13: on synp_p0,
# phi = log(one column), near-monotone features at |rho| 0.957-0.999 passed the
# 0.999 rule; on p0 no feature can beat the raw column, so p0 = 0 by construction).
RAW_GAIN_MARGIN = 0.01

FAMILY_SPEARMAN = "spearman"
FAMILY_RULE = "rule"  # Spearman OR NMI (P3, P5, P7)


# --------------------------------------------------------------------------- #
# Criterion primitives (also used by the pairwise CPU screen)
# --------------------------------------------------------------------------- #
def _clean_pair(f, phi):
    f = np.asarray(f, dtype=float)
    phi = np.asarray(phi, dtype=float)
    ok = np.isfinite(f) & np.isfinite(phi)
    if ok.mean() < _MIN_VALID_FRACTION or ok.sum() < 10:
        return None
    return f[ok], phi[ok]


def spearman_abs(f, phi) -> float:
    """|Spearman rho| (average ranks for ties); 0.0 for degenerate input."""
    pair = _clean_pair(f, phi)
    if pair is None:
        return 0.0
    a, b = pair
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = math.sqrt(float(ra @ ra) * float(rb @ rb))
    if den == 0.0:
        return 0.0
    return abs(float(ra @ rb) / den)


def _discretise(x: np.ndarray, bins: int = NMI_BINS) -> np.ndarray:
    """Integer codes: low-cardinality columns keep their levels, others are cut
    into ``bins`` quantile bins (duplicate edges dropped)."""
    uniq = np.unique(x)
    if uniq.size <= bins:
        return np.searchsorted(uniq, x)
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    return np.searchsorted(edges, x, side="right")


def _entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def normalized_mi(f, phi, bins: int = NMI_BINS) -> float:
    """I(f; phi) / mean(H(f), H(phi)) (natural log, arithmetic-mean
    normalisation as scikit-learn's default); both sides discretised with
    :func:`_discretise`.  0.0 for degenerate input."""
    pair = _clean_pair(f, phi)
    if pair is None:
        return 0.0
    a, b = pair
    ca, cb = _discretise(a, bins), _discretise(b, bins)
    na, nb = int(ca.max()) + 1, int(cb.max()) + 1
    joint = np.bincount(ca * nb + cb, minlength=na * nb).reshape(na, nb).astype(float)
    ha, hb = _entropy(joint.sum(1)), _entropy(joint.sum(0))
    if ha == 0.0 or hb == 0.0:
        return 0.0
    mi = ha + hb - _entropy(joint.ravel())
    return float(max(0.0, mi) / (0.5 * (ha + hb)))


def meets_criterion(f, phi, family: str = FAMILY_SPEARMAN) -> tuple[bool, float, float]:
    """Return ``(recovered, |rho|, nmi)``; NMI only counts for the rule family."""
    rho = spearman_abs(f, phi)
    if family == FAMILY_RULE:
        nmi = normalized_mi(f, phi)
        return (rho >= SPEARMAN_THRESHOLD or nmi >= NMI_THRESHOLD), rho, nmi
    return rho >= SPEARMAN_THRESHOLD, rho, float("nan")


# --------------------------------------------------------------------------- #
# Exclusions: raw columns and single-column monotone transforms
# --------------------------------------------------------------------------- #
def _is_weakly_monotone_function_of(f: np.ndarray, c: np.ndarray) -> bool:
    """True iff ``f`` is a (weakly) monotone single-valued function of ``c``."""
    ok = np.isfinite(f) & np.isfinite(c)
    if ok.sum() < 3:
        return False
    f, c = f[ok], c[ok]
    order = np.argsort(c, kind="stable")
    cs, fs = c[order], f[order]
    gid = np.cumsum(np.concatenate(([True], cs[1:] != cs[:-1]))) - 1
    tol = 1e-9 * max(1.0, float(np.max(np.abs(fs))))
    gmin = np.full(gid[-1] + 1, np.inf)
    gmax = np.full(gid[-1] + 1, -np.inf)
    np.minimum.at(gmin, gid, fs)
    np.maximum.at(gmax, gid, fs)
    if np.any(gmax - gmin > tol):
        return False
    d = np.diff(gmin)
    return bool(np.all(d >= -tol) or np.all(d <= tol))


def single_column_monotone_source(f, raw_numeric: pd.DataFrame) -> str | None:
    """Name of a raw numeric column of which ``f`` is a monotone transform
    (identity included), else None.  Non-monotone single-column functions
    (e.g. sin of hours) are *not* excluded."""
    f = np.asarray(f, dtype=float)
    for col in raw_numeric.columns:
        c = raw_numeric[col].to_numpy(dtype=float)
        if spearman_abs(f, c) < _MONOTONE_PREFILTER:
            continue
        if _is_weakly_monotone_function_of(f, c):
            return str(col)
    return None


@dataclass(frozen=True)
class TaskDetector:
    family: str = FAMILY_SPEARMAN

    @classmethod
    def from_meta(cls, meta: Mapping[str, Any]) -> "TaskDetector":
        return cls(family=meta.get("detector", {}).get("family", FAMILY_SPEARMAN))


def load_phi(meta_dir, task_id: str, df_raw: pd.DataFrame):
    """Recompute phi on ``df_raw`` from ``synp_meta/<task_id>.json``.

    Imported lazily: the generator lives under ``scripts/agent2027/synp``,
    which is kept out of agent images."""
    meta = json.loads((Path(meta_dir) / f"{task_id}.json").read_text(encoding="utf-8"))
    try:
        from scripts.agent2027.synp.tasks import phi_from_meta  # noqa: PLC0415
    except ImportError:  # same evaluator as synp.tasks.eval_phi
        phi_from_meta = _phi_from_meta_local
    return phi_from_meta(meta, df_raw), TaskDetector.from_meta(meta)


def _phi_from_meta_local(meta: Mapping[str, Any], df: pd.DataFrame) -> np.ndarray:
    spec = meta["phi"]
    ns: dict[str, Any] = {"np": np}
    for var, col in spec["cols"].items():
        ns[var] = df[col].to_numpy()
    ns.update(spec.get("consts", {}))
    out = eval(spec["expr"], {"__builtins__": {}}, ns)  # noqa: S307 - trusted local metadata
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- #
# Episode detection
# --------------------------------------------------------------------------- #
def attribute_is_fallback(row: Mapping[str, Any]) -> bool:
    for key in ("is_fallback", "Is Fallback", "Fallback"):
        val = row.get(key)
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            return bool(val)
    desc = row.get("Description", row.get("description", ""))
    return isinstance(desc, str) and desc.lstrip().lower().startswith("fallback")


@dataclass
class EpisodeResult:
    rec_any: int
    rec_sel: int
    iter_first_rec: int | None
    selected_iter: int | None
    per_feature: pd.DataFrame = field(repr=False)
    hits_by_iter: dict[int, list[str]] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {"rec_any": self.rec_any, "rec_sel": self.rec_sel,
                "iter_first_rec": self.iter_first_rec}


_COLS = ["feature", "rho", "nmi", "recovered", "excluded", "counted"]


def evaluate_features(
    features: pd.DataFrame,
    phi_train: np.ndarray,
    raw_train: pd.DataFrame,
    detector: TaskDetector,
    *,
    fallback: Collection[str] = (),
    names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Score columns of ``features`` (training rows) against phi.

    ``recovered`` is the pure criterion; ``counted`` additionally requires that
    the feature is not a fallback feature and not a raw/monotone single-column
    transform (``excluded`` says why)."""
    raw_numeric = raw_train.select_dtypes(include=[np.number]).drop(
        columns=[c for c in ("id",) if c in raw_train.columns]
    )
    fb = set(fallback)
    rows = []
    for name in names if names is not None else list(features.columns):
        if name == "id":
            continue
        excluded = "fallback" if name in fb else ""
        vals = None
        try:
            vals = pd.to_numeric(features[name], errors="raise").to_numpy(dtype=float)
        except Exception:  # noqa: BLE001
            excluded = excluded or "non_numeric"
        rho, nmi, recovered = 0.0, float("nan"), False
        if vals is not None:
            recovered, rho, nmi = meets_criterion(vals, phi_train, detector.family)
            if recovered and not excluded:
                src = single_column_monotone_source(vals, raw_numeric)
                if src is not None:
                    excluded = f"raw_or_monotone:{src}"
        rows.append({"feature": name, "rho": rho, "nmi": nmi, "recovered": bool(recovered),
                     "excluded": excluded, "counted": bool(recovered and not excluded)})
    return pd.DataFrame(rows, columns=_COLS)


def detect_episode(
    df_raw: pd.DataFrame,
    attributes: pd.DataFrame,
    train_mask: np.ndarray,
    phi: np.ndarray,
    detector: TaskDetector,
    active_by_iter: Mapping[int, Sequence[str]],
    selected_iter: int | None,
    *,
    fallback: Collection[str] | None = None,
    amended: bool = True,
) -> EpisodeResult:
    """Detect recovery for one fold-episode.

    Args:
        df_raw: raw table (``id`` + raw columns, no target), all rows.
        attributes: the run's ``df_attributes_explanations`` (``Attribute``,
            ``Pandas Command``; ``Description``/``is_fallback`` flag fallbacks).
        train_mask: boolean mask over ``df_raw`` = this fold's training rows.
        phi: planted feature over all rows of ``df_raw`` (same order).
        active_by_iter: iteration -> attribute names active in that iteration.
        selected_iter: inner-CV-selected iteration (None -> rec_sel = 0).
        fallback: extra attribute names to ignore.
        amended: apply the Part III.2 raw-column exclusions (``raw:``,
            ``monotone_rho:``, ``no_gain_over_raw:``) via
            :func:`evaluate_feature_table` (default).  ``False`` reproduces the
            registered pre-amendment rule.
    """
    from rogueone.utils.fold_safe_features import (  # noqa: PLC0415
        materialize_fold_safe_features,
    )

    train_mask = np.asarray(train_mask, dtype=bool)
    fb = set(fallback or ())
    for _, row in attributes.iterrows():
        if attribute_is_fallback(row.to_dict()):
            fb.add(row["Attribute"])
    feats = materialize_fold_safe_features(df_raw, attributes, train_mask)
    names = list(dict.fromkeys(a for a in attributes["Attribute"] if a in feats.columns and a != "id"))
    phi_train = np.asarray(phi, dtype=float)[train_mask]
    raw_train = df_raw[train_mask].reset_index(drop=True)
    if amended:
        pf = evaluate_feature_table(feats[names].reset_index(drop=True), phi_train, raw_train, detector)
        is_fb = pf["feature"].isin(fb)
        pf.loc[is_fb, "excluded"] = "fallback"
        pf["counted"] = pf["recovered"].astype(bool) & (pf["excluded"] == "")
        per_feature = pf.set_index("feature")
    else:
        per_feature = evaluate_features(
            feats, phi_train, raw_train, detector, fallback=fb, names=names,
        ).set_index("feature")

    hits: dict[int, list[str]] = {}
    for it, names in sorted(active_by_iter.items()):
        h = [n for n in names if n in per_feature.index and per_feature.loc[n, "counted"]]
        if h:
            hits[int(it)] = h
    return EpisodeResult(
        rec_any=int(bool(hits)),
        rec_sel=int(selected_iter is not None and int(selected_iter) in hits),
        iter_first_rec=min(hits) if hits else None,
        selected_iter=selected_iter,
        per_feature=per_feature.reset_index(),
        hits_by_iter=hits,
    )


def train_mask_for_fold(
    df_raw: pd.DataFrame, labels: pd.DataFrame, fold_seed: int, fold_id: int,
    *, k_folds: int = 5, stratified: bool = False,
) -> np.ndarray:
    """Training-row mask of outer fold ``fold_id`` (same splitter call as
    ``rogue_one_main.py``)."""
    from rogueone.utils.train_test_splitter import (  # noqa: PLC0415
        compute_entity_duplicate_groups,
        deterministic_kfold_indices,
    )

    ids = df_raw["id"].tolist()
    strat = None
    if stratified:
        strat = df_raw[["id"]].merge(labels[["id", "target"]], on="id", how="left")["target"].to_numpy()
    groups = compute_entity_duplicate_groups(df_raw, ids, exclude_cols=("id",))
    folds = deterministic_kfold_indices(
        n_rows=len(df_raw), n_splits=k_folds, random_state=fold_seed,
        stratify_labels=strat, dedup_groups=groups,
    )
    mask = np.zeros(len(df_raw), dtype=bool)
    mask[folds[fold_id][0]] = True
    return mask


def run_episodes_file(episodes_json, tasks_root, meta_dir, *, amended: bool = True) -> pd.DataFrame:
    """Score fold-episodes described in a JSON list (hook for E11b).

    Episode: ``{"episode_id", "task_id", "task_type", "fold_seed", "fold_id",
    "stratified", "attributes": [{"Attribute", "Pandas Command", "Description"?}],
    "active_by_iter": {"0": [...]}, "selected_iter"}``.  Failed or aborted
    episodes must be added by the caller as ITT zero rows."""
    out = []
    for ep in json.loads(Path(episodes_json).read_text(encoding="utf-8")):
        task_dir = Path(tasks_root) / ep["task_type"] / ep["task_id"]
        df_raw = pd.read_csv(task_dir / "data.csv")
        labels = pd.read_csv(task_dir / "labels.csv")
        phi, det = load_phi(meta_dir, ep["task_id"], df_raw)
        mask = train_mask_for_fold(df_raw, labels, int(ep["fold_seed"]), int(ep["fold_id"]),
                                   stratified=bool(ep.get("stratified", False)))
        res = detect_episode(df_raw, pd.DataFrame(ep["attributes"]), mask, phi, det,
                             {int(k): v for k, v in ep["active_by_iter"].items()},
                             ep.get("selected_iter"), amended=amended)
        out.append({"episode_id": ep.get("episode_id"), "task_id": ep["task_id"],
                    "fold_seed": ep["fold_seed"], "fold_id": ep["fold_id"], **res.as_row()})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# --features-table mode (A08-T8): brute-force comparators' dumped feature tables
# --------------------------------------------------------------------------- #
TABLE_NAME_RE = re.compile(r"^(?P<dataset>.+)_(?P<arm>[^_]+)_fold(?P<fold>\d+)\.csv(?:\.gz)?$")
_SEED_DIR_RE = re.compile(r"^f(?P<seed>\d+)$")

EPISODE_COLUMNS = [
    "dataset_id", "arm", "fold_seed", "fold", "status", "rec_any", "rec_sel",
    "n_features", "n_excluded_monotone", "n_eligible", "n_recovering", "first_rec_rank",
    "best_feature", "best_score", "best_rho", "best_nmi", "best_counted", "n_train_rows",
    "family", "source", "error",
]


def discover_feature_tables(root, default_fold_seed: int | None = None) -> list[dict[str, Any]]:
    """Find ``<dataset_id>_<arm>_fold<k>.csv[.gz]`` under ``root`` (a file or a
    directory, searched recursively).  The fold seed comes from the nearest
    ``f<seed>`` directory between the file and ``root`` (e.g.
    ``<root>/f42/features/x.csv.gz``; ``root`` itself may be ``f42``), else
    ``default_fold_seed``."""
    root = Path(root)
    files = [root] if root.is_file() else sorted(
        p for p in root.rglob("*") if p.is_file() and TABLE_NAME_RE.match(p.name))
    out = []
    for p in files:
        m = TABLE_NAME_RE.match(p.name)
        if m is None:
            raise ValueError(f"not a feature-table file name: {p.name}")
        seed = default_fold_seed
        for parent in p.parents:
            sm = _SEED_DIR_RE.match(parent.name)
            if sm:
                seed = int(sm.group("seed"))
                break
            if parent == root:
                break
        out.append({"path": p, "dataset_id": m.group("dataset"), "arm": m.group("arm"),
                    "fold": int(m.group("fold")), "fold_seed": seed})
    return out


def encode_raw_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Raw columns (``id`` dropped) as floats, the way ``classicfe_arm.prep``
    hands them to OpenFE/AutoFeat: non-numeric columns -> sorted-level codes."""
    out = {}
    for col in df_raw.columns:
        if col == "id":
            continue
        s = df_raw[col]
        if pd.api.types.is_bool_dtype(s):
            out[col] = s.astype(float)
        elif pd.api.types.is_numeric_dtype(s):
            out[col] = s.astype(float)
        else:
            levels = sorted(s.dropna().astype(str).unique().tolist())
            codes = {v: float(i) for i, v in enumerate(levels)}
            out[col] = s.map(lambda v: np.nan if pd.isna(v) else codes[str(v)]).astype(float)
    return pd.DataFrame(out, index=df_raw.index)


class _RawRankIndex:
    """|Spearman| of a vector against every raw column (vectorised when both
    sides are fully finite, :func:`spearman_abs` otherwise)."""

    def __init__(self, raw_numeric: pd.DataFrame):
        self.cols = [str(c) for c in raw_numeric.columns]
        self.raw = raw_numeric.to_numpy(dtype=float)
        finite = np.isfinite(self.raw).all(axis=0)
        self.fast = np.flatnonzero(finite)
        self.slow = np.flatnonzero(~finite)
        r = np.apply_along_axis(rankdata, 0, self.raw[:, self.fast]) if self.fast.size else \
            np.zeros((self.raw.shape[0], 0))
        r = r - r.mean(axis=0) if r.size else r
        nrm = np.sqrt((r * r).sum(axis=0))
        nrm[nrm == 0] = np.inf  # constant raw column -> rho 0
        self.r = r / nrm

    def max_abs(self, f: np.ndarray) -> tuple[str | None, float]:
        if not self.cols:
            return None, 0.0
        rho = np.zeros(len(self.cols))
        slow = np.arange(len(self.cols))
        if np.isfinite(f).all() and len(f) >= 10:
            slow = self.slow
            rf = rankdata(f)
            rf = rf - rf.mean()
            n = math.sqrt(float(rf @ rf))
            if n > 0 and self.fast.size:
                rho[self.fast] = np.abs(self.r.T @ rf) / n
        for j in slow:
            rho[j] = spearman_abs(f, self.raw[:, j])
        j = int(np.argmax(rho))
        return self.cols[j], float(rho[j])


def evaluate_feature_table(
    features: pd.DataFrame,
    phi_train: np.ndarray,
    raw_train: pd.DataFrame,
    detector: TaskDetector,
    *,
    monotone_rho: float = MONOTONE_RHO_THRESHOLD,
) -> pd.DataFrame:
    """Score every non-``id`` column of a training-row feature table (rows
    aligned with ``raw_train`` and ``phi_train``).

    Exclusions (``excluded``, first reason wins): ``non_numeric``; for columns
    meeting the criterion, ``raw_or_monotone:<col>`` (the exact weakly-monotone
    check of formula mode); ``raw:<col>`` (named like a raw column);
    ``monotone_rho:<col>`` (|Spearman| >= ``monotone_rho`` with one raw column);
    ``no_gain_over_raw:<col>`` (a Spearman hit no better than one raw column).
    ``rank`` is the 1-based column position in the table."""
    raw_numeric = encode_raw_columns(raw_train)
    index = _RawRankIndex(raw_numeric)
    raw_names = set(map(str, raw_numeric.columns))
    names = [c for c in features.columns if c != "id"]
    pf = evaluate_features(features, phi_train, raw_numeric, detector, names=names)
    pf.insert(1, "rank", np.arange(1, len(pf) + 1))
    max_col: list[str | None] = []
    max_rho: list[float] = []
    for k, name in enumerate(names):
        col, rho = None, float("nan")
        if pf.at[k, "excluded"] != "non_numeric":
            vals = pd.to_numeric(features[name], errors="coerce").to_numpy(dtype=float)
            col, rho = index.max_abs(vals)
            if not pf.at[k, "excluded"]:
                if str(name) in raw_names:
                    pf.at[k, "excluded"] = f"raw:{name}"
                elif rho >= monotone_rho:
                    pf.at[k, "excluded"] = f"monotone_rho:{col}"
                elif (pf.at[k, "recovered"] and pd.notna(pf.at[k, "rho"])
                      and pf.at[k, "rho"] >= SPEARMAN_THRESHOLD
                      and pf.at[k, "rho"] - rho < RAW_GAIN_MARGIN):
                    pf.at[k, "excluded"] = f"no_gain_over_raw:{col}"
        max_col.append(col)
        max_rho.append(rho)
    pf["raw_rho_max"] = max_rho
    pf["raw_rho_col"] = max_col
    pf["counted"] = pf["recovered"].astype(bool) & (pf["excluded"] == "")
    return pf


def _score(row: Mapping[str, Any]) -> float:
    nmi = row["nmi"]
    return float(max(row["rho"], 0.0 if pd.isna(nmi) else nmi))


def score_feature_table(
    table: pd.DataFrame, df_raw: pd.DataFrame, phi: np.ndarray, detector: TaskDetector,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """One episode: join ``table`` (``id`` + features on training rows) to the
    raw table by ``id``; return (episode summary, per-feature frame).

    ``best_*`` describes the highest-scoring counted column if any, else the
    highest-scoring eligible (non-excluded) column; score = max(|rho|, NMI)."""
    if "id" not in table.columns:
        raise ValueError("feature table has no 'id' column")
    pos = pd.Series(np.arange(len(df_raw)), index=df_raw["id"].to_numpy())
    if not pos.index.is_unique:
        raise ValueError("data.csv ids are not unique")
    missing = ~table["id"].isin(pos.index)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} feature-table ids not in data.csv")
    if table["id"].duplicated().any():
        raise ValueError("duplicate ids in feature table")
    rows = pos.loc[table["id"].to_numpy()].to_numpy()
    raw_train = df_raw.iloc[rows].reset_index(drop=True)
    phi_train = np.asarray(phi, dtype=float)[rows]
    pf = evaluate_feature_table(table.reset_index(drop=True), phi_train, raw_train, detector)
    excl = pf["excluded"].astype(str)
    eligible = pf[excl == ""]
    counted = pf[pf["counted"]]
    pool = counted if len(counted) else eligible
    best = pool.loc[pool.apply(_score, axis=1).idxmax()] if len(pool) else None
    rec = int(len(counted) > 0)
    summary = {
        "rec_any": rec,
        "rec_sel": rec,  # brute force: every generated feature reaches XGBoost
        "n_features": int(len(pf)),
        "n_excluded_monotone": int(excl.str.match(r"^(raw|monotone_rho|raw_or_monotone|no_gain_over_raw):").sum()),
        "n_eligible": int(len(eligible)),
        "n_recovering": int(len(counted)),
        "first_rec_rank": int(counted["rank"].min()) if len(counted) else None,
        "best_feature": None if best is None else str(best["feature"]),
        "best_score": None if best is None else _score(best),
        "best_rho": None if best is None else float(best["rho"]),
        "best_nmi": None if best is None or pd.isna(best["nmi"]) else float(best["nmi"]),
        "best_counted": None if best is None else bool(best["counted"]),
        "n_train_rows": int(len(table)),
        "family": detector.family,
    }
    return summary, pf


def _find_task_dir(tasks_root: Path, dataset_id: str, meta: Mapping[str, Any]) -> Path:
    d = tasks_root / str(meta.get("task_type", "")) / dataset_id
    if (d / "data.csv").exists():
        return d
    hits = sorted(tasks_root.glob(f"*/{dataset_id}/data.csv")) + sorted(tasks_root.glob(f"{dataset_id}/data.csv"))
    if not hits:
        raise FileNotFoundError(f"no data.csv for {dataset_id} under {tasks_root}")
    return hits[0].parent


def run_feature_tables(
    features_root, tasks_root, meta_dir, *, default_fold_seed: int | None = None,
    datasets: Collection[str] | None = None, arms: Collection[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every dumped feature table under ``features_root``.

    Returns (episodes, per_feature).  A table that cannot be scored gets a
    ``status = "error"`` row with ``rec_any = rec_sel = 0`` (ITT).  Missing
    cells (no file) are not invented here; the caller adds them."""
    tasks_root = Path(tasks_root)
    cache: dict[str, tuple[pd.DataFrame, np.ndarray, TaskDetector]] = {}
    episodes, per_feature = [], []
    for t in discover_feature_tables(features_root, default_fold_seed):
        if datasets and t["dataset_id"] not in datasets:
            continue
        if arms and t["arm"] not in arms:
            continue
        key = {k: t[k] for k in ("dataset_id", "arm", "fold_seed", "fold")}
        row: dict[str, Any] = dict.fromkeys(EPISODE_COLUMNS)
        row.update(key, source=str(t["path"]), status="ok", error="")
        try:
            ds = t["dataset_id"]
            if ds not in cache:
                meta = json.loads((Path(meta_dir) / f"{ds}.json").read_text(encoding="utf-8"))
                df_raw = pd.read_csv(_find_task_dir(tasks_root, ds, meta) / "data.csv")
                phi, det = load_phi(meta_dir, ds, df_raw)
                cache[ds] = (df_raw, phi, det)
            df_raw, phi, det = cache[ds]
            summary, pf = score_feature_table(pd.read_csv(t["path"]), df_raw, phi, det)
            row.update(summary)
            per_feature.append(pf.assign(**key))
        except Exception as exc:  # noqa: BLE001 - recorded, scored as ITT zero
            row.update(status="error", error=f"{type(exc).__name__}: {exc}", rec_any=0, rec_sel=0)
        episodes.append(row)
    ep = pd.DataFrame(episodes, columns=EPISODE_COLUMNS)
    if len(ep):
        ep = ep.sort_values(["dataset_id", "arm", "fold_seed", "fold"], na_position="last",
                            kind="stable").reset_index(drop=True)
    pf_all = pd.concat(per_feature, ignore_index=True) if per_feature else pd.DataFrame()
    return ep, pf_all


def write_feature_table_outputs(out_dir, episodes: pd.DataFrame, per_feature: pd.DataFrame) -> None:
    """``episodes.csv``, ``episodes.json`` (list of records) and
    ``per_feature.csv.gz`` (one row per scored column) in ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(out / "episodes.csv", index=False)
    recs = json.loads(episodes.to_json(orient="records"))
    (out / "episodes.json").write_text(json.dumps(recs, indent=1), encoding="utf-8")
    if len(per_feature):
        lead = ["dataset_id", "arm", "fold_seed", "fold"]
        per_feature[lead + [c for c in per_feature.columns if c not in lead]].to_csv(
            out / "per_feature.csv.gz", index=False)


def _ensure_repo_on_path() -> None:
    """Run as a script, only this directory is on sys.path; the lazy
    ``scripts.agent2027`` / ``rogueone`` imports need the repository root."""
    root = str(Path(__file__).resolve().parents[3])
    if root not in sys.path:
        sys.path.insert(0, root)


def main(argv=None) -> None:  # pragma: no cover - thin CLI
    import argparse

    _ensure_repo_on_path()
    ap = argparse.ArgumentParser(description="SYN-P minimal recovery detector")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--episodes", help="formula mode: JSON list of fold-episodes")
    mode.add_argument("--features-table", metavar="IN",
                      help="table mode: one <dataset>_<arm>_fold<k>.csv[.gz] or a directory searched "
                           "recursively; fold seed from f<seed>/ directories")
    ap.add_argument("--tasks-root", "--tasks", dest="tasks_root", required=True)
    ap.add_argument("--meta-dir", "--meta", dest="meta_dir", required=True)
    ap.add_argument("--out", required=True,
                    help="formula mode: CSV file; table mode: directory (episodes.csv, "
                         "episodes.json, per_feature.csv.gz)")
    ap.add_argument("--fold-seed", type=int, default=None,
                    help="table mode: seed for files outside any f<seed>/ directory")
    ap.add_argument("--datasets", nargs="*", default=None, help="table mode: only these dataset ids")
    ap.add_argument("--arms", nargs="*", default=None, help="table mode: only these arms")
    ap.add_argument("--registered-formula", action="store_true",
                    help="formula mode: registered pre-amendment rule (no Part III.2 raw-column "
                         "exclusions); only for reproducing earlier read-outs")
    a = ap.parse_args(argv)
    if a.episodes:
        run_episodes_file(a.episodes, a.tasks_root, a.meta_dir,
                          amended=not a.registered_formula).to_csv(a.out, index=False)
        return
    ep, pf = run_feature_tables(a.features_table, a.tasks_root, a.meta_dir,
                                default_fold_seed=a.fold_seed, datasets=a.datasets, arms=a.arms)
    write_feature_table_outputs(a.out, ep, pf)
    n_err = int((ep["status"] != "ok").sum()) if len(ep) else 0
    print(f"{len(ep)} episodes, rec_any={int(ep['rec_any'].sum()) if len(ep) else 0}, "
          f"errors={n_err} -> {a.out}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
