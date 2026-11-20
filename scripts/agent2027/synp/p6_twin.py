"""P6b: the exact anonymised twin of P6a (decision D8, option A with hidden rescaling).

Same rows, same row order, same ``id``, byte-identical ``labels.csv``, same
generator seed (nothing is regenerated: P6b is a *transformation* of P6a):

* columns renamed ``f01..fNN`` by the rank of the P6a name under Python
  ``sorted()`` (zero-padded), so the alphabetically sorted column list in the
  Extractor prompt has every column at the same position in both twins; the
  ``data.csv`` column order is P6a's;
* categorical levels renamed ``c01..cKK`` per column by the rank of the P6a level
  string under ``sorted()`` (so alphabetical ordinal codes are identical);
* a neutral ``context_and_goal`` from the same template, length within +-2%;
* if phi is a monomial, *every* numeric column is multiplied by a hidden positive
  constant (no shifts): this hides value ranges but keeps every rank and every
  scale-equivariant feature (Spearman with phi is unchanged).

The name map, level map and constants live only in the off-mount metadata.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.agent2027.synp import context_template as ctx
from scripts.agent2027.synp import tasks as T

FORMULA_NAMES = {
    "reynolds": ["reynolds", "reynolds number"],
    "ideal_gas": ["ideal gas", "gas law", "pv=nrt"],
    "cockcroft_gault": ["cockcroft", "gault", "creatinine clearance"],
    "bmi_like": ["bmi", "body mass index", "quetelet"],
}


def pad_width(k: int) -> int:
    return max(2, len(str(k)))


def name_map(columns: list[str]) -> dict[str, str]:
    """P6a name -> fNN by sorted rank."""
    w = pad_width(len(columns))
    return {c: f"f{r + 1:0{w}d}" for r, c in enumerate(sorted(columns))}


def level_map(levels: tuple[str, ...]) -> dict[str, str]:
    w = pad_width(len(levels))
    return {lv: f"c{r + 1:0{w}d}" for r, lv in enumerate(sorted(levels))}


def hidden_constants(columns: list[str], seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    out = {}
    for c in columns:
        v = float(np.exp(rng.uniform(np.log(0.05), np.log(50.0))))
        out[c] = float(f"{v:.6g}")
    return out


def make_twin(p6a: T.GeneratedTask, *, rescale: bool | None = None, master_seed: int = T.MASTER_SEED) -> T.GeneratedTask:
    """Build P6b from a generated P6a.  ``rescale=None`` means: rescale iff phi is
    a monomial (the D8 rule); pass True/False to force either branch."""
    mono = p6a.built.monomial
    do_rescale = (mono is not None) if rescale is None else bool(rescale)
    if do_rescale and mono is None:
        raise ValueError("rescaling a non-monomial phi changes its rank structure")
    feat_cols = [c for c in p6a.data.columns if c != "id"]
    nmap = name_map(feat_cols)
    lmaps = {c: level_map(p6a.levels[c]) for c in p6a.categorical}
    numeric = [c for c in feat_cols if c not in p6a.categorical]
    consts = hidden_constants(numeric, T.derive_seed(master_seed, "p6b", "scale")) if do_rescale else {}

    data = p6a.data.copy()
    for c, k in consts.items():
        data[c] = data[c].astype(float) * k
    for c, m in lmaps.items():
        data[c] = data[c].map(m)
    data = data.rename(columns=nmap)

    columns_neutral = [(nmap[c], c in p6a.categorical) for c in feat_cols]
    context = ctx.render_neutral_context(
        n_rows=len(data), columns=columns_neutral, task_type=p6a.task_type,
        target_length=len(p6a.context), classes=p6a.built.classes,
    )
    scale_factor = 1.0
    if do_rescale:
        for c, e in mono.items():
            scale_factor *= consts[c] ** e

    meta = copy.deepcopy(p6a.meta)
    p6a_phi_spec = copy.deepcopy(p6a.meta["phi"])
    phi_b = {"expr": p6a_phi_spec["expr"], "cols": {v: nmap[c] for v, c in p6a_phi_spec["cols"].items()},
             "consts": p6a_phi_spec.get("consts", {})}
    phi_b_vals = T.eval_phi(phi_b, data)
    meta.update({
        "task_id": "synp_p6b", "key": "p6b", "twin_of": "synp_p6a",
        "phi": phi_b, "phi_pandas_expr": T.pandas_expr(phi_b),
        "phi_p6a": p6a_phi_spec,
        "phi_scale_factor": scale_factor,  # phi_b = scale_factor * phi_p6a (monomial) else 1
        "name_map": nmap, "level_map": lmaps, "scale_constants": consts, "rescaled": do_rescale,
        "relevant_columns": [nmap[c] for c in p6a.meta["relevant_columns"]],
        "column_roles": {nmap[c]: r for c, r in p6a.meta["column_roles"].items()},
        "oracle_focus": T.oracle_focus(p6a_phi_spec, "p6b", colmap=nmap),
        "p6a_data_sha256": p6a.meta["data_sha256"],
        "p6a_phi_sha256": p6a.meta["phi_sha256"],  # "same phi": hash of phi on the P6a rows
        "phi_sha256": p6a.meta["phi_sha256"],
        "p6b_phi_sha256": T.phi_sha256(phi_b_vals),
        "data_sha256": T.sha256(T.data_csv_bytes(data)),
        "context_chars": len(context),
    })
    column_desc = [(nmap[c], ctx.NEUTRAL_CATEGORICAL if c in p6a.categorical else ctx.NEUTRAL_NUMERIC)
                   for c in feat_cols]
    return T.GeneratedTask(
        key="p6b", task_id="synp_p6b", task_type=p6a.task_type, data=data, labels=p6a.labels.copy(),
        phi=p6a.phi.copy(), context=context, meta=meta, column_desc=column_desc,
        categorical=tuple(nmap[c] for c in p6a.categorical),
        levels={nmap[c]: tuple(lmaps[c][lv] for lv in p6a.levels[c]) for c in p6a.categorical},
        built=p6a.built,
    )


# --------------------------------------------------------------------------- #
# A15-T2 checks, A15-T7 term list, A15-T8 static leak check
# --------------------------------------------------------------------------- #
def invert_twin(data_b: pd.DataFrame, meta_b: dict[str, Any]) -> pd.DataFrame:
    """Undo names, levels and scales: returns the P6a-equivalent data frame."""
    inv_n = {v: k for k, v in meta_b["name_map"].items()}
    out = data_b.rename(columns=inv_n).copy()
    for c, m in meta_b["level_map"].items():
        out[c] = out[c].map({v: k for k, v in m.items()})
    for c, k in meta_b["scale_constants"].items():
        out[c] = out[c].astype(float) / k
    return out


def verify_twin(dir_a: Path, dir_b: Path, meta_b: dict[str, Any]) -> dict[str, bool]:
    """The A15-T2 data checks on two task folders.  Returns check -> passed."""
    from rogueone.utils.train_test_splitter import deterministic_kfold_indices  # noqa: PLC0415
    import hashlib
    import yaml

    a, b = pd.read_csv(dir_a / "data.csv"), pd.read_csv(dir_b / "data.csv")
    la, lb = (dir_a / "labels.csv").read_bytes(), (dir_b / "labels.csv").read_bytes()
    inv = invert_twin(b, meta_b)[list(a.columns)]
    num = [c for c in a.columns if c != "id" and pd.api.types.is_numeric_dtype(a[c])]
    cat = [c for c in a.columns if c not in num and c != "id"]
    numeric_ok = all(
        np.allclose(inv[c].to_numpy(float), a[c].to_numpy(float), rtol=1e-9, atol=0.0) for c in num
    )
    ca = yaml.safe_load((dir_a / "config.yml").read_text(encoding="utf-8"))["context_and_goal"]
    cb = yaml.safe_load((dir_b / "config.yml").read_text(encoding="utf-8"))["context_and_goal"]
    folds_ok = all(
        [sorted(t) for _, t in deterministic_kfold_indices(len(a), 5, s)]
        == [sorted(t) for _, t in deterministic_kfold_indices(len(b), 5, s)]
        for s in (42, 4, 5)
    )
    phi_a = T.eval_phi(meta_b["phi_p6a"], a)
    phi_b = T.eval_phi(meta_b["phi"], b)
    return {
        "labels_bytes_equal": la == lb,
        "labels_sha256_equal": hashlib.sha256(la).hexdigest() == hashlib.sha256(lb).hexdigest(),
        "ids_identical": bool((a["id"].to_numpy() == b["id"].to_numpy()).all()),
        "column_order_kept": list(b.columns) == ["id"] + [meta_b["name_map"][c] for c in a.columns if c != "id"],
        "numeric_inverts": bool(numeric_ok),
        "categorical_equal": all((inv[c] == a[c]).all() for c in cat),
        "phi_hash_matches": T.phi_sha256(phi_a) == meta_b["p6a_phi_sha256"] == meta_b["phi_sha256"],
        "phi_scales_exactly": bool(np.allclose(phi_b, phi_a * meta_b["phi_scale_factor"], rtol=1e-9, atol=0.0)),
        "no_duplicate_rows": not b.drop(columns="id").duplicated().any(),
        "folds_identical": bool(folds_ok),
        "context_len_within_2pct": abs(len(cb) - len(ca)) <= 0.02 * len(ca),
    }


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^A-Za-z]+", name) if len(t) > 2]


GENERIC_TOKENS = {"the", "and", "per", "for", "with", "from", "mean", "total", "number", "value", "rate"}


def domain_terms(p6a: T.GeneratedTask, candidate: str) -> dict[str, list[str]]:  # noqa: C901
    """A15-T7 term list: strong (formula name and synonyms) and weak (P6a column
    names and tokens, level strings, unit/domain nouns of the P6a context)."""
    strong = list(FORMULA_NAMES.get(candidate, []))
    weak: set[str] = set()
    for c, d in p6a.column_desc:
        weak.add(c)
        weak.update(_tokens(c))
        weak.update(t.lower() for t in _tokens(d))
    for lv in p6a.levels.values():
        weak.update(lv)
    weak.update(t.lower() for t in _tokens(p6a.built.intro + " " + p6a.built.unit + " " + p6a.built.target))
    template_text = " ".join([ctx.CONTEXT_TEMPLATE, ctx.REGRESSION_GOAL, ctx.CLASSIFICATION_GOAL, ctx.NEUTRAL_INTRO,
                              ctx.NEUTRAL_NUMERIC, ctx.NEUTRAL_CATEGORICAL, ctx.NEUTRAL_TARGET, ctx.NEUTRAL_UNIT,
                              *ctx.NEUTRAL_FILLER]).lower()
    weak = {w for w in weak if w and w.lower() not in GENERIC_TOKENS and w.lower() not in {"ratio"}}
    # twin-invariant template words (e.g. "relative" in "relative scale") are not leaks
    const = sorted(w for w in weak if re.search(rf"\b{re.escape(w.lower())}\b", template_text))
    return {"strong": strong, "weak": sorted(weak - set(const)), "template_constant": const}


def mark_template_constant(terms: list[str], files: list[Path]) -> list[str]:
    """Terms that occur in the twin-invariant template text (agent role prompts etc.)."""
    texts = [p.read_text(encoding="utf-8", errors="ignore").lower() for p in files if p.exists()]
    return [t for t in terms if any(re.search(rf"\b{re.escape(t.lower())}\b", x) for x in texts)]


def _find(term: str, text: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", text, flags=re.IGNORECASE) is not None


def static_leak_check(terms: dict[str, list[str]], dir_a: Path, dir_b: Path, template_constant: set[str] = frozenset()) -> dict[str, list[str]]:
    """A15-T8: case-insensitive word-boundary grep of the term list over P6b's
    config.yml, data.csv header + categorical values and labels.csv header, and of
    the strong terms over P6a's context.  Template-constant terms are ignored."""
    import yaml

    all_terms = [t for t in terms["strong"] + terms["weak"] if t.lower() not in {x.lower() for x in template_constant}]
    cfg_b = (dir_b / "config.yml").read_text(encoding="utf-8")
    data_b = pd.read_csv(dir_b / "data.csv")
    pieces = [cfg_b, ",".join(data_b.columns), (dir_b / "labels.csv").read_text(encoding="utf-8").splitlines()[0]]
    for c in data_b.columns:
        if not pd.api.types.is_numeric_dtype(data_b[c]):
            pieces.append(" ".join(map(str, data_b[c].unique())))
    blob = "\n".join(pieces)
    hits_b = [t for t in all_terms if _find(t, blob)]
    ctx_a = yaml.safe_load((dir_a / "config.yml").read_text(encoding="utf-8"))["context_and_goal"]
    hits_a = [t for t in terms["strong"] if _find(t, ctx_a)]
    return {"p6b_hits": hits_b, "p6a_context_strong_hits": hits_a}
