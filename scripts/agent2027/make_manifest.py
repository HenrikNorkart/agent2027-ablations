#!/usr/bin/env python3
"""Ablation manifests and session files for the AGENT 2027 study (E1c, 01 §5.3).

Three modes (run on the developer machine, never on the CAIR host)::

    # 1. ablation manifest from a row spec (default spec path:
    #    scripts/agent2027/manifest_specs/<AID>.yaml)
    python scripts/agent2027/make_manifest.py --ablation A02
        -> manifests/agent2027/A02_M4.csv

    # 2. session file: rows picked from the committed ablation manifests, plus
    #    session_id and a random order_key (RNG seeded from the session ID)
    python scripts/agent2027/make_manifest.py --plan-session S1013a --rows A02-001 … A02-024
        -> manifests/agent2027/sessions/S1013a.csv  (+ S1013a.plan.txt, the seed and order)

    # 3. validate existing manifests / session files
    python scripts/agent2027/make_manifest.py --check manifests/agent2027/A02_M4.csv

Every write validates every row (01 §7 R7): header in R7 order, unique ``row_id``
(``<AID>-<nnn>``), known ``task_type``, integer seeds, ``env_overrides`` a flat JSON
object of registered R3/R4 flags with plain string values, ``cap_hours`` > 0.

HARD RULE: ``covtype`` is excluded from every AGENT 2027 study. Any row whose
``dataset_id`` is covtype is an error, in every mode, and no dataset set defined here
contains it.

Row ranges accepted by ``--rows``: ``A02-001 … A02-024`` (separate tokens, ``…`` or
``...``), ``A02-001…A02-024``, ``A10-001…014`` (abbreviated end), ``A02-001..024``,
or single IDs. Ranges expand only to rows that exist in the manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO_ROOT / "manifests" / "agent2027"
SESSIONS_DIR = MANIFEST_DIR / "sessions"
SPEC_DIR = Path(__file__).resolve().parent / "manifest_specs"

#: 01 §7 R7, in order.
R7_HEADER: list[str] = [
    "row_id", "ablation_id", "arm_id", "dataset_id", "task_type", "fold_seed",
    "llm_seed", "replicate", "is_sentinel", "session_id", "order_key", "image_tag",
    "rung_S", "rung_E", "rung_T", "port_S", "port_E", "port_T", "env_overrides",
    "cap_hours", "expected_u", "notes",
]

#: 01 §7 R1 short IDs (manifest filename ``<AID>_<short>.csv``, hyphens verbatim).
ABLATION_SHORT: dict[str, str] = {
    "A01": "W0", "A02": "M4", "A03": "M1", "A04": "CAP-0", "A05": "M0", "A06": "M0-S",
    "A07": "HC1", "A08": "M5", "A09": "RT3", "A10": "LF1", "A11": "LF2-L0", "A12": "FX",
    "A13": "KT1-Off", "A14": "KT3-P7", "A15": "KT2-min", "A16": "RT2", "A17": "CAP-1",
    "A18": "RP1", "A19": "CAP-2", "A20": "CAP-3", "A21": "M2", "A22": "M3", "A23": "CAP-6",
}

#: 01 §7 R9 rungs (empty is allowed for CPU / comparator rows).
RUNGS = ("Q0.8", "Q2", "Q4", "Q9", "Q122", "O20", "O120")
SMALL_RUNGS = ("Q0.8", "Q2", "Q4", "Q9")

EXCLUDED_DATASETS = frozenset({"covtype"})

#: Comparator / CPU rows (A08, A22) run outside Rogue One: their image_tag names a
#: comparator image, they pass no LLM seed (llm_seed may be empty) and the Rogue One
#: launcher never runs them.
COMPARATOR_IMAGE_PREFIXES = ("classicfe:", "llmfe:", "caafe:")


def is_comparator_row(row: dict) -> bool:
    return str(row.get("image_tag", "")).strip().startswith(COMPARATOR_IMAGE_PREFIXES)

# --------------------------------------------------------------------------- R10 datasets
CORE10 = ("balance-scale", "junglechess", "vehicle", "heart", "credit-g", "myocardial",
          "housing", "insurance", "crab", "wine")
CAP_REAL4 = ("junglechess", "vehicle", "heart", "myocardial")
HC1_6 = ("myocardial", "vehicle", "housing", "crab", "heart", "insurance")
W0_8 = tuple(d for d in CORE10 if d not in ("balance-scale", "wine"))
PILOT = ("blood", "communities")
SYNP8 = ("synp_p1", "synp_p2", "synp_p3", "synp_p4", "synp_p5", "synp_p6a", "synp_p6b", "synp_p7")
SYNP7 = tuple(d for d in SYNP8 if d != "synp_p7")
SYNP_ALL = ("synp_p0",) + SYNP8

DATASET_SETS: dict[str, tuple[str, ...]] = {
    "CORE10": CORE10, "CAP_REAL4": CAP_REAL4, "HC1_6": HC1_6, "W0_8": W0_8,
    "PILOT": PILOT, "SYNP8": SYNP8, "SYNP7": SYNP7, "SYNP_ALL": SYNP_ALL,
}

#: dataset_id -> task_type. Legacy real datasets (rogue_one_main's list, minus covtype)
#: plus SYN-P (proposed types, R10; E8a's registry.yaml confirms them).
TASK_TYPES: dict[str, str] = {
    **{d: "classification" for d in (
        "balance-scale", "pc1", "tic-tac-toe", "junglechess", "communities", "eucalyptus",
        "blood", "car", "arrhythmia", "bank", "breast-w", "diabetes", "cmc", "adult",
        "heart", "vehicle", "credit-g", "myocardial")},
    **{d: "regression" for d in (
        "forest-fires", "wine", "plasma_retinol", "housing", "insurance", "crab", "bike",
        "airfoil_self_noice", "diamonds")},
    **{f"synp_{k}": "classification" for k in ("p3", "p4", "p5", "p7")},
    **{f"synp_{k}": "regression" for k in ("p0", "p1", "p2", "p6a", "p6b")},
}

for _name, _set in DATASET_SETS.items():  # the hard rule, enforced at import
    assert not EXCLUDED_DATASETS.intersection(_set), f"{_name} contains an excluded dataset"
assert not EXCLUDED_DATASETS.intersection(TASK_TYPES)


# --------------------------------------------------------------------------- flag registry
def _load_flag_registry():
    """Load rogueone/utils/flag_registry.py by path (no rogueone package side effects)."""
    path = REPO_ROOT / "rogueone" / "utils" / "flag_registry.py"
    spec = importlib.util.spec_from_file_location("_ro27_flag_registry", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_FR = _load_flag_registry()

#: Keys the launcher sets per row; an env_override may not touch them.
LAUNCHER_OWNED_KEYS = frozenset({
    "ROGUE_ONE_DATASETS", "ROGUE_ONE_FOLD_SEED", "ROGUE_ONE_LLM_SEED",
    "ROGUE_ONE_TASK_REGISTRY", "ROGUE_ONE_START_VLLM",
    "LLM_MODEL", "LLM_PORT", "LLM_API_KEY",
    "ROGUE_ONE_SCIENTIST_MODEL", "ROGUE_ONE_SCIENTIST_PORT",
    "ROGUE_ONE_EXTRACTOR_MODEL", "ROGUE_ONE_EXTRACTOR_PORT",
    "ROGUE_ONE_TESTER_MODEL", "ROGUE_ONE_TESTER_PORT",
    "WANDB_API_KEY", "WANDB_PROJECT_NAME", "WANDB_ENTITY", "SERPER_API_KEY",
})
OVERRIDABLE_KEYS = frozenset(_FR.FLAG_KEYS) - LAUNCHER_OWNED_KEYS

#: Mixed-routing rows only (rung_S/E/T or port_S/E/T not all equal): the global
#: LLM_MODEL / LLM_PORT that no role should reach (A04 §4 session C "third model" rule;
#: A19 §4 global = the big rung). Both must be given together; the launcher writes them
#: instead of its rung_E default (ro27_clients_manifest.sh, mixed branch). Uniform rows
#: may never set them (the launcher owns them there). Declared deviation from R7 (A04 §8,
#: A19 §4): R3 keys in env_overrides.
MIXED_GLOBAL_KEYS = frozenset({"LLM_MODEL", "LLM_PORT"})

#: Non-run rows (A18 replay targets): dataset_id -> task_type. They are never run by
#: ro27_clients_manifest.sh (no task folder), only by their own clients script.
NON_RUN_TASK_TYPES: dict[str, str] = {"rp1set": "mixed"}

ROW_ID_RE = re.compile(r"^(A\d{2})-(\d{3})$")
SESSION_ID_RE = re.compile(r"^S\d{4}[a-z]$")
_BAD_VALUE_CHARS = re.compile(r'["\\\n\r\t]')


class ManifestError(ValueError):
    pass


# --------------------------------------------------------------------------- validation
def canonical_overrides(value: Any, *, mixed: bool = False) -> str:
    """Validate ``env_overrides`` and return its canonical compact JSON.

    The launcher parses this field without Python (awk on the host), so the format is
    restricted: a flat object, keys registered R3/R4 flags the launcher does not own,
    values scalars rendered as strings with no quote, backslash, tab or newline.
    ``mixed=True`` (a mixed-routing row) additionally admits the pair
    ``LLM_MODEL`` + ``LLM_PORT`` (:data:`MIXED_GLOBAL_KEYS`).
    """
    if isinstance(value, str):
        text = value.strip() or "{}"
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"env_overrides is not JSON: {text!r} ({exc})") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"env_overrides must be a JSON object, got {type(value).__name__}")
    out: dict[str, str] = {}
    for k, v in value.items():
        if k in MIXED_GLOBAL_KEYS and mixed:
            pass
        elif k in MIXED_GLOBAL_KEYS:
            raise ManifestError(f"env_overrides may set {k} only on a mixed-routing row "
                                "(rung_S/E/T or port_S/E/T not all equal)")
        elif k in LAUNCHER_OWNED_KEYS:
            raise ManifestError(f"env_overrides may not set launcher-owned key {k}")
        elif k not in OVERRIDABLE_KEYS:
            raise ManifestError(f"env_overrides key {k} is not a registered R3/R4 flag")
        if isinstance(v, bool):
            v = "True" if v else "False"
        if isinstance(v, (dict, list)) or v is None:
            raise ManifestError(f"env_overrides[{k}] must be a scalar")
        sv = str(v)
        if _BAD_VALUE_CHARS.search(sv):
            raise ManifestError(f"env_overrides[{k}]={sv!r} contains a quote, backslash, tab or newline")
        out[k] = sv
    given = MIXED_GLOBAL_KEYS.intersection(out)
    if given and given != MIXED_GLOBAL_KEYS:
        raise ManifestError("env_overrides must set LLM_MODEL and LLM_PORT together (mixed global)")
    if "LLM_PORT" in out and not out["LLM_PORT"].isdigit():
        raise ManifestError(f"env_overrides[LLM_PORT]={out['LLM_PORT']!r} is not a port number")
    if "LLM_MODEL" in out and not out["LLM_MODEL"].strip():
        raise ManifestError("env_overrides[LLM_MODEL] is empty")
    return json.dumps(out, separators=(",", ":"), sort_keys=True) if out else "{}"


def is_mixed_row(row: dict) -> bool:
    """A row whose roles do not all use one rung and one port (per-role routing)."""
    rungs = {str(row.get(c, "")).strip() for c in ("rung_S", "rung_E", "rung_T")}
    ports = {str(row.get(c, "")).strip() for c in ("port_S", "port_E", "port_T")}
    return len(rungs) > 1 or len(ports) > 1


def _as_int(row: dict, col: str, *, allow_empty: bool = False) -> None:
    v = str(row.get(col, "")).strip()
    if v == "" and allow_empty:
        return
    try:
        int(v)
    except ValueError as exc:
        raise ManifestError(f"{row.get('row_id')}: {col}={v!r} is not an integer") from exc


def validate_row(row: dict, *, session: bool = False) -> dict:
    """Validate (and normalise) one manifest row; returns the normalised row."""
    overrides = row.get("env_overrides")
    row = {c: ("" if row.get(c) is None else str(row.get(c))) for c in R7_HEADER}
    row["env_overrides"] = overrides if isinstance(overrides, dict) else row["env_overrides"]
    rid = row["row_id"].strip()
    m = ROW_ID_RE.match(rid)
    if not m:
        raise ManifestError(f"row_id {rid!r} is not <AID>-<nnn>")
    if row["ablation_id"].strip() != m.group(1):
        raise ManifestError(f"{rid}: ablation_id {row['ablation_id']!r} does not match the row_id prefix")
    ds = row["dataset_id"].strip()
    if ds.lower() in EXCLUDED_DATASETS:
        raise ManifestError(f"{rid}: dataset_id {ds!r} is excluded from every AGENT 2027 study (covtype rule)")
    if not ds:
        raise ManifestError(f"{rid}: dataset_id is empty")
    if not row["arm_id"].strip():
        raise ManifestError(f"{rid}: arm_id is empty")
    if ds in NON_RUN_TASK_TYPES:
        if row["task_type"] != NON_RUN_TASK_TYPES[ds]:
            raise ManifestError(f"{rid}: {ds} rows have task_type {NON_RUN_TASK_TYPES[ds]!r}")
    elif row["task_type"] not in ("classification", "regression"):
        raise ManifestError(f"{rid}: task_type {row['task_type']!r}")
    known = TASK_TYPES.get(ds)
    if known and known != row["task_type"]:
        raise ManifestError(f"{rid}: {ds} is {known}, not {row['task_type']}")
    for col in ("fold_seed", "replicate"):
        _as_int(row, col)
    _as_int(row, "llm_seed", allow_empty=is_comparator_row(row))
    if row["is_sentinel"] not in ("0", "1"):
        raise ManifestError(f"{rid}: is_sentinel must be 0/1")
    sid = row["session_id"].strip()
    if sid and not SESSION_ID_RE.match(sid):
        raise ManifestError(f"{rid}: session_id {sid!r} is not S<MMDD><a-z>")
    if session and not sid:
        raise ManifestError(f"{rid}: session file row without session_id")
    _as_int(row, "order_key", allow_empty=not session)
    for col in ("rung_S", "rung_E", "rung_T"):
        if row[col] and row[col] not in RUNGS:
            raise ManifestError(f"{rid}: {col}={row[col]!r} is not an R9 rung")
    for col in ("port_S", "port_E", "port_T"):
        _as_int(row, col, allow_empty=True)
    row["env_overrides"] = canonical_overrides(row["env_overrides"], mixed=is_mixed_row(row))
    try:
        cap = float(row["cap_hours"])
    except ValueError as exc:
        raise ManifestError(f"{rid}: cap_hours={row['cap_hours']!r}") from exc
    if cap <= 0:
        raise ManifestError(f"{rid}: cap_hours must be > 0")
    if row["expected_u"]:
        float(row["expected_u"])
    if any(ch in row["notes"] for ch in "\n\r\t"):
        raise ManifestError(f"{rid}: notes may not contain tabs or newlines")
    return row


def validate_rows(rows: list[dict], *, session: bool = False) -> list[dict]:
    out = [validate_row(r, session=session) for r in rows]
    seen: set[str] = set()
    for r in out:
        if r["row_id"] in seen:
            raise ManifestError(f"duplicate row_id {r['row_id']}")
        seen.add(r["row_id"])
    if session:
        keys = [int(r["order_key"]) for r in out]
        if len(set(keys)) != len(keys):
            raise ManifestError("duplicate order_key in session file")
        sids = {r["session_id"] for r in out}
        if len(sids) != 1:
            raise ManifestError(f"session file mixes session ids {sorted(sids)}")
    return out


# --------------------------------------------------------------------------- CSV io
def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=R7_HEADER, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, "") for c in R7_HEADER})
    path.write_bytes(buf.getvalue().encode("utf-8"))


def read_csv(path: Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != R7_HEADER:
        raise ManifestError(f"{path}: header is not R7 (got {reader.fieldnames})")
    return [dict(r) for r in reader]


# --------------------------------------------------------------------------- --ablation
def llm_seed_for(fold_seed: int, replicate: int, is_sentinel: bool, rung: str) -> int:
    """01 §7 R11: r0 -> fold_seed; A/A (r1) -> fold_seed + 1000; O120 sentinel k ->
    5000 + k; small-rung sentinel k -> 6000 + k."""
    if is_sentinel:
        return (6000 if rung in SMALL_RUNGS else 5000) + int(replicate)
    if int(replicate) == 0:
        return int(fold_seed)
    if int(replicate) == 1:
        return int(fold_seed) + 1000
    raise ManifestError(f"no R11 llm_seed rule for replicate={replicate}; set llm_seed explicitly")


def _datasets(spec_value: Any) -> list[str]:
    if isinstance(spec_value, str):
        if spec_value.startswith("@"):
            return list(DATASET_SETS[spec_value[1:]])
        return [spec_value]
    out: list[str] = []
    for v in spec_value:
        out.extend(_datasets(v))
    return out


def _listify(v: Any) -> list:
    return list(v) if isinstance(v, (list, tuple)) else [v]


_PARAM_RE = re.compile(r"\$\{([A-Za-z0-9_]+)\}")


def _subst(value: Any, values: dict) -> Any:
    """Replace ``${NAME}`` in every string of a (nested) spec value."""
    if isinstance(value, str):
        def rep(m: "re.Match[str]") -> str:
            k = m.group(1)
            if k not in values:
                raise ManifestError(f"${{{k}}} is not a declared spec parameter")
            if values[k] in (None, ""):
                raise ManifestError(f"spec parameter {k} is unset; pass --param {k}=VALUE "
                                    "or list it under the block's requires:")
            return str(values[k])
        return _PARAM_RE.sub(rep, value)
    if isinstance(value, list):
        return [_subst(v, values) for v in value]
    if isinstance(value, dict):
        return {k: _subst(v, values) for k, v in value.items()}
    return value


def expand_spec(spec: dict, params: Optional[dict] = None, log=None) -> list[dict]:
    """Expand an ablation spec (YAML mapping) into R7 rows.

    Spec shape::

        ablation_id: A02
        defaults: {image_tag: agent2027-pre.N, rung: O120, port: 8010, cap_hours: 6}
        blocks:
          - arm_id: R10
            start: 1                      # first row number (default: next free)
            datasets: "@SYNP8"            # or a list; "@SET" names a set above
            fold_seeds: [42]
            replicates: [0]
            env_overrides: {ROGUE_ONE_EXTRACTOR_ROUNDS: "10"}
            env_overrides_by_task_type: {classification: {ROGUE_ONE_STRATIFIED_FOLDS: "True"}}
            expected_u: 0.85
            notes: ""

    Any block key may override a default (``image_tag``, ``rung``/``rung_S|E|T``,
    ``port``/``port_S|E|T``, ``cap_hours``, ``expected_u``, ``notes``, ``is_sentinel``,
    ``llm_seed``, ``task_type``). Loop order: fold seed, replicate, dataset (innermost).

    Parameters: a spec may declare ``params: {NAME: default}`` and use ``${NAME}`` in any
    string value; ``--param NAME=VALUE`` overrides the default. A block that lists
    ``requires: [NAME, ...]`` is skipped (with a note) while any of them is unset (null);
    an unset parameter used by a block that is not skipped is an error;
    ``param_choices: {NAME: "@SET" | [..]}`` restricts a parameter's value. This is how a
    value fixed later (e.g. the G2 sentinel task, A06 §4.3) stays out of the rows until
    it is known. ``llm_seed: ""`` leaves the seed empty (comparator rows only).
    """
    declared = dict(spec.get("params") or {})
    for k in (params or {}):
        if k not in declared:
            raise ManifestError(f"--param {k}: the spec declares no such parameter ({sorted(declared)})")
    values = {**declared, **(params or {})}
    for k, choices in (spec.get("param_choices") or {}).items():
        allowed = _datasets(choices)
        if values.get(k) not in (None, "") and values[k] not in allowed:
            raise ManifestError(f"{spec['ablation_id']}: {k}={values[k]!r} is not one of {allowed}")
    log = log or (lambda *_: None)
    aid = spec["ablation_id"]
    defaults = dict(spec.get("defaults") or {})
    rows: list[dict] = []
    skipped: list[tuple[str, str, tuple]] = []
    next_n = 1
    for block in spec["blocks"]:
        b = {**defaults, **block}
        missing = [k for k in _listify(b.pop("requires", []) or []) if values.get(k) in (None, "")]
        if missing:
            skipped.append((str(b.get("arm_id")), str(b.get("start", next_n)), tuple(missing)))
            continue
        b = _subst(b, values)
        n = int(b.get("start", next_n))
        datasets = _datasets(b["datasets"])
        for fs in _listify(b.get("fold_seeds", [42])):
            for rep in _listify(b.get("replicates", [0])):
                for ds in datasets:
                    if ds.lower() in EXCLUDED_DATASETS:
                        raise ManifestError(f"{aid}: dataset {ds} is excluded (covtype rule)")
                    task_type = b.get("task_type") or TASK_TYPES.get(ds)
                    if not task_type:
                        raise ManifestError(f"{aid}: unknown task_type for {ds}; set task_type in the block")
                    rung_s = b.get("rung_S", b.get("rung", ""))
                    rung_e = b.get("rung_E", b.get("rung", ""))
                    rung_t = b.get("rung_T", b.get("rung", ""))
                    sentinel = bool(int(b.get("is_sentinel", 0)))
                    llm_seed = b.get("llm_seed")
                    if llm_seed is None:
                        llm_seed = llm_seed_for(int(fs), int(rep), sentinel, rung_e or rung_s)
                    elif str(llm_seed).strip() != "":
                        llm_seed = int(llm_seed)
                    ov = dict(b.get("env_overrides") or {})
                    ov.update((b.get("env_overrides_by_task_type") or {}).get(task_type, {}))
                    rows.append({
                        "row_id": f"{aid}-{n:03d}",
                        "ablation_id": aid,
                        "arm_id": b["arm_id"],
                        "dataset_id": ds,
                        "task_type": task_type,
                        "fold_seed": int(fs),
                        "llm_seed": llm_seed,
                        "replicate": int(rep),
                        "is_sentinel": int(sentinel),
                        "session_id": "",
                        "order_key": "",
                        "image_tag": b.get("image_tag", ""),
                        "rung_S": rung_s, "rung_E": rung_e, "rung_T": rung_t,
                        "port_S": b.get("port_S", b.get("port", "")),
                        "port_E": b.get("port_E", b.get("port", "")),
                        "port_T": b.get("port_T", b.get("port", "")),
                        "env_overrides": ov,
                        "cap_hours": b.get("cap_hours", 6),
                        "expected_u": b.get("expected_u", ""),
                        "notes": b.get("notes", ""),
                    })
                    n += 1
        next_n = n
    if skipped:
        params_unset = sorted({k for *_, ks in skipped for k in ks})
        log(f"{aid}: skipped {len(skipped)} block(s) (arm {', '.join(sorted({a for a, *_ in skipped}))}; "
            f"starts {', '.join(st for _, st, _ in skipped)}): parameter(s) {', '.join(params_unset)} "
            "unset (pass --param NAME=VALUE)")
    return validate_rows(rows)


def manifest_path(aid: str, manifests_dir: Path = MANIFEST_DIR) -> Path:
    return manifests_dir / f"{aid}_{ABLATION_SHORT[aid]}.csv"


def make_ablation(aid: str, spec_path: Optional[Path], out: Optional[Path],
                  manifests_dir: Path = MANIFEST_DIR, image_tag: Optional[str] = None,
                  params: Optional[dict] = None, log=print) -> Path:
    import yaml

    if aid not in ABLATION_SHORT:
        raise ManifestError(f"unknown ablation {aid}")
    spec_path = spec_path or (SPEC_DIR / f"{aid}.yaml")
    if not spec_path.is_file():
        raise ManifestError(f"no row spec for {aid} at {spec_path}; write one (see expand_spec)")
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if spec.get("ablation_id") != aid:
        raise ManifestError(f"{spec_path} is for {spec.get('ablation_id')}, not {aid}")
    if image_tag:
        spec.setdefault("defaults", {})["image_tag"] = image_tag
    rows = expand_spec(spec, params=params, log=log)
    out = out or manifest_path(aid, manifests_dir)
    write_csv(out, rows)
    return out


# --------------------------------------------------------------------------- --plan-session
def _row_num(rid: str) -> tuple[str, int]:
    m = ROW_ID_RE.match(rid)
    if not m:
        raise ManifestError(f"bad row id {rid!r}")
    return m.group(1), int(m.group(2))


def expand_row_tokens(tokens: Iterable[str], available: Iterable[str]) -> list[str]:
    """Resolve ``--rows`` tokens (IDs and ranges) against the available row IDs."""
    avail = list(available)
    avail_set = set(avail)
    toks: list[str] = []
    for t in tokens:
        toks.extend(x for x in re.split(r"[\s,]+", t.strip()) if x)

    def _range(a: str, b: str) -> list[str]:
        aid, lo = _row_num(a)
        if ROW_ID_RE.match(b):
            bid, hi = _row_num(b)
            if bid != aid:
                raise ManifestError(f"range {a}…{b} spans two ablations")
        elif b.isdigit():
            hi = int(b)
        else:
            raise ManifestError(f"bad range end {b!r}")
        if hi < lo:
            raise ManifestError(f"empty range {a}…{b}")
        got = [f"{aid}-{n:03d}" for n in range(lo, hi + 1) if f"{aid}-{n:03d}" in avail_set]
        if not got:
            raise ManifestError(f"range {a}…{b} matches no manifest row")
        return got

    out: list[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in ("…", "...", ".."):
            if not out or i + 1 >= len(toks):
                raise ManifestError("dangling range marker in --rows")
            start = out.pop()
            out.extend(_range(start, toks[i + 1]))
            i += 2
            continue
        parts = re.split(r"…|\.\.\.|\.\.", t)
        if len(parts) == 2:
            out.extend(_range(parts[0], parts[1]))
        elif len(parts) == 1:
            if t not in avail_set:
                raise ManifestError(f"row {t} is in no ablation manifest")
            out.append(t)
        else:
            raise ManifestError(f"bad --rows token {t!r}")
        i += 1
    dup = {r for r in out if out.count(r) > 1}
    if dup:
        raise ManifestError(f"rows listed twice: {sorted(dup)}")
    return out


def load_ablation_rows(manifests_dir: Path = MANIFEST_DIR) -> dict[str, dict]:
    """Every row of every ablation manifest in ``manifests_dir`` (sessions/ excluded)."""
    rows: dict[str, dict] = {}
    for path in sorted(manifests_dir.glob("A??_*.csv")):
        for r in read_csv(path):
            rid = r["row_id"]
            if rid in rows:
                raise ManifestError(f"row {rid} appears in two manifests")
            rows[rid] = r
    return rows


def session_seed(session_id: str) -> int:
    """RNG seed for a session's order_key: first 8 bytes of sha256(session_id)."""
    return int.from_bytes(hashlib.sha256(session_id.encode("utf-8")).digest()[:8], "big")


def plan_session(session_id: str, row_tokens: list[str], *, sentinel_first: bool = False,
                 no_sentinel: bool = False, manifests_dir: Path = MANIFEST_DIR,
                 out: Optional[Path] = None, log=print,
                 sentinel_image_tag: Optional[str] = None, port: Optional[int] = None,
                 proxy_port: Optional[int] = None) -> Path:
    """Write ``sessions/<SID>.csv``.

    ``sentinel_image_tag`` stamps the host session's image on its sentinel rows (A06
    §4.2: "image_tag = the host's tag, written into the row when it is planned"); a
    sentinel row must end up with an image_tag. ``port`` stamps the session server's
    port on every Rogue One row with planned ports, ``proxy_port`` the X5 token-proxy
    port on every comparator row with planned ports (the launchers use the live server
    port anyway; the stamp keeps the session file truthful).
    """
    if not SESSION_ID_RE.match(session_id):
        raise ManifestError(f"session id {session_id!r} is not S<MMDD><a-z>")
    pool = load_ablation_rows(manifests_dir)
    row_ids = expand_row_tokens(row_tokens, pool.keys())
    rows = [dict(pool[r]) for r in row_ids]
    for r in rows:
        if r["session_id"].strip() and r["session_id"].strip() != session_id:
            raise ManifestError(f"{r['row_id']} is already pinned to session {r['session_id']}")
    for r in rows:
        if str(r["is_sentinel"]) == "1" and sentinel_image_tag:
            r["image_tag"] = sentinel_image_tag
        if str(r["is_sentinel"]) == "1" and not str(r["image_tag"]).strip():
            raise ManifestError(f"{r['row_id']} is a sentinel without image_tag: pass "
                                "--sentinel-image-tag <host image tag> (A06 §4.2)")
        stamp = proxy_port if is_comparator_row(r) else port
        # Mixed-routing rows keep their per-role ports: they ARE the live ports of the
        # session's several servers (the launcher routes mixed rows by them).
        if stamp is not None and not is_mixed_row(r):
            for col in ("port_S", "port_E", "port_T"):
                if str(r[col]).strip():
                    r[col] = str(int(stamp))
    seed = session_seed(session_id)
    rng = random.Random(seed)
    keys = list(range(1, len(rows) + 1))
    rng.shuffle(keys)
    for r, k in zip(rows, keys):
        r["session_id"] = session_id
        r["order_key"] = k
    sentinels = [r for r in rows if str(r["is_sentinel"]) == "1"]
    if sentinel_first and sentinels:
        # The sentinel takes the session minimum ("first wave"); swap keys, so the
        # rest of the permutation is unchanged.
        for i, s in enumerate(sorted(sentinels, key=lambda r: r["row_id"])):
            want = i + 1
            holder = next(r for r in rows if int(r["order_key"]) == want)
            holder["order_key"], s["order_key"] = s["order_key"], want
    if not sentinels and not no_sentinel:
        log(f"WARNING: session {session_id} has no sentinel row (pass --no-sentinel if intended, "
            "e.g. smoke / pilot sessions, A06 §4.5)")
    rows.sort(key=lambda r: int(r["order_key"]))
    rows = validate_rows(rows, session=True)
    out = out or (manifests_dir / "sessions" / f"{session_id}.csv")
    write_csv(out, rows)
    plan_log = out.with_suffix(".plan.txt")
    lines = [
        f"session_id={session_id}",
        f"rng=random.Random(int.from_bytes(sha256(session_id)[:8], 'big')) seed={seed}",
        f"sentinel_first={int(sentinel_first)} sentinels={','.join(r['row_id'] for r in sentinels) or 'none'}",
        *([f"sentinel_image_tag={sentinel_image_tag}"] if sentinel_image_tag else []),
        *([f"port={port}"] if port is not None else []),
        *([f"proxy_port={proxy_port}"] if proxy_port is not None else []),
        f"rows={len(rows)}",
        "order_key\trow_id\tarm_id\tdataset_id\tis_sentinel",
        *[f"{r['order_key']}\t{r['row_id']}\t{r['arm_id']}\t{r['dataset_id']}\t{r['is_sentinel']}" for r in rows],
    ]
    plan_log.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    log(f"wrote {out} ({len(rows)} rows; order seed {seed}; log {plan_log.name})")
    return out


def container_name(session_id: str, row_id: str) -> str:
    """01 §7 R8: ``ro27-<session_id>-<row_id>``, lower-case."""
    return f"ro27-{session_id}-{row_id}".lower()


# --------------------------------------------------------------------------- CLI
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ablation", metavar="AID")
    mode.add_argument("--plan-session", metavar="SESSION_ID")
    mode.add_argument("--check", nargs="+", metavar="CSV")
    ap.add_argument("--spec", type=Path, help="row spec YAML for --ablation")
    ap.add_argument("--image-tag", help="--ablation: override the spec's default image_tag")
    ap.add_argument("--rows", nargs="+", default=[], help="row ids / ranges for --plan-session")
    ap.add_argument("--sentinel-first", action="store_true",
                    help="give the sentinel row(s) the session minimum order_key (guard-3 session)")
    ap.add_argument("--no-sentinel", action="store_true", help="silence the no-sentinel warning")
    ap.add_argument("--sentinel-image-tag", help="--plan-session: the host image tag for its sentinel row(s)")
    ap.add_argument("--port", type=int, help="--plan-session: stamp this server port on Rogue One rows")
    ap.add_argument("--proxy-port", type=int, help="--plan-session: stamp this X5 proxy port on comparator rows")
    ap.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                    help="--ablation: set a spec parameter (repeatable)")
    ap.add_argument("--manifests-dir", type=Path, default=MANIFEST_DIR)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    try:
        if a.ablation:
            params = {}
            for kv in a.param:
                if "=" not in kv:
                    ap.error(f"--param {kv!r} is not NAME=VALUE")
                k, v = kv.split("=", 1)
                params[k.strip()] = v.strip()
            out = make_ablation(a.ablation, a.spec, a.out, a.manifests_dir, a.image_tag, params=params)
            print(f"wrote {out}")
        elif a.plan_session:
            if not a.rows:
                ap.error("--plan-session needs --rows")
            plan_session(a.plan_session, a.rows, sentinel_first=a.sentinel_first,
                         no_sentinel=a.no_sentinel, manifests_dir=a.manifests_dir, out=a.out,
                         sentinel_image_tag=a.sentinel_image_tag, port=a.port,
                         proxy_port=a.proxy_port)
        else:
            for p in a.check:
                path = Path(p)
                rows = read_csv(path)
                validate_rows(rows, session=path.parent.name == "sessions")
                print(f"OK {path} ({len(rows)} rows)")
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
