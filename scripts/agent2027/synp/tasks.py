"""SYN-P task definitions P0-P7 and the deterministic table generator (E8a).

Everything here is CPU-only and deterministic given ``(master_seed, task key)``.
The *metadata* (phi expression, constants, seeds, oracle focus, detector) is
returned in ``GeneratedTask.meta`` and written by ``generate_synp.py`` to a
directory that is never mounted into an agent container.

Planted-signal model
--------------------
* regression:      y = z(phi) + noise * eps, standardised to unit SD
                   (mean ``TARGET_OFFSET``, see below), eps ~ N(0, 1)
* classification:  P(y = 1) = sigmoid(beta * (z(phi) - z_cut)); z_cut is the midpoint
                   of the two levels for a binary phi, else the (1 - prevalence)
                   quantile of z(phi) (prevalence 0.5 unless set); y = 1[u < P]
* P5 (deterministic rule): y = class of sign(LW*LD^2 - RW*RD^2), optional flip noise

``eps`` and ``u`` are fixed per task (own RNG stream), so calibration over
``noise`` / ``beta`` only moves the signal strength, never the rows.

``z(phi)`` is plain standardisation (mean 0, SD 1) of the planted feature.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from scripts.agent2027.analysis.recovery_detector import spearman_abs
from scripts.agent2027.synp import context_template as ctx
from scripts.agent2027.synp import frame as fr

N_ROWS = 2000
MASTER_SEED = 2027
# Standardised regression targets have SD 1.  The repo's n_RMSE is RMSE / mean(target)
# (tester/agent.py), which is undefined for a mean-0 target, so the standardised
# target is shifted to a positive mean.  The shift does not affect SHAP magnitudes
# (scale only).  See the E8a report ("decisions for the author").
TARGET_OFFSET = 5.0

TASK_KEYS = ("p0", "p1", "p2", "p3", "p4", "p5", "p6a", "p6b", "p7")
GENERATED_KEYS = ("p0", "p1", "p2", "p3", "p4", "p5", "p6a", "p7")  # p6b is derived
# Proposed types (01 R10); the binding file is synp_meta/registry.yaml
TASK_TYPES = {
    "p0": "regression", "p1": "regression", "p2": "regression", "p3": "classification",
    "p4": "classification", "p5": "classification", "p6a": "regression",
    "p6b": "regression", "p7": "classification",
}
HEADROOM_TARGET = {"classification": 0.06, "regression": 0.15}  # balanced acc / relative RMSE
# calibration aims slightly above the pass line
HEADROOM_AIM = {"classification": 0.075, "regression": 0.18}

# Signal strength per task (calibrated by calibrate_synp.py; see the report)
SIGNAL_DEFAULTS: dict[str, dict[str, float]] = {
    "p0": {"noise": 0.5},
    "p1": {"noise": 0.1},
    "p2": {"noise": 0.25},
    "p3": {"beta": 1.5},
    "p4": {"beta": 0.5, "prevalence": 0.5},
    "p5": {"flip": 0.0},
    "p6a": {"noise": 0.25},
    "p7": {"beta": 1000.0, "prevalence": 0.1},
}


def derive_seed(master: int, key: str, stream: str = "data") -> int:
    h = hashlib.sha256(f"{master}:{key}:{stream}".encode()).digest()
    return int.from_bytes(h[:4], "big")


# --------------------------------------------------------------------------- #
# phi: expression + column map + constants (all in the metadata)
# --------------------------------------------------------------------------- #
def eval_phi(spec: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    ns: dict[str, Any] = {"np": np}
    for var, col in spec["cols"].items():
        ns[var] = df[col].to_numpy()
    ns.update(spec.get("consts", {}))
    out = eval(spec["expr"], {"__builtins__": {}}, ns)  # noqa: S307 - trusted local metadata
    return np.asarray(out, dtype=float)


def phi_from_meta(meta: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    return eval_phi(meta["phi"], df)


def pandas_expr(spec: dict[str, Any], colmap: dict[str, str] | None = None) -> str:
    """phi as a pandas expression over ``df[...]`` (``colmap`` renames columns)."""
    expr = spec["expr"]
    consts = spec.get("consts", {})
    for var, col in spec["cols"].items():
        col = (colmap or {}).get(col, col)
        expr = re.sub(rf"\b{re.escape(var)}\b", f"df[{col!r}]", expr)
    for name, val in consts.items():
        expr = re.sub(rf"\b{re.escape(name)}\b", repr(val), expr)
    return f"({expr}).astype(float)" if spec.get("boolean") else expr


# --------------------------------------------------------------------------- #
# Signal
# --------------------------------------------------------------------------- #
def _std(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def compute_target(
    task_type: str, phi: np.ndarray, params: dict[str, float],
    eps: np.ndarray, u: np.ndarray, *, deterministic_classes: np.ndarray | None = None,
) -> np.ndarray:
    z = _std(np.asarray(phi, dtype=float))
    if task_type == "regression":
        y = z + float(params["noise"]) * eps
        return _std(y) + TARGET_OFFSET
    if deterministic_classes is not None:  # P5
        y = deterministic_classes.copy()
        flip = float(params.get("flip", 0.0))
        if flip > 0:
            n_cls = int(y.max()) + 1
            m = u < flip
            y[m] = (y[m] + 1 + (eps[m] > 0).astype(int)) % n_cls
        return y
    beta = float(params["beta"])
    levels = np.unique(np.round(z, 9))
    if len(levels) <= 2:  # binary phi: cut half-way between the two levels
        z_cut = float(levels.mean())
    else:  # continuous phi: cut at the (1 - prevalence) quantile
        z_cut = float(np.quantile(z, 1.0 - float(params.get("prevalence", 0.5))))
    c = -beta * z_cut
    return (u < _sigmoid(beta * z + c)).astype(int)


# --------------------------------------------------------------------------- #
# Task builders
# --------------------------------------------------------------------------- #
@dataclass
class Built:
    asm: fr.Assembled
    phi: dict[str, Any]
    intro: str
    unit: str
    target: str
    classes: str = ""
    detector_family: str = "spearman"
    depth: str = ""
    extra_meta: dict[str, Any] = field(default_factory=dict)
    monomial: dict[str, int] | None = None
    deterministic_classes: np.ndarray | None = None


def _ln(rng, n, median, sigma):
    return np.exp(np.log(median) + sigma * rng.standard_normal(n))


def _simple(rng, n, pool, cols, *, n_cats=None, relevant_cats=None, relevant_extra=None, corr_ab=0.0):
    """cols: list of (name, desc, median, sigma) lognormal relevant columns.

    ``corr_ab`` > 0 correlates the (log of the) first two columns (latent correlation):
    a ratio of two correlated quantities varies only along the thin direction
    orthogonal to the data cloud, which axis-aligned trees approximate poorly."""
    z = rng.standard_normal((len(cols), n))
    if corr_ab:
        z[1] = corr_ab * z[0] + np.sqrt(1 - corr_ab**2) * z[1]
    rel = {name: snap(np.exp(np.log(med) + sg * z[i])) for i, (name, _d, med, sg) in enumerate(cols)}
    rel_desc = {name: d for name, d, _m, _s in cols}
    if relevant_extra:
        for name, (vals, d) in relevant_extra.items():
            rel[name] = vals
            rel_desc[name] = d
    return fr.assemble(rng, n, rel, rel_desc, pool, relevant_cats=relevant_cats, n_cats=n_cats)


snap = fr.snap


def build_p0(rng, n, **_):
    cols = [("building_floor_area_m2", "heated floor area of the building, square metres", 800.0, 0.6)]
    asm = _simple(rng, n, fr.INDUSTRIAL, cols)
    return Built(
        asm, {"expr": "np.log(a)", "cols": {"a": "building_floor_area_m2"}},
        "Energy audits of commercial buildings.", "building", "the annual heating demand index",
        depth="negative control (monotone transform of one column)",
    )


def build_p1(rng, n, **_):
    cols = [
        ("energy_consumed_kwh", "energy drawn from the battery during the trip, kWh", 40.0, 0.4),
        ("distance_km", "distance driven during the trip, km", 180.0, 0.4),
    ]
    asm = _simple(rng, n, fr.INDUSTRIAL, cols)
    return Built(
        asm, {"expr": "a / b", "cols": {"a": "energy_consumed_kwh", "b": "distance_km"}},
        "Trip logs from a fleet of electric delivery vans.", "trip", "the battery health score",
        depth="D1 algebraic ratio",
    )


def build_p2(rng, n, **_):
    cols = [
        ("axial_load_kn", "axial load applied to the member, kN", 120.0, 0.4),
        ("cross_section_cm2", "cross-section area of the member, cm2", 30.0, 0.4),
        ("member_length_m", "length of the member, metres", 4.0, 0.4),
    ]
    asm = _simple(rng, n, fr.INDUSTRIAL, cols)
    return Built(
        asm,
        {"expr": "(a / b) * c", "cols": {"a": "axial_load_kn", "b": "cross_section_cm2", "c": "member_length_m"}},
        "Load tests of steel members in a structures laboratory.", "test specimen",
        "the measured deformation index", depth="D2 ratio x product",
    )


P3_CORR_AB = 0.85  # loan amount and income are correlated (thin ratio direction)
P3_RATIO_QUANTILE = 0.8  # threshold t = this quantile of the ratio, rounded to 3 significant digits


def build_p3(rng, n, **_):
    cols = [
        ("loan_amount_k", "requested loan amount, thousand NOK", 250.0, 0.5),
        ("annual_income_k", "gross annual income, thousand NOK", 70.0, 0.5),
    ]
    lv = ("salaried", "self_employed", "retired")
    emp = np.asarray(lv)[rng.choice(3, size=n, p=[0.35, 0.45, 0.20])]
    asm = _simple(
        rng, n, fr.FINANCE, cols, n_cats=2, relevant_cats={"employment_type": lv}, corr_ab=P3_CORR_AB,
        relevant_extra={"employment_type": (emp, "employment type of the applicant")},
    )
    ratio = asm.frame["loan_amount_k"].to_numpy() / asm.frame["annual_income_k"].to_numpy()
    thr = float(snap(np.array([np.quantile(ratio, P3_RATIO_QUANTILE)]), 3)[0])
    return Built(
        asm,
        {"expr": "(a / b >= thr) & (e == lvl)",
         "cols": {"a": "loan_amount_k", "b": "annual_income_k", "e": "employment_type"},
         "consts": {"thr": thr, "lvl": "self_employed"}, "boolean": True},
        "Loan applications at a retail bank.", "application", "whether the loan defaulted",
        classes="1 = defaulted, 0 = repaid", detector_family="rule",
        depth="D3 threshold-AND of a ratio",
    )


def build_p4(rng, n, **_):
    t = snap(rng.uniform(0.0, 720.0, size=n))
    asm = fr.assemble(
        rng, n, {"hours_since_start": t},
        {"hours_since_start": "hours elapsed since the monitoring system was switched on"},
        fr.INDUSTRIAL,
    )
    return Built(
        asm, {"expr": "np.sin(2 * np.pi * a / per)", "cols": {"a": "hours_since_start"}, "consts": {"per": 24.0}},
        "Readings from a continuously running monitoring system.", "reading",
        "whether an alarm was raised", classes="1 = alarm raised, 0 = no alarm",
        depth="cyclic / temporal (period 24 on an elapsed-hours counter)",
    )


BALANCE_NAMES = ("Left-Weight", "Left-Distance", "Right-Weight", "Right-Distance")
BALANCE_DESC = {
    "Left-Weight": "weight placed on the left pan (1 to 5)",
    "Left-Distance": "distance of the left weight from the fulcrum (1 to 5)",
    "Right-Weight": "weight placed on the right pan (1 to 5)",
    "Right-Distance": "distance of the right weight from the fulcrum (1 to 5)",
}


def build_p5(rng, n, **_):
    """Real balance-scale rows: the full 5^4 grid once (625 rows, real names)
    plus ``n - 625`` rows resampled from the same grid (see the E8a report)."""
    grid = np.array(np.meshgrid(*[np.arange(1, 6)] * 4, indexing="ij")).reshape(4, -1).T
    extra = max(0, n - len(grid))
    rows = np.vstack([grid, grid[rng.integers(0, len(grid), size=extra)]])[:n]
    rows = rows[rng.permutation(len(rows))]
    rel = {nm: rows[:, i].astype(int) for i, nm in enumerate(BALANCE_NAMES)}
    asm = fr.assemble(rng, n, rel, dict(BALANCE_DESC), fr.INDUSTRIAL)
    lw, ld, rw, rd = (rows[:, i].astype(float) for i in range(4))
    diff = np.sign(lw * ld**2 - rw * rd**2)
    classes = np.where(diff > 0, 0, np.where(diff == 0, 1, 2)).astype(int)
    return Built(
        asm,
        {"expr": "np.sign(a * b ** 2 - c * d ** 2)",
         "cols": {"a": "Left-Weight", "b": "Left-Distance", "c": "Right-Weight", "d": "Right-Distance"}},
        "Experiments with a lever-arm balance scale.", "experiment", "which side of the scale tips",
        classes="0 = left side tips, 1 = balanced, 2 = right side tips", detector_family="rule",
        depth="counterfactual rule (squared distance)",
        extra_meta={
            "phi_textbook_prior": {"expr": "np.sign(a * b - c * d)",
                                   "cols": {"a": "Left-Weight", "b": "Left-Distance",
                                            "c": "Right-Weight", "d": "Right-Distance"}},
            "class_codes": {"0": "L", "1": "B", "2": "R"},
        },
        deterministic_classes=classes,
    )


# P6a candidates (decision D8; the pairwise screen decides).  dist: ("ln", median, sigma) | ("unif", lo, hi)
P6A_CANDIDATES: dict[str, dict[str, Any]] = {
    "reynolds": dict(
        cols=[
            ("flow_velocity_m_s", "mean flow velocity in the test section, m/s", ("ln", 1.5, 0.4)),
            ("pipe_diameter_mm", "inner diameter of the pipe, mm", ("ln", 80.0, 0.4)),
            ("kinematic_viscosity_cst", "kinematic viscosity of the fluid, cSt", ("ln", 3.0, 0.4)),
        ],
        expr="(a * b) / c", monomial={"a": 1, "b": 1, "c": -1},
        intro="Measurements from an industrial pipe-flow test loop.", unit="test run",
        target="the flow resistance index", depth="named composite (monomial)",
    ),
    "ideal_gas": dict(
        cols=[
            ("amount_mol", "amount of gas in the vessel, mol", ("ln", 12.0, 0.4)),
            ("temperature_k", "gas temperature, kelvin", ("ln", 320.0, 0.4)),
            ("volume_l", "vessel volume, litres", ("ln", 250.0, 0.4)),
        ],
        expr="(a * b) / c", monomial={"a": 1, "b": 1, "c": -1},
        intro="Measurements from sealed gas vessels in a process plant.", unit="vessel reading",
        target="the vessel stress index", depth="named composite (monomial)",
    ),
    "cockcroft_gault": dict(
        cols=[
            ("age_years", "patient age, years", ("unif", 25.0, 85.0)),
            ("body_weight_kg", "body weight, kg", ("ln", 75.0, 0.2)),
            ("serum_creatinine_umol_l", "serum creatinine, umol/L", ("ln", 85.0, 0.3)),
        ],
        expr="((140 - a) * b) / (72 * c)", monomial=None,
        intro="Routine laboratory records from an outpatient clinic.", unit="visit",
        target="the renal function index", depth="named composite (non-monomial)",
    ),
    # negative control: w / h^2 with realistic ranges is rank-identical to w / h
    "bmi_like": dict(
        cols=[
            ("body_weight_kg", "body weight, kg", ("ln", 75.0, 0.2)),
            ("height_m", "standing height, metres", ("ln", 1.70, 0.07)),
        ],
        expr="a / b ** 2", monomial={"a": 1, "b": -2},
        intro="Health-check records from an occupational clinic.", unit="check-up",
        target="the body composition index", depth="named composite (fails the screen)",
    ),
}
DEFAULT_P6A = "reynolds"


def build_p6a(rng, n, p6a_candidate: str = DEFAULT_P6A, **_):
    cand = P6A_CANDIDATES[p6a_candidate]
    rel, desc = {}, {}
    for name, d, dist in cand["cols"]:
        if dist[0] == "ln":
            rel[name] = snap(_ln(rng, n, dist[1], dist[2]))
        else:
            rel[name] = snap(rng.uniform(dist[1], dist[2], size=n))
        desc[name] = d
    asm = fr.assemble(rng, n, rel, desc, fr.INDUSTRIAL)
    var_names = "abcdefgh"
    cols = {var_names[i]: c[0] for i, c in enumerate(cand["cols"])}
    mono = None
    if cand["monomial"]:
        mono = {cols[v]: e for v, e in cand["monomial"].items()}
    return Built(
        asm, {"expr": cand["expr"], "cols": cols}, cand["intro"], cand["unit"], cand["target"],
        depth=cand["depth"], extra_meta={"p6a_candidate": p6a_candidate}, monomial=mono,
    )


P7_ROUND_QUANTILES = (0.25, 1 / 3, 0.4, 0.5, 0.6, 2 / 3, 0.75)
P7_WRONG_CANDIDATES = (
    ("(a * b) * (c >= thr)", "product instead of ratio"),
    ("(a / b) * (c < thr)", "complementary support"),
    ("(b / a) * (c >= thr)", "inverted ratio"),
    ("(a / c) * (b >= thr_b)", "rotated roles"),
)


def secret_rng(secret: str) -> np.random.Generator:
    return np.random.default_rng(int.from_bytes(hashlib.sha256(secret.encode()).digest()[:8], "big"))


P7_CORR_AB = 0.85  # tension and anchor spacing are correlated


def build_p7(rng, n, p7_secret: str = "", p7_constants: dict[str, Any] | None = None, **_):
    cols = [
        ("strand_tension_kn", "measured tension in the cable strand, kN", 90.0, 0.4),
        ("anchor_spacing_m", "spacing between cable anchor points, metres", 12.0, 0.4),
        ("corrosion_depth_mm", "deepest measured corrosion pit on the strand, mm", 1.2, 0.5),
    ]
    asm = _simple(rng, n, fr.INDUSTRIAL, cols, corr_ab=P7_CORR_AB)
    c = asm.frame["corrosion_depth_mm"].to_numpy()
    if p7_constants:
        thr, q7 = float(p7_constants["thr"]), float(p7_constants["q"])
    else:
        sr = secret_rng(p7_secret or f"default-{MASTER_SEED}")
        while True:  # unguessable: far from every round quantile
            q7 = float(sr.uniform(0.30, 0.70))
            if all(abs(q7 - r) > 0.04 for r in P7_ROUND_QUANTILES):
                break
        thr = float(snap(np.array([np.quantile(c, q7)]), 9)[0])
    phi = {
        "expr": "(a / b) * (c >= thr)",
        "cols": {"a": "strand_tension_kn", "b": "anchor_spacing_m", "c": "corrosion_depth_mm"},
        "consts": {"thr": thr},
    }
    # phi_wrong: first candidate with |rho(phi_wrong, phi)| < 0.5 on all rows
    df = asm.frame
    phi_vals = eval_phi(phi, df)
    wrong = None
    for expr, why in P7_WRONG_CANDIDATES:
        w_consts = {"thr": thr, "thr_b": float(np.median(df["anchor_spacing_m"]))}
        spec = {"expr": expr, "cols": dict(phi["cols"]), "consts": w_consts}
        if spearman_abs(eval_phi(spec, df), phi_vals) < 0.5:
            wrong = {**spec, "why": why}
            break
    return Built(
        asm, phi,
        "Inspection records for the stay cables of a suspension bridge. The inspection office "
        "ranks cable segments with its in-house 'Marenko index' (not defined in this table).",
        "cable segment", "whether the segment was flagged for inspection",
        classes="1 = flagged, 0 = not flagged", detector_family="rule",
        depth="knowledge-only fictional composite (ratio x threshold, unguessable constant)",
        extra_meta={"phi_wrong": wrong, "p7_constants": {"thr": thr, "q": q7}},
    )


BUILDERS = {
    "p0": build_p0, "p1": build_p1, "p2": build_p2, "p3": build_p3, "p4": build_p4,
    "p5": build_p5, "p6a": build_p6a, "p7": build_p7,
}


# --------------------------------------------------------------------------- #
# Oracle focus string (01 / A09 section 4.2 template)
# --------------------------------------------------------------------------- #
def oracle_focus(spec: dict[str, Any], key: str, colmap: dict[str, str] | None = None) -> str:
    expr = pandas_expr(spec, colmap)
    s = (f"Build exactly this feature with `append_new_attribute`: "
         f"oracle_feature = {expr}.")
    if key in ("p3", "p5", "p7"):
        s += " Build it in steps: first the intermediate quantities, then combine them into oracle_feature."
    return s


# --------------------------------------------------------------------------- #
# Generated task
# --------------------------------------------------------------------------- #
@dataclass
class GeneratedTask:
    key: str
    task_id: str
    task_type: str
    data: pd.DataFrame  # id + features
    labels: pd.DataFrame  # id, target
    phi: np.ndarray
    context: str
    meta: dict[str, Any]
    column_desc: list[tuple[str, str]]
    categorical: tuple[str, ...]
    levels: dict[str, tuple[str, ...]]
    built: Built = field(repr=False, default=None)  # type: ignore[assignment]


def data_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(None, index=False, float_format="%.12g", lineterminator="\n").encode()


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def phi_sha256(phi: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(np.asarray(phi, dtype="<f8")).tobytes())


def generate_task(
    key: str,
    *,
    master_seed: int = MASTER_SEED,
    n: int = N_ROWS,
    params: dict[str, float] | None = None,
    p7_secret: str = "",
    p7_constants: dict[str, Any] | None = None,
    p6a_candidate: str = DEFAULT_P6A,
) -> GeneratedTask:
    if key not in BUILDERS:
        raise KeyError(f"unknown generated task {key!r} (p6b is derived via p6_twin)")
    task_type = TASK_TYPES[key]
    data_seed = derive_seed(master_seed, key, "data")
    noise_seed = derive_seed(master_seed, key, "noise")
    rng = np.random.default_rng(data_seed)
    built = BUILDERS[key](rng, n, p6a_candidate=p6a_candidate, p7_secret=p7_secret, p7_constants=p7_constants)
    frame = built.asm.frame.copy()
    phi = eval_phi(built.phi, frame)
    nrng = np.random.default_rng(noise_seed)
    eps, u = nrng.standard_normal(n), nrng.random(n)
    sig = dict(SIGNAL_DEFAULTS[key])
    if params:
        sig.update(params)
    y = compute_target(task_type, phi, sig, eps, u, deterministic_classes=built.deterministic_classes)
    data = frame.copy()
    data.insert(0, "id", np.arange(n))
    labels = pd.DataFrame({"id": np.arange(n), "target": y})
    if data.drop(columns="id").duplicated().any():
        raise AssertionError(f"{key}: generated table has exact-duplicate rows")

    columns = [(c, built.asm.descriptions[c]) for c in frame.columns]
    context = ctx.render_context(
        intro=built.intro, unit=built.unit, n_rows=n, columns=columns,
        task_type=task_type, target=built.target, classes=built.classes,
    )
    task_id = f"synp_{key}"
    meta: dict[str, Any] = {
        "task_id": task_id, "key": key, "task_type": task_type, "family": built.depth,
        "master_seed": master_seed, "data_seed": data_seed, "noise_seed": noise_seed,
        "n_rows": n, "n_columns": int(frame.shape[1]),
        "phi": built.phi, "phi_pandas_expr": pandas_expr(built.phi),
        "relevant_columns": [c for c, r in built.asm.roles.items() if r == "relevant"],
        "column_roles": built.asm.roles,
        "signal_params": sig, "target_offset": TARGET_OFFSET,
        "detector": {"family": built.detector_family, "spearman_threshold": 0.95, "nmi_threshold": 0.9},
        "headroom_target": HEADROOM_TARGET[task_type] if key != "p0" else None,
        "oracle_focus": oracle_focus(built.phi, key),
        "monomial_exponents": built.monomial,
        "data_sha256": sha256(data_csv_bytes(data)),
        "labels_sha256": sha256(labels.to_csv(None, index=False, lineterminator="\n").encode()),
        "phi_sha256": phi_sha256(phi),
        "context_chars": len(context),
        **built.extra_meta,
    }
    return GeneratedTask(
        key=key, task_id=task_id, task_type=task_type, data=data, labels=labels, phi=phi,
        context=context, meta=meta, column_desc=columns,
        categorical=tuple(built.asm.levels), levels=dict(built.asm.levels), built=built,
    )
