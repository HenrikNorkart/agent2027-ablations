"""Column-assembly helpers for the SYN-P generator (E8a).

A SYN-P table has: the *relevant* columns (inputs of the planted phi), correlated
decoys (latent correlation 0.55-0.85 with one relevant column, never a near
copy), near-collinear pairs (same quantity in two units plus ~1% noise, built
from irrelevant quantities only), pure-noise numeric columns and a few
noise categoricals.  Column order is shuffled so position does not reveal
relevance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

# (name, description)
Col = tuple[str, str]
# (name_a, desc_a, name_b, desc_b, factor)
PairCol = tuple[str, str, str, str, float]
# (name, desc, levels)
CatCol = tuple[str, str, tuple[str, ...]]

N_CORR, N_PAIRS, N_NOISE = 5, 2, 8


@dataclass(frozen=True)
class Pool:
    corr: tuple[Col, ...]
    pairs: tuple[PairCol, ...]
    noise: tuple[Col, ...]
    cats: tuple[CatCol, ...]

    def all_names(self) -> set[str]:
        names = {c[0] for c in self.corr} | {c[0] for c in self.noise}
        for p in self.pairs:
            names |= {p[0], p[2]}
        return names | {c[0] for c in self.cats}


INDUSTRIAL = Pool(
    corr=(
        ("ambient_temperature_c", "ambient temperature at the site, degrees Celsius"),
        ("supply_voltage_v", "supply voltage, volts"),
        ("vibration_rms_mm_s", "RMS vibration velocity, mm/s"),
        ("operating_hours_total", "accumulated operating hours"),
        ("line_pressure_kpa", "line pressure, kPa"),
    ),
    pairs=(
        ("site_altitude_m", "site altitude, metres", "site_altitude_ft", "site altitude, feet", 3.28084),
        ("coolant_flow_l_min", "coolant flow, litres per minute", "coolant_flow_gal_min",
         "coolant flow, US gallons per minute", 0.264172),
    ),
    noise=(
        ("relative_humidity_pct", "relative humidity, percent"),
        ("maintenance_interval_days", "days between scheduled maintenance"),
        ("sensor_age_months", "age of the installed sensor, months"),
        ("shift_length_h", "length of the working shift, hours"),
        ("noise_level_db", "background noise level, dB"),
        ("dust_concentration_mg_m3", "airborne dust concentration, mg per cubic metre"),
        ("power_factor", "electrical power factor"),
        ("inspection_count", "number of previous inspections"),
    ),
    cats=(
        ("site_region", "region of the site", ("north", "south", "east", "west", "central")),
        ("operator_shift", "shift during which the record was taken", ("day", "evening", "night")),
        ("sensor_vendor", "vendor of the installed sensor", ("vendor_a", "vendor_b", "vendor_c", "vendor_d")),
    ),
)

FINANCE = Pool(
    corr=(
        ("credit_score", "credit bureau score"),
        ("num_open_accounts", "number of open credit accounts"),
        ("months_at_address", "months at the current address"),
        ("loan_term_months", "requested loan term, months"),
        ("age_years", "applicant age, years"),
    ),
    pairs=(
        ("monthly_payment_k", "scheduled monthly payment, thousand NOK", "monthly_payment_usd",
         "scheduled monthly payment, USD", 94.0),
        ("account_age_months", "age of the oldest account, months", "account_age_years",
         "age of the oldest account, years", 1 / 12),
    ),
    noise=(
        ("num_late_payments", "late payments in the last 24 months"),
        ("utilisation_pct", "revolving credit utilisation, percent"),
        ("num_inquiries", "credit inquiries in the last 12 months"),
        ("dependents", "number of dependents"),
        ("savings_balance_k", "savings balance, thousand NOK"),
        ("num_cards", "number of payment cards held"),
        ("branch_distance_km", "distance to the nearest branch, km"),
        ("months_since_last_login", "months since the last online login"),
    ),
    cats=(
        ("loan_purpose", "stated purpose of the loan", ("car", "home", "education", "other")),
        ("region", "region of residence", ("north", "south", "east", "west")),
    ),
)

_COUNT_PREFIXES = ("num_", "dependents", "inspection_count")


def snap(x: np.ndarray, sig: int = 6) -> np.ndarray:
    """Round to ``sig`` significant digits *through the decimal string*, so the
    float is exactly what ``pd.read_csv`` returns for its printed form."""
    return np.array([float(f"{v:.{sig}g}") for v in np.asarray(x, dtype=float)])


def latent(x: np.ndarray) -> np.ndarray:
    """Rank-Gaussianised latent of a column (ties broken by average rank)."""
    return norm.ppf((rankdata(x) - 0.5) / len(x))


def _from_latent(rng: np.random.Generator, z: np.ndarray, is_count: bool) -> np.ndarray:
    kind = rng.integers(0, 3)
    if is_count:
        lam = float(rng.uniform(1.5, 8.0))
        u = norm.cdf(z)
        return np.floor(-lam * np.log1p(-np.clip(u, 1e-9, 1 - 1e-9))).astype(float)
    if kind == 0:  # lognormal
        return np.exp(rng.uniform(0.0, 4.0) + rng.uniform(0.3, 0.7) * z)
    if kind == 1:  # normal
        loc = rng.uniform(10, 100)
        return loc + loc * rng.uniform(0.1, 0.3) * z
    lo = rng.uniform(0, 50)  # uniform
    return lo + rng.uniform(10, 100) * norm.cdf(z)


@dataclass
class Assembled:
    frame: pd.DataFrame  # without id
    roles: dict[str, str]  # name -> relevant|corr_decoy|collinear|noise|noise_cat
    descriptions: dict[str, str]
    levels: dict[str, tuple[str, ...]] = field(default_factory=dict)  # categorical columns


def assemble(
    rng: np.random.Generator,
    n: int,
    relevant: dict[str, np.ndarray | pd.Series],
    relevant_desc: dict[str, str],
    pool: Pool,
    *,
    relevant_cats: dict[str, tuple[str, ...]] | None = None,
    n_cats: int | None = None,
    shuffle: bool = True,
) -> Assembled:
    """Add decoys/noise to the relevant columns and shuffle the column order.

    ``n_cats`` noise categoricals are taken from ``pool.cats`` (default: all)."""
    clash = set(relevant) & pool.all_names()
    if clash:
        raise ValueError(f"relevant column names clash with the decoy pool: {sorted(clash)}")
    cols: dict[str, np.ndarray] = {}
    roles: dict[str, str] = {}
    desc: dict[str, str] = dict(relevant_desc)
    levels: dict[str, tuple[str, ...]] = dict(relevant_cats or {})
    for name, vals in relevant.items():
        cols[name] = np.asarray(vals)
        roles[name] = "relevant"
    num_rel = [k for k, v in cols.items() if k not in levels and np.issubdtype(np.asarray(v).dtype, np.number)]
    zs = [latent(cols[k].astype(float)) for k in num_rel]

    for j, (name, d) in enumerate(pool.corr[:N_CORR]):
        r = j % len(zs)
        rho = float(rng.uniform(0.55, 0.85))
        z = rho * zs[r] + np.sqrt(1 - rho**2) * rng.standard_normal(n)
        cols[name] = _from_latent(rng, z, name.startswith(_COUNT_PREFIXES))
        roles[name], desc[name] = "corr_decoy", d
    for name_a, d_a, name_b, d_b, k in pool.pairs[:N_PAIRS]:
        base = _from_latent(rng, rng.standard_normal(n), False)
        cols[name_a] = base
        cols[name_b] = base * k * (1.0 + 0.012 * rng.standard_normal(n))
        roles[name_a] = roles[name_b] = "collinear"
        desc[name_a], desc[name_b] = d_a, d_b
    for name, d in pool.noise[:N_NOISE]:
        cols[name] = _from_latent(rng, rng.standard_normal(n), name.startswith(_COUNT_PREFIXES))
        roles[name], desc[name] = "noise", d
    for name, d, lv in pool.cats[: n_cats if n_cats is not None else len(pool.cats)]:
        p = rng.dirichlet(np.ones(len(lv)) * 3.0)
        cols[name] = np.asarray(lv)[rng.choice(len(lv), size=n, p=p)]
        roles[name], desc[name], levels[name] = "noise_cat", d, lv

    order = list(cols)
    if shuffle:
        order = [order[i] for i in rng.permutation(len(order))]
    out: dict[str, np.ndarray] = {}
    for name in order:
        v = cols[name]
        if name in levels:
            out[name] = v
        elif np.allclose(v, np.round(v)) and name.startswith(_COUNT_PREFIXES):
            out[name] = v.astype(int)
        else:
            out[name] = snap(v) if not np.issubdtype(v.dtype, np.integer) else v
    return Assembled(pd.DataFrame(out), roles, desc, levels)
