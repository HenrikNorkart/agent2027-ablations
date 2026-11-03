"""AGENT2027 final read-out (2026-11-02): every pre-registered contrast, the Holm family, and the outside-Holm
analyses, computed from the episode, row and funnel tables built on CAIR.

Binding rules: docs/agent2027/preregistration.md Part IV (family, margins, ITT) and Part V (per ablation), as amended
(Amendments 2-19 and the dated decisions). Two stages, because the GLMMs of record are fitted in R (lme4/logistf,
container ro27-rstats:4.4.1) and the exact tests here:

    python final_tests.py prep   --in <dir> --x6 X6_A08_llmfe_episodes.csv --out <dir>   # writes glmm_*.csv + specs
    Rscript final_glmm.R <dir>                                                            # writes glmm_results.json
    python final_tests.py report --in <dir> --x6 ... --rv3 RV3_A12_glmm.json --g2 g2_constants.json --out <dir>

Inputs in --in: episodes_final.csv (recovery_rows.py over every SYN-P row), rows_final.csv (final_rows.py),
funnel_episodes.csv (funnel.py; A17 rows), mflat/*.csv (manifests of record).

Row-of-record rule (pareto.py docstring, Amendments 11-19): pre-registered rows are row numbers < 100 without
smoke/pilot/repilot/cancelled notes; A17 and A19 use the 4xx relaunch rows (Amendments 17, 18); A20 keeps 0xx as
the pre-registered rows with 4xx as the comparable-harness sensitivity; 8xx power replicates are a labelled
sensitivity (Amendment 11 rule: computed after the pre-registered version, never replacing it); 5xx inf-guard
re-runs are an exploratory sensitivity that replaces the row they re-run.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

P6 = ["synp_p2", "synp_p3", "synp_p4", "synp_p5", "synp_p6a", "synp_p6b"]
D3 = ["synp_p3", "synp_p5", "synp_p2"]
SYNP7 = ["synp_p1"] + P6
LOG10_ACTIVE = {"f2_Q0.8": -0.097, "f2_Q2": 0.301, "f2_Q4": 0.602, "f2_Q9": 0.954, "f2_Q122": 1.000}
DELTA = 0.167
BAD_NOTES = ("smoke", "pilot", "repilot", "cancel", "probe")


# --------------------------------------------------------------------------------------------- helpers
def rownum(rid: str) -> int:
    return int(rid.split("-")[1])


def load_manifests(d: Path) -> pd.DataFrame:
    man = pd.concat([pd.read_csv(f, dtype=str) for f in sorted((d / "mflat").glob("*.csv"))], ignore_index=True)
    man["notes"] = man["notes"].fillna("")
    man["n"] = man["row_id"].map(rownum)
    man["bad"] = man["notes"].str.lower().str.contains("|".join(BAD_NOTES))
    return man


def wilson(k, n, z):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe_unpaired(k1, n1, k0, n0, level=0.95):
    """Newcombe (1998) method 10 for an unpaired difference p1 - p0 (Wilson score hybrid)."""
    z = stats.norm.ppf(0.5 + level / 2)
    p1, p0 = k1 / n1, k0 / n0
    l1, u1 = wilson(k1, n1, z)
    l0, u0 = wilson(k0, n0, z)
    d = p1 - p0
    return d, d - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)


def newcombe_paired(n11, n10, n01, n00, level=0.95):
    """Newcombe (1998) paired method 10: CI for p(B) - p(A) = (n01 - n10)/n with A in rows, B in columns."""
    n = n11 + n10 + n01 + n00
    z = stats.norm.ppf(0.5 + level / 2)
    pa, pb = (n11 + n10) / n, (n11 + n01) / n
    la, ua = wilson(n11 + n10, n, z)
    lb, ub = wilson(n11 + n01, n, z)
    d = pb - pa
    num = n11 * n00 - n10 * n01
    dd = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    if num > 0:
        num = max(num - n / 2, 0)
    phi = num / math.sqrt(dd) if dd > 0 else 0.0
    dl = math.sqrt((pb - lb) ** 2 - 2 * phi * (pb - lb) * (ua - pa) + (ua - pa) ** 2)
    du = math.sqrt((ub - pb) ** 2 - 2 * phi * (ub - pb) * (pa - la) + (pa - la) ** 2)
    return d, d - dl, d + du


def exact_stratified(tables):
    """Exact conditional test of a common OR = 1 over 2x2 strata, two-sided by the probability rule
    (Part V.14): tables = [(a_k successes in arm 1, n1_k, c_k successes in arm 0, n0_k)].
    Returns T = sum a_k, its null mean, and p = sum P(t) over t with P(t) <= P(T_obs)."""
    dist = {0: 1.0}
    t_obs = 0
    for a, n1, c, n0 in tables:
        m = a + c
        t_obs += a
        lo, hi = max(0, m - n0), min(n1, m)
        pk = {t: stats.hypergeom.pmf(t, n1 + n0, n1, m) for t in range(lo, hi + 1)}
        new = {}
        for s, ps in dist.items():
            for t, pt in pk.items():
                new[s + t] = new.get(s + t, 0.0) + ps * pt
        dist = new
    p_obs = dist.get(t_obs, 0.0)
    p = sum(v for v in dist.values() if v <= p_obs * (1 + 1e-7))
    mean = sum(k * v for k, v in dist.items())
    p_lo = sum(v for k, v in dist.items() if k <= t_obs)
    p_hi = sum(v for k, v in dist.items() if k >= t_obs)
    return {"T": t_obs, "E_T": mean, "p_two_sided": min(1.0, p), "p_lower": p_lo, "p_upper": p_hi}


def exact_ca_trend(strata):
    """Exact Cochran-Armitage trend test stratified: strata = [[(x_i, y_i), ...], ...]. Conditional on each
    stratum's success count, T = sum x_i y_i; null by convolution of the per-stratum subset-sum distributions.
    Two-sided: P(|T - E T| >= |t - E T|)."""
    dist = {0.0: 1.0}
    t_obs = 0.0
    for s in strata:
        xs = [round(x, 6) for x, _ in s]
        m = sum(y for _, y in s)
        t_obs += sum(x for x, y in s if y)
        combs = list(itertools.combinations(range(len(xs)), m))
        pk = {}
        for c in combs:
            v = round(sum(xs[i] for i in c), 6)
            pk[v] = pk.get(v, 0) + 1 / len(combs)
        new = {}
        for a, pa in dist.items():
            for b, pb in pk.items():
                k = round(a + b, 6)
                new[k] = new.get(k, 0) + pa * pb
        dist = new
    e = sum(k * v for k, v in dist.items())
    dev = abs(round(t_obs, 6) - e)
    p = sum(v for k, v in dist.items() if abs(k - e) >= dev - 1e-9)
    return {"T": t_obs, "E_T": e, "p_two_sided": min(1.0, p), "n_strata": len(strata),
            "n_informative_strata": sum(1 for s in strata if 0 < sum(y for _, y in s) < len(s))}


def rerun_map(rows, abl):
    """{row of record: latest re-run} for exploratory re-runs (Amendments 19-19f), following chains such as
    A17-442 <- A17-513 <- A17-521 <- A17-529 through the 're-run of' / 'retry of' notes; latest row number wins."""
    sub = rows[(rows.ablation_id == abl) & rows.n.between(500, 699)]
    parent = {}
    for _, r in sub.iterrows():
        m = re.search(r"(?:re-run|retry) of (A\d\d-\d+)", str(r.notes))
        if m:
            parent[r.row_id] = m.group(1)
    out = {}
    for rid in sorted(parent, key=rownum):
        root = parent[rid]
        while root in parent:
            root = parent[root]
        out[root] = rid
    return out


def holm(pv: dict) -> dict:
    keys = sorted(pv, key=lambda k: pv[k])
    m, run, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        run = max(run, (m - i) * pv[k])
        out[k] = min(1.0, run)
    return out


def sign_flip_p(y):
    y = np.asarray(y, float)
    obs = abs(y.mean())
    pats = list(itertools.product([1, -1], repeat=len(y)))
    return float(np.mean([abs((y * np.array(s)).mean()) >= obs - 1e-12 for s in pats]))


# --------------------------------------------------------------------------------------------- data
def load(ind: Path):
    man = load_manifests(ind)
    ep = pd.read_csv(ind / "episodes_final.csv").drop_duplicates(["row_id", "fold"], keep="last")
    rows = pd.read_csv(ind / "rows_final.csv")
    meta = man.set_index("row_id")[["notes", "n", "bad", "llm_seed"]]
    ep = ep.join(meta, on="row_id")
    rows = rows.drop(columns=[c for c in ("notes", "llm_seed") if c in rows]).join(meta, on="row_id")
    return man, ep, rows


def sel(ep, abl, arms=None, tasks=None, nums=None, seeds=None, reps=None):
    d = ep[(ep.ablation_id == abl) & (~ep.bad)]
    if arms is not None:
        d = d[d.arm_id.isin(arms)]
    if tasks is not None:
        d = d[d.dataset_id.isin(tasks)]
    if nums is not None:
        d = d[d.n.between(*nums)]
    if seeds is not None:
        d = d[d.fold_seed.isin(seeds)]
    if reps is not None:
        d = d[d.replicate.astype(int).isin(reps)]
    return d


def full_a05(ep, combos, power=False):
    """A05 Full fold-episodes for the (fold_seed, replicate) combos; power adds the 8xx anchors."""
    d = sel(ep, "A05", arms=["full"], tasks=P6, nums=(0, 99))
    d = d[d.apply(lambda r: (int(r.fold_seed), int(r.replicate)) in combos, axis=1)]
    if power:
        p = sel(ep, "A05", arms=["full"], tasks=P6, nums=(800, 899))
        d = pd.concat([d, p])
    return d


def pack(arm1: pd.DataFrame, arm0: pd.DataFrame, outcome="rec_sel", label1="arm", label0="full"):
    a = arm1.assign(arm=1, arm_name=label1)
    b = arm0.assign(arm=0, arm_name=label0)
    d = pd.concat([a, b], ignore_index=True)
    d["y"] = d[outcome].astype(int)
    d["task"] = d["dataset_id"]
    d["seed"] = d["fold_seed"].astype(str)
    d["task_seed"] = d["task"] + ":" + d["seed"]
    d["task_fold"] = d["task"] + ":" + d["fold"].astype(str)
    d["run"] = d["row_id"]
    return d[["row_id", "arm", "arm_name", "task", "seed", "fold", "task_seed", "task_fold", "run", "y", "itt_imputed"]]


def describe(d):
    k1, n1 = int(d[d.arm == 1].y.sum()), int((d.arm == 1).sum())
    k0, n0 = int(d[d.arm == 0].y.sum()), int((d.arm == 0).sum())
    rd, l95, u95 = newcombe_unpaired(k1, n1, k0, n0, 0.95) if n1 and n0 else (np.nan,) * 3
    _, l90, u90 = newcombe_unpaired(k1, n1, k0, n0, 0.90) if n1 and n0 else (np.nan,) * 3
    tabs = []
    for t, g in d.groupby("task"):
        tabs.append((int(g[g.arm == 1].y.sum()), int((g.arm == 1).sum()), int(g[g.arm == 0].y.sum()), int((g.arm == 0).sum())))
    ex = exact_stratified(tabs)
    per_task = {t: f"{int(g[g.arm==1].y.sum())}/{int((g.arm==1).sum())} vs {int(g[g.arm==0].y.sum())}/{int((g.arm==0).sum())}"
                for t, g in d.groupby("task")}
    sep = (k1 in (0, n1)) or (k0 in (0, n0))
    return {"k1": k1, "n1": n1, "k0": k0, "n0": n0, "rate1": k1 / n1 if n1 else None, "rate0": k0 / n0 if n0 else None,
            "rd": rd, "rd_ci95": [l95, u95], "rd_ci90": [l90, u90], "exact_task": ex, "per_task": per_task,
            "separation_raw": bool(sep), "itt_runs1": int(d[(d.arm == 1)].groupby("run").itt_imputed.max().sum()),
            "itt_runs0": int(d[(d.arm == 0)].groupby("run").itt_imputed.max().sum())}


# --------------------------------------------------------------------------------------------- prep
def prep(a):
    ind, out = Path(a.indir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man, ep, rows = load(ind)
    specs, sets = [], {}
    FULL3 = {(42, 0), (42, 1), (4, 0)}

    def add(cid, d, formula, term="arm", lrt=False, power=False):
        f = out / f"glmm_{cid}.csv"
        d.to_csv(f, index=False)
        specs.append({"id": cid, "file": f.name, "formula": formula, "term": term, "lrt": lrt})
        sets[cid] = describe(d) if "arm" in d and d["arm"].nunique() == 2 else {}

    # C1a  SET-phased vs Full (A16), SYN-P 7 minus P1 x {42, 4}
    for pw in (False, True):
        s = sel(ep, "A16", arms=["set_phased"], tasks=P6, nums=(0, 99) if not pw else (0, 899))
        add("C1a" + ("_power" if pw else ""), pack(s, full_a05(ep, FULL3, pw), label1="set_phased"),
            "y ~ arm + (1|task) + (1|task_seed)")
    # C1b  LLM-FE vs Full (A08), seed-42 partition, Full r0 + r1, three modes
    x6 = pd.read_csv(a.x6)
    x6 = x6[x6.dataset_id.isin(P6)]
    full42 = full_a05(ep, {(42, 0), (42, 1)})
    for mode, outc in (("best3", "rec_sel"), ("notebook", "rec_sel"), ("any", "rec_any")):
        l = x6[x6.arm.str.endswith(mode)].copy()
        l["row_id"] = "LLMFE-" + l.arm.str.extract(r"(r\d)")[0] + "-" + l.dataset_id
        l["itt_imputed"] = 0
        l["fold_seed"] = 42
        fo = full42.copy()
        add(f"C1b_{mode}", pack(l, fo, outcome=outc, label1=f"llmfe_{mode}"), "y ~ arm + (1|task) + (1|task_fold)")
    # C3  restarts vs Full r0 (A10), D3 population, task fixed, LR test
    for pw in (False, True):
        r_ = sel(ep, "A10", arms=["restarts"], tasks=D3, nums=(0, 99) if not pw else (0, 899))
        f_ = sel(ep, "A05", arms=["full"], tasks=D3, nums=(0, 99) if not pw else (0, 899))
        f_ = f_[(f_.replicate.astype(int) == 0) | (f_.n >= 800)] if pw else f_[f_.replicate.astype(int) == 0]
        add("C3" + ("_power" if pw else ""), pack(r_, f_, label1="restarts"), "y ~ arm + task + (1|task_seed)", lrt=True)
    # A11 L0 vs Full r0 (outside Holm), same population and model as C3
    l0 = sel(ep, "A11", arms=["l0"], tasks=D3, nums=(0, 99))
    f_ = sel(ep, "A05", arms=["full"], tasks=D3, nums=(0, 99))
    add("A11", pack(l0, f_[f_.replicate.astype(int) == 0], label1="l0"), "y ~ arm + task + (1|task_seed)", lrt=True)
    # A09 oracle / misdirected vs Full (outside Holm, G3)
    for arm in ("oracle", "misdirected"):
        s = sel(ep, "A09", arms=[arm], tasks=P6, nums=(0, 99))
        add(f"A09_{arm}", pack(s, full_a05(ep, FULL3), label1=arm), "y ~ arm + (1|task) + (1|task_seed)")
    # C5b  mix_esmall vs f2_Q122 (A19 4xx vs A17 4xx)
    q122 = sel(ep, "A17", arms=["f2_Q122"], tasks=P6, nums=(400, 499))
    for arm in ("mix_esmall", "mix_1b"):
        s = sel(ep, "A19", arms=[arm], tasks=P6, nums=(400, 499))
        add(f"C5b_{arm}" if arm == "mix_esmall" else "A19_mix_1b", pack(s, q122, label1=arm, label0="f2_Q122"),
            "y ~ arm + (1|task) + (1|task_seed)")
    # A20  harness x rung interaction (outside Holm): Full/minimal at O120 and at Q4, seed 42, P6
    o_full = full_a05(ep, {(42, 0)})
    o_min = sel(ep, "A12", arms=["fx_pajM"], tasks=P6, nums=(0, 99), seeds=[42])
    q_full = sel(ep, "A17", arms=["f2_Q4"], tasks=P6, nums=(400, 499), seeds=[42])
    for lab, rng in (("A20", (0, 99)), ("A20_relaunch", (400, 499))):
        q_min = sel(ep, "A20", arms=["fx_pajM_Q4"] if rng[0] == 0 else ["fx_pajM_Q4", "fx_pajM_Q4_cap12"], tasks=P6, nums=rng, seeds=[42])
        parts = []
        for H, R, dd in ((1, 1, o_full), (-1, 1, o_min), (1, -1, q_full), (-1, -1, q_min)):
            x = pack(dd, dd.iloc[0:0]).assign(H=H, R=R)
            parts.append(x)
        d = pd.concat(parts, ignore_index=True)
        f = out / f"glmm_{lab}.csv"
        d.to_csv(f, index=False)
        specs.append({"id": lab, "file": f.name, "formula": "y ~ H * R + (1|task)", "term": "H:R", "lrt": False})
        sets[lab] = {f"{h}{r}": f"{int(g.y.sum())}/{len(g)}" for (h, r), g in d.groupby(["H", "R"])}
    # C5a: run-level func_success (fold rule 3/5) and the Firth sensitivity input
    fun = pd.read_csv(ind / "funnel_episodes.csv")
    r17 = rows[(rows.ablation_id == "A17") & rows.n.between(400, 499) & rows.dataset_id.isin(SYNP7) & rows.arm_id.isin(LOG10_ACTIVE)]
    fs = func_success(r17, fun, 3).merge(r17[["row_id", "arm_id", "dataset_id", "fold_seed"]], on="row_id")
    fs["x"] = fs.arm_id.map(LOG10_ACTIVE)
    fs.to_csv(out / "c5a_func_success.csv", index=False)
    fs.rename(columns={"func_success": "y", "dataset_id": "task", "fold_seed": "seed"}).to_csv(out / "glmm_C5a_firth.csv", index=False)
    specs.append({"id": "C5a_firth", "file": "glmm_C5a_firth.csv", "formula": "y ~ x + task + factor(seed)", "term": "x", "lrt": False, "firth": True})
    (out / "glmm_specs.json").write_text(json.dumps(specs, indent=1))
    (out / "describe.json").write_text(json.dumps(sets, indent=1, default=float))
    print(f"prep: {len(specs)} GLMM specs")


# --------------------------------------------------------------------------------------------- C5a
def func_success(rows, fun, rule=3):
    """A17 run-level func_success: run finished (summary present) and, in >= `rule` of 5 folds, the selected
    trial's held-out score passes the success bound AND >= 1 accepted non-fallback LLM feature is active in it
    (funnel fun_selected >= 1). Capped / failed runs are 0 (ITT)."""
    out = []
    for _, r in rows.iterrows():
        ok = int(r.row_ok) == 1 and int(r.cap_hit) == 0
        n_pass = 0
        if ok and pd.notna(r.base_mean):
            for f in range(5):
                sc = r.get(f"fold{f}_score")
                if pd.isna(sc):
                    continue
                good = sc >= r.base_mean - 2 * r.base_sd if r.task_type == "classification" else sc <= r.base_mean + 2 * r.base_sd
                fs = fun[(fun.row_id == r.row_id) & (fun.fold == f)]
                nsel = int(fs.fun_selected.iloc[0]) if len(fs) else 0
                n_pass += int(good and nsel >= 1)
        out.append({"row_id": r.row_id, "n_folds_pass": n_pass, "func_success": int(ok and n_pass >= rule)})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------------- report
def report(a):
    ind, out = Path(a.indir), Path(a.out)
    man, ep, rows = load(ind)
    glmm = json.loads((out / "glmm_results.json").read_text()) if (out / "glmm_results.json").exists() else {}
    desc = json.loads((out / "describe.json").read_text())
    rv3 = json.loads(Path(a.rv3).read_text())
    g2 = json.loads(Path(a.g2).read_text())
    res = {"generated": pd.Timestamp.utcnow().isoformat(), "delta": DELTA, "contrasts": {}, "outside_holm": {}}

    def glmm_or_exact(cid, one_sided=None):
        g = glmm.get(cid, {})
        d = desc.get(cid, {})
        msgs = g.get("conv_messages") or []
        msgs = [msgs] if isinstance(msgs, str) else msgs
        conv_fail = (not g) or g.get("note") is not None or any(not str(m).startswith("boundary (singular)") for m in msgs)
        sep = bool(d.get("separation_raw")) or conv_fail or abs(g.get("estimate", 0) or 0) > 10
        ex = d.get("exact_task", {})
        if sep:
            p = ex.get("p_two_sided")
            if one_sided == "lower":
                p = ex.get("p_lower")
            elif one_sided == "upper":
                p = ex.get("p_upper")
            how = "exact conditional test stratified by task (separation / non-convergence fallback)"
        else:
            p = g.get("p_lrt") if g.get("p_lrt") is not None else g.get("p_wald")
            if one_sided:
                z = (g["estimate"] or 0) / g["se"]
                p = stats.norm.cdf(z) if one_sided == "lower" else stats.norm.sf(z)
            how = "GLMM " + g.get("formula", "") + (" (LR test)" if g.get("p_lrt") is not None else " (Wald)")
        return {"p": p, "test": how, "glmm": g, "desc": d}

    C = {}
    C["C1a"] = glmm_or_exact("C1a")
    C["C1a"]["sensitivity_power"] = glmm_or_exact("C1a_power")
    modes = {m: glmm_or_exact(f"C1b_{m}") for m in ("best3", "notebook", "any")}
    C["C1b"] = dict(modes["best3"])
    C["C1b"]["modes"] = {m: {"rd": v["desc"].get("rd"), "p": v["p"], "rates": [v["desc"].get("rate1"), v["desc"].get("rate0")]} for m, v in modes.items()}
    signs = [np.sign(v["desc"].get("rd") or 0) for v in modes.values()]
    C["C1b"]["three_mode_agreement"] = bool(all(s == signs[0] for s in signs) and signs[0] != 0)
    for k, lab in (("C2a", "A"), ("C2b", "J"), ("C2c", "P")):
        e = rv3["effects"][lab]
        C[k] = {"p": e["p_tost"] if k != "C2c" else e["p_wald"], "test": "RV3 12-cell beta-binomial (analysis of record, G4 addendum)" + (", TOST +-0.167" if k != "C2c" else ", two-sided Wald"),
                "Delta": e["Delta"], "Delta_ci90": e["Delta_ci90"], "Delta_ci95": e["Delta_ci95"], "p_wald": e["p_wald"], "p_tost": e["p_tost"]}
    C["C3"] = glmm_or_exact("C3")
    C["C3"]["sensitivity_power"] = glmm_or_exact("C3_power")

    # C4a  A14 Local-correct vs Off on P7, exact conditional by fold seed
    k = sel(ep, "A14", tasks=["synp_p7"], nums=(0, 99))
    tabs, per = [], {}
    for s, g in k.groupby("fold_seed"):
        a1 = g[g.arm_id == "kb_local_correct"]
        a0 = g[g.arm_id == "kb_off"]
        if len(a1) and len(a0):
            tabs.append((int(a1.rec_sel.sum()), len(a1), int(a0.rec_sel.sum()), len(a0)))
            per[int(s)] = f"{int(a1.rec_sel.sum())}/{len(a1)} vs {int(a0.rec_sel.sum())}/{len(a0)}"
    ex = exact_stratified(tabs)
    arms_k = {arm: f"{int(g.rec_sel.sum())}/{len(g)}" for arm, g in k.groupby("arm_id")}
    k1 = sum(t[0] for t in tabs); n1 = sum(t[1] for t in tabs); k0 = sum(t[2] for t in tabs); n0 = sum(t[3] for t in tabs)
    rd = newcombe_unpaired(k1, n1, k0, n0) if n1 and n0 else (None, None, None)
    sec = {}
    for other in ("kb_parametric", "kb_local_wrong"):
        tb = []
        for s, g in k.groupby("fold_seed"):
            a1, a0 = g[g.arm_id == "kb_local_correct"], g[g.arm_id == other]
            if len(a1) and len(a0):
                tb.append((int(a1.rec_sel.sum()), len(a1), int(a0.rec_sel.sum()), len(a0)))
        sec[other] = exact_stratified(tb) if tb else None
    C["C4a"] = {"p": ex["p_two_sided"], "test": "exact conditional test stratified by fold seed (Part V.14)", "exact": ex,
                "per_seed": per, "arms": arms_k, "rd": rd[0], "rd_ci95": rd[1:], "H2_secondary": sec}

    # C4b  A15 on A05 data: P6b - P6a, 15 pairs (fold seed, replicate, fold), exact McNemar
    a5 = sel(ep, "A05", arms=["full"], tasks=["synp_p6a", "synp_p6b"], nums=(0, 99))
    a5 = a5[a5.apply(lambda r: (int(r.fold_seed), int(r.replicate)) in FULL3_SET, axis=1)]
    pa = a5[a5.dataset_id == "synp_p6a"].set_index(["fold_seed", "replicate", "fold"]).rec_sel
    pb = a5[a5.dataset_id == "synp_p6b"].set_index(["fold_seed", "replicate", "fold"]).rec_sel
    jj = pd.concat([pa.rename("a"), pb.rename("b")], axis=1).dropna()
    n11 = int(((jj.a == 1) & (jj.b == 1)).sum()); n10 = int(((jj.a == 1) & (jj.b == 0)).sum())
    n01 = int(((jj.a == 0) & (jj.b == 1)).sum()); n00 = int(((jj.a == 0) & (jj.b == 0)).sum())
    disc = n10 + n01
    pm = float(min(1.0, 2 * stats.binom.cdf(min(n10, n01), disc, 0.5))) if disc else 1.0
    d, lo, hi = newcombe_paired(n11, n10, n01, n00)
    C["C4b"] = {"p": pm, "test": "exact McNemar, 15 pairs (Part V.15)", "n_pairs": len(jj), "table": [n11, n10, n01, n00],
                "rate_p6a": float(jj.a.mean()), "rate_p6b": float(jj.b.mean()), "delta": d, "delta_ci95": [lo, hi]}

    # C5a  func_success trend over the Qwen rungs (A17 4xx rows of record)
    fun = pd.read_csv(ind / "funnel_episodes.csv")
    r17 = rows[(rows.ablation_id == "A17") & rows.n.between(400, 499) & rows.dataset_id.isin(SYNP7) & rows.arm_id.isin(LOG10_ACTIVE)]
    res_c5 = {}
    for rule in (3, 5, 1):
        fs = func_success(r17, fun, rule).merge(r17[["row_id", "arm_id", "dataset_id", "fold_seed"]], on="row_id")
        strata = [[(LOG10_ACTIVE[r.arm_id], int(r.func_success)) for _, r in g.iterrows()] for _, g in fs.groupby(["dataset_id", "fold_seed"])]
        res_c5[f"rule_{rule}of5"] = {"exact": exact_ca_trend(strata),
                                     "by_rung": {arm: f"{int(g.func_success.sum())}/{len(g)}" for arm, g in fs.groupby("arm_id")}}
    # first-attempt sensitivity: a row whose first attempt was not OK counts 0
    fa = r17[r17.first_status == "OK"]
    fs1 = func_success(r17, fun, 3).merge(r17[["row_id", "arm_id", "dataset_id", "fold_seed", "first_status"]], on="row_id")
    fs1.loc[fs1.first_status != "OK", "func_success"] = 0
    strata = [[(LOG10_ACTIVE[r.arm_id], int(r.func_success)) for _, r in g.iterrows()] for _, g in fs1.groupby(["dataset_id", "fold_seed"])]
    res_c5["first_attempt"] = {"exact": exact_ca_trend(strata), "by_rung": {arm: f"{int(g.func_success.sum())}/{len(g)}" for arm, g in fs1.groupby("arm_id")}}
    # inf-guard sensitivity (5xx replace the row they re-run)
    r5 = rows[(rows.ablation_id == "A17") & rows.n.between(500, 599)]
    fun5 = pd.concat([pd.read_csv(p) for p in (ind / "funnel_episodes_5xx.csv", ind / "funnel_episodes_late.csv") if p.exists()]
                     or [fun.iloc[0:0]], ignore_index=True)
    rep_map = rerun_map(rows, "A17")
    if rep_map:
        swap = r17.copy()
        for orig, new in rep_map.items():
            if orig in set(swap.row_id) and new in set(r5.row_id):
                nr = r5[r5.row_id == new].iloc[0].copy()
                nr["arm_id"] = swap[swap.row_id == orig].arm_id.iloc[0]
                swap = pd.concat([swap[swap.row_id != orig], nr.to_frame().T], ignore_index=True)
        fs5 = func_success(swap, pd.concat([fun, fun5]), 3).merge(swap[["row_id", "arm_id", "dataset_id", "fold_seed"]], on="row_id")
        strata = [[(LOG10_ACTIVE[r.arm_id], int(r.func_success)) for _, r in g.iterrows()] for _, g in fs5.groupby(["dataset_id", "fold_seed"])]
        res_c5["inf_guard"] = {"replaced": rep_map, "exact": exact_ca_trend(strata),
                               "by_rung": {arm: f"{int(g.func_success.sum())}/{len(g)}" for arm, g in fs5.groupby("arm_id")}}
    firth = glmm.get("C5a_firth", {})
    C["C5a"] = {"p": res_c5["rule_3of5"]["exact"]["p_two_sided"], "test": "exact Cochran-Armitage trend stratified by task x seed (Part V.17)",
                **res_c5, "firth": firth}
    C["C5b"] = glmm_or_exact("C5b_mix_esmall")
    # C5b inf-guard sensitivity (Amendments 19, 19c): every re-run replaces its row of record in both arms
    m17, m19 = rerun_map(rows, "A17"), rerun_map(rows, "A19")
    def swap_eps(e_rec, mp, arm):
        out = e_rec
        for orig, new in mp.items():
            if orig in set(e_rec.row_id):
                ne = ep[ep.row_id == new].copy()
                if len(ne):
                    ne["arm_id"] = arm
                    out = pd.concat([out[out.row_id != orig], ne], ignore_index=True)
        return out
    s_es = swap_eps(sel(ep, "A19", arms=["mix_esmall"], tasks=P6, nums=(400, 499)), m19, "mix_esmall")
    s_q = swap_eps(sel(ep, "A17", arms=["f2_Q122"], tasks=P6, nums=(400, 499)), m17, "f2_Q122")
    C["C5b"]["inf_guard"] = {"replaced_A19": {k: v for k, v in m19.items() if k in set(sel(ep, "A19", arms=["mix_esmall"], nums=(400, 499)).row_id)},
                             "replaced_A17_Q122": {k: v for k, v in m17.items() if k in set(sel(ep, "A17", arms=["f2_Q122"], nums=(400, 499)).row_id)},
                             "desc": describe(pack(s_es, s_q, label1="mix_esmall", label0="f2_Q122"))}
    s_1b = swap_eps(sel(ep, "A19", arms=["mix_1b"], tasks=P6, nums=(400, 499)), m19, "mix_1b")
    C["C5b"]["inf_guard_mix_1b"] = describe(pack(s_1b, s_q, label1="mix_1b", label0="f2_Q122"))
    # Amendment 19e retries of A12 ITT rows (exploratory; the RV3 analysis of record is unchanged)
    res["a12_retries"] = {r.row_id: {"retries": str(r.notes)[:80], "status": r.final_status,
                                     "rec_sel": f"{int(ep[ep.row_id == r.row_id].rec_sel.sum())}/{len(ep[ep.row_id == r.row_id])}"}
                          for _, r in rows[(rows.ablation_id == "A12") & rows.n.between(781, 799)].iterrows()}

    pv = {k: float(v["p"]) for k, v in C.items()}
    hp = holm(pv)
    for k in C:
        C[k]["p_holm"] = hp[k]
        C[k]["holm_reject"] = bool(hp[k] < 0.05)
    res["contrasts"] = C

    # ----------------------------------------------------------------------------- outside Holm
    O = {}
    O["A09_oracle"] = glmm_or_exact("A09_oracle", one_sided="upper")
    O["A09_misdirected"] = glmm_or_exact("A09_misdirected", one_sided="lower")
    r1 = O["A09_misdirected"]["p"] < 0.025
    r2 = (O["A09_oracle"]["p"] < 0.025) and not r1
    O["G3_planner_half"] = {"route1_misdirected_lt_full": bool(r1), "route2_oracle_gt_full": bool(r2), "pass": bool(r1 or r2)}
    O["A11"] = glmm_or_exact("A11")
    O["A19_mix_1b"] = glmm_or_exact("A19_mix_1b")
    O["A20"] = {"test": "Firth logistic with task fixed effects (zero cell, GLMM did not converge)", "firth": glmm.get("A20_firth"), "glmm": glmm.get("A20"), "cells": desc.get("A20")}
    O["A20_relaunch"] = {"test": "Firth logistic with task fixed effects", "firth": glmm.get("A20_relaunch_firth"), "glmm": glmm.get("A20_relaunch"), "cells": desc.get("A20_relaunch")}

    # A07 HC1: log-ratio of errors per accepted, hidden vs shown (pairs with A05 full r0, same dataset and seed)
    a7 = rows[(rows.ablation_id == "A07") & (~rows.bad)]
    pairs = []
    for _, r in a7.iterrows():
        f = rows[(rows.ablation_id == "A05") & (rows.arm_id == "full") & (rows.dataset_id == r.dataset_id) & (rows.fold_seed == r.fold_seed) & (rows.replicate.astype(int) == 0) & (rows.n < 100)]
        if not len(f):
            continue
        f = f.iloc[0]
        complete = int(r.row_ok) == 1 and int(f.row_ok) == 1 and pd.notna(r.n_tool_errors) and pd.notna(f.n_tool_errors)
        y = None
        if complete:
            y = math.log((r.n_tool_errors + .5) / (r.n_accepted + .5)) - math.log((f.n_tool_errors + .5) / (f.n_accepted + .5))
        pairs.append({"dataset": r.dataset_id, "seed": int(r.fold_seed), "complete": complete, "y": y,
                      "e_hid": r.n_tool_errors, "a_hid": r.n_accepted, "e_shw": f.n_tool_errors, "a_shw": f.n_accepted,
                      "itt_hid": int(r.itt), "itt_shw": int(f.itt)})
    pp = pd.DataFrame(pairs)
    a7r = {"pairs": pairs}
    cp = pp[pp.complete] if len(pp) else pp
    if len(cp):
        yd = cp.groupby("dataset").y.mean()
        D = len(yd)
        mu = float(cp.y.mean())
        if D >= 2:
            half = stats.t.ppf(0.975, D - 1) * yd.std(ddof=1) / math.sqrt(D)
            ci = [float(yd.mean() - half), float(yd.mean() + half)]
        else:
            ci = [None, None]
        rho = math.exp(-mu)
        rho_ci = [math.exp(-ci[1]), math.exp(-ci[0])] if ci[0] is not None else [None, None]
        p_perm = sign_flip_p(cp.y.values)
        med_err = float(cp.e_shw.median())
        harm = int((pp.itt_hid.sum()) - (pp.itt_shw.sum()))
        if len(cp) < 6:
            verdict = "too few complete pairs: descriptive only"
        elif med_err < 20:
            verdict = "channel rarely exercised in v2"
        elif rho <= 0.70 and rho_ci[1] < 1 and p_perm < 0.05:
            verdict = "supported: showing error text reduces errors per accepted feature"
        elif rho_ci[1] < 1:
            verdict = "shown text reduces errors, smaller or less robust"
        elif rho_ci[0] > 1:
            verdict = "opposite: error text increased errors"
        else:
            verdict = f"inconclusive: no detectable effect at {len(cp)} pairs"
        a7r.update({"n_complete": int(len(cp)), "D": D, "mu": mu, "mu_ci95": ci, "rho_shown_over_hidden": rho, "rho_ci95": rho_ci,
                    "p_signflip": p_perm, "median_errors_control": med_err, "harm_flag_diff": harm, "verdict": verdict})
    O["A07"] = a7r

    # A13 KT1-Off: cost ratio and no-harm on Core-10 (control = mean of A05 f42 r0 + r1)
    sig = g2["sigma_d"]
    a13 = rows[(rows.ablation_id == "A13") & (~rows.bad) & (rows.n < 100)]
    recs = []
    for _, r in a13.iterrows():
        f = rows[(rows.ablation_id == "A05") & (rows.arm_id == "full") & (rows.dataset_id == r.dataset_id) & (rows.fold_seed == 42) & (rows.replicate.astype(int).isin([0, 1])) & (rows.n < 100)]
        if len(f) < 2:
            continue
        ft = f.tok_compl_total.astype(float).mean()
        fs_ = f.heldout_score.astype(float).mean()
        sc = r.heldout_score if int(r.itt) == 0 else r.base_mean
        dlt = (sc - fs_) if r.task_type == "classification" else (fs_ - sc)
        recs.append({"dataset": r.dataset_id, "ratio": float(r.tok_compl_total) / ft if pd.notna(r.tok_compl_total) else None,
                     "delta": dlt, "delta_std": dlt / sig.get(r.dataset_id, np.nan), "itt": int(r.itt)})
    rr = pd.DataFrame(recs)
    a13r = {"rows": recs}
    if len(rr):
        lr = np.log(rr.ratio.dropna())
        n = len(lr)
        gm = float(np.exp(lr.mean()))
        h = stats.t.ppf(0.975, n - 1) * lr.std(ddof=1) / math.sqrt(n)
        ci = [float(np.exp(lr.mean() - h)), float(np.exp(lr.mean() + h))]
        p_sf = sign_flip_p(lr.values)
        if ci[0] > 1:
            kv = "K1 more expensive"
        elif ci[1] < 1 and gm <= 0.95:
            kv = "K2 clearly cheaper"
        elif gm > 0.95:
            kv = "K3 small cost centre"
        else:
            kv = "K4 inconclusive"
        md = float(rr.delta_std.mean())
        lo, hi = md - 0.637, md + 0.637
        nh = "no harm beyond M" if lo > -1 else ("harm" if hi < -1 else "inconclusive")
        a13r.update({"tok_ratio_gm": gm, "tok_ratio_ci95": ci, "p_signflip": p_sf, "cost_verdict": kv,
                     "mean_delta_std": md, "nh_bounds": [lo, hi], "no_harm_verdict": nh, "n": n})
    O["A13"] = a13r

    # A17 descriptive per rung (Part V.17, A17-T16), and the functional threshold
    desc17 = {}
    for arm, g in r17.groupby("arm_id"):
        e = ep[ep.row_id.isin(g.row_id)]
        fsg = pd.read_csv(out / "c5a_func_success.csv")
        fsg = fsg[fsg.row_id.isin(g.row_id)]
        desc17[arm] = {"runs": int(len(g)), "ok": int(g.row_ok.sum()), "cap_hit": int(g.cap_hit.sum()),
                       "failed": int(((g.row_ok == 0) & (g.cap_hit == 0)).sum()),
                       "inf_crash": int(g.final_reason.astype(str).str.contains("no-fold_aggregation").sum()),
                       "success": int(g.success.sum()), "func_success": int(fsg.func_success.sum()),
                       "rec_sel": f"{int(e.rec_sel.sum())}/{len(e)}", "rec_sel_rate": float(e.rec_sel.mean()) if len(e) else None,
                       "tok_compl_median": float(g.tok_compl_total.median()) if g.tok_compl_total.notna().any() else None,
                       "loop_guard_median": float(g.loop_guard.median()) if "loop_guard" in g and g.loop_guard.notna().any() else None,
                       "llm_calls_median": float(g.llm_calls_total.median()) if g.llm_calls_total.notna().any() else None}
    order = ["f2_Q0.8", "f2_Q2", "f2_Q4", "f2_Q9", "f2_Q122"]
    thr = None
    for i, rung in enumerate(order):
        d_ = desc17.get(rung)
        if d_ and all((desc17.get(o, {}).get("func_success", 0) / max(desc17.get(o, {}).get("runs", 1), 1)) >= 0.8 for o in order[i:]):
            thr = rung
            break
    O["A17_descriptive"] = {"by_rung": desc17, "functional_threshold": thr or "no functional rung at this cap"}

    # A16 engineering-collapse overlay; A19 big-token ratio
    s16 = rows[(rows.ablation_id == "A16") & (rows.arm_id == "set_phased") & rows.dataset_id.isin(SYNP7) & (rows.n < 100) & (~rows.bad)]
    O["A16_overlay"] = {"set_synp_runs": int(len(s16)), "itt_runs": int(s16.itt.sum()), "overlay_fires": bool(s16.itt.sum() >= 3)}

    # costs per arm (median completion tokens), for the design table
    cost = {}
    for (abl, arm), g in rows[rows.dataset_id.isin(SYNP7) & (~rows.bad)].groupby(["ablation_id", "arm_id"]):
        if g.tok_compl_total.notna().any():
            cost[f"{abl}:{arm}"] = {"runs": int(len(g)), "tok_compl_median": float(g.tok_compl_total.median()),
                                    "itt": int(g.itt.sum()), "cap_hit": int(g.cap_hit.sum())}
    O["cost_synp"] = cost
    res["outside_holm"] = O
    (out / "final_readout.json").write_text(json.dumps(res, indent=1, default=lambda x: None if x is None or (isinstance(x, float) and math.isnan(x)) else (float(x) if isinstance(x, (np.floating, np.integer)) else str(x))))
    tab = pd.DataFrame([{"contrast": k, "test": v["test"], "p_raw": v["p"], "p_holm": v["p_holm"], "reject": v["holm_reject"]} for k, v in C.items()])
    tab.to_csv(out / "holm_table.csv", index=False)
    print(tab.to_string(index=False))


FULL3_SET = {(42, 0), (42, 1), (4, 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "report"])
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--x6", required=True)
    ap.add_argument("--rv3")
    ap.add_argument("--g2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prep(a) if a.cmd == "prep" else report(a)


if __name__ == "__main__":
    main()
