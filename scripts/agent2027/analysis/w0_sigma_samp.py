"""A01 (W0): LLM-sampling variance of the old image, from 16 t17-partnered replicate runs.

Spec: docs/ablations/A01_W0_old_image_replicate.md, sections 2 (Definitions), 3 and 6.
Each W0 run re-ran t17's exact system (rogue_one:nestedcv9, same vLLM image, same launchers,
same fold seed) on one of the W0-8 datasets, so it shares its outer-fold partition with its
t17 partner. The pair difference therefore carries LLM sampling (plus live-search drift)
only, while t17's seed SD carries sampling and partition together.

Only numpy, pandas, scipy and wandb are imported, so this runs inside rogue_one:nestedcv9:

    python3 w0_sigma_samp.py --selftest
    python3 w0_sigma_samp.py --w0-long ... --t17-long ... --t17-sd ... --baselines ... \
        --logs DIR --w0-roots /w0/s42,/w0/s4 --t17-roots /t17/s42,/t17/s4 --out DIR [--wandb]

Every pooled number, both drift tests, the verdict and every section 6 rule use the W0-8
pairs only. The optional balance-scale and wine rows (if present) go to a separate
descriptive table.
"""
import argparse
import datetime as _dt
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats

W0_8 = ["credit-g", "heart", "junglechess", "myocardial", "vehicle", "crab", "housing", "insurance"]
OPTIONAL = ["balance-scale", "wine"]
REG = {"crab", "housing", "insurance", "wine"}
REG3 = ["crab", "housing", "insurance"]
SEEDS = [42, 4]

# section 3 / section 6 thresholds, pre-declared Wed 7 Oct
HOPED_MAX = 0.36
NULL_MIN = 0.72
DRIFT_ALPHA = 0.05
IDENTICAL_LOWER_BOUND_MIN = 3
USABLE_FULL, USABLE_MIN = 12, 8


def task_type(ds):
    return "regression" if ds in REG else "classification"


# ----------------------------------------------------------------------------- statistics

def sigma_samp_sq(deltas):
    """sigma^2_samp,d = sum(delta^2) / (2 k): a pair difference of two draws has variance 2 sigma^2."""
    d = np.asarray(deltas, dtype=float)
    return float(np.sum(d ** 2) / (2 * len(d))) if len(d) else float("nan")


def satterthwaite(w, k):
    """Effective df of sum(v_d) (k_d pairs each) and sum(w_d) (2 df each), weighted by w = t17 variances."""
    w, k = np.asarray(w, float), np.asarray(k, float)
    s = w.sum()
    return float(s ** 2 / np.sum(w ** 2 / k)), float(s ** 2 / np.sum(w ** 2 / 2.0))


def share_with_ci(v, w, k):
    """Z_v1 = sum v / sum w, with the F-based CI at Satterthwaite df and the nominal-df version."""
    v, w, k = np.asarray(v, float), np.asarray(w, float), np.asarray(k, float)
    z = float(v.sum() / w.sum())
    nu1, nu2 = satterthwaite(w, k)
    nn1, nn2 = float(k.sum()), 2.0 * len(k)
    out = dict(
        z=z, r_raw=math.sqrt(z), nu1=nu1, nu2=nu2,
        ci_low=z / stats.f.ppf(0.975, nu1, nu2), ci_high=z * stats.f.ppf(0.975, nu2, nu1),
        var_inflation_threshold=float(stats.f.ppf(0.975, nu1, nu2)),
        nu1_nominal=nn1, nu2_nominal=nn2,
        ci_low_nominal=z / stats.f.ppf(0.975, nn1, nn2), ci_high_nominal=z * stats.f.ppf(0.975, nn2, nn1),
        var_inflation_threshold_nominal=float(stats.f.ppf(0.975, nn1, nn2)),
    )
    out["var_inflation"] = bool(z > out["var_inflation_threshold"])
    return out


def wilcoxon_exact(z):
    """Two-sided exact signed-rank p by enumerating all 2^n sign flips (mid-ranks; zeros dropped)."""
    z = np.asarray(z, float)
    z = z[z != 0]
    n = len(z)
    if n == 0:
        return 1.0, 0.0, 0
    ranks = stats.rankdata(np.abs(z))  # mid-ranks for ties
    obs = float(ranks[z > 0].sum())
    signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(float)
    wplus = signs @ ranks
    eps = 1e-9
    lo = np.mean(wplus <= obs + eps)
    hi = np.mean(wplus >= obs - eps)
    return float(min(1.0, 2 * min(lo, hi))), obs, n


def verdict(z):
    if z <= HOPED_MAX:
        return "hoped"
    if z >= NULL_MIN:
        return "null"
    return "partial"


def wording(cat, lo, hi):
    """Section 6 rule 3: state the category unhedged only if the CI excludes the opposite one."""
    if cat == "hoped" and hi < NULL_MIN:
        return "unhedged"
    if cat == "null" and lo > HOPED_MAX:
        return "unhedged"
    spans = [t for t in (HOPED_MAX, NULL_MIN) if lo < t < hi]
    return "consistent with %s (CI spans %s)" % (cat, ", ".join(str(t) for t in spans) or "none")


# ----------------------------------------------------------------------------- inputs

def read_logs(log_dir):
    """Per (dataset, seed): rc and wall_s from the LAST start/done pair across all session logs."""
    out = {}
    start_re = re.compile(r"^--- start rogueone (\S+) \(seed=(\d+) .*\) (\S+Z)$")
    done_re = re.compile(r"^--- done rogueone (\S+) rc=(\d+)(.*?)(\d{4}-\d\d-\d\dT\S+Z)$")
    for path in sorted(glob.glob(os.path.join(log_dir, "*.log"))):
        starts = {}
        nstart = {}
        seed_of = {}
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                m = start_re.match(line)
                if m:
                    ds, seed, ts = m.group(1), int(m.group(2)), m.group(3)
                    starts[ds] = ts
                    seed_of[ds] = seed
                    nstart[ds] = nstart.get(ds, 0) + 1
                    continue
                m = done_re.match(line)
                if m and m.group(1) in starts:
                    ds = m.group(1)
                    rc = int(m.group(2))
                    ok = rc == 0 and "BUT NO" not in m.group(3)
                    t0 = _dt.datetime.strptime(starts[ds], "%Y-%m-%dT%H:%M:%SZ")
                    t1 = _dt.datetime.strptime(m.group(4), "%Y-%m-%dT%H:%M:%SZ")
                    out[(ds, seed_of[ds])] = dict(
                        rc=rc, rc_ok=ok, wall_s=(t1 - t0).total_seconds(),
                        retried=nstart[ds] > 1, log=os.path.basename(path))
    return out


def run_dir(root, ds):
    return os.path.join(root, "tasks", task_type(ds), ds, "output")


def root_for_seed(roots, seed):
    for r in roots:
        if re.search(r"(s|_)%d$" % seed, r.rstrip("/")) or r.rstrip("/").endswith("/s%d" % seed):
            return r
    return None


def attr_list(outdir):
    f = os.path.join(outdir, "df_attributes_explanations_final.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f)
    cols = [c for c in ("Attribute", "Pandas Command") if c in df.columns]
    return [tuple(str(x) for x in r) for r in df[cols].itertuples(index=False)]


def tester_share(out_a, out_b):
    same = 0
    for i in range(50):
        fa = os.path.join(out_a, "tester_reports", "tester_report_iteration_%d.md" % i)
        fb = os.path.join(out_b, "tester_reports", "tester_report_iteration_%d.md" % i)
        if os.path.exists(fa) and os.path.exists(fb):
            with open(fa, "rb") as a, open(fb, "rb") as b:
                same += hashlib.sha256(a.read()).digest() == hashlib.sha256(b.read()).digest()
    return same / 50.0


def per_fold_selected(outdir, isreg):
    """The section 2 selection rule per outer fold (collect_seed_sweep_t17.py:88-97)."""
    f = os.path.join(outdir, "df_test_pool_final.csv")
    if not os.path.exists(f):
        return {}
    df = pd.read_csv(f)
    df = df[df["status"] == "completed"].copy()
    mcol, icol = ("n_RMSE", "inner_cv_RMSE") if isreg else ("accuracy", "inner_cv_accuracy")
    for c in (mcol, icol, "test_fold_id"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    res = {}
    for fold in sorted(df["test_fold_id"].dropna().unique()):
        g = df[df["test_fold_id"] == fold]
        if not g[icol].notna().any():
            continue
        i = g[icol].idxmin() if isreg else g[icol].idxmax()
        res[int(fold)] = float(g.loc[i, mcol])
    return res


def long_ok(row):
    return (row is not None and int(row["folds"]) == 5
            and int(row["inner_populated"]) == int(row["completed_trials"]))


def lookup(df, ds, seed):
    m = df[(df["dataset"] == ds) & (df["seed"] == seed)]
    return None if m.empty else m.iloc[0]


# ----------------------------------------------------------------------------- analysis

def analyse(a):
    os.makedirs(a.out, exist_ok=True)
    w0_long = pd.read_csv(a.w0_long)
    t17_long = pd.read_csv(a.t17_long)
    t17_sd = pd.read_csv(a.t17_sd).set_index("dataset")
    base = pd.read_csv(a.baselines)
    base = base[(base["baseline"] == "base_no_fe") & (base["fold_scheme"] == "kfold")]
    logs = read_logs(a.logs)
    w0_roots = [r for r in a.w0_roots.split(",") if r]
    t17_roots = [r for r in a.t17_roots.split(",") if r]

    # --rerun dataset:seed:long_csv:root overrides the first-session W0 result for that pair
    reruns = {}
    for spec in a.rerun or []:
        ds, seed, csv, root = spec.split(":", 3)
        reruns[(ds, int(seed))] = (pd.read_csv(csv), root)

    # output folder sizes (printed first, per the spec)
    sizes = {}
    for ds in W0_8 + OPTIONAL:
        for seed in SEEDS:
            root = reruns[(ds, seed)][1] if (ds, seed) in reruns else root_for_seed(w0_roots, seed)
            if root and os.path.isdir(run_dir(root, ds)):
                d = run_dir(root, ds)
                sizes[d] = int(subprocess.run(["du", "-sb", d], capture_output=True, text=True).stdout.split()[0])
                print("du %8.1f MB  %s" % (sizes[d] / 1e6, d))
    total = sum(sizes.values())
    print("du total %.2f GB" % (total / 1e9))

    pairs, fold_tables = [], []
    for ds in W0_8 + OPTIONAL:
        b = base[base["dataset"] == ds]
        if ds in W0_8:
            assert len(b) == 1, "expected exactly one base_no_fe kfold row for %s, got %d" % (ds, len(b))
        s_d = float(b["value_sd"].iloc[0]) if len(b) == 1 else float("nan")
        isreg = ds in REG
        for seed in SEEDS:
            src, root = (reruns[(ds, seed)] if (ds, seed) in reruns else (w0_long, root_for_seed(w0_roots, seed)))
            w = lookup(src, ds, seed)
            t = lookup(t17_long, ds, seed)
            if w is None and ds in OPTIONAL:
                continue
            lg = logs.get((ds, seed), {})
            w_out = run_dir(root, ds) if root else None
            t_root = root_for_seed(t17_roots, seed)
            t_out = run_dir(t_root, ds) if t_root else None
            w_complete = (long_ok(w) and lg.get("rc_ok", False)
                          and w_out is not None and os.path.exists(os.path.join(w_out, "fold_aggregation_summary.csv")))
            t_complete = long_ok(t) and (t_out is None or os.path.exists(os.path.join(t_out, "fold_aggregation_summary.csv")))
            ws = float(w["value"]) if w is not None else float("nan")
            ts = float(t["value"]) if t is not None else float("nan")
            delta = (ts - ws) if isreg else (ws - ts)
            al_w = attr_list(w_out) if w_out else None
            al_t = attr_list(t_out) if t_out else None
            ident = bool(al_w is not None and al_t is not None and al_w == al_t)
            if al_w is not None and al_t is not None:
                cw, ct = {c[-1] for c in al_w}, {c[-1] for c in al_t}
                jac = len(cw & ct) / len(cw | ct) if (cw | ct) else 1.0
            else:
                jac = float("nan")
            tsh = tester_share(w_out, t_out) if (w_out and t_out) else float("nan")
            pairs.append(dict(
                dataset_id=ds, task_type=task_type(ds), fold_seed=seed, w0_score=ws, t17_score=ts,
                delta_raw=delta, s_d=s_d, z=delta / s_d if s_d else float("nan"),
                pair_identical=ident, jaccard=jac, tester_identical_share=tsh,
                complete=bool(w_complete and t_complete), w0_complete=bool(w_complete),
                t17_complete=bool(t_complete), wall_s=lg.get("wall_s"), retried=lg.get("retried"),
                rc=lg.get("rc"), w0_dir_bytes=sizes.get(w_out), population="W0-8" if ds in W0_8 else "optional"))
            if ds in ("junglechess", "insurance") and w_out and t_out:
                fw, ft = per_fold_selected(w_out, isreg), per_fold_selected(t_out, isreg)
                for f in sorted(set(fw) | set(ft)):
                    a_, b_ = fw.get(f, np.nan), ft.get(f, np.nan)
                    fold_tables.append(dict(dataset_id=ds, fold_seed=seed, fold=f, w0=a_, t17=b_,
                                            delta=(b_ - a_) if isreg else (a_ - b_)))
    P = pd.DataFrame(pairs)
    FT = pd.DataFrame(fold_tables)
    C = P[(P["population"] == "W0-8") & P["complete"]]

    # per-dataset table
    rows = []
    for ds in W0_8 + OPTIONAL:
        g = P[(P["dataset_id"] == ds) & P["complete"]]
        allg = P[P["dataset_id"] == ds]
        if allg.empty:
            continue
        k = len(g)
        sd17 = float(t17_sd.loc[ds, "sd_across_seeds"])
        v = sigma_samp_sq(g["delta_raw"]) if k else float("nan")
        ss = math.sqrt(v) if k else float("nan")
        rho = ss / sd17 if (k and sd17 > 0) else float("nan")
        r = dict(dataset_id=ds, task_type=task_type(ds), metric="n_RMSE" if ds in REG else "accuracy",
                 population="W0-8" if ds in W0_8 else "optional", k_d=k, t17_seed_sd=sd17,
                 s_d=float(allg["s_d"].iloc[0]), sigma_samp=ss, rho_d=rho,
                 samp_share_d=min(rho ** 2, 1.0) if not math.isnan(rho) else float("nan"))
        for seed in SEEDS:
            x = allg[allg["fold_seed"] == seed]
            if len(x):
                x = x.iloc[0]
                r.update({"w0_f%d" % seed: x["w0_score"], "t17_f%d" % seed: x["t17_score"],
                          "delta_f%d" % seed: x["delta_raw"], "complete_f%d" % seed: x["complete"],
                          "wall_s_f%d" % seed: x["wall_s"], "retried_f%d" % seed: x["retried"]})
        rows.append(r)
    D = pd.DataFrame(rows)
    D8 = D[(D["population"] == "W0-8") & (D["k_d"] >= 1)]

    S = dict(generated_utc=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             n_complete_pairs=int(len(C)), n_pairs_attempted=int((P["population"] == "W0-8").sum()),
             k_d={r.dataset_id: int(r.k_d) for r in D.itertuples()},
             incomplete_pairs=P[(P["population"] == "W0-8") & ~P["complete"]][["dataset_id", "fold_seed", "w0_complete", "t17_complete"]].to_dict("records"))

    def pooled(dsub):
        sh = share_with_ci((dsub["sigma_samp"] ** 2).values, (dsub["t17_seed_sd"] ** 2).values, dsub["k_d"].values)
        return sh

    sh = pooled(D8)
    sh_nj = pooled(D8[D8["dataset_id"] != "junglechess"])
    S.update({"samp_share_v1": sh["z"], "r_raw": sh["r_raw"], "z_v1_ci_low": sh["ci_low"], "z_v1_ci_high": sh["ci_high"],
              "z_v1_nu1": sh["nu1"], "z_v1_nu2": sh["nu2"], "var_inflation_threshold": sh["var_inflation_threshold"],
              "var_inflation": sh["var_inflation"],
              "z_v1_ci_low_nominal": sh["ci_low_nominal"], "z_v1_ci_high_nominal": sh["ci_high_nominal"],
              "z_v1_nu1_nominal": sh["nu1_nominal"], "z_v1_nu2_nominal": sh["nu2_nominal"],
              "var_inflation_threshold_nominal": sh["var_inflation_threshold_nominal"],
              "var_inflation_nominal": bool(sh["z"] > sh["var_inflation_threshold_nominal"]),
              "no_junglechess": sh_nj})

    def pooled_sd(sub):
        zz = (sub["delta_raw"] / sub["s_d"]).values
        return dict(value=float(math.sqrt(np.sum(zz ** 2) / (2 * len(zz)))) if len(zz) else float("nan"), df=int(len(zz)))

    S["sigma_samp_pooled_sd"] = pooled_sd(C)
    S["sigma_samp_pooled_sd_no_junglechess"] = pooled_sd(C[C["dataset_id"] != "junglechess"])
    S["sigma_samp_pooled_sd_reg3"] = pooled_sd(C[C["dataset_id"].isin(REG3)])
    p, wplus, n_nz = wilcoxon_exact((C["delta_raw"] / C["s_d"]).values)
    S.update(drift_p=p, wilcoxon_w_plus=wplus, wilcoxon_n_nonzero=n_nz,
             mean_z=float((C["delta_raw"] / C["s_d"]).mean()) if len(C) else float("nan"),
             median_z=float((C["delta_raw"] / C["s_d"]).median()) if len(C) else float("nan"))

    # section 6, in order
    n = len(C)
    S["rule1_usable"] = "full" if n >= USABLE_FULL else ("df_stated" if n >= USABLE_MIN else "failed")
    mean_shift = p < DRIFT_ALPHA
    S["rule2_drift"] = bool(mean_shift or sh["var_inflation"])
    S["rule2_trigger"] = [t for t, on in (("mean_shift", mean_shift), ("var_inflation", sh["var_inflation"])) if on]
    cat = verdict(sh["z"])
    S["rule3_verdict"] = cat
    S["rule3_wording"] = wording(cat, sh["ci_low"], sh["ci_high"])
    S["rule3_verdict_used"] = not S["rule2_drift"] and S["rule1_usable"] != "failed"
    S["rho_le_0.6_count"] = int((D8["rho_d"] <= 0.6).sum())
    S["rho_ge_0.85_count"] = int((D8["rho_d"] >= 0.85).sum())
    n_ident = int(C["pair_identical"].sum())
    S["n_pair_identical"] = n_ident
    S["rule4_sigma_is_lower_bound"] = n_ident >= IDENTICAL_LOWER_BOUND_MIN
    S["median_tester_identical_share"] = float(C["tester_identical_share"].median()) if n else float("nan")
    S["median_jaccard"] = float(C["jaccard"].median()) if n else float("nan")
    walls = P[(P["population"] == "W0-8") & P["w0_complete"]]["wall_s"].dropna()
    S["median_wall_s"] = float(walls.median()) if len(walls) else float("nan")
    S["runs_over_6h"] = P[(P["wall_s"].fillna(0) > 6 * 3600)][["dataset_id", "fold_seed", "wall_s"]].to_dict("records")
    S["a03_handoff"] = {
        "sigma_samp_by_dataset": {r.dataset_id: (r.sigma_samp if r.k_d == 2 else None) for r in D8.itertuples()},
        "fallback_t17_sd": [r.dataset_id for r in D.itertuples() if r.k_d < 2 or r.population != "W0-8"] + [x for x in OPTIONAL if x not in set(D["dataset_id"])],
        "drift": S["rule2_drift"], "drift_trigger": S["rule2_trigger"],
        "independence_lower_bound": S["rule4_sigma_is_lower_bound"],
    }
    if total > 20e9:
        S["upload_warning"] = "output folders total %.1f GB > 20 GB; files > 500 MB excluded" % (total / 1e9)
    if FT is not None and not FT.empty:
        jc = FT[FT["dataset_id"] == "junglechess"]
        S["junglechess_folds_ge_0.95"] = {
            "f%d" % s: {"w0": int((jc[jc.fold_seed == s]["w0"] >= 0.95).sum()), "t17": int((jc[jc.fold_seed == s]["t17"] >= 0.95).sum())}
            for s in SEEDS}

    D.to_csv(os.path.join(a.out, "results_w0_t17rep.csv"), index=False)
    P.to_csv(os.path.join(a.out, "results_w0_t17rep_pairs.csv"), index=False)
    if not FT.empty:
        FT.to_csv(os.path.join(a.out, "results_w0_t17rep_folds.csv"), index=False)
    with open(os.path.join(a.out, "results_w0_t17rep_summary.json"), "w") as fh:
        json.dump(S, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    pd.set_option("display.width", 200)
    print(D[["dataset_id", "k_d", "t17_seed_sd", "s_d", "sigma_samp", "rho_d", "delta_f42", "delta_f4"]].to_string(index=False))
    print(json.dumps({k: S[k] for k in ("n_complete_pairs", "samp_share_v1", "z_v1_ci_low", "z_v1_ci_high", "z_v1_nu1",
                                        "z_v1_nu2", "var_inflation_threshold", "var_inflation", "drift_p", "rule1_usable",
                                        "rule2_drift", "rule3_verdict", "rule3_wording", "n_pair_identical", "median_wall_s")},
                     indent=1, default=str))

    if a.wandb:
        upload(a, S, P, D, sizes)
    return S


def upload(a, S, P, D, sizes):
    import wandb
    name = "ANALYSIS__w0_sigma_samp__%s" % _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d")
    run = wandb.init(entity=a.wandb_entity, project=a.wandb_project, group="ANALYSIS", name=name,
                     job_type="analysis", config=dict(ablation_id="A01", script="w0_sigma_samp.py"))
    run.summary.update({k: v for k, v in S.items() if isinstance(v, (int, float, str, bool))})
    run.log({"pairs": wandb.Table(dataframe=P.astype(str)), "datasets": wandb.Table(dataframe=D.astype(str))})
    art = wandb.Artifact("a01-results", type="analysis")
    for f in ("results_w0_t17rep.csv", "results_w0_t17rep_pairs.csv", "results_w0_t17rep_summary.json", "results_w0_t17rep_folds.csv"):
        fp = os.path.join(a.out, f)
        if os.path.exists(fp):
            art.add_file(fp)
    art.add_file(a.w0_long, name="results_w0_t17rep_long.csv")
    run.log_artifact(art)
    big = sum(sizes.values()) > 20e9
    out = wandb.Artifact("a01-run-outputs", type="run-outputs")
    excluded = []
    for d in sizes:
        parts = d.rstrip("/").split("/")
        prefix = "%s/%s" % (parts[-5] if len(parts) >= 5 else "w0", parts[-2])  # <root>/<dataset>
        for dirpath, _, files in os.walk(d):
            for fn in files:
                fp = os.path.join(dirpath, fn)
                if big and os.path.getsize(fp) > 500e6:
                    excluded.append(fp)
                    continue
                out.add_file(fp, name=os.path.join(prefix, os.path.relpath(fp, d)))
    run.log_artifact(out)
    if excluded:
        run.summary["upload_excluded"] = excluded
    run.finish()
    print("W&B run", name, "logged")


# ----------------------------------------------------------------------------- selftest

T17_SD_COMMITTED = {"credit-g": 0.0055, "heart": 0.0189, "junglechess": 0.0695, "myocardial": 0.0076,
                    "vehicle": 0.0294, "crab": 0.0242, "housing": 0.0213, "insurance": 0.0630}
S_D = {"credit-g": 0.0498, "heart": 0.0315, "junglechess": 0.0021, "myocardial": 0.0331,
       "vehicle": 0.0375, "crab": 0.0096, "housing": 0.0070, "insurance": 0.0193}


def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        print("%-4s %s" % ("ok" if cond else "FAIL", name))
        ok &= bool(cond)

    sd = np.array([T17_SD_COMMITTED[d] for d in W0_8])
    sdd = np.array([S_D[d] for d in W0_8])
    w = sd ** 2
    k = np.full(8, 2.0)

    # 1. W0 = t17 -> sigma_samp = 0 and p = 1
    zeros = np.zeros(16)
    check("W0 = t17: sigma_samp = 0", sigma_samp_sq(zeros) == 0.0)
    check("W0 = t17: p = 1", wilcoxon_exact(zeros)[0] == 1.0)

    # 2. +3 s_d shift on all 16 pairs, distinct small jitter -> p < 0.001
    rng = np.random.default_rng(0)
    z = 3.0 + rng.uniform(-0.01, 0.01, 16)
    p = wilcoxon_exact(z)[0]
    check("+3 s_d shift: p < 0.001 (p = %.2e, min 2/2^16 = %.2e)" % (p, 2 / 2 ** 16), p < 1e-3 and abs(p - 2 / 2 ** 16) < 1e-12)
    # sanity of the exact test against scipy at small n without ties
    zt = rng.normal(0.3, 1, 10)
    check("exact p matches scipy wilcoxon (n=10)",
          abs(wilcoxon_exact(zt)[0] - stats.wilcoxon(zt, mode="exact").pvalue) < 1e-9)

    # 3. draws with a known sigma reproduce the sigma_samp formula within 1e-9
    sig = 0.0123
    d = rng.normal(0, math.sqrt(2) * sig, 2)
    check("sigma_samp formula", abs(sigma_samp_sq(d) - (d[0] ** 2 + d[1] ** 2) / 4) < 1e-9)
    big = rng.normal(0, math.sqrt(2) * sig, 400000)
    check("sigma_samp recovers sigma on large draws", abs(math.sqrt(sigma_samp_sq(big)) - sig) / sig < 0.01)

    # 4. var-inflation flag just above / just below the threshold
    thr = share_with_ci(w, w, k)["var_inflation_threshold"]
    check("var_inflation above threshold flags", share_with_ci(w * thr * 1.001, w, k)["var_inflation"])
    check("var_inflation below threshold does not", not share_with_ci(w * thr * 0.999, w, k)["var_inflation"])

    # 5. Satterthwaite df and the nominal factor
    nu1, nu2 = satterthwaite(w, k)
    check("nu1 = nu2 within 0.1 of 6.1 (nu1 = %.3f, nu2 = %.3f)" % (nu1, nu2), abs(nu1 - 6.1) < 0.1 and abs(nu2 - 6.1) < 0.1)
    check("f.ppf(0.975, 16, 16) = 2.761 +- 0.001 (%.4f)" % stats.f.ppf(0.975, 16, 16), abs(stats.f.ppf(0.975, 16, 16) - 2.761) < 1e-3)
    sh = share_with_ci(w, w, k)
    check("CI factor at the full design ~ 5.7 (%.2f)" % (sh["ci_high"] / sh["z"]), 5.2 < sh["ci_high"] / sh["z"] < 6.2)

    # 6. Monte Carlo, sampling share 1: flag rate and CI coverage
    rng = np.random.default_rng(1)
    M = 20000
    w_hat = w * rng.chisquare(2, (M, 8)) / 2           # t17 SD^2 from 3 seeds (2 df)
    v_hat = w * rng.chisquare(2, (M, 8)) / 2           # two pairs: sum(delta^2)/4, delta ~ N(0, 2 sigma^2)
    zs = v_hat.sum(1) / w_hat.sum(1)
    flag = np.mean(zs > thr)
    lo = zs / stats.f.ppf(0.975, nu1, nu2)
    hi = zs * stats.f.ppf(0.975, nu2, nu1)
    cov = np.mean((lo <= 1) & (1 <= hi))
    check("MC flag rate in [0.5%%, 3%%] (%.2f%%)" % (100 * flag), 0.005 <= flag <= 0.03)
    check("MC coverage in [0.94, 0.99] (%.3f)" % cov, 0.94 <= cov <= 0.99)

    # 7. verdict and wording
    check("verdict thresholds", verdict(0.36) == "hoped" and verdict(0.72) == "null" and verdict(0.5) == "partial")
    check("wording rule", wording("hoped", 0.05, 0.7) == "unhedged" and wording("hoped", 0.05, 2.0).startswith("consistent"))

    # 8. log parser: failed first pass then solo retry -> last pair wins, retried flag
    import tempfile
    tmp = tempfile.mkdtemp()
    with open(os.path.join(tmp, "s.log"), "w") as fh:
        fh.write("--- start rogueone crab (seed=42 image=rogue_one:nestedcv9) 2026-10-09T08:32:39Z\n"
                 "--- done rogueone crab rc=0 BUT NO fold_aggregation_summary.csv -- treating as FAILED 2026-10-09T09:54:29Z\n"
                 "--- start rogueone crab (seed=42 image=rogue_one:nestedcv9) 2026-10-09T12:01:54Z\n"
                 "--- done rogueone crab rc=0 2026-10-09T13:23:17Z\n"
                 "--- start rogueone heart (seed=42 image=rogue_one:nestedcv9) 2026-10-09T08:32:39Z\n"
                 "--- done rogueone heart rc=0 BUT NO fold_aggregation_summary.csv -- treating as FAILED 2026-10-09T09:00:00Z\n")
    L = read_logs(tmp)
    check("log parser: retry wins", L[("crab", 42)]["rc_ok"] and L[("crab", 42)]["retried"] and L[("crab", 42)]["wall_s"] == 4883)
    check("log parser: BUT NO is a failure", not L[("heart", 42)]["rc_ok"])

    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--w0-long")
    ap.add_argument("--t17-long")
    ap.add_argument("--t17-sd")
    ap.add_argument("--baselines")
    ap.add_argument("--logs")
    ap.add_argument("--w0-roots", default="")
    ap.add_argument("--t17-roots", default="")
    ap.add_argument("--rerun", action="append", help="dataset:seed:long_csv:root")
    ap.add_argument("--out")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-entity", default="PlanBench")
    ap.add_argument("--wandb-project", default="RogueOne-AGENT2027")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    analyse(a)


if __name__ == "__main__":
    main()
