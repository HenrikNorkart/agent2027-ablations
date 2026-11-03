"""E11b: GLMMs, TOST and Holm for the AGENT2027 confirmatory claims.

Pre-registration Part IV.3 / V.12 (A12, C2a-c): run-level beta-binomial with +-1 coding,

    k ~ BetaBinomial(n, mu, phi),  logit(mu) = X beta + u_task,  u_task ~ N(0, s^2)

fitted by maximum likelihood with the task intercept integrated out by Gauss-Hermite quadrature (one random
effect, so quadrature is exact up to the node count; no Laplace step). Wald SEs come from a finite-difference
Hessian of the marginal log-likelihood. ``--check-r`` refits the same model with glmmTMB (betabinomial family)
in the ``ro27-rstats`` R container as a cross-check.

Delta_X ("predicted mean X+ minus X-, tasks equally weighted"): for each task t, with the task's conditional
mode u_t and every other factor at 0 (the fraction is balanced), d_t = expit(b0 + u_t + b_X) - expit(b0 + u_t - b_X);
Delta_X = mean_t d_t. Its SE is by the delta method in beta (u_t held at its mode).

Entry points:
    python glmm_holm.py toy                       # reproduces the toy example (Done-when of E11b)
    python glmm_holm.py g4 --episodes a12_episodes.csv --out G4_glmm.json [--delta 0.167]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

# ---------------------------------------------------------------- beta-binomial GLMM, one random intercept


def _betabin_logpmf(k, n, mu, phi):
    a, b = mu * phi, (1.0 - mu) * phi
    return (special.gammaln(n + 1) - special.gammaln(k + 1) - special.gammaln(n - k + 1)
            + special.betaln(k + a, n - k + b) - special.betaln(a, b))


@dataclass
class BetaBinFit:
    names: list[str]
    beta: np.ndarray
    se: np.ndarray
    log_phi: float
    log_sd: float
    loglik: float
    converged: bool
    groups: list[str]
    modes: np.ndarray
    cov: np.ndarray = field(repr=False)

    @property
    def phi(self) -> float:
        return math.exp(self.log_phi)

    @property
    def sd(self) -> float:
        return math.exp(self.log_sd)

    def wald(self) -> pd.DataFrame:
        z = self.beta / self.se
        return pd.DataFrame({"term": self.names, "beta": self.beta, "se": self.se, "z": z,
                             "p_wald": 2 * stats.norm.sf(np.abs(z))})


def fit_betabin_glmm(k, n, X, group, names, n_nodes: int = 40) -> BetaBinFit:
    k, n, X = np.asarray(k, float), np.asarray(n, float), np.asarray(X, float)
    group = np.asarray(group).astype(str)
    levels = sorted(set(group))
    gidx = np.array([levels.index(g) for g in group])
    nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)  # for int exp(-x^2) f(x)
    p = X.shape[1]

    def negll(theta):
        beta, log_phi, log_sd = theta[:p], theta[p], theta[p + 1]
        phi, sd = math.exp(log_phi), math.exp(log_sd)
        eta = X @ beta
        u = math.sqrt(2.0) * sd * nodes  # (Q,)
        mu = special.expit(eta[:, None] + u[None, :])  # (N, Q)
        mu = np.clip(mu, 1e-12, 1 - 1e-12)
        lp = _betabin_logpmf(k[:, None], n[:, None], mu, phi)  # (N, Q)
        per_group = np.zeros((len(levels), len(nodes)))
        np.add.at(per_group, gidx, lp)
        lg = special.logsumexp(per_group + np.log(weights)[None, :], axis=1) - 0.5 * math.log(math.pi)
        return -float(lg.sum())

    ybar = min(max(k.sum() / n.sum(), 1e-3), 1 - 1e-3)
    best = None
    for log_sd0, log_phi0 in ((-1.0, 1.0), (0.5, 3.0)):
        th0 = np.zeros(p + 2)
        th0[0] = math.log(ybar / (1 - ybar))
        th0[p], th0[p + 1] = log_phi0, log_sd0
        res = optimize.minimize(negll, th0, method="BFGS", options={"gtol": 1e-6, "maxiter": 5000})
        if best is None or res.fun < best.fun:
            best = res
    th = best.x
    # BFGS often stops on precision loss at the optimum; judge convergence by the gradient instead
    grad = optimize.approx_fprime(th, negll, 1e-6)
    converged = bool(best.success or np.max(np.abs(grad)) < 1e-2)
    H = _hessian(negll, th)
    cov = np.linalg.pinv(H)
    se = np.sqrt(np.clip(np.diag(cov)[:p], 0, None))

    # conditional modes of u_t given theta (for Delta_X)
    beta, phi, sd = th[:p], math.exp(th[p]), math.exp(th[p + 1])
    eta = X @ beta
    modes = np.zeros(len(levels))
    for j in range(len(levels)):
        m = gidx == j
        f = lambda u: -(float(_betabin_logpmf(k[m], n[m], np.clip(special.expit(eta[m] + u), 1e-12, 1 - 1e-12),
                                              phi).sum()) - 0.5 * (u / sd) ** 2)
        modes[j] = optimize.minimize_scalar(f, bounds=(-10 * sd - 5, 10 * sd + 5), method="bounded").x
    return BetaBinFit(list(names), beta, se, float(th[p]), float(th[p + 1]), -float(best.fun),
                      converged, levels, modes, cov)


def _hessian(f, x, h=1e-4):
    x = np.asarray(x, float)
    d = len(x)
    H = np.zeros((d, d))
    for i in range(d):
        for j in range(i, d):
            ei, ej = np.zeros(d), np.zeros(d)
            ei[i], ej[j] = h, h
            v = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
            H[i, j] = H[j, i] = v
    return H


def delta_x(fit: BetaBinFit, term: str) -> tuple[float, float]:
    """Delta_X = mean over tasks of expit(b0+u_t+b_X) - expit(b0+u_t-b_X); delta-method SE in (b0, b_X)."""
    i0, ix = fit.names.index("Intercept"), fit.names.index(term)
    b0, bx = fit.beta[i0], fit.beta[ix]
    hi, lo = special.expit(b0 + fit.modes + bx), special.expit(b0 + fit.modes - bx)
    d = float(np.mean(hi - lo))
    g0 = float(np.mean(hi * (1 - hi) - lo * (1 - lo)))
    gx = float(np.mean(hi * (1 - hi) + lo * (1 - lo)))
    V = fit.cov[np.ix_([i0, ix], [i0, ix])]
    g = np.array([g0, gx])
    return d, float(math.sqrt(max(g @ V @ g, 0.0)))


# ---------------------------------------------------------------- TOST and Holm


def tost_p(est: float, se: float, delta: float) -> float:
    """Two one-sided z tests against +-delta; p = the larger one-sided p."""
    return float(max(stats.norm.sf((est + delta) / se), stats.norm.cdf((est - delta) / se)))


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down: adjusted p and reject flag per hypothesis."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    stop = False
    for i, (name, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        rej = (not stop) and p <= alpha / (m - i)
        stop = stop or not rej
        out[name] = {"p": p, "p_holm": running, "reject": rej}
    return out


# ---------------------------------------------------------------- A12 fraction (G4)

CELLS = {"fx_pAJM": "-+++", "fx_PaJM": "+-++", "fx_PAjM": "++-+", "fx_PAJm": "+++-",
         "fx_pajM": "---+", "fx_paJm": "--+-", "fx_pAjm": "-+--", "fx_Pajm": "+---"}
TERMS = ["Intercept", "P", "A", "J", "M", "PA", "PJ", "PM"]


def a12_runs(episodes: pd.DataFrame) -> pd.DataFrame:
    """Pre-registered fraction rows only (A12-001..056, the 8 cells), SYN-P 7 minus P1, ITT zeros kept."""
    ep = episodes[episodes.row_id.str.match(r"^A12-0(0[1-9]|[1-4][0-9]|5[0-6])$") & episodes.arm_id.isin(CELLS)
                  & (episodes.dataset_id != "synp_p1")]
    run = ep.groupby(["row_id", "arm_id", "dataset_id", "status"], as_index=False).agg(k=("rec_sel", "sum"),
                                                                                        n=("rec_sel", "size"))
    for i, f in enumerate("PAJM"):
        run[f] = run.arm_id.map(lambda a, i=i: 1.0 if CELLS[a][i] == "+" else -1.0)
    run["PA"], run["PJ"], run["PM"] = run.P * run.A, run.P * run.J, run.P * run.M
    run["Intercept"] = 1.0
    return run


def g4(episodes_csv: str, delta: float) -> dict:
    run = a12_runs(pd.read_csv(episodes_csv))
    fit = fit_betabin_glmm(run.k, run.n, run[TERMS].to_numpy(), run.dataset_id, TERMS)
    w = fit.wald()
    effects = {}
    for t in TERMS[1:]:
        d, se = delta_x(fit, t)
        effects[t] = {"beta": float(w.set_index("term").beta[t]), "se": float(w.set_index("term").se[t]),
                      "p_wald": float(w.set_index("term").p_wald[t]), "Delta": d, "Delta_se": se,
                      "Delta_ci90": [d - 1.6449 * se, d + 1.6449 * se], "Delta_ci95": [d - 1.96 * se, d + 1.96 * se],
                      "p_tost": tost_p(d, se, delta)}
    pair_p = {t: effects[t]["p_wald"] for t in ("PA", "PJ", "PM")}
    return {
        "n_runs": int(len(run)), "n_episodes": int(run.n.sum()), "itt_runs": int((run.status != "OK").sum()),
        "tasks": fit.groups, "converged": fit.converged, "loglik": fit.loglik, "phi": fit.phi, "sd_task": fit.sd,
        "delta": delta, "effects": effects,
        "g4_pair_p": pair_p, "g4_trigger_p": bool(min(pair_p.values()) < 0.05),
        "cell_task_k": run.pivot_table(index="arm_id", columns="dataset_id", values="k", aggfunc="sum")
                          .reindex(list(CELLS)).fillna(-1).astype(int).to_dict(),
    }


RV3_CELLS = {"fx_paJM": "--++", "fx_pAjM": "-+-+", "fx_pAJm": "-++-", "fx_pajm": "----"}
RV3_TERMS = TERMS + ["AJ", "AM", "JM", "block"]


def rv3_runs(episodes: pd.DataFrame) -> pd.DataFrame:
    """A12-T23 data: the fraction rows (A12-001..056) plus the RV3 foldover rows (A12-066..093), minus P1.
    block = +1 for the foldover session, -1 for the fraction."""
    cells = {**CELLS, **RV3_CELLS}
    rid = episodes.row_id.str.extract(r"^A12-(\d{3})$")[0].astype(float)
    ep = episodes[(rid.between(1, 56) | rid.between(66, 93)) & episodes.arm_id.isin(cells)
                  & (episodes.dataset_id != "synp_p1")]
    run = ep.groupby(["row_id", "arm_id", "dataset_id", "status"], as_index=False).agg(k=("rec_sel", "sum"),
                                                                                        n=("rec_sel", "size"))
    for i, f in enumerate("PAJM"):
        run[f] = run.arm_id.map(lambda a, i=i: 1.0 if cells[a][i] == "+" else -1.0)
    for a, b in ("PA", "PJ", "PM", "AJ", "AM", "JM"):
        run[a + b] = run[a] * run[b]
    run["block"] = np.where(run.row_id.str.slice(4).astype(int) >= 66, 1.0, -1.0)
    run["Intercept"] = 1.0
    return run


def rv3(episodes_csv: list[str], delta: float) -> dict:
    run = rv3_runs(pd.concat([pd.read_csv(p) for p in episodes_csv]).drop_duplicates(["row_id", "fold"]))
    fit = fit_betabin_glmm(run.k, run.n, run[RV3_TERMS].to_numpy(), run.dataset_id, RV3_TERMS)
    w = fit.wald().set_index("term")
    effects = {}
    for t in RV3_TERMS[1:]:
        d, se = delta_x(fit, t)
        effects[t] = {"beta": float(w.beta[t]), "se": float(w.se[t]), "p_wald": float(w.p_wald[t]), "Delta": d,
                      "Delta_se": se, "Delta_ci90": [d - 1.6449 * se, d + 1.6449 * se],
                      "Delta_ci95": [d - 1.96 * se, d + 1.96 * se], "p_tost": tost_p(d, se, delta)}
    return {"n_runs": int(len(run)), "n_fraction": int((run.block < 0).sum()), "n_foldover": int((run.block > 0).sum()),
            "n_episodes": int(run.n.sum()), "itt_runs": int((run.status != "OK").sum()), "tasks": fit.groups,
            "converged": fit.converged, "loglik": fit.loglik, "phi": fit.phi, "sd_task": fit.sd, "delta": delta,
            "effects": effects,
            "cell_task_k": run.pivot_table(index="arm_id", columns="dataset_id", values="k", aggfunc="sum")
                              .fillna(-1).astype(int).to_dict()}


# ---------------------------------------------------------------- toy example (E11b Done-when)


def toy(seed: int = 7) -> dict:
    """Simulate the A12 design (8 cells x 6 tasks x 5 folds) with known coefficients and check recovery."""
    rng = np.random.default_rng(seed)
    true = {"Intercept": -0.3, "P": 0.6, "A": 0.0, "J": 0.1, "M": 0.2, "PA": -0.5, "PJ": 0.0, "PM": 0.0}
    rows = []
    u = rng.normal(0, 0.8, 6)  # one intercept per task, shared by every replicate
    for rep in range(20):  # 20 replicates of the fraction so recovery is tight
        for t in range(6):
            for cell, code in CELLS.items():
                x = {f: (1.0 if code[i] == "+" else -1.0) for i, f in enumerate("PAJM")}
                x.update(PA=x["P"] * x["A"], PJ=x["P"] * x["J"], PM=x["P"] * x["M"], Intercept=1.0)
                mu = special.expit(sum(true[k] * x[k] for k in TERMS) + u[t])
                p = rng.beta(mu * 8, (1 - mu) * 8)
                rows.append({**x, "k": rng.binomial(5, p), "n": 5, "task": f"t{t}"})
    df = pd.DataFrame(rows)
    fit = fit_betabin_glmm(df.k, df.n, df[TERMS].to_numpy(), df.task, TERMS)
    w = fit.wald().set_index("term")
    ok = all(abs(w.beta[t] - true[t]) < 3 * w.se[t] + 0.05 for t in TERMS[1:])
    h = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    holm_ok = h["a"]["reject"] and not h["b"]["reject"] and not h["c"]["reject"] and abs(h["a"]["p_holm"] - 0.03) < 1e-12
    return {"beta_hat": w.beta.round(3).to_dict(), "se": w.se.round(3).to_dict(), "sd_task": round(fit.sd, 3),
            "phi": round(fit.phi, 2), "recovered_within_3se": ok, "holm_toy_ok": holm_ok, "converged": fit.converged}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("toy")
    g = sub.add_parser("g4")
    g.add_argument("--episodes", required=True)
    g.add_argument("--delta", type=float, default=0.167)
    g.add_argument("--out")
    r = sub.add_parser("rv3")
    r.add_argument("--episodes", nargs="+", required=True)
    r.add_argument("--delta", type=float, default=0.167)
    r.add_argument("--out")
    a = ap.parse_args(argv)
    res = toy() if a.cmd == "toy" else g4(a.episodes, a.delta) if a.cmd == "g4" else rv3(a.episodes, a.delta)
    txt = json.dumps(res, indent=2, default=float)
    if getattr(a, "out", None):
        open(a.out, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    sys.exit(main())
