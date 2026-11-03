"""Feature funnel (E11b; 01 section 4.6 "Feature funnel", R13 ``fun_*``), post hoc from run folders.

Definition (01 section 4.6): proposed -> executes -> numeric, non-constant, non-duplicate (|r| < 0.999) ->
replay OK -> survives pruning -> in the selected iteration. v2.0 logs no ``fun_*`` keys (prereg Amendment 6), so
every stage is reconstructed here from ``output/`` of the row's final attempt. Unit: one LLM append proposal of
the Extractor (``append_new_attribute``) in one outer fold; counts are per fold-episode, summed per row.

Stage rules (each stage is a subset of the previous one; ``rej_*`` columns say where proposals left):

* ``fun_proposed``  every ``append_new_attribute`` call that carries code = rows added by the Extractor tool
  (``Added`` = ``Trial k``, not ``Seeded``/``Fallback``) in any of the fold's attribute tables (for the append-only
  pool that is the final table; A10 restarts discard the pool, so each proposal is taken from the last trail in which
  it appears and is judged there) + that fold's append rejections
  in ``agent_toolcall_errors.jsonl``. Auto-unprune calls run no code and are not proposals (A18 ``rp1_executes``
  leaves them out too); they are only in the run-level ``append_restored`` check column.
* ``fun_executes``  minus ``rej_dup_name`` (name already Active: the command never runs), ``rej_exec_error``
  (exception) and ``rej_no_result``.
* ``fun_valid``     minus the harness rejections ``rej_nan_inf`` and ``rej_non_numeric`` (= accepted appends), minus
  ``rej_constant`` (one distinct value) and ``rej_near_dup`` (|Pearson r| >= 0.999 with a feature Active when it was
  appended: A18 ``rp1_usable`` = ``fun_valid``). Values = the run's own authoring values
  (``df_entities_attributes_trail_<k>.csv``, all rows). "Active when appended" = Active in the fold's previous
  trail (seeded rows at the fold's first iteration) or appended earlier in the same trial.
* ``fun_replay_ok`` minus ``rej_replay``: the command is replayed in table order on the fold's training rows and on
  its test rows alone, exactly as ``fold_safe_features.materialize_fold_safe_features`` does for the Tester; a replay
  that raises or cannot be stored (the code's NaN-fill path, counter ``replay_nan_fills``) fails.
* ``fun_survives_prune`` minus ``rej_pruned``: Status Active in the last attribute table in which it appears (the
  fold's final table; being discarded by an A10 restart is not pruning).
* ``fun_selected``  Active in the inner-CV-selected iteration (``fold_aggregation_best_iterations.csv``) **and** in
  the previous stage. A feature pruned after the selected iteration is therefore neither surviving nor selected;
  ``sel_any_valid`` counts selected features among ``fun_replay_ok`` without the survival condition.

Optional phi stage (``--meta``; SYN-P): the furthest stage that any accepted LLM feature reaching the detector
criterion (amended detector of record, authoring values on the fold's training rows) reached: ``none`` (no accepted
feature recovers phi) < ``valid`` < ``replay_ok`` < ``survives_prune`` < ``selected``; ``invalid`` = only constant /
near-duplicate hits. This is the "stage where phi is lost" (A09, A12, A20).

ITT: rows with no output contribute no funnel (status column says why); capped rows give the folds they reached.

    python scripts/agent2027/analysis/funnel.py --runs <session dirs> --manifest <manifests...> \
        --ablations A05 A17 --tasks <tasks root> [--meta <synp meta>] [--no-replay] --out-dir <dir>

Run on CAIR only inside a rogue_one container (it executes the agents' recorded pandas commands).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

N_FOLDS = 5
N_ITER = 10
DUP_R = 0.999
STAGES = ("fun_proposed", "fun_executes", "fun_valid", "fun_replay_ok", "fun_survives_prune", "fun_selected")
REJ = ("rej_dup_name", "rej_exec_error", "rej_no_result", "rej_nan_inf", "rej_non_numeric", "rej_constant",
       "rej_near_dup", "rej_replay", "rej_pruned", "rej_not_selected")
PHI_ORDER = ("none", "invalid", "valid", "replay_ok", "survives_prune", "selected")
_TRIAL_RE = re.compile(r"^Trial (\d+)$")


# --------------------------------------------------------------------------- pure helpers (unit-tested)
def classify_append_error(msg: str) -> str:
    """``append_new_attribute`` rejection kind from the logged message (``ext/agent.py`` append tool)."""
    m = str(msg)
    if "already exists and is active" in m:
        return "rej_dup_name"
    if "did not set 'result'" in m:
        return "rej_no_result"
    if "NaN or Inf" in m:
        return "rej_nan_inf"
    if "non-numeric" in m:
        return "rej_non_numeric"
    return "rej_exec_error"


def _flag(v: Any) -> bool:
    x = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    return bool(x) if pd.notna(x) else str(v).strip().lower() in {"true", "yes"}


def llm_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Rows appended by the Extractor's tool: added in a trial, with code, not seeded and not fallback."""
    t = table.copy()
    added = t.get("Added", pd.Series([""] * len(t))).astype(str)
    seeded = t["Seeded"].map(_flag) if "Seeded" in t else pd.Series(False, index=t.index)
    fb = t["Fallback"].map(_flag) if "Fallback" in t else pd.Series(False, index=t.index)
    cmd = t["Pandas Command"].fillna("").astype(str).str.strip() != ""
    return t[added.str.match(r"^Trial \d+$") & ~seeded & ~fb & cmd]


def validity(values: pd.DataFrame, names: Sequence[str], refs: Mapping[str, Sequence[str]],
             dup_r: float = DUP_R) -> dict[str, str]:
    """``""`` (valid), ``rej_constant`` or ``rej_near_dup`` per feature; a feature without values counts as valid."""
    out: dict[str, str] = {}
    for n in names:
        if n not in values.columns:
            out[n] = ""
            continue
        v = pd.to_numeric(values[n], errors="coerce").to_numpy(float)
        if np.unique(v[np.isfinite(v)]).size <= 1:
            out[n] = "rej_constant"
            continue
        cols = [c for c in refs.get(n, ()) if c in values.columns and c != n]
        if cols:
            m = values[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            ok = np.isfinite(v)[:, None] & np.isfinite(m)
            with np.errstate(invalid="ignore", divide="ignore"):
                if ok.all():  # fast path: Pearson r of v against every column at once
                    zv = (v - v.mean()) / v.std()
                    zm = (m - m.mean(0)) / m.std(0)
                    r = zm.T @ zv / len(v)
                else:
                    r = [np.corrcoef(v[ok[:, j]], m[ok[:, j], j])[0, 1] if ok[:, j].sum() > 2 else 0.0
                         for j in range(m.shape[1])]
            r = np.nan_to_num(np.abs(np.asarray(r, float)), nan=0.0)
            if r.size and r.max() >= dup_r:
                out[n] = "rej_near_dup"
                continue
        out[n] = ""
    return out


def funnel_counts(proposed: Sequence[str], rejections: Mapping[str, int], invalid: Mapping[str, str],
                  replay_ok: Optional[Mapping[str, bool]], final_active: Iterable[str],
                  selected_active: Optional[Iterable[str]]) -> dict[str, Any]:
    """Stage counts of one fold. ``proposed`` = accepted LLM feature names; ``rejections`` = pre-acceptance
    rejections by kind; ``replay_ok`` None -> replay not run (stage NaN, later stages computed on ``fun_valid``);
    ``selected_active`` None -> no selected iteration (``fun_selected`` = 0)."""
    rej = Counter({k: int(rejections.get(k, 0)) for k in REJ})
    acc = list(dict.fromkeys(proposed))
    n_prop = len(acc) + sum(rej[k] for k in ("rej_dup_name", "rej_exec_error", "rej_no_result",
                                                 "rej_nan_inf", "rej_non_numeric"))
    n_exec = n_prop - rej["rej_dup_name"] - rej["rej_exec_error"] - rej["rej_no_result"]
    for n in acc:
        if invalid.get(n):
            rej[invalid[n]] += 1
    valid = [n for n in acc if not invalid.get(n)]
    if replay_ok is None:
        rep, n_rep = valid, float("nan")
    else:
        rep = [n for n in valid if replay_ok.get(n, False)]
        rej["rej_replay"] = len(valid) - len(rep)
        n_rep = len(rep)
    fa = set(final_active)
    surv = [n for n in rep if n in fa]
    rej["rej_pruned"] = len(rep) - len(surv)
    sa = set(selected_active) if selected_active is not None else set()
    sel = [n for n in surv if n in sa]
    rej["rej_not_selected"] = len(surv) - len(sel)
    out = {"fun_proposed": n_prop, "fun_executes": n_exec, "fun_valid": len(valid), "fun_replay_ok": n_rep,
           "fun_survives_prune": len(surv), "fun_selected": len(sel),
           "sel_any_valid": len([n for n in rep if n in sa])}
    out.update({k: int(rej[k]) for k in REJ})
    return out


def phi_stage(phi_hits: Iterable[str], invalid: Mapping[str, str], replay_ok: Optional[Mapping[str, bool]],
              final_active: Iterable[str], selected_active: Optional[Iterable[str]]) -> str:
    """Furthest funnel stage reached by any accepted feature that meets the phi criterion."""
    fa, sa = set(final_active), set(selected_active or ())
    best = 0
    for n in phi_hits:
        if invalid.get(n):
            k = 1
        elif replay_ok is not None and not replay_ok.get(n, False):
            k = 2
        elif n not in fa:
            k = 3
        elif n not in sa:
            k = 4
        else:
            k = 5
        best = max(best, k)
    return PHI_ORDER[best]


def active_refs(trails: Sequence[Mapping[str, Any]], llm: pd.DataFrame) -> dict[str, list[str]]:
    """Per LLM feature: names Active just before it was appended (see module docstring)."""
    by_iter = {int(t["iteration_in_fold"]): t for t in trails}
    trial_to_iter = {str(t["trial"]): int(t["iteration_in_fold"]) for t in trails}
    first = trails[-1]["attributes"] if trails else pd.DataFrame(columns=["Attribute"])
    seeded = [str(a) for a, s in zip(first["Attribute"], first.get("Seeded", [0] * len(first))) if _flag(s)]
    order = list(trails[-1]["attributes"]["Attribute"].astype(str)) if trails else []
    added_trial = dict(zip(llm["Attribute"].astype(str), llm["Added"].astype(str)))
    refs: dict[str, list[str]] = {}
    for n, tr in added_trial.items():
        it = trial_to_iter.get(tr)
        if it is None:  # skipped-iteration trial: no trail of its own; use the trial number
            m = _TRIAL_RE.match(tr)
            it = int(m.group(1)) % N_ITER if m else 0
        prev = [i for i in by_iter if i < it]
        base = list(by_iter[max(prev)]["active"]) if prev else seeded
        same = [a for a in order[: order.index(n)] if added_trial.get(a) == tr] if n in order else []
        refs[n] = list(dict.fromkeys(base + same))
    return refs


# --------------------------------------------------------------------------- run-folder I/O
def read_errors(out_dir: Path) -> list[dict[str, Any]]:
    p = out_dir / "agent_toolcall_errors.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("tool_name") == "append_new_attribute":
            rows.append(r)
    return rows


def read_iteration_log(out_dir: Path) -> list[dict[str, Any]]:
    p = out_dir / "iteration_log.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def fold_structure(out_dir: Path) -> list[dict[str, Any]]:
    """Per outer fold: its trails and the selected trail (tolerates capped runs without the best-iteration file)."""
    from scripts.agent2027.analysis import a02_readout as a02  # noqa: PLC0415

    if (out_dir / "fold_aggregation_best_iterations.csv").exists() and (out_dir / "iteration_log.jsonl").exists():
        eps = a02.build_row_episodes(out_dir)
        trails = a02.read_trails(out_dir)
        for ep in eps:
            ep["selected_active"] = next((t["active"] for t in trails if t["trial"] == ep["selected_trial"]), None)
        return eps
    trails = a02.read_trails(out_dir)
    run_to_test = {int(r["fold_run_id"]): int(r["test_fold_id"]) for r in read_iteration_log(out_dir)}
    eps = []
    for fr in sorted({t["fold_run_id"] for t in trails}):
        eps.append({"fold": run_to_test.get(fr, fr), "fold_run_id": fr,
                    "trails": [t for t in trails if t["fold_run_id"] == fr],
                    "selected_iter": None, "selected_trial": None, "selected_active": None})
    return eps


def replay_status(df_raw: pd.DataFrame, table: pd.DataFrame, mask: np.ndarray) -> dict[str, bool]:
    """Mirror of ``materialize_fold_safe_features`` that records which attributes replay without the NaN fill."""
    from rogueone.utils.pandas_exec import exec_user_command  # noqa: PLC0415

    sub = df_raw[mask].reset_index(drop=True)
    feats = pd.DataFrame(index=sub.index)
    if "id" in sub.columns:
        feats["id"] = sub["id"].to_numpy()
    ok: dict[str, bool] = {}
    for _, row in table.iterrows():
        name, cmd = str(row["Attribute"]), row.get("Pandas Command", "")
        good = isinstance(cmd, str) and bool(cmd.strip())
        if good:
            try:
                res, _ = exec_user_command(cmd, sub, feats)
                if isinstance(res, (pd.Series, pd.DataFrame)):
                    res = res.reset_index(drop=True).values
                feats[name] = res
            except Exception:  # noqa: BLE001 - the code's own NaN-fill path
                good = False
        if not good:
            feats[name] = np.nan
        ok[name] = good
    return ok


def _set_exec_compat(row_dir: Path) -> None:
    """Replay under the row's own ``ROGUE_ONE_EXEC_COMPAT`` (Qwen rows, Amendment 17)."""
    try:
        from rogueone.utils import rouge_one_cfg  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return
    val = False
    env = row_dir / "env.list"
    if env.exists():
        for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("ROGUE_ONE_EXEC_COMPAT="):
                val = line.split("=", 1)[1].strip().lower() in {"1", "true", "yes", "on"}
    if hasattr(rouge_one_cfg, "exec_compat"):
        rouge_one_cfg.exec_compat = val


def row_funnel(out_dir: Path, *, df_raw: Optional[pd.DataFrame] = None, labels: Optional[pd.DataFrame] = None,
               fold_seed: int = 42, stratified: bool = False, replay: bool = True,
               phi=None, det=None) -> list[dict[str, Any]]:
    """Funnel per fold-episode of one row."""
    from scripts.agent2027.analysis import recovery_detector as rd  # noqa: PLC0415

    idx_to_fold = {int(r["index"]): int(r["fold_run_id"]) for r in read_iteration_log(out_dir)}
    rej_by_fold: dict[int, Counter] = {}
    for e in read_errors(out_dir):
        step = e.get("step")
        if step is None:
            continue
        fr = idx_to_fold.get(int(step), int(step) // N_ITER)
        rej_by_fold.setdefault(fr, Counter())[classify_append_error(e.get("message", ""))] += 1
    out = []
    for ep in fold_structure(out_dir):
        trails = ep["trails"]
        rec: dict[str, Any] = {"fold": ep["fold"], "fold_run_id": ep["fold_run_id"],
                               "selected_iter": ep["selected_iter"], "n_trails": len(trails)}
        if not trails:
            rec.update(funnel_counts([], rej_by_fold.get(ep["fold_run_id"], {}), {}, None, [], None))
            out.append(rec)
            continue
        mask = None
        if df_raw is not None and labels is not None and ep["fold"] is not None:
            mask = rd.train_mask_for_fold(df_raw, labels, fold_seed, int(ep["fold"]), stratified=stratified)
        # every LLM proposal of the fold, judged in the last trail in which it appears: the final table for the
        # append-only pool; for A10 restarts (pool reset) each discarded pool's last trail
        last_pos: dict[str, tuple[int, str]] = {}
        for pos, t in enumerate(trails):
            for _, r in llm_rows(t["attributes"].drop_duplicates("Attribute", keep="last")).iterrows():
                last_pos[f"{r['Attribute']}@{r['Added']}"] = (pos, str(r["Attribute"]))
        keys = list(last_pos)
        invalid: dict[str, str] = {}
        rok: Optional[dict[str, bool]] = {} if (replay and mask is not None) else None
        surviving: list[str] = []
        hits: list[str] = []
        n_missing = 0
        for pos in sorted({v[0] for v in last_pos.values()}):
            t = trails[pos]
            table = t["attributes"].drop_duplicates("Attribute", keep="last")
            grp = {k: n for k, (q, n) in last_pos.items() if q == pos}
            llm = llm_rows(table)
            vals_p = out_dir / "intermediate_steps" / f"df_entities_attributes_trail_{t['index']}.csv"
            values = pd.read_csv(vals_p) if vals_p.exists() else pd.DataFrame()
            inv = validity(values, list(grp.values()), active_refs(trails[: pos + 1], llm))
            st = dict(zip(table["Attribute"].astype(str), table["Status"].astype(str)))
            if rok is not None:
                a, b = replay_status(df_raw, table, mask), replay_status(df_raw, table, ~mask)
            for k, n in grp.items():
                invalid[k] = inv.get(n, "")
                if rok is not None:
                    rok[k] = a.get(n, False) and b.get(n, False)
                if st.get(n) == "Active":
                    surviving.append(k)
            n_missing += sum(1 for n in grp.values() if n not in values.columns)
            have = [n for n in grp.values() if n in values.columns]
            if phi is not None and mask is not None and "id" in values.columns and have:
                tr = df_raw.loc[mask, ["id"]].merge(values[["id"] + have], on="id", how="left")
                pf = rd.evaluate_feature_table(tr[have].reset_index(drop=True), np.asarray(phi, float)[mask],
                                               df_raw[mask].reset_index(drop=True), det)
                hit_names = set(pf.loc[pf["counted"].astype(bool), "feature"].astype(str))
                hits += [k for k, n in grp.items() if n in hit_names]
        sel = None
        if ep.get("selected_trial") is not None:
            stt = next((t for t in trails if t["trial"] == ep["selected_trial"]), None)
            if stt is not None:
                tab = stt["attributes"].drop_duplicates("Attribute", keep="last")
                sel = [f"{r['Attribute']}@{r['Added']}" for _, r in tab.iterrows() if str(r["Status"]) == "Active"]
        rec.update(funnel_counts(keys, rej_by_fold.get(ep["fold_run_id"], {}), invalid, rok, surviving, sel))
        rec["n_values_missing"] = n_missing
        rec["n_pool_tables"] = len({v[0] for v in last_pos.values()})
        if phi is not None and mask is not None:
            rec["phi_hits"] = ";".join(k.split("@")[0] for k in hits)
            rec["phi_stage"] = phi_stage(hits, invalid, rok, surviving, sel)
        out.append(rec)
    return out


def final_status(root: Path, row_id: str) -> str:
    st = root / "status.tsv"
    last = "MISSING"
    if st.exists():
        for r in csv.DictReader(st.open(encoding="utf-8"), delimiter="\t"):
            if r.get("row_id") == row_id:
                last = r.get("status", "?")
    return last


def locate(roots: Sequence[Path], rid: str) -> tuple[Optional[Path], str, Optional[Path]]:
    """(row folder, final status, output dir): prefer the session where the row ended OK, else the last one."""
    have = [r for r in roots if (r / rid).exists()]
    root = next((r for r in have if final_status(r, rid) == "OK"), have[-1] if have else None)
    if root is None:
        return None, "MISSING", None
    outs = sorted((root / rid).glob("tasks/*/*/output"))
    return root / rid, final_status(root, rid), (outs[0] if outs else None)


def rung_of(m: Mapping[str, str]) -> str:
    r = [m.get("rung_S", ""), m.get("rung_E", ""), m.get("rung_T", "")]
    return r[0] if len(set(r)) == 1 else "/".join(r)


def summarise(rows: pd.DataFrame, by: Sequence[str] = ("ablation_id", "arm_id", "rung")) -> pd.DataFrame:
    """Per group: summed stage counts, mean per run, and stage-to-stage conversion (ratio of sums)."""
    num = [c for c in rows.columns if c in STAGES or c in REJ or c == "sel_any_valid"]
    have = rows.dropna(subset=["fun_proposed"])
    g = have.groupby(list(by), dropna=False)
    s = g[num].sum(min_count=1)
    s.insert(0, "n_rows", g.size())
    for a, b in zip(STAGES[:-1], STAGES[1:]):
        s[f"conv_{b[4:]}"] = s[b] / s[a].replace(0, np.nan)
    s["conv_overall"] = s["fun_selected"] / s["fun_proposed"].replace(0, np.nan)
    for c in STAGES:
        s[f"{c}_per_run"] = s[c] / s["n_rows"]
    return s.reset_index()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", help="session folders (…/rogueone-runs-agent2027/S*)")
    ap.add_argument("--manifest", nargs="+")
    ap.add_argument("--merge", nargs="+", default=None,
                    help="merge earlier --out-dir folders (row shards) into --out-dir instead of computing")
    ap.add_argument("--rows", nargs="*", default=None, help="row ids (default: every row of --ablations)")
    ap.add_argument("--ablations", nargs="*", default=None)
    ap.add_argument("--tasks", default=None, help="tasks root (<type>/<dataset>/data.csv, labels.csv); needed for replay")
    ap.add_argument("--meta", default=None, help="synp meta dir: adds the phi stage")
    ap.add_argument("--no-replay", action="store_true")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    a = ap.parse_args(argv)
    if a.merge:
        def _read(d, f):
            q = Path(d) / f
            return pd.read_csv(q) if q.exists() and q.stat().st_size > 1 else pd.DataFrame(columns=["row_id"])
        rws = [_read(d, "funnel_rows.csv") for d in a.merge]
        eps = [_read(d, "funnel_episodes.csv") for d in a.merge]
        last_src = {rid: i for i, d in enumerate(rws) for rid in d.row_id}  # later folders win
        r = pd.concat([x[x.row_id.map(last_src) == i] for i, x in enumerate(rws)], ignore_index=True)
        e = pd.concat([x[x.row_id.map(last_src) == i] for i, x in enumerate(eps)], ignore_index=True)
        write_outputs(e, r, Path(a.out_dir))
        return 0
    if not a.runs or not a.manifest:
        ap.error("--runs and --manifest are required unless --merge is given")
    if a.repo not in sys.path:
        sys.path.insert(0, a.repo)
    from scripts.agent2027.analysis import recovery_detector as rd  # noqa: PLC0415

    man: dict[str, dict[str, str]] = {}
    for f in a.manifest:
        for r in csv.DictReader(open(f, newline="", encoding="utf-8")):
            man[r["row_id"]] = r
    rows = a.rows or [k for k, m in man.items() if not a.ablations or m["ablation_id"] in a.ablations]
    roots = [Path(r) for r in a.runs]
    cache: dict[str, Any] = {}
    ep_out, row_out = [], []
    for rid in rows:
        m = man[rid]
        row_dir, status, out_dir = locate(roots, rid)
        meta = {"row_id": rid, "ablation_id": m["ablation_id"], "arm_id": m["arm_id"], "rung": rung_of(m),
                "dataset_id": m["dataset_id"], "fold_seed": int(m["fold_seed"]), "replicate": m["replicate"],
                "status": status}
        if out_dir is None:
            row_out.append(meta)
            continue
        ds, seed = m["dataset_id"], int(m["fold_seed"])
        env = json.loads(m.get("env_overrides") or "{}")
        strat = str(env.get("ROGUE_ONE_STRATIFIED_FOLDS", "False")) == "True"
        df_raw = labels = phi = det = None
        if a.tasks:
            if ds not in cache:
                tdir = Path(a.tasks) / m["task_type"] / ds
                d = (pd.read_csv(tdir / "data.csv"), pd.read_csv(tdir / "labels.csv"))
                pd_ = (None, None)
                if a.meta and (Path(a.meta) / f"{ds}.json").exists():
                    pd_ = rd.load_phi(a.meta, ds, d[0])
                cache[ds] = d + pd_
            df_raw, labels, phi, det = cache[ds]
        _set_exec_compat(row_dir)
        try:
            eps = row_funnel(out_dir, df_raw=df_raw, labels=labels, fold_seed=seed, stratified=strat,
                             replay=not a.no_replay, phi=phi, det=det)
        except Exception as exc:  # noqa: BLE001
            print(f"[funnel] {rid}: {type(exc).__name__}: {exc}", file=sys.stderr)
            row_out.append({**meta, "error": f"{type(exc).__name__}: {exc}"[:300]})
            continue
        for e in eps:
            ep_out.append({**meta, **e})
        df = pd.DataFrame(eps)
        agg = {c: (df[c].sum(min_count=1) if c in df else np.nan) for c in STAGES + REJ + ("sel_any_valid",)}
        inst = out_dir / "instrumentation_summary.json"
        s = json.loads(inst.read_text(encoding="utf-8")) if inst.exists() else {}
        chk = {"append_attempts": s.get("append_attempts"), "append_accepted": s.get("append_accepted"),
               "append_restored": s.get("append_restored"), "replay_nan_fills": s.get("replay_nan_fills"),
               "n_folds": len(df)}
        if chk["append_attempts"] is not None and chk["append_restored"] is not None:
            chk["chk_proposed_minus_attempts"] = agg["fun_proposed"] - (chk["append_attempts"] - chk["append_restored"])
        if "phi_stage" in df:
            chk["phi_stage_counts"] = json.dumps(df["phi_stage"].value_counts().to_dict())
        row_out.append({**meta, **agg, **chk})
        print(f"[funnel] {rid} {ds} {status} " + " ".join(f"{k[4:]}={agg[k]}" for k in STAGES), file=sys.stderr,
              flush=True)
    write_outputs(pd.DataFrame(ep_out), pd.DataFrame(row_out), Path(a.out_dir))
    return 0


def write_outputs(e: pd.DataFrame, rdf: pd.DataFrame, od: Path) -> None:
    """funnel_episodes.csv, funnel_rows.csv, funnel_summary.csv (per ablation x arm x rung), funnel_phi_stage.csv."""
    od.mkdir(parents=True, exist_ok=True)
    e.to_csv(od / "funnel_episodes.csv", index=False)
    rdf.to_csv(od / "funnel_rows.csv", index=False)
    if "fun_proposed" in rdf:
        summarise(rdf).to_csv(od / "funnel_summary.csv", index=False)
    if len(e) and "phi_stage" in e:
        ct = e.dropna(subset=["phi_stage"]).groupby(["ablation_id", "arm_id", "rung", "phi_stage"]).size()
        ct = ct.unstack(fill_value=0)
        ct.reindex(columns=[c for c in PHI_ORDER if c in ct.columns]).to_csv(od / "funnel_phi_stage.csv")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
