"""E11b process metrics per row and per arm (01 section 4.6 and R13 "Process"; A07 section 2; A10/A11
section 2; A12-T17).

Columns (one row per manifest row; aggregated per ablation x arm):

* **errors** (A07 section 2, prereg Part V.7): ``err_per_accepted`` = ``n_tool_errors`` / ``n_accepted``
  (both reported, all roles and Extractor only), ``err_repeat_rate``, ``silent_rejections``,
  ``append_success`` -- read from E3's ``instrumentation_summary.json``; ``recover_within3`` = share
  of Extractor tool errors (``agent_toolcall_errors.jsonl``, sites h1-h9, v1, v3) followed by an
  accepted append (tool result "... added to df_features.") within the next 3 Extractor tool calls
  of the same iteration, reconstructed from ``llm_traces.jsonl.gz``; Extractor tokens per accepted
  feature.
* **pool** (01 4.6): ``reproposal_ratio`` = share of new LLM features (``Added`` = this trial,
  not seeded, not fallback) that duplicate an earlier row of the fold's pool (active or pruned):
  identical normalised AST of ``Pandas Command``, or |rho| >= 0.99 on the fold's training rows
  (value check only with ``--tasks``); ``reproposal_ratio_llm_pool`` counts only matches with an
  earlier LLM row (not an E6-seeded raw column); ``restore_pruned_count`` (E3 counter, restorations of
  pruned features counted separately); ``reproposal_ratio_a10`` (A10 variant: reference = every
  feature proposed earlier in the same fold across the per-iteration tables; AST only);
  ``prune_rate`` = sum over iterations of features pruned at that trial / features scored at that
  trial (``Scored`` = 1); ``floor_activations`` (E3); ``prune_jaccard`` = mean per Tester call of
  the Jaccard between the LLM's prunes and J-a's -- from ``pruning_shadow.jsonl`` when the run had
  the shadow (FX J+ cells), else post hoc (``prune_jaccard_basis = posthoc``) from the metric table
  logged in the Tester's assessment request, with the J-a rules of ``rogueone/utils/fx_switches.py``
  applied once without the VIF recompute and without the floor (descriptive only, 01 4.6);
  ``prune_jaccard_posthoc`` is computed on every row with a trace, so shadow rows validate it.
* **planner**: ``focus_novelty`` = share of focuses (iteration >= 1) whose maximum cosine to an
  earlier focus of the same fold is >= 0.9 (repeat share, A11 section 2), with the embedder named in
  ``focus_embedder``; ``focus_implementable`` = share of focuses whose iteration appended >= 1
  accepted non-seeded, non-fallback feature (rule label; kappa sheet below);
  ``report_read_rate`` (E3 counter: Scientist pandas calls touching ``feature_assessment``);
  ``lineage_share_cmd`` (with ``--meta``): share of fold-episodes with a phi-recovery whose
  earliest recovering feature's command references an earlier engineered (non-seed) feature;
  the focus half of ``lineage_share`` needs hand coding (lineage sheet).
* coverage and harness: ``raw_coverage`` (iteration 0, min), ``fallback_share``,
  ``rescues_per100_<family>``, ``context_errors``.

Hand-coded sheets (never auto-labelled): ``kappa_implementability_sheet.csv`` (60 focuses
stratified by the rule label, two blank rater columns; the rule label is only in the ``_key``
file), ``kappa_recovery_sheet.csv`` (60 fold-episodes stratified by detector hit/miss, with
``--meta``) and ``lineage_sheet.csv``.

    python process_metrics.py --runs <runroot>/S* --manifest ~/ro27_manifests/*.csv \
        [--rows A06-001 ...] [--ablations A05 A06] [--tasks <tasks root> --meta <synp meta>] \
        [--embed tfidf|st:<model>] --out-dir <dir>

Run on CAIR only inside a CPU container with the run folders mounted read-only.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

ACCEPT_TEXT = "added to df_features."
TOOL_ERROR_SITES = {f"h{i}" for i in range(1, 10)} | {"v1", "v3"}
RHO_DUP = 0.99
FOCUS_REPEAT_COS = 0.9
KAPPA_N = 60
METRIC_TABLE_RE = re.compile(r"\|\s*feature_name\s*\|\s*mutual_information\s*\|\s*vif\s*\|\s*pfi_mean\s*\|"
                             r"\s*pfi_std\s*\|\s*noise_sensitivity\s*\|\s*mean_abs_shap\s*\|")
# J-a constants (rogueone/utils/fx_switches.py; A12-T7)
VIF_CLUSTER, VIF_HIGH, SHAP_KEEP = 1e10, 10.0, 0.1
SUMMARY_KEYS = ["err_per_accepted", "n_tool_errors", "n_accepted", "err_role_extractor", "err_repeat_rate",
                "silent_rejections", "append_success", "append_attempts", "append_accepted", "append_restored",
                "restore_pruned_count", "floor_activations", "floor_restored_features", "report_read_rate",
                "report_reads", "scientist_pandas_calls", "fallback_share", "context_errors", "max_prompt_tokens",
                "tok_compl_extractor", "rescues_per100_gpt-oss", "rescues_per100_qwen", "loop_guard",
                "fx_prune_candidates", "fx_pruned", "fx_shadow_iterations", "iterations_total"]


# =========================================================================== pure helpers
def norm_ast(cmd: Any) -> Optional[str]:
    """Normalised AST of a pandas command (formatting, comments and quoting removed); None if it
    does not parse."""
    if not isinstance(cmd, str) or not cmd.strip():
        return None
    try:
        return ast.dump(ast.parse(cmd.strip()), annotate_fields=False, include_attributes=False)
    except SyntaxError:
        return None


def _flag(v: Any) -> bool:
    try:
        return bool(int(float(v)))
    except (TypeError, ValueError):
        return False


def new_llm_rows(table: pd.DataFrame, trial: Optional[str] = None) -> pd.Series:
    """Rows proposed by the LLM (not E6-seeded, not deterministic fallback); optionally only those
    added at ``trial`` (``Trial <index>``)."""
    m = ~table["Seeded"].map(_flag) & ~table.get("Fallback", pd.Series(0, index=table.index)).map(_flag)
    m &= table["Added"].astype(str) != "Seed"
    if trial is not None:
        m &= table["Added"].astype(str) == trial
    return m


def reproposals(table: pd.DataFrame, values: Optional[pd.DataFrame] = None, rho: float = RHO_DUP) -> pd.DataFrame:
    """For each new LLM row of an append-only fold table: does it duplicate an earlier row (any
    status, seeds included) by normalised AST, or by |Pearson r| >= ``rho`` on training-row values
    (``values``: one column per attribute name, training rows only)?"""
    t = table.reset_index(drop=True)
    asts = [norm_ast(c) for c in t["Pandas Command"]]
    is_new = new_llm_rows(t)
    seeded = t["Seeded"].map(_flag) | (t["Added"].astype(str) == "Seed")
    out = []
    for i in range(len(t)):
        if not is_new.iloc[i]:
            continue
        name = str(t.loc[i, "Attribute"])
        match_rows = [j for j in range(i) if asts[i] is not None and asts[j] == asts[i]]
        if values is not None and name in values.columns:
            v = pd.to_numeric(values[name], errors="coerce").to_numpy(float)
            best = float("nan")
            for j in range(i):
                other = str(t.loc[j, "Attribute"])
                if other == name or other not in values.columns:
                    continue
                r = abs_corr(v, pd.to_numeric(values[other], errors="coerce").to_numpy(float))
                if not math.isnan(r):
                    best = r if math.isnan(best) else max(best, r)
                    if r >= rho:
                        match_rows.append(j)
        else:
            best = float("nan")
        dup_ast = any(asts[i] is not None and asts[j] == asts[i] for j in range(i))
        dup_val = (not math.isnan(best)) and best >= rho
        out.append({"attribute": name, "added": str(t.loc[i, "Added"]), "dup_ast": bool(dup_ast),
                    "dup_value": bool(dup_val), "max_abs_r": best, "reproposal": bool(dup_ast or dup_val),
                    "reproposal_of_llm_row": any(not seeded.iloc[j] for j in match_rows)})
    return pd.DataFrame(out, columns=["attribute", "added", "dup_ast", "dup_value", "max_abs_r", "reproposal",
                                      "reproposal_of_llm_row"])


def abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3:
        return float("nan")
    a, b = a[ok], b[ok]
    if np.std(a) == 0 or np.std(b) == 0:
        return 1.0 if np.std(a) == 0 and np.std(b) == 0 else float("nan")
    return float(abs(np.corrcoef(a, b)[0, 1]))


def reproposal_a10(trails: Sequence[pd.DataFrame]) -> tuple[int, int]:
    """A10 variant (AST only): reference = every row of earlier per-iteration tables of the fold plus
    earlier rows of the same table. Returns (reproposals, new rows)."""
    seen: set[str] = set()
    n_new = n_dup = 0
    for t in trails:
        trial = str(t["Trial"].iloc[0]) if "Trial" in t.columns and len(t) else None
        is_new = new_llm_rows(t, trial)
        for i, (cmd, new) in enumerate(zip(t["Pandas Command"], is_new)):
            a = norm_ast(cmd)
            if new:
                n_new += 1
                n_dup += int(a is not None and a in seen)
            if a is not None:
                seen.add(a)
    return n_dup, n_new


def prune_counts(trail: pd.DataFrame) -> tuple[int, int]:
    """(pruned at this trial, scored at this trial) for one per-iteration table."""
    trial = str(trail["Trial"].iloc[0])
    pruned = int((trail["Pruned Iteration"].astype(str) == trial).sum())
    scored = int(trail["Scored"].map(_flag).sum()) if "Scored" in trail.columns else 0
    return pruned, scored


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = set(map(str, a)), set(map(str, b))
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def parse_metric_table(text: str) -> Optional[pd.DataFrame]:
    """The Tester's pre-computed metric table (markdown) from an assessment request."""
    m = METRIC_TABLE_RE.search(text or "")
    if not m:
        return None
    rows = []
    for line in text[m.start():].splitlines()[2:]:
        line = line.strip()
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 7:
            break

        def num(x):
            try:
                return float(x)
            except ValueError:
                return None
        rows.append({"feature": cells[0], "mi": num(cells[1]), "vif": num(cells[2]), "pfi_mean": num(cells[3]),
                     "pfi_std": num(cells[4]), "shap": num(cells[6])})
    return pd.DataFrame(rows)


def ja_posthoc(mt: pd.DataFrame, protected: Iterable[str] = ()) -> list[str]:
    """J-a decisions from a logged metric table, single pass: at most one ``ja_vif_cluster`` prune
    (lowest SHAP, ties lowest MI then name, among VIF >= 1e10 and SHAP <= 0.1; the VIF recompute
    after it needs the training matrix, which is not logged), then ``ja_vif_pfi``: VIF in [10, 1e10),
    |pfi_mean| <= pfi_std and SHAP < 0.1. Seeded rows are protected; no floor."""
    prot = set(map(str, protected))
    rows = mt[~mt.feature.isin(prot)]
    out = []
    cl = rows[rows.vif.notna() & (rows.vif >= VIF_CLUSTER) & rows.shap.notna() & (rows.shap <= SHAP_KEEP)]
    if len(cl):
        cl = cl.assign(_mi=cl.mi.fillna(math.inf)).sort_values(["shap", "_mi", "feature"])
        out.append(str(cl.feature.iloc[0]))
    for r in rows.itertuples():
        if r.feature in out or any(v is None or (isinstance(v, float) and math.isnan(v))
                                   for v in (r.vif, r.pfi_mean, r.pfi_std, r.shap)):
            continue
        if VIF_HIGH <= r.vif < VIF_CLUSTER and abs(r.pfi_mean) <= r.pfi_std and r.shap < SHAP_KEEP:
            out.append(str(r.feature))
    return out


def recover_within3(calls: Sequence[Mapping[str, Any]], window: int = 3) -> tuple[int, int]:
    """``calls``: Extractor tool calls of one iteration in order, each with ``error`` (bool) and
    ``accepted`` (bool). Returns (errors followed by an accepted append within the next ``window``
    calls, errors)."""
    n_err = n_rec = 0
    for i, c in enumerate(calls):
        if not c.get("error"):
            continue
        n_err += 1
        n_rec += int(any(x.get("accepted") for x in calls[i + 1:i + 1 + window]))
    return n_rec, n_err


def focus_repeat(focuses: Sequence[tuple[int, int, str]], embed) -> dict[str, Any]:
    """``focuses``: (fold, iteration, text). For each focus after the first of its fold, the maximum
    cosine to an earlier focus of the same fold; repeat iff >= 0.9."""
    by: dict[int, list[tuple[int, str]]] = {}
    for f, it, txt in focuses:
        by.setdefault(f, []).append((it, txt or ""))
    maxcos, details = [], []
    for f, lst in by.items():
        lst.sort()
        if len(lst) < 2:
            continue
        V = embed([t for _, t in lst])
        n = np.linalg.norm(V, axis=1, keepdims=True)
        n[n == 0] = 1.0
        V = V / n
        S = V @ V.T
        for i in range(1, len(lst)):
            mc = float(np.max(S[i, :i]))
            maxcos.append(mc)
            details.append({"fold": f, "iteration": lst[i][0], "max_cos_prev": mc})
    rep = [m >= FOCUS_REPEAT_COS for m in maxcos]
    return {"focus_novelty": float(np.mean(rep)) if rep else float("nan"),
            "focus_max_cos_mean": float(np.mean(maxcos)) if maxcos else float("nan"),
            "n_focus_compared": len(maxcos), "details": details}


def tfidf_embedder(texts: Sequence[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415
    try:
        return TfidfVectorizer(sublinear_tf=True).fit_transform(texts).toarray()
    except ValueError:  # empty vocabulary
        return np.zeros((len(texts), 1))


def make_embedder(spec: str):
    if spec == "tfidf":
        return tfidf_embedder, "tfidf (proxy; the embedding model is not named in the pre-registration)"
    if spec.startswith("st:"):
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        model = SentenceTransformer(spec[3:])
        return (lambda xs: np.asarray(model.encode(list(xs)))), spec
    raise ValueError(f"unknown embedder {spec}")


def stratified_sample(df: pd.DataFrame, label: str, n: int = KAPPA_N, seed: int = 20261101) -> pd.DataFrame:
    """Half from each label value where possible (the rest from the other), shuffled."""
    if not len(df):
        return df
    rng = np.random.default_rng(seed)
    pos, neg = df[df[label] == 1], df[df[label] != 1]
    k_pos = min(len(pos), max(n // 2, n - len(neg)))
    k_neg = min(len(neg), n - k_pos)
    pick = pd.concat([pos.iloc[rng.permutation(len(pos))[:k_pos]], neg.iloc[rng.permutation(len(neg))[:k_neg]]])
    return pick.iloc[rng.permutation(len(pick))].reset_index(drop=True)


# =========================================================================== run-folder readers
def _read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def read_trails(out: Path) -> list[pd.DataFrame]:
    tr = []
    for f in (out / "intermediate_steps").glob("df_attributes_explanations_trail_*.csv"):
        df = pd.read_csv(f)
        if len(df):
            tr.append(df)
    return sorted(tr, key=lambda d: int(str(d["Trial"].iloc[0]).split()[-1]))


def extractor_calls(trace: Path, errors: Sequence[Mapping[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Extractor tool calls per iteration index, in order, with ``accepted`` (tool result says the
    attribute was added) and ``error`` (matched to a tool-error event of that step: the latest call
    emitted before the event's timestamp, preferring a call whose result contains the message)."""
    calls: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    results: dict[str, str] = {}
    with gzip.open(trace, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if '"ExtractorAgent"' not in line[:600]:
                continue
            r = json.loads(line)
            if r.get("agent_name") != "ExtractorAgent":
                continue
            idx, ts = r.get("index"), r.get("ts_end") or r.get("ts_start")
            msgs = list(r.get("messages") or [])
            for m in msgs:
                if m.get("role") == "tool" and m.get("tool_call_id"):
                    results[m["tool_call_id"]] = str(m.get("content") or "")
            resp = r.get("message") or {}
            for pos, tc in enumerate(resp.get("tool_calls") or []):
                cid = tc.get("id") or f"{r.get('seq')}:{pos}"
                if cid not in calls:
                    calls[cid] = {"id": cid, "index": idx, "ts": ts, "seq": r.get("seq"), "pos": pos,
                                  "name": (tc.get("function") or {}).get("name")}
                    order.append(cid)
    by: dict[int, list[dict[str, Any]]] = {}
    for cid in order:
        c = calls[cid]
        res = results.get(cid, "")
        c["accepted"] = c["name"] == "append_new_attribute" and ACCEPT_TEXT in res
        c["error"] = False
        c["result"] = res
        by.setdefault(c["index"], []).append(c)
    for lst in by.values():
        lst.sort(key=lambda c: (c["ts"] or 0, c["seq"] or 0, c["pos"]))
    for e in errors:
        if e.get("agent_name") != "ExtractorAgent" or (e.get("context") or {}).get("site") not in TOOL_ERROR_SITES:
            continue
        try:
            t = datetime.fromisoformat(str(e["timestamp_utc"])).timestamp()
        except (KeyError, ValueError):
            continue
        lst = [c for c in by.get(e.get("step"), []) if (c["ts"] or 0) <= t and not c["error"]]
        if not lst:
            continue
        msg = str(e.get("message") or "")[:60]
        cand = [c for c in lst if msg and msg in c["result"] and c["name"] == e.get("tool_name")]
        (cand[-1] if cand else lst[-1])["error"] = True
    return by


# =========================================================================== per row
def row_metrics(out: Path, *, embed=None, embed_name: str = "", tasks: Optional[Path] = None,
                meta: Optional[Path] = None, dataset: str = "", task_type: str = "", fold_seed: int = 42,
                stratified: bool = False, sheets: Optional[dict] = None, row_id: str = "") -> dict[str, Any]:
    rec: dict[str, Any] = {}
    summ = _read_json(out / "instrumentation_summary.json") or {}
    for k in SUMMARY_KEYS:
        rec[k] = summ.get(k)
    na = summ.get("n_accepted")
    rec["tok_compl_extractor_per_accepted"] = (summ.get("tok_compl_extractor") / na) if na else None
    if summ.get("fx_prune_candidates"):
        rec["fx_prune_rate"] = summ.get("fx_pruned", 0) / summ["fx_prune_candidates"]

    # errors: recover_within3
    trace = out / "llm_traces.jsonl.gz"
    errs = _jsonl(out / "agent_toolcall_errors.jsonl")
    rec["n_tool_errors_extractor_jsonl"] = sum(1 for e in errs if e.get("agent_name") == "ExtractorAgent"
                                               and (e.get("context") or {}).get("site") in TOOL_ERROR_SITES)
    if trace.exists():
        try:
            by = extractor_calls(trace, errs)
            n_rec = n_err = 0
            for lst in by.values():
                a, b = recover_within3(lst)
                n_rec, n_err = n_rec + a, n_err + b
            rec.update({"recover_within3": n_rec / n_err if n_err else None, "recover_within3_num": n_rec,
                        "recover_within3_den": n_err,
                        "accepted_appends_trace": sum(c["accepted"] for lst in by.values() for c in lst)})
        except Exception as exc:  # noqa: BLE001
            rec["recover_within3_error"] = f"{type(exc).__name__}: {exc}"

    # pool
    trails = read_trails(out)
    it_log = _jsonl(out / "iteration_log.jsonl")
    run_to_fold = {int(r["fold_run_id"]): int(r["test_fold_id"]) for r in it_log if r.get("fold_run_id") is not None}
    cov0 = [r.get("raw_coverage") for r in it_log if r.get("iteration_in_fold") == 0 and r.get("raw_coverage") is not None]
    rec["raw_coverage_iter0_min"] = min(cov0) if cov0 else None
    by_fold: dict[int, list[pd.DataFrame]] = {}
    for t in trails:
        by_fold.setdefault(int(t["fold_run_id"].iloc[0]), []).append(t)
    n_new = n_dup = n_dup_ast = n_dup_val = n_dup_llm = 0
    a10_dup = a10_new = 0
    pr_num = pr_den = 0
    value_basis = "ast"
    det_cache = None
    for fr, lst in by_fold.items():
        lst.sort(key=lambda d: int(d["iteration_in_fold"].iloc[0]))
        for t in lst:
            p, s = prune_counts(t)
            pr_num, pr_den = pr_num + p, pr_den + s
        d, n = reproposal_a10(lst)
        a10_dup, a10_new = a10_dup + d, a10_new + n
        last = lst[-1]
        values = None
        if tasks is not None and dataset:
            try:
                if det_cache is None:
                    det_cache = _load_task(tasks, dataset, task_type)
                df_raw, labels = det_cache
                values = _materialize(df_raw, labels, last, fold_seed, run_to_fold.get(fr, fr), stratified)
                value_basis = "ast+value"
            except Exception as exc:  # noqa: BLE001
                rec["reproposal_value_error"] = f"{type(exc).__name__}: {exc}"
        rp = reproposals(last, values)
        n_new += len(rp)
        n_dup += int(rp.reproposal.sum())
        n_dup_ast += int(rp.dup_ast.sum())
        n_dup_val += int(rp.dup_value.sum())
        n_dup_llm += int(rp.reproposal_of_llm_row.sum())
    rec.update({"reproposal_ratio": n_dup / n_new if n_new else None, "reproposal_n": n_dup, "new_llm_features": n_new,
                "reproposal_ratio_llm_pool": n_dup_llm / n_new if n_new else None,
                "reproposal_ast_n": n_dup_ast, "reproposal_value_n": n_dup_val, "reproposal_basis": value_basis,
                "reproposal_ratio_a10": a10_dup / a10_new if a10_new else None,
                "prune_rate": pr_num / pr_den if pr_den else None, "pruned_n": pr_num, "scored_n": pr_den})

    # prune_jaccard
    shadow = _jsonl(out / "pruning_shadow.jsonl")
    if shadow:
        rec["prune_jaccard"] = float(np.mean([s["jaccard"] for s in shadow]))
        rec["prune_jaccard_basis"] = "shadow"
        rec["prune_jaccard_n"] = len(shadow)
    if trace.exists() and trails:  # post hoc always (on shadow rows it validates the approximation)
        try:
            tables = tester_metric_tables(trace)
            tr_by_idx = {int(str(t["Trial"].iloc[0]).split()[-1]): t for t in trails}
            jac = []
            for idx, mt in tables.items():
                t = tr_by_idx.get(idx)
                if t is None or mt is None or not len(mt):
                    continue
                llm = t.loc[t["Pruned Iteration"].astype(str) == f"Trial {idx}", "Attribute"].astype(str)
                seeded = t.loc[t["Seeded"].map(_flag), "Attribute"].astype(str)
                jac.append(jaccard(llm, ja_posthoc(mt, seeded)))
            rec["prune_jaccard_posthoc"] = float(np.mean(jac)) if jac else None
            rec["prune_jaccard_posthoc_n"] = len(jac)
            if not shadow:
                rec["prune_jaccard"] = rec["prune_jaccard_posthoc"]
                rec["prune_jaccard_basis"] = "posthoc"
                rec["prune_jaccard_n"] = len(jac)
        except Exception as exc:  # noqa: BLE001
            rec["prune_jaccard_error"] = f"{type(exc).__name__}: {exc}"

    # planner
    focus = _jsonl(out / "focus_log.jsonl")
    if focus and embed is not None:
        fr = focus_repeat([(int(f["fold_run_id"]), int(f["iteration_in_fold"]), str(f.get("focus") or ""))
                           for f in focus], embed)
        rec.update({k: v for k, v in fr.items() if k != "details"})
        rec["focus_embedder"] = embed_name
    tr_by_idx = {int(str(t["Trial"].iloc[0]).split()[-1]): t for t in trails}
    impl = []
    for f in focus:
        idx = int(f["index"])
        t = tr_by_idx.get(idx)
        if t is None:
            continue
        added = t[new_llm_rows(t, f"Trial {idx}")]
        lab = int(len(added) > 0)
        impl.append(lab)
        if sheets is not None:
            sheets["impl"].append({"row_id": row_id, "index": idx, "fold_run_id": f.get("fold_run_id"),
                                   "iteration_in_fold": f.get("iteration_in_fold"), "focus_source": f.get("source"),
                                   "focus": f.get("focus"),
                                   "features_added": "\n".join(f"{a}: {c}" for a, c in zip(added["Attribute"], added["Pandas Command"])),
                                   "rule_label": lab})
    rec["focus_implementable"] = float(np.mean(impl)) if impl else None
    rec["focus_n"] = len(focus)
    rec["focus_fallback_n"] = sum(1 for f in focus if str(f.get("source")) != "llm")

    # lineage (needs the detector)
    if meta is not None and tasks is not None and dataset.startswith("synp_"):
        try:
            rec.update(lineage(out, tasks, meta, dataset, task_type, fold_seed, stratified, focus, sheets, row_id))
        except Exception as exc:  # noqa: BLE001
            rec["lineage_error"] = f"{type(exc).__name__}: {exc}"
    return rec


def tester_metric_tables(trace: Path) -> dict[int, pd.DataFrame]:
    """First logged metric table per iteration index (Tester assessment request)."""
    out: dict[int, pd.DataFrame] = {}
    with gzip.open(trace, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if '"TesterAgent"' not in line[:600] or "mutual_information" not in line:
                continue
            r = json.loads(line)
            idx = r.get("index")
            if idx is None or idx in out:
                continue
            for m in r.get("messages") or []:
                if m.get("role") == "user":
                    mt = parse_metric_table(str(m.get("content") or ""))
                    if mt is not None:
                        out[int(idx)] = mt
                        break
    return out


def _load_task(tasks: Path, dataset: str, task_type: str):
    tdir = Path(tasks) / task_type / dataset
    return pd.read_csv(tdir / "data.csv"), pd.read_csv(tdir / "labels.csv")


def _materialize(df_raw, labels, table, fold_seed, fold, stratified):
    from scripts.agent2027.analysis import recovery_detector as rd  # noqa: PLC0415
    from rogueone.utils.fold_safe_features import materialize_fold_safe_features  # noqa: PLC0415
    mask = rd.train_mask_for_fold(df_raw, labels, fold_seed, fold, stratified=stratified)
    feats = materialize_fold_safe_features(df_raw, table, mask)
    return feats[np.asarray(mask, bool)].reset_index(drop=True) if len(feats) == len(df_raw) else feats


def _engineered_refs(cmd: str, engineered: Iterable[str]) -> list[str]:
    refs = set(re.findall(r"df_features\s*\[\s*['\"]([^'\"]+)['\"]\s*\]", cmd or ""))
    return sorted(refs & set(engineered))


def lineage(out, tasks, meta, dataset, task_type, fold_seed, stratified, focus, sheets, row_id) -> dict[str, Any]:
    """phi-recoveries per fold-episode (detector of record, amended) and whether the earliest
    recovering feature's command references an earlier engineered feature. Also feeds the kappa
    recovery sheet (hits and misses) and the lineage sheet (focus half, hand-coded)."""
    from scripts.agent2027.analysis import a02_readout as a02  # noqa: PLC0415
    from scripts.agent2027.analysis import recovery_detector as rd  # noqa: PLC0415
    df_raw, labels = _load_task(tasks, dataset, task_type)
    phi, det = rd.load_phi(meta, dataset, df_raw)
    focus_by_idx = {int(f["index"]): f for f in focus}
    n_rec = n_cmd = 0
    for ep in a02.build_row_episodes(out):
        r = a02.score_episode(ep, df_raw, labels, phi, det, fold_seed, stratified, amended=True)
        last = ep["trails"][-1]["attributes"] if ep["trails"] else None
        if sheets is not None:
            sel = next((t for t in ep["trails"] if t["iteration_in_fold"] == ep["selected_iter"]), None)
            act = sel["attributes"][sel["attributes"]["Attribute"].isin(sel["active"])] if sel else None
            sheets["rec"].append({"row_id": row_id, "dataset_id": dataset, "fold": ep["fold"],
                                  "selected_iter": ep["selected_iter"],
                                  "active_features_selected_iter": "" if act is None else
                                  "\n".join(f"{a}: {c}" for a, c in zip(act["Attribute"], act["Pandas Command"])),
                                  "detector_rec_sel": int(r["rec_sel"]), "detector_hits_sel": r.get("hits_sel", "")})
        if not r["rec_any"] or last is None:
            continue
        n_rec += 1
        hits = [h for h in str(r.get("hits_any", "")).split(";") if h]
        t = last.reset_index(drop=True)
        pos = {str(a): i for i, a in enumerate(t["Attribute"].astype(str))}
        hits = sorted((h for h in hits if h in pos), key=lambda h: pos[h])
        if not hits:
            continue
        h = hits[0]
        i = pos[h]
        eng = t.loc[:i - 1][new_llm_rows(t.loc[:i - 1])]["Attribute"].astype(str) if i > 0 else []
        refs = _engineered_refs(str(t.loc[i, "Pandas Command"]), eng)
        n_cmd += int(bool(refs))
        if sheets is not None:
            added = str(t.loc[i, "Added"])
            idx = int(added.split()[-1]) if added.startswith("Trial") else None
            fo = focus_by_idx.get(idx, {}) if idx is not None else {}
            earlier = [focus_by_idx[j]["focus"] for j in sorted(focus_by_idx)
                       if idx is not None and j < idx and focus_by_idx[j].get("fold_run_id") == fo.get("fold_run_id")]
            sheets["lineage"].append({"row_id": row_id, "fold": ep["fold"], "feature": h,
                                      "command": t.loc[i, "Pandas Command"], "added": added,
                                      "cmd_refs_engineered": ";".join(refs), "focus_of_iteration": fo.get("focus"),
                                      "earlier_focuses_same_fold": "\n---\n".join(map(str, earlier)),
                                      "rater1_focus_builds_on_earlier": "", "rater2_focus_builds_on_earlier": ""})
    return {"phi_recoveries": n_rec, "lineage_share_cmd": n_cmd / n_rec if n_rec else None}


# =========================================================================== driver
def _status(run_roots: Sequence[Path]) -> pd.DataFrame:
    frames = []
    for r in run_roots:
        p = Path(r) / "status.tsv"
        if p.exists():
            frames.append(pd.read_csv(p, sep="\t", dtype=str).assign(session=Path(r).name, root=str(r)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


AGG_COLS = ["err_per_accepted", "err_repeat_rate", "recover_within3", "silent_rejections", "append_success",
            "reproposal_ratio", "reproposal_ratio_llm_pool", "reproposal_ratio_a10", "restore_pruned_count",
            "prune_rate", "floor_activations", "prune_jaccard", "prune_jaccard_posthoc", "focus_novelty", "focus_implementable", "report_read_rate", "lineage_share_cmd",
            "raw_coverage_iter0_min", "fallback_share", "context_errors", "tok_compl_extractor_per_accepted"]


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (abl, arm), g in rows.groupby(["ablation_id", "arm_id"]):
        rec = {"ablation_id": abl, "arm_id": arm, "n_rows": len(g), "n_with_output": int(g.has_output.sum())}
        for c in AGG_COLS:
            if c in g.columns:
                v = pd.to_numeric(g[c], errors="coerce").dropna()
                rec[f"{c}_mean"] = float(v.mean()) if len(v) else None
                rec[f"{c}_median"] = float(v.median()) if len(v) else None
                rec[f"{c}_n"] = int(len(v))
        out.append(rec)
    return pd.DataFrame(out)


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - I/O driver; pure parts unit-tested
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--rows", nargs="*", help="row ids (default: every manifest row found in the runs)")
    ap.add_argument("--ablations", nargs="*", help="restrict to these ablation ids")
    ap.add_argument("--tasks", help="tasks root (<type>/<dataset>/data.csv): value-based re-proposal, lineage")
    ap.add_argument("--meta", help="SYN-P phi metadata: lineage and the kappa recovery sheet")
    ap.add_argument("--embed", default="tfidf", help="tfidf | st:<sentence-transformers model>")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    if a.repo not in sys.path:
        sys.path.insert(0, a.repo)
    man = pd.concat([pd.read_csv(m, dtype=str) for m in a.manifest if Path(m).exists()], ignore_index=True)
    man = man[man.columns[man.columns.isin(["row_id", "ablation_id", "arm_id", "dataset_id", "task_type",
                                             "fold_seed", "replicate", "env_overrides", "is_sentinel", "notes"])]]
    man = man.dropna(subset=["row_id"]).drop_duplicates("row_id", keep="last")
    man = man[~man.dataset_id.astype(str).str.contains("covtype")]  # never analysed (registry ban)
    st = _status([Path(r) for r in a.runs])
    if a.rows:
        man = man[man.row_id.isin(a.rows)]
    if a.ablations:
        man = man[man.ablation_id.isin(a.ablations)]
    man = man[man.row_id.isin(set(st.row_id))] if len(st) else man.iloc[0:0]
    embed, embed_name = make_embedder(a.embed)
    sheets = {"impl": [], "rec": [], "lineage": []}
    rows = []
    for m in man.itertuples():
        s = st[st.row_id == m.row_id].copy()
        s["attempt"] = s["attempt"].astype(int)
        s = s.sort_values(["session", "attempt"])
        final = s.iloc[-1]
        rdir = Path(final["root"]) / m.row_id
        outs = sorted(rdir.glob("tasks/*/*/output"))
        rec = {"row_id": m.row_id, "ablation_id": m.ablation_id, "arm_id": m.arm_id, "dataset_id": m.dataset_id,
               "fold_seed": m.fold_seed, "replicate": m.replicate, "session": final["session"],
               "final_status": final["status"], "cap_hit": int((s["status"] == "CAP_HIT").any()),
               "has_output": int(bool(outs))}
        if outs:
            env = json.loads(m.env_overrides) if isinstance(m.env_overrides, str) and m.env_overrides.strip() else {}
            strat = str(env.get("ROGUE_ONE_STRATIFIED_FOLDS", "False")) == "True"
            try:
                rec.update(row_metrics(outs[0], embed=embed, embed_name=embed_name,
                                       tasks=Path(a.tasks) if a.tasks else None, meta=Path(a.meta) if a.meta else None,
                                       dataset=m.dataset_id, task_type=m.task_type, fold_seed=int(m.fold_seed),
                                       stratified=strat, sheets=sheets, row_id=m.row_id))
            except Exception as exc:  # noqa: BLE001
                rec["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(rec)
        print(f"[process_metrics] {m.row_id} {m.dataset_id} {rec['final_status']}", file=sys.stderr, flush=True)
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(od / "process_rows.csv", index=False)
    if len(df):
        aggregate(df).to_csv(od / "process_by_arm.csv", index=False)
    write_sheets(sheets, od)
    print(f"[process_metrics] {len(df)} rows -> {od}", file=sys.stderr)
    return 0


def write_sheets(sheets: Mapping[str, list], od: Path) -> None:
    """kappa sheets: the rater-facing file has blank label columns and no rule/detector label; the
    ``_key`` file maps ``sheet_id`` to the automatic label."""
    for name, src, label in (("kappa_implementability", "impl", "rule_label"),
                             ("kappa_recovery", "rec", "detector_rec_sel")):
        df = pd.DataFrame(sheets.get(src, []))
        if not len(df):
            continue
        smp = stratified_sample(df, label)
        smp.insert(0, "sheet_id", [f"{name[6:9]}{i:03d}" for i in range(1, len(smp) + 1)])
        hidden = [c for c in (label, "detector_hits_sel") if c in smp.columns]
        smp[["sheet_id"] + hidden + ["row_id"]].to_csv(od / f"{name}_sheet_key.csv", index=False)
        rater = smp.drop(columns=hidden)
        rater["rater1_label"] = ""
        rater["rater2_label"] = ""
        rater["consensus_label"] = ""
        rater.to_csv(od / f"{name}_sheet.csv", index=False)
    if sheets.get("lineage"):
        pd.DataFrame(sheets["lineage"]).to_csv(od / "lineage_sheet.csv", index=False)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
