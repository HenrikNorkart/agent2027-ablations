"""A5 exploratory: what does LLM pruning do differently from the rule pruner?

For the 72 confirmatory A12 factorial runs (P2-P6b, fold seed 42):
  J+ runs (LLM pruning): per iteration, the LLM's prunes vs the rule pruner's shadow
  decisions on the same table (output/pruning_shadow.jsonl).
  J- runs (rules): the rules' actual prunes (Prune Rationale ja_*).
For each fold run, the planted-feature matches are found with the registered detector
(recovery_detector.detect_episode, amended, 'counted'), over every attribute the fold run
ever created, so pruned features are scored too.
Also scored: 'components' of phi* = features with |Spearman| >= 0.8 with phi* that are not
matches (partial hypotheses, e.g. load/area on P2).
"""
import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/code")
from scripts.agent2027.analysis import a02_readout as a02  # noqa: E402
from scripts.agent2027.analysis import recovery_detector as rd  # noqa: E402

RUNS = Path("/runs")
runs = [r for r in csv.DictReader(open("/work/a12_runs.csv")) if r["J"] == "1"]
man = {}
for f in Path("/code/manifests").glob("A12*.csv"):
    for r in csv.DictReader(open(f, newline="", encoding="utf-8")):
        man[r["row_id"]] = r

cache = {}
per_iter, per_feat = [], []
for r in runs:
    rid, ds, tt, J = r["row_id"], r["dataset_id"], r["task_type"], int(r["J"])
    m = man[rid]
    strat = json.loads(m["env_overrides"] or "{}").get("ROGUE_ONE_STRATIFIED_FOLDS", "False") == "True"
    outs = sorted((RUNS / r["session"] / rid).glob("tasks/*/*/output"))
    if not outs:
        print("missing", rid, file=sys.stderr); continue
    out = outs[0]
    if ds not in cache:
        tdir = Path("/tasks") / tt / ds
        df_raw = pd.read_csv(tdir / "data.csv"); labels = pd.read_csv(tdir / "labels.csv")
        phi, det = rd.load_phi("/meta", ds, df_raw)
        cache[ds] = (df_raw, labels, phi, det)
    df_raw, labels, phi, det = cache[ds]
    trails = a02.read_trails(out)
    it_log = [json.loads(l) for l in (out / "iteration_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    run_to_test = {int(x["fold_run_id"]): int(x["test_fold_id"]) for x in it_log}
    by_run = defaultdict(list)
    for t in trails:
        by_run[t["fold_run_id"]].append(t)
    status = {}  # (fold_run, feature) -> planted / component / other
    active_post = {}
    for fr, ts in by_run.items():
        fold = run_to_test.get(fr)
        if fold is None:
            continue
        attrs = pd.concat([t["attributes"] for t in ts]).drop_duplicates("Attribute", keep="last")
        mask = rd.train_mask_for_fold(df_raw, labels, 42, fold, stratified=strat)
        try:
            res = rd.detect_episode(df_raw, attrs, mask, phi, det, {}, None, amended=True)
        except Exception as exc:  # noqa: BLE001
            print(rid, fr, "detector error", exc, file=sys.stderr); continue
        pf = res.per_feature
        for _, f in pf.iterrows():
            rho = abs(float(f["rho"])) if pd.notna(f["rho"]) else 0.0
            kind = "planted" if f["counted"] else ("component" if (rho >= 0.8 and not str(f["excluded"]).startswith("fallback")) else "other")
            status[(fr, str(f["feature"]))] = kind
        for t in ts:
            act = [str(x) for x in t["active"]]
            active_post[(fr, int(t["iteration_in_fold"]))] = (
                sum(status.get((fr, n)) == "planted" for n in act),
                sum(status.get((fr, n)) == "component" for n in act), len(act))
        last = ts[-1]["attributes"]
        for _, a in last.iterrows():
            name = str(a["Attribute"])
            per_feat.append({"row_id": rid, "dataset_id": ds, "J": J, "fold_run": fr, "feature": name,
                             "kind": status.get((fr, name), "other"), "status": a.get("Status"),
                             "rationale": str(a.get("Prune Rationale", ""))[:300],
                             "pruned_iter": a.get("Pruned Iteration")})
    if J:
        sh = out / "pruning_shadow.jsonl"
        if not sh.exists():
            print("no shadow", rid, file=sys.stderr); continue
        for line in sh.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            s = json.loads(line)
            fr = int(s["fold_run_id"])
            L, R = set(s.get("llm_pruned") or []), set(s.get("shadow_pruned") or [])
            rule_of = {d["feature"]: d["rule"] for d in s.get("shadow_decisions") or []}
            k = lambda n: status.get((fr, n), "other")
            per_iter.append({
                "row_id": rid, "dataset_id": ds, "fold_run": fr, "iter": s["iteration_in_fold"],
                "pre_active": s.get("pre_active"), "n_llm": len(L), "n_rules": len(R), "n_both": len(L & R),
                "llm_planted": sum(k(n) == "planted" for n in L), "rules_planted": sum(k(n) == "planted" for n in R),
                "llm_comp": sum(k(n) == "component" for n in L), "rules_comp": sum(k(n) == "component" for n in R),
                "rules_only_planted": ";".join(n for n in R - L if k(n) == "planted"),
                "llm_only_planted": ";".join(n for n in L - R if k(n) == "planted"),
                "rules_only_comp": ";".join(n for n in R - L if k(n) == "component"),
                "llm_only_comp": ";".join(n for n in L - R if k(n) == "component"),
                "post_planted": active_post.get((fr, int(s["iteration_in_fold"])), (None,))[0],
                "post_comp": active_post.get((fr, int(s["iteration_in_fold"])), (None, None))[1],
                "post_n": active_post.get((fr, int(s["iteration_in_fold"])), (None, None, None))[2],
                "rules_only_rule": ";".join(f"{n}:{rule_of.get(n)}" for n in R - L),
            })
    print("done", rid, ds, "J" if J else "j", file=sys.stderr, flush=True)

pd.DataFrame(per_iter).to_csv("/work/shadow_per_iter2.csv", index=False)
pd.DataFrame(per_feat).to_csv("/work/final_features_J.csv", index=False)
print("wrote", len(per_iter), "shadow iterations,", len(per_feat), "final feature rows")
