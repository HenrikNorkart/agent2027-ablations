"""RQ3 (capacity) - EXPLORATORY. Per-role, per-iteration profile of the live Qwen arms (A17 4xx, A19 4xx) from
iteration_log.jsonl and agent_toolcall_errors.jsonl copied read-only from
CAIR-gpu14:~/rogueone-runs-agent2027/<session>/<row>/tasks/<type>/<task>/output/ into cair_live/.
All attempt directories (<row>.attempt1.<ts> and <row>) are included; metrics are per logged iteration, so a run
that hit the 12 h cap contributes the iterations it finished. Output: derived/t11_live_per_iteration_roles.csv,
derived/t11b_live_toolerr_by_agent.csv.  Run: python -X utf8 live_iteration_profile.py
"""
import json
from pathlib import Path

import pandas as pd

CAP = Path(__file__).resolve().parent
LIVE = CAP / "cair_live"
OUT = CAP / "derived"
REPO = Path(__file__).resolve().parents[5]
rows = pd.read_csv(REPO / "docs/agent2027/results/final_20261112/rows_final.csv")
arm_of = dict(zip(rows.row_id, rows.arm_id))
ROLES = ["scientist", "extractor", "tester"]

it = []
for f in LIVE.glob("*/*/tasks/*/*/output/iteration_log.jsonl"):
    rdir = f.relative_to(LIVE).parts[1]
    row = rdir.split(".")[0]
    arm = arm_of.get(row)
    task = f.relative_to(LIVE).parts[4]
    if arm is None or not task.startswith("synp"):
        continue
    for line in open(f, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        tk, w = r.get("tokens") or {}, r.get("wall_s") or {}
        d = {"arm": arm, "row": row, "dir": rdir, "task": task, "status": r.get("status"),
             "active_features": r.get("active_features"), "pruned_features": r.get("pruned_features")}
        for k in ROLES:
            d[f"calls_{k}"] = (tk.get(k) or {}).get("calls")
            d[f"prompt_{k}"] = (tk.get(k) or {}).get("prompt")
            d[f"compl_{k}"] = (tk.get(k) or {}).get("compl")
            d[f"wall_{k}"] = w.get(k)
        it.append(d)
I = pd.DataFrame(it)
I["prompt_per_call_extractor"] = I.prompt_extractor / I.calls_extractor
agg = {"runs": ("row", "nunique"), "iterations": ("row", "size")}
for c in ([f"{m}_{k}" for m in ("calls", "prompt", "compl", "wall") for k in ROLES]
          + ["active_features", "pruned_features", "prompt_per_call_extractor"]):
    agg[c] = (c, "median")
t = I.groupby("arm").agg(**agg)
ws = I[[f"wall_{k}" for k in ROLES]].fillna(0)
I["wall_share_extractor"] = ws.wall_extractor / ws.sum(axis=1)
t["wall_share_extractor_median"] = I.groupby("arm").wall_share_extractor.median()
t["iters_with_zero_pruned_share"] = I.assign(z=I.pruned_features.fillna(0) == 0).groupby("arm").z.mean()
t.to_csv(OUT / "t11_live_per_iteration_roles.csv")
print(t.T.round(2).to_string())

er = []
for f in LIVE.glob("*/*/tasks/*/*/output/agent_toolcall_errors.jsonl"):
    rdir = f.relative_to(LIVE).parts[1]
    row = rdir.split(".")[0]
    arm = arm_of.get(row)
    task = f.relative_to(LIVE).parts[4]
    if arm is None or not task.startswith("synp"):
        continue
    for line in open(f, encoding="utf-8"):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        er.append({"arm": arm, "agent": e.get("agent_name"), "tool": e.get("tool_name"),
                   "exc": e.get("exception_type")})
E = pd.DataFrame(er)
t2 = E.groupby(["arm", "agent"]).size().unstack(fill_value=0)
t2["iterations_logged"] = I.groupby("arm").size()
for c in [c for c in t2.columns if c.endswith("Agent")]:
    t2[c + "_per_iter"] = t2[c] / t2.iterations_logged
t2.to_csv(OUT / "t11b_live_toolerr_by_agent.csv")
print(t2.round(2).to_string())
top = E.groupby(["arm", "agent", "tool", "exc"]).size().rename("n").reset_index().sort_values("n", ascending=False)
top.to_csv(OUT / "t11c_live_toolerr_top.csv", index=False)
print(top.head(30).to_string())
