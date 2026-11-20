"""RQ3 (capacity) - EXPLORATORY, post hoc, outside the pre-registered Holm family.

Inputs (read-only):
  REPO/docs/agent2027/results/final_20261102/{rows_final.csv, episodes_final.csv, final_readout.json}
  REPO/docs/agent2027/results/G2_token_shares.csv
  CAP/cair_a18/final_1102/a18_verdicts/{rp1_floor_rates.csv, rp1_floors.csv, rp1_verdicts.csv, rp1_jaccard.csv}
      (copied read-only from CAIR-gpu14:~/ro27_analysis/final_1102/a18_verdicts/, written 2026-11-02)
  CAP/cair_a18/funnel_episodes.csv (CAIR ~/ro27_analysis/final_1102/funnel_episodes.csv)
  SCR/mast_data/<session>/<row>/tasks/<type>/<task>/output/{llm_traces.jsonl.gz, agent_toolcall_errors.jsonl,
      iteration_log.jsonl}  (57 synthetic runs: 30 A05 Full, 27 A16 set_phased)
Outputs: CAP/derived/*.csv, CAP/derived/numbers.json
Run: python -X utf8 capacity_analysis.py
"""
from __future__ import annotations

import gzip
import os
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[5]
RES = REPO / "docs/agent2027/results"
FIN = RES / "final_20261112"  # superset of the 11-02 read-out used originally
CAP = Path(__file__).resolve().parent
DATA = Path(os.environ.get("MAST_DATA", CAP.parent / "mast_data"))  # raw traces, not released
OUT = CAP / "derived"
OUT.mkdir(exist_ok=True)
A18 = CAP / "cair_a18/final_1102/a18_verdicts"
NUM: dict = {}

ROLES = ["scientist", "extractor", "tester"]
# Active / total parameters per model (active from scripts/agent2027/analysis/pareto.py:55 ACTIVE_PARAMS;
# total from the model cards: gpt-oss-120b 117B, Qwen3.5-122B-A10B 122B, dense Qwen rungs = active).
ACTIVE = {"Q0.8": 0.8e9, "Q2": 2e9, "Q4": 4e9, "Q9": 9e9, "Q122": 10e9, "O120": 5.1e9, "code": 0.0}
TOTAL = {"Q0.8": 0.8e9, "Q2": 2e9, "Q4": 4e9, "Q9": 9e9, "Q122": 122e9, "O120": 117e9, "code": 0.0}


def q(s, p):
    return float(s.quantile(p))


# ------------------------------------------------------------------ 1. rows_final: role tokens/calls per arm
rows = pd.read_csv(FIN / "rows_final.csv")
rows["n"] = rows.row_id.str.split("-").str[1].astype(int)
rows["family"] = rows.dataset_id.str.startswith("synp").map({True: "synthetic", False: "real"})
tc = [f"tok_compl_{r}" for r in ROLES]
cc = [f"llm_calls_{r}" for r in ROLES]

full = rows[(rows.ablation_id == "A05") & (rows.arm_id == "full") & (rows.final_status == "OK")].copy()
for r in ROLES:
    full[f"s_{r}"] = full[f"tok_compl_{r}"] / full.tok_compl_total
    full[f"c_{r}"] = full[f"llm_calls_{r}"] / full[cc].sum(axis=1)
    full[f"tpc_{r}"] = full[f"tok_compl_{r}"] / full[f"llm_calls_{r}"]
full["s_tester_assess"] = full.tok_compl_tester_assess / full.tok_compl_total
full["s_tester_prune"] = full.tok_compl_tester_prune / full.tok_compl_total
recs = []
for fam, g in list(full.groupby("family")) + [("all", full)]:
    d = {"family": fam, "n_runs": len(g)}
    for r in ROLES:
        d[f"share_compl_{r}_median"] = g[f"s_{r}"].median()
        d[f"share_compl_{r}_q25"] = q(g[f"s_{r}"], .25)
        d[f"share_compl_{r}_q75"] = q(g[f"s_{r}"], .75)
        d[f"share_compl_{r}_pooled"] = g[f"tok_compl_{r}"].sum() / g.tok_compl_total.sum()
        d[f"share_calls_{r}_median"] = g[f"c_{r}"].median()
        d[f"calls_{r}_median"] = g[f"llm_calls_{r}"].median()
        d[f"compl_per_call_{r}_median"] = g[f"tpc_{r}"].median()
    d["share_compl_tester_assess_median"] = g.s_tester_assess.median()
    d["share_compl_tester_prune_median"] = g.s_tester_prune.median()
    d["tok_compl_total_median"] = g.tok_compl_total.median()
    d["tok_prompt_total_median"] = g.tok_prompt_total.median()
    d["max_prompt_tokens_median"] = g.max_prompt_tokens.median()
    recs.append(d)
t1 = pd.DataFrame(recs)
t1.to_csv(OUT / "t1_full_role_shares_rows_final.csv", index=False)
NUM["t1_full_role_shares"] = t1.to_dict(orient="records")

# Live arms: per-arm role profile (Full O120, single agent, A17 ladder 4xx/5xx relaunch rows, A19, A20, A12 code-Tester)
def arm_sel(r):
    a, arm, n = r.ablation_id, r.arm_id, r.n
    if a == "A05" and arm == "full":
        return "A05 full (O120)"
    if a == "A16" and arm == "set_phased":
        return "A16 set_phased (O120, single agent)"
    # rows of record: A17/A19 relaunch rows 4xx (Amendments 17, 18); 5xx inf-guard re-runs are sensitivity rows
    if a == "A17" and 400 <= n < 500 and arm.startswith("f2_") and not arm.endswith("cap12"):
        return f"A17 {arm} (all roles)"
    if a == "A19" and 400 <= n < 500:
        return f"A19 {arm}"
    if a == "A20" and 400 <= n < 500:
        return "A20 fx_pajM_Q4 (no S, code T, E on Q4)"
    if a == "A12" and arm in ("fx_PajM", "fx_Pajm"):
        return "A12 fx_Paj* (S+E LLM on O120, T = code)"
    if a == "A12" and arm in ("fx_PAJM", "fx_PAJm"):
        return "A12 fx_PAJ* (all LLM, O120)"
    return None


rows["arm_label"] = rows.apply(arm_sel, axis=1)
live = rows[rows.arm_label.notna() & rows.dataset_id.str.startswith("synp")].copy()
live["nonext_tool_err"] = live.n_tool_errors - live.err_role_extractor
agg = {"runs": ("row_id", "size"), "ok": ("final_status", lambda s: int((s == "OK").sum())),
       "cap_hit": ("final_status", lambda s: int((s == "CAP_HIT").sum())),
       "failed": ("final_status", lambda s: int((s == "FAILED").sum()))}
for c in tc + cc + ["tok_prompt_total", "max_prompt_tokens", "n_tool_errors", "err_role_extractor", "nonext_tool_err",
                     "n_accepted", "err_per_accepted", "fallback_share", "loop_guard", "iterations_total", "flop_proxy"]:
    agg[c + "_med"] = (c, "median")
t2 = live.groupby("arm_label").agg(**agg)
ep = pd.read_csv(FIN / "episodes_final.csv")
ep["n"] = ep.row_id.str.split("-").str[1].astype(int)
ep = ep.merge(rows[["row_id", "arm_label"]], on="row_id", how="left")
ep = ep[ep.arm_label.notna() & ep.dataset_id.str.startswith("synp")]
rec = ep.groupby("arm_label").agg(ep_n=("rec_sel", "size"), rec_sel=("rec_sel", "sum"), rec_any=("rec_any", "sum"))
t2 = t2.join(rec)
t2["iter_per_run_med"] = t2["iterations_total_med"]
t2.to_csv(OUT / "t2_live_arm_role_profile.csv")
NUM["t2_live_arm_role_profile"] = json.loads(t2.reset_index().to_json(orient="records"))

# Funnel for live arms: where do generated features die? (fold-episode sums)
fun = pd.read_csv(CAP / "cair_a18/funnel_episodes.csv")
fun = fun.merge(rows[["row_id", "arm_label"]], on="row_id", how="left")
fun = fun[fun.arm_label.notna()]
fcols = ["fun_proposed", "fun_executes", "fun_valid", "fun_survives_prune", "fun_selected", "rej_exec_error",
         "rej_pruned", "rej_dup_name"]
t2f = fun.groupby("arm_label")[fcols].sum()
t2f["exec_rate"] = t2f.fun_executes / t2f.fun_proposed
t2f["valid_rate"] = t2f.fun_valid / t2f.fun_proposed
t2f["prune_rate_of_valid"] = t2f.rej_pruned / t2f.fun_valid
t2f["episodes"] = fun.groupby("arm_label").size()
t2f.to_csv(OUT / "t2b_live_arm_funnel.csv")
NUM["t2b_live_arm_funnel"] = json.loads(t2f.reset_index().to_json(orient="records"))

# ------------------------------------------------------------------ 2. traces: per role/stage prompt sizes, wall time
def load_runs():
    for f in sorted(DATA.glob("*/*/tasks/*/*/output/llm_traces.jsonl.gz")):
        parts = f.relative_to(DATA).parts
        yield parts[0], parts[1], parts[4], f.parent


calls = []
toolerr = []
itwall = []
toolnames = Counter()
for sess, row, task, od in load_runs():
    arm = "full" if row.startswith("A05") else "single"
    with gzip.open(od / "llm_traces.jsonl.gz", "rt", encoding="utf-8") as h:
        for line in h:
            r = json.loads(line)
            u = r.get("usage") or {}
            role, stage = r.get("role"), r.get("stage")
            unit = {"focus": "S", "append": "E", "assessment": "T/assess", "pruning": "T/prune"}.get(stage, None)
            if stage == "search":
                unit = "W(" + str(r.get("caller")) + ")"
            if unit is None:
                unit = f"{role}/{stage}"
            m = r.get("message") or {}
            n_tc = 0
            for t in (m.get("tool_calls") or []):
                if isinstance(t, dict):
                    toolnames[(arm, unit, (t.get("function") or {}).get("name"))] += 1
                    n_tc += 1
            calls.append({"arm": arm, "row": row, "task": task, "role": role, "unit": unit,
                          "prompt": u.get("prompt_tokens"), "compl": u.get("completion_tokens"),
                          "dur": r.get("duration_s"), "n_toolcalls": n_tc, "rerolls": r.get("rerolls") or 0,
                          "finish": r.get("finish_reason")})
    fe = od / "agent_toolcall_errors.jsonl"
    if fe.exists():
        with open(fe, encoding="utf-8") as h:
            for line in h:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = str(e.get("exception_message") or e.get("message") or "")
                toolerr.append({"arm": arm, "row": row, "task": task, "agent": e.get("agent_name"),
                                "tool": e.get("tool_name"), "exc": e.get("exception_type"), "msg": msg[:120]})
    fi = od / "iteration_log.jsonl"
    if fi.exists():
        with open(fi, encoding="utf-8") as h:
            for line in h:
                it = json.loads(line)
                w = it.get("wall_s") or {}
                tk = it.get("tokens") or {}
                itwall.append({"arm": arm, "row": row, "task": task, "index": it.get("index"),
                               **{f"wall_{k}": w.get(k) for k in ROLES},
                               **{f"prompt_{k}": (tk.get(k) or {}).get("prompt") for k in ROLES},
                               **{f"compl_{k}": (tk.get(k) or {}).get("compl") for k in ROLES},
                               **{f"calls_{k}": (tk.get(k) or {}).get("calls") for k in ROLES}})

C = pd.DataFrame(calls)
NUM["trace_runs"] = C.groupby("arm").row.nunique().to_dict()
NUM["trace_calls"] = C.groupby("arm").size().to_dict()

# per-unit call-level prompt sizes and run-level shares
g = C.groupby(["arm", "unit"])
t3 = g.agg(calls=("prompt", "size"), prompt_median=("prompt", "median"),
           prompt_p90=("prompt", lambda s: q(s, .9)), prompt_max=("prompt", "max"),
           compl_median=("compl", "median"), compl_sum=("compl", "sum"), prompt_sum=("prompt", "sum"),
           dur_sum_s=("dur", "sum"), dur_median_s=("dur", "median"))
tot = C.groupby("arm").agg(P=("prompt", "sum"), Cc=("compl", "sum"), D=("dur", "sum"))
t3 = t3.join(tot, on="arm")
t3["share_prompt"] = t3.prompt_sum / t3.P
t3["share_compl"] = t3.compl_sum / t3.Cc
t3["share_llm_time"] = t3.dur_sum_s / t3.D
t3 = t3.drop(columns=["P", "Cc", "D"])
t3.to_csv(OUT / "t3_trace_unit_stats.csv")
NUM["t3_trace_unit_stats"] = json.loads(t3.reset_index().to_json(orient="records"))

# per-run max / median prompt per call, Full vs single
pr = C.groupby(["arm", "row"]).prompt.agg(["max", "median", "sum"]).reset_index()
t4 = pr.groupby("arm").agg(runs=("row", "size"), run_max_prompt_median=("max", "median"),
                           run_max_prompt_max=("max", "max"), run_median_prompt_median=("median", "median"),
                           run_prompt_sum_median=("sum", "median"))
t4.to_csv(OUT / "t4_prompt_full_vs_single.csv")
NUM["t4_prompt_full_vs_single"] = json.loads(t4.reset_index().to_json(orient="records"))
# by role in single vs full (role field): median prompt per call
t4b = C.groupby(["arm", "role"]).prompt.agg(["size", "median", lambda s: q(s, .9), "max"])
t4b.columns = ["calls", "median", "p90", "max"]
t4b.to_csv(OUT / "t4b_prompt_by_role_full_vs_single.csv")
NUM["t4b_prompt_by_role"] = json.loads(t4b.reset_index().to_json(orient="records"))

# wall-time per role from iteration_log (includes tool/XGBoost time, not only LLM)
W = pd.DataFrame(itwall)
wr = W.groupby(["arm", "row"])[[f"wall_{k}" for k in ROLES]].sum()
wr = wr.div(wr.sum(axis=1), axis=0)
t5 = wr.groupby("arm").median()
t5.columns = [c.replace("wall_", "share_wall_") for c in t5.columns]
t5.to_csv(OUT / "t5_wall_share_iteration_log.csv")
NUM["t5_wall_share"] = json.loads(t5.reset_index().to_json(orient="records"))

# tool errors per agent
E = pd.DataFrame(toolerr)
t6 = E.groupby(["arm", "agent"]).size().unstack(fill_value=0)
t6["runs"] = C.groupby("arm").row.nunique()
t6.to_csv(OUT / "t6_toolerr_by_agent.csv")
E["kind"] = E.msg.str.replace(r"[0-9_A-Za-z]+_[0-9A-Za-z_]+", "<name>", regex=True).str[:60]
t6b = E.groupby(["arm", "agent", "tool", "exc"]).size().rename("n").reset_index().sort_values("n", ascending=False)
t6b.to_csv(OUT / "t6b_toolerr_by_agent_tool_exc.csv", index=False)
NUM["t6_toolerr_by_agent"] = json.loads(t6.reset_index().to_json(orient="records"))

# tool-error rate per tool call, by agent (Full vs single)
tcall = C.assign(agent=C.role.map({"scientist": "ScientistAgent", "extractor": "ExtractorAgent", "tester": "TesterAgent"}))
tcall = tcall[~tcall.unit.str.startswith("W(")].groupby(["arm", "agent"]).agg(llm_calls=("n_toolcalls", "size"),
                                                                              tool_calls=("n_toolcalls", "sum"))
e_nonrunner = E[E.tool != "runner"].groupby(["arm", "agent"]).size().rename("tool_errors")
e_runner = E[E.tool == "runner"].groupby(["arm", "agent"]).size().rename("runner_errors")
t6c = tcall.join(e_nonrunner).join(e_runner).fillna(0)
t6c["tool_err_per_tool_call"] = t6c.tool_errors / t6c.tool_calls
t6c.to_csv(OUT / "t6c_toolerr_rate_by_agent.csv")
NUM["t6c_toolerr_rate_by_agent"] = json.loads(t6c.reset_index().to_json(orient="records"))

# tool usage per unit (what kind of work)
tn = pd.Series(toolnames).rename("n").reset_index()
tn.columns = ["arm", "unit", "tool", "n"]
tn.sort_values(["arm", "unit", "n"], ascending=[True, True, False]).to_csv(OUT / "t7_tool_usage_by_unit.csv",
                                                                          index=False)

# ------------------------------------------------------------------ 3. replay floors (A18), compact table
fr = pd.read_csv(A18 / "rp1_floor_rates.csv")
keep = fr[fr.variant.isin(["primary", "hq315"]) & fr.unit.isin(["S", "E", "T", "T/assessment", "T/pruning", "W"])]
keep = keep.assign(cell=keep.num.astype(int).astype(str) + "/" + keep.den.astype(int).astype(str))
t8 = keep.pivot_table(index=["variant", "unit", "score"], columns="target", values="rate", aggfunc="first")
t8 = t8[["recorded_O120", "O120", "Q0.8", "Q2", "Q4", "Q9", "Q122"]]
t8.to_csv(OUT / "t8_replay_rates.csv")
fl = pd.read_csv(A18 / "rp1_floors.csv")
fl = fl[fl.variant.isin(["primary", "hq315"]) & fl.unit.isin(["S", "E", "T", "W", "T/assessment", "T/pruning"])]
fl.to_csv(OUT / "t8b_replay_floors.csv", index=False)

# ------------------------------------------------------------------ 4. cost projection
# Role shares from Full O120 synthetic traces (prompt + completion per unit). W follows its caller's model
# in live runs (ROGUE_ONE_WEBSEARCH_ROUTING=caller); A18 W floor is <= Q0.8, but we keep W with its caller.
F = C[C.arm == "full"].copy()
F["role_unit"] = F.unit.map(lambda u: "S" if u in ("S", "W(scientist)") else "E" if u in ("E", "W(extractor)")
                            else "T" if u.startswith("T") or u == "W(tester)" else "other")
sh = F.groupby("role_unit")[["prompt", "compl"]].sum()
shp = (sh / sh.sum()).to_dict()
NUM["cost_role_shares_full_synthetic_traces"] = {"prompt": shp["prompt"], "compl": shp["compl"],
                                                 "abs_sum": sh.to_dict()}
# also rows_final completion shares (pooled over all OK Full rows, syn+real)
rf = {r[0].upper(): float(full[f"tok_compl_{r}"].sum() / full.tok_compl_total.sum()) for r in ROLES}

# (scenario -> (assignment, baseline)) ; baselines: everything on one model
BASE = {"O120": {"S": "O120", "E": "O120", "T": "O120"}, "Q122": {"S": "Q122", "E": "Q122", "T": "Q122"}}
scen = {
    "A floors primary (S=Q4, E=O120, T=Q0.8)": ({"S": "Q4", "E": "O120", "T": "Q0.8"}, "O120"),
    "B floors primary (S=Q4, E=Q122, T=Q0.8)": ({"S": "Q4", "E": "Q122", "T": "Q0.8"}, "Q122"),
    "C floors hq315 decoded (S=Q4, E=Q4, T=Q0.8)": ({"S": "Q4", "E": "Q4", "T": "Q0.8"}, "Q122"),
    "D only Tester small (S=O120, E=O120, T=Q0.8)": ({"S": "O120", "E": "O120", "T": "Q0.8"}, "O120"),
    "E Tester as code (S=O120, E=O120, T=code)": ({"S": "O120", "E": "O120", "T": "code"}, "O120"),
    "F Tester as code + S=Q4 (E=O120)": ({"S": "Q4", "E": "O120", "T": "code"}, "O120"),
    "G live mix_esmall (S=Q122, E=Q4, T=Q122)": ({"S": "Q122", "E": "Q4", "T": "Q122"}, "Q122"),
    "H live mix_1b (S=Q0.8, E=Q122, T=Q0.8)": ({"S": "Q0.8", "E": "Q122", "T": "Q0.8"}, "Q122"),
}


def cost(assign, params, tokens):
    return sum(params[assign[r]] * tokens[r] for r in ("S", "E", "T"))


res = []
bases = (("prompt+compl (traces, Full syn, n=30)", {r: float(sh.prompt[r] + sh.compl[r]) for r in "SET"}),
         ("compl only (traces, Full syn, n=30)", {r: float(sh.compl[r]) for r in "SET"}),
         ("compl only (rows_final, Full OK syn+real)", rf))
for basis_name, tok in bases:
    for pname, params in (("active", ACTIVE), ("total", TOTAL)):
        for sname, (a, bname) in scen.items():
            b = cost(BASE[bname], params, tok)
            c = cost(a, params, tok)
            moved = sum(tok[r] for r in "SET" if a[r] != BASE[bname][r]) / sum(tok.values())
            res.append({"token_basis": basis_name, "param_proxy": pname, "baseline": "all " + bname,
                        "scenario": sname, "share_tokens_off_big_model": moved,
                        "cost_ratio_vs_baseline": c / b, "reduction": 1 - c / b})
t9 = pd.DataFrame(res)
t9.to_csv(OUT / "t9_cost_projection.csv", index=False)
NUM["t9_cost_projection"] = t9.to_dict(orient="records")

# measured flop_proxy (E3 instrumentation, rows_final col flop_proxy) for live arms, OK rows only and all rows
mfp = live.groupby("arm_label").agg(flop_med_all=("flop_proxy", "median"),
                                    tokc_med_all=("tok_compl_total", "median"))
mfp_ok = live[live.final_status == "OK"].groupby("arm_label").agg(flop_med_ok=("flop_proxy", "median"),
                                                                    tokc_med_ok=("tok_compl_total", "median"),
                                                                    n_ok=("row_id", "size"))
t10 = mfp.join(mfp_ok)
t10.to_csv(OUT / "t10_measured_flop_proxy_live.csv")
NUM["t10_measured_flop_proxy_live"] = json.loads(t10.reset_index().to_json(orient="records"))

# Evidence-gap check: any live arm with a small Tester while S and E stay big?
NUM["arms_in_rows_final"] = sorted(rows.ablation_id.str.cat(rows.arm_id, sep=":").unique().tolist())

(OUT / "numbers.json").write_text(json.dumps(NUM, indent=1, default=float), encoding="utf-8")
print("ok", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in NUM.items()})
