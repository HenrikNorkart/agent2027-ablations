"""A18-T18: ``rp1_verdicts.py`` (floors, RP1-D1…D6 verdicts) on hand-computed toy tables, and the
HQ #315 scoring sensitivity (JSON-string object arguments) in ``replay_tool`` / ``rp1_score``."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.agent2027 import replay_tool as rt

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "agent2027" / "analysis"))
import rp1_verdicts as rv  # noqa: E402

pytestmark = [pytest.mark.unit]

LAD = ("Q0.8", "Q2", "Q4", "Q9", "Q122")
T_ = True
F_ = False


# =========================================================================== toy builder
def _rows(spec: dict, n: int = 10, recorded: dict = None, extra: list = ()) -> list[dict]:
    """``spec[(target, role)] = {score: list of n bools}``; prompt i of a role lies in cluster i % 5.

    Missing scores default to all True. ``recorded[role] = {score: list}`` adds ``recorded_O120``
    (raw is None there, as in ``rp1_score``)."""
    rows = []
    for (target, role), scores in spec.items():
        for i in range(n):
            r = {"target": target, "prompt_id": f"{role}{i}", "role": role,
                 "stage": "assessment" if (role == "T" and i < n // 2) else ("pruning" if role == "T" else "x"),
                 "fold_episode": f"run{i % 5}#f0", "common_set": True, "clipped": False,
                 "suspicious": False, "flags": "", "error_kind": "", "infra_error": False,
                 "context_error": False, "auto_unprune": False}
            for s in rv.SCORES:
                r[s] = scores.get(s, [T_] * n)[i]
            rows.append(r)
    for role, scores in (recorded or {}).items():
        for i in range(n):
            r = {"target": "recorded_O120", "prompt_id": f"{role}{i}", "role": role, "stage": "x",
                 "fold_episode": f"run{i % 5}#f0", "common_set": True, "rp1_valid_call_raw": None}
            for s in ("rp1_valid_call_rescued", "rp1_executes", "rp1_usable"):
                r[s] = scores.get(s, [T_] * n)[i]
            rows.append(r)
    return rows + list(extra)


def _k(k: int, n: int = 10) -> list[bool]:
    """First k prompts pass."""
    return [i < k for i in range(n)]


def _toy() -> list[dict]:
    """Primary toy (hand-computed below).

    usable vs O120 (= 1.0 everywhere, so the D1 bar is 0.8):
      E: Q0.8 0.0, Q2 0.0, Q4 0.7, Q9 0.8 (boundary passes), Q122 1.0 -> floor Q9 (position 4)
      S: Q0.8 0.0, else 1.0                                          -> floor Q2 (position 2)
      T: all 1.0                                                     -> floor <= Q0.8 (position 1)
      W: Q122 0.7, else 1.0                                          -> floor > Q122 (position 6)
    rescued: E at Q0.8 0.0 (transport-limited), else 1.0; W at O120 0.8 (D5(a) flag).
    """
    spec = {}
    e_use = {"O120": _k(10), "Q0.8": _k(0), "Q2": _k(0), "Q4": _k(7), "Q9": _k(8), "Q122": _k(10)}
    for t, u in e_use.items():
        spec[(t, "E")] = {"rp1_usable": u, "rp1_valid_call_rescued": _k(0) if t == "Q0.8" else _k(10)}
    for t in ("O120",) + LAD:
        spec[(t, "S")] = {"rp1_usable": _k(0) if t == "Q0.8" else _k(10)}
        spec[(t, "T")] = {}
        spec[(t, "W")] = {"rp1_usable": _k(7) if t == "Q122" else _k(10),
                          "rp1_valid_call_rescued": _k(8) if t == "O120" else _k(10)}
    # recorded_O120: E usable 0.8 (diff -0.2 -> D5(b) flags E); T usable 0.9 (diff exactly 10 pp: no flag)
    recorded = {"S": {}, "E": {"rp1_usable": _k(8)}, "T": {"rp1_usable": _k(9)},
                "W": {"rp1_valid_call_rescued": _k(8)}}  # W matches its O120 rate
    return _rows(spec, recorded=recorded)


def _floor(res, variant, unit, score):
    return next(f for f in res["variants"][variant]["floors"] if f["unit"] == unit and f["score"] == score)


def _rate(res, variant, unit, target, score):
    return next(r for r in res["variants"][variant]["rates"]
                if r["unit"] == unit and r["target"] == target and r["score"] == score)


# =========================================================================== statistics
def test_wilson_and_mcnemar_hand_values():
    lo, hi = rv.wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-4) and hi == pytest.approx(0.9433, abs=1e-4)
    assert rv.wilson(0, 0) == (None, None)
    assert rv.mcnemar_exact(0, 10) == pytest.approx(2 / 1024)
    assert rv.mcnemar_exact(3, 3) == 1.0
    assert rv.mcnemar_exact(0, 0) == 1.0


def test_floor_positions_and_labels():
    p = np.array([[0, 1, 0, 1, 1],    # Q4 fails -> Q9 (4)
                  [1, 1, 1, 1, 1],    # all pass -> <= Q0.8 (1)
                  [1, 1, 1, 1, 0],    # Q122 fails -> > Q122 (6)
                  [0, 0, 0, 0, 0],    # none passes -> > Q122 (6)
                  [1, np.nan, 1, 1, 1]], dtype=float)
    pos = rv.floor_positions(p)
    assert list(pos[:4]) == [4, 1, 6, 6] and np.isnan(pos[4])
    assert [rv.floor_label(x, LAD) for x in (4, 1, 6)] == ["Q9", "<= Q0.8", "> Q122"]
    # reduced ladder (ED2 / A04 drop): positions on the measured rungs only
    red = ("Q0.8", "Q4", "Q9")
    assert rv.floor_positions(np.array([0, 1, 1.0])) == 2 and rv.floor_label(2, red) == "Q4"
    assert rv.floor_label(4, red) == "> Q9"


def test_d1_pass_boundary_and_undefined():
    assert rv.d1_pass(0.8, 1.0) == 1.0  # >= 0.8 x O120
    assert rv.d1_pass(0.79, 1.0) == 0.0
    assert np.isnan(rv.d1_pass(np.nan, 1.0)) and np.isnan(rv.d1_pass(0.5, np.nan))


# =========================================================================== D2 order verdict
@pytest.mark.parametrize("fs,fe,ft,want", [
    (2, 4, 1, "Hoped"),      # E >= S >= T and E - T = 3 >= 2
    (2, 2, 2, "Null"),
    (3, 2, 1, "Opposite"),   # E < max(S, T)
    (1, 3, 2, "Partial"),    # S < T breaks the chain, E not below either
    (2, 3, 2, "Partial"),    # chain holds but E - T = 1 < 2
    (None, 3, 2, None),
])
def test_order_verdict(fs, fe, ft, want):
    assert rv.order_verdict(fs, fe, ft) == want


# =========================================================================== primary toy end to end
def test_toy_floors_rates_and_verdicts():
    rows = _toy()
    res = rv.run_all(rows, n_boot=200, seed=42)
    assert res["meta"]["ladder"] == list(LAD) and res["meta"]["n_clusters"] == 5
    assert _floor(res, "primary", "E", "rp1_usable")["floor"] == "Q9"
    assert _floor(res, "primary", "E", "rp1_usable")["position"] == 4
    assert _floor(res, "primary", "S", "rp1_usable")["floor"] == "Q2"
    assert _floor(res, "primary", "T", "rp1_usable")["floor"] == "<= Q0.8"
    assert _floor(res, "primary", "W", "rp1_usable")["floor"] == "> Q122"
    assert _floor(res, "primary", "E", "rp1_valid_call_rescued")["floor"] == "Q2"  # transport floor
    r = _rate(res, "primary", "E", "Q9", "rp1_usable")
    assert (r["num"], r["den"], r["rate"], r["ratio_to_ref"], r["d1_pass"]) == (8, 10, 0.8, 0.8, True)
    assert r["wilson_lo"] == pytest.approx(0.4902, abs=1e-4)
    assert r["ratio_lo"] is not None and r["ratio_lo"] <= 0.8 <= r["ratio_hi"]
    assert _rate(res, "primary", "E", "Q4", "rp1_usable")["d1_pass"] is False
    # T stages are units of their own (5 assessment + 5 pruning prompts)
    assert _rate(res, "primary", "T/assessment", "Q2", "rp1_usable")["den"] == 5
    v = res["variants"]["primary"]["verdicts"]
    assert v["D2_point"] == "Hoped"
    assert v["D2_floor_positions"] == {"S": 2, "E": 4, "T": 1}
    assert v["D4"] == "1B cell predicted to fail"  # floor(S) = Q2 > Q0.8
    # D3a at E's failing rungs: Q0.8 rescued fails; Q2/Q4 rescued passes, usable fails
    assert v["D3a"]["E"] == {"Q0.8": "transport-limited", "Q2": "code/planning-limited",
                             "Q4": "code/planning-limited", "Q9": "pass", "Q122": "pass"}
    assert v["D3a"]["S"]["Q0.8"] == "code/planning-limited"
    # D3b: E's rescued passes D1 at Q2 -> not bottlenecked
    assert v["D3b"]["Qwen"]["verdict"] == "not transport-bottlenecked"
    assert v["D3b"]["gpt-oss"]["verdict"].startswith("not evaluable")
    # D5(a): W's O120 self-replay rescued = 0.8 < 0.9
    assert v["D5a"]["flag"] is True and v["D5a"]["by_role"]["W"]["ok"] is False
    assert v["D5a"]["by_role"]["S"]["ok"] is True
    # D5(b): E usable recorded 0.8 vs 1.0 (20 pp) flags; T 0.9 vs 1.0 (exactly 10 pp) does not
    assert v["D5b"]["affected_roles"] == ["E"]
    assert v["D5b"]["by_role_score"]["T:rp1_usable"]["flag"] is False
    assert v["D2_without_affected"] == "not evaluable without role(s) E (D5(b))"
    # McNemar S usable Q0.8 vs Q2: 0 lower-only, 10 upper-only -> p = 2 / 2^10
    m = next(x for x in res["mcnemar"] if x["unit"] == "S" and x["score"] == "rp1_usable"
             and x["lower"] == "Q0.8")
    assert (m["n_pairs"], m["lower_only"], m["upper_only"]) == (10, 0, 10)
    assert m["p_exact"] == pytest.approx(2 / 1024)


def test_floor_stability_share_and_unstable_verdict():
    """Hand-made resamples: row 0 = the full sample; the others keep only cluster 0 (prompts 0, 5).
    On cluster 0 alone E at Q4 is 2/2 -> E's floor becomes Q4 (3) instead of Q9 (4)."""
    rows = _toy()
    clusters = sorted({r["fold_episode"] for r in rows})
    full, c0 = np.ones(5), np.array([1.0, 0, 0, 0, 0])
    lad = rv.ladder_of(rows)
    unstable = rv.evaluate(rows, "primary", clusters=clusters, ladder=lad,
                           weights=np.stack([full, c0, c0, c0]))
    f = next(x for x in unstable["floors"] if x["unit"] == "E" and x["score"] == "rp1_usable")
    assert f["stability_share"] == pytest.approx(0.25)
    assert unstable["verdicts"]["D2"] == "uninformative (unstable floors) [Hoped]"
    stable = rv.evaluate(rows, "primary", clusters=clusters, ladder=lad,
                         weights=np.stack([full, full, full, c0]))
    assert stable["verdicts"]["D2_stability"]["E"] == pytest.approx(0.75)
    assert stable["verdicts"]["D2"] == "Hoped"


def test_d4_pass_and_d2_null():
    spec = {}
    for t in ("O120",) + LAD:
        for role in ("S", "E", "T", "W"):
            spec[(t, role)] = {}
    res = rv.run_all(_rows(spec), n_boot=50)
    v = res["variants"]["primary"]["verdicts"]
    assert v["D2"] == "Null" and v["D4"] == "1B cell predicted to pass"


def test_d3b_transport_bottleneck_and_min_denominator():
    def spec_for(n_exec_den):
        spec = {("O120", "E"): {}}
        for t in LAD:
            ex = [i < 36 for i in range(40)]
            ex = [x if i < n_exec_den else None for i, x in enumerate(ex)]
            spec[(t, "E")] = {"rp1_valid_call_rescued": _k(20, 40), "rp1_executes": ex}
        return spec
    res = rv.run_all(_rows(spec_for(40), n=40), n_boot=0)
    d = res["variants"]["primary"]["verdicts"]["D3b"]["Qwen"]
    assert d["verdict"] == "transport-bottlenecked"
    assert d["rungs"]["Q0.8"] == {"rescued_pass": False, "executes_den": 40, "ref_executes_den": 40,
                                  "executes_pass": True}
    res = rv.run_all(_rows(spec_for(20), n=40), n_boot=0)  # executes den 20 < 30 at every rung
    assert res["variants"]["primary"]["verdicts"]["D3b"]["Qwen"]["verdict"] == "not evaluable"


def test_d6_kappa_substitute_and_transport_only():
    rows = _toy()
    # S: executes fails at Q2 (0.5) while rescued passes -> joint pass fails at Q2 -> floor Q4
    for r in rows:
        if r["role"] == "S" and r["target"] == "Q2":
            r["rp1_executes"] = int(r["prompt_id"][1:]) < 5
    res = rv.run_all(rows, n_boot=0, kappa=0.5)
    v = res["variants"]["primary"]["verdicts"]
    assert v["d6_kappa_substitute_S"] is True
    assert _floor(res, "primary", "S", "S_joint_rescued_executes")["floor"] == "Q4"
    assert v["D2_floor_positions"]["S"] == 3
    assert rv.run_all(rows, n_boot=0, kappa=0.7)["variants"]["primary"]["verdicts"]["D2_floor_positions"]["S"] == 2
    for r in rows:
        r["rp1_usable"] = None if r["target"] != "recorded_O120" else r["rp1_usable"]
        r["rp1_executes"] = None
    v = rv.run_all(rows, n_boot=0)["variants"]["primary"]["verdicts"]
    assert v["d6_transport_only"] is True and v["D2"].startswith("not evaluated (RP1-D6")


def test_sensitivity_populations():
    base = dict(role="S", stage="x", fold_episode="run0#f0", flags="", suspicious=False, clipped=False,
                infra_error=False, error_kind="", rp1_valid_call_raw=False, rp1_executes=None)
    extra = [  # prompt S10 hit a context error at Q0.8, so it is outside the common set everywhere
        dict(base, target="Q0.8", prompt_id="S10", common_set=False, context_error=True,
             error_kind="context", rp1_valid_call_rescued=False, rp1_usable=False)] + [
        dict(base, target=t, prompt_id="S10", common_set=False, context_error=False,
             rp1_valid_call_rescued=True, rp1_usable=True) for t in ("O120",) + LAD[1:]]
    rows = _toy() + extra
    for r in rows:  # one clipped and one suspicious turn at Q2 (S)
        if r["target"] == "Q2" and r["prompt_id"] == "S0":
            r["clipped"] = True
        if r["target"] == "Q2" and r["prompt_id"] == "S1":
            r["flags"] = "reconstruction_partial"
    res = rv.run_all(rows, n_boot=0)
    assert _rate(res, "primary", "S", "Q0.8", "rp1_usable")["den"] == 10
    assert _rate(res, "no_common_set", "S", "Q0.8", "rp1_usable")["den"] == 11  # own context error = invalid
    assert _rate(res, "no_common_set", "S", "Q2", "rp1_usable")["num"] == 11
    assert _rate(res, "drop_clipped", "S", "Q2", "rp1_usable")["den"] == 9
    assert _rate(res, "drop_suspicious_partial", "S", "Q2", "rp1_usable")["den"] == 9
    assert _rate(res, "drop_clipped", "S", "Q4", "rp1_usable")["den"] == 10


def test_cli_writes_side_by_side_with_hq315(tmp_path):
    rows = _toy()
    rows_315 = [dict(r) for r in rows]
    for r in rows_315:  # under #315 E's Q2 calls lose validity (decoded args miss a required key)
        if r["role"] == "E" and r["target"] == "Q2":
            r["rp1_valid_call_rescued"] = False
    cols = sorted({k for r in rows for k in r})

    def dump(path, rr):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows([{k: ("" if r.get(k) is None else r.get(k)) for k in cols} for r in rr])
    dump(tmp_path / "p.csv", rows)
    dump(tmp_path / "q.csv", rows_315)
    assert rv.main(["--scored", str(tmp_path / "p.csv"), "--scored-315", str(tmp_path / "q.csv"),
                    "--out", str(tmp_path / "out"), "--n-boot", "100"]) == 0
    res = json.loads((tmp_path / "out" / "rp1_floors.json").read_text(encoding="utf-8"))
    assert res["variants"]["primary"]["verdicts"]["D3a"]["E"]["Q2"] == "code/planning-limited"
    assert res["variants"]["hq315"]["verdicts"]["D3a"]["E"]["Q2"] == "transport-limited"
    assert res["variants"]["hq315"]["verdicts"]["D2_point"] == "Hoped"
    with open(tmp_path / "out" / "rp1_verdicts.csv", encoding="utf-8") as fh:
        sbs = list(csv.DictReader(fh))
    assert list(sbs[0])[:3] == ["item", "primary", "hq315"]
    fl = next(r for r in sbs if r["item"] == "floor E rp1_valid_call_rescued")
    assert fl["primary"].startswith("Q2") and fl["hq315"].startswith("Q4")
    for name in ("rp1_floor_rates.csv", "rp1_floors.csv", "rp1_mcnemar.csv", "rp1_jaccard.csv"):
        assert (tmp_path / "out" / name).exists()


# =========================================================================== HQ #315 (replay_tool)
E_REF_TOOLS = [{"type": "function", "function": {"name": "generic_pandas_tool", "parameters": {
    "$defs": {"Cmd": {"type": "object", "required": ["command", "explanation"],
                      "properties": {"command": {"type": "string"}, "explanation": {"type": "string"}}}},
    "type": "object", "required": ["args"], "properties": {"args": {"$ref": "#/$defs/Cmd"}}}}}]
SCHEMA = E_REF_TOOLS[0]["function"]["parameters"]


def _stringified(inner: dict) -> str:
    return json.dumps({"args": json.dumps(inner)})


def test_missing_required_keys_json_string_default_and_decoded():
    args = {"args": json.dumps({"command": "df.shape"})}  # stringified object, 'explanation' missing
    assert rt.missing_required_keys(args, SCHEMA) == []  # default: nested check skipped (as before)
    assert rt.missing_required_keys(args, SCHEMA, decode_json_strings=True) == ["args.explanation"]
    ok = {"args": json.dumps({"command": "df.shape", "explanation": "x"})}
    assert rt.missing_required_keys(ok, SCHEMA, decode_json_strings=True) == []
    # not valid JSON, or a JSON list: left as is (no decode, no extra missing key)
    assert rt.missing_required_keys({"args": "{not json"}, SCHEMA, decode_json_strings=True) == []
    assert rt.missing_required_keys({"args": "[1, 2]"}, SCHEMA, decode_json_strings=True) == []
    # a string-typed property holding JSON text is never decoded
    s_schema = {"type": "object", "required": ["command"],
                "properties": {"command": {"type": "string"}}}
    assert rt.missing_required_keys({"command": '{"a": 1}'}, s_schema, decode_json_strings=True) == []
    # dict args: identical either way
    d = {"args": {"command": "x"}}
    assert rt.missing_required_keys(d, SCHEMA) == rt.missing_required_keys(d, SCHEMA, decode_json_strings=True)


def test_classify_move_hq315_both_settings():
    calls = [{"name": "generic_pandas_tool", "arguments": _stringified({"command": "df.shape"})}]
    default = rt.classify_move("tool_calls", calls, None, E_REF_TOOLS)
    assert default["valid"] is True and default["reason"] == "ok"
    assert default == rt.classify_move("tool_calls", calls, None, E_REF_TOOLS, decode_json_strings=False)
    dec = rt.classify_move("tool_calls", calls, None, E_REF_TOOLS, decode_json_strings=True)
    assert dec["valid"] is False and dec["reason"] == "missing_required:args.explanation"
    full = [{"name": "generic_pandas_tool",
             "arguments": _stringified({"command": "df.shape", "explanation": "e"})}]
    assert rt.classify_move("tool_calls", full, None, E_REF_TOOLS, decode_json_strings=True)["valid"] is True


def test_rp1_score_hq315_passthrough():
    import rp1_score as rp  # noqa: PLC0415

    call = {"name": "append_new_attribute",
            "arguments": _stringified({"name": "f", "command": "result = 1", "argumentation": "a"})}
    assert rp.call_args(call) == {"args": json.dumps({"name": "f", "command": "result = 1",
                                                      "argumentation": "a"})}
    assert rp.call_args(call, decode_json_strings=True) == {"name": "f", "command": "result = 1",
                                                            "argumentation": "a"}
    rec = {"prompt_id": "p1", "role": "E", "stage": "append", "stratum": "E_other", "recorded_move": "action",
           "request": {"tools": E_REF_TOOLS}, "state": {}, "provenance": {}}
    msg = [{"name": "generic_pandas_tool", "arguments": _stringified({"command": "df.shape"})}]
    move = {"final": {"finish_reason": "tool_calls", "content": None, "tool_calls": msg},
            "raw": {"finish_reason": "tool_calls", "content": None, "tool_calls": msg},
            "error_kind": None, "clipped": False}
    a = rp.score_move(rec, move, None, execute=False)
    b = rp.score_move(rec, move, None, execute=False, decode_json_strings=False)
    assert a == b and a["rp1_valid_call_rescued"] is True and a["rp1_valid_call_raw"] is True
    c = rp.score_move(rec, move, None, execute=False, decode_json_strings=True)
    assert c["rp1_valid_call_rescued"] is False and c["rp1_valid_call_raw"] is False
    assert c["valid_reason"] == "missing_required:args.explanation"
