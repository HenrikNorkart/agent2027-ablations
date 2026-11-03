"""Unit tests for the pre-registered A02 R rule (a02_readout.py, A02-T12)."""

import re

import pandas as pd
import pytest

from scripts.agent2027.analysis import a02_readout as ar

pytestmark = pytest.mark.unit

TASKS = list(ar.TASKS)


def _task_arm(r10, r3, r1=None):
    rows = []
    for arm, vals in (("R10", r10), ("R3", r3), ("R1", r1 or [0] * 8)):
        rows += [{"arm": arm, "dataset_id": t, "rec_sel": v} for t, v in zip(TASKS, vals)]
    return pd.DataFrame(rows)


def _rows(succ10=8, succ3=8, u10=1.0, u3=0.7, succ1=8, u1=0.1):
    out = []
    for arm, s, u in (("R10", succ10, u10), ("R3", succ3, u3), ("R1", succ1, u1)):
        for i, t in enumerate(TASKS):
            out.append({"arm": arm, "dataset_id": t, "success": int(i < s), "u_run": u})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ criterion 1
def test_c1_boundary_loss_of_4_passes_5_fails():
    assert ar.criterion1(20, 16)["pass"] is True
    assert ar.criterion1(20, 15)["pass"] is False
    assert ar.criterion1(10, 14)["pass"] is True  # reverse outcome: R3 recovers more
    assert ar.criterion1(20, 16)["value"] == 4


def test_c1_is_net_loss_not_discordant_count():
    # P7 concordant 0/5 and offsetting task moves do not count against R3
    ta = _task_arm([5, 5, 0, 3, 2, 1, 1, 0], [0, 5, 5, 3, 2, 1, 1, 0])
    res = ar.apply_r_rule(ta, _rows())
    c1 = res["criteria"][0]
    assert c1["value"] == 0 and c1["pass"]


# ------------------------------------------------------------------ criterion 2
def test_c2_drop_from_3_to_0_fails():
    c = ar.criterion2({"a": 3, "b": 5}, {"a": 0, "b": 5})
    assert c["pass"] is False and "a 3->0" in c["detail"]


def test_c2_drop_to_1_or_from_2_passes():
    assert ar.criterion2({"a": 3}, {"a": 1})["pass"] is True
    assert ar.criterion2({"a": 2}, {"a": 0})["pass"] is True


def test_c2_missing_r3_task_counts_as_zero():
    assert ar.criterion2({"a": 4}, {})["pass"] is False


# ------------------------------------------------------------------ criterion 3, 4
def test_c3_success_counts():
    assert ar.criterion3(7, 7)["pass"] is True
    assert ar.criterion3(8, 7)["pass"] is True
    assert ar.criterion3(6, 7)["pass"] is False


def test_c4_ratio_boundary():
    assert ar.criterion4(0.75, 1.0)["pass"] is True
    assert ar.criterion4(0.7501, 1.0)["pass"] is False
    assert ar.criterion4(0.6, 0.8)["value"] == pytest.approx(0.75)


# ------------------------------------------------------------------ decision
def test_all_pass_adopts_r3():
    ta = _task_arm([5, 4, 3, 2, 1, 0, 0, 0], [5, 4, 2, 2, 1, 0, 0, 0])
    assert ar.apply_r_rule(ta, _rows())["decision_R"] == 3


@pytest.mark.parametrize("kw,ta_r3", [
    ({}, [0, 0, 3, 2, 1, 0, 0, 0]),                       # c1 fails (loss 9) and c2 fails
    ({}, [5, 0, 3, 2, 1, 1, 1, 1]),                       # c2 only: p2 4->0, net loss 1
    ({"succ3": 7}, [5, 4, 3, 2, 1, 0, 0, 0]),             # c3
    ({"u3": 0.76}, [5, 4, 3, 2, 1, 0, 0, 0]),             # c4
])
def test_any_failure_keeps_r10(kw, ta_r3):
    ta = _task_arm([5, 4, 3, 2, 1, 0, 0, 0], ta_r3)
    res = ar.apply_r_rule(ta, _rows(**kw))
    assert res["decision_R"] == 10
    assert not all(c["pass"] for c in res["criteria"])


def test_c2_alone_can_block_adoption():
    ta = _task_arm([5, 4, 3, 2, 1, 0, 0, 0], [5, 0, 3, 2, 1, 1, 1, 1])
    res = ar.apply_r_rule(ta, _rows())
    passes = [c["pass"] for c in res["criteria"]]
    assert passes == [True, False, True, True] and res["decision_R"] == 10


def test_r1_never_adopted_even_if_dominant():
    # R1 recovers everything at a tenth of the cost; R3 fails c4 -> R=10, never 1
    ta = _task_arm([2] * 8, [2] * 8, r1=[5] * 8)
    res = ar.apply_r_rule(ta, _rows(u3=0.9, u1=0.1))
    assert res["decision_R"] == 10
    for c in res["criteria"]:
        assert not re.search(r"\bR1\b", c["detail"])


def test_r1_rows_do_not_influence_rule():
    ta = _task_arm([3] * 8, [3] * 8, r1=[0] * 8)
    a = ar.apply_r_rule(ta, _rows(succ1=0, u1=99.0))
    b = ar.apply_r_rule(ta, _rows(succ1=8, u1=0.01))
    assert a == b


def test_decide_requires_four_criteria():
    with pytest.raises(ValueError):
        ar.decide_r([{"id": 1, "pass": True}])
    assert ar.decide_r([{"id": i, "pass": True} for i in (1, 2, 3, 4)]) in (3,)


# ------------------------------------------------------------------ ITT, success, u
def test_itt_failed_row_gives_five_zero_episodes():
    ep = pd.DataFrame({"fold": range(5), "rec_sel": 1, "rec_any": 1})
    out = ar.itt_episode_rows(False, ep)
    assert len(out) == 5 and out["rec_sel"].sum() == 0 and out["itt_imputed"].all()


def test_itt_ok_row_keeps_scores_and_fills_missing_fold():
    ep = pd.DataFrame({"fold": [0, 1, 2, 4], "rec_sel": [1, 1, 0, 1], "rec_any": [1, 1, 1, 1]})
    out = ar.itt_episode_rows(True, ep)
    assert out["fold"].tolist() == [0, 1, 2, 3, 4]
    assert out["rec_sel"].tolist() == [1, 1, 0, 0, 1]
    assert out["itt_imputed"].tolist() == [0, 0, 0, 1, 0]


def test_success_direction_aware():
    # regression nRMSE: worse = higher
    assert ar.is_success("regression", 0.12, 0.10, 0.01, True) == 1
    assert ar.is_success("regression", 0.1201, 0.10, 0.01, True) == 0
    assert ar.is_success("regression", 0.05, 0.10, 0.01, True) == 1
    # classification balanced accuracy: worse = lower
    assert ar.is_success("classification", 0.80, 0.82, 0.01, True) == 1
    assert ar.is_success("classification", 0.79, 0.82, 0.01, True) == 0
    # cap hit / failed run / missing base
    assert ar.is_success("classification", 0.99, 0.82, 0.01, False) == 0
    assert ar.is_success("regression", None, 0.1, 0.01, True) == 0
    assert ar.is_success("regression", 0.1, None, 0.01, True) == 0


def test_u_run_sums_attempts_and_caps_at_6h():
    assert ar.u_run([3600, 7200], [False, False], 3600) == pytest.approx(3.0)
    assert ar.u_run([100, 50], [True, False], 3600) == pytest.approx(6.0 + 50 / 3600)


# ------------------------------------------------------------------ M5S
def _a08(hits):
    rows = []
    for (ds, arm), k in hits.items():
        for i in range(15):
            rows.append({"dataset_id": ds, "arm": arm, "rec_any": int(i < k)})
    return pd.DataFrame(rows)


def test_m5s_gate_counts_families_and_p6_once():
    ep = _a08({("synp_p1", "openfe"): 15, ("synp_p1", "autofeat"): 15, ("synp_p6a", "autofeat"): 14,
               ("synp_p6b", "openfe"): 14, ("synp_p2", "openfe"): 13, ("synp_p0", "openfe"): 0,
               ("synp_p0", "raw"): 0})
    g = ar.m5s_gate(ep)
    assert g["nondisc_tasks"] == ["synp_p1", "synp_p6a", "synp_p6b"]
    assert g["nondisc_families"] == ["P1", "P6"]
    assert g["gate_pass"] is True and g["p0_recoveries"] == 0


def test_m5s_gate_fails_on_p0_recovery_or_four_families():
    assert ar.m5s_gate(_a08({("synp_p0", "openfe"): 1}))["gate_pass"] is False
    four = {(f"synp_p{i}", "autofeat"): 15 for i in (1, 2, 3, 4)}
    assert ar.m5s_gate(_a08(four))["gate_pass"] is False


def test_flag_invariant_detects_difference_and_wrong_rounds():
    man = pd.DataFrame([
        {"row_id": "A", "env_overrides": '{"ROGUE_ONE_EXTRACTOR_ROUNDS":"10"}'},
        {"row_id": "B", "env_overrides": '{"ROGUE_ONE_EXTRACTOR_ROUNDS":"3","ROGUE_ONE_STRATIFIED_FOLDS":"True"}'},
    ])
    good = {"A": {"flags/ROGUE_ONE_EXTRACTOR_ROUNDS": "10", "flags/ROGUE_ONE_STRATIFIED_FOLDS": "False", "flags/X": 1},
            "B": {"flags/ROGUE_ONE_EXTRACTOR_ROUNDS": "3", "flags/ROGUE_ONE_STRATIFIED_FOLDS": "True", "flags/X": 1}}
    assert ar.flag_invariant(good, man)["pass"] is True
    bad = {k: dict(v) for k, v in good.items()}
    bad["B"]["flags/X"] = 2
    bad["A"]["flags/ROGUE_ONE_EXTRACTOR_ROUNDS"] = "3"
    r = ar.flag_invariant(bad, man)
    assert r["pass"] is False and "flags/X" in r["differing"] and r["by_arm_mismatch"]
    # the dataset id and the 2xTP2 vLLM port differ by design: reported, not a violation
    design = {k: dict(v) for k, v in good.items()}
    design["A"].update({"flags/LLM_PORT": "8010", "flags/ROGUE_ONE_DATASETS": "synp_p1"})
    design["B"].update({"flags/LLM_PORT": "8011", "flags/ROGUE_ONE_DATASETS": "synp_p3"})
    r = ar.flag_invariant(design, man)
    assert r["pass"] is True and set(r["differing_by_design"]) == {"flags/LLM_PORT", "flags/ROGUE_ONE_DATASETS"}
