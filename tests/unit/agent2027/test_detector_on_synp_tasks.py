"""End-to-end: generated SYN-P folders + off-mount meta + detector episode file (E8a)."""

import json

import pytest

from scripts.agent2027.analysis import recovery_detector as rd
from scripts.agent2027.synp import generate_synp as G

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("det")
    out = G.build_all(root / "tasks", root / "meta", n=500, p7_secret="det-secret")
    return root, out


def _episode(task_id, task_type, attrs, active, selected, fold_seed=42, fold_id=0, stratified=False):
    return {"episode_id": f"{task_id}-{fold_seed}-{fold_id}", "task_id": task_id, "task_type": task_type,
            "fold_seed": fold_seed, "fold_id": fold_id, "stratified": stratified,
            "attributes": attrs, "active_by_iter": active, "selected_iter": selected}


def test_oracle_expression_from_meta_is_recovered_on_every_task(world, tmp_path):
    root, out = world
    eps = []
    for key, g in out.items():
        meta = json.loads((root / "meta" / f"{g.task_id}.json").read_text())
        cmd = "result = " + meta["phi_pandas_expr"].replace("df[", "df[")
        attrs = [{"Attribute": "oracle_feature", "Pandas Command": cmd, "Description": "LLM"}]
        if key == "p0":  # P0: phi is a monotone transform of one column -> never counts
            eps.append(_episode(g.task_id, g.task_type, attrs, {"0": ["oracle_feature"]}, 0))
        else:
            eps.append(_episode(g.task_id, g.task_type, attrs, {"0": ["oracle_feature"]}, 0,
                                stratified=g.task_type == "classification"))
    f = tmp_path / "episodes.json"
    f.write_text(json.dumps(eps))
    res = rd.run_episodes_file(f, root / "tasks", root / "meta").set_index("task_id")
    assert res.loc["synp_p0", ["rec_any", "rec_sel"]].tolist() == [0, 0]
    for tid in res.index:
        if tid != "synp_p0":
            assert res.loc[tid, ["rec_any", "rec_sel"]].tolist() == [1, 1], tid


def test_wrong_features_and_textbook_prior_do_not_recover(world, tmp_path):
    root, out = world
    g5 = out["p5"]
    textbook = ("result = ((df['Left-Weight'] * df['Left-Distance'] - df['Right-Weight'] * df['Right-Distance'])"
                ".apply(lambda v: 0 if v == 0 else (1 if v > 0 else -1))).astype(float)")
    attrs = [{"Attribute": "torque", "Pandas Command": textbook, "Description": "LLM"},
             {"Attribute": "raw", "Pandas Command": "result = df['Left-Weight']", "Description": "LLM"}]
    f = tmp_path / "e.json"
    f.write_text(json.dumps([_episode("synp_p5", "classification", attrs, {"0": ["torque", "raw"]}, 0,
                                      stratified=True)]))
    res = rd.run_episodes_file(f, root / "tasks", root / "meta")
    assert res.loc[0, "rec_any"] == 0  # the textbook LW*LD rule is the wrong prior (P5)
    # and the P7 phi_wrong never counts as phi
    g7 = out["p7"]
    meta7 = json.loads((root / "meta" / "synp_p7.json").read_text())
    w = meta7["phi_wrong"]
    assert w and w["expr"] != meta7["phi"]["expr"]
    assert g5.task_type == "classification"
