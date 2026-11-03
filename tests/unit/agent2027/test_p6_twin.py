"""P6a/P6b exact-twin tests (E8a, decision D8 option A)."""

import json

import numpy as np
import pytest
import yaml

from scripts.agent2027.synp import generate_synp as G
from scripts.agent2027.synp import p6_twin as P
from scripts.agent2027.synp import tasks as T

pytestmark = pytest.mark.unit


def make(tmp_path, candidate, **kw):
    tasks, meta = tmp_path / "tasks", tmp_path / "meta"
    out = G.build_all(tasks, meta, keys=("p6a", "p6b"), p6a_candidate=candidate, **kw)
    return tasks, meta, out


@pytest.mark.parametrize("candidate", ["reynolds", "ideal_gas", "cockcroft_gault"])
def test_twin_checks_pass_both_branches(tmp_path, candidate):
    tasks, meta, out = make(tmp_path, candidate)
    mb = json.loads((meta / "synp_p6b.json").read_text())
    assert mb["rescaled"] == (candidate != "cockcroft_gault")  # monomial -> hidden rescale, else none
    res = P.verify_twin(tasks / "regression" / "synp_p6a", tasks / "regression" / "synp_p6b", mb)
    assert all(res.values()), res


def test_naming_rule_sorted_rank_and_padding(tmp_path):
    _, _, out = make(tmp_path, "reynolds")
    a, b = out["p6a"], out["p6b"]
    cols = [c for c in a.data.columns if c != "id"]
    nm = b.meta["name_map"]
    assert sorted(cols, key=lambda c: nm[c]) == sorted(cols)  # f-order == alphabetical order of P6a
    assert sorted(nm.values()) == [f"f{i:02d}" for i in range(1, len(cols) + 1)]
    assert [nm[c] for c in cols] == [c for c in b.data.columns if c != "id"]  # P6a column order kept
    for c, lm in b.meta["level_map"].items():
        assert sorted(lm.values()) == [f"c{i:02d}" for i in range(1, len(lm) + 1)]
        assert sorted(lm, key=lm.get) == sorted(lm)


def test_rows_labels_ids_identical_and_hashes(tmp_path):
    _, _, out = make(tmp_path, "reynolds")
    a, b = out["p6a"], out["p6b"]
    assert a.labels.equals(b.labels) and (a.data["id"] == b.data["id"]).all()
    assert b.meta["p6a_data_sha256"] == a.meta["data_sha256"]
    assert b.meta["phi_sha256"] == a.meta["phi_sha256"] == b.meta["p6a_phi_sha256"]
    assert b.meta["data_seed"] == a.meta["data_seed"]


def test_hidden_rescale_is_positive_all_numeric_no_shift(tmp_path):
    _, _, out = make(tmp_path, "reynolds")
    a, b = out["p6a"], out["p6b"]
    k = b.meta["scale_constants"]
    num = [c for c in a.data.columns if c != "id" and c not in a.categorical]
    assert set(k) == set(num) and all(v > 0 for v in k.values())
    inv = P.invert_twin(b.data, b.meta)
    for c in num:
        np.testing.assert_allclose(inv[c].to_numpy(float), a.data[c].to_numpy(float), rtol=1e-9)
        assert (a.data[c].rank().to_numpy() == b.data[b.meta["name_map"][c]].rank().to_numpy()).all()
    assert b.meta["phi_scale_factor"] > 0


def test_forced_rescale_branches():
    a = T.generate_task("p6a", n=300, p6a_candidate="reynolds")
    no = P.make_twin(a, rescale=False)
    yes = P.make_twin(a, rescale=True)
    assert not no.meta["scale_constants"] and yes.meta["scale_constants"]
    cg = T.generate_task("p6a", n=300, p6a_candidate="cockcroft_gault")
    with pytest.raises(ValueError):
        P.make_twin(cg, rescale=True)
    assert not P.make_twin(cg).meta["rescaled"]


def test_neutral_context_length_within_2pct():
    for cand in ("reynolds", "ideal_gas", "cockcroft_gault"):
        a = T.generate_task("p6a", n=300, p6a_candidate=cand)
        b = P.make_twin(a)
        assert abs(len(b.context) - len(a.context)) <= 0.02 * len(a.context)
        assert "Columns:\n- f" in b.context and "Goal: predict the target variable" in b.context


def test_static_leak_check_clean_and_detects_planted_leak(tmp_path):
    tasks, meta, out = make(tmp_path, "reynolds")
    terms = P.domain_terms(out["p6a"], "reynolds")
    d_a, d_b = tasks / "regression" / "synp_p6a", tasks / "regression" / "synp_p6b"
    res = P.static_leak_check(terms, d_a, d_b)
    assert res == {"p6b_hits": [], "p6a_context_strong_hits": []}, res
    cfg = yaml.safe_load((d_b / "config.yml").read_text())
    cfg["context_and_goal"] += " Reynolds number"
    (d_b / "config.yml").write_text(yaml.safe_dump(cfg))
    assert "reynolds" in [t.lower() for t in P.static_leak_check(terms, d_a, d_b)["p6b_hits"]]


def test_p6b_has_no_name_leak_in_data(tmp_path):
    _, _, out = make(tmp_path, "reynolds")
    b = out["p6b"]
    assert all(c == "id" or c.startswith("f") for c in b.data.columns)
    for c in b.categorical:
        assert all(v.startswith("c") for v in b.data[c].unique())
    assert b.meta["task_id"] == "synp_p6b" and b.task_type == "regression"
