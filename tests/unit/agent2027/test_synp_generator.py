"""Unit tests for the SYN-P generator, folder format, metadata separation (E8a)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.agent2027.synp import generate_synp as G
from scripts.agent2027.synp import tasks as T

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("synp")
    tasks, meta = root / "tasks", root / "meta"
    out = G.build_all(tasks, meta, p7_secret="unit-test-secret", registry_prefix="/workspace/tasks")
    return tasks, meta, out


def test_nine_folders_with_exactly_three_files(built):
    tasks, _, out = built
    assert set(out) == set(T.TASK_KEYS)
    for g in out.values():
        d = tasks / g.task_type / g.task_id
        assert sorted(p.name for p in d.iterdir()) == sorted(G.TASK_FILES)


def test_sizes_and_format(built):
    _, _, out = built
    for key, g in out.items():
        assert len(g.data) == 2000 and list(g.labels.columns) == ["id", "target"]
        p = g.data.shape[1] - 1
        assert 20 <= p <= 30, (key, p)
        assert not g.data.drop(columns="id").duplicated().any()
        assert g.data["id"].is_unique and (g.data["id"] == g.labels["id"]).all()
        assert g.data.drop(columns="id").isna().sum().sum() == 0


def test_regression_targets_standardised_classification_stratifiable(built):
    _, _, out = built
    for key, g in out.items():
        y = g.labels["target"]
        if g.task_type == "regression":
            assert y.std(ddof=0) == pytest.approx(1.0, abs=1e-9)
            assert y.mean() == pytest.approx(T.TARGET_OFFSET, abs=1e-9)
        else:
            assert pd.api.types.is_integer_dtype(y)
            assert y.value_counts().min() >= 5  # StratifiedKFold(5) viable


def test_folders_load_through_experiment_config(built, tmp_path, monkeypatch):
    from rogueone.utils.config import ExperimentConfig

    tasks, _, out = built
    monkeypatch.chdir(tmp_path)  # ExperimentConfig creates ./output
    for g in out.values():
        cfg = ExperimentConfig.from_yaml(tasks / g.task_type / g.task_id / "config.yml")
        assert cfg.task_type == g.task_type and cfg.experiment_name == g.task_id
        assert cfg.feature_file_path.exists() and cfg.label_file_path.exists()
        assert cfg.context_and_goal == g.context


def test_task_folders_contain_no_metadata(built):
    tasks, meta, out = built
    p7_thr = str(out["p7"].meta["p7_constants"]["thr"])
    for g in out.values():
        d = tasks / g.task_type / g.task_id
        blob = "\n".join((d / f).read_text(encoding="utf-8") for f in G.TASK_FILES)
        assert p7_thr not in blob
        low = blob.lower()
        for needle in ("seed", "oracle", "np.sin", "sha256", "synp_meta", "oracle_feature"):
            assert needle not in low, (g.task_id, needle)
        assert g.meta["phi"]["expr"] not in blob
        assert g.meta["oracle_focus"] not in blob


def test_meta_files_written_off_mount(built):
    tasks, meta, out = built
    for g in out.values():
        m = json.loads((meta / f"{g.task_id}.json").read_text(encoding="utf-8"))
        for k in ("phi", "data_seed", "noise_seed", "oracle_focus", "detector", "data_sha256", "phi_sha256"):
            assert k in m, (g.task_id, k)
    assert not list(tasks.rglob("*.json"))
    reg = yaml.safe_load((meta / "registry.yaml").read_text())
    assert {k: v["task_type"] for k, v in reg.items()} == {f"synp_{k}": T.TASK_TYPES[k] for k in T.TASK_KEYS}
    cfgs = yaml.safe_load((meta / "task_registry.yml").read_text())
    assert len(cfgs) == 9 and all(c.startswith("/workspace/tasks/") and c.endswith("config.yml") for c in cfgs)


def test_meta_dir_must_be_off_mount_and_off_repo(tmp_path):
    with pytest.raises(ValueError):
        G.check_meta_dir(tmp_path / "tasks" / "meta", tmp_path / "tasks")
    with pytest.raises(ValueError):
        G.check_meta_dir(tmp_path, tmp_path / "tasks")
    with pytest.raises(ValueError):
        G.check_meta_dir(REPO / "synp_meta", tmp_path / "tasks")
    G.check_meta_dir(tmp_path / "meta", tmp_path / "tasks")  # fine


def test_deterministic_given_seed():
    a = T.generate_task("p3", n=300)
    b = T.generate_task("p3", n=300)
    c = T.generate_task("p3", n=300, master_seed=7)
    assert a.meta["data_sha256"] == b.meta["data_sha256"] and a.meta["labels_sha256"] == b.meta["labels_sha256"]
    assert a.meta["data_sha256"] != c.meta["data_sha256"]


def test_p7_constant_is_derived_from_secret_and_then_read_back(tmp_path):
    out1 = G.build_all(tmp_path / "t1", tmp_path / "m", keys=("p7",), n=400, p7_secret="s1")
    thr1 = out1["p7"].meta["p7_constants"]["thr"]
    out2 = G.build_all(tmp_path / "t2", tmp_path / "m", keys=("p7",), n=400, p7_secret="OTHER")
    assert out2["p7"].meta["p7_constants"]["thr"] == thr1  # read back from the meta dir
    out3 = G.build_all(tmp_path / "t3", tmp_path / "m3", keys=("p7",), n=400, p7_secret="OTHER")
    assert out3["p7"].meta["p7_constants"]["thr"] != thr1
    assert str(thr1) not in Path(T.__file__).read_text()  # never hard-coded in the repo


def test_phi_replays_from_csv_and_meta(built):
    tasks, meta, out = built
    for g in out.values():
        m = json.loads((meta / f"{g.task_id}.json").read_text())
        df = pd.read_csv(tasks / g.task_type / g.task_id / "data.csv")
        phi = T.phi_from_meta(m, df)
        if g.key != "p6b":
            assert T.phi_sha256(phi) == m["phi_sha256"], g.task_id
        assert np.isfinite(phi).all()


def test_p7_phi_wrong_defined_and_decorrelated(built):
    _, _, out = built
    from scripts.agent2027.analysis.recovery_detector import spearman_abs

    g = out["p7"]
    w = g.meta["phi_wrong"]
    assert w is not None
    assert spearman_abs(T.eval_phi(w, g.data), g.phi) < 0.5


def test_context_template_shared_and_clean(built):
    _, _, out = built
    for g in out.values():
        c = g.context
        assert "Columns:\n- " in c and "\nGoal: " in c and f"{len(g.data)} records" in c
        assert c.count("\n- ") == g.data.shape[1] - 1
    assert "Marenko" in out["p7"].context
    forbidden = ["a / b", "ratio of", "np.", "threshold", "reynolds", "formula"]
    for g in out.values():
        low = g.context.lower()
        assert not any(t in low for t in forbidden), g.task_id
    assert str(out["p7"].meta["p7_constants"]["thr"])[:6] not in out["p7"].context


def test_p5_uses_real_balance_scale_grid(built):
    _, _, out = built
    g = out["p5"]
    names = list(T.BALANCE_NAMES)
    assert all(n in g.data.columns for n in names)
    assert len(g.data[names].drop_duplicates()) == 625  # every cell of the real 5^4 grid occurs
    lw, ld, rw, rd = (g.data[n].to_numpy(float) for n in names)
    y = np.sign(lw * ld**2 - rw * rd**2)
    expect = np.where(y > 0, 0, np.where(y == 0, 1, 2))
    assert (g.labels["target"].to_numpy() == expect).all()
    assert g.meta["phi_textbook_prior"]["expr"] == "np.sign(a * b - c * d)"


def test_oracle_focus_strings(built):
    _, _, out = built
    assert "df['energy_consumed_kwh'] / df['distance_km']" in out["p1"].meta["oracle_focus"]
    assert str(out["p7"].meta["p7_constants"]["thr"]) in out["p7"].meta["oracle_focus"]
    assert "in steps" in out["p3"].meta["oracle_focus"] and "in steps" not in out["p1"].meta["oracle_focus"]
    b = out["p6b"].meta["oracle_focus"]
    assert "oracle_feature" in b and "df['f" in b and "flow_velocity" not in b


def test_signal_params_monotone_in_strength():
    weak = T.generate_task("p1", n=500, params={"noise": 2.0})
    strong = T.generate_task("p1", n=500, params={"noise": 0.2})
    assert weak.data.equals(strong.data)  # rows unchanged by the signal parameter

    def r(g):
        return abs(np.corrcoef(g.phi, g.labels["target"])[0, 1])

    assert r(strong) > r(weak)
