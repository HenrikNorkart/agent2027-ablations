"""E1a flag registry and E1b task registration.

* the registry reads effective (post-dotenv) values, including a .dev.env that
  overrides a docker -e value (config.py's override=True);
* init_wandb_run merges flags/<KEY> and the identity keys and takes the R12 name;
* ROGUE_ONE_TASK_REGISTRY unset -> the legacy list; set -> the YAML list;
  ROGUE_ONE_SMOKE_FOLDS truncates total_trials to n x iterations.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

for _k, _v in {"EMBEDD_API_KEY": "x", "EMBEDD_MODEL": "x", "EMBEDD_PORT": "1", "LLM_MODEL": "x"}.items():
    os.environ.setdefault(_k, _v)

import pytest

from rogueone.utils import flag_registry as fr
from rogueone.utils import task_registry as tr

pytestmark = [pytest.mark.unit]

IDENTITY = {
    "ROGUE_ONE_ABLATION_ID": "A02", "ROGUE_ONE_ARM_ID": "R3", "ROGUE_ONE_REPLICATE": "0",
    "ROGUE_ONE_SESSION_ID": "S1012a", "ROGUE_ONE_MANIFEST_ROW": "A02-902",
    "ROGUE_ONE_IS_SENTINEL": "0", "ROGUE_ONE_CODE_PHASE": "smoke",
    "ROGUE_ONE_IMAGE_TAG": "agent2027-pre.1", "ROGUE_ONE_GIT_SHA": "abc1234",
    "ROGUE_ONE_SERVER_TOPOLOGY": "uniform:O120@8010",
    "ROGUE_ONE_RUNG_S": "O120", "ROGUE_ONE_RUNG_E": "O120", "ROGUE_ONE_RUNG_T": "O120",
    "ROGUE_ONE_IMAGE_ID": "sha256:img", "ROGUE_ONE_VLLM_IMAGE_ID": "sha256:vllm",
    "ROGUE_ONE_PREREG_TAG": "agent2027-prereg", "ROGUE_ONE_PARALLEL": "6",
}

# The 28 configs rogue_one_main.py had inline at origin/agent2027/v2 (63681ff), minus
# covtype (never run, hard rule 2026-10-16).
LEGACY = [
    "classification/balance-scale", "classification/pc1",
    "classification/tic-tac-toe", "classification/junglechess", "classification/communities",
    "classification/eucalyptus", "classification/blood", "classification/car",
    "classification/arrhythmia", "classification/bank", "classification/breast-w",
    "classification/diabetes", "classification/cmc", "classification/adult",
    "classification/heart", "classification/vehicle", "classification/credit-g",
    "classification/myocardial", "regression/forest-fires", "regression/wine",
    "regression/plasma_retinol", "regression/housing", "regression/insurance",
    "regression/crab", "regression/bike", "regression/airfoil_self_noice",
    "regression/diamonds",
]


# --------------------------------------------------------------------------- E1a
def test_registry_lists_every_r3_r4_and_identity_key():
    for key in ("ROGUE_ONE_NUM_ITERATIONS", "ROGUE_ONE_STRATIFIED_FOLDS", "LLM_TEMPERATURE",
                "SCIENTIST_AGENT_MAX_TOKENS", "ROGUE_ONE_TESTER_PORT", "CONSOLE_DISABLE",
                "ROGUE_ONE_EXTRACTOR_ROUNDS", "ROGUE_ONE_TOOL_ERROR_FEEDBACK",
                "ROGUE_ONE_TASK_REGISTRY", "ROGUE_ONE_SMOKE_FOLDS", "ROGUE_ONE_PLANNER_MODE",
                "ROGUE_ONE_SANDBOX", "ROGUE_ONE_SCIENTIST_SIGNAL", "ROGUE_ONE_FALLBACKS"):
        assert key in fr.FLAG_KEYS
    assert len(fr.FLAG_KEYS) == len(set(fr.FLAG_KEYS))
    assert set(IDENTITY) - set(fr.LAUNCHER_META_KEYS) == set(fr.IDENTITY_KEYS)


def test_effective_values_follow_the_dotenv_chain_incl_dev_env_override(tmp_path, monkeypatch):
    # docker -e values already in the environment before config.py's dotenv chain runs
    monkeypatch.setenv("ROGUE_ONE_K_FOLDS", "5")
    monkeypatch.setenv("LLM_PORT", "8010")
    monkeypatch.delenv("ROGUE_ONE_EXTRACTOR_ROUNDS", raising=False)
    monkeypatch.delenv("ROGUE_ONE_TOOL_OUTPUT_MAX_CHARS", raising=False)
    (tmp_path / ".env").write_text("LLM_PORT=9999\nROGUE_ONE_EXTRACTOR_ROUNDS=3\n")
    (tmp_path / ".dev.env").write_text("# baked dev overrides\nLLM_PORT=8000\nROGUE_ONE_K_FOLDS=2\n")
    (tmp_path / ".secret.env").write_text("ROGUE_ONE_K_FOLDS=7\nSERPER_API_KEY=s3cret\n")
    for k in ("SERPER_API_KEY",):
        monkeypatch.delenv(k, raising=False)
    saved_state = dict(fr.DOTENV_STATE)
    saved_env = dict(os.environ)  # load_dotenv writes os.environ directly
    try:
        state = fr.load_dotenv_chain(tmp_path)
        flags = fr.effective_flags()
        # .env never beats an existing value; .dev.env (override=True) does; .secret.env does not
        assert flags["LLM_PORT"] == "8000"
        assert flags["ROGUE_ONE_K_FOLDS"] == "2"
        assert flags["ROGUE_ONE_EXTRACTOR_ROUNDS"] == "3"           # only .env set it
        assert flags["ROGUE_ONE_TOOL_OUTPUT_MAX_CHARS"] == "0"      # unset -> code default
        assert "ROGUE_ONE_TOOL_OUTPUT_MAX_CHARS" in fr.unset_flags()
        assert flags["SERPER_API_KEY"] == "<set>"                   # secrets masked
        assert state["files_loaded"] == [".env", ".dev.env", ".secret.env"]
        assert state["dev_env_keys"] == ["LLM_PORT", "ROGUE_ONE_K_FOLDS"]
    finally:
        os.environ.clear()
        os.environ.update(saved_env)
        fr.DOTENV_STATE.clear()
        fr.DOTENV_STATE.update(saved_state)


def test_wandb_kwargs_without_identity_keep_legacy_name():
    env = {"ROGUE_ONE_EXTRACTOR_ROUNDS": "3"}
    out = fr.wandb_run_kwargs(dataset_id="heart", task_type="classification", fold_seed=42,
                              llm_seed=None, environ=env)
    assert set(out) == {"config"}
    assert out["config"]["flags/ROGUE_ONE_EXTRACTOR_ROUNDS"] == "3"


def test_init_wandb_run_merges_flags_identity_and_r12_name(monkeypatch, tmp_path):
    from rogueone.utils import wandb_utils

    for k, v in IDENTITY.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("ROGUE_ONE_EXTRACTOR_ROUNDS", "3")
    monkeypatch.setenv("WANDB_ENTITY", "")
    monkeypatch.setattr(wandb_utils.rouge_one_cfg, "do_logging", True)
    monkeypatch.setattr(wandb_utils.rouge_one_cfg, "fold_seed", 42)
    monkeypatch.setattr(wandb_utils.rouge_one_cfg, "llm_seed", 42)
    monkeypatch.setattr(wandb_utils.wandb_cfg, "entity", "")
    monkeypatch.setattr(wandb_utils, "sign_in_wandb", lambda: None)
    captured = {}

    def _fake_init(**kw):
        captured.update(kw)
        return SimpleNamespace(name=kw.get("name"), id="x")

    monkeypatch.setattr(wandb_utils.wandb, "init", _fake_init)
    cfg = SimpleNamespace(
        experiment_name="synp_p1",
        feature_file_path=Path("/workspace/tasks/regression/synp_p1/data.csv"),
        task_type="regression",
        to_dict=lambda: {"experiment_name": "synp_p1", "task_type": "regression"},
    )
    wandb_utils.init_wandb_run(cfg)

    assert captured["name"] == "A02__R3__synp_p1__f42__r0"
    assert captured["entity"] == "PlanBench"
    assert captured["group"] == "A02" and captured["job_type"] == "R3"
    assert captured["tags"] == ["S1012a", "smoke", "agent2027-pre.1"]
    conf = captured["config"]
    for k in fr.FLAG_KEYS:
        assert f"flags/{k}" in conf
    assert conf["flags/ROGUE_ONE_EXTRACTOR_ROUNDS"] == "3"
    for key in ("ablation_id", "arm_id", "dataset_id", "task_type", "fold_seed", "llm_seed",
                "replicate", "is_sentinel", "session_id", "manifest_row", "code_phase", "git_sha",
                "image_tag", "image_id", "vllm_image_id", "prereg_tag", "rung_S", "rung_E",
                "rung_T", "model_S", "model_E", "model_T", "model_W", "server_topology", "parallel"):
        assert key in conf, key
    assert conf["manifest_row"] == "A02-902" and conf["dataset_id"] == "synp_p1"
    assert conf["experiment_name"] == "synp_p1"  # cfg.to_dict() kept


def test_init_wandb_run_legacy_name_and_entity_without_identity(monkeypatch):
    from rogueone.utils import wandb_utils

    for k in IDENTITY:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(wandb_utils.rouge_one_cfg, "do_logging", True)
    monkeypatch.setattr(wandb_utils.wandb_cfg, "entity", "brainflux")
    monkeypatch.setattr(wandb_utils, "sign_in_wandb", lambda: None)
    captured = {}
    monkeypatch.setattr(wandb_utils.wandb, "init",
                        lambda **kw: captured.update(kw) or SimpleNamespace(name=kw["name"], id="x"))
    cfg = SimpleNamespace(experiment_name="heart", task_type="classification",
                          feature_file_path=Path("/workspace/tasks/classification/heart/data.csv"),
                          to_dict=lambda: {})
    wandb_utils.init_wandb_run(cfg)
    assert captured["name"] == "heart" and captured["entity"] == "brainflux"
    assert "group" not in captured and "tags" not in captured
    assert "flags/ROGUE_ONE_NUM_ITERATIONS" in captured["config"]


# --------------------------------------------------------------------------- E1b
def test_unset_registry_selects_the_legacy_list():
    paths = tr.task_paths({})
    assert [f"{p.parent.parent.name}/{p.parent.name}" for p in paths] == LEGACY
    assert all(str(p).replace("\\", "/").startswith("/workspace/tasks/") for p in paths)
    assert tr.task_paths({"ROGUE_ONE_TASK_REGISTRY": "  "}) == paths


def test_set_registry_selects_the_yaml_list(tmp_path):
    reg = tmp_path / "task_registry.yml"
    reg.write_text("- /workspace/tasks/regression/synp_p1/config.yml\n"
                   "- /workspace/tasks/classification/synp_p3/config.yml\n"
                   "- relative/heart/config.yml\n")
    paths = tr.task_paths({"ROGUE_ONE_TASK_REGISTRY": str(reg)})
    names = [p.parent.name for p in paths]
    assert names == ["synp_p1", "synp_p3", "heart"]
    assert paths[2] == tmp_path / "relative" / "heart" / "config.yml"
    # select_paths still filters by ROGUE_ONE_DATASETS
    missing = []
    chosen = tr.select_paths(paths, ["synp_p3", "nope"], on_missing=missing.append)
    assert [p.parent.name for p in chosen] == ["synp_p3"] and missing == ["nope"]
    assert tr.select_paths(paths, []) == paths


def test_broken_registry_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        tr.task_paths({"ROGUE_ONE_TASK_REGISTRY": str(tmp_path / "missing.yml")})
    bad = tmp_path / "bad.yml"
    bad.write_text("tasks: 3\n")
    with pytest.raises(ValueError):
        tr.task_paths({"ROGUE_ONE_TASK_REGISTRY": str(bad)})


def test_smoke_folds_truncates_total_trials():
    assert tr.total_trials(5, 10, None) == 50
    assert tr.total_trials(5, 10, tr.parse_smoke_folds("1")) == 10
    assert tr.total_trials(5, 10, tr.parse_smoke_folds("2")) == 20
    assert tr.total_trials(5, 10, tr.parse_smoke_folds("9")) == 50  # never more than k
    assert tr.parse_smoke_folds("") is None and tr.parse_smoke_folds(None) is None
    assert tr.evaluated_fold_count(5, 2) == 2
    with pytest.raises(ValueError):
        tr.parse_smoke_folds("0")


def test_rogueone_config_reads_smoke_folds(monkeypatch):
    from rogueone.utils.config import rogueoneConfig
    from rogueone.utils.singelton import Singleton

    saved = Singleton._instances.pop(rogueoneConfig, None)
    try:
        monkeypatch.setenv("ROGUE_ONE_SMOKE_FOLDS", "1")
        assert rogueoneConfig().smoke_folds == 1
        Singleton._instances.pop(rogueoneConfig, None)
        monkeypatch.delenv("ROGUE_ONE_SMOKE_FOLDS")
        assert rogueoneConfig().smoke_folds is None
    finally:
        Singleton._instances.pop(rogueoneConfig, None)
        if saved is not None:
            Singleton._instances[rogueoneConfig] = saved


# --------------------------------------------------------------------------- covtype ban
def test_covtype_is_never_selected(tmp_path, monkeypatch):
    from rogueone.utils import task_registry as tr

    assert not any("covtype" in str(p).lower() for p in tr.LEGACY_TASK_PATHS)
    reg = tmp_path / "reg.yml"
    reg.write_text("- /workspace/tasks/classification/covtype/config.yml" + chr(10)
                   + "- /workspace/tasks/classification/heart/config.yml" + chr(10), encoding="utf-8")
    paths = tr.task_paths({"ROGUE_ONE_TASK_REGISTRY": str(reg)})
    assert [p.parent.name for p in paths] == ["heart"]
    assert tr.select_paths([Path("/w/tasks/classification/covtype/config.yml")], []) == []
    with pytest.raises(tr.BannedDatasetError):
        tr.select_paths(paths, ["covtype"])
    with pytest.raises(tr.BannedDatasetError):
        tr.select_paths(paths, ["heart", "CovType"])
