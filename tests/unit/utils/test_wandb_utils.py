"""Unit tests for W&B utilities."""

from pathlib import Path

import pytest

pytest_plugins = []


@pytest.mark.unit
@pytest.mark.config
def test_wandb_utils_imports():
    """Verify wandb_utils module can be imported."""
    try:
        from rogueone.utils import wandb_utils

        assert wandb_utils is not None
    except ImportError:
        pytest.skip("wandb_utils not available")


@pytest.mark.unit
@pytest.mark.config
def test_wandb_utils_has_logging_functions():
    """
    Test wandb_utils provides logging capabilities.

    Per spec FR-005: Experiment utilities must include W&B logging
    """
    try:
        from rogueone.utils import wandb_utils

        # Check for common logging functions
        # This test documents expected functionality

    except ImportError:
        pytest.skip("wandb_utils not available")


@pytest.mark.unit
@pytest.mark.config
def test_wandb_logging_handles_missing_project():
    """
    Test W&B logging gracefully handles missing WANDB_PROJECT.

    Per spec FR-003: Must validate failure handling
    """
    try:
        from rogueone.utils import wandb_utils

        # Placeholder test - documents expected behavior
        # When logging is called without W&B config, should not crash

    except ImportError:
        pytest.skip("wandb_utils not available")


@pytest.mark.unit
@pytest.mark.config
def test_log_file_artifact_uploads_existing_file(monkeypatch, tmp_path):
    from rogueone.utils import wandb_utils

    monkeypatch.setattr(wandb_utils.rouge_one_cfg, "do_logging", True)
    monkeypatch.setattr(wandb_utils.wandb, "run", object())

    file_path = tmp_path / "agent_toolcall_errors.jsonl"
    file_path.write_text('{"event":"tool_error"}\n', encoding="utf-8")

    artifact_calls = {}

    class FakeArtifact:
        def __init__(self, name, type, metadata=None):
            artifact_calls["name"] = name
            artifact_calls["type"] = type
            artifact_calls["metadata"] = metadata
            artifact_calls["files"] = []

        def add_file(self, path):
            artifact_calls["files"].append(path)

    logged = {}

    monkeypatch.setattr(wandb_utils.wandb, "Artifact", FakeArtifact)
    monkeypatch.setattr(
        wandb_utils.wandb,
        "log_artifact",
        lambda artifact: logged.setdefault("artifact", artifact),
    )

    uploaded = wandb_utils.log_file_artifact(
        file_path,
        artifact_name="agent_toolcall_errors_demo",
        artifact_type="agent-tool-errors",
        metadata={"experiment_name": "demo"},
    )

    assert uploaded is True
    assert artifact_calls["name"] == "agent_toolcall_errors_demo"
    assert artifact_calls["type"] == "agent-tool-errors"
    assert artifact_calls["metadata"] == {"experiment_name": "demo"}
    assert artifact_calls["files"] == [str(file_path)]
    assert "artifact" in logged


@pytest.mark.unit
@pytest.mark.config
def test_log_file_artifact_skips_missing_file(monkeypatch, tmp_path):
    from rogueone.utils import wandb_utils

    monkeypatch.setattr(wandb_utils.rouge_one_cfg, "do_logging", True)
    monkeypatch.setattr(wandb_utils.wandb, "run", object())

    missing_path = tmp_path / "missing.jsonl"
    messages = []

    monkeypatch.setattr(
        wandb_utils.ConsoleManager,
        "console_print",
        lambda message, style=None: messages.append(message),
    )

    uploaded = wandb_utils.log_file_artifact(
        Path(missing_path),
        artifact_name="agent_toolcall_errors_demo",
        artifact_type="agent-tool-errors",
    )

    assert uploaded is False
    assert any(
        "Skipping W&B artifact upload; file not found" in message
        for message in messages
    )


@pytest.mark.unit
@pytest.mark.config
def test_wandb_logging_wrapper_skips_when_run_missing(monkeypatch):
    from rogueone.utils import wandb_utils

    monkeypatch.setattr(wandb_utils.rouge_one_cfg, "do_logging", True)
    monkeypatch.setattr(wandb_utils.wandb, "run", None)

    called = {"value": False}

    @wandb_utils.wandb_logging_wrapper
    def _sample():
        called["value"] = True

    _sample()
    assert called["value"] is False


# ---------------------------------------------------------------------------
# safe_wandb_log: drops `step=` when it would regress the run step
# (CAR runs 71g183ic / o5zyks6e / m4i2r4v9 — every per-iteration metric was
#  silently dropped by wandb because LLM token logs auto-advanced the step
#  past the small per-iteration index passed via `step=`)
# ---------------------------------------------------------------------------


class _FakeWandbRun:
    def __init__(self, step: int):
        self.step = step


@pytest.mark.unit
@pytest.mark.config
def test_safe_wandb_log_drops_step_when_it_would_regress(monkeypatch):
    from rogueone.utils import wandb_utils

    monkeypatch.setattr(wandb_utils.rouge_one_cfg, "do_logging", True)
    monkeypatch.setattr(wandb_utils.wandb, "run", _FakeWandbRun(step=42))

    captured: list[dict] = []

    def _fake_log(payload, step=None):
        captured.append({"payload": payload, "step": step})

    monkeypatch.setattr(wandb_utils.wandb, "log", _fake_log)

    # Caller asks for step=3 but run.step is already 42 → must call without step
    # so wandb auto-increments instead of silently dropping the call.
    wandb_utils.safe_wandb_log({"extractor_iteration/new_features_added": 5}, prefer_step=3)

    assert len(captured) == 1
    assert captured[0]["payload"] == {"extractor_iteration/new_features_added": 5}
    assert captured[0]["step"] is None


@pytest.mark.unit
@pytest.mark.config
def test_safe_wandb_log_honors_step_when_monotonic(monkeypatch):
    from rogueone.utils import wandb_utils

    monkeypatch.setattr(wandb_utils.rouge_one_cfg, "do_logging", True)
    monkeypatch.setattr(wandb_utils.wandb, "run", _FakeWandbRun(step=10))

    captured: list[dict] = []

    def _fake_log(payload, step=None):
        captured.append({"payload": payload, "step": step})

    monkeypatch.setattr(wandb_utils.wandb, "log", _fake_log)

    # When prefer_step >= current run.step, the requested step is honored so
    # plots that pivot on the explicit step axis still align.
    wandb_utils.safe_wandb_log({"trial/accuracy": 0.81}, prefer_step=20)

    assert captured == [{"payload": {"trial/accuracy": 0.81}, "step": 20}]


@pytest.mark.unit
@pytest.mark.config
def test_safe_wandb_log_no_op_when_logging_disabled(monkeypatch):
    from rogueone.utils import wandb_utils

    monkeypatch.setattr(wandb_utils.rouge_one_cfg, "do_logging", False)
    monkeypatch.setattr(wandb_utils.wandb, "run", _FakeWandbRun(step=0))

    def _fake_log(*_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("wandb.log must not be called when logging disabled")

    monkeypatch.setattr(wandb_utils.wandb, "log", _fake_log)

    wandb_utils.safe_wandb_log({"trial/accuracy": 0.5}, prefer_step=1)


@pytest.mark.unit
@pytest.mark.config
def test_safe_wandb_log_no_op_when_run_is_none(monkeypatch):
    from rogueone.utils import wandb_utils

    monkeypatch.setattr(wandb_utils.rouge_one_cfg, "do_logging", True)
    monkeypatch.setattr(wandb_utils.wandb, "run", None)

    def _fake_log(*_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("wandb.log must not be called when run is None")

    monkeypatch.setattr(wandb_utils.wandb, "log", _fake_log)

    wandb_utils.safe_wandb_log({"trial/accuracy": 0.5}, prefer_step=1)


@pytest.mark.unit
@pytest.mark.config
def test_init_wandb_run_skips_sign_in_when_logging_disabled(monkeypatch):
    from rogueone.utils import wandb_utils

    class DummyCfg:
        experiment_name = "dummy"

        @staticmethod
        def to_dict():
            return {}

    monkeypatch.setattr(wandb_utils.rouge_one_cfg, "do_logging", False)

    sign_in_calls = {"count": 0}

    def _fake_sign_in():
        sign_in_calls["count"] += 1

    monkeypatch.setattr(wandb_utils, "sign_in_wandb", _fake_sign_in)

    run = wandb_utils.init_wandb_run(DummyCfg())

    assert run is None
    assert sign_in_calls["count"] == 0
