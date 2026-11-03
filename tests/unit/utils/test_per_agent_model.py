"""Unit tests for per-agent model/endpoint overrides (checklist item T-1.4).

`REBUTTAL_FINAL.md` (reply to baGY, Q3) states that per-agent model scale was not
configurable — one model setting served all three agents. These tests pin the
behaviour of the knobs that make it configurable:

* each core agent picks up its own ``ROGUE_ONE_<ROLE>_MODEL`` override,
* an unset or empty override falls back to ``LLM_MODEL``,
* the same holds for ``ROGUE_ONE_<ROLE>_PORT`` against ``LLM_PORT``,
* an unknown role raises rather than silently returning the global model.

``LLMConfig`` is a singleton that reads the environment once in ``__init__``, so
each test constructs a fresh instance by evicting the cached one first.
"""

import pytest

from rogueone.utils.config import AGENT_ROLES, LLMConfig
from rogueone.utils.singelton import Singleton


def _fresh_config(monkeypatch, **env) -> LLMConfig:
    """Build an LLMConfig that sees exactly the env in ``env`` for the LLM keys."""
    monkeypatch.setenv("LLM_MODEL", env.pop("LLM_MODEL", "gpt-oss-120b"))
    monkeypatch.setenv("LLM_PORT", env.pop("LLM_PORT", "8010"))
    for role in AGENT_ROLES:
        for suffix in ("MODEL", "PORT"):
            monkeypatch.delenv(f"ROGUE_ONE_{role.upper()}_{suffix}", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    Singleton._instances.pop(LLMConfig, None)
    cfg = LLMConfig()
    # Don't leave the test's instance cached for the next test / for imports that
    # already hold `llm_cfg`.
    Singleton._instances.pop(LLMConfig, None)
    return cfg


@pytest.mark.unit
@pytest.mark.config
@pytest.mark.parametrize("role", AGENT_ROLES)
def test_role_model_override_is_picked_up(monkeypatch, role):
    """Each agent reads its own override and leaves the other two alone."""
    cfg = _fresh_config(
        monkeypatch,
        LLM_MODEL="gpt-oss-120b",
        **{f"ROGUE_ONE_{role.upper()}_MODEL": "gpt-oss-20b"},
    )

    assert cfg.model_for(role) == "gpt-oss-20b"
    for other in AGENT_ROLES:
        if other != role:
            assert cfg.model_for(other) == "gpt-oss-120b"


@pytest.mark.unit
@pytest.mark.config
def test_named_properties_match_model_for(monkeypatch):
    """The convenience properties the agents use resolve to the same thing."""
    cfg = _fresh_config(
        monkeypatch,
        LLM_MODEL="gpt-oss-120b",
        ROGUE_ONE_SCIENTIST_MODEL="scientist-model",
        ROGUE_ONE_EXTRACTOR_MODEL="extractor-model",
        ROGUE_ONE_TESTER_MODEL="tester-model",
    )

    assert cfg.scientist_model == "scientist-model"
    assert cfg.extractor_model == "extractor-model"
    assert cfg.tester_model == "tester-model"
    # The global stays untouched, so the retrieval helpers keep using it.
    assert cfg.model == "gpt-oss-120b"


@pytest.mark.unit
@pytest.mark.config
def test_unset_override_falls_back_to_llm_model(monkeypatch):
    """No override anywhere reproduces the pre-T-1.4 behaviour exactly."""
    cfg = _fresh_config(monkeypatch, LLM_MODEL="gpt-oss-120b")

    for role in AGENT_ROLES:
        assert cfg.model_for(role) == "gpt-oss-120b"


@pytest.mark.unit
@pytest.mark.config
def test_empty_override_falls_back_to_llm_model(monkeypatch):
    """An empty/whitespace value means "unset", not "use the empty model name"."""
    cfg = _fresh_config(
        monkeypatch,
        LLM_MODEL="gpt-oss-120b",
        ROGUE_ONE_EXTRACTOR_MODEL="   ",
        ROGUE_ONE_TESTER_MODEL="",
    )

    assert cfg.extractor_model == "gpt-oss-120b"
    assert cfg.tester_model == "gpt-oss-120b"


@pytest.mark.unit
@pytest.mark.config
def test_role_port_override_and_fallback(monkeypatch):
    """A role pointed at a second vLLM server gets its own endpoint URL.

    A model override without a port override is unusable in practice — one vLLM
    server serves one model — so the port knob is part of the same feature.
    """
    cfg = _fresh_config(
        monkeypatch,
        LLM_PORT="8010",
        ROGUE_ONE_EXTRACTOR_PORT="8011",
    )

    assert cfg.endpoint_for("extractor") == "http://localhost:8011/v1"
    assert cfg.endpoint_for("scientist") == "http://localhost:8010/v1"
    assert cfg.endpoint_for("tester") == "http://localhost:8010/v1"
    assert cfg.endpoint == "http://localhost:8010/v1"


@pytest.mark.unit
@pytest.mark.config
def test_unknown_role_raises(monkeypatch):
    """A typo'd role must fail loudly, not look like a working override."""
    cfg = _fresh_config(monkeypatch)

    with pytest.raises(KeyError):
        cfg.model_for("orchestrator")
    with pytest.raises(KeyError):
        cfg.endpoint_for("orchestrator")


@pytest.mark.unit
@pytest.mark.config
def test_role_lookup_is_case_insensitive(monkeypatch):
    cfg = _fresh_config(monkeypatch, ROGUE_ONE_TESTER_MODEL="gpt-oss-20b")

    assert cfg.model_for("TESTER") == "gpt-oss-20b"
    assert cfg.model_for("Tester") == "gpt-oss-20b"
