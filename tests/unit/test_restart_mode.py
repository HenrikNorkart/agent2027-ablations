"""E9a: ROGUE_ONE_RESTART_MODE=off|every_iteration (AGENT 2027 A10-T6).

The real ``forward_pass`` / ``_forward_pass`` / ``_reset_fold_state`` run on a stub
network; the Scientist is the real ``ScientistAgent.determine_focus`` with ``Runner``
replaced (``tests/unit/agent2027/e9_scientist_harness.py``), the Extractor is a stub
that adds one feature, and the iteration stops at the data split (``skipped``), so every
iteration still appends one ``df_test_pool`` row.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

for _k, _v in {"EMBEDD_API_KEY": "x", "EMBEDD_MODEL": "x", "EMBEDD_PORT": "1", "LLM_MODEL": "x",
               "LLM_API_KEY": "x", "LLM_PORT": "1"}.items():
    os.environ.setdefault(_k, _v)

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "unit" / "agent2027"))

import e9_scientist_harness as H  # noqa: E402
from rogueone.llm import clients  # noqa: E402
from rogueone.utils import flag_registry as fr  # noqa: E402
from rogueone.utils import loop_switches as ls  # noqa: E402
from rogueone.utils import rouge_one_cfg  # noqa: E402
from rogueone.utils.config import rogueoneConfig  # noqa: E402

pytestmark = [pytest.mark.unit]

N_ITER = 10
LLM_SEED = 42


@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.setattr(rouge_one_cfg, "num_iterations", N_ITER)
    monkeypatch.setattr(rouge_one_cfg, "k_folds", 5)
    monkeypatch.setattr(rouge_one_cfg, "llm_seed", LLM_SEED)
    monkeypatch.setattr(rouge_one_cfg, "feature_pool_mode", "replace")
    monkeypatch.setattr(rouge_one_cfg, "feedback_level", "full")
    monkeypatch.setattr(rouge_one_cfg, "restart_mode", "off")
    return rouge_one_cfg


def _network(tmp_path, monkeypatch):
    """Stub network whose loop methods are the real ones; returns (net, calls)."""
    from rogue_one_main import RogueOneAgentNetwork
    import rogueone.llm.agents.core.scientist.agent as sci_mod

    monkeypatch.chdir(tmp_path)  # E3 logs default to ./output
    net = MagicMock(spec=RogueOneAgentNetwork)
    for name in ("forward_pass", "_forward_pass"):
        setattr(net, name, getattr(RogueOneAgentNetwork, name).__get__(net))
    resets: list[int] = []
    real_reset = RogueOneAgentNetwork._reset_fold_state.__get__(net)

    def _reset():
        resets.append(net._current_index)
        real_reset()

    net._reset_fold_state = _reset
    net.df = H.make_raw_df()
    net.cfg = SimpleNamespace(experiment_name="harness", output_dir=tmp_path,
                              task_type="classification")
    net.df_test_pool = None
    net._tabular_skip_reason = "e9a unit test: stop after the extractor"
    net._log_iteration = MagicMock()
    net._log_extractor_iteration_activity = MagicMock()
    net.update_attribute_pool = AsyncMock()
    net.dump_data = AsyncMock()
    net.save_report = AsyncMock()
    net.scientist_agent = H._new_scientist(tmp_path)
    net._current_index = -1

    calls: list[dict] = []

    class FakeRunner:
        async def run(self, agent, prompt, **kwargs):
            sent = {"model": "x", "messages": []}
            clients._merge_extra_body_for_vllm(sent)
            calls.append({
                "index": net._current_index,
                "instructions": agent.instructions,
                "prompt": prompt,
                "seed": sent.get("seed"),
                "feedback_rows": len(net.df_scientist_feedback_pool),
                "attributes": net.df_attributes_explanations.copy(),
                "entity_columns": list(net.df_entities_attributes.columns),
                "focus_history": list(net.focus_history),
                "note_book": net.scientist_agent._note_book,
            })
            # The Scientist leaves a note, as Full's Scientist does.
            tools = {t.name: t for t in agent.tools}
            from agents.tool_context import ToolContext

            payload = json.dumps({"note": f"note at {net._current_index}"})
            await tools["take_a_note"].on_invoke_tool(
                ToolContext(context=None, tool_name="take_a_note", tool_call_id="c",
                            tool_arguments=payload), payload)
            return SimpleNamespace(final_output=H.FIXED_FOCUS)

    import rogue_one_main

    # No progress bar is started outside run(); the update would assert.
    monkeypatch.setattr(rogue_one_main.ConsoleManager, "progress_bar_update",
                        staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(sci_mod, "Runner", FakeRunner)
    monkeypatch.setattr(sci_mod, "create_openai_client_for_vllm", lambda **_: MagicMock())

    class FakeExtractor:
        async def generate_patient_attributes(self, df, df_entities, df_expl, focus, step):
            name = f"feat_{step}"
            ents = df_entities.copy()
            ents[name] = df["a"].values
            row = pd.DataFrame([{
                "Attribute": name, "Description": "stub", "Pandas Command": "df['a']",
                "Status": "Active", "Added": f"Trial {step}", "Pruned Iteration": None,
                "Prune Rationale": None,
            }])
            # What update_attribute_pool and the feedback row would carry forward.
            net.df_attributes_explanations = pd.concat(
                [net.df_attributes_explanations, row], ignore_index=True)
            net.df_scientist_feedback_pool = pd.concat(
                [net.df_scientist_feedback_pool,
                 pd.DataFrame([{"id": f"Trial {step}", "feature_assessment": "r"}])],
                ignore_index=True)
            return ents, row

    net.extractor_agent = FakeExtractor()
    return net, calls, resets


def _run(net, indices):
    import asyncio

    async def _go():
        out = []
        for i in indices:
            net._current_index = i
            out.append(await net.forward_pass(i))
        return out

    return asyncio.run(_go())


# --------------------------------------------------------------------------- flag
def test_flag_default_and_validation(monkeypatch):
    assert fr.FLAG_DEFAULTS["ROGUE_ONE_RESTART_MODE"] == "off"
    assert ls.parse_restart_mode(None) == "off"
    assert ls.parse_restart_mode("") == "off"
    assert ls.parse_restart_mode(" Every_Iteration ") == "every_iteration"
    for bad in ("on", "every", "1", "restart"):
        with pytest.raises(ValueError):
            ls.parse_restart_mode(bad)
    c = rogueoneConfig()
    monkeypatch.delenv("ROGUE_ONE_RESTART_MODE", raising=False)
    c.__init__()
    assert c.restart_mode == "off"
    monkeypatch.setenv("ROGUE_ONE_RESTART_MODE", "every_iteration")
    c.__init__()
    assert c.restart_mode == "every_iteration"
    monkeypatch.setenv("ROGUE_ONE_RESTART_MODE", "bogus")
    with pytest.raises(ValueError):
        c.__init__()
    monkeypatch.delenv("ROGUE_ONE_RESTART_MODE", raising=False)
    c.__init__()


def test_v2_defaults_keeps_switch_off():
    env = (ROOT / "scripts" / "agent2027" / "v2_defaults.env").read_text(encoding="utf-8")
    assert "ROGUE_ONE_RESTART_MODE=off" in env.splitlines()


# --------------------------------------------------------------------------- every_iteration
def test_every_iteration_resets_state_and_keeps_test_pool(cfg, tmp_path, monkeypatch):
    cfg.restart_mode = "every_iteration"
    net, calls, resets = _network(tmp_path, monkeypatch)
    statuses = _run(net, range(15))

    assert statuses == ["skipped"] * 15
    assert resets == list(range(15))  # before every iteration
    # (1) df_test_pool gains one row per iteration and survives every reset.
    assert len(net.df_test_pool) == 15
    assert list(net.df_test_pool["id"]) == [f"Trial {i}" for i in range(15)]
    # (2) attribute table, feedback pool, focus history and notebook empty at each start.
    for c in calls:
        assert c["feedback_rows"] == 0
        assert len(c["attributes"]) == 0
        assert c["entity_columns"] == ["id"]
        assert c["focus_history"] == []
        assert c["note_book"] == ""
    # (4) the iteration-5 Scientist instructions and first user message equal iteration 0's.
    assert calls[5]["instructions"] == calls[0]["instructions"]
    assert calls[5]["prompt"] == calls[0]["prompt"]
    assert calls[5]["prompt"].startswith("Round 1: ")
    assert calls[12]["instructions"] == calls[0]["instructions"]
    # (5) restart seed rule: llm_seed + global index (fold f, restart k -> llm_seed + 10f + k).
    assert [c["seed"] for c in calls] == [LLM_SEED + i for i in range(15)]
    assert clients.current_llm_seed_offset() == 0  # reset after every iteration


def test_every_iteration_reseeds_augment_rows(cfg, tmp_path, monkeypatch):
    from rogueone.utils.fold_safe_features import SEEDED_COLUMN, seed_raw_feature_rows

    cfg.restart_mode = "every_iteration"
    cfg.feature_pool_mode = "augment"
    net, calls, _ = _network(tmp_path, monkeypatch)
    _run(net, range(7))
    raw = H.make_raw_df()
    _, seeded = seed_raw_feature_rows(
        raw, pd.DataFrame({"id": raw["id"]}), calls[0]["attributes"].iloc[0:0])
    raw_cols = list(seeded["Attribute"])
    assert len(raw_cols) == 4  # a, b, c and the ordinal-encoded cat
    for c in calls:
        att = c["attributes"]
        # (3) every restart starts from the seeded raw pool, and only from it.
        assert list(att["Attribute"]) == raw_cols
        assert att[SEEDED_COLUMN].astype(bool).all()
        assert c["entity_columns"] == ["id"] + raw_cols
    assert calls[5]["instructions"] == calls[0]["instructions"]
    assert calls[5]["prompt"] == calls[0]["prompt"]


# --------------------------------------------------------------------------- off (guard 2)
@pytest.mark.parametrize("mode_env", [None, "off"])
def test_off_resets_only_at_fold_start_and_keeps_seed(cfg, tmp_path, monkeypatch, mode_env):
    if mode_env is None:
        monkeypatch.delenv("ROGUE_ONE_RESTART_MODE", raising=False)
    else:
        monkeypatch.setenv("ROGUE_ONE_RESTART_MODE", mode_env)
    cfg.restart_mode = ls.parse_restart_mode(os.environ.get("ROGUE_ONE_RESTART_MODE"))
    net, calls, resets = _network(tmp_path, monkeypatch)
    _run(net, range(15))

    # (6) the reset runs only at iteration_in_fold == 0.
    assert resets == [0, 10]
    assert len(net.df_test_pool) == 15
    # State carries within the fold, exactly as Full-v2.
    assert [c["feedback_rows"] for c in calls] == list(range(10)) + list(range(5))
    assert [len(c["focus_history"]) for c in calls] == list(range(10)) + list(range(5))
    assert calls[5]["prompt"].startswith("Round 6/10: ")
    assert calls[5]["note_book"].startswith("Trail 0:\n")
    # (5) off and unset: every request carries llm_seed itself.
    assert [c["seed"] for c in calls] == [LLM_SEED] * 15


def test_seed_offset_is_inert_without_llm_seed(cfg):
    cfg.llm_seed = None
    tok = clients.set_llm_seed_offset(7)
    try:
        sent = {}
        clients._merge_extra_body_for_vllm(sent)
        assert "seed" not in sent
        cfg.llm_seed = 42
        sent = {"seed": 3}
        clients._merge_extra_body_for_vllm(sent)
        assert sent["seed"] == 3  # an explicit caller seed still wins
    finally:
        clients.reset_llm_seed_offset(tok)


# --------------------------------------------------------------------------- sessions
def test_no_sqlite_session_has_a_db_path():
    """(8) Every SQLiteSession is in-memory: one positional argument, no db_path."""
    found = 0
    for path in (ROOT / "rogueone").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "SQLiteSession":
                found += 1
                assert len(node.args) == 1 and not node.keywords, f"{path}:{node.lineno}"
    assert found >= 4
