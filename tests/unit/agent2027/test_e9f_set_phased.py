"""E9f SET-phased topology (A16-T5, T6, T7): ``ROGUE_ONE_TOPOLOGY=three_agent|set_phased``.

Guards (01 §2.2 post-freeze rule, guard 2): with the flag unset or ``three_agent`` every
role keeps its own session (Full names), system prompt and first message; no SET file
or summary key appears. Under ``set_phased``:

* T5: one ``SQLiteSession`` per iteration shared by S, every E round, T-A and T-B; a new,
  empty one at the next iteration; the phase header + role block once per phase in the
  stored history, also after an injected retry; error records carry ``phase``;
* T6: the frozen table reproduces the SET phase messages and the neutral system prompt
  byte for byte from the Full-v2.0 renderings (``fixtures/e9f``), and the forbidden-word
  grep is clean outside ``context_and_goal``;
* T7: the generating phases (S, E) never hold the Tester's labelled executor; an
  off-phase tool call and an HTTP 400 are counted for the right phase.

Agents run with a fake ``Runner`` that persists each run's input in the session at run
start, as the SDK does (openai-agents 0.9.1 ``run.py``), and the real ``SQLiteSession``.
"""

import asyncio
import importlib
import importlib.util
import json
import os
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

for _k, _v in {"EMBEDD_API_KEY": "x", "EMBEDD_MODEL": "x", "EMBEDD_PORT": "1", "LLM_MODEL": "x",
               "LLM_API_KEY": "x", "LLM_PORT": "1"}.items():
    os.environ.setdefault(_k, _v)

import numpy as np
import pandas as pd
import pytest
from agents.exceptions import ModelBehaviorError
from agents.memory.sqlite_session import SQLiteSession
from agents.tool_context import ToolContext

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests" / "unit" / "agent2027"))

import e9_scientist_harness as H  # noqa: E402
import rogueone.llm.agents.core.extractor.agent as ext_mod  # noqa: E402
import rogueone.llm.agents.core.scientist.agent as sci_mod  # noqa: E402
import rogueone.llm.agents.core.tester.agent as tst_mod  # noqa: E402
from rogueone.llm.prompts import set_phased as sp  # noqa: E402
from rogueone.utils import config_keys, console_cfg, rouge_one_cfg  # noqa: E402
from rogueone.utils import flag_registry as fr  # noqa: E402
from rogueone.utils import instrumentation as ins  # noqa: E402
from rogueone.utils import topology as topo  # noqa: E402
from rogueone.utils.config import rogueoneConfig  # noqa: E402

pytestmark = [pytest.mark.unit]

FIX = ROOT / "tests" / "unit" / "agent2027" / "fixtures" / "e9f"
STEMS = {
    "S": "scientist__focus__iter000__c01",
    "E": "extractor__append__iter000__c01",
    "T-A": "tester__tester-agent-assessment__iter000__c01",
    "T-B": "tester__tester-agent-pruning__iter000__c01",
}
#: The harness task's context_and_goal (``fixtures/e9f``; synthetic, no real task text).
HARNESS_CONTEXT = (
    "Synthetic harness task (E9f). One row per entity with the columns loan_amount_k,\n"
    "annual_income_k, credit_score and region. Goal: predict the binary target.\n"
)
HARNESS_COLUMNS = ["id", "loan_amount_k", "annual_income_k", "credit_score", "region"]
LABELLED_EXECUTOR = "generic_python_executor_tool"
FORBIDDEN = re.compile(r"(?i)you are the|team of agents|scientist|extractor|tester")
OTHER_SWITCHES_OFF = {
    "ROGUE_ONE_PLANNER_MODE": "llm", "ROGUE_ONE_MEMORY_MODE": "full",
    "ROGUE_ONE_ASSESSMENT_MODE": "llm", "ROGUE_ONE_PRUNING_MODE": "llm",
    "ROGUE_ONE_PRUNING_SHADOW": "0", "ROGUE_ONE_RESTART_MODE": "off",
    "ROGUE_ONE_FEEDBACK_LEVEL": "full", "ROGUE_ONE_KNOWLEDGE_BACKEND": "web",
}


def _read(name: str, kind: str, sub: str) -> str:
    return (FIX / sub / f"{name}.{kind}.txt").read_bytes().decode("utf-8")


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.delenv("ROGUE_ONE_TOPOLOGY", raising=False)
    monkeypatch.delenv("ROGUE_ONE_PROMPT_DUMP", raising=False)
    saved = (rouge_one_cfg.extractor_rounds, console_cfg.disable,
             getattr(rouge_one_cfg, "topology", "three_agent"))
    console_cfg.disable = True
    ins.configure(tmp_path / "out")
    yield
    rouge_one_cfg.extractor_rounds, console_cfg.disable, rouge_one_cfg.topology = saved
    ins.reset()


# =========================================================================== flag
def test_flag_default_registry_and_config(monkeypatch):
    assert fr.FLAG_DEFAULTS["ROGUE_ONE_TOPOLOGY"] == "three_agent"
    assert config_keys.OPTIONAL_ENV_DEFAULTS["ROGUE_ONE_TOPOLOGY"] == "three_agent"
    env = (ROOT / "scripts" / "agent2027" / "v2_defaults.env").read_text(encoding="utf-8")
    assert "ROGUE_ONE_TOPOLOGY=three_agent" in env.splitlines()
    assert topo.parse_topology(None) == topo.parse_topology("") == "three_agent"
    assert topo.parse_topology(" SET_Phased ") == "set_phased"
    for bad in ("set", "solo", "1", "three"):
        with pytest.raises(ValueError):
            topo.parse_topology(bad)
    c = rogueoneConfig()
    try:
        c.__init__()
        assert c.topology == "three_agent"
        monkeypatch.setenv("ROGUE_ONE_TOPOLOGY", "set_phased")
        c.__init__()
        assert c.topology == "set_phased"
        monkeypatch.setenv("ROGUE_ONE_TOPOLOGY", "bogus")
        with pytest.raises(ValueError):
            c.__init__()
    finally:
        monkeypatch.delenv("ROGUE_ONE_TOPOLOGY", raising=False)
        c.__init__()


@pytest.mark.parametrize("flag,value", [
    ("ROGUE_ONE_PLANNER_MODE", "none"), ("ROGUE_ONE_PLANNER_MODE", "oracle"),
    ("ROGUE_ONE_MEMORY_MODE", "amnesic"), ("ROGUE_ONE_ASSESSMENT_MODE", "template"),
    ("ROGUE_ONE_PRUNING_MODE", "rules"), ("ROGUE_ONE_PRUNING_SHADOW", "1"),
    ("ROGUE_ONE_RESTART_MODE", "every_iteration"), ("ROGUE_ONE_FEEDBACK_LEVEL", "l0"),
    ("ROGUE_ONE_KNOWLEDGE_BACKEND", "off"), ("ROGUE_ONE_KNOWLEDGE_BACKEND", "local"),
])
def test_set_phased_never_combines(flag, value):
    env = {**OTHER_SWITCHES_OFF, flag: value}
    topo.check_exclusive("three_agent", env)  # Full and other arms: no restriction
    with pytest.raises(ValueError, match="does not combine"):
        topo.check_exclusive("set_phased", env)


def test_set_phased_with_full_values_or_unset_is_allowed():
    topo.check_exclusive("set_phased", OTHER_SWITCHES_OFF)
    topo.check_exclusive("set_phased", {})


def test_summary_keys_only_under_set_phased(monkeypatch):
    assert topo.summary_keys({"set_offphase_e": 3}) == {}
    assert not [k for k in ins.compute_summary() if "offphase" in k or k.startswith("set_")]
    monkeypatch.setenv("ROGUE_ONE_TOPOLOGY", "set_phased")
    s = topo.summary_keys({"set_offphase_e": 3, "set_offphase_tb": 1})
    assert s["offphase_tool_calls"] == 4 and s["offphase_tool_calls_e"] == 3


# =========================================================================== T6 prompts
def test_frozen_table_has_23_rows():
    ids = [r[0] for r in sp.SUBSTITUTIONS]
    assert len(ids) == 23 and len(set(ids)) == 23


@pytest.mark.parametrize("phase", list(STEMS))
def test_frozen_rule_reproduces_set_phase_messages(phase):
    """Full-v2.0 rendering + frozen table -> the SET dump, byte for byte."""
    full_instr = _read(STEMS[phase], "system", "full_v20")
    trailing = _read(STEMS[phase], "user", "full_v20")
    expected = _read(STEMS[phase], "user", "set_phased")
    assert sp.phase_message(phase, full_instr, trailing) == expected
    assert expected.startswith(sp.phase_header(phase) + "\n\n")
    assert expected.endswith("\n\n" + trailing)


def test_system_prompt_identical_across_phases_and_rebuilt():
    texts = {_read(stem, "system", "set_phased") for stem in STEMS.values()}
    assert len(texts) == 1
    built = sp.system_prompt(HARNESS_CONTEXT, sp.raw_columns_str(HARNESS_COLUMNS))
    assert texts == {built}
    lines = built.split("\n")
    assert lines[0].startswith("The overall goal is to discover")
    assert lines[2] == sp.PHASES_LINE
    assert built.endswith(sp.OUTPUT_STYLE_LINE)


def test_raw_column_list_is_the_extractors():
    full_ext = _read(STEMS["E"], "system", "full_v20")
    assert sp.RAW_COLUMNS_HEADER in full_ext
    assert sp.raw_columns_str(HARNESS_COLUMNS) in full_ext


def test_forbidden_words_absent_outside_context():
    for stem in STEMS.values():
        for kind in ("system", "user"):
            text = _read(stem, kind, "set_phased").replace(HARNESS_CONTEXT.rstrip("\n"), "")
            assert not FORBIDDEN.findall(text), (stem, kind, FORBIDDEN.findall(text))


@pytest.mark.parametrize("phase", list(STEMS))
def test_role_block_is_verbatim_full_text_and_deleted_lines_are_gone(phase):
    full_instr = _read(STEMS[phase], "system", "full_v20")
    lines = full_instr.split("\n")
    raw = "\n\n".join(sp._segment(lines, a, b, phase) for a, b in sp.ROLE_BLOCKS[phase])
    for seg in raw.split("\n\n"):
        assert seg in full_instr  # every segment is a contiguous byte slice of Full
    block = sp.role_block(phase, full_instr)
    for gone in ("You are the Scientist Agent", "You are the Tester Agent",
                 "Your current stage is", "# Team context:", "You all work in a loop",
                 "# The Setup:", "# Role", "Global goal:", "# Task context",
                 sp.RAW_COLUMNS_HEADER):
        assert gone not in block


def test_mismatched_anchor_or_row_raises():
    with pytest.raises(sp.SetPhasedPromptError):
        sp.role_block("S", "no anchors here")
    full = _read(STEMS["E"], "system", "full_v20").replace(
        "The Tester will prune it if it is uninformative.", "")
    with pytest.raises(sp.SetPhasedPromptError, match="E2"):
        sp.role_block("E", full)


# =========================================================================== T5 / T7 plumbing
class _Recorder:
    """Fake Runner: stores the run input in the session first (SDK behaviour), records
    (role, input, instructions, tools, session), optionally raises on given calls."""

    def __init__(self, raise_on=None):
        self.calls = []
        self.raise_on = raise_on or {}

    def runner(self):
        rec = self

        class FakeRunner:  # S/E call Runner().run, the Tester calls Runner.run
            @staticmethod
            async def run(agent, inp, *args, **kwargs):
                return await rec._run(agent, inp, **kwargs)

        return FakeRunner

    async def _run(self, agent, inp, **kwargs):
        sess = kwargs.get("session")
        if sess is not None:
            await sess.add_items([{"role": "user", "content": str(inp)}])
        role = agent.name
        tools = {t.name: t for t in agent.tools}
        n = sum(1 for c in self.calls if c["agent"] == role)
        self.calls.append({"agent": role, "input": str(inp), "instructions": agent.instructions,
                           "tools": sorted(tools), "session": sess,
                           "filter": getattr(kwargs.get("run_config"), "call_model_input_filter", None)})
        exc = self.raise_on.get((role, n))
        if exc is not None:
            raise exc
        if role == "Extractor Agent":
            return SimpleNamespace(final_output="added", new_items=[])
        if "attribute_pruning_tool" in tools:
            payload = json.dumps({"attribute_names_list": ["f3"]})
            await tools["attribute_pruning_tool"].on_invoke_tool(
                ToolContext(context=None, tool_name="attribute_pruning_tool", tool_call_id="c",
                            tool_arguments=payload), payload)
            return SimpleNamespace(final_output="pruned f3", new_items=[])
        if role.startswith("Tester"):
            return SimpleNamespace(final_output="# Assessment\nfine", new_items=[])
        return SimpleNamespace(final_output=H.FIXED_FOCUS, new_items=[])


FEATURES = ["f0", "f1", "f2", "f3"]


def _folds(k=5, n=12, seed=0):
    rng = np.random.default_rng(seed)
    out, nxt = [], 0
    for _ in range(k):
        d = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
        d["target"] = rng.integers(0, 2, size=n)
        d["id"] = range(nxt, nxt + n)
        nxt += n
        out.append(d)
    return out


def _expl():
    return pd.DataFrame({"Attribute": FEATURES, "Status": ["Active"] * 4,
                         "Pruned Iteration": [pd.NA] * 4, "Prune Rationale": [pd.NA] * 4})


def _iteration(tmp_path, monkeypatch, rec, shared, index=0, rounds=3):
    """One S -> E (R rounds) -> T-A -> T-B pass with the real agents and a fake Runner."""
    rouge_one_cfg.extractor_rounds = rounds
    fake = rec.runner()
    for m in (sci_mod, ext_mod, tst_mod):
        monkeypatch.setattr(m, "Runner", fake)
    for m in (sci_mod, ext_mod, tst_mod):
        monkeypatch.setattr(m, "create_openai_client_for_vllm", lambda **_: MagicMock())
    kw = {"shared_session": shared} if shared is not None else {}
    out = Path(tmp_path) / "out"
    sci = H._new_scientist(out)
    ext = ext_mod.ExtractorAgent.__new__(ext_mod.ExtractorAgent)
    ext.cfg_experiment = SimpleNamespace(output_dir=out, experiment_description="d",
                                         context_and_goal="g", experiment_name="e",
                                         task_type="classification", modality="tabular")
    ext.knowledge_agent = SimpleNamespace()
    tst = tst_mod.TesterAgent(SimpleNamespace(
        output_dir=out, experiment_description="d", context_and_goal="g", experiment_name="e",
        task_type="classification", modality="tabular"), include_report=True, enable_wandb=False)
    df = pd.DataFrame({"id": [1, 2], "v": [1.0, 2.0]})

    async def go():
        await sci.determine_focus(df, None, pd.DataFrame(columns=["Attribute", "Status"]), [],
                                  index=index, max_index=10, iteration_in_fold=index, **kw)
        await ext.generate_patient_attributes(df, pd.DataFrame({"id": [1, 2]}),
                                              pd.DataFrame({"Attribute": [], "Status": []}),
                                              "THE FOCUS", step=index, **kw)
        await tst.test_hypotheses(_folds(), _expl(), step=0, trial_index=index, **kw)

    asyncio.run(go())


def _shared(index, tmp_path):
    return topo.SharedIterationSession(
        index, sp.system_prompt("CTX", sp.raw_columns_str(["id", "v"])),
        output_dir=Path(tmp_path) / "out")


def _items(session):
    return asyncio.run(session.get_items())


def test_three_agent_path_unchanged(tmp_path, monkeypatch):
    """Guard 2: no shared_session -> own Full sessions, Full instructions, trailing inputs."""
    rec = _Recorder()
    _iteration(tmp_path, monkeypatch, rec, None)
    names = [(c["agent"], c["session"].session_id) for c in rec.calls]
    assert names == [("Researcher Agent", "scientist_agent_session_step_0")] + [
        ("Extractor Agent", "extractor_agent_session_step_0")] * 3 + [
        ("Tester Agent (Assessment)", "tester_agent_assessment_step_0"),
        ("Tester Agent (Pruning)", "tester_agent_pruning_step_0")]
    assert len({id(c["session"]) for c in rec.calls}) == 4  # S, E, T-A, T-B each their own
    assert rec.calls[0]["instructions"].lstrip().startswith("The setting:")
    assert "You are the Tester Agent. Your current stage is PRUNING ONLY." in rec.calls[-1]["instructions"]
    assert not any(c["input"].startswith("### PHASE") for c in rec.calls)
    assert [c["input"] for c in rec.calls[1:4]] == ["System now wants to focus on: THE FOCUS."] * 3
    assert rec.calls[-1]["filter"].__name__ == "_strip_reasoning"  # the Tester's own filter
    assert not (Path(tmp_path) / "out" / topo.LOG_FILE).exists()
    errs = Path(tmp_path) / "out" / "agent_toolcall_errors.jsonl"
    assert not errs.exists() or "phase" not in errs.read_text(encoding="utf-8")


def test_one_shared_session_per_iteration_header_once_after_retry(tmp_path, monkeypatch):
    rec = _Recorder(raise_on={
        ("Researcher Agent", 0): RuntimeError("transient"),  # S retry
        ("Tester Agent (Assessment)", 0): RuntimeError("transient"),  # T-A retry
    })
    s0 = _shared(0, tmp_path)
    assert _items(s0.session) == []  # new object starts empty
    _iteration(tmp_path, monkeypatch, rec, s0)
    # (a) the same session object for S, every E round, T-A and T-B
    assert {id(c["session"]) for c in rec.calls} == {id(s0.session)}
    assert s0.session.session_id == "set_phased_trial_0"
    assert [c["agent"] for c in rec.calls] == ["Researcher Agent"] * 2 + ["Extractor Agent"] * 3 + [
        "Tester Agent (Assessment)"] * 2 + ["Tester Agent (Pruning)"]
    # one neutral system prompt for every phase and attempt
    assert {c["instructions"] for c in rec.calls} == {s0.system_prompt}
    # (b) header + role block exactly once per phase in the stored history
    stored = [it["content"] for it in _items(s0.session)]
    assert len(stored) == 8
    for phase in topo.PHASES:
        assert sum(t.startswith(sp.phase_header(phase)) for t in stored) == 1
    # retries / rounds 2..R re-send the trailing input only (what Full re-sends)
    assert stored[1] == rec.calls[1]["input"] and stored[1].startswith("Round 1:")
    assert stored[3] == stored[4] == "System now wants to focus on: THE FOCUS."
    assert stored[2].endswith("\n\nSystem now wants to focus on: THE FOCUS.")
    assert stored[6].startswith("Assess the current feature set")
    # every phase uses the S/E input filter
    assert {c["filter"].__name__ for c in rec.calls} == {"_strip_reasoning", "strip_reasoning_and_special_tokens"}
    assert rec.calls[-1]["filter"] is topo.strip_reasoning_and_special_tokens
    # next iteration: a new, empty object
    s1 = _shared(1, tmp_path)
    assert s1.session is not s0.session and _items(s1.session) == []
    assert s1.session.session_id == "set_phased_trial_1"
    row = s0.close(0, 0)
    assert row["phases"]["S"]["messages"] == 2 and row["phases"]["E"]["messages"] == 3
    assert row["phases"]["T-A"]["messages"] == 2 and row["phases"]["T-B"]["messages"] == 1
    logged = (Path(tmp_path) / "out" / topo.LOG_FILE).read_text(encoding="utf-8").splitlines()
    assert json.loads(logged[0])["session_name"] == "set_phased_trial_0"


def test_generating_phases_never_hold_labelled_executor(tmp_path, monkeypatch):
    """A16-T7 leakage guard: S and E never get the Tester's labelled executor, and the
    history S and E see holds no Tester content (T runs last; new session per iteration)."""
    rec = _Recorder()
    s0 = _shared(0, tmp_path)
    seen_by = {}

    async def snap(role):
        seen_by[role] = [it["content"] for it in await s0.session.get_items()]

    orig = rec._run

    async def spy(agent, inp, **kw):
        if agent.name not in seen_by:
            await snap(agent.name)
        return await orig(agent, inp, **kw)

    rec._run = spy
    _iteration(tmp_path, monkeypatch, rec, s0)
    by_role = {}
    for c in rec.calls:
        by_role.setdefault(c["agent"], set()).update(c["tools"])
    assert LABELLED_EXECUTOR not in by_role["Researcher Agent"]
    assert LABELLED_EXECUTOR not in by_role["Extractor Agent"]
    assert LABELLED_EXECUTOR in by_role["Tester Agent (Assessment)"]
    assert LABELLED_EXECUTOR in by_role["Tester Agent (Pruning)"]
    # the generating phases' tool lists are exactly Full's (per-phase tool lists kept)
    rec_full = _Recorder()
    _iteration(tmp_path / "full", monkeypatch, rec_full, None)
    full_by_role = {}
    for c in rec_full.calls:
        full_by_role.setdefault(c["agent"], set()).update(c["tools"])
    assert by_role == full_by_role
    # S saw an empty history; E saw only the PLAN phase; neither saw ASSESS/PRUNE content
    assert seen_by["Researcher Agent"] == []
    assert all(not t.startswith(sp.phase_header("T-A")) and not t.startswith(sp.phase_header("T-B"))
               for t in seen_by["Extractor Agent"])
    assert seen_by["Extractor Agent"][0].startswith(sp.phase_header("S"))
    # a new iteration's session holds nothing from the previous one (incl. Tester outputs)
    assert _items(_shared(1, tmp_path).session) == []


def test_offphase_call_and_context_error_are_counted_per_phase(tmp_path, monkeypatch):
    monkeypatch.setenv("ROGUE_ONE_TOPOLOGY", "set_phased")
    rec = _Recorder(raise_on={
        ("Extractor Agent", 0): ModelBehaviorError(
            f"Tool {LABELLED_EXECUTOR} not found in agent Extractor Agent"),
        ("Tester Agent (Pruning)", 0): ModelBehaviorError(
            "Tool append_new_attribute not found in agent Tester Agent (Pruning)"),
    })
    s0 = _shared(0, tmp_path)
    _iteration(tmp_path, monkeypatch, rec, s0)
    assert s0.offphase == {"S": 0, "E": 1, "T-A": 0, "T-B": 1}
    # an HTTP 400 during the E phase -> context_errors with phase E
    ins.set_phase("E")
    exc = RuntimeError("prompt too long")
    exc.status_code = 400
    ins.end_call(ins.begin_call("ExtractorAgent"), None, exc)
    ins.set_phase(None)
    s = ins.compute_summary()
    assert s["offphase_tool_calls"] == 2
    assert s["offphase_tool_calls_e"] == 1 and s["offphase_tool_calls_tb"] == 1
    assert s["context_errors_e"] == 1 and s["context_errors_http400"] == 1
    assert s["context_errors_s"] == 0
    row = s0.close(0, 0)
    assert row["phases"]["E"]["context_errors"] == 1
    # the fake E appends nothing -> deterministic fallback + a context error = S2 exhausted
    assert row["phases"]["E"]["fallbacks"] == ["deterministic"]
    assert row["phases"]["E"]["exhausted"] is True
    assert row["phases"]["T-B"]["exhausted"] is False  # off-phase error is not a context error
    # error records carry the phase and the shared session name
    recs = [json.loads(l) for l in (Path(tmp_path) / "out" / "agent_toolcall_errors.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    runner = [r for r in recs if r["tool_name"] == "runner"]
    assert {(r["context"]["phase"], r["context"]["session_name"]) for r in runner} == {
        ("E", "set_phased_trial_0"), ("T-B", "set_phased_trial_0")}


def test_fallback_with_context_error_is_s2_exhaustion(tmp_path, monkeypatch):
    rec = _Recorder()
    rouge_one_cfg.extractor_rounds = 1
    s0 = _shared(0, tmp_path)
    ins.set_phase("T-B")
    ins.end_call(ins.begin_call("TesterAgent"), SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="length", message=None)], usage=None, model="m"), None)
    s0.note_fallback("T-B", "no_pruning_summary")
    row = s0.close(0, 0)
    assert row["phases"]["T-B"]["exhausted"] is True
    assert ins.get_counter("set_phase_exhausted") == 1
    assert rec.calls == []


# =========================================================================== main
_stubbed = importlib.util.find_spec("git") is None
if _stubbed:
    sys.modules["git"] = types.SimpleNamespace(Tree=object)
try:
    rom = importlib.import_module("rogue_one_main")
finally:
    if _stubbed:
        sys.modules.pop("git", None)
        sys.modules.pop("rogue_one_main", None)

COLS = ["Attribute", "Description", "Pandas Command", "Status", "Added", "Pruned Iteration", "Prune Rationale"]


def _drive(tmp_path, index):
    net = object.__new__(rom.RogueOneAgentNetwork)
    net.cfg = SimpleNamespace(output_dir=Path(tmp_path) / "out", experiment_name="e",
                              task_type="classification", context_and_goal="CTX")
    net.enable_wandb = False
    net.df = pd.DataFrame({"id": [1, 2], "v": [1.0, 2.0]})
    net.df_entities_attributes = pd.DataFrame({"id": [1, 2]})
    net.df_attributes_explanations = pd.DataFrame(columns=COLS)
    net.df_scientist_feedback_pool = pd.DataFrame()
    net.focus_history = []
    net.df_test_pool = None
    net._scored_before_test = None
    seen = []

    async def determine_focus(*a, **k):
        seen.append(("S", k.get("shared_session", "absent")))
        return "focus"

    async def generate(*a, **k):
        seen.append(("E", k.get("shared_session", "absent")))
        return net.df_entities_attributes, pd.DataFrame()  # -> iteration_skipped

    net.scientist_agent = SimpleNamespace(determine_focus=determine_focus, _note_book="")
    net.extractor_agent = SimpleNamespace(generate_patient_attributes=generate)
    status = asyncio.run(net.forward_pass(index))
    return status, seen


def test_main_three_agent_passes_no_shared_session(tmp_path):
    rouge_one_cfg.topology = "three_agent"
    status, seen = _drive(tmp_path, 1)
    assert status == "iteration_skipped"
    assert seen == [("S", "absent"), ("E", "absent")]
    assert not (Path(tmp_path) / "out" / topo.LOG_FILE).exists()


def test_main_set_phased_shares_one_session_and_closes_it(tmp_path):
    rouge_one_cfg.topology = "set_phased"
    status, seen = _drive(tmp_path, 1)
    assert status == "iteration_skipped"
    (s_role, s_obj), (e_role, e_obj) = seen
    assert s_obj is e_obj and isinstance(s_obj, topo.SharedIterationSession)
    assert s_obj.session_name == "set_phased_trial_1" and s_obj.closed
    assert s_obj.system_prompt == sp.system_prompt("CTX", "`id`, `v`")
    (row,) = [json.loads(l) for l in (Path(tmp_path) / "out" / topo.LOG_FILE)
              .read_text(encoding="utf-8").splitlines()]
    assert row["index"] == 1 and set(row["phases"]) == set(topo.PHASES)
    _, seen2 = _drive(tmp_path, 2)
    assert seen2[0][1] is not s_obj and seen2[0][1].session is not s_obj.session
