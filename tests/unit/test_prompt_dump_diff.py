"""Unit tests for AGENT 2027 E5: rendered-prompt dump, diff tool, neutralisation."""
from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Both modules are dependency-free; load by path so the tests run without the
# heavy rogueone.utils package import (openai, wandb, ...).
prompt_dump = _load("rogueone_prompt_dump", REPO / "rogueone" / "utils" / "prompt_dump.py")
prompt_diff = _load("prompt_diff", REPO / "scripts" / "agent2027" / "prompt_diff.py")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(prompt_dump.DIR_ENV, raising=False)
    monkeypatch.delenv(prompt_dump.FLAG, raising=False)
    prompt_dump.reset_counters()
    yield
    prompt_dump.reset_counters()


def _render_run(out_dir: Path, system_suffix: str = "") -> None:
    prompt_dump.reset_counters()
    tool = SimpleNamespace(name="t1", description="desc", params_json_schema={"type": "object"})
    prompt_dump.dump_prompt("scientist", "focus", 3, "SYS A" + system_suffix, "USER A", [tool], out_dir)
    prompt_dump.dump_prompt("extractor", "append", 3, "SYS B", "USER B", [tool], out_dir)
    prompt_dump.dump_prompt("tester", "Tester Agent (Pruning)", 3, "SYS C", "USER C", [], out_dir)
    prompt_dump.dump_prompt("websearch", "search", 0, "SYS D", "q", [], out_dir, once=True)
    prompt_dump.dump_prompt("websearch", "search", 0, "SYS D", "q2", [], out_dir, once=True)


# ---------------------------------------------------------------- dump

def test_dump_disabled_by_default(tmp_path):
    assert prompt_dump.dump_prompt("scientist", "focus", 0, "s", "u", None, tmp_path) is None
    assert not (tmp_path / "prompts").exists()


def test_dump_writes_expected_files(tmp_path, monkeypatch):
    monkeypatch.setenv(prompt_dump.FLAG, "1")
    _render_run(tmp_path)
    d = tmp_path / "prompts"
    names = sorted(p.name for p in d.iterdir())
    assert "scientist__focus__iter003__c01.system.txt" in names
    assert "scientist__focus__iter003__c01.user.txt" in names
    assert "scientist__focus__iter003__c01.tools.json" in names
    assert "extractor__append__iter003__c01.system.txt" in names
    assert "tester__tester-agent-pruning__iter003__c01.user.txt" in names
    # once=True: only the first web-search call is written
    assert [n for n in names if n.startswith("websearch") and n.endswith(".user.txt")] == [
        "websearch__search__iter000__c01.user.txt"
    ]
    assert (d / "scientist__focus__iter003__c01.system.txt").read_bytes() == b"SYS A"
    assert '"name": "t1"' in (d / "scientist__focus__iter003__c01.tools.json").read_text()


def test_dump_repeat_calls_get_sequence_numbers(tmp_path, monkeypatch):
    monkeypatch.setenv(prompt_dump.FLAG, "1")
    for _ in range(2):
        prompt_dump.dump_prompt("extractor", "append", 1, "s", "u", None, tmp_path)
    names = {p.name for p in (tmp_path / "prompts").iterdir()}
    assert "extractor__append__iter001__c02.system.txt" in names


def test_dump_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv(prompt_dump.FLAG, "1")
    monkeypatch.setenv(prompt_dump.DIR_ENV, str(tmp_path / "custom"))
    prompt_dump.dump_prompt("scientist", "focus", 0, "s", "u", None, tmp_path / "ignored")
    assert (tmp_path / "custom" / "scientist__focus__iter000__c01.system.txt").exists()


def test_dump_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setenv(prompt_dump.FLAG, "1")
    _render_run(tmp_path / "a")
    _render_run(tmp_path / "b")
    assert prompt_diff.load_dump(tmp_path / "a") == prompt_diff.load_dump(tmp_path / "b")


# ---------------------------------------------------------------- diff

def test_diff_identical_reports_zero(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(prompt_dump.FLAG, "1")
    _render_run(tmp_path / "a")
    _render_run(tmp_path / "b")
    rc = prompt_diff.main([str(tmp_path / "a"), str(tmp_path / "b")])
    assert rc == 0
    assert "IDENTICAL" in capsys.readouterr().out


def test_diff_accepts_parent_of_prompts_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(prompt_dump.FLAG, "1")
    _render_run(tmp_path / "a")
    assert prompt_diff.main([str(tmp_path / "a"), str(tmp_path / "a" / "prompts")]) == 0


def test_diff_flags_changed_token(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(prompt_dump.FLAG, "1")
    _render_run(tmp_path / "a")
    _render_run(tmp_path / "b", system_suffix=" x")
    rc = prompt_diff.main([str(tmp_path / "a"), str(tmp_path / "b")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "CHANGED: role=scientist phase=focus iter=3" in out
    assert "DIFFERENT" in out


def test_diff_flags_missing_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(prompt_dump.FLAG, "1")
    _render_run(tmp_path / "a")
    _render_run(tmp_path / "b")
    (tmp_path / "b" / "prompts" / "extractor__append__iter003__c01.user.txt").unlink()
    assert prompt_diff.main([str(tmp_path / "a"), str(tmp_path / "b")]) == 1
    assert "ONLY IN A" in capsys.readouterr().out


def test_scan_finds_and_allows_terms(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(prompt_dump.FLAG, "1")
    prompt_dump.dump_prompt("scientist", "focus", 0, "task about CAR and torque", "u", None, tmp_path)
    assert prompt_diff.main(["--scan", str(tmp_path)]) == 1
    allow = tmp_path / "ctx.txt"
    allow.write_text("task about CAR and torque", encoding="utf-8")
    assert prompt_diff.main(["--scan", str(tmp_path), "--allow-text", str(allow)]) == 0
    capsys.readouterr()


# ------------------------------------------------- neutralisation (static)

FORBIDDEN = {
    "torque": re.compile(r"\btorque\b", re.I),
    "CAR": re.compile(r"\bCAR\b"),
    "buying": re.compile(r"\bbuying\b", re.I),
    "maint": re.compile(r"\bmaint\b", re.I),
    "safety": re.compile(r"\bsafety\b", re.I),
    "winner": re.compile(r"\bwinner\b", re.I),
    "minimax": re.compile(r"\bminimax\b", re.I),
    "unacc/vhigh": re.compile(r"\bunacc\b|\bvhigh\b"),
    "tic-tac-toe squares": re.compile(r"(top|middle|bottom)-(left|middle|right)-square"),
    "balance scale": re.compile(r"balance scale", re.I),
    "rF sin": re.compile(r"rF sin"),
    "run id": re.compile(r"wi5r0oki"),
}
# helper docstrings that are never sent to a model
NOT_RENDERED = {"_ordinal_encode_categorical"}


def _prompt_strings(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in NOT_RENDERED:
            if node.body and isinstance(node.body[0], ast.Expr):
                skip.add(id(node.body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            yield node.lineno, node.value


@pytest.mark.parametrize(
    "rel",
    [
        "rogueone/llm/agents/core/scientist/agent.py",
        "rogueone/llm/agents/core/extractor/agent.py",
        "rogueone/llm/agents/core/tester/agent.py",
        "rogueone/llm/agents/core/knowledge/web_search.py",
        "rogueone/llm/agents/core/knowledge/agent.py",
        "rogueone/llm/prompts/agent_roles.py",
    ],
)
def test_no_dataset_specific_strings_in_prompts(rel):
    offenders = [
        f"{rel}:{lineno}: {name}"
        for lineno, text in _prompt_strings(REPO / rel)
        for name, pat in FORBIDDEN.items()
        if pat.search(text)
    ]
    assert not offenders, "\n".join(offenders)
