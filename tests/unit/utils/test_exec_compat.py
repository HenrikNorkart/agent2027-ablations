"""ROGUE_ONE_EXEC_COMPAT: lenient exec of LLM-authored code (A17 Q4 smoke)."""
import pandas as pd
import pytest

from rogueone.utils import pandas_exec, rouge_one_cfg


@pytest.fixture
def compat(monkeypatch):
    monkeypatch.setattr(rouge_one_cfg, "exec_compat", True)


@pytest.fixture
def default(monkeypatch):
    monkeypatch.setattr(rouge_one_cfg, "exec_compat", False)


VALID_WITH_ESCAPE = 'x = 1\nresult = "a" + "\\n" + "b"'


def test_default_still_unescapes_inside_literals(default):
    out = pandas_exec.normalize_user_command(VALID_WITH_ESCAPE)
    with pytest.raises(SyntaxError):
        compile(out, "<t>", "exec")


def test_compat_keeps_escape_in_valid_code(compat):
    out = pandas_exec.normalize_user_command(VALID_WITH_ESCAPE)
    scope = {}
    exec(out, scope)
    assert scope["result"] == "a\nb"


def test_compat_still_unescapes_double_escaped_code(compat):
    out = pandas_exec.normalize_user_command("a = 1\\nresult = a + 1")
    scope = {}
    exec(out, scope)
    assert scope["result"] == 2


def test_tester_normalizer_follows_flag(compat):
    from rogueone.llm.agents.core.tester.agent import _normalize_user_command

    scope = {}
    exec(_normalize_user_command(VALID_WITH_ESCAPE), scope)
    assert scope["result"] == "a\nb"


def test_last_expression_sets_result_only_under_compat(monkeypatch):
    df = pd.DataFrame({"a": [1, 2]})
    monkeypatch.setattr(rouge_one_cfg, "exec_compat", False)
    assert pandas_exec.exec_user_command("df['a'].sum()", df, df)[0] is None
    monkeypatch.setattr(rouge_one_cfg, "exec_compat", True)
    assert pandas_exec.exec_user_command("df['a'].sum()", df, df)[0] == 3
    assert pandas_exec.exec_user_command("b = 2\ndf['a'].sum() * b", df, df)[0] == 6
    # an explicit result wins over the trailing expression
    assert pandas_exec.exec_user_command("result = 7\ndf['a'].sum()", df, df)[0] == 7
    assert pandas_exec.exec_user_command("result = df['a'].max()", df, df)[0] == 2
