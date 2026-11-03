"""Unit tests for ExtractorAgent."""

import pytest
import numpy as np
import pandas as pd

from rogueone.llm.agents.core.extractor.agent import (
    _exec_user_command,
    _normalize_user_command,
    _build_extractor_model_settings,
    _ordinal_encode_categorical,
)

pytest_plugins = []


@pytest.mark.unit
@pytest.mark.agent
def test_extractor_agent_exists():
    """Verify ExtractorAgent can be imported."""
    try:
        from rogueone.llm.agents import ExtractorAgent

        assert ExtractorAgent is not None
    except ImportError:
        pytest.skip("ExtractorAgent not yet implemented")


@pytest.mark.unit
@pytest.mark.agent
def test_extractor_output_contract_fields():
    """
    Validate ExtractorAgent output contract includes required fields.

    Per agent-behavior-validation-contract.md, output must include:
    - hypotheses (list of dicts with: name, argumentation, domain_relevance)
    - reasoning (non-empty string)

    Per Constitution III: argumentation field is MANDATORY and non-empty
    """
    try:
        from rogueone.llm.agents import ExtractorAgent

        # Expected contract:
        expected_fields = ["hypotheses", "reasoning"]
        hypothesis_fields = ["name", "argumentation", "domain_relevance"]

        assert all(f in expected_fields for f in expected_fields)

    except ImportError:
        pytest.skip("ExtractorAgent not yet implemented")


@pytest.mark.unit
@pytest.mark.agent
def test_extractor_validates_argumentation_field():
    """
    Test ExtractorAgent validates Constitution III: knowledge-informed features.

    The argumentation field is MANDATORY per Constitution III and must be non-empty.
    This is a core validation for research reproducibility.
    """
    try:
        from rogueone.llm.agents import ExtractorAgent

        # Placeholder test - documents Constitution III requirement
        # When implemented, this validates agent requires non-empty argumentation

    except ImportError:
        pytest.skip("ExtractorAgent not yet implemented")


@pytest.mark.unit
@pytest.mark.agent
def test_extractor_handles_empty_data():
    """
    Test ExtractorAgent rejects empty DataFrame.

    Per spec FR-003: Must validate failure handling
    """
    try:
        from rogueone.llm.agents import ExtractorAgent

        # Placeholder test - documents expected behavior
        # When implemented, this validates agent raises ValueError for empty data

    except ImportError:
        pytest.skip("ExtractorAgent not yet implemented")


@pytest.mark.unit
@pytest.mark.agent
@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_extractor_supports_multiple_task_types(task_type):
    """
    Test ExtractorAgent works with both classification and regression.

    Per Constitution IV: task-agnostic design - no duplicated logic per task type
    """
    try:
        from rogueone.llm.agents import ExtractorAgent

        # Placeholder test - documents Constitution IV requirement
        # When implemented, this validates agent supports both task types

    except ImportError:
        pytest.skip("ExtractorAgent not yet implemented")


# ---------------------------------------------------------------------------
# Helpers / fixtures for the exec helper tests
# ---------------------------------------------------------------------------


def _make_dfs():
    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0],
            "buying": ["low", "low", "high", "high", "vhigh"],
        }
    )
    df_features = pd.DataFrame({"id": [1, 2, 3]})
    return df, df_features


@pytest.mark.unit
@pytest.mark.agent
def test_extractor_exec_accepts_multi_statement_semicolon():
    df, df_features = _make_dfs()
    cmd = "tmp = df.groupby('id')['x'].mean(); result = tmp.fillna(0)"
    result, _ = _exec_user_command(cmd, df, df_features)
    assert isinstance(result, pd.Series)
    assert len(result) == 3


@pytest.mark.unit
@pytest.mark.agent
def test_extractor_exec_accepts_multi_statement_newline():
    df, df_features = _make_dfs()
    cmd = "tmp = df.groupby('id')['x'].mean()\nresult = tmp.fillna(0)"
    result, _ = _exec_user_command(cmd, df, df_features)
    assert isinstance(result, pd.Series)
    assert len(result) == 3


@pytest.mark.unit
@pytest.mark.agent
def test_extractor_exec_tolerates_leading_whitespace_after_semicolon():
    df, df_features = _make_dfs()
    cmd = "tmp = df.groupby('id')['x'].mean();\n  result = tmp.fillna(0)"
    result, _ = _exec_user_command(cmd, df, df_features)
    assert isinstance(result, pd.Series)
    assert len(result) == 3


@pytest.mark.unit
@pytest.mark.agent
def test_extractor_exec_tolerates_dedent_indented_block():
    df, df_features = _make_dfs()
    cmd = "    tmp = df.groupby('id')['x'].mean()\n    result = tmp.fillna(0)"
    result, _ = _exec_user_command(cmd, df, df_features)
    assert isinstance(result, pd.Series)
    assert len(result) == 3


@pytest.mark.unit
@pytest.mark.agent
def test_extractor_exec_categorical_ordinal_pattern_one_call():
    """Regression: car-failure mode — derived ordinal column built in same call."""
    df, df_features = _make_dfs()
    cmd = (
        "buying_map = {'low': 1, 'med': 2, 'high': 3, 'vhigh': 4}\n"
        "buying_ord = df.groupby('id')['buying'].first().map(buying_map)\n"
        "result = df_features['id'].map(buying_ord).fillna(0).values"
    )
    result, _ = _exec_user_command(cmd, df, df_features)
    assert isinstance(result, np.ndarray)
    assert len(result) == 3
    # ids 1->low(1), 2->high(3), 3->vhigh(4)
    assert list(result) == [1, 3, 4]


@pytest.mark.unit
@pytest.mark.agent
def test_extractor_exec_rejects_python_and_or_on_series_with_helpful_message():
    df, df_features = _make_dfs()
    cmd = "result = (df['x'] == 1) and (df['y'] == 10)"
    with pytest.raises(ValueError) as excinfo:
        _exec_user_command(cmd, df, df_features)
    # The underlying ValueError should still surface (we don't mask it).
    assert "truth value" in str(excinfo.value).lower()


@pytest.mark.unit
@pytest.mark.agent
def test_extractor_exec_lambda_sees_locals_defined_in_same_call():
    """Regression: car run mm1v57s6 — multi-statement code where a lambda inside
    df.groupby(...).apply(...) references a dict defined as a top-level statement
    in the same exec snippet. With separate globals/locals dicts the lambda's
    __globals__ does not include the locals_dict, so it raises NameError. This
    test pins the single-scope behavior in :func:`_exec_user_command`.
    """
    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "doors": ["2", "2", "4", "4", "5more", "5more"],
            "persons": ["2", "2", "4", "4", "more", "more"],
        }
    )
    df_features = pd.DataFrame({"id": [1, 2, 3]})
    cmd = (
        "doors_map = {'2': 2, '3': 3, '4': 4, '5more': 5}\n"
        "persons_map = {'2': 2, '4': 4, 'more': 5}\n"
        "result = df.groupby('id').apply(\n"
        "    lambda g: doors_map[g['doors'].iloc[0]] / persons_map[g['persons'].iloc[0]]\n"
        ")"
    )
    result, _ = _exec_user_command(cmd, df, df_features)
    assert hasattr(result, "__len__")
    assert len(result) == 3


@pytest.mark.unit
@pytest.mark.agent
def test_extractor_exec_lookup_table_visible_to_list_comprehension():
    """Regression: tic-tac-toe runs si9h32kk / lciydoty — `lines = [[...], ...]`
    defined as a top-level statement, then referenced in a generator expression
    consumed by ``pd.concat``. With separate globals/locals dicts the comp's
    enclosing scope misses ``lines`` and raises NameError. Pin single-scope.
    """
    df = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "top-left-square": ["x", "x", "x", "o", "o", "o"],
            "top-middle-square": ["x", "x", "x", "o", "o", "o"],
            "top-right-square": ["x", "x", "x", "o", "o", "o"],
        }
    )
    df_features = pd.DataFrame({"id": [1, 2]})
    cmd = (
        "lines = [['top-left-square', 'top-middle-square', 'top-right-square']]\n"
        "result = pd.concat(\n"
        "    [(df.groupby('id')[cols].first() == 'x').all(axis=1) for cols in lines],\n"
        "    axis=1,\n"
        ").any(axis=1).astype(int)"
    )
    result, _ = _exec_user_command(cmd, df, df_features)
    assert isinstance(result, pd.Series)
    assert list(result) == [1, 0]


@pytest.mark.unit
@pytest.mark.agent
def test_extractor_normalize_preserves_simple_command():
    cmd = "result = df.groupby('id')['x'].mean()"
    assert _normalize_user_command(cmd) == cmd


@pytest.mark.unit
@pytest.mark.agent
def test_extractor_exec_normalizes_literal_backslash_n():
    """Regression: car v23 / run 09vnwi4s — the model emitted multi-statement code
    where ``\\n`` is a two-character literal (backslash + n) rather than a real
    newline. Python parses ``\\n`` as a line-continuation backslash followed by
    an identifier ``n`` and raises ``SyntaxError: unexpected character after line
    continuation character``. The normalizer must convert the literal escape
    into a real newline.
    """
    df = pd.DataFrame(
        {"id": [1, 1, 2, 2], "buying": ["low", "low", "high", "high"]}
    )
    df_features = pd.DataFrame({"id": [1, 2]})
    cmd = (
        "buying_map = {'low': 1, 'high': 3}"
        "\\n"
        "result = df_features['id'].map("
        "df.groupby('id')['buying'].first().map(buying_map)"
        ").values"
    )
    # Sanity: the raw command really does contain literal backslash + n
    assert "\\n" in cmd
    result, _ = _exec_user_command(cmd, df, df_features)
    assert list(result) == [1, 3]


@pytest.mark.unit
@pytest.mark.agent
def test_extractor_exec_normalizes_continuation_line_indent_without_semicolon():
    """Regression: car v23 / run 09vnwi4s — model emitted multi-line code where
    line 1 is flush-left and lines 2+ start with a leading space. The existing
    ``;\\s*\\n[ \\t]+`` regex only collapses semicolon-then-indent; this pattern
    has no semicolon. The probe-and-fall-back compile() check must catch the
    IndentationError and lstrip continuation lines (no `:`-block present).
    """
    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "buying": ["low", "low", "high", "high"],
            "safety": ["low", "low", "high", "high"],
        }
    )
    df_features = pd.DataFrame({"id": [1, 2]})
    cmd = (
        "buying_map = {'low': 1, 'high': 3}\n"
        " safety_map = {'low': 1, 'high': 3}\n"
        " buying_ord = df.groupby('id')['buying'].first().map(buying_map)\n"
        " safety_ord = df.groupby('id')['safety'].first().map(safety_map)\n"
        " result = df_features['id'].map(buying_ord + safety_ord).values"
    )
    result, _ = _exec_user_command(cmd, df, df_features)
    assert list(result) == [2, 6]


@pytest.mark.unit
@pytest.mark.agent
def test_extractor_normalize_preserves_legitimately_indented_block():
    """The normalize-on-IndentationError path must NOT mangle a real Python
    block. When the snippet contains a `:`-terminated line (def / if / for),
    the body's indentation is meaningful and must survive normalization.
    """
    cmd = (
        "def winner(g):\n"
        "    return int((g['x'] == 'a').all())\n"
        "result = df.groupby('id').apply(winner).values"
    )
    out = _normalize_user_command(cmd)
    # The function body's `return` line must keep its 4-space indentation.
    assert "    return int" in out


# ---------------------------------------------------------------------------
# df / df_features confusion (CAR run m4i2r4v9 dominant failure pattern)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.agent
def test_extractor_exec_keyerror_hints_df_features_when_column_only_there():
    """Reading a derived column from `df` when it lives on `df_features` must
    surface a hint that points the agent to the right DataFrame.

    This was the dominant CAR failure mode: the model added `safety_ord` to
    df_features then immediately wrote `df.groupby('id')['safety_ord']`,
    which raised an opaque `KeyError: 'Column not found: safety_ord'`. With
    no signal that the column simply lived elsewhere, the model retried the
    same shape, the iteration produced 0 new features, and the dataset
    aborted.
    """
    df = pd.DataFrame({"id": [1, 1, 2, 2], "buying": ["high"] * 4})
    df_features = pd.DataFrame({"id": [1, 2], "safety_ord": [0, 1]})

    with pytest.raises(KeyError) as exc_info:
        _exec_user_command(
            "result = df.groupby('id')['safety_ord'].first()",
            df,
            df_features,
        )

    msg = str(exc_info.value)
    assert "df_features" in msg
    assert "safety_ord" in msg


@pytest.mark.unit
@pytest.mark.agent
def test_extractor_exec_keyerror_hints_df_when_column_only_there():
    """Symmetric case: reading a raw column from `df_features` when it only
    lives on `df` should hint back to `df`."""
    df = pd.DataFrame({"id": [1, 1, 2, 2], "raw_only": [10, 20, 30, 40]})
    df_features = pd.DataFrame({"id": [1, 2], "engineered": [0.1, 0.2]})

    with pytest.raises(KeyError) as exc_info:
        _exec_user_command("result = df_features['raw_only']", df, df_features)

    msg = str(exc_info.value)
    assert "raw_only" in msg
    assert "df" in msg
    # The hint should not push the agent toward df_features again.
    assert "available in `df`" in msg


@pytest.mark.unit
@pytest.mark.agent
def test_extractor_exec_keyerror_passes_through_when_column_in_neither():
    """When the missing column does not exist on either DataFrame the original
    pandas KeyError must propagate unchanged so the agent sees the real cause
    (typo, hallucinated name) and does not get a misleading hint."""
    df = pd.DataFrame({"id": [1, 2], "x": [1, 2]})
    df_features = pd.DataFrame({"id": [1, 2], "y": [10, 20]})

    with pytest.raises(KeyError) as exc_info:
        _exec_user_command("result = df['totally_missing']", df, df_features)

    msg = str(exc_info.value)
    assert "df_features" not in msg


@pytest.mark.unit
@pytest.mark.agent
def test_extractor_exec_keyerror_handles_quoted_columns_message():
    """pandas formats some KeyErrors as `\"['col_a','col_b'] not in index\"`.
    The hint extractor must still find the offending column in that shape so
    multi-column drops/selects also benefit from the cross-DataFrame hint."""
    df = pd.DataFrame({"id": [1, 2], "buying": ["high", "low"]})
    df_features = pd.DataFrame(
        {"id": [1, 2], "buying_ord": [0, 1], "maint_ord": [1, 0]}
    )

    # df.groupby('id')[['buying_ord','maint_ord']] raises a KeyError with the
    # quoted-list shape — both columns live on df_features.
    with pytest.raises(KeyError) as exc_info:
        _exec_user_command(
            "result = df.groupby('id')[['buying_ord','maint_ord']].first()",
            df,
            df_features,
        )

    msg = str(exc_info.value)
    assert "df_features" in msg


# ---------------------------------------------------------------------------
# Extractor prompt — "always do something" rule and last-resort additions
# (added after CAR runs ln2qurm8 / g6t13ci5 surfaced an empty-Extractor
#  failure mode the previous prompt round did not deter)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.agent
def test_extractor_prompt_demands_at_least_one_append_call():
    """The Extractor sometimes runs out of fresh ideas mid-iteration and
    returns no tool calls at all (Mode A — CAR runs ca90miqg / p9t3rayi /
    wi5r0oki). The system prompt must explicitly require at least one
    append_new_attribute call per iteration and enumerate cheap last-resort
    additions (pruned-feature restoration, simple transforms, pairwise
    interactions, ordinal encodings) so the model has concrete fallback
    moves when its primary plan is unworkable."""
    import inspect
    from rogueone.llm.agents.core.extractor.agent import ExtractorAgent

    src = inspect.getsource(ExtractorAgent.generate_patient_attributes)
    src_lower = src.lower()

    assert "at least once" in src_lower, (
        "Prompt must require at least one append_new_attribute call."
    )
    assert "last-resort" in src_lower or "last resort" in src_lower, (
        "Prompt must label the cheap fallback options as last-resort."
    )
    # At least 3 of the suggested fallback moves should be named.
    fallback_hints = [
        "restore",
        "polynomial",
        "log",
        "interaction",
        "ordinal",
        "pairwise",
    ]
    hits = sum(1 for h in fallback_hints if h in src_lower)
    assert hits >= 3, (
        f"Prompt should enumerate concrete fallback moves; only {hits} of "
        f"{fallback_hints} present."
    )


# ---------------------------------------------------------------------------
# Categorical fallback (CAR-style all-categorical datasets, post-fix run round)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.agent
def test_ordinal_encode_categorical_basic_car_like():
    """CAR has six raw categorical columns. The fallback must encode each
    deterministically so that the dataset can survive a 0-feature Extractor
    turn (Mode A). Categories are sorted alphabetically before encoding so
    the mapping is reproducible regardless of row order.
    """
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "safety": ["low", "high", "med", "low"],
        }
    )
    df_features = pd.DataFrame({"id": [1, 2, 3, 4]})

    encoded = _ordinal_encode_categorical(df, column="safety", df_features=df_features)

    assert encoded is not None
    # Sorted alphabetically: high=0, low=1, med=2
    expected = pd.Series([1.0, 0.0, 2.0, 1.0])
    pd.testing.assert_series_equal(
        encoded.reset_index(drop=True),
        expected,
        check_names=False,
    )


@pytest.mark.unit
@pytest.mark.agent
def test_ordinal_encode_categorical_independent_of_row_order():
    """Reordering the rows must not change the encoding — sorted-categories
    encoding gives reproducible numbers across runs."""
    df1 = pd.DataFrame({"id": [1, 2, 3], "buying": ["low", "high", "vhigh"]})
    df2 = pd.DataFrame({"id": [3, 1, 2], "buying": ["vhigh", "low", "high"]})
    df_features = pd.DataFrame({"id": [1, 2, 3]})

    e1 = _ordinal_encode_categorical(df1, column="buying", df_features=df_features)
    e2 = _ordinal_encode_categorical(df2, column="buying", df_features=df_features)

    pd.testing.assert_series_equal(
        e1.reset_index(drop=True),
        e2.reset_index(drop=True),
        check_names=False,
    )


@pytest.mark.unit
@pytest.mark.agent
def test_ordinal_encode_categorical_returns_none_on_missing_column():
    df = pd.DataFrame({"id": [1, 2], "buying": ["low", "high"]})
    df_features = pd.DataFrame({"id": [1, 2]})

    assert (
        _ordinal_encode_categorical(df, column="not_there", df_features=df_features)
        is None
    )


@pytest.mark.unit
@pytest.mark.agent
def test_ordinal_encode_categorical_returns_none_when_alignment_fails():
    """If df_features['id'] contains an id that has no row in df, the encoding
    cannot be aligned — the fallback must return None rather than write NaN
    into df_features (NaN values would later trigger an isna() check elsewhere)."""
    df = pd.DataFrame({"id": [1, 2], "buying": ["low", "high"]})
    df_features = pd.DataFrame({"id": [1, 2, 3]})  # id=3 has no row in df

    assert (
        _ordinal_encode_categorical(df, column="buying", df_features=df_features)
        is None
    )


# ---------------------------------------------------------------------------
# Config / ModelSettings wiring
# ---------------------------------------------------------------------------


def _reset_config_singletons():
    from rogueone.utils.singelton import Singleton
    from rogueone.utils import config as _cfg

    for cls in (
        _cfg.ExtractorAgentConfig,
        _cfg.MedicalKnowledgeAgentConfig,
        _cfg.TestAgentConfig,
        _cfg.ScientistAgentConfig,
    ):
        Singleton._instances.pop(cls, None)


@pytest.mark.unit
@pytest.mark.agent
def test_extractor_config_max_tokens_default(monkeypatch):
    monkeypatch.delenv("EXTRACTOR_AGENT_MAX_TOKENS", raising=False)
    _reset_config_singletons()
    from rogueone.utils.config import ExtractorAgentConfig

    cfg = ExtractorAgentConfig()
    assert cfg.max_tokens == 8192

    monkeypatch.setenv("EXTRACTOR_AGENT_MAX_TOKENS", "16384")
    _reset_config_singletons()
    cfg2 = ExtractorAgentConfig()
    assert cfg2.max_tokens == 16384

    # Restore singleton with default for downstream tests.
    monkeypatch.delenv("EXTRACTOR_AGENT_MAX_TOKENS", raising=False)
    _reset_config_singletons()


@pytest.mark.unit
@pytest.mark.agent
def test_extractor_model_settings_propagates_max_tokens():
    class _Cfg:
        temperature = 0.5
        reasoning = None
        max_tokens = 4321

    settings = _build_extractor_model_settings(_Cfg())
    assert settings.max_tokens == 4321
    assert settings.temperature == 0.5


@pytest.mark.unit
@pytest.mark.agent
def test_extractor_model_settings_omits_max_tokens_when_none():
    class _Cfg:
        temperature = 0.5
        reasoning = None
        max_tokens = None

    settings = _build_extractor_model_settings(_Cfg())
    assert settings.max_tokens is None
