"""E13: ROGUE_ONE_SANDBOX restricts the three generation-side exec scopes."""

from __future__ import annotations

import builtins

import numpy as np
import pandas as pd
import pytest

from rogueone.utils import flag_registry as fr
from rogueone.utils import sandbox as sb
from rogueone.utils.pandas_exec import exec_user_command
from rogueone.llm.agents.core.scientist.agent import build_pandas_tool_scope
from rogueone.llm.agents.core.tester.agent import build_python_executor_scope


# --------------------------------------------------------------------------- fixtures
@pytest.fixture
def labels_path(tmp_path):
    p = tmp_path / "workspace" / "tasks" / "classification" / "car" / "labels.csv"
    p.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2, 3], "target": [0, 1, 0]}).to_csv(p, index=False)
    return str(p)


@pytest.fixture
def frames():
    df = pd.DataFrame(
        {"id": [1, 1, 2, 2, 3], "persons": ["2", "2", "4", "4", "more"], "x": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    df_features = pd.DataFrame({"id": [1, 2, 3]})
    return df, df_features


@pytest.fixture
def sandbox_on(monkeypatch):
    monkeypatch.setenv("ROGUE_ONE_SANDBOX", "on")


@pytest.fixture
def sandbox_off(monkeypatch):
    monkeypatch.delenv("ROGUE_ONE_SANDBOX", raising=False)


def _tester_scope():
    df_attributes = pd.DataFrame({"f1": [0.1, 0.2, 0.3, 0.4], "target": [0, 1, 0, 1]})
    return build_python_executor_scope(df_attributes, assessment_model=None)


def _scientist_scope(frames):
    df, _ = frames
    df_test_results = pd.DataFrame({"id": ["Trial 1", "Trial 2"], "train_accuracy": [0.7, 0.8]})
    return build_pandas_tool_scope(df_test_results, df)


def _run_extractor(code, frames):
    df, df_features = frames
    result, _ = exec_user_command(code, df, df_features)
    return result


def _run_scope(code, scope, out="result"):
    exec(code, scope)
    return scope.get(out)


# --------------------------------------------------------------------------- flag
def test_flag_registered_with_code_default_off(monkeypatch):
    assert fr.FLAG_DEFAULTS["ROGUE_ONE_SANDBOX"] == "off"
    monkeypatch.delenv("ROGUE_ONE_SANDBOX", raising=False)
    assert sb.sandbox_enabled() is False
    for v in ("on", "ON", "1", "true"):
        assert sb.sandbox_enabled({"ROGUE_ONE_SANDBOX": v}) is True
    for v in ("off", "0", "", "no"):
        assert sb.sandbox_enabled({"ROGUE_ONE_SANDBOX": v}) is False


# --------------------------------------------------------------------------- off == pre-E13
def test_off_scopes_are_the_pre_e13_dicts(sandbox_off, frames):
    sci = _scientist_scope(frames)
    assert set(sci) == {"pd", "np", "df_test_results", "df_raw_data"}
    assert sci["pd"] is pd and sci["np"] is np
    tst = _tester_scope()
    assert set(tst) == {"df_attributes", "assessment_model", "pd", "np", "sklearn",
                        "shap", "sm", "spy", "xgb"}
    assert tst["pd"] is pd and "__builtins__" not in tst


def test_off_open_still_works_everywhere(sandbox_off, frames, labels_path):
    code = f"result = open({labels_path!r}).read()"
    assert "target" in _run_extractor(code, frames)
    assert "target" in _run_scope(code, _scientist_scope(frames))
    assert "target" in _run_scope(code, _tester_scope())


# --------------------------------------------------------------------------- on: file access fails
BLOCKED_SNIPPETS = [
    "result = open({p!r}).read()",
    "result = pd.read_csv({p!r})",
    "from pandas import read_csv\nresult = read_csv({p!r})",
    "import pandas\nresult = pandas.read_csv({p!r})",
    "result = np.loadtxt({p!r}, delimiter=',', skiprows=1)",
    "import os\nresult = os.listdir('/')",
    "result = __import__('os').getcwd()",
    "import io\nresult = io.open({p!r}).read()",
    "import pathlib\nresult = pathlib.Path({p!r}).read_text()",
    "result = eval('1+1')",
    "exec('result = 1')",
    "result = compile('1', 's', 'eval')",
    "import builtins\nresult = builtins.open({p!r}).read()",
]


@pytest.mark.parametrize("snippet", BLOCKED_SNIPPETS)
def test_on_blocks_file_access_in_extractor(sandbox_on, frames, labels_path, snippet):
    with pytest.raises((NameError, ImportError, AttributeError)):
        _run_extractor(snippet.format(p=labels_path), frames)


@pytest.mark.parametrize("snippet", BLOCKED_SNIPPETS)
def test_on_blocks_file_access_in_scientist(sandbox_on, frames, labels_path, snippet):
    with pytest.raises((NameError, ImportError, AttributeError)):
        _run_scope(snippet.format(p=labels_path), _scientist_scope(frames))


@pytest.mark.parametrize(
    "snippet",
    BLOCKED_SNIPPETS
    + [
        "result = spy.io.loadmat({p!r})",
        "result = sklearn.datasets.load_svmlight_file({p!r})",
        "from sklearn.datasets import load_files\nresult = 1",
    ],
)
def test_on_blocks_file_access_in_tester(sandbox_on, labels_path, snippet):
    with pytest.raises((NameError, ImportError, AttributeError)):
        _run_scope(snippet.format(p=labels_path), _tester_scope(), out="result")


def test_on_does_not_leak_into_process_builtins(sandbox_on, frames, labels_path):
    with pytest.raises(NameError):
        _run_extractor(f"result = open({labels_path!r}).read()", frames)
    with open(labels_path) as fh:  # the host process is unaffected
        assert "target" in fh.read()
    assert builtins.open is open and pd.read_csv(labels_path).shape == (3, 2)


# --------------------------------------------------------------------------- on: normal code works
GROUPBY_PATTERN = (
    "persons_map = {'2': 2, '4': 4, 'more': 5}\n"
    "result = df.groupby('id').apply(\n"
    "    lambda g: persons_map[g['persons'].iloc[0]]\n"
    ")"
)


def test_on_extractor_normal_pandas_and_groupby_pattern(sandbox_on, frames):
    res = _run_extractor("result = df.groupby('id')['x'].mean()", frames)
    assert list(res) == [1.5, 3.5, 5.0]
    res = _run_extractor(GROUPBY_PATTERN, frames)
    assert list(res) == [2, 4, 5]
    res = _run_extractor(
        "tmp = df.groupby('id')['x'].agg(['mean', 'std']).reset_index()\n"
        "result = pd.merge(df_features, tmp, on='id', how='left')\n"
        "result['z'] = np.where(result['mean'] > 2, 1, 0)",
        frames,
    )
    assert list(result_col for result_col in res.columns) == ["id", "mean", "std", "z"]
    res = _run_extractor(
        "import numpy as np2\nimport math\nfrom collections import Counter\n"
        "result = np2.log1p(df['x']).round(3).tolist() + [math.pi > 3, len(Counter(df['persons']))]",
        frames,
    )
    assert res[-1] == 3 and res[-2] is True
    # pd.api / pd.Series / isinstance checks still resolve through the proxy
    res = _run_extractor(
        "result = isinstance(df, pd.DataFrame) and pd.api.types.is_numeric_dtype(df['x'])",
        frames,
    )
    assert res is True


def test_on_scientist_normal_pandas(sandbox_on, frames):
    scope = _scientist_scope(frames)
    assert _run_scope("result = df_test_results['train_accuracy'].mean()", scope) == pytest.approx(0.75)
    scope = _scientist_scope(frames)
    res = _run_scope(GROUPBY_PATTERN.replace("df.", "df_raw_data."), scope)
    assert list(res) == [2, 4, 5]


def test_on_tester_normal_analysis_and_in_memory_labels(sandbox_on):
    code = (
        "from sklearn.linear_model import LogisticRegression\n"
        "from scipy import stats\n"
        "X = df_attributes[['f1']]; y = df_attributes['target']\n"
        "m = LogisticRegression().fit(X, y)\n"
        "r, _ = stats.pearsonr(df_attributes['f1'], y)\n"
        "output = (len(m.coef_[0]), round(float(r), 3), int(y.sum()))"
    )
    out = _run_scope(code, _tester_scope(), out="output")
    assert out[0] == 1 and out[2] == 2
    # stdout capture via print() still works
    out = _run_scope("print('hi')\noutput = sm.__name__ + spy.__name__ + xgb.__name__", _tester_scope(), out="output")
    assert out == "statsmodelsscipyxgboost"


def test_on_guarded_module_is_read_only(sandbox_on, frames):
    with pytest.raises(AttributeError):
        _run_extractor("pd.read_csv = None\nresult = 1", frames)
    assert callable(pd.read_csv)
