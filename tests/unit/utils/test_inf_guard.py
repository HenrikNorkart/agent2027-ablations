"""ROGUE_ONE_INF_GUARD: clip +-inf in the Tester's fold matrices (A17 relaunch)."""
import numpy as np
import pytest
import xgboost as xgb

from rogueone.llm.agents.core.tester import agent as tester
from rogueone.utils import instrumentation, rouge_one_cfg


def test_default_is_off():
    assert rouge_one_cfg.inf_guard is False


def test_raw_inf_matrix_crashes_xgboost():
    X = np.array([[1.0, np.inf], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0]])
    with pytest.raises(xgb.core.XGBoostError):
        xgb.XGBRegressor(n_estimators=2).fit(X, np.arange(4.0))


def test_guard_clips_inf_keeps_nan_and_fits():
    X = np.array([[1.0, np.inf], [2.0, -np.inf], [3.0, np.nan], [4.0, 1e300]])
    before = instrumentation.get_counter(tester.INF_GUARD_COUNTER)
    out = tester._inf_guard(X)
    assert out.dtype == np.float32
    assert out[0, 1] == 1e6 and out[1, 1] == -1e6 and out[3, 1] == 1e6
    assert np.isnan(out[2, 1])
    assert instrumentation.get_counter(tester.INF_GUARD_COUNTER) == before + 1
    xgb.XGBRegressor(n_estimators=2).fit(out, np.arange(4.0))


def test_finite_matrix_unchanged_and_not_counted():
    X = np.array([[1.0, 2.0], [3.0, np.nan]])
    before = instrumentation.get_counter(tester.INF_GUARD_COUNTER)
    out = tester._inf_guard(X)
    np.testing.assert_array_equal(out, X.astype(np.float32))
    assert instrumentation.get_counter(tester.INF_GUARD_COUNTER) == before


def test_non_numeric_matrix_passed_through():
    X = np.array([["a", "b"]], dtype=object)
    assert tester._inf_guard(X) is not None
