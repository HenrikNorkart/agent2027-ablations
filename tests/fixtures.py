"""Shared test fixtures and factories for RougeOne tests."""

import pandas as pd
from typing import Dict, Any


def sample_dataframe() -> pd.DataFrame:
    """Create a minimal sample DataFrame for testing."""
    return pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature2": [5.0, 4.0, 3.0, 2.0, 1.0],
        "feature3": [1.5, 2.5, 3.5, 4.5, 5.5],
        "target": [0, 1, 0, 1, 0]
    })


def sample_classification_target() -> pd.Series:
    """Sample target for classification (binary)."""
    return pd.Series([0, 1, 0, 1, 0], name="target")


def sample_regression_target() -> pd.Series:
    """Sample target for regression (continuous)."""
    return pd.Series([1.5, 2.3, 3.1, 4.2, 5.0], dtype=float, name="target")


def minimal_valid_config() -> Dict[str, Any]:
    """Minimal valid ExperimentConfig dict for testing."""
    return {
        "task_type": "classification",
        "task_name": "test_task",
        "data_path": "tasks/test_classification/data.csv",
        "config_path": "tasks/test_classification/config.yml",
        "target": "target",
        "features": "all",
        "model_name": "qwen2-7b",
        "llm_endpoint": "http://localhost:8000",
        "llm_api_key": "rogueone-dev",
        "random_seed": 42,
    }
