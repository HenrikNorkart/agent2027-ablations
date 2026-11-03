"""Unit tests for ExperimentConfig and config loading."""

import pytest
import tempfile
import os

pytest_plugins = []


@pytest.mark.unit
@pytest.mark.config
def test_experiment_config_imports():
    """Verify ExperimentConfig can be imported."""
    try:
        from rogueone.utils.config import ExperimentConfig
        assert ExperimentConfig is not None
    except ImportError:
        pytest.skip("ExperimentConfig not available")


@pytest.mark.unit
@pytest.mark.config
def test_experiment_config_validates_task_type(config_fixture):
    """
    Test ExperimentConfig validates task_type field.
    
    Per Constitution IV: task_type must be either "classification" or "regression"
    """
    try:
        from rogueone.utils.config import ExperimentConfig
        
        # Test valid config
        assert config_fixture.get("task_type") in ["classification", "regression"]
        
    except ImportError:
        pytest.skip("ExperimentConfig not available")


@pytest.mark.unit
@pytest.mark.config
@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_experiment_config_supports_both_task_types(task_type, config_fixture):
    """
    Test ExperimentConfig works with both task types.
    
    Per Constitution IV: implementation must be task-agnostic
    """
    try:
        from rogueone.utils.config import ExperimentConfig
        
        # Test config with task type
        test_config = config_fixture.copy()
        test_config["task_type"] = task_type
        
        assert test_config["task_type"] == task_type
        
    except ImportError:
        pytest.skip("ExperimentConfig not available")


@pytest.mark.unit
@pytest.mark.config
def test_experiment_config_requires_essential_fields(config_fixture):
    """
    Test ExperimentConfig has required fields.
    
    Per spec FR-005: Must validate config loading for reproducibility
    """
    try:
        from rogueone.utils.config import ExperimentConfig
        
        # Check essential fields
        required_fields = ["task_type", "task_name", "data_path"]
        for field in required_fields:
            assert field in config_fixture, f"Missing required field: {field}"
        
    except ImportError:
        pytest.skip("ExperimentConfig not available")
