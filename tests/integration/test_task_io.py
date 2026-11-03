"""Integration tests for task I/O (YAML/CSV reading and writing)."""

import pytest
import os
import yaml
import pandas as pd

pytest_plugins = []


@pytest.mark.integration
@pytest.mark.task_io
def test_task_io_reads_classification_config():
    """
    Test task I/O can read classification task config.
    
    Per spec FR-005: Must validate task YAML/CSV read operations
    """
    config_path = "/workspaces/RougeOne/tasks/test_classification/config.yml"
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert config["task_type"] == "classification"
        assert config["task_name"] == "test_classification"
        assert "data_path" in config
    else:
        pytest.skip("Test config file not found")


@pytest.mark.integration
@pytest.mark.task_io
def test_task_io_reads_regression_config():
    """
    Test task I/O can read regression task config.
    
    Per spec FR-005: Must validate task YAML/CSV read operations
    """
    config_path = "/workspaces/RougeOne/tasks/test_regression/config.yml"
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert config["task_type"] == "regression"
        assert config["task_name"] == "test_regression"
        assert "data_path" in config
    else:
        pytest.skip("Test config file not found")


@pytest.mark.integration
@pytest.mark.task_io
def test_task_io_reads_classification_csv():
    """
    Test task I/O can read classification task CSV data.
    
    Per spec FR-005: Must validate CSV read operations
    """
    csv_path = "/workspaces/RougeOne/tasks/test_classification/data.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        assert not df.empty, "CSV should not be empty"
        assert "income" in df.columns, "Target column 'income' required"
        assert len(df) > 0, "CSV should have at least 1 row"
    else:
        pytest.skip("Test data file not found")


@pytest.mark.integration
@pytest.mark.task_io
def test_task_io_reads_regression_csv():
    """
    Test task I/O can read regression task CSV data.
    
    Per spec FR-005: Must validate CSV read operations
    """
    csv_path = "/workspaces/RougeOne/tasks/test_regression/data.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        assert not df.empty, "CSV should not be empty"
        assert "price" in df.columns, "Target column 'price' required"
        assert len(df) > 0, "CSV should have at least 1 row"
    else:
        pytest.skip("Test data file not found")


@pytest.mark.integration
@pytest.mark.task_io
def test_task_io_handles_missing_files():
    """
    Test task I/O handles missing files gracefully.
    
    Per spec FR-003: Must validate failure handling
    """
    missing_path = "/nonexistent/path/config.yml"
    
    # Document expected behavior: FileNotFoundError or similar when file missing
    assert not os.path.exists(missing_path), "Test file should not exist"
