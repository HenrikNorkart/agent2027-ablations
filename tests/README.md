# RougeOne Test Suite

Complete automated pytest testing infrastructure for the RougeOne project. This guide covers test organization, execution, markers, fixtures, and best practices.

## Directory Structure

```
tests/
├── README.md                    # This file: test guide & best practices
├── conftest.py                  # Session & function-scope shared fixtures
├── fixtures.py                  # Reusable test data factories & mocks
├── pytest.ini                   # Pytest configuration (markers, timeouts)
│
├── unit/                        # Unit tests (no external dependencies)
│   ├── agents/                  # Agent contract validation
│   │   ├── test_scientist.py    # Scientist agent tests
│   │   ├── test_extractor.py    # Extractor agent tests
│   │   ├── test_tester.py       # Tester agent tests
│   │   ├── test_knowledge.py    # Knowledge agent tests
│   │   └── test_web_search.py   # Web Search agent tests
│   │
│   ├── utils/                   # Utility & configuration tests
│   │   ├── test_config.py       # ExperimentConfig tests
│   │   ├── test_wandb_utils.py  # W&B utilities tests
│   │   └── test_train_test_splitter.py # Train/test split tests
│   │
│   ├── examples/                # Example tests for contributors [T077]
│   │   └── test_example_agent_test.py
│   │
│   ├── __init__.py
│   └── conftest.py              # Unit-specific fixtures (optional)
│
├── integration/                 # Integration tests (multi-component)
│   ├── test_agent_workflow.py   # Agent-to-agent communication [requires_llm]
│   ├── test_task_io.py          # YAML/CSV read-write functionality
│   ├── test_orchestration.py    # rogue_one_main.py end-to-end integration
│   ├── test_localhost_availability.py # LLM endpoint health checks
│   ├── __init__.py
│   └── conftest.py              # Integration-specific fixtures (optional)
│
└── __init__.py
```

## Running Tests

### Quick Start

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=rogueone --cov-report=term-missing

# Run specific test suite
pytest tests/unit/agents/
pytest tests/integration/
```

### By Test Type

```bash
# Unit tests only (no LLM service required)
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Only tests that don't require LLM
pytest -m "not requires_llm"
```

### By Functionality

```bash
# All agent tests (unit + integration)
pytest -m agent

# All configuration tests
pytest -m config

# All LLM-required tests
pytest -m requires_llm

# Slow tests only
pytest -m slow
```

### Advanced Options

```bash
# Stop on first failure
pytest tests/ -x

# Show local variables on failure
pytest tests/ -l

# Run with short traceback
pytest tests/ --tb=short

# Parallel execution (requires pytest-xdist)
pytest tests/ -n auto

# Run specific test by name
pytest tests/ -k "test_scientist"

# Without capturing output
pytest tests/ -s
```

## Test Markers

Markers enable filtering and organization. Defined in `pytest.ini`:

| Marker | Purpose | Example |
|--------|---------|---------|
| `@pytest.mark.unit` | Unit tests (no external dependencies) | `test_config.py` |
| `@pytest.mark.integration` | Integration tests (multi-component) | `test_agent_workflow.py` |
| `@pytest.mark.agent` | Agent-specific tests | All `tests/unit/agents/*` |
| `@pytest.mark.config` | Configuration tests | `test_config.py` |
| `@pytest.mark.task_io` | Task I/O functionality | `test_task_io.py` |
| `@pytest.mark.requires_llm` | Need localhost:8000 LLM service | `test_agent_workflow.py` |
| `@pytest.mark.slow` | Long-running tests | Reserved for future use |

Examples:

```bash
# Run all agent tests
pytest -m agent

# Run all tests EXCEPT those requiring LLM
pytest -m "not requires_llm"

# Run integration AND agent tests
pytest -m "integration and agent"
```

## Shared Fixtures

### Session Scope (Initialized Once at Startup)

#### `llm_endpoint_health`
Returns endpoint availability status before any tests run.

```python
{
  "available": bool,           # True if endpoint healthy
  "endpoint_url": str,         # "http://localhost:8000"
  "latency_ms": float,         # Response time in milliseconds
  "error": str                 # Error message if unavailable
}
```

Behavior:
- Checks `http://localhost:8000` at session startup
- Retries 3 times with 1-second delay between attempts
- Logs explicit "LLM ENDPOINT UNAVAILABLE" message with instructions when endpoint down
- Returns status dict to all tests

### Function Scope (Reset for Each Test)

#### `llm_required`
Validates LLM endpoint available before running tests marked `@pytest.mark.requires_llm`.

```python
@pytest.mark.requires_llm
def test_agent_with_llm(llm_required):
    # Test only runs if llm_endpoint_health["available"] == True
    # Otherwise test fails with explicit error
    pass
```

#### `config_fixture`
Generic ExperimentConfig instance for testing.

```python
def test_config_loading(config_fixture):
    assert config_fixture.task_type in ["classification", "regression"]
```

#### `classification_task_config` & `regression_task_config`
Pre-configured task-specific config fixtures.

```python
def test_both_task_types(classification_task_config, regression_task_config):
    assert classification_task_config.target == "income"
    assert regression_task_config.target == "price"
```

#### `sample_dataframe`, `sample_classification_target`, `sample_regression_target`
Test datasets for parametrized tests.

```python
@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_with_data(sample_dataframe, task_type):
    assert len(sample_dataframe) >= 10
```

See `conftest.py` for detailed fixture implementations.

## Test Data

Minimal test data is provided in `/tasks/test_classification/` and `/tasks/test_regression/`:

| File | Type | Rows | Target | Purpose |
|------|------|------|--------|---------|
| `test_classification/config.yml` | YAML | - | income | Classification task config |
| `test_classification/data.csv` | CSV | 10 | income | Sample classification data |
| `test_regression/config.yml` | YAML | - | price | Regression task config |
| `test_regression/data.csv` | CSV | 10 | price | Sample regression data |

These are intentionally minimal to ensure fast test execution.

## Writing Tests

### 1. Unit Test Template (No LLM)

```python
import pytest
from rogueone.llm.agents.scientist import ScientistAgent

@pytest.mark.unit
@pytest.mark.agent
class TestScientistAgent:
    
    def test_scientist_agent_exists(self):
        """Agent must be importable"""
        assert ScientistAgent is not None
    
    def test_scientist_output_contract(self):
        """Output must match contract schema"""
        agent = ScientistAgent()
        result = agent.run(prompt="test")
        
        # Validate required fields from agent-behavior-validation-contract.md
        assert isinstance(result, dict)
        assert "hypotheses" in result
        assert isinstance(result["hypotheses"], list)
    
    def test_scientist_handles_empty_input(self):
        """Graceful degradation: empty input does not crash"""
        agent = ScientistAgent()
        result = agent.run(prompt="")
        assert result is not None
```

### 2. Parametrized Test (Multi-Task Support)

```python
@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_extractor_both_task_types(task_type, config_fixture):
    """Constitution IV: Verify task-agnostic behavior"""
    config_fixture.task_type = task_type
    agent = ExtractorAgent(config=config_fixture)
    result = agent.run(data=sample_data)
    
    # Verify output valid for both task types
    assert "features" in result
    assert result["task_type"] == task_type
```

### 3. Integration Test (With LLM)

```python
@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.requires_llm
def test_agent_workflow_with_llm(llm_required):
    """Full workflow with actual LLM service"""
    from rogueone.rogue_one_main import main
    
    # llm_required fixture fails test if endpoint unavailable
    assert llm_required is not None
    
    # Run full workflow
    result = main(task_config="tasks/test_classification/config.yml")
    assert result["success"] == True
```


## Coverage Goals

| Component | Target | Current | Status |
|-----------|--------|---------|--------| Agents | 80% | [See report] | Phase dependent |
| Utilities | 70% | [See report] | Phase dependent |
| Overall | 60% | [See report] | Phase dependent |

Generate coverage report:

```bash
pytest tests/ --cov=rogueone --cov-report=html --cov-report=term-missing

# Open in browser
open htmlcov/index.html
```

## Running Tests in Docker

The project includes a `Dockerfile` configured for testing. Run tests in the exact CI environment:

```bash
# Build image with all dependencies
docker build -t rougeone:test .

# Run tests in container (uses localhost:8000 if available in docker network)
docker run --rm \
    -v $(pwd):/workspaces/RougeOne \
    --network host \
    rougeone:test \
    pytest tests/ -v

# Run with coverage
docker run --rm \
    -v $(pwd):/workspaces/RougeOne \
    --network host \
    rougeone:test \
    pytest tests/ --cov=rogueone
```

For LLM service in Docker, use `--network host` or `--link` to allow access to localhost:8000:

```bash
# Terminal 1: Start LLM service
docker run -p 8000:8000 vllm/vllm-openai:v0.11.0 \
    --model qwen/qwen2-7b \
    --api-key rogueone-dev

# Terminal 2: Run tests with LLM access
docker run --rm \
    -v $(pwd):/workspaces/RougeOne \
    --network host \
    rougeone:test \
    pytest tests/ -m "requires_llm" -v
```

## Troubleshooting

### LLM Endpoint Unavailable

**Symptom**: Tests fail with "LLM ENDPOINT UNAVAILABLE"

**Solution**: 
1. Start LLM service: `docker run -p 8000:8000 vllm/vllm-openai:v0.11.0 --model qwen/qwen2-7b --api-key rogueone-dev`
2. Verify endpoint: `curl -H "Authorization: Bearer rogueone-dev" http://localhost:8000/health`
3. Run non-LLM tests: `pytest -m "not requires_llm"`

### Import Errors in Tests

**Symptom**: `ModuleNotFoundError: No module named 'rogueone'`

**Solution**:
1. Ensure package installed: `pip install -e .` from project root
2. Check PYTHONPATH: `export PYTHONPATH=/workspaces/RougeOne:$PYTHONPATH`
3. Verify conftest.py location (must be in tests/ directory)

### Timeout Errors

**Symptom**: `TimeoutError: test session interrupted by timeout after 300.0 seconds`

**Solution**:
1. Increase timeout in `pytest.ini`: Change `timeout = 300` to higher value
2. Mark slow tests: `@pytest.mark.slow`, then run without them: `pytest -m "not slow"`
3. Use `pytest -k "specific_test"` to run individual tests

### Marker Not Registered

**Symptom**: `PytestUnknownMarkWarning: unknown pytest.mark.custom_marker`

**Solution**:
1. Register marker in `pytest.ini` under `[pytest]` `markers =` list
2. Use only markers from `[pytest]` configuration

## Best Practices

1. **Test Isolation**: Each test must be independent; no test should depend on order
2. **Descriptive Names**: Use clear names: `test_scientist_handles_empty_input` not `test_1`
3. **One Assert Per Test** (ideally): Multiple asserts OK if validating one behavior
4. **Fixtures > Globals**: Use fixtures for setup/teardown, not module-level globals
5. **Mock External Services**: Use `unittest.mock` to avoid real API calls
6. **Skip vs. Xfail**: 
   - `pytest.skip("reason")` - Test not yet implemented
   - `pytest.xfail("reason")` - Known failure, expect to fail
   - Skip preferred for contract validation tests
7. **Parametrize for Variants**: Use `@pytest.mark.parametrize` for multi-task testing
8. **Fast Tests**: Keep unit tests <100ms; use minimal test data
9. **Explicit Failure**: Always include error message in assertions: `assert x, "Expected x to be..."`
10. **Document Contracts**: Add docstring explaining which contract(s) each test validates

## Contributing Tests

1. **Before Merging**: All tests must pass: `pytest tests/ -v`
2. **Add Tests First**: Follow TDD: write test before implementation
3. **Update this README**: If new test patterns or requirements added
4. **Check Coverage**: `pytest --cov=rogueone --cov-report=term-missing`
5. **Run in Docker**: Validate in container: `docker run ... pytest tests/ -v`
6. **Document Your Markers**: If using new markers, add to pytest.ini and this README

## Resources

- [pytest.ini](../pytest.ini) — Marker definitions & configuration
- [conftest.py](./conftest.py) — Shared fixture implementations
- [examples/test_example_agent_test.py](./unit/examples/test_example_agent_test.py) — Example test patterns
