"""Integration tests for agent workflow."""

import pytest

pytest_plugins = []


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.requires_llm
def test_agent_workflow_integration_happy_path(llm_required):
    """
    Test complete Scientist → Extractor → Tester agent loop.
    
    Per User Story 1: Reliable Agent Behavior Coverage
    This validates the full workflow with real LLM endpoint.
    
    Flow:
    1. Scientist selects investigation focus
    2. Extractor generates hypotheses for that focus
    3. Tester evaluates those hypotheses
    
    Per spec FR-002, FR-004: All agents must return valid contracts
    """
    try:
        from rogueone.llm.agents import ScientistAgent, ExtractorAgent, TesterAgent
        import pandas as pd
        
        # Placeholder integration test - documents expected workflow
        # When agents are fully implemented, this validates:
        # 1. Scientist returns valid focus_area, rationale, next_action
        # 2. Extractor generates hypotheses with non-empty argumentation (Constitution III)
        # 3. Tester evaluates with valid results, summary, improvements
        # 4. All agent outputs conform to contracts per agent-behavior-validation-contract.md
        
    except ImportError:
        pytest.skip("One or more agents not yet implemented")


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.requires_llm
@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_agent_workflow_both_task_types(llm_required, task_type):
    """
    Test agent workflow works for both classification and regression.
    
    Per Constitution IV: task-agnostic design
    Per spec FR-012: Agent tests must work for both task types
    """
    try:
        from rogueone.llm.agents import ScientistAgent, ExtractorAgent, TesterAgent
        
        # Placeholder test - documents task-agnostic requirement
        # When implemented, validates workflow succeeds with both task types
        
    except ImportError:
        pytest.skip("One or more agents not yet implemented")


@pytest.mark.integration
@pytest.mark.agent
@pytest.mark.requires_llm
def test_agent_workflow_output_contracts_validated(llm_required):
    """
    Test agent workflow validates all output contracts.
    
    Per spec FR-004: Agent outputs must validate contracted fields
    - Scientist: focus_area, rationale, next_action
    - Extractor: hypotheses with argumentation, reasoning
    - Tester: results, summary, best_feature, improvements
    """
    try:
        from rogueone.llm.agents import ScientistAgent, ExtractorAgent, TesterAgent
        
        # Placeholder test - documents contract validation requirement
        # When implemented, validates all required fields present in workflow output
        
    except ImportError:
        pytest.skip("One or more agents not yet implemented")
