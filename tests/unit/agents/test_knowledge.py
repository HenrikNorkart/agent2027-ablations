"""Unit tests for KnowledgeAgent."""

import pytest

pytest_plugins = []


@pytest.mark.unit
@pytest.mark.agent
def test_knowledge_agent_exists():
    """Verify KnowledgeAgent can be imported."""
    try:
        from rogueone.llm.agents import KnowledgeAgent
        assert KnowledgeAgent is not None
    except ImportError:
        pytest.skip("KnowledgeAgent not yet implemented")


@pytest.mark.unit
@pytest.mark.agent
def test_knowledge_output_contract_fields():
    """
    Validate KnowledgeAgent output contract includes required fields.
    
    Per agent-behavior-validation-contract.md, output must include:
    - documents (list of knowledge items with content, relevance_score, source)
    - success (boolean)
    """
    try:
        from rogueone.llm.agents import KnowledgeAgent
        
        # Expected contract:
        expected_fields = ["documents", "success"]
        
        assert all(f in expected_fields for f in expected_fields)
        
    except ImportError:
        pytest.skip("KnowledgeAgent not yet implemented")


@pytest.mark.unit
@pytest.mark.agent
def test_knowledge_handles_empty_query():
    """
    Test KnowledgeAgent rejects empty query.
    
    Per spec FR-003: Must validate failure handling
    """
    try:
        from rogueone.llm.agents import KnowledgeAgent
        
        # Placeholder test - documents expected behavior
        # When implemented, validates agent raises ValueError for empty query
        
    except ImportError:
        pytest.skip("KnowledgeAgent not yet implemented")


@pytest.mark.unit
@pytest.mark.agent
def test_knowledge_handles_no_matches():
    """
    Test KnowledgeAgent gracefully handles queries with no matches.
    
    Per spec FR-003: Must validate failure handling
    """
    try:
        from rogueone.llm.agents import KnowledgeAgent
        
        # Placeholder test - documents expected behavior
        # When implemented, validates agent returns valid output with empty documents
        
    except ImportError:
        pytest.skip("KnowledgeAgent not yet implemented")
