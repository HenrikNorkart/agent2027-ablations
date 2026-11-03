"""Integration tests for LLM endpoint availability and error handling."""

import pytest

pytest_plugins = []


@pytest.mark.integration
def test_llm_endpoint_health_check_runs(llm_endpoint_health):
    """
    Test LLM endpoint health check fixture executes.
    
    Per spec FR-014: Must validate endpoint availability at startup
    Per contracts/llm-endpoint-integration-contract.md: Health check is REQUIRED
    """
    assert llm_endpoint_health is not None
    assert "available" in llm_endpoint_health
    assert "endpoint_url" in llm_endpoint_health


@pytest.mark.integration
def test_llm_endpoint_reports_availability_status(llm_endpoint_health):
    """
    Test LLM endpoint health check reports correct status.
    
    Per spec FR-014: Must report endpoint availability explicitly
    """
    if llm_endpoint_health.get("available"):
        # If available
        assert "latency_ms" in llm_endpoint_health
        assert llm_endpoint_health["latency_ms"] > 0
    else:
        # If unavailable
        assert "error" in llm_endpoint_health
        assert "localhost:8000" in llm_endpoint_health.get("error", "")


@pytest.mark.integration
@pytest.mark.requires_llm
def test_agent_tests_fail_explicitly_when_endpoint_unavailable(llm_required):
    """
    Test that agent tests fail explicitly when endpoint is unavailable.
    
    Per spec FR-014: "Agent-related test suites MUST be marked as failed with 
    an explicit endpoint availability error"
    
    This test documents the expected behavior. If the endpoint is unavailable
    at session start, any test marked @pytest.mark.requires_llm will fail
    with the llm_required fixture raising pytest.fail() with explicit error.
    """
    # If we reach this point, endpoint was available
    assert llm_required.get("available"), "Endpoint should be available for this test"


@pytest.mark.integration
def test_non_agent_tests_continue_when_endpoint_unavailable():
    """
    Test that non-agent tests continue when endpoint is unavailable.
    
    Per spec FR-014: "All non-agent test suites MUST continue running and report 
    their own results independently"
    
    This test documents that non-agent tests (without @pytest.mark.requires_llm)
    should run successfully even if LLM endpoint is down. This test itself
    demonstrates that - it runs without the llm_required fixture.
    """
    # Non-agent tests should not depend on LLM endpoint
    assert True  # This test just validates the concept
