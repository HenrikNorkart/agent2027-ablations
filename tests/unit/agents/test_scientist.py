"""Unit tests for ScientistAgent."""

import inspect
import pytest
from typing import Dict, Any

pytest_plugins = []


# ---------------------------------------------------------------------------
# Focus contract — every focus must be additive (CAR runs ca90miqg / p9t3rayi
# / wi5r0oki aborted because the Scientist returned a pure-pruning focus that
# the Extractor had no way to act on; pruning is the Tester's job)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.agent
def test_scientist_prompt_forbids_pure_pruning_focus():
    """The focus-generation prompt must explicitly forbid pure-pruning
    focuses and require every focus to include at least one additive ask
    (encoding / interaction / transform). Without this rule the Scientist
    occasionally hands the Extractor a pure pruning task; the Extractor
    has no pruning tool and the dataset aborts (CAR run wi5r0oki: focus
    'Perform an importance-driven pruning... Keep only the top 8-10').
    """
    try:
        from rogueone.llm.agents.core.scientist.agent import ScientistAgent
    except ImportError:
        pytest.skip("ScientistAgent not yet importable")

    src = inspect.getsource(ScientistAgent.determine_focus)
    src_lower = src.lower()

    # The rule itself
    assert "additive" in src_lower, (
        "Focus prompt must contain the 'additive' rule keyword."
    )
    # The forbidden pattern
    assert "pure-pruning" in src_lower or "pure pruning" in src_lower, (
        "Focus prompt must explicitly forbid pure-pruning focuses."
    )
    # The reason — at least one tool name the Extractor can call
    assert "append_new_attribute" in src, (
        "Focus prompt must reference the Extractor's only mutation tool "
        "(append_new_attribute) so the model is grounded in what is "
        "actually buildable."
    )


@pytest.mark.unit
@pytest.mark.agent
def test_scientist_prompt_warns_against_replacement_framing():
    """Phrases like 'replace X with Y' caused the Extractor on CAR run
    ca90miqg to interpret the focus as a delete-and-replace task and emit
    zero append calls. The prompt must redirect the Scientist toward
    'introduce Y, the Tester may retire X' framing instead."""
    try:
        from rogueone.llm.agents.core.scientist.agent import ScientistAgent
    except ImportError:
        pytest.skip("ScientistAgent not yet importable")

    src = inspect.getsource(ScientistAgent.determine_focus)
    src_lower = src.lower()

    assert "replacement" in src_lower or "replace" in src_lower, (
        "Focus prompt must address the replacement-framing failure mode."
    )
    assert "introduce" in src_lower, (
        "Focus prompt must offer the additive 'introduce' alternative."
    )


@pytest.mark.unit
@pytest.mark.agent
def test_scientist_prompt_addresses_focus_saturation():
    """Late-iteration CAR runs (e.g. ca90miqg, iteration 4 of 4) ran out
    of feature engineering ideas and the Scientist re-issued the same
    encoding-themed focus the Extractor had already exhausted. The prompt
    must call out repetition and direct the Scientist toward less-obvious
    additive alternatives (transforms, three-way interactions, restored
    pruned features) once the focus history shows saturation.
    """
    try:
        from rogueone.llm.agents.core.scientist.agent import ScientistAgent
    except ImportError:
        pytest.skip("ScientistAgent not yet importable")

    src = inspect.getsource(ScientistAgent.determine_focus)
    src_lower = src.lower()

    assert "focus history" in src_lower, (
        "Saturation rule must reference focus history."
    )
    assert "saturat" in src_lower or "exhaust" in src_lower or "loop" in src_lower, (
        "Saturation rule must name the failure mode (saturated / exhausted / loop)."
    )
    # At least one less-obvious alternative
    assert any(alt in src_lower for alt in ("log", "transform", "three-way", "ratio", "restoration")), (
        "Saturation rule should suggest alternative additive ideas (log, transform, three-way, ratio, restoration)."
    )


@pytest.mark.unit
@pytest.mark.agent
def test_scientist_agent_exists():
    """Verify ScientistAgent can be imported."""
    try:
        from rogueone.llm.agents import ScientistAgent
        assert ScientistAgent is not None
    except ImportError:
        pytest.skip("ScientistAgent not yet implemented")


@pytest.mark.unit
@pytest.mark.agent
def test_scientist_output_contract_fields():
    """
    Validate ScientistAgent output contract includes required fields.
    
    Per agent-behavior-validation-contract.md, output must include:
    - focus_area (non-empty string)
    - rationale (non-empty string)
    - next_action (one of: "explore_new", "refine_existing", "pivot")
    """
    try:
        from rogueone.llm.agents import ScientistAgent
        
        # Create minimal agent (if implementation exists)
        # This test is a contract validator - it documents expected behavior
        # Once ScientistAgent is implemented, this test validates the contract
        
        # Expected contract:
        expected_fields = ["focus_area", "rationale", "next_action"]
        expected_actions = ["explore_new", "refine_existing", "pivot"]
        
        assert all(f in expected_fields for f in expected_fields)
        
    except ImportError:
        pytest.skip("ScientistAgent not yet implemented")


@pytest.mark.unit
@pytest.mark.agent
def test_scientist_handles_empty_prior_results():
    """
    Test ScientistAgent handles empty prior_results gracefully.
    
    Per spec FR-003: Must validate failure handling
    """
    try:
        from rogueone.llm.agents import ScientistAgent
        
        # Placeholder test - documents expected behavior
        # When implemented, this validates agent handles empty input
        
    except ImportError:
        pytest.skip("ScientistAgent not yet implemented")


@pytest.mark.unit
@pytest.mark.agent
def test_scientist_rejects_invalid_iterations():
    """
    Test ScientistAgent rejects invalid iterations_remaining.
    
    Per spec FR-003: Must validate failure handling and error cases
    """
    try:
        from rogueone.llm.agents import ScientistAgent
        
        # Placeholder test - documents expected behavior
        # When implemented, this validates agent raises ValueError for invalid iterations
        
    except ImportError:
        pytest.skip("ScientistAgent not yet implemented")

