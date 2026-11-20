"""Compatibility wrapper for knowledge-agent internals."""

from rogueone.llm.agents.core.knowledge.agent import (
    KnowledgeAgent,
    _run_explanation_agent,
)

__all__ = ["KnowledgeAgent", "_run_explanation_agent"]
