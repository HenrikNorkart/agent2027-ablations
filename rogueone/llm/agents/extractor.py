"""Compatibility wrapper for ExtractorAgent.

Public import path is preserved:
`from rogueone.llm.agents import ExtractorAgent`
"""

from rogueone.llm.agents.core.extractor.agent import (
    AttributeExplainingArgs,
    ExtractorAgent,
    PandasCommandArgs,
)

__all__ = ["ExtractorAgent", "AttributeExplainingArgs", "PandasCommandArgs"]
