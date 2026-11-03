"""Compatibility wrapper for TesterAgent.

Public import path is preserved:
`from rogueone.llm.agents import TesterAgent`
"""

from rogueone.llm.agents.core.tester.agent import (
    TestResultClassification,
    TestResultRegression,
    TesterAgent,
)
from rogueone.llm.agents.core.tester.metrics import FeatureMetricProfile

__all__ = [
    "TesterAgent",
    "TestResultClassification",
    "TestResultRegression",
    "FeatureMetricProfile",
]
