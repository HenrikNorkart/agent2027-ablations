# Agent Core Modules

This directory contains internal implementations for RougeOne agents.

Compatibility policy:
- External imports should continue to use `rogueone.llm.agents`.
- Top-level modules in `rogueone/llm/agents/` act as wrappers/re-exports.

Structure:
- `extractor/agent.py`
- `scientist/agent.py`
- `tester/agent.py`
- `knowledge/agent.py`
- `knowledge/web_search.py`
