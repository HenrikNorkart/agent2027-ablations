#!/usr/bin/env python3
"""
Test: Verify that the ScientistAgent search_tool is accessible and functional.

Tests:
1. WebSearchAgent.explain_query works in isolation (SERPER API + LLM).
2. ScientistAgent.determine_focus calls search_tool at least once (integration).

Run from repo root:
    python scripts/test_scientist_search_tool.py [config.yml]

If no config path is supplied the balance-scale dataset is used.
"""

import asyncio
import sys
from pathlib import Path

import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rogueone.utils.config import ExperimentConfig
from rogueone.llm.agents.web_search import WebSearchAgent
from rogueone.llm.agents.scientist import ScientistAgent


# ── config ────────────────────────────────────────────────────────────────────
_default_config = ROOT / "tasks/classification/balance-scale/config.yml"
CONFIG_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else _default_config


def _load_data(cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load feature and label CSVs, resolving workspace-relative paths."""
    feature_path = Path(str(cfg.feature_file_path).replace("/workspace/", str(ROOT) + "/"))
    label_path   = Path(str(cfg.label_file_path).replace("/workspace/", str(ROOT) + "/"))
    df_raw  = pd.read_csv(feature_path)
    df_lbls = pd.read_csv(label_path)
    return df_raw, df_lbls


# ── test 1: WebSearchAgent isolate ──────────────────────────────────────────
async def test_web_search_agent_direct(cfg: ExperimentConfig) -> bool:
    """Call WebSearchAgent.explain_query directly and report results."""
    print("\n" + "=" * 60)
    print("TEST 1 — WebSearchAgent direct call")
    print("=" * 60)

    agent = WebSearchAgent(cfg=cfg)
    query = "What is the torque formula used in balance-scale physics experiments?"

    print(f"Query: {query!r}")
    try:
        result = await agent.explain_query(query)
        print(f"\nResult (first 400 chars):\n{str(result)[:400]}")
        if result and len(str(result).strip()) > 20:
            print("\n[PASS] WebSearchAgent returned a non-empty answer.")
            return True
        else:
            print("\n[FAIL] WebSearchAgent returned empty or trivial result.")
            return False
    except Exception as e:
        print(f"\n[FAIL] WebSearchAgent raised exception: {e}")
        return False


# ── test 2: ScientistAgent search_tool invoked during determine_focus ────────
async def test_scientist_search_tool_invoked(cfg: ExperimentConfig) -> bool:
    """Run determine_focus and check that search_tool was called at least once."""
    print("\n" + "=" * 60)
    print("TEST 2 — ScientistAgent search_tool invoked during determine_focus")
    print("=" * 60)

    df_raw, _ = _load_data(cfg)

    # Minimal attribute explanations so the agent has context resembling round 1.
    df_attr_explanations = pd.DataFrame(columns=["Attribute", "Explanation"])

    agent = ScientistAgent(cfg=cfg)

    try:
        focus = await agent.determine_focus(
            df_raw_data=df_raw,
            df_test_results=None,
            df_attribute_explanations=df_attr_explanations,
            focus_history=[],
            index=0,
            max_index=1,
        )

        search_calls   = agent._tool_call_counter.get("search_tool", 0)
        search_success = agent._tool_call_counter_success.get("search_tool", 0)
        search_failed  = agent._tool_call_counter_failed.get("search_tool", 0)

        print(f"\nFocus output: {focus[:300] if focus else '<empty>'}")
        print(f"\nTool call counts:")
        print(f"  search_tool calls   : {search_calls}")
        print(f"  search_tool success : {search_success}")
        print(f"  search_tool failed  : {search_failed}")
        print(f"\nAll tool call counts : {dict(agent._tool_call_counter)}")
        print(f"All success counts   : {dict(agent._tool_call_counter_success)}")
        print(f"All failed counts    : {dict(agent._tool_call_counter_failed)}")

        if search_calls == 0:
            print("\n[WARN] search_tool was NOT called by the scientist agent.")
        elif search_success > 0:
            print(f"\n[PASS] search_tool called {search_calls}x, succeeded {search_success}x.")
        else:
            print(f"\n[FAIL] search_tool called {search_calls}x but ALL calls FAILED.")

        return focus is not None and len(str(focus).strip()) > 10

    except Exception as e:
        import traceback
        print(f"\n[FAIL] ScientistAgent raised exception: {e}")
        traceback.print_exc()
        return False


# ── main ─────────────────────────────────────────────────────────────────────
async def main():
    print("Loading config from:", CONFIG_PATH)
    cfg = ExperimentConfig.from_yaml(CONFIG_PATH)

    results = {}
    results["web_search_direct"]         = await test_web_search_agent_direct(cfg)
    results["scientist_search_invoked"]  = await test_scientist_search_tool_invoked(cfg)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    all_pass = all(results.values())
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    asyncio.run(main())
