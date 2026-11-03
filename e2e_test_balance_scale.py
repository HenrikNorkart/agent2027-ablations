#!/usr/bin/env python3
"""
E2E Test: Balance-scale classification with fixed fold iteration logic.

This script runs the RogueOneAgentNetwork against an already-running LLM
on port 8000 without attempting to start vLLM servers.
"""
import asyncio
from pathlib import Path
from rogueone.utils.config import ExperimentConfig
from rogue_one_main import RogueOneAgentNetwork

async def main():
    config_path = Path("tasks/classification/balance-scale/config.yml")
    cfg = ExperimentConfig.from_yaml(config_path)
    network = RogueOneAgentNetwork(cfg)
    await network.main()

if __name__ == "__main__":
    asyncio.run(main())
