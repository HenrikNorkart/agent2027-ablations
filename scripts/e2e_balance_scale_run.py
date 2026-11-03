import asyncio
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rogue_one_main import RogueOneAgentNetwork
from rogueone.utils.config import ExperimentConfig


def main() -> None:
    cfg = ExperimentConfig.from_yaml("tasks/classification/balance-scale/config.yml")
    agent = RogueOneAgentNetwork(cfg, enable_wandb=False)
    try:
        asyncio.run(agent.main())
        print("RUN_OK")
    except Exception as exc:
        print("RUN_FAIL", type(exc).__name__, exc)
        traceback.print_exc()


if __name__ == "__main__":
    main()
