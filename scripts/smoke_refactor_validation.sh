#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CLASSIFICATION_CFG="tasks/classification/adult/config.yml"
REGRESSION_CFG="tasks/regression/bike/config.yml"
TMP_CLASSIFICATION_CFG="/tmp/rougeone-smoke-classification.yml"
TMP_REGRESSION_CFG="/tmp/rougeone-smoke-regression.yml"

if [[ ! -f "$CLASSIFICATION_CFG" ]]; then
  echo "Missing classification config: $CLASSIFICATION_CFG" >&2
  exit 1
fi

if [[ ! -f "$REGRESSION_CFG" ]]; then
  echo "Missing regression config: $REGRESSION_CFG" >&2
  exit 1
fi

# Normalize repository path prefixes for container layouts where configs use /workspace
sed "s#/workspace/#$ROOT_DIR/#g" "$CLASSIFICATION_CFG" > "$TMP_CLASSIFICATION_CFG"
sed "s#/workspace/#$ROOT_DIR/#g" "$REGRESSION_CFG" > "$TMP_REGRESSION_CFG"

python3 - <<'PY'
import asyncio
from pathlib import Path

from rogueone.utils.config import ExperimentConfig
from rogue_one_main import RogueOneAgentNetwork

async def run_one(path: str):
    cfg = ExperimentConfig.from_yaml(Path(path))
    network = RogueOneAgentNetwork(cfg)
    await network.forward_pass(0)

async def main():
  await run_one("/tmp/rougeone-smoke-classification.yml")
  await run_one("/tmp/rougeone-smoke-regression.yml")

asyncio.run(main())
PY

echo "Smoke validation completed for classification and regression configs."
