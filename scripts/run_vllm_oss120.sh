#!/bin/bash

# set -euo pipefail

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/../.env}"

# if [[ -f "${ENV_FILE}" ]]; then
#     set -a
#     source "${ENV_FILE}"
#     set +a
# fi

# : "${LLM_API_KEY:?LLM_API_KEY must be set in the environment or ${ENV_FILE}}"
# : "${LLM_PORT:?LLM_PORT must be set in the environment or ${ENV_FILE}}"
# : "${LLM_MODEL:?LLM_MODEL must be set in the environment or ${ENV_FILE}}"
# : "${LLM_GPUS:?LLM_GPUS must be set in the environment or ${ENV_FILE}}"

source .env

API_KEY="${LLM_API_KEY}"
PORT="${LLM_PORT}"
MODEL_NAME="${LLM_MODEL}"
GPUS="${LLM_GPUS}"


CUDA_VISIBLE_DEVICES=$GPUS vllm serve \
    --tensor-parallel-size $(( $(echo "$GPUS" | awk -F',' '{print NF}') )) \
    --api-key "${API_KEY}" \
    --dtype auto \
    --port "${PORT}" \
    "${MODEL_NAME}"