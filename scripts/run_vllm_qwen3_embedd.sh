#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/../.env}"

if [[ -f "${ENV_FILE}" ]]; then
    set -a
    source "${ENV_FILE}"
    set +a
fi

: "${EMBEDD_API_KEY:?EMBEDD_API_KEY must be set in the environment or ${ENV_FILE}}"
: "${EMBEDD_PORT:?EMBEDD_PORT must be set in the environment or ${ENV_FILE}}"
: "${EMBEDD_MODEL:?EMBEDD_MODEL must be set in the environment or ${ENV_FILE}}"
: "${EMBEDD_GPUS:?EMBEDD_GPUS must be set in the environment or ${ENV_FILE}}"

API_KEY="${EMBEDD_API_KEY}"
PORT="${EMBEDD_PORT}"
MODEL_NAME="${EMBEDD_MODEL}"
GPUS="${EMBEDD_GPUS}"

CUDA_VISIBLE_DEVICES=$GPUS vllm serve \
    --tensor-parallel-size $(( $(echo "$GPUS" | awk -F',' '{print NF}') )) \
    --api-key "${API_KEY}" \
    --dtype auto \
    --port "${PORT}" \
    --task embedding \
    --gpu_memory_utilization 0.5 \
    "${MODEL_NAME}"