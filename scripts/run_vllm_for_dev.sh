
GPUS="6,7"
PORT="8000"
API_KEY="rogueone-dev"
MODEL_NAME="openai/gpt-oss-120b"

CUDA_VISIBLE_DEVICES=$GPUS vllm serve \
    --tensor-parallel-size $(( $(echo "$GPUS" | awk -F',' '{print NF}') )) \
    --api-key "${API_KEY}" \
    --dtype auto \
    --port "${PORT}" \
    "${MODEL_NAME}"