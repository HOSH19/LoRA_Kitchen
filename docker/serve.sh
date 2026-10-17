#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_LORAS="${MAX_LORAS:-3}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
ADAPTER_ROOT="${ADAPTER_ROOT:-/adapters}"

LORAS=()
if [[ -d "${ADAPTER_ROOT}/flame" ]]; then
  LORAS+=(--lora-modules "flame=${ADAPTER_ROOT}/flame")
fi
if [[ -d "${ADAPTER_ROOT}/butter" ]]; then
  LORAS+=(--lora-modules "butter=${ADAPTER_ROOT}/butter")
fi
if [[ -d "${ADAPTER_ROOT}/chaos" ]]; then
  LORAS+=(--lora-modules "chaos=${ADAPTER_ROOT}/chaos")
fi

if [[ ${#LORAS[@]} -eq 0 ]]; then
  echo "No adapters found under ${ADAPTER_ROOT}/{flame,butter,chaos}" >&2
  exit 1
fi

exec vllm serve "${BASE_MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --enable-lora \
  --max-loras "${MAX_LORAS}" \
  --max-lora-rank "${MAX_LORA_RANK}" \
  --enable-prefix-caching \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code \
  "${LORAS[@]}"
