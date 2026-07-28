#!/usr/bin/env bash
set -euo pipefail

MODEL="${FRAMEGUARD_MODEL:-Qwen/Qwen2.5-Omni-3B}"
PORT="${FRAMEGUARD_MODEL_PORT:-8091}"
HOST="${FRAMEGUARD_MODEL_HOST:-0.0.0.0}"

exec vllm serve "$MODEL" \
  --omni \
  --host "$HOST" \
  --port "$PORT"
