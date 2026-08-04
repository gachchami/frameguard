#!/usr/bin/env bash
set -Eeuo pipefail

# Start the complete local FrameGuard stack:
#   1. Qwen2.5-Omni through the AMD/ROCm vLLM environment
#   2. FrameGuard through the repository's uv environment
#
# Place this file at:
#   <frameguard-repo>/scripts/start_frameguard_stack.sh
#
# Optional overrides:
#   FRAMEGUARD_MODEL_PATH=/path/to/Qwen2.5-Omni-3B
#   FRAMEGUARD_LOG_LEVEL=DEBUG
#   FRAMEGUARD_HOST=127.0.0.1
#   FRAMEGUARD_PORT=7860
#   FRAMEGUARD_API_PORT=8091
#   FRAMEGUARD_SYNC=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

VLLM_BIN="${VLLM_BIN:-/opt/venv/bin/vllm}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"

FRAMEGUARD_HOST="${FRAMEGUARD_HOST:-127.0.0.1}"
FRAMEGUARD_PORT="${FRAMEGUARD_PORT:-7860}"
FRAMEGUARD_API_HOST="${FRAMEGUARD_API_HOST:-127.0.0.1}"
FRAMEGUARD_API_PORT="${FRAMEGUARD_API_PORT:-8091}"
FRAMEGUARD_LOG_LEVEL="${FRAMEGUARD_LOG_LEVEL:-INFO}"
FRAMEGUARD_SYNC="${FRAMEGUARD_SYNC:-0}"

FACE_MODEL="${FRAMEGUARD_FACE_MODEL:-${REPO_ROOT}/models/face_detection_yunet_2023mar.onnx}"
FACE_RECOGNITION_MODEL="${FRAMEGUARD_FACE_RECOGNITION_MODEL:-${REPO_ROOT}/models/face_recognition_sface_2021dec.onnx}"

if [[ -n "${FRAMEGUARD_MODEL_PATH:-}" ]]; then
    MODEL_PATH="${FRAMEGUARD_MODEL_PATH}"
elif [[ -d "/workspace/persistent/Qwen2.5-Omni-3B" ]]; then
    MODEL_PATH="/workspace/persistent/Qwen2.5-Omni-3B"
elif [[ -d "/persistent/Qwen2.5-Omni-3B" ]]; then
    MODEL_PATH="/persistent/Qwen2.5-Omni-3B"
else
    echo "ERROR: Qwen2.5-Omni model directory was not found." >&2
    echo "Set FRAMEGUARD_MODEL_PATH to its local directory." >&2
    exit 1
fi

RUNTIME_DIR="${REPO_ROOT}/outputs/runtime"
mkdir -p "${RUNTIME_DIR}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
VLLM_LOG="${RUNTIME_DIR}/vllm_${RUN_STAMP}.log"
APP_LOG="${RUNTIME_DIR}/frameguard_${RUN_STAMP}.log"
VLLM_PID_FILE="${RUNTIME_DIR}/vllm.pid"
APP_PID_FILE="${RUNTIME_DIR}/frameguard.pid"

VLLM_PID=""
APP_PID=""
STARTED_VLLM=0

log() {
    printf '[FrameGuard] %s\n' "$*"
}

fail() {
    printf '[FrameGuard] ERROR: %s\n' "$*" >&2
    exit 1
}

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM

    if [[ -n "${APP_PID}" ]] && kill -0 "${APP_PID}" 2>/dev/null; then
        log "Stopping FrameGuard (PID ${APP_PID})..."
        kill "${APP_PID}" 2>/dev/null || true
        wait "${APP_PID}" 2>/dev/null || true
    fi

    if [[ "${STARTED_VLLM}" == "1" ]] && [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        log "Stopping vLLM (PID ${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi

    rm -f "${APP_PID_FILE}"
    if [[ "${STARTED_VLLM}" == "1" ]]; then
        rm -f "${VLLM_PID_FILE}"
    fi

    exit "${exit_code}"
}

trap cleanup EXIT INT TERM

[[ -x "${VLLM_BIN}" ]] || fail "vLLM executable not found: ${VLLM_BIN}"
[[ -n "${UV_BIN}" ]] || fail "uv is not installed or not on PATH"
[[ -f "${REPO_ROOT}/app.py" ]] || fail "app.py not found under ${REPO_ROOT}"
[[ -f "${FACE_MODEL}" ]] || fail "YuNet model not found: ${FACE_MODEL}"
[[ -f "${FACE_RECOGNITION_MODEL}" ]] || fail "SFace model not found: ${FACE_RECOGNITION_MODEL}"
command -v curl >/dev/null 2>&1 || fail "curl is required"
command -v ffmpeg >/dev/null 2>&1 || fail "ffmpeg is required"
command -v ffprobe >/dev/null 2>&1 || fail "ffprobe is required"
command -v tesseract >/dev/null 2>&1 || fail "tesseract is required"

cd "${REPO_ROOT}"

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if [[ "${FRAMEGUARD_SYNC}" == "1" ]] || [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    log "Synchronizing the FrameGuard uv environment..."
    "${UV_BIN}" sync --link-mode=copy
fi

# Prevent all optional telemetry and remote model lookups at runtime.
export GRADIO_ANALYTICS_ENABLED=False
export GRADIO_SHARE=False
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1

# Required for this ROCm FlashAttention package.
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

export FRAMEGUARD_DETECTOR=qwen
export FRAMEGUARD_API_BASE="http://${FRAMEGUARD_API_HOST}:${FRAMEGUARD_API_PORT}/v1"
export FRAMEGUARD_MODEL="${MODEL_PATH}"
export FRAMEGUARD_FACE_MODEL="${FACE_MODEL}"
export FRAMEGUARD_FACE_RECOGNITION_MODEL="${FACE_RECOGNITION_MODEL}"
export FRAMEGUARD_LOG_LEVEL
export FRAMEGUARD_HOST
export FRAMEGUARD_PORT

HEALTH_URL="http://${FRAMEGUARD_API_HOST}:${FRAMEGUARD_API_PORT}/health"
MODELS_URL="http://${FRAMEGUARD_API_HOST}:${FRAMEGUARD_API_PORT}/v1/models"

if curl -fsS --max-time 2 "${HEALTH_URL}" >/dev/null 2>&1; then
    log "Using the existing Omni server at ${HEALTH_URL}"
else
    log "Starting Qwen2.5-Omni through vLLM..."
    log "vLLM log: ${VLLM_LOG}"

    "${VLLM_BIN}" serve \
        "${MODEL_PATH}" \
        --host "${FRAMEGUARD_API_HOST}" \
        --port "${FRAMEGUARD_API_PORT}" \
        --dtype bfloat16 \
        >"${VLLM_LOG}" 2>&1 &

    VLLM_PID=$!
    STARTED_VLLM=1
    printf '%s\n' "${VLLM_PID}" > "${VLLM_PID_FILE}"

    # Model loading and graph capture can take several minutes.
    deadline=$((SECONDS + 300))
    until curl -fsS --max-time 2 "${HEALTH_URL}" >/dev/null 2>&1; do
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            tail -n 100 "${VLLM_LOG}" >&2 || true
            fail "vLLM exited before becoming healthy"
        fi

        if (( SECONDS >= deadline )); then
            tail -n 100 "${VLLM_LOG}" >&2 || true
            fail "Timed out waiting for vLLM health after 300 seconds"
        fi

        sleep 2
    done

    log "Omni server is healthy."
fi

if ! curl -fsS --max-time 5 "${MODELS_URL}" | grep -Fq "${MODEL_PATH}"; then
    log "WARNING: the served model ID did not exactly match ${MODEL_PATH}."
    log "Inspect ${MODELS_URL} if FrameGuard receives a model-not-found response."
fi

log "Starting FrameGuard on http://${FRAMEGUARD_HOST}:${FRAMEGUARD_PORT}"
log "FrameGuard log: ${APP_LOG}"
log "Use your SSH tunnel and open http://127.0.0.1:${FRAMEGUARD_PORT} on the Mac."
log "Press Ctrl+C here to stop processes started by this script."

"${UV_BIN}" run --no-sync python app.py \
    > >(tee -a "${APP_LOG}") \
    2>&1 &
APP_PID=$!
printf '%s\n' "${APP_PID}" > "${APP_PID_FILE}"

wait "${APP_PID}"
