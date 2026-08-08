#!/usr/bin/env bash
set -Eeuo pipefail

# Start FrameGuard on macOS.
#
# This script starts the React/FastAPI application locally and connects it to an
# OpenAI-compatible Qwen endpoint. The endpoint may already be running locally,
# may be exposed through an SSH tunnel, or may be started with a locally
# installed vLLM/vLLM-Metal executable.
#
# Recommended remote-endpoint usage:
#   FRAMEGUARD_REMOTE_HOST=<host> ./scripts/start-mac.sh
#
# Existing API usage:
#   FRAMEGUARD_API_BASE=http://127.0.0.1:8091/v1 ./scripts/start-mac.sh
#
# By default the launcher starts Qwen/Qwen2.5-Omni-3B locally using PyTorch
# with Metal acceleration. The first run downloads the checkpoint.
#
# Useful overrides:
#   FRAMEGUARD_SYNC=1
#   FRAMEGUARD_LOG_LEVEL=DEBUG
#   FRAMEGUARD_PORT=7861
#   FRAMEGUARD_OPEN_BROWSER=0
#   FRAMEGUARD_MODEL=Qwen/Qwen2.5-Omni-3B
#   FRAMEGUARD_START_LOCAL_QWEN=0
#   FRAMEGUARD_REMOTE_USER=root
#   FRAMEGUARD_REMOTE_SSH_PORT=31092
#   FRAMEGUARD_SSH_KEY="$HOME/.ssh/radeon_key"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

UV_BIN="${UV_BIN:-$(command -v uv || true)}"

FRAMEGUARD_HOST="${FRAMEGUARD_HOST:-127.0.0.1}"
FRAMEGUARD_PORT="${FRAMEGUARD_PORT:-7860}"
FRAMEGUARD_API_HOST="${FRAMEGUARD_API_HOST:-127.0.0.1}"
FRAMEGUARD_API_PORT="${FRAMEGUARD_API_PORT:-8091}"
FRAMEGUARD_API_BASE="${FRAMEGUARD_API_BASE:-http://${FRAMEGUARD_API_HOST}:${FRAMEGUARD_API_PORT}/v1}"
FRAMEGUARD_LOG_LEVEL="${FRAMEGUARD_LOG_LEVEL:-INFO}"
FRAMEGUARD_CHUNK_SECONDS="${FRAMEGUARD_CHUNK_SECONDS:-3}"
FRAMEGUARD_SYNC="${FRAMEGUARD_SYNC:-0}"
FRAMEGUARD_OPEN_BROWSER="${FRAMEGUARD_OPEN_BROWSER:-1}"
FRAMEGUARD_START_LOCAL_QWEN="${FRAMEGUARD_START_LOCAL_QWEN:-1}"
FRAMEGUARD_START_LOCAL_VLLM="${FRAMEGUARD_START_LOCAL_VLLM:-0}"
FRAMEGUARD_DETECTOR="${FRAMEGUARD_DETECTOR:-qwen}"

FRAMEGUARD_REMOTE_HOST="${FRAMEGUARD_REMOTE_HOST:-}"
FRAMEGUARD_REMOTE_USER="${FRAMEGUARD_REMOTE_USER:-root}"
FRAMEGUARD_REMOTE_SSH_PORT="${FRAMEGUARD_REMOTE_SSH_PORT:-31092}"
FRAMEGUARD_REMOTE_API_HOST="${FRAMEGUARD_REMOTE_API_HOST:-127.0.0.1}"
FRAMEGUARD_REMOTE_API_PORT="${FRAMEGUARD_REMOTE_API_PORT:-8091}"
FRAMEGUARD_SSH_KEY="${FRAMEGUARD_SSH_KEY:-${HOME}/.ssh/radeon_key}"

FACE_MODEL="${FRAMEGUARD_FACE_MODEL:-${REPO_ROOT}/models/face_detection_yunet_2023mar.onnx}"
FACE_RECOGNITION_MODEL="${FRAMEGUARD_FACE_RECOGNITION_MODEL:-${REPO_ROOT}/models/face_recognition_sface_2021dec.onnx}"

RUNTIME_DIR="${REPO_ROOT}/outputs/runtime"
mkdir -p "${RUNTIME_DIR}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
APP_LOG="${RUNTIME_DIR}/frameguard_mac_${RUN_STAMP}.log"
VLLM_LOG="${RUNTIME_DIR}/qwen_local_${RUN_STAMP}.log"
SSH_LOG="${RUNTIME_DIR}/qwen_tunnel_${RUN_STAMP}.log"
APP_PID_FILE="${RUNTIME_DIR}/frameguard.pid"

APP_PID=""
SSH_PID=""
VLLM_PID=""
STARTED_TUNNEL=0
STARTED_VLLM=0

log() {
    printf '[FrameGuard] %s\n' "$*"
}

warn() {
    printf '[FrameGuard] WARNING: %s\n' "$*" >&2
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

    if [[ "${STARTED_TUNNEL}" == "1" ]] && [[ -n "${SSH_PID}" ]] && kill -0 "${SSH_PID}" 2>/dev/null; then
        log "Stopping Qwen SSH tunnel (PID ${SSH_PID})..."
        kill "${SSH_PID}" 2>/dev/null || true
        wait "${SSH_PID}" 2>/dev/null || true
    fi

    if [[ "${STARTED_VLLM}" == "1" ]] && [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        log "Stopping local Qwen server (PID ${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi

    rm -f "${APP_PID_FILE}"
    exit "${exit_code}"
}

trap cleanup EXIT INT TERM

api_root() {
    local base="${1%/}"
    printf '%s\n' "${base%/v1}"
}

api_is_healthy() {
    local root
    root="$(api_root "$1")"
    curl -fsS --max-time 3 "${root}/health" >/dev/null 2>&1
}

wait_for_api() {
    local base="$1"
    local process_pid="$2"
    local description="$3"
    local timeout_seconds="${4:-300}"
    local status_log="${5:-}"
    local root
    root="$(api_root "${base}")"
    local deadline=$((SECONDS + timeout_seconds))
    local started_at=${SECONDS}
    local next_status=$((SECONDS + 30))

    until curl -fsS --max-time 3 "${root}/health" >/dev/null 2>&1; do
        if [[ -n "${process_pid}" ]] && ! kill -0 "${process_pid}" 2>/dev/null; then
            fail "${description} exited before the API became healthy"
        fi
        if (( SECONDS >= deadline )); then
            fail "Timed out waiting for ${description} at ${root}/health"
        fi
        if (( SECONDS >= next_status )); then
            log "Still waiting for ${description} ($((SECONDS - started_at))s elapsed)..."
            if [[ -n "${status_log}" ]]; then
                log "Live details: tail -f ${status_log}"
            fi
            next_status=$((SECONDS + 30))
        fi
        sleep 2
    done
}

port_is_listening() {
    local port="$1"
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
}

resolve_vllm_bin() {
    if [[ -n "${VLLM_BIN:-}" ]] && [[ -x "${VLLM_BIN}" ]]; then
        printf '%s\n' "${VLLM_BIN}"
        return 0
    fi

    local candidate
    for candidate in \
        "${HOME}/.venv-vllm-metal/bin/vllm" \
        "${REPO_ROOT}/.venv-vllm-metal/bin/vllm" \
        "$(command -v vllm 2>/dev/null || true)"; do
        if [[ -n "${candidate}" ]] && [[ -x "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    return 1
}

resolve_served_model_id() {
    if [[ -n "${FRAMEGUARD_MODEL:-}" ]]; then
        printf '%s\n' "${FRAMEGUARD_MODEL}"
        return 0
    fi

    local models_url
    models_url="$(api_root "${FRAMEGUARD_API_BASE}")/v1/models"

    curl -fsS --max-time 8 "${models_url}" | python3 -c '
import json
import sys

payload = json.load(sys.stdin)
models = payload.get("data", [])
if not models:
    raise SystemExit(1)
model_id = models[0].get("id")
if not model_id:
    raise SystemExit(1)
print(model_id)
'
}

[[ "$(uname -s)" == "Darwin" ]] || fail "start-mac.sh must be run on macOS"
[[ -n "${UV_BIN}" ]] || fail "uv is not installed. Install it with: brew install uv"
[[ -f "${REPO_ROOT}/app.py" ]] || fail "app.py was not found under ${REPO_ROOT}"
[[ -f "${FACE_MODEL}" ]] || fail "YuNet model not found: ${FACE_MODEL}"
[[ -f "${FACE_RECOGNITION_MODEL}" ]] || fail "SFace model not found: ${FACE_RECOGNITION_MODEL}"

for command_name in curl ffmpeg ffprobe tesseract lsof python3 npm; do
    command -v "${command_name}" >/dev/null 2>&1 || fail "${command_name} is required"
done

cd "${REPO_ROOT}"

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if [[ "${FRAMEGUARD_SYNC}" == "1" ]] || [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    log "Synchronizing the FrameGuard uv environment..."
    "${UV_BIN}" sync --link-mode=copy
fi

# Disable optional telemetry and accidental runtime model downloads.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

# The smoke-test detector does not need a model server.
if [[ "${FRAMEGUARD_DETECTOR}" == "mock" ]]; then
    SERVED_MODEL_ID="demo-mock-no-llm"
    log "Using the controlled local smoke-test detector"

# Reuse an existing API first.
elif api_is_healthy "${FRAMEGUARD_API_BASE}"; then
    log "Using the existing Qwen API at ${FRAMEGUARD_API_BASE}"

# Otherwise create a tunnel to an already-running remote Qwen server.
elif [[ -n "${FRAMEGUARD_REMOTE_HOST}" ]]; then
    command -v ssh >/dev/null 2>&1 || fail "ssh is required for remote Qwen mode"

    if port_is_listening "${FRAMEGUARD_API_PORT}"; then
        fail "Local port ${FRAMEGUARD_API_PORT} is already occupied, but its API health check failed"
    fi

    ssh_args=(
        -N
        -o ExitOnForwardFailure=yes
        -o ServerAliveInterval=30
        -o ServerAliveCountMax=3
        -p "${FRAMEGUARD_REMOTE_SSH_PORT}"
        -L "${FRAMEGUARD_API_HOST}:${FRAMEGUARD_API_PORT}:${FRAMEGUARD_REMOTE_API_HOST}:${FRAMEGUARD_REMOTE_API_PORT}"
    )

    if [[ -n "${FRAMEGUARD_SSH_KEY}" ]]; then
        [[ -f "${FRAMEGUARD_SSH_KEY}" ]] || fail "SSH key not found: ${FRAMEGUARD_SSH_KEY}"
        ssh_args+=( -i "${FRAMEGUARD_SSH_KEY}" )
    fi

    ssh_args+=( "${FRAMEGUARD_REMOTE_USER}@${FRAMEGUARD_REMOTE_HOST}" )

    log "Opening the Qwen API tunnel..."
    log "Tunnel log: ${SSH_LOG}"
    ssh "${ssh_args[@]}" >"${SSH_LOG}" 2>&1 &
    SSH_PID=$!
    STARTED_TUNNEL=1

    wait_for_api "${FRAMEGUARD_API_BASE}" "${SSH_PID}" "Qwen SSH tunnel" 30
    log "Qwen API tunnel is healthy at ${FRAMEGUARD_API_BASE}"

# Start the complete Qwen2.5-Omni model locally with Metal acceleration.
elif [[ "${FRAMEGUARD_START_LOCAL_QWEN}" == "1" ]]; then
    if port_is_listening "${FRAMEGUARD_API_PORT}"; then
        fail "Local port ${FRAMEGUARD_API_PORT} is already occupied"
    fi

    export FRAMEGUARD_MODEL="${FRAMEGUARD_MODEL:-Qwen/Qwen2.5-Omni-3B}"
    log "Starting ${FRAMEGUARD_MODEL} locally (the first download can take a while)..."
    log "Qwen log: ${VLLM_LOG}"
    "${UV_BIN}" run --no-sync uvicorn scripts.qwen_mac_server:app \
        --host "${FRAMEGUARD_API_HOST}" \
        --port "${FRAMEGUARD_API_PORT}" \
        > >(tee -a "${VLLM_LOG}") \
        2>&1 &
    VLLM_PID=$!
    STARTED_VLLM=1

    wait_for_api \
        "${FRAMEGUARD_API_BASE}" \
        "${VLLM_PID}" \
        "local Qwen service" \
        1800 \
        "${VLLM_LOG}"
    log "Local Qwen API is healthy at ${FRAMEGUARD_API_BASE}"

# Optionally start a local vLLM/vLLM-Metal service.
elif [[ "${FRAMEGUARD_START_LOCAL_VLLM}" == "1" ]]; then
    MODEL_PATH="${FRAMEGUARD_MODEL_PATH:-}"
    [[ -n "${MODEL_PATH}" ]] || fail "Set FRAMEGUARD_MODEL_PATH for local vLLM mode"
    [[ -d "${MODEL_PATH}" ]] || fail "Local model directory not found: ${MODEL_PATH}"

    VLLM_EXECUTABLE="$(resolve_vllm_bin || true)"
    [[ -n "${VLLM_EXECUTABLE}" ]] || fail \
        "vLLM was not found. Install vLLM-Metal or set VLLM_BIN explicitly"

    if port_is_listening "${FRAMEGUARD_API_PORT}"; then
        fail "Local port ${FRAMEGUARD_API_PORT} is already occupied"
    fi

    log "Starting the local vLLM service..."
    log "vLLM log: ${VLLM_LOG}"
    "${VLLM_EXECUTABLE}" serve \
        "${MODEL_PATH}" \
        --host "${FRAMEGUARD_API_HOST}" \
        --port "${FRAMEGUARD_API_PORT}" \
        --dtype auto \
        >"${VLLM_LOG}" 2>&1 &
    VLLM_PID=$!
    STARTED_VLLM=1

    wait_for_api "${FRAMEGUARD_API_BASE}" "${VLLM_PID}" "local vLLM service" 600
    log "Local Qwen API is healthy at ${FRAMEGUARD_API_BASE}"

else
    cat >&2 <<ERROR_MESSAGE
[FrameGuard] ERROR: No healthy Qwen API was found at:
[FrameGuard]   ${FRAMEGUARD_API_BASE}
[FrameGuard]
[FrameGuard] Use one of these startup modes:
[FrameGuard]
[FrameGuard] 1. Existing OpenAI-compatible endpoint:
[FrameGuard]    FRAMEGUARD_API_BASE=http://host:port/v1 ./scripts/start-mac.sh
[FrameGuard]
[FrameGuard] 2. Radeon server through an automatic SSH tunnel:
[FrameGuard]    FRAMEGUARD_REMOTE_HOST=<host> ./scripts/start-mac.sh
[FrameGuard]
[FrameGuard] 3. Experimental local vLLM/vLLM-Metal:
[FrameGuard]    FRAMEGUARD_START_LOCAL_VLLM=1 \\
[FrameGuard]    FRAMEGUARD_MODEL_PATH=/path/to/Qwen2.5-Omni-3B \\
[FrameGuard]    ./scripts/start-mac.sh
[FrameGuard]
[FrameGuard] Local Qwen is enabled by default. Set FRAMEGUARD_START_LOCAL_QWEN=1
[FrameGuard] unless you explicitly disabled it.
ERROR_MESSAGE
    exit 1
fi

if [[ -z "${SERVED_MODEL_ID:-}" ]]; then
    SERVED_MODEL_ID="$(resolve_served_model_id || true)"
    [[ -n "${SERVED_MODEL_ID}" ]] || fail \
        "The Qwen endpoint is healthy, but no model ID was returned by /v1/models. Set FRAMEGUARD_MODEL explicitly."
fi

export FRAMEGUARD_DETECTOR
export FRAMEGUARD_API_BASE
export FRAMEGUARD_MODEL="${SERVED_MODEL_ID}"
export FRAMEGUARD_FACE_MODEL="${FACE_MODEL}"
export FRAMEGUARD_FACE_RECOGNITION_MODEL="${FACE_RECOGNITION_MODEL}"
export FRAMEGUARD_LOG_LEVEL
export FRAMEGUARD_CHUNK_SECONDS
export FRAMEGUARD_HOST
export FRAMEGUARD_PORT

if port_is_listening "${FRAMEGUARD_PORT}"; then
    fail "FrameGuard port ${FRAMEGUARD_PORT} is already in use"
fi

log "Building the React interface..."
npm --prefix "${REPO_ROOT}/frontend" install --no-audit --no-fund
npm --prefix "${REPO_ROOT}/frontend" run build

APP_URL="http://${FRAMEGUARD_HOST}:${FRAMEGUARD_PORT}"
log "Qwen model: ${FRAMEGUARD_MODEL}"
log "Starting FrameGuard at ${APP_URL}"
log "FrameGuard log: ${APP_LOG}"
log "Press Ctrl+C to stop processes started by this script."

"${UV_BIN}" run --no-sync python app.py \
    > >(tee -a "${APP_LOG}") \
    2>&1 &
APP_PID=$!
printf '%s\n' "${APP_PID}" > "${APP_PID_FILE}"

# Wait briefly for FrameGuard before opening the browser.
deadline=$((SECONDS + 45))
until curl -fsS --max-time 2 "${APP_URL}/" >/dev/null 2>&1; do
    if ! kill -0 "${APP_PID}" 2>/dev/null; then
        tail -n 100 "${APP_LOG}" >&2 || true
        fail "FrameGuard exited before its web interface became available"
    fi
    if (( SECONDS >= deadline )); then
        warn "The browser was not opened because FrameGuard did not become reachable within 45 seconds"
        break
    fi
    sleep 1
done

if [[ "${FRAMEGUARD_OPEN_BROWSER}" == "1" ]] && curl -fsS --max-time 2 "${APP_URL}/" >/dev/null 2>&1; then
    open "${APP_URL}" >/dev/null 2>&1 || true
fi

wait "${APP_PID}"
