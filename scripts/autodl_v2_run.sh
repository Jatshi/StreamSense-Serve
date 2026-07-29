#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${PORTFOLIO_V2_MODE:-smoke}"
VENV_PATH="${STREAMSENSE_VENV:-${PROJECT_ROOT}/.venv-v2}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_PATH}/bin/python}"
API_HOST="${STREAMSENSE_API_HOST:-127.0.0.1}"
API_PORT="${STREAMSENSE_API_PORT:-8000}"
ACTION="${STREAMSENSE_V2_ACTION:-serve}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

cd "${PROJECT_ROOT}"
export STREAMSENSE_BACKEND_CONFIG="${STREAMSENSE_BACKEND_CONFIG:-configs/backends.json}"
export STREAMSENSE_MODEL_MANIFEST="${STREAMSENSE_MODEL_MANIFEST:-models/serve_manifest.json}"
export STREAMSENSE_DATABASE="${STREAMSENSE_DATABASE:-data/events_v2.db}"
export STREAMSENSE_FEEDBACK_DATABASE="${STREAMSENSE_FEEDBACK_DATABASE:-data/feedback_v2.db}"
export STREAMSENSE_FEEDBACK_EXPORT_DIR="${STREAMSENSE_FEEDBACK_EXPORT_DIR:-artifacts/training_candidates}"

PORTFOLIO_V2_MODE="${MODE}" PYTHON_BIN="${PYTHON_BIN}" \
  bash scripts/autodl_v2_preflight.sh

if [[ "${ACTION}" != "benchmark" && "${ACTION}" != "serve" ]]; then
  echo "STREAMSENSE_V2_ACTION must be benchmark or serve" >&2
  exit 2
fi

if [[ "${MODE}" == "smoke" ]]; then
  "${PYTHON_BIN}" -m pytest -q
  "${PYTHON_BIN}" -m uvicorn streamsense.api:app --host 127.0.0.1 --port "${API_PORT}" \
    >runs/smoke_api.log 2>&1 &
  API_PID=$!
  cleanup() { kill "${API_PID}" 2>/dev/null || true; }
  trap cleanup EXIT
  for _ in $(seq 1 30); do
    if "${PYTHON_BIN}" -c \
      "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${API_PORT}/health', timeout=1)" \
      >/dev/null 2>&1; then
      echo "StreamSense-Serve v2 smoke passed."
      exit 0
    fi
    sleep 1
  done
  echo "API smoke start timed out; see runs/smoke_api.log" >&2
  exit 1
fi

"${PYTHON_BIN}" -m streamsense.backend_launcher \
  --config "${STREAMSENSE_BACKEND_CONFIG}" \
  --profile "${STREAMSENSE_BACKEND_PROFILE:-}" \
  --execute >runs/backend.log 2>&1 &
BACKEND_PID=$!
cleanup() {
  if [[ -n "${API_PID:-}" ]]; then
    kill "${API_PID}" 2>/dev/null || true
    wait "${API_PID}" 2>/dev/null || true
  fi
  kill "${BACKEND_PID}" 2>/dev/null || true
  wait "${BACKEND_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Backend PID ${BACKEND_PID}; log: runs/backend.log"
BACKEND_BASE_URL="$("${PYTHON_BIN}" -m streamsense.backend_launcher \
  --config "${STREAMSENSE_BACKEND_CONFIG}" \
  --profile "${STREAMSENSE_BACKEND_PROFILE:-}" \
  --print-base-url)"
for _ in $(seq 1 240); do
  if ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
    echo "Backend exited during startup; see runs/backend.log" >&2
    exit 1
  fi
  if "${PYTHON_BIN}" -c \
    'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2)' \
    "${BACKEND_BASE_URL}/health" >/dev/null 2>&1; then
    echo "Backend is healthy at ${BACKEND_BASE_URL}."
    break
  fi
  sleep 5
done
if ! "${PYTHON_BIN}" -c \
  'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2)' \
  "${BACKEND_BASE_URL}/health" >/dev/null 2>&1; then
  echo "Backend startup timed out after 20 minutes; see runs/backend.log" >&2
  exit 1
fi

"${PYTHON_BIN}" -m uvicorn streamsense.api:app --host "${API_HOST}" --port "${API_PORT}" \
  >runs/api.log 2>&1 &
API_PID=$!
for _ in $(seq 1 60); do
  if ! kill -0 "${API_PID}" 2>/dev/null; then
    echo "API exited during startup; see runs/api.log" >&2
    exit 1
  fi
  if "${PYTHON_BIN}" -c \
    'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2)' \
    "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
    echo "API is healthy at http://127.0.0.1:${API_PORT}."
    break
  fi
  sleep 1
done
if ! "${PYTHON_BIN}" -c \
  'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2)' \
  "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
  echo "API startup timed out; see runs/api.log" >&2
  exit 1
fi

if [[ "${ACTION}" == "serve" ]]; then
  echo "StreamSense-Serve is running in persistent serve mode."
  wait "${API_PID}"
  exit $?
fi

BACKEND_MODEL="$("${PYTHON_BIN}" -m streamsense.backend_launcher \
  --config "${STREAMSENSE_BACKEND_CONFIG}" \
  --profile "${STREAMSENSE_BACKEND_PROFILE:-}" \
  --print-model)"
BENCHMARK_DIR="${STREAMSENSE_BENCHMARK_DIR:-benchmarks/results/v2}"
CHAT_OUTPUT="${STREAMSENSE_BENCHMARK_OUTPUT:-${BENCHMARK_DIR}/backend_chat.json}"
API_OUTPUT="${STREAMSENSE_API_BENCHMARK_OUTPUT:-${BENCHMARK_DIR}/api_health.json}"
REQUESTS="${STREAMSENSE_BENCHMARK_REQUESTS:-20}"
CONCURRENCY="${STREAMSENSE_BENCHMARK_CONCURRENCY:-4}"
MAX_TOKENS="${STREAMSENSE_BENCHMARK_MAX_TOKENS:-64}"
GPU_ARGS=()
if [[ "${STREAMSENSE_BENCHMARK_SAMPLE_GPU:-1}" == "1" ]]; then
  GPU_ARGS+=(--sample-gpu)
fi

"${PYTHON_BIN}" scripts/load_test.py \
  --base-url "http://127.0.0.1:${API_PORT}" \
  --endpoint health \
  --requests "${REQUESTS}" \
  --concurrency "${CONCURRENCY}" \
  --output "${API_OUTPUT}"

"${PYTHON_BIN}" scripts/load_test.py \
  --base-url "${BACKEND_BASE_URL}" \
  --endpoint chat \
  --model "${BACKEND_MODEL}" \
  --requests "${REQUESTS}" \
  --concurrency "${CONCURRENCY}" \
  --max-tokens "${MAX_TOKENS}" \
  "${GPU_ARGS[@]}" \
  --output "${CHAT_OUTPUT}"

echo "Benchmark complete: ${API_OUTPUT}, ${CHAT_OUTPUT}"
