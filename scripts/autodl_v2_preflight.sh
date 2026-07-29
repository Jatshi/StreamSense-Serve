#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${PORTFOLIO_V2_MODE:-smoke}"
VENV_PATH="${STREAMSENSE_VENV:-${PROJECT_ROOT}/.venv-v2}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_PATH}/bin/python}"

if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
  echo "PORTFOLIO_V2_MODE must be smoke or full" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
"${PYTHON_BIN}" -m streamsense.backend_launcher \
  --config "${STREAMSENSE_BACKEND_CONFIG:-configs/backends.json}" >/dev/null
"${PYTHON_BIN}" scripts/validate_v2_config.py \
  --backends "${STREAMSENSE_BACKEND_CONFIG:-configs/backends.json}" \
  --models "${STREAMSENSE_MODEL_MANIFEST:-models/serve_manifest.json}"

mkdir -p data runs artifacts
test -w data
test -w runs

if [[ "${MODE}" == "full" ]]; then
  command -v nvidia-smi >/dev/null
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  "${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'
fi

if [[ "${STREAMSENSE_API_HOST:-127.0.0.1}" != "127.0.0.1" ]]; then
  if [[ -z "${STREAMSENSE_ADMIN_TOKEN:-}" || -z "${STREAMSENSE_FEEDBACK_TOKEN:-}" || -z "${STREAMSENSE_INFERENCE_TOKEN:-}" ]]; then
    echo "ADMIN, FEEDBACK, and INFERENCE tokens are required beyond loopback" >&2
    exit 3
  fi
fi

echo "StreamSense-Serve v2 preflight passed (${MODE})."
