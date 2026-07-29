#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${PORTFOLIO_V2_MODE:-smoke}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_PATH="${STREAMSENSE_VENV:-${PROJECT_ROOT}/.venv-v2}"
BACKEND_RUNTIME="${STREAMSENSE_BACKEND_RUNTIME:-vllm}"
SGLANG_VERSION="${STREAMSENSE_SGLANG_VERSION:-0.5.10}"
FLASHINFER_JIT_CACHE_SPEC="${STREAMSENSE_FLASHINFER_JIT_CACHE_SPEC:-flashinfer_jit_cache==0.6.7.post2}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/.cache/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/autodl-tmp/.cache/uv}"
export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/.cache/torch}"
mkdir -p "${HF_HOME}" "${PIP_CACHE_DIR}" "${UV_CACHE_DIR}" "${TORCH_HOME}"

"${PYTHON_BIN}" -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install -e "${PROJECT_ROOT}[dev,media,asr]"

if [[ "${MODE}" == "full" ]]; then
  if [[ "${BACKEND_RUNTIME}" == "vllm" ]]; then
    python -m pip install "vllm==0.15.1"
  elif [[ "${BACKEND_RUNTIME}" == "sglang" ]]; then
    # SGLang >=0.5.11 switched its default wheels to Torch 2.11/CUDA 13.
    # AutoDL's 560-series driver cannot initialize that runtime. Keep SGLang
    # in its own environment and pin the last Torch 2.9/CUDA 12.8 line.
    python -m pip install "uv>=0.11,<1"
    uv pip install \
      --python "${VENV_PATH}/bin/python" \
      --force-reinstall \
      --no-build-isolation \
      --default-index http://mirrors.aliyun.com/pypi/simple \
      --index-strategy unsafe-best-match \
      --prerelease allow \
      --extra-index-url https://download.pytorch.org/whl/cu128 \
      --extra-index-url https://flashinfer.ai/whl/cu128 \
      "sglang[all]==${SGLANG_VERSION}" \
      "torch==2.9.1+cu128" \
      "triton==3.5.1" \
      "${FLASHINFER_JIT_CACHE_SPEC}"
  else
    echo "STREAMSENSE_BACKEND_RUNTIME must be vllm or sglang" >&2
    exit 2
  fi
fi

python -m pip check
PORTFOLIO_V2_MODE="${MODE}" PYTHON_BIN="${VENV_PATH}/bin/python" \
  bash "${PROJECT_ROOT}/scripts/autodl_v2_preflight.sh"
echo "Bootstrap complete: ${VENV_PATH}"
