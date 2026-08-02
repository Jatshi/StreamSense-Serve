#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PORTFOLIO_V2_MODE=full
export STREAMSENSE_BACKEND_RUNTIME=vllm
# vLLM 0.26 resolves to Torch 2.11/CUDA 13 wheels.  AutoDL's 560-series
# driver cannot initialize that line, so keep the already validated CUDA 12.8
# stack used by the other portfolio projects.
export STREAMSENSE_VLLM_VERSION="${STREAMSENSE_VLLM_VERSION:-0.12.0}"
export STREAMSENSE_VENV="${STREAMSENSE_V3_VLLM_VENV:-/root/autodl-tmp/portfolio-v3/envs/streamsense-vllm}"
mkdir -p "$ROOT/artifacts/v3"
bash "$ROOT/scripts/autodl_v2_bootstrap.sh"
source "$STREAMSENSE_VENV/bin/activate"
python -m pip freeze > "$ROOT/artifacts/v3/vllm-environment.freeze.txt"
