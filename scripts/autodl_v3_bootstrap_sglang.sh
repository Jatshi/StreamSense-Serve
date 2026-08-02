#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PORTFOLIO_V2_MODE=full
export STREAMSENSE_BACKEND_RUNTIME=sglang
export STREAMSENSE_SGLANG_VERSION="${STREAMSENSE_SGLANG_VERSION:-0.5.10}"
export STREAMSENSE_VENV="${STREAMSENSE_V3_SGLANG_VENV:-/root/autodl-tmp/portfolio-v3/envs/streamsense-sglang}"
mkdir -p "$ROOT/artifacts/v3"
bash "$ROOT/scripts/autodl_v2_bootstrap.sh"
source "$STREAMSENSE_VENV/bin/activate"
python -m pip freeze > "$ROOT/artifacts/v3/sglang-environment.freeze.txt"
