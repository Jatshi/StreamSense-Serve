#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${STREAMSENSE_V3_RUNTIME:-vllm}"
ARTIFACT_DIR="${STREAMSENSE_V3_SMOKE_DIR:-$ROOT/artifacts/v3/smoke-$RUNTIME}"
if [[ "$RUNTIME" == "vllm" ]]; then
  VENV="${STREAMSENSE_V3_VLLM_VENV:-/root/autodl-tmp/portfolio-v3/envs/streamsense-vllm}"
  PROFILE="vllm-qwen25-vl-3b"
elif [[ "$RUNTIME" == "sglang" ]]; then
  VENV="${STREAMSENSE_V3_SGLANG_VENV:-/root/autodl-tmp/portfolio-v3/envs/streamsense-sglang}"
  PROFILE="sglang-qwen25-vl-3b"
else
  echo "STREAMSENSE_V3_RUNTIME must be vllm or sglang" >&2
  exit 2
fi
source "$VENV/bin/activate"
cd "$ROOT"
mkdir -p "$ARTIFACT_DIR"
python -m pytest -q tests/test_slo_v3.py
PORTFOLIO_V2_MODE=full \
STREAMSENSE_VENV="$VENV" \
STREAMSENSE_V2_ACTION=benchmark \
STREAMSENSE_BACKEND_PROFILE="$PROFILE" \
STREAMSENSE_BENCHMARK_REQUESTS="${STREAMSENSE_BENCHMARK_REQUESTS:-4}" \
STREAMSENSE_BENCHMARK_CONCURRENCY="${STREAMSENSE_BENCHMARK_CONCURRENCY:-1}" \
STREAMSENSE_BENCHMARK_MAX_TOKENS="${STREAMSENSE_BENCHMARK_MAX_TOKENS:-32}" \
STREAMSENSE_BENCHMARK_DIR="$ARTIFACT_DIR" \
STREAMSENSE_BENCHMARK_OUTPUT="$ARTIFACT_DIR/backend_chat.json" \
STREAMSENSE_API_BENCHMARK_OUTPUT="$ARTIFACT_DIR/api_health.json" \
  bash scripts/autodl_v2_run.sh
python scripts/write_v3_smoke_manifest.py \
  --runtime "$RUNTIME" \
  --chat-report "$ARTIFACT_DIR/backend_chat.json" \
  --api-report "$ARTIFACT_DIR/api_health.json" \
  --output "$ARTIFACT_DIR/run_manifest.json"
