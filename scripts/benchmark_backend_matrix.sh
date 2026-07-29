#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${STREAMSENSE_BACKEND_CONFIG:-${PROJECT_ROOT}/configs/backends.json}"
RESULT_ROOT="${STREAMSENSE_MATRIX_OUTPUT:-${PROJECT_ROOT}/benchmarks/results/v2/matrix}"
REQUESTS="${STREAMSENSE_MATRIX_REQUESTS:-64}"
MAX_TOKENS="${STREAMSENSE_MATRIX_MAX_TOKENS:-64}"
STARTUP_POLLS="${STREAMSENSE_MATRIX_STARTUP_POLLS:-240}"
VLLM_VENV="${STREAMSENSE_VLLM_VENV:-/root/autodl-tmp/streamsense-vllm-venv}"
SGLANG_VENV="${STREAMSENSE_SGLANG_VENV:-/root/autodl-tmp/streamsense-sglang-venv}"
PROFILES="${STREAMSENSE_MATRIX_PROFILES:-vllm-qwen25-vl-3b vllm-qwen25-vl-3b-fp8 sglang-qwen25-vl-3b}"
CONCURRENCIES="${STREAMSENSE_MATRIX_CONCURRENCIES:-1 4 8 16 32}"

mkdir -p "${RESULT_ROOT}"
cd "${PROJECT_ROOT}"

backend_pid=""
cleanup_backend() {
  if [[ -n "${backend_pid}" ]]; then
    kill "${backend_pid}" 2>/dev/null || true
    wait "${backend_pid}" 2>/dev/null || true
    backend_pid=""
  fi
}
trap cleanup_backend EXIT

for profile in ${PROFILES}; do
  case "${profile}" in
    vllm-*) venv="${VLLM_VENV}" ;;
    sglang-*) venv="${SGLANG_VENV}" ;;
    *)
      echo "Unsupported matrix profile: ${profile}" >&2
      exit 2
      ;;
  esac
  python_bin="${venv}/bin/python"
  if [[ ! -x "${python_bin}" ]]; then
    echo "Missing backend Python: ${python_bin}" >&2
    exit 2
  fi

  profile_dir="${RESULT_ROOT}/${profile}"
  mkdir -p "${profile_dir}"
  base_url="$("${python_bin}" -m streamsense.backend_launcher \
    --config "${CONFIG}" --profile "${profile}" --print-base-url)"
  model="$("${python_bin}" -m streamsense.backend_launcher \
    --config "${CONFIG}" --profile "${profile}" --print-model)"

  "${python_bin}" -m streamsense.backend_launcher \
    --config "${CONFIG}" --profile "${profile}" --execute \
    >"${profile_dir}/backend.log" 2>&1 &
  backend_pid=$!

  healthy=0
  for _ in $(seq 1 "${STARTUP_POLLS}"); do
    if ! kill -0 "${backend_pid}" 2>/dev/null; then
      echo "Backend ${profile} exited during startup" >&2
      tail -n 100 "${profile_dir}/backend.log" >&2 || true
      exit 1
    fi
    if "${python_bin}" -c \
      'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2)' \
      "${base_url}/health" >/dev/null 2>&1; then
      healthy=1
      break
    fi
    sleep 5
  done
  if [[ "${healthy}" != "1" ]]; then
    echo "Backend ${profile} did not become healthy" >&2
    tail -n 100 "${profile_dir}/backend.log" >&2 || true
    exit 1
  fi

  "${python_bin}" - <<'PY' "${CONFIG}" "${profile}" "${base_url}" "${model}" \
    "${profile_dir}/environment.json"
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

config_path, profile, base_url, model, output = sys.argv[1:]
packages = {}
for name in ("torch", "transformers", "vllm", "sglang"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
gpu = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ],
    capture_output=True,
    text=True,
    check=False,
).stdout.strip()
config = json.loads(Path(config_path).read_text(encoding="utf-8"))
selected = next(item for item in config["profiles"] if item["name"] == profile)
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "profile": selected,
    "base_url": base_url,
    "model": model,
    "python": platform.python_version(),
    "packages": packages,
    "gpu_after_load": gpu,
}
Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

  "${python_bin}" scripts/load_test.py \
    --base-url "${base_url}" --endpoint chat --model "${model}" \
    --requests 4 --concurrency 1 --max-tokens "${MAX_TOKENS}" \
    --sample-gpu --output "${profile_dir}/warmup.json"

  # A model can miss a quality case without invalidating the performance run.
  # The non-zero exit is preserved in the machine-readable report and summary.
  "${python_bin}" scripts/quality_benchmark.py \
    --base-url "${base_url}" --model "${model}" \
    --output "${profile_dir}/quality.json" || true

  for concurrency in ${CONCURRENCIES}; do
    if (( concurrency > REQUESTS )); then
      echo "Concurrency ${concurrency} exceeds requests ${REQUESTS}" >&2
      exit 2
    fi
    "${python_bin}" scripts/load_test.py \
      --base-url "${base_url}" --endpoint chat --model "${model}" \
      --requests "${REQUESTS}" --concurrency "${concurrency}" \
      --max-tokens "${MAX_TOKENS}" --sample-gpu \
      --output "${profile_dir}/concurrency_${concurrency}.json"
  done

  cleanup_backend
  # Ensure CUDA contexts from the backend are gone before changing precision/runtime.
  for _ in $(seq 1 30); do
    if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
      | grep -q '[0-9]'; then
      break
    fi
    sleep 2
  done
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | grep -q '[0-9]'; then
    echo "CUDA process remained after ${profile}; refusing to contaminate the next profile." >&2
    exit 1
  fi
done

"${VLLM_VENV}/bin/python" - <<'PY' "${RESULT_ROOT}"
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for profile_dir in sorted(path for path in root.iterdir() if path.is_dir()):
    environment = json.loads((profile_dir / "environment.json").read_text(encoding="utf-8"))
    quality = json.loads((profile_dir / "quality.json").read_text(encoding="utf-8"))
    for report_path in sorted(profile_dir.glob("concurrency_*.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        gpu_samples = report.get("gpu", {}).get("samples", [])
        rows.append(
            {
                "profile": profile_dir.name,
                "runtime": environment["profile"]["kind"],
                "quantization": environment["profile"]["engine"]["quantization"],
                "concurrency": report["concurrency"],
                "requests": report["requests"],
                "successes": report["successes"],
                "error_rate": report["error_rate"],
                "throughput_requests_per_second": report[
                    "throughput_requests_per_second"
                ],
                "reported_output_tokens_per_second": report.get(
                    "reported_output_tokens_per_second"
                ),
                "ttft_p50_ms": (report.get("ttft_ms") or {}).get("p50"),
                "ttft_p95_ms": (report.get("ttft_ms") or {}).get("p95"),
                "tpot_p50_ms": (report.get("tpot_ms") or {}).get("p50"),
                "latency_p95_ms": (report.get("latency_ms") or {}).get("p95"),
                "quality_pass_rate": quality["pass_rate"],
                "peak_gpu_memory_mib": max(
                    (sample["memory_used_mib"] for sample in gpu_samples),
                    default=None,
                ),
            }
        )
summary = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "rows": rows,
}
(root / "matrix_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "STREAMSENSE_BACKEND_MATRIX_OK ${RESULT_ROOT}"
