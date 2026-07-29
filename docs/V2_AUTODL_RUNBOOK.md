# StreamSense-Serve 2.0 AutoDL runbook

## 1. Upload and bootstrap

Upload the repository without `data/`, `runs/`, `.venv*`, or model caches. From the repository
root:

```bash
export PORTFOLIO_V2_MODE=smoke
bash scripts/autodl_v2_bootstrap.sh
bash scripts/autodl_v2_run.sh
```

Smoke mode uses no model or GPU. It validates config, runs the test suite, starts FastAPI on
loopback, and probes `/health`.

For a full 4090 run:

```bash
source .venv-v2/bin/activate
export PORTFOLIO_V2_MODE=full
export STREAMSENSE_BACKEND_RUNTIME=vllm
export STREAMSENSE_BACKEND_PROFILE=vllm-qwen25-vl-3b
export STREAMSENSE_ADMIN_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export STREAMSENSE_FEEDBACK_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export STREAMSENSE_INFERENCE_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export STREAMSENSE_V2_ACTION=benchmark
bash scripts/autodl_v2_bootstrap.sh
bash scripts/autodl_v2_run.sh
```

The bootstrap pins the previously verified vLLM 0.15.1 runtime. SGLang support is selected with
`STREAMSENSE_BACKEND_RUNTIME=sglang` and its corresponding profile. The SGLang environment is
separate and pins SGLang 0.5.10 with Torch 2.9.1/CUDA 12.8. SGLang 0.5.11 and newer changed their
default wheel line to Torch 2.11/CUDA 13, which cannot initialize CUDA on the 560-series AutoDL
driver used for this run. Every environment must pass a real `torch.cuda.is_available()` and CUDA
matrix-operation preflight before a model server is launched.

If the 1.2 GiB FlashInfer JIT-cache wheel is slow or times out when fetched from GitHub, download
it once with a resumable, segmented downloader and point the bootstrap at the verified local file:

```bash
export STREAMSENSE_FLASHINFER_JIT_CACHE_SPEC=/root/autodl-tmp/wheels/flashinfer_jit_cache.whl
bash scripts/autodl_v2_bootstrap.sh
```

When this variable is unset, the bootstrap retains the reproducible
`flashinfer_jit_cache==0.6.7.post2` package pin. The local override changes only the transport
path, not the package version or benchmark configuration. The SGLang resolver uses the Aliyun
PyPI mirror as its default index and persists metadata/wheels under
`/root/autodl-tmp/.cache/uv`; interrupted runs therefore resume without repeating the full
overseas-index metadata scan.

`STREAMSENSE_V2_ACTION=benchmark` starts the backend and API, waits for both health probes, writes
`benchmarks/results/v2/api_health.json` and `backend_chat.json`, then terminates both processes.
The request count, concurrency, output-token cap, GPU sampling, and output paths are configurable
through the `STREAMSENSE_BENCHMARK_*` variables in `.env.example`. Use
`STREAMSENSE_V2_ACTION=serve` only when a resident service is intentionally required.

## 2. Backend profiles

`configs/backends.json` is executable configuration, not a benchmark claim. Each profile records:

- runtime and base URL;
- served model name and immutable model revision;
- dtype and quantization mode;
- tensor parallelism, maximum context, and GPU-memory fraction;
- retry, timeout, temperature, and output limits.

Inspect the exact launch argv without starting a model:

```bash
python -m streamsense.backend_launcher \
  --config configs/backends.json \
  --profile vllm-qwen25-vl-3b
```

Do not set `trust_remote_code` unless you have reviewed the referenced repository revision.

## 3. EvidenceAgent call

The endpoint is `POST /v2/evidence-agent/query` with
`Authorization: Bearer $STREAMSENSE_INFERENCE_TOKEN`. Evidence IDs are the trust boundary:

```json
{
  "request_id": "meeting-001-q1",
  "question": "谁提出了两阶段检索方案?",
  "evidence": [
    {
      "evidence_id": "seg-0007",
      "modality": "transcript",
      "text": "张老师: 我建议采用两阶段检索。",
      "start_ms": 12000,
      "end_ms": 15500,
      "speaker": "speaker-02",
      "score": 0.94
    }
  ]
}
```

An `answer` without citations, a citation to any other ID, or a clarify/abstain result without a
missing-evidence explanation is rejected as invalid backend output.

## 4. Feedback and training candidates

Submit a correction with `Authorization: Bearer $STREAMSENSE_FEEDBACK_TOKEN` to
`POST /v2/feedback`. Exact duplicates return the existing ID with `duplicate=true`.
`consent_for_training` defaults to `false`. A record enters any training export only when consent
is explicitly true, `source_license` is present, and either `corrected_response` or the legacy
`corrected_answer` is usable. Prefer `corrected_response`: it preserves the three-state decision,
citations, confidence, and missing-evidence contract.

Export with the admin token:

```bash
curl -X POST http://127.0.0.1:8000/v2/feedback/export \
  -H "Authorization: Bearer ${STREAMSENSE_ADMIN_TOKEN}"
```

Outputs:

- `artifacts/training_candidates/sft_candidates.jsonl`
- `artifacts/training_candidates/dpo_candidates.jsonl`
- `artifacts/training_candidates/evidenceagent_bridge.jsonl`
- `artifacts/training_candidates/consented_feedback_raw.jsonl`
- `artifacts/training_candidates/export_manifest.json`

The manifest records schema version, consenting source content hashes, a source-manifest hash,
per-output SHA-256, and example counts. The bridge contains only structured corrections; legacy
plain-text corrections remain available in SFT/DPO/raw outputs but are not represented as if they
contained a verified three-state/citation target. These are candidates, not automatically trusted
labels. Review speaker identity, citations, license, private information, and corrections before
training.

## 5. Model activation and rollback

`models/serve_manifest.json` is the source of truth. Only `validated` entries can be activated.
Reading `GET /v2/models` also requires the admin token because the manifest can contain local
adapter paths and model revisions. `GET /v2/inference/health` requires the inference token because
it exposes backend/model configuration.
Activation requires:

- exact `expected_revision`;
- configured backend profile;
- reachable health endpoint;
- matching served model when `/v1/models` advertises model IDs.

```bash
curl -X POST http://127.0.0.1:8000/v2/models/activate \
  -H "Authorization: Bearer ${STREAMSENSE_ADMIN_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"model_id":"qwen25-vl-3b-baseline","expected_revision":"66285546d2b821cf421d4f5eb2576359d3770cd3","reason":"manual health and regression gate passed"}'
```

Rollback:

```bash
curl -X POST http://127.0.0.1:8000/v2/models/rollback \
  -H "Authorization: Bearer ${STREAMSENSE_ADMIN_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"reason":"citation regression detected"}'
```

The registry switches selection atomically. It does not rewrite model files or silently start a
different process.

## 6. Measured load tests

Backend chat streaming:

```bash
python scripts/load_test.py \
  --base-url http://127.0.0.1:8001 \
  --endpoint chat \
  --model qwen25-vl-3b-streamsense \
  --requests 100 \
  --concurrency 8 \
  --sample-gpu \
  --output benchmarks/results/v2/vllm_c8.json
```

Repeat with concurrency 1, 4, 8, 16, and 32. TTFT is first non-empty content delta. TPOT uses
subsequent streaming deltas, which are explicitly named output units because server chunks are
not guaranteed to equal tokenizer tokens. If `nvidia-smi` is unavailable, the JSON records
`available=false`; it never inserts a guessed GPU value.

## 7. Network and recovery

- Default binding is `127.0.0.1`; use SSH port forwarding for access.
- Do not expose the service directly to the internet.
- A non-loopback full run requires all three independent tokens during preflight.
- Backend and API logs are under `runs/`.
- If backend startup fails, inspect `runs/backend.log`, fix only the selected profile or runtime,
  rerun preflight, then restart.
- SQLite, active-model state, and exports live under `data/` and `artifacts/`; back them up before
  deleting a rental instance.
