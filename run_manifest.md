# Run manifest

## Version 2.0 local preparation

- Scope: OpenAI/vLLM/SGLang profiles, EvidenceAgent adapter, feedback data flywheel, streaming
  load test, atomic model activation/rollback, and AutoDL entry scripts.
- Config validation: `python scripts/validate_v2_config.py`
- Offline quality gate: `python -m ruff check .`, `python -m ruff format --check .`,
  `python -m pytest --cov=streamsense --cov-report=term-missing`
- Local result: 48 tests passed, 84.79% statement coverage, Ruff lint/format PASS, config validator
  PASS, and `PORTFOLIO_V2_MODE=smoke` API startup/probe PASS.
- Wheel/Docker note: local wheel build could not be re-executed because the offline Windows
  environment lacks `hatchling`; Docker CLI is unavailable. AutoDL bootstrap installs build/runtime
  dependencies, while CI remains the authoritative image-build gate.
- AutoDL entry points: `scripts/autodl_v2_preflight.sh`,
  `scripts/autodl_v2_bootstrap.sh`, `scripts/autodl_v2_run.sh`
- Full-mode action contract: `STREAMSENSE_V2_ACTION=benchmark` measures API health and backend
  streaming load to JSON and exits cleanly; `serve` intentionally remains resident.
- GPU status: ENVIRONMENT_VERIFIED / MATRIX_PENDING. No 2.0 TTFT, TPOT, throughput,
  quantization, or memory result is asserted before the three-profile matrix finishes.
- Expected new artifacts: `benchmarks/results/v2/{api_health,backend_chat}.json`;
  consent-gated `artifacts/training_candidates/{sft_candidates,dpo_candidates,evidenceagent_bridge,
  consented_feedback_raw}.jsonl` plus `export_manifest.json`

## V2 AutoDL environment preflight — 2026-07-29

- Hardware: NVIDIA GeForce RTX 4090 24GB; driver 560.35.03.
- Model: Qwen2.5-VL-3B-Instruct at revision
  `66285546d2b821cf421d4f5eb2576359d3770cd3`.
- vLLM environment: vLLM 0.15.1 with a CUDA 12.8-capable PyTorch runtime.
- SGLang environment: SGLang 0.5.10, PyTorch 2.9.1+cu128, 214 resolved packages.
- SGLang acceptance: `torch.cuda.is_available()` and a real 64×64 CUDA matrix
  multiplication passed.
- Matrix contract: vLLM BF16, vLLM dynamic FP8 with BF16 compute, and SGLang BF16;
  64 requests at concurrency 1/4/8/16/32, fixed 64-token cap and shared quality fixture.
- Lifecycle guard: the launcher uses process replacement, and the matrix refuses to start the
  next profile while any CUDA PID remains.
- Current status: the full matrix is queued behind Audio-Codec-LLM training so both workloads
  never contend for the single GPU.

## Local quality run

- Environment: Windows, Python 3.11.7
- Install: `python -m pip install -e ".[dev]"`
- Quality: `ruff check . && ruff format --check .`
- Test: `pytest --cov=streamsense --cov-report=term-missing`
- Result: 32 tests passed, 87% statement coverage
- GPU: not used
- Artifacts: terminal output only; remote/GPU run pending

## AutoDL RTX 4090 ASR run

- Remote Python: 3.10.8; GPU: NVIDIA GeForce RTX 4090 24 GB; driver: 570.124.04
- Model: faster-whisper small, CUDA FP16
- Sample: 11-second JFK WAV, SHA-256 `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`
- Cold run including model download/load: 304.440 s
- Warm median: 0.356 s; real-time factor: 0.0323; observed post-inference VRAM: 1019 MiB
- Robustness: WER 0.0 (clean/20/10 dB), 0.091 (0 dB), 0.318 (-5 dB)
- Artifacts: `benchmarks/results/asr_small_jfk_4090.json`, `benchmarks/results/asr_small_jfk_noise_4090.json`

## AutoDL API end-to-end and load run

- Media flow: upload -> audio energy + GPU ASR -> SQLite events -> grounded query
- Result: 2/2 events persisted; query cited the 0.000-10.340 s transcript and did not abstain
- End-to-end media analysis latency: 3033.7 ms after the model was resident
- Loopback load: 500/500 successful health requests at concurrency 25
- Observed throughput: 195.6 requests/s; p50/p95/p99: 81.8/334.8/491.2 ms
- Scope: single-host engineering checks, not production capacity or corpus accuracy claims
- Artifacts: `benchmarks/results/jfk_api_end_to_end_4090.json`, `benchmarks/results/api_health_load_4090.json`

## Curated routing trade-off run

- Fixture: 20 hand-authored engineering cases; 10 escalation-positive
- Rule router: recall 1.000, precision 0.909, escalation rate 0.550
- Always-escalate baseline: recall 1.000, precision 0.500, escalation rate 1.000
- Fixture GPU cost: 3.85 s versus 7.00 s (45% reduction at equal fixture recall)
- Scope: thresholds were inspected against this fixture; this is a behavior regression test, not
  an unbiased deployment estimate
- Artifact: `benchmarks/results/routing_fixture.json`

## AutoDL local VLM run

- Runtime: vLLM 0.15.1, PyTorch 2.9.1+cu128, Qwen2.5-VL-3B-Instruct FP16
- Model revision: `66285546d2b821cf421d4f5eb2576359d3770cd3`
- Synthetic flow: MP4 decode -> 2.000 s change frame -> router -> strict JSON VLM -> SQLite
- Pipeline result: 1 event, 1 successful escalation, 0 human-review fallbacks, 4167.8 ms
- Stream latency after grammar warmup: median TTFT 54.8 ms; median total 313.6 ms
- First request with the JSON grammar: TTFT 2541.6 ms; total 2802.7 ms
- Observed resident GPU memory: 17855 MiB / 24564 MiB
- Scope: one deterministic synthetic frame; not a visual-accuracy or capacity claim
- Artifact: `benchmarks/results/vlm_qwen25_3b_4090.json`

## Web console quality gate

- Chromium rendering: 1440 x 1100 with a persisted grounded event
- Layout audit: 6/6 PASS
- AI-tell lint: CLEAN
- Artifact: `docs/assets/dashboard.png`

## Release-candidate gate

- AutoDL/Linux: `make test`, `make smoke`, and `make benchmark` passed in the reference venv
- Local/Windows: 32 tests passed at 87% coverage; Ruff lint and format checks passed
- Wheel: built from source with the dashboard, Apache-2.0 license, and NOTICE included
- Clean wheel install: isolated Python 3.11 environment reported no broken requirements; CLI,
  `/health`, and dashboard smoke checks passed
- CI: Python 3.10/3.11/3.12 tests plus Docker image build
