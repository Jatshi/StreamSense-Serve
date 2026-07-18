# Run manifest

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

## Web console quality gate

- Chromium rendering: 1440 x 1100 with a persisted grounded event
- Layout audit: 6/6 PASS
- AI-tell lint: CLEAN
- Artifact: `docs/assets/dashboard.png`
