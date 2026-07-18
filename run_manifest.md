# Run manifest

## Local foundation run

- Environment: Windows, Python 3.11.7
- Install: `python -m pip install -e ".[dev]"`
- Quality: `ruff check . && ruff format --check .`
- Test: `pytest --cov=streamsense --cov-report=term-missing`
- Result: 10 tests passed, 84% statement coverage
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

## Web console quality gate

- Chromium rendering: 1440 x 1100 with a persisted grounded event
- Layout audit: 6/6 PASS
- AI-tell lint: CLEAN
- Artifact: `docs/assets/dashboard.png`
