# StreamSense-Serve 2.0 implementation plan

## Goal and acceptance boundary

Version 2.0 is the serving and data-flywheel layer for EvidenceAgent-MM. It must be runnable on
one AutoDL RTX 4090 without writing code after the rental starts, while remaining fully testable
without a GPU. The existing evidence/event API remains supported.

Completion requires:

1. validated vLLM, SGLang, and generic OpenAI-compatible profiles;
2. an EvidenceAgent request that can carry audio/transcript/OCR/frame evidence and a response
   whose citations can only reference supplied evidence IDs;
3. durable, deduplicated feedback with explicit training consent/license, SFT/DPO candidates,
   structured EvidenceAgent bridge data, raw audit data, and a hashed export manifest;
4. load-test JSON containing measured TTFT, TPOT, throughput, error rate, and optional raw GPU
   samples;
5. revision-guarded activation and rollback with atomic local state;
6. one-command smoke plus finite benchmark and persistent serve AutoDL paths;
7. offline unit tests and explicit separation between verified local behavior and pending GPU
   results.

## Architecture

```text
EvidenceAgent request
  -> schema validation
  -> active model + backend profile resolution
  -> OpenAI-compatible /v1/chat/completions
  -> strict JSON parsing + citation allow-list
  -> EvidenceAgent response
  -> token-protected user feedback
  -> SQLite deduplication
  -> SFT/DPO candidate JSONL
```

The backend profile owns endpoint, timeout, model name, engine launch settings, quantization, and
memory limits. The model manifest owns semantic model identity, immutable revision, validation
state, and backend-profile binding. These are deliberately separate: a backend may restart
without changing the selected model, and a model may not be activated merely because a process
is reachable.

## Safety invariants

- Non-abstained v1 events still require replayable evidence.
- A v2 `answer` must contain a citation; every cited ID must have appeared in the request.
- A `clarify` or `abstain` response must state missing evidence.
- Candidate or deprecated model artifacts cannot be activated.
- Activation requires an exact revision and a reachable backend; an advertised-model mismatch
  is rejected.
- State changes are written to a temporary file, flushed, and atomically replaced.
- Feedback SQL is parameterized, duplicate payloads share one content hash, and exports are
  replaced atomically.
- Feedback defaults to no training consent. Export fails closed unless consent is true, a source
  license is recorded, and a correction exists. Structured corrections preserve answer/clarify/
  abstain state, citations, confidence, and missing-evidence fields.
- Inference, admin, and feedback APIs use independent bearer tokens. Full mode cannot bind beyond
  loopback without all three.
- No benchmark value enters documentation unless the raw output file exists.

## Local gate

```bash
python -m pip install -e ".[dev]"
python scripts/validate_v2_config.py
python -m ruff check .
python -m ruff format --check .
python -m pytest --cov=streamsense --cov-report=term-missing
```

The smoke runner executes the tests, launches the API on loopback, and probes `/health`.

## AutoDL full gate

1. Run `scripts/autodl_v2_bootstrap.sh` with `PORTFOLIO_V2_MODE=full`.
2. Review the selected profile and immutable model revision.
3. Set `STREAMSENSE_V2_ACTION=benchmark` and run `scripts/autodl_v2_run.sh`; it waits up to
   20 minutes for the backend, starts and probes the API, measures both, then terminates both
   child processes.
4. Submit a fixed EvidenceAgent fixture and verify citations.
5. Run chat load tests at concurrency 1, 4, 8, 16, and 32.
6. Preserve every JSON under `benchmarks/results/v2/`; do not summarize failed or partial runs as
   successful capacity.
7. Export reviewed hard cases and manually redact them before training or publication.

## GPU experiment matrix still to execute

| Variable | Values |
|---|---|
| Runtime | vLLM, SGLang |
| Precision/quantization | FP16 baseline, then only supported AWQ/GPTQ/FP8 variants |
| Concurrency | 1, 4, 8, 16, 32 |
| Context | short, medium, long evidence packs |
| Metrics | success/error rate, TTFT, TPOT, request/s, output units/s, GPU memory/utilization |

Quantized profiles must be added only after a real compatible artifact is selected. The template
supports the settings, but the repository does not claim an unexecuted quantized result.

## Release gate

- local tests/lint/format pass;
- Docker image builds and `/health` responds;
- smoke runner passes on the target machine;
- full run artifacts contain environment, model revision, command, and raw measurements;
- README numbers agree with raw artifacts;
- `.light/passport.yaml` and `run_manifest.md` distinguish PASS, PENDING_GPU, and limitations.
