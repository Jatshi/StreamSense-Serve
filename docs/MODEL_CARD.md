# Model card

## Components

| Component | Pinned artifact | Purpose |
|---|---|---|
| Audio activity | `rms-v1` | Transparent candidate-window baseline |
| ASR | `Systran/faster-whisper-small` at the revision in `models.lock` | Timestamped speech transcription |
| Visual change | `mad-v1` | Mean absolute pixel-change evidence |
| Visual grounding | `Qwen/Qwen2.5-VL-3B-Instruct` at the revision in `models.lock` | Conservative frame description |
| Router | `rules-v1` | Risk, uncertainty, conflict, grounding, and exploration decisions |

The v2 RTX 4090 matrix pins Qwen2.5-VL-3B-Instruct revision
`66285546d2b821cf421d4f5eb2576359d3770cd3` and uses an 8,192-token context
limit with 88% configured GPU-memory utilization. Its profiles are:

- vLLM 0.15.1 + PyTorch 2.9.1/cu128 in BF16;
- the same vLLM runtime with dynamic FP8 quantization and BF16 compute;
- SGLang 0.5.10 + PyTorch 2.9.1/cu128 in BF16.

All profiles use the same model revision, quality fixture, 64-request loads,
64-token output cap, and concurrency levels 1/4/8/16/32. They run sequentially;
the supervisor refuses to start the next profile while any previous CUDA PID
remains. ASR and VLM benchmarks also run in mutually exclusive GPU modes.

## Verified RTX 4090 serving matrix

All 15 cells completed 64/64 requests with zero errors (960/960 total).
Load-test responses matched the fixed `benchmark-ok` contract in every cell.
The separate 12-case quality fixture passed 8/12 for vLLM BF16 and 7/12 for
vLLM dynamic FP8 and SGLang BF16.

| Profile | Concurrency | Requests/s | Reported completion tokens/s | p50/p95 TTFT (ms) | p50 TPOT (ms) | p95 latency (ms) | Peak memory (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM BF16 | 1 | 24.828 | 74.484 | 23.508 / 27.303 | 16.160 | 42.866 | 21,677 |
| vLLM BF16 | 32 | 176.806 | 530.417 | 124.098 / 161.209 | 32.309 | 194.168 | 21,677 |
| vLLM dynamic FP8 | 1 | 26.291 | 78.872 | 24.370 / 31.005 | 12.429 | 43.627 | 21,683 |
| vLLM dynamic FP8 | 32 | 189.843 | 569.530 | 123.866 / 158.425 | 24.236 | 193.499 | 21,683 |
| SGLang BF16 | 1 | 24.294 | 72.883 | 27.598 / 28.420 | 13.301 | 42.017 | 22,563 |
| SGLang BF16 | 32 | 144.422 | 433.267 | 131.241 / 160.483 | 68.406 | 218.769 | 22,623 |

The complete 15-row machine-readable result is
`docs/benchmark_matrix_4090.json`. Dynamic FP8 was fastest at concurrency 32
in this short-output synthetic workload, but it did not reduce the configured
server memory reservation and lost one quality-fixture item. The SGLang
concurrency-4 run also contained a p95 TTFT outlier of 412.955 ms. Results are
reported without deleting or rerunning unfavorable cells.

## Intended use

The system is an evidence-oriented engineering demonstrator for meetings, classroom recordings,
and consented equipment-inspection footage. It creates reviewable event candidates and grounded
answers. It is not an autonomous safety, medical, employment, policing, or surveillance decision
system.

## Output constraints

The VLM prompt forbids identity and intent inference. Its output must validate against a bounded
JSON schema. Non-abstained stored events require replayable evidence. Missing or failed optional
VLM inference becomes `human_review`; unsupported questions should abstain.

## Limitations

- A scene-change frame does not establish causality or intent.
- ASR confidence is an operational score, not a calibrated probability of transcript correctness.
- Text inside synthetic frames may be easier than real OCR under blur, glare, or occlusion.
- The rule-router fixture is too small for deployment-level threshold selection.
- Model licenses and acceptable-use terms remain those of their upstream publishers.
