# Changelog

## 2.0.0 - 2026-08-02

- Added versioned vLLM, SGLang, and OpenAI-compatible backend profiles and health probes.
- Added EvidenceAgent-MM request/response adaptation with citation allow-list validation.
- Added token-protected feedback persistence, deduplication, consent/license-gated SFT/DPO
  export, structured EvidenceAgent bridge data, raw audit JSONL, and hashed export manifests.
- Added revision-guarded atomic model activation and rollback.
- Added streaming TTFT/TPOT/throughput/error/GPU sampling and unified AutoDL v2 scripts.
- Verified the complete vLLM BF16, vLLM dynamic FP8, and SGLang BF16 matrix on RTX 4090:
  15 cells, five concurrency levels, and 960/960 successful requests.
- Added a reproducible animated README demo, complete release notes, and an implementation/failure
  diary in the Chinese learning manual.

## 0.1.0 - 2026-07-18

- Added evidence-grounded audio activity, GPU ASR, video-change, and local VLM analysis.
- Added explainable adaptive routing with exploration and safe human-review fallback.
- Added SQLite persistence, abstaining retrieval, FastAPI, CLI, web console, metrics, tracing,
  Docker, and multi-version CI.
- Published reproducible RTX 4090 ASR, robustness, routing, load, end-to-end, and VLM artifacts.
- Added model/data cards, bilingual documentation, security guidance, pinned model revisions, and
  third-party license notices.
