# Model card

## Components

| Component | Pinned artifact | Purpose |
|---|---|---|
| Audio activity | `rms-v1` | Transparent candidate-window baseline |
| ASR | `Systran/faster-whisper-small` at the revision in `models.lock` | Timestamped speech transcription |
| Visual change | `mad-v1` | Mean absolute pixel-change evidence |
| Visual grounding | `Qwen/Qwen2.5-VL-3B-Instruct` at the revision in `models.lock` | Conservative frame description |
| Router | `rules-v1` | Risk, uncertainty, conflict, grounding, and exploration decisions |

The 4090 reference uses vLLM 0.15.1, PyTorch 2.9.1+cu128, FP16, a 4,096-token context limit,
and 72% GPU memory utilization. ASR and VLM benchmarks run in mutually exclusive GPU modes.

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

