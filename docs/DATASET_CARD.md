# Dataset card

## Scope

StreamSense-Serve v0.1 does not bundle a training corpus and does not claim corpus-level
accuracy. The committed benchmark artifacts come from two small, auditable engineering
fixtures. Raw media is excluded from Git to avoid silently redistributing third-party content.

## Fixtures

### JFK speech sample

- Source: `whisper.cpp/samples/jfk.wav`, fetched by `scripts/download_jfk_sample.py`.
- SHA-256: `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`.
- Use: ASR latency and deterministic Gaussian-noise stress checks only.
- Duration: 11 seconds; English speech.
- Redistribution: the downloader stores it under ignored `data/public/`; users remain
  responsible for checking upstream terms for their use case.

### Synthetic visual-change fixture

- Source: generated locally by `scripts/generate_visual_fixture.py`.
- Content: a deterministic two-scene status display changing from a green normal state to a
  synthetic open-door alert.
- Use: video decoding, frame-change detection, adaptive VLM escalation, schema validation, and
  evidence persistence.
- Personal data: none.

## Routing fixture

`benchmarks/data/router_fixture.jsonl` contains 20 curated feature vectors with engineering
oracle labels. It tests routing behavior and cost accounting; it is not a statistical sample of
real incidents.

## Known gaps

The release does not include the proposed 100-300 clip public evaluation corpus, dual-annotator
labels, demographics, or real surveillance footage. Therefore evidence validity, risk recall,
and fairness cannot be generalized beyond the committed fixtures. Any future corpus must split
by source video before perturbation and record source, license, hash, and consent metadata.

