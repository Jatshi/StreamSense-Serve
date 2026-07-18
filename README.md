# StreamSense-Serve

Evidence-first audiovisual event inference with adaptive escalation to a vision-language model.

StreamSense-Serve turns time-aligned audio and video observations into structured events. Every
non-abstained result carries replayable evidence, and a configurable router escalates only risky,
uncertain, conflicting, or visually grounded requests to an expensive VLM worker.

## Current status

The repository is under active development. The first public milestone provides the event schema,
SQLite event store, deterministic adaptive router, evidence-constrained query API, tests, and a
GPU-ready extension interface. Model-backed media ingestion and benchmark results are tracked in
the project plan.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,media]"
pytest
streamsense serve --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the API documentation.

To enable timestamped ASR, install `.[asr]` and set `STREAMSENSE_ASR_MODEL=small`. The model is
loaded lazily on the first media request. Video scene-change analysis uses the `media` extra.

## Safety and privacy

The project does not perform identity recognition. Use only media that you are licensed and
authorized to process. Outputs are decision support and must not be used as autonomous medical,
safety, or surveillance decisions.

## License

Apache-2.0. Third-party model and dataset licenses remain in force.
