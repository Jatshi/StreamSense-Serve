from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from streamsense.api import create_app


def main() -> int:
    with TemporaryDirectory() as directory:
        app = create_app(Path(directory) / "events.db")
        client = TestClient(app)
        assert client.get("/health").json()["status"] == "ok"
        event = {
            "event_id": "evt_smoke_001",
            "stream_id": "smoke",
            "event_type": "sound",
            "start_ms": 1000,
            "end_ms": 2500,
            "summary": "A synthetic impact sound is present.",
            "labels": [{"name": "impact", "score": 0.8}],
            "evidence": [{"kind": "audio", "uri": "smoke.wav#t=1,2.5", "score": 0.8}],
            "route": "lightweight",
        }
        assert client.post("/v1/events", json=event).status_code == 201
        answer = client.post("/v1/query", json={"question": "When was the impact sound?"})
        assert answer.status_code == 200
        assert answer.json()["event_ids"] == ["evt_smoke_001"]
    print("StreamSense smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
