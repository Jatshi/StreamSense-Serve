from fastapi.testclient import TestClient

from streamsense.api import create_app


def test_health_and_demo_pipeline(tmp_path) -> None:
    app = create_app(database_path=tmp_path / "events.db")
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    response = client.post(
        "/v1/events",
        json={
            "event_id": "evt_api_001",
            "stream_id": "demo",
            "event_type": "speech_claim",
            "start_ms": 0,
            "end_ms": 2_000,
            "summary": "The speaker proposes an adaptive router.",
            "labels": [{"name": "proposal", "score": 0.9}],
            "evidence": [{"kind": "audio", "uri": "demo.wav#t=0,2", "score": 0.9}],
            "route": "lightweight",
        },
    )
    assert response.status_code == 201
    answer = client.post("/v1/query", json={"question": "What did the speaker propose?"})
    assert answer.status_code == 200
    assert answer.json()["event_ids"] == ["evt_api_001"]
