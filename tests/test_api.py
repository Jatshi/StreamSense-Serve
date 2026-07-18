from fastapi.testclient import TestClient

from streamsense.api import create_app
from tests.test_media import write_test_wave


def test_health_and_demo_pipeline(tmp_path) -> None:
    app = create_app(database_path=tmp_path / "events.db")
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.headers["x-trace-id"]
    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert "StreamSense" in dashboard.text
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "streamsense_http_requests_total" in metrics.text

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


def test_upload_wave_runs_media_pipeline(tmp_path) -> None:
    source = tmp_path / "source.wav"
    write_test_wave(source)
    app = create_app(database_path=tmp_path / "events.db", media_dir=tmp_path / "media")
    client = TestClient(app)
    with source.open("rb") as media:
        response = client.post(
            "/v1/media/analyze",
            data={"stream_id": "upload-demo"},
            files={"file": ("source.wav", media, "audio/wav")},
        )
    assert response.status_code == 201
    payload = response.json()
    assert payload["events_created"] >= 1
    events = client.get("/v1/events", params={"stream_id": "upload-demo"}).json()
    assert events[0]["evidence"][0]["kind"] == "audio"
    evidence = client.get(f"/v1/evidence/{events[0]['event_id']}/0")
    assert evidence.status_code == 200
    assert evidence.content[:4] == b"RIFF"
    metrics = client.get("/metrics").text
    assert 'streamsense_route_decisions_total{route="lightweight"}' in metrics


def test_upload_rejects_unsupported_media(tmp_path) -> None:
    app = create_app(database_path=tmp_path / "events.db", media_dir=tmp_path / "media")
    client = TestClient(app)
    response = client.post(
        "/v1/media/analyze",
        data={"stream_id": "bad"},
        files={"file": ("bad.exe", b"not media", "application/octet-stream")},
    )
    assert response.status_code == 415
