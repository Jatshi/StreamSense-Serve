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


def test_v2_feedback_export_requires_admin_token(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("STREAMSENSE_ADMIN_TOKEN", "test-secret")
    monkeypatch.setenv("STREAMSENSE_FEEDBACK_TOKEN", "feedback-secret")
    app = create_app(
        database_path=tmp_path / "events.db",
        feedback_database_path=tmp_path / "feedback.db",
        feedback_export_dir=tmp_path / "exports",
        model_manifest_path=tmp_path / "missing.json",
    )
    client = TestClient(app)
    submission = {
        "request": {
            "request_id": "req-api-feedback",
            "question": "Who proposed the plan?",
            "evidence": [
                {
                    "evidence_id": "seg-1",
                    "modality": "transcript",
                    "text": "Alice proposed the plan.",
                }
            ],
        },
        "response": {
            "request_id": "req-api-feedback",
            "state": "answer",
            "answer": "Bob proposed it.",
            "citations": [{"evidence_id": "seg-1", "claim": "Bob proposed it."}],
            "confidence": 0.6,
            "missing_evidence": [],
            "model": "test",
            "backend_profile": "vllm-test",
        },
        "rating": "correction",
        "corrected_answer": "Alice proposed the plan. [seg-1]",
        "consent_for_training": True,
        "source_license": "user-consented-private-evaluation",
    }
    created = client.post(
        "/v2/feedback",
        json=submission,
        headers={"Authorization": "Bearer feedback-secret"},
    )
    assert created.status_code == 201
    duplicate = client.post(
        "/v2/feedback",
        json=submission,
        headers={"Authorization": "Bearer feedback-secret"},
    )
    assert duplicate.json()["duplicate"] is True
    assert client.post("/v2/feedback/export").status_code == 401
    exported = client.post(
        "/v2/feedback/export",
        headers={"Authorization": "Bearer test-secret"},
    )
    assert exported.status_code == 200
    assert exported.json()["sft_examples"] == 1


def test_v2_inference_requires_its_own_token(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("STREAMSENSE_INFERENCE_TOKEN", "inference-secret")
    monkeypatch.delenv("STREAMSENSE_BACKEND_CONFIG", raising=False)
    app = create_app(
        database_path=tmp_path / "events.db",
        model_manifest_path=tmp_path / "missing.json",
    )
    client = TestClient(app)
    payload = {
        "request_id": "req-auth-001",
        "question": "What is supported?",
        "evidence": [],
    }
    assert client.post("/v2/evidence-agent/query", json=payload).status_code == 401
    response = client.post(
        "/v2/evidence-agent/query",
        json=payload,
        headers={"Authorization": "Bearer inference-secret"},
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "inference backend is not configured"


def test_v2_inference_health_requires_token(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("STREAMSENSE_INFERENCE_TOKEN", raising=False)
    monkeypatch.delenv("STREAMSENSE_BACKEND_CONFIG", raising=False)
    app = create_app(
        database_path=tmp_path / "events.db",
        model_manifest_path=tmp_path / "missing.json",
    )
    client = TestClient(app)
    assert client.get("/v2/inference/health").status_code == 503
    monkeypatch.setenv("STREAMSENSE_INFERENCE_TOKEN", "inference-secret")
    assert client.get("/v2/inference/health").status_code == 401
    response = client.get(
        "/v2/inference/health",
        headers={"Authorization": "Bearer inference-secret"},
    )
    assert response.status_code == 200
    assert response.json()["configured"] is False


def test_v2_models_requires_admin_token(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("STREAMSENSE_ADMIN_TOKEN", raising=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """{
          "schema_version": 1,
          "models": [{
            "model_id": "safe-model",
            "served_model_name": "safe-model",
            "revision": "revision-1",
            "backend_profile": "vllm-test",
            "base_model": "org/model",
            "status": "validated"
          }]
        }""",
        encoding="utf-8",
    )
    app = create_app(
        database_path=tmp_path / "events.db",
        model_manifest_path=manifest,
        model_state_path=tmp_path / "active.json",
    )
    client = TestClient(app)
    assert client.get("/v2/models").status_code == 503
    monkeypatch.setenv("STREAMSENSE_ADMIN_TOKEN", "admin-secret")
    assert client.get("/v2/models").status_code == 401
    response = client.get(
        "/v2/models",
        headers={"Authorization": "Bearer admin-secret"},
    )
    assert response.status_code == 200
    assert response.json()["manifest"]["models"][0]["model_id"] == "safe-model"
