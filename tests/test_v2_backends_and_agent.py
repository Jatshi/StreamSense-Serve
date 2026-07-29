from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

from streamsense import backend_launcher
from streamsense.backend_launcher import build_command
from streamsense.backends import (
    BackendProfile,
    BackendProfiles,
    EngineConfig,
    OpenAICompatibleClient,
)
from streamsense.evidence_agent import (
    AgentEvidence,
    EvidenceAgentBackendAdapter,
    EvidenceAgentRequest,
)


def profile() -> BackendProfile:
    return BackendProfile(
        name="vllm-test",
        kind="vllm",
        base_url="http://backend.local",
        model="test-model",
        engine=EngineConfig(
            model_path="org/model",
            served_model_name="test-model",
            dtype="float16",
        ),
    )


def test_backend_health_and_evidence_agent_adapter() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "test-model"}]})
        payload = json.loads(request.content)
        assert payload["response_format"]["json_schema"]["strict"] is True
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "state": "answer",
                                    "answer": "方案在第3页提出。",
                                    "citations": [
                                        {"evidence_id": "slide-3", "claim": "方案在第3页提出。"}
                                    ],
                                    "confidence": 0.91,
                                    "missing_evidence": [],
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
        )

    client = OpenAICompatibleClient(
        profile(),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    health = client.health()
    assert health.reachable is True
    assert health.advertised_models == ["test-model"]
    response = EvidenceAgentBackendAdapter(client).answer(
        EvidenceAgentRequest(
            request_id="req-001",
            question="方案在哪里提出?",
            evidence=[
                AgentEvidence(
                    evidence_id="slide-3",
                    modality="ocr",
                    text="第三页: 建议采用分层检索方案",
                    page=3,
                )
            ],
        )
    )
    assert response.state == "answer"
    assert response.citations[0].evidence_id == "slide-3"


def test_adapter_rejects_unknown_citation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "state": "answer",
                                    "answer": "unsupported",
                                    "citations": [
                                        {"evidence_id": "invented", "claim": "unsupported"}
                                    ],
                                    "confidence": 0.9,
                                    "missing_evidence": [],
                                }
                            )
                        }
                    }
                ]
            },
        )

    client = OpenAICompatibleClient(
        profile(),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    request = EvidenceAgentRequest(
        request_id="req-002",
        question="What happened?",
        evidence=[AgentEvidence(evidence_id="real", modality="transcript", text="Nothing.")],
    )
    with pytest.raises(ValueError, match="unknown evidence"):
        EvidenceAgentBackendAdapter(client).answer(request)


def test_vllm_launcher_command_is_structured() -> None:
    command = build_command(profile())
    assert command[:3] == [
        command[0],
        "-m",
        "vllm.entrypoints.openai.api_server",
    ]
    assert command[command.index("--model") + 1] == "org/model"
    assert "--quantization" not in command


def test_execute_replaces_launcher_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "backends.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "default_profile": "vllm-test",
                "profiles": [profile().model_dump(mode="json")],
            }
        ),
        encoding="utf-8",
    )
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setenv("PATH", "sentinel-path")

    def fake_execv(executable: str, command: list[str]) -> None:
        calls.append((executable, command))
        raise SystemExit(17)

    monkeypatch.setattr(backend_launcher.os, "execv", fake_execv)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "streamsense-backend",
            "--config",
            str(config_path),
            "--profile",
            "vllm-test",
            "--execute",
        ],
    )

    with pytest.raises(SystemExit, match="17"):
        backend_launcher.main()

    assert calls[0][0] == sys.executable
    assert calls[0][1][1:3] == ["-m", "vllm.entrypoints.openai.api_server"]
    path_parts = backend_launcher.os.environ["PATH"].split(backend_launcher.os.pathsep)
    assert path_parts[0] == str(Path(sys.executable).absolute().parent)
    assert path_parts[1] == "sentinel-path"


def test_checked_in_gpu_matrix_uses_bfloat16_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    profiles = BackendProfiles.load(repo_root / "configs" / "backends.json")
    selected = {
        name: profiles.get(name)
        for name in (
            "vllm-qwen25-vl-3b",
            "vllm-qwen25-vl-3b-fp8",
            "sglang-qwen25-vl-3b",
        )
    }

    assert all(item.engine is not None for item in selected.values())
    assert all(item.engine.dtype == "bfloat16" for item in selected.values() if item.engine)
    assert selected["vllm-qwen25-vl-3b"].engine.quantization == "none"
    assert selected["sglang-qwen25-vl-3b"].engine.quantization == "none"
    assert selected["vllm-qwen25-vl-3b-fp8"].engine.quantization == "fp8"
