from __future__ import annotations

import json

import pytest

from streamsense.evidence_agent import (
    AgentCitation,
    AgentEvidence,
    EvidenceAgentRequest,
    EvidenceAgentResponse,
)
from streamsense.feedback import FeedbackStore, FeedbackSubmission
from streamsense.model_registry import ModelRegistry


def feedback_submission() -> FeedbackSubmission:
    request = EvidenceAgentRequest(
        request_id="req-feedback-001",
        question="谁提出了方案?",
        evidence=[
            AgentEvidence(
                evidence_id="seg-1",
                modality="transcript",
                text="张老师: 我建议采用两阶段检索。",
                start_ms=1000,
                end_ms=4000,
            )
        ],
    )
    response = EvidenceAgentResponse(
        request_id=request.request_id,
        state="answer",
        answer="李老师提出了方案。",
        citations=[AgentCitation(evidence_id="seg-1", claim="李老师提出了方案。")],
        confidence=0.7,
        model="test",
        backend_profile="vllm-test",
    )
    return FeedbackSubmission(
        request=request,
        response=response,
        rating="correction",
        corrected_answer="张老师提出了两阶段检索方案。[seg-1]",
        consent_for_training=True,
        source_license="user-consented-private-evaluation",
        reason_codes=["wrong_speaker"],
    )


def test_feedback_deduplicates_and_exports_candidates(tmp_path) -> None:
    store = FeedbackStore(tmp_path / "feedback.db")
    first = store.add(feedback_submission())
    duplicate = store.add(feedback_submission())
    assert first.duplicate is False
    assert duplicate.duplicate is True
    assert duplicate.feedback_id == first.feedback_id

    summary = store.export_training_data(tmp_path / "export")
    assert summary.source_records == 1
    assert summary.sft_examples == 1
    assert summary.dpo_examples == 1
    assert summary.bridge_examples == 0
    assert summary.eligible_records == 1
    sft = json.loads((tmp_path / "export" / "sft_candidates.jsonl").read_text("utf-8"))
    dpo = json.loads((tmp_path / "export" / "dpo_candidates.jsonl").read_text("utf-8"))
    manifest = json.loads((tmp_path / "export" / "export_manifest.json").read_text("utf-8"))
    assert sft["messages"][-1]["content"].startswith("张老师")
    assert dpo["rejected"] == "李老师提出了方案。"
    assert manifest["schema_version"] == 2
    assert manifest["source"]["content_hashes"] == [first.content_hash]
    assert len(manifest["outputs"]["sft"]["sha256"]) == 64


def test_feedback_without_explicit_consent_is_not_exported(tmp_path) -> None:
    submission = feedback_submission().model_copy(
        update={"consent_for_training": False, "source_license": None}
    )
    store = FeedbackStore(tmp_path / "feedback.db")
    store.add(submission)
    summary = store.export_training_data(tmp_path / "export")
    assert summary.source_records == 1
    assert summary.eligible_records == 0
    assert summary.skipped_no_consent == 1
    assert summary.sft_examples == 0
    assert summary.dpo_examples == 0
    assert (tmp_path / "export" / "consented_feedback_raw.jsonl").read_text("utf-8") == ""


def test_training_consent_requires_source_license() -> None:
    with pytest.raises(ValueError, match="source_license"):
        FeedbackSubmission.model_validate(
            {
                **feedback_submission().model_dump(mode="json"),
                "source_license": None,
            }
        )


def test_structured_correction_exports_evidenceagent_bridge(tmp_path) -> None:
    base = feedback_submission()
    corrected_response = EvidenceAgentResponse(
        request_id=base.request.request_id,
        state="answer",
        answer="张老师提出了两阶段检索方案。",
        citations=[AgentCitation(evidence_id="seg-1", claim="张老师提出了两阶段检索方案。")],
        confidence=0.98,
        model="human-reviewed",
        backend_profile="human-review",
    )
    structured = base.model_copy(
        update={"corrected_answer": None, "corrected_response": corrected_response}
    )
    store = FeedbackStore(tmp_path / "feedback.db")
    store.add(structured)
    summary = store.export_training_data(tmp_path / "export")
    assert summary.bridge_examples == 1
    bridge = json.loads((tmp_path / "export" / "evidenceagent_bridge.jsonl").read_text("utf-8"))
    assert bridge["schema_version"] == 2
    assert bridge["target"]["state"] == "answer"
    assert bridge["target"]["citations"][0]["evidence_id"] == "seg-1"
    assert bridge["metadata"]["source_license"] == "user-consented-private-evaluation"
    assert bridge["metadata"]["created_at"]
    raw = json.loads((tmp_path / "export" / "consented_feedback_raw.jsonl").read_text("utf-8"))
    assert raw["created_at"] == bridge["metadata"]["created_at"]


def test_registry_activation_and_rollback_are_revision_guarded(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "model_id": "model-a",
                        "served_model_name": "served-a",
                        "revision": "rev-a",
                        "backend_profile": "vllm-test",
                        "base_model": "org/a",
                        "status": "validated",
                    },
                    {
                        "model_id": "model-b",
                        "served_model_name": "served-b",
                        "revision": "rev-b",
                        "backend_profile": "vllm-test",
                        "base_model": "org/b",
                        "status": "validated",
                    },
                    {
                        "model_id": "model-c",
                        "served_model_name": "served-c",
                        "revision": "rev-c",
                        "backend_profile": "vllm-test",
                        "base_model": "org/c",
                        "status": "candidate",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    state_path = tmp_path / "active.json"
    registry = ModelRegistry(manifest, state_path)
    registry.activate("model-a", expected_revision="rev-a", reason="initial validation")
    active = registry.activate("model-b", expected_revision="rev-b", reason="quality gate passed")
    assert active.active_model_id == "model-b"
    assert state_path.is_file()
    rolled_back = registry.rollback(reason="regression detected")
    assert rolled_back.active_model_id == "model-a"

    with pytest.raises(ValueError, match="validated"):
        registry.activate("model-c", expected_revision="rev-c", reason="not ready")
    with pytest.raises(ValueError, match="expected_revision"):
        registry.activate("model-a", expected_revision="wrong", reason="stale command")
