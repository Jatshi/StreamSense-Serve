from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .backends import OpenAICompatibleClient

AnswerState = Literal["answer", "clarify", "abstain"]
EvidenceModality = Literal["audio", "video", "frame", "ocr", "transcript", "metadata"]


class AgentEvidence(BaseModel):
    evidence_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]{1,128}$")
    modality: EvidenceModality
    text: str = Field(min_length=1, max_length=8_000)
    uri: str | None = Field(default=None, max_length=2_048)
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, gt=0)
    speaker: str | None = Field(default=None, max_length=256)
    page: int | None = Field(default=None, ge=1)
    score: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_interval(self) -> AgentEvidence:
        if self.start_ms is not None and self.end_ms is not None and self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self


class EvidenceAgentRequest(BaseModel):
    request_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]{3,128}$")
    question: str = Field(min_length=2, max_length=2_000)
    evidence: list[AgentEvidence] = Field(default_factory=list, max_length=100)
    conversation_id: str | None = Field(default=None, max_length=128)
    stream_id: str | None = Field(default=None, max_length=128)
    language: str = Field(default="zh-CN", max_length=32)


class AgentCitation(BaseModel):
    evidence_id: str
    claim: str = Field(min_length=1, max_length=2_000)


class EvidenceAgentResponse(BaseModel):
    request_id: str
    state: AnswerState
    answer: str = Field(max_length=8_000)
    citations: list[AgentCitation] = Field(default_factory=list, max_length=50)
    confidence: float = Field(ge=0.0, le=1.0)
    missing_evidence: list[str] = Field(default_factory=list, max_length=20)
    model: str
    backend_profile: str

    @model_validator(mode="after")
    def enforce_grounding(self) -> EvidenceAgentResponse:
        if self.state == "answer" and not self.citations:
            raise ValueError("answer state requires at least one citation")
        if self.state in {"clarify", "abstain"} and not self.missing_evidence:
            raise ValueError("clarify/abstain states require missing_evidence")
        return self


class _ModelOutput(BaseModel):
    state: AnswerState
    answer: str = Field(max_length=8_000)
    citations: list[AgentCitation] = Field(default_factory=list, max_length=50)
    confidence: float = Field(ge=0.0, le=1.0)
    missing_evidence: list[str] = Field(default_factory=list, max_length=20)


class EvidenceAgentBackendAdapter:
    SYSTEM_PROMPT = (
        "You are an evidence-constrained multimodal assistant. Use only the provided evidence. "
        "Every factual claim in an answer must cite an evidence_id. If evidence is insufficient, "
        "choose clarify or abstain and name the missing evidence. Never invent timestamps, "
        "speakers, pages, or sources. Return only JSON matching the supplied schema."
    )

    def __init__(self, client: OpenAICompatibleClient) -> None:
        self.client = client

    def answer(
        self, request: EvidenceAgentRequest, *, model: str | None = None
    ) -> EvidenceAgentResponse:
        evidence_payload = [item.model_dump(mode="json") for item in request.evidence]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": request.question,
                        "language": request.language,
                        "evidence": evidence_payload,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        body = self.client.chat(
            messages,
            model=model,
            response_schema=_ModelOutput.model_json_schema(),
        )
        try:
            content = body["choices"][0]["message"]["content"]
        except (IndexError, KeyError, TypeError) as error:
            raise ValueError("backend response is missing choices[0].message.content") from error
        if not isinstance(content, str):
            raise ValueError("backend message content must be text")
        parsed = self._parse_json(content)
        output = _ModelOutput.model_validate(parsed)
        allowed_ids = {item.evidence_id for item in request.evidence}
        cited_ids = {citation.evidence_id for citation in output.citations}
        unknown = cited_ids - allowed_ids
        if unknown:
            raise ValueError(f"backend cited unknown evidence ids: {sorted(unknown)}")
        return EvidenceAgentResponse(
            request_id=request.request_id,
            **output.model_dump(),
            model=model or self.client.profile.model,
            backend_profile=self.client.profile.name,
        )

    @staticmethod
    def _parse_json(content: str) -> dict[str, object]:
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ValueError("agent response must be a JSON object")
        return parsed
