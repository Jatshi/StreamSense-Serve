from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

EvidenceKind = Literal["audio", "frame", "video", "transcript", "metadata"]
RouteName = Literal["lightweight", "vlm_escalated", "abstained", "human_review"]


class EventLabel(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    score: float = Field(ge=0.0, le=1.0)


class Evidence(BaseModel):
    kind: EvidenceKind
    uri: str = Field(min_length=1, max_length=2_048)
    score: float = Field(ge=0.0, le=1.0)
    description: str | None = Field(default=None, max_length=1_000)


class EventRecord(BaseModel):
    event_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]{3,128}$")
    stream_id: str = Field(min_length=1, max_length=128)
    event_type: str = Field(min_length=1, max_length=100)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    summary: str = Field(min_length=1, max_length=4_000)
    labels: list[EventLabel] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    route: RouteName
    model_versions: dict[str, str] = Field(default_factory=dict)
    latency_ms: float | None = Field(default=None, ge=0.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    review_state: Literal["unreviewed", "accepted", "rejected"] = "unreviewed"

    @model_validator(mode="after")
    def validate_interval_and_grounding(self) -> EventRecord:
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        if self.route != "abstained" and not self.evidence:
            raise ValueError("non-abstained events require at least one evidence item")
        return self

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def is_grounded(self) -> bool:
        return bool(self.evidence)


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2_000)
    stream_id: str | None = Field(default=None, max_length=128)
    limit: int = Field(default=5, ge=1, le=20)


class GroundedAnswer(BaseModel):
    answer: str
    abstained: bool
    event_ids: list[str]
    citations: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
