"""SLO-aware admission and backend routing contracts for StreamSense v3."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SLORequest:
    request_id: str
    deadline_ms: float
    min_quality: float
    privacy_required: bool = False
    required_capabilities: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id cannot be empty")
        if self.deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive")
        if not 0.0 <= self.min_quality <= 1.0:
            raise ValueError("min_quality must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class BackendState:
    name: str
    healthy: bool
    predicted_latency_ms: float
    quality: float
    cost_per_request: float
    privacy_preserving: bool = True
    capabilities: set[str] = field(default_factory=set)
    queue_utilization: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("backend name cannot be empty")
        if self.predicted_latency_ms < 0 or self.cost_per_request < 0:
            raise ValueError("latency and cost cannot be negative")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be in [0, 1]")
        if not 0.0 <= self.queue_utilization <= 1.0:
            raise ValueError("queue_utilization must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class SLOPolicy:
    deadline_safety_margin_ms: float = 0.0
    quality_weight: float = 1.0
    latency_weight: float = 0.35
    cost_weight: float = 0.15
    queue_weight: float = 0.25


@dataclass(frozen=True, slots=True)
class SLODecision:
    admitted: bool
    backend: str | None
    score: float | None
    rejection_reasons: tuple[str, ...] = ()


class SLORouter:
    def __init__(self, policy: SLOPolicy) -> None:
        self.policy = policy

    def decide(self, request: SLORequest, backends: list[BackendState]) -> SLODecision:
        candidates: list[tuple[float, BackendState]] = []
        rejected: set[str] = set()
        available_deadline = request.deadline_ms - self.policy.deadline_safety_margin_ms
        for backend in backends:
            reasons: list[str] = []
            if not backend.healthy:
                reasons.append("unhealthy")
            if backend.predicted_latency_ms > available_deadline:
                reasons.append("deadline")
            if backend.quality < request.min_quality:
                reasons.append("quality")
            if request.privacy_required and not backend.privacy_preserving:
                reasons.append("privacy")
            if not request.required_capabilities <= backend.capabilities:
                reasons.append("capability")
            if reasons:
                rejected.update(reasons)
                continue
            latency_ratio = backend.predicted_latency_ms / request.deadline_ms
            score = (
                self.policy.quality_weight * backend.quality
                - self.policy.latency_weight * latency_ratio
                - self.policy.cost_weight * backend.cost_per_request
                - self.policy.queue_weight * backend.queue_utilization
            )
            candidates.append((score, backend))
        if not candidates:
            return SLODecision(
                admitted=False,
                backend=None,
                score=None,
                rejection_reasons=tuple(sorted(rejected or {"no_backend"})),
            )
        score, selected = max(candidates, key=lambda item: (item[0], item[1].name))
        return SLODecision(admitted=True, backend=selected.name, score=score)
