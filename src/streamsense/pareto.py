"""Latency/cost/quality/safety Pareto analysis for serving experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ServingPoint:
    name: str
    latency_ms: float
    cost: float
    quality: float
    safety: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("point name cannot be empty")
        if self.latency_ms < 0 or self.cost < 0:
            raise ValueError("latency and cost cannot be negative")
        if not 0.0 <= self.quality <= 1.0 or not 0.0 <= self.safety <= 1.0:
            raise ValueError("quality and safety must be in [0, 1]")


def _dominates(left: ServingPoint, right: ServingPoint) -> bool:
    weakly_better = (
        left.latency_ms <= right.latency_ms
        and left.cost <= right.cost
        and left.quality >= right.quality
        and left.safety >= right.safety
    )
    strictly_better = (
        left.latency_ms < right.latency_ms
        or left.cost < right.cost
        or left.quality > right.quality
        or left.safety > right.safety
    )
    return weakly_better and strictly_better


def pareto_frontier(points: list[ServingPoint]) -> list[ServingPoint]:
    if len({point.name for point in points}) != len(points):
        raise ValueError("serving point names must be unique")
    frontier = [
        candidate
        for candidate in points
        if not any(_dominates(other, candidate) for other in points if other is not candidate)
    ]
    return sorted(frontier, key=lambda item: (item.latency_ms, -item.quality, item.name))
