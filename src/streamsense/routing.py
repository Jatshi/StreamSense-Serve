from __future__ import annotations

import random
from dataclasses import dataclass

from pydantic import BaseModel, Field


class RouteFeatures(BaseModel):
    risk_score: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    cross_modal_conflict: float = Field(ge=0.0, le=1.0)
    needs_visual_grounding: bool = False


class RouterConfig(BaseModel):
    risk_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    uncertainty_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    conflict_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    exploration_rate: float = Field(default=0.02, ge=0.0, le=1.0)
    seed: int = 17


@dataclass(frozen=True)
class RouteDecision:
    route: str
    reasons: tuple[str, ...]


class RuleRouter:
    def __init__(self, config: RouterConfig) -> None:
        self.config = config
        self._random = random.Random(config.seed)

    def decide(self, features: RouteFeatures) -> RouteDecision:
        reasons: list[str] = []
        if features.risk_score >= self.config.risk_threshold:
            reasons.append("risk")
        if features.uncertainty >= self.config.uncertainty_threshold:
            reasons.append("uncertainty")
        if features.cross_modal_conflict >= self.config.conflict_threshold:
            reasons.append("cross_modal_conflict")
        if features.needs_visual_grounding:
            reasons.append("visual_grounding")
        if not reasons and self._random.random() < self.config.exploration_rate:
            reasons.append("exploration")
        if reasons:
            return RouteDecision(route="vlm_escalated", reasons=tuple(reasons))
        return RouteDecision(route="lightweight", reasons=("confident",))
