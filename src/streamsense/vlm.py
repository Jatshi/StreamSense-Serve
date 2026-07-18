from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from .media import Observation, calibrated_uncertainty
from .schema import EventLabel, Evidence


class VLMDescription(BaseModel):
    summary: str = Field(min_length=1, max_length=1_000)
    label: str = Field(min_length=1, max_length=100)
    confidence: float = Field(ge=0.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=1.0)


class OpenAIVLMEnhancer:
    """Enhance a visual observation through an OpenAI-compatible local server."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = client or httpx.Client(timeout=timeout_seconds)

    def enhance(self, observation: Observation) -> Observation:
        if observation.evidence.kind != "frame":
            return observation
        evidence_path = Path(observation.evidence.uri.split("#", 1)[0])
        image_url = self._data_url(evidence_path)
        response = self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "temperature": 0,
                "max_tokens": 200,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Describe only visible evidence. Return JSON with keys summary, "
                            "label, confidence, risk_score. Do not identify people or infer intent."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "This frame follows a detected visual change. Describe the "
                                    "visible change conservatively and assign a short label."
                                ),
                            },
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ],
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        description = VLMDescription.model_validate(self._parse_json(content))
        return Observation(
            event_type="vlm_visual_event",
            start_ms=observation.start_ms,
            end_ms=observation.end_ms,
            summary=description.summary,
            label=EventLabel(name=description.label, score=description.confidence),
            evidence=Evidence(
                kind="frame",
                uri=observation.evidence.uri,
                score=description.confidence,
                description=description.summary,
            ),
            risk_score=description.risk_score,
            uncertainty=calibrated_uncertainty(description.confidence),
            cross_modal_conflict=observation.cross_modal_conflict,
            needs_visual_grounding=False,
            model_name="vlm",
            model_version=self.model,
        )

    @staticmethod
    def _parse_json(content: str) -> dict[str, object]:
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ValueError("VLM response must be a JSON object")
        return parsed

    @staticmethod
    def _data_url(path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError(path)
        media_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".ppm": "image/x-portable-pixmap",
            ".webp": "image/webp",
        }.get(path.suffix.lower(), "application/octet-stream")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{media_type};base64,{encoded}"
