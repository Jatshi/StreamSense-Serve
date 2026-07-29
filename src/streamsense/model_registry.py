from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ModelArtifact(BaseModel):
    model_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]{2,128}$")
    served_model_name: str = Field(min_length=1, max_length=256)
    revision: str = Field(min_length=1, max_length=256)
    backend_profile: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    base_model: str = Field(min_length=1, max_length=512)
    adapter_path: str | None = Field(default=None, max_length=2_048)
    artifact_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    status: Literal["candidate", "validated", "deprecated"] = "candidate"
    notes: str = Field(default="", max_length=2_000)


class ModelManifest(BaseModel):
    schema_version: Literal[1] = 1
    models: list[ModelArtifact] = Field(min_length=1)

    @field_validator("models")
    @classmethod
    def unique_model_ids(cls, models: list[ModelArtifact]) -> list[ModelArtifact]:
        ids = [model.model_id for model in models]
        if len(ids) != len(set(ids)):
            raise ValueError("model_id values must be unique")
        return models

    def get(self, model_id: str) -> ModelArtifact:
        for model in self.models:
            if model.model_id == model_id:
                return model
        raise KeyError(f"model not found: {model_id}")


class ActivationState(BaseModel):
    schema_version: Literal[1] = 1
    active_model_id: str
    previous_model_ids: list[str] = Field(default_factory=list, max_length=20)
    updated_at: datetime
    reason: str = Field(min_length=1, max_length=1_000)


class ModelRegistry:
    """Atomic model-selection contract; process launch stays outside the API."""

    def __init__(self, manifest_path: str | Path, state_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.state_path = Path(state_path)
        self._lock = threading.RLock()
        self.manifest = ModelManifest.model_validate_json(
            self.manifest_path.read_text(encoding="utf-8")
        )

    def state(self) -> ActivationState | None:
        if not self.state_path.is_file():
            return None
        return ActivationState.model_validate_json(self.state_path.read_text(encoding="utf-8"))

    def active_model(self) -> ModelArtifact | None:
        state = self.state()
        return self.manifest.get(state.active_model_id) if state else None

    def activate(
        self,
        model_id: str,
        *,
        expected_revision: str,
        reason: str,
    ) -> ActivationState:
        with self._lock:
            candidate = self.manifest.get(model_id)
            if candidate.status != "validated":
                raise ValueError("only validated model artifacts may be activated")
            if candidate.revision != expected_revision:
                raise ValueError("expected_revision does not match the manifest")
            current = self.state()
            previous = list(current.previous_model_ids) if current else []
            if current and current.active_model_id != model_id:
                previous.append(current.active_model_id)
            state = ActivationState(
                active_model_id=model_id,
                previous_model_ids=previous[-20:],
                updated_at=datetime.now(timezone.utc),
                reason=reason,
            )
            self._write_state(state)
            return state

    def rollback(self, *, reason: str) -> ActivationState:
        with self._lock:
            current = self.state()
            if current is None or not current.previous_model_ids:
                raise ValueError("no previous model is available for rollback")
            target_id = current.previous_model_ids[-1]
            target = self.manifest.get(target_id)
            if target.status == "deprecated":
                raise ValueError("refusing to roll back to a deprecated model")
            state = ActivationState(
                active_model_id=target_id,
                previous_model_ids=current.previous_model_ids[:-1],
                updated_at=datetime.now(timezone.utc),
                reason=reason,
            )
            self._write_state(state)
            return state

    def _write_state(self, state: ActivationState) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(f"{self.state_path.suffix}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            output.write(state.model_dump_json(indent=2))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, self.state_path)
        if os.name != "nt":
            self.state_path.chmod(0o600)


class ActivationRequest(BaseModel):
    model_id: str
    expected_revision: str
    reason: str = Field(min_length=3, max_length=1_000)


class RollbackRequest(BaseModel):
    reason: str = Field(min_length=3, max_length=1_000)
