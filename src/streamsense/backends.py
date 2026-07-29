from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, field_validator, model_validator

BackendKind = Literal["openai", "vllm", "sglang"]
QuantizationMode = Literal["none", "auto", "awq", "gptq", "bitsandbytes", "fp8"]


class EngineConfig(BaseModel):
    """Launch-time settings shared by the documented vLLM/SGLang profiles."""

    model_path: str = Field(min_length=1, max_length=2_048)
    served_model_name: str = Field(min_length=1, max_length=256)
    revision: str | None = Field(default=None, max_length=256)
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    quantization: QuantizationMode = "none"
    tensor_parallel_size: int = Field(default=1, ge=1, le=8)
    max_model_len: int = Field(default=8192, ge=512, le=262_144)
    gpu_memory_utilization: float = Field(default=0.88, gt=0.1, le=0.98)
    trust_remote_code: bool = False
    extra_args: list[str] = Field(default_factory=list, max_length=64)

    @field_validator("extra_args")
    @classmethod
    def validate_extra_args(cls, args: list[str]) -> list[str]:
        for arg in args:
            if not arg.startswith("--") or any(char in arg for char in "\r\n;&|`$"):
                raise ValueError("extra_args must contain safe long-form command arguments")
        return args


class BackendProfile(BaseModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    kind: BackendKind
    base_url: str = Field(min_length=8, max_length=2_048)
    model: str = Field(min_length=1, max_length=256)
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]{1,127}$")
    timeout_seconds: float = Field(default=120.0, ge=1.0, le=3_600.0)
    max_retries: int = Field(default=1, ge=0, le=5)
    default_max_tokens: int = Field(default=512, ge=1, le=32_768)
    default_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    health_path: str = Field(default="/health", pattern=r"^/[A-Za-z0-9_./-]*$")
    engine: EngineConfig | None = None

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        value = value.rstrip("/")
        if not re.match(r"^https?://", value):
            raise ValueError("base_url must use http or https")
        return value

    def authorization_headers(self) -> dict[str, str]:
        if not self.api_key_env:
            return {}
        api_key = os.environ.get(self.api_key_env)
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}


class BackendProfiles(BaseModel):
    schema_version: Literal[1] = 1
    default_profile: str
    profiles: list[BackendProfile] = Field(min_length=1)

    @field_validator("profiles")
    @classmethod
    def unique_names(cls, profiles: list[BackendProfile]) -> list[BackendProfile]:
        names = [profile.name for profile in profiles]
        if len(names) != len(set(names)):
            raise ValueError("backend profile names must be unique")
        return profiles

    @model_validator(mode="after")
    def validate_default_profile(self) -> BackendProfiles:
        if self.default_profile not in {profile.name for profile in self.profiles}:
            raise ValueError("default_profile must reference a configured profile")
        return self

    def get(self, name: str | None = None) -> BackendProfile:
        selected = name or self.default_profile
        for profile in self.profiles:
            if profile.name == selected:
                return profile
        raise KeyError(f"backend profile not found: {selected}")

    @classmethod
    def load(cls, path: str | Path) -> BackendProfiles:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class BackendHealth(BaseModel):
    profile: str
    kind: BackendKind
    base_url: str
    configured_model: str
    reachable: bool
    status_code: int | None = None
    advertised_models: list[str] = Field(default_factory=list)
    detail: str


class OpenAICompatibleClient:
    """Small synchronous client usable with OpenAI, vLLM, and SGLang APIs."""

    def __init__(
        self,
        profile: BackendProfile,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.profile = profile
        self.client = client or httpx.Client(
            timeout=profile.timeout_seconds,
            headers=profile.authorization_headers(),
            transport=httpx.HTTPTransport(retries=profile.max_retries),
        )

    def health(self) -> BackendHealth:
        try:
            health_response = self.client.get(
                f"{self.profile.base_url}{self.profile.health_path}",
                headers=self.profile.authorization_headers(),
            )
            if health_response.status_code >= 400:
                health_response.raise_for_status()
            models_response = self.client.get(
                f"{self.profile.base_url}/v1/models",
                headers=self.profile.authorization_headers(),
            )
            advertised_models: list[str] = []
            if models_response.is_success:
                payload = models_response.json()
                advertised_models = [
                    str(item["id"])
                    for item in payload.get("data", [])
                    if isinstance(item, dict) and item.get("id")
                ]
            return BackendHealth(
                profile=self.profile.name,
                kind=self.profile.kind,
                base_url=self.profile.base_url,
                configured_model=self.profile.model,
                reachable=True,
                status_code=health_response.status_code,
                advertised_models=advertised_models,
                detail="backend health endpoint responded",
            )
        except (httpx.HTTPError, json.JSONDecodeError, TypeError, ValueError) as error:
            return BackendHealth(
                profile=self.profile.name,
                kind=self.profile.kind,
                base_url=self.profile.base_url,
                configured_model=self.profile.model,
                reachable=False,
                detail=f"{type(error).__name__}: {error}",
            )

    def close(self) -> None:
        self.client.close()

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self.profile.model,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens or self.profile.default_max_tokens,
            "temperature": (
                self.profile.default_temperature if temperature is None else temperature
            ),
        }
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "evidence_agent_response",
                    "strict": True,
                    "schema": response_schema,
                },
            }
        response = self.client.post(
            f"{self.profile.base_url}/v1/chat/completions",
            json=payload,
            headers=self.profile.authorization_headers(),
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("backend response must be a JSON object")
        return body


def load_backend_profiles(path: str | Path | None = None) -> BackendProfiles | None:
    configured = path or os.environ.get("STREAMSENSE_BACKEND_CONFIG")
    if not configured:
        return None
    return BackendProfiles.load(configured)
