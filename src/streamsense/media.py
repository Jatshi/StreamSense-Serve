from __future__ import annotations

import hashlib
import math
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .routing import RouteFeatures, RuleRouter
from .schema import EventLabel, EventRecord, Evidence, RouteName
from .store import EventStore


@dataclass(frozen=True)
class Observation:
    event_type: str
    start_ms: int
    end_ms: int
    summary: str
    label: EventLabel
    evidence: Evidence
    risk_score: float
    uncertainty: float
    cross_modal_conflict: float = 0.0
    needs_visual_grounding: bool = False
    model_name: str = "unknown"
    model_version: str = "unknown"


class MediaAnalyzer(Protocol):
    def analyze(self, media_path: Path, *, stream_id: str) -> list[Observation]: ...


class ObservationEscalator(Protocol):
    def enhance(self, observation: Observation) -> Observation: ...


class ObservationEscalationError(RuntimeError):
    """Raised when an optional heavyweight model cannot enhance evidence."""


@dataclass(frozen=True)
class PipelineResult:
    stream_id: str
    media_path: str
    observations: int
    events_created: int
    lightweight_events: int
    escalated_events: int
    human_review_events: int
    elapsed_ms: float


class AudioEnergyAnalyzer:
    """Detect sustained audio activity in uncompressed 16-bit PCM WAV files.

    This is deliberately a transparent signal-level baseline. It finds candidate
    windows for downstream ASR/audio-event models without assigning a semantic
    sound label that the evidence cannot support.
    """

    def __init__(
        self,
        *,
        window_seconds: float = 1.0,
        hop_seconds: float = 0.5,
        activity_threshold: float = 0.35,
    ) -> None:
        if window_seconds <= 0 or hop_seconds <= 0:
            raise ValueError("window_seconds and hop_seconds must be positive")
        if not 0.0 <= activity_threshold <= 1.0:
            raise ValueError("activity_threshold must be between 0 and 1")
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        self.activity_threshold = activity_threshold

    def analyze(self, media_path: Path, *, stream_id: str) -> list[Observation]:
        del stream_id  # Reserved for analyzers with stream-specific calibration.
        if media_path.suffix.lower() != ".wav":
            raise ValueError("AudioEnergyAnalyzer currently accepts .wav input")
        samples, sample_rate = self._read_pcm(media_path)
        if samples.size == 0:
            return []

        window = max(1, round(self.window_seconds * sample_rate))
        hop = max(1, round(self.hop_seconds * sample_rate))
        active: list[tuple[int, int, float]] = []
        for start in range(0, samples.size, hop):
            end = min(start + window, samples.size)
            if end - start < max(1, window // 4):
                break
            rms = float(np.sqrt(np.mean(np.square(samples[start:end]), dtype=np.float64)))
            score = min(0.999, rms / (rms + 0.05))
            if score >= self.activity_threshold:
                active.append(
                    (
                        round(start * 1000 / sample_rate),
                        round(end * 1000 / sample_rate),
                        score,
                    )
                )
        merged = self._merge(active, max_gap_ms=round(self.hop_seconds * 1000))
        return [self._to_observation(media_path, interval) for interval in merged]

    @staticmethod
    def supports(media_path: Path) -> bool:
        return media_path.suffix.lower() == ".wav"

    @staticmethod
    def _read_pcm(media_path: Path) -> tuple[np.ndarray, int]:
        with wave.open(str(media_path), "rb") as source:
            if source.getcomptype() != "NONE" or source.getsampwidth() != 2:
                raise ValueError("input must be uncompressed 16-bit PCM WAV")
            channels = source.getnchannels()
            sample_rate = source.getframerate()
            frames = source.readframes(source.getnframes())
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        return samples, sample_rate

    @staticmethod
    def _merge(
        active: list[tuple[int, int, float]], *, max_gap_ms: int
    ) -> list[tuple[int, int, float]]:
        if not active:
            return []
        merged: list[tuple[int, int, float]] = [active[0]]
        for start, end, score in active[1:]:
            last_start, last_end, last_score = merged[-1]
            if start <= last_end + max_gap_ms:
                merged[-1] = (last_start, max(last_end, end), max(last_score, score))
            else:
                merged.append((start, end, score))
        return merged

    @staticmethod
    def _to_observation(media_path: Path, interval: tuple[int, int, float]) -> Observation:
        start_ms, end_ms, score = interval
        uncertainty = 1.0 - min(1.0, abs(score - 0.5) * 2.0)
        return Observation(
            event_type="audio_activity",
            start_ms=start_ms,
            end_ms=end_ms,
            summary=f"Elevated audio energy detected from {start_ms / 1000:.2f}s to "
            f"{end_ms / 1000:.2f}s.",
            label=EventLabel(name="audio_activity", score=score),
            evidence=Evidence(
                kind="audio",
                uri=f"{media_path.as_posix()}#t={start_ms / 1000:.3f},{end_ms / 1000:.3f}",
                score=score,
                description="PCM waveform interval supporting the activity detection.",
            ),
            risk_score=0.15,
            uncertainty=uncertainty,
            model_name="audio_energy",
            model_version="rms-v1",
        )


class MediaPipeline:
    def __init__(
        self,
        *,
        analyzers: list[MediaAnalyzer],
        router: RuleRouter,
        store: EventStore,
        escalator: ObservationEscalator | None = None,
    ) -> None:
        if not analyzers:
            raise ValueError("at least one analyzer is required")
        self.analyzers = analyzers
        self.router = router
        self.store = store
        self.escalator = escalator

    def analyze(self, media_path: str | Path, *, stream_id: str) -> PipelineResult:
        started = time.perf_counter()
        resolved = Path(media_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        observations: list[Observation] = []
        for analyzer in self.analyzers:
            supports = getattr(analyzer, "supports", None)
            if supports is not None and not supports(resolved):
                continue
            observations.extend(analyzer.analyze(resolved, stream_id=stream_id))
        lightweight = 0
        escalated = 0
        human_review = 0
        for observation in observations:
            decision = self.router.decide(
                RouteFeatures(
                    risk_score=observation.risk_score,
                    uncertainty=observation.uncertainty,
                    cross_modal_conflict=observation.cross_modal_conflict,
                    needs_visual_grounding=observation.needs_visual_grounding,
                )
            )
            route: RouteName = decision.route
            if route == "vlm_escalated":
                if self.escalator is None:
                    route = "human_review"
                else:
                    try:
                        observation = self.escalator.enhance(observation)
                    except ObservationEscalationError:
                        route = "human_review"
            lightweight += route == "lightweight"
            escalated += route == "vlm_escalated"
            human_review += route == "human_review"
            event_id = self._event_id(stream_id, observation)
            elapsed_ms = (time.perf_counter() - started) * 1000
            self.store.upsert(
                EventRecord(
                    event_id=event_id,
                    stream_id=stream_id,
                    event_type=observation.event_type,
                    start_ms=observation.start_ms,
                    end_ms=observation.end_ms,
                    summary=observation.summary,
                    labels=[observation.label],
                    evidence=[observation.evidence],
                    route=route,
                    model_versions={observation.model_name: observation.model_version},
                    latency_ms=elapsed_ms,
                )
            )
        return PipelineResult(
            stream_id=stream_id,
            media_path=str(resolved),
            observations=len(observations),
            events_created=len(observations),
            lightweight_events=lightweight,
            escalated_events=escalated,
            human_review_events=human_review,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    @staticmethod
    def _event_id(stream_id: str, observation: Observation) -> str:
        payload = (
            f"{stream_id}|{observation.event_type}|{observation.start_ms}|"
            f"{observation.end_ms}|{observation.model_version}"
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()[:20]
        return f"evt_{digest}"


def calibrated_uncertainty(probability: float) -> float:
    """Return normalized binary uncertainty, 1 at p=.5 and 0 at p in {0, 1}."""
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be finite and between 0 and 1")
    return 1.0 - abs(probability - 0.5) * 2.0
