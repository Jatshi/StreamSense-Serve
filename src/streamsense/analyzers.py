from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from .media import Observation, calibrated_uncertainty
from .schema import EventLabel, Evidence


class FasterWhisperAnalyzer:
    """Timestamped ASR adapter with lazy model initialization."""

    supported_suffixes = frozenset(
        {".wav", ".mp3", ".flac", ".m4a", ".mp4", ".mov", ".mkv", ".webm"}
    )

    def __init__(
        self,
        *,
        model: Any | None = None,
        model_name: str = "small",
        device: str = "cuda",
        compute_type: str = "float16",
        cache_dir: str | Path | None = None,
        language: str | None = None,
    ) -> None:
        self._model = model
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.cache_dir = str(cache_dir) if cache_dir else None
        self.language = language

    def supports(self, media_path: Path) -> bool:
        return media_path.suffix.lower() in self.supported_suffixes

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as error:
                raise RuntimeError(
                    "faster-whisper is required; install streamsense-serve[asr]"
                ) from error
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.cache_dir,
            )
        return self._model

    def analyze(self, media_path: Path, *, stream_id: str) -> list[Observation]:
        del stream_id
        if not self.supports(media_path):
            return []
        segments, _info = self._get_model().transcribe(
            str(media_path),
            language=self.language,
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
        )
        observations: list[Observation] = []
        for segment in segments:
            text = str(segment.text).strip()
            if not text:
                continue
            start_ms = max(0, round(float(segment.start) * 1000))
            end_ms = max(start_ms + 1, round(float(segment.end) * 1000))
            confidence = math.exp(min(0.0, float(segment.avg_logprob)))
            confidence *= 1.0 - float(segment.no_speech_prob)
            confidence = min(0.999, max(0.001, confidence))
            observations.append(
                Observation(
                    event_type="speech_transcript",
                    start_ms=start_ms,
                    end_ms=end_ms,
                    summary=f"Transcript: {text}",
                    label=EventLabel(name="speech", score=confidence),
                    evidence=Evidence(
                        kind="transcript",
                        uri=(
                            f"{media_path.as_posix()}#t={start_ms / 1000:.3f},{end_ms / 1000:.3f}"
                        ),
                        score=confidence,
                        description=text,
                    ),
                    risk_score=0.1,
                    uncertainty=calibrated_uncertainty(confidence),
                    model_name="asr",
                    model_version=f"faster-whisper:{self.model_name}:{self.compute_type}",
                )
            )
        return observations


class FrameChangeAnalyzer:
    """Sample video frames and persist evidence for large scene changes."""

    supported_suffixes = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi"})

    def __init__(
        self,
        *,
        evidence_dir: str | Path,
        sample_fps: float = 1.0,
        change_threshold: float = 0.18,
        max_frames: int = 3_600,
    ) -> None:
        if sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        if not 0.0 <= change_threshold <= 1.0:
            raise ValueError("change_threshold must be between 0 and 1")
        self.evidence_dir = Path(evidence_dir)
        self.sample_fps = sample_fps
        self.change_threshold = change_threshold
        self.max_frames = max_frames

    def supports(self, media_path: Path) -> bool:
        return media_path.suffix.lower() in self.supported_suffixes

    def analyze(self, media_path: Path, *, stream_id: str) -> list[Observation]:
        if not self.supports(media_path):
            return []
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError(
                "OpenCV is required for video; install streamsense-serve[media]"
            ) from error
        capture = cv2.VideoCapture(str(media_path))
        if not capture.isOpened():
            raise ValueError(f"cannot decode video: {media_path}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
        stride = max(1, round(source_fps / self.sample_fps))
        frames: list[np.ndarray] = []
        timestamps: list[int] = []
        index = 0
        try:
            while len(frames) < self.max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                if index % stride == 0:
                    frames.append(frame)
                    timestamps.append(round(index * 1000 / source_fps))
                index += 1
        finally:
            capture.release()
        return self.detect_changes(frames=frames, timestamps_ms=timestamps, stream_id=stream_id)

    def detect_changes(
        self,
        *,
        frames: list[np.ndarray],
        timestamps_ms: list[int],
        stream_id: str,
    ) -> list[Observation]:
        if not frames or len(frames) != len(timestamps_ms):
            raise ValueError("frames and timestamps must have the same non-zero length")
        observations: list[Observation] = []
        for previous, current, start_ms, end_ms in zip(
            frames[:-1],
            frames[1:],
            timestamps_ms[:-1],
            timestamps_ms[1:],
            strict=True,
        ):
            if previous.shape != current.shape:
                raise ValueError("all frames must share the same shape")
            previous_gray = previous.astype(np.float32).mean(axis=2)
            current_gray = current.astype(np.float32).mean(axis=2)
            score = float(np.mean(np.abs(current_gray - previous_gray)) / 255.0)
            if score < self.change_threshold:
                continue
            evidence_path = self._write_ppm(current, stream_id=stream_id, timestamp_ms=end_ms)
            observations.append(
                Observation(
                    event_type="visual_change",
                    start_ms=start_ms,
                    end_ms=max(start_ms + 1, end_ms),
                    summary=f"Large visual change detected near {end_ms / 1000:.2f}s.",
                    label=EventLabel(name="visual_change", score=min(0.999, score)),
                    evidence=Evidence(
                        kind="frame",
                        uri=f"{evidence_path.as_posix()}#t={end_ms / 1000:.3f}",
                        score=min(0.999, score),
                        description="Evidence frame after a large mean absolute pixel change.",
                    ),
                    risk_score=0.2,
                    uncertainty=calibrated_uncertainty(min(0.999, score)),
                    needs_visual_grounding=True,
                    model_name="frame_change",
                    model_version="mad-v1",
                )
            )
        return observations

    def _write_ppm(self, frame: np.ndarray, *, stream_id: str, timestamp_ms: int) -> Path:
        safe_stream = re.sub(r"[^A-Za-z0-9_.-]+", "_", stream_id)[:80] or "stream"
        directory = self.evidence_dir / safe_stream
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"frame_{timestamp_ms:012d}.ppm"
        rgb = frame[..., ::-1] if frame.shape[2] == 3 else frame
        height, width = rgb.shape[:2]
        with path.open("wb") as output:
            output.write(f"P6\n{width} {height}\n255\n".encode())
            output.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        return path
