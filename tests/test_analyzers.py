from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from streamsense.analyzers import FasterWhisperAnalyzer, FrameChangeAnalyzer


@dataclass
class FakeSegment:
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float


class FakeWhisperModel:
    def transcribe(self, _path: str, **_kwargs):
        return (
            [FakeSegment(1.0, 2.5, " adaptive routing works ", -0.1, 0.02)],
            object(),
        )


def test_whisper_adapter_emits_timestamped_transcript(tmp_path) -> None:
    path = tmp_path / "demo.wav"
    path.write_bytes(b"placeholder")
    analyzer = FasterWhisperAnalyzer(model=FakeWhisperModel(), model_name="fake-whisper")
    observations = analyzer.analyze(path, stream_id="demo")
    assert len(observations) == 1
    assert observations[0].start_ms == 1000
    assert observations[0].end_ms == 2500
    assert "adaptive routing works" in observations[0].summary
    assert observations[0].evidence.kind == "transcript"


def test_frame_change_detector_returns_only_large_changes(tmp_path) -> None:
    analyzer = FrameChangeAnalyzer(evidence_dir=tmp_path / "frames", change_threshold=0.2)
    black = np.zeros((20, 20, 3), dtype=np.uint8)
    almost_black = np.full((20, 20, 3), 5, dtype=np.uint8)
    white = np.full((20, 20, 3), 255, dtype=np.uint8)
    changes = analyzer.detect_changes(
        frames=[black, almost_black, white],
        timestamps_ms=[0, 1000, 2000],
        stream_id="camera",
    )
    assert len(changes) == 1
    assert changes[0].start_ms == 1000
    assert changes[0].end_ms == 2000
    assert changes[0].needs_visual_grounding is True
    assert Path(changes[0].evidence.uri.split("#", 1)[0]).is_file()


def test_frame_change_validates_input_lengths(tmp_path) -> None:
    analyzer = FrameChangeAnalyzer(evidence_dir=tmp_path / "frames")
    try:
        analyzer.detect_changes(
            frames=[np.zeros((2, 2, 3), dtype=np.uint8)],
            timestamps_ms=[],
            stream_id="bad",
        )
    except ValueError as error:
        assert "same non-zero length" in str(error)
    else:
        raise AssertionError("invalid aligned input must fail")
