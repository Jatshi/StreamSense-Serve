from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

from streamsense.media import AudioEnergyAnalyzer, MediaPipeline
from streamsense.routing import RouterConfig, RuleRouter
from streamsense.store import EventStore


def write_test_wave(path: Path) -> None:
    sample_rate = 16_000
    silence = np.zeros(sample_rate, dtype=np.float32)
    time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    tone = 0.7 * np.sin(2 * math.pi * 880 * time)
    signal = np.concatenate([silence, tone, silence])
    pcm = np.clip(signal * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def test_audio_energy_analyzer_finds_active_interval(tmp_path) -> None:
    media_path = tmp_path / "tone.wav"
    write_test_wave(media_path)
    observations = AudioEnergyAnalyzer(window_seconds=0.5, hop_seconds=0.25).analyze(
        media_path, stream_id="tone-demo"
    )
    assert observations
    assert min(item.start_ms for item in observations) <= 1_250
    assert max(item.end_ms for item in observations) >= 2_750
    assert all(item.evidence.kind == "audio" for item in observations)


def test_media_pipeline_persists_grounded_events(tmp_path) -> None:
    media_path = tmp_path / "tone.wav"
    write_test_wave(media_path)
    store = EventStore(tmp_path / "events.db")
    pipeline = MediaPipeline(
        analyzers=[AudioEnergyAnalyzer(window_seconds=0.5, hop_seconds=0.25)],
        router=RuleRouter(RouterConfig(exploration_rate=0.0)),
        store=store,
    )
    result = pipeline.analyze(media_path, stream_id="tone-demo")
    assert result.events_created >= 1
    persisted = store.list(stream_id="tone-demo")
    assert len(persisted) == result.events_created
    assert all(event.is_grounded for event in persisted)
    assert all(event.model_versions["audio_energy"] == "rms-v1" for event in persisted)


def test_audio_analyzer_rejects_unsupported_pcm_width(tmp_path) -> None:
    media_path = tmp_path / "invalid.wav"
    with wave.open(str(media_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(1)
        output.setframerate(8_000)
        output.writeframes(bytes([128] * 800))

    analyzer = AudioEnergyAnalyzer()
    try:
        analyzer.analyze(media_path, stream_id="invalid")
    except ValueError as error:
        assert "16-bit PCM" in str(error)
    else:
        raise AssertionError("unsupported sample width must fail at the input boundary")
