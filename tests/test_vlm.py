from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from streamsense.analyzers import FrameChangeAnalyzer
from streamsense.media import ObservationEscalationError
from streamsense.vlm import OpenAIVLMEnhancer


def test_openai_vlm_enhancer_preserves_evidence_and_parses_json(tmp_path) -> None:
    frame_analyzer = FrameChangeAnalyzer(evidence_dir=tmp_path, change_threshold=0.1)
    observation = frame_analyzer.detect_changes(
        frames=[
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.full((4, 4, 3), 255, dtype=np.uint8),
        ],
        timestamps_ms=[0, 1000],
        stream_id="demo",
    )[0]

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        image_url = payload["messages"][1]["content"][1]["image_url"]["url"]
        assert image_url.startswith("data:image/x-portable-pixmap;base64,")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "```json\n"
                                '{"summary":"A bright scene replaces a dark frame.",'
                                '"label":"lighting_change","confidence":0.92,'
                                '"risk_score":0.1}\n```'
                            )
                        }
                    }
                ]
            },
        )

    enhancer = OpenAIVLMEnhancer(
        base_url="http://vlm.local",
        model="test-vlm",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    enhanced = enhancer.enhance(observation)
    assert enhanced.summary == "A bright scene replaces a dark frame."
    assert enhanced.label.name == "lighting_change"
    assert enhanced.evidence.uri == observation.evidence.uri
    assert enhanced.model_version == "test-vlm"


def test_vlm_enhancer_skips_non_frame_observation(tmp_path) -> None:
    from streamsense.media import AudioEnergyAnalyzer
    from tests.test_media import write_test_wave

    media = tmp_path / "tone.wav"
    write_test_wave(media)
    observation = AudioEnergyAnalyzer().analyze(media, stream_id="audio")[0]
    enhancer = OpenAIVLMEnhancer(base_url="http://unused", model="unused")
    assert enhancer.enhance(observation) is observation


def test_vlm_enhancer_retries_invalid_schema_once(tmp_path) -> None:
    observation = FrameChangeAnalyzer(evidence_dir=tmp_path, change_threshold=0.1).detect_changes(
        frames=[
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.full((4, 4, 3), 255, dtype=np.uint8),
        ],
        timestamps_ms=[0, 1000],
        stream_id="retry",
    )[0]
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"summary":"Visible change.","label":"change",'
                                '"confidence":0.8,"risk_score":0.1}'
                            )
                        }
                    }
                ]
            },
        )

    enhancer = OpenAIVLMEnhancer(
        base_url="http://vlm.local",
        model="test-vlm",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert enhancer.enhance(observation).label.name == "change"
    assert calls == 2


def test_vlm_enhancer_wraps_persistent_invalid_output(tmp_path) -> None:
    observation = FrameChangeAnalyzer(evidence_dir=tmp_path, change_threshold=0.1).detect_changes(
        frames=[
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.full((4, 4, 3), 255, dtype=np.uint8),
        ],
        timestamps_ms=[0, 1000],
        stream_id="invalid",
    )[0]

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"choices": []})

    enhancer = OpenAIVLMEnhancer(
        base_url="http://vlm.local",
        model="test-vlm",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ObservationEscalationError):
        enhancer.enhance(observation)
