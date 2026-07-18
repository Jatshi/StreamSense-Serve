from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from streamsense.schema import EventLabel, EventRecord, Evidence


def test_event_requires_valid_time_and_evidence() -> None:
    event = EventRecord(
        event_id="evt_demo_001",
        stream_id="demo",
        event_type="sound",
        start_ms=1_000,
        end_ms=2_500,
        summary="A short impact sound is present.",
        labels=[EventLabel(name="impact", score=0.8)],
        evidence=[Evidence(kind="audio", uri="media/demo.wav#t=1,2.5", score=0.8)],
        route="lightweight",
        created_at=datetime.now(UTC),
    )
    assert event.duration_ms == 1_500
    assert event.is_grounded


def test_event_rejects_inverted_interval() -> None:
    with pytest.raises(ValidationError):
        EventRecord(
            event_id="evt_bad",
            stream_id="demo",
            event_type="sound",
            start_ms=2_000,
            end_ms=1_000,
            summary="invalid",
            evidence=[Evidence(kind="audio", uri="a.wav", score=0.5)],
            route="lightweight",
        )


def test_non_abstained_event_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        EventRecord(
            event_id="evt_bad",
            stream_id="demo",
            event_type="visual",
            start_ms=0,
            end_ms=100,
            summary="unsupported",
            evidence=[],
            route="lightweight",
        )
