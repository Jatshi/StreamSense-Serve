from streamsense.agent import EvidenceAgent
from streamsense.schema import EventLabel, EventRecord, Evidence
from streamsense.store import EventStore


def sample_event() -> EventRecord:
    return EventRecord(
        event_id="evt_glass_001",
        stream_id="camera-1",
        event_type="fused_alert",
        start_ms=12_000,
        end_ms=15_000,
        summary="Glass-breaking sound followed by a person entering the frame.",
        labels=[EventLabel(name="glass_break", score=0.91)],
        evidence=[
            Evidence(kind="audio", uri="audio.wav#t=12,15", score=0.91),
            Evidence(kind="frame", uri="frames/13000.jpg", score=0.82),
        ],
        route="vlm_escalated",
    )


def test_store_roundtrip_and_search(tmp_path) -> None:
    store = EventStore(tmp_path / "events.db")
    original = sample_event()
    store.upsert(original)
    restored = store.get("evt_glass_001")
    assert restored == original
    assert store.search("glass", limit=5)[0].event_id == "evt_glass_001"


def test_agent_returns_citations(tmp_path) -> None:
    store = EventStore(tmp_path / "events.db")
    store.upsert(sample_event())
    answer = EvidenceAgent(store).answer("When did the glass break?")
    assert answer.abstained is False
    assert answer.event_ids == ["evt_glass_001"]
    assert answer.citations[0].startswith("audio.wav")


def test_agent_abstains_without_evidence(tmp_path) -> None:
    store = EventStore(tmp_path / "events.db")
    answer = EvidenceAgent(store).answer("Who is the person?")
    assert answer.abstained is True
    assert answer.citations == []
