from __future__ import annotations

import re

from .schema import GroundedAnswer
from .store import EventStore

_STOP_WORDS = {
    "a",
    "an",
    "did",
    "do",
    "does",
    "is",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "who",
}


def _stems(text: str) -> set[str]:
    words = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
    stems: set[str] = set()
    for word in words:
        if word in _STOP_WORDS:
            continue
        if len(word) > 4 and word.endswith("ing"):
            word = word[:-3]
        elif len(word) > 3 and word.endswith("es"):
            word = word[:-2]
        elif len(word) > 3 and word.endswith("s"):
            word = word[:-1]
        stems.add(word)
    return stems


class EvidenceAgent:
    """A deterministic evidence-first retriever.

    Generative adapters can later rewrite the response, but citations are selected
    here and remain mandatory.
    """

    def __init__(self, store: EventStore) -> None:
        self.store = store

    def answer(
        self, question: str, *, stream_id: str | None = None, limit: int = 5
    ) -> GroundedAnswer:
        question_terms = _stems(question)
        candidates = self.store.list(stream_id=stream_id, limit=1_000)
        ranked: list[tuple[int, float, object]] = []
        for event in candidates:
            document = " ".join(
                [event.summary, event.event_type, *(label.name for label in event.labels)]
            )
            overlap = len(question_terms & _stems(document))
            if overlap:
                evidence_score = max(item.score for item in event.evidence)
                ranked.append((overlap, evidence_score, event))
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2].start_ms))
        selected = [item[2] for item in ranked[:limit]]
        if not selected:
            return GroundedAnswer(
                answer="I cannot answer from the available evidence.",
                abstained=True,
                event_ids=[],
                citations=[],
                confidence=0.0,
            )

        citations = list(dict.fromkeys(ev.uri for event in selected for ev in event.evidence))
        statements = [
            f"[{event.start_ms / 1000:.1f}s-{event.end_ms / 1000:.1f}s] {event.summary}"
            for event in selected
        ]
        confidence = min(
            1.0,
            sum(max(ev.score for ev in event.evidence) for event in selected) / len(selected),
        )
        return GroundedAnswer(
            answer=" ".join(statements),
            abstained=False,
            event_ids=[event.event_id for event in selected],
            citations=citations,
            confidence=confidence,
        )
