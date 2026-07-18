from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, status

from .agent import EvidenceAgent
from .routing import RouteDecision, RouteFeatures, RouterConfig, RuleRouter
from .schema import EventRecord, GroundedAnswer, QueryRequest
from .store import EventStore


def create_app(database_path: str | Path | None = None) -> FastAPI:
    resolved_path = Path(database_path or os.environ.get("STREAMSENSE_DATABASE", "data/events.db"))
    store = EventStore(resolved_path)
    router = RuleRouter(RouterConfig())
    agent = EvidenceAgent(store)

    app = FastAPI(
        title="StreamSense-Serve",
        version="0.1.0",
        description="Evidence-first audiovisual event service with adaptive VLM routing.",
    )
    app.state.store = store
    app.state.router = router

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "database": str(resolved_path)}

    @app.post("/v1/events", response_model=EventRecord, status_code=status.HTTP_201_CREATED)
    def create_event(event: EventRecord) -> EventRecord:
        store.upsert(event)
        return event

    @app.get("/v1/events", response_model=list[EventRecord])
    def list_events(
        stream_id: str | None = None,
        limit: int = Query(default=100, ge=1, le=1_000),
    ) -> list[EventRecord]:
        return store.list(stream_id=stream_id, limit=limit)

    @app.get("/v1/events/{event_id}", response_model=EventRecord)
    def get_event(event_id: str) -> EventRecord:
        event = store.get(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")
        return event

    @app.post("/v1/route", response_model=RouteDecision)
    def route(features: RouteFeatures) -> RouteDecision:
        return router.decide(features)

    @app.post("/v1/query", response_model=GroundedAnswer)
    def query(request: QueryRequest) -> GroundedAnswer:
        return agent.answer(
            request.question,
            stream_id=request.stream_id,
            limit=request.limit,
        )

    return app


app = create_app()
