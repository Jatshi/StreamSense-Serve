from __future__ import annotations

import os
import wave
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status

from .agent import EvidenceAgent
from .analyzers import FasterWhisperAnalyzer, FrameChangeAnalyzer
from .media import AudioEnergyAnalyzer, MediaPipeline, PipelineResult
from .routing import RouteDecision, RouteFeatures, RouterConfig, RuleRouter
from .schema import EventRecord, GroundedAnswer, QueryRequest
from .store import EventStore
from .vlm import OpenAIVLMEnhancer


def create_app(
    database_path: str | Path | None = None,
    media_dir: str | Path | None = None,
) -> FastAPI:
    resolved_path = Path(database_path or os.environ.get("STREAMSENSE_DATABASE", "data/events.db"))
    store = EventStore(resolved_path)
    router = RuleRouter(RouterConfig())
    agent = EvidenceAgent(store)
    resolved_media_dir = Path(
        media_dir or os.environ.get("STREAMSENSE_MEDIA_DIR", "data/media")
    ).resolve()
    resolved_media_dir.mkdir(parents=True, exist_ok=True)
    analyzers = [
        AudioEnergyAnalyzer(),
        FrameChangeAnalyzer(evidence_dir=resolved_media_dir / "evidence"),
    ]
    if asr_model := os.environ.get("STREAMSENSE_ASR_MODEL"):
        analyzers.append(
            FasterWhisperAnalyzer(
                model_name=asr_model,
                device=os.environ.get("STREAMSENSE_ASR_DEVICE", "cuda"),
                compute_type=os.environ.get("STREAMSENSE_ASR_COMPUTE_TYPE", "float16"),
                cache_dir=os.environ.get("STREAMSENSE_MODEL_CACHE"),
                language=os.environ.get("STREAMSENSE_ASR_LANGUAGE") or None,
            )
        )
    vlm_base_url = os.environ.get("STREAMSENSE_VLM_BASE_URL")
    vlm_model = os.environ.get("STREAMSENSE_VLM_MODEL")
    escalator = (
        OpenAIVLMEnhancer(base_url=vlm_base_url, model=vlm_model)
        if vlm_base_url and vlm_model
        else None
    )
    media_pipeline = MediaPipeline(
        analyzers=analyzers,
        router=router,
        store=store,
        escalator=escalator,
    )

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

    @app.post(
        "/v1/media/analyze",
        response_model=PipelineResult,
        status_code=status.HTTP_201_CREATED,
    )
    async def analyze_media(
        stream_id: Annotated[str, Form(min_length=1, max_length=128)],
        file: Annotated[UploadFile, File()],
    ) -> PipelineResult:
        original_name = Path(file.filename or "upload").name
        suffix = Path(original_name).suffix.lower()
        allowed_suffixes = {".wav", ".mp4", ".mov", ".mkv", ".webm", ".avi"}
        if suffix not in allowed_suffixes:
            await file.close()
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"supported media suffixes: {', '.join(sorted(allowed_suffixes))}",
            )
        destination = resolved_media_dir / f"{uuid4().hex}{suffix}"
        written = 0
        max_bytes = 200 * 1024 * 1024
        try:
            with destination.open("xb") as output:
                while chunk := await file.read(1024 * 1024):
                    written += len(chunk)
                    if written > max_bytes:
                        raise HTTPException(status_code=413, detail="media exceeds 200 MiB")
                    output.write(chunk)
            return media_pipeline.analyze(destination, stream_id=stream_id)
        except HTTPException:
            destination.unlink(missing_ok=True)
            raise
        except (ValueError, wave.Error) as error:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=503, detail=str(error)) from error
        finally:
            await file.close()

    return app


app = create_app()
