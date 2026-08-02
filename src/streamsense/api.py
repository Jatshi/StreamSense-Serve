from __future__ import annotations

import os
import secrets
import wave
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import httpx
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from .agent import EvidenceAgent
from .analyzers import FasterWhisperAnalyzer, FrameChangeAnalyzer
from .backends import OpenAICompatibleClient, load_backend_profiles
from .evidence_agent import (
    EvidenceAgentBackendAdapter,
    EvidenceAgentRequest,
    EvidenceAgentResponse,
)
from .feedback import ExportSummary, FeedbackStore, FeedbackSubmission, StoredFeedback
from .media import AudioEnergyAnalyzer, MediaPipeline, PipelineResult
from .model_registry import (
    ActivationRequest,
    ActivationState,
    ModelRegistry,
    RollbackRequest,
)
from .routing import RouteDecision, RouteFeatures, RouterConfig, RuleRouter
from .schema import EventRecord, GroundedAnswer, QueryRequest
from .store import EventStore
from .telemetry import ROUTE_DECISIONS, configure_telemetry
from .vlm import OpenAIVLMEnhancer


def create_app(
    database_path: str | Path | None = None,
    media_dir: str | Path | None = None,
    *,
    backend_config_path: str | Path | None = None,
    feedback_database_path: str | Path | None = None,
    model_manifest_path: str | Path | None = None,
    model_state_path: str | Path | None = None,
    feedback_export_dir: str | Path | None = None,
) -> FastAPI:
    resolved_path = Path(database_path or os.environ.get("STREAMSENSE_DATABASE", "data/events.db"))
    store = EventStore(resolved_path)
    feedback_store = FeedbackStore(
        feedback_database_path
        or os.environ.get("STREAMSENSE_FEEDBACK_DATABASE", resolved_path.with_name("feedback.db"))
    )
    resolved_export_dir = Path(
        feedback_export_dir
        or os.environ.get("STREAMSENSE_FEEDBACK_EXPORT_DIR", resolved_path.parent / "exports")
    )
    backend_profiles = load_backend_profiles(backend_config_path)
    manifest_path = Path(
        model_manifest_path
        or os.environ.get("STREAMSENSE_MODEL_MANIFEST", "models/serve_manifest.json")
    )
    state_path = Path(
        model_state_path
        or os.environ.get("STREAMSENSE_MODEL_STATE", resolved_path.parent / "active_model.json")
    )
    model_registry = ModelRegistry(manifest_path, state_path) if manifest_path.is_file() else None
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
        version="2.0.0",
        description=(
            "Evidence-first audiovisual event service with adaptive routing, "
            "OpenAI-compatible inference, and a feedback data flywheel."
        ),
    )
    app.state.store = store
    app.state.feedback_store = feedback_store
    app.state.backend_profiles = backend_profiles
    app.state.model_registry = model_registry
    app.state.router = router
    configure_telemetry(app)

    @app.get("/health")
    def health() -> dict[str, str | bool]:
        return {
            "status": "ok",
            "database": str(resolved_path),
            "backend_configured": backend_profiles is not None,
            "model_registry_configured": model_registry is not None,
        }

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "index.html")

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

    @app.get("/v1/evidence/{event_id}/{evidence_index}")
    def get_evidence(event_id: str, evidence_index: int) -> FileResponse:
        event = store.get(event_id)
        if event is None or not 0 <= evidence_index < len(event.evidence):
            raise HTTPException(status_code=404, detail="evidence not found")
        evidence_path = Path(event.evidence[evidence_index].uri.split("#", 1)[0]).resolve()
        if not evidence_path.is_relative_to(resolved_media_dir) or not evidence_path.is_file():
            raise HTTPException(status_code=404, detail="evidence file is unavailable")
        return FileResponse(evidence_path)

    @app.post("/v1/route", response_model=RouteDecision)
    def route(features: RouteFeatures) -> RouteDecision:
        decision = router.decide(features)
        ROUTE_DECISIONS.labels(decision.route).inc()
        return decision

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
            result = await run_in_threadpool(
                media_pipeline.analyze,
                destination,
                stream_id=stream_id,
            )
            for route_name, count in (
                ("lightweight", result.lightweight_events),
                ("vlm_escalated", result.escalated_events),
                ("human_review", result.human_review_events),
            ):
                if count:
                    ROUTE_DECISIONS.labels(route_name).inc(count)
            return result
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

    def require_admin(authorization: str | None = Header(default=None)) -> None:
        configured_token = os.environ.get("STREAMSENSE_ADMIN_TOKEN")
        if not configured_token:
            raise HTTPException(status_code=503, detail="admin API is disabled")
        supplied = authorization.removeprefix("Bearer ").strip() if authorization else ""
        if not supplied or not secrets.compare_digest(supplied, configured_token):
            raise HTTPException(status_code=401, detail="invalid admin token")

    def require_feedback_token(
        authorization: str | None = Header(default=None),
    ) -> None:
        configured_token = os.environ.get("STREAMSENSE_FEEDBACK_TOKEN")
        if not configured_token:
            raise HTTPException(status_code=503, detail="feedback API is disabled")
        supplied = authorization.removeprefix("Bearer ").strip() if authorization else ""
        if not supplied or not secrets.compare_digest(supplied, configured_token):
            raise HTTPException(status_code=401, detail="invalid feedback token")

    def require_inference_token(
        authorization: str | None = Header(default=None),
    ) -> None:
        configured_token = os.environ.get("STREAMSENSE_INFERENCE_TOKEN")
        if not configured_token:
            raise HTTPException(status_code=503, detail="v2 inference API is disabled")
        supplied = authorization.removeprefix("Bearer ").strip() if authorization else ""
        if not supplied or not secrets.compare_digest(supplied, configured_token):
            raise HTTPException(status_code=401, detail="invalid inference token")

    def backend_selection() -> tuple[OpenAICompatibleClient, str]:
        if backend_profiles is None:
            raise HTTPException(status_code=503, detail="inference backend is not configured")
        profile_name: str | None = None
        model_name: str | None = None
        if model_registry is not None and (active := model_registry.active_model()) is not None:
            profile_name = active.backend_profile
            model_name = active.served_model_name
        try:
            profile = backend_profiles.get(profile_name)
        except KeyError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return OpenAICompatibleClient(profile), model_name or profile.model

    @app.get(
        "/v2/inference/health",
        dependencies=[Depends(require_inference_token)],
    )
    def inference_health(profile: str | None = None) -> dict[str, object]:
        if backend_profiles is None:
            return {"configured": False, "detail": "set STREAMSENSE_BACKEND_CONFIG"}
        try:
            selected = backend_profiles.get(profile)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        client = OpenAICompatibleClient(selected)
        try:
            health_result = client.health()
        finally:
            client.close()
        return {
            "configured": True,
            "health": health_result.model_dump(mode="json"),
        }

    @app.post(
        "/v2/evidence-agent/query",
        response_model=EvidenceAgentResponse,
        dependencies=[Depends(require_inference_token)],
    )
    def evidence_agent_query(request: EvidenceAgentRequest) -> EvidenceAgentResponse:
        client, model_name = backend_selection()
        try:
            return EvidenceAgentBackendAdapter(client).answer(request, model=model_name)
        except httpx.HTTPError as error:
            raise HTTPException(
                status_code=502,
                detail=f"backend request failed: {type(error).__name__}",
            ) from error
        except (ValueError, KeyError) as error:
            raise HTTPException(
                status_code=502, detail=f"invalid backend output: {error}"
            ) from error
        finally:
            client.close()

    @app.post(
        "/v2/feedback",
        response_model=StoredFeedback,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(require_feedback_token)],
    )
    def submit_feedback(submission: FeedbackSubmission) -> StoredFeedback:
        return feedback_store.add(submission)

    @app.post(
        "/v2/feedback/export",
        response_model=ExportSummary,
        dependencies=[Depends(require_admin)],
    )
    def export_feedback() -> ExportSummary:
        return feedback_store.export_training_data(resolved_export_dir)

    @app.get("/v2/models", dependencies=[Depends(require_admin)])
    def models() -> dict[str, object]:
        if model_registry is None:
            raise HTTPException(status_code=503, detail="model registry is not configured")
        state = model_registry.state()
        return {
            "manifest": model_registry.manifest.model_dump(mode="json"),
            "activation": state.model_dump(mode="json") if state else None,
        }

    @app.post(
        "/v2/models/activate",
        response_model=ActivationState,
        dependencies=[Depends(require_admin)],
    )
    def activate_model(request: ActivationRequest) -> ActivationState:
        if model_registry is None:
            raise HTTPException(status_code=503, detail="model registry is not configured")
        try:
            candidate = model_registry.manifest.get(request.model_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        if backend_profiles is None:
            raise HTTPException(status_code=503, detail="backend profiles are not configured")
        try:
            profile = backend_profiles.get(candidate.backend_profile)
        except KeyError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        client = OpenAICompatibleClient(profile)
        try:
            health = client.health()
        finally:
            client.close()
        if not health.reachable:
            raise HTTPException(status_code=503, detail=f"backend is unreachable: {health.detail}")
        if health.advertised_models and candidate.served_model_name not in health.advertised_models:
            raise HTTPException(
                status_code=409,
                detail="backend does not advertise the requested served_model_name",
            )
        try:
            return model_registry.activate(
                request.model_id,
                expected_revision=request.expected_revision,
                reason=request.reason,
            )
        except (KeyError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post(
        "/v2/models/rollback",
        response_model=ActivationState,
        dependencies=[Depends(require_admin)],
    )
    def rollback_model(request: RollbackRequest) -> ActivationState:
        if model_registry is None:
            raise HTTPException(status_code=503, detail="model registry is not configured")
        try:
            return model_registry.rollback(reason=request.reason)
        except (KeyError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    return app


app = create_app()
