from __future__ import annotations

import os
import time
import uuid

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

HTTP_REQUESTS = Counter(
    "streamsense_http_requests_total",
    "HTTP requests handled by StreamSense.",
    ("method", "path", "status"),
)
HTTP_LATENCY = Histogram(
    "streamsense_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ("method", "path"),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
ROUTE_DECISIONS = Counter(
    "streamsense_route_decisions_total",
    "Adaptive routing decisions.",
    ("route",),
)


def configure_telemetry(app: FastAPI) -> None:
    @app.middleware("http")
    async def observe_request(request: Request, call_next):
        started = time.perf_counter()
        trace_id = request.headers.get("x-trace-id") or uuid.uuid4().hex
        try:
            response = await call_next(request)
        except Exception:
            HTTP_REQUESTS.labels(request.method, request.url.path, "500").inc()
            HTTP_LATENCY.labels(request.method, request.url.path).observe(
                time.perf_counter() - started
            )
            raise
        response.headers["X-Trace-ID"] = trace_id
        HTTP_REQUESTS.labels(request.method, request.url.path, str(response.status_code)).inc()
        HTTP_LATENCY.labels(request.method, request.url.path).observe(time.perf_counter() - started)
        return response

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        _enable_opentelemetry(app)


def _enable_opentelemetry(app: FastAPI) -> None:
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as error:
        raise RuntimeError(
            "OTLP endpoint is configured but observability dependencies are missing"
        ) from error

    provider = TracerProvider(resource=Resource.create({"service.name": "streamsense-serve"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
