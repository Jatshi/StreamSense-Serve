from __future__ import annotations

from streamsense.faults import FaultKind, FaultSchedule
from streamsense.pareto import ServingPoint, pareto_frontier
from streamsense.slo import BackendState, SLOPolicy, SLORequest, SLORouter
from streamsense.telemetry import sanitize_genai_attributes


def test_slo_router_selects_only_backend_that_meets_deadline_and_privacy() -> None:
    router = SLORouter(SLOPolicy(deadline_safety_margin_ms=10))
    request = SLORequest(
        request_id="req-1",
        deadline_ms=150,
        min_quality=0.7,
        privacy_required=True,
        required_capabilities={"vision"},
    )
    backends = [
        BackendState(
            name="remote-fast",
            healthy=True,
            predicted_latency_ms=60,
            quality=0.9,
            cost_per_request=0.01,
            privacy_preserving=False,
            capabilities={"vision"},
        ),
        BackendState(
            name="local-vllm",
            healthy=True,
            predicted_latency_ms=120,
            quality=0.82,
            cost_per_request=0.02,
            privacy_preserving=True,
            capabilities={"vision"},
        ),
    ]

    decision = router.decide(request, backends)

    assert decision.backend == "local-vllm"
    assert decision.admitted


def test_slo_router_rejects_when_no_backend_can_meet_contract() -> None:
    router = SLORouter(SLOPolicy())
    request = SLORequest(request_id="req-2", deadline_ms=50, min_quality=0.95)
    decision = router.decide(
        request,
        [
            BackendState(
                name="overloaded",
                healthy=True,
                predicted_latency_ms=300,
                quality=0.8,
                cost_per_request=0.01,
            )
        ],
    )
    assert not decision.admitted
    assert decision.backend is None
    assert "deadline" in decision.rejection_reasons


def test_genai_telemetry_drops_prompts_and_unknown_attributes() -> None:
    sanitized = sanitize_genai_attributes(
        {
            "gen_ai.request.model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "gen_ai.prompt": "private patient conversation",
            "user.email": "private@example.com",
            "gen_ai.usage.input_tokens": 42,
        }
    )
    assert sanitized == {
        "gen_ai.request.model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "gen_ai.usage.input_tokens": 42,
    }


def test_fault_schedule_is_deterministic_and_explicit() -> None:
    schedule = FaultSchedule(every_n_requests=3, fault=FaultKind.TIMEOUT)
    assert [schedule.fault_for(index) for index in range(1, 7)] == [
        None,
        None,
        FaultKind.TIMEOUT,
        None,
        None,
        FaultKind.TIMEOUT,
    ]


def test_pareto_frontier_removes_dominated_serving_points() -> None:
    points = [
        ServingPoint(name="balanced", latency_ms=100, cost=0.02, quality=0.85, safety=0.95),
        ServingPoint(name="dominated", latency_ms=130, cost=0.03, quality=0.80, safety=0.90),
        ServingPoint(name="quality", latency_ms=180, cost=0.04, quality=0.95, safety=0.98),
    ]
    assert {point.name for point in pareto_frontier(points)} == {"balanced", "quality"}
