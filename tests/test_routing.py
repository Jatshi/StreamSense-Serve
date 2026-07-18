from streamsense.routing import RouteDecision, RouteFeatures, RouterConfig, RuleRouter


def test_high_risk_always_escalates() -> None:
    router = RuleRouter(RouterConfig(risk_threshold=0.7, uncertainty_threshold=0.6))
    decision = router.decide(
        RouteFeatures(risk_score=0.9, uncertainty=0.1, cross_modal_conflict=0.0)
    )
    assert decision.route == "vlm_escalated"
    assert "risk" in decision.reasons


def test_low_risk_confident_event_stays_lightweight() -> None:
    router = RuleRouter(RouterConfig(exploration_rate=0.0))
    decision = router.decide(
        RouteFeatures(risk_score=0.1, uncertainty=0.1, cross_modal_conflict=0.1)
    )
    assert decision == RouteDecision(route="lightweight", reasons=("confident",))


def test_visual_grounding_request_escalates() -> None:
    router = RuleRouter(RouterConfig())
    decision = router.decide(
        RouteFeatures(
            risk_score=0.1,
            uncertainty=0.1,
            cross_modal_conflict=0.1,
            needs_visual_grounding=True,
        )
    )
    assert decision.route == "vlm_escalated"
    assert "visual_grounding" in decision.reasons
