from streamsense.evaluation import (
    RoutingExample,
    evaluate_router,
    normalize_words,
    percentile,
    word_error_rate,
)
from streamsense.routing import RouteFeatures, RouterConfig, RuleRouter


def test_word_error_rate_exact_match_ignores_case_and_punctuation() -> None:
    assert word_error_rate("Hello, WORLD!", "hello world") == 0.0


def test_word_error_rate_counts_substitution_insertion_and_deletion() -> None:
    assert word_error_rate("a b c", "a x c") == 1 / 3
    assert word_error_rate("a b", "a b c") == 1 / 2
    assert word_error_rate("a b c", "a c") == 1 / 3


def test_empty_reference_behavior() -> None:
    assert normalize_words("...") == []
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "unexpected") == 1.0


def test_routing_metrics_track_quality_and_cost() -> None:
    router = RuleRouter(RouterConfig(exploration_rate=0.0))
    examples = [
        RoutingExample(
            RouteFeatures(risk_score=0.9, uncertainty=0.1, cross_modal_conflict=0),
            True,
            0.4,
        ),
        RoutingExample(
            RouteFeatures(risk_score=0.1, uncertainty=0.1, cross_modal_conflict=0),
            False,
            0.4,
        ),
        RoutingExample(
            RouteFeatures(risk_score=0.1, uncertainty=0.9, cross_modal_conflict=0),
            True,
            0.4,
        ),
    ]
    metrics = evaluate_router(router, examples)
    assert metrics.recall == 1.0
    assert metrics.precision == 1.0
    assert metrics.escalation_rate == 2 / 3
    assert metrics.gpu_seconds == 0.8


def test_percentile_interpolates_and_validates() -> None:
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    try:
        percentile([], 0.95)
    except ValueError:
        pass
    else:
        raise AssertionError("empty percentile input must fail")
