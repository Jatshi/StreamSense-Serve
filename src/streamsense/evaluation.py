from __future__ import annotations

import re
from dataclasses import dataclass

from .routing import RouteFeatures, RuleRouter


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower())


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = normalize_words(reference)
    hypothesis_words = normalize_words(hypothesis)
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    previous = list(range(len(hypothesis_words) + 1))
    for reference_word in reference_words:
        current = [previous[0] + 1]
        for column, hypothesis_word in enumerate(hypothesis_words, start=1):
            substitution = previous[column - 1] + (reference_word != hypothesis_word)
            insertion = current[column - 1] + 1
            deletion = previous[column] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1] / len(reference_words)


@dataclass(frozen=True)
class RoutingExample:
    features: RouteFeatures
    oracle_escalate: bool
    escalation_gpu_seconds: float


@dataclass(frozen=True)
class RoutingMetrics:
    examples: int
    escalations: int
    true_positives: int
    false_positives: int
    false_negatives: int
    recall: float
    precision: float
    escalation_rate: float
    gpu_seconds: float


def evaluate_router(router: RuleRouter, examples: list[RoutingExample]) -> RoutingMetrics:
    if not examples:
        raise ValueError("routing evaluation requires at least one example")
    escalations = true_positives = false_positives = false_negatives = 0
    gpu_seconds = 0.0
    for example in examples:
        predicted = router.decide(example.features).route == "vlm_escalated"
        escalations += predicted
        gpu_seconds += example.escalation_gpu_seconds if predicted else 0.0
        true_positives += predicted and example.oracle_escalate
        false_positives += predicted and not example.oracle_escalate
        false_negatives += not predicted and example.oracle_escalate
    positives = true_positives + false_negatives
    predicted_positives = true_positives + false_positives
    return RoutingMetrics(
        examples=len(examples),
        escalations=escalations,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        recall=true_positives / positives if positives else 1.0,
        precision=true_positives / predicted_positives if predicted_positives else 1.0,
        escalation_rate=escalations / len(examples),
        gpu_seconds=gpu_seconds,
    )


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
