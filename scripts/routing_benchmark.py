from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from streamsense.evaluation import RoutingExample, evaluate_router
from streamsense.routing import RouteFeatures, RouterConfig, RuleRouter


def load_examples(path: Path) -> list[RoutingExample]:
    examples: list[RoutingExample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        try:
            examples.append(
                RoutingExample(
                    features=RouteFeatures(
                        risk_score=item["risk_score"],
                        uncertainty=item["uncertainty"],
                        cross_modal_conflict=item["cross_modal_conflict"],
                        needs_visual_grounding=item["needs_visual_grounding"],
                    ),
                    oracle_escalate=bool(item["oracle_escalate"]),
                    escalation_gpu_seconds=float(item["gpu_seconds"]),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid fixture at line {line_number}") from error
    return examples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    examples = load_examples(args.fixture)
    systems = {
        "rule_router": RuleRouter(RouterConfig(exploration_rate=0.0)),
        "uncertainty_only": RuleRouter(
            RouterConfig(
                risk_threshold=1.0,
                uncertainty_threshold=0.65,
                conflict_threshold=1.0,
                exploration_rate=0.0,
            )
        ),
    }
    results = {name: asdict(evaluate_router(router, examples)) for name, router in systems.items()}
    oracle_positives = sum(example.oracle_escalate for example in examples)
    full_gpu = sum(example.escalation_gpu_seconds for example in examples)
    results["never_escalate"] = {
        "examples": len(examples),
        "escalations": 0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": oracle_positives,
        "recall": 0.0,
        "precision": 1.0,
        "escalation_rate": 0.0,
        "gpu_seconds": 0.0,
    }
    results["always_escalate"] = {
        "examples": len(examples),
        "escalations": len(examples),
        "true_positives": oracle_positives,
        "false_positives": len(examples) - oracle_positives,
        "false_negatives": 0,
        "recall": 1.0,
        "precision": oracle_positives / len(examples),
        "escalation_rate": 1.0,
        "gpu_seconds": full_gpu,
    }
    payload = {
        "schema_version": 1,
        "fixture": args.fixture.as_posix(),
        "warning": "Curated engineering fixture; not a corpus-level quality benchmark.",
        "systems": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
