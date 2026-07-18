from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import uvicorn

from .analyzers import FasterWhisperAnalyzer, FrameChangeAnalyzer
from .media import AudioEnergyAnalyzer, MediaPipeline
from .routing import RouteFeatures, RouterConfig, RuleRouter
from .store import EventStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="streamsense")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    route = subparsers.add_parser("route", help="Evaluate one routing decision")
    route.add_argument("--risk", type=float, required=True)
    route.add_argument("--uncertainty", type=float, required=True)
    route.add_argument("--conflict", type=float, required=True)
    route.add_argument("--visual-grounding", action="store_true")

    analyze = subparsers.add_parser("analyze", help="Analyze one local media file")
    analyze.add_argument("media", type=Path)
    analyze.add_argument("--stream-id", required=True)
    analyze.add_argument("--database", type=Path, default=Path("data/events.db"))
    analyze.add_argument("--evidence-dir", type=Path, default=Path("data/evidence"))
    analyze.add_argument("--asr-model")
    analyze.add_argument("--device", default="cuda")
    analyze.add_argument("--compute-type", default="float16")
    analyze.add_argument("--model-cache", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "serve":
        uvicorn.run("streamsense.api:app", host=args.host, port=args.port)
        return 0
    if args.command == "analyze":
        analyzers = [
            AudioEnergyAnalyzer(),
            FrameChangeAnalyzer(evidence_dir=args.evidence_dir),
        ]
        if args.asr_model:
            analyzers.append(
                FasterWhisperAnalyzer(
                    model_name=args.asr_model,
                    device=args.device,
                    compute_type=args.compute_type,
                    cache_dir=args.model_cache,
                )
            )
        result = MediaPipeline(
            analyzers=analyzers,
            router=RuleRouter(RouterConfig()),
            store=EventStore(args.database),
        ).analyze(args.media, stream_id=args.stream_id)
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return 0
    decision = RuleRouter(RouterConfig()).decide(
        RouteFeatures(
            risk_score=args.risk,
            uncertainty=args.uncertainty,
            cross_modal_conflict=args.conflict,
            needs_visual_grounding=args.visual_grounding,
        )
    )
    print(json.dumps({"route": decision.route, "reasons": decision.reasons}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
