from __future__ import annotations

import argparse
import json

import uvicorn

from .routing import RouteFeatures, RouterConfig, RuleRouter


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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "serve":
        uvicorn.run("streamsense.api:app", host=args.host, port=args.port)
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
