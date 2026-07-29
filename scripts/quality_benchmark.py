from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


def _extract_json(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def score_output(case: dict[str, Any], output: str) -> dict[str, Any]:
    expected = [str(value) for value in case.get("expected_contains", [])]
    forbidden = [str(value) for value in case.get("forbidden_contains", [])]
    contains_pass = all(value in output for value in expected)
    forbidden_pass = all(value not in output for value in forbidden)
    expected_json = case.get("expected_json")
    json_pass: bool | None = None
    parsed_json: dict[str, Any] | None = None
    if isinstance(expected_json, dict):
        parsed_json = _extract_json(output)
        json_pass = parsed_json is not None and all(
            parsed_json.get(key) == value for key, value in expected_json.items()
        )
    passed = contains_pass and forbidden_pass and (json_pass is not False)
    return {
        "passed": passed,
        "contains_pass": contains_pass,
        "forbidden_pass": forbidden_pass,
        "json_pass": json_pass,
        "parsed_json": parsed_json,
    }


def run_benchmark(
    *,
    base_url: str,
    model: str,
    fixture: dict[str, Any],
    max_tokens: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    with httpx.Client(base_url=base_url, timeout=timeout_seconds) as client:
        for case in fixture["cases"]:
            started = time.perf_counter()
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": case["prompt"]}],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            response.raise_for_status()
            payload = response.json()
            output = str(payload["choices"][0]["message"]["content"]).strip()
            score = score_output(case, output)
            results.append(
                {
                    "id": case["id"],
                    "prompt": case["prompt"],
                    "output": output,
                    "elapsed_ms": elapsed_ms,
                    "usage": payload.get("usage"),
                    **score,
                }
            )
    passes = sum(bool(result["passed"]) for result in results)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": {"base_url": base_url, "model": model},
        "cases": len(results),
        "passes": passes,
        "pass_rate": passes / len(results) if results else 0.0,
        "mean_latency_ms": (
            sum(float(result["elapsed_ms"]) for result in results) / len(results)
            if results
            else None
        ),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic contract-quality regression for OpenAI-compatible backends"
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("benchmarks/fixtures/backend_quality_v2.json"),
    )
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_tokens < 1 or args.timeout_seconds <= 0:
        raise ValueError("max_tokens and timeout_seconds must be positive")
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    if fixture.get("schema_version") != 1 or not isinstance(fixture.get("cases"), list):
        raise ValueError("invalid quality fixture")
    report = run_benchmark(
        base_url=args.base_url,
        model=args.model,
        fixture=fixture,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passes"] == report["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
