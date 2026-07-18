from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from streamsense.evaluation import percentile


async def run_load(base_url: str, requests: int, concurrency: int) -> dict[str, object]:
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    statuses: list[int] = []

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:

        async def one_request() -> None:
            async with semaphore:
                started = time.perf_counter()
                response = await client.get("/health")
                latencies.append((time.perf_counter() - started) * 1000)
                statuses.append(response.status_code)

        started = time.perf_counter()
        await asyncio.gather(*(one_request() for _ in range(requests)))
        elapsed = time.perf_counter() - started

    successes = sum(status == 200 for status in statuses)
    return {
        "requests": requests,
        "concurrency": concurrency,
        "successes": successes,
        "errors": requests - successes,
        "elapsed_seconds": elapsed,
        "throughput_requests_per_second": requests / elapsed,
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.requests < 1 or not 1 <= args.concurrency <= args.requests:
        raise ValueError("require requests >= concurrency >= 1")
    result = asyncio.run(run_load(args.base_url, args.requests, args.concurrency))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
