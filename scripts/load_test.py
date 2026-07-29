from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from streamsense.evaluation import percentile


def parse_sse_data(line: str) -> dict[str, Any] | None:
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("SSE data payload must be a JSON object")
    return parsed


async def _sample_gpu(stop: asyncio.Event, interval_seconds: float) -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    samples: list[dict[str, object]] = []
    try:
        while not stop.is_set():
            result = await asyncio.to_thread(
                subprocess.run,
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if result.returncode != 0:
                return {
                    "available": False,
                    "reason": result.stderr.strip() or "nvidia-smi failed",
                    "samples": [],
                }
            for line in result.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) == 5:
                    samples.append(
                        {
                            "timestamp": parts[0],
                            "gpu_index": int(parts[1]),
                            "memory_used_mib": float(parts[2]),
                            "memory_total_mib": float(parts[3]),
                            "utilization_percent": float(parts[4]),
                        }
                    )
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
            except asyncio.TimeoutError:
                pass
    except (FileNotFoundError, subprocess.SubprocessError, ValueError) as error:
        return {"available": False, "reason": str(error), "samples": []}
    return {"available": True, "samples": samples}


async def _health_request(client: httpx.AsyncClient) -> dict[str, object]:
    started = time.perf_counter()
    response = await client.get("/health")
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    return {
        "status": response.status_code,
        "total_ms": elapsed_ms,
        "ttft_ms": None,
        "tpot_ms": None,
        "output_units": 0,
    }


async def _chat_request(
    client: httpx.AsyncClient,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    expected_response: str | None,
) -> dict[str, object]:
    started = time.perf_counter()
    first_content_at: float | None = None
    output_units = 0
    output_tokens: int | None = None
    content_parts: list[str] = []
    status_code = 0
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        status_code = response.status_code
        response.raise_for_status()
        async for line in response.aiter_lines():
            event = parse_sse_data(line)
            if event is None:
                continue
            try:
                content = event["choices"][0]["delta"].get("content")
            except (IndexError, KeyError, TypeError):
                content = None
            if content:
                now = time.perf_counter()
                first_content_at = first_content_at or now
                output_units += 1
                content_parts.append(str(content))
            usage = event.get("usage")
            if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
                output_tokens = int(usage["completion_tokens"])
    finished = time.perf_counter()
    total_ms = (finished - started) * 1000
    ttft_ms = (first_content_at - started) * 1000 if first_content_at is not None else None
    tpot_ms = None
    if ttft_ms is not None and output_units > 1:
        tpot_ms = (total_ms - ttft_ms) / (output_units - 1)
    output_text = "".join(content_parts).strip()
    normalized_expected = expected_response.strip() if expected_response else None
    return {
        "status": status_code,
        "total_ms": total_ms,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        # A streaming delta is not guaranteed to equal a tokenizer token.
        "output_units": output_units,
        "output_tokens": output_tokens,
        "response_exact": (
            output_text == normalized_expected if normalized_expected is not None else None
        ),
        "response_contains": (
            normalized_expected in output_text if normalized_expected is not None else None
        ),
    }


def _distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


async def run_load(
    base_url: str,
    requests: int,
    concurrency: int,
    *,
    endpoint: str = "health",
    model: str | None = None,
    prompt: str = "Reply with exactly: benchmark-ok",
    expected_response: str | None = "benchmark-ok",
    max_tokens: int = 64,
    api_key: str | None = None,
    sample_gpu: bool = False,
    gpu_sample_interval: float = 0.5,
) -> dict[str, object]:
    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    stop_gpu = asyncio.Event()
    gpu_task = (
        asyncio.create_task(_sample_gpu(stop_gpu, gpu_sample_interval)) if sample_gpu else None
    )

    async with httpx.AsyncClient(base_url=base_url, timeout=180.0, headers=headers) as client:

        async def one_request(index: int) -> None:
            async with semaphore:
                try:
                    if endpoint == "health":
                        result = await _health_request(client)
                    else:
                        if not model:
                            raise ValueError("model is required for chat load tests")
                        result = await _chat_request(
                            client,
                            model=model,
                            prompt=prompt,
                            max_tokens=max_tokens,
                            expected_response=expected_response,
                        )
                    results.append(result)
                except (httpx.HTTPError, json.JSONDecodeError, ValueError) as error:
                    errors.append({"request_index": str(index), "error": type(error).__name__})

        started = time.perf_counter()
        await asyncio.gather(*(one_request(index) for index in range(requests)))
        elapsed = time.perf_counter() - started

    stop_gpu.set()
    gpu = await gpu_task if gpu_task else {"available": False, "reason": "sampling disabled"}
    total_latencies = [float(result["total_ms"]) for result in results]
    ttft = [float(result["ttft_ms"]) for result in results if result["ttft_ms"] is not None]
    tpot = [float(result["tpot_ms"]) for result in results if result["tpot_ms"] is not None]
    output_units = sum(int(result["output_units"]) for result in results)
    reported_output_tokens = [
        int(result["output_tokens"])
        for result in results
        if isinstance(result.get("output_tokens"), int)
    ]
    exact_results = [
        bool(result["response_exact"])
        for result in results
        if result.get("response_exact") is not None
    ]
    contains_results = [
        bool(result["response_contains"])
        for result in results
        if result.get("response_contains") is not None
    ]
    return {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": {"base_url": base_url, "endpoint": endpoint, "model": model},
        "requests": requests,
        "concurrency": concurrency,
        "successes": len(results),
        "errors": len(errors),
        "error_rate": len(errors) / requests,
        "error_types": errors[:100],
        "elapsed_seconds": elapsed,
        "throughput_requests_per_second": len(results) / elapsed if elapsed else 0.0,
        "stream_output_units_per_second": output_units / elapsed if elapsed else 0.0,
        "reported_output_tokens": (
            sum(reported_output_tokens)
            if len(reported_output_tokens) == len(results) and results
            else None
        ),
        "reported_output_tokens_per_second": (
            sum(reported_output_tokens) / elapsed
            if len(reported_output_tokens) == len(results) and results and elapsed
            else None
        ),
        "quality": {
            "expected_response": expected_response,
            "exact_match_rate": (
                sum(exact_results) / len(exact_results) if exact_results else None
            ),
            "contains_rate": (
                sum(contains_results) / len(contains_results) if contains_results else None
            ),
        },
        "metric_semantics": {
            "ttft": "time to first non-empty streamed content delta",
            "tpot": "time after first content divided by remaining content deltas",
            "output_units": "streamed content deltas; not tokenizer-exact tokens",
            "reported_output_tokens": (
                "sum of server-reported completion_tokens; null unless every successful "
                "response reports usage"
            ),
        },
        "latency_ms": _distribution(total_latencies),
        "ttft_ms": _distribution(ttft),
        "tpot_ms": _distribution(tpot),
        "gpu": gpu,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", choices=["health", "chat"], default="health")
    parser.add_argument("--model")
    parser.add_argument("--prompt", default="Reply with exactly: benchmark-ok")
    parser.add_argument("--expected-response", default="benchmark-ok")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--api-key-env")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=25)
    parser.add_argument("--sample-gpu", action="store_true")
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.requests < 1 or not 1 <= args.concurrency <= args.requests:
        raise ValueError("require requests >= concurrency >= 1")
    if args.endpoint == "chat" and not args.model:
        raise ValueError("--model is required for --endpoint chat")
    if args.max_tokens < 1 or args.gpu_sample_interval <= 0:
        raise ValueError("max_tokens and gpu_sample_interval must be positive")
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    result = asyncio.run(
        run_load(
            args.base_url,
            args.requests,
            args.concurrency,
            endpoint=args.endpoint,
            model=args.model,
            prompt=args.prompt,
            expected_response=args.expected_response,
            max_tokens=args.max_tokens,
            api_key=api_key,
            sample_gpu=args.sample_gpu,
            gpu_sample_interval=args.gpu_sample_interval,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
