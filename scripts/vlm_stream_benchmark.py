from __future__ import annotations

import argparse
import base64
import json
import statistics
import time
from pathlib import Path

import httpx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure streaming TTFT for one visual fixture")
    parser.add_argument("image", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return parser


def _image_url(path: Path) -> str:
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".ppm": "image/x-portable-pixmap",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "application/octet-stream")
    return f"data:{media_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _run(client: httpx.Client, *, base_url: str, model: str, image_url: str) -> dict:
    started = time.perf_counter()
    first_content_at: float | None = None
    content: list[str] = []
    usage: dict | None = None
    with client.stream(
        "POST",
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json={
            "model": model,
            "temperature": 0,
            "max_tokens": 200,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Describe only visible evidence. Return JSON with keys summary, label, "
                        "confidence, risk_score. Do not identify people or infer intent."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this status frame conservatively."},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
        },
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            choices = chunk.get("choices") or []
            token = choices[0].get("delta", {}).get("content") if choices else None
            if token:
                if first_content_at is None:
                    first_content_at = time.perf_counter()
                content.append(token)
    ended = time.perf_counter()
    if first_content_at is None:
        raise RuntimeError("stream completed without a content token")
    return {
        "ttft_seconds": first_content_at - started,
        "total_seconds": ended - started,
        "usage": usage,
        "content": "".join(content),
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.runs < 1:
        raise ValueError("runs must be positive")
    image = args.image.expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    image_url = _image_url(image)
    with httpx.Client(timeout=180.0) as client:
        runs = [
            _run(client, base_url=args.base_url, model=args.model, image_url=image_url)
            for _ in range(args.runs)
        ]
    result = {
        "model": args.model,
        "image": image.name,
        "runs": runs,
        "summary": {
            "median_ttft_seconds": statistics.median(run["ttft_seconds"] for run in runs),
            "median_total_seconds": statistics.median(run["total_seconds"] for run in runs),
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
