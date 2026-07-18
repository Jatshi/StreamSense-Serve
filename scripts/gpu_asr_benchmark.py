from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
import wave
from pathlib import Path

from streamsense.analyzers import FasterWhisperAnalyzer


def gpu_snapshot() -> dict[str, str]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        line = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"available": "false"}
    name, driver, used, total, utilization = [item.strip() for item in line.split(",")]
    return {
        "available": "true",
        "name": name,
        "driver": driver,
        "memory_used_mib": used,
        "memory_total_mib": total,
        "utilization_percent": utilization,
    }


def wav_duration_seconds(path: Path) -> float | None:
    if path.suffix.lower() != ".wav":
        return None
    with wave.open(str(path), "rb") as source:
        return source.getnframes() / source.getframerate()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("media", type=Path)
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be at least 1")
    analyzer = FasterWhisperAnalyzer(
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        cache_dir=args.cache_dir,
    )
    before = gpu_snapshot()
    runs: list[dict[str, object]] = []
    for index in range(args.repeats):
        started = time.perf_counter()
        observations = analyzer.analyze(args.media.resolve(), stream_id="benchmark")
        elapsed = time.perf_counter() - started
        runs.append(
            {
                "index": index,
                "elapsed_seconds": elapsed,
                "segments": len(observations),
                "transcript": " ".join(item.evidence.description or "" for item in observations),
                "mean_confidence": statistics.fmean([item.evidence.score for item in observations])
                if observations
                else 0.0,
            }
        )
    duration = wav_duration_seconds(args.media)
    warm_runs = runs[1:] if len(runs) > 1 else runs
    median_elapsed = statistics.median(float(run["elapsed_seconds"]) for run in warm_runs)
    result = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "gpu_before": before,
            "gpu_after": gpu_snapshot(),
        },
        "configuration": {
            "model": args.model,
            "device": args.device,
            "compute_type": args.compute_type,
            "media": args.media.name,
            "media_duration_seconds": duration,
            "repeats": args.repeats,
        },
        "summary": {
            "median_warm_elapsed_seconds": median_elapsed,
            "real_time_factor": median_elapsed / duration if duration else None,
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
