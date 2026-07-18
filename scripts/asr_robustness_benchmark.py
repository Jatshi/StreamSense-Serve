from __future__ import annotations

import argparse
import json
import math
import time
import wave
from pathlib import Path

import numpy as np

from streamsense.analyzers import FasterWhisperAnalyzer
from streamsense.evaluation import word_error_rate


def read_pcm(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("benchmark requires mono 16-bit PCM WAV")
        sample_rate = source.getframerate()
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    return samples.astype(np.float32) / 32768.0, sample_rate


def write_pcm(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def add_white_noise(samples: np.ndarray, *, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signal_power = float(np.mean(np.square(samples), dtype=np.float64))
    if signal_power <= 0:
        raise ValueError("cannot add calibrated noise to a silent signal")
    noise_power = signal_power / math.pow(10.0, snr_db / 10.0)
    noise = rng.normal(0.0, math.sqrt(noise_power), size=samples.shape)
    return np.asarray(samples + noise, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("media", type=Path)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--snr", type=float, nargs="+", default=[20, 10, 0, -5])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model", default="small")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples, sample_rate = read_pcm(args.media)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    analyzer = FasterWhisperAnalyzer(
        model_name=args.model,
        device="cuda",
        compute_type="float16",
        cache_dir=args.cache_dir,
    )
    conditions: list[tuple[str, Path, float | None]] = [("clean", args.media, None)]
    for snr_db in args.snr:
        path = args.work_dir / f"snr_{snr_db:g}db.wav"
        write_pcm(path, add_white_noise(samples, snr_db=snr_db, seed=args.seed), sample_rate)
        conditions.append((f"snr_{snr_db:g}db", path, snr_db))

    results: list[dict[str, object]] = []
    for name, path, snr_db in conditions:
        started = time.perf_counter()
        observations = analyzer.analyze(path, stream_id=name)
        elapsed = time.perf_counter() - started
        transcript = " ".join(item.evidence.description or "" for item in observations)
        results.append(
            {
                "condition": name,
                "snr_db": snr_db,
                "elapsed_seconds": elapsed,
                "transcript": transcript,
                "word_error_rate": word_error_rate(args.reference, transcript),
                "segments": len(observations),
            }
        )
    payload = {
        "schema_version": 1,
        "configuration": {
            "model": args.model,
            "seed": args.seed,
            "reference": args.reference,
            "sample_rate": sample_rate,
            "samples": int(samples.size),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
