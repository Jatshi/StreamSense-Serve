"""Validate v3 serving smoke artifacts and write a machine manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--chat-report", type=Path, required=True)
    parser.add_argument("--api-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.chat_report, args.api_report):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    reports = {
        "chat": json.loads(args.chat_report.read_text(encoding="utf-8")),
        "api": json.loads(args.api_report.read_text(encoding="utf-8")),
    }
    package = "vllm" if args.runtime == "vllm" else "sglang"
    payload = {
        "schema_version": "streamsense.smoke.v3",
        "status": "completed",
        "runtime": args.runtime,
        "runtime_version": importlib.metadata.version(package),
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "artifacts": {
            name: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in (("chat", args.chat_report), ("api", args.api_report))
        },
        "reports": reports,
        "claim_boundary": "single_gpu_smoke_not_dp_or_tp_evidence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
