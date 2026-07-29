from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .backends import BackendProfile, BackendProfiles


def build_command(profile: BackendProfile) -> list[str]:
    engine = profile.engine
    if profile.kind == "openai" or engine is None:
        raise ValueError(f"profile {profile.name!r} is client-only and cannot be launched")
    if profile.kind == "vllm":
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            engine.model_path,
            "--served-model-name",
            engine.served_model_name,
            "--host",
            "127.0.0.1",
            "--port",
            str(_port(profile.base_url)),
            "--dtype",
            engine.dtype,
            "--tensor-parallel-size",
            str(engine.tensor_parallel_size),
            "--max-model-len",
            str(engine.max_model_len),
            "--gpu-memory-utilization",
            str(engine.gpu_memory_utilization),
        ]
        if engine.revision:
            command.extend(["--revision", engine.revision])
        if engine.quantization not in {"none", "auto"}:
            command.extend(["--quantization", engine.quantization])
        if engine.trust_remote_code:
            command.append("--trust-remote-code")
        command.extend(engine.extra_args)
        return command
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        engine.model_path,
        "--served-model-name",
        engine.served_model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(_port(profile.base_url)),
        "--dtype",
        engine.dtype,
        "--tp-size",
        str(engine.tensor_parallel_size),
        "--context-length",
        str(engine.max_model_len),
        "--mem-fraction-static",
        str(engine.gpu_memory_utilization),
    ]
    if engine.revision:
        command.extend(["--revision", engine.revision])
    if engine.quantization not in {"none", "auto"}:
        command.extend(["--quantization", engine.quantization])
    if engine.trust_remote_code:
        command.append("--trust-remote-code")
    command.extend(engine.extra_args)
    return command


def _port(base_url: str) -> int:
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    if parsed.port:
        return parsed.port
    return 443 if parsed.scheme == "https" else 80


def main() -> int:
    parser = argparse.ArgumentParser(description="Render or execute one safe backend profile")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("STREAMSENSE_BACKEND_CONFIG", "configs/backends.json")),
    )
    parser.add_argument("--profile")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--print-base-url", action="store_true")
    parser.add_argument("--print-model", action="store_true")
    args = parser.parse_args()
    profiles = BackendProfiles.load(args.config)
    profile = profiles.get(args.profile or os.environ.get("STREAMSENSE_BACKEND_PROFILE"))
    if args.print_base_url:
        print(profile.base_url)
        return 0
    if args.print_model:
        print(profile.model)
        return 0
    command = build_command(profile)
    if not args.execute:
        print(json.dumps(command, ensure_ascii=False))
        return 0
    # Replace the launcher process so its PID is the actual model server PID.
    # Benchmark supervisors can then terminate and reap the backend reliably
    # without leaving an orphaned CUDA process behind.
    # A venv Python can be invoked by absolute path without activating its
    # shell environment. Put its bin directory first so backend JIT compilers
    # can discover console tools installed in the same environment (for
    # example SGLang's pinned `ninja` executable).
    # Do not call Path.resolve(): venv/bin/python is commonly a symlink to the
    # base interpreter, and resolving it would prepend the base bin directory
    # instead of the venv directory that actually contains console tools.
    executable_dir = str(Path(command[0]).absolute().parent)
    inherited_path = os.environ.get("PATH", "")
    os.environ["PATH"] = (
        executable_dir if not inherited_path else executable_dir + os.pathsep + inherited_path
    )
    os.execv(command[0], command)


if __name__ == "__main__":
    raise SystemExit(main())
