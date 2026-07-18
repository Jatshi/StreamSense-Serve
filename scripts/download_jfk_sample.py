from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/ggerganov/whisper.cpp/master/samples/jfk.wav"
SHA256 = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download the pinned ASR engineering sample")
    parser.add_argument("output", type=Path, nargs="?", default=Path("data/public/jfk.wav"))
    return parser


def main() -> int:
    output = build_parser().parse_args().output
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(URL, headers={"User-Agent": "StreamSense-Serve/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != SHA256:
        raise RuntimeError(f"sample hash mismatch: expected {SHA256}, received {actual}")
    output.write_bytes(payload)
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
