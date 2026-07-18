from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a deterministic visual-change fixture")
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seconds-per-scene", type=int, default=2)
    return parser


def _scene(*, alert: bool) -> np.ndarray:
    frame = np.full((360, 640, 3), (24, 28, 32), dtype=np.uint8)
    if alert:
        frame[:] = (235, 235, 235)
        cv2.rectangle(frame, (255, 75), (385, 305), (45, 45, 190), thickness=10)
        cv2.line(frame, (270, 95), (370, 285), (45, 45, 190), thickness=8)
        cv2.circle(frame, (355, 190), 8, (20, 20, 20), thickness=-1)
        label, color = "ALERT: DOOR OPEN", (30, 30, 190)
    else:
        cv2.circle(frame, (320, 175), 55, (45, 185, 75), thickness=-1)
        label, color = "STATUS: NORMAL", (80, 220, 110)
    cv2.putText(
        frame,
        label,
        (145, 335),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame


def main() -> int:
    args = build_parser().parse_args()
    if args.fps <= 0 or args.seconds_per_scene <= 0:
        raise ValueError("fps and seconds-per-scene must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (640, 360),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not initialize the mp4v encoder")
    try:
        for alert in (False, True):
            frame = _scene(alert=alert)
            for _ in range(args.fps * args.seconds_per_scene):
                writer.write(frame)
    finally:
        writer.release()
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
