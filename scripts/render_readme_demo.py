"""Render the README GIF from the committed serving benchmark matrix."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - documentation utility
    raise SystemExit("Install Pillow first: python -m pip install pillow") from exc


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets" / "streamsense_v2_demo.gif"
SIZE = (960, 540)
BG, PANEL, WHITE, MUTED = "#080C14", "#111827", "#F8FAFC", "#94A3B8"
BLUE, CYAN, ORANGE, RED = "#60A5FA", "#22D3EE", "#FB923C", "#FB7185"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    size: int,
    color: str = WHITE,
    bold: bool = False,
) -> None:
    draw.text(xy, value, font=font(size, bold), fill=color)


def panel(draw: ImageDraw.ImageDraw, x: int, title: str, value: str, accent: str) -> None:
    draw.rounded_rectangle(
        (x, 220, x + 270, 335), radius=18, fill=PANEL, outline="#26344D", width=2
    )
    label(draw, (x + 20, 239), title, 14, MUTED, True)
    label(draw, (x + 20, 274), value, 23, accent, True)


def matrix_facts() -> dict[str, tuple[float, float]]:
    rows = json.loads((ROOT / "docs" / "benchmark_matrix_4090.json").read_text(encoding="utf-8"))[
        "rows"
    ]
    result: dict[str, tuple[float, float]] = {}
    for row in rows:
        if row["concurrency"] == 32:
            result[row["profile"]] = (row["throughput_requests_per_second"], row["ttft_p50_ms"])
    return result


def render() -> list[Image.Image]:
    facts = matrix_facts()
    scenes = [
        (
            "INGEST EVENTS",
            ("AUDIO", "timestamped ASR", CYAN),
            ("VIDEO", "scene evidence", BLUE),
            ("STORE", "typed + replayable", ORANGE),
        ),
        (
            "ADAPTIVE ROUTE",
            ("LOW RISK", "local path", BLUE),
            ("VISUAL / CONFLICT", "VLM escalate", ORANGE),
            ("FIXTURE", "10/10 recall", CYAN),
        ),
        (
            "vLLM BF16",
            ("RPS @ C32", f"{facts['vllm-qwen25-vl-3b'][0]:.1f}", CYAN),
            ("TTFT P50", f"{facts['vllm-qwen25-vl-3b'][1]:.1f} ms", BLUE),
            ("QUALITY", "8 / 12", ORANGE),
        ),
        (
            "vLLM DYNAMIC FP8",
            ("RPS @ C32", f"{facts['vllm-qwen25-vl-3b-fp8'][0]:.1f}", CYAN),
            ("TTFT P50", f"{facts['vllm-qwen25-vl-3b-fp8'][1]:.1f} ms", BLUE),
            ("QUALITY", "7 / 12", RED),
        ),
        (
            "SGLANG BF16",
            ("RPS @ C32", f"{facts['sglang-qwen25-vl-3b'][0]:.1f}", CYAN),
            ("TTFT P50", f"{facts['sglang-qwen25-vl-3b'][1]:.1f} ms", BLUE),
            ("LIFECYCLE", "clean shutdown", ORANGE),
        ),
        (
            "VERIFY + LEARN",
            ("REQUESTS", "960 / 960", CYAN),
            ("CITATIONS", "strict IDs", BLUE),
            ("FEEDBACK", "consent + license", ORANGE),
        ),
    ]
    frames: list[Image.Image] = []
    for index, (heading, *cards) in enumerate(scenes, start=1):
        image = Image.new("RGB", SIZE, BG)
        draw = ImageDraw.Draw(image)
        for x in range(0, 960, 48):
            draw.line((x, 0, x, 540), fill="#111827", width=1)
        label(draw, (42, 28), "STREAMSENSE SERVE", 24, WHITE, True)
        label(draw, (42, 61), "EVIDENCE-FIRST INFERENCE", 13, BLUE, True)
        label(draw, (800, 36), "v2.0.0", 16, CYAN, True)
        label(draw, (42, 108), f"0{index}  {heading}", 18, MUTED, True)
        label(draw, (42, 151), "Route  /  stream  /  verify  /  observe  /  learn", 23, WHITE, True)
        for i, (title, value, accent) in enumerate(cards):
            panel(draw, 42 + i * 292, title, value, accent)
        draw.rounded_rectangle(
            (42, 374, 918, 458), radius=18, fill="#101D35", outline="#2453A6", width=2
        )
        label(draw, (66, 394), "Qwen2.5-VL-3B  |  vLLM + SGLang  |  RTX 4090", 19, BLUE, True)
        label(
            draw, (66, 427), "TTFT  |  TPOT  |  throughput  |  error rate  |  peak VRAM", 14, MUTED
        )
        draw.rounded_rectangle((42, 500, 918, 505), radius=3, fill="#202C42")
        draw.rounded_rectangle((42, 500, 42 + int(876 * index / 6), 505), radius=3, fill=BLUE)
        frames.append(image)
    return frames


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frames = render()
    frames[0].save(
        OUTPUT, save_all=True, append_images=frames[1:], duration=1150, loop=0, optimize=True
    )
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
