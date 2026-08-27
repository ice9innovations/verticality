#!/usr/bin/env python3
"""Conservatively detect and non-destructively crop photos on white scan pages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, UnidentifiedImageError
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
REPORT_FIELDS = ("source_path", "album", "relative_path", "status", "confidence",
                 "left", "top", "right", "bottom", "width", "height", "reason", "preview", "error")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def longest_run(active: np.ndarray) -> tuple[int, int] | None:
    padded = np.pad(active.astype(np.int8), 1)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    if not len(starts):
        return None
    index = int(np.argmax(ends - starts))
    return int(starts[index]), int(ends[index])


def detect_photo(image: Image.Image, analysis_size: int = 1200,
                 color_distance: int = 24, min_occupancy: float = 0.025) -> dict:
    """Return a conservative axis-aligned crop proposal in original pixels."""
    original_width, original_height = image.size
    preview = image.convert("RGB")
    preview.thumbnail((analysis_size, analysis_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(preview, dtype=np.int16)
    height, width = pixels.shape[:2]
    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    background = np.median(border, axis=0)
    border_distance = np.max(np.abs(border - background), axis=1)
    white_background = float(background.mean()) >= 200 and float((border_distance < color_distance).mean()) >= 0.65
    if not white_background:
        return {"status": "unchanged", "confidence": 1.0,
                "reason": "no uniform light page border",
                "box": (0, 0, original_width, original_height)}

    distance = np.max(np.abs(pixels - background), axis=2).clip(0, 255).astype(np.uint8)
    # Full-page scans often contain isolated dust, paper texture, or JPEG noise.
    # Median filtering removes those dots before row/column projections without
    # erasing a real photograph boundary.
    smoothed = np.asarray(Image.fromarray(distance).filter(ImageFilter.MedianFilter(5)))
    foreground = smoothed >= color_distance
    row_run = longest_run(foreground.mean(axis=1) >= min_occupancy)
    col_run = longest_run(foreground.mean(axis=0) >= min_occupancy)
    if row_run is None or col_run is None:
        return {"status": "uncertain", "confidence": 0.0,
                "reason": "no single rectangular content region",
                "box": (0, 0, original_width, original_height)}
    top, bottom = row_run
    left, right = col_run
    margin = max(2, round(min(width, height) * 0.004))
    left, top = max(0, left - margin), max(0, top - margin)
    right, bottom = min(width, right + margin), min(height, bottom + margin)
    retained = (right - left) * (bottom - top) / (width * height)
    margins = (left / width, top / height, (width - right) / width, (height - bottom) / height)
    if retained < 0.08 or right - left < 40 or bottom - top < 40:
        status, confidence, reason = "uncertain", 0.0, "detected region is implausibly small"
    elif max(margins) < 0.02 or 1 - retained < 0.03:
        status, confidence, reason = "unchanged", 1.0, "content already fills the scan"
        left, top, right, bottom = 0, 0, width, height
    else:
        outside = np.ones((height, width), dtype=bool)
        outside[top:bottom, left:right] = False
        outside_white = float((~foreground[outside]).mean()) if outside.any() else 0.0
        confidence = max(0.0, min(1.0, (outside_white - 0.75) / 0.25))
        status = "crop" if confidence >= 0.70 else "uncertain"
        reason = "uniform whitespace outside photo" if status == "crop" else "outside region is not uniformly blank"
    scale_x, scale_y = original_width / width, original_height / height
    box = (round(left * scale_x), round(top * scale_y),
           round(right * scale_x), round(bottom * scale_y))
    return {"status": status, "confidence": confidence, "reason": reason, "box": box}


def preview_path(workspace: Path, source: Path) -> Path:
    digest = hashlib.sha1(str(source).encode()).hexdigest()
    return workspace / "previews" / digest[:2] / f"{digest}.jpg"


def analyze_one(task) -> dict:
    source_text, album, relative_text, workspace_text, analysis_size = task
    source, workspace = Path(source_text), Path(workspace_text)
    row = {"source_path": str(source), "album": album, "relative_path": relative_text,
           "status": "error", "confidence": 0.0, "left": "", "top": "", "right": "",
           "bottom": "", "width": "", "height": "", "reason": "", "preview": "", "error": ""}
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
            warnings.filterwarnings("ignore", message=r"Corrupt EXIF data.*", category=UserWarning)
            with Image.open(source) as opened:
                image = opened.convert("RGB")
                image.load()
        result = detect_photo(image, analysis_size=analysis_size)
        box = result["box"]
        row.update({"status": result["status"], "confidence": f'{result["confidence"]:.4f}',
                    "left": box[0], "top": box[1], "right": box[2], "bottom": box[3],
                    "width": image.width, "height": image.height, "reason": result["reason"]})
        shown = image.copy()
        shown.thumbnail((700, 700), Image.Resampling.LANCZOS)
        if result["status"] in ("crop", "uncertain"):
            scale_x, scale_y = shown.width / image.width, shown.height / image.height
            draw = ImageDraw.Draw(shown)
            draw.rectangle(tuple(round(value * scale) for value, scale in zip(
                box, (scale_x, scale_y, scale_x, scale_y))), outline=(255, 45, 45), width=5)
        destination = preview_path(workspace, source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shown.save(destination, "JPEG", quality=82)
        row["preview"] = destination.relative_to(workspace).as_posix()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def source_tasks(inputs: list[Path], workspace: Path, analysis_size: int) -> list[tuple]:
    tasks = []
    for root in inputs:
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        for source in sorted(root.rglob("*")):
            if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
                tasks.append((str(source.resolve()), root.name, source.relative_to(root).as_posix(),
                              str(workspace.resolve()), analysis_size))
    if not tasks:
        raise RuntimeError("No supported scans found")
    return tasks


def analyze(args) -> None:
    args.workspace.mkdir(parents=True, exist_ok=True)
    tasks = source_tasks(args.input, args.workspace, args.analysis_size)
    if args.workers == 1:
        results = map(analyze_one, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        results = executor.map(analyze_one, tasks)
    rows = list(tqdm(results, total=len(tasks), desc="analyze scans"))
    if args.workers != 1:
        executor.shutdown()
    report = args.workspace / "crop-report.csv"
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    cards = []
    for row in sorted(rows, key=lambda item: ({"uncertain": 0, "crop": 1, "error": 2,
                                               "unchanged": 3}.get(item["status"], 4),
                                              item["album"], item["relative_path"])):
        image = (f'<img loading="lazy" src="{html.escape(row["preview"])}">'
                 if row["preview"] else "")
        cards.append(f'<article class="{html.escape(row["status"])}">{image}'
                     f'<b>{html.escape(row["status"])}</b> '
                     f'<span>{html.escape(row["confidence"])}</span>'
                     f'<div>{html.escape(row["reason"])}</div>'
                     f'<div>{html.escape(row["album"] + "/" + row["relative_path"])}</div></article>')
    gallery = """<!doctype html><meta charset="utf-8"><title>Scan crop review</title>
<style>body{margin:20px;background:#151719;color:#eee;font:14px system-ui}h1{font-size:22px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}
article{border:2px solid #444;background:#222;padding:8px;overflow:hidden}article.uncertain{border-color:#e0aa45}
article.crop{border-color:#63b887}article.error{border-color:#d86868}img{width:100%;height:260px;object-fit:contain;background:#111}
b{display:inline-block;margin:6px 8px 2px 0}span{color:#aaa}article div{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}</style>
<h1>Scan crop review</h1><p>Red rectangles show proposed boundaries. Uncertain cases appear first.</p><div class="grid">"""
    (args.workspace / "index.html").write_text(gallery + "".join(cards) + "</div>", encoding="utf-8")
    counts = {status: sum(row["status"] == status for row in rows)
              for status in ("crop", "unchanged", "uncertain", "error")}
    print(" ".join(f"{key}={value:,}" for key, value in counts.items()))
    print(f"report: {report}\ngallery: {args.workspace / 'index.html'}")


def save_cropped(source: Path, destination: Path, box: tuple[int, int, int, int]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
        with Image.open(source) as opened:
            cropped = opened.crop(box)
            save_options = {}
            if source.suffix.lower() in (".tif", ".tiff"):
                save_options["compression"] = "tiff_lzw"
            if "dpi" in opened.info:
                save_options["dpi"] = opened.info["dpi"]
            cropped.save(destination, **save_options)


def apply_report(args) -> None:
    with args.report.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    written = copied = skipped = 0
    for row in tqdm(rows, desc="write output"):
        source = Path(row["source_path"])
        destination = args.output / row["album"] / Path(row["relative_path"])
        if row["status"] == "crop":
            box = tuple(int(row[key]) for key in ("left", "top", "right", "bottom"))
            save_cropped(source, destination, box)
            written += 1
        elif row["status"] == "unchanged" and args.copy_unchanged:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied += 1
        else:
            skipped += 1
    print(f"cropped={written:,} copied_unchanged={copied:,} skipped={skipped:,} output={args.output}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("analyze", help="detect crops and create private previews")
    scan.add_argument("--input", type=Path, action="append", required=True,
                      help="source directory; repeat for multiple albums")
    scan.add_argument("--workspace", type=Path, default=Path("private-crops"))
    scan.add_argument("--workers", type=positive_int, default=1, metavar="N")
    scan.add_argument("--analysis-size", type=positive_int, default=1200)
    scan.set_defaults(function=analyze)
    apply = commands.add_parser("apply", help="write approved crop proposals to a separate tree")
    apply.add_argument("--report", type=Path, default=Path("private-crops/crop-report.csv"))
    apply.add_argument("--output", type=Path, required=True)
    apply.add_argument("--copy-unchanged", action="store_true")
    apply.set_defaults(function=apply_report)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)
