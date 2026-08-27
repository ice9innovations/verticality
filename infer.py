#!/usr/bin/env python3
"""Predict corrective orientation for one image or a recursive directory."""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from dataset import CORRECTION_DEGREES, IMAGE_EXTENSIONS, ImagePreprocessor, image_paths, rotate_clockwise
from model import load_model
from utils import choose_device

CSV_FIELDS = [
    "path", "predicted_correction", "status", "probability_0", "probability_90",
    "probability_180", "probability_270", "confidence", "error",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="an image or directory (searched recursively)")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--output-dir", type=Path, help="write corrected copies here; sources are never overwritten")
    parser.add_argument("--csv", type=Path, help="CSV path (default: <output-dir>/predictions.csv or predictions.csv)")
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.confidence_threshold <= 1:
        raise SystemExit("--confidence-threshold must be between 0 and 1")
    if args.input.is_file():
        if args.input.suffix.lower() not in IMAGE_EXTENSIONS:
            raise SystemExit(f"Unsupported image extension: {args.input.suffix}")
        paths, base = [args.input], args.input.parent
    else:
        paths, base = image_paths(args.input), args.input
    device = choose_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    preprocess = ImagePreprocessor(size=int(checkpoint.get("image_size", 224)), augment=False)
    rows = []
    for path in paths:
        try:
            # EXIF is deliberately ignored, so malformed EXIF warnings are not useful.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"Corrupt EXIF data.*", category=UserWarning)
                with Image.open(path) as opened:
                    image = opened.convert("RGB")
                    image.load()  # Surface lazy decoder failures here so the run can continue.
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"{path}\nstatus: error\nerror: {error}")
            rows.append({"path": str(path), "predicted_correction": "", "status": "error",
                         "probability_0": "", "probability_90": "", "probability_180": "",
                         "probability_270": "", "confidence": "", "error": error})
            continue
        tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            probabilities = model(tensor).softmax(1)[0].cpu().tolist()
        label = max(range(4), key=probabilities.__getitem__)
        correction, confidence = CORRECTION_DEGREES[label], probabilities[label]
        status = "ok" if confidence >= args.confidence_threshold else "uncertain"
        print(f"{path}\ncorrection: {correction}° ({status})\nconfidence: {confidence:.3f}")
        print("\n".join(f"{degrees}°: {probability:.3f}" for degrees, probability in zip(CORRECTION_DEGREES, probabilities)))
        rows.append({"path": str(path), "predicted_correction": correction, "status": status,
                     "probability_0": probabilities[0], "probability_90": probabilities[1],
                     "probability_180": probabilities[2], "probability_270": probabilities[3],
                     "confidence": confidence, "error": ""})
        if args.output_dir:
            relative = path.relative_to(base)
            destination = args.output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            rotate_clockwise(image, correction).save(destination)
    csv_path = args.csv or ((args.output_dir / "predictions.csv") if args.output_dir else Path("predictions.csv"))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote CSV: {csv_path}")


if __name__ == "__main__":
    main()
