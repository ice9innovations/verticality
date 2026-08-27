"""COCO-image datasets and orientation-safe preprocessing."""

from __future__ import annotations

import random
import csv
import hashlib
from pathlib import Path
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CORRECTION_DEGREES = (0, 90, 180, 270)


def image_paths(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise RuntimeError(f"No supported images found under {root}")
    return paths


def rotate_clockwise(image: Image.Image, degrees: int) -> Image.Image:
    """Rotate by an exact clockwise multiple of 90 degrees."""
    methods = {
        0: None,
        90: Image.Transpose.ROTATE_270,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_90,
    }
    degrees %= 360
    if degrees not in methods:
        raise ValueError("degrees must be a multiple of 90")
    return image.copy() if degrees == 0 else image.transpose(methods[degrees])


class ImagePreprocessor:
    """Letterbox without distortion, then normalize for MobileNetV3.

    Letterboxing happens *before* the synthetic rotation. Consequently the square
    canvas, including its padding, rotates with the photograph and cannot encode a
    shortcut based on which class received padding on which sides.
    """

    def __init__(self, size: int = 224, augment: bool = False):
        self.size = size
        self.augment = augment

    def prepare_square(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")  # Intentionally does not apply EXIF orientation.
        if self.augment:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.9, 1.1))
        width, height = image.size
        scale = self.size / max(width, height)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = image.resize(new_size, Image.Resampling.BILINEAR)
        # A constant ImageNet-mean-ish fill is rotationally invariant.
        canvas = Image.new("RGB", (self.size, self.size), (124, 116, 104))
        canvas.paste(image, ((self.size - new_size[0]) // 2, (self.size - new_size[1]) // 2))
        return canvas

    @staticmethod
    def to_tensor(image: Image.Image) -> torch.Tensor:
        tensor = TF.to_tensor(image)
        return TF.normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __call__(self, image: Image.Image, observed_clockwise: int = 0) -> torch.Tensor:
        return self.to_tensor(rotate_clockwise(self.prepare_square(image), observed_clockwise))


class OrientationDataset(Dataset[tuple[torch.Tensor, int]]):
    """Synthetic orientation classification over normally upright source images.

    Labels are corrective clockwise rotations. If an image is synthetically
    rotated clockwise by r quarter-turns, its label is (-r) mod 4.
    Validation exposes every source image at all four rotations exactly once.
    """

    def __init__(self, root: str | Path, training: bool, size: int = 224):
        self.paths = image_paths(root)
        self.training = training
        self.preprocess = ImagePreprocessor(size=size, augment=training)

    def __len__(self) -> int:
        return len(self.paths) if self.training else 4 * len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if self.training:
            path = self.paths[index]
            observed_quarters = random.randrange(4)
        else:
            path = self.paths[index // 4]
            observed_quarters = index % 4
        with Image.open(path) as image:
            tensor = self.preprocess(image, observed_clockwise=observed_quarters * 90)
        corrective_label = (-observed_quarters) % 4
        return tensor, corrective_label


def review_group_key(relative_path: str) -> str:
    path = Path(relative_path)
    stem = path.stem[:-2] if path.stem.lower().endswith("_a") else path.stem
    return (path.parent / f"{stem}{path.suffix.lower()}").as_posix()


def review_thumbnail_path(workspace: str | Path, relative_path: str) -> Path:
    digest = hashlib.sha1(Path(relative_path).as_posix().encode()).hexdigest()
    return Path(workspace) / "thumbnails" / digest[:2] / f"{digest}.jpg"


class ReviewOrientationDataset(Dataset[tuple[torch.Tensor, int]]):
    """Synthetic rotations of human-approved upright review thumbnails."""

    def __init__(self, workspace: str | Path, labels: str | Path, training: bool,
                 size: int = 224, val_fraction: float = 0.2, seed: int = 42):
        if not 0 < val_fraction < 1:
            raise ValueError("val_fraction must be between 0 and 1")
        rows = []
        with Path(labels).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("unknown", "0") == "1" or row.get("error", ""):
                    continue
                thumbnail = review_thumbnail_path(workspace, row["relative_path"])
                if not thumbnail.is_file():
                    continue
                group = review_group_key(row["relative_path"])
                digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
                in_validation = int.from_bytes(digest[:8], "big") / 2**64 < val_fraction
                if in_validation != (not training):
                    continue
                rows.append((thumbnail, int(row["predicted_correction"]),
                             int(row["selected_correction"]), group))
        if not rows:
            split = "training" if training else "validation"
            raise RuntimeError(f"No usable {split} review thumbnails found")
        self.rows = rows
        self.training = training
        self.preprocess = ImagePreprocessor(size=size, augment=training)

    def __len__(self) -> int:
        # Four draws per source per training epoch gives the small review set
        # enough updates; validation deterministically covers all rotations.
        return 4 * len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if self.training:
            thumbnail, predicted, selected, _ = self.rows[index % len(self.rows)]
            observed_quarters = random.randrange(4)
        else:
            thumbnail, predicted, selected, _ = self.rows[index // 4]
            observed_quarters = index % 4
        with Image.open(thumbnail) as image:
            # Stored thumbnails show the model proposal. Rotate by the human
            # correction delta first to reconstruct the approved upright view.
            upright = rotate_clockwise(image.convert("RGB"), selected - predicted)
            tensor = self.preprocess(upright, observed_clockwise=observed_quarters * 90)
        return tensor, (-observed_quarters) % 4
