# Experimental image-orientation classifier

A deliberately small PyTorch experiment: fine-tune ImageNet-pretrained MobileNetV3-Small on clean COCO photos with synthetic quarter-turn rotations, then test whether it generalizes to unrelated photographs.

## Label convention and preprocessing

Every class is the **clockwise correction to apply to the observed input**: class 0 = 0°, class 1 = 90°, class 2 = 180°, class 3 = 270°. If an upright source is synthetically rotated clockwise by `r` quarter-turns, the target is `(-r) mod 4`. For example, a 90°-clockwise synthetic rotation gets class 3 (correct clockwise by 270°).

Images retain their aspect ratio and are centered on a square constant-color canvas. Crucially, this happens before synthetic rotation, so the complete square canvas rotates with the photo and padding placement cannot reveal the label. No EXIF auto-orientation is performed: pixels are classified exactly as stored.

Training samples one random orientation per source image per epoch. Validation evaluates all four rotations of every `val2017` image exactly once, so it is balanced and reproducible. Official COCO splits keep every variant of a source photograph in one split.

## Setup and COCO

Python 3.10+ is recommended. Install a PyTorch build suitable for your CUDA version if needed, then install the remaining requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download COCO 2017 images (about 19 GB compressed/uncompressed combined):

```bash
chmod +x download_coco.sh
./download_coco.sh data/coco
```

Equivalent manual commands are in `download_coco.sh`. Annotations are not used. The expected directories are `data/coco/train2017` and `data/coco/val2017`.

## Train and validate

```bash
python train.py \
  --train-dir data/coco/train2017 \
  --val-dir data/coco/val2017 \
  --epochs 10 --batch-size 128 --workers 8 \
  --checkpoint checkpoints/best.pt
```

The best validation-accuracy checkpoint is saved. Each epoch reports train/validation loss and accuracy, four per-class accuracies, a confusion matrix, and average confidence on correct versus incorrect predictions. `--device auto` selects CUDA, then Apple MPS, then CPU; it can be overridden with `--device cuda:0`, for example.

Evaluate a saved checkpoint without training:

```bash
python train.py --eval-only --val-dir data/coco/val2017 \
  --checkpoint checkpoints/best.pt --batch-size 128 --workers 8
```

## Arbitrary photographs

One image (also writes a CSV):

```bash
python infer.py photos/example.jpg --checkpoint checkpoints/best.pt \
  --confidence-threshold 0.75 --csv results/example.csv
```

A recursive directory, CSV, and corrected copies:

```bash
python infer.py photos --checkpoint checkpoints/best.pt \
  --confidence-threshold 0.75 --output-dir results/corrected \
  --csv results/predictions.csv
```

Low-confidence results are marked `uncertain`. Corrected files are written under a separate output tree and sources are never modified. Predictions still appear in the CSV so threshold behavior can be analyzed later.

Corrupt images and formats unsupported by the installed Pillow build are reported with `status=error` and an explanation in the CSV's `error` column. They are skipped without stopping a recursive directory run. Because EXIF orientation is intentionally ignored, malformed EXIF warnings are suppressed. The CSV header is created before inference and every result is flushed immediately, preserving completed work if the process is interrupted later.

### CPU inference and large source images

Inference selects CUDA, then Apple MPS, then CPU by default. To force CPU inference, even on a machine with a supported GPU:

```bash
python infer.py photos \
  --checkpoint checkpoints/best.pt \
  --device cpu \
  --confidence-threshold 0.75 \
  --csv results/predictions.csv
```

CPU inference is slower but otherwise produces the same output format. Image decoding and resizing already occur on the CPU regardless of the model device.

Large JPEG and TIFF sources are converted to RGB, resized with their aspect ratio preserved so the longest side is 224 pixels, and letterboxed to 224×224 before classification. Model memory therefore does not scale with source dimensions. Pillow must still decode the complete source image before resizing it, so a very large TIFF can temporarily require substantial CPU memory and take noticeably longer to load. Images are processed sequentially during inference rather than retained as a batch.

For a large collection, a CSV-only first pass avoids duplicating every photograph:

```bash
python infer.py photos \
  --checkpoint checkpoints/best.pt \
  --device cpu \
  --confidence-threshold 0.75 \
  --csv results/predictions.csv
```

Add `--output-dir results/corrected` only when corrected copies are wanted. Keep the output directory outside the input tree, and ensure enough storage is available. Source photographs are never overwritten.

## Experimental caveats

COCO is only assumed to be normally oriented; occasional mislabeled source orientation becomes label noise. A confidence threshold is not a calibration guarantee. Evaluate on a separately curated wild-photo set (including all four rotations per source) before interpreting generalization, and inspect confusion between 0°/180° and 90°/270° rather than relying only on overall accuracy.
