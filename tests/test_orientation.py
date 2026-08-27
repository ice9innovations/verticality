from PIL import Image

import csv

from dataset import (OrientationDataset, ReviewOrientationDataset,
                     review_thumbnail_path, rotate_clockwise)


def test_clockwise_rotation_moves_top_left_to_top_right():
    image = Image.new("RGB", (2, 2), "black")
    image.putpixel((0, 0), (255, 0, 0))
    rotated = rotate_clockwise(image, 90)
    assert rotated.getpixel((1, 0)) == (255, 0, 0)


def test_validation_labels_are_balanced_and_correct(tmp_path):
    Image.new("RGB", (4, 2), "white").save(tmp_path / "one.jpg")
    dataset = OrientationDataset(tmp_path, training=False, size=8)
    labels = [dataset[index][1] for index in range(4)]
    assert labels == [0, 3, 2, 1]


def test_review_dataset_normalizes_proposed_thumbnail_and_splits_groups(tmp_path):
    workspace = tmp_path / "review"
    labels = workspace / "labels.csv"
    workspace.mkdir()
    rows = []
    for index in range(20):
        for suffix in ("", "_a"):
            relative = f"album/photo_{index:02d}{suffix}.tif"
            thumbnail = review_thumbnail_path(workspace, relative)
            thumbnail.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 4), "white").save(thumbnail)
            rows.append({"relative_path": relative, "predicted_correction": "90",
                         "selected_correction": "180", "unknown": "0", "error": ""})
    with labels.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    train = ReviewOrientationDataset(workspace, labels, True, size=8, seed=7)
    val = ReviewOrientationDataset(workspace, labels, False, size=8, seed=7)
    train_groups = {row[3] for row in train.rows}
    val_groups = {row[3] for row in val.rows}
    assert train_groups.isdisjoint(val_groups)
    assert train_groups | val_groups == {f"album/photo_{index:02d}.tif" for index in range(20)}
    assert len(train) == 4 * len(train.rows)
    assert [val[index][1] for index in range(4)] == [0, 3, 2, 1]
