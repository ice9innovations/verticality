from PIL import Image

from dataset import OrientationDataset, rotate_clockwise


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

