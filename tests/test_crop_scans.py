from PIL import Image, ImageDraw

from crop_scans import detect_photo


def test_detects_photo_inside_white_scan_page():
    page = Image.new("RGB", (1000, 800), "white")
    ImageDraw.Draw(page).rectangle((180, 120, 820, 680), fill=(70, 100, 130))
    result = detect_photo(page)
    assert result["status"] == "crop"
    left, top, right, bottom = result["box"]
    assert abs(left - 180) < 10
    assert abs(top - 120) < 10
    assert abs(right - 821) < 10
    assert abs(bottom - 681) < 10


def test_leaves_edge_to_edge_photo_unchanged():
    photo = Image.new("RGB", (1000, 800), (60, 90, 120))
    assert detect_photo(photo)["status"] == "unchanged"
