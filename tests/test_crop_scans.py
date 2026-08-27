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


def test_ignores_scanner_speckles_across_white_page():
    page = Image.new("RGB", (1000, 800), "white")
    draw = ImageDraw.Draw(page)
    draw.rectangle((100, 80, 450, 500), fill=(80, 110, 130))
    for x in range(10, 1000, 23):
        for y in range((x * 7) % 29, 800, 97):
            draw.point((x, y), fill=(180, 180, 180))
    result = detect_photo(page)
    assert result["status"] == "crop"
    assert result["box"][2] < 500
