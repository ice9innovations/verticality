from PIL import Image, ImageDraw

from crop_scans import detect_photo


def test_detects_photo_inside_white_scan_page():
    page = Image.new("RGB", (1000, 800), "white")
    ImageDraw.Draw(page).rectangle((180, 120, 820, 680), fill=(70, 100, 130))
    result = detect_photo(page)
    assert result["status"] == "crop"
    left, top, right, bottom = result["box"]
    assert abs(left - 180) < 3
    assert abs(top - 120) < 3
    assert abs(right - 821) < 3
    assert abs(bottom - 681) < 3


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


def test_detects_photo_touching_top_and_left_page_edges():
    page = Image.new("RGB", (1000, 800), (248, 247, 244))
    ImageDraw.Draw(page).rectangle((0, 0, 430, 360), fill=(110, 70, 55))
    result = detect_photo(page)
    assert result["status"] == "crop"
    left, top, right, bottom = result["box"]
    assert left == 0
    assert top == 0
    assert abs(right - 431) < 10
    assert abs(bottom - 361) < 10
