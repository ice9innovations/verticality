import argparse
import csv
import json
import sqlite3
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from PIL import Image

import review_app


def make_workspace(tmp_path):
    photos = tmp_path / "photos"
    album = photos / "Family" / "1990s"
    album.mkdir(parents=True)
    source = album / "photo.png"
    Image.new("RGB", (40, 20), "red").save(source)
    enhanced = album / "photo_a.png"
    Image.new("RGB", (40, 20), "orange").save(enhanced)
    predictions = tmp_path / "predictions.csv"
    with predictions.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "path", "predicted_correction", "status", "probability_0",
            "probability_90", "probability_180", "probability_270",
            "confidence", "error",
        ))
        writer.writeheader()
        writer.writerow({"path": source, "predicted_correction": 90, "status": "ok",
                         "confidence": 0.9})
        writer.writerow({"path": enhanced, "predicted_correction": 180, "status": "ok",
                         "confidence": 0.8})
    workspace = tmp_path / "private-review"
    args = argparse.Namespace(images=photos, predictions=predictions,
                              workspace=workspace, thumbnail_size=32, workers=1)
    review_app.prepare(args)
    return args, workspace


def test_prepare_parser_accepts_arbitrary_worker_count():
    args = review_app.parser().parse_args([
        "prepare", "--images", "photos", "--predictions", "predictions.csv", "--workers", "13",
    ])
    assert args.workers == 13


def test_prepare_preserves_album_rotation_and_manual_review(tmp_path):
    args, workspace = make_workspace(tmp_path)
    with sqlite3.connect(workspace / "review.sqlite3") as db:
        row = db.execute("SELECT album, thumbnail, predicted_correction, selected_correction FROM images WHERE relative_path='Family/1990s/photo.png'").fetchone()
        assert row[0] == "Family/1990s"
        assert row[2:] == (90, 90)
        thumbnail = workspace / "thumbnails" / row[1]
        assert Image.open(thumbnail).height > Image.open(thumbnail).width
        db.execute("UPDATE images SET selected_correction=180, reviewed=1")
        db.commit()
    # A repeated preparation must preserve manual state.
    review_app.prepare(args)
    with sqlite3.connect(workspace / "review.sqlite3") as db:
        assert db.execute("SELECT DISTINCT selected_correction, reviewed FROM images").fetchall() == [(180, 1)]


def test_review_api_persists_each_action(tmp_path):
    _, workspace = make_workspace(tmp_path)
    handler = type("TestReviewHandler", (review_app.ReviewHandler,), {
        "workspace": workspace,
        "static_dir": review_app.Path(__file__).parents[1] / "review_static",
    })
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        albums = json.load(urllib.request.urlopen(base + "/api/albums"))
        assert albums["totals"]["total"] == 1
        images = json.load(urllib.request.urlopen(base + "/api/images?album=Family%2F1990s"))
        assert len(images["images"]) == 1
        assert images["images"][0]["variant_count"] == 2
        image_id = images["images"][0]["id"]
        body = json.dumps({"group_key": images["images"][0]["group_key"],
                           "selected_correction": 270, "reviewed": True}).encode()
        request = urllib.request.Request(base + "/api/review", data=body,
                                         headers={"Content-Type": "application/json"}, method="POST")
        updated = json.load(urllib.request.urlopen(request))
        assert updated["selected_correction"] == 270
        assert updated["reviewed"] == 1
        assert updated["variant_count"] == 2
        with sqlite3.connect(workspace / "review.sqlite3") as db:
            assert db.execute("SELECT DISTINCT selected_correction, reviewed FROM images").fetchall() == [(270, 1)]
    finally:
        server.shutdown()
        thread.join()
