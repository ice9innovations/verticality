#!/usr/bin/env python3
"""Prepare and review a private, album-organized photo orientation dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import sqlite3
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

DEFAULT_WORKSPACE = Path("private-review")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY,
    source_path TEXT NOT NULL UNIQUE,
    relative_path TEXT NOT NULL UNIQUE,
    group_key TEXT NOT NULL,
    album TEXT NOT NULL,
    thumbnail TEXT,
    predicted_correction INTEGER,
    selected_correction INTEGER,
    confidence REAL,
    prediction_status TEXT NOT NULL,
    reviewed INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS images_album_idx ON images(album, relative_path);
"""


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def connect(workspace: Path) -> sqlite3.Connection:
    workspace.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(workspace / "review.sqlite3")
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(images)")}
    if "group_key" not in columns:
        connection.execute("ALTER TABLE images ADD COLUMN group_key TEXT NOT NULL DEFAULT ''")
        for row in connection.execute("SELECT id, relative_path FROM images"):
            connection.execute("UPDATE images SET group_key=? WHERE id=?",
                               (logical_group_key(Path(row["relative_path"])), row["id"]))
        connection.commit()
    connection.execute("CREATE INDEX IF NOT EXISTS images_group_idx ON images(group_key)")
    return connection


def load_predictions(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {str(Path(row["path"]).resolve()): row for row in csv.DictReader(handle)}


def image_paths(root: Path) -> list[Path]:
    paths = sorted(path for path in root.rglob("*")
                   if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise RuntimeError(f"No supported images found under {root}")
    return paths


def rotate_clockwise(image: Image.Image, degrees: int) -> Image.Image:
    methods = {0: None, 90: Image.Transpose.ROTATE_270,
               180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_90}
    degrees %= 360
    if degrees not in methods:
        raise ValueError("degrees must be a multiple of 90")
    return image.copy() if degrees == 0 else image.transpose(methods[degrees])


def thumbnail_name(relative_path: Path) -> str:
    digest = hashlib.sha1(relative_path.as_posix().encode()).hexdigest()
    return f"{digest[:2]}/{digest}.jpg"


def logical_group_key(relative_path: Path) -> str:
    """Collapse scanner-generated *_a enhancements into their base photo."""
    stem = relative_path.stem
    if stem.lower().endswith("_a"):
        stem = stem[:-2]
    return (relative_path.parent / f"{stem}{relative_path.suffix.lower()}").as_posix()


def generate_thumbnail(task: tuple[str, str, int, int]) -> str:
    """Decode and resize one image in a worker process; return an error message or ''."""
    source, destination, correction, thumbnail_size = task
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Corrupt EXIF data.*", category=UserWarning)
            with Image.open(source) as opened:
                image = opened.convert("RGB")
                image.load()
        image.thumbnail((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)
        image = rotate_clockwise(image, correction)
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination_path, "JPEG", quality=85)
        return ""
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return f"{type(exc).__name__}: {exc}"


def prepare(args) -> None:
    root = args.images.resolve()
    predictions = load_predictions(args.predictions)
    connection = connect(args.workspace)
    thumbnails = args.workspace / "thumbnails"
    paths = image_paths(root)
    records = []
    tasks = []
    for path in paths:
        absolute = path.resolve()
        relative = absolute.relative_to(root)
        group_key = logical_group_key(relative)
        prediction = predictions.get(str(absolute), {})
        predicted = prediction.get("predicted_correction", "")
        correction = int(predicted) if predicted not in (None, "") else 0
        status = prediction.get("status", "missing")
        confidence = prediction.get("confidence", "")
        confidence_value = float(confidence) if confidence not in (None, "") else None
        thumb_rel = thumbnail_name(relative)
        thumb_path = thumbnails / thumb_rel
        error = prediction.get("error", "")
        existing = connection.execute(
            "SELECT predicted_correction, thumbnail FROM images WHERE source_path=?", (str(absolute),)
        ).fetchone()
        regenerate = (not thumb_path.exists() or existing is None
                      or existing["predicted_correction"] != correction or not existing["thumbnail"])
        records.append((absolute, relative, group_key, correction, status, confidence_value,
                        thumb_rel, error, regenerate))
        if regenerate:
            tasks.append((str(absolute), str(thumb_path), correction, args.thumbnail_size))

    workers = getattr(args, "workers", 1)
    if workers == 1:
        task_errors = map(generate_thumbnail, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        task_errors = executor.map(generate_thumbnail, tasks)
    generated_errors = iter(tqdm(task_errors, total=len(tasks), desc="thumbnails"))

    prepared = errors = 0
    for index, record in enumerate(records, 1):
        absolute, relative, group_key, correction, status, confidence_value, thumb_rel, error, regenerate = record
        thumbnail_error = next(generated_errors) if regenerate else ""
        if thumbnail_error:
            thumb_rel = None
            error = thumbnail_error
            status = "error"
            errors += 1
        else:
            prepared += 1
        album = relative.parent.as_posix()
        if album == ".":
            album = "(root)"
        connection.execute("""
            INSERT INTO images (
                source_path, relative_path, group_key, album, thumbnail, predicted_correction,
                selected_correction, confidence, prediction_status, reviewed, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            ON CONFLICT(source_path) DO UPDATE SET
                relative_path=excluded.relative_path, group_key=excluded.group_key, album=excluded.album,
                thumbnail=excluded.thumbnail, predicted_correction=excluded.predicted_correction,
                confidence=excluded.confidence, prediction_status=excluded.prediction_status,
                error=excluded.error
        """, (str(absolute), relative.as_posix(), group_key, album, thumb_rel, correction, correction,
              confidence_value, status, error))
        if index % 100 == 0:
            connection.commit()
    if executor is not None:
        executor.shutdown()
    connection.commit()
    print(f"indexed={len(paths):,} thumbnails={prepared:,} errors={errors:,} workspace={args.workspace}")


def rows_as_dicts(rows) -> list[dict]:
    return [dict(row) for row in rows]


class ReviewHandler(BaseHTTPRequestHandler):
    workspace: Path
    static_dir: Path

    def connection_db(self) -> sqlite3.Connection:
        return connect(self.workspace)

    def json_response(self, value, status=HTTPStatus.OK) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def file_response(self, path: Path, content_type: str | None = None) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        payload = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store" if path.suffix == ".html" else "private, max-age=86400")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.file_response(self.static_dir / "index.html", "text/html; charset=utf-8")
            return
        if parsed.path == "/api/albums":
            with self.connection_db() as db:
                albums = rows_as_dicts(db.execute("""
                    WITH photo_groups AS (
                        SELECT album, group_key, MIN(reviewed) AS reviewed,
                               MAX(prediction_status = 'uncertain') AS uncertain,
                               MAX(prediction_status = 'error') AS errors
                        FROM images GROUP BY album, group_key
                    )
                    SELECT album, COUNT(*) AS total, SUM(reviewed) AS reviewed,
                           SUM(uncertain) AS uncertain, SUM(errors) AS errors
                    FROM photo_groups GROUP BY album ORDER BY album COLLATE NOCASE
                """))
                totals = dict(db.execute("""
                    WITH photo_groups AS (
                        SELECT group_key, MIN(reviewed) AS reviewed,
                               MAX(prediction_status = 'uncertain') AS uncertain,
                               MAX(prediction_status = 'error') AS errors
                        FROM images GROUP BY group_key
                    )
                    SELECT COUNT(*) AS total, SUM(reviewed) AS reviewed,
                           SUM(uncertain) AS uncertain, SUM(errors) AS errors FROM photo_groups
                """).fetchone())
            self.json_response({"albums": albums, "totals": totals})
            return
        if parsed.path == "/api/images":
            query = parse_qs(parsed.query)
            album = query.get("album", [""])[0]
            state = query.get("state", ["all"])[0]
            page = max(1, int(query.get("page", ["1"])[0]))
            page_size = min(200, max(1, int(query.get("page_size", ["60"])[0])))
            clauses, having, values = [], [], []
            if album:
                clauses.append("album = ?")
                values.append(album)
            if state == "unreviewed":
                having.append("MIN(reviewed) = 0")
            elif state == "uncertain":
                having.append("MAX(prediction_status IN ('uncertain', 'error', 'missing')) = 1")
            where = " WHERE " + " AND ".join(clauses) if clauses else ""
            having_sql = " HAVING " + " AND ".join(having) if having else ""
            with self.connection_db() as db:
                groups_sql = (f"SELECT group_key, album, MIN(reviewed) AS reviewed FROM images{where} "
                              f"GROUP BY group_key, album{having_sql}")
                total = db.execute(f"SELECT COUNT(*) FROM ({groups_sql})", values).fetchone()[0]
                groups = db.execute(
                    f"{groups_sql} ORDER BY group_key COLLATE NOCASE LIMIT ? OFFSET ?",
                    (*values, page_size, (page - 1) * page_size),
                ).fetchall()
                images = []
                for group in groups:
                    variants = db.execute("""
                        SELECT * FROM images WHERE group_key=?
                        ORDER BY CASE WHEN lower(relative_path) GLOB '*_a.*' THEN 1 ELSE 0 END,
                                 relative_path COLLATE NOCASE
                    """, (group["group_key"],)).fetchall()
                    representative = dict(variants[0])
                    representative["reviewed"] = group["reviewed"]
                    representative["variant_count"] = len(variants)
                    representative["variant_paths"] = [row["relative_path"] for row in variants]
                    statuses = {row["prediction_status"] for row in variants}
                    for candidate in ("error", "missing", "uncertain"):
                        if candidate in statuses:
                            representative["prediction_status"] = candidate
                            break
                    images.append(representative)
            self.json_response({"images": images, "total": total,
                                "page": page, "page_size": page_size})
            return
        if parsed.path.startswith("/thumbnails/"):
            relative = unquote(parsed.path.removeprefix("/thumbnails/"))
            candidate = (self.workspace / "thumbnails" / relative).resolve()
            root = (self.workspace / "thumbnails").resolve()
            if root not in candidate.parents:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            self.file_response(candidate, "image/jpeg")
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/review":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            correction = int(payload["selected_correction"]) % 360
            if correction not in (0, 90, 180, 270):
                raise ValueError("correction must be a quarter turn")
            reviewed = 1 if payload.get("reviewed", True) else 0
            with self.connection_db() as db:
                if "group_key" in payload:
                    group_key = str(payload["group_key"])
                else:
                    image_id = int(payload["id"])
                    found = db.execute("SELECT group_key FROM images WHERE id=?", (image_id,)).fetchone()
                    if found is None:
                        raise KeyError(image_id)
                    group_key = found["group_key"]
                cursor = db.execute(
                    "UPDATE images SET selected_correction=?, reviewed=? WHERE group_key=?",
                    (correction, reviewed, group_key),
                )
                if cursor.rowcount == 0:
                    raise KeyError(group_key)
                row = db.execute("SELECT * FROM images WHERE group_key=? ORDER BY relative_path LIMIT 1",
                                 (group_key,)).fetchone()
                db.commit()
            response = dict(row)
            response["variant_count"] = cursor.rowcount
            self.json_response(response)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self.json_response({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args) -> None:
        if getattr(self.server, "verbose", False):
            super().log_message(format, *args)


def serve(args) -> None:
    handler = type("WorkspaceReviewHandler", (ReviewHandler,), {
        "workspace": args.workspace.resolve(),
        "static_dir": (Path(__file__).parent / "review_static").resolve(),
    })
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.verbose = args.verbose
    print(f"review app: http://{args.host}:{args.port}")
    print(f"private workspace: {args.workspace.resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


def export(args) -> None:
    connection = connect(args.workspace)
    query = "SELECT * FROM images" + (" WHERE reviewed = 1" if args.reviewed_only else "") + " ORDER BY relative_path"
    rows = rows_as_dicts(connection.execute(query))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source_path", "relative_path", "album", "predicted_correction",
              "selected_correction", "confidence", "prediction_status", "reviewed", "error"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    print(f"exported {len(rows):,} records to {args.output}")
    if args.corrected_dir:
        for row in tqdm(rows, desc="corrected originals"):
            if row["error"]:
                continue
            destination = args.corrected_dir / PurePosixPath(row["relative_path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                with Image.open(row["source_path"]) as opened:
                    image = opened.convert("RGB")
                    image.load()
                rotate_clockwise(image, row["selected_correction"]).save(destination)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                print(f"skip {row['source_path']}: {exc}", file=sys.stderr)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="index photos and create proposed-upright thumbnails")
    prepare_parser.add_argument("--images", type=Path, required=True)
    prepare_parser.add_argument("--predictions", type=Path, required=True)
    prepare_parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    prepare_parser.add_argument("--thumbnail-size", type=int, default=320)
    prepare_parser.add_argument("--workers", type=positive_int, default=1, metavar="N",
                                help="use N parallel image-decoding workers")
    prepare_parser.set_defaults(function=prepare)
    serve_parser = commands.add_parser("serve", help="serve the private review UI")
    serve_parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8080)
    serve_parser.add_argument("--verbose", action="store_true")
    serve_parser.set_defaults(function=serve)
    export_parser = commands.add_parser("export", help="export reviewed corrections")
    export_parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    export_parser.add_argument("--output", type=Path, default=DEFAULT_WORKSPACE / "reviewed-orientations.csv")
    export_parser.add_argument("--reviewed-only", action="store_true")
    export_parser.add_argument("--corrected-dir", type=Path)
    export_parser.set_defaults(function=export)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)
