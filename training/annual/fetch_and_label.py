"""
fetch_and_label.py  (training/annual/)
---------------------------------------
Path A — Step 1. Run this every year after the race.

Fetches race images for a previous year from the API, runs the pretrained
YOLOv8n person detector on each image to generate bounding-box labels,
and writes them into dataset/ in YOLO format.

No manual annotation needed — the person detector does the labelling.

Usage:
    python training/annual/fetch_and_label.py --years 23
    python training/annual/fetch_and_label.py --years 22 23  # multiple years

Requires detection/.env to be filled in (IMAGE_BASE_URL, GET_ALBUMS_URL,
GET_IMAGE_LIST_URL, TAGGER_ID).
"""

import argparse
import os
import re
import sys
import random
import cv2
import numpy as np
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'detection'))
import config  # type: ignore[import-untyped]

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DATASET_DIR    = Path("dataset")
TRAIN_RATIO    = 0.85   # fraction of images that go to train split
PERSON_CLASS   = 0      # COCO class index for "person"
CONF_THRESH    = 0.40   # minimum YOLO confidence to keep a person box
WORKERS        = 16     # concurrent download threads
SKIP_EXISTING  = True   # skip images already in dataset/
LABELLER_MODEL = "yolov8n.pt"  # downloads automatically on first run (~6 MB)

# All URLs are read from detection/.env — never constructed in code
TAGGER_ID          = os.getenv("TAGGER_ID", "")
IMAGE_BASE_URL     = os.getenv("IMAGE_BASE_URL", "").rstrip("/")
GET_ALBUMS_URL     = os.getenv("GET_ALBUMS_URL", "")
GET_IMAGE_LIST_URL = os.getenv("GET_IMAGE_LIST_URL", "")

# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------
_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
_session.mount("https://", _adapter)


def fetch_albums(year: str) -> list:
    """
    Fetch album list from the API and patch the year segment in each album_url.
    The API returns album_url values like "images/23/albums/raus" — we replace
    the year so we can reuse the same API call for any year.
    """
    resp = _session.get(GET_ALBUMS_URL, timeout=config.IMAGE_FETCH_TIMEOUT)
    resp.raise_for_status()
    albums = resp.json()
    for album in albums:
        album["album_url"] = re.sub(r"/(\d{2})/", f"/{year}/", album["album_url"])
    return albums


def fetch_image_list(album_url: str) -> list:
    """
    POST to get-image-list.php for the given album_url.
    Returns list of filenames, e.g. ["10000.jpg", "10001.jpg", ...]
    """
    resp = _session.post(
        GET_IMAGE_LIST_URL,
        json={"album": album_url, "tagger": TAGGER_ID},
        timeout=config.IMAGE_FETCH_TIMEOUT,
    )
    resp.raise_for_status()
    # Response is {"2": "10000.jpg", "3": "10001.jpg", ...} — we only need the filenames
    return list(resp.json().values())


def fetch_image(url: str):
    try:
        resp = _session.get(url, timeout=config.IMAGE_FETCH_TIMEOUT)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        arr = np.frombuffer(resp.content, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def yolo_bbox(box_xyxy, img_w, img_h):
    """Convert absolute xyxy → YOLO normalised xywh."""
    x1, y1, x2, y2 = box_xyxy
    xc = ((x1 + x2) / 2) / img_w
    yc = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return xc, yc, w, h


def label_and_save(filename: str, album: dict, year: str, model: YOLO, split: str):
    """Fetch one image, detect persons, write label file + save image."""
    album_slug = album["album_url"].split("/")[-1]   # e.g. "raus"
    stem       = Path(filename).stem                   # e.g. "10000"
    img_name   = f"y{year}_{album_slug}_{stem}"
    img_dest   = DATASET_DIR / "images" / split / f"{img_name}.jpg"
    label_dest = DATASET_DIR / "labels" / split / f"{img_name}.txt"

    if SKIP_EXISTING and img_dest.exists():
        return "skipped"

    url = f"{IMAGE_BASE_URL}/{album['album_url']}/{filename}"
    img = fetch_image(url)
    if img is None:
        return "404"

    h, w = img.shape[:2]
    results = model.predict(img, imgsz=640, verbose=False, classes=[PERSON_CLASS])

    lines = []
    for box in results[0].boxes:
        if float(box.conf[0]) < CONF_THRESH:
            continue
        xc, yc, bw, bh = yolo_bbox(box.xyxy[0].tolist(), w, h)
        lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    if not lines:
        return "no_person"

    img_dest.parent.mkdir(parents=True, exist_ok=True)
    label_dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_dest), img)
    label_dest.write_text("\n".join(lines))
    return "ok"


def process_year(year: str, model: YOLO):
    print(f"\n{'='*50}")
    print(f"  Year: 20{year}")
    print(f"{'='*50}")

    print("  Fetching album list from API...")
    albums = fetch_albums(year)
    print(f"  Found {len(albums)} albums")

    stats: dict = {}

    for album in albums:
        album_slug = album["album_url"].split("/")[-1]
        print(f"\n  Fetching image list: {album['name']} ({album_slug})...")
        try:
            filenames = fetch_image_list(album["album_url"])
        except Exception as e:
            print(f"  Warning: failed to fetch image list — {e}")
            continue

        if not filenames:
            print("  No images found, skipping")
            continue

        # Deterministic train/val split
        shuffled = filenames[:]
        random.seed(42)
        random.shuffle(shuffled)
        split_at  = int(len(shuffled) * TRAIN_RATIO)
        train_set = set(shuffled[:split_at])
        tasks     = [(f, "train" if f in train_set else "val") for f in filenames]

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {
                pool.submit(label_and_save, fn, album, year, model, split): fn
                for fn, split in tasks
            }
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"  {album['name']}", leave=False):
                result = fut.result()
                stats[result] = stats.get(result, 0) + 1

    print(f"\n  Results: {stats.get('ok', 0)} labelled  |  "
          f"{stats.get('skipped', 0)} skipped  |  "
          f"{stats.get('404', 0)} not found  |  "
          f"{stats.get('no_person', 0)} no person detected")
    return stats.get("ok", 0)


def main():
    missing = [k for k, v in {
        "TAGGER_ID": TAGGER_ID,
        "IMAGE_BASE_URL": IMAGE_BASE_URL,
        "GET_ALBUMS_URL": GET_ALBUMS_URL,
        "GET_IMAGE_LIST_URL": GET_IMAGE_LIST_URL,
    }.items() if not v]
    if missing:
        print(f"Error: missing env vars in detection/.env: {', '.join(missing)}")
        return

    parser = argparse.ArgumentParser(description="Auto-label previous years' race images")
    parser.add_argument(
        "--years", nargs="+", default=["23"],
        help="Two-digit year suffixes to process, e.g. --years 22 23"
    )
    parser.add_argument(
        "--model", default=LABELLER_MODEL,
        help="COCO pretrained YOLO model to use for person detection"
    )
    args = parser.parse_args()

    print(f"Loading labeller model: {args.model}")
    model = YOLO(args.model)

    total_new = 0
    for year in args.years:
        total_new += process_year(year, model)

    print(f"\nDone. {total_new:,} new labelled images added to dataset/")
    print("Next step: run  python training/resume_training.py  to fine-tune.")


if __name__ == "__main__":
    main()
