"""
evaluate.py
-----------
Runs the detector against race images and compares results to ground truth.

Supports two ground truth formats:
  - .txt  — (image_id, 'bibnr', albumnr) tuples  [2024 format]
  - .json — phpMyAdmin export with {id, bibnr, albumnr} rows  [2023 format]

Outputs:
  - evaluation_results.csv  — one row per image: ground truth, detected, verdict
  - failures/               — saved images for every miss and false positive

Usage:
    # 2024 ground truth (default)
    python training/evaluate.py

    # 2023 ground truth
    python training/evaluate.py --ground-truth models/marathon_raw_data_2023.json

    # Limit to one album for a quick sanity check
    python training/evaluate.py --album finish --limit 200

GCP:
    python training/evaluate.py --workers 32
"""

import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import requests
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'detection'))
import config  # type: ignore[import-untyped]
from detector import BibDetector  # type: ignore[import-untyped]
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'detection', '.env'))

# All URLs are read from detection/.env — same as fetch_and_label.py
TAGGER_ID          = os.getenv("TAGGER_ID", "")
IMAGE_BASE_URL     = os.getenv("IMAGE_BASE_URL", "").rstrip("/")
GET_ALBUMS_URL     = os.getenv("GET_ALBUMS_URL", "")
GET_IMAGE_LIST_URL = os.getenv("GET_IMAGE_LIST_URL", "")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_GROUND_TRUTH = "models/marathon_raw_data.txt"
OUTPUT_CSV            = "evaluation_results.csv"
FAILURES_DIR          = Path("failures")

# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------
_session = requests.Session()
_session.headers["User-Agent"] = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "Chrome/120.0.0.0 Safari/537.36"
)
_adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
_session.mount("https://", _adapter)


def fetch_albums(year: str) -> list:
    resp = _session.get(GET_ALBUMS_URL, timeout=config.IMAGE_FETCH_TIMEOUT)
    resp.raise_for_status()
    albums = resp.json()
    for album in albums:
        album["album_url"] = re.sub(r"/(\d{2})/", f"/{year}/", album["album_url"])
    return albums


def fetch_image_list(album_url: str) -> list:
    resp = _session.post(
        GET_IMAGE_LIST_URL,
        json={"album": album_url, "tagger": TAGGER_ID},
        timeout=config.IMAGE_FETCH_TIMEOUT,
    )
    resp.raise_for_status()
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


# ---------------------------------------------------------------------------
# Ground truth parsing
# ---------------------------------------------------------------------------
def load_ground_truth(path: str) -> dict:
    """
    Parse ground truth into {image_id: set(bib_strings)}.

    Supports two formats:
      .txt  — (image_id, 'bibnr', albumnr) tuples
      .json — phpMyAdmin export: [{type, data: [{id, bibnr, albumnr}, ...]}, ...]
    """
    gt = {}
    if path.endswith(".json"):
        with open(path) as f:
            raw = json.load(f)
        rows = next(
            (e["data"] for e in raw if e.get("type") == "table" and "data" in e), []
        )
        for row in rows:
            image_id = int(row["id"])
            bib_str  = row.get("bibnr", "").strip()
            bibs     = set(b.strip() for b in bib_str.split(",") if b.strip()) if bib_str else set()
            gt[image_id] = bibs
    else:
        with open(path) as f:
            content = f.read()
        for match in re.finditer(r'\((\d+),\s*\'(.*?)\',\s*(\d+)\)', content):
            image_id = int(match.group(1))
            bib_str  = match.group(2).strip()
            bibs     = set(b.strip() for b in bib_str.split(',') if b.strip()) if bib_str else set()
            gt[image_id] = bibs
    return gt


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------
def verdict(gt_bibs: set, detected_bibs: set) -> str:
    if not gt_bibs and not detected_bibs:
        return "true_negative"     # no bibs, none detected — correct
    if not gt_bibs and detected_bibs:
        return "false_positive"    # detected bibs that aren't there
    if gt_bibs and not detected_bibs:
        return "false_negative"    # missed all bibs
    if gt_bibs == detected_bibs:
        return "exact_match"       # perfect
    if gt_bibs & detected_bibs:
        return "partial_match"     # got some, missed or added others
    return "wrong"                 # detected something completely different


FAILURE_VERDICTS = {"false_positive", "false_negative", "partial_match", "wrong"}


# ---------------------------------------------------------------------------
# Per-image worker
# ---------------------------------------------------------------------------
def evaluate_one(image_id: int, album_slug: str, url: str, gt_bibs: set,
                 detector: BibDetector, save_failures: bool):
    img = fetch_image(url)

    if img is None:
        return {
            "image_id": image_id,
            "album":    album_slug,
            "gt_bibs":  ",".join(sorted(gt_bibs)),
            "detected": "",
            "verdict":  "fetch_failed",
            "url":      url,
        }

    detections    = detector.detect_bibs_in_image(img)
    detected_bibs = set(d["bib_number"] for d in detections)
    v             = verdict(gt_bibs, detected_bibs)

    if save_failures and v in FAILURE_VERDICTS:
        FAILURES_DIR.mkdir(exist_ok=True)
        out_path = FAILURES_DIR / f"{v}_{album_slug}_{image_id}.jpg"
        cv2.imwrite(str(out_path), img)

    return {
        "image_id": image_id,
        "album":    album_slug,
        "gt_bibs":  ",".join(sorted(gt_bibs)),
        "detected": ",".join(sorted(detected_bibs)),
        "verdict":  v,
        "url":      url,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate detector against ground truth")
    parser.add_argument("--ground-truth", default=DEFAULT_GROUND_TRUTH,
                        help="Path to ground truth file (.txt or .json)")
    parser.add_argument("--album",   default=None, help="Restrict to one album (e.g. finish)")
    parser.add_argument("--limit",   type=int, default=None, help="Max images per album")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--year", required=True,
                        help="2-digit year to evaluate (e.g. 23) — same as fetch_and_label --years")
    parser.add_argument("--no-save-failures", action="store_true",
                        help="Don't save failure images to disk")
    args = parser.parse_args()

    print(f"Loading ground truth: {args.ground_truth}")
    gt        = load_ground_truth(args.ground_truth)
    print(f"Ground truth: {len(gt):,} images  ({sum(1 for b in gt.values() if b):,} with bibs)")

    print("Loading detector model...")
    detector = BibDetector()
    # Warm up: one dummy predict so the model is fully fused before threads start
    detector.detect_bibs_in_image(np.zeros((64, 64, 3), dtype=np.uint8))

    # Fetch album + image list from API — same as fetch_and_label.py
    print(f"Fetching album list for year {args.year}...")
    api_albums = fetch_albums(args.year)
    if args.album:
        api_albums = [a for a in api_albums if a["album_url"].split("/")[-1] == args.album]

    tasks = []  # (image_id, album_slug, url, gt_bibs)
    for album in api_albums:
        album_slug = album["album_url"].split("/")[-1]
        print(f"  Fetching image list: {album_slug}...")
        try:
            filenames = fetch_image_list(album["album_url"])
        except Exception as e:
            print(f"  WARNING: could not fetch image list for {album_slug}: {e}")
            continue
        if args.limit:
            filenames = filenames[:args.limit]
        for filename in filenames:
            image_id = int(Path(filename).stem)
            if image_id not in gt:
                continue
            url = f"{IMAGE_BASE_URL}/{album['album_url']}/{filename}"
            tasks.append((image_id, album_slug, url, gt[image_id]))

    print(f"Evaluating {len(tasks):,} images with {args.workers} workers...\n")

    save_failures = not args.no_save_failures
    results       = []
    counts        = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(evaluate_one, img_id, album_slug, url, gt_bibs, detector, save_failures): img_id
            for img_id, album_slug, url, gt_bibs in tasks
        }
        for fut in tqdm(as_completed(futures), total=len(futures)):
            row = fut.result()
            results.append(row)
            counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1

    # Write CSV
    results.sort(key=lambda r: r["image_id"])
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_id", "album", "gt_bibs", "detected", "verdict", "url"])
        writer.writeheader()
        writer.writerows(results)

    # Summary
    total    = len(results)
    fetched  = total - counts.get("fetch_failed", 0)
    correct  = counts.get("exact_match", 0) + counts.get("true_negative", 0)
    accuracy = correct / fetched * 100 if fetched else 0

    print(f"\n{'='*50}")
    print(f"  Results: {total:,} images evaluated")
    print(f"{'='*50}")
    print(f"  Exact match:      {counts.get('exact_match',    0):>6,}")
    print(f"  True negative:    {counts.get('true_negative',  0):>6,}  (no bibs, none detected)")
    print(f"  Partial match:    {counts.get('partial_match',  0):>6,}  ← missed some bibs")
    print(f"  False positive:   {counts.get('false_positive', 0):>6,}  ← detected non-existent bib")
    print(f"  False negative:   {counts.get('false_negative', 0):>6,}  ← missed all bibs")
    print(f"  Wrong:            {counts.get('wrong',          0):>6,}  ← completely wrong number")
    print(f"  Fetch failed:     {counts.get('fetch_failed',   0):>6,}")
    print(f"{'='*50}")
    print(f"  Accuracy:         {accuracy:.1f}%  ({correct:,} / {fetched:,})")
    if save_failures:
        n_failures = sum(counts.get(v, 0) for v in FAILURE_VERDICTS)
        print(f"  Failure images:   {n_failures:,} saved to failures/")
    print(f"  Full results:     {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

