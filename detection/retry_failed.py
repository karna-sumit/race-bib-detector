"""
retry_failed.py
---------------
Re-runs detection on image IDs that are present in bib_results.csv but had
no bib numbers detected. Useful after adjusting OCR confidence thresholds.

Usage:
    cd detection && python retry_failed.py
"""
import os
import sys
import csv
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
import config
import utils
from detector import BibDetector
from tqdm import tqdm


def build_image_index() -> dict:
    """Return {image_id: (album, filename)} by walking every album's image list."""
    index = {}
    for album in utils.fetch_albums():
        try:
            filenames = utils.fetch_image_list(album["album_url"])
        except Exception as e:  # noqa: BLE001
            album_slug = album["album_url"].split("/")[-1]
            print(f"Warning: could not fetch image list for {album_slug}: {e}")
            continue
        for filename in filenames:
            try:
                index[int(Path(filename).stem)] = (album, filename)
            except ValueError:
                pass
    return index


def main():
    if not os.path.exists(config.OUTPUT_CSV):
        print(f"No results file found: {config.OUTPUT_CSV}")
        return

    # Find image IDs that were processed but had empty bib lists
    empty_ids = []
    with open(config.OUTPUT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            bibs = row.get("bibs", "").strip()
            if not bibs:
                try:
                    empty_ids.append(int(row["image_id"]))
                except (KeyError, ValueError):
                    pass

    if not empty_ids:
        print("No empty results to retry.")
        return

    print("Building image index...")
    index = build_image_index()

    print(f"Retrying {len(empty_ids)} images with empty bib results...")
    detector = BibDetector()

    for img_id in tqdm(empty_ids, desc="Retrying"):
        entry = index.get(img_id)
        if entry is None:
            continue
        album, filename = entry
        url = f"{config.IMAGE_BASE_URL}/{album['album_url']}/{filename}"
        img = utils.fetch_image(url)
        if img is None:
            continue
        detections = detector.detect_bibs_in_image(img)
        if detections:
            utils.post_results(img_id, album["id"], detections)

    print("Retry complete.")


if __name__ == "__main__":
    main()
