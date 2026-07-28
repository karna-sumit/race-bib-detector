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
sys.path.insert(0, os.path.dirname(__file__))
import config
import utils
from detector import BibDetector
from tqdm import tqdm


def find_album(image_id: int) -> dict | None:
    """Return the album dict for a given image_id, or None if not found."""
    for album in config.albums:
        if album["startImageId"] <= image_id < album["startImageId"] + album["noOfImages"]:
            return album
    return None


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

    print(f"Retrying {len(empty_ids)} images with empty bib results...")
    detector = BibDetector()

    for img_id in tqdm(empty_ids, desc="Retrying"):
        album = find_album(img_id)
        if album is None:
            continue
        url = config.GET_IMAGE_URL.format(album_name=album["name"], image_id=img_id)
        img = utils.fetch_image(url)
        if img is None:
            continue
        detections = detector.detect_bibs_in_image(img)
        if detections:
            utils.post_results(img_id, album["albumNr"], detections)

    print("Retry complete.")


if __name__ == "__main__":
    main()
