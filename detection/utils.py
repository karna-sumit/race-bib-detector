import requests
import config
import csv
import logging
import threading
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# Persistent HTTP session with connection pooling
_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)

# Thread-safe CSV lock
_csv_lock = threading.Lock()

# Initialise CSV with header if it doesn't exist
if not os.path.exists(config.OUTPUT_CSV):
    with open(config.OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "image_id", "album_number", "bibs"])


def load_processed_ids() -> set:
    """Return a set of image_ids already written to the output CSV (resume support)."""
    done = set()
    if not os.path.exists(config.OUTPUT_CSV):
        return done
    with open(config.OUTPUT_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                done.add(int(row["image_id"]))
            except (KeyError, ValueError):
                pass
    return done


def remove_duplicate_dicts(detections: list) -> list:
    """Deduplicate detections by bib_number, keeping the highest-confidence result."""
    seen = {}
    for d in detections:
        bib = d["bib_number"]
        if bib not in seen or d["yolo_conf"] > seen[bib]["yolo_conf"]:
            seen[bib] = d
    return list(seen.values())


def post_results(image_id: int, album_number: int, bib_numbers: list):
    """Post detections to the API and append to CSV. Thread-safe."""
    bib_numbers = [d["bib_number"] if isinstance(d, dict) else d for d in bib_numbers]
    try:
        payload = {
            "id": f"{image_id}.jpg",
            "bibNr": ",".join(bib_numbers),
            "albumNr": album_number,
            "tagger": os.getenv("TAGGER_ID"),
        }
        headers = {"accept": "application/json", "content-type": "application/json"}
        _session.post(config.ADD_IMAGE_URL, headers=headers, json=payload, timeout=10)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with _csv_lock:
            with open(config.OUTPUT_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, image_id, album_number, ",".join(bib_numbers)])
    except Exception as e:
        logger.warning("post_results failed for image %s: %s", image_id, e)


def fetch_image(url: str):
    """Fetch a JPEG from url and return a BGR numpy array, or None on failure."""
    import cv2
    import numpy as np
    try:
        resp = _session.get(url, timeout=config.IMAGE_FETCH_TIMEOUT)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        arr = np.frombuffer(resp.content, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

def _remove_substring_bibs(detections: list) -> list:
    """Drop detections whose bib_number is a substring of another detection.

    e.g. if both '123' and '1234' are detected, '123' is dropped because
    it is likely a partial read of the same bib.
    """
    bibs = [d["bib_number"] for d in detections]
    bibs_sorted = sorted(set(bibs), key=len, reverse=True)
    keep = []
    for bib in bibs_sorted:
        if not any(bib in longer for longer in keep):
            keep.append(bib)
    keep_set = set(keep)
    return [d for d in detections if d["bib_number"] in keep_set]


def remove_duplicate_dicts(detections: list) -> list:
    """Deduplicate detections by bib_number (exact), keeping highest yolo_conf,
    then drop any bib that is a pure substring of a longer detected bib."""
    seen = {}
    for d in detections:
        bib = d["bib_number"]
        if bib not in seen or d["yolo_conf"] > seen[bib]["yolo_conf"]:
            seen[bib] = d
    return _remove_substring_bibs(list(seen.values()))
