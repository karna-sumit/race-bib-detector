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
# Some upstream endpoints reject the default python-requests User-Agent.
_session.headers["User-Agent"] = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "Chrome/120.0.0.0 Safari/537.36"
)
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


_fetch_status_counts: dict = {}
_fetch_status_lock = threading.Lock()


def _record_status(code: int):
    with _fetch_status_lock:
        _fetch_status_counts[code] = _fetch_status_counts.get(code, 0) + 1


def fetch_status_snapshot() -> dict:
    with _fetch_status_lock:
        return dict(_fetch_status_counts)


def fetch_image(url: str):
    """Fetch a JPEG from url and return a BGR numpy array, or None on failure."""
    import cv2
    import numpy as np
    import time
    try:
        resp = _session.get(url, timeout=config.IMAGE_FETCH_TIMEOUT)
        _record_status(resp.status_code)
        if resp.status_code in (429, 503):
            retry_after = resp.headers.get("Retry-After")
            logger.warning(
                "RATE_LIMIT status=%s retry_after=%s url=%s",
                resp.status_code, retry_after, url,
            )
            try:
                sleep_s = float(retry_after) if retry_after else 5.0
            except ValueError:
                sleep_s = 5.0
            time.sleep(min(sleep_s, 30.0))
            return None
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        arr = np.frombuffer(resp.content, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        _record_status(-1)
        logger.debug("fetch_image failed url=%s err=%s", url, e)
        return None


def remove_duplicate_dicts(detections: list) -> list:
    """Deduplicate detections by bib_number, keeping the highest yolo_conf result."""
    seen = {}
    for d in detections:
        bib = d["bib_number"]
        if bib not in seen or d["yolo_conf"] > seen[bib]["yolo_conf"]:
            seen[bib] = d
    return list(seen.values())


def fetch_albums() -> list:
    """Fetch the current season's album metadata from the API."""
    resp = _session.get(config.GET_ALBUMS_URL, timeout=config.IMAGE_FETCH_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_image_list(album_url: str) -> list:
    """POST to get-image-list.php and return the list of filenames for an album."""
    resp = _session.post(
        config.GET_IMAGE_LIST_URL,
        json={"album": album_url, "tagger": config.TAGGER_ID},
        timeout=config.IMAGE_FETCH_TIMEOUT,
    )
    resp.raise_for_status()
    return list(resp.json().values())
