from ultralytics import YOLO
import utils
import config
import logging
import re
import threading
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import easyocr

logger = logging.getLogger(__name__)

# EasyOCR reader - loaded once, shared across threads.
# readtext() is not thread-safe internally, so all OCR calls go through a lock.
_ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
_ocr_lock   = threading.Lock()


def _ocr_roi(roi):
    """Run EasyOCR digit-only recognition on a small crop. Thread-safe via lock.
    Returns (text, bbox, conf) - text and bbox are empty/None on miss.
    """
    with _ocr_lock:
        results = _ocr_reader.readtext(
            roi,
            allowlist='0123456789',
            detail=1,
            paragraph=False,
        )
    if not results:
        return "", None, 0.0
    best = max(results, key=lambda r: r[2])
    bbox, text, conf = best[0], best[1], best[2]
    if conf < config.OCR_CONF_THRESHOLD:
        return "", None, 0.0
    return "".join(filter(str.isdigit, text)), bbox, float(conf)


class BibDetector:
    def __init__(self):
        self.model = YOLO(config.MODEL_PATH)
        self.model.to(config.DEVICE)

    # ==================== Public Methods ====================
    def detect_bibs_in_image(self, img):
        """Detect bib numbers in a single image."""
        results = self.model.predict(
            img, imgsz=config.IMAGE_SIZE, device=config.DEVICE, verbose=False
        )
        all_detections = []
        for result in results:
            all_detections.extend(self._process_result_boxes(result, img))
        return utils.remove_duplicate_dicts(all_detections)

    def batch_detect_bibs(self):
        """Run detection on all images using a thread pool for concurrent fetching + processing."""
        workers = getattr(config, "WORKERS", 16)
        albums = utils.fetch_albums()
        for album in albums:
            self._process_album_concurrent(album, workers)

    # ==================== Private Helpers ====================
    def _process_one(self, filename, album):
        """Fetch and process a single image. Designed to run in a thread pool."""
        img_id = int(Path(filename).stem)
        url = f"{config.IMAGE_BASE_URL}/{album['album_url']}/{filename}"
        img = utils.fetch_image(url)
        if img is None:
            return
        try:
            detections = self.detect_bibs_in_image(img)
            if detections:
                utils.post_results(img_id, album["albumNr"], detections)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed image %s: %s", img_id, e)

    def _process_album_concurrent(self, album, workers):
        album_name = album.get("name") or album["album_url"].split("/")[-1]
        try:
            filenames = utils.fetch_image_list(album["album_url"])
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to fetch image list for %s: %s", album_name, e)
            return

        done = utils.load_processed_ids()
        pending = [fn for fn in filenames if int(Path(fn).stem) not in done]
        if not pending:
            print(f"Album '{album_name}' already complete, skipping.")
            return

        print(f"Processing album '{album_name}' - {len(pending)} remaining / {len(filenames)} total")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self._process_one, fn, album): fn for fn in pending}
            for _ in tqdm(as_completed(futures), total=len(futures), desc=album_name, leave=False):
                pass

    def _process_result_boxes(self, result, img):
        """Process YOLO prediction boxes for a single result.

        Filters to person boxes above threshold, sorts by descending YOLO
        confidence, and caps at 10 to bound OCR cost per image.
        """
        person_boxes = [b for b in result.boxes if self._is_valid_person_box(b)]
        person_boxes.sort(key=lambda b: float(b.conf[0]), reverse=True)
        detections = []
        for box in person_boxes[:10]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = img[y1:y2, x1:x2]
            if roi.size > 0:
                detections.extend(self._extract_bibs_from_roi(roi, x1, y1, x2, y2, float(box.conf[0])))
        return detections

    def _is_valid_person_box(self, box):
        """Check if the box represents a person with high enough confidence."""
        return int(box.cls[0]) == 0 and float(box.conf[0]) > config.CONF_THRESHOLD

    def _extract_bibs_from_roi(self, roi, x1, y1, x2, y2, yolo_conf):
        """Run OCR on the full person crop and return any bib found.

        The OCR confidence threshold filters noise from shoes/shorts logos,
        and the margin check drops reads that hug the box edges (typically
        partial reads of clothing text bleeding out of the crop).
        """
        if roi.size == 0:
            return []
        clean_text, ocr_bbox, ocr_conf = _ocr_roi(roi)
        if not clean_text or not re.fullmatch(r"\d{1,4}", clean_text):
            return []

        # Margin check - drop text detected within 15% of any person-box edge
        if ocr_bbox is not None:
            rh, rw = roi.shape[:2]
            margin = min(rw, rh) * 0.15
            xs = [pt[0] for pt in ocr_bbox]
            ys = [pt[1] for pt in ocr_bbox]
            if min(xs) < margin or min(ys) < margin or \
               max(xs) > rw - margin or max(ys) > rh - margin:
                return []

        return [{
            "bib_number": clean_text,
            "yolo_conf": yolo_conf,
            "ocr_conf": ocr_conf,
            "bbox": [x1, y1, x2, y2]
        }]
