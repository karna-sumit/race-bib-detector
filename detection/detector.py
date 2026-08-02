from ultralytics import YOLO
import utils
import config
import logging
import re
import threading
import cv2
import numpy as np
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
        for album in config.albums:
            self._process_album_concurrent(album, workers)

    # ==================== Private Helpers ====================
    def _process_one(self, img_id, album):
        """Fetch and process a single image. Designed to run in a thread pool."""
        url = config.GET_IMAGE_URL.format(album_name=album["name"], image_id=img_id)
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
        album_name = album["name"]
        start_id = album["startImageId"]
        end_id = start_id + album["noOfImages"]
        done = utils.load_processed_ids()

        ids = [i for i in range(start_id, end_id) if i not in done]
        if not ids:
            print(f"Album '{album_name}' already complete, skipping.")
            return

        print(f"Processing album '{album_name}' - {len(ids)} remaining / {album['noOfImages']} total")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self._process_one, img_id, album): img_id for img_id in ids}
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
        """Run OCR on the upper torso of the person crop and return any bib found.

        Bibs sit on the chest - roughly the top 55% of a person bounding box.
        Scanning the full box causes false positives from race signage, shorts
        text, and shoe logos in the lower half (the classic '10' artefact).

        The margin check drops any OCR result whose bounding box sits too close
        to the edge of the torso crop - likely a partial read of clothing text
        rather than the bib number.
        """
        h, w = roi.shape[:2]
        torso = roi[:int(h * 0.55), :]
        if torso.size == 0:
            return []
        clean_text, ocr_bbox, ocr_conf = _ocr_roi(torso)
        if not clean_text or not re.fullmatch(r"\d{1,4}", clean_text):
            return []

        # Margin check - drop text detected within 15% of any torso edge
        if ocr_bbox is not None:
            th, tw = torso.shape[:2]
            margin = min(tw, th) * 0.15
            xs = [pt[0] for pt in ocr_bbox]
            ys = [pt[1] for pt in ocr_bbox]
            if min(xs) < margin or min(ys) < margin or \
               max(xs) > tw - margin or max(ys) > th - margin:
                return []

        return [{
            "bib_number": clean_text,
            "yolo_conf": yolo_conf,
            "ocr_conf": ocr_conf,
            "bbox": [x1, y1, x2, y2]
        }]
