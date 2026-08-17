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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OCR engine selection
# ---------------------------------------------------------------------------
# Both engines expose the same read(roi) -> (digits, conf) interface so the
# rest of the pipeline is engine-agnostic. Selection is driven by
# config.OCR_ENGINE. Neither library is fully thread-safe, so all calls go
# through _ocr_lock.
_ocr_lock = threading.Lock()


class _EasyOCREngine:
    name = "easyocr"

    def __init__(self):
        import easyocr
        self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    def read(self, roi):
        results = self.reader.readtext(
            roi,
            allowlist='0123456789',
            detail=1,
            paragraph=False,
        )
        if not results:
            return "", 0.0
        best = max(results, key=lambda r: r[2])
        _, text, conf = best
        conf = float(conf)
        if conf < config.OCR_CONF_THRESHOLD:
            return "", 0.0
        return "".join(filter(str.isdigit, text)), conf


class _PaddleOCREngine:
    name = "paddleocr"

    def __init__(self):
        from paddleocr import PaddleOCR
        # PP-OCRv4 English recognition; angle classifier off (bibs are upright).
        self.reader = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)

    def read(self, roi):
        try:
            results = self.reader.ocr(roi, cls=False)
        except Exception:  # noqa: BLE001
            return "", 0.0
        # PaddleOCR returns [[[bbox, (text, conf)], ...]] or [None]
        if not results or not results[0]:
            return "", 0.0
        cands = []
        for item in results[0]:
            if item is None:
                continue
            _bbox, (text, conf) = item
            conf = float(conf)
            if conf < config.OCR_CONF_THRESHOLD:
                continue
            digits = "".join(filter(str.isdigit, text))
            if not digits:
                continue
            cands.append((digits, conf))
        if not cands:
            return "", 0.0
        return max(cands, key=lambda c: c[1])


def _build_engine():
    choice = config.OCR_ENGINE
    if choice == "paddle" or choice == "paddleocr":
        logger.info("Using PaddleOCR")
        return _PaddleOCREngine()
    if choice == "easyocr" or choice == "easy":
        logger.info("Using EasyOCR")
        return _EasyOCREngine()
    raise ValueError(f"Unknown OCR_ENGINE: {config.OCR_ENGINE!r}")


_ocr_engine = _build_engine()


def _ocr_roi(roi):
    """Run digit OCR on a crop. Thread-safe via lock.
    Returns (digits, conf) - digits is "" on miss.
    """
    with _ocr_lock:
        return _ocr_engine.read(roi)


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
    def _process_one(self, filename, album, batcher):
        """Fetch and process a single image. Designed to run in a thread pool."""
        img_id = int(Path(filename).stem)
        url = f"{config.IMAGE_BASE_URL}/{album['album_url']}/{filename}"
        img = utils.fetch_image(url)
        if img is None:
            return
        try:
            detections = self.detect_bibs_in_image(img)
            if detections:
                # CSV write disabled; no resume support until re-enabled.
                # utils.append_csv_row(img_id, album["id"], detections)
                batcher.add(img_id, detections)
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
        batcher = utils.PostBatcher(album["id"], batch_size=config.POST_BATCH_SIZE)
        completed = 0
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(self._process_one, fn, album, batcher): fn for fn in pending}
                for _ in tqdm(as_completed(futures), total=len(futures), desc=album_name, leave=False):
                    completed += 1
                    if completed % 250 == 0:
                        logger.info(
                            "GET: %s | POST: %s",
                            utils.fetch_status_snapshot(),
                            utils.post_status_snapshot(),
                        )
        finally:
            batcher.flush()
        logger.info(
            "album '%s' done. GET: %s | POST: %s",
            album_name,
            utils.fetch_status_snapshot(),
            utils.post_status_snapshot(),
        )

    def _process_result_boxes(self, result, img):
        """Filter person boxes, skip frame-clipped ones, cap at 10, run OCR."""
        h_img, w_img = img.shape[:2]
        edge_margin = max(4, int(config.EDGE_MARGIN_FRAC * max(h_img, w_img)))
        person_boxes = [b for b in result.boxes if self._is_valid_person_box(b)]
        person_boxes.sort(key=lambda b: float(b.conf[0]), reverse=True)
        detections = []
        for box in person_boxes[:10]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if (x1 <= edge_margin or y1 <= edge_margin
                    or x2 >= w_img - edge_margin or y2 >= h_img - edge_margin):
                continue
            roi = img[y1:y2, x1:x2]
            if roi.size > 0:
                detections.extend(self._extract_bibs_from_roi(roi, x1, y1, x2, y2, float(box.conf[0])))
        return detections

    def _is_valid_person_box(self, box):
        """Check if the box represents a person with high enough confidence."""
        return int(box.cls[0]) == 0 and float(box.conf[0]) > config.CONF_THRESHOLD

    def _extract_bibs_from_roi(self, roi, x1, y1, x2, y2, yolo_conf):
        """Run OCR on the detected bib crop and return any bib found.

        The detector (v2) predicts tight bib bboxes directly, so we upscale
        small crops to give EasyOCR enough pixels to work with. No edge-margin
        check: with a tight bib crop, digits legitimately fill the ROI.
        """
        if roi.size == 0:
            return []

        # Upscale small crops so EasyOCR has enough pixels. Bibs seen from a
        # distance can be only ~30px tall in the source image.
        rh, rw = roi.shape[:2]
        min_h = 96
        if rh < min_h:
            scale = min_h / rh
            roi = cv2.resize(roi, (int(rw * scale), min_h), interpolation=cv2.INTER_CUBIC)

        clean_text, ocr_conf = _ocr_roi(roi)
        if not clean_text or not re.fullmatch(r"\d{1,4}", clean_text):
            return []
        if clean_text.startswith("0"):
            return []

        return [{
            "bib_number": clean_text,
            "yolo_conf": yolo_conf,
            "ocr_conf": ocr_conf,
            "bbox": [x1, y1, x2, y2]
        }]
