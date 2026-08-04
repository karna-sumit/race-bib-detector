"""Trace the bib-detection pipeline on a single image and dump every stage.

Usage:
    python3 scripts/trace_one.py <image-url-or-path> [--outdir /tmp/bib-inspect]

Emits, into outdir/<stem>/:
    00_input.jpg              full input
    01_yolo_boxes.jpg         input with person boxes drawn
    person_<i>_full.jpg       each person crop
    person_<i>_torso.jpg      torso crop (top 55%)
    person_<i>_full_ocr.jpg   full crop with all raw OCR boxes drawn
    person_<i>_torso_ocr.jpg  torso crop with all raw OCR boxes drawn

Prints, per person:
    - YOLO conf, box coords, crop shape
    - RAW OCR results on the FULL person crop (no filtering)
    - RAW OCR results on the TORSO crop (no filtering)
    - What the current pipeline (55% + margin + regex) would return
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import requests
from ultralytics import YOLO
import easyocr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "detection"))
import config  # noqa: E402


def load_image(src: str) -> np.ndarray:
    if src.startswith(("http://", "https://")):
        r = requests.get(src, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        arr = np.frombuffer(r.content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(src)
    if img is None:
        raise SystemExit(f"Failed to load {src}")
    return img


def draw_ocr(crop: np.ndarray, results) -> np.ndarray:
    out = crop.copy()
    for bbox, text, conf in results:
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 0), 2)
        x, y = pts[0]
        cv2.putText(out, f"{text} ({conf:.2f})", (int(x), max(int(y) - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


def apply_current_pipeline(torso: np.ndarray, ocr_results) -> str:
    """Replicate detector._extract_bibs_from_roi's filters."""
    import re
    if not ocr_results:
        return "(no OCR result)"
    best = max(ocr_results, key=lambda r: r[2])
    bbox, text, conf = best[0], best[1], best[2]
    reason = []
    reason.append(f"best raw text='{text}' conf={conf:.3f}")
    if conf < config.OCR_CONF_THRESHOLD:
        reason.append(f"REJECTED: conf < OCR_CONF_THRESHOLD={config.OCR_CONF_THRESHOLD}")
        return " | ".join(reason)
    digits = "".join(c for c in text if c.isdigit())
    reason.append(f"digits='{digits}'")
    if not re.fullmatch(r"\d{1,4}", digits):
        reason.append("REJECTED: not 1-4 digits")
        return " | ".join(reason)
    th, tw = torso.shape[:2]
    margin = min(tw, th) * 0.15
    xs = [pt[0] for pt in bbox]
    ys = [pt[1] for pt in bbox]
    if (min(xs) < margin or min(ys) < margin or
            max(xs) > tw - margin or max(ys) > th - margin):
        reason.append(f"REJECTED: margin check (torso {tw}x{th}, margin={margin:.0f}, "
                      f"bbox x=[{min(xs):.0f},{max(xs):.0f}] y=[{min(ys):.0f},{max(ys):.0f}])")
        return " | ".join(reason)
    reason.append(f"ACCEPTED: {digits}")
    return " | ".join(reason)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="image URL or local path")
    ap.add_argument("--outdir", default="/tmp/bib-inspect")
    args = ap.parse_args()

    stem = Path(args.src).stem
    out = Path(args.outdir) / stem
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.src}")
    img = load_image(args.src)
    print(f"Image shape: {img.shape}")
    cv2.imwrite(str(out / "00_input.jpg"), img)

    print(f"Loading YOLO model: {config.MODEL_PATH}")
    model = YOLO(str(REPO / config.MODEL_PATH))
    model.to(config.DEVICE)

    print(f"Loading EasyOCR")
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    print(f"\nRunning YOLO (imgsz={config.IMAGE_SIZE}, conf_thr={config.CONF_THRESHOLD})...")
    results = model.predict(img, imgsz=config.IMAGE_SIZE, device=config.DEVICE, verbose=False)

    annot = img.copy()
    person_boxes = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 0 and conf > config.CONF_THRESHOLD:
                person_boxes.append(box)
    person_boxes.sort(key=lambda b: float(b.conf[0]), reverse=True)

    print(f"Persons above threshold: {len(person_boxes)}")
    for i, box in enumerate(person_boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cv2.rectangle(annot, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(annot, f"p{i} {conf:.2f}", (x1, max(y1 - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(str(out / "01_yolo_boxes.jpg"), annot)

    for i, box in enumerate(person_boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        crop = img[y1:y2, x1:x2]
        h, w = crop.shape[:2]
        torso = crop[:int(h * 0.55), :]

        cv2.imwrite(str(out / f"person_{i}_full.jpg"), crop)
        cv2.imwrite(str(out / f"person_{i}_torso.jpg"), torso)

        print(f"\n=== Person {i}  yolo_conf={conf:.3f}  box=({x1},{y1})->({x2},{y2})  crop={w}x{h}  torso={w}x{int(h*0.55)} ===")

        raw_full = reader.readtext(crop, allowlist='0123456789', detail=1, paragraph=False)
        print(f"  RAW OCR on FULL crop ({len(raw_full)} results):")
        for bbox, text, conf_o in raw_full:
            print(f"    text='{text}'  conf={conf_o:.3f}  bbox={[[int(p[0]),int(p[1])] for p in bbox]}")
        cv2.imwrite(str(out / f"person_{i}_full_ocr.jpg"), draw_ocr(crop, raw_full))

        raw_torso = reader.readtext(torso, allowlist='0123456789', detail=1, paragraph=False)
        print(f"  RAW OCR on TORSO (top 55%) ({len(raw_torso)} results):")
        for bbox, text, conf_o in raw_torso:
            print(f"    text='{text}'  conf={conf_o:.3f}  bbox={[[int(p[0]),int(p[1])] for p in bbox]}")
        cv2.imwrite(str(out / f"person_{i}_torso_ocr.jpg"), draw_ocr(torso, raw_torso))

        verdict = apply_current_pipeline(torso, raw_torso)
        print(f"  Current pipeline verdict: {verdict}")

    print(f"\nAll artifacts saved to: {out}")


if __name__ == "__main__":
    main()
