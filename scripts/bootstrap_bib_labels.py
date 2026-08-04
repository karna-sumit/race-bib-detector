"""Bootstrap bib bounding-box labels from confident detections.

Reads an evaluation_results CSV, re-runs YOLO+OCR on each image, and keeps
any OCR read whose digits exactly match a ground-truth bib. Writes a
YOLO-format labeled dataset for training a bib-region detector.

Output layout (importable by CVAT / label-studio / roboflow):
    <outdir>/
        images/<image_id>.jpg
        labels/<image_id>.txt   # class cx cy w h (normalized 0-1)
        classes.txt             # single line: "bib"
        review.csv              # image_id, gt_bibs, matched, dropped, note

Usage:
    python scripts/bootstrap_bib_labels.py \\
        --csv evaluation_results.crop_fix.csv \\
        --outdir autolabels \\
        --limit 500
"""

import argparse
import csv
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import requests
from tqdm import tqdm
from ultralytics import YOLO
import easyocr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "detection"))
import config  # noqa: E402


_ocr_lock = threading.Lock()


def load_image(url: str):
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        arr = np.frombuffer(r.content, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def extract_bib_boxes(model, reader, img, gt_bibs, pad):
    """Return list of (bib_text, x1, y1, x2, y2) in full-image coords for OCR
    reads whose text exactly matches one of the ground truth bibs. The bbox
    is padded by `pad` (fraction of its own width/height) on each side and
    clipped to image bounds."""
    ih, iw = img.shape[:2]
    matches = []
    results = model.predict(img, imgsz=config.IMAGE_SIZE,
                            device=config.DEVICE, verbose=False)
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) != 0 or float(box.conf[0]) <= config.CONF_THRESHOLD:
                continue
            px1, py1, px2, py2 = map(int, box.xyxy[0])
            roi = img[py1:py2, px1:px2]
            if roi.size == 0:
                continue
            with _ocr_lock:
                ocr = reader.readtext(roi, allowlist='0123456789',
                                      detail=1, paragraph=False)
            for bbox, text, conf in ocr:
                digits = "".join(c for c in text if c.isdigit())
                if digits in gt_bibs and conf >= config.OCR_CONF_THRESHOLD:
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    x1 = px1 + min(xs)
                    y1 = py1 + min(ys)
                    x2 = px1 + max(xs)
                    y2 = py1 + max(ys)
                    bw = x2 - x1
                    bh = y2 - y1
                    x1 = int(max(0, x1 - bw * pad))
                    y1 = int(max(0, y1 - bh * pad))
                    x2 = int(min(iw, x2 + bw * pad))
                    y2 = int(min(ih, y2 + bh * pad))
                    matches.append((digits, x1, y1, x2, y2))
    return matches


def to_yolo(x1, y1, x2, y2, w, h):
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh


def process_row(row, model, reader, out_images, out_labels, pad):
    gt_bibs = set(b.strip() for b in row["gt_bibs"].split(",") if b.strip())
    if not gt_bibs:
        return None
    img = load_image(row["url"])
    if img is None:
        return {"image_id": row["image_id"], "gt_bibs": row["gt_bibs"],
                "matched": "", "dropped": "", "note": "fetch_failed"}
    matches = extract_bib_boxes(model, reader, img, gt_bibs, pad)
    matched_texts = set(m[0] for m in matches)
    dropped = gt_bibs - matched_texts
    if not matches:
        return {"image_id": row["image_id"], "gt_bibs": row["gt_bibs"],
                "matched": "", "dropped": ",".join(sorted(dropped)),
                "note": "no_matches"}

    h, w = img.shape[:2]
    img_path = out_images / f"{row['image_id']}.jpg"
    label_path = out_labels / f"{row['image_id']}.txt"
    cv2.imwrite(str(img_path), img)
    with open(label_path, "w") as f:
        for _, x1, y1, x2, y2 in matches:
            cx, cy, bw, bh = to_yolo(x1, y1, x2, y2, w, h)
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    return {"image_id": row["image_id"], "gt_bibs": row["gt_bibs"],
            "matched": ",".join(sorted(matched_texts)),
            "dropped": ",".join(sorted(dropped)),
            "note": "ok"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="evaluation_results.crop_fix.csv")
    ap.add_argument("--outdir", default="autolabels")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap number of candidate rows to process")
    ap.add_argument("--verdicts", default="exact_match,partial_match",
                    help="Comma-separated verdicts to include")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--pad", type=float, default=0.3,
                    help="Padding around OCR bbox as fraction of box size "
                         "(0 = tight digits, 0.3 = whole-bib rectangle)")
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="If set, shuffle rows with this seed before applying --limit "
                         "(useful for stratified sampling across albums)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    out_images = outdir / "images"
    out_labels = outdir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    (outdir / "classes.txt").write_text("bib\n")

    keep = set(args.verdicts.split(","))
    with open(args.csv) as f:
        rows = [r for r in csv.DictReader(f)
                if r["verdict"] in keep and r["gt_bibs"].strip()]
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(rows)
    if args.limit:
        rows = rows[:args.limit]
    print(f"Loaded {len(rows):,} candidate rows from {args.csv} "
          f"(verdicts: {sorted(keep)})")

    print(f"Loading YOLO model: {config.MODEL_PATH}")
    model = YOLO(str(REPO / config.MODEL_PATH))
    model.to(config.DEVICE)
    print("Loading EasyOCR")
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    review = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_row, r, model, reader, out_images, out_labels, args.pad): r
                   for r in rows}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            res = fut.result()
            if res is not None:
                review.append(res)

    review.sort(key=lambda r: int(r["image_id"]))
    with open(outdir / "review.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "gt_bibs", "matched",
                                          "dropped", "note"])
        w.writeheader()
        w.writerows(review)

    ok = sum(1 for r in review if r["note"] == "ok")
    no_match = sum(1 for r in review if r["note"] == "no_matches")
    fetch_fail = sum(1 for r in review if r["note"] == "fetch_failed")
    total_boxes = sum(len(list((out_labels / f"{r['image_id']}.txt").open()))
                      for r in review if r["note"] == "ok")

    print()
    print(f"Labeled images: {ok:,} ({total_boxes:,} bib boxes)")
    print(f"No matches:     {no_match:,}")
    print(f"Fetch failed:   {fetch_fail:,}")
    print(f"Output:         {outdir}/")
    print(f"Review CSV:     {outdir}/review.csv")


if __name__ == "__main__":
    main()
