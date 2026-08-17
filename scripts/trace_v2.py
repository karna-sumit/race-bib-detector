"""Trace the v2 bib pipeline on one URL/path. Dumps YOLO boxes, edge-filter
decisions, and PaddleOCR raw output. Mirrors detection/detector.py logic.

Usage:
    python3 scripts/trace_v2.py <url-or-path> [<url-or-path> ...] [--outdir /tmp/bib-inspect]
"""
import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import requests
from ultralytics import YOLO

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


def build_ocr():
    from paddleocr import PaddleOCR
    return PaddleOCR(use_angle_cls=False, lang="en", show_log=False)


def paddle_read(reader, roi):
    try:
        results = reader.ocr(roi, cls=False)
    except Exception as e:
        return [], f"exception: {e}"
    if not results or not results[0]:
        return [], "no results"
    out = []
    for item in results[0]:
        if item is None:
            continue
        bbox, (text, conf) = item
        out.append((bbox, text, float(conf)))
    return out, f"{len(out)} raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("srcs", nargs="+")
    ap.add_argument("--outdir", default="/tmp/bib-inspect")
    args = ap.parse_args()

    print(f"Loading YOLO: {config.MODEL_PATH}")
    model = YOLO(str(REPO / config.MODEL_PATH))
    model.to(config.DEVICE)
    print(f"Loading PaddleOCR")
    reader = build_ocr()

    for src in args.srcs:
        stem = Path(src).stem
        out = Path(args.outdir) / stem
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n########## {src} -> {out} ##########")
        img = load_image(src)
        h_img, w_img = img.shape[:2]
        edge_margin = max(4, int(config.EDGE_MARGIN_FRAC * max(h_img, w_img)))
        print(f"image={w_img}x{h_img} edge_margin={edge_margin}")
        cv2.imwrite(str(out / "00_input.jpg"), img)

        results = model.predict(img, imgsz=config.IMAGE_SIZE, device=config.DEVICE, verbose=False)
        annot = img.copy()
        boxes = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                boxes.append((cls, conf, list(map(int, box.xyxy[0]))))
        print(f"YOLO returned {len(boxes)} boxes total (all classes, all confs)")
        for i, (cls, conf, xyxy) in enumerate(boxes):
            x1, y1, x2, y2 = xyxy
            keep = cls == 0 and conf > config.CONF_THRESHOLD
            clipped = (x1 <= edge_margin or x2 >= w_img - edge_margin)
            color = (0, 255, 0) if keep and not clipped else (0, 0, 255)
            cv2.rectangle(annot, (x1, y1), (x2, y2), color, 2)
            label = f"b{i} c{cls} {conf:.2f}"
            if not keep:
                label += " KEEP=NO"
            elif clipped:
                label += " CLIPPED"
            cv2.putText(annot, label, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            print(f"  b{i} cls={cls} conf={conf:.3f} xyxy={xyxy} keep={keep} clipped={clipped}")

        cv2.imwrite(str(out / "01_yolo.jpg"), annot)

        kept = [(i, xyxy, conf) for i, (cls, conf, xyxy) in enumerate(boxes)
                if cls == 0 and conf > config.CONF_THRESHOLD]
        kept.sort(key=lambda t: t[2], reverse=True)

        for i, xyxy, yolo_conf in kept[:10]:
            x1, y1, x2, y2 = xyxy
            clipped = (x1 <= edge_margin or x2 >= w_img - edge_margin)
            if clipped:
                print(f"\n== box {i} SKIPPED (clipped) ==")
                continue
            roi = img[y1:y2, x1:x2]
            rh, rw = roi.shape[:2]
            aspect = rh / rw if rw else 0
            print(f"\n== box {i} yolo_conf={yolo_conf:.3f} xyxy={xyxy} roi={rw}x{rh} h/w={aspect:.2f} ==")
            cv2.imwrite(str(out / f"box_{i}_raw.jpg"), roi)

            # Test multiple crop strategies to see which OCR reads best
            variants = [("full", roi)]
            if aspect >= 3.0:
                variants.append(("top50", roi[: int(rh * 0.50)]))
            elif aspect >= 2.0:
                variants.append(("top65", roi[: int(rh * 0.65)]))
            elif aspect >= 1.5:
                variants.append(("top80", roi[: int(rh * 0.80)]))
            # Always also try top50 for comparison so we can see if the heuristic bands are wrong
            if aspect < 3.0 and aspect >= 1.5:
                variants.append(("top50_forcedcompare", roi[: int(rh * 0.50)]))

            for name, v in variants:
                vh, vw = v.shape[:2]
                min_h = 96
                if vh < min_h:
                    scale = min_h / vh
                    v_up = cv2.resize(v, (int(vw * scale), min_h), interpolation=cv2.INTER_CUBIC)
                else:
                    v_up = v
                cv2.imwrite(str(out / f"box_{i}_{name}_ocr_input.jpg"), v_up)
                ocr, note = paddle_read(reader, v_up)
                cands = []
                for bbox, text, conf in ocr:
                    digits = "".join(c for c in text if c.isdigit())
                    accept = (conf >= config.OCR_CONF_THRESHOLD
                              and digits and re.fullmatch(r"\d{1,4}", digits)
                              and not digits.startswith("0"))
                    cands.append((text, digits, conf, accept))
                good = [c for c in cands if c[3]]
                if good:
                    best = max(good, key=lambda c: c[2])
                    verdict = f"bib={best[1]} conf={best[2]:.3f}"
                elif cands:
                    verdict = "no acceptable | raw=" + ", ".join(
                        f"{t!r}({c:.2f})" for t, _, c, _ in cands[:3]
                    )
                else:
                    verdict = "no OCR results"
                print(f"  [{name:22s} {v_up.shape[1]}x{v_up.shape[0]}] {note} -> {verdict}")

        print(f"artifacts: {out}")


if __name__ == "__main__":
    main()
