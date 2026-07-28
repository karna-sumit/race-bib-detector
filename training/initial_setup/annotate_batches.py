"""
annotate_batches.py  (training/initial_setup/)
-----------------------------------------------
Path B — Step 2. One-time setup only.

Runs the current detector (models/best.pt) over each manually-collected
image batch and writes the resulting bounding-box coordinates into
batch_metadata.json alongside each image.

Expects batches under marathon_annotation_batches/ in the format:
    batch_1/
        batch_metadata.json   ← {"images": [{"filename": "x.jpg"}, ...]}
        image_x.jpg

Usage:
    python training/initial_setup/annotate_batches.py
"""

import os
import json
import sys
from pathlib import Path
import cv2
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'detection'))
from detector import BibDetector  # type: ignore[import]

BATCHES_ROOT = "marathon_annotation_batches"

def process_batch(batch_path, detector):
    metadata_file = Path(batch_path) / "batch_metadata.json"
    if not metadata_file.exists():
        print(f"⚠️ Skipping {batch_path}, no metadata.json")
        return

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    images = metadata.get("images", [])
    updated_images = []

    for img_info in tqdm(images, desc=f"Processing {batch_path.name}"):
        img_path = Path(batch_path) / img_info["filename"]
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        detections = detector.detect_bibs_in_image(img)

        img_info["detections"] = detections
        updated_images.append(img_info)

    metadata["images"] = updated_images

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Updated {metadata_file} with detections")


def process_all_batches():
    detector = BibDetector()
    for batch_dir in sorted(os.listdir(BATCHES_ROOT)):
        full_path = Path(BATCHES_ROOT) / batch_dir
        if full_path.is_dir():
            process_batch(full_path, detector)

    print("🎉 All batches updated with detections")


if __name__ == "__main__":
    process_all_batches()
