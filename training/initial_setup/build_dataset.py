"""
build_dataset.py  (training/initial_setup/)
--------------------------------------------
Path B - Step 3. One-time setup only.

Reads all annotated batches from marathon_annotation_batches/, converts
bounding boxes to YOLO normalised format, and writes the final dataset
into dataset/ split 80% train / 20% val.

Already-processed batches are skipped (tracked in processed_batches.json)
so it's safe to re-run if new batches are added.

Usage:
    python training/initial_setup/build_dataset.py
"""

import os
import json
import shutil
import cv2
import argparse
from tqdm import tqdm
import random

# ===== CONFIG =====
RAW_IMAGES_ROOT = "marathon_annotation_batches"  # root folder with batch_1..batch_x
YOLO_DATASET_DIR = "dataset"                    # output dataset folder
TRAIN_RATIO = 0.8                               # train/val split ratio
PROCESSED_BATCHES_FILE = "processed_batches.json"
DATA_YAML = "data.yaml"                         # YOLO training config

# Normalize bbox to YOLO format
def normalize_bbox(x1, y1, x2, y2, img_w, img_h):
    xc = (x1 + x2) / 2.0 / img_w
    yc = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return xc, yc, w, h

def load_processed_batches():
    if os.path.exists(PROCESSED_BATCHES_FILE):
        with open(PROCESSED_BATCHES_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_processed_batches(processed):
    with open(PROCESSED_BATCHES_FILE, "w") as f:
        json.dump(sorted(list(processed)), f, indent=2)

def gather_images_from_batches(specific_batches=None):
    all_images = []
    updated_batches = []

    for batch_folder in sorted(os.listdir(RAW_IMAGES_ROOT)):
        if not batch_folder.startswith("batch_"):
            continue

        batch_nr = int(batch_folder.split("_")[1])
        if specific_batches and batch_nr not in specific_batches:
            continue

        batch_path = os.path.join(RAW_IMAGES_ROOT, batch_folder)
        metadata_path = os.path.join(batch_path, "batch_metadata.json")
        if not os.path.exists(metadata_path):
            continue

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        has_detections = any(img.get("detections") for img in metadata.get("images", []))
        if not has_detections:
            continue

        updated_batches.append(batch_nr)

        for img_data in metadata.get("images", []):
            img_file = os.path.join(batch_path, img_data["filename"])
            if os.path.exists(img_file) and img_data.get("detections"):
                all_images.append((img_data["filename"].split(".")[0], img_file, img_data["detections"], batch_nr))

    return all_images, updated_batches

def process_images(img_list, img_dest, label_dest):
    os.makedirs(img_dest, exist_ok=True)
    os.makedirs(label_dest, exist_ok=True)

    for img_id, img_path, detections, batch_nr in tqdm(img_list, desc=f"Processing {img_dest}"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        yolo_lines = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            xc, yc, bw, bh = normalize_bbox(x1, y1, x2, y2, w, h)
            yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if not yolo_lines:
            continue

        # Label
        label_path = os.path.join(label_dest, f"{img_id}.txt")
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

        # Image
        dest_img_path = os.path.join(img_dest, f"{img_id}.jpg")
        shutil.copy(img_path, dest_img_path)

def cleanup_batch(batch_nr):
    """Remove old images/labels for a given batch_nr before refreshing"""
    for split in ["train", "val"]:
        img_dir = os.path.join(YOLO_DATASET_DIR, "images", split)
        label_dir = os.path.join(YOLO_DATASET_DIR, "labels", split)
        if not os.path.exists(img_dir):
            continue
        for f in os.listdir(img_dir):
            if f.endswith(".jpg") and f.startswith(str(batch_nr) + "_"):
                os.remove(os.path.join(img_dir, f))
        if not os.path.exists(label_dir):
            continue
        for f in os.listdir(label_dir):
            if f.endswith(".txt") and f.startswith(str(batch_nr) + "_"):
                os.remove(os.path.join(label_dir, f))

def write_data_yaml():
    content = {
        "names": ["bib"],
        "nc": 1,
        "train": os.path.join(YOLO_DATASET_DIR, "images/train"),
        "val": os.path.join(YOLO_DATASET_DIR, "images/val"),
    }
    with open(DATA_YAML, "w") as f:
        yaml_str = (
            f"names:\n- bib\n"
            f"nc: 1\n"
            f"train: {content['train']}\n"
            f"val: {content['val']}\n"
        )
        f.write(yaml_str)
    print(f"✅ Wrote {DATA_YAML}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--incremental", action="store_true", help="Update only new or re-processed batches")
    args = parser.parse_args()

    processed_batches = load_processed_batches()

    if args.incremental:
        all_images, updated_batches = gather_images_from_batches()
        # Always refresh (even if already processed)
        target_batches = updated_batches

        if not target_batches:
            print("✅ No updated batches found.")
            return

        print(f"📦 Incrementally refreshing {len(target_batches)} batches: {target_batches}")

        for batch_nr in target_batches:
            cleanup_batch(batch_nr)  # remove old data first
            batch_images, _ = gather_images_from_batches(specific_batches=[batch_nr])
            for img_id, img_path, detections, _ in batch_images:
                if random.random() < TRAIN_RATIO:
                    process_images([(img_id, img_path, detections, batch_nr)],
                                   os.path.join(YOLO_DATASET_DIR, "images/train"),
                                   os.path.join(YOLO_DATASET_DIR, "labels/train"))
                else:
                    process_images([(img_id, img_path, detections, batch_nr)],
                                   os.path.join(YOLO_DATASET_DIR, "images/val"),
                                   os.path.join(YOLO_DATASET_DIR, "labels/val"))

        processed_batches.update(target_batches)
        save_processed_batches(processed_batches)
        write_data_yaml()

    else:
        print("♻️ Rebuilding dataset from scratch...")
        if os.path.exists(YOLO_DATASET_DIR):
            shutil.rmtree(YOLO_DATASET_DIR)
        for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
            os.makedirs(os.path.join(YOLO_DATASET_DIR, sub), exist_ok=True)

        all_images, updated_batches = gather_images_from_batches()
        all_images.sort()
        split_idx = int(len(all_images) * TRAIN_RATIO)
        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]

        print(f"✅ Found {len(updated_batches)} updated batches: {updated_batches}")
        print(f"📸 Total images with detections: {len(all_images)}")

        process_images(train_images,
                       os.path.join(YOLO_DATASET_DIR, "images/train"),
                       os.path.join(YOLO_DATASET_DIR, "labels/train"))
        process_images(val_images,
                       os.path.join(YOLO_DATASET_DIR, "images/val"),
                       os.path.join(YOLO_DATASET_DIR, "labels/val"))

        processed_batches = set(updated_batches)
        save_processed_batches(processed_batches)
        write_data_yaml()
        print("✅ YOLO dataset rebuilt successfully.")

if __name__ == "__main__":
    main()
