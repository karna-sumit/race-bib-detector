"""
train_detector.py  (training/annual/)
--------------------------------------
Path A — Step 2. Run this after fetch_and_label.py has added new images.

Fine-tunes models/best.pt on the current dataset/ for a fixed number of
epochs. Stops early if validation loss doesn't improve for 10 epochs.
Checkpoints are saved to runs/detect/trainN/weights/.

Usage:
    python training/annual/train_detector.py                   # Apple Silicon
    python training/annual/train_detector.py --device 0 --batch 32  # GCP T4
"""

import argparse
import subprocess
import os


CHECKPOINT = "models/best.pt"
DATA_YAML  = "training/data.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of additional epochs (default 30)")
    parser.add_argument("--device", default="mps",
                        help="Device: mps (Apple Silicon), 0 (CUDA GPU), cpu")
    parser.add_argument("--batch",  type=int, default=16,
                        help="Batch size — use 32 on T4, 16 on MPS")
    parser.add_argument("--imgsz",  type=int, default=640)
    args = parser.parse_args()

    if not os.path.exists(CHECKPOINT):
        print(f"Checkpoint not found: {CHECKPOINT}")
        print("Run:  cp runs/detect/train23/weights/best.pt models/best.pt")
        return

    print(f"Fine-tuning from: {CHECKPOINT}")
    print(f"Epochs: {args.epochs}  |  Device: {args.device}  |  Batch: {args.batch}")

    cmd = [
        "yolo", "detect", "train",
        f"data={DATA_YAML}",
        f"model={CHECKPOINT}",
        f"epochs={args.epochs}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"device={args.device}",
        "patience=10",        # stop early if no improvement for 10 epochs
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

