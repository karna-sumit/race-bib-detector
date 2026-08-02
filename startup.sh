#!/bin/bash
# Startup script - runs automatically when the VM boots.
# Fetches 2023 images, fine-tunes the model, then shuts down the VM.

set -e
cd /home/zumitkrn/race-bib-detector

# Activate venv
source venv/bin/activate

# Run training pipeline
python3 training/annual/fetch_and_label.py --years 23
python3 training/annual/train_detector.py --device 0 --batch 32

# Shut down VM when done so it stops billing
sudo shutdown -h now
