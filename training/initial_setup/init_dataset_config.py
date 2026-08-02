"""
init_dataset_config.py  (training/initial_setup/)
---------------------------------------------------
Path B - Step 4. One-time setup only.

Writes training/data.yaml pointing at the dataset/ folder.
Only needs to be run once when first setting up the project.

Usage:
    python training/initial_setup/init_dataset_config.py
"""

import yaml
import os

# Paths
DATASET_DIR = "dataset"  # root of your YOLO dataset
YAML_FILE = "data.yaml"

# Build dictionary
data = {
    "train": os.path.join(DATASET_DIR, "images/train"),
    "val": os.path.join(DATASET_DIR, "images/val"),
    "nc": 1,
    "names": ["bib"]
}

# Save as YAML
with open(YAML_FILE, "w") as f:
    yaml.dump(data, f, default_flow_style=False)

print(f"✅ Created {YAML_FILE} pointing to {DATASET_DIR}")
