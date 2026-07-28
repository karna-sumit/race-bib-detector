# Bib Detection Service

Automatically detects and reads race bib numbers from marathon photos and posts them to a photo tagging API.

The model was trained on ~20,000 manually tagged images from the 2024 race. The same pipeline was then applied to tag runners in the 2025 photos — processing all ~20,000 images in around 15 minutes.

---

## How it works

A YOLOv8 model detects each person in a photo. EasyOCR then reads the bib number from the upper torso region only — this avoids false positives from background signage and text on shorts. Detections are posted to the photo tagging API in real time, with 16 concurrent workers to keep things fast.

```
Fetch image → YOLO detects persons → crop upper torso → EasyOCR reads digits → POST to API
```

The model itself only detects persons (a task it handles well out-of-the-box from COCO pretraining). The bib number is read purely by OCR. Retraining is only needed if image quality or bib design changes significantly between years — for most years, adding the previous year's images and fine-tuning for 30 epochs is sufficient.

---

## Project Structure

```
bib-detection-service/
├── detection/     # Race day pipeline — see detection/README.md
├── training/      # Model training scripts — see training/README.md
├── evaluation/    # Accuracy evaluation scripts
├── models/        # Weights (gitignored — store in GCS)
├── dataset/       # YOLO training data (gitignored)
└── runs/          # YOLO training outputs (gitignored)
```

---

## Where to go next

- **Running detection on race day** → [detection/README.md](detection/README.md)
- **Retraining or evaluating the model** → [training/README.md](training/README.md)
