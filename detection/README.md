# Detection Guide

This directory contains the live bib detection pipeline - the code you run on race day.

**~20,000 images processed in ~15 minutes** using YOLOv8 person detection + EasyOCR with 16 concurrent workers.

---

## How it works

```
Fetch image → YOLO detects persons → crop upper torso → EasyOCR reads digits → POST to API
```

- YOLO restricts OCR to the chest area only, eliminating false positives from background signage and race route markers
- 16 concurrent worker threads overlap network I/O with inference
- Progress is checkpointed to `bib_results.csv` after every image - restarts resume automatically

---

## Setup

### 1. Install dependencies

```bash
# From repo root
python -m venv venv && source venv/bin/activate
pip install -r detection/requirements.txt
```

### 2. Get model weights

`models/best.pt` is not in git. Pull it from GCS:

```bash
gsutil cp gs://YOUR_BUCKET/models/best.pt models/best.pt
```

### 3. Configure environment

Copy the template and fill in your values:

```bash
cp detection/.env.example detection/.env
```

Required variables:

| Variable            | Description                                              |
| ------------------- | -------------------------------------------------------- |
| `TAGGER_ID`         | Your photographer/tagger ID for the API                  |
| `ADD_IMAGE_URL`     | API endpoint for posting bib detections                  |
| `IMAGE_BASE_URL`    | Base domain for image hosting (no trailing slash)        |
| `GET_ALBUMS_URL`    | API endpoint to list albums                              |
| `GET_IMAGE_LIST_URL`| API endpoint to list images in an album                  |

Optional variables:

| Variable  | Default | Description                                      |
| --------- | ------- | ------------------------------------------------ |
| `DEVICE`  | `cpu`   | `cpu`, `mps` (Apple Silicon), or `cuda` (NVIDIA) |
| `WORKERS` | `16`    | Number of concurrent threads                     |

---

## Running Detection

```bash
# From repo root
cd detection && python run.py
```

The script fetches the album list from `GET_ALBUMS_URL`, iterates the images in each album, and appends results to `bib_results.csv`.

If interrupted, re-running the same command resumes from the last processed image - no images are processed twice.

---

## Retrying Failed Images

Some images may return no detections on the first pass (network timeouts, low-contrast bibs, unusual lighting). Run the retry script after the main run completes:

```bash
cd detection && python retry_failed.py
```

This reads `bib_results.csv`, identifies images with empty bib columns, and reruns the detector on those images only.

---

## Configuration Reference (`config.py`)

| Setting               | Default          | Env var   | Description                                    |
| --------------------- | ---------------- | --------- | ---------------------------------------------- |
| `MODEL_PATH`          | `models/best.pt` | -         | Path to trained weights                        |
| `WORKERS`             | `16`             | `WORKERS` | Concurrent image threads                       |
| `CONF_THRESHOLD`      | `0.25`           | -         | YOLO person detection minimum confidence       |
| `OCR_CONF_THRESHOLD`  | `0.3`            | -         | EasyOCR minimum confidence to accept a reading |
| `IMAGE_SIZE`          | `320`            | -         | YOLO inference resolution (lower = faster)     |
| `DEVICE`              | `cpu`            | `DEVICE`  | Inference device                               |
| `IMAGE_FETCH_TIMEOUT` | `10`             | -         | HTTP timeout per image in seconds              |

---

## Running on GCP

A Spot VM is the cheapest option - spin up, run, delete.

```bash
gcloud compute instances create bib-detector \
  --machine-type=e2-standard-4 \
  --provisioning-model=SPOT \
  --zone=europe-north1-a \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --scopes=storage-ro

gcloud compute ssh bib-detector --zone=europe-north1-a
```

On the VM:

```bash
git clone https://github.com/YOUR_USERNAME/bib-detection-service.git && cd bib-detection-service
gsutil cp gs://YOUR_BUCKET/models/best.pt models/best.pt
python -m venv venv && source venv/bin/activate
pip install -r detection/requirements.txt

# Create .env with your credentials
cp detection/.env.example detection/.env
# Edit detection/.env and fill in TAGGER_ID, ADD_IMAGE_URL, IMAGE_BASE_URL,
# GET_ALBUMS_URL, GET_IMAGE_LIST_URL

cd detection && python run.py
```

**Delete the VM when done - Spot VMs continue billing while running:**

```bash
gcloud compute instances delete bib-detector --zone=europe-north1-a
```

---

## Files

| File               | Purpose                                             |
| ------------------ | --------------------------------------------------- |
| `run.py`           | Entry point - processes all albums                  |
| `retry_failed.py`  | Re-runs images with no detections                   |
| `detector.py`      | `BibDetector` class (YOLO + EasyOCR pipeline)       |
| `utils.py`         | HTTP session, image fetch, API post, CSV checkpoint |
| `config.py`        | All configuration constants                         |
| `.env.example`     | Template for required environment variables         |
| `requirements.txt` | Python dependencies                                 |
