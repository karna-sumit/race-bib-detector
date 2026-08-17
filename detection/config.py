# Configuration constants for the Bib Detector project

# API endpoints - set in detection/.env or as environment variables
import os
from dotenv import load_dotenv
_env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(_env_path)

ADD_IMAGE_URL      = os.getenv("ADD_IMAGE_URL", "")
ADD_IMAGES_URL     = os.getenv("ADD_IMAGES_URL", "")
TAGGER_ID          = os.getenv("TAGGER_ID", "")
IMAGE_BASE_URL     = os.getenv("IMAGE_BASE_URL", "").rstrip("/")
GET_ALBUMS_URL     = os.getenv("GET_ALBUMS_URL", "")
GET_IMAGE_LIST_URL = os.getenv("GET_IMAGE_LIST_URL", "")

# Path to the trained model weights
MODEL_PATH = "models/best.pt"

# Image fetch settings
IMAGE_FETCH_TIMEOUT = 10  # seconds

# YOLO inference
IMAGE_SIZE       = 640
CONF_THRESHOLD   = 0.25

# Skip bib detections whose bbox touches the image edge (frame-clipped = incomplete).
EDGE_MARGIN_FRAC = 0.005

# Max images per batched POST to ADD_IMAGES_URL.
POST_BATCH_SIZE = int(os.getenv("POST_BATCH_SIZE", "50"))

# OCR
OCR_ENGINE = os.getenv("OCR_ENGINE", "paddle").lower()
OCR_CONF_THRESHOLD = 0.3

# Output
OUTPUT_CSV = "bib_results.csv"

# Device: "mps" on Apple Silicon, "cuda" on GCP GPU, "cpu" on GCP CPU VM
DEVICE = os.getenv("DEVICE", "cpu")

WORKERS = int(os.getenv("WORKERS", "16"))
