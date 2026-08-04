# Configuration constants for the Bib Detector project

# API endpoints - set in detection/.env or as environment variables
import os
from dotenv import load_dotenv
_env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(_env_path)

ADD_IMAGE_URL      = os.getenv("ADD_IMAGE_URL", "")
TAGGER_ID          = os.getenv("TAGGER_ID", "")
IMAGE_BASE_URL     = os.getenv("IMAGE_BASE_URL", "").rstrip("/")
GET_ALBUMS_URL     = os.getenv("GET_ALBUMS_URL", "")
GET_IMAGE_LIST_URL = os.getenv("GET_IMAGE_LIST_URL", "")

# Path to the trained model weights
MODEL_PATH = "models/best.pt"

# Image fetch settings
IMAGE_FETCH_TIMEOUT = 10  # seconds

# YOLO inference
IMAGE_SIZE       = 320
CONF_THRESHOLD   = 0.25   # YOLO bbox confidence threshold

# OCR
OCR_CONF_THRESHOLD = 0.3  # EasyOCR minimum confidence to accept a result

# Output
OUTPUT_CSV = "bib_results.csv"

# Device: "mps" on Apple Silicon, "cuda" on GCP GPU, "cpu" on GCP CPU VM
DEVICE = os.getenv("DEVICE", "cpu")

# Concurrent image-fetch + processing threads
# 16 is safe on Apple Silicon; 32 works well on GCP e2-standard-4
WORKERS = int(os.getenv("WORKERS", "16"))
