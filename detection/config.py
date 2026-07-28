# Configuration constants for the Bib Detector project

# API endpoints — set in detection/.env or as environment variables
import os
from dotenv import load_dotenv
_env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(_env_path)

ADD_IMAGE_URL = os.getenv("ADD_IMAGE_URL", "")
GET_IMAGE_URL = os.getenv("GET_IMAGE_URL", "")

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

albums = [
    {"albumNr": 1,  "name": "start",            "startImageId": 1000,  "noOfImages": 123},
    {"albumNr": 2,  "name": "centrum",           "startImageId": 2000,  "noOfImages": 34},
    {"albumNr": 5,  "name": "ramlosa",           "startImageId": 6000,  "noOfImages": 1377},
    {"albumNr": 6,  "name": "jordbodalen",       "startImageId": 8000,  "noOfImages": 30},
    {"albumNr": 7,  "name": "faltabacken",       "startImageId": 9000,  "noOfImages": 145},
    {"albumNr": 8,  "name": "jonkopingsgatan",   "startImageId": 10000, "noOfImages": 38},
    {"albumNr": 9,  "name": "fredriksdal",       "startImageId": 11000, "noOfImages": 46},
    {"albumNr": 10, "name": "olympia",           "startImageId": 12000, "noOfImages": 28},
    {"albumNr": 11, "name": "karnan",            "startImageId": 13000, "noOfImages": 1660},
    {"albumNr": 12, "name": "tinkarp",           "startImageId": 15000, "noOfImages": 4000},
    {"albumNr": 13, "name": "strandpromenaden",  "startImageId": 20000, "noOfImages": 2879},
    {"albumNr": 14, "name": "finish",            "startImageId": 23000, "noOfImages": 9530},
    {"albumNr": 15, "name": "dkr",              "startImageId": 33000, "noOfImages": 547},
    {"albumNr": 16, "name": "groningen",         "startImageId": 34000, "noOfImages": 170},
]
