import os
import sys
import logging
sys.path.insert(0, os.path.dirname(__file__))
from detector import BibDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    if not os.getenv("TAGGER_ID"):
        print("TAGGER_ID not set. Set it in detection/.env or as an environment variable.")
        return

    print("Starting Bib Detection Pipeline")
    detector = BibDetector()
    detector.batch_detect_bibs()
    print("Done.")


if __name__ == "__main__":
    main()
