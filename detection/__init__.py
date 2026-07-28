# This file initializes the bib_detector package.
# It can be used to define what is exported when the package is imported.

from detector import BibDetector
from utils import fetch_image
import config

__all__ = ['BibDetector', 'fetch_image']