"""
OCR utilities for extracting text from images
Supports both Tesseract and EasyOCR
"""

import cv2 as cv
import numpy as np
from typing import Optional
import re

# Try to import OCR libraries
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    reader = easyocr.Reader(['en'], gpu=False)
except ImportError:
    EASYOCR_AVAILABLE = False
    reader = None
    print("Warning: EasyOCR not available. Install with: pip install easyocr")

try:
    import pytesseract
    # Test if Tesseract is actually installed by checking if pytesseract can find it
    try:
        pytesseract.get_tesseract_version()
        TESSERACT_AVAILABLE = True
    except Exception:
        TESSERACT_AVAILABLE = False
        print("Warning: Tesseract executable not found. Using EasyOCR only.")
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available. Using EasyOCR only.")


def extract_text_easyocr(image: np.ndarray) -> str:
    """Extract text from image using EasyOCR."""
    if not EASYOCR_AVAILABLE:
        raise RuntimeError("EasyOCR is not available")
    
    output = reader.readtext(image)
    text = ' '.join([item[1] for item in output])
    return text.strip()


def extract_text_tesseract(image: np.ndarray) -> str:
    """Extract text from image using Tesseract OCR."""
    if not TESSERACT_AVAILABLE:
        raise RuntimeError("Tesseract is not available")
    
    text = pytesseract.image_to_string(image)
    return text.strip()


def extract_text(image: np.ndarray, method: str = 'auto') -> str:
    """Extract text from image using available OCR method."""
    if method == 'auto':
        if EASYOCR_AVAILABLE:
            return extract_text_easyocr(image)
        elif TESSERACT_AVAILABLE:
            try:
                return extract_text_tesseract(image)
            except Exception as e:
                raise RuntimeError(f"Tesseract failed: {e}. Please ensure Tesseract is installed.")
        else:
            raise RuntimeError("No OCR library available. Install easyocr or ensure Tesseract is installed.")
    elif method == 'easyocr':
        return extract_text_easyocr(image)
    elif method == 'tesseract':
        if not TESSERACT_AVAILABLE:
            raise RuntimeError("Tesseract not available. Use EasyOCR or install Tesseract.")
        return extract_text_tesseract(image)
    else:
        raise ValueError(f"Unknown OCR method: {method}")


def extract_header_text(image: np.ndarray, header_coords: list) -> str:
    """Extract text from image header region."""
    header_region = image[header_coords[0]:header_coords[1], 
                         header_coords[2]:header_coords[3]]
    return extract_text(header_region)


def clean_header_text(text: str) -> str:
    """Clean extracted header text."""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


def match_header(extracted_text: str, expected_headers: list) -> Optional[str]:
    """Match extracted text against expected headers."""
    extracted_text = clean_header_text(extracted_text)
    
    # Try exact match first
    for header in expected_headers:
        if header in extracted_text:
            return header
    
    # Try fuzzy match based on key words
    extracted_words = set(extracted_text.lower().split())
    
    for header in expected_headers:
        header_words = set(header.lower().split())
        common_words = extracted_words.intersection(header_words)
        if len(common_words) >= len(header_words) * 0.6:  # 60% match
            return header
    
    return None
