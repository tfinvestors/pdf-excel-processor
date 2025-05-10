# app/core/config.py
import os
import platform
from pathlib import Path


class Settings:
    # Tesseract path based on platform
    if os.name == 'nt':  # Windows
        TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    elif 'STREAMLIT_SHARING' in os.environ:
        # Streamlit Cloud - tesseract installed via packages.txt
        TESSERACT_PATH = '/usr/bin/tesseract'
    else:
        # Default for Linux/Mac
        TESSERACT_PATH = '/usr/bin/tesseract'

    # Poppler path for pdf2image
    if os.name == 'nt':  # Windows
        POPPLER_PATH = r'C:\poppler-24.08.0\Library\bin'
    elif 'STREAMLIT_SHARING' in os.environ:
        # Streamlit Cloud - poppler installed via packages.txt
        POPPLER_PATH = None  # Use system path
    else:
        # Default for Linux/Mac
        POPPLER_PATH = None

    # Other settings
    DEBUG_MODE = False
    USE_HYBRID_EXTRACTION = True
    EXTRACT_TABLES = True
    OCR_DPI = 300


settings = Settings()