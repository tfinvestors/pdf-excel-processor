# config.py
import os


class Config:
    # PDF Text Extraction API Configuration
    PDF_EXTRACTION_API_URL = os.environ.get(
        'PDF_EXTRACTION_API_URL',
        'http://localhost:8000/extract'  # Appending '/extract' as the specific endpoint
    )

    # Fallback and Debugging Options
    DEBUG_MODE = os.environ.get('DEBUG_MODE', 'False').lower() == 'true'

    # Poppler Path (for PDF to image conversion)
    POPPLER_PATH = os.environ.get(
        'POPPLER_PATH',
        r'C:\poppler-24.08.0\Library\bin'  # Default path, can be overridden by environment variable
    )

    # ML Model Configuration
    USE_ML_MODEL = os.environ.get('USE_ML_MODEL', 'True').lower() == 'true'

    # API Call Configuration
    MAX_API_RETRIES = int(os.environ.get('MAX_API_RETRIES', 3))
    API_TIMEOUT = int(os.environ.get('API_TIMEOUT', 60))