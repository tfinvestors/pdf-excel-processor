# config.py 
import os
import platform
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


class Config:
    # PDF Text Extraction API Configuration
    if 'STREAMLIT_SHARING' in os.environ:
        # For Streamlit Cloud, use a public API endpoint or disable API extraction
        PDF_EXTRACTION_API_URL = os.environ.get(
            'PDF_EXTRACTION_API_URL',
            None  # Set to None to force local extraction in cloud
        )
    else:
        # For local development
        PDF_EXTRACTION_API_URL = os.environ.get(
            'PDF_EXTRACTION_API_URL',
            'http://localhost:8000/api/v1/documents/upload'
        )

    # Fallback and Debugging Options
    DEBUG_MODE = os.environ.get('DEBUG_MODE', 'False').lower() == 'true'

    # Poppler Path (for PDF to image conversion)
    # Use platform-specific default paths
    POPPLER_PATH = os.environ.get('POPPLER_PATH',
                                  # Default paths based on operating system
                                  r'C:\poppler-24.08.0\Library\bin' if platform.system() == 'Windows' else
                                  # On Linux (Streamlit Cloud), poppler is installed via packages.txt
                                  # and available on the system path
                                  ''
                                  )

    # ML Model Configuration
    USE_ML_MODEL = os.environ.get('USE_ML_MODEL', 'True').lower() == 'true'

    # API Call Configuration
    MAX_API_RETRIES = int(os.environ.get('MAX_API_RETRIES', 3))
    API_TIMEOUT = int(os.environ.get('API_TIMEOUT', 60))

    # Define common paths for application
    APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.environ.get('DATA_DIR', os.path.join(APP_ROOT, 'data'))
    MODEL_DIR = os.environ.get('MODEL_DIR', os.path.join(APP_ROOT, 'models'))

    # Create directories if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Streamlit specific settings
    STREAMLIT_MODE = 'STREAMLIT_SHARING' in os.environ or os.environ.get('STREAMLIT_MODE', 'False').lower() == 'true'

    # OCR settings
    TESSERACT_PATH = os.environ.get('TESSERACT_PATH',
                                    # Platform specific default paths
                                    r'C:\Program Files\Tesseract-OCR\tesseract.exe' if platform.system() == 'Windows' else
                                    '/usr/bin/tesseract'  # Default path on Linux including Streamlit Cloud
                                    )