import os
import sys
import logging

# Suppress verbose logging from specific libraries
logging.getLogger('pdfminer').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('PyPDF2').setLevel(logging.WARNING)
logging.getLogger('pdfplumber').setLevel(logging.WARNING)

# Configure your main logging
logging.basicConfig(
    level=logging.INFO,  # or logging.ERROR if you want even less output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Ensure the project root is in Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.pdf_processor import PDFProcessor
from config import Config


def test_extraction(source):
    """
    Unified test function for both local file and URL extraction
    """
    # Validate file exists for local files
    if not source.startswith(('http://', 'https://')):
        if not os.path.exists(source):
            print(f"❌ File does not exist: {source}")
            return None

    # Initialize processor with API configuration
    pdf_processor = PDFProcessor(
        use_ml=Config.USE_ML_MODEL,
        debug_mode=True,
        poppler_path=Config.POPPLER_PATH,
        text_extraction_api_url=Config.PDF_EXTRACTION_API_URL
    )

    try:
        # Determine extraction method based on source
        if source.startswith(('http://', 'https://')):
            # URL Extraction
            print(f"\n--- URL PDF Extraction Test ---")
            print(f"Source URL: {source}")
            print(f"API URL: {Config.PDF_EXTRACTION_API_URL}")

            # Use extract_from_url for web sources
            extracted_text = pdf_processor.text_extractor.extract_from_url(source)
            extraction_method = "URL API/Local Extraction"
        else:
            # Local File Extraction
            print(f"\n--- Local PDF Extraction Test ---")
            print(f"Source File: {source}")
            print(f"API URL: {Config.PDF_EXTRACTION_API_URL}")

            # Print the actual method being used
            extracted_text = pdf_processor.text_extractor.extract_from_file(source)
            extraction_method = "Local File Extraction or API Extraction"

        # Print extraction results
        print(f"Extraction Method: {extraction_method}")
        print(f"Text Length: {len(extracted_text)} characters")
        print("\nFirst 500 characters:")
        print(extracted_text[:500])

        # Additional checks
        if not extracted_text:
            print("❌ No text extracted!")
        else:
            print("✅ Text extraction successful!")

        return extracted_text

    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        # Log the full traceback
        import traceback
        traceback.print_exc()
        return None


def main():
    # Test Sources - IMPORTANT: Replace with ACTUAL file paths or URLs
    test_sources = [
        # REPLACE with an ACTUAL PDF file path on your system
        r"C:\Users\Lenovo\Downloads\2425-21837.pdf"

        # Uncomment and replace with an actual PDF URL if testing URL extraction
        # "https://example.com/sample.pdf"
    ]

    # Test each source
    for source in test_sources:
        print("\n" + "=" * 50)
        test_extraction(source)


if __name__ == "__main__":
    main()