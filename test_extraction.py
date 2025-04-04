import os
import sys
import requests
import logging

# Ensure the project root is in Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
from app.pdf_processor import PDFProcessor
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_pdf_extraction(pdf_path):
    """
    Comprehensive PDF extraction test with robust error handling
    """
    try:
        # Initialize processor with API configuration
        pdf_processor = PDFProcessor(
            use_ml=getattr(Config, 'USE_ML_MODEL', True),
            debug_mode=True,
            poppler_path=getattr(Config, 'POPPLER_PATH', None),
            text_extraction_api_url=getattr(Config, 'PDF_EXTRACTION_API_URL',
                                            'http://localhost:8000/api/v1/documents/upload')
        )

        # Attempt text extraction
        try:
            extracted_text = pdf_processor.text_extractor.extract_from_file(pdf_path)

            if extracted_text:
                print("✅ Extraction Successful:")
                print(f"Text Length: {len(extracted_text)} characters")
                print("\nFirst 500 characters:")
                # print(extracted_text[:500])
                print(extracted_text)

            else:
                print("❌ No text extracted. Fell back to local extraction.")

        except Exception as extraction_error:
            print(f"Extraction Error: {extraction_error}")
            import traceback
            traceback.print_exc()

    except Exception as setup_error:
        print(f"Setup Error: {setup_error}")
        import traceback
        traceback.print_exc()


def manual_api_test(pdf_path):
    """
    Comprehensive API diagnostic test
    """
    base_url = 'http://localhost:8000/api/v1'

    try:
        # Read file content
        with open(pdf_path, 'rb') as f:
            file_content = f.read()

        # Upload file
        upload_url = f'{base_url}/documents/upload'
        files = {'file': (os.path.basename(pdf_path), file_content, 'application/pdf')}
        data = {
            'process_immediately': 'true',
            'process_directly': 'true'
        }

        print("Attempting direct API upload...")
        upload_response = requests.post(upload_url, files=files, data=data, timeout=10)

        print("\n=== UPLOAD RESPONSE ===")
        print(f"Status Code: {upload_response.status_code}")
        upload_result = upload_response.json()
        document_id = upload_result.get('document_id')

        print("\n=== DOCUMENT DETAILS ===")
        print(f"Document ID: {document_id}")

        # Check status
        status_url = f'{base_url}/extract/{document_id}/status'
        status_response = requests.get(status_url, timeout=10)
        status_data = status_response.json()

        print("\n=== STATUS CHECK ===")
        print(f"Status: {status_data}")

        # Retrieve text
        if status_data.get('status') == 'completed':
            text_url = f'{base_url}/documents/{document_id}/consolidated-text'
            text_response = requests.get(text_url, timeout=10)
            text_result = text_response.json()

            print("\n=== EXTRACTED TEXT ===")
            print(f"Text Length: {len(text_result.get('consolidated_text', ''))}")
            print("First 500 characters:")
            print(text_result.get('consolidated_text', '')[:500])

    except Exception as e:
        print(f"Error during API test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Replace with your actual PDF path
    pdf_path = r"D:\My Downloads\Cera Sanitaryware Ltd.pdf"

    print("=== Manual API Test ===")
    manual_api_test(pdf_path)

    print("\n=== PDF Extraction Test ===")
    test_pdf_extraction(pdf_path)