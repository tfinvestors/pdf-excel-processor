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
    Manual API test to diagnose connection issues
    """
    try:
        upload_url = 'http://localhost:8000/api/v1/documents/upload'

        with open(pdf_path, 'rb') as f:
            files = {'file': (os.path.basename(pdf_path), f, 'application/pdf')}
            data = {
                'process_immediately': 'true',
                'process_directly': 'true'
            }

            print("Attempting direct API upload...")
            try:
                response = requests.post(upload_url, files=files, data=data, timeout=5)

                print("\n=== UPLOAD RESPONSE ===")
                print(f"Status Code: {response.status_code}")
                print("Response Headers:")
                for key, value in response.headers.items():
                    print(f"  {key}: {value}")

                try:
                    print("\nResponse JSON:")
                    print(response.json())
                except ValueError:
                    print("Response is not JSON")
                    print("Response Text:")
                    print(response.text)

            except requests.ConnectionError as conn_error:
                print(f"Connection Error: {conn_error}")
                print("Possible reasons:")
                print("1. API service not running")
                print("2. Incorrect port")
                print("3. Firewall blocking connection")
                print("4. Service not started")

            except requests.Timeout:
                print("Request timed out")

            except Exception as e:
                print(f"Unexpected error: {e}")
                import traceback
                traceback.print_exc()

    except Exception as e:
        print(f"Error in manual API test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Replace with your actual PDF path
    pdf_path = r"D:\My Downloads\Cera Sanitaryware Ltd.pdf"

    print("=== Manual API Test ===")
    manual_api_test(pdf_path)

    print("\n=== PDF Extraction Test ===")
    test_pdf_extraction(pdf_path)