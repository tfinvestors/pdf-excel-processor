import os
import requests
import time
import json

def test_pdf_extraction(pdf_path):
    """
    Comprehensive PDF extraction test
    """
    base_url = 'http://localhost:8000/api/v1'
    upload_url = f'{base_url}/documents/upload'

    try:
        # Upload file
        with open(pdf_path, 'rb') as f:
            files = {'file': (os.path.basename(pdf_path), f, 'application/pdf')}
            data = {
                'process_immediately': 'true',
                'process_directly': 'true'
            }
            print(f"Uploading file: {pdf_path}")
            upload_response = requests.post(upload_url, files=files, data=data)

        print("\n=== UPLOAD RESPONSE ===")
        print(f"Status Code: {upload_response.status_code}")
        upload_result = upload_response.json()
        print(json.dumps(upload_result, indent=2))

        # Extract document ID
        document_id = upload_result.get('document_id')
        if not document_id:
            print("No document ID received")
            return

        # Poll for status
        timeout = 300  # 5 minutes
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check status
            status_response = requests.get(f'{base_url}/extract/{document_id}/status')
            status_data = status_response.json()
            print("\nStatus Check:")
            print(json.dumps(status_data, indent=2))

            if status_data['status'] == 'completed':
                break
            elif status_data['status'] == 'failed':
                print("Document processing failed")
                return

            time.sleep(2)

        # Fetch consolidated text
        text_response = requests.get(f'{base_url}/documents/{document_id}/consolidated-text')

        print("\n=== TEXT RETRIEVAL ===")
        print(f"Status Code: {text_response.status_code}")
        text_result = text_response.json()
        print("Text Response:")
        print(json.dumps(text_result, indent=2))

        # Extract and print text
        extracted_text = text_result.get('consolidated_text', '')
        print("\n=== EXTRACTED TEXT ===")
        print(f"Text Length: {len(extracted_text)}")
        print("First 500 characters:")
        # print(extracted_text[:500])
        print(extracted_text)

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Replace with your actual PDF path
    pdf_path = r"D:\My Downloads\Cera Sanitaryware Ltd.pdf"
    test_pdf_extraction(pdf_path)