import requests
import time
import traceback


def extract_local_pdf_text(file_path, base_url="http://localhost:8000", timeout=300):
    """
    Extract text from a local PDF file by uploading it to the service.

    Args:
        file_path (str): Path to the local PDF file
        base_url (str): Base URL of the PDF extraction service
        timeout (int): Maximum time to wait for processing

    Returns:
        str or None: Extracted text, or None if extraction fails
    """
    try:
        # Prepare the file for upload
        with open(file_path, 'rb') as file:
            files = {'file': (file_path.split('\\')[-1], file, 'application/pdf')}

            print("Attempting to upload file...")
            # Send file to upload endpoint
            upload_response = requests.post(
                f"{base_url}/api/v1/documents/upload",
                files=files,
                data={
                    'process_immediately': 'true',
                    'process_directly': 'true'  # Use direct processing
                }
            )

            # Print full response details
            print(f"Upload Response Status: {upload_response.status_code}")
            print(f"Upload Response Text: {upload_response.text}")

            # Check upload response
            upload_response.raise_for_status()

            # Parse upload response
            upload_result = upload_response.json()
            document_id = upload_result.get('document_id')

            print(f"Received Document ID: {document_id}")

            if not document_id:
                print("No document ID received")
                return None

            # Poll for document status
            start_time = time.time()
            while time.time() - start_time < timeout:
                # Check document status
                status_response = requests.get(
                    f"{base_url}/api/v1/extract/{document_id}/status"
                )

                print(f"Status Check Response: {status_response.text}")

                status_data = status_response.json()

                if status_data['status'] == 'completed':
                    break
                elif status_data['status'] == 'failed':
                    print("Document processing failed")
                    return None

                # Wait before next check
                time.sleep(2)

            # Fetch consolidated text
            print("Fetching consolidated text...")
            text_response = requests.get(
                f"{base_url}/api/v1/documents/{document_id}/consolidated-text"
            )

            # Print full response details
            print(f"Text Response Status: {text_response.status_code}")
            print(f"Text Response Text: {text_response.text}")

            text_response.raise_for_status()

            text_result = text_response.json()
            return text_result.get('consolidated_text')

    except Exception as e:
        print("Full Error Details:")
        print(traceback.format_exc())
        print(f"Error extracting text: {str(e)}")
        return None


# Example usage
local_pdf_path = r"C:\Users\Lenovo\Downloads\2425-21837.pdf"
extracted_text = extract_local_pdf_text(local_pdf_path)

if extracted_text:
    print("Extracted Text:")
    print(extracted_text)
    print(f"\nText Length: {len(extracted_text)} characters")
else:
    print("Failed to extract text")