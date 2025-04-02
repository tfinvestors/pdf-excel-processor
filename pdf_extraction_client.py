import requests
import time


class PDFExtractionClient:
    """
    Client for extracting text from PDFs using the PDF Extraction Service API.

    Args:
        base_url (str): Base URL of the PDF extraction service
        api_prefix (str, optional): API prefix path
        api_key (str, optional): API key for authentication
    """

    def __init__(self, base_url="http://localhost:8000", api_prefix="/api/v1", api_key=None):
        self.base_url = base_url.rstrip('/')
        self.api_prefix = api_prefix
        self.headers = {
            'Content-Type': 'application/json'
        }
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'

    def extract_from_url(self, pdf_url, category=None, wait=True, timeout=300):
        """
        Extract text from a PDF URL.

        Args:
            pdf_url (str): URL to the PDF file
            category (str, optional): Document category for specialized processing
            wait (bool): Whether to wait for processing to complete
            timeout (int): Max time to wait in seconds if wait=True

        Returns:
            dict: Extraction result or job information
        """
        endpoint = f"{self.base_url}{self.api_prefix}/extract/url"
        payload = {
            "url": pdf_url,
            "category": category,
            "wait_for_result": wait
        }

        try:
            response = requests.post(endpoint, json=payload, headers=self.headers)
            response.raise_for_status()
            result = response.json()

            if wait and result["status"] == "queued":
                # If the server didn't wait but we want to, poll until completion
                document_id = result["document_id"]
                return self._wait_for_completion(document_id, timeout)

            return result

        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def get_extraction_status(self, document_id):
        """
        Get the status of an extraction job.

        Args:
            document_id (int): ID of the document to check

        Returns:
            dict: Status information of the document
        """
        try:
            endpoint = f"{self.base_url}{self.api_prefix}/extract/{document_id}/status"
            response = requests.get(endpoint, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "document_id": document_id,
                "status": "error",
                "error": str(e)
            }

    def get_extraction_text(self, document_id):
        """
        Get the extracted text for a document.

        Args:
            document_id (int): ID of the document to retrieve text

        Returns:
            dict: Extracted text information
        """
        try:
            endpoint = f"{self.base_url}{self.api_prefix}/extract/{document_id}/text"
            response = requests.get(endpoint, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "document_id": document_id,
                "status": "error",
                "error": str(e)
            }

    def _wait_for_completion(self, document_id, timeout=300):
        """
        Poll until document processing is complete or timeout is reached.

        Args:
            document_id (int): ID of the document to wait for
            timeout (int): Maximum time to wait in seconds

        Returns:
            dict: Final extraction result or status
        """
        start_time = time.time()
        poll_interval = 2  # seconds

        while (time.time() - start_time) < timeout:
            try:
                status = self.get_extraction_status(document_id)

                # Handle different possible status scenarios
                if status.get("status") in ["completed", "failed", "error"]:
                    if status["status"] == "completed":
                        # Fetch and return the text
                        text_result = self.get_extraction_text(document_id)
                        return {
                            "document_id": document_id,
                            "status": "completed",
                            "text": text_result.get("text")
                        }
                    else:
                        return {
                            "document_id": document_id,
                            "status": status["status"],
                            "error": status.get("error", "Processing failed")
                        }

                # Wait before next poll with exponential backoff
                time.sleep(poll_interval)
                poll_interval = min(poll_interval * 1.5, 10)

            except Exception as e:
                # Handle any unexpected errors during polling
                return {
                    "document_id": document_id,
                    "status": "error",
                    "error": str(e)
                }

        # Timeout occurred
        return {
            "document_id": document_id,
            "status": "timeout",
            "error": "Processing timed out"
        }


def extract_pdf_text(pdf_url, base_url, category=None, timeout=600):
    """
    Convenience function to extract text from a PDF URL.

    Args:
        pdf_url (str): URL of the PDF to extract text from
        base_url (str): Base URL of the PDF extraction service
        category (str, optional): Category of the document
        timeout (int, optional): Maximum time to wait for extraction

    Returns:
        str or None: Extracted text, or None if extraction fails
    """
    client = PDFExtractionClient(base_url)

    try:
        result = client.extract_from_url(
            pdf_url,
            category=category,
            wait=True,
            timeout=timeout
        )

        if result['status'] == 'completed':
            return result['text']
        else:
            print(f"Text extraction failed: {result.get('error', 'Unknown error')}")
            return None

    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        return None


# Example usage demonstration
if __name__ == "__main__":
    # Example of how to use the client
    BASE_URL = "http://localhost:8000"  # Replace with your actual service URL
    PDF_URL = "https://example.com/sample.pdf"

    # Direct text extraction
    extracted_text = extract_pdf_text(PDF_URL, BASE_URL, category="financial_report")
    if extracted_text:
        print("Extracted Text:")
        print(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)

    # Detailed client usage
    client = PDFExtractionClient(BASE_URL)

    # Extract with more control
    extraction_result = client.extract_from_url(
        PDF_URL,
        category="financial_report",
        wait=True,
        timeout=300
    )

    print("\nDetailed Extraction Result:")
    print(extraction_result)