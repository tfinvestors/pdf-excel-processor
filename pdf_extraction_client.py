# pdf_extraction_client.py
import requests
import time


class PDFExtractionClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.api_prefix = "/api/v1"

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

        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        result = response.json()

        if wait and result["status"] == "queued":
            # If the server didn't wait but we want to, poll until completion
            document_id = result["document_id"]
            return self._wait_for_completion(document_id, timeout)

        return result

    def get_extraction_status(self, document_id):
        """Get the status of an extraction job."""
        endpoint = f"{self.base_url}{self.api_prefix}/extract/{document_id}/status"
        response = requests.get(endpoint)
        response.raise_for_status()
        return response.json()

    def get_extraction_text(self, document_id):
        """Get the extracted text for a document."""
        endpoint = f"{self.base_url}{self.api_prefix}/extract/{document_id}/text"
        response = requests.get(endpoint)
        response.raise_for_status()
        return response.json()

    def _wait_for_completion(self, document_id, timeout=300):
        """Poll until document processing is complete or timeout is reached."""
        start_time = time.time()
        poll_interval = 2  # seconds

        while (time.time() - start_time) < timeout:
            status = self.get_extraction_status(document_id)
            if status["status"] in ["completed", "failed"]:
                if status["status"] == "completed":
                    return self.get_extraction_text(document_id)
                else:
                    return {"document_id": document_id, "status": "failed"}

            time.sleep(poll_interval)
            # Gradually increase polling interval
            poll_interval = min(poll_interval * 1.5, 10)

        return {"document_id": document_id, "status": "timeout"}