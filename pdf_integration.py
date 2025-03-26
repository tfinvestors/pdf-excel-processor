from pdf_extraction_client import PDFExtractionClient


def process_pdf_documents(pdf_urls):
    """Process a list of PDF URLs and get their extracted text."""
    client = PDFExtractionClient("http://pdf-extraction-service.example.com")

    results = []
    for url in pdf_urls:
        try:
            # Extract text from PDF
            extraction = client.extract_from_url(url, wait=True, timeout=600)

            if extraction["status"] == "completed" and extraction.get("text"):
                # Process the extracted text
                processed_text = process_text(extraction["text"])
                results.append({
                    "url": url,
                    "success": True,
                    "processed_text": processed_text
                })
            else:
                results.append({
                    "url": url,
                    "success": False,
                    "error": "Extraction failed or timed out"
                })
        except Exception as e:
            results.append({
                "url": url,
                "success": False,
                "error": str(e)
            })

    return results


def process_text(text):
    """Process the extracted text - implement your logic here."""
    # Your custom text processing logic
    return text