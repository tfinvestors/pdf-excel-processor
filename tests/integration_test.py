# integration_test.py
import os
import sys

# Add the parent directory to the path to import from the main project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import directly from your PDF processor
from app.pdf_processor import PDFProcessor
from app.utils.pdf_text_extractor import PDFTextExtractor


def process_pdf_documents(pdf_paths):
    """Process a list of PDF paths and get their extracted text."""
    # Create a PDF processor instance
    pdf_processor = PDFProcessor(use_ml=True)
    text_extractor = PDFTextExtractor()

    results = []
    for path in pdf_paths:
        try:
            # Extract text from PDF
            extracted_text = text_extractor.extract_from_file(path)

            if extracted_text:
                # Extract data points
                unique_id, data_points, table_data, detected_provider = pdf_processor.extract_data_points(
                    extracted_text)

                results.append({
                    "path": path,
                    "success": True,
                    "text": extracted_text,
                    "unique_id": unique_id,
                    "data_points": data_points,
                    "table_data": table_data,
                    "detected_provider": detected_provider
                })
            else:
                results.append({
                    "path": path,
                    "success": False,
                    "error": "Failed to extract text from PDF"
                })
        except Exception as e:
            results.append({
                "path": path,
                "success": False,
                "error": str(e)
            })

    return results


# Example usage
if __name__ == "__main__":
    pdf_paths = [
        r"D:/My Downloads/Bajaj Allianz General Insurance Co. Ltd..pdf",
        r"D:/My Downloads/Cera Sanitaryware Ltd.pdf"
    ]

    results = process_pdf_documents(pdf_paths)
    print(f"Processed {len(results)} PDF documents")

    for result in results:
        if result["success"]:
            print(f"Successfully processed: {result['path']}")
            print(f"Unique ID: {result['unique_id']}")
            print(f"Extracted data points: {result['data_points']}")
            # Print only the first 200 characters of text to avoid flooding the console
            print(f"Text excerpt: {result['text'][:200]}...")
        else:
            print(f"Failed to process: {result['path']}, Error: {result.get('error')}")