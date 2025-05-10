import pdfplumber
import fitz
import camelot
from pdf2image import convert_from_path
import pytesseract
import time
import logging
import os
import re
from config import Config
from app.processors.ocr_processor import OCRProcessor
from app.processors.table_processor import TableProcessor

logger = logging.getLogger(__name__)


class PdfReader:
    def __init__(self):
        """Initialize the PDF Reader with specialized processors."""
        self.ocr_processor = OCRProcessor()
        self.table_processor = TableProcessor()

    def determine_extraction_method(self, page):
        """Determine the best extraction method for a page."""
        text = page.get_text("text")
        word_count = len(text.split())

        if word_count < 5:  # Almost no text detected
            return "ocr"
        elif word_count < 20:  # Some text but likely incomplete
            return "hybrid"
        else:
            return "text"

    def process_page(self, pdf_path, page_num):
        # Existing OCR processing code
        text, table_data, ocr_time, confidence = self.ocr_processor.process_page(pdf_path, page_num)

        # Log raw OCR text
        logger.debug(f"Raw OCR Text for Page {page_num + 1}: {text[:300]}")

        # Clean OCR text
        cleaned_text = self.ocr_processor.clean_ocr_text(text)

        # Log cleaned OCR text
        logger.debug(f"Cleaned OCR Text for Page {page_num + 1}: {cleaned_text[:300]}")

        return cleaned_text, table_data, ocr_time, confidence

    def extract_with_hybrid_approach(self, pdf_path, page_num):
        """Extract text using multiple methods and combine the best results."""
        try:
            # Get text using native extraction (PyMuPDF)
            with fitz.open(pdf_path) as pdf:
                page = pdf[page_num]
                text_native = page.get_text("text")

            # Get text using OCR - using a safer approach with array indexing
            ocr_result = self.ocr_processor.process_page(pdf_path, page_num)
            text_ocr = ""
            if ocr_result and isinstance(ocr_result, (list, tuple)) and len(ocr_result) > 0:
                text_ocr = ocr_result[0] or ""

            # Get text using pdfplumber (good for certain structures)
            plumber_result = self.table_processor.process_with_pdfplumber(pdf_path, page_num)
            text_plumber = ""
            table_data = None
            if plumber_result and isinstance(plumber_result, (list, tuple)):
                if len(plumber_result) > 0:
                    text_plumber = plumber_result[0] or ""
                if len(plumber_result) > 1:
                    table_data = plumber_result[1]

            # Combine results using the best parts of each
            combined_text = self.combine_extraction_results(text_native or "", text_ocr or "")
            if not combined_text and text_plumber:
                combined_text = text_plumber

            # Process any detected tables if none found yet
            if not table_data:
                camelot_result = self.table_processor.process_with_camelot(pdf_path, page_num)
                if camelot_result and isinstance(camelot_result, (list, tuple)) and len(camelot_result) > 0:
                    table_data = camelot_result[0]

            return combined_text, table_data
        except Exception as e:
            import traceback
            logger.error(f"Error in hybrid extraction for page {page_num + 1}: {str(e)}")
            logger.error(traceback.format_exc())
            return "", None

    def combine_extraction_results(self, text_native, text_ocr):
        """Combine results from native extraction and OCR."""
        # If one is empty, return the other
        if not text_native:
            return text_ocr
        if not text_ocr:
            return text_native

        # If both have content, use the longer one as it likely has more information
        if len(text_ocr) > len(text_native) * 1.5:  # OCR found significantly more text
            return text_ocr
        return text_native  # Default to native extraction as it's usually more accurate

    def is_scanned_page(self, page):
        """Check if a page is scanned by detecting if it has selectable text."""
        text = page.get_text("text")
        return not bool(text.strip())  # If no text, it's likely scanned

    def has_table_structure(self, page_text):
        """Detect if a page contains table-like structures."""
        lines = page_text.split('\n')

        # Skip pages with too few lines
        if len(lines) < 3:
            return False

        # Count lines with consistent spacing patterns (potential table rows)
        table_pattern_count = 0
        space_pattern = []

        for line in lines:
            # Look for lines with multiple whitespace clusters
            spaces = [match.start() for match in re.finditer(r'\s{2,}', line)]
            if len(spaces) >= 2:
                if not space_pattern:
                    space_pattern = spaces
                    table_pattern_count += 1
                else:
                    # If the spacing pattern is similar to previous lines, likely a table
                    matches = sum(1 for pos in spaces if any(abs(pos - p) <= 3 for p in space_pattern))
                    if matches >= len(spaces) * 0.5:  # 50% of positions match
                        table_pattern_count += 1

        # If at least 3 lines follow a similar spacing pattern, likely a table
        return table_pattern_count >= 3

    def process_pdf(self, pdf_path, output_file=None):
        """Process each page dynamically and write results to a file or return results."""
        results = []
        start_time = time.time()

        try:
            with fitz.open(pdf_path) as pdf:
                for page_num in range(len(pdf)):
                    page_result = {
                        "page_number": page_num + 1,
                        "extraction_method": None,
                        "text": None,
                        "table_data": None,
                        "processing_time": None
                    }

                    page_start_time = time.time()
                    page = pdf[page_num]

                    # Use the method to determine extraction approach
                    extraction_method = self.determine_extraction_method(page)

                    if extraction_method == "ocr":
                        # Use OCR only
                        logger.info(f"Processing page {page_num + 1} with OCR only")
                        ocr_result = self.ocr_processor.process_page(pdf_path, page_num)
                        text = ocr_result[0] if ocr_result and len(ocr_result) > 0 else ""
                        confidence_score = ocr_result[3] if ocr_result and len(ocr_result) > 3 else None

                        page_result["extraction_method"] = "ocr"
                        page_result["text"] = text
                        page_result["confidence_score"] = confidence_score

                    elif extraction_method == "hybrid":
                        # Use both methods and combine results
                        logger.info(f"Processing page {page_num + 1} with hybrid approach")
                        try:
                            combined_text, table_data = self.extract_with_hybrid_approach(pdf_path, page_num)

                            page_result["text"] = combined_text
                            page_result["extraction_method"] = "hybrid"
                            page_result["table_data"] = table_data
                        except Exception as e:
                            logger.error(f"Error in hybrid approach for page {page_num + 1}: {str(e)}")
                            # Fallback to basic text extraction
                            page_result["text"] = page.get_text()
                            page_result["extraction_method"] = "text_fallback"

                    else:
                        # Use native text extraction
                        logger.info(f"Processing page {page_num + 1} with native text extraction")
                        plumber_result = self.table_processor.process_with_pdfplumber(pdf_path, page_num)

                        if plumber_result and len(plumber_result) > 0:
                            text = plumber_result[0]
                            if len(plumber_result) > 1:
                                table_data = plumber_result[1]
                            else:
                                table_data = None
                        else:
                            text = page.get_text()
                            table_data = None

                        # Try Camelot for structured tables
                        camelot_result = self.table_processor.process_with_camelot(pdf_path, page_num)
                        camelot_data = camelot_result[0] if camelot_result and len(camelot_result) > 0 else None

                        page_result["text"] = text
                        page_result["extraction_method"] = "text"
                        page_result["table_data"] = camelot_data if camelot_data else table_data

                    page_result["processing_time"] = time.time() - page_start_time
                    results.append(page_result)

            total_processing_time = time.time() - start_time

            return {
                "results": results,
                "total_pages": len(results),
                "total_processing_time": total_processing_time
            }

        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return {
                "results": [],
                "total_pages": 0,
                "total_processing_time": time.time() - start_time,
                "error": str(e)
            }