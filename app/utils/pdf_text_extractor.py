# app/utils/pdf_text_extractor.py
import os
import re
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
from pdf2image import convert_from_path, convert_from_bytes
import pytesseract
import logging
import tempfile
import io
import requests
import concurrent.futures
from PIL import Image, ImageEnhance
import signal
from contextlib import contextmanager
import platform
import time
import json
import traceback

try:
    import cv2
    import numpy as np

    HAS_OPENCV = True
except ImportError:
    # Create mock CV2 for compatibility
    class MockCV2:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

        # Mock essential functions used in your code
        def fastNlMeansDenoising(self, *args, **kwargs):
            return args[0]  # Return the input image unchanged

        def adaptiveThreshold(self, src, maxValue, adaptiveMethod, thresholdType, blockSize, C):
            return src  # Return the input image unchanged

        # Add constants used in your code
        ADAPTIVE_THRESH_GAUSSIAN_C = 0
        THRESH_BINARY = 0


    cv2 = MockCV2()
    import numpy as np  # Still need numpy

    HAS_OPENCV = False
    logger.warning("OpenCV (cv2) import failed. Some image processing features will be limited.")


@contextmanager
def timeout(seconds):
    """Context manager for timing out operations."""
    # Only works on Unix-based systems (not Windows)
    if platform.system() == 'Windows':
        yield
        return

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    # Set the timeout handler
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)

# OCR Corrections Dictionary
OCR_CORRECTIONS = {
    "0rienta1": "Oriental",
    "0riental": "Oriental",
    "0rient": "Orient",
    "lnsurance": "Insurance",
    "1nsurance": "Insurance",
    "lndia": "India",
    "1ndia": "India",
    "c1aim": "claim",
    "pol1cy": "policy",
    "po1icy": "policy",
    "c1ient": "client",
    "rece1pt": "receipt",
    "remittance": "remittance",
    "HSBC;": "HSBC",
    "HS8C": "HSBC",
    "H5BC": "HSBC",
    "UNlTED": "UNITED",
    "UN1TED": "UNITED",
    "lnvoice": "Invoice",
    "lnv": "Inv",
    "TD5": "TDS",
    "td5": "tds",
    "5ECT0R": "SECTOR",
    "5ect0r": "Sector",
    "IN5URANCE": "INSURANCE",
    "In5urance": "Insurance",
    "5URVEY0R5": "SURVEYORS",
    "5urvey0r5": "Surveyors",
    "L055": "LOSS",
    "l055": "loss",
    "A55E550R5": "ASSESSORS",
    "a55e550r5": "assessors",
    "0C": "OC"
}

# Configure logging
logger = logging.getLogger("pdf_extractor")


class PDFTextExtractor:
    """
    Unified PDF text extraction class used by both training and production code.
    This ensures consistent text extraction between model training and inference.
    """

    def __init__(self, poppler_path=None, debug_mode=False):
        """Initialize the PDF text extractor."""
        self.poppler_path = poppler_path
        self.debug_mode = debug_mode
        self.debug_dir = None

        # Set platform-specific paths
        if os.name == 'nt':  # Windows
            self.poppler_path = poppler_path or r'C:\poppler-24.08.0\Library\bin'
        elif 'STREAMLIT_SHARING' in os.environ:
            # On Streamlit Cloud, poppler is installed via packages.txt
            self.poppler_path = None  # Use system path
        else:
            # Other Linux/Mac systems
            self.poppler_path = poppler_path

        # Set Tesseract path based on platform
        try:
            if os.name == 'nt':  # Windows
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            elif 'STREAMLIT_SHARING' in os.environ:
                # We're on Streamlit Cloud
                pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

            # Verify Tesseract is available
            tesseract_version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {tesseract_version}")
        except Exception as e:
            logger.warning(f"Could not set Tesseract path: {str(e)}")
            logger.warning("OCR functionality may be limited")

        # Create debug directory if needed
        if self.debug_mode:
            self.debug_dir = os.path.join(os.path.expanduser("~"), "Downloads", "PDF_Debug")
            os.makedirs(self.debug_dir, exist_ok=True)
            logger.info(f"Debug mode enabled. Outputs will be saved to {self.debug_dir}")

    # 1. Public methods first
    def _cleanup_memory(self):
        """Force garbage collection to free memory."""
        import gc
        gc.collect()

    # 2. Main internal extraction method
    def _extract_text_from_pdf(self, pdf_path=None, pdf_content=None, source_type="file", progress_callback=None):
        """Enhanced text extraction with better error recovery and progress tracking."""

        all_text = []
        doc = None  # Initialize doc variable

        try:
            # Import the specialized processors
            from app.processors.pdf_reader import PdfReader
            from app.processors.ocr_processor import OCRProcessor
            from app.processors.table_processor import TableProcessor

            # Initialize processors
            pdf_reader = PdfReader()
            ocr_processor = OCRProcessor()
            table_processor = TableProcessor()

            # Open PDF based on source type
            if source_type == "file":
                if not os.path.exists(pdf_path):
                    logger.error(f"PDF file not found: {pdf_path}")
                    return ""
                doc = fitz.open(pdf_path)
            else:
                doc = fitz.open(stream=pdf_content, filetype="pdf")

            # Process each page with error recovery
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    page_text = ""

                    # Determine the best extraction method for the page
                    extraction_method = pdf_reader.determine_extraction_method(page)

                    if extraction_method == "ocr":
                        # OCR only
                        logger.info(f"Processing page {page_num + 1} with OCR only")
                        ocr_result = ocr_processor.process_page(pdf_path if pdf_path else None, page_num)
                        if ocr_result and isinstance(ocr_result, (list, tuple)) and len(ocr_result) > 0:
                            page_text = ocr_result[0] or ""

                    elif extraction_method == "hybrid":
                        # Hybrid approach
                        logger.info(f"Processing page {page_num + 1} with hybrid approach")
                        combined_text, table_data = pdf_reader.extract_with_hybrid_approach(
                            pdf_path if pdf_path else None, page_num
                        )
                        page_text = combined_text

                        # Add table data if found
                        if table_data:
                            page_text += "\n\n[TABLE DATA]\n" + table_data

                    else:
                        # Native extraction
                        logger.info(f"Processing page {page_num + 1} with native text extraction")
                        text = page.get_text("text")
                        word_count = len(text.split())

                        if word_count < 5:
                            # Fallback to OCR
                            ocr_result = ocr_processor.process_page(pdf_path if pdf_path else None, page_num)
                            if ocr_result and len(ocr_result) > 0:
                                page_text = ocr_result[0] or ""
                        else:
                            page_text = text

                    # Check for tables if we haven't found any yet
                    if pdf_reader.has_table_structure(page_text):
                        # Try camelot first
                        camelot_result = table_processor.process_with_camelot(
                            pdf_path if pdf_path else None, page_num
                        )
                        if camelot_result and camelot_result[0]:
                            page_text += "\n\n[TABLE DATA]\n" + camelot_result[0]
                        else:
                            # Try pdfplumber as fallback
                            plumber_result = table_processor.process_with_pdfplumber(
                                pdf_path if pdf_path else None, page_num
                            )
                            if plumber_result and len(plumber_result) > 1 and plumber_result[1]:
                                page_text += "\n\n[TABLE DATA]\n" + plumber_result[1]

                    all_text.append(page_text)

                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {e}")
                    # Continue with next page instead of failing completely
                    all_text.append("")

                # Report progress
                if progress_callback:
                    progress = (page_num + 1) / len(doc) * 100
                    progress_callback(progress)

            # Combine all pages
            combined_text = "\n\n--- PAGE BREAK ---\n\n".join(all_text)

            # Apply post-processing only if we have text
            if combined_text:
                try:
                    from app.processors.text_post_processor import TextPostProcessor
                    post_processor = TextPostProcessor()
                    processed_text = post_processor.process(combined_text)
                    final_text = self.clean_text(processed_text)
                except Exception as e:
                    logger.error(f"Post-processing failed: {e}")
                    final_text = combined_text
            else:
                final_text = ""

            doc.close()
            return final_text

        except Exception as e:
            logger.error(f"Error in text extraction: {str(e)}")
            # Last resort: try OCR on all pages
            return self._fallback_ocr_extraction(pdf_path, pdf_content)

        finally:
            # Clean up resources
            if doc:
                doc.close()
            # Clean up memory
            self._cleanup_memory()
    # 3. Specific extraction methods
    def _extract_text_ocr_from_page(self, pdf_path, page_num):
        """Extract text from a single page using OCR with timeout protection."""
        try:
            # Add timeout protection
            with timeout(30):  # 30 seconds timeout per page
                logger.info(f"Converting page {page_num + 1} to image for OCR")

                # Convert to high DPI for better OCR
                images = convert_from_path(
                    pdf_path,
                    first_page=page_num + 1,
                    last_page=page_num + 1,
                    dpi=300,
                    poppler_path=self.poppler_path
                )

                if not images:
                    return ""

                # Preprocess image
                preprocessed_img = self._preprocess_image_enhanced(images[0])

                # Apply OCR with custom config
                custom_config = r'--oem 3 --psm 1 -l eng --dpi 300 -c preserve_interword_spaces=1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_/., -c language_model_penalty_non_dict_word=0.5 -c tessedit_do_invert=0'
                text = pytesseract.image_to_string(preprocessed_img, config=custom_config)

                # Apply OCR cleaning
                cleaned_text = self._clean_ocr_text_enhanced(text)

                return cleaned_text

        except TimeoutError:
            logger.error(f"OCR timeout for page {page_num + 1}")
            return ""
        except Exception as e:
            logger.error(f"OCR failed for page {page_num + 1}: {str(e)}")
            return ""

    def _fallback_ocr_extraction(self, pdf_path=None, pdf_content=None):
        """Fallback OCR extraction when other methods fail."""
        try:
            logger.info("Using fallback OCR extraction")

            if pdf_path:
                images = convert_from_path(pdf_path, dpi=300, poppler_path=self.poppler_path)
            else:
                images = convert_from_bytes(pdf_content, dpi=300)

            texts = []
            for i, img in enumerate(images):
                try:
                    # Add timeout for each page
                    with timeout(30):
                        # Preprocess image
                        processed_img = self._preprocess_image_enhanced(img)
                        # Extract text
                        text = pytesseract.image_to_string(processed_img)
                        texts.append(text)
                except TimeoutError:
                    logger.error(f"Fallback OCR timeout for page {i}")
                    texts.append("")
                except Exception as e:
                    logger.error(f"Fallback OCR failed for page {i}: {e}")
                    texts.append("")

            return "\n\n".join(texts)

        except Exception as e:
            logger.error(f"Fallback OCR extraction failed: {e}")
            return ""

    # 4. Helper methods
    def combine_extraction_results(self, text_native, text_ocr):
        """Intelligently combine results from native extraction and OCR."""
        # If one is empty, return the other
        if not text_native:
            return text_ocr
        if not text_ocr:
            return text_native

        # If both have content, use the longer one (likely has more information)
        if len(text_ocr) > len(text_native) * 1.5:
            return text_ocr

        return text_native

    def has_table_structure(self, page_text):
        """Detect if a page contains table-like structures."""
        lines = page_text.split('\n')

        if len(lines) < 3:
            return False

        # Count lines with consistent spacing patterns
        table_pattern_count = 0
        space_pattern = []

        for line in lines:
            spaces = [match.start() for match in re.finditer(r'\s{2,}', line)]
            if len(spaces) >= 2:
                if not space_pattern:
                    space_pattern = spaces
                    table_pattern_count += 1
                else:
                    # Check if spacing pattern is similar
                    matches = sum(1 for pos in spaces if any(abs(pos - p) <= 3 for p in space_pattern))
                    if matches >= len(spaces) * 0.5:
                        table_pattern_count += 1

        return table_pattern_count >= 3

    def extract_tables_with_camelot(self, pdf_path, page_num):
        """Extract tables using camelot with timeout protection."""
        try:
            with timeout(20):  # 20 seconds timeout for table extraction
                # Check if camelot is available
                import camelot

                tables = camelot.read_pdf(pdf_path, pages=str(page_num + 1), flavor='stream')
                if tables.n == 0:
                    # Try lattice flavor
                    tables = camelot.read_pdf(pdf_path, pages=str(page_num + 1), flavor='lattice')

                if tables.n > 0:
                    table_texts = []
                    for table in tables:
                        table_texts.append(table.df.to_string())
                    return "\n\n".join(table_texts)

        except TimeoutError:
            logger.warning(f"Table extraction timeout for page {page_num + 1}")
        except ImportError:
            logger.warning("Camelot not available. Table extraction skipped.")
        except Exception as e:
            logger.warning(f"Camelot failed on page {page_num + 1}: {e}")

        return None

    # 5. Processing methods
    def _preprocess_image_enhanced(self, img):
        """Enhanced image preprocessing for better OCR results."""
        try:
            # Convert to grayscale
            img = img.convert('L')

            # Resize if image is too large
            max_size = 2000
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Apply basic enhancements
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)

            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.5)

            # Convert to numpy array for OpenCV operations if available
            if HAS_OPENCV:
                img_np = np.array(img)

                # Apply denoising
                img_np = cv2.fastNlMeansDenoising(img_np, None, 20, 7, 21)

                # Apply adaptive thresholding
                img_np = cv2.adaptiveThreshold(
                    img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )

                # Convert back to PIL Image
                img = Image.fromarray(img_np)

            return img

        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            return img  # Return original image if preprocessing fails

    def _post_process_text(self, text):
        """Apply post-processing to improve text quality."""
        from app.utils.text_post_processor import TextPostProcessor

        post_processor = TextPostProcessor()

        # Apply all post-processing steps
        processed_text = post_processor.process(text)

        return processed_text

    def clean_text(self, text):
        """
        Clean and standardize extracted text with improved line handling.

        Args:
            text (str): Extracted text

        Returns:
            str: Cleaned text
        """
        if not text:
            return ""

        # EMERGENCY FIX: Directly join Bajaj claim numbers that got split across lines
        text = re.sub(r'(OC-\d+-\d+-\d+)[\s\n]+(\d+)', r'\1-\2', text)

        # Join broken lines in claim numbers - critical for Bajaj Allianz documents
        text = re.sub(r'(OC-\d+-\d+-\d+)[ \t]*\n[ \t]*(\d+)', r'\1-\2', text)

        # Remove excessive whitespace (but preserve newlines for some processing)
        text = re.sub(r'[ \t]+', ' ', text)

        # Remove non-printable characters
        text = ''.join(c for c in text if c.isprintable() or c in ['\n', '\t'])

        # Apply OCR corrections for common misrecognitions
        for error, correction in OCR_CORRECTIONS.items():
            text = text.replace(error, correction)

        return text.strip()

    def _extract_text_ocr_from_file(self, pdf_path):
        """
        Extract text from PDF file using OCR.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            str: Extracted text
        """
        try:
            # Convert PDF to images
            images = convert_from_path(
                pdf_path,
                dpi=300,  # Higher DPI for better quality
                fmt="jpeg",
                poppler_path=self.poppler_path,
                thread_count=2
            )

            # Process images with OCR in parallel
            return self._process_images_with_ocr(images)

        except Exception as e:
            logger.error(f"Error in OCR extraction from file: {str(e)}")
            return ""

    def _extract_text_ocr_from_bytes(self, pdf_content):
        """
        Extract text from PDF bytes using OCR.

        Args:
            pdf_content (bytes): PDF content as bytes

        Returns:
            str: Extracted text
        """
        try:
            # Create a temporary directory for storing images
            with tempfile.TemporaryDirectory() as temp_dir:
                # Convert PDF to images
                images = convert_from_bytes(
                    pdf_content,
                    output_folder=temp_dir,
                    dpi=300,  # Higher DPI for better quality
                    fmt="jpeg",
                    thread_count=2
                )

                # Process images with OCR in parallel
                return self._process_images_with_ocr(images)

        except Exception as e:
            logger.error(f"Error in OCR extraction from bytes: {str(e)}")
            return ""

    def _process_images_with_ocr(self, images):
        """
        Process images with OCR in parallel.

        Args:
            images (list): List of PIL.Image objects

        Returns:
            str: Combined OCR text
        """
        # Maximum number of worker threads
        max_workers = min(4, len(images))

        try:
            # Process images in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                extracted_texts = list(executor.map(self._ocr_process_image, images))

            # Combine the results
            combined_text = "\n\n".join([t for t in extracted_texts if t.strip()])

            # Save OCR results for debugging if enabled
            if self.debug_mode and self.debug_dir:
                debug_ocr_path = os.path.join(self.debug_dir, "ocr_extraction.txt")
                with open(debug_ocr_path, 'w', encoding='utf-8') as f:
                    f.write(combined_text)
                logger.debug(f"Saved OCR results to: {debug_ocr_path}")

            return combined_text

        except Exception as e:
            logger.error(f"Error processing images with OCR: {str(e)}")
            return ""

    def _ocr_process_image(self, img):
        """
        Process a single image with OCR.

        Args:
            img (PIL.Image): Input image

        Returns:
            str: Extracted text
        """
        try:
            # Preprocess the image
            preprocessed_img = self._preprocess_image(img)

            # Save preprocessed image for debugging if enabled
            if self.debug_mode and self.debug_dir:
                debug_img_path = os.path.join(self.debug_dir, f"preprocessed_{id(img)}.png")
                preprocessed_img.save(debug_img_path)
                logger.debug(f"Saved preprocessed image: {debug_img_path}")

            # Custom config for financial documents
            custom_config = r'--oem 3 --psm 6 -l eng --dpi 300 -c preserve_interword_spaces=1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_/., -c language_model_penalty_non_dict_word=0.5 -c tessedit_do_invert=0'

            # Extract text
            text = pytesseract.image_to_string(preprocessed_img, config=custom_config)

            return text
        except Exception as e:
            logger.error(f"Error in OCR processing: {str(e)}")
            return ""

    def _preprocess_image(self, img):
        """
        Enhance image for better OCR accuracy.

        Args:
            img (PIL.Image): The input image

        Returns:
            PIL.Image: Enhanced image
        """
        try:
            # Convert to grayscale
            img = img.convert('L')

            # Apply contrast enhancement
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)

            # Check if OpenCV is available for advanced preprocessing
            if HAS_OPENCV:
                # Convert to numpy array for OpenCV operations
                img_np = np.array(img)

                # Apply noise reduction
                img_np = cv2.fastNlMeansDenoising(img_np, None, 20, 7, 21)

                # Apply adaptive thresholding for better text extraction
                img_np = cv2.adaptiveThreshold(
                    img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )

                # Return as PIL Image
                return Image.fromarray(img_np)
            else:
                # If OpenCV is not available, return the contrast-enhanced image
                logger.info("OpenCV not available. Using basic image preprocessing.")
                return img

        except Exception as e:
            logger.warning(f"Image preprocessing error: {str(e)}")
            return img  # Return original image if preprocessing fails

# Usage example (if run directly)
if __name__ == "__main__":
    # Configure logging when run as standalone
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Simple command-line interface for testing
    import argparse

    parser = argparse.ArgumentParser(description='Extract text from PDF files.')
    parser.add_argument('source', help='PDF file path or URL')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--poppler', help='Path to poppler binaries (for pdf2image)')

    args = parser.parse_args()

    # Create extractor
    extractor = PDFTextExtractor(poppler_path=args.poppler, debug_mode=args.debug)

    # Print first 500 characters of the result
    print(f"Extracted {len(text)} characters of text.")
    print("\nPreview of extracted text:")
    print(text[:500] + "..." if len(text) > 500 else text)