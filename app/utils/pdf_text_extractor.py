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
import time
import json
import traceback

# Configure logging
logger = logging.getLogger("pdf_extractor")

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


class PDFTextExtractor:
    """
    Unified PDF text extraction class used by both training and production code.
    This ensures consistent text extraction between model training and inference.
    """

    def __init__(self, api_url=None, poppler_path=None, debug_mode=False):
        """
        Initialize the PDF text extractor.

        Args:
            poppler_path (str): Path to poppler binaries for pdf2image
            debug_mode (bool): Enable debug mode for extra logging
        """
        self.api_url = api_url
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
        """Extract tables using camelot for structured tables."""
        try:
            tables = camelot.read_pdf(pdf_path, pages=str(page_num + 1))
            if tables.n > 0:
                return "\n".join([table.df.to_string() for table in tables])
        except Exception as e:
            logger.warning(f"Camelot failed on page {page_num + 1}: {e}")
        return None

    def _extract_via_api(self, pdf_path=None, pdf_content=None):
        """
        Extract text using external API service with comprehensive error handling.
        """
        logger.info(f"Attempting API extraction")
        base_url = 'http://localhost:8000/api/v1'

        try:
            # Upload file
            upload_url = f'{base_url}/documents/upload'

            # Prepare files and data
            if pdf_path:
                if not os.path.exists(pdf_path):
                    logger.error(f"PDF file not found: {pdf_path}")
                    return ""

                # Read file content instead of keeping file open
                with open(pdf_path, 'rb') as f:
                    file_content = f.read()

                files = {'file': (os.path.basename(pdf_path), file_content, 'application/pdf')}
                data = {
                    'process_immediately': 'true',
                    'process_directly': 'true'
                }
            elif pdf_content:
                files = {'file': ('document.pdf', pdf_content, 'application/pdf')}
                data = {
                    'process_immediately': 'true',
                    'process_directly': 'true'
                }
            else:
                logger.error("No PDF source provided for API extraction")
                return ""

            # Attempt API extraction with explicit error handling
            try:
                upload_response = requests.post(
                    upload_url,
                    files=files,
                    data=data,
                    timeout=10
                )
            except (requests.ConnectionError, requests.Timeout) as conn_error:
                logger.error(f"API Connection Error: {conn_error}")
                logger.warning("Falling back to local extraction due to API unavailability")
                return ""
            except Exception as e:
                logger.error(f"Unexpected error during API upload: {e}")
                return ""

            # Check upload response
            if upload_response.status_code != 200:
                logger.error(f"API upload failed. Status code: {upload_response.status_code}")
                logger.error(f"Response content: {upload_response.text}")
                return ""

            # Parse upload response
            try:
                upload_result = upload_response.json()
                document_id = upload_result.get('document_id')
            except ValueError:
                logger.error("Failed to parse API response")
                return ""

            if not document_id:
                logger.error("No document ID received")
                return ""

            # Status and text retrieval with error handling
            try:
                # Check status
                status_url = f'{base_url}/extract/{document_id}/status'

                # Wait and poll for completion
                max_attempts = 20 if 'STREAMLIT_SHARING' in os.environ else 10
                initial_wait = 5 if 'STREAMLIT_SHARING' in os.environ else 3

                for attempt in range(max_attempts):
                    try:
                        status_response = requests.get(status_url, timeout=10)
                        status_data = status_response.json()

                        logger.info(f"Status Check (Attempt {attempt + 1}): {status_data}")

                        # Check if processing is complete
                        if status_data.get('status') == 'completed':
                            # Retrieve text
                            text_url = f'{base_url}/documents/{document_id}/consolidated-text'
                            text_response = requests.get(text_url, timeout=10)

                            text_result = text_response.json()
                            extracted_text = text_result.get('consolidated_text', '')

                            logger.info(f"API Extraction Successful. Text Length: {len(extracted_text)}")
                            return extracted_text

                        # If not completed, wait and retry
                        elif status_data.get('status') in ['queued', 'processing', 'pending']:
                            logger.info("Document still processing. Waiting...")
                            wait_time = initial_wait if attempt < 3 else min(initial_wait * 1.5, 10)
                            time.sleep(wait_time)  # Increased wait time
                        else:
                            logger.warning(f"Unexpected status: {status_data.get('status')}")
                            break

                    except requests.RequestException as status_error:
                        logger.error(f"Error checking status: {status_error}")
                        # Wait before retrying
                        time.sleep(wait_time)

                logger.warning("Maximum attempts reached. Processing may have failed.")
                return ""

            except Exception as e:
                logger.error(f"Error during status/text retrieval: {e}")
                return ""

        except Exception as e:
            logger.error(f"Unexpected error during API extraction: {e}")
            return ""

    def extract_from_file(self, pdf_path):
        """
        Extract text from a local PDF file with API-first approach.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            str: Extracted text
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return ""

        try:
            logger.info(f"Attempting extraction for: {pdf_path}")
            logger.info(f"API URL: {self.api_url}")

            # First, try API extraction
            api_text = self._extract_via_api(pdf_path=pdf_path)
            if api_text:
                logger.info("✅ API Extraction Successful!")
                return api_text

            # Fallback to local extraction
            logger.warning("Falling back to local extraction")
            return self._extract_text_from_pdf(
                pdf_path=pdf_path,
                pdf_content=None,
                source_type="file"
            )
        except Exception as e:
            logger.error(f"Error extracting text from PDF file {pdf_path}: {str(e)}")
            return ""

    def extract_from_url(self, url, timeout=60):
        """
        Extract text from a PDF file at the given URL with comprehensive handling.

        Args:
            url (str): URL to the PDF file
            timeout (int): Timeout for URL requests in seconds

        Returns:
            str: Extracted text
        """
        try:
            # Handle Google Drive URL variations
            if 'drive.google.com' in url:
                # Convert sharing URL to direct download URL
                file_id = None

                # Comprehensive Google Drive URL parsing
                url_parsing_strategies = [
                    lambda: url.split('/file/d/')[1].split('/')[0] if '/file/d/' in url else None,
                    lambda: url.split('id=')[1].split('&')[0] if 'id=' in url else None,
                    lambda: url.split('/open?id=')[1].split('&')[0] if '/open?id=' in url else None,
                    lambda: url.split('open?')[1].split('&')[0] if 'open?' in url else None
                ]

                # Try each parsing strategy
                for strategy in url_parsing_strategies:
                    try:
                        file_id = strategy()
                        if file_id:
                            break
                    except Exception:
                        continue

                if file_id:
                    # Direct download URL format with multiple variations
                    direct_url_formats = [
                        f"https://drive.google.com/uc?export=download&id={file_id}",
                        f"https://drive.google.com/file/d/{file_id}/view",
                        f"https://drive.google.com/open?id={file_id}"
                    ]
                else:
                    logger.error(f"Could not parse Google Drive URL: {url}")
                    return ""
            else:
                direct_url = url
                direct_url_formats = [url]

            # Download the PDF with comprehensive retry and fallback mechanism
            max_retries = 3
            retry_delay = 2
            response = None

            # Try different URL formats and retry mechanisms
            for direct_url in direct_url_formats:
                for attempt in range(max_retries):
                    try:
                        # Comprehensive request headers to mimic browser
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                            'Accept': 'application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                            'Accept-Language': 'en-US,en;q=0.5',
                            'Referer': direct_url
                        }

                        logger.info(f"Downloading PDF from {direct_url} (attempt {attempt + 1}/{max_retries})")

                        # Use streaming to handle large files and prevent memory issues
                        response = requests.get(
                            direct_url,
                            headers=headers,
                            timeout=timeout,
                            stream=True
                        )

                        # Validate response
                        if response.status_code == 200:
                            # Verify content type is PDF
                            content_type = response.headers.get('Content-Type', '').lower()
                            if 'application/pdf' not in content_type and 'pdf' not in content_type:
                                logger.warning(f"Non-PDF content type: {content_type}")
                                continue

                            # Read content, limiting to prevent excessive memory usage
                            pdf_content = response.content
                            if len(pdf_content) < 10:  # Minimum PDF file size check
                                logger.warning("Downloaded content is too small to be a valid PDF")
                                continue

                            break

                        logger.warning(f"Failed to download PDF: Status code {response.status_code}. Retrying...")
                        time.sleep(retry_delay)

                    except requests.exceptions.RequestException as e:
                        logger.warning(f"Request error: {str(e)}. Retrying...")
                        time.sleep(retry_delay)

                        # Last attempt for this URL
                        if attempt == max_retries - 1:
                            logger.error(f"Failed to download PDF from {direct_url} after {max_retries} attempts")
                            break

                # If successful download, proceed with extraction
                if response and response.status_code == 200:
                    break

            # Final validation
            if not response or response.status_code != 200:
                logger.error(f"Exhausted all download attempts for URL: {url}")
                return ""

            # Attempt API extraction first
            api_text = self._extract_via_api(pdf_content=pdf_content)
            if api_text:
                return api_text

            # Fallback to local extraction
            logger.info(f"Falling back to local extraction for URL: {url}")
            extracted_text = self._extract_text_from_pdf(
                pdf_path=None,
                pdf_content=pdf_content,
                source_type="url"
            )

            # Additional logging for extracted text
            if not extracted_text:
                logger.warning(f"No text extracted from PDF URL: {url}")
            else:
                logger.info(f"Successfully extracted text from PDF URL. Length: {len(extracted_text)} characters")

            return extracted_text

        except Exception as e:
            # Comprehensive error logging
            logger.error(f"Unhandled error extracting text from PDF URL {url}: {str(e)}")
            logger.error(traceback.format_exc())
            return ""

    def extract_from_bytes(self, pdf_bytes):
        """
        Extract text from PDF bytes.

        Args:
            pdf_bytes (bytes): PDF content as bytes

        Returns:
            str: Extracted text
        """
        try:
            logger.info("Extracting text from PDF bytes")

            # First, try API extraction
            api_text = self._extract_via_api(pdf_content=pdf_bytes)
            if api_text:
                return api_text

                # Fallback to local extraction
                logger.info("Falling back to local extraction from bytes")
                return self._extract_text_from_pdf(
                    pdf_path=None,
                    pdf_content=pdf_bytes,
                    source_type="bytes"
                )

        except Exception as e:
            logger.error(f"Error extracting text from PDF bytes: {str(e)}")
            return ""


    def _extract_text_ocr_from_page(self, pdf_path, page_num):
        """Extract text from a single page using OCR with enhanced processing."""
        try:
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
            custom_config = r'--oem 3 --psm 1 -l eng --dpi 300 -c preserve_interword_spaces=1'
            text = pytesseract.image_to_string(preprocessed_img, config=custom_config)

            # Apply OCR cleaning
            cleaned_text = self._clean_ocr_text_enhanced(text)

            return cleaned_text
        except Exception as e:
            logger.error(f"OCR failed for page {page_num + 1}: {str(e)}")
            return ""

    def _preprocess_image_enhanced(self, img):
        """Enhanced image preprocessing for better OCR results."""
        # Convert to grayscale
        img = img.convert('L')

        # Apply contrast enhancement
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)

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

    def _extract_text_from_pdf(self, pdf_path=None, pdf_content=None, source_type="file"):
        """Enhanced text extraction with multi-method approach."""

        # Initialize components
        from app.utils.text_post_processor import TextPostProcessor
        post_processor = TextPostProcessor()

        all_text = []

        try:
            # Open PDF based on source type
            if source_type == "file":
                doc = fitz.open(pdf_path)
            else:
                doc = fitz.open(stream=pdf_content, filetype="pdf")

            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]

                # Determine extraction method
                text = page.get_text("text")
                word_count = len(text.split())

                if word_count < 5:
                    # OCR only
                    page_text = self._extract_text_ocr_from_page(pdf_path, page_num)
                elif word_count < 20:
                    # Hybrid approach
                    text_native = text
                    text_ocr = self._extract_text_ocr_from_page(pdf_path, page_num)
                    page_text = self.combine_extraction_results(text_native, text_ocr)
                else:
                    # Native extraction with table detection
                    page_text = text

                    # Check for tables
                    if self.has_table_structure(page_text):
                        table_data = self.extract_tables_with_camelot(pdf_path, page_num)
                        if table_data:
                            page_text += "\n\n[TABLE DATA]\n" + table_data

                all_text.append(page_text)

            # Combine all pages
            combined_text = "\n\n--- PAGE BREAK ---\n\n".join(all_text)

            # Apply post-processing
            processed_text = post_processor.process(combined_text)

            # Clean the text further
            final_text = self.clean_text(processed_text)

            doc.close()

            return final_text

        except Exception as e:
            logger.error(f"Error in enhanced text extraction: {str(e)}")
            return ""

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

    def _post_process_text(self, text):
        """Apply post-processing to improve text quality."""
        from app.utils.text_post_processor import TextPostProcessor

        post_processor = TextPostProcessor()

        # Apply all post-processing steps
        processed_text = post_processor.process(text)

        return processed_text

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

    # Extract text
    if args.source.startswith(('http://', 'https://')):
        text = extractor.extract_from_url(args.source)
    else:
        text = extractor.extract_from_file(args.source)

    # Print first 500 characters of the result
    print(f"Extracted {len(text)} characters of text.")
    print("\nPreview of extracted text:")
    print(text[:500] + "..." if len(text) > 500 else text)