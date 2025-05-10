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

        # Detect if running in Streamlit Cloud
        self.is_cloud = 'STREAMLIT_SHARING' in os.environ

        # Adjust settings for cloud environment
        if self.is_cloud:
            self.max_workers = 1  # Reduce parallelism
            self.dpi = 150  # Lower DPI for images
        else:
            self.max_workers = 4  # Default for local
            self.dpi = 300  # Higher quality locally

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
                max_attempts = 10
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
                        if status_data.get('status') in ['queued', 'processing', 'pending']:
                            logger.info("Document still processing. Waiting...")
                            time.sleep(3)  # Increased wait time
                        else:
                            logger.warning(f"Unexpected status: {status_data.get('status')}")
                            break

                    except requests.RequestException as status_error:
                        logger.error(f"Error checking status: {status_error}")
                        # Wait before retrying
                        time.sleep(3)

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

    def _extract_text_from_pdf(self, pdf_path=None, pdf_content=None, source_type="file"):
        """
        Core text extraction logic used by all extraction methods.

        Args:
            pdf_path (str, optional): Path to the PDF file
            pdf_content (bytes, optional): PDF content as bytes
            source_type (str): Source type ('file', 'url', or 'bytes')

        Returns:
            str: Extracted text
        """
        # Extraction results from different methods
        extraction_methods = []

        # 1. Try PyMuPDF (fitz) extraction - often the best for non-scanned PDFs
        try:
            if source_type == "file":
                with fitz.open(pdf_path) as doc:
                    text_pymupdf = ""
                    for page_num, page in enumerate(doc):
                        text_pymupdf += page.get_text("text") + "\n"

                        # Save images for debugging if enabled
                        if self.debug_mode and self.debug_dir:
                            pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
                            debug_img_path = os.path.join(
                                self.debug_dir,
                                f"{os.path.basename(pdf_path) if pdf_path else 'pdf'}_page{page_num + 1}_pymupdf.png"
                            )
                            pix.save(debug_img_path)
            else:  # url or bytes
                with fitz.open(stream=pdf_content, filetype="pdf") as doc:
                    text_pymupdf = ""
                    for page in doc:
                        text_pymupdf += page.get_text("text") + "\n"

            extraction_methods.append(("PyMuPDF", text_pymupdf))
            logger.debug(f"PyMuPDF extraction: {len(text_pymupdf)} characters")
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {str(e)}")

        # 2. Try PyPDF2 extraction
        try:
            if source_type == "file":
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text_pypdf2 = ""
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text_pypdf2 += page.extract_text() + "\n"
            else:  # url or bytes
                pdf_file = io.BytesIO(pdf_content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text_pypdf2 = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text_pypdf2 += page.extract_text() + "\n"

            extraction_methods.append(("PyPDF2", text_pypdf2))
            logger.debug(f"PyPDF2 extraction: {len(text_pypdf2)} characters")
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {str(e)}")

        # 3. Try pdfplumber extraction
        try:
            if source_type == "file":
                with pdfplumber.open(pdf_path) as pdf:
                    text_pdfplumber = ""
                    for page_num, page in enumerate(pdf.pages):
                        page_text = page.extract_text()
                        if page_text:
                            text_pdfplumber += page_text + "\n"

                            # Save images for debugging if enabled
                            if self.debug_mode and self.debug_dir:
                                img = page.to_image()
                                debug_img_path = os.path.join(
                                    self.debug_dir,
                                    f"{os.path.basename(pdf_path) if pdf_path else 'pdf'}_page{page_num + 1}_plumber.png"
                                )
                                img.save(debug_img_path)
            else:  # url or bytes
                pdf_file = io.BytesIO(pdf_content)
                with pdfplumber.open(pdf_file) as pdf:
                    text_pdfplumber = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_pdfplumber += page_text + "\n"

            extraction_methods.append(("pdfplumber", text_pdfplumber))
            logger.debug(f"pdfplumber extraction: {len(text_pdfplumber)} characters")
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {str(e)}")

        # 4. Try OCR if direct extraction methods yielded limited text
        best_text_so_far = max([text for _, text in extraction_methods], key=len, default="")
        if len(best_text_so_far) < 500:  # Only use OCR if direct methods produced little text
            logger.info("Standard extraction yielded limited text. Trying OCR...")

            try:
                if source_type == "file":
                    text_ocr = self._extract_text_ocr_from_file(pdf_path)
                else:  # url or bytes
                    text_ocr = self._extract_text_ocr_from_bytes(pdf_content)

                if text_ocr:
                    extraction_methods.append(("OCR", text_ocr))
                    logger.debug(f"OCR extraction: {len(text_ocr)} characters")
            except Exception as e:
                logger.warning(f"OCR extraction failed: {str(e)}")

        # Use the best result (longest text)
        extraction_methods.sort(key=lambda x: len(x[1]), reverse=True)

        if extraction_methods:
            best_method, best_text = extraction_methods[0]
            logger.info(f"Best extraction method: {best_method} with {len(best_text)} characters")

            # If the best text is still very short, combine all methods
            if len(best_text) < 200 and len(extraction_methods) > 1:
                logger.info("Combining results from all extraction methods")
                combined_text = "\n\n".join([text for _, text in extraction_methods])
            else:
                combined_text = best_text

            # Clean and standardize the text
            combined_text = self.clean_text(combined_text)

            # Save extracted text for debugging if enabled
            if self.debug_mode and self.debug_dir:
                filename = os.path.basename(pdf_path) if pdf_path else "pdf_extraction"
                debug_text_path = os.path.join(self.debug_dir, f"{filename}_extracted_text.txt")
                with open(debug_text_path, 'w', encoding='utf-8') as f:
                    f.write(combined_text)
                logger.debug(f"Saved extracted text to: {debug_text_path}")

            return combined_text
        else:
            logger.error("All text extraction methods failed")
            return ""

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
                dpi=self.dpi,  # Higher DPI for better quality
                fmt="jpeg",
                poppler_path=self.poppler_path,
                thread_count=1 if self.is_cloud else 2
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
        max_workers = min(self.max_workers, len(images))

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