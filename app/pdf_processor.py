import os
import re
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
from pdf2image import convert_from_path, convert_from_bytes
import pytesseract
import pandas as pd
import spacy
import joblib
import logging
from pathlib import Path
import tempfile
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageOps
import concurrent.futures
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pdf_processing.log")
    ]
)
logger = logging.getLogger("pdf_processor")


class PDFProcessor:
    def __init__(self, use_ml=True, debug_mode=False, poppler_path=None):
        """
        Initialize the PDF processor.

        Args:
            use_ml (bool): Whether to use ML model for extraction enhancement
            debug_mode (bool): Enable debug mode for visualization and extra logging
            poppler_path (str): Path to poppler binaries for pdf2image
        """
        self.use_ml = use_ml
        self.debug_mode = debug_mode
        self.debug_dir = None
        self.poppler_path = poppler_path

        if self.debug_mode:
            # Create debug directory for visualizations
            self.debug_dir = os.path.join(os.path.expanduser("~"), "Downloads", "PDF_Debug")
            os.makedirs(self.debug_dir, exist_ok=True)
            logger.info(f"Debug mode enabled. Visualizations will be saved to {self.debug_dir}")

        # Load NLP model for text analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            # If model not found, download it
            import subprocess
            subprocess.call(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        # Load ML model if available and use_ml is True
        self.ml_model = None
        if self.use_ml and os.path.exists('models/pdf_extractor_model.joblib'):
            try:
                self.ml_model = joblib.load('models/pdf_extractor_model.joblib')
                logger.info("ML model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load ML model: {str(e)}")

        # Setup pytesseract
        if os.name == 'nt':  # For Windows
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        # Import field mappings from excel_handler's expected fields
        # These should ideally be in a shared configuration
        self.field_mappings = {
            'receipt_date': ['Receipt Date', 'Receipt Date', 'Payment Date', 'Value Date', 'Date', 'Payment Ini Date',
                             'Value date', 'Advice sending date', 'Settlement Date', 'Value Date', 'VALUE DATE'],
            'receipt_amount': ['Receipt Amount', 'Receipt Amt', 'Payment Amount', 'Amount', 'Value', 'Amount',
                               'Remittance amount', 'Net Paid Amount', 'REMITTANCE AMOUNT', 'Net Amount',
                               'Amount (INR)', 'AMOUNT', 'Payment amount', 'TRF AMOUNT'],
            'tds': ['TDS', 'TDS Amount', 'Tax Deducted at Source', 'Tax Amount', 'TDS Amount', 'TDS', 'TDS Amt',
                    'Payment Details 4', 'Less : TDS'],
            'tds_computed': ['TDS Computed?', 'TDS Computed', 'Is TDS Computed', 'Computed TDS']
        }

        # Import ID column patterns from excel_handler
        self.invoice_columns = [
            'Invoice No.', 'Invoice No', 'Invoice Number', 'Invoice',
            'Inv No.', 'Inv No', 'Inv Number', 'Invoice #', 'Bill No.',
            'Bill No', 'Payment Details 5', 'ThirdPartyInv', 'ILA_REF_NO',
            'Expense Paid', 'Invoice'
        ]

        self.client_ref_columns = [
            'Client Ref. No.', 'Client Ref. No', 'Client Reference Number',
            'Client Ref', 'Ref. No.', 'Reference No.', 'Reference Number', 'Claim#', 'Claim Number', 'CLAIM NUMBER',
            'Claim number', 'Sub Claim No', 'INV REF:Claim No', 'Claim No', 'DESC', 'Payment Details 7',
            'ClaimNo', 'Claim_Ref_No', 'CLAIM_REF_NO', 'Invoice Details'
        ]

    def preprocess_image(self, img):
        """
        Enhances image for better OCR accuracy with advanced preprocessing.

        Args:
            img (PIL.Image): The input image

        Returns:
            PIL.Image: Enhanced image
        """
        # Convert to grayscale
        img = img.convert('L')

        # Apply contrast enhancement
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)

        # Convert to numpy array for OpenCV operations
        img_np = np.array(img)

        # Apply noise reduction
        img_np = cv2.fastNlMeansDenoising(img_np, None, 20, 7, 21)  # Reduced strength from 30 to 20

        # Apply slight Gaussian blur
        img_np = cv2.GaussianBlur(img_np, (3, 3), 0)

        # Apply Otsu's thresholding (more adaptive than fixed threshold)
        _, img_np = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Return as PIL Image
        return Image.fromarray(img_np)

    def detect_text_structure(self, image):
        """
        Dynamically selects best OCR mode based on image analysis.

        Args:
            image (PIL.Image): The input image

        Returns:
            str: The best PSM mode for Tesseract
        """
        try:
            img_np = np.array(image)
            edges = cv2.Canny(img_np, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            num_contours = len(contours)

            if num_contours > 100:  # Multi-column text
                return "--psm 4"
            elif num_contours < 20:  # Sparse text
                return "--psm 11"
            else:
                return "--psm 6"
        except Exception as e:
            logger.warning(f"Error detecting text structure: {str(e)}")
            return "--psm 6"  # Default to single block of text

    def ocr_process_image(self, img):
        """
        Runs OCR on a single image with advanced preprocessing.

        Args:
            img (PIL.Image): The input image

        Returns:
            str: Extracted text
        """
        try:
            # Preprocessing steps
            preprocessed_img = self.preprocess_image(img)

            # Additional image enhancement
            preprocessed_img = ImageOps.autocontrast(preprocessed_img, cutoff=1)
            preprocessed_img = ImageEnhance.Sharpness(preprocessed_img).enhance(2.0)
            preprocessed_img = ImageEnhance.Contrast(preprocessed_img).enhance(1.5)

            # Detect text structure for PSM selection
            selected_psm = self.detect_text_structure(preprocessed_img)

            # Save preprocessed image for debugging
            if self.debug_mode and self.debug_dir:
                debug_img_path = os.path.join(self.debug_dir, f"preprocessed_{id(img)}.png")
                preprocessed_img.save(debug_img_path)
                logger.debug(f"Saved preprocessed image: {debug_img_path}")
                logger.debug(f"Using PSM mode: {selected_psm}")

            # Enhanced custom config for financial documents
            custom_config = (
                f'{selected_psm} '
                '-l eng '
                '--oem 3 '
                '--dpi 300 '
                '-c preserve_interword_spaces=1 '
                '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_/., '
                '-c language_model_penalty_non_dict_word=0.5 '
                '-c tessedit_do_invert=0 '
                '-c tessedit_enable_doc_dict=1'
            )

            # Perform OCR
            extracted_text = pytesseract.image_to_string(preprocessed_img, config=custom_config)

            # Additional post-processing
            extracted_text = self.correct_ocr_text(extracted_text)

            return extracted_text

        except Exception as e:
            logger.error(f"Error in OCR process: {str(e)}")
            return ""

    def extract_text_pymupdf(self, pdf_path):
        """
        Extract text from PDF using PyMuPDF (fitz).

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            str: Extracted text
        """
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc):
                    page_text = page.get_text("text")
                    text += page_text + "\n"

                    if self.debug_mode:
                        logger.debug(f"PyMuPDF extraction - Page {page_num + 1} sample: {page_text[:200]}")

                        # Save page as image for debugging
                        if self.debug_dir:
                            pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
                            debug_img_path = os.path.join(self.debug_dir,
                                                          f"{os.path.basename(pdf_path)}_page{page_num + 1}_pymupdf.png")
                            pix.save(debug_img_path)
                            logger.debug(f"Saved PyMuPDF page image: {debug_img_path}")
        except Exception as e:
            logger.error(f"Error using PyMuPDF on {pdf_path}: {str(e)}")
        return text.strip()

    def extract_text_pypdf2(self, pdf_path):
        """Extract text from PDF using PyPDF2."""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    text += page_text + "\n"

                    if self.debug_mode:
                        logger.debug(f"PyPDF2 extraction - Page {page_num + 1} sample: {page_text[:200]}")
        except Exception as e:
            logger.error(f"Error using PyPDF2 on {pdf_path}: {str(e)}")
        return text.strip()

    def extract_text_pdfplumber(self, pdf_path):
        """Extract text from PDF using pdfplumber."""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

                        if self.debug_mode:
                            logger.debug(f"pdfplumber extraction - Page {page_num + 1} sample: {page_text[:200]}")

                            # Save page image for debugging
                            if self.debug_dir:
                                img = page.to_image()
                                debug_img_path = os.path.join(self.debug_dir,
                                                              f"{os.path.basename(pdf_path)}_page{page_num + 1}_plumber.png")
                                img.save(debug_img_path)
                                logger.debug(f"Saved pdfplumber page image: {debug_img_path}")
        except Exception as e:
            logger.error(f"Error using pdfplumber on {pdf_path}: {str(e)}")
        return text.strip()

    def extract_text_ocr(self, pdf_path):
        """
        Extract text from PDF using OCR with enhanced preprocessing and multi-threading.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            str: Extracted text
        """
        text = ""
        try:
            # Create a temporary directory for storing images
            with tempfile.TemporaryDirectory() as temp_dir:
                # Convert PDF to images with higher DPI for better OCR
                images = convert_from_path(
                    pdf_path,
                    output_folder=temp_dir,
                    dpi=350,  # Higher DPI for better quality
                    poppler_path=self.poppler_path
                )

                # Log number of pages converted
                logger.debug(f"Converted PDF to {len(images)} images for OCR")

                # Process images in parallel using multi-threading
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(images))) as executor:
                    extracted_texts = list(executor.map(self.ocr_process_image, images))

                # Combine the results
                text = "\n\n".join([t for t in extracted_texts if t.strip()])

                if self.debug_mode and self.debug_dir:
                    # Save full OCR results to file
                    debug_ocr_path = os.path.join(self.debug_dir, f"{os.path.basename(pdf_path)}_ocr_full.txt")
                    with open(debug_ocr_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    logger.debug(f"Saved full OCR results to: {debug_ocr_path}")

        except Exception as e:
            logger.error(f"Error using OCR on {pdf_path}: {str(e)}")
            logger.error(f"OCR error details: {str(e)}", exc_info=True)
        return text.strip()

    def correct_ocr_text(self, text):
        """
        Correct common OCR errors using a domain-specific dictionary.

        Args:
            text (str): OCR-extracted text

        Returns:
            str: Corrected text
        """
        corrections = {
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
            "0C":"OC"
        }

        for error, correction in corrections.items():
            text = text.replace(error, correction)

        return text

    def extract_text(self, pdf_path):
        """
        Extract text from PDF using multiple methods and combine results.
        Uses direct extraction first, falling back to OCR if needed.

        Returns:
            str: Combined extracted text
        """
        logger.info(f"Processing PDF: {pdf_path}")

        # Try direct extraction using PyMuPDF first (fastest method)
        text_pymupdf = self.extract_text_pymupdf(pdf_path)

        # Log the results
        logger.debug(f"Text length from PyMuPDF: {len(text_pymupdf)}")

        # If PyMuPDF yields sufficient text, use it directly
        if len(text_pymupdf) > 100:
            logger.info("Using PyMuPDF extraction as primary source")
            combined_text = text_pymupdf
        else:
            # Try alternate direct extraction methods
            text_pypdf2 = self.extract_text_pypdf2(pdf_path)
            text_pdfplumber = self.extract_text_pdfplumber(pdf_path)

            # Log the lengths of extracted text from each method
            logger.debug(f"Text length from PyPDF2: {len(text_pypdf2)}")
            logger.debug(f"Text length from pdfplumber: {len(text_pdfplumber)}")

            # If direct extraction methods yield limited text, use OCR
            if (len(text_pymupdf) < 100 and len(text_pypdf2) < 100 and len(text_pdfplumber) < 100):
                logger.info("Standard extraction methods yielded limited text. Switching to OCR.")

                # Use enhanced OCR with preprocessing
                text_ocr = self.extract_text_ocr(pdf_path)
                logger.debug(f"Text length from enhanced OCR: {len(text_ocr)}")

                if len(text_ocr) > 0:
                    combined_text = text_ocr

                    # Also append any text that might contain unique information
                    # from direct extraction
                    if len(text_pymupdf) > 50:
                        combined_text += "\n" + text_pymupdf
                    if len(text_pypdf2) > 50:
                        combined_text += "\n" + text_pypdf2
                    if len(text_pdfplumber) > 50:
                        combined_text += "\n" + text_pdfplumber
                else:
                    # Fallback to the best direct extraction result if OCR fails
                    logger.warning("OCR failed. Using best direct extraction result.")
                    combined_text = max([text_pymupdf, text_pypdf2, text_pdfplumber], key=len)
            else:
                # Use the best direct extraction result
                logger.info("Using best direct extraction result.")
                combined_text = max([text_pymupdf, text_pypdf2, text_pdfplumber], key=len)

        # Clean up the text
        combined_text = self.clean_text(combined_text)

        # Apply dictionary-based corrections
        combined_text = self.correct_ocr_text(combined_text)

        # Save combined text for debugging
        if self.debug_mode and self.debug_dir:
            debug_text_path = os.path.join(self.debug_dir, f"{os.path.basename(pdf_path)}_combined_text.txt")
            with open(debug_text_path, 'w', encoding='utf-8') as f:
                f.write(combined_text)
            logger.debug(f"Saved combined extracted text: {debug_text_path}")

        return combined_text

    def clean_text(self, text):
        """Clean up extracted text."""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove non-printable characters
        text = ''.join(c for c in text if c.isprintable() or c in ['\n', '\t'])

        # Fix common OCR errors
        text = text.replace('l', '1').replace('O', '0').replace('S', '5')

        return text.strip()

    # Add a method for generic pattern extraction
    def extract_generic_data(self, text):
        """
        Extract data using generic patterns when provider-specific patterns fail.
        """
        data = {}

        # Generic patterns for unique identifiers
        unique_id_patterns = [
            # Claim number formats
            r'claim\s+(?:no|number|#)\.?\s*:?\s*(\S+)',
            r'claim\s*[:.\s]\s*(\S+)',
            # Invoice number formats
            r'invoice\s+(?:no|number|#)\.?\s*:?\s*(\S+)',
            r'inv\s+(?:no|number|#)\.?\s*:?\s*(\S+)',
            # Policy number formats
            r'policy\s+(?:no|number|#)\.?\s*:?\s*(\S+)',
            # Reference number formats
            r'ref(?:erence)?\s+(?:no|number|#)\.?\s*:?\s*(\S+)',
            r'our\s+ref(?:erence)?\s*:?\s*(\S+)',
            r'your\s+ref(?:erence)?\s*:?\s*(\S+)',
            # UTR/Payment reference
            r'utr\s+(?:no|number|#)\.?\s*:?\s*(\S+)',
            r'payment\s+ref(?:erence)?\s*:?\s*(\S+)',
            # Common claim number formats
            r'\d{6}/\d{2}/\d{4}/\d+',  # Format like 510000/11/2025/000000
            r'\d{19}',  # Long numeric ID
            r'\d{10}[cC]\d{8}',  # Format like 2502002624C050122001
            r'2425-\d{5}',  # Invoice format
            r'OC-\d{2}-\d{4}-\d{3}-\d+',  # Specific to Bajaj Allianz format
            r'Claim\s*No\s*[:.]?\s*(OC-\d{2}-\d{4}-\d{3}-\d+)',
            r'Claim\s*No\s*[:.]?\s*(\d+)',
        ]

        # Try to extract unique ID using patterns
        for pattern in unique_id_patterns:
            try:
                if '(' in pattern:  # Pattern has a capture group
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        data['unique_id'] = match.group(1).strip()
                        logger.debug(f"Extracted unique ID with pattern {pattern}: {data['unique_id']}")
                        break
                else:  # Pattern is a direct match without capture groups
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        data['unique_id'] = match.group(0).strip()
                        logger.debug(f"Extracted unique ID with direct pattern: {data['unique_id']}")
                        break
            except Exception as e:
                logger.warning(f"Error with pattern {pattern}: {str(e)}")

        # Generic patterns for amount
        amount_patterns = [
            r'amount\s*:?\s*(?:rs\.?|inr)?\s*([0-9,.]+)',
            r'remittance\s+amount\s*:?\s*(?:rs\.?|inr)?\s*([0-9,.]+)',
            r'payment\s+amount\s*:?\s*(?:rs\.?|inr)?\s*([0-9,.]+)',
            r'net\s+amount\s*:?\s*(?:rs\.?|inr)?\s*([0-9,.]+)',
            r'gross\s+amount\s*:?\s*(?:rs\.?|inr)?\s*([0-9,.]+)',
            r'(?:rs\.?|inr)\s*([0-9,.]+)',
            r'rupees\s+([a-zA-Z\s]+)\s+only',  # For amounts in words
        ]

        # Try to extract amount
        for pattern in amount_patterns:
            amount_match = re.search(pattern, text, re.IGNORECASE)
            if amount_match:
                # Check if the pattern captures the amount or if it's in words
                if 'rupees' in pattern.lower():
                    # Convert words to numbers - would need a more sophisticated conversion in practice
                    words_amount = amount_match.group(1).strip().lower()
                    # This is a placeholder - a real implementation would convert word amounts to numbers
                    logger.warning(f"Amount in words detected but not converted: {words_amount}")
                else:
                    data['receipt_amount'] = amount_match.group(1).strip().replace(',', '')
                    logger.debug(f"Extracted amount: {data['receipt_amount']}")
                break

        # Generic patterns for date
        date_patterns = [
            r'date\s*:?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
            r'date\s*:?\s*(\d{2,4}[-/\.]\d{1,2}[-/\.]\d{1,2})',
            r'value\s+date\s*:?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
            r'payment\s+date\s*:?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
            r'(?:settlement|transaction)\s+date\s*:?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
            r'date\s*:?\s*(\d{1,2}\s+[a-zA-Z]{3}\s+\d{4})',  # Format like "11 Feb 2025"
        ]

        # Try to extract date
        for pattern in date_patterns:
            date_match = re.search(pattern, text, re.IGNORECASE)
            if date_match:
                data['receipt_date'] = date_match.group(1).strip()
                logger.debug(f"Extracted date: {data['receipt_date']}")
                break

        # Generic patterns for TDS
        tds_patterns = [
            r'tds\s*:?\s*(?:rs\.?|inr)?\s*([0-9,.]+)',
            r'tds\s+amount\s*:?\s*(?:rs\.?|inr)?\s*([0-9,.]+)',
            r'less\s*:?\s*tds\s*:?\s*(?:rs\.?|inr)?\s*([0-9,.]+)',
            r'tax\s+deducted\s+at\s+source\s*:?\s*(?:rs\.?|inr)?\s*([0-9,.]+)',
        ]

        # Try to extract TDS
        for pattern in tds_patterns:
            tds_match = re.search(pattern, text, re.IGNORECASE)
            if tds_match:
                data['tds'] = tds_match.group(1).strip().replace(',', '')
                data['tds_computed'] = 'No'  # TDS already present in the document
                logger.debug(f"Extracted TDS: {data['tds']}")
                break

        # Log the results
        logger.debug(f"Generic extraction results: {data}")

        return data

    # Expand provider identification in extract_bank_specific_data
    def extract_bank_specific_data(self, text):
        """
        Extract data specifically from banking documents (payment advices).
        Enhanced to support multiple insurance providers.
        """
        data = {}

        # Normalize text for better pattern matching
        normalized_text = ' '.join(text.lower().split())
        logger.debug(f"Normalized text for pattern matching: {normalized_text[:200]}...")

        # Create a list of all insurance providers to check
        insurance_providers = {
            "oriental": ["hsbc", "oriental insurance", "oicl", "the oriental insurance"],
            "united": ["axis bank", "united india insurance", "united india", "uiic"],
            "tata_aig": ["tata aig", "tataaig", "noreply", "claim number"],
            "reliance": ["reliance general", "sterlit", "sector-1"],
            "hdfc_ergo": ["hdfc ergo", "business suraksha", "hdfcn5"],
            "future_generali": ["future generali", "embassy 247", "hdfcn5"],
            "iffco_tokio": ["iffco", "tokio", "paid claims notification"],
            "liberty": ["liberty", "international center", "senapati"],
            "national": ["national insurance", "paid claims notification"],
            "universal_sompo": ["universal sompo", "ackruti star", "protocol house"],
            "new_india": ["new india", "payment of rs", "assurance"],
            "bajaj_allianz": ["bajaj allianz", "standard chartered", "scbln"],
            "cholamandalam": ["cholamandalam", "ms genera", "protocol house"]
        }

        # Determine the insurance provider
        detected_provider = None
        for provider, keywords in insurance_providers.items():
            if any(keyword in normalized_text for keyword in keywords):
                detected_provider = provider
                logger.info(f"Detected insurance provider: {provider}")
                break

        # If no provider detected, try to extract using generic patterns
        if not detected_provider:
            logger.info("No specific insurance provider detected. Using generic patterns.")
            return self.extract_generic_data(text)

        # Provider-specific extraction patterns
        if detected_provider == "oriental":
            # HSBC/Oriental Insurance format
            logger.debug("Extracting data using Oriental Insurance patterns")

            # Look for claim number in various formats
            claim_patterns = [
                r'claim\s+number.*?(\d{6}/\d{2}/\d{4}/\d+)',
                r'510000/11/(\d{4}/\d+)',
                r'claim\s+number\s*510000/11/\d{4}/\d+',
                r'(\d{6}/\d{2}/\d{4}/\d+)',
                r'claim\s+number.*?\n.*?(\d+/\d+/\d+/\d+)'
            ]

            for pattern in claim_patterns:
                claim_match = re.search(pattern, normalized_text, re.IGNORECASE)
                if claim_match:
                    if "510000/11" in claim_match.group(0):
                        data['unique_id'] = claim_match.group(0).strip()
                    else:
                        data['unique_id'] = claim_match.group(1).strip()
                    logger.debug(f"Extracted claim number: {data['unique_id']}")
                    break

            # Extract remittance amount
            amount_patterns = [
                r'remittance\s+amount\s*:?\s*(?:inr)?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'debit\s+amount\s*:?\s*(?:inr)?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'amount.*?(\d{6}(?:\.\d{2})?)',
                r'amount\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            ]

            for pattern in amount_patterns:
                amount_match = re.search(pattern, normalized_text, re.IGNORECASE)
                if amount_match:
                    data['receipt_amount'] = amount_match.group(1).strip().replace(',', '')
                    logger.debug(f"Extracted amount: {data['receipt_amount']}")
                    break

            # Extract date
            date_patterns = [
                r'value\s+date\s*:?\s*(\d{1,2}\s+[a-z]+\s+\d{4})',
                r'advice\s+sending\s+date\s*:?\s*(\d{1,2}\s+[a-z]+\s+\d{4})',
                r'\d{1,2}\s+[a-z]+\s+\d{4}',
            ]

            for pattern in date_patterns:
                date_match = re.search(pattern, normalized_text, re.IGNORECASE)
                if date_match:
                    data['receipt_date'] = date_match.group(0).strip()
                    logger.debug(f"Extracted date: {data['receipt_date']}")
                    break

        elif detected_provider == "united":
            # AXIS/United India format
            logger.debug("Extracting data using United India Insurance patterns")

            # Extract claim number
            claim_patterns = [
                r'claim#\s*.*?(\d{10}c\d{8})',
                r'claim\s*[\s#:]*\s*(\d{10}c\d{8})',
                r'(\d{10}c\d{8})',
                r'claim#\s*(5004\d{10})',
                r'payment\s+ref\.\s+no\.\s*:\s*(\d{18})',
                r'(5004\d{10})',
                r'claim#\s*(\d{19})',
                r'enrichment:[\s\S]*?claim#\s*(\S+)',
                r'2502\d+C\d+',
            ]

            for pattern in claim_patterns:
                claim_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if claim_match:
                    if "5004" in claim_match.group(0) or "c" in claim_match.group(
                            0).lower() or "2502" in claim_match.group(0):
                        data['unique_id'] = claim_match.group(0).strip()
                    elif '(' in pattern:
                        data['unique_id'] = claim_match.group(1).strip()
                    else:
                        data['unique_id'] = claim_match.group(0).strip()
                    logger.debug(f"Extracted claim number: {data['unique_id']}")
                    break

            # Extract amount
            amount_patterns = [
                r'amount\s*:?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:inr|rs\.?)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:rs\.?|inr)\s*(\d+\.\d{2})',
                r'amount\s*:?\s*(\d+\.\d{2})',
                r'invoice\s+amount\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                r'neft.*?amount\s*:\s*([0-9,.]+)',
                r'remittance amount\s*:\s*([0-9,.]+)',
            ]

            for pattern in amount_patterns:
                amount_match = re.search(pattern, normalized_text, re.IGNORECASE)
                if amount_match:
                    amt = amount_match.group(1) if '(' in pattern else amount_match.group(0)
                    data['receipt_amount'] = amt.strip().replace(',', '')
                    if data['receipt_amount'].startswith('rs') or data['receipt_amount'].startswith('inr'):
                        data['receipt_amount'] = re.sub(r'[^\d.]', '', data['receipt_amount'])
                    logger.debug(f"Extracted amount: {data['receipt_amount']}")
                    break

            # Extract date
            date_patterns = [
                r'payment\s+ini\s+date\s*:?\s*(\d{2}-\d{2}-\d{4})',
                r'date\s*:?\s*(\d{2}-\d{2}-\d{4})',
                r'invoice\s+date\s*(\d{2}-\d{2}-\d{4})',
                r'value\s+date\s*:?\s*(\d{2}-\d{2}-\d{4})',
                r'(\d{2}-\d{2}-\d{4})',
            ]

            for pattern in date_patterns:
                date_match = re.search(pattern, normalized_text, re.IGNORECASE)
                if date_match:
                    if '(' in pattern:
                        data['receipt_date'] = date_match.group(1).strip()
                    else:
                        data['receipt_date'] = date_match.group(0).strip()
                    logger.debug(f"Extracted date: {data['receipt_date']}")
                    break

        elif detected_provider == "tata_aig":
            # TATA AIG format
            logger.debug("Extracting data using TATA AIG patterns")

            # Look for claim number pattern
            claim_patterns = [
                r'claim\s+number\s*[–\-:]\s*(\d+)',
                r'payment\s+details\s+for\s*[–\-:]\s*claim\s+number\s*[–\-:]\s*(\d+)',
                r'payment of :-.*?s fees for invoice (\d+-\d+)',
            ]

            for pattern in claim_patterns:
                claim_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if claim_match:
                    data['unique_id'] = claim_match.group(1).strip()
                    logger.debug(f"Extracted TATA AIG claim/invoice number: {data['unique_id']}")
                    break

            # Extract amount
            amount_patterns = [
                r'net\s+payable\s+amount\s*(?:rs\.?|inr)?\s*([0-9,.]+)',
                r'payment\s+amount:\s*(?:rs\.?|inr)?\s*([0-9,.]+)',
                r'rs\.\s*([0-9,.]+)',
            ]

            for pattern in amount_patterns:
                amount_match = re.search(pattern, normalized_text, re.IGNORECASE)
                if amount_match:
                    data['receipt_amount'] = amount_match.group(1).strip().replace(',', '')
                    logger.debug(f"Extracted TATA AIG amount: {data['receipt_amount']}")
                    break

            # Extract date
            date_patterns = [
                r'date:\s*(\d{2}/\d{2}/\d{4})',
                r'date:\s*(\d{2}-\d{2}-\d{4})',
                r'value\s+date:\s*(\d{2}-\d{2}-\d{4})',
            ]

            for pattern in date_patterns:
                date_match = re.search(pattern, text, re.IGNORECASE)
                if date_match:
                    data['receipt_date'] = date_match.group(1).strip()
                    logger.debug(f"Extracted TATA AIG date: {data['receipt_date']}")
                    break

            # Extract TDS if available
            tds_patterns = [
                r'tds\s+amount\s*(?:rs\.?|inr)?\s*([0-9,.]+)',
                r'tds\s*:\s*(?:rs\.?|inr)?\s*([0-9,.]+)',
            ]

            for pattern in tds_patterns:
                tds_match = re.search(pattern, normalized_text, re.IGNORECASE)
                if tds_match:
                    data['tds'] = tds_match.group(1).strip().replace(',', '')
                    data['tds_computed'] = 'No'  # TDS already present in the document
                    logger.debug(f"Extracted TATA AIG TDS: {data['tds']}")
                    break

        elif detected_provider == "hdfc_ergo":
            # HDFC ERGO format
            logger.debug("Extracting data using HDFC ERGO patterns")

            # Extract claim/policy number
            claim_patterns = [
                r'claim\s+no\s*\|\s*(\S+)',
                r'policy\s+no\s*\|\s*(\S+)',
                r'C\d+\-\d+',
            ]

            for pattern in claim_patterns:
                claim_match = re.search(pattern, text, re.IGNORECASE)
                if claim_match:
                    if '|' in pattern:
                        data['unique_id'] = claim_match.group(1).strip()
                    else:
                        data['unique_id'] = claim_match.group(0).strip()
                    logger.debug(f"Extracted HDFC ERGO claim number: {data['unique_id']}")
                    break

            # Extract amount
            amount_patterns = [
                r'amount\s*:\s*([0-9,.]+)',
                r'gross\s+amt\s*\|\s*([0-9,.]+)',
                r'net_pay\s*\|\s*([0-9,.]+)',
            ]

            for pattern in amount_patterns:
                amount_match = re.search(pattern, normalized_text, re.IGNORECASE)
                if amount_match:
                    data['receipt_amount'] = amount_match.group(1).strip().replace(',', '')
                    logger.debug(f"Extracted HDFC ERGO amount: {data['receipt_amount']}")
                    break

            # Extract date
            date_patterns = [
                r'value\s+date\s*:?\s*(\d{2}/\d{2}/\d{4})',
                r'value\s+date\s*:?\s*(\d{2}-\d{2}-\d{4})',
                r'date\s*(\d{2}/\d{2}/\d{4})',
            ]

            for pattern in date_patterns:
                date_match = re.search(pattern, text, re.IGNORECASE)
                if date_match:
                    data['receipt_date'] = date_match.group(1).strip()
                    logger.debug(f"Extracted HDFC ERGO date: {data['receipt_date']}")
                    break

            # Extract TDS
            tds_patterns = [
                r'less\s*:\s*tds\s*\|\s*([0-9,.]+)',
                r'tds\s+amount\s*\|\s*([0-9,.]+)',
                r'tds\s+amt\s*\|\s*([0-9,.]+)',
            ]

            for pattern in tds_patterns:
                tds_match = re.search(pattern, normalized_text, re.IGNORECASE)
                if tds_match:
                    data['tds'] = tds_match.group(1).strip().replace(',', '')
                    data['tds_computed'] = 'No'
                    logger.debug(f"Extracted HDFC ERGO TDS: {data['tds']}")
                    break

        elif detected_provider == "future_generali":
            # Future Generali format
            logger.debug("Extracting data using Future Generali patterns")

            # Extract claim/reference number
            claim_patterns = [
                r'our\s+ref\s*:\s*(\S+)',
                r'your\s+ref:\s*(\S+)',
                r'b\d+\s+claims\s+payment',
                r'invoice\s+details\s*.*?(\d+-\d+)',
            ]

            for pattern in claim_patterns:
                claim_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if claim_match:
                    if 'ref' in pattern.lower():
                        data['unique_id'] = claim_match.group(1).strip()
                    elif 'invoice' in pattern.lower():
                        data['unique_id'] = claim_match.group(1).strip()
                    else:
                        data['unique_id'] = claim_match.group(0).strip()
                    logger.debug(f"Extracted Future Generali reference: {data['unique_id']}")
                    break

            # Extract amount
            amount_patterns = [
                r'amount\s*:\s*([0-9,.]+)',
                r'amount\s+in\s+words\s*:\s*.*?([0-9,.]+)',
                r'rupees.*?([0-9,.]+).*?only',
            ]

            for pattern in amount_patterns:
                amount_match = re.search(pattern, normalized_text, re.IGNORECASE)
                if amount_match:
                    data['receipt_amount'] = amount_match.group(1).strip().replace(',', '')
                    logger.debug(f"Extracted Future Generali amount: {data['receipt_amount']}")
                    break

            # Extract date
            date_patterns = [
                r'value\s+date\s*:?\s*(\d{2}/\d{2}/\d{4})',
                r'value\s+date\s*:?\s*(\d{2}-\d{2}-\d{4})',
                r'chq\s*/dd/ft\s+no\s*:.*?value\s+date\s*:(\d{2}/\d{2}/\d{4})',
            ]

            for pattern in date_patterns:
                date_match = re.search(pattern, text, re.IGNORECASE)
                if date_match:
                    data['receipt_date'] = date_match.group(1).strip()
                    logger.debug(f"Extracted Future Generali date: {data['receipt_date']}")
                    break

        elif detected_provider in ["iffco_tokio", "national"]:
            # IFFCO Tokio or National Insurance format
            logger.debug(f"Extracting data using {detected_provider} patterns")

            # Extract claim/policy number
            claim_patterns = [
                r'sub\s+claim\s+no\s*(?:\/|\\|\.)*\s*(\S+)',
                r'policy\s+number\s*(?:\/|\\|\.)*\s*(\S+)',
                r'\d+\d+\d+\-\d+',
            ]

            for pattern in claim_patterns:
                claim_match = re.search(pattern, text, re.IGNORECASE)
                if claim_match:
                    if '(' in pattern:
                        data['unique_id'] = claim_match.group(1).strip()
                    else:
                        data['unique_id'] = claim_match.group(0).strip()
                    logger.debug(f"Extracted {detected_provider} claim number: {data['unique_id']}")
                    break

            # Extract amount
            amount_patterns = [
                r'net\s+paid\s+amount\s*(?:Rs\.?|INR)?\s*([0-9,.]+)',
                r'शुद्ध भुगतान राशि\s*(?:Rs\.?|INR)?\s*([0-9,.]+)',
            ]

            for pattern in amount_patterns:
                amount_match = re.search(pattern, text, re.IGNORECASE)
                if amount_match:
                    data['receipt_amount'] = amount_match.group(1).strip().replace(',', '')
                    logger.debug(f"Extracted {detected_provider} amount: {data['receipt_amount']}")
                    break

            # Extract date
            date_patterns = [
                r'निपटान की तारीख\s*(?:\/|\\|\.)*\s*(\d{2}-\d{2}-\d{4})',
                r'settlement\s+date\s*(?:\/|\\|\.)*\s*(\d{2}-\d{2}-\d{4})',
            ]

            for pattern in date_patterns:
                date_match = re.search(pattern, text, re.IGNORECASE)
                if date_match:
                    data['receipt_date'] = date_match.group(1).strip()
                    logger.debug(f"Extracted {detected_provider} date: {data['receipt_date']}")
                    break

            # Extract TDS
            tds_patterns = [
                r'टीडीएस राशि\s*(?:\/|\\|\.)*\s*([0-9,.]+)',
                r'tds\s+amount\s*(?:\/|\\|\.)*\s*([0-9,.]+)',
            ]

            for pattern in tds_patterns:
                tds_match = re.search(pattern, text, re.IGNORECASE)
                if tds_match:
                    data['tds'] = tds_match.group(1).strip().replace(',', '')
                    data['tds_computed'] = 'No'
                    logger.debug(f"Extracted {detected_provider} TDS: {data['tds']}")
                    break

        elif detected_provider == "liberty":
            # Liberty format
            logger.debug("Extracting data using Liberty patterns")

            # Extract reference number
            ref_patterns = [
                r'your\s+reference\s*:\s*(\d+)',
                r'misc\s+reference\s*:\s*(\d+)',
                r'\d{15,}',
            ]

            for pattern in ref_patterns:
                ref_match = re.search(pattern, text, re.IGNORECASE)
                if ref_match:
                    if '(' in pattern:
                        data['unique_id'] = ref_match.group(1).strip()
                    else:
                        data['unique_id'] = ref_match.group(0).strip()
                    logger.debug(f"Extracted Liberty reference: {data['unique_id']}")
                    break

            # Extract amount
            amount_patterns = [
                r'payment\s+amount\s*:\s*([0-9,.]+)',
                r'payment\s+amount:\s*([0-9,.]+)',
                r'gross\s+amount\s*([0-9,.]+)',
            ]

            for pattern in amount_patterns:
                amount_match = re.search(pattern, normalized_text, re.IGNORECASE)
                if amount_match:
                    data['receipt_amount'] = amount_match.group(1).strip().replace(',', '')
                    logger.debug(f"Extracted Liberty amount: {data['receipt_amount']}")
                    break

            # Extract date
            date_patterns = [
                r'value\s+date\s*:\s*(\d{2}-\d{2}-\d{4})',
                r'date\s*:\s*(\d{2}-\d{2}-\d{4})',
            ]

            for pattern in date_patterns:
                date_match = re.search(pattern, text, re.IGNORECASE)
                if date_match:
                    data['receipt_date'] = date_match.group(1).strip()
                    logger.debug(f"Extracted Liberty date: {data['receipt_date']}")
                    break

            # Extract TDS
            tds_patterns = [
                r'tds\s*([0-9,.]+)',
            ]

            for pattern in tds_patterns:
                tds_match = re.search(pattern, normalized_text, re.IGNORECASE)
                if tds_match:
                    data['tds'] = tds_match.group(1).strip().replace(',', '')
                    data['tds_computed'] = 'No'
                    logger.debug(f"Extracted Liberty TDS: {data['tds']}")
                    break

        elif detected_provider == "universal_sompo":
            # Universal Sompo format
            logger.debug("Extracting data using Universal Sompo patterns")

            # Extract reference number
            ref_patterns = [
                r'client\s+ref\s+no\s*:\s*(\d+)',
                r'payment\s+ref\.\s+no\.\s*:\s*(\d+)',
                r'CL\d+',
            ]

            for pattern in ref_patterns:
                ref_match = re.search(pattern, text, re.IGNORECASE)
                if ref_match:
                    if '(' in pattern:
                        data['unique_id'] = ref_match.group(1).strip()
                    else:
                        data['unique_id'] = ref_match.group(0).strip()
                    logger.debug(f"Extracted Universal Sompo reference: {data['unique_id']}")
                    break

            # Extract amount
            amount_patterns = [
                r'amount\s*:\s*([0-9,.]+)',
                r'(?:Rs\.?|INR)?\s*([0-9,.]+)',
                r'rupees.*?([0-9,.]+).*?only',
            ]

            for pattern in amount_patterns:
                amount_match = re.search(pattern, normalized_text, re.IGNORECASE)
                if amount_match:
                    data['receipt_amount'] = amount_match.group(1).strip().replace(',', '')
                    logger.debug(f"Extracted Universal Sompo amount: {data['receipt_amount']}")
                    break

            # Extract date
            date_patterns = [
                r'value\s+date\s*:?\s*(\d{2}/\d{2}/\d{4})',
                r'value\s+date\s*:?\s*(\d{2}-\d{2}-\d{4})',
                r'date\s*:\s*(\d{2}-\d{2}-\d{4})',
            ]

            for pattern in date_patterns:
                date_match = re.search(pattern, text, re.IGNORECASE)
                if date_match:
                    data['receipt_date'] = date_match.group(1).strip()
                    logger.debug(f"Extracted Universal Sompo date: {data['receipt_date']}")
                    break

        elif detected_provider == "new_india":
            # New India Assurance format
            logger.debug("Extracting data using New India Assurance patterns")

            # Extract claim/policy number
            claim_patterns = [
                r'claim\s+no\s*(?:\/|\\|\.)*\s*(\S+)',
                r'policy\s+no\s*(?:\/|\\|\.)*\s*(\S+)',
                r'invoice\s+number.*?(\d+-\d+)',
                r'\d{12,}',
            ]

            for pattern in claim_patterns:
                claim_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if claim_match:
                    if '(' in pattern:
                        data['unique_id'] = claim_match.group(1).strip()
                    else:
                        data['unique_id'] = claim_match.group(0).strip()
                    logger.debug(f"Extracted New India Assurance claim number: {data['unique_id']}")
                    break

            # Extract amount
            amount_patterns = [
                r'we have instructed our bank to remit an amount of Rs\.([0-9,.]+)',
                r'amount\s*(?:Rs\.?|INR)?\s*([0-9,.]+)',
                r'amt\s*(?:Rs\.?|INR)?\s*([0-9,.]+)',
            ]

            for pattern in amount_patterns:
                amount_match = re.search(pattern, text, re.IGNORECASE)
                if amount_match:
                    data['receipt_amount'] = amount_match.group(1).strip().replace(',', '')
                    logger.debug(f"Extracted New India Assurance amount: {data['receipt_amount']}")
                    break

            # Extract date
            date_match = re.search(r'Date\s*:\s*\w{3}\s+\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\w{3}\s+(\d{4})', text)
            if date_match:
                year = date_match.group(1)
                # Try to find day and month
                date_parts_match = re.search(r'Date\s*:\s*\w{3}\s+(\w{3})\s+(\d{1,2})', text)
                if date_parts_match:
                    month = date_parts_match.group(1)
                    day = date_parts_match.group(2)
                    data['receipt_date'] = f"{day}-{month}-{year}"
                    logger.debug(f"Extracted New India Assurance date: {data['receipt_date']}")

            # Extract TDS if present
            tds_match = re.search(r'(-\d+\.\d+)', text)
            if tds_match:
                tds_value = tds_match.group(1).strip()
                if tds_value.startswith('-'):
                    tds_value = tds_value[1:]  # Remove the minus sign
                    data['tds'] = tds_value
                    data['tds_computed'] = 'No'
                    logger.debug(f"Extracted New India Assurance TDS: {data['tds']}")

        elif detected_provider == "bajaj_allianz":
            # Extract unique ID patterns
            unique_id_patterns = [
                r'(OC-\d{2}-\d{4}-\d{3}-\d+)',  # Matches OC-23-1901-401-301521 format
                r'Claim\s*No\s*[:.]?\s*(OC-\d{2}-\d{4}-\d{3}-\d+)',
                r'Claim\s*No\s*[:.]?\s*(\d+)',
            ]

            for pattern in unique_id_patterns:
                unique_id_match = re.search(pattern, text, re.IGNORECASE)
                if unique_id_match:
                    data['unique_id'] = unique_id_match.group(1).strip()
                    logger.debug(f"Extracted Bajaj Allianz unique ID: {data['unique_id']}")
                    break

        elif detected_provider in ["cholamandalam"]:
            # Bajaj Allianz/Cholamandalam format
            logger.debug(f"Extracting data using {detected_provider} patterns")

            # Extract claim/reference number
            claim_patterns = [
                r'claim\s+no\s*(?:\/|\\|\.)*\s*(\S+)',
                r'customer\s+reference\s*(?:\/|\\|\.)*\s*:\s*(\S+)',
                r'oc-\d+-\d+-\d+',
                r'inv\s+ref:claim\s+no\s*:\s*(\d+)',
                r'settlement\s+reference\s*:\s*(\S+)',
            ]

            for pattern in claim_patterns:
                claim_match = re.search(pattern, text, re.IGNORECASE)
                if claim_match:
                    if '(' in pattern:
                        data['unique_id'] = claim_match.group(1).strip()
                    else:
                        data['unique_id'] = claim_match.group(0).strip()
                    logger.debug(f"Extracted {detected_provider} claim number: {data['unique_id']}")
                    break

            # Extract amount
            amount_patterns = [
                r'remittance\s+amount\s*:\s*([0-9,.]+)',
                r'amount\s*\(inr\)\s*([0-9,.]+)',
                r'amount\s*:\s*([0-9,.]+)',
                r'gross\s+amt\s*(?:ser\s+tax)?\s*([0-9,.]+)',
            ]

            for pattern in amount_patterns:
                amount_match = re.search(pattern, normalized_text, re.IGNORECASE)
                if amount_match:
                    data['receipt_amount'] = amount_match.group(1).strip().replace(',', '')
                    logger.debug(f"Extracted {detected_provider} amount: {data['receipt_amount']}")
                    break

            # Extract date
            date_patterns = [
                r'value\s+date\s*:?\s*(\d{2}-[a-zA-Z]{3}-\d{4})',
                r'value\s+date\s*:?\s*(\d{2}/\d{2}/\d{4})',
                r'value\s+date\s*:?\s*(\d{2}-\d{2}-\d{4})',
                r'date\s*:\s*(\d{2}/\d{1,2}/\d{4})',
                r'advice\s+date\s*:\s*(\d{2}-[a-zA-Z]{3}-\d{4})',
            ]

            for pattern in date_patterns:
                date_match = re.search(pattern, text, re.IGNORECASE)
                if date_match:
                    data['receipt_date'] = date_match.group(1).strip()
                    logger.debug(f"Extracted {detected_provider} date: {data['receipt_date']}")
                    break

            # Extract TDS
            tds_patterns = [
                r'tds\s+amt\s*([0-9,.]+)',
                r'tds\s+amount\s*([0-9,.]+)',
            ]

            for pattern in tds_patterns:
                tds_match = re.search(pattern, normalized_text, re.IGNORECASE)
                if tds_match:
                    data['tds'] = tds_match.group(1).strip().replace(',', '')
                    data['tds_computed'] = 'No'
                    logger.debug(f"Extracted {detected_provider} TDS: {data['tds']}")
                    break

            # If we have not found a unique ID yet, try the invoice number patterns as a fallback
        if 'unique_id' not in data or not data['unique_id']:
            invoice_patterns = [
                r'invoice\s+no\.?\s*:?\s*(\S+)',
                r'bill\s+no\.?\s*:?\s*(\S+)',
                r'invoice\s+number\s*:?\s*(\S+)',
                r'payment\s+details\s+\d+\s*:?\s*(\S+)',
                r'payment\s+details\s+\d+\s*:?\s*(\d{4}-\d{5})',
            ]

            for pattern in invoice_patterns:
                invoice_match = re.search(pattern, text, re.IGNORECASE)
                if invoice_match:
                    data['unique_id'] = invoice_match.group(1).strip()
                    logger.debug(f"Extracted invoice number as unique ID: {data['unique_id']}")
                    break

            # Log extracted data for debugging
        if data:
            logger.debug(f"Provider-specific extraction results: {data}")
        else:
            logger.debug("No data extracted using provider-specific patterns")

        return data

    def extract_data_points(self, text, expected_fields=None):
        """
        Extract required data points from text with enhanced handling for all document types.
        Combines targeted claim number extraction with provider-specific and generic patterns.
        """
        if not text:
            logger.warning("No text provided for data extraction")
            return None, {}, []

        # Log text length for debugging
        logger.info(f"Text length for extraction: {len(text)}")

        # Initialize data extraction results
        data_points = {}
        unique_id = None

        # Check if this is an HSBC/Oriental document
        is_hsbc_doc = any(
            keyword.lower() in text.lower() for keyword in ["HSBC", "Oriental Insurance", "Oriental", "0rienta1"])

        # FIRST PRIORITY: For HSBC/Oriental documents, use the specialized approach
        if is_hsbc_doc:
            logger.info("HSBC/Oriental Insurance document detected, using specialized extraction")

            # Get all ID patterns matching the claim number format
            id_pattern = r'\d+/\d+/\d+/\d+'
            all_ids = re.findall(id_pattern, text)

            if len(all_ids) >= 2:
                # Typically in these documents, the second match is the claim number
                policy_no = all_ids[0]
                claim_no = all_ids[1]

                logger.info(f"Found multiple ID patterns. First: {policy_no}, Second: {claim_no}")

                # We'll set the claim number as our unique ID
                unique_id = claim_no

                # Also check if we can find a direct indicator of the claim number
                policy_label_pos = text.find("Policy no")
                claim_label_pos = text.find("Claim number")

                if policy_label_pos >= 0 and claim_label_pos >= 0:
                    logger.info(f"Found 'Policy no' at {policy_label_pos} and 'Claim number' at {claim_label_pos}")

                    # Get text chunk that should contain both identifiers
                    chunk_text = text[
                                 min(policy_label_pos, claim_label_pos):max(policy_label_pos, claim_label_pos) + 500]

                    # Extract both IDs in the order they appear
                    chunk_ids = re.findall(id_pattern, chunk_text)

                    if len(chunk_ids) >= 2:
                        # Policy number should be first, claim number should be second
                        unique_id = chunk_ids[1]
                        logger.info(f"From chunk analysis, selected claim number: {unique_id}")

            # Extract specific fields using HSBC/Oriental patterns
            if unique_id:
                # Amount patterns
                amount_patterns = [
                    r'Remittance\s+amount\s*:?\s*(?:INR)?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                    r'Amount.*?(\d{6}(?:\.\d{2})?)',
                    r'Debit\s+amount\s*:?\s*(?:INR)?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                ]

                for pattern in amount_patterns:
                    amount_match = re.search(pattern, text, re.IGNORECASE)
                    if amount_match:
                        data_points['receipt_amount'] = amount_match.group(1).strip().replace(',', '')
                        logger.debug(f"Extracted HSBC amount: {data_points['receipt_amount']}")
                        break

                # Date patterns
                date_patterns = [
                    r'Value\s+date\s*:?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
                    r'Advice\s+sending\s+date\s*:?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
                ]

                for pattern in date_patterns:
                    date_match = re.search(pattern, text, re.IGNORECASE)
                    if date_match:
                        data_points['receipt_date'] = date_match.group(1).strip()
                        logger.debug(f"Extracted HSBC date: {data_points['receipt_date']}")
                        break

        # SECOND PRIORITY: If HSBC/Oriental approach didn't work or it's another document type,
        # try the bank-specific extraction approach
        if not unique_id or not data_points:
            logger.info("Using bank-specific extraction approach")
            bank_data = self.extract_bank_specific_data(text)

            if bank_data:
                # Get unique ID if available
                if 'unique_id' in bank_data and bank_data['unique_id']:
                    unique_id = bank_data['unique_id']
                    logger.info(f"Found unique ID from bank-specific extraction: {unique_id}")

                # Copy extracted fields to data_points
                for key, value in bank_data.items():
                    if key != 'unique_id' and value:
                        data_points[key] = value
                        logger.debug(f"Bank-specific extraction provided value for {key}: {value}")

        # THIRD PRIORITY: If bank-specific extraction didn't find everything,
        # use the original field pattern extraction
        if not unique_id or len(data_points) < 2:  # If we're missing ID or most fields
            logger.info("Using pattern-based extraction for missing fields")

            # If we still don't have a unique ID, try generic ID patterns
            if not unique_id:
                # Define patterns for unique identifier
                id_patterns = [
                    # Claim number patterns with higher priority
                    r'Policy\s+no.*?Claim\s+number.*?\n.*?(?:\d+/\d+/\d+/\d+)\s+(\d+/\d+/\d+/\d+)',
                    r'Claim\s+number.*?\n.*?(\d+/\d+/\d+/\d+)',
                    r'Claim\s+number.*?:\s*([A-Z0-9/-]+)',
                    r'Claim#\s*[:.]?\s*([A-Z0-9/_-]+)',
                    r'Claim\s+No\s*[:.]?\s*([A-Z0-9/_-]+)',

                    # Specific format patterns
                    r'(\d{10}C\d{8})',  # UNITED format
                    r'(5004\d{10})',  # UNITED format alternate

                    # Generic claim/invoice numbers
                    r'Claim(?:\s+No\.?|\s*Number|\s*#)(?:[:\s]*|[.\s]*|[:-]\s*)([A-Z0-9-_/]+)',
                    r'Invoice(?:\s+No\.?|\s*Number|\s*#)(?:[:\s]*|[.\s]*|[:-]\s*)([A-Z0-9-_/]+)',

                    # Other reference patterns
                    r'(?:Our|Your)\s+Ref\s*[:.]?\s*([A-Z0-9-_/]+)',
                    r'Policy(?:\s+No\.?|\s*Number|\s*#)(?:[:\s]*|[.\s]*|[:-]\s*)([A-Z0-9-_/]+)',

                    # Generic pattern - use with caution for HSBC documents
                    r'(\d+/\d+/\d+/\d+)',

                    # Invoice patterns for other PDFs
                    r'2425-\d{5}',
                    r'Bill\s+No\.?\s*(\d{4}-\d{5})',
                ]

                # Try to find the unique identifier using patterns
                for pattern in id_patterns:
                    try:
                        if '(' in pattern:  # Pattern has a capture group
                            matches = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                            if matches and matches.group(1):
                                potential_id = matches.group(1).strip()
                                # Don't match on just common labels
                                if potential_id.lower() not in ['invoice', 'claim', 'ref', 'reference']:
                                    # Skip advice reference for HSBC docs
                                    if is_hsbc_doc and "advice reference" in pattern.lower():
                                        continue
                                    unique_id = potential_id
                                    logger.debug(f"Found unique ID using pattern: {unique_id}")
                                    break
                        else:  # Pattern without capture groups
                            matches = re.search(pattern, text, re.IGNORECASE)
                            if matches:
                                unique_id = matches.group(0).strip()
                                logger.debug(f"Found unique ID using direct pattern: {unique_id}")
                                break
                    except (IndexError, AttributeError) as e:
                        logger.warning(f"Error with pattern {pattern}: {str(e)}")
                        continue

            # Define field patterns for missing data points
            field_patterns = {
                # Amount fields (only check if not already found)
                'receipt_amount': [
                    r'(?:Receipt|Payment|Remittance)\s+[Aa]mount\s*[:.]?\s*(?:Rs\.?|INR)?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                    r'Amount\s*[:.]?\s*(?:Rs\.?|INR)?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                    r'Gross\s+Amount\s*[:.]?\s*(?:Rs\.?|INR)?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                    r'[Nn]et\s+[Aa]mount\s*[:.]?\s*(?:Rs\.?|INR)?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                    r'(?:Rs\.?|INR)\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)(?!\s*%)',  # Amount with currency symbol
                    r'Remittance amount\s*:?\s*INR\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                    r'Amount.*?(\d{6}(?:\.\d{2})?)',
                    r'Invoice Amount\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                ],

                # Date fields
                'receipt_date': [
                    r'(?:Receipt|Payment|Value)\s+[Dd]ate\s*[:.]?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
                    r'(?:Receipt|Payment|Value)\s+[Dd]ate\s*[:.]?\s*(\d{2,4}[-/\.]\d{1,2}[-/\.]\d{1,2})',
                    r'[Dd]ate\s*[:.]?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
                    r'[Dd]ate\s*[:.]?\s*(\d{2,4}[-/\.]\d{1,2}[-/\.]\d{1,2})',
                    r'Value\s+Date\s*[:-]?\s*(\d{1,2}[-/\.]?\d{1,2}[-/\.]?\d{2,4})',
                    r'Advice\s+Date\s*[:-]?\s*(\d{1,2}[-/\.]?\d{1,2}[-/\.]?\d{2,4})',
                    r'Value\s+date\s*:?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
                    r'Advice\s+sending\s+date\s*:?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
                    r'Payment\s+Ini\s+Date\s*:?\s*(\d{2}-\d{2}-\d{4})',
                    r'Invoice\s+Date\s*(\d{2}-\d{2}-\d{4})',
                ],

                # TDS fields
                'tds': [
                    r'TDS\s+(?:Amount)?\s*[:.]?\s*(?:Rs\.?|INR)?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                    r'(?:Less|Deduction)[:\s]*TDS\s*[:.]?\s*(?:Rs\.?|INR)?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                    r'TDS\s*[:-]?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                    r'Tax\s*(?:Amount)?\s*[:.]?\s*(?:Rs\.?|INR)?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                ],
            }

            # Extract data points using patterns only if they're not already set
            for field, pattern_list in field_patterns.items():
                if field not in data_points or not data_points[field]:
                    for pattern in pattern_list:
                        matches = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                        if matches:
                            if '(' in pattern:
                                value = matches.group(1).strip()
                            else:
                                value = matches.group(0).strip()

                            # Clean up the value
                            if field == 'receipt_amount' or field == 'tds':
                                value = re.sub(r'[^0-9.]', '', value)

                            data_points[field] = value
                            logger.debug(f"Extracted {field}: {value} using pattern")
                            break

        # FOURTH PRIORITY: If still missing fields, try to extract using generic patterns
        if not unique_id or len(data_points) < 2:
            logger.info("Using generic pattern extraction for remaining fields")
            generic_data = self.extract_generic_data(text)

            # Get unique ID if still missing
            if not unique_id and 'unique_id' in generic_data and generic_data['unique_id']:
                unique_id = generic_data['unique_id']
                logger.info(f"Found unique ID using generic extraction: {unique_id}")

            # Fill in missing fields from generic extraction
            for key, value in generic_data.items():
                if key != 'unique_id' and value and (key not in data_points or not data_points[key]):
                    data_points[key] = value
                    logger.debug(f"Generic extraction provided value for {key}: {value}")

        # FIFTH PRIORITY: Use ML model to enhance extraction if available
        if self.ml_model and self.use_ml:
            try:
                # Use model to predict additional fields or correct existing ones
                ml_enhanced_data = self.ml_model.predict([text])[0]

                # Update data points with ML predictions where standard patterns failed
                for field in ml_enhanced_data:
                    if field not in data_points or not data_points[field]:
                        data_points[field] = ml_enhanced_data[field]
                        logger.debug(f"ML model provided value for {field}: {ml_enhanced_data[field]}")
            except Exception as e:
                logger.error(f"Error using ML model: {str(e)}")

        # Extract table data which might contain multiple entries
        table_data = self.extract_table_data(text)

        # FINAL VALIDATION for HSBC documents
        if is_hsbc_doc and "Policy no" in text and "Claim number" in text:
            # Look for the specific pattern in HSBC documents
            claim_pattern = r'(\d+/\d+/\d+/\d+)'
            claim_matches = re.findall(claim_pattern, text)

            if len(claim_matches) >= 2:
                # For documents with policy/claim numbers in a table
                policy_no = claim_matches[0]
                claim_no = claim_matches[1]

                # Check if our unique ID is wrong or missing
                if not unique_id or unique_id == policy_no:
                    logger.warning(f"Final validation correction: changing {unique_id} to {claim_no}")
                    unique_id = claim_no

        # Add TDS computation if needed
        if 'receipt_amount' in data_points and data_points['receipt_amount'] and \
                ('tds' not in data_points or not data_points['tds']):
            try:
                amount = float(data_points['receipt_amount'])
                tds, is_computed = self.compute_tds(amount, text)
                if is_computed:
                    data_points['tds'] = str(tds)
                    data_points['tds_computed'] = 'Yes'
                    logger.info(f"TDS computed: {data_points['tds']}")
            except Exception as e:
                logger.error(f"Error computing TDS: {str(e)}")

        # Validate extracted data before returning
        self.validate_extracted_data(unique_id, data_points)

        # Return the results
        return unique_id, data_points, table_data

    def validate_extracted_data(self, unique_id, data_points):
        """
        Validate extracted data to ensure correct format.

        Args:
            unique_id (str): The extracted unique identifier
            data_points (dict): Dictionary of extracted data points
        """
        # Validate and correct receipt amount
        if 'receipt_amount' in data_points:
            amount = data_points['receipt_amount']
            # Clean up and ensure proper format
            try:
                # Remove any non-numeric characters except dot
                amount = re.sub(r'[^0-9.]', '', amount)
                # Convert to float and back to string to ensure valid number
                amount_float = float(amount)
                # If amount seems too small for a payment (e.g., 201 instead of 201156)
                if len(amount) <= 3 and '.' not in amount:
                    logger.warning(f"Receipt amount {amount} seems too small, may be incomplete")
                # Update with cleaned value
                data_points['receipt_amount'] = str(amount_float)
                logger.debug(f"Validated receipt amount: {data_points['receipt_amount']}")
            except ValueError:
                logger.warning(f"Invalid receipt amount format: {amount}")

        # Validate receipt date
        if 'receipt_date' in data_points:
            date_str = data_points['receipt_date']
            # Try to parse and standardize the date format
            try:
                # Different date formats we might encounter
                date_formats = [
                    '%d %b %Y',  # 11 Feb 2025
                    '%d-%m-%Y',  # 12-02-2025
                    '%d/%m/%Y',  # 12/02/2025
                    '%Y-%m-%d',  # 2025-02-12
                    '%Y/%m/%d',  # 2025/02/12
                    '%d.%m.%Y',  # 12.02.2025
                    '%b %d, %Y',  # Feb 12, 2025
                ]

                parsed_date = None
                for date_format in date_formats:
                    try:
                        parsed_date = datetime.strptime(date_str, date_format)
                        break
                    except ValueError:
                        continue

                if parsed_date:
                    # Standardize to dd-mm-yyyy format
                    data_points['receipt_date'] = parsed_date.strftime('%d-%m-%Y')
                    logger.debug(f"Validated receipt date: {data_points['receipt_date']}")
                else:
                    logger.warning(f"Could not parse date format: {date_str}")
            except Exception as e:
                logger.warning(f"Error validating date {date_str}: {str(e)}")

        # Log validation results
        logger.info(
            f"Data validation complete. Unique ID: {unique_id}, Fields: {', '.join(data_points.keys())}")

    # Enhancement 5: Updated extract_table_data to handle more table formats
    def extract_table_data(self, text):
        """
        Extract data from tables in the PDF with enhanced pattern recognition.
        """
        table_data = []

        # Check for insurance payment advice formats
        # 1. HSBC/Oriental format table
        if "HSBC" in text or "Oriental Insurance" in text or "claim payment" in text.lower():
            table_match = re.search(r'Policy\s*no\s+Claim\s*number.*?(\n.*?\d+/\d+/\d+/\d+.*?\d+)', text,
                                    re.IGNORECASE | re.DOTALL)
            if table_match:
                # Extract row data
                row_text = table_match.group(1).strip()
                policy_match = re.search(r'(\d+/\d+/\d+/\d+)', row_text)
                claim_match = re.search(r'\d+/\d+/\d+/\d+\s+(\d+/\d+/\d+/\d+)', row_text)
                amount_match = re.search(r'(\d+)$', row_text)

                if claim_match:
                    row_data = {
                        'unique_id': claim_match.group(1).strip(),
                        'policy_no': policy_match.group(1).strip() if policy_match else "",
                    }

                    if amount_match:
                        row_data['receipt_amount'] = amount_match.group(1).strip()

                    table_data.append(row_data)

        # 2. United India & Other Structured Tables
        # Look for tables with columns for number, date, amount
        table_headers = re.search(r'(invoice.*?(?:number|no|date|amount|net|gross).*?(?:tds|tax)?.*?)\n', text,
                                  re.IGNORECASE)

        if table_headers:
            headers = table_headers.group(1).lower()
            header_positions = {}

            # Map header positions
            for header in ["invoice", "claim", "date", "amount", "tds"]:
                pos = headers.find(header)
                if pos >= 0:
                    header_positions[header] = pos

            # Get content after headers
            content_after_headers = text[table_headers.end():].strip()
            rows = content_after_headers.split('\n')

            # Process each row based on header positions
            for row in rows[:20]:  # Limit to first 20 rows for performance
                if not row.strip() or not any(c.isdigit() for c in row):
                    continue

                row_data = {}

                # Extract data based on header positions
                for header, pos in header_positions.items():
                    if header in ["invoice", "claim"]:
                        # Look for ID patterns
                        id_match = re.search(r'(\d{4}-\d{5}|\d{4,19}|[A-Z0-9-_/]+)', row)
                        if id_match:
                            row_data['unique_id'] = id_match.group(1).strip()
                    elif header == "date":
                        # Look for date patterns
                        date_match = re.search(r'(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{2,4}[-/\.]\d{1,2}[-/\.]\d{1,2})',
                                               row)
                        if date_match:
                            row_data['receipt_date'] = date_match.group(1).strip()
                    elif header == "amount":
                        # Look for amount patterns
                        amount_match = re.search(r'(?:Rs\.?|INR)?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)', row)
                        if amount_match:
                            row_data['receipt_amount'] = amount_match.group(1).strip()
                    elif header == "tds":
                        # Look for TDS amount
                        tds_match = re.search(r'(\d+(?:,\d{3})*(?:\.\d{1,2})?)', row)
                        if tds_match:
                            row_data['tds'] = tds_match.group(1).strip()
                            row_data['tds_computed'] = 'No'

                # Only add row if it has sufficient data
                if 'unique_id' in row_data and ('receipt_date' in row_data or 'receipt_amount' in row_data):
                    table_data.append(row_data)

        # 3. Cera Sanitaryware multi-invoice table
        if "cera sanitaryware" in text.lower() or "gross amount" in text.lower():
            # Look for table with Document No., Bill No., Bill Date, Amount, and TDS columns
            bill_rows = re.findall(
                r'(\d{10,11})\s+(2425-\d{5})\s+(\d{2}\.\d{2}\.\d{4})\s+(\d+\.\d{2})\s+(\d+\.\d{2})\s+(\d+\.\d{2})',
                text)

            for row in bill_rows:
                row_data = {
                    'unique_id': row[1].strip(),  # Bill No. as unique ID
                    'document_no': row[0].strip(),
                    'receipt_date': row[2].strip(),
                    'gross_amount': row[3].strip(),
                    'tds': row[4].strip(),
                    'receipt_amount': row[5].strip(),
                    'tds_computed': 'No'
                }
                table_data.append(row_data)

        # 4. Zion reference table (spotted in sample 34)
        zion_match = re.search(r'DATE\s+ZION\s+REF\s+NO\.\s+bill\s+no\s+tds\s+amt', text, re.IGNORECASE)
        if zion_match:
            # Extract rows in format: DATE | ZION REF NO. | bill no | tds | amt
            zion_rows = re.findall(r'(\d{1,2}-\d{1,2}-\d{4})\s+(\S+)\s+(2425-\d{5})\s+(\d+)\s+(\d+)', text)

            for row in zion_rows:
                row_data = {
                    'unique_id': row[2].strip(),  # Bill No. as unique ID
                    'receipt_date': row[0].strip(),
                    'reference_no': row[1].strip(),
                    'tds': row[3].strip(),
                    'receipt_amount': row[4].strip(),
                    'tds_computed': 'No'
                }
                table_data.append(row_data)

        # 5. Special handling for Bajaj Allianz style table
        bajaj_table_pattern = r'(OC-\d{2}-\d{4}-\d{3}-\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)'
        bajaj_matches = re.findall(bajaj_table_pattern, text)

        for match in bajaj_matches:
            row_data = {
                'unique_id': match[0],
                'claim_no': match[1],
                'gross_amt': match[2],
                'ser_tax': match[3],
                'tds_amt': match[4]
            }
            table_data.append(row_data)

        # 6. Look for table with invoice numbers and amounts (generic pattern)
        invoice_rows = re.findall(r'(2425[-\s]\d{5})[^\n]*?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)', text)
        for row in invoice_rows:
            # Check if this invoice is already in table_data
            if not any(td.get('unique_id') == row[0].strip() for td in table_data):
                row_data = {
                    'unique_id': row[0].strip().replace(' ', '-'),  # Clean up any spacing issues
                    'receipt_amount': row[1].strip().replace(',', '')
                }

                # Try to find a date near this invoice
                context = text[max(0, text.find(row[0]) - 50):min(len(text), text.find(row[0]) + 100)]
                date_match = re.search(r'(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})', context)
                if date_match:
                    row_data['receipt_date'] = date_match.group(1).strip()

                table_data.append(row_data)

        # 7. Generic pattern for invoice numbers followed by amounts
        generic_rows = re.findall(r'(?:Invoice|Bill|Ref).*?(\d{4}-\d{5}).*?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)', text,
                                  re.IGNORECASE)
        for row in generic_rows:
            # Check if this invoice is already in table_data
            if not any(td.get('unique_id') == row[0].strip() for td in table_data):
                row_data = {
                    'unique_id': row[0].strip(),
                    'receipt_amount': row[1].strip().replace(',', '')
                }
                table_data.append(row_data)

        # Log extracted table data
        if table_data:
            logger.info(f"Extracted {len(table_data)} table rows")
            for i, row in enumerate(table_data):
                logger.debug(f"Table row {i + 1}: {row}")
        else:
            logger.info("No table data found")

        return table_data

    def process_pdf(self, pdf_path, expected_fields=None):
        """
        Process a single PDF and extract relevant data.

        Args:
            pdf_path (str): Path to the PDF file
            expected_fields (list): List of expected field names

        Returns:
            tuple: (unique_id, data_points_dict, table_data)
        """
        if not os.path.exists(pdf_path):
            logger.error(f"File not found: {pdf_path}")
            return None, {}, []

        # Extract text from PDF
        extracted_text = self.extract_text(pdf_path)

        if not extracted_text:
            logger.warning(f"No text extracted from {pdf_path}")
            return None, {}, []

        # Extract unique ID, data points, and table data
        unique_id, data_points, table_data = self.extract_data_points(extracted_text, expected_fields)

        # Log the extracted data
        logger.info(f"Extracted data from {pdf_path}: ID={unique_id}, Fields={len(data_points)}")
        if table_data:
            logger.info(f"Found {len(table_data)} table rows")

        return unique_id, data_points, table_data