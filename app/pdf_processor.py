import os
import re
import PyPDF2
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import pandas as pd
import spacy
import joblib
import logging
from pathlib import Path
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pdf_processing.log")
    ]
)
logger = logging.getLogger("pdf_processor")


class PDFProcessor:
    def __init__(self, use_ml=True):
        """
        Initialize the PDF processor.

        Args:
            use_ml (bool): Whether to use ML model for extraction enhancement
        """
        self.use_ml = use_ml

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

    def extract_text_pypdf2(self, pdf_path):
        """Extract text from PDF using PyPDF2."""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Error using PyPDF2 on {pdf_path}: {str(e)}")
        return text.strip()

    def extract_text_pdfplumber(self, pdf_path):
        """Extract text from PDF using pdfplumber."""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.error(f"Error using pdfplumber on {pdf_path}: {str(e)}")
        return text.strip()

    def extract_text_ocr(self, pdf_path):
        """Extract text from PDF using OCR (Tesseract)."""
        text = ""
        try:
            # Create a temporary directory for storing images
            with tempfile.TemporaryDirectory() as temp_dir:
                # Convert PDF to images
                images = convert_from_path(pdf_path, output_folder=temp_dir)

                # Extract text from each image
                for i, image in enumerate(images):
                    page_text = pytesseract.image_to_string(image)
                    text += page_text + "\n"
        except Exception as e:
            logger.error(f"Error using OCR on {pdf_path}: {str(e)}")
        return text.strip()

    def extract_text(self, pdf_path):
        """
        Extract text from PDF using multiple methods and combine results.

        Returns:
            str: Combined extracted text
        """
        logger.info(f"Processing PDF: {pdf_path}")

        # Try multiple extraction methods
        text_pypdf2 = self.extract_text_pypdf2(pdf_path)
        text_pdfplumber = self.extract_text_pdfplumber(pdf_path)

        # If both methods fail or have very little text, use OCR
        if (len(text_pypdf2) < 100 and len(text_pdfplumber) < 100):
            logger.info("Standard extraction methods yielded limited text. Using OCR.")
            text_ocr = self.extract_text_ocr(pdf_path)
            # Combine all text (with more weight to OCR results in this case)
            combined_text = text_ocr + "\n" + text_pypdf2 + "\n" + text_pdfplumber
        else:
            # Combine results from PyPDF2 and pdfplumber
            combined_text = text_pypdf2 + "\n" + text_pdfplumber

        # Clean up the text
        combined_text = self.clean_text(combined_text)

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

    def extract_data_points(self, text, expected_fields=None):
        """
        Extract required data points from text.

        Args:
            text (str): Extracted text from PDF
            expected_fields (list): List of expected field names

        Returns:
            tuple: (unique_id, data_points_dict)
        """
        if not text:
            return None, {}

        # Parse the text with spaCy for better entity recognition
        doc = self.nlp(text)

        # Initialize data extraction results
        data = {
            'unique_id': None,
            'data_points': {}
        }

        # Define patterns for unique identifier
        # Update these patterns based on your specific data format
        id_patterns = [
            r'ID[:\s]*([A-Z0-9-]+)',
            r'Identifier[:\s]*([A-Z0-9-]+)',
            r'Reference[:\s]*([A-Z0-9-]+)',
            r'Document\s*ID[:\s]*([A-Z0-9-]+)',
            r'Invoice\s*Number[:\s]*([A-Z0-9-]+)',
            r'Case\s*Number[:\s]*([A-Z0-9-]+)',
            # Add more patterns as needed for your use case
        ]

        # Try to find the unique identifier using patterns
        for pattern in id_patterns:
            matches = re.search(pattern, text, re.IGNORECASE)
            if matches:
                data['unique_id'] = matches.group(1).strip()
                break

        # If no ID found, try to use NLP entities
        if not data['unique_id']:
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'NORP']:
                    # Check if entity contains alphanumeric characters
                    potential_id = ''.join(c for c in ent.text if c.isalnum())
                    if len(potential_id) >= 4 and any(c.isdigit() for c in potential_id):
                        data['unique_id'] = ent.text.strip()
                        break

        # Define patterns for expected data fields
        # Customize these patterns based on the expected data in your PDFs
        field_patterns = {
            'date': r'Date[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]+\s+\d{2,4})',
            'amount': r'Amount[:\s]*[\$£€]?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            'name': r'Name[:\s]*([A-Za-z\s]+)',
            'address': r'Address[:\s]*([A-Za-z0-9\s,]+)',
            'contact': r'(?:Contact|Phone|Tel)[:\s]*(\+?[\d\-\(\)\s]{7,})',
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            # Add more fields as needed
        }

        # Extract data points using patterns
        for field, pattern in field_patterns.items():
            matches = re.search(pattern, text, re.IGNORECASE)
            if matches:
                data['data_points'][field] = matches.group(1).strip()

        # Check for email addresses specifically
        email_matches = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
        if email_matches:
            data['data_points']['email'] = email_matches[0]

        # Use ML model to enhance extraction (if available)
        if self.ml_model and self.use_ml:
            try:
                # Use model to predict additional fields or correct existing ones
                ml_enhanced_data = self.ml_model.predict([text])[0]

                # Update data points with ML predictions where standard patterns failed
                for field in ml_enhanced_data:
                    if field not in data['data_points'] or not data['data_points'][field]:
                        data['data_points'][field] = ml_enhanced_data[field]
            except Exception as e:
                logger.error(f"Error using ML model: {str(e)}")

        return data['unique_id'], data['data_points']

    def process_pdf(self, pdf_path, expected_fields=None):
        """
        Process a single PDF and extract relevant data.

        Args:
            pdf_path (str): Path to the PDF file
            expected_fields (list): List of expected field names

        Returns:
            tuple: (unique_id, data_points_dict)
        """
        if not os.path.exists(pdf_path):
            logger.error(f"File not found: {pdf_path}")
            return None, {}

        # Extract text from PDF
        extracted_text = self.extract_text(pdf_path)

        if not extracted_text:
            logger.warning(f"No text extracted from {pdf_path}")
            return None, {}

        # Extract unique ID and data points
        unique_id, data_points = self.extract_data_points(extracted_text, expected_fields)

        # Log the extracted data
        logger.info(f"Extracted data from {pdf_path}: ID={unique_id}, Fields={len(data_points)}")

        return unique_id, data_points