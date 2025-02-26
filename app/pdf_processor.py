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
            tuple: (unique_id, data_points_dict, table_data)
        """
        if not text:
            return None, {}, []

        # Parse the text with spaCy for better entity recognition
        doc = self.nlp(text)

        # Initialize data extraction results
        data = {
            'unique_id': None,
            'data_points': {}
        }

        # Define patterns for unique identifier (based on sample PDFs)
        id_patterns = [
            # Claim numbers
            r'Claim(?:\s+No\.?|\s*Number|\s*#)(?:[:\s]*|[.\s]*|[:-]\s*)([A-Z0-9-_/]+)',
            r'ClaimNo\s*[:.]?\s*([A-Z0-9-_/]+)',
            r'Sub\s+Claim\s+No.*?[:]\s*([A-Z0-9-_/]+)',
            r'CLAIM_REF_NO\s*([A-Z0-9-_/]+)',
            # Invoice/reference numbers
            r'(?:Invoice|Ref|Reference)(?:\s+No\.?|\s*Number|\s*#)(?:[:\s]*|[.\s]*|[:-]\s*)([A-Z0-9-_/]+)',
            r'(?:Our|Your)\s+Ref\s*[:.]?\s*([A-Z0-9-_/]+)',
            r'Msg\s+Ref\s+Number\s*[:]\s*([A-Z0-9-_/]+)',
            # Policy numbers
            r'Policy(?:\s+No\.?|\s*Number|\s*#)(?:[:\s]*|[.\s]*|[:-]\s*)([A-Z0-9-_/]+)',
            r'PolicyNo\s*[:.]?\s*([A-Z0-9-_/]+)',
            # Bank reference numbers
            r'Bank\s+Reference\s*[:]\s*([A-Z0-9-_/]+)',
            r'Bank\s+Ref\s+No\s*[:]\s*([A-Z0-9-_/]+)',
            r'UTR\s+Number\s*[:]\s*([A-Z0-9-_/]+)',
            r'UETR\s*[:]\s*([A-Z0-9-_/]+)',
            # Settlement references
            r'SETTLEMENT\s+REFERENCE\s*[:]\s*([A-Z0-9-_/]+)',
            r'(?:Settlement|Clearing)\s+(?:document|No|Reference)\s*[:.]?\s*([A-Z0-9-_/]+)',
        ]

        # Try to find the unique identifier using patterns
        for pattern in id_patterns:
            matches = re.search(pattern, text, re.IGNORECASE)
            if matches:
                data['unique_id'] = matches.group(1).strip()
                break

        # If no ID found through patterns, try to use NLP entities
        if not data['unique_id']:
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'NORP', 'CARDINAL']:
                    # Check if entity contains alphanumeric characters
                    potential_id = ''.join(c for c in ent.text if c.isalnum())
                    if len(potential_id) >= 4 and any(c.isdigit() for c in potential_id):
                        data['unique_id'] = ent.text.strip()
                        break

        # Define patterns for expected data fields
        field_patterns = {
            # Amount fields
            'receipt_amount': [
                r'(?:Receipt|Payment|Remittance)\s+[Aa]mount\s*[:.]?\s*(?:Rs\.?|INR)?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                r'Amount\s*[:.]?\s*(?:Rs\.?|INR)?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                r'Gross\s+Amount\s*[:.]?\s*(?:Rs\.?|INR)?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                r'[Nn]et\s+[Aa]mount\s*[:.]?\s*(?:Rs\.?|INR)?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                r'(?:Rs\.?|INR)\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)(?!\s*%)',  # Amount with currency symbol
            ],

            # Date fields
            'receipt_date': [
                r'(?:Receipt|Payment|Value)\s+[Dd]ate\s*[:.]?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
                r'(?:Receipt|Payment|Value)\s+[Dd]ate\s*[:.]?\s*(\d{2,4}[-/\.]\d{1,2}[-/\.]\d{1,2})',
                r'[Dd]ate\s*[:.]?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
                r'[Dd]ate\s*[:.]?\s*(\d{2,4}[-/\.]\d{1,2}[-/\.]\d{1,2})',
                r'Value\s+Date\s*[:-]?\s*(\d{1,2}[-/\.]?\d{1,2}[-/\.]?\d{2,4})',
                r'Advice\s+Date\s*[:-]?\s*(\d{1,2}[-/\.]?\d{1,2}[-/\.]?\d{2,4})',
                r'(?:Document|Bill)\s+(?:No\.|Date).*?(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
            ],

            # TDS fields
            'tds': [
                r'TDS\s+(?:Amount)?\s*[:.]?\s*(?:Rs\.?|INR)?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                r'(?:Less|Deduction)[:\s]*TDS\s*[:.]?\s*(?:Rs\.?|INR)?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                r'TDS\s*[:-]?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
            ],

            # Additional fields that might be useful
            'invoice_no': [
                r'Invoice\s+(?:No\.|Number)\s*[:.]?\s*([A-Z0-9-_/]+)',
                r'Bill\s+No\.\s*[:.]?\s*([A-Z0-9-_/]+)',
                r'InvoiceNo\s*[:.]?\s*([A-Z0-9-_/]+)',
            ],

            'client_ref': [
                r'Claim\s+(?:No\.|Number)\s*[:.]?\s*([A-Z0-9-_/]+)',
                r'Reference\s+(?:No\.|Number)\s*[:.]?\s*([A-Z0-9-_/]+)',
                r'Ref\s+(?:No\.|Number)\s*[:.]?\s*([A-Z0-9-_/]+)',
            ],
        }

        # Extract data points using patterns
        for field, pattern_list in field_patterns.items():
            for pattern in pattern_list:
                matches = re.search(pattern, text, re.IGNORECASE)
                if matches:
                    value = matches.group(1).strip()
                    data['data_points'][field] = value
                    break  # Use the first successful pattern match for each field

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

        # Try to extract data from tables in the PDF
        table_data = self.extract_table_data(text)

        # Return the results
        return data['unique_id'], data['data_points'], table_data

    def extract_table_data(self, text):
        """
        Extract data from tables in the PDF.

        Args:
            text (str): Extracted text from PDF

        Returns:
            list: List of dictionaries, each representing a row of data
        """
        table_data = []

        # Look for table-like structures in the text
        # Common patterns in financial documents include tables with claim numbers, dates, and amounts

        # Pattern 1: Try to find tables with claim/invoice numbers followed by dates and amounts
        table_pattern = r'(?:Claim(?:\s+No\.?|\s*Number|\s*#)|Invoice(?:\s+No\.?|\s*Number|\s*#))\s*.*?\n((?:.*?\d+.*?\n)+)'
        table_matches = re.findall(table_pattern, text, re.IGNORECASE)

        for table_text in table_matches:
            rows = table_text.strip().split('\n')
            for row in rows:
                if not row.strip():
                    continue

                # Try to extract data from each row
                row_data = {}

                # Look for claim/invoice number
                claim_match = re.search(r'([A-Z0-9-_/]+)', row)
                if claim_match:
                    row_data['unique_id'] = claim_match.group(1).strip()

                # Look for date
                date_match = re.search(r'(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{2,4}[-/\.]\d{1,2}[-/\.]\d{1,2})', row)
                if date_match:
                    row_data['receipt_date'] = date_match.group(1).strip()

                # Look for amount
                amount_match = re.search(r'(?:Rs\.?|INR)?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)', row)
                if amount_match:
                    row_data['receipt_amount'] = amount_match.group(1).strip()

                # Look for TDS
                tds_match = re.search(r'TDS\s*:?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)', row, re.IGNORECASE)
                if tds_match:
                    row_data['tds'] = tds_match.group(1).strip()
                    row_data['tds_computed'] = 'No'

                # Only add the row if we have at least a unique ID and one data point
                if 'unique_id' in row_data and (
                        'receipt_date' in row_data or 'receipt_amount' in row_data or 'tds' in row_data):
                    table_data.append(row_data)

        # Pattern 2: Try to find table with headers and values (like in sample PDF 3)
        # This assumes a more structured table with columns
        header_pattern = r'((?:Claim|Invoice).*?(?:Amount|Date).*?(?:TDS|Tax).*?)\n'
        header_match = re.search(header_pattern, text, re.IGNORECASE)

        if header_match:
            headers = header_match.group(1).strip()

            # Try to identify column positions based on headers
            claim_pos = headers.lower().find('claim')
            invoice_pos = headers.lower().find('invoice')
            date_pos = headers.lower().find('date')
            amount_pos = headers.lower().find('amount')
            tds_pos = headers.lower().find('tds')

            # Get the table content after the header
            table_content = text[header_match.end():].strip()

            # Split into rows
            data_rows = table_content.split('\n')

            for row in data_rows:
                if not row.strip() or not re.search(r'\d+', row):
                    continue

                row_data = {}

                # Extract data based on column positions
                if claim_pos >= 0:
                    # Extract claim number (typically at the beginning of the row or at claim_pos)
                    claim_match = re.search(r'([A-Z0-9-_/]+)', row)
                    if claim_match:
                        row_data['unique_id'] = claim_match.group(1).strip()

                if invoice_pos >= 0 and invoice_pos != claim_pos:
                    # Extract invoice number
                    invoice_part = row[invoice_pos:].split()[0] if invoice_pos < len(row) else ""
                    if invoice_part:
                        row_data['invoice_no'] = invoice_part.strip()

                if date_pos >= 0:
                    # Extract date
                    date_match = re.search(r'(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{2,4}[-/\.]\d{1,2}[-/\.]\d{1,2})',
                                           row)
                    if date_match:
                        row_data['receipt_date'] = date_match.group(1).strip()

                if amount_pos >= 0:
                    # Extract amount
                    amount_match = re.search(r'(?:Rs\.?|INR)?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)', row)
                    if amount_match:
                        row_data['receipt_amount'] = amount_match.group(1).strip()

                if tds_pos >= 0:
                    # Extract TDS
                    tds_part = row[tds_pos:].split()[0] if tds_pos < len(row) else ""
                    if tds_part and re.match(r'\d', tds_part):
                        row_data['tds'] = tds_part.strip()
                        row_data['tds_computed'] = 'No'

                # Only add the row if we have at least a unique ID and one data point
                if ('unique_id' in row_data or 'invoice_no' in row_data) and (
                        'receipt_date' in row_data or 'receipt_amount' in row_data or 'tds' in row_data):
                    table_data.append(row_data)

        # Pattern 3: Look for specific table patterns in the sample PDFs
        # This pattern is especially for documents like sample PDF 11 with a list of claim numbers and amounts
        claim_list_pattern = r'((?:Claim\s+Number|Invoice\s+Number)(?:.*?\n)+(?:\d+.*?\n)+)'
        claim_list_matches = re.findall(claim_list_pattern, text, re.IGNORECASE)

        for claim_list in claim_list_matches:
            lines = claim_list.strip().split('\n')

            # Skip the header line
            for line in lines[1:]:
                if not re.search(r'\d+', line):
                    continue

                row_data = {}

                # Extract claim/invoice number
                claim_match = re.search(r'([A-Z0-9-_/]+)', line)
                if claim_match:
                    row_data['unique_id'] = claim_match.group(1).strip()

                # Extract date
                date_match = re.search(r'(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{2,4}[-/\.]\d{1,2}[-/\.]\d{1,2})', line)
                if date_match:
                    row_data['receipt_date'] = date_match.group(1).strip()

                # Extract amount - look for numbers with decimal points as they're likely amounts
                amount_matches = re.findall(r'(\d+(?:,\d{3})*\.\d{2})', line)
                if amount_matches:
                    # First amount is usually the receipt amount
                    if len(amount_matches) >= 1:
                        row_data['receipt_amount'] = amount_matches[0].strip()

                    # Second amount is often the TDS
                    if len(amount_matches) >= 2:
                        row_data['tds'] = amount_matches[1].strip()
                        row_data['tds_computed'] = 'No'

                # Only add the row if we have at least a unique ID and one data point
                if 'unique_id' in row_data and (
                        'receipt_date' in row_data or 'receipt_amount' in row_data or 'tds' in row_data):
                    table_data.append(row_data)

        return table_data

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