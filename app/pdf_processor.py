import os
import re
import joblib
import logging
import pandas as pd
import spacy
from datetime import datetime
import numpy as np
import concurrent.futures
import json

# Import unified PDF text extractor
from app.utils.pdf_text_extractor import PDFTextExtractor

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

        # Initialize the unified PDF text extractor
        self.text_extractor = PDFTextExtractor(
            poppler_path=self.poppler_path,
            debug_mode=self.debug_mode
        )

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

    def extract_text(self, pdf_path):
        """
        Extract text from PDF using the unified PDF text extractor.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            str: Extracted text
        """
        logger.info(f"Processing PDF: {pdf_path}")

        # Use the unified text extractor
        extracted_text = self.text_extractor.extract_from_file(pdf_path)

        # Save combined text for debugging
        if self.debug_mode and self.debug_dir:
            debug_text_path = os.path.join(self.debug_dir, f"{os.path.basename(pdf_path)}_combined_text.txt")
            with open(debug_text_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            logger.debug(f"Saved combined extracted text: {debug_text_path}")

        return extracted_text

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
            "bajaj_allianz": ["bajaj allianz", "bajaj", "allianz", "BAJAJ", "ALLIANZ", "BAJAJ ALLIANZ",
                              "BAJAJ ALLIANZ GENERAL", "BAJAJ ALLIANZ GENERAL IN",
                              "BAJAJ ALLIANZ GENERAL IN***********"],
            "cera": ["CERA", "Cera", "Sanitaryware", "cera", "sanitaryware", "Cera Sanitaryware Ltd",
                     "CERA Sanitaryware Limited"],
            "cholamandalam": ["cholamandalam", "ms genera", "cholamandalam ms genera", "CHOLAMANDALAM MS GENERA",
                              "CHOLAMANDALAM MS GENERA***********"],
            "future_generali": ["FUTURE GENERALI INDIA INSURANCE CO", "FUTURE", "GENERALI", "future generali",
                                "future", "generali"],
            "hdfc_ergo": ["HDFC ERGO GENERAL INSURANCE COM LTD", "HDFC", "ERGO", "HDFC ERGO", "hdfc ergo", "ergo"],
            "icici_lombard": ["ICICI Lombard", "ICICI LOMBARD", "ICICI", "Lombard", "LOMBARD", "CLAIM_REF_NO",
                              "LAE Invoice No"],
            "iffco_tokio": ["IFFCO", "IFFCO TOKIO", "iffco", "iffco tokio", "tokio", "TOKIO", "IFFCO TOWER", "TOWER",
                            "iffco tower", "tower", "ITG", "IFFCOXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"],
            "liberty": ["liberty", "LIBERTY", "liber", "LIBER", "LIBERXXXXXXXXXXXXXXXXXXXXXXXX"],
            "new_india": ["new india", "newindia.co.in", "NIAHO@newindia.co.in", "The New India Assurance Co. Ltd",
                          "The New India Assurance Co. Ltd(510000)"],
            "national": ["national insurance", " National Insurance", " National Insurance Company Limited"],
            "oriental": ["oriental insurance", "oicl", "the oriental insurance", "Oriental Insurance Co Ltd",
                         "the Oriental Insurance Co Ltd"],
            "rgcargo": ["RG Cargo Services Private Limited", "rg cargo services private limited", "RG Cargo",
                        "rg cargo"],
            "reliance": ["reliance general", "RELIANCE GENERAL", "reliance general insuraance",
                         "RELIANCE GENERAL INSURAANCE"],
            "tata_aig": ["tata aig", "tataaig", "tataaig.com", "TATA AIG General Insurance Company Ltd",
                         "TATA AIG General Insurance Company Ltd.", "noreplyclaims@tataaig.com"],
            "united": ["united india insurance", "united india", "uiic", "UNITED INDIA INSURANCE COMPANY LIMITED",
                       "united india insurance company limited", "UNITED INDIA"],
            "universal_sompo": ["universal sompo", "sompo", "UNIVERSAL SOMPO GENERAL INSURANCE COMPANY LTD",
                                "UNIVERSAL SOMPO", "SOMPO"],
            "zion": ["zion", "ZION", "ZION REF NO."]
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
            logger.debug("Extracting data using Universal Sompo patterns")

            # Look for claim reference in CL format
            claim_patterns = [
                r'CL\d{8}',
                r'Payment\s+Ref\.\s+No\.\s*:\s*(\d{10})',
                r'Claim\s+number\s*:\s*(\S+)',
            ]

            for pattern in claim_patterns:
                claim_match = re.search(pattern, text, re.IGNORECASE)
                if claim_match:
                    if '(' in pattern:
                        data['unique_id'] = claim_match.group(1).strip()
                    else:
                        data['unique_id'] = claim_match.group(0).strip()
                    logger.debug(f"Extracted Universal Sompo claim reference: {data['unique_id']}")
                    break

            # Extract amount
            amount_patterns = [
                r'Amount\s*:\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)',
                r'INR\s+(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)',
            ]

            for pattern in amount_patterns:
                amount_match = re.search(pattern, text, re.IGNORECASE)
                if amount_match:
                    data['receipt_amount'] = amount_match.group(1).strip().replace(',', '')
                    logger.debug(f"Extracted Universal Sompo amount: {data['receipt_amount']}")
                    break

            # Extract date
            date_patterns = [
                r'Payment\s+Ini\.\s+Date\s*:\s*(\d{2}-\d{2}-\d{4})',
                r'Value\s+Date\s*:\s*(\d{2}-\d{2}-\d{4})',
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

        elif detected_provider == "icici_lombard":
            logger.debug("Extracting data using ICICI Lombard patterns")

            # Extract claim reference numbers in ENG format
            claim_patterns = [
                r'((?:MAR|FIR|ENG|MSC|LIA)\d{9})',  # Captures all five possible prefixes followed by 9 digits
                r'CLAIM_REF_NO\s*(\S+)',  # Generic pattern for claim reference
                r'(?:MAR|FIR|ENG|MSC|LIA)\d{9}',  # Direct match without capture group
            ]

            for pattern in claim_patterns:
                claim_match = re.search(pattern, text, re.IGNORECASE)
                if claim_match:
                    data['unique_id'] = claim_match.group(1).strip() if '(' in pattern else claim_match.group(0).strip()
                    logger.debug(f"Extracted ICICI claim reference: {data['unique_id']}")
                    break

            # Extract invoice number (LAE format) with dynamic financial year prefix
            invoice_patterns = [
                r'LAE\s+Invoice\s+No\s*[:.\s]*\s*(\d{4}-\d{5})',  # Match any 4-digit year prefix
                r'\d{4}-\d{5}',  # Direct match for any 4-digit prefix followed by dash and 5 digits
            ]

            for pattern in invoice_patterns:
                invoice_match = re.search(pattern, text, re.IGNORECASE)
                if invoice_match:
                    data['invoice_no'] = invoice_match.group(1).strip() if '(' in pattern else invoice_match.group(
                        0).strip()
                    logger.debug(f"Extracted ICICI invoice number: {data['invoice_no']}")
                    break

            # Extract receipt amount
            amount_patterns = [
                r'TRF\s+AMOUNT\s*[:.\s]*\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                r'Invoice\s+Amt\s*[:.\s]*\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
            ]

            for pattern in amount_patterns:
                amount_match = re.search(pattern, text, re.IGNORECASE)
                if amount_match:
                    data['receipt_amount'] = amount_match.group(1).strip().replace(',', '')
                    logger.debug(f"Extracted ICICI amount: {data['receipt_amount']}")
                    break

            # Extract receipt date
            date_patterns = [
                r'RECEIPT\s+DATE\s*[:.\s]*\s*(\d{2}-\d{2}-\d{4})',
                r'Bill\s+Date\s*[:.\s]*\s*(\d{1,2}/\d{1,2}/\d{4})',
            ]

            for pattern in date_patterns:
                date_match = re.search(pattern, text, re.IGNORECASE)
                if date_match:
                    date_str = date_match.group(1).strip()
                    # Convert to standard format if needed
                    if '/' in date_str:
                        parts = date_str.split('/')
                        if len(parts) == 3:
                            date_str = f"{parts[0].zfill(2)}-{parts[1].zfill(2)}-{parts[2]}"
                    data['receipt_date'] = date_str
                    logger.debug(f"Extracted ICICI date: {data['receipt_date']}")
                    break

        elif detected_provider == "zion":
            logger.debug("Extracting data using Zion patterns")

            # Extract reference number from Zion's format
            ref_patterns = [
                r'JPB\d+/\d+-\d+',
                r'ZION\s+REF\s+NO\.\s*(\S+)',
            ]

            for pattern in ref_patterns:
                ref_match = re.search(pattern, text, re.IGNORECASE)
                if ref_match:
                    if '(' in pattern:
                        data['unique_id'] = ref_match.group(1).strip()
                    else:
                        data['unique_id'] = ref_match.group(0).strip()
                    logger.debug(f"Extracted Zion reference: {data['unique_id']}")
                    break

            # Extract bill number (invoice) which is usually in 2425-XXXXX format
            bill_patterns = [
                r'bill\s+no\s*[:.\s]*\s*(\d{2}\d{2}-\d{5})',
                # Match any 4-digit year prefix followed by dash and 5 digits
                r'\d{4}-\d{5}',  # More generic pattern for any 4-digit prefix
                r'bill\s+no\s*[:.\s]*\s*([A-Z0-9]{4}-\d{5})',  # Even more flexible to handle alphanumeric prefixes
            ]

            for pattern in bill_patterns:
                bill_match = re.search(pattern, text, re.IGNORECASE)
                if bill_match:
                    if '(' in pattern:
                        data['invoice_no'] = bill_match.group(1).strip()
                    else:
                        data['invoice_no'] = bill_match.group(0).strip()
                    # If unique_id not found, use invoice number
                    if 'unique_id' not in data:
                        data['unique_id'] = data['invoice_no']
                    logger.debug(f"Extracted Zion bill number: {data['invoice_no']}")
                    break

            # Extract amount with Zion specific patterns
            amount_patterns = [
                r'Bill\s+Amount\s*[:.\s]*\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                r'amt\s*[:.\s]*\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                r'(\d{4})(?:\s+|\s*)\d+\s+\d+',  # Matches amount pattern in Zion table
            ]

            for pattern in amount_patterns:
                amount_match = re.search(pattern, text, re.IGNORECASE)
                if amount_match:
                    data['receipt_amount'] = amount_match.group(1).strip().replace(',', '')
                    logger.debug(f"Extracted Zion amount: {data['receipt_amount']}")
                    break

            # Extract date with Zion specific patterns
            date_patterns = [
                r'Receipt\s+Date\s*[:.\s]*\s*(\d{2}-\d{2}-\d{4})',
                r'(\d{2}-\d{2}-\d{4})',
            ]

            for pattern in date_patterns:
                date_match = re.search(pattern, text, re.IGNORECASE)
                if date_match:
                    data['receipt_date'] = date_match.group(1).strip()
                    logger.debug(f"Extracted Zion date: {data['receipt_date']}")
                    break

            # Extract TDS specific to Zion
            tds_patterns = [
                r'tds\s*[:.\s]*\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                r'TDS\s+Amount\s*[:.\s]*\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
            ]

            for pattern in tds_patterns:
                tds_match = re.search(pattern, text, re.IGNORECASE)
                if tds_match:
                    data['tds'] = tds_match.group(1).strip().replace(',', '')
                    data['tds_computed'] = 'No'  # TDS is provided in the document
                    logger.debug(f"Extracted Zion TDS: {data['tds']}")
                    break

        elif detected_provider in ["bajaj_allianz", "cholamandalam"]:
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
        return unique_id, data_points, table_data, detected_provider

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
                    '%d-%b-%Y',  # 12-Feb-2025
                    '%d/%b/%Y',  # 12/Feb/2025
                    '%d %B %Y',  # 11 February 2025
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

    # In compute_tds method, add company-specific rates
    def compute_tds(self, amount, text):
        """
        Compute TDS based on the provider and amount.
        """
        try:
            # Normalize the text for better matching
            normalized_text = ' '.join(text.lower().split())

            # Define insurance companies with their TDS rates
            insurance_rates = {
                "oriental": 0.11111111,  # 11.111111%
                "united_india": 0.11111111,  # 11.111111%
                "new_india": {
                    "default": 0.11111111,  # 11.111111% for amounts > 300000
                    "below_threshold": 0.09259259  # 9.259259% for amounts <= 300000
                },
                "national": 0.11111111,  # 11.111111%
                "hdfc_ergo": 0.09259259,  # 9.259259%
                "icici_lombard": 0.09259259,  # 9.259259%
                "iffco_tokio": 0.09259259,  # 9.259259%
                "universal_sompo": 0.09259259,  # 9.259259%
                "future_generali": 0.09259259,  # 9.259259%
                "tata_aig": 0.09259259,  # 9.259259%
                "bajaj_allianz": 0.09259259,  # 9.259259%
                "liberty": 0.09259259,  # 9.259259%
                "cholamandalam": 0.09259259,  # 9.259259%
                "reliance": 0.09259259,  # 9.259259%
                "zion": 0.09259259,  # 9.259259%
                "cera": 0.09259259,  # 9.259259%  (for non-insurance)
            }

            # Default rate for other companies
            default_rate = 0.09259259  # 9.259259%

            # Detect company
            detected_company = None
            for company, keywords in self.insurance_providers.items():
                if any(keyword in normalized_text for keyword in keywords):
                    detected_company = company
                    logger.info(f"Detected company for TDS computation: {company}")
                    break

            # Apply appropriate TDS rate
            if detected_company:
                if detected_company == "new_india":
                    # Special handling for New India Assurance with threshold
                    if amount <= 300000:
                        tds_rate = insurance_rates["new_india"]["below_threshold"]
                        logger.info(f"Using below threshold rate for New India: {tds_rate}")
                    else:
                        tds_rate = insurance_rates["new_india"]["default"]
                        logger.info(f"Using above threshold rate for New India: {tds_rate}")
                elif detected_company in insurance_rates:
                    tds_rate = insurance_rates[detected_company]
                    logger.info(f"Using company-specific rate for {detected_company}: {tds_rate}")
                else:
                    tds_rate = default_rate
                    logger.info(f"Using default rate for {detected_company}: {tds_rate}")
            else:
                tds_rate = default_rate
                logger.info(f"Using default rate (no company detected): {tds_rate}")

            # Calculate TDS
            tds = round(amount * tds_rate, 2)
            logger.info(f"Computed TDS: {tds} ({tds_rate * 100}% of {amount})")

            return tds, True

        except Exception as e:
            logger.error(f"Error computing TDS: {str(e)}")
            return 0.0, True

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
        if "cera" in text.lower() or "gross amount" in text.lower():
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

        # 4. ICICI Lombard reference table
        icici_table_match = re.search(
            r'CLAIM_REF_NO\s+LAE\s+Invoice\s+No\s+Bill\s+Date\s+Invoice\s+Amt\s+TDS\s+TRF\s+AMOUNT', text,
            re.IGNORECASE)
        if icici_table_match:
            # Extract rows using regex pattern
            rows = re.findall(
                r'(ENG\d{9})\s+(2425-\d{5})\s+(\d{2}/\d{2}/\d{4})\s+(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s+(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s+(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)',
                text)

            for row in rows:
                claim_ref, invoice_no, bill_date, invoice_amt, tds, trf_amount = row

                # Convert date format from DD/MM/YYYY to DD-MM-YYYY
                date_parts = bill_date.split('/')
                formatted_date = f"{date_parts[0]}-{date_parts[1]}-{date_parts[2]}" if len(
                    date_parts) == 3 else bill_date

                table_data.append({
                    'unique_id': claim_ref.strip(),
                    'invoice_no': invoice_no.strip(),
                    'receipt_date': formatted_date.strip(),
                    'invoice_amt': invoice_amt.strip().replace(',', ''),
                    'tds': tds.strip().replace(',', ''),
                    'receipt_amount': trf_amount.strip().replace(',', ''),
                    'tds_computed': 'No'  # TDS already present in document
                })

            logger.info(f"Extracted {len(table_data)} rows from ICICI Lombard table")

        # 5. Zion reference table (spotted in sample 34)
        zion_table_match = re.search(r'DATE\s+ZION\s+REF\s+NO\.\s+bill\s+no\s+(?:amt|Bill\s+Amount)', text,
                                     re.IGNORECASE)
        if zion_table_match:
            # Find the table content lines
            table_content = text[zion_table_match.end():]
            table_end = table_content.find("TOTAL PAID AMT")
            if table_end > 0:
                table_content = table_content[:table_end]

            # Extract rows using regex pattern matching
            rows = re.findall(r'(\d{2}-\d{2}-\d{4})\s+(JPB[\d/\-]+)\s+(2425-\d{5})\s+(\d+)\s+(\d+)\s+(\d+)',
                              table_content)

            for row in rows:
                date, ref_no, bill_no, amount, tds, receipt_amount = row
                table_data.append({
                    'unique_id': bill_no.strip(),
                    'reference_no': ref_no.strip(),
                    'receipt_date': date.strip(),
                    'bill_amount': amount.strip(),
                    'tds': tds.strip(),
                    'receipt_amount': receipt_amount.strip(),
                    'tds_computed': 'No'  # TDS already present in document
                })

            logger.info(f"Extracted {len(table_data)} rows from Zion table")

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