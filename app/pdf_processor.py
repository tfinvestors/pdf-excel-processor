import os
import re
import joblib
import logging
import pandas as pd
from datetime import datetime
import numpy as np
import concurrent.futures
import json
import requests
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


# Import unified PDF text extractor
from app.utils.pdf_text_extractor import PDFTextExtractor
from app.utils.insurance_providers import INSURANCE_PROVIDERS, SPECIFIC_TDS_RATE_PROVIDERS, NEW_INDIA_THRESHOLD
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(stream=sys.stdout),  # Use stdout with UTF-8 config
        logging.FileHandler("pdf_processing.log", encoding='utf-8')  # Add encoding here
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
        """Extract text from PDF using the unified approach."""
        logger.info(f"Processing PDF: {pdf_path}")

        try:
            # Import the PDF reader
            from app.processors.pdf_reader import PdfReader

            # Initialize and process
            pdf_reader = PdfReader()
            result = pdf_reader.process_pdf(pdf_path)

            # Combine all page results
            combined_text = ""
            for page_result in result["results"]:
                if page_result["text"]:
                    combined_text += page_result["text"] + "\n"
                if page_result["table_data"]:
                    combined_text += "\n[TABLE DATA]\n" + page_result["table_data"] + "\n"

            # Apply post-processing
            if combined_text:
                from app.processors.text_post_processor import TextPostProcessor
                post_processor = TextPostProcessor()
                processed_text = post_processor.process(combined_text)
                return self.text_extractor.clean_text(processed_text)

            return combined_text

        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            # Fallback to original approach
            return self.text_extractor.extract_from_file(pdf_path)

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

    def extract_cera_invoice_numbers(self, text):
        """
        Specialized method to extract Cera invoice numbers and table data from OCR text.
        Uses specific pattern matching for Cera's structured table format.
        """
        logger.info("Using specialized Cera invoice extraction")

        invoice_numbers = []
        table_data = []

        # 1. Extract document date with flexible pattern matching
        date_patterns = [
            r'Document\s*No\.\s*/\s*Date\s*.*?(\d{1,2}\.\d{1,2}\.\d{4})',
            r'payment\s*made\s*on\s*(\d{1,2}\.\d{1,2}\.\d{4})',
            r'Date\s*[:.]?\s*(\d{1,2}[-/.]\d{1,2}[-/.]\d{4})'
        ]

        document_date = None
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                document_date = match.group(1).strip()
                date_parts = document_date.split('.')
                if len(date_parts) == 3:
                    document_date = f"{date_parts[0]}-{date_parts[1]}-{date_parts[2]}"
                logger.info(f"Extracted document date: {document_date}")
                break

        # 2. First look specifically for the Cera structured table format
        structured_table_pattern = r'Document\s+No\.\s+Bill\s+No\.\s+Bill\s+Date\s+Gross\s+Amount\s+TDS\s+Amount\s+Net\s+Amount'
        structured_table_match = re.search(structured_table_pattern, text, re.IGNORECASE)

        if structured_table_match:
            logger.info("Found structured Cera invoice table")
            table_content = text[structured_table_match.end():]

            # Find where the table ends (typically at "Net Total")
            table_end = re.search(r'Net\s+Total', table_content)
            if table_end:
                table_content = table_content[:table_end.start()]

            # Look for rows with exact column structure matching Cera invoices
            # This pattern specifically captures DocNo, BillNo, Date, Gross, TDS, Net in that order
            row_pattern = r'(\d+)\s+(\d{4}-\d{5})\s+(\d{1,2}\.\d{1,2}\.\d{4})\s+(\d{1,3}(?:[,.]\d{3})*(?:\.\d{2}))\s+(\d{1,3}(?:[,.]\d{3})*(?:\.\d{2}))\s+(\d{1,3}(?:[,.]\d{3})*(?:\.\d{2}))'

            # Find all matching rows - each match will have 6 groups: docno, billno, date, gross, tds, net
            rows = re.findall(row_pattern, table_content)

            if rows:
                logger.info(f"Found {len(rows)} structured Cera invoice rows")
                for row in rows:
                    doc_no, bill_no, bill_date, gross_amount, tds_amount, net_amount = row

                    # Clean and format the values
                    date_parts = bill_date.split('.')
                    if len(date_parts) == 3:
                        formatted_date = f"{date_parts[0]}-{date_parts[1]}-{date_parts[2]}"
                    else:
                        formatted_date = bill_date

                    # Create table data with the correct NET amount (not TDS)
                    row_data = {
                        'unique_id': bill_no,
                        'bill_date': formatted_date,  # Store Bill Date separately
                        'receipt_date': document_date.replace('.', '-') if document_date else formatted_date,
                        'receipt_amount': net_amount.replace(',', ''),  # NET AMOUNT (not TDS)
                        'tds': tds_amount.replace(',', ''),
                        'tds_computed': 'No'  # We have the actual TDS from the document
                    }

                    invoice_numbers.append(bill_no)
                    table_data.append(row_data)
                    logger.info(f"Created structured row for bill {bill_no} with net amount {net_amount}")

                # If we've found and processed structured rows, return immediately
                if table_data:
                    return invoice_numbers, table_data

        # 2. Extract the Net Total - it's a reliable anchor point
        net_total_patterns = [
            r'Net\s*Total\s*(?:Rs\.?|INR)?\s*(\d{1,3}(?:[,.]\d{3})*(?:[,.]\d{1,2})?)',
            r'NetTotal\s*T?Y?\s*(\d{1,3}(?:[,.]\d{3})*(?:[,.]\d{1,2})?)'
        ]

        net_total = None
        for pattern in net_total_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                net_total = match.group(1).strip().replace(',', '.')
                logger.info(f"Extracted Net Total amount: {net_total}")
                break

        # 4. Find the table section by looking for key headers
        table_section = None
        table_markers = [
            "Bill No.", "Document No.", "Gross Amount", "TDS Amount", "Net Amount",
            "TDSAmount", "NetAmount", "BillNo", "BllDate"
        ]

        for marker in table_markers:
            pos = text.find(marker)
            if pos >= 0:
                # Extract from marker position to the end of document or Net Total
                end_pos = text.find("Net Total", pos)
                if end_pos < 0:
                    end_pos = len(text)

                table_section = text[max(0, pos - 50):end_pos + 50]
                logger.info(f"Found table section using marker '{marker}'")
                break

        # 5. Try targeted pattern extraction for table section if we have one
        if table_section:
            logger.info("Attempting to extract from table section using targeted pattern")
            # This pattern specifically targets the table structure in Cera documents
            # Looking for DocNo BillNo Date Gross TDS Net format
            targeted_pattern = r'(\d+)\s+(\d{4}-\d{5})\s+(\d{1,2}\.\d{1,2}\.\d{4})\s+(\d+\.\d{2})\s+(\d+\.\d{2})\s+(\d+\.\d{2})'
            targeted_matches = re.findall(targeted_pattern, table_section)

            if targeted_matches:
                logger.info(f"Found {len(targeted_matches)} rows using targeted pattern")
                for match in targeted_matches:
                    doc_no, bill_no, date, gross, tds, net = match
                    # Format date
                    date_parts = date.split('.')
                    if len(date_parts) == 3:
                        formatted_date = f"{date_parts[0]}-{date_parts[1]}-{date_parts[2]}"
                    else:
                        formatted_date = date

                    row_data = {
                        'unique_id': bill_no,
                        'bill_date': formatted_date,  # Store Bill Date separately
                        'receipt_date': document_date.replace('.', '-') if document_date else formatted_date,
                        'receipt_amount': net,  # IMPORTANT: Use NET AMOUNT
                        'tds': tds,
                        'tds_computed': 'No'
                    }

                    # Only add if not already in table_data
                    if bill_no not in [row.get('unique_id') for row in table_data]:
                        table_data.append(row_data)
                        if bill_no not in invoice_numbers:
                            invoice_numbers.append(bill_no)
                        logger.info(f"Added targeted row for bill {bill_no} with net amount {net}")

        # 6. Use document structure analysis to identify the bill number pattern in use
        # First look for any pattern that appears to be a bill number
        # (typically 4 digits, dash, 5 digits)
        bill_patterns = [
            r'\d{4}-\d{5}',  # Standard format (e.g., 2425-13520)
            r'[a-zA-Z0-9]{4}-[a-zA-Z0-9]{5}'  # Format allowing for OCR errors
        ]

        all_potential_bills = []

        # Only look for more bill numbers if we haven't found any yet
        if not invoice_numbers:
            for pattern in bill_patterns:
                matches = re.findall(pattern, text)
                all_potential_bills.extend(matches)

            # Clean up potential bill numbers
            if all_potential_bills:
                for bill in all_potential_bills:
                    # Apply OCR error corrections
                    clean_bill = self._correct_ocr_errors(bill)

                    # Only add if it seems like a valid bill number
                    if re.match(r'\d{4}-\d{5}', clean_bill) and clean_bill not in invoice_numbers:
                        invoice_numbers.append(clean_bill)
                        logger.info(f"Extracted bill number: {clean_bill}")

        # 7. If we still don't have invoice numbers, try to identify them from context
        if not invoice_numbers and table_section:
            # Look for the Bill No. column and extract values
            bill_column_pattern = r'Bill\s*No\.'
            bill_column_match = re.search(bill_column_pattern, table_section)

            if bill_column_match:
                # Look for values in this column format
                column_values_pattern = r'(?:Bill\s*No\..*?)(\S+)(?:\s|$)'
                column_values = re.findall(column_values_pattern, table_section)

                for value in column_values:
                    # Clean up potential OCR errors
                    clean_value = self._correct_ocr_errors(value)

                    # Only add if it seems like a valid bill number
                    if re.match(r'\d{4}-\d{5}', clean_value) and clean_value not in invoice_numbers:
                        invoice_numbers.append(clean_value)
                        logger.info(f"Extracted bill number from column: {clean_value}")

        # 8. For any invoice numbers without table data, try to find bill-amount pairs
        if invoice_numbers and len(invoice_numbers) > len(table_data):
            # Try to find bill-amount pairs specifically for Cera
            bill_amount_pattern = r'(\d{4}-\d{5})(?:[^\n]*?)(\d{1,2}\.\d{1,2}\.\d{4})(?:[^\n]*?)(\d+\.\d{2})(?:[^\n]*?)(\d+\.\d{2})(?:[^\n]*?)(\d+\.\d{2})'
            bill_amount_matches = re.findall(bill_amount_pattern, table_section if table_section else text)

            if bill_amount_matches:
                logger.info(f"Found {len(bill_amount_matches)} bill-amount pairs")
                for match in bill_amount_matches:
                    bill_no, date, gross, tds, net = match
                    # Skip if we already have this bill in table_data
                    if any(row.get('unique_id') == bill_no for row in table_data):
                        continue

                    # Format date
                    date_parts = date.split('.')
                    if len(date_parts) == 3:
                        formatted_date = f"{date_parts[0]}-{date_parts[1]}-{date_parts[2]}"
                    else:
                        formatted_date = date

                    row_data = {
                        'unique_id': bill_no,
                        'bill_date': formatted_date,  # Store Bill Date separately
                        'receipt_date': document_date.replace('.', '-') if document_date else formatted_date,
                        'receipt_amount': net,  # Use NET AMOUNT
                        'tds': tds,
                        'tds_computed': 'No'
                    }
                    table_data.append(row_data)
                    logger.info(f"Created table row for bill {bill_no} with net amount {net}")

        # 9. Filter out phone/fax numbers by checking for common patterns
        if invoice_numbers:
            filtered_invoice_numbers = []
            for inv in invoice_numbers:
                # Skip phone/fax patterns (usually start with 2764-, etc.)
                if inv.startswith('2764-') or inv.startswith('0144-'):
                    logger.info(f"Skipping likely phone/fax number: {inv}")
                    continue

                # Keep only those matching expected patterns for bill numbers
                if not inv.startswith(('2764-', '0144-')) and re.match(r'(\d{4})-\d{5}', inv):
                    filtered_invoice_numbers.append(inv)

            invoice_numbers = filtered_invoice_numbers

        # 10. As a last resort, if we have invoice numbers but no table data,
        # create entries based on the document date and extract net total if available
        if invoice_numbers and not table_data and (document_date or net_total):
            for inv in invoice_numbers:
                # Create minimal data
                row_data = {
                    'unique_id': inv,
                    'receipt_date': document_date.replace('.', '-') if document_date else None,
                }

                # If we have multiple invoice numbers and a net total,
                # split the total proportionally (as a last resort)
                if net_total and len(invoice_numbers) > 1:
                    # Equal distribution
                    per_invoice = float(net_total.replace(',', '')) / len(invoice_numbers)
                    row_data['receipt_amount'] = str(per_invoice)
                    row_data['tds_computed'] = 'Yes'
                elif net_total:
                    # Just one invoice, use the full amount
                    row_data['receipt_amount'] = net_total.replace(',', '')
                    row_data['tds_computed'] = 'Yes'

                table_data.append(row_data)
                logger.info(f"Created fallback row for invoice {inv}")

        return invoice_numbers, table_data

    def _correct_ocr_errors(self, text):
        """
        Helper method to correct common OCR errors in text without hardcoding specific values.

        Args:
            text (str): Text to correct

        Returns:
            str: Corrected text
        """
        if not text:
            return text

        # Convert to lowercase for consistent processing
        corrected = text.lower()

        # Common OCR confusions
        ocr_corrections = {
            'a': '4',  # OCR often confuses 'a' with '4'
            'l': '1',  # OCR often confuses 'l' with '1'
            'i': '1',  # OCR often confuses 'i' with '1'
            'o': '0',  # OCR often confuses 'o' with '0'
            'z': '2',  # OCR often confuses 'z' with '2'
            'g': '9',  # OCR can confuse 'g' with '9'
            's': '5',  # OCR can confuse 's' with '5'
            'e': '3',  # OCR can confuse 'e' with '3'
        }

        # Apply corrections
        for error, correction in ocr_corrections.items():
            if error in corrected:
                # Only replace if it's likely a digit context
                # Look for patterns like 'a' within digits
                digit_contexts = re.findall(r'\d*' + error + r'\d*', corrected)
                for context in digit_contexts:
                    fixed_context = context.replace(error, correction)
                    corrected = corrected.replace(context, fixed_context)

        # Ensure proper formatting for bill numbers
        if '-' in corrected:
            parts = corrected.split('-')
            if len(parts) == 2:
                # Fix first part if it looks like a bill prefix (usually 4 digits)
                if len(parts[0]) == 4 and parts[0].isdigit():
                    pass  # Already clean
                elif len(parts[0]) == 4:
                    # Convert any remaining letters to most likely digits
                    parts[0] = ''.join(c if c.isdigit() else '4' for c in parts[0])

                # Fix second part if it looks like a bill number (usually 5 digits)
                if len(parts[1]) == 5 and parts[1].isdigit():
                    pass  # Already clean
                elif len(parts[1]) == 5:
                    # Convert any remaining letters to most likely digits
                    parts[1] = ''.join(c if c.isdigit() else '1' for c in parts[1])

                corrected = f"{parts[0]}-{parts[1]}"

        return corrected

    # Expand provider identification in extract_bank_specific_data
    def extract_bank_specific_data(self, text):
        """
        Extract data specifically from banking documents (payment advices).
        Enhanced to support multiple insurance providers.
        """
        data = {}
        detected_provider = None

        # Normalize text for better pattern matching
        normalized_text = ' '.join(text.lower().split())
        logger.debug(f"Normalized text for pattern matching: {normalized_text[:200]}...")

        # Determine the insurance provider
        for provider, keywords in INSURANCE_PROVIDERS.items():
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
                r'Claim#\s*.*?(\d{10}[cC]\d{9})',  # Updated to capture 9 digits after C
                r'claim\s*[\s#:]*\s*(\d{10}[cC]\d{9})',  # Updated pattern
                r'(\d{10}[cC]\d{9})',  # Updated to capture complete number
                r'Enrichment:[\s\S]*?Claim#\s*[\s\n]*(\d{10}[cC]\d{9})',  # New pattern to capture across newlines
                r'claim#\s*(5004\d{10})',
                r'payment\s+ref\.\s+no\.\s*:\s*(\d{18})',
                r'(5004\d{10})',
                r'claim#\s*(\d{19})',
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

            # FIRST CHECK THE TABLE FORMAT - Table Pattern
            if 'Net Payable amount' in text or 'Net Payable\namount' in text:
                # Try to extract from the table format
                table_pattern = r'Gross\s+amount\s+TDS\s+amount\s+Net\s+Payable\s+amount\s*[\r\n]*\s*([\d\.,]+)\s+([\d\.,]+)\s+([\d\.,]+)'
                table_match = re.search(table_pattern, text, re.IGNORECASE | re.DOTALL)
                if table_match:
                    data['receipt_amount'] = table_match.group(3).strip().replace(',', '')
                    logger.debug(f"Extracted TATA AIG net amount from table: {data['receipt_amount']}")
                    # Also extract TDS since we have it
                    data['tds'] = table_match.group(2).strip().replace(',', '')
                    data['tds_computed'] = 'No'  # TDS already present in the document
                    logger.debug(f"Extracted TATA AIG TDS from table: {data['tds']}")

            # ONLY IF TABLE FORMAT DOESN'T MATCH, try other patterns - amount_patterns
            if 'receipt_amount' not in data:
                # Extract amount using improved patterns
                amount_patterns = [
                    r'Net\s+Payable\s+amount\s*(?:rs\.?|inr)?\s*([0-9,.]+)',
                    r'Net\s+Payable\s+Amount\s*(?:rs\.?|inr)?\s*([0-9,.]+)',
                    r'net\s+payable\s+amount\s*(?:rs\.?|inr)?\s*([0-9,.]+)',
                    r'Payment\s+amoun[rt]:\s*(?:rs\.?|inr|₹)?\s*([0-9,.]+)',
                    r'Payment\s+amoun[rt]\s*:?\s*(?:rs\.?|inr|₹)?\s*([0-9,.]+)',
                    r'(?:Rs\.?|INR|₹)\s*([0-9,.]+)',
                    r'(?:rs\.?|inr|₹)\s*([0-9,.]+)',
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

            # First look specifically for F0036935 pattern in the table section
            claim_table_pattern = r'Not\s+Applic\/F(\d{7})\s+Claims\s+Payment'
            claim_table_match = re.search(claim_table_pattern, text, re.IGNORECASE)

            if claim_table_match:
                data['unique_id'] = f"F{claim_table_match.group(1)}"
                logger.debug(f"Extracted Future Generali claim number from table: {data['unique_id']}")

            # If not found in table, then try the other patterns
            elif not 'unique_id' in data or not data['unique_id']:
                # Try different invoice/claim patterns
                invoice_patterns = [
                    r'F\d{7}(?=\s+Claims\s+Payment)',  # Match F followed by 7 digits before Claims Payment
                    r'Invoice\s+Details.*?(?:F\d{7})',  # Look for F-numbers in Invoice Details
                ]

                for pattern in invoice_patterns:
                    invoice_match = re.search(pattern, text, re.IGNORECASE)
                    if invoice_match:
                        data['unique_id'] = invoice_match.group(0).strip()
                        logger.debug(f"Extracted Future Generali invoice/claim number: {data['unique_id']}")
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

        elif detected_provider == "national":
            logger.debug("Extracting data using National Insurance patterns")

            # Try parsing the full table row structure first - most reliable method
            table_row_pattern = r'(\d+)\s+(\d{18})\s+(\d+-\d+)\s+([A-Z\s]+)\s+(\d{2}-\d{2}-\d{4})\s+(\d+)(?:\s*CMS\d+)?\s+(\d+)\s+(\d+)\s+(\d+)'
            table_match = re.search(table_row_pattern, text, re.DOTALL)

            if table_match:
                # Extract all values from the table row
                processing_code, policy_number, sub_claim_no, insured_name, settlement_date, payment_ref, gross_amount, tds_amount, net_amount = table_match.groups()

                # Look for Payment Date specifically in the header, which is more reliable
                payment_date_pattern = r'भुगतान\s+सलाह\s+की\s+तारीख/Payment\s+Date:(\d{2}/\d{2}/\d{4})'
                payment_date_match = re.search(payment_date_pattern, text)

                data['unique_id'] = sub_claim_no.strip()
                data['policy_no'] = policy_number.strip()
                # Use Payment Date from the header if found, otherwise fall back to settlement date
                if payment_date_match:
                    data['receipt_date'] = payment_date_match.group(1).strip()
                else:
                    data['receipt_date'] = settlement_date.strip()
                data['receipt_amount'] = net_amount.strip()  # Use Net Amount as receipt amount
                data['gross_amount'] = gross_amount.strip()
                data['tds'] = tds_amount.strip()
                data['tds_computed'] = 'No'

                logger.debug(f"Extracted complete National Insurance payment row")
                return data, detected_provider

            # If direct table pattern fails, fall back to individual field extraction
            # Extract claim number
            sub_claim_pattern = r'उप\s+दावा\s+सं(?:[^\n]*?)(\d+(?:-\d+)+)'
            alt_pattern = r'Sub\s+Claim\s+No(?:[^\n]*?)(\d+(?:-\d+)+)'

            for pattern in [sub_claim_pattern, alt_pattern]:
                claim_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if claim_match:
                    data['unique_id'] = claim_match.group(1).strip()
                    logger.debug(f"Extracted National Insurance sub claim number: {data['unique_id']}")
                    break

            # If claim number not found via specific fields, try the structured pattern
            if 'unique_id' not in data or not data['unique_id']:
                structured_pattern = r'(\d{18})\s+(\d+-\d+)'
                structured_match = re.search(structured_pattern, text)
                if structured_match:
                    data['unique_id'] = structured_match.group(2).strip()
                    logger.debug(f"Extracted National Insurance structured claim number: {data['unique_id']}")

            # Extract Net Paid Amount - look for the specific pattern at the end of a line
            amount_patterns = [
                # Pattern to find the three numeric values at the end of a line (Gross, TDS, Net)
                r'(\d{5})\s+(\d{4})\s+(\d{5})$',
                r'शुद्ध\s+भुगतान\s+राशि.*?(\d{5})',
                r'Net\s+Paid\s+Amount.*?(\d{5})',
                # More general pattern
                r'Net\s+Paid\s+Amount[^0-9]*(\d+)',
            ]

            for pattern in amount_patterns:
                amount_match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if amount_match:
                    # If it's the pattern with 3 groups, use the last one (Net Amount)
                    if len(amount_match.groups()) == 3:
                        data['receipt_amount'] = amount_match.group(3).strip()
                        # Also extract the other values while we're here
                        data['gross_amount'] = amount_match.group(1).strip()
                        data['tds'] = amount_match.group(2).strip()
                        data['tds_computed'] = 'No'
                    else:
                        data['receipt_amount'] = amount_match.group(1).strip()
                    logger.debug(f"Extracted National Insurance amount: {data['receipt_amount']}")
                    break

            # Extract date from Settlement Date field or Payment Date
            date_patterns = [
                r'भुगतान\s+सलाह\s+की\s+तारीख.*?(\d{2}/\d{2}/\d{4})',
                r'Payment\s+Date.*?(\d{2}/\d{2}/\d{4})',
                r'िन(?:[^\n]*?)तारीख(?:[^0-9]*?)(\d{2}-\d{2}-\d{4})',
                r'Settlement\s+Date(?:[^0-9]*?)(\d{2}-\d{2}-\d{4})',
            ]

            for pattern in date_patterns:
                date_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if date_match:
                    data['receipt_date'] = date_match.group(1).strip()
                    logger.debug(f"Extracted National Insurance date: {data['receipt_date']}")
                    break

            # Extract TDS if not already extracted
            if 'tds' not in data:
                tds_patterns = [
                    r'टीडीएस\s+रािश(?:[^0-9]*?)(\d+)',
                    r'TDS\s+Amount(?:[^0-9]*?)(\d+)',
                    r'/TDS\s+Amount\s*\n(\d+)',
                ]

                for pattern in tds_patterns:
                    tds_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                    if tds_match:
                        data['tds'] = tds_match.group(1).strip()
                        data['tds_computed'] = 'No'
                        logger.debug(f"Extracted National Insurance TDS: {data['tds']}")
                        break

        elif detected_provider in ["iffco_tokio"]:
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
                r'net\s+amount\s*([0-9,.]+)',
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
                r'(\d{2}-\d{2}-\d{4})',
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
                r'Amount\s*:\s*([\d,\.]+)',  # More permissive pattern
                r'Amount\s*:\s*(\d{1,2},\d{2},\d{3}\.\d{2})',  # Pattern for Indian format like 1,49,054.00
                r'for an amount of INR\s+([\d,\.]+)',  # Look for amount in the text paragraph
                r'INR\s+([\d,\.]+)',
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

            # Check for table format with headers
            if "INVOICE NUMBER POLICY NUMBER CLAIM NUMBER AMOUNT" in text:
                # This is a structured table - extract from correct column
                table_start = text.find("INVOICE NUMBER")
                table_end = text.find("TOTAL:", table_start)

                if table_start >= 0 and table_end >= 0:
                    table_content = text[table_start:table_end]
                    # Parse the table row: invoice, policy, claim, amount
                    row_pattern = r'(\S+)\s+(\d{20})\s+(\d{20})\s+([\d,\.]+)'
                    row_match = re.search(row_pattern, table_content)

                    if row_match:
                        # Extract claim number (3rd column)
                        data['unique_id'] = row_match.group(3).strip()
                        logger.debug(f"Extracted New India claim number from table: {data['unique_id']}")

                        # Extract amount (4th column)
                        data['receipt_amount'] = row_match.group(4).strip().replace(',', '')
                        logger.debug(f"Extracted amount from table: {data['receipt_amount']}")
            else:
                # Fallback to original patterns for non-table documents
                claim_patterns = [
                    r'claim\s+number.*?(\d{14,})',
                    r'claim\s+no\s*(?:\/|\\|\.)*\s*(\S+)',
                    r'CLAIM NUMBER.*?\n.*?(?:\d+)\s+(\d+)\s+[\d,\.]+',
                ]

                for pattern in claim_patterns:
                    claim_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                    if claim_match:
                        data['unique_id'] = claim_match.group(1).strip()
                        logger.debug(f"Extracted New India claim number: {data['unique_id']}")
                        break

            # Extract amount
            amount_patterns = [
                r'TOTAL:\s*([\d,\.]+)',  # Add this pattern to prioritize the TOTAL amount
                r'an amount of Rs\.([\d,\.]+)',
                r'remit an amount of Rs\.([\d,\.]+)',
                r'we have instructed our bank to remit an amount of rs\.([\d,\.]+)',
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

        elif detected_provider == "cera":
            logger.debug("Extracting data using Cera Sanitaryware patterns")

            # Use specialized method to extract invoice numbers and table data
            invoice_numbers, table_rows = self.extract_cera_invoice_numbers(text)

            if invoice_numbers:
                # Use the first invoice number as the unique ID
                data['unique_id'] = invoice_numbers[0]
                logger.debug(f"Using first invoice number as unique ID: {data['unique_id']}")

                # If we have table data, use it to populate data
                if table_rows and len(table_rows) > 0:
                    # Use the first table row for data
                    row = table_rows[0]

                    # Copy relevant fields to data (NOT data_points - this was the error)
                    for field in ['receipt_date', 'receipt_amount', 'tds', 'tds_computed', 'gross_amount']:
                        if field in row and row[field]:
                            data[field] = row[field]
                            logger.debug(f"Using {field} from table data: {row[field]}")

            # Extract additional fields from text if needed
            date_match = re.search(r'Document\s*No\.\s*/\s*Date\s*.*?(\d{1,2}\.\d{1,2}\.\d{4})', text, re.IGNORECASE)
            if date_match and ('receipt_date' not in data or not data['receipt_date']):
                data['receipt_date'] = date_match.group(1).strip()
                logger.debug(f"Extracted Cera document date: {data['receipt_date']}")

            # Extract Net Total
            net_total_match = re.search(r'Net\s*Total\s*(?:Rs\.?|INR)?\s*(\d{1,3}(?:[,.]\d{3})*(?:[,.]\d{1,2})?)',
                                        text, re.IGNORECASE)
            if net_total_match and ('receipt_amount' not in data or not data['receipt_amount']):
                data['receipt_amount'] = net_total_match.group(1).strip().replace(',', '')
                logger.debug(f"Extracted Cera Net Total: {data['receipt_amount']}")

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
                # This pattern should be first to prioritize Receipt Date
                r'(\d{2}-\d{2}-\d{4})',  # General date pattern as fallback
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


        elif detected_provider == "bajaj_allianz":
            logger.debug("Extracting data using Bajaj Allianz patterns")

            # (1) Try to capture “OC” + three groups of digits + exactly eight digits,
            #     but allow ANY spaces/newlines between digits. Then strip all whitespace.
            full_oc_match = re.search(
                r'(OC[-\s]*\d+[-\s]*\d+[-\s]*\d+[-\s]*(?:\d[-\s]*){8})',
                text,
                re.IGNORECASE
            )
            if full_oc_match:
                raw = full_oc_match.group(1)
                # Remove any whitespace so that “O C - 24 - 1501 - 4089  00000009” → “OC-24-1501-4089-00000009”
                cleaned = re.sub(r'\s+', '', raw).upper()
                data['unique_id'] = cleaned
                logger.info(f"Extracted complete claim ID (full‐pattern): {data['unique_id']}")
                return data, detected_provider

            # (2) If that didn’t match, look for the first three groups (“OC-xxx-xxx-xxx”)
            #     and then exactly eight digits—allowing whitespace between each digit of those eight.
            three_plus_eight = re.search(
                r'(OC[-\s]*\d+[-\s]*\d+[-\s]*\d+)[^\d]*((?:\d[-\s]*){8})',
                text,
                re.IGNORECASE
            )
            if three_plus_eight:
                prefix_raw = three_plus_eight.group(1)
                suffix_raw = three_plus_eight.group(2)
                # Normalize the prefix to uppercase and ensure single hyphens
                prefix = re.sub(r'[-\s]+', '-', prefix_raw.strip()).upper()
                # Remove whitespace from the eight‐digit part (“0 0 0 0 0 0 0 9” → “00000009”)
                suffix = re.sub(r'\s+', '', suffix_raw)
                data['unique_id'] = f"{prefix}-{suffix}"
                logger.info(f"Extracted claim ID (split‐pattern): {data['unique_id']}")
                return data, detected_provider

            # (3) Only if neither of those worked do we fall back to Claim No / Reference No…
            claim_patterns = [
                r'claim\s+no\s*(?:\/|\\|\.)*\s*[:\-]?\s*(\S+)',
                r'customer\s+reference\s*(?:\/|\\|\.)*\s*[:\-]?\s*(\S+)',
                r'settlement\s+reference\s*[:\-]?\s*(\S+)',
            ]
            for pattern in claim_patterns:
                claim_match = re.search(pattern, text, re.IGNORECASE)
                if claim_match:
                    candidate = claim_match.group(1).strip()
                    if candidate.lower() in ("no", "gross"):
                        logger.info(f"Ignoring invalid fallback ID: {candidate!r}")
                        continue
                    data['unique_id'] = candidate
                    logger.debug(f"Extracted Bajaj Allianz fallback ID: {data['unique_id']}")
                    break

            # (4) Extract amount & date & TDS as before…
            amount_patterns = [
                r'remittance\s+amount\s*:?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                r'amount\s*\(inr\)\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                r'amount\s*:?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                r'Amount\s*\(INR\)\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
            ]
            for pattern in amount_patterns:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    data['receipt_amount'] = m.group(1).replace(',', '')
                    logger.debug(f"Extracted Bajaj Allianz amount: {data['receipt_amount']}")
                    break

            date_patterns = [
                r'value\s+date\s*[»:]\s*(\d{2}-[A-Za-z]{3}-\d{4})',  # e.g. 02-Jan-2025
                r'advice\s+date\s*:?\s*(\d{2}-[A-Za-z]{3}-\d{4})',
                r'value\s+date\s*:?\s*(\d{2}/\d{2}/\d{4})',
                r'value\s+date\s*:?\s*(\d{2}-\d{2}-\d{4})',
                r'(\d{2}/\d{2}/\d{2,4})\s+\d+\s+OC-',  # date near OC
            ]
            for pattern in date_patterns:
                d = re.search(pattern, text, re.IGNORECASE)
                if d:
                    data['receipt_date'] = d.group(1).strip()
                    logger.debug(f"Extracted Bajaj Allianz date: {data['receipt_date']}")
                    break

            tds_patterns = [
                r'TDS\s+Amt\s+(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                r'tds\s+amt\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
                r'tds\s+amount\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
            ]
            for pattern in tds_patterns:
                t = re.search(pattern, text, re.IGNORECASE)
                if t:
                    data['tds'] = t.group(1).replace(',', '')
                    data['tds_computed'] = 'No'
                    logger.debug(f"Extracted Bajaj Allianz TDS: {data['tds']}")
                    break

            return data, detected_provider




        elif detected_provider == "rgcargo":
            logger.debug("Extracting data using RG Cargo patterns")

            # Extract the invoice numbers
            invoice_pattern = r'(\d{4}-\d{5})'
            invoice_matches = re.findall(invoice_pattern, text)

            if invoice_matches:
                # Use the first invoice number as unique ID
                data['unique_id'] = invoice_matches[0].strip()
                logger.debug(f"Extracted RG Cargo invoice number: {data['unique_id']}")

                # Find corresponding date for this invoice
                date_pattern = f"{data['unique_id']}\\s+(\\d{{2}}-\\d{{2}}-\\d{{4}})"
                date_match = re.search(date_pattern, text)
                if date_match:
                    data['receipt_date'] = date_match.group(1).strip()

                # Extract TDS amount - in your document it's the second-to-last column
                tds_pattern = f"{data['unique_id']}.*?\\d+\\s+\\d+\\s+\\d+\\s+(\\d+)\\s+-\\d+"
                tds_match = re.search(tds_pattern, text)
                if tds_match:
                    data['tds'] = tds_match.group(1).strip()
                    data['tds_computed'] = 'No'

                # Extract net amount (negative in your document)
                net_pattern = f"{data['unique_id']}.*?\\d+\\s+\\d+\\s+\\d+\\s+\\d+\\s+(-\\d+)"
                net_match = re.search(net_pattern, text)
                if net_match:
                    data['receipt_amount'] = net_match.group(1).strip()

        elif detected_provider == "reliance":
            logger.debug("Extracting data using Reliance General Insurance patterns")

            # First look for invoice number in Payment Details 5 which is the actual identifier
            invoice_pattern = r'Payment\s+Details\s+5\s*:?\s*(\d{4}-\d{4,5})'
            invoice_match = re.search(invoice_pattern, text, re.IGNORECASE)
            if invoice_match:
                data['unique_id'] = invoice_match.group(1).strip()
                logger.debug(f"Extracted Reliance invoice number: {data['unique_id']}")

            # Look for claim/reference details in other fields as fallback
            if 'unique_id' not in data:
                ref_patterns = [
                    r'Payment\s+Details\s+1\s*:?\s*(\S+)',
                    r'Client\s+Ref\s+No\s*:?\s*(\S+)',
                    r'Payment\s+Details\s+7\s*:?\s*(\S+)'
                ]

                for pattern in ref_patterns:
                    ref_match = re.search(pattern, text, re.IGNORECASE)
                    if ref_match:
                        data['unique_id'] = ref_match.group(1).strip()
                        logger.debug(f"Extracted Reliance reference as fallback: {data['unique_id']}")
                        break

            # Extract amount
            amount_patterns = [
                r'amount\s+of\s*Rs\.?(\d+)',
                r'Rs\.(\d+)\s+through',
                r'amount\s*:?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)'
            ]

            for pattern in amount_patterns:
                amount_match = re.search(pattern, text, re.IGNORECASE)
                if amount_match:
                    data['receipt_amount'] = amount_match.group(1).strip().replace(',', '')
                    logger.debug(f"Extracted Reliance amount: {data['receipt_amount']}")
                    break

            # Extract date
            date_patterns = [
                r'Date\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})',
                r'Date\s*:?\s*(\d{1,2}-\d{1,2}-\d{4})'
            ]

            for pattern in date_patterns:
                date_match = re.search(pattern, text, re.IGNORECASE)
                if date_match:
                    data['receipt_date'] = date_match.group(1).strip()
                    logger.debug(f"Extracted Reliance date: {data['receipt_date']}")
                    break

            # Extract TDS if available (Payment Details 4)
            tds_pattern = r'Payment\s+Details\s+4\s*:?\s*(\d+)'
            tds_match = re.search(tds_pattern, text, re.IGNORECASE)
            if tds_match:
                data['tds'] = tds_match.group(1).strip()
                data['tds_computed'] = 'No'  # TDS already in document
                logger.debug(f"Extracted Reliance TDS: {data['tds']}")

        elif detected_provider in ["cholamandalam"]:

            logger.debug(f"Extracting data using {detected_provider} patterns")

            # Extract claim/reference number
            claim_patterns = [
                r'claim\s+no\s*(?:\/|\\|\.)*\s*(\S+)',
                r'customer\s+reference\s*(?:\/|\\|\.)*\s*:\s*(\S+)',
                r'inv\s+ref:claim\s+no:\s*(\d+)',
                r'inv\s+ref:\s*claim\s+no:\s*(\d+)',
                r'claim\s+no:\s*(\d+)',
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

        return data, detected_provider

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

        # Debug: Log a sample of the text to see what we're working with
        logger.debug(f"Text sample (first 500 chars): {text[:500]}")

        # Check specifically for the problematic pattern
        if "OC-24-1501-4089" in text:
            logger.info("Found OC-24-1501-4089 in text")
            # Try to extract the full pattern
            full_pattern_match = re.search(r'OC-24-1501-4089[-\s\n]*(\d{8})', text, re.DOTALL)
            if full_pattern_match:
                logger.info(f"Full pattern match found: {full_pattern_match.group(0)}")

        # Initialize data extraction results
        data_points = {}
        unique_id = None
        detected_provider = None
        numeric_part = None
        table_data = []

        if not text:
            logger.warning("No text provided for data extraction")
            return None, {}, [], None

        # Check if this is an HSBC/Oriental document
        is_hsbc_doc = any(
            keyword.lower() in text.lower() for keyword in ["HSBC", "Oriental Insurance", "Oriental", "0rienta1"])

        if unique_id:
            numeric_part = re.sub(r'[^0-9]', '', unique_id)
            if len(numeric_part) >= 6:  # Only use if it's a substantial number
                logger.info(f"Numeric part of ID: {numeric_part}")
            else:
                numeric_part = None


        if numeric_part and len(numeric_part) >= 6:
            # Check if this might be a phone number
            if re.search(r'ph.*?\d+[-\s]?\d+', text.lower()):
                phone_number = re.search(r'ph.*?(\d+[-\s]?\d+)', text.lower())
                if phone_number and phone_number.group(1).replace('-', '').replace(' ', '') == numeric_part:
                    logger.info(f"Skipping potential phone number: {numeric_part}")
                    numeric_part = None
            logger.info(f"Numeric part of ID: {numeric_part}")
        else:
            numeric_part = None

        if unique_id:
            # Check if unique ID appears near phone-related keywords
            phone_indicators = ["ph", "phone", "tel", "fax"]

            # Look for the unique ID appearing near any phone indicator within ~50 chars
            for indicator in phone_indicators:
                # Check if the unique ID is mentioned near a phone indicator
                indicator_pos = text.lower().find(indicator)
                if indicator_pos >= 0:
                    # Check if unique_id is within 50 chars of the phone indicator
                    context = text.lower()[max(0, indicator_pos - 10):indicator_pos + 50]
                    if unique_id.lower() in context or unique_id.lower().replace('-', '') in context:
                        logger.warning(f"ID {unique_id} appears near '{indicator}' - likely a phone number")
                        unique_id = None
                        break

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
            bank_result  = self.extract_bank_specific_data(text)

            # Unpack the tuple correctly
            if isinstance(bank_result, tuple) and len(bank_result) == 2:
                bank_data, detected_provider = bank_result
            else:
                # For backward compatibility in case the return value format changes
                bank_data = bank_result
                detected_provider = None

            if bank_data:
                # Get unique ID if available
                if 'unique_id' in bank_data and bank_data['unique_id']:
                    unique_id = bank_data['unique_id']
                    logger.info(f"Found unique ID from bank-specific extraction: {unique_id}")
                    # If we literally captured "No" (e.g. from "Claim No"), drop it now:
                    if unique_id.strip().lower() == "no":
                        unique_id = None
                        logger.info("Discarding invalid unique_id='No' from bank-specific extraction")

                # Copy extracted fields to data_points
                for key, value in bank_data.items():
                    if key != 'unique_id' and value:
                        data_points[key] = value
                        logger.debug(f"Bank-specific extraction provided value for {key}: {value}")

        # Special check for New India Assurance payment advice format
        if "INVOICE NUMBER POLICY NUMBER CLAIM NUMBER AMOUNT" in text and "newindia.co.in" in text:
            logger.info("New India Assurance payment advice format detected - direct extraction")

            # Direct pattern to extract the claim number from the table
            table_pattern = r'(\S+)\s+(\d{20})\s+(\d{20})\s+([\d,\.]+)'
            table_match = re.search(table_pattern, text[text.find("INVOICE NUMBER"):])

            if table_match:
                unique_id = table_match.group(3)  # Claim number is in the 3rd column
                logger.info(f"Directly extracted claim number: {unique_id}")

                # Extract amount from the same row
                data_points['receipt_amount'] = table_match.group(4).replace(',', '')

                # Set detected provider
                detected_provider = "new_india"

        # Fallback for table structure (especially for National Insurance docs)
        if not unique_id and 'Sub Claim No' in text:
            # Look for a pattern with multiple columns in a row - policy number, sub claim, etc.
            table_pattern = r'(\d{18})\s+(\d+-\d+)\s+([A-Z\s]+)'
            table_match = re.search(table_pattern, text)
            if table_match:
                unique_id = table_match.group(2).strip()
                logger.info(f"Found unique ID using table structure extraction: {unique_id}")

                # Also try to extract other data
                if 'receipt_amount' not in data_points:
                    # Look ahead for amounts in the same row
                    amount_match = re.search(fr'{re.escape(unique_id)}.*?(\d+)\s+(\d+)\s+(\d+)', text)
                    if amount_match:
                        data_points['receipt_amount'] = amount_match.group(3).strip()  # Net amount is usually last
                        data_points['tds'] = amount_match.group(2).strip()  # TDS is usually middle
                        data_points['tds_computed'] = 'No'

        # ICICI Lombard specific method to extract data points
        if "CLAIM_REF_NO" in text and "LAE Invoice No" in text and "TRF AMOUNT" in text:
            logger.info("Detected ICICI Lombard claim table format")

            # Extract first row's data from the table for primary identification
            claim_ref_match = re.search(r'CLAIM_REF_NO.*?\n((?:ENG|MAR|FIR|MSC|LIA)\d+)', text, re.IGNORECASE)
            invoice_match = re.search(r'LAE Invoice No.*?\n.*?(2425-\d{5})', text, re.IGNORECASE)

            if claim_ref_match:
                unique_id = claim_ref_match.group(1).strip()
                logger.info(f"Using claim reference number as unique ID: {unique_id}")

                # Also store invoice number for fallback matching
                if invoice_match:
                    data_points['invoice_no'] = invoice_match.group(1).strip()
                    logger.info(f"Extracted invoice number: {data_points['invoice_no']}")

                # Set the provider for TDS calculation
                detected_provider = "icici_lombard"

        # EMERGENCY FIX FOR STANDARD CHARTERED BAJAJ DOCUMENT - AS FIRST LAYER FOR BAJAJ ALLIANZ
        if ("standard" in text.lower() and "chartered" in text.lower()) and "bajaj allianz" in text.lower():
            # One regex to grab “OC-<digits>-<digits>-<digits>” plus 8 digits, regardless of separators
            match = re.search(r'(OC-\d+-\d+-\d+)\D+(\d{8})', text, re.IGNORECASE)
            if match:
                unique_id = f"{match.group(1)}-{match.group(2)}"
                logger.info(f"EMERGENCY FIX: Extracted full claim ID: {unique_id}")

                # Extract other data from REMITTANCE AMOUNT field (reliable anchor)
                amount_match = re.search(r'REMITTANCE AMOUNT\s*:\s*(\d+(?:\.\d+)?)', text)
                if amount_match:
                    data_points['receipt_amount'] = amount_match.group(1)

                date_match = re.search(r'VALUE DATE\s*[»:]?\s*(\d{2}-[A-Za-z]{3}-\d{4})', text)
                if date_match:
                    data_points['receipt_date'] = date_match.group(1)

                detected_provider = "bajaj_allianz"

        # Special handling for Bajaj Allianz documents with table data
        if detected_provider == "bajaj_allianz":
            # Try specialized table extraction for Bajaj
            table_data_bajaj = self.extract_table_data_from_bajaj(text)

            # If we found table data with valid claim numbers, use the first one as unique_id
            if table_data_bajaj and any('unique_id' in row for row in table_data_bajaj):
                for row in table_data_bajaj:
                    if 'unique_id' in row:
                        # Only override the unique_id if it's not already set or it looks invalid
                        if not unique_id or unique_id == "Gross" or len(unique_id) < 5:
                            unique_id = row['unique_id']
                            logger.info(f"Using claim number from specialized table extraction: {unique_id}")

                        # Also use other data from the table row
                        if 'receipt_amount' in row:
                            data_points['receipt_amount'] = row['receipt_amount']
                        if 'receipt_date' in row:
                            data_points['receipt_date'] = row['receipt_date']
                        if 'tds' in row:
                            data_points['tds'] = row['tds']
                            data_points['tds_computed'] = row.get('tds_computed', 'No')

                        # Also add this to the regular table_data for processing later
                        if not any(td.get('unique_id') == row['unique_id'] for td in table_data):
                            table_data.append(row)
                        break

        # Special debugging for Bajaj Allianz documents
        if detected_provider == "bajaj_allianz":
            logger.info(f"Final Bajaj claim ID check: {unique_id}")

            # Specific debug for this document
            if "OC-24-1501-4089" in text:
                logger.info("Document contains OC-24-1501-4089 pattern")

            # Check for the known format in one line
            if re.search(r'OC-24-1501-4089-00000009', text):
                logger.info("Document contains full claim ID on single line")

            # Look for the split pattern
            if re.search(r'OC-24-1501-4089[\s\n]+00000009', text):
                logger.info("Document contains split claim ID pattern")

        # Reliance - Add this as a fallback check in extract_data_points, just before returning
        if not unique_id or unique_id.startswith('*'):
            # Try to find a standard invoice number pattern (2425-13641)
            invoice_pattern = r'Payment\s+Details\s+5\s*:?\s*(\d{4}-\d{4,5})'
            invoice_match = re.search(invoice_pattern, text, re.IGNORECASE)
            if invoice_match:
                unique_id = invoice_match.group(1).strip()
                logger.info(f"Fallback: Using invoice number from Payment Details 5: {unique_id}")
            else:
                # More general pattern to catch any invoice number
                general_pattern = r'\d{4}-\d{5}'
                general_match = re.search(general_pattern, text)
                if general_match:
                    unique_id = general_match.group(0).strip()
                    logger.info(f"Fallback: Found invoice pattern: {unique_id}")

        # SPECIAL HANDLING for RG Cargo: For tabular invoice documents with header "Invoice No."
        if "Invoice No." in text and len(
                re.findall(r'\d{4}-\d{5}', text)) > 5:  # Many invoice numbers in XXXX-XXXXX format
            logger.info("Detected tabular invoice document with multiple invoice numbers")

            # Extract the first invoice number as the primary identifier
            invoice_matches = re.findall(r'(\d{4}-\d{5})', text)
            if invoice_matches:
                unique_id = invoice_matches[0]  # Use the first invoice number
                logger.info(f"Using first invoice number as unique ID: {unique_id}")

                # Also populate table_data with all invoice numbers
                table_data = self.extract_table_data(text)

                return unique_id, data_points, table_data, detected_provider

        # Add Universal Sompo specific extraction for the full amount
        if detected_provider == "universal_sompo":
            # Try to find the complete amount format with lakhs
            full_amount_match = re.search(r'INR\s+([\d,\.]+)', text, re.IGNORECASE)
            if full_amount_match:
                data_points['receipt_amount'] = full_amount_match.group(1).replace(',', '')
                logger.info(f"Found full amount in text: {data_points['receipt_amount']}")

            # For Universal Sompo, verify against Amount in Words
            if "lakh" in text.lower():
                words_match = re.search(r'Amount in Words\s*:\s*(.*?)(?:Only|Rs)', text, re.IGNORECASE)
                if words_match:
                    amount_words = words_match.group(1).lower()
                    # Check if there's a significant discrepancy
                    try:
                        current_amount = float(data_points.get('receipt_amount', '0'))
                        if "lakh" in amount_words and current_amount < 100000:
                            logger.warning(
                                f"Possible incorrect amount extraction: {data_points.get('receipt_amount')}")

                            # Try to extract from payment paragraph which is more reliable
                            payment_match = re.search(r'payment to RBI.*?amount of INR\s+([\d,\.]+)', text,
                                                      re.IGNORECASE | re.DOTALL)
                            if payment_match:
                                corrected_amount = payment_match.group(1).replace(',', '')
                                logger.info(f"Corrected amount from payment text: {corrected_amount}")
                                data_points['receipt_amount'] = corrected_amount
                    except ValueError:
                        logger.error(
                            f"Could not parse amount for verification: {data_points.get('receipt_amount')}")

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

        # Cholamandalam bank-specific extraction
        if not unique_id or unique_id == ':':
            # Try specific format for payment details section
            payment_details_match = re.search(r'INV REF:Claim No:\s*(\d+)', text, re.IGNORECASE)
            if payment_details_match:
                unique_id = payment_details_match.group(1).strip()
                logger.info(f"Found unique ID from Cholamandalam payment details: {unique_id}")

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

        # Check if F0036935 pattern exists in any of the table rows
        f_pattern_rows = [row for row in table_data if
                          'unique_id' in row and re.match(r'F\d{7}', row['unique_id']) and 'Claims Payment' in row.get(
                              'payment_type', '')]

        if f_pattern_rows:
            # Use the first match as the primary unique ID
            unique_id = f_pattern_rows[0]['unique_id']
            logger.info(f"Found primary Future Generali claim ID in table: {unique_id}")

            # Also update receipt_amount if needed
            if 'receipt_amount' in f_pattern_rows[0]:
                cleaned_amount = f_pattern_rows[0]['receipt_amount'].lstrip('0')
                if cleaned_amount.startswith('.'):
                    cleaned_amount = '0' + cleaned_amount
                data_points['receipt_amount'] = cleaned_amount

        # Ensure we're using the correct unique ID for Excel matching
        if table_data and unique_id:
            # Force the table_data entries to use the validated unique ID
            for row in table_data:
                if 'unique_id' in row:
                    row['unique_id'] = unique_id

        if unique_id and re.search(r'ph.*?' + re.escape(unique_id), text.lower()):
            logger.warning(f"Identified ID {unique_id} is likely a phone number - skipping")
            unique_id = None

        if not unique_id and table_data:
            # Use the first entry's unique_id from table_data
            unique_id = table_data[0].get('unique_id')
            logger.info(f"Using unique ID from table data: {unique_id}")

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
        self.validate_extracted_data(unique_id, data_points, text)

        # Return the results
        return unique_id, data_points, table_data, detected_provider

    def validate_extracted_data(self, unique_id, data_points, text=None):
        """
        Validate extracted data to ensure correct format.

        Args:
            unique_id (str): The extracted unique identifier
            data_points (dict): Dictionary of extracted data points
            text (str, optional): The full PDF text for context
        """
        # Validate receipt amount
        if 'receipt_amount' in data_points:
            amount = data_points['receipt_amount']
            try:
                # Handle European number format (periods as thousands separators)
                if '.' in amount and amount.count('.') > 1:
                    # Format like "9.989.00" - replace thousands separator and keep decimal
                    parts = amount.split('.')
                    if len(parts) == 3:  # Like 9.989.00
                        clean_amount = parts[0] + parts[1] + '.' + parts[2]
                        amount = clean_amount

                # Clean any remaining non-numeric characters
                amount = re.sub(r'[^0-9.]', '', amount)

                # Convert to float
                amount_float = float(amount)

                # Format with 2 decimal places
                data_points['receipt_amount'] = str(amount_float)

            except ValueError:
                logger.warning(f"Invalid receipt amount format: {amount}")

                # For Cera documents, check for specific patterns indicating possible errors
                if text and 'cera' in text.lower():
                    # Issue 1: Account number mistaken for amount
                    account_pattern = r'Your\s*account\s*with\s*us\s*(\d{6})'
                    account_match = re.search(account_pattern, text)

                    if account_match and str(int(amount_float)) == account_match.group(1):
                        logger.warning(f"Detected account number mistakenly used as amount: {amount_float}")

                        # Look for Net Total as a better source
                        net_total_match = re.search(
                            r'Net\s*Total\s*(?:Rs\.?|INR)?\s*(\d{1,3}(?:[,.]\d{3})*(?:[,.]\d{1,2})?)',
                            text, re.IGNORECASE)
                        if net_total_match:
                            corrected_amount = net_total_match.group(1).replace(',', '.')
                            data_points['receipt_amount'] = corrected_amount
                            logger.info(f"Corrected amount using Net Total: {corrected_amount}")
                            amount_float = float(corrected_amount)

                    # Issue 2: Amount seems unreasonably large
                    elif amount_float > 50000:  # Most Cera invoices unlikely to be this large
                        # Search for Net Total
                        net_total_match = re.search(
                            r'Net\s*Total\s*(?:Rs\.?|INR)?\s*(\d{1,3}(?:[,.]\d{3})*(?:[,.]\d{1,2})?)',
                            text, re.IGNORECASE)
                        if net_total_match:
                            corrected_amount = net_total_match.group(1).replace(',', '.')
                            logger.warning(
                                f"Amount {amount_float} suspiciously large, using Net Total: {corrected_amount}")
                            data_points['receipt_amount'] = corrected_amount
                            amount_float = float(corrected_amount)

                # Update with validated value
                data_points['receipt_amount'] = str(amount_float)
                logger.debug(f"Validated receipt amount: {data_points['receipt_amount']}")
            except ValueError:
                logger.warning(f"Invalid receipt amount format: {amount}")

                # Recovery strategy - try alternative formatting
                try:
                    # Try simple cleanup as last resort
                    clean_amount = amount.replace('.', '')
                    if ',' in clean_amount:
                        clean_amount = clean_amount.replace(',', '.')
                    amount_float = float(clean_amount)
                    data_points['receipt_amount'] = str(amount_float)
                    logger.info(f"Recovered amount using cleanup: {amount} -> {data_points['receipt_amount']}")
                except:
                    # If all else fails, keep the original string
                    logger.error(f"Could not convert amount: {amount}")

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
                    '%d/%m/%Y',  # 01/01/2025 (dd/mm/yyyy)
                    '%d/%m/%y',  # 01/01/25' (dd/mm/yy)
                    '%m/%d/%Y',  # 01/01/2025 (mm/dd/yyyy)
                    '%m/%d/%y',  # 01/01/25 (mm/dd/yy)
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
        Compute TDS based on the logic provided.
        """
        try:
            # Skip unnecessary import check
            from app.utils.insurance_providers import INSURANCE_PROVIDERS, SPECIFIC_TDS_RATE_PROVIDERS, \
                NEW_INDIA_THRESHOLD

            # Normalize text for matching
            normalized_text = ' '.join(text.lower().split())
            logger.debug(f"Normalized text sample for TDS: {normalized_text[:100]}...")

            # Default to non-specific rate
            contains_insurance_company = False

            # Check if any provider keywords are in the text
            for provider, keywords in INSURANCE_PROVIDERS.items():
                if any(keyword.lower() in normalized_text for keyword in keywords):
                    contains_insurance_company = True
                    detected_provider = provider
                    logger.info(f"Detected provider for TDS: {provider}")
                    break

            # Apply appropriate TDS calculation
            if contains_insurance_company and detected_provider in SPECIFIC_TDS_RATE_PROVIDERS:
                # Special handling for New India with threshold
                if detected_provider == "new_india" and amount <= NEW_INDIA_THRESHOLD:
                    tds = round(amount * 0.09259259, 2)
                    logger.info(f"TDS computed for New India (<=300000): {tds} (9.259259% of {amount})")
                else:
                    tds = round(amount * 0.11111111, 2)
                    logger.info(f"TDS computed for specific provider: {tds} (11.111111% of {amount})")
            else:
                tds = round(amount * 0.09259259, 2)
                logger.info(f"TDS computed for non-specific provider: {tds} (9.259259% of {amount})")

            return tds, True
        except Exception as e:
            logger.error(f"Error computing TDS: {str(e)}")
            return 0.0, True

    def extract_table_data(self, text):
        """
        Extract data from tables in the PDF with enhanced pattern recognition.
        """
        table_data = []

        # 0. Check for insurance payment advice formats
        if "INVOICE NUMBER" in text and "CLAIM NUMBER" in text:
            # Get the table structure
            start_idx = max(0, text.find("INVOICE NUMBER") - 10)  # Include some context
            end_idx = text.find("TOTAL:", start_idx)
            if end_idx == -1:  # If "TOTAL:" not found, try another boundary
                end_idx = text.find("=====", start_idx)
            if end_idx == -1:  # If still not found, just get 500 chars
                end_idx = start_idx + 500

            # Get a reasonable sample
            debug_sample = text[start_idx:end_idx]
            # Print each line separately for better structure visibility
            logger.info("DEBUG: Table structure:")
            for line_num, line in enumerate(debug_sample.split('\n')):
                logger.info(f"Line {line_num}: {line}")

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
        if "cera sanitaryware" in text.lower() or "cerasanitaryware" in text.lower():
            # Reuse the specialized extraction to avoid duplication
            _, cera_table_data = self.extract_cera_invoice_numbers(text)
            table_data.extend(cera_table_data)
            return table_data  # Return early to avoid duplicate processing

            # Use specialized extraction for Cera invoice numbers
            invoice_numbers = self.extract_cera_invoice_numbers(text)

            # Find the table contents with amounts
            table_content_pattern = r'Document\s*No.*?Bill\s*No.*?Bill\s*Date.*?TDS\s*Amount.*?Net\s*Amount'
            table_match = re.search(table_content_pattern, text, re.IGNORECASE | re.DOTALL)

            table_text = text
            if table_match:
                table_text = text[table_match.start():]

            # Process each extracted invoice number
            for invoice_no in invoice_numbers:
                # Look for amount patterns near the invoice number
                invoice_pos = table_text.find(invoice_no)
                if invoice_pos == -1:
                    # Try finding the original (uncleaned) version
                    for i in range(invoice_pos - 100, invoice_pos + 100):
                        if i >= 0 and i < len(table_text):
                            chunk = table_text[i:i + 20]
                            if re.search(r'\d\w{3}-\d\w{4}', chunk):
                                invoice_pos = i
                                break

                if invoice_pos >= 0:
                    # Extract 150 characters after the invoice position - should include amounts
                    context = table_text[invoice_pos:invoice_pos + 150]

                    # Look for amount patterns
                    amount_matches = re.findall(r'(\d+[,\.\d]+)', context)

                    # Need at least 3 numbers for gross, TDS, and net
                    if len(amount_matches) >= 3:
                        # Convert to proper formats
                        amounts = [amt.replace(',', '') for amt in amount_matches]

                        row_data = {
                            'unique_id': invoice_no,
                            'gross_amount': amounts[0] if len(amounts) > 0 else "",
                            'tds': amounts[1] if len(amounts) > 1 else "",
                            'receipt_amount': amounts[2] if len(amounts) > 2 else ""
                        }

                        table_data.append(row_data)
                        logger.debug(f"Extracted Cera table row for invoice: {invoice_no}")

        # 4. ICICI Lombard reference table
        icici_table_match = re.search(
            r'CLAIM_REF_NO\s+LAE\s+Invoice\s+No\s+Bill\s+Date\s+Invoice\s+Amt\s+TDS\s+TRF\s+AMOUNT\s+RECEIPT\s+DATE',
            text, re.IGNORECASE)

        if icici_table_match:
            logger.info("Detected ICICI Lombard table format with receipt date")

            # Use a more flexible pattern that can handle different date formats and number formats
            rows = re.findall(
                r'((?:ENG|MAR|FIR|MSC|LIA)\d+)\s+(2425-\d{5})\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+([\d,\.]+)\s+([\d,\.]+)\s+([\d,\.]+)\s+(\d{2}-\d{2}-\d{4})',
                text)

            if rows:
                logger.info(f"Extracted {len(rows)} rows from ICICI Lombard table")

                for row in rows:
                    claim_ref, invoice_no, bill_date, invoice_amt, tds, trf_amount, receipt_date = row

                    # Transform amounts - remove commas and ensure proper format
                    receipt_amount = trf_amount.replace(',', '')
                    tds_amount = tds.replace(',', '')

                    # Create row data with both identifiers
                    row_data = {
                        'unique_id': claim_ref.strip(),  # Primary ID is the claim reference
                        'invoice_no': invoice_no.strip(),  # Store invoice number for fallback matching
                        'receipt_date': receipt_date.strip(),
                        'receipt_amount': receipt_amount.strip(),
                        'tds': tds_amount.strip(),
                        'tds_computed': 'No'  # TDS is already in the document
                    }

                    table_data.append(row_data)
                    logger.info(f"Added ICICI Lombard table row: {claim_ref} / {invoice_no}")

        # 5. Zion reference table (spotted in sample 34)
        zion_table_match = re.search(r'DATE\s+ZION\s+REF\s+NO\.\s+bill\s+no\s+(?:amt|Bill\s+Amount)', text,
                                     re.IGNORECASE)
        if zion_table_match:
            # Find the table content lines
            table_content = text[zion_table_match.end():]
            table_end = table_content.find("TOTAL PAID AMT")
            if table_end > 0:
                table_content = table_content[:table_end]

            # Extract rows using regex pattern matching - WITH RECEIPT DATE
            rows = re.findall(
                r'(\d{2}-\d{2}-\d{4})\s+(JPB[\d/\-]+)\s+(\d{4}-\d{5})\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d{2}-\d{2}-\d{4})',
                table_content)

            for row in rows:
                # Unpack 7 values including receipt date
                date, ref_no, bill_no, amount, tds, receipt_amount, receipt_date = row
                table_data.append({
                    'unique_id': bill_no.strip(),
                    'reference_no': ref_no.strip(),
                    'receipt_date': receipt_date.strip(),  # Use the explicit Receipt Date
                    'bill_amount': amount.strip(),
                    'tds': tds.strip(),
                    'receipt_amount': receipt_amount.strip(),
                    'tds_computed': 'No'  # TDS already present in document
                })

            # Check for any rows with a different format (e.g., missing receipt date column)
            if not rows:
                # Alternative pattern without receipt date
                alt_rows = re.findall(r'(\d{2}-\d{2}-\d{4})\s+(JPB[\d/\-]+)\s+(\d{4}-\d{5})\s+(\d+)\s+(\d+)\s+(\d+)',
                                      table_content)

                for row in alt_rows:
                    # Make sure to unpack only 6 values here
                    date, ref_no, bill_no, amount, tds, receipt_amount = row

                    # Search for receipt date in the nearby text
                    receipt_date_match = re.search(r'Receipt\s+Date\s*[:.\s]*\s*(\d{2}-\d{2}-\d{4})',
                                                   table_content[
                                                   table_content.find(bill_no):table_content.find(bill_no) + 100])

                    receipt_date = receipt_date_match.group(
                        1).strip() if receipt_date_match else "08-01-2025"  # Default from document

                    table_data.append({
                        'unique_id': bill_no.strip(),
                        'reference_no': ref_no.strip(),
                        'receipt_date': receipt_date,
                        'bill_amount': amount.strip(),
                        'tds': tds.strip(),
                        'receipt_amount': receipt_amount.strip(),
                        'tds_computed': 'No'
                    })

            logger.info(f"Extracted {len(table_data)} rows from Zion table")

        # 6. Look for Liberty payment advice tables with Gross, TDS and Net columns
        liberty_table_match = re.search(r'(\d+)\s+Gross\s+Amount\s+TDS\s+Net\s+Amount\s*\n\s*(\d+)\s+(\d+)\s+(\d+)',
                                        text, re.IGNORECASE)
        if not liberty_table_match:
            # Try alternative pattern
            liberty_table_match = re.search(r'Gross\s+Amount\s+TDS\s+Net\s+Amount\s*\n\s*(\d+)\s+(\d+)\s+(\d+)', text,
                                            re.IGNORECASE)

        if liberty_table_match:
            # Get the values based on which pattern matched
            if len(liberty_table_match.groups()) == 4:
                ref_id = liberty_table_match.group(1).strip()
                gross_amount = liberty_table_match.group(2).strip()
                tds_amount = liberty_table_match.group(3).strip()
                net_amount = liberty_table_match.group(4).strip()
            else:
                gross_amount = liberty_table_match.group(1).strip()
                tds_amount = liberty_table_match.group(2).strip()
                net_amount = liberty_table_match.group(3).strip()
                ref_id = None  # Will use any existing unique_id

            row_data = {
                'unique_id': ref_id if ref_id else unique_id,
                'receipt_amount': net_amount,  # Use net amount as receipt_amount
                'tds': tds_amount,
                'tds_computed': 'No'  # TDS directly extracted
            }
            table_data.append(row_data)
            logger.info(f"Extracted Liberty payment table with Net Amount: {net_amount}, TDS: {tds_amount}")

        # 7. Special handling for Future Generali table format
        future_generali_table = re.search(r'Ref\s+NO\.\.\.\/Registration\s+No\.\/Invoice\s+Details.*?Amount', text,
                                          re.IGNORECASE)
        if future_generali_table:
            # Extract rows using the specific table format of Future Generali
            table_content = text[future_generali_table.end():]
            # Find the end of table
            end_pos = table_content.find("Corporate Identity")
            if end_pos > 0:
                table_content = table_content[:end_pos]

            # Look for patterns like: HDF_20250108/Not Applic/F0036935 Claims Payment 000011547.00
            fg_rows = re.findall(
                r'(HDF_\d+/Not\s+Applic/(?:F\d{7}|[^/\s]+))\s+(Claims\s+Payment|Intermediary[^0-9]+)\s+(\d+\.\d{2})',
                table_content)

            for row in fg_rows:
                ref_with_invoice = row[0].strip()
                payment_type = row[1].strip()
                amount = row[2].strip()

                # Extract the invoice/claim number part (F0036935)
                invoice_match = re.search(r'F(\d{7})', ref_with_invoice)
                if invoice_match:
                    row_data = {
                        'unique_id': f"F{invoice_match.group(1)}",
                        'payment_type': payment_type
                    }

                    # Use the actual amount from the table, not the overall document amount
                    if "Claims Payment" in payment_type:
                        row_data['receipt_amount'] = amount
                    elif "Intermediary" in payment_type and "TDS" in text[table_content.find(
                            ref_with_invoice) - 100:table_content.find(ref_with_invoice)]:
                        row_data['tds'] = amount
                        row_data['tds_computed'] = 'No'  # Directly extracted

                    # Only add if it has useful data and not already present
                    if 'receipt_amount' in row_data and not any(
                            td.get('unique_id') == row_data['unique_id'] for td in table_data):
                        table_data.append(row_data)
                        logger.info(f"Extracted Future Generali table row: {row_data}")

        # 8. Add specific check for HDFC ERGO format with invoice number in PAY_TYPE
        if "hdfc ergo" in text.lower():
            logger.info(f"Started table data exxtraction for HDFC ERGO table")
            # More flexible pattern to match various financial year prefixes
            # This will match patterns like 2425-XXXXX, 2526-XXXXX, etc.
            paytype_pattern = r'PAY_TYPE\s*\|\s*Expense\s+Paid(\d{4}-\d{5})'
            paytype_match = re.search(paytype_pattern, text)

            if paytype_match:
                invoice_no = paytype_match.group(1)
                logger.info(f"Found HDFC ERGO invoice number: {invoice_no}")

                # Extract claim number as well for potential matching
                claim_pattern = r'Claim\s+No\s*\|\s*(C\d+(-\d+)?)'
                claim_match = re.search(claim_pattern, text)
                claim_no = None

                if claim_match:
                    claim_no = claim_match.group(1)
                    logger.info(f"Found HDFC ERGO claim number: {claim_no}")

                # Don't override the unique_id - we'll try both identifiers in find_row_by_identifiers

                # Add both identifiers to table_data
                row_data = {
                    'unique_id': claim_no if claim_no else unique_id,  # Primary ID remains claim number if available
                    'invoice_no': invoice_no,  # Also store invoice number
                }

                # Add this row to table data
                table_data.append(row_data)

        return table_data

        # 9. Add specific handling for New India Assurance table format
        if "INVOICE NUMBER POLICY NUMBER CLAIM NUMBER AMOUNT" in text:
            logger.info("Detected New India Assurance table format")

            # Extract the table section
            table_start = text.find("INVOICE NUMBER")
            table_end = text.find("TOTAL:", table_start)

            if table_start >= 0 and table_end >= 0:
                table_section = text[table_start:table_end]

                # Pattern to match table rows: invoice policy claim amount
                table_row_pattern = r'(51\d+\w+)\s+(\d{20})\s+(\d{20})\s+([\d,\.]+)'
                rows = re.findall(table_row_pattern, table_section)

                for row in rows:
                    invoice_no, policy_no, claim_no, amount = row

                    # Clean and validate amount
                    clean_amount = amount.replace(',', '')
                    try:
                        amount_val = float(clean_amount)
                        if amount_val > 0:  # Only positive amounts
                            table_data.append({
                                'unique_id': claim_no,  # Use claim number (3rd column)
                                'invoice_no': invoice_no,
                                'policy_no': policy_no,
                                'receipt_amount': clean_amount
                            })
                            logger.info(f"Added New India claim {claim_no} with amount {clean_amount}")
                    except ValueError:
                        logger.warning(f"Invalid amount format: {amount}")
                        continue

        # 10. Check for RG Cargo table format (specific to your document)
        if "Invoice No. Invoice Date F.Y. Client" in text and "RG Cargo Services Private Limited" in text:
            logger.info("Detected RG Cargo Services invoice table format")

            # Extract rows with invoice pattern
            rows = re.findall(
                r'(\d{4}-\d{5})\s+(\d{2}-\d{2}-\d{4})\s+(\d{4}-\d{2})\s+([^\n]+?)(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(-\d+)',
                text)

            for row in rows:
                invoice_no, invoice_date, fy, client, total, igst, total_amount, tds, net = row

                row_data = {
                    'unique_id': invoice_no.strip(),
                    'receipt_date': invoice_date.strip(),
                    'receipt_amount': net.strip(),  # Note: The net amount is negative in your document
                    'tds': tds.strip(),
                    'tds_computed': 'No'  # TDS is already in the document
                }

                table_data.append(row_data)

            logger.info(f"Extracted {len(table_data)} rows from RG Cargo invoice table")

        # 11. Look for table with invoice numbers and amounts (generic pattern)
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

        # 11. Generic pattern for invoice numbers followed by amounts
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

    def extract_table_data_from_bajaj(self, text):
        """
        Extract table data specifically from Bajaj Allianz format PDFs where data might be split across lines.

        Args:
            text (str): Extracted PDF text

        Returns:
            list: List of dictionaries containing extracted table data
        """
        table_data = []

        # Look for the table pattern specific to Bajaj Allianz advices
        table_pattern = r'(?:Appr|Ref)\s+Date\s+Description\s+Claim\s+No\s+Gross\s+(?:Amt|Amount)\s+(?:Ser\s+Tax|Service\s+Tax)\s+TDS\s+(?:Amt|Amount)\s+Amount\s+\(INR\)'
        table_match = re.search(table_pattern, text, re.IGNORECASE)

        if table_match:
            # Extract the content after the table header
            table_content = text[table_match.end():]

            # Find the end of the table content
            end_markers = ["Note:", "----", "This is a system"]
            for marker in end_markers:
                end_pos = table_content.find(marker)
                if end_pos > 0:
                    table_content = table_content[:end_pos].strip()
                    break

            # Split the content into lines
            lines = table_content.strip().split('\n')

            # Process rows by combining lines if needed
            i = 0
            while i < len(lines):
                line = lines[i].strip()

                # Skip empty lines
                if not line:
                    i += 1
                    continue

                # Check if this looks like the start of a data row (reference number)
                if re.match(r'^\d{4}\s+\d{2}/\d{2}/\d', line):
                    # This is likely a row start
                    full_row = line

                    # Check if the next line might be a continuation
                    if i + 1 < len(lines) and not re.match(r'^\d{4}\s+\d{2}/\d{2}/\d', lines[i + 1]):
                        full_row += " " + lines[i + 1].strip()
                        i += 2
                    else:
                        i += 1

                    # Now extract the data using a more specific pattern
                    # Looking for values in a specific order
                    row_data = {}

                    # Extract reference number and date
                    ref_match = re.search(r'(\d{4,})\s+(\d{2}/\d{2}/\d{1,4})', full_row)
                    if ref_match:
                        row_data['reference'] = ref_match.group(1).strip()
                        row_data['receipt_date'] = ref_match.group(2).strip()

                    # Extract description number
                    desc_match = re.search(r'(\d{2}/\d{2}/\d{1,4})\s+(\d+)', full_row)
                    if desc_match:
                        row_data['description'] = desc_match.group(2).strip()

                    # Extract claim number - CRITICAL FIX HERE
                    claim_pattern = r'OC-\d+-\d+-\d+-\d+'  # Pattern to match entire claim number including splits
                    claim_match = re.search(claim_pattern, full_row)
                    if claim_match:
                        row_data['unique_id'] = claim_match.group(0).strip()
                    else:
                        # Try a more flexible pattern to handle split claim numbers
                        claim_parts = re.findall(r'OC-\d+-\d+-\d+[-\s]*(\d+)?', full_row)
                        if claim_parts:
                            # If we have a partial match, look for continuation in adjacent lines
                            partial_id = re.search(r'(OC-\d+-\d+-\d+)', full_row).group(1)

                            # Check next line for remaining digits if we don't have a complete ID
                            continuation = ""
                            if i < len(lines) and not re.match(r'^\d{4}\s+\d{2}/\d{2}/\d', lines[i].strip()):
                                continuation = re.search(r'^\s*(\d+)', lines[i].strip())
                                if continuation:
                                    continuation = continuation.group(1)
                                    i += 1  # Advance if we consumed the next line

                            # Combine parts to form complete ID
                            row_data['unique_id'] = f"{partial_id}-{continuation}" if continuation else partial_id

                    # Extract amount values using positions or patterns
                    amounts = re.findall(r'(\d+(?:\.\d+)?)', full_row)
                    if len(amounts) >= 4:  # We need at least reference, date, description, and values
                        idx = 3  # Start from the 4th number (after ref, date, desc)
                        if len(amounts) > idx:
                            row_data['gross_amount'] = amounts[idx]
                            idx += 1
                        if len(amounts) > idx:
                            row_data['service_tax'] = amounts[idx]
                            idx += 1
                        if len(amounts) > idx:
                            row_data['tds'] = amounts[idx]
                            row_data['tds_computed'] = 'No'  # TDS is directly from document
                            idx += 1
                        if len(amounts) > idx:
                            row_data['receipt_amount'] = amounts[idx]

                    # Add the row data if we have the essential fields
                    if 'unique_id' in row_data and 'receipt_amount' in row_data:
                        table_data.append(row_data)
                        logger.info(f"Extracted table row with claim ID: {row_data['unique_id']}")
                else:
                    i += 1

        return table_data

    def process_pdf(self, pdf_path, expected_fields=None):
        """Process a single PDF with enhanced error handling."""
        if not os.path.exists(pdf_path):
            logger.error(f"File not found: {pdf_path}")
            return None, {}, []

        try:
            # Extract text from PDF
            extracted_text = self.extract_text(pdf_path)

            if not extracted_text:
                logger.warning(f"No text extracted from {pdf_path}")
                # Try alternative extraction methods
                extracted_text = self.text_extractor._fallback_ocr_extraction(pdf_path)

                if not extracted_text:
                    logger.error(f"All extraction methods failed for {pdf_path}")
                    return None, {}, []

            # Extract data points
            unique_id, data_points, table_data, detected_provider = self.extract_data_points(
                extracted_text, expected_fields
            )

            return unique_id, data_points, table_data

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return None, {}, []