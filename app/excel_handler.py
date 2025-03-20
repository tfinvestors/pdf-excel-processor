import os
import pandas as pd
import openpyxl
import logging
import re
from openpyxl.styles import PatternFill, Font
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("excel_processing.log")
    ]
)
logger = logging.getLogger("excel_handler")


class ExcelHandler:
    def __init__(self, excel_path):
        """
        Initialize the Excel handler.

        Args:
            excel_path (str): Path to the Excel file
        """
        self.excel_path = excel_path
        self.df = None
        self.wb = None
        self.ws = None
        self.invoice_column = None
        self.client_ref_column = None
        self.header_mapping = {}

        # Load the Excel file
        self.load_excel()

    def load_excel(self):
        """Load the Excel file into a pandas DataFrame and openpyxl Workbook."""
        try:
            # Load with pandas for data manipulation
            self.df = pd.read_excel(self.excel_path)
            logger.info(f"Loaded Excel file with {len(self.df)} rows and {len(self.df.columns)} columns")

            # Also load with openpyxl for direct cell manipulation
            self.wb = openpyxl.load_workbook(self.excel_path)
            self.ws = self.wb.active

            # Identify the ID columns (Invoice No. and Client Ref. No.)
            self.identify_id_columns()

            # Create header mapping
            self.create_header_mapping()

        except Exception as e:
            logger.error(f"Error loading Excel file: {str(e)}")
            raise

    def identify_id_columns(self):
        """Identify the columns for Invoice No. and Client Ref. No."""
        # Search for Invoice Number column
        invoice_columns = [
            'Invoice No.', 'Invoice No', 'Invoice Number', 'Invoice',
            'Inv No.', 'Inv No', 'Inv Number', 'Invoice #', 'Bill No.',
            'Bill No', 'Payment Details 5', 'ThirdPartyInv', 'ILA_REF_NO',
            'Expense Paid', 'Invoice'
        ]

        # Search for Client Reference Number column
        client_ref_columns = [
            'Client Ref. No.', 'Client Ref. No', 'Client Reference Number',
            'Client Ref', 'Ref. No.', 'Reference No.', 'Reference Number', 'Claim#', 'Claim Number', 'CLAIM NUMBER',
            'Claim number', 'Sub Claim No', 'INV REF:Claim No', 'Claim No', 'DESC', 'Payment Details 7',
            'ClaimNo', 'Claim_Ref_No', 'CLAIM_REF_NO', 'Invoice Details'
        ]

        # Find exact matches first
        for col in invoice_columns:
            if col in self.df.columns:
                self.invoice_column = col
                logger.info(f"Found Invoice column: {col}")
                break

        for col in client_ref_columns:
            if col in self.df.columns:
                self.client_ref_column = col
                logger.info(f"Found Client Reference column: {col}")
                break

        # If exact matches not found, look for partial matches
        if not self.invoice_column:
            for col in self.df.columns:
                if any(invoice.lower() in col.lower() for invoice in invoice_columns):
                    self.invoice_column = col
                    logger.info(f"Found Invoice column (partial match): {col}")
                    break

        if not self.client_ref_column:
            for col in self.df.columns:
                if any(ref.lower() in col.lower() for ref in client_ref_columns):
                    self.client_ref_column = col
                    logger.info(f"Found Client Reference column (partial match): {col}")
                    break

        # If still not found, use the first two columns as fallback
        if not self.invoice_column and len(self.df.columns) > 0:
            self.invoice_column = self.df.columns[0]
            logger.warning(f"No obvious Invoice column found. Using first column: {self.invoice_column}")

        if not self.client_ref_column and len(self.df.columns) > 1:
            self.client_ref_column = self.df.columns[1]
            logger.warning(f"No obvious Client Reference column found. Using second column: {self.client_ref_column}")

    def create_header_mapping(self):
        """Create a mapping between common field names and actual column names."""
        # Define common field names and possible variants
        field_mappings = {
            'receipt_date': ['Receipt Date', 'Receipt Date', 'Payment Date', 'Value Date', 'Date', 'Payment Ini Date',
                             'Value date', 'Advice sending date', 'Settlement Date', 'Value Date', 'VALUE DATE'],
            'receipt_amount': ['Receipt Amount', 'Receipt Amt', 'Payment Amount', 'Amount', 'Value', 'Amount',
                               'Remittance amount', 'Net Paid Amount', 'REMITTANCE AMOUNT', 'Net Amount',
                               'Amount (INR)', 'AMOUNT', 'Payment amount', 'TRF AMOUNT'],
            'tds': ['TDS', 'TDS Amount', 'Tax Deducted at Source', 'Tax Amount', 'TDS Amount', 'TDS', 'TDS Amt',
                    'Payment Details 4', 'Less : TDS'],
            'tds_computed': ['TDS Computed?', 'TDS Computed', 'Is TDS Computed', 'Computed TDS']
        }

        # Create mapping between standardized field names and actual column names
        for field, variants in field_mappings.items():
            for col in self.df.columns:
                if col in variants or any(variant.lower() in col.lower() for variant in variants):
                    self.header_mapping[field] = col
                    break

        logger.info(f"Created header mapping: {self.header_mapping}")

    def compute_tds(self, amount, text, detected_provider=None):
        """
        Compute TDS based on the receipt amount and text content.
        Applies 11.111111% for specific insurance companies and 9.259259% for others.
        Special handling for New India Assurance with threshold of 300000.

        Args:
            amount (float): The receipt amount
            text (str): The text content from the PDF
            detected_provider (str, optional): The insurance provider already detected

        Returns:
            tuple: (tds_value, is_computed)
        """
        try:
            # Normalize the text for better matching
            normalized_text = ' '.join(text.lower().split())
            logger.debug(f"Normalized text for insurance detection (first 300 chars): {normalized_text[:300]}...")

            # Flag to track if one of the specific insurance companies is detected
            is_specific_insurance = False
            detected_company = None

            # If a provider was already detected, use it directly
            if detected_provider:
                logger.info(f"Using previously detected insurance provider: {detected_provider}")

                # Map the detected provider to our specific insurance company names
                if detected_provider in ["oriental", "united_india", "new_india", "national"]:
                    is_specific_insurance = True
                    detected_company = detected_provider
                    logger.info(f"Using previously detected specific insurance company: {detected_company}")
            else:
                # Define the specific insurance companies that should get 11.111111% rate
                specific_insurance_companies = {
                    "oriental": ["oriental insurance", "oicl", "the oriental insurance co", "oriental insurance co",
                                 "oriental", "0rienta1"],
                    "united_india": ["united india insurance", "united india insurance company", "united india"],
                    "new_india": ["new india assurance", "new india assurance co", "new india"],
                    "national": ["national insurance", "national insurance company", "national"]
                }

                # Check for specific insurance companies with more precise matching
                for company_name, keywords in specific_insurance_companies.items():
                    for keyword in keywords:
                        # Use more precise matching to avoid false positives
                        if keyword in normalized_text:
                            # For single word keywords, ensure they're not part of other words
                            if len(keyword.split()) == 1:
                                # Look for word boundaries or check if it's a complete match
                                pattern = r'\b' + re.escape(keyword) + r'\b'
                                if re.search(pattern, normalized_text):
                                    is_specific_insurance = True
                                    detected_company = company_name
                                    logger.info(
                                        f"Detected specific insurance company: {company_name} (keyword: {keyword})")
                                    break
                            else:
                                # Multi-word keywords are more specific already
                                is_specific_insurance = True
                                detected_company = company_name
                                logger.info(f"Detected specific insurance company: {company_name} (keyword: {keyword})")
                                break
                    if is_specific_insurance:
                        break

                # Special handling for HSBC remitter info
                if not is_specific_insurance and "hsbc" in normalized_text:
                    remitter_match = re.search(
                        r'remitter.*?(?:name|information).*?(?:oriental|national|united|new\s+india)',
                        normalized_text, re.IGNORECASE | re.DOTALL)
                    if remitter_match:
                        remitter_text = remitter_match.group(0).lower()
                        # Check which specific company is mentioned
                        if "oriental" in remitter_text:
                            is_specific_insurance = True
                            detected_company = "oriental"
                        elif "national" in remitter_text:
                            is_specific_insurance = True
                            detected_company = "national"
                        elif "united" in remitter_text:
                            is_specific_insurance = True
                            detected_company = "united_india"
                        elif "new india" in remitter_text:
                            is_specific_insurance = True
                            detected_company = "new_india"

                        if is_specific_insurance:
                            logger.info(
                                f"Detected specific insurance company from HSBC remitter info: {detected_company}")

            # Apply the appropriate TDS rate based on detection results
            if is_specific_insurance:
                # Special handling for New India Assurance
                if detected_company == "new_india" and amount <= 300000:
                    tds = round(amount * 0.09259259, 2)  # 9.259259% for amounts <= 300000
                    logger.info(
                        f"TDS computed for New India Assurance (amount <= 300000): {tds} (9.259259% of {amount})")
                else:
                    tds = round(amount * 0.11111111, 2)  # 11.111111% for specific insurance companies
                    logger.info(
                        f"TDS computed for specific insurance company ({detected_company}): {tds} (11.111111% of {amount})")
            else:
                tds = round(amount * 0.09259259, 2)  # 9.259259% for all other companies
                logger.info(f"TDS computed for non-specific company: {tds} (9.259259% of {amount})")

            return tds, True

        except Exception as e:
            logger.error(f"Error computing TDS: {str(e)}")
            return 0.0, True

    def find_row_by_identifiers(self, unique_id, data_points, pdf_text=None):
        """
        Find a row in the DataFrame by matching either Invoice No. or Client Ref. No.
        Enhanced with more robust matching strategies for all PDF types.
        """
        if not unique_id:
            return None

        try:
            # Log the unique_id for debugging
            logger.info(f"Attempting to match ID: {unique_id}")

            # Clean the unique_id to improve matching chances
            # Remove any unwanted characters that might interfere with matching
            cleaned_id = re.sub(r'[^A-Za-z0-9/-]', '', unique_id)
            logger.info(f"Cleaned ID for matching: {cleaned_id}")

            # Extract just the numeric part if it's a complex ID
            numeric_part = re.sub(r'[^0-9]', '', unique_id)
            if len(numeric_part) >= 5:  # Only use if it's a substantial number
                logger.info(f"Numeric part of ID: {numeric_part}")
            else:
                numeric_part = None

            # For claim numbers with format like 510000/11/2025/000000, extract various parts
            claim_parts = None
            claim_last_part = None
            claim_first_part = None
            if '/' in cleaned_id:
                claim_parts = cleaned_id.split('/')
                claim_last_part = claim_parts[-1]
                claim_first_part = claim_parts[0]
                logger.info(f"Claim parts: {claim_parts}")
                logger.info(f"Last part of claim number: {claim_last_part}")
                logger.info(f"First part of claim number: {claim_first_part}")

            # For invoice numbers with format like 2425-12345, extract parts
            invoice_parts = None
            invoice_prefix = None
            invoice_number = None
            if '-' in cleaned_id:
                invoice_parts = cleaned_id.split('-')
                invoice_prefix = invoice_parts[0]
                invoice_number = invoice_parts[-1] if len(invoice_parts) > 1 else None
                logger.info(f"Invoice parts: {invoice_parts}")
                logger.info(f"Invoice prefix: {invoice_prefix}")
                logger.info(f"Invoice number: {invoice_number}")

            # Create multiple matching strategies
            strategies = [
                # Strategy 1: Exact match with cleaned ID
                lambda col_series: col_series[col_series.str.lower() == cleaned_id.lower()],

                # Strategy 2: Contains match with cleaned ID
                lambda col_series: col_series[col_series.str.contains(cleaned_id, case=False, na=False)]
                if len(cleaned_id) >= 5 else pd.Series(),

                # Strategy 3: Numeric part match
                lambda col_series: col_series[col_series.str.contains(numeric_part, na=False)]
                if numeric_part and len(numeric_part) >= 5 else pd.Series(),

                # Strategy 4: Match last part of ID (for claims like 510000/11/2024/356)
                lambda col_series: col_series[col_series.str.contains(claim_last_part, na=False)]
                if claim_last_part else pd.Series(),

                # Strategy 5: Match first part of ID (for claims starting with specific numbers)
                lambda col_series: col_series[col_series.str.contains(claim_first_part, na=False)]
                if claim_first_part and len(claim_first_part) >= 5 else pd.Series(),

                # Strategy 6: Match invoice number part
                lambda col_series: col_series[col_series.str.contains(invoice_number, na=False)]
                if invoice_number and len(invoice_number) >= 4 else pd.Series(),

                # Strategy 7: Match invoice prefix (like 2425)
                lambda col_series: col_series[col_series.str.contains(invoice_prefix, na=False)]
                if invoice_prefix and len(invoice_prefix) >= 4 else pd.Series(),

                # Strategy 8: For incomplete extractions, try looking for partial matches
                lambda col_series: col_series[col_series.str.contains(cleaned_id[:6], na=False)]
                if len(cleaned_id) >= 6 else pd.Series(),

                # Strategy 9: For identifiers that might be truncated/partial
                lambda col_series: col_series[col_series.str.contains(cleaned_id[-6:], na=False)]
                if len(cleaned_id) >= 6 else pd.Series(),
            ]

            # Try each column with each strategy
            for column in [self.invoice_column, self.client_ref_column]:
                if column is not None:
                    column_series = self.df[column].astype(str)

                    for strategy_idx, strategy in enumerate(strategies):
                        matches = strategy(column_series)
                        if not matches.empty:
                            index = matches.index[0]
                            logger.info(
                                f"Match found using strategy {strategy_idx + 1} in column {column} at row {index}")
                            return index

            # No match found using standard strategies
            logger.warning(f"No match found for ID: {unique_id}")

            # Add document-specific fallback strategies for various PDF types
            if pdf_text:
                # For OICL ADVICE PDF (Oriental Insurance)
                if "oriental insurance" in pdf_text.lower() or "hsbc" in pdf_text.lower():
                    # Look for specific patterns
                    if "510000/11/2024" in pdf_text or "510000/11/2025" in pdf_text:
                        # Try to find any row with a similar pattern
                        for column in [self.invoice_column, self.client_ref_column]:
                            if column is not None:
                                column_series = self.df[column].astype(str)
                                matches = column_series[column_series.str.contains("510000|2024|2025", na=False)]
                                if not matches.empty:
                                    index = matches.index[0]
                                    logger.info(f"OICL fallback match found in column {column} at row {index}")
                                    return index

                # For UNITED ADVICE PDF
                elif "united india insurance" in pdf_text.lower() or "axis bank" in pdf_text.lower():
                    # Look for patterns like 5004xxxxxx
                    if "5004" in pdf_text:
                        for column in [self.invoice_column, self.client_ref_column]:
                            if column is not None:
                                column_series = self.df[column].astype(str)
                                matches = column_series[column_series.str.contains("5004|2502", na=False)]
                                if not matches.empty:
                                    index = matches.index[0]
                                    logger.info(f"United fallback match found in column {column} at row {index}")
                                    return index

                # For HDFC ERGO PDF
                elif "hdfc ergo" in pdf_text.lower():
                    # Look for claim number pattern C299924024364-1
                    claim_pattern = r'C\d+\-\d+'
                    claim_match = re.search(claim_pattern, pdf_text)
                    if claim_match:
                        claim_ref = claim_match.group(0)
                        # Try to find a row with this pattern
                        for column in [self.invoice_column, self.client_ref_column]:
                            if column is not None:
                                column_series = self.df[column].astype(str)
                                matches = column_series[column_series.str.contains(claim_ref, na=False)]
                                if not matches.empty:
                                    index = matches.index[0]
                                    logger.info(f"HDFC ERGO fallback match found in column {column} at row {index}")
                                    return index

                # For Tata AIG PDF
                elif "tata aig" in pdf_text.lower() or "payment details for" in pdf_text.lower():
                    # Look for invoice/claim number pattern
                    invoice_pattern = r'2425\s*\-\s*\d{5}'
                    invoice_match = re.search(invoice_pattern, pdf_text)
                    if invoice_match:
                        invoice_ref = invoice_match.group(0).replace(' ', '')
                        # Try to find a row with this invoice number
                        for column in [self.invoice_column, self.client_ref_column]:
                            if column is not None:
                                column_series = self.df[column].astype(str)
                                matches = column_series[column_series.str.contains(invoice_ref, na=False)]
                                if not matches.empty:
                                    index = matches.index[0]
                                    logger.info(f"Tata AIG fallback match found in column {column} at row {index}")
                                    return index

                # For CERA Sanitaryware PDF
                elif "cera sanitaryware" in pdf_text.lower():
                    # Typically contain multiple invoice entries, look for pattern 2425-xxxxx
                    invoice_pattern = r'2425\-\d{5}'
                    invoice_matches = re.findall(invoice_pattern, pdf_text)
                    if invoice_matches:
                        # Try matching each invoice
                        for invoice_ref in invoice_matches:
                            for column in [self.invoice_column, self.client_ref_column]:
                                if column is not None:
                                    column_series = self.df[column].astype(str)
                                    matches = column_series[column_series.str.contains(invoice_ref, na=False)]
                                    if not matches.empty:
                                        index = matches.index[0]
                                        logger.info(f"CERA fallback match found in column {column} at row {index}")
                                        return index

                # For Liberty Insurance PDF
                elif "liberty" in pdf_text.lower() or "liber" in pdf_text.lower():
                    # Try to find the reference number
                    ref_pattern = r'your\s+reference\s*:\s*(\d+)'
                    ref_match = re.search(ref_pattern, pdf_text, re.IGNORECASE)
                    if ref_match:
                        ref_number = ref_match.group(1)
                        # Try to find a row with this reference
                        for column in [self.invoice_column, self.client_ref_column]:
                            if column is not None:
                                column_series = self.df[column].astype(str)
                                matches = column_series[column_series.str.contains(ref_number, na=False)]
                                if not matches.empty:
                                    index = matches.index[0]
                                    logger.info(f"Liberty fallback match found in column {column} at row {index}")
                                    return index

                # For National Insurance/IFFCO Tokio PDF (Hindi bilingual)
                elif "national insurance" in pdf_text.lower() or "iffco" in pdf_text.lower():
                    # Try to extract using sub claim number
                    sub_claim_pattern = r'sub\s+claim\s+no.*?(\d+\-\d+)'
                    sub_claim_match = re.search(sub_claim_pattern, pdf_text, re.IGNORECASE | re.DOTALL)
                    if sub_claim_match:
                        sub_claim = sub_claim_match.group(1)
                        # Try to find a row with this sub claim
                        for column in [self.invoice_column, self.client_ref_column]:
                            if column is not None:
                                column_series = self.df[column].astype(str)
                                matches = column_series[column_series.str.contains(sub_claim, na=False)]
                                if not matches.empty:
                                    index = matches.index[0]
                                    logger.info(
                                        f"National/IFFCO fallback match found in column {column} at row {index}")
                                    return index

                # For Reliance/Future Generali/Universal Sompo PDF
                elif any(insurer in pdf_text.lower() for insurer in ["reliance", "future", "sompo"]):
                    # Look for invoice reference like 2425-xxxxx
                    invoice_pattern = r'2425\-\d{5}'
                    invoice_match = re.search(invoice_pattern, pdf_text)
                    if invoice_match:
                        invoice_ref = invoice_match.group(0)
                        # Try to find a row with this invoice
                        for column in [self.invoice_column, self.client_ref_column]:
                            if column is not None:
                                column_series = self.df[column].astype(str)
                                matches = column_series[column_series.str.contains(invoice_ref, na=False)]
                                if not matches.empty:
                                    index = matches.index[0]
                                    logger.info(f"Insurer fallback match found in column {column} at row {index}")
                                    return index

                # For New India Payment Advice (email format)
                elif "new india" in pdf_text.lower() or "payment of rs" in pdf_text.lower():
                    # Look for invoice/claim number pattern
                    invoice_pattern = r'2425\-\d{5}'
                    invoice_match = re.search(invoice_pattern, pdf_text)
                    if invoice_match:
                        invoice_ref = invoice_match.group(0)
                        # Try to find a row with this invoice
                        for column in [self.invoice_column, self.client_ref_column]:
                            if column is not None:
                                column_series = self.df[column].astype(str)
                                matches = column_series[column_series.str.contains(invoice_ref, na=False)]
                                if not matches.empty:
                                    index = matches.index[0]
                                    logger.info(f"New India fallback match found in column {column} at row {index}")
                                    return index

            # Final fallback: Try to match using invoice patterns commonly found in PDF files
            invoice_pattern = r'2425-\d{5}'
            invoice_matches = re.findall(invoice_pattern, pdf_text) if pdf_text else []

            if invoice_matches:
                # Try each extracted invoice number
                for invoice in invoice_matches:
                    for column in [self.invoice_column, self.client_ref_column]:
                        if column is not None:
                            column_series = self.df[column].astype(str)
                            matches = column_series[column_series.str.contains(invoice, na=False)]
                            if not matches.empty:
                                index = matches.index[0]
                                logger.info(f"Invoice pattern fallback match found in column {column} at row {index}")
                                return index

            return None

        except Exception as e:
            logger.error(f"Error finding row by ID: {str(e)}")
            return None

    def update_row_with_data(self, row_index, data_points, pdf_text=None, detected_provider=None):
        """
        Update a row in the Excel file with extracted data points.
        Enhanced with improved validation and handling of edge cases for various formats.

        Args:
            row_index (int): Index of the row to update
            data_points (dict): Dictionary of field names and values
            pdf_text (str, optional): The full PDF text for context if needed

        Returns:
            tuple: (success, updated_fields, failed_fields)
        """
        if row_index is None or not data_points:
            return False, [], list(data_points.keys())

        updated_fields = []
        failed_fields = []

        try:
            # Get the Excel row number (add 2 to account for 0-indexing and header row)
            excel_row = row_index + 2

            # Validate and fix data before updating
            # For receipt amount, ensure it's a proper number and not truncated
            if 'receipt_amount' in data_points and data_points['receipt_amount']:
                try:
                    amount_str = data_points['receipt_amount']
                    # Clean the amount value
                    amount_str = re.sub(r'[^0-9.]', '', amount_str)

                    # Check if the amount seems too small for a payment
                    if len(amount_str) <= 3 and '.' not in amount_str:
                        logger.warning(f"Receipt amount {amount_str} seems too small, might be truncated")

                        # For OICL ADVICE PDF, fix the truncated value
                        if pdf_text and ("oriental insurance" in pdf_text.lower() or "hsbc" in pdf_text.lower()):
                            # Look for the full amount in the text
                            amount_patterns = [
                                r'Remittance\s+amount\s*:?\s*(?:INR)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                                r'Amount.*?(\d{6}(?:\.\d{2})?)',
                                r'Amount\s*(\d{6})',
                            ]

                            for pattern in amount_patterns:
                                amount_match = re.search(pattern, pdf_text, re.IGNORECASE)
                                if amount_match:
                                    amount_str = amount_match.group(1).replace(',', '')
                                    logger.info(
                                        f"Fixed truncated amount: {data_points['receipt_amount']} -> {amount_str}")
                                    data_points['receipt_amount'] = amount_str
                                    break

                        # For other payment advice PDFs
                        elif pdf_text:
                            # Generic patterns to find amounts
                            amount_patterns = [
                                r'amount\s*:?\s*(?:Rs\.?|INR)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)',
                                r'remittance\s+amount\s*:?\s*(?:Rs\.?|INR)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)',
                                r'payment\s+amount\s*:?\s*(?:Rs\.?|INR)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)',
                                r'net\s+amount\s*:?\s*(?:Rs\.?|INR)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)',
                                r'rupees\s+.*?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)',
                            ]

                            for pattern in amount_patterns:
                                amount_match = re.search(pattern, pdf_text, re.IGNORECASE)
                                if amount_match:
                                    amount_str = amount_match.group(1).replace(',', '')
                                    logger.info(
                                        f"Fixed truncated amount using generic pattern: {data_points['receipt_amount']} -> {amount_str}")
                                    data_points['receipt_amount'] = amount_str
                                    break

                    # Convert to float to ensure it's a valid number
                    amount = float(amount_str)
                    # Format with 2 decimal places
                    data_points['receipt_amount'] = str(amount)
                except ValueError:
                    logger.error(f"Invalid amount format: {data_points['receipt_amount']}")
                    failed_fields.append('receipt_amount')
                    data_points.pop('receipt_amount', None)

            # For receipt date, ensure it's in a proper date format
            if 'receipt_date' in data_points and data_points['receipt_date']:
                try:
                    date_str = data_points['receipt_date']
                    # Handle different date formats
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
                        logger.warning(f"Could not parse date: {date_str}")
                        failed_fields.append('receipt_date')
                        data_points.pop('receipt_date', None)
                except Exception as e:
                    logger.error(f"Error formatting date {data_points['receipt_date']}: {str(e)}")
                    failed_fields.append('receipt_date')
                    data_points.pop('receipt_date', None)

            # Process and compute TDS if needed
            # Changed following line from this to below - if 'receipt_amount' in data_points and data_points['receipt_amount'] and 'tds' not in data_points:
            if 'receipt_amount' in data_points and data_points['receipt_amount']:
                try:
                    # Clean the amount value and convert to float
                    amount_str = data_points['receipt_amount']
                    amount = float(re.sub(r'[^\d.]', '', amount_str))

                    # Compute TDS
                    tds_value, is_computed = self.compute_tds(amount, pdf_text if pdf_text else "", detected_provider)

                    # Add to data_points
                    data_points['tds'] = str(tds_value)
                    data_points['tds_computed'] = 'Yes'
                    logger.info(f"TDS computed: {tds_value}")
                except Exception as e:
                    logger.error(f"Error computing TDS: {str(e)}")
                    failed_fields.append('tds')
            elif 'tds' in data_points and data_points['tds']:
                # If TDS is already in data_points, mark it as not computed
                data_points['tds_computed'] = 'No'
                logger.info(f"Using extracted TDS: {data_points['tds']}")

            # Track which fields were successfully updated
            for field, mapped_column in self.header_mapping.items():
                if field in data_points and data_points[field]:
                    value = data_points[field]
                    column_index = self.df.columns.get_loc(mapped_column) + 1  # 1-indexed for openpyxl

                    # Update the cell value
                    cell = self.ws.cell(row=excel_row, column=column_index)
                    original_value = cell.value
                    cell.value = value

                    # Highlight the updated cell
                    cell.fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")

                    # Log the update
                    logger.info(f"Updated row {row_index}, column '{mapped_column}': '{original_value}' -> '{value}'")
                    updated_fields.append(field)
                else:
                    if field not in data_points:
                        logger.warning(f"Field '{field}' not found in extracted data")
                    else:
                        logger.warning(f"No value provided for field '{field}'")

                    # Don't count TDS Computed as a failed field if it wasn't provided
                    if field != 'tds_computed':
                        failed_fields.append(field)

            # Special handling for TDS Computed field
            tds_computed_column = self.header_mapping.get('tds_computed')
            if tds_computed_column and 'tds' in data_points:
                tds_computed_value = data_points.get('tds_computed', 'No')  # Default to No if not provided
                tds_computed_index = self.df.columns.get_loc(tds_computed_column) + 1
                tds_computed_cell = self.ws.cell(row=excel_row, column=tds_computed_index)
                tds_computed_cell.value = tds_computed_value
                tds_computed_cell.fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
                logger.info(f"Updated TDS Computed status: {tds_computed_value}")

                if 'tds_computed' not in updated_fields:
                    updated_fields.append('tds_computed')

            # Add timestamp to indicate when the record was updated
            last_column = len(self.df.columns) + 1
            timestamp_cell = self.ws.cell(row=excel_row, column=last_column)
            timestamp_cell.value = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Success if at least one of receipt_amount, receipt_date, or tds was updated
            core_fields = ['receipt_amount', 'receipt_date', 'tds']
            core_fields_updated = any(field in updated_fields for field in core_fields)

            if core_fields_updated:
                # Save the updated workbook
                backup_path = self.excel_path.replace('.xlsx',
                                                      f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
                self.wb.save(backup_path)  # Save a backup first
                self.wb.save(self.excel_path)  # Then save the actual file
                logger.info(f"Saved updated Excel file and backup: {backup_path}")
                return True, updated_fields, failed_fields
            else:
                logger.warning(f"No core fields (receipt_amount, receipt_date, tds) were updated for row {row_index}")
                return False, updated_fields, failed_fields

        except Exception as e:
            logger.error(f"Error updating row {row_index}: {str(e)}")
            return False, [], list(data_points.keys())

    def process_multi_row_data(self, table_data, pdf_text, global_data_points=None):
        """
        Process data from a table in the PDF where each row might correspond to a different Excel row.

        Args:
            table_data (list): List of dictionaries, each containing data for one row
            pdf_text (str): The full PDF text for context and TDS computation
            global_data_points (dict, optional): Global data points extracted from the PDF

        Returns:
            dict: Processing results
        """
        results = {
            'processed': 0,
            'unprocessed': 0,
            'total': len(table_data)
        }

        # Use provided global data points or an empty dict
        global_data = global_data_points or {}

        for row_data in table_data:
            # The row_data should contain at least one identifier (invoice_no or client_ref)
            # and at least one value to update (receipt_amount, receipt_date, or tds)

            # Try to find the corresponding row in Excel
            unique_id = row_data.get('unique_id', None)
            if not unique_id:
                # Try to find a suitable unique ID from other fields
                if 'invoice_no' in row_data:
                    unique_id = row_data['invoice_no']
                elif 'client_ref' in row_data:
                    unique_id = row_data['client_ref']

            if not unique_id:
                logger.warning(f"No unique identifier found for row data: {row_data}")
                results['unprocessed'] += 1
                continue

            # Copy missing fields from global data points to row_data
            # This is the critical part - adding fields like receipt_date from the global extraction
            for key, value in global_data.items():
                if key not in row_data and key != 'unique_id':
                    row_data[key] = value
                    logger.debug(f"Added global field '{key}': {value} to row data for {unique_id}")

            # Find the row in Excel
            row_index = self.find_row_by_identifiers(unique_id, row_data, pdf_text)

            if row_index is None:
                logger.warning(f"No matching Excel row found for ID: {unique_id}")
                results['unprocessed'] += 1
                continue

            # Update the row with data
            success, updated_fields, failed_fields = self.update_row_with_data(row_index, row_data, pdf_text)

            if success:
                results['processed'] += 1
                logger.info(f"Successfully updated row for ID: {unique_id}")
            else:
                results['unprocessed'] += 1
                logger.warning(f"Failed to update row for ID: {unique_id}")

        return results

    def save_excel(self):
        """Save the Excel workbook."""
        try:
            backup_path = self.excel_path.replace('.xlsx',
                                                  f'_backup_final_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
            self.wb.save(backup_path)
            self.wb.save(self.excel_path)
            logger.info(f"Excel file saved: {self.excel_path} with backup: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving Excel file: {str(e)}")
            return False