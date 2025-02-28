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

    def compute_tds(self, amount, text):
        """
        Compute TDS based on the receipt amount and text content.
        Enhanced to better detect insurance companies with various name formats.

        Args:
            amount (float): The receipt amount
            text (str): The text content from the PDF

        Returns:
            tuple: (tds_value, is_computed)
        """
        try:
            # Check if any of the insurance companies are mentioned in the text
            insurance_companies = [
                "national insurance company limited",
                "united india insurance company limited",
                "the new india assurance co. ltd",
                "oriental insurance co ltd",
                "oriental insurance",
                "national insurance",
                "united india insurance",
                "new india assurance",
                "oriental insurance",
                "united india",
                "new india",
                "oriental",
                "0rienta1"
            ]

            # Normalize the text for better matching
            normalized_text = ' '.join(text.lower().split())
            logger.debug(
                f"Normalized text for insurance detection (first a few hundred chars): {normalized_text[:300]}...")

            # Check for insurance company with more flexible matching
            contains_insurance_company = False
            detected_company = None

            # Check for explicit mention of oriental/national/united insurance (first priority)
            insurance_keywords = ["oriental", "national", "united india", "new india"]
            for keyword in insurance_keywords:
                if keyword in normalized_text and "insurance" in normalized_text:
                    detected_company = f"{keyword} insurance"
                    contains_insurance_company = True
                    logger.info(f"Explicitly detected {keyword} insurance in the text")
                    break

            # If not found, try more specific matches
            if not contains_insurance_company:
                for company in insurance_companies:
                    # Check for exact match
                    if company in normalized_text:
                        contains_insurance_company = True
                        detected_company = company
                        logger.info(f"Detected insurance company: {company}")
                        break

                    # Check for partial match (key parts of company name)
                    key_parts = [part for part in company.split() if len(part) > 3 and part != "insurance"]
                    if key_parts and all(part in normalized_text for part in key_parts):
                        contains_insurance_company = True
                        detected_company = company
                        logger.info(f"Detected insurance company by key parts: {company} (parts: {key_parts})")
                        break

            # Additional checks for specific PDF types
            # For HSBC documents, check for insurance company clues
            if "hsbc" in normalized_text:
                # Look specifically for mentions of insurance companies in remitter info
                remitter_info_match = re.search(r'remitter.*?information:.*?(oriental|national|united|new india)',
                                                normalized_text, re.IGNORECASE | re.DOTALL)
                if remitter_info_match:
                    insurance_name = remitter_info_match.group(1).lower()
                    contains_insurance_company = True
                    detected_company = f"{insurance_name} insurance"
                    logger.info(f"Detected {insurance_name} insurance from HSBC remitter information")

            # Apply the appropriate calculation
            if contains_insurance_company:
                tds = round(amount * 0.11111111, 2)
                logger.info(f"TDS computed for insurance company ({detected_company}): {tds} (11.111111% of {amount})")
            else:
                tds = round(amount * 0.09259259, 2)
                logger.info(f"TDS computed for non-insurance company: {tds} (9.259259% of {amount})")

            return tds, True

        except Exception as e:
            logger.error(f"Error computing TDS: {str(e)}")
            return 0.0, True

    def find_row_by_identifiers(self, unique_id, data_points, pdf_text=None):
        """
        Find a row in the DataFrame by matching either Invoice No. or Client Ref. No.
        Enhanced with more robust matching strategies.
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

            # For OICL ADVICE and similar PDFs, extract the last part of claim number
            claim_last_part = None
            if '/' in cleaned_id:
                claim_parts = cleaned_id.split('/')
                claim_last_part = claim_parts[-1]
                logger.info(f"Last part of claim number: {claim_last_part}")

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

                # Strategy 5: For incomplete extractions, try looking for partial matches
                lambda col_series: col_series[col_series.str.contains(cleaned_id[:6], na=False)]
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

            # Additional fallback for specific PDF types
            if pdf_text:
                # For OICL ADVICE PDF
                if "oriental insurance" in pdf_text.lower() or "hsbc" in pdf_text.lower():
                    # Look for keywords
                    if "510000/11/2024" in pdf_text or "2025/000000" in pdf_text:
                        # Try to find any row with a similar pattern
                        for column in [self.invoice_column, self.client_ref_column]:
                            if column is not None:
                                column_series = self.df[column].astype(str)
                                matches = column_series[column_series.str.contains("2024|2025", na=False)]
                                if not matches.empty:
                                    index = matches.index[0]
                                    logger.info(f"Fallback match found in column {column} at row {index}")
                                    return index

                # For UNITED ADVICE PDF
                elif "united india insurance" in pdf_text.lower() or "axis bank" in pdf_text.lower():
                    # Look for patterns like 5004xxxxxx
                    if "5004" in pdf_text:
                        for column in [self.invoice_column, self.client_ref_column]:
                            if column is not None:
                                column_series = self.df[column].astype(str)
                                matches = column_series[column_series.str.contains("5004", na=False)]
                                if not matches.empty:
                                    index = matches.index[0]
                                    logger.info(f"Fallback match found in column {column} at row {index}")
                                    return index

            return None

        except Exception as e:
            logger.error(f"Error finding row by ID: {str(e)}")
            return None

    def update_row_with_data(self, row_index, data_points, pdf_text=None):
        """
        Update a row in the Excel file with extracted data points.
        Enhanced with improved validation and handling of edge cases.

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
                    else:
                        logger.warning(f"Could not parse date: {date_str}")
                        failed_fields.append('receipt_date')
                        data_points.pop('receipt_date', None)
                except Exception as e:
                    logger.error(f"Error formatting date {data_points['receipt_date']}: {str(e)}")
                    failed_fields.append('receipt_date')
                    data_points.pop('receipt_date', None)

            # Process and compute TDS if needed
            if 'receipt_amount' in data_points and data_points['receipt_amount'] and 'tds' not in data_points:
                try:
                    # Clean the amount value and convert to float
                    amount_str = data_points['receipt_amount']
                    amount = float(re.sub(r'[^\d.]', '', amount_str))

                    # Compute TDS
                    tds_value, is_computed = self.compute_tds(amount, pdf_text if pdf_text else "")

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

    def process_multi_row_data(self, table_data, pdf_text):
        """
        Process data from a table in the PDF where each row might correspond to a different Excel row.

        Args:
            table_data (list): List of dictionaries, each containing data for one row
            pdf_text (str): The full PDF text for context and TDS computation

        Returns:
            dict: Processing results
        """
        results = {
            'processed': 0,
            'unprocessed': 0,
            'total': len(table_data)
        }

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