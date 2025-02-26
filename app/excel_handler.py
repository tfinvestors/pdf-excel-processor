import os
import pandas as pd
import openpyxl
import logging
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
        self.id_column = None
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

            # Try to identify the ID column automatically
            self.identify_id_column()

            # Create header mapping
            self.create_header_mapping()

        except Exception as e:
            logger.error(f"Error loading Excel file: {str(e)}")
            raise

    def identify_id_column(self):
        """Try to identify the column containing unique identifiers."""
        potential_id_columns = [
            'ID', 'Id', 'id', 'identifier', 'Identifier',
            'reference', 'Reference', 'ref', 'Ref',
            'doc_id', 'Doc_ID', 'Document ID', 'document id',
            'invoice', 'Invoice', 'Invoice Number', 'invoice number',
            'case', 'Case', 'Case Number', 'case number'
        ]

        # Check for exact matches in column names
        for col in potential_id_columns:
            if col in self.df.columns:
                self.id_column = col
                logger.info(f"Found ID column: {col}")
                return

        # If no exact match, check for partial matches
        for col in self.df.columns:
            for id_col in potential_id_columns:
                if id_col.lower() in col.lower():
                    self.id_column = col
                    logger.info(f"Found ID column (partial match): {col}")
                    return

        # If still not found, use the first column as a fallback
        self.id_column = self.df.columns[0]
        logger.warning(f"No obvious ID column found. Using first column: {self.id_column}")

    def create_header_mapping(self):
        """Create a mapping between common field names and actual column names."""
        # Define common field names and possible variants
        field_mappings = {
            'date': ['date', 'dt', 'day', 'invoice date', 'transaction date', 'document date'],
            'amount': ['amount', 'amt', 'value', 'price', 'cost', 'total', 'sum', 'invoice amount'],
            'name': ['name', 'customer name', 'client name', 'full name', 'person', 'contact name'],
            'address': ['address', 'addr', 'location', 'customer address', 'client address'],
            'contact': ['contact', 'phone', 'telephone', 'tel', 'mobile', 'cell', 'contact number'],
            'email': ['email', 'mail', 'e-mail', 'email address', 'contact email']
            # Add more mappings as needed
        }

        # Create mapping between standardized field names and actual column names
        for field, variants in field_mappings.items():
            for col in self.df.columns:
                if any(variant.lower() in col.lower() for variant in variants):
                    self.header_mapping[field] = col
                    break

        logger.info(f"Created header mapping: {self.header_mapping}")

    def find_row_by_id(self, unique_id):
        """
        Find a row in the DataFrame by its unique identifier.

        Args:
            unique_id (str): The unique identifier to search for

        Returns:
            int or None: Index of the matching row or None if not found
        """
        if not unique_id or self.id_column is None:
            return None

        try:
            # Convert ID column and search value to string for matching
            id_series = self.df[self.id_column].astype(str)
            unique_id_str = str(unique_id)

            # Try exact match first
            matches = id_series[id_series == unique_id_str]

            if not matches.empty:
                index = matches.index[0]
                logger.info(f"Found exact match for ID {unique_id} at row {index}")
                return index

            # If no exact match, try case-insensitive match
            matches = id_series[id_series.str.lower() == unique_id_str.lower()]

            if not matches.empty:
                index = matches.index[0]
                logger.info(f"Found case-insensitive match for ID {unique_id} at row {index}")
                return index

            # If still no match, try partial match (if ID is long enough)
            if len(unique_id_str) >= 5:
                # Check if the ID column contains the search value
                partial_matches = id_series[id_series.str.contains(unique_id_str, case=False, na=False)]

                if not partial_matches.empty:
                    index = partial_matches.index[0]
                    logger.warning(f"Found partial match for ID {unique_id} at row {index}")
                    return index

            # No match found
            logger.warning(f"No match found for ID {unique_id}")
            return None

        except Exception as e:
            logger.error(f"Error finding row by ID: {str(e)}")
            return None

    def update_row_with_data(self, row_index, data_points):
        """
        Update a row in the Excel file with extracted data points.

        Args:
            row_index (int): Index of the row to update
            data_points (dict): Dictionary of field names and values

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

            # Track which fields were successfully updated
            for field, value in data_points.items():
                if field in self.header_mapping and value:
                    column_name = self.header_mapping[field]
                    column_index = self.df.columns.get_loc(column_name) + 1  # 1-indexed for openpyxl

                    # Update the cell value
                    cell = self.ws.cell(row=excel_row, column=column_index)
                    original_value = cell.value
                    cell.value = value

                    # Highlight the updated cell
                    cell.fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")

                    # Log the update
                    logger.info(f"Updated row {row_index}, column '{column_name}': '{original_value}' -> '{value}'")
                    updated_fields.append(field)
                else:
                    if field not in self.header_mapping:
                        logger.warning(f"Field '{field}' not found in header mapping")
                    else:
                        logger.warning(f"No value provided for field '{field}'")
                    failed_fields.append(field)

            # Add timestamp to indicate when the record was updated
            if 'last_updated' in self.df.columns:
                update_col = self.df.columns.get_loc('last_updated') + 1
                self.ws.cell(row=excel_row, column=update_col).value = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Save the updated workbook
            success = len(updated_fields) > 0
            if success:
                backup_path = self.excel_path.replace('.xlsx',
                                                      f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
                self.wb.save(backup_path)  # Save a backup first
                self.wb.save(self.excel_path)  # Then save the actual file
                logger.info(f"Saved updated Excel file and backup: {backup_path}")

            return success, updated_fields, failed_fields

        except Exception as e:
            logger.error(f"Error updating row {row_index}: {str(e)}")
            return False, [], list(data_points.keys())

    def save_excel(self):
        """Save the Excel workbook."""
        try:
            self.wb.save(self.excel_path)
            logger.info(f"Excel file saved: {self.excel_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving Excel file: {str(e)}")
            return False