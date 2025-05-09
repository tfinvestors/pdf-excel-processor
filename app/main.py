import os
import shutil
from datetime import datetime
import logging
import traceback
import json
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from config import Config

from app.pdf_processor import PDFProcessor
from app.excel_handler import ExcelHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(stream=sys.stdout),  # Use stdout with UTF-8 config
        logging.FileHandler("pdf_processing.log", encoding='utf-8')  # Add encoding here
    ]
)
logger = logging.getLogger("main")


def move_to_appropriate_folder(pdf_path, success, processed_dir, unprocessed_dir):
    """
    Move PDF to the appropriate folder based on processing success.

    Args:
        pdf_path (str): Path to the PDF file
        success (bool): Whether processing was successful
        processed_dir (str): Path to the processed directory
        unprocessed_dir (str): Path to the unprocessed directory

    Returns:
        str: Path to the destination where file was copied
    """
    pdf_file = os.path.basename(pdf_path)

    if success:
        # Only copy to processed folder
        destination = os.path.join(processed_dir, pdf_file)
        shutil.copy2(pdf_path, destination)
        return destination
    else:
        # Only copy to unprocessed folder
        destination = os.path.join(unprocessed_dir, pdf_file)
        shutil.copy2(pdf_path, destination)
        return destination

# Process_files function for better multi-PDF handling
def process_files(excel_path, pdf_folder, progress_callback=None, status_callback=None, debug_mode=False):
    """
    Process PDF files and update Excel with extracted data.
    Enhanced with improved logging, debugging, and multi-document handling.

    Args:
        excel_path (str): Path to the Excel file
        pdf_folder (str): Path to the folder containing PDF files
        progress_callback (function): Function to update progress (current, total)
        status_callback (function): Function to update status messages
        debug_mode (bool): Enable debug mode for visualizations and extra logging

    Returns:
        dict: Processing results summary
    """
    # Initialize processors with API URL from config
    pdf_processor = PDFProcessor(
        use_ml=Config.USE_ML_MODEL,
        debug_mode=Config.DEBUG_MODE,
        poppler_path=Config.POPPLER_PATH,
        text_extraction_api_url=Config.PDF_EXTRACTION_API_URL
    )

    results = {
        'total': 0,
        'processed': 0,
        'unprocessed': 0,
        'files': {
            'processed': [],
            'unprocessed': []
        }
    }

    # Create output directories in user's Downloads folder
    # Get platform-neutral output directories
    processed_dir, unprocessed_dir = get_output_dirs()  # Add this function

    # Define the function if it doesn't exist
    def get_output_dirs():
        if 'STREAMLIT_SHARING' in os.environ:
            # In cloud environment, use directories in the app folder
            processed_dir = os.path.join('processed_pdf')
            unprocessed_dir = os.path.join('unprocessed_pdf')
        else:
            # In local environment, use Downloads directories
            processed_dir = os.path.join(os.path.expanduser("~"), "Downloads", "Processed PDF")
            unprocessed_dir = os.path.join(os.path.expanduser("~"), "Downloads", "Unprocessed PDF")

        # Ensure directories exist
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(unprocessed_dir, exist_ok=True)

        return processed_dir, unprocessed_dir

    # Call the function to get the directories
    processed_dir, unprocessed_dir = get_output_dirs()

    # Create debug directory if debug mode is enabled
    debug_dir = None
    if debug_mode:
        debug_dir = os.path.join(download_dir, "PDF_Debug")
        os.makedirs(debug_dir, exist_ok=True)
        if status_callback:
            status_callback(f"Debug mode enabled. Visualizations will be saved to {debug_dir}")

    for dir_path in [processed_dir, unprocessed_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Clear the output directories before processing
    for dir_path in [processed_dir, unprocessed_dir]:
        if os.path.exists(dir_path):
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    # Initialize processors
    try:
        # Check for poppler on the system - more paths to try
        poppler_paths = [
            "C:\\poppler-24.08.0\\Library\\bin",
            "C:\\poppler\\bin",
            "C:\\Program Files\\poppler\\bin",
            "C:\\Program Files (x86)\\poppler\\bin",
            "/usr/bin",
            "/usr/local/bin",
            "/opt/homebrew/bin"
        ]

        poppler_path = None
        for path in poppler_paths:
            if os.path.exists(path):
                if os.path.exists(os.path.join(path, "pdfinfo.exe")) or os.path.exists(os.path.join(path, "pdfinfo")):
                    poppler_path = path
                    break

        pdf_processor = PDFProcessor(
            use_ml=True,
            debug_mode=debug_mode,
            poppler_path=poppler_path
        )
        excel_handler = ExcelHandler(excel_path)

        # Get list of PDF files in the folder
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
        total_files = len(pdf_files)
        results['total'] = total_files

        if status_callback:
            status_callback(f"Found {total_files} PDF files to process")

        # Process each PDF file
        for idx, pdf_file in enumerate(pdf_files):
            pdf_path = os.path.join(pdf_folder, pdf_file)

            # Update progress
            if progress_callback:
                progress_callback(idx + 1, total_files)

            if status_callback:
                status_callback(f"Processing file {idx + 1}/{total_files}: {pdf_file}")

            try:
                # Extract text from PDF
                extracted_text = pdf_processor.extract_text(pdf_path)

                if not extracted_text:
                    if status_callback:
                        status_callback(f"❌ Failed to extract text from {pdf_file}")

                    # Move to unprocessed folder
                    move_to_appropriate_folder(pdf_path, False, processed_dir, unprocessed_dir)
                    results['unprocessed'] += 1
                    results['files']['unprocessed'].append(pdf_file)
                    continue

                # National Insurance - After extracting text, before calling extract_data_points
                if "Sub Claim No" in extracted_text and "National Insurance" in extracted_text:
                    # Direct extraction of Sub Claim No for National Insurance
                    claim_match = re.search(r'(\d+-\d+)\s+[A-Z\s]+\s+\d{2}-\d{2}-\d{4}', extracted_text)
                    if claim_match:
                        direct_claim_id = claim_match.group(1)
                        logger.info(f"Direct extraction found claim ID: {direct_claim_id}")

                        # Use this ID with extract_data_points
                        extraction_result = pdf_processor.extract_data_points(extracted_text)

                        # Override the unique_id if needed
                        if isinstance(extraction_result, tuple) and len(extraction_result) >= 3:
                            unique_id, data_points, table_data = extraction_result[:3]
                            if not unique_id or "बीिमत" in unique_id:
                                logger.info(f"Overriding extracted ID with direct match: {direct_claim_id}")
                                unique_id = direct_claim_id

                # Special handling for ICICI Lombard table documents
                if "CLAIM_REF_NO" in extracted_text and "LAE Invoice No" in extracted_text and "TRF AMOUNT" in extracted_text:
                    if status_callback:
                        status_callback(f"📊 Detected ICICI Lombard claim table in {pdf_file}")

                    # Extract table data directly from the structured format
                    table_data = []

                    # Use the regex pattern to extract rows
                    rows = re.findall(
                        r'((?:ENG|MAR|FIR|MSC|LIA)\d+)\s+(2425-\d{5})\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+([\d,\.]+)\s+([\d,\.]+)\s+([\d,\.]+)\s+(\d{2}-\d{2}-\d{4})',
                        extracted_text)

                    for row in rows:
                        claim_ref, invoice_no, bill_date, invoice_amt, tds, trf_amount, receipt_date = row

                        # Add to table data
                        table_data.append({
                            'unique_id': claim_ref.strip(),
                            'invoice_no': invoice_no.strip(),
                            'receipt_date': receipt_date.strip(),
                            'receipt_amount': trf_amount.strip().replace(',', ''),
                            'tds': tds.strip().replace(',', ''),
                            'tds_computed': 'No'  # TDS already in document
                        })

                    if table_data:
                        if status_callback:
                            status_callback(f"📋 Found {len(table_data)} claim entries in table")

                        # Process the table data
                        table_results = excel_handler.process_multi_row_data(table_data, extracted_text, {})

                        if table_results['processed'] > 0:
                            if status_callback:
                                status_callback(
                                    f"✅ Processed {table_results['processed']} of {table_results['total']} rows from table")

                            # Move to processed folder if at least one row was successful
                            move_to_appropriate_folder(pdf_path, True, processed_dir, unprocessed_dir)
                            results['processed'] += 1
                            results['files']['processed'].append(pdf_file)
                            continue

                # Extract data with proper error handling
                try:
                    # Extract data points with better error handling
                    extraction_result = pdf_processor.extract_data_points(extracted_text)

                    # Handle different return types to ensure compatibility
                    if isinstance(extraction_result, tuple):
                        if len(extraction_result) == 4:
                            unique_id, data_points, table_data, detected_provider = extraction_result
                        elif len(extraction_result) == 3:
                            unique_id, data_points, table_data = extraction_result
                            detected_provider = None
                        else:
                            logger.error(f"Unexpected return format from extract_data_points: {extraction_result}")
                            raise ValueError("Invalid return format from data extraction")
                    else:
                        logger.error(f"Unexpected return type from extract_data_points: {type(extraction_result)}")
                        raise ValueError("Invalid return type from data extraction")

                except Exception as e:
                    logger.error(f"Error extracting data from {pdf_file}: {str(e)}")
                    logger.error(traceback.format_exc())

                    # Move to unprocessed folder
                    move_to_appropriate_folder(pdf_path, False, processed_dir, unprocessed_dir)
                    results['unprocessed'] += 1
                    results['files']['unprocessed'].append(pdf_file)
                    continue

                # Log first 500 chars of extracted text for debugging
                logger.debug(f"Extracted text sample: {extracted_text[:500000000000]}")

                # If in debug mode, save the full extracted text
                if debug_mode and debug_dir:
                    text_path = os.path.join(debug_dir, f"{pdf_file}_extracted_text.txt")
                    with open(text_path, 'w', encoding='utf-8') as f:
                        f.write(extracted_text)
                    logger.debug(f"Saved full extracted text to {text_path}")

                # Extract data points and potential table data
                unique_id, data_points, table_data, detected_provider = pdf_processor.extract_data_points(extracted_text)

                # Log extracted data
                logger.info(f"Extracted unique ID: {unique_id}")
                logger.info(f"Extracted data points: {data_points}")
                logger.info(f"Found {len(table_data)} table rows")

                # If in debug mode, save the extracted data as JSON for analysis
                if debug_mode and debug_dir:
                    data_path = os.path.join(debug_dir, f"{pdf_file}_extracted_data.json")
                    with open(data_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            "unique_id": unique_id,
                            "data_points": data_points,
                            "table_data": table_data
                        }, f, indent=2)
                    logger.debug(f"Saved extracted data to {data_path}")

                # Case 1: The PDF contains a table with multiple rows of data
                if table_data and len(table_data) > 0:
                    if status_callback:
                        status_callback(f"📊 Found table with {len(table_data)} rows in {pdf_file}")

                    # For Future Generali specifically, find the right claim number
                    if detected_provider == "future_generali":
                        # Look for the specific format with F0036935 pattern
                        claim_rows = [row for row in table_data if 'unique_id' in row and
                                      re.match(r'F\d{7}', row['unique_id']) and
                                      'Claims Payment' in row.get('payment_type', '')]

                        if claim_rows:
                            # Use this row's unique_id for the whole document
                            unique_id = claim_rows[0]['unique_id']
                            logger.info(f"Using Future Generali claim ID from table: {unique_id}")

                            # Set the correct receipt_amount in data_points
                            receipt_amount = claim_rows[0]['receipt_amount'].lstrip('0')
                            if receipt_amount.startswith('.'):
                                receipt_amount = '0' + receipt_amount
                            data_points['receipt_amount'] = receipt_amount
                            logger.info(f"Using receipt amount from Claims Payment row: {receipt_amount}")

                            # Update all table rows to use this unique_id
                            for row in table_data:
                                row['unique_id'] = unique_id
                                # Set the correct receipt_amount in all table rows too
                                row['receipt_amount'] = receipt_amount

                            # For each row in table_data, update the unique_id to this claim ID
                            # This ensures all rows get matched to the same Excel row
                            for row in table_data:
                                row['unique_id'] = unique_id

                    # Process the table data - pass the extracted data_points
                    table_results = excel_handler.process_multi_row_data(table_data, extracted_text, data_points)



                    if table_results['processed'] > 0:
                        if status_callback:
                            status_callback(
                                f"✅ Successfully processed {table_results['processed']} of {table_results['total']} rows from {pdf_file}")

                        # Move to processed folder if at least one row processed successfully
                        move_to_appropriate_folder(pdf_path, True, processed_dir, unprocessed_dir)
                        results['processed'] += 1
                        results['files']['processed'].append(pdf_file)
                    else:
                        if status_callback:
                            status_callback(f"❌ Failed to process any rows from {pdf_file}")

                        # Move to unprocessed folder
                        move_to_appropriate_folder(pdf_path, False, processed_dir, unprocessed_dir)
                        results['unprocessed'] += 1
                        results['files']['unprocessed'].append(pdf_file)

                    continue  # Skip the single row processing since we've processed the table



                # Case 2: The PDF contains a single row of data (standard case)
                if not unique_id:
                    if status_callback:
                        status_callback(f"❌ Failed to extract ID from {pdf_file}")

                    # Move to unprocessed folder
                    move_to_appropriate_folder(pdf_path, False, processed_dir, unprocessed_dir)
                    results['unprocessed'] += 1
                    results['files']['unprocessed'].append(pdf_file)
                    continue

                # Find corresponding row in Excel
                row_index = excel_handler.find_row_by_identifiers(unique_id, data_points, extracted_text)

                if row_index is None:
                    if status_callback:
                        status_callback(f"❌ Could not find matching row for ID: {unique_id} in {pdf_file}")

                    # Move to unprocessed folder
                    move_to_appropriate_folder(pdf_path, False, processed_dir, unprocessed_dir)
                    results['unprocessed'] += 1
                    results['files']['unprocessed'].append(pdf_file)
                    continue

                # First log the data that will be passed to the Excel handler
                logger.debug(f"Data points to be passed to Excel handler: {data_points}")

                # Specifically check the receipt_amount format
                if 'receipt_amount' in data_points:
                    try:
                        logger.info(
                            f"IMPORTANT DEBUG - Initial receipt_amount in data_points in main.py code file: {data_points['receipt_amount']}")
                        test_float = float(data_points['receipt_amount'])
                        # If valid, make sure it's stored as a string with the correct format
                        data_points['receipt_amount'] = str(test_float)
                    except ValueError:
                        # If we can't convert it, log this severe issue
                        logger.error(
                            f"Invalid receipt_amount format before Excel update: {data_points['receipt_amount']}")
                        # Apply one final emergency fix
                        if '.' in data_points['receipt_amount'] and data_points['receipt_amount'].count('.') > 1:
                            # Fix European format
                            fixed_amount = re.sub(r'\.(?=\d{3})', '', data_points['receipt_amount'])
                            logger.info(
                                f"Emergency fix on receipt_amount: {data_points['receipt_amount']} -> {fixed_amount}")
                            data_points['receipt_amount'] = fixed_amount

                logger.info(
                    f"IMPORTANT DEBUG - Final receipt_amount in data_points in main.py code file just before calling update_row_with_data function: {data_points['receipt_amount']}")

                # Update Excel row with extracted data
                success, updated_fields, failed_fields = excel_handler.update_row_with_data(row_index, data_points,
                                                                                            extracted_text, detected_provider)

                if success:
                    if status_callback:
                        status_callback(
                            f"✅ Successfully updated row for ID: {unique_id} with fields: {', '.join(updated_fields)}")

                    # Move to processed folder
                    move_to_appropriate_folder(pdf_path, True, processed_dir, unprocessed_dir)
                    results['processed'] += 1
                    results['files']['processed'].append(pdf_file)
                else:
                    if len(updated_fields) > 0:
                        if status_callback:
                            status_callback(f"⚠️ Partially updated row for ID: {unique_id}")
                            status_callback(f"   Updated fields: {', '.join(updated_fields)}")
                            status_callback(f"   Failed fields: {', '.join(failed_fields)}")
                    else:
                        if status_callback:
                            status_callback(f"❌ Failed to update row for ID: {unique_id}")

                    # Move to unprocessed folder since not all fields were updated
                    move_to_appropriate_folder(pdf_path, False, processed_dir, unprocessed_dir)
                    results['unprocessed'] += 1
                    results['files']['unprocessed'].append(pdf_file)

            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
                logger.error(traceback.format_exc())

                if status_callback:
                    status_callback(f"❌ Error processing {pdf_file}: {str(e)}")

                # Move to unprocessed folder
                move_to_appropriate_folder(pdf_path, False, processed_dir, unprocessed_dir)
                results['unprocessed'] += 1
                results['files']['unprocessed'].append(pdf_file)

        # Save final Excel file
        excel_handler.save_excel()

        if status_callback:
            status_callback(
                f"Processing complete. {results['processed']} files processed successfully, {results['unprocessed']} files failed.")

        return results

    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}")
        logger.error(traceback.format_exc())

        if status_callback:
            status_callback(f"❌ Critical error: {str(e)}")

        return results


if __name__ == "__main__":
    # Test the processing functionality
    excel_path = "path/to/your/excel/file.xlsx"
    pdf_folder = "path/to/your/pdf/folder"

    results = process_files(excel_path, pdf_folder)
    print(f"Processed: {results['processed']}/{results['total']}")