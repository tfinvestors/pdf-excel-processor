import os
import shutil
from datetime import datetime
import logging
import traceback
from app.pdf_processor import PDFProcessor
from app.excel_handler import ExcelHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pdf_excel_processing.log")
    ]
)
logger = logging.getLogger("main")


def process_files(excel_path, pdf_folder, progress_callback=None, status_callback=None):
    """
    Process PDF files and update Excel with extracted data.

    Args:
        excel_path (str): Path to the Excel file
        pdf_folder (str): Path to the folder containing PDF files
        progress_callback (function): Function to update progress (current, total)
        status_callback (function): Function to update status messages

    Returns:
        dict: Processing results summary
    """
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
    download_dir = os.path.join(os.path.expanduser("~"), "Downloads")
    processed_dir = os.path.join(download_dir, "Processed PDF")
    unprocessed_dir = os.path.join(download_dir, "Unprocessed PDF")

    for dir_path in [processed_dir, unprocessed_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Initialize processors
    try:
        pdf_processor = PDFProcessor(use_ml=True)
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
                # Extract data from PDF
                unique_id, data_points = pdf_processor.process_pdf(pdf_path)

                if not unique_id:
                    if status_callback:
                        status_callback(f"❌ Failed to extract ID from {pdf_file}")

                    # Move to unprocessed folder
                    shutil.copy2(pdf_path, os.path.join(unprocessed_dir, pdf_file))
                    results['unprocessed'] += 1
                    results['files']['unprocessed'].append(pdf_file)
                    continue

                # Find corresponding row in Excel
                row_index = excel_handler.find_row_by_id(unique_id)

                if row_index is None:
                    if status_callback:
                        status_callback(f"❌ Could not find matching row for ID: {unique_id} in {pdf_file}")

                    # Move to unprocessed folder
                    shutil.copy2(pdf_path, os.path.join(unprocessed_dir, pdf_file))
                    results['unprocessed'] += 1
                    results['files']['unprocessed'].append(pdf_file)
                    continue

                # Update Excel row with extracted data
                success, updated_fields, failed_fields = excel_handler.update_row_with_data(row_index, data_points)

                if success and len(failed_fields) == 0:
                    if status_callback:
                        status_callback(
                            f"✅ Successfully updated row for ID: {unique_id} with fields: {', '.join(updated_fields)}")

                    # Move to processed folder
                    shutil.copy2(pdf_path, os.path.join(processed_dir, pdf_file))
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
                    shutil.copy2(pdf_path, os.path.join(unprocessed_dir, pdf_file))
                    results['unprocessed'] += 1
                    results['files']['unprocessed'].append(pdf_file)

            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
                logger.error(traceback.format_exc())

                if status_callback:
                    status_callback(f"❌ Error processing {pdf_file}: {str(e)}")

                # Move to unprocessed folder
                shutil.copy2(pdf_path, os.path.join(unprocessed_dir, pdf_file))
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