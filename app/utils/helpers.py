import os
import re
import logging
import pandas as pd
import shutil
from pathlib import Path
import pytesseract
import importlib.util
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("utils.log")
    ]
)
logger = logging.getLogger("utils")


def check_dependencies():
    """
    Check if all required dependencies are installed and available.

    Returns:
        tuple: (success, missing_dependencies)
    """
    required_packages = [
        'PyPDF2', 'pdfplumber', 'pytesseract', 'pdf2image',
        'openpyxl', 'pandas', 'pillow', 'customtkinter',
        'spacy', 'scikit-learn', 'joblib'
    ]

    missing = []

    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing.append(package)

    # Check for special cases like Tesseract OCR
    try:
        pytesseract.get_tesseract_version()
    except:
        missing.append('tesseract-ocr (external)')

    success = len(missing) == 0

    if not success:
        logger.warning(f"Missing dependencies: {', '.join(missing)}")

    return success, missing


def install_missing_dependencies(missing_dependencies):
    """
    Install missing Python dependencies.

    Args:
        missing_dependencies (list): List of missing package names

    Returns:
        bool: True if installation was successful
    """
    try:
        import subprocess

        # Filter out external dependencies
        python_packages = [pkg for pkg in missing_dependencies if 'external' not in pkg]

        if python_packages:
            logger.info(f"Installing missing packages: {', '.join(python_packages)}")
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + python_packages)

        # Handle special cases
        if 'tesseract-ocr (external)' in missing_dependencies:
            if os.name == 'nt':  # Windows
                logger.warning(
                    "Please install Tesseract OCR manually from: https://github.com/UB-Mannheim/tesseract/wiki")
            else:  # Linux/Mac
                logger.info("Attempting to install Tesseract OCR...")
                if os.name == 'posix':  # Linux
                    subprocess.check_call(["apt-get", "update"])
                    subprocess.check_call(["apt-get", "install", "-y", "tesseract-ocr"])
                elif sys.platform == 'darwin':  # Mac
                    subprocess.check_call(["brew", "install", "tesseract"])

        return True

    except Exception as e:
        logger.error(f"Error installing dependencies: {str(e)}")
        return False


def validate_excel_file(excel_path):
    """
    Validate that the Excel file has the expected structure.

    Args:
        excel_path (str): Path to the Excel file

    Returns:
        tuple: (is_valid, issues)
    """
    issues = []

    try:
        # Check if file exists
        if not os.path.exists(excel_path):
            issues.append("Excel file does not exist")
            return False, issues

        # Try to load the file
        df = pd.read_excel(excel_path)

        # Check if it has content
        if len(df) == 0:
            issues.append("Excel file has no data rows")

        if len(df.columns) == 0:
            issues.append("Excel file has no columns")

        # Check for potential ID columns
        potential_id_cols = [
            'id', 'identifier', 'reference', 'doc_id', 'invoice', 'case',
            'ID', 'Id', 'Reference', 'Doc_ID', 'Invoice', 'Case'
        ]

        has_id_column = any(col in df.columns or
                            any(id_col.lower() in col.lower() for id_col in potential_id_cols)
                            for col in df.columns)

        if not has_id_column:
            issues.append("Excel file does not appear to have an ID column")

        # Check for duplicates in potential ID columns
        for col in df.columns:
            if any(id_col.lower() in col.lower() for id_col in potential_id_cols):
                dup_count = df[col].duplicated().sum()
                if dup_count > 0:
                    issues.append(f"Column '{col}' (potential ID column) has {dup_count} duplicate values")

        return len(issues) == 0, issues

    except Exception as e:
        issues.append(f"Error validating Excel file: {str(e)}")
        return False, issues


def validate_pdf_folder(pdf_folder):
    """
    Validate that the PDF folder has PDF files.

    Args:
        pdf_folder (str): Path to the PDF folder

    Returns:
        tuple: (is_valid, issues, pdf_count)
    """
    issues = []
    pdf_count = 0

    try:
        # Check if folder exists
        if not os.path.exists(pdf_folder):
            issues.append("PDF folder does not exist")
            return False, issues, 0

        # Check if it's a directory
        if not os.path.isdir(pdf_folder):
            issues.append("Path is not a directory")
            return False, issues, 0

        # Count PDF files
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
        pdf_count = len(pdf_files)

        if pdf_count == 0:
            issues.append("No PDF files found in the folder")

        # Test a sample PDF file
        if pdf_count > 0:
            sample_pdf = os.path.join(pdf_folder, pdf_files[0])

            # Try to open the PDF (check if it's not corrupt)
            try:
                import PyPDF2
                with open(sample_pdf, 'rb') as f:
                    PyPDF2.PdfReader(f)
            except Exception as e:
                issues.append(f"Error opening sample PDF file: {str(e)}")

        return len(issues) == 0, issues, pdf_count

    except Exception as e:
        issues.append(f"Error validating PDF folder: {str(e)}")
        return False, issues, 0


def create_output_folders():
    """
    Create output folders for processed and unprocessed PDFs.

    Returns:
        tuple: (success, folder_paths)
    """
    try:
        # Create folders in user's Downloads directory
        download_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        processed_dir = os.path.join(download_dir, "Processed PDF")
        unprocessed_dir = os.path.join(download_dir, "Unprocessed PDF")

        for dir_path in [processed_dir, unprocessed_dir]:
            os.makedirs(dir_path, exist_ok=True)

        return True, (processed_dir, unprocessed_dir)

    except Exception as e:
        logger.error(f"Error creating output folders: {str(e)}")
        return False, (None, None)


def get_file_info(file_path):
    """
    Get basic information about a file.

    Args:
        file_path (str): Path to the file

    Returns:
        dict: File information
    """
    info = {
        'exists': False,
        'size': 0,
        'created': None,
        'modified': None,
        'extension': None,
        'is_readable': False
    }

    try:
        if os.path.exists(file_path):
            info['exists'] = True
            info['size'] = os.path.getsize(file_path)
            info['created'] = os.path.getctime(file_path)
            info['modified'] = os.path.getmtime(file_path)
            info['extension'] = os.path.splitext(file_path)[1].lower()

            # Check if file is readable
            try:
                with open(file_path, 'rb') as f:
                    f.read(1)
                info['is_readable'] = True
            except:
                info['is_readable'] = False
    except Exception as e:
        logger.error(f"Error getting file info for {file_path}: {str(e)}")

    return info


def clean_filename(filename):
    """
    Clean a filename to ensure it's valid on all operating systems.

    Args:
        filename (str): Original filename

    Returns:
        str: Cleaned filename
    """
    # Replace invalid characters with underscore
    invalid_chars = r'[<>:"/\\|?*]'
    cleaned = re.sub(invalid_chars, '_', filename)

    # Ensure filename is not empty
    if not cleaned or cleaned.isspace():
        cleaned = "unnamed_file"

    # Trim to reasonable length
    if len(cleaned) > 200:
        base, ext = os.path.splitext(cleaned)
        cleaned = base[:196] + ext

    return cleaned