import os
import logging
import json
import pandas as pd
from pathlib import Path
from app.pdf_processor import PDFProcessor
from app.excel_handler import ExcelHandler
from app.main import process_files

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("regression_tests.log")
    ]
)
logger = logging.getLogger("regression_tests")


class PDFExtractionTester:
    def __init__(self, test_data_dir="test_data"):
        """
        Initialize the PDF extraction tester.

        Args:
            test_data_dir (str): Directory containing test data
        """
        self.test_data_dir = test_data_dir
        self.results = {
            "passed": 0,
            "failed": 0,
            "total": 0,
            "details": []
        }

        # Create test data directory if it doesn't exist
        os.makedirs(self.test_data_dir, exist_ok=True)

        # Initialize the PDF processor with debug mode enabled
        self.pdf_processor = PDFProcessor(use_ml=True, debug_mode=True)

    def add_test_case(self, pdf_path, expected_data):
        """
        Add a test case to the test data.

        Args:
            pdf_path (str): Path to the PDF file
            expected_data (dict): Expected extraction results
        """
        # Create a JSON file with expected data
        pdf_name = os.path.basename(pdf_path)
        json_path = os.path.join(self.test_data_dir, f"{os.path.splitext(pdf_name)[0]}_expected.json")

        with open(json_path, 'w') as f:
            json.dump(expected_data, f, indent=2)

        logger.info(f"Added test case for {pdf_name}")

    def run_tests(self, pdf_dir):
        """
        Run tests on all PDFs in the given directory.

        Args:
            pdf_dir (str): Directory containing PDF files to test

        Returns:
            dict: Test results
        """
        # Get list of PDFs
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        self.results["total"] = len(pdf_files)

        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            test_result = self.test_pdf(pdf_path)
            self.results["details"].append(test_result)

            if test_result["passed"]:
                self.results["passed"] += 1
            else:
                self.results["failed"] += 1

        logger.info(f"Tests completed: {self.results['passed']} passed, {self.results['failed']} failed")
        return self.results

    def test_pdf(self, pdf_path):
        """
        Test a single PDF against expected results.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            dict: Test result details
        """
        pdf_name = os.path.basename(pdf_path)
        result = {
            "pdf": pdf_name,
            "passed": False,
            "extracted_data": {},
            "expected_data": {},
            "differences": []
        }

        # Check if expected data exists
        expected_json = os.path.join(self.test_data_dir, f"{os.path.splitext(pdf_name)[0]}_expected.json")
        if not os.path.exists(expected_json):
            logger.warning(f"No expected data found for {pdf_name}")
            result["differences"].append("No expected data")
            return result

        # Load expected data
        with open(expected_json, 'r') as f:
            expected_data = json.load(f)
            result["expected_data"] = expected_data

        # Process the PDF
        unique_id, data_points, table_data = self.pdf_processor.process_pdf(pdf_path)

        # Store extracted data
        extracted_data = {
            "unique_id": unique_id,
            "data_points": data_points,
            "table_data": table_data
        }
        result["extracted_data"] = extracted_data

        # Compare results
        differences = []

        # Check unique ID
        if unique_id != expected_data.get("unique_id"):
            differences.append(f"Unique ID: Expected '{expected_data.get('unique_id')}', got '{unique_id}'")

        # Check data points
        expected_data_points = expected_data.get("data_points", {})
        for field, expected_value in expected_data_points.items():
            if field not in data_points:
                differences.append(f"Missing field: {field}")
            elif data_points[field] != expected_value:
                differences.append(f"Field {field}: Expected '{expected_value}', got '{data_points[field]}'")

        # Store differences
        result["differences"] = differences

        # Mark as passed if no differences
        result["passed"] = len(differences) == 0

        # Log result
        if result["passed"]:
            logger.info(f"✓ {pdf_name} passed")
        else:
            logger.warning(f"✗ {pdf_name} failed")
            for diff in differences:
                logger.warning(f"  - {diff}")

        return result

    def test_integration(self, excel_path, pdf_dir):
        """
        Test the full integration flow - processing PDFs and updating Excel.

        Args:
            excel_path (str): Path to the Excel file
            pdf_dir (str): Directory containing PDF files

        Returns:
            dict: Test results
        """
        logger.info(f"Running integration test with Excel: {excel_path} and PDFs from: {pdf_dir}")

        # Create a backup of the original Excel file
        backup_path = excel_path.replace('.xlsx', f'_test_backup.xlsx')
        excel_df = pd.read_excel(excel_path)
        excel_df.to_excel(backup_path, index=False)
        logger.info(f"Created Excel backup: {backup_path}")

        # Process the files and capture results
        results = process_files(excel_path, pdf_dir)

        # Analyze results
        test_results = {
            "total_pdfs": results['total'],
            "processed_pdfs": results['processed'],
            "unprocessed_pdfs": results['unprocessed'],
            "processed_files": results['files']['processed'],
            "unprocessed_files": results['files']['unprocessed'],
            "success_rate": (results['processed'] / results['total'] * 100) if results['total'] > 0 else 0
        }

        # Restore original Excel file
        restore_df = pd.read_excel(backup_path)
        restore_df.to_excel(excel_path, index=False)
        logger.info(f"Restored Excel from backup")

        return test_results


# Create test cases for specific PDFs
def create_test_cases():
    tester = PDFExtractionTester()

    # Test case for OICL ADVICE.pdf
    oicl_expected = {
        "unique_id": "510000/11/2025/000000",
        "data_points": {
            "receipt_amount": "201156",
            "receipt_date": "11-02-2025",
            "tds": "22350.67"  # 11.111111% of 201,156
        }
    }
    tester.add_test_case("OICL ADVICE.pdf", oicl_expected)

    # Test case for UNITED ADVICE.pdf
    united_expected = {
        "unique_id": "5004002624C05021400",
        "data_points": {
            "receipt_amount": "7738.00",
            "receipt_date": "12-02-2025",
            "tds": "859.77"  # 11.111111% of 7,738.00
        }
    }
    tester.add_test_case("UNITED ADVICE (1).pdf", united_expected)

    logger.info("Created test cases for sample PDFs")


# Run the tests
def run_regression_tests(excel_path, pdf_dir):
    create_test_cases()

    tester = PDFExtractionTester()

    # Run individual PDF tests
    pdf_results = tester.run_tests(pdf_dir)

    # Run integration test
    integration_results = tester.test_integration(excel_path, pdf_dir)

    # Output results
    print("\n===== PDF EXTRACTION TESTS =====")
    print(f"Total: {pdf_results['total']}")
    print(f"Passed: {pdf_results['passed']}")
    print(f"Failed: {pdf_results['failed']}")
    print(f"Success Rate: {pdf_results['passed'] / pdf_results['total'] * 100:.2f}%")

    print("\n===== INTEGRATION TEST =====")
    print(f"Total PDFs: {integration_results['total_pdfs']}")
    print(f"Successfully Processed: {integration_results['processed_pdfs']}")
    print(f"Failed to Process: {integration_results['unprocessed_pdfs']}")
    print(f"Success Rate: {integration_results['success_rate']:.2f}%")


if __name__ == "__main__":
    # Set the paths for your test files
    excel_path = r"D:\backup 22.9.21\PSEPL\CDMS\Automation Project\PDF to Excel\UAT\pdf_to_excel_testing_data.xlsx"
    pdf_dir = r"D:\backup 22.9.21\PSEPL\CDMS\Automation Project\PDF to Excel\UAT\testing_invoice_pdf"

    run_regression_tests(excel_path, pdf_dir)