from models.train_model import train_model_with_excel_data, test_model_on_new_pdf
import logging
import os
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("model_training.log")
    ]
)
logger = logging.getLogger("model_training")

# Path to your training data
excel_path = r"C:\Users\Lenovo\Downloads\pdf_to_excel_training_data.xlsx"

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

if __name__ == "__main__":
    start_time = time.time()
    logger.info(f"Starting model training with Excel data: {excel_path}")

    try:
        # Train with default settings (with OCR)
        extractor = train_model_with_excel_data(
            excel_path,
            use_ocr=True,  # Enable OCR for scanned PDFs
            use_grid_search=False,  # Set to True for hyperparameter tuning (much slower)
            threads=4  # Number of parallel threads
        )

        logger.info(f"Model training completed in {time.time() - start_time:.2f} seconds")

        # Uncomment and update the following lines to test the model on a sample PDF
        # test_pdf_url = "https://example.com/sample.pdf"  # Replace with a test PDF URL
        # logger.info(f"Testing model on PDF: {test_pdf_url}")
        # result = test_model_on_new_pdf(extractor, test_pdf_url)
        # logger.info(f"Extraction results: {result}")

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())