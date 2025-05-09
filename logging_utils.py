# logging_utils.py
import logging
import sys
import os
from datetime import datetime


def setup_logging(app_name="pdf_extraction_app"):
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    # Create a timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"{app_name}_{timestamp}.log")

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )

    # Create dedicated logger for the app
    logger = logging.getLogger(app_name)
    logger.setLevel(logging.DEBUG)

    return logger, log_file