# logging_utils.py
import logging
import sys
import os
from datetime import datetime
import io


class StreamlitCompatibleHandler(logging.StreamHandler):
    """Custom handler that works well with Streamlit"""

    def __init__(self, stream=None):
        super().__init__(stream)
        self.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

    def emit(self, record):
        try:
            # Clean the message of problematic unicode characters
            if hasattr(record, 'msg') and isinstance(record.msg, str):
                # Replace problematic emoji characters
                record.msg = record.msg.replace('📊', '[TABLE]')
                record.msg = record.msg.replace('❌', '[ERROR]')
                record.msg = record.msg.replace('✅', '[SUCCESS]')
                record.msg = record.msg.replace('⚠️', '[WARNING]')
                record.msg = record.msg.replace('🚀', '[PROCESS]')
                record.msg = record.msg.replace('📦', '[DOWNLOAD]')
                record.msg = record.msg.replace('📥', '[DOWNLOAD]')

            super().emit(record)
        except (UnicodeEncodeError, UnicodeDecodeError):
            # If encoding fails, create a safe version
            try:
                safe_msg = str(record.msg).encode('ascii', 'ignore').decode('ascii')
                record.msg = safe_msg
                super().emit(record)
            except:
                pass  # If all else fails, skip this log


def setup_logging(app_name="streamlit_pdf_app"):
    """Setup logging with Streamlit compatibility"""

    # Clear any existing handlers to avoid conflicts
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    # Create a timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"{app_name}_{timestamp}.log")

    # Configure root logger with safe level
    root_logger.setLevel(logging.INFO)

    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler with custom StreamlitCompatibleHandler
    console_handler = StreamlitCompatibleHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler with explicit UTF-8 encoding
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create file handler: {e}")

    # Create dedicated logger for the app
    logger = logging.getLogger(app_name)
    logger.setLevel(logging.DEBUG)

    # Test the logger
    logger.info(f"Logging initialized for {app_name}")
    logger.info(f"Log file: {log_file}")

    return logger, log_file


def get_log_contents(log_file_path, max_lines=100):
    """Safely read log file contents"""
    try:
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                return lines[-max_lines:] if len(lines) > max_lines else lines
        else:
            return [f"Log file not found: {log_file_path}"]
    except Exception as e:
        return [f"Error reading log file: {str(e)}"]