# streamlit_app.py
import streamlit as st
from streamlit_config import configure_streamlit_environment

configure_streamlit_environment()

import os
import logging
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import io
import zipfile
import datetime
import json
import sys
from io import StringIO
import platform
from dotenv import load_dotenv
from logging_utils import setup_logging


class StreamlitLogger:
    def __init__(self):
        self.log_capture = StringIO()

    def setup(self):
        # Create a custom handler that captures logs
        handler = logging.StreamHandler(self.log_capture)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # Add handler to the root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

    def get_logs(self, last_n_lines=50):
        # Get the captured logs
        self.log_capture.seek(0)
        lines = self.log_capture.readlines()
        return lines[-last_n_lines:] if lines else []

# THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="PDF Processing & Excel Update Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize at the start of your app
if 'streamlit_logger' not in st.session_state:
    st.session_state.streamlit_logger = StreamlitLogger()
    st.session_state.streamlit_logger.setup()

# Setup logging at the start of your app
logger, log_file = setup_logging("streamlit_pdf_app")
logger.info("Streamlit application started")
logger.info("TEST LOG - Application started")



st.write("Direct test log message written")

# Load environment variables from .env file if it exists
load_dotenv()

# Initialize platform-specific paths and create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Set Tesseract and Poppler paths based on platform
if platform.system() == 'Windows':
    os.environ['TESSERACT_PATH'] = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    os.environ['POPPLER_PATH'] = r'C:\poppler-24.08.0\Library\bin'
else:
    # On Streamlit Cloud, these are available through packages.txt
    os.environ['TESSERACT_PATH'] = '/usr/bin/tesseract'
    os.environ['POPPLER_PATH'] = ''  # Empty means use system path


# Define platform-neutral output directories
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


# Import your processing modules
from app.pdf_processor import PDFProcessor
from app.excel_handler import ExcelHandler
from app.main import process_files
from app.auth.user_manager import UserManager

# Initialize user manager with appropriate DB path
db_path = 'data/users.db'
os.makedirs(os.path.dirname(db_path), exist_ok=True)
user_manager = UserManager(db_path=db_path)

# Session state initialization
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'login'
if 'admin_tab' not in st.session_state:
    st.session_state.admin_tab = 'users'

# Custom CSS
st.markdown("""
    <style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        color: #4F8BF9;
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-align: center;
    }

    /* Sub header styling */
    .sub-header {
        font-size: 1.5rem;
        color: #36454F;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    /* Card styling */
    .card {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        background: white;
        transition: transform 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
    }

    /* Button styling */
    .stButton>button {
        border-radius: 50px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    /* Form styling */
    input, textarea, select {
        border-radius: 8px !important;
        border: 1px solid #E0E0E0 !important;
    }

    /* Success message */
    .success-msg {
        color: #4CAF50;
        font-weight: bold;
        background-color: #E8F5E9;
        padding: 10px;
        border-radius: 8px;
        border-left: 5px solid #4CAF50;
    }

    /* Error message */
    .error-msg {
        color: #F44336;
        font-weight: bold;
        background-color: #FFEBEE;
        padding: 10px;
        border-radius: 8px;
        border-left: 5px solid #F44336;
    }

    /* Info box */
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 5px solid #2196F3;
    }

    /* Divider */
    .divider {
        margin: 2rem 0;
        border-bottom: 1px solid #E0E0E0;
    }

    /* Dashboard metrics */
    .metric-card {
        background: linear-gradient(135deg, #6B73FF 0%, #000DFF 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }

    /* Custom file uploader */
    .stFileUploader > div {
        border-radius: 10px !important;
        border: 2px dashed #BDBDBD !important;
        padding: 2rem !important;
    }
    </style>
""", unsafe_allow_html=True)


# Helper functions
def navigate_to(page):
    """Navigate to a different page in the app."""
    st.session_state.current_page = page
    # Reset any page-specific state
    if page == 'login':
        st.session_state.authenticated = False
        st.session_state.user_data = None


def log_user_activity(activity_type, details):
    """Log user activity if authenticated."""
    if st.session_state.authenticated and st.session_state.user_data:
        logger.info(f"User activity: {activity_type} - {details}")
        user_manager.log_activity(st.session_state.user_data['id'], activity_type, details)


# Authentication pages
def show_login_page():
    """Display the login page."""
    # Add logs view to sidebar
    with st.sidebar:
        st.write(f"Logs are being written to: {log_file}")
        # Optional: Add a button to view logs
        # Replace the existing log viewing button code in the sidebar
        if st.button("View Recent Logs"):
            logs = st.session_state.streamlit_logger.get_logs()
            if logs:
                st.code(''.join(logs), language="text")
            else:
                st.info("No logs captured in this session")
            try:
                log_path = log_file
                st.write(f"Attempting to read log file: {log_path}")

                # Check if file exists
                if not os.path.exists(log_path):
                    st.error(f"Log file does not exist at: {log_path}")
                else:
                    # Check file size
                    file_size = os.path.getsize(log_path)
                    st.write(f"Log file exists. Size: {file_size} bytes")

                    if file_size == 0:
                        st.warning("Log file exists but is empty")
                    else:
                        with open(log_path, 'r') as f:
                            log_content = f.readlines()[-50:]  # Show last 50 lines
                            if not log_content:
                                st.warning("No content found in log file (file may be empty)")
                            else:
                                st.code(''.join(log_content), language="text")
            except Exception as e:
                st.error(f"Error reading log file: {str(e)}")
                # Print more detailed exception info
                import traceback
                st.code(traceback.format_exc(), language="python")

    # Use columns for overall page layout
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Create a container for the entire login card
        with st.container():
            # Add the main title outside the card but within the center column
            st.markdown("<h1 class='main-header'>PDF Processing & Excel Update Tool</h1>", unsafe_allow_html=True)

            # Create the card with proper styling
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h2 class='sub-header'>Login</h2>", unsafe_allow_html=True)

            # Login form
            with st.form("login_form"):
                username = st.text_input("Username or Email")
                password = st.text_input("Password", type="password")
                submit_button = st.form_submit_button("Login")

                if submit_button:
                    logger.info(f"Login attempt for user: {username}")
                    if username and password:
                        success, result = user_manager.authenticate_user(username, password)

                        if success:
                            # Set session state
                            st.session_state.authenticated = True
                            st.session_state.user_data = result
                            st.session_state.current_page = 'main'
                            logger.info(f"Login successful for user: {username}")

                            # Force a rerun to update the UI
                            st.rerun()
                        else:
                            logger.warning(f"Login failed for user: {username} - {result}")
                            st.markdown(f"<div class='error-msg'>Login failed: {result}</div>", unsafe_allow_html=True)
                    else:
                        logger.warning("Login attempt with empty username or password")
                        st.markdown("<div class='error-msg'>Please enter both username and password</div>",
                                    unsafe_allow_html=True)

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            # Registration link - Fix for needing double-click
            st.markdown("<h3>New User?</h3>", unsafe_allow_html=True)
            register_btn = st.button("Register", key="register_button")
            if register_btn:
                logger.info("User navigating to registration page")
                st.session_state.current_page = 'register'
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)


def show_register_page():
    """Display the registration page."""
    # Add logs view to sidebar
    with st.sidebar:
        st.write(f"Logs are being written to: {log_file}")
        # Optional: Add a button to view logs
        if st.button("View Recent Logs"):
            try:
                with open(log_file, 'r') as f:
                    log_content = f.readlines()[-50:]  # Show last 50 lines
                    st.code(''.join(log_content), language="text")
            except Exception as e:
                st.error(f"Couldn't read log file: {str(e)}")
                logger.error(f"Error reading log file: {str(e)}")

    st.markdown("<h1 class='main-header'>PDF Processing & Excel Update Tool</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("<h2 class='sub-header'>Register</h2>", unsafe_allow_html=True)

        # Registration form
        with st.form("register_form"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            full_name = st.text_input("Full Name")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")

            submit_button = st.form_submit_button("Register")

            if submit_button:
                logger.info(f"Registration attempt for username: {username}, email: {email}")
                if not all([username, email, password, confirm_password]):
                    logger.warning("Registration attempt with missing fields")
                    st.error("Please fill in all required fields")
                elif password != confirm_password:
                    logger.warning("Registration attempt with mismatched passwords")
                    st.error("Passwords do not match")
                else:
                    success, result = user_manager.register_user(
                        username=username,
                        email=email,
                        password=password,
                        full_name=full_name
                    )

                    if success:
                        logger.info(f"Registration successful for user: {username}")
                        st.success("Registration successful! You can now login.")
                        # Automatically navigate to login page after a short delay
                        st.info("Redirecting to login page...")
                        st.session_state.current_page = 'login'
                        st.rerun()
                    else:
                        logger.warning(f"Registration failed for user: {username} - {result}")
                        st.error(f"Registration failed: {result}")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Login link
        st.markdown("<h3>Already have an account?</h3>", unsafe_allow_html=True)
        if st.button("Login"):
            logger.info("User navigating to login page")
            navigate_to('login')


# Application pages
def show_main_page():
    """Display the main application page."""
    # Get environment-specific output directories
    processed_dir, unprocessed_dir = get_output_dirs()

    # Sidebar
    with st.sidebar:
        st.image(
            "https://www.computerworld.com/wp-content/uploads/2024/03/cw-pdf-to-excel-100928235-orig.jpg?quality=50&strip=all",
            width=100)
        st.markdown(f"**👤 Logged in as:** {st.session_state.user_data['username']}")

        # Add log viewing capability
        st.write(f"Logs are being written to: {log_file}")
        # Optional: Add a button to view logs
        if st.button("View Recent Logs"):
            try:
                with open(log_file, 'r') as f:
                    log_content = f.readlines()[-50:]  # Show last 50 lines
                    st.code(''.join(log_content), language="text")
            except Exception as e:
                st.error(f"Couldn't read log file: {str(e)}")
                logger.error(f"Error reading log file: {str(e)}")

        # Debug toggle
        if st.checkbox("Enable Debug Logging"):
            logger.setLevel(logging.DEBUG)
            st.info("Debug logging enabled")
        else:
            logger.setLevel(logging.INFO)

        # Navigation
        st.markdown("### 🧭 Navigation")
        if st.button("🏠 Dashboard"):
            logger.info("User navigating to dashboard")
            st.session_state.current_page = 'main'

        # Admin section
        if st.session_state.user_data.get('is_admin'):
            st.markdown("### 👑 Administration")
            if st.button("👥 User Management"):
                logger.info("Admin navigating to user management")
                st.session_state.current_page = 'admin'
                st.rerun()

        # Account
        st.markdown("### 🔐 Account")
        if st.button("🔑 Change Password"):
            logger.info("User navigating to change password")
            st.session_state.current_page = 'change_password'
            st.rerun()

        # Logout
        if st.button("🚪 Logout"):
            logger.info(f"User logging out: {st.session_state.user_data['username']}")
            if st.session_state.user_data and 'session_token' in st.session_state.user_data:
                user_manager.logout_user(st.session_state.user_data['session_token'])
            navigate_to('login')
            st.rerun()

        # Environment info (only shown to admins)
        if st.session_state.user_data.get('is_admin'):
            st.markdown("### ⚙️ Environment")
            env_type = "Cloud" if 'STREAMLIT_SHARING' in os.environ else "Local"
            st.info(f"Running in: {env_type}")
            st.info(f"Output folders:\n- {processed_dir}\n- {unprocessed_dir}")

    # Main content
    st.markdown("<h1 class='main-header'>PDF Processing & Excel Update Tool</h1>", unsafe_allow_html=True)

    st.markdown(
        "<div class='info-box'>This application extracts data from PDF files and updates matching records in an Excel file.</div>",
        unsafe_allow_html=True)

    # File uploader for Excel file - ENHANCED VERSION
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Upload your Excel file")
    st.markdown("<p>Select the Excel file containing the data you want to update.</p>", unsafe_allow_html=True)
    excel_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"], key="excel_uploader")
    if excel_file:
        logger.info(f"Excel file uploaded: {excel_file.name}")
    st.markdown("</div>", unsafe_allow_html=True)

    # File uploader for PDF files - ENHANCED VERSION
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📄 Upload your PDF files")
    st.markdown("<p>Select the PDF files you want to process and extract data from.</p>", unsafe_allow_html=True)
    pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True, key="pdf_uploader")
    if pdf_files:
        logger.info(f"PDF files uploaded: {len(pdf_files)} files")
        for pdf in pdf_files:
            logger.debug(f"PDF file: {pdf.name}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Process button - ENHANCED VERSION
    process_disabled = (excel_file is None or not pdf_files)
    if process_disabled:
        st.markdown("<div class='info-box'>Please upload both Excel and PDF files to continue</div>",
                    unsafe_allow_html=True)

    if st.button("🚀 Process Files", disabled=process_disabled, key="process_button"):
        if excel_file and pdf_files:
            log_user_activity("PROCESSING_STARTED", f"Processing {len(pdf_files)} PDF files")
            logger.info(f"Starting to process {len(pdf_files)} PDF files with Excel file: {excel_file.name}")

            with st.spinner("Processing files..."):
                try:
                    # Create temporary directories
                    temp_dir = tempfile.mkdtemp()
                    pdf_folder = os.path.join(temp_dir, "pdfs")
                    os.makedirs(pdf_folder, exist_ok=True)
                    logger.debug(f"Created temporary directories: {temp_dir}, {pdf_folder}")

                    # Save Excel file to temp directory
                    excel_path = os.path.join(temp_dir, excel_file.name)
                    with open(excel_path, "wb") as f:
                        f.write(excel_file.getvalue())
                    logger.info(f"Saved Excel file to: {excel_path}")

                    # Save PDF files to temp directory
                    for pdf in pdf_files:
                        pdf_path = os.path.join(pdf_folder, pdf.name)
                        with open(pdf_path, "wb") as f:
                            f.write(pdf.getvalue())
                        logger.debug(f"Saved PDF file: {pdf_path}")
                    logger.info(f"Saved {len(pdf_files)} PDF files to: {pdf_folder}")

                    # Set up progress bar and status
                    progress_bar = st.progress(0)
                    status_area = st.empty()

                    # Define callback functions for progress updates
                    def update_progress(current, total):
                        progress = current / total
                        progress_bar.progress(progress)
                        logger.debug(f"Progress update: {current}/{total} ({progress:.2%})")

                    def update_status(message):
                        status_area.text(message)
                        logger.info(f"Status update: {message}")

                    # Process the files
                    try:
                        logger.info("Calling process_files function")
                        results = process_files(excel_path, pdf_folder,
                                                progress_callback=update_progress,
                                                status_callback=update_status)
                        logger.info(f"Processing completed successfully: {results}")
                    except Exception as e:
                        error_msg = f"Error in process_files: {str(e)}"
                        logger.error(error_msg, exc_info=True)
                        raise RuntimeError(error_msg)

                    # Show results
                    st.success("Processing complete!")
                    logger.info("Processing complete!")

                    # Enhanced results display
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.subheader("📊 Processing Results")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total PDFs", results['total'])
                    with col2:
                        st.metric("Successfully Processed", results['processed'],
                                  delta=f"{results['processed'] / results['total'] * 100:.1f}%" if results[
                                                                                                       'total'] > 0 else None)
                    with col3:
                        st.metric("Failed to Process", results['unprocessed'])

                    st.markdown("</div>", unsafe_allow_html=True)

                    # Log the processing results
                    log_user_activity(
                        "PROCESSING_COMPLETED",
                        f"Processed: {results['processed']}, Unprocessed: {results['unprocessed']}"
                    )

                    # Create a zip file with the results
                    with io.BytesIO() as zip_buffer:
                        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                            # Add the updated Excel file
                            zip_file.write(excel_path, os.path.basename(excel_path))
                            logger.debug(f"Added Excel file to ZIP: {os.path.basename(excel_path)}")

                            # Add processed PDFs
                            if os.path.exists(processed_dir):
                                processed_files = [f for f in os.listdir(processed_dir) if f.lower().endswith('.pdf')]
                                if processed_files:
                                    for file in processed_files:
                                        file_path = os.path.join(processed_dir, file)
                                        zip_file.write(file_path, os.path.join("Processed PDF", file))
                                        logger.debug(f"Added processed PDF to ZIP: {file}")
                                else:
                                    # Create an empty directory marker
                                    zip_info = zipfile.ZipInfo(os.path.join("Processed PDF", "/"))
                                    zip_info.external_attr = 0o755 << 16  # permissions
                                    zip_file.writestr(zip_info, "")
                                    logger.debug("Created empty processed PDF directory in ZIP")

                            # Add unprocessed PDFs
                            if os.path.exists(unprocessed_dir):
                                unprocessed_files = [f for f in os.listdir(unprocessed_dir) if
                                                     f.lower().endswith('.pdf')]
                                if unprocessed_files:
                                    for file in unprocessed_files:
                                        file_path = os.path.join(unprocessed_dir, file)
                                        zip_file.write(file_path, os.path.join("Unprocessed PDF", file))
                                        logger.debug(f"Added unprocessed PDF to ZIP: {file}")
                                else:
                                    # Create an empty directory marker
                                    zip_info = zipfile.ZipInfo(os.path.join("Unprocessed PDF", "/"))
                                    zip_info.external_attr = 0o755 << 16  # permissions
                                    zip_file.writestr(zip_info, "")
                                    logger.debug("Created empty unprocessed PDF directory in ZIP")

                        # Offer download of the zip file - ENHANCED VERSION
                        zip_buffer.seek(0)
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.subheader("📦 Download Results")
                        st.markdown(
                            "<p>Download a zip file containing the updated Excel file and processed/unprocessed PDFs.</p>",
                            unsafe_allow_html=True)
                        st.download_button(
                            label="📥 Download Results",
                            data=zip_buffer,
                            file_name="pdf_processing_results.zip",
                            mime="application/zip"
                        )
                        logger.info("Created ZIP file for download with processing results")
                        st.markdown("</div>", unsafe_allow_html=True)

                    # Clean up temporary directory
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temporary directory: {temp_dir}")

                except Exception as e:
                    error_msg = f"An error occurred during processing: {str(e)}"
                    st.error(error_msg)
                    logger.error(error_msg, exc_info=True)
                    import traceback
                    st.exception(traceback.format_exc())
                    log_user_activity("PROCESSING_ERROR", f"Error: {str(e)}")


def show_change_password_page():
    """Display the change password page."""
    # Sidebar
    with st.sidebar:
        st.markdown(f"**Logged in as:** {st.session_state.user_data['username']}")

        # Add log viewing capability
        st.write(f"Logs are being written to: {log_file}")
        # Optional: Add a button to view logs
        if st.button("View Recent Logs"):
            try:
                with open(log_file, 'r') as f:
                    log_content = f.readlines()[-50:]  # Show last 50 lines
                    st.code(''.join(log_content), language="text")
            except Exception as e:
                st.error(f"Couldn't read log file: {str(e)}")
                logger.error(f"Error reading log file: {str(e)}")

        # Navigation
        st.markdown("### Navigation")
        if st.button("PDF Processing"):
            logger.info("User navigating to PDF processing")
            st.session_state.current_page = 'main'
            st.rerun()

        # Admin section
        if st.session_state.user_data.get('is_admin'):
            st.markdown("### Administration")
            if st.button("User Management"):
                logger.info("Admin navigating to user management")
                st.session_state.current_page = 'admin'
                st.rerun()

        # Logout
        if st.button("Logout"):
            logger.info(f"User logging out: {st.session_state.user_data['username']}")
            if st.session_state.user_data and 'session_token' in st.session_state.user_data:
                user_manager.logout_user(st.session_state.user_data['session_token'])
            navigate_to('login')
            st.rerun()

    # Main content
    st.markdown("<h1 class='main-header'>Change Password</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Change password form
        with st.form("change_password_form"):
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")

            submit_button = st.form_submit_button("Change Password")

            if submit_button:
                logger.info(f"Password change attempt for user: {st.session_state.user_data['username']}")
                if not all([current_password, new_password, confirm_password]):
                    logger.warning("Password change attempt with missing fields")
                    st.error("Please fill in all fields")
                elif new_password != confirm_password:
                    logger.warning("Password change attempt with mismatched passwords")
                    st.error("New passwords do not match")
                else:
                    success, message = user_manager.change_password(
                        st.session_state.user_data['id'],
                        current_password,
                        new_password
                    )

                    if success:
                        logger.info(f"Password changed successfully for user: {st.session_state.user_data['username']}")
                        st.success("Password changed successfully!")
                        log_user_activity("PASSWORD_CHANGED", "User changed their password")
                    else:
                        logger.warning(f"Password change failed: {message}")
                        st.error(f"Failed to change password: {message}")


def show_admin_page():
    """Display the admin page."""
    if not st.session_state.user_data.get('is_admin', False):
        logger.warning(f"Non-admin user attempting to access admin page: {st.session_state.user_data['username']}")
        st.error("Access denied. Admin privileges required.")
        st.button("Back to Main", on_click=lambda: navigate_to('main'))
        return

    # Sidebar
    with st.sidebar:
        st.markdown(f"**Logged in as:** {st.session_state.user_data['username']} (Admin)")

        # Add log viewing capability
        st.write(f"Logs are being written to: {log_file}")
        # Optional: Add a button to view logs
        if st.button("View Recent Logs"):
            try:
                with open(log_file, 'r') as f:
                    log_content = f.readlines()[-50:]  # Show last 50 lines
                    st.code(''.join(log_content), language="text")
            except Exception as e:
                st.error(f"Couldn't read log file: {str(e)}")
                logger.error(f"Error reading log file: {str(e)}")

        # Navigation
        st.markdown("### Navigation")
        if st.button("PDF Processing"):
            logger.info("Admin navigating to PDF processing")
            st.session_state.current_page = 'main'
            st.rerun()

        # Account
        st.markdown("### Account")
        if st.button("Change Password"):
            logger.info("Admin navigating to change password")
            st.session_state.current_page = 'change_password'
            st.rerun()

        # Logout
        if st.button("Logout"):
            logger.info(f"Admin logging out: {st.session_state.user_data['username']}")
            if st.session_state.user_data and 'session_token' in st.session_state.user_data:
                user_manager.logout_user(st.session_state.user_data['session_token'])
            navigate_to('login')
            st.rerun()

        # Database info for admins
        st.markdown("### Database Info")
        db_path = 'data/users.db'
        st.info(f"Database path: {os.path.abspath(db_path)}")

        # Environment info
        st.markdown("### Environment")
        env_type = "Cloud" if 'STREAMLIT_SHARING' in os.environ else "Local"
        st.info(f"Running in: {env_type}")

    # Main content
    st.markdown("<h1 class='main-header'>Administration</h1>", unsafe_allow_html=True)

    # Admin tabs
    admin_tab = st.radio("Select tab:", ["User Management", "Activity Log"],
                         index=0 if st.session_state.admin_tab == 'users' else 1,
                         horizontal=True)

    if admin_tab == "User Management":
        st.session_state.admin_tab = 'users'
        show_user_management()
    else:
        st.session_state.admin_tab = 'activity'
        show_activity_log()


def show_user_management():
    """Display the user management section."""
    st.subheader("User Management")
    logger.info("Admin viewing user management page")

    # Get user list
    try:
        success, users = user_manager.get_user_list(st.session_state.user_data['id'])

        if not success:
            logger.error(f"Failed to get user list: {users}")
            st.error(f"Failed to get user list: {users}")
            return

        # Convert users to DataFrame for display
        user_df = pd.DataFrame(users)

        # Add action buttons
        st.dataframe(user_df[['id', 'username', 'email', 'full_name', 'is_admin', 'is_active', 'last_login']])
        logger.debug(f"Displaying {len(users)} users in management table")

        # User status management
        st.subheader("Change User Status")

        col1, col2 = st.columns(2)

        with col1:
            user_id = st.selectbox("Select User", options=[user['id'] for user in users],
                                   format_func=lambda x: next((u['username'] for u in users if u['id'] == x), ""))

        with col2:
            selected_user = next((u for u in users if u['id'] == user_id), None)
            current_status = "Active" if selected_user and selected_user['is_active'] else "Inactive"
            st.write(f"Current Status: {current_status}")

            action = "Deactivate" if selected_user and selected_user['is_active'] else "Activate"
            if st.button(action, key=f"status_{user_id}"):
                if selected_user:
                    logger.info(
                        f"Admin attempting to {action.lower()} user: {selected_user['username']} (ID: {user_id})")
                    # Prevent deactivating self
                    if user_id == st.session_state.user_data['id'] and action == "Deactivate":
                        logger.warning("Admin attempted to deactivate their own account")
                        st.error("You cannot deactivate your own account")
                    else:
                        success, message = user_manager.update_user_status(
                            st.session_state.user_data['id'],
                            user_id,
                            not selected_user['is_active']
                        )

                        if success:
                            logger.info(
                                f"Successfully {action.lower()}d user: {selected_user['username']} (ID: {user_id})")
                            st.success(message)
                            log_user_activity("USER_STATUS_CHANGED", f"Changed status for user {user_id}")
                            st.rerun()
                        else:
                            logger.error(f"Failed to {action.lower()} user: {message}")
                            st.error(f"Failed to update user status: {message}")
    except Exception as e:
        error_msg = f"Error in user management: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)


def show_activity_log():
    """Display the activity log section."""
    st.subheader("Activity Log")
    logger.info("Admin viewing activity log")

    # Here you would add code to retrieve and display the activity log
    # from your database, perhaps with filtering options

    st.info("Activity log feature is under development")


# Main app function
def main():
    """Main application entry point."""
    # Ensure logging available in global scope
    import logging

    # Handle page navigation
    if st.session_state.current_page == 'login':
        show_login_page()
    elif st.session_state.current_page == 'register':
        show_register_page()
    elif st.session_state.current_page == 'main':
        if st.session_state.authenticated:
            show_main_page()
        else:
            logger.warning("Unauthenticated user attempted to access main page")
            navigate_to('login')
    elif st.session_state.current_page == 'change_password':
        if st.session_state.authenticated:
            show_change_password_page()
        else:
            logger.warning("Unauthenticated user attempted to access change password page")
            navigate_to('login')
    elif st.session_state.current_page == 'admin':
        if st.session_state.authenticated and st.session_state.user_data.get('is_admin'):
            show_admin_page()
        else:
            logger.warning("Unauthorized user attempted to access admin page")
            navigate_to('login')
    else:
        logger.warning(f"Unknown page requested: {st.session_state.current_page}")
        navigate_to('login')


if __name__ == "__main__":
    main()