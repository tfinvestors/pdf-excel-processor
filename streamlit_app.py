import streamlit as st
import os
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import io
import zipfile
import datetime
import json
import platform
from dotenv import load_dotenv

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

# Set page config
st.set_page_config(
    page_title="PDF Processing & Excel Update Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        user_manager.log_activity(st.session_state.user_data['id'], activity_type, details)


# Authentication pages
def show_login_page():
    """Display the login page."""
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
                    if username and password:
                        success, result = user_manager.authenticate_user(username, password)

                        if success:
                            # Set session state
                            st.session_state.authenticated = True
                            st.session_state.user_data = result
                            st.session_state.current_page = 'main'

                            # Force a rerun to update the UI
                            st.rerun()
                        else:
                            st.markdown(f"<div class='error-msg'>Login failed: {result}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='error-msg'>Please enter both username and password</div>",
                                    unsafe_allow_html=True)

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            # Registration link - Fix for needing double-click
            st.markdown("<h3>New User?</h3>", unsafe_allow_html=True)
            register_btn = st.button("Register", key="register_button")
            if register_btn:
                st.session_state.current_page = 'register'
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)


def show_register_page():
    """Display the registration page."""
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
                if not all([username, email, password, confirm_password]):
                    st.error("Please fill in all required fields")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    success, result = user_manager.register_user(
                        username=username,
                        email=email,
                        password=password,
                        full_name=full_name
                    )

                    if success:
                        st.success("Registration successful! You can now login.")
                        # Automatically navigate to login page after a short delay
                        st.info("Redirecting to login page...")
                        st.session_state.current_page = 'login'
                        st.rerun()
                    else:
                        st.error(f"Registration failed: {result}")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Login link
        st.markdown("<h3>Already have an account?</h3>", unsafe_allow_html=True)
        if st.button("Login"):
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

        # Navigation
        st.markdown("### 🧭 Navigation")
        if st.button("🏠 Dashboard"):
            st.session_state.current_page = 'main'

        # Admin section
        if st.session_state.user_data.get('is_admin'):
            st.markdown("### 👑 Administration")
            if st.button("👥 User Management"):
                st.session_state.current_page = 'admin'
                st.rerun()

        # Account
        st.markdown("### 🔐 Account")
        if st.button("🔑 Change Password"):
            st.session_state.current_page = 'change_password'
            st.rerun()

        # Logout
        if st.button("🚪 Logout"):
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
    st.markdown("</div>", unsafe_allow_html=True)

    # File uploader for PDF files - ENHANCED VERSION
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📄 Upload your PDF files")
    st.markdown("<p>Select the PDF files you want to process and extract data from.</p>", unsafe_allow_html=True)
    pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True, key="pdf_uploader")
    st.markdown("</div>", unsafe_allow_html=True)

    # Process button - ENHANCED VERSION
    process_disabled = (excel_file is None or not pdf_files)
    if process_disabled:
        st.markdown("<div class='info-box'>Please upload both Excel and PDF files to continue</div>",
                    unsafe_allow_html=True)

    if st.button("🚀 Process Files", disabled=process_disabled, key="process_button"):
        if excel_file and pdf_files:
            log_user_activity("PROCESSING_STARTED", f"Processing {len(pdf_files)} PDF files")

            with st.spinner("Processing files..."):
                try:
                    # Create temporary directories
                    temp_dir = tempfile.mkdtemp()
                    pdf_folder = os.path.join(temp_dir, "pdfs")
                    os.makedirs(pdf_folder, exist_ok=True)

                    # Save Excel file to temp directory
                    excel_path = os.path.join(temp_dir, excel_file.name)
                    with open(excel_path, "wb") as f:
                        f.write(excel_file.getvalue())

                    # Save PDF files to temp directory
                    for pdf in pdf_files:
                        pdf_path = os.path.join(pdf_folder, pdf.name)
                        with open(pdf_path, "wb") as f:
                            f.write(pdf.getvalue())

                    # Set up progress bar and status
                    progress_bar = st.progress(0)
                    status_area = st.empty()

                    # Define callback functions for progress updates
                    def update_progress(current, total):
                        progress_bar.progress(current / total)

                    def update_status(message):
                        status_area.text(message)

                    # Process the files
                    results = process_files(excel_path, pdf_folder,
                                            progress_callback=update_progress,
                                            status_callback=update_status)

                    # Show results
                    st.success("Processing complete!")

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

                            # Add processed PDFs
                            if os.path.exists(processed_dir):
                                processed_files = [f for f in os.listdir(processed_dir) if f.lower().endswith('.pdf')]
                                if processed_files:
                                    for file in processed_files:
                                        file_path = os.path.join(processed_dir, file)
                                        zip_file.write(file_path, os.path.join("Processed PDF", file))
                                else:
                                    # Create an empty directory marker
                                    zip_info = zipfile.ZipInfo(os.path.join("Processed PDF", "/"))
                                    zip_info.external_attr = 0o755 << 16  # permissions
                                    zip_file.writestr(zip_info, "")

                            # Add unprocessed PDFs
                            if os.path.exists(unprocessed_dir):
                                unprocessed_files = [f for f in os.listdir(unprocessed_dir) if
                                                     f.lower().endswith('.pdf')]
                                if unprocessed_files:
                                    for file in unprocessed_files:
                                        file_path = os.path.join(unprocessed_dir, file)
                                        zip_file.write(file_path, os.path.join("Unprocessed PDF", file))
                                else:
                                    # Create an empty directory marker
                                    zip_info = zipfile.ZipInfo(os.path.join("Unprocessed PDF", "/"))
                                    zip_info.external_attr = 0o755 << 16  # permissions
                                    zip_file.writestr(zip_info, "")

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
                        st.markdown("</div>", unsafe_allow_html=True)

                    # Clean up temporary directory
                    shutil.rmtree(temp_dir)

                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
                    import traceback
                    st.exception(traceback.format_exc())
                    log_user_activity("PROCESSING_ERROR", f"Error: {str(e)}")


def show_change_password_page():
    """Display the change password page."""
    # Sidebar
    with st.sidebar:
        st.markdown(f"**Logged in as:** {st.session_state.user_data['username']}")

        # Navigation
        st.markdown("### Navigation")
        if st.button("PDF Processing"):
            st.session_state.current_page = 'main'
            st.rerun()

        # Admin section
        if st.session_state.user_data.get('is_admin'):
            st.markdown("### Administration")
            if st.button("User Management"):
                st.session_state.current_page = 'admin'
                st.rerun()

        # Logout
        if st.button("Logout"):
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
                if not all([current_password, new_password, confirm_password]):
                    st.error("Please fill in all fields")
                elif new_password != confirm_password:
                    st.error("New passwords do not match")
                else:
                    success, message = user_manager.change_password(
                        st.session_state.user_data['id'],
                        current_password,
                        new_password
                    )

                    if success:
                        st.success("Password changed successfully!")
                        log_user_activity("PASSWORD_CHANGED", "User changed their password")
                    else:
                        st.error(f"Failed to change password: {message}")


def show_admin_page():
    """Display the admin page."""
    if not st.session_state.user_data.get('is_admin', False):
        st.error("Access denied. Admin privileges required.")
        st.button("Back to Main", on_click=lambda: navigate_to('main'))
        return

    # Sidebar
    with st.sidebar:
        st.markdown(f"**Logged in as:** {st.session_state.user_data['username']} (Admin)")

        # Navigation
        st.markdown("### Navigation")
        if st.button("PDF Processing"):
            st.session_state.current_page = 'main'
            st.rerun()

        # Account
        st.markdown("### Account")
        if st.button("Change Password"):
            st.session_state.current_page = 'change_password'
            st.rerun()

        # Logout
        if st.button("Logout"):
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

    # Get user list
    success, users = user_manager.get_user_list(st.session_state.user_data['id'])

    if not success:
        st.error(f"Failed to get user list: {users}")
        return

    # Convert users to DataFrame for display
    user_df = pd.DataFrame(users)

    # Add action buttons
    st.dataframe(user_df[['id', 'username', 'email', 'full_name', 'is_admin', 'is_active', 'last_login']])

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
                # Prevent deactivating self
                if user_id == st.session_state.user_data['id'] and action == "Deactivate":
                    st.error("You cannot deactivate your own account")
                else:
                    success, message = user_manager.update_user_status(
                        st.session_state.user_data['id'],
                        user_id,
                        not selected_user['is_active']
                    )

                    if success:
                        st.success(message)
                        log_user_activity("USER_STATUS_CHANGED", f"Changed status for user {user_id}")
                        st.rerun()
                    else:
                        st.error(f"Failed to update user status: {message}")


def show_activity_log():
    """Display the activity log section."""
    st.subheader("Activity Log")

    # Here you would add code to retrieve and display the activity log
    # from your database, perhaps with filtering options

    st.info("Activity log feature is under development")


# Main app function
def main():
    """Main application entry point."""
    # Handle page navigation
    if st.session_state.current_page == 'login':
        show_login_page()
    elif st.session_state.current_page == 'register':
        show_register_page()
    elif st.session_state.current_page == 'main':
        if st.session_state.authenticated:
            show_main_page()
        else:
            navigate_to('login')
    elif st.session_state.current_page == 'change_password':
        if st.session_state.authenticated:
            show_change_password_page()
        else:
            navigate_to('login')
    elif st.session_state.current_page == 'admin':
        if st.session_state.authenticated and st.session_state.user_data.get('is_admin'):
            show_admin_page()
        else:
            navigate_to('login')
    else:
        navigate_to('login')


if __name__ == "__main__":
    main()