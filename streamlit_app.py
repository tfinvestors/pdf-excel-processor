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

# Import your processing modules
from app.pdf_processor import PDFProcessor
from app.excel_handler import ExcelHandler
from app.main import process_files
from app.auth.user_manager import UserManager

# Initialize user manager
user_manager = UserManager()

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
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .centered {
        display: flex;
        justify-content: center;
    }
    .success-msg {
        color: #4CAF50;
        font-weight: bold;
    }
    .error-msg {
        color: #F44336;
        font-weight: bold;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .divider {
        margin: 2rem 0;
        border-bottom: 1px solid #E0E0E0;
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
    st.markdown("<h1 class='main-header'>PDF Processing & Excel Update Tool</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
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
                        st.experimental_rerun()
                    else:
                        st.error(f"Login failed: {result}")
                else:
                    st.error("Please enter both username and password")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Registration link
        st.markdown("<h3>New User?</h3>", unsafe_allow_html=True)
        if st.button("Register"):
            navigate_to('register')


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
                        st.experimental_rerun()
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
    # Sidebar
    with st.sidebar:
        st.markdown(f"**Logged in as:** {st.session_state.user_data['username']}")

        # Navigation
        st.markdown("### Navigation")
        if st.button("PDF Processing"):
            st.session_state.current_page = 'main'

        # Admin section
        if st.session_state.user_data.get('is_admin'):
            st.markdown("### Administration")
            if st.button("User Management"):
                st.session_state.current_page = 'admin'

        # Account
        st.markdown("### Account")
        if st.button("Change Password"):
            st.session_state.current_page = 'change_password'

        # Logout
        if st.button("Logout"):
            if st.session_state.user_data and 'session_token' in st.session_state.user_data:
                user_manager.logout_user(st.session_state.user_data['session_token'])
            navigate_to('login')

    # Main content
    st.markdown("<h1 class='main-header'>PDF Processing & Excel Update Tool</h1>", unsafe_allow_html=True)

    st.markdown(
        "<div class='info-box'>This application extracts data from PDF files and updates matching records in an Excel file.</div>",
        unsafe_allow_html=True)

    # File uploader for Excel file
    st.subheader("Step 1: Upload your Excel file")
    excel_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

    # File uploader for PDF files
    st.subheader("Step 2: Upload your PDF files")
    pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    # Process button
    process_disabled = (excel_file is None or not pdf_files)
    if st.button("Process Files", disabled=process_disabled):
        if excel_file and pdf_files:
            log_user_activity("PROCESSING_STARTED", f"Processing {len(pdf_files)} PDF files")

            with st.spinner("Processing files..."):
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

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total PDFs", results['total'])
                    st.metric("Successfully Processed", results['processed'])

                with col2:
                    st.metric("Failed to Process", results['unprocessed'])

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
                        processed_dir = os.path.join(os.path.expanduser("~"), "Downloads", "Processed PDF")
                        if os.path.exists(processed_dir):
                            for file in os.listdir(processed_dir):
                                file_path = os.path.join(processed_dir, file)
                                zip_file.write(file_path, os.path.join("Processed PDF", file))

                        # Add unprocessed PDFs
                        unprocessed_dir = os.path.join(os.path.expanduser("~"), "Downloads", "Unprocessed PDF")
                        if os.path.exists(unprocessed_dir):
                            for file in os.listdir(unprocessed_dir):
                                file_path = os.path.join(unprocessed_dir, file)
                                zip_file.write(file_path, os.path.join("Unprocessed PDF", file))

                    # Offer download of the zip file
                    zip_buffer.seek(0)
                    st.download_button(
                        label="Download Results",
                        data=zip_buffer,
                        file_name="pdf_processing_results.zip",
                        mime="application/zip"
                    )

                # Clean up temporary directory
                shutil.rmtree(temp_dir)


def show_change_password_page():
    """Display the change password page."""
    # Sidebar
    with st.sidebar:
        st.markdown(f"**Logged in as:** {st.session_state.user_data['username']}")

        # Navigation
        st.markdown("### Navigation")
        if st.button("PDF Processing"):
            st.session_state.current_page = 'main'
            st.experimental_rerun()

        # Admin section
        if st.session_state.user_data.get('is_admin'):
            st.markdown("### Administration")
            if st.button("User Management"):
                st.session_state.current_page = 'admin'
                st.experimental_rerun()

        # Logout
        if st.button("Logout"):
            if st.session_state.user_data and 'session_token' in st.session_state.user_data:
                user_manager.logout_user(st.session_state.user_data['session_token'])
            navigate_to('login')
            st.experimental_rerun()

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
            st.experimental_rerun()

        # Account
        st.markdown("### Account")
        if st.button("Change Password"):
            st.session_state.current_page = 'change_password'
            st.experimental_rerun()

        # Logout
        if st.button("Logout"):
            if st.session_state.user_data and 'session_token' in st.session_state.user_data:
                user_manager.logout_user(st.session_state.user_data['session_token'])
            navigate_to('login')
            st.experimental_rerun()

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
                        st.experimental_rerun()
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