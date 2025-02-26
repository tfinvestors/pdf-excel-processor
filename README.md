# PDF Processing & Excel Update Platform

A robust platform for extracting data from PDF files and updating Excel records with high accuracy. The application includes user authentication to ensure secure access.

## Features

- User-friendly web interface using Streamlit
- User authentication system with login/registration
- Advanced PDF text extraction using multiple methods
- Machine learning enhancement for improved data extraction accuracy
- Intelligent matching of PDF data to Excel records
- Automated organization of processed and unprocessed PDFs
- Detailed logging and status tracking
- Admin panel for user management

## Installation

### Prerequisites

- Python 3.7 or higher
- Tesseract OCR engine

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/pdf-excel-processor.git
cd pdf-excel-processor
```

### Step 2: Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate the virtual environment:

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Tesseract OCR

**Windows:**
1. Download and install Tesseract from [UB-Mannheim's GitHub repository](https://github.com/UB-Mannheim/tesseract/wiki)
2. Add Tesseract to your PATH or update the path in the code

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install tesseract-ocr
```

### Step 5: Install spaCy language model

```bash
python -m spacy download en_core_web_sm
```

## Usage

### Running the Application

To run the application locally:

```bash
streamlit run streamlit_app.py
```

This will start the application and open it in your default web browser.

### User Authentication

1. **Default Admin Account**
   - On first run, a default admin account is created:
     - Username: `admin`
     - Password: `admin123`
   - ⚠️ **Important**: Change this password immediately after first login

2. **Registration**
   - New users can register by clicking the "Register" button on the login page
   - Fill in the required details: username, email, full name, and password

3. **Login**
   - Existing users can log in with their username/email and password

### Using the Application

1. **Upload Excel File**
   - Click "Browse" next to "Upload Excel File"
   - Select your Excel file containing the data to be updated

2. **Upload PDF Files**
   - Click "Browse" next to "Upload PDF Files"
   - Select multiple PDF files for processing

3. **Process Files**
   - Click the "Process Files" button
   - The application will extract data from each PDF and update the Excel file
   - Progress will be displayed in real-time

4. **Download Results**
   - After processing, click "Download Results" to get a ZIP file containing:
     - Updated Excel file
     - Organized PDFs in "Processed PDF" and "Unprocessed PDF" folders

### Administration

Administrators have access to additional features:

1. **User Management**
   - View all registered users
   - Activate or deactivate user accounts
   - Monitor user activity

2. **Account Settings**
   - All users can change their passwords

## Deployment Options

### Local Network Deployment

To make the application available on your local network:

```bash
streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501
```

Users on the same network can access it via: `http://your-computer-ip:8501`

### Streamlit Cloud Deployment

For public access, you can deploy to Streamlit Cloud:

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect your GitHub repository
4. Follow the deployment instructions

### Heroku Deployment

To deploy to Heroku:

1. Create a `Procfile` with:
   ```
   web: streamlit run streamlit_app.py
   ```

2. Create a `setup.sh` file with:
   ```bash
   mkdir -p ~/.streamlit/
   echo "[server]
   headless = true
   port = $PORT
   enableCORS = false
   " > ~/.streamlit/config.toml
   ```

3. Deploy using the Heroku CLI:
   ```bash
   heroku create
   git push heroku main
   ```

## Security Considerations

1. **Database Security**
   - User credentials are stored with salted password hashing
   - Session tokens are securely generated and managed

2. **Data Privacy**
   - Files are processed locally and not stored permanently
   - All temporary files are cleaned up after processing

3. **Access Control**
   - Role-based permissions (admin vs. regular users)
   - Session management with automatic expiration

4. **Production Deployment**
   - For production environments, consider using HTTPS/SSL
   - Consider adding two-factor authentication for sensitive deployments

## Troubleshooting

### Common Issues

1. **PDF Extraction Issues**
   - Ensure PDFs are text-based rather than scanned images when possible
   - For scanned PDFs, ensure they have good image quality for OCR

2. **Authentication Problems**
   - Clear browser cookies if experiencing login issues
   - Use the password reset functionality if needed

3. **File Upload Limits**
   - By default, Streamlit has a 200MB upload limit
   - For larger files, adjust the settings in `.streamlit/config.toml`

### Logs

Check the log files for detailed information:

- `application.log` - General application logs
- `pdf_processing.log` - PDF extraction logs
- `excel_processing.log` - Excel update logs
- `auth.log` - Authentication and user activity logs

## License

[MIT License](LICENSE)

## Contact

For support or questions, please contact [your-email@example.com](mailto:your-email@example.com).