import os
import pandas as pd
import numpy as np
import re
import spacy
import joblib
import logging
import requests
import io
import PyPDF2
import pdfplumber
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ml_training.log")
    ]
)
logger = logging.getLogger("ml_model")


class PDFDataExtractor:
    def __init__(self):
        """Initialize the PDF data extractor model."""
        self.nlp = spacy.load("en_core_web_sm")
        self.model = None
        self.fields = []

        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)

    def preprocess_text(self, text):
        """Preprocess text for feature extraction."""
        if not text or not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text)

        # Process with spaCy to keep only important tokens
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]

        return " ".join(tokens)

    def prepare_training_data(self, training_data):
        """
        Prepare training data for model training.

        Args:
            training_data (pd.DataFrame): DataFrame with 'text' column and target field columns

        Returns:
            tuple: (X, y) feature matrix and target matrix
        """
        # Store field names (excluding 'text' and columns we don't want to predict directly)
        self.fields = [col for col in training_data.columns if col not in ['text', 'tds_computed']]

        # Preprocess text
        X = training_data['text'].apply(self.preprocess_text)

        # Prepare target matrix
        y = training_data[self.fields].values

        return X, y

    def train_model(self, training_data, test_size=0.2, random_state=42):
        """
        Train the extraction model.

        Args:
            training_data (pd.DataFrame): DataFrame with 'text' column and target field columns
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility

        Returns:
            float: Test accuracy score
        """
        logger.info("Starting model training")

        # Prepare data
        X, y = self.prepare_training_data(training_data)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Create pipeline with TF-IDF and Random Forest
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=random_state)))
        ])

        # Train the model
        logger.info("Fitting the model")
        pipeline.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test.flatten(), y_pred.flatten())

        logger.info(f"Model training complete. Accuracy: {accuracy:.4f}")

        # Print detailed report
        for i, field in enumerate(self.fields):
            logger.info(f"Field: {field}")
            logger.info(classification_report(y_test[:, i], y_pred[:, i]))

        # Save the model
        self.model = pipeline
        model_path = os.path.join('models', 'pdf_extractor_model.joblib')
        joblib.dump(pipeline, model_path)
        logger.info(f"Model saved to {model_path}")

        return accuracy

    def compute_tds(self, text, amount):
        """
        Compute TDS based on the logic provided.

        Args:
            text (str): The text from the PDF
            amount (float): The receipt amount

        Returns:
            tuple: (tds_value, is_computed)
        """
        try:
            # Check if any of the insurance companies are mentioned in the text
            insurance_companies = [
                "national insurance company limited",
                "united india insurance company limited",
                "the new india assurance co. ltd",
                "oriental insurance co ltd"
            ]

            contains_insurance_company = any(company.lower() in text.lower() for company in insurance_companies)

            # Apply the appropriate calculation
            if contains_insurance_company:
                tds = round(amount * 0.11111111, 2)
                logger.info(f"TDS computed for insurance company: {tds} (11.111111% of {amount})")
            else:
                tds = round(amount * 0.09259259, 2)
                logger.info(f"TDS computed for non-insurance company: {tds} (9.259259% of {amount})")

            return tds, True

        except Exception as e:
            logger.error(f"Error computing TDS: {str(e)}")
            return 0.0, True

    def predict(self, texts, apply_tds_logic=True):
        """
        Predict extraction fields from text and apply TDS computation logic if needed.

        Args:
            texts (list): List of text strings to extract from
            apply_tds_logic (bool): Whether to apply TDS computation logic

        Returns:
            list: List of dictionaries with extracted fields
        """
        if self.model is None:
            try:
                model_path = os.path.join('models', 'pdf_extractor_model.joblib')
                self.model = joblib.load(model_path)
            except:
                logger.error("No model found. Please train the model first.")
                return []

        # Preprocess the text
        processed_texts = [self.preprocess_text(text) for text in texts]

        # Make predictions
        predictions = self.model.predict(processed_texts)

        # Convert to list of dictionaries
        results = []
        for i, pred in enumerate(predictions):
            result = {}
            for j, field in enumerate(self.fields):
                result[field] = pred[j]

            # Apply TDS computation logic if requested
            if apply_tds_logic and 'amount' in result:
                # Check if TDS should be computed
                original_text = texts[i]
                amount_str = result.get('amount', '0')

                try:
                    # Remove currency symbols and commas
                    amount = float(re.sub(r'[^\d.]', '', amount_str)) if amount_str else 0

                    # Check if TDS is missing or seems wrong
                    model_tds = result.get('tds', None)

                    if not model_tds or model_tds == '' or model_tds == '0' or float(
                            re.sub(r'[^\d.]', '', model_tds)) == 0:
                        computed_tds, is_computed = self.compute_tds(original_text, amount)
                        result['tds'] = str(computed_tds)
                        result['tds_computed'] = 'Yes'
                    else:
                        result['tds_computed'] = 'No'
                except Exception as e:
                    logger.error(f"Error in TDS computation during prediction: {str(e)}")

            results.append(result)

        return results

    def extract_table_data(self, text):
        """
        Extract data from tables in the PDF.

        Args:
            text (str): Extracted text from PDF

        Returns:
            list: List of dictionaries, each representing a row of data
        """
        table_data = []

        # Look for table-like structures in the text
        # Common patterns in financial documents include tables with claim numbers, dates, and amounts

        # Pattern 1: Try to find tables with claim/invoice numbers followed by dates and amounts
        table_pattern = r'(?:Claim(?:\s+No\.?|\s*Number|\s*#)|Invoice(?:\s+No\.?|\s*Number|\s*#))\s*.*?\n((?:.*?\d+.*?\n)+)'
        table_matches = re.findall(table_pattern, text, re.IGNORECASE)

        for table_text in table_matches:
            rows = table_text.strip().split('\n')
            for row in rows:
                if not row.strip():
                    continue

                # Try to extract data from each row
                row_data = {}

                # Look for claim/invoice number
                claim_match = re.search(r'([A-Z0-9-_/]+)', row)
                if claim_match:
                    row_data['unique_id'] = claim_match.group(1).strip()

                # Look for date
                date_match = re.search(r'(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{2,4}[-/\.]\d{1,2}[-/\.]\d{1,2})', row)
                if date_match:
                    row_data['date'] = date_match.group(1).strip()

                # Look for amount
                amount_match = re.search(r'(?:Rs\.?|INR)?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)', row)
                if amount_match:
                    row_data['amount'] = amount_match.group(1).strip()

                # Look for TDS
                tds_match = re.search(r'TDS\s*:?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)', row, re.IGNORECASE)
                if tds_match:
                    row_data['tds'] = tds_match.group(1).strip()
                    row_data['tds_computed'] = 'No'

                # Only add the row if we have at least a unique ID and one data point
                if 'unique_id' in row_data and ('date' in row_data or 'amount' in row_data or 'tds' in row_data):
                    table_data.append(row_data)

        # Pattern 2: Try to find table with headers and values
        header_pattern = r'((?:Claim|Invoice).*?(?:Amount|Date).*?(?:TDS|Tax).*?)\n'
        header_match = re.search(header_pattern, text, re.IGNORECASE)

        if header_match:
            headers = header_match.group(1).strip()

            # Try to identify column positions based on headers
            claim_pos = headers.lower().find('claim')
            invoice_pos = headers.lower().find('invoice')
            date_pos = headers.lower().find('date')
            amount_pos = headers.lower().find('amount')
            tds_pos = headers.lower().find('tds')

            # Get the table content after the header
            table_content = text[header_match.end():].strip()

            # Split into rows
            data_rows = table_content.split('\n')

            for row in data_rows:
                if not row.strip() or not re.search(r'\d+', row):
                    continue

                row_data = {}

                # Extract data based on column positions
                if claim_pos >= 0:
                    # Extract claim number
                    claim_match = re.search(r'([A-Z0-9-_/]+)', row)
                    if claim_match:
                        row_data['unique_id'] = claim_match.group(1).strip()

                if invoice_pos >= 0 and invoice_pos != claim_pos:
                    # Extract invoice number
                    invoice_part = row[invoice_pos:].split()[0] if invoice_pos < len(row) else ""
                    if invoice_part:
                        row_data['invoice_no'] = invoice_part.strip()

                if date_pos >= 0:
                    # Extract date
                    date_match = re.search(r'(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{2,4}[-/\.]\d{1,2}[-/\.]\d{1,2})',
                                           row)
                    if date_match:
                        row_data['date'] = date_match.group(1).strip()

                if amount_pos >= 0:
                    # Extract amount
                    amount_match = re.search(r'(?:Rs\.?|INR)?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)', row)
                    if amount_match:
                        row_data['amount'] = amount_match.group(1).strip()

                if tds_pos >= 0:
                    # Extract TDS
                    tds_part = row[tds_pos:].split()[0] if tds_pos < len(row) else ""
                    if tds_part and re.match(r'\d', tds_part):
                        row_data['tds'] = tds_part.strip()
                        row_data['tds_computed'] = 'No'

                # Only add the row if we have at least a unique ID and one data point
                if ('unique_id' in row_data or 'invoice_no' in row_data) and (
                        'date' in row_data or 'amount' in row_data or 'tds' in row_data):
                    table_data.append(row_data)

        # Pattern 3: Look for specific table patterns with a list of claim numbers and amounts
        claim_list_pattern = r'((?:Claim\s+Number|Invoice\s+Number)(?:.*?\n)+(?:\d+.*?\n)+)'
        claim_list_matches = re.findall(claim_list_pattern, text, re.IGNORECASE)

        for claim_list in claim_list_matches:
            lines = claim_list.strip().split('\n')

            # Skip the header line
            for line in lines[1:]:
                if not re.search(r'\d+', line):
                    continue

                row_data = {}

                # Extract claim/invoice number
                claim_match = re.search(r'([A-Z0-9-_/]+)', line)
                if claim_match:
                    row_data['unique_id'] = claim_match.group(1).strip()

                # Extract date
                date_match = re.search(r'(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{2,4}[-/\.]\d{1,2}[-/\.]\d{1,2})', line)
                if date_match:
                    row_data['date'] = date_match.group(1).strip()

                # Extract amount - look for numbers with decimal points as they're likely amounts
                amount_matches = re.findall(r'(\d+(?:,\d{3})*\.\d{2})', line)
                if amount_matches:
                    # First amount is usually the receipt amount
                    if len(amount_matches) >= 1:
                        row_data['amount'] = amount_matches[0].strip()

                    # Second amount is often the TDS
                    if len(amount_matches) >= 2:
                        row_data['tds'] = amount_matches[1].strip()
                        row_data['tds_computed'] = 'No'

                # Only add the row if we have at least a unique ID and one data point
                if 'unique_id' in row_data and ('date' in row_data or 'amount' in row_data or 'tds' in row_data):
                    table_data.append(row_data)

        return table_data


def extract_text_from_pdf_url(url):
    """
    Extract text from a PDF file at the given URL.

    Args:
        url (str): URL to the PDF file

    Returns:
        str: Extracted text
    """
    try:
        # Check if it's a Google Drive URL
        if 'drive.google.com' in url:
            # Convert sharing URL to direct download URL
            file_id = None

            # Handle various Google Drive URL formats
            if '/file/d/' in url:
                file_id = url.split('/file/d/')[1].split('/')[0]
            elif 'id=' in url:
                file_id = url.split('id=')[1].split('&')[0]
            elif '/open?id=' in url:
                file_id = url.split('/open?id=')[1].split('&')[0]

            if file_id:
                # Direct download URL format
                direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            else:
                logger.error(f"Could not parse Google Drive URL: {url}")
                return ""
        else:
            direct_url = url

        # Download the PDF
        response = requests.get(direct_url)
        if response.status_code != 200:
            logger.error(f"Failed to download PDF from URL: {url}, Status code: {response.status_code}")
            return ""

        pdf_content = io.BytesIO(response.content)

        # Extract text using PyPDF2
        text_pypdf2 = ""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_content)
            for page_num in range(len(pdf_reader.pages)):
                text_pypdf2 += pdf_reader.pages[page_num].extract_text() + "\n"
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {str(e)}")

        # Reset the BytesIO position
        pdf_content.seek(0)

        # Extract text using pdfplumber
        text_pdfplumber = ""
        try:
            with pdfplumber.open(pdf_content) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_pdfplumber += page_text + "\n"
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {str(e)}")

        # Combine the results
        combined_text = text_pypdf2 + "\n" + text_pdfplumber

        # Clean up the text
        combined_text = re.sub(r'\s+', ' ', combined_text)
        combined_text = combined_text.strip()

        return combined_text

    except Exception as e:
        logger.error(f"Error extracting text from PDF URL {url}: {str(e)}")
        return ""


def load_training_data_from_excel(excel_path):
    """
    Load training data from an Excel file and extract text from PDF URLs.

    Args:
        excel_path (str): Path to the Excel file

    Returns:
        pd.DataFrame: DataFrame with text and target columns
    """
    try:
        # Read the Excel file
        df = pd.read_excel(excel_path)

        # Check if the required columns exist
        required_columns = ['Document URL', 'Unique Identifier', 'Receipt Amount', 'Receipt Date', 'TDS',
                            'TDS Computed?']

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing columns in Excel file: {missing_columns}")
            return None

        # Create a new DataFrame for the training data
        training_data = {
            'text': [],
            'unique_id': [],
            'invoice_no': [],
            'client_ref': [],
            'amount': [],
            'date': [],
            'tds': [],
            'tds_computed': []
        }

        # Process each row to extract text and target values
        total_rows = len(df)
        for idx, row in df.iterrows():
            logger.info(f"Processing row {idx + 1}/{total_rows}: {row['Document URL']}")

            # Extract text from PDF URL
            pdf_text = extract_text_from_pdf_url(row['Document URL'])

            if not pdf_text:
                logger.warning(f"Could not extract text from PDF: {row['Document URL']}")
                continue

            # Add the data to the training dataset
            training_data['text'].append(pdf_text)
            training_data['unique_id'].append(str(row['Unique Identifier']))

            # Use empty strings for invoice_no and client_ref if not available in Excel
            training_data['invoice_no'].append("")
            training_data['client_ref'].append("")

            training_data['amount'].append(str(row['Receipt Amount']))
            training_data['date'].append(str(row['Receipt Date']))
            training_data['tds'].append(str(row['TDS']))
            training_data['tds_computed'].append(str(row['TDS Computed?']))

            # Also try to extract table data from the PDF
            extractor = PDFDataExtractor()
            table_rows = extractor.extract_table_data(pdf_text)

            if table_rows:
                logger.info(f"Found {len(table_rows)} table rows in PDF: {row['Document URL']}")

                # Add each table row as a separate training example
                for table_row in table_rows:
                    # Use values from table row, falling back to the Excel row for missing values
                    training_data['text'].append(pdf_text)  # Use the same full text
                    training_data['unique_id'].append(table_row.get('unique_id', str(row['Unique Identifier'])))
                    training_data['invoice_no'].append(table_row.get('invoice_no', ""))
                    training_data['client_ref'].append(table_row.get('client_ref', ""))
                    training_data['amount'].append(table_row.get('amount', str(row['Receipt Amount'])))
                    training_data['date'].append(table_row.get('date', str(row['Receipt Date'])))
                    training_data['tds'].append(table_row.get('tds', str(row['TDS'])))
                    training_data['tds_computed'].append(table_row.get('tds_computed', str(row['TDS Computed?'])))

        # Create DataFrame from the collected data
        training_df = pd.DataFrame(training_data)

        # Log the size of the training dataset
        logger.info(f"Created training dataset with {len(training_df)} samples")

        return training_df

    except Exception as e:
        logger.error(f"Error loading training data from Excel: {str(e)}")
        return None


def train_model_with_excel_data(excel_path):
    """
    Train a model using data from an Excel file containing PDF URLs and target values.

    Args:
        excel_path (str): Path to the Excel file

    Returns:
        PDFDataExtractor: Trained extractor model
    """
    # Load training data from Excel
    training_data = load_training_data_from_excel(excel_path)

    if training_data is None or len(training_data) == 0:
        logger.error("Failed to load training data or no valid samples found")
        return None

    # Initialize the extractor
    extractor = PDFDataExtractor()

    # Train the model
    accuracy = extractor.train_model(training_data)

    logger.info(f"Model trained with accuracy: {accuracy:.4f}")

    return extractor


if __name__ == "__main__":
    # Path to your Excel file with training data
    excel_path = "path/to/your/training_data.xlsx"

    # Train the model
    extractor = train_model_with_excel_data(excel_path)

    if extractor:
        # Test with a new example
        test_url = "https://drive.google.com/your-test-pdf-url"
        test_text = extract_text_from_pdf_url(test_url)

        if test_text:
            predictions = extractor.predict([test_text])

            if predictions:
                logger.info("Test Prediction:")
                for field, value in predictions[0].items():
                    logger.info(f"{field}: {value}")
            else:
                logger.error("Prediction failed.")
        else:
            logger.error("Failed to extract text from test PDF.")
    else:
        logger.error("Failed to train the model.")