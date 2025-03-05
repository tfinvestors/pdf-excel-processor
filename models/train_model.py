import os
import pandas as pd
import numpy as np
import re
import spacy
import joblib
import logging
import concurrent.futures
from urllib.parse import urlparse
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from pathlib import Path
import time
import warnings
import tqdm

# Import unified PDF text extractor
from app.utils.pdf_text_extractor import PDFTextExtractor

# Suppress low-level warnings
warnings.filterwarnings("ignore", category=UserWarning)

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

# Known insurance companies for TDS calculation
INSURANCE_COMPANIES = [
    "national insurance company limited",
    "united india insurance company limited",
    "the new india assurance co. ltd",
    "oriental insurance co ltd",
    "hdfc ergo",
    "tata aig",
    "iffco tokio",
    "future generali",
    "reliance general",
    "liberty general",
    "bajaj allianz",
    "universal sompo",
    "cholamandalam"
]


class PDFDataExtractor:
    def __init__(self):
        """Initialize the PDF data extractor model."""
        self.nlp = None
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.info("Downloading spaCy model...")
            import subprocess
            subprocess.call(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        self.model = None
        self.fields = []
        self.grid_search = None
        self.best_params = None

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
        try:
            doc = self.nlp(text[:1000000])  # Limit text length to avoid memory issues
            tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
            return " ".join(tokens)
        except Exception as e:
            logger.warning(f"spaCy processing error: {str(e)}")
            # Fallback to simple tokenization if spaCy fails
            return " ".join([word for word in text.split() if len(word) > 2])

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
        logger.info("Preprocessing text data...")
        X = training_data['text'].apply(self.preprocess_text)

        # Prepare target matrix
        y = training_data[self.fields].values

        return X, y

    def train_model_with_grid_search(self, training_data, test_size=0.2, random_state=42):
        """
        Train the extraction model with hyperparameter tuning using GridSearchCV.

        Args:
            training_data (pd.DataFrame): DataFrame with 'text' column and target field columns
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility

        Returns:
            float: Test accuracy score
        """
        logger.info("Starting model training with hyperparameter tuning...")

        # Prepare data
        X, y = self.prepare_training_data(training_data)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', MultiOutputClassifier(RandomForestClassifier(random_state=random_state)))
        ])

        # Define parameter grid
        param_grid = {
            'tfidf__max_features': [3000, 5000, 7000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'classifier__estimator__n_estimators': [50, 100],
            'classifier__estimator__max_depth': [None, 10, 20],
            'classifier__estimator__min_samples_split': [2, 5]
        }

        # Create GridSearchCV
        logger.info("Starting GridSearchCV (this may take a while)...")
        self.grid_search = GridSearchCV(
            pipeline, param_grid, cv=3, scoring='f1_weighted', verbose=1, n_jobs=-1
        )

        # Fit the grid search
        self.grid_search.fit(X_train, y_train)

        # Get best parameters
        self.best_params = self.grid_search.best_params_
        logger.info(f"Best parameters: {self.best_params}")

        # Save best model
        self.model = self.grid_search.best_estimator_
        best_model_path = os.path.join('models', 'pdf_extractor_model_best.joblib')
        joblib.dump(self.model, best_model_path)
        logger.info(f"Best model saved to {best_model_path}")

        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test.flatten(), y_pred.flatten())
        f1 = f1_score(y_test.flatten(), y_pred.flatten(), average='weighted')

        logger.info(f"Model training complete. Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

        # Print detailed report
        for i, field in enumerate(self.fields):
            logger.info(f"Field: {field}")
            logger.info(classification_report(y_test[:, i], y_pred[:, i]))

        # Also save just the best parameters for reference
        params_path = os.path.join('models', 'best_parameters.joblib')
        joblib.dump(self.best_params, params_path)

        return accuracy

    def train_model(self, training_data, test_size=0.2, random_state=42, use_grid_search=False):
        """
        Train the extraction model.

        Args:
            training_data (pd.DataFrame): DataFrame with 'text' column and target field columns
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            use_grid_search (bool): Whether to use GridSearchCV for hyperparameter tuning

        Returns:
            float: Test accuracy score
        """
        if use_grid_search:
            return self.train_model_with_grid_search(training_data, test_size, random_state)

        logger.info("Starting model training with default parameters...")

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
        logger.info("Fitting the model...")
        pipeline.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test.flatten(), y_pred.flatten())
        f1 = f1_score(y_test.flatten(), y_pred.flatten(), average='weighted')

        logger.info(f"Model training complete. Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

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

    def compute_tds(self, text, amount, insurance_company=None):
        """
        Compute TDS based on the logic provided.

        Args:
            text (str): The text from the PDF
            amount (float): The receipt amount
            insurance_company (str, optional): The insurance company name

        Returns:
            tuple: (tds_value, is_computed)
        """
        try:
            contains_insurance_company = False

            # First check if insurance_company is provided directly
            if insurance_company:
                for known_company in INSURANCE_COMPANIES:
                    if known_company.lower() in insurance_company.lower():
                        contains_insurance_company = True
                        logger.info(f"Insurance company identified from metadata: {insurance_company}")
                        break

            # If not found in metadata, check the text
            if not contains_insurance_company:
                contains_insurance_company = any(company.lower() in text.lower() for company in INSURANCE_COMPANIES)
                if contains_insurance_company:
                    logger.info("Insurance company identified from text content")

            # Apply the appropriate calculation
            if contains_insurance_company:
                # Special handling for New India Assurance with threshold
                if "new india" in text.lower() or (insurance_company and "new india" in insurance_company.lower()):
                    if amount <= 300000:
                        tds = round(amount * 0.09259259, 2)  # 9.259259% for <= 300000
                        logger.info(
                            f"TDS computed for New India Assurance (amount <= 300000): {tds} (9.259259% of {amount})")
                    else:
                        tds = round(amount * 0.11111111, 2)  # 11.111111% for > 300000
                        logger.info(
                            f"TDS computed for New India Assurance (amount > 300000): {tds} (11.111111% of {amount})")
                else:
                    tds = round(amount * 0.11111111, 2)  # 11.111111% for other insurance companies
                    logger.info(f"TDS computed for insurance company: {tds} (11.111111% of {amount})")
            else:
                tds = round(amount * 0.09259259, 2)  # 9.259259% for non-insurance companies
                logger.info(f"TDS computed for non-insurance company: {tds} (9.259259% of {amount})")

            return tds, True

        except Exception as e:
            logger.error(f"Error computing TDS: {str(e)}")
            return 0.0, False

    def predict(self, texts, apply_tds_logic=True, insurance_companies=None):
        """
        Predict extraction fields from text and apply TDS computation logic if needed.

        Args:
            texts (list): List of text strings to extract from
            apply_tds_logic (bool): Whether to apply TDS computation logic
            insurance_companies (list, optional): List of insurance company names

        Returns:
            list: List of dictionaries with extracted fields
        """
        if self.model is None:
            try:
                model_path = os.path.join('models', 'pdf_extractor_model.joblib')
                if os.path.exists(model_path):
                    self.model = joblib.load(model_path)
                else:
                    best_model_path = os.path.join('models', 'pdf_extractor_model_best.joblib')
                    if os.path.exists(best_model_path):
                        self.model = joblib.load(best_model_path)
                    else:
                        logger.error("No model found. Please train the model first.")
                        return []
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                return []

        # Preprocess the text
        logger.info(f"Preprocessing {len(texts)} text samples for prediction...")
        processed_texts = [self.preprocess_text(text) for text in texts]

        # Make predictions
        logger.info("Making predictions...")
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
                insurance_company = None if insurance_companies is None else insurance_companies[i]

                try:
                    # Remove currency symbols and commas
                    amount = float(re.sub(r'[^\d.]', '', amount_str)) if amount_str else 0

                    # Check if TDS is missing or seems wrong
                    model_tds = result.get('tds', None)

                    if not model_tds or model_tds == '' or model_tds == '0' or float(
                            re.sub(r'[^\d.]', '', model_tds)) == 0:
                        computed_tds, is_computed = self.compute_tds(original_text, amount, insurance_company)
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

                # Look for date with expanded formats
                date_match = re.search(
                    r'(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{2,4}[-/\.]\d{1,2}[-/\.]\d{1,2}|\d{1,2}\s+[A-Za-z]{3}\s+\d{4})',
                    row
                )
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
                    # Extract date with expanded formats
                    date_match = re.search(
                        r'(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{2,4}[-/\.]\d{1,2}[-/\.]\d{1,2}|\d{1,2}\s+[A-Za-z]{3}\s+\d{4})',
                        row
                    )
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

                # Extract date with expanded formats
                date_match = re.search(
                    r'(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{2,4}[-/\.]\d{1,2}[-/\.]\d{1,2}|\d{1,2}\s+[A-Za-z]{3}\s+\d{4})',
                    line
                )
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


def load_training_data_from_excel(excel_path, use_ocr=True, threads=4):
    """
    Load training data from an Excel file and extract text from PDF URLs with parallel processing.
    Uses the unified PDFTextExtractor for consistent text extraction.

    Args:
        excel_path (str): Path to the Excel file
        use_ocr (bool): Whether to use OCR for text extraction
        threads (int): Number of threads for parallel processing

    Returns:
        pd.DataFrame: DataFrame with text and target columns
    """
    try:
        # Check if Excel file exists
        if not os.path.exists(excel_path):
            logger.error(f"Excel file not found: {excel_path}")
            return None

        # Read the Excel file
        logger.info(f"Reading Excel file: {excel_path}")
        df = pd.read_excel(excel_path)
        logger.info(f"Excel file contains {len(df)} rows")

        # Check if the required columns exist
        required_columns = ['Document URL', 'Unique Identifier', 'Receipt Amount', 'Receipt Date', 'TDS',
                            'TDS Computed?']

        # Check for Insurance Company Name column (optional but recommended)
        has_insurance_column = 'Insurance Company Name' in df.columns
        if has_insurance_column:
            logger.info("Insurance Company Name column found in Excel")
        else:
            logger.info("Insurance Company Name column not found in Excel (optional)")

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns in Excel file: {missing_columns}")
            return None

        # Create a new DataFrame for the training data
        training_data = {
            'text': [],
            'unique_id': [],
            'invoice_no': [],
            'client_ref': [],
            'insurance_company': [] if has_insurance_column else None,
            'amount': [],
            'date': [],
            'tds': [],
            'tds_computed': []
        }

        # Remove None values from the dictionary
        training_data = {k: v for k, v in training_data.items() if v is not None}

        # Create a single text extractor instance
        pdf_extractor = PDFTextExtractor()

        # Define a function to process a single row for parallel execution
        def process_row(row_data):
            url = row_data['Document URL']
            if not url or not isinstance(url, str):
                logger.warning(f"Invalid URL for row: {row_data}")
                return None

            try:
                # Extract text from PDF URL using unified extractor
                pdf_text = pdf_extractor.extract_from_url(url)

                if not pdf_text or len(pdf_text.strip()) < 50:  # Minimum text length check
                    logger.warning(f"Insufficient text extracted from PDF: {url}")
                    return None

                # Create a result dictionary with extracted text and targets
                result = {
                    'text': pdf_text,
                    'unique_id': str(row_data['Unique Identifier']),
                    'invoice_no': "",  # Not available in Excel
                    'client_ref': "",  # Not available in Excel
                    'amount': str(row_data['Receipt Amount']),
                    'date': str(row_data['Receipt Date']),
                    'tds': str(row_data['TDS']),
                    'tds_computed': str(row_data['TDS Computed?'])
                }

                # Add insurance company if available
                if has_insurance_column:
                    result['insurance_company'] = str(row_data['Insurance Company Name'])

                # Extract table data to generate additional training examples
                extractor = PDFDataExtractor()
                table_rows = extractor.extract_table_data(pdf_text)

                if table_rows:
                    logger.info(f"Found {len(table_rows)} table rows in PDF: {url}")

                    # Return main result and additional table rows
                    additional_results = []
                    for table_row in table_rows:
                        table_result = result.copy()
                        # Update with table values, falling back to the Excel row for missing values
                        table_result['unique_id'] = table_row.get('unique_id', str(row_data['Unique Identifier']))
                        table_result['invoice_no'] = table_row.get('invoice_no', "")
                        table_result['client_ref'] = table_row.get('client_ref', "")
                        table_result['amount'] = table_row.get('amount', str(row_data['Receipt Amount']))
                        table_result['date'] = table_row.get('date', str(row_data['Receipt Date']))
                        table_result['tds'] = table_row.get('tds', str(row_data['TDS']))
                        table_result['tds_computed'] = table_row.get('tds_computed', str(row_data['TDS Computed?']))
                        additional_results.append(table_result)

                    return [result] + additional_results

                return [result]

            except Exception as e:
                logger.error(f"Error processing row with URL {url}: {str(e)}")
                return None

        # Process each row with progress bar
        logger.info(f"Processing {len(df)} rows with {threads} threads")
        all_results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            # Convert DataFrame rows to dictionaries for easier processing
            rows = df.to_dict('records')

            # Process rows with progress tracking
            futures = [executor.submit(process_row, row) for row in rows]

            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    result = future.result()
                    if result:
                        all_results.extend(result)
                    # Print progress every 10%
                    if (i + 1) % max(1, len(df) // 10) == 0 or i + 1 == len(df):
                        logger.info(f"Processed {i + 1}/{len(df)} rows ({(i + 1) / len(df) * 100:.1f}%)")
                except Exception as e:
                    logger.error(f"Error in future: {str(e)}")

        logger.info(f"Extracted {len(all_results)} total samples (including table rows)")

        # Convert to DataFrame
        if not all_results:
            logger.error("No valid samples found")
            return None

        # Create DataFrame from all results
        for result in all_results:
            for key, value in result.items():
                if key in training_data:
                    training_data[key].append(value)

        # Create DataFrame
        training_df = pd.DataFrame(training_data)

        # Remove rows with missing or invalid data
        original_len = len(training_df)
        training_df = training_df.dropna(subset=['text', 'unique_id'])
        if len(training_df) < original_len:
            logger.warning(f"Removed {original_len - len(training_df)} rows with missing text or unique_id")

        # Log the size of the training dataset
        logger.info(f"Created training dataset with {len(training_df)} samples")

        return training_df

    except Exception as e:
        logger.error(f"Error loading training data from Excel: {str(e)}")
        return None


def train_model_with_excel_data(excel_path, use_ocr=True, use_grid_search=False, threads=4):
    """
    Train a model using data from an Excel file containing PDF URLs and target values.

    Args:
        excel_path (str): Path to the Excel file
        use_ocr (bool): Whether to use OCR for text extraction
        use_grid_search (bool): Whether to use GridSearchCV for hyperparameter tuning
        threads (int): Number of threads for parallel processing

    Returns:
        PDFDataExtractor: Trained extractor model
    """
    start_time = time.time()
    logger.info(f"Starting model training with Excel data: {excel_path}")
    logger.info(f"OCR enabled: {use_ocr}, Grid search: {use_grid_search}, Threads: {threads}")

    # Load training data from Excel
    training_data = load_training_data_from_excel(excel_path, use_ocr=use_ocr, threads=threads)

    if training_data is None or len(training_data) == 0:
        logger.error("Failed to load training data or no valid samples found")
        return None

    # Initialize the extractor
    extractor = PDFDataExtractor()

    # Train the model
    logger.info("Training model...")
    accuracy = extractor.train_model(training_data, use_grid_search=use_grid_search)

    elapsed_time = time.time() - start_time
    logger.info(f"Model trained with accuracy: {accuracy:.4f}")
    logger.info(f"Total training time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")

    return extractor


def test_model_on_new_pdf(model, pdf_url, use_ocr=True):
    """
    Test the trained model on a new PDF.

    Args:
        model (PDFDataExtractor): Trained model
        pdf_url (str): URL to the PDF file
        use_ocr (bool): Whether to use OCR for text extraction

    Returns:
        dict: Extraction results
    """
    logger.info(f"Testing model on PDF: {pdf_url}")

    # Create text extractor
    pdf_extractor = PDFTextExtractor()

    # Extract text from the PDF using unified extractor
    text = pdf_extractor.extract_from_url(pdf_url)

    if not text:
        logger.error(f"Failed to extract text from PDF: {pdf_url}")
        return None

    # Make predictions
    predictions = model.predict([text])

    if predictions:
        result = predictions[0]
        logger.info("Extraction results:")
        for field, value in result.items():
            logger.info(f"  {field}: {value}")
        return result
    else:
        logger.error("Prediction failed.")
        return None


if __name__ == "__main__":
    print("=" * 80)
    print(" PDF Data Extraction Model Training")
    print("=" * 80)
    print("\nThis script will train a machine learning model to extract data from PDF files.")
    print("The training data should be provided in an Excel file with the following columns:")
    print("  - Document URL: URL to the PDF file")
    print("  - Unique Identifier: Identifier for the document")
    print("  - Insurance Company Name: Name of the insurance company (optional)")
    print("  - Receipt Amount: Amount to be extracted")
    print("  - Receipt Date: Date to be extracted")
    print("  - TDS: Tax Deducted at Source")
    print("  - TDS Computed?: Whether TDS was computed")
    print("\nThe trained model will be saved to models/pdf_extractor_model.joblib")

    # Get Excel file path
    default_path = "training_data.xlsx"
    excel_path = input(f"\nEnter the path to your Excel file [{default_path}]: ").strip() or default_path

    # Get OCR option
    use_ocr = input("Use OCR for scanned PDFs? (y/n) [y]: ").lower().strip() != 'n'

    # Get grid search option
    use_grid_search = input("Use grid search for hyperparameter tuning? (y/n) [n]: ").lower().strip() == 'y'
    if use_grid_search:
        print("\nWARNING: Grid search can take a very long time, especially with many PDFs.")
        print("You might want to start with a smaller dataset or set use_grid_search=False for initial testing.")

    # Get threads option
    default_threads = os.cpu_count() or 4
    try:
        threads = int(
            input(f"Number of threads for parallel processing [{default_threads}]: ").strip() or default_threads)
    except ValueError:
        threads = default_threads

    print("\nStarting training...")

    # Train the model
    extractor = train_model_with_excel_data(
        excel_path,
        use_ocr=use_ocr,
        use_grid_search=use_grid_search,
        threads=threads
    )

    if extractor:
        print("\nModel training completed successfully!")

        # Ask if user wants to test the model
        test_model = input("\nTest the model on a new PDF? (y/n) [n]: ").lower().strip() == 'y'

        if test_model:
            test_url = input("Enter the URL to a test PDF: ").strip()
            if test_url:
                test_result = test_model_on_new_pdf(extractor, test_url, use_ocr=use_ocr)

                if test_result:
                    print("\nExtraction Results:")
                    for field, value in test_result.items():
                        print(f"  {field}: {value}")
                else:
                    print("\nFailed to extract data from the test PDF.")
            else:
                print("No test URL provided.")
    else:
        print("\nModel training failed. Check the logs for details.")