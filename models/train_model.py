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
        # Store field names
        self.fields = [col for col in training_data.columns if col != 'text']

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

    def predict(self, texts):
        """
        Predict extraction fields from text.

        Args:
            texts (list): List of text strings to extract from

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
            results.append(result)

        return results


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
        required_columns = ['Document URL', 'Unique Identifier', 'Claim No.', 'Inv. No.',
                            'Receipt Amount', 'Receipt Date', 'TDS']

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing columns in Excel file: {missing_columns}")
            return None

        # Create a new DataFrame for the training data
        training_data = {
            'text': [],
            'unique_id': [],
            'claim_no': [],
            'invoice_no': [],
            'amount': [],
            'date': [],
            'tds': []
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
            training_data['claim_no'].append(str(row['Claim No.']))
            training_data['invoice_no'].append(str(row['Inv. No.']))
            training_data['amount'].append(str(row['Receipt Amount']))
            training_data['date'].append(str(row['Receipt Date']))
            training_data['tds'].append(str(row['TDS']))

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