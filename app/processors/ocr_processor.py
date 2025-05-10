import pytesseract
from pdf2image import convert_from_path
import time
import re
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class OCRProcessor:
    def __init__(self):
        # Set Tesseract path from configuration
        pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_PATH

    def recognize_indian_currency(self, text):
        """Improve recognition of Indian Rupee symbol and currency format."""
        # Keep track of original text for comparison
        original_text = text

        # STEP 1: Fix true currency symbols at beginning of amounts
        # These are highly likely to be rupee symbols: %, R, Rs, र, ?, T, F, P, t, r
        text = re.sub(r'(?<!\w)([%RTFPrtr?₨])\s*(\d)', '₹\\2', text)

        # STEP 2: Handle variations of "Rs." before numbers
        text = re.sub(r'(?i)(Rs|RS|R5|F5|P5|FS)[\s\.]*(\d)', '₹\\2', text)

        # STEP 3: Process potential rupee symbol cases with '7' - but only in monetary contexts
        # Collect all potential matches to evaluate with context
        def replace_7_with_rupee(match):
            full_match = match.group(0)
            prefix = match.group(1) or ""

            # Check if this appears to be a monetary amount context
            monetary_context = False

            # Check surrounding words (before match) for monetary indicators
            before_context = original_text[:match.start()].lower().split()[-3:]
            monetary_words = ['amount', 'rs', 'rs.', 'rupee', 'rupees', 'fee', 'price', 'cost', 'payment',
                              'paid', 'pay', 'value', 'charge', 'sum', 'total', 'gst', 'tax']

            # Check if any monetary indicators are in the preceding context
            if any(word in monetary_words for word in before_context):
                monetary_context = True

            # Format checks: Indian currency usually has specific patterns
            # Check for Indian money format (typically has lakhs, crores)
            if re.match(r'7\s*\d{1,2},\d{2},\d{3}', full_match):  # Lakhs format (e.g., 7 1,77,000)
                monetary_context = True

            # Replace only if in monetary context
            if monetary_context:
                return prefix + '₹' + full_match[1 + len(prefix):]
            else:
                return full_match

        # Apply the contextual replacement
        text = re.sub(r'(\s|^|[(])7\s*(\d+,\d+)', replace_7_with_rupee, text)

        # STEP 4: Preserve Indian number format
        text = re.sub(r'(\d)\.(\d\d),', '\\1,\\2,', text)  # Fix 1.77,000 to 1,77,000
        text = re.sub(r'(\d)\.(\d\d)\.(\d)', '\\1,\\2,\\3', text)  # Fix 1.77.000 to 1,77,000

        # STEP 5: Handle specific patterns for monetary amounts in parentheses (like GST)
        # Example: (71,50,000+18%GST) → (₹1,50,000+18%GST)
        text = re.sub(r'\(\s*7(\d+,\d+)', r'(₹\1', text)

        return text

    def clean_ocr_text(self, text):
        """
        Enhanced method to preserve date formats during OCR text cleaning
        while maintaining currency and number corrections
        """
        # Preserve date formats first
        date_patterns = [
            r'\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b',  # DD.MM.YYYY
            r'\b(\d{4})\.(\d{1,2})\.(\d{1,2})\b'  # YYYY.MM.DD
        ]

        placeholders = {}

        # Create unique placeholders for dates
        for idx, pattern in enumerate(date_patterns):
            def replace_date(match):
                key = f"__DATE_PLACEHOLDER_{idx}_{hash(match.group(0))}__"
                placeholders[key] = match.group(0)
                return key

            text = re.sub(pattern, replace_date, text)

        # Regular expression to fix common OCR errors with currency
        # Example: Replace "S 100" with "$100" or "Rs, 100" with "Rs. 100"
        text = re.sub(r'(\$|€|£|Rs|₹)\s+(\d)', r'\1\2', text)

        # Fix common OCR errors with decimal numbers
        # text = re.sub(r'(\d),(\d)', r'\1.\2', text)  # Replace comma with dot in numbers

        # Apply special handling for Indian currency
        text = self.recognize_indian_currency(text)

        # Restore date formats
        for placeholder, original in placeholders.items():
            text = text.replace(placeholder, original)

        return text

    def process_page(self, pdf_path, page_num):
        """Process a single page with OCR."""
        start_time = time.time()

        try:
            logger.info(f"Converting page {page_num + 1} to image for OCR")
            # Use higher DPI for better OCR results
            images = convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1, dpi=300)

            if not images:
                logger.warning(f"Failed to convert page {page_num + 1} to image")
                return None, None, 0, None

            logger.info(f"Running OCR on page {page_num + 1}")
            # custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
            custom_config = r'--oem 3 --psm 1 -l eng --dpi 300'

            # Get detailed OCR data including confidence scores
            ocr_data = pytesseract.image_to_data(images[0], config=custom_config, output_type=pytesseract.Output.DICT)

            # Construct text from the OCR data to ensure alignment with confidence scores
            words = []
            for i in range(len(ocr_data['text'])):
                if ocr_data['conf'][i] > 0:  # Only include words with confidence > 0
                    words.append(ocr_data['text'][i])
                    # Add space after word except at end of line
                    if i + 1 < len(ocr_data['text']) and ocr_data['line_num'][i] == ocr_data['line_num'][i + 1]:
                        words.append(' ')
                    elif i + 1 < len(ocr_data['text']) and ocr_data['line_num'][i] != ocr_data['line_num'][i + 1]:
                        words.append('\n')

            text = ''.join(words)
            text = self.clean_ocr_text(text)

            # Calculate average confidence if available
            confidence_score = None
            if 'conf' in ocr_data and ocr_data['conf']:
                # Filter out -1 values which indicate no confidence available
                valid_scores = [conf for conf in ocr_data['conf'] if conf != -1]
                if valid_scores:
                    confidence_score = sum(valid_scores) / len(valid_scores)

            processing_time = time.time() - start_time
            logger.info(f"OCR completed for page {page_num + 1} in {processing_time:.2f} seconds")

            return text, None, processing_time, confidence_score

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"OCR failed for page {page_num + 1}: {str(e)}")
            return None, None, processing_time, None