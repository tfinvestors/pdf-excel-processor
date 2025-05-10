import camelot
import pdfplumber
import time
import logging
import re
from pdf2image import convert_from_path
import pytesseract

logger = logging.getLogger(__name__)


class TableProcessor:
    def extract_form_fields(self, pdf_path, page_num):
        """Extract form-like key-value pairs using region-based detection."""
        try:
            # Convert the page to an image for processing
            images = convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1, dpi=300)
            if not images:
                return None, None

            image = images[0]

            # Convert to grayscale for processing
            import numpy as np
            from PIL import Image
            img_np = np.array(image)

            # Use OpenCV if available, otherwise use PIL
            try:
                import cv2
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

                # Find horizontal and vertical lines (potential table structures)
                horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
                horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)

                vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
                vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)

                # Detect cells/grid intersections
                grid = cv2.add(horizontal_lines, vertical_lines)
                contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Extract potential form fields based on contours
                form_fields = {}
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Extract text from this region using OCR
                    cell_img = gray[y:y + h, x:x + w]
                    cell_text = pytesseract.image_to_string(cell_img, config='--psm 6')

                    # Store by position for later analysis
                    form_fields[(y, x)] = cell_text.strip()

                # Analyze and pair keys with values
                # Simple approach: keys likely on left, values on right
                return self._pair_key_values(form_fields)

            except ImportError:
                # Fallback to simpler method if OpenCV is not available
                logger.warning("OpenCV not available for region-based extraction")
                return None, None

        except Exception as e:
            logger.error(f"Region-based extraction failed: {str(e)}")
            return None, None

    def _pair_key_values(self, form_fields):
        """Pair keys with values from the extracted form fields."""
        if not form_fields or not isinstance(form_fields, dict):
            return {}

        # Sort fields by y-coordinate (row)
        rows = {}
        for (y, x), text in form_fields.items():
            row_key = y // 20  # Group by approximate rows (20px tolerance)
            if row_key not in rows:
                rows[row_key] = []
            rows[row_key].append((x, text))

        # For each row, assume leftmost is key, rightmost is value
        key_value_pairs = {}
        for row, items in rows.items():
            if len(items) >= 2:
                items.sort()  # Sort by x-coordinate
                key = items[0][1]
                value = items[-1][1]
                if key and value:
                    key_value_pairs[key] = value

        return key_value_pairs

    def process_with_camelot(self, pdf_path, page_num):
        """Extract tables using camelot (good for bordered tables)."""
        start_time = time.time()

        try:
            logger.info(f"Processing page {page_num + 1} with Camelot")
            tables = camelot.read_pdf(pdf_path, pages=str(page_num + 1))

            if tables.n == 0:
                logger.info(f"No tables found on page {page_num + 1} with Camelot")
                return None, time.time() - start_time

            table_data = "\n".join([table.df.to_string() for table in tables])
            processing_time = time.time() - start_time

            logger.info(
                f"Extracted {tables.n} tables from page {page_num + 1} with Camelot in {processing_time:.2f} seconds")
            return table_data, processing_time

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Camelot failed on page {page_num + 1}: {str(e)}")
            return None, processing_time

    def process_with_pdfplumber(self, pdf_path, page_num):
        """Extract tables using pdfplumber (good for looser table structures)."""
        start_time = time.time()

        try:
            logger.info(f"Processing page {page_num + 1} with pdfplumber")

            with pdfplumber.open(pdf_path) as pdf:
                if page_num >= len(pdf.pages):
                    logger.warning(f"Page {page_num + 1} out of range for document with {len(pdf.pages)} pages")
                    return None, None, time.time() - start_time

                page = pdf.pages[page_num]
                text = page.extract_text()
                table = page.extract_table()

                if table:
                    for i, row in enumerate(table):
                        for j, cell in enumerate(row):
                            if cell and isinstance(cell, str) and re.match(r'^[\d,.]+$', cell.strip()):
                                # This is a numeric cell, ensure it's properly formatted
                                # Remove any spaces in numbers
                                table[i][j] = re.sub(r'\s+', '', cell)

                table_data = None
                if table:
                    # Format the table, handling None values
                    table_data = "\n".join([
                        " | ".join(cell if cell is not None else "" for cell in row)
                        for row in table if row
                    ])

                processing_time = time.time() - start_time
                logger.info(
                    f"Extracted text and table from page {page_num + 1} with pdfplumber in {processing_time:.2f} seconds")

                return text, table_data, processing_time

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"pdfplumber failed on page {page_num + 1}: {str(e)}")
            return None, None, processing_time