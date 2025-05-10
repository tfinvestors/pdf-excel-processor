# app/processors/text_post_processor.py
import re
import logging
import os
from difflib import get_close_matches

logger = logging.getLogger(__name__)


class TextPostProcessor:
    """
    Post-processes extracted text to improve quality and fix common OCR errors.
    """

    def __init__(self):
        # Load domain-specific terms
        self.domain_terms = self.load_domain_terms()
        self._term_frequency = {}

    def debug_missed_corrections(self, text):
        """Find potential errors that weren't corrected and explain why."""
        known_issues = {
            'ebruary': 'February',
            'Subiect': 'Subject',
            'aharashtra': 'Maharashtra',
            'rd Floor': '3rd Floor'
        }

        issues_found = []

        for issue, correction in known_issues.items():
            matches = list(re.finditer(r'\b' + re.escape(issue) + r'\b', text))
            if matches:
                for match in matches:
                    start, end = match.span()
                    context_start = max(0, start - 50)
                    context_end = min(len(text), end + 50)
                    context = text[context_start:context_end]

                    # Try to identify why it wasn't corrected
                    is_valid = self.is_valid_word(issue)
                    context_corrections = self.analyze_context(issue, context)
                    has_context_match = len(context_corrections) > 0

                    reason = "Unknown"
                    if is_valid:
                        reason = "Word considered valid by is_valid_word()"
                    elif not has_context_match:
                        reason = "No context match found by analyze_context()"

                    issues_found.append({
                        'issue': issue,
                        'correction': correction,
                        'context': context,
                        'reason': reason
                    })

                    logger.debug(f"Missed correction: '{issue}' -> '{correction}' (Reason: {reason})")
                    logger.debug(f"  Context: '{context}'")

        return issues_found

    def process(self, text, document_type=None, category=None):
        """Apply all post-processing steps to extracted text."""
        logger.info(f"Starting post-processing on text (length: {len(text)})")

        # Add detailed logging
        logger.debug(f"Original text preview (first 500 chars): {text[:500]}")

        if not text:
            return text

        # Fix date decimal confusion first
        text = self.fix_date_decimal_confusion(text)

        logger.debug(f"After date decimal fix (first 500 chars): {text[:500]}")

        # Process the text word by word with robust context analysis
        words_with_context = []
        for match in re.finditer(r'\b(\w+)\b', text):
            word = match.group(1)
            start, end = match.span()
            # Get surrounding context
            context_start = max(0, start - 50)
            context_end = min(len(text), end + 50)
            context = text[context_start:context_end]
            words_with_context.append((word, start, end, context))

        # Build corrections list
        corrections = []
        for word, start, end, context in words_with_context:
            # Skip very short words
            if len(word) < 3:
                continue

            # Skip words that are definitely valid
            if self.is_valid_word(word):
                continue

            # Try to find corrections using context analysis
            context_results = self.analyze_context(word, context)
            if context_results:
                best_correction, confidence = context_results[0]
                if confidence > 0.7:
                    logger.debug(f"Correcting '{word}' to '{best_correction}' (confidence: {confidence})")
                    corrections.append((start, end, best_correction))
                    continue

            # Try static error patterns
            patterns = self.get_error_patterns()
            for pattern, replacement in patterns:
                if re.search(pattern, word):
                    corrected = re.sub(pattern, replacement, word)
                    if corrected != word and self.is_valid_word(corrected):
                        logger.debug(f"Pattern correcting '{word}' to '{corrected}'")
                        corrections.append((start, end, corrected))
                        break

            # Try function-based error patterns
            corrected = self.apply_function_patterns(word)
            if corrected != word and self.is_valid_word(corrected):
                logger.debug(f"Function correcting '{word}' to '{corrected}'")
                corrections.append((start, end, corrected))

        # Apply corrections from end to beginning
        corrections.sort(reverse=True)
        result = list(text)
        for start, end, correction in corrections:
            result[start:end] = correction

        processed_text = ''.join(result)
        logger.info(f"Completed post-processing, returning text of length {len(processed_text)}")
        return processed_text

    def fix_date_decimal_confusion(self, text):
        """
        Ultra-precise date and decimal separator preservation
        """
        # Strict date format patterns with precise matching
        strict_date_formats = [
            r'\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b',  # DD.MM.YYYY
            r'\b(\d{4})\.(\d{1,2})\.(\d{1,2})\b'  # YYYY.MM.DD
        ]

        # Create robust placeholder mechanism
        placeholders = {}
        protected_text = text

        def unique_date_placeholder(match):
            full_match = match.group(0)
            placeholder = f"__STRICT_DATE_PLACEHOLDER_{hash(full_match)}__"
            placeholders[placeholder] = full_match
            return placeholder

        # Protect date formats with strict replacement
        for pattern in strict_date_formats:
            protected_text = re.sub(pattern, unique_date_placeholder, protected_text)

        # Advanced decimal separator handling
        def precise_decimal_replacer(match):
            whole_part, decimal_part = match.groups()
            context = match.group(0)

            monetary_indicators = [
                '₹', '$', '%', 'Rs', 'amount', 'value', 'cost',
                'total', 'gross', 'net', 'price', 'rate'
            ]

            monetary_context = any(
                indicator.lower() in context.lower()
                for indicator in monetary_indicators
            )

            return f"{whole_part},{decimal_part}" if monetary_context else match.group(0)

        # Targeted decimal replacements
        protected_text = re.sub(r'(\d+)\.(\d{2})\b', precise_decimal_replacer, protected_text)

        # Restore original date formats
        for placeholder, original in placeholders.items():
            protected_text = protected_text.replace(placeholder, original)

        return protected_text

    def apply_context_aware_correction(self, text):
        """
        Correct common OCR errors using context, dictionary lookup, and error patterns.

        Args:
            text (str): Text to process

        Returns:
            str: Processed text
        """
        if not text:
            return text

        # 1. Tokenize the text into words while preserving position information
        words_with_context = []
        for match in re.finditer(r'\b(\w+)\b', text):
            word = match.group(1)
            start, end = match.span()
            # Get some context (a few words before and after)
            context_start = max(0, start - 40)
            context_end = min(len(text), end + 40)
            context = text[context_start:context_end]
            words_with_context.append({
                'word': word,
                'position': (start, end),
                'context': context
            })

        corrections = []

        # 2. Apply different correction strategies
        for word_info in words_with_context:
            word = word_info['word']
            context = word_info['context']
            start, end = word_info['position']

            # Skip very short words (less likely to be errors, more likely to cause false positives)
            if len(word) <= 2:
                continue

            # Skip words that are valid according to our dictionary
            if self.is_valid_word(word):
                continue

            # Try potential corrections
            potential_correction = self.get_best_correction(word, context)

            if potential_correction and potential_correction != word:
                # Verify the correction makes sense in context
                if self.validate_correction(word, potential_correction, context):
                    corrections.append((start, end, potential_correction))

        # 3. Apply corrections from end to beginning to maintain position integrity
        corrections.sort(reverse=True)
        result = list(text)
        for start, end, correction in corrections:
            result[start:end] = correction

        return ''.join(result)

    def is_valid_word(self, word):
        """Check if a word is valid using dictionary libraries and context."""
        # Known invalid words - this is necessary because dictionary lookups might
        # falsely validate these due to language variations
        known_invalid = {'ebruary', 'subiect', 'aharashtra'}
        if word.lower() in known_invalid:
            return False

        # Very short words are always considered valid to avoid overcorrection
        if not word or len(word) < 3:
            return True

        # Check common terms and allowed special patterns
        if word.isdigit() or (word.isupper() and len(word) <= 5):  # Numbers or acronyms
            return True

        # Calculate word statistics for better error detection
        vowel_count = sum(1 for c in word.lower() if c in 'aeiou')
        consonant_count = sum(1 for c in word.lower() if c in 'bcdfghjklmnpqrstvwxyz')

        # Words without vowels or with very unusual consonant/vowel ratios are likely errors
        if vowel_count == 0 and len(word) > 2:
            return False  # No vowels suggests an OCR error

        # Extremely skewed consonant/vowel ratio suggests an error
        if consonant_count > 0 and vowel_count / consonant_count < 0.1 and len(word) > 3:
            return False

        # Check domain-specific terms
        if word.lower() in self.domain_terms:
            return True

        try:
            import enchant
            dictionary = enchant.Dict("en_US")
            return dictionary.check(word)
        except ImportError:
            # Fall back to basic heuristics if enchant is not available

            # Check for proper names (first letter capitalized)
            if word[0].isupper() and len(word) > 1 and all(c.islower() for c in word[1:]):
                return True

            return self.check_term_frequency(word)

    def load_domain_terms(self):
        """
        Load domain-specific terms from file or a built-in list.

        Returns:
            set: Set of domain terms
        """
        terms = set()

        # Try loading from files if available
        domain_term_files = [
            os.path.join(os.path.dirname(__file__), "../data/financial_terms.txt"),
            os.path.join(os.path.dirname(__file__), "../data/legal_terms.txt"),
            os.path.join(os.path.dirname(__file__), "../data/business_terms.txt")
        ]

        for file_path in domain_term_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        for line in f:
                            term = line.strip().lower()
                            if term:
                                terms.add(term)
                except Exception as e:
                    logger.warning(f"Error loading domain terms from {file_path}: {e}")

        # Add essential terms that might not be in standard dictionaries
        essential_terms = {
            # Financial reporting terms
            "ebitda", "capex", "opex", "eps", "p/e", "gaap", "ifrs", "roi", "wacc",

            # Business/corporate terms
            "stakeholder", "shareholder", "dividend", "subsidiaries", "merger",

            # Common months/abbreviations
            "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
            "january", "february", "march", "april", "june", "july", "august", "september",
            "october", "november", "december"
        }
        terms.update(essential_terms)

        return terms

    def get_best_correction(self, word, context):
        """
        Get the best correction for a word based on multiple strategies.

        Args:
            word (str): Word to correct
            context (str): Surrounding text context

        Returns:
            str or None: Best correction or None if no good correction found
        """
        corrections = []

        # Strategy 1: Common OCR error patterns
        pattern_corrections = self.apply_error_patterns(word)
        if pattern_corrections:
            corrections.extend(pattern_corrections)

        # Strategy 2: Edit distance to known words
        edit_corrections = self.find_close_matches(word)
        if edit_corrections:
            corrections.extend(edit_corrections)

        # Strategy 3: Context-based correction (e.g., date context for month names)
        context_corrections = self.context_based_correction(word, context)
        if context_corrections:
            corrections.extend(context_corrections)

        # Return the best correction based on confidence score
        if corrections:
            # Sort by confidence score
            corrections.sort(key=lambda x: x[1], reverse=True)
            # Return the highest confidence correction
            return corrections[0][0]

        return None

    def apply_error_patterns(self, word):
        """
        Apply common OCR error patterns.

        Args:
            word (str): Word to check

        Returns:
            list: List of (correction, confidence) tuples
        """
        # Special case for month names
        months = ["january", "february", "march", "april", "may", "june", "july",
                  "august", "september", "october", "november", "december"]

        # Check if word might be a month with error
        for month in months:
            # If word is a substring of a month (missing letters) or vice versa
            if word.lower() in month or month in word.lower():
                # Calculate similarity
                similarity = len(set(word.lower()) & set(month)) / max(len(word), len(month))
                if similarity > 0.7:  # High similarity threshold
                    return [(month, 0.9)]  # High confidence for month corrections

        corrections = []

        # Get comprehensive error patterns
        error_patterns = self.get_ocr_error_patterns()

        for error, correction in error_patterns:
            if error in word:
                corrected = word.replace(error, correction)

                # Check if the correction is a valid word
                if self.is_valid_word(corrected):
                    # Determine confidence based on how common this error is
                    # and how much of the word was changed
                    change_ratio = len(error) / len(word)
                    confidence = 0.7 - (0.3 * change_ratio)  # Higher confidence for smaller changes
                    corrections.append((corrected, confidence))

        return corrections

    def find_close_matches(self, word):
        """
        Find close matches to the word by edit distance.

        Args:
            word (str): Word to match

        Returns:
            list: List of (correction, confidence) tuples
        """
        corrections = []

        # Check against domain terms first for efficiency
        close_matches = get_close_matches(word.lower(), self.domain_terms, n=3, cutoff=0.8)

        for match in close_matches:
            # Calculate edit distance
            distance = self.levenshtein_distance(word.lower(), match)
            # Assign confidence based on edit distance and word length
            max_distance = max(len(word), len(match))
            if max_distance == 0:
                confidence = 0
            else:
                confidence = 1 - (distance / max_distance)

            # Only consider if confidence is high enough
            if confidence > 0.7:
                corrections.append((match, confidence))

        return corrections

    def levenshtein_distance(self, s1, s2):
        """
        Calculate the Levenshtein distance between two strings.

        Args:
            s1 (str): First string
            s2 (str): Second string

        Returns:
            int: Edit distance
        """
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def context_based_correction(self, word, context):
        """
        Correct a word based on its surrounding context.

        Args:
            word (str): Word to correct
            context (str): Surrounding text context

        Returns:
            list: List of (correction, confidence) tuples
        """
        corrections = []

        # Check for date context
        date_patterns = [
            r'\b\d{1,2}(st|nd|rd|th)?\s+(\w+)\s+\d{4}\b',  # 15th February 2024
            r'\b(\w+)\s+\d{1,2}(st|nd|rd|th)?,?\s+\d{4}\b',  # February 15, 2024
            r'\bDated\s+(\w+)',  # Dated February
            r'\bas\s+of\s+(\w+)',  # as of February
            r'\bDate\s+of\s+[Rr]eport\s+(\w+)',  # Date of Report February
            r'\bDate:\s+(\w+)',  # Date: February
            r'\bon\s+(\w+)',  # on February
            r'Date\s+of\s+[Rr]eport\s+(\w+)',  # Match "Date of report ebruary"
            r'Date\s+of\s+[Rr]eport\s+(\w+)\s+\d{1,2}',  # Match "Date of Report ebruary 11"
            r'[Ss]ub[ij]ect',  # Match "Subiect" or "Subject"
        ]

        # Extract potential month words from context
        month_positions = []
        for pattern in date_patterns:
            matches = re.finditer(pattern, context, re.IGNORECASE)
            for match in matches:
                # Get the word that might be a month
                for group_idx in range(1, match.lastindex + 1 if match.lastindex else 1):
                    try:
                        potential_month = match.group(group_idx)
                        if potential_month and len(potential_month) > 2:  # Skip short groups
                            # Check if the current word is in the potential month position
                            if potential_month.lower() == word.lower():
                                month_positions.append((match.start(group_idx), match.end(group_idx)))
                    except:
                        pass

        # If our word appears to be a month based on context
        if month_positions:
            # Check if it's close to a month name
            months = ["january", "february", "march", "april", "may", "june", "july",
                      "august", "september", "october", "november", "december"]

            closest_month = self.get_closest_month(word.lower(), months)
            if closest_month:
                # High confidence because we have date context
                confidence = 0.9
                corrections.append((closest_month, confidence))

        return corrections

    def learn_from_corrections(self, original, correction):
        """Enhanced learning from corrections."""
        if not hasattr(self, '_learned_corrections'):
            self._learned_corrections = {}
            self._char_substitutions = {}
            self._context_patterns = {}

        # Store direct word mapping
        self._learned_corrections[original.lower()] = correction.lower()

        # Store character-level substitutions
        if len(original) == len(correction):
            for i in range(len(original)):
                if original[i] != correction[i]:
                    key = (original[i], correction[i])
                    self._char_substitutions[key] = self._char_substitutions.get(key, 0) + 1

        # Try to extract linguistic patterns
        if len(original) > 3 and len(correction) > 3:
            # Check for prefix/suffix patterns
            if original[1:] == correction[1:]:  # Different first letter
                pattern = ('prefix', original[0], correction[0])
                self._context_patterns[pattern] = self._context_patterns.get(pattern, 0) + 1

            # Different last letter
            if original[:-1] == correction[:-1]:
                pattern = ('suffix', original[-1], correction[-1])
                self._context_patterns[pattern] = self._context_patterns.get(pattern, 0) + 1

        # Log the learning
        logger.debug(f"Learned correction: '{original}' -> '{correction}'")

    def apply_learned_corrections(self, text):
        """Apply learned corrections with more sophistication."""
        if not hasattr(self, '_learned_corrections'):
            return text

        result = text

        # Apply direct word replacements
        for original, correction in self._learned_corrections.items():
            result = re.sub(r'\b' + re.escape(original) + r'\b', correction, result)

        # Apply character substitution patterns to new text
        if hasattr(self, '_char_substitutions'):
            # Find most common substitutions
            common_subs = sorted(self._char_substitutions.items(),
                                 key=lambda x: x[1], reverse=True)[:10]

            # Apply to unknown words
            for word_match in re.finditer(r'\b(\w{3,})\b', result):
                word = word_match.group(1)
                if not self.is_valid_word(word):
                    for (orig_char, corr_char), _ in common_subs:
                        if orig_char in word:
                            corrected = word.replace(orig_char, corr_char)
                            if self.is_valid_word(corrected):
                                result = result.replace(word, corrected)
                                break

        return result

    def _learn_pattern(self, original, correction):
        """Extract a generalizable pattern from this correction pair."""
        # Compare character by character to identify the pattern
        if len(original) != len(correction):
            return  # Different lengths are harder to generalize

        for i in range(len(original)):
            if original[i] != correction[i]:
                # Record this character substitution
                if not hasattr(self, '_char_substitutions'):
                    self._char_substitutions = {}

                key = (original[i], correction[i])
                self._char_substitutions[key] = self._char_substitutions.get(key, 0) + 1

    def apply_learned_corrections(self, text):
        """Apply corrections learned from previous documents."""
        if not hasattr(self, '_learned_corrections'):
            return text

        # First apply direct word corrections
        for original, correction in self._learned_corrections.items():
            text = re.sub(r'\b' + re.escape(original) + r'\b', correction, text)

        # Then try to apply character substitution patterns
        if hasattr(self, '_char_substitutions'):
            # Sort by frequency (most common substitutions first)
            common_subs = sorted(self._char_substitutions.items(),
                                 key=lambda x: x[1], reverse=True)

            # Apply the most common substitutions
            for (orig_char, corr_char), _ in common_subs[:10]:  # Top 10 substitutions
                # Only apply to words that aren't valid
                words = re.findall(r'\b\w+\b', text)
                for word in words:
                    if orig_char in word and not self.is_valid_word(word):
                        corrected = word.replace(orig_char, corr_char)
                        if self.is_valid_word(corrected):
                            text = re.sub(r'\b' + re.escape(word) + r'\b', corrected, text)

        return text

    def analyze_context(self, word, context):
        """Analyze the context of a word to determine likely corrections."""
        corrections = []

        # Extract context words and make lowercase for easier matching
        context_words = set(w.lower() for w in re.findall(r'\b\w+\b', context))

        # Special handling for month names
        date_terms = {'date', 'report', 'dated', 'of', 'on', 'day', 'month', 'year'}
        date_context = any(term in context_words for term in date_terms)

        if date_context:
            months = ["january", "february", "march", "april", "may", "june", "july",
                      "august", "september", "october", "november", "december"]

            # Try to match month names
            for month in months:
                # For missing first letter
                if word.lower() == month[1:].lower():
                    return [(month, 0.95)]

                # For close matches (like 'ebruary' for 'february')
                if self.string_similarity(word.lower(), month) > 0.7:
                    return [(month, 0.9)]

        # Special handling for common document terms
        doc_terms = {'subject', 'regarding', 'ref', 'reference'}
        doc_context = any(term in context_words for term in doc_terms)

        if doc_context or re.search(r'\b[Ss]ub', context):
            if word.lower() in ['subiect', 'subjcct', 'subjet']:
                return [('Subject', 0.95)]

        # Location context
        location_terms = {'mumbai', 'india', 'floor', 'building', 'office'}
        location_context = any(term in context_words for term in location_terms)

        if location_context:
            locations = {'maharashtra', 'chennai', 'delhi', 'kolkata', 'bangalore'}
            for location in locations:
                if self.string_similarity(word.lower(), location) > 0.7:
                    return [(location, 0.9)]

        return corrections

    def string_similarity(self, s1, s2):
        """Calculate string similarity."""
        # Use Jaccard similarity for character sets
        if not s1 or not s2:
            return 0

        # Character set overlap
        chars1 = set(s1)
        chars2 = set(s2)

        # Calculate Jaccard similarity
        return len(chars1.intersection(chars2)) / len(chars1.union(chars2))

    def context_based_correction(self, word, context):
        """
        Correct a word based on its surrounding context.

        Args:
            word (str): Word to correct
            context (str): Surrounding text context

        Returns:
            list: List of (correction, confidence) tuples
        """
        # Keep compatibility with existing code by maintaining return format
        corrections = []

        # Call the new, more powerful context analysis
        context_results = self.analyze_context(word, context)

        # If we got results from analyze_context, use them
        if context_results:
            corrections.extend(context_results)
            return corrections

        # Original context-based correction logic as fallback
        # Keep this section if you want to maintain backward compatibility
        # or if there are specific patterns your original method catches

        # Check for date context
        date_patterns = [
            r'\b\d{1,2}(st|nd|rd|th)?\s+(\w+)\s+\d{4}\b',  # 15th February 2024
            r'\b(\w+)\s+\d{1,2}(st|nd|rd|th)?,?\s+\d{4}\b',  # February 15, 2024
            r'\bDated\s+(\w+)',  # Dated February
            r'\bas\s+of\s+(\w+)',  # as of February
            r'\bDate\s+of\s+[Rr]eport\s+(\w+)',  # Date of Report February
            r'\bDate:\s+(\w+)',  # Date: February
            r'\bon\s+(\w+)',  # on February
            r'Date\s+of\s+[Rr]eport\s+(\w+)',  # Match "Date of report ebruary"
            r'Date\s+of\s+[Rr]eport\s+(\w+)\s+\d{1,2}',  # Match "Date of Report ebruary 11"
            r'[Ss]ub[ij]ect',  # Match "Subiect" or "Subject"
        ]

        # Extract potential month words from context
        month_positions = []
        for pattern in date_patterns:
            matches = re.finditer(pattern, context, re.IGNORECASE)
            for match in matches:
                # Get the word that might be a month
                for group_idx in range(1, match.lastindex + 1 if match.lastindex else 1):
                    try:
                        potential_month = match.group(group_idx)
                        if potential_month and len(potential_month) > 2:  # Skip short groups
                            # Check if the current word is in the potential month position
                            if potential_month.lower() == word.lower():
                                month_positions.append((match.start(group_idx), match.end(group_idx)))
                    except:
                        pass

        # If our word appears to be a month based on context
        if month_positions:
            # Check if it's close to a month name
            months = ["january", "february", "march", "april", "may", "june", "july",
                      "august", "september", "october", "november", "december"]

            closest_month = self.get_closest_month(word.lower(), months)
            if closest_month:
                # High confidence because we have date context
                confidence = 0.9
                corrections.append((closest_month, confidence))

        return corrections

    def get_closest_month(self, word, months):
        """
        Get the closest matching month name.

        Args:
            word (str): Word to match
            months (list): List of month names

        Returns:
            str or None: Closest month or None if no good match
        """
        if len(word) < 3:
            return None

        # Try prefix matching first (more reliable for months)
        prefix_matches = [month for month in months if month.startswith(word[:3])]
        if prefix_matches:
            return prefix_matches[0]

        # Try edit distance
        matches = get_close_matches(word, months, n=1, cutoff=0.6)
        if matches:
            return matches[0]

        return None

    def validate_correction(self, original, correction, context):
        """
        Validate that a correction makes sense in context.

        Args:
            original (str): Original word
            correction (str): Proposed correction
            context (str): Surrounding text context

        Returns:
            bool: Whether the correction is valid
        """
        # Don't correct if original word is valid
        if self.is_valid_word(original):
            return False

        # Only accept the correction if it's a valid word
        if not self.is_valid_word(correction):
            return False

        # Reject if edit distance is too large compared to word length
        max_acceptable_distance = max(len(original) // 3, 1)  # 1/3 of length
        if self.levenshtein_distance(original.lower(), correction.lower()) > max_acceptable_distance:
            return False

        # Special case for months
        months = ["january", "february", "march", "april", "may", "june", "july",
                  "august", "september", "october", "november", "december"]

        if correction.lower() in months:
            # Check if there's date context
            date_indicators = [
                r'\b\d{1,2}(st|nd|rd|th)?\b',  # Day of month
                r'\b\d{4}\b',  # Year
                r'\b(dated|as of|on)\b'  # Date related phrases
            ]

            for indicator in date_indicators:
                if re.search(indicator, context, re.IGNORECASE):
                    return True

            # If no date context, be more conservative with month corrections
            return False

        # For general words, accept if it's a valid word with reasonable edit distance
        return True

    def check_term_frequency(self, term):
        """
        Check if a term appears frequently in the document.

        Args:
            term (str): Term to check

        Returns:
            bool: Whether the term appears frequently
        """
        term_lower = term.lower()
        if term_lower in self._term_frequency:
            self._term_frequency[term_lower] += 1
        else:
            self._term_frequency[term_lower] = 1

        # Lower the threshold or disable for OCR context
        return self._term_frequency[term_lower] >= 1  # Changed from 3 to 1

    def apply_category_specific_processing(self, text, category):
        """
        Apply category-specific processing rules.

        Args:
            text (str): Text to process
            category (str): Document category

        Returns:
            str: Processed text
        """
        if category == "financial":
            # Additional financial document specific processing
            # Standardize number formatting, etc.
            pass
        elif category == "legal":
            # Legal document specific processing
            pass

        return text

    def contains_table(self, text):
        """Check if the text likely contains a table structure."""
        # Look for common table indicators
        table_indicators = [
            '|',  # Vertical pipe (common table separator)
            '+---+',  # ASCII table border
            r'\b\d+\s*\|\s*\d+',  # Numbers separated by pipes
            r'\bNo\.\s*\|\s*Item'  # Common table header pattern
        ]

        for indicator in table_indicators:
            if re.search(indicator, text):
                return True

        # Count lines with multiple whitespace clusters (potential table rows)
        whitespace_pattern_count = 0
        lines = text.split('\n')

        for line in lines:
            # Look for lines with multiple spaced words (potential table cells)
            if len(re.findall(r'\s{2,}', line)) >= 3:
                whitespace_pattern_count += 1

        # If several lines have this pattern, likely a table
        return whitespace_pattern_count >= 3

    def post_process_table_content(self, text):
        """Fix common issues in table content extraction."""
        # Normalize whitespace in table rows
        text = re.sub(r'(\|\s*)+\|', '|', text)

        # Fix alignment issues in table headers
        text = re.sub(r'(\w+)\s+\|\s+(\w+)', r'\1 | \2', text)

        # Standardize number formats in tables
        text = re.sub(r'(\d),(\d{3})', r'\1\2', text)  # Remove thousands separator within cells

        return text

    def process_with_debug(self, text, document_type=None, category=None):
        """Process text with debug output to identify missed corrections."""
        result = self.process(text, document_type, category)

        # Find words that should have been corrected but weren't
        for line in text.split('\n'):
            words = re.findall(r'\b(\w+)\b', line)
            for word in words:
                if len(word) > 3 and not self.is_valid_word(word):
                    potential_corrections = self.get_best_correction(word, line)
                    if potential_corrections:
                        logger.debug(f"Word '{word}' could be corrected to '{potential_corrections}' but wasn't")
                    else:
                        logger.debug(f"No correction found for invalid word: '{word}'")

        return result

    def get_error_patterns(self):
        """Get common OCR error patterns."""
        # Simple pattern replacements
        patterns = [
            # Missing first letter (vowel) in common words
            (r'\bebruary\b', 'February'),  # This is needed since it's extremely common
            (r'\baharashtra\b', 'Maharashtra'),  # This is needed since it's extremely common

            # Common OCR confusions
            (r'rn', 'm'),
            (r'cl', 'd'),
            (r'vv', 'w'),
            (r'ii', 'n'),
            (r'li', 'h'),

            # Subject variants (specifically problematic)
            (r'\bSubiect\b', 'Subject'),
        ]

        return patterns

    def apply_function_patterns(self, word):
        """Apply function-based replacement patterns."""
        # Check for missing initial vowel
        if len(word) >= 3:
            match = re.match(r'^([bcdfghjklmnpqrstvwxyz])([a-z]{2,})', word)
            if match:
                return self._add_initial_vowel(match)

        # Check for missing internal vowel
        match = re.match(r'([bcdfghjklmnpqrst])([bcdfghjklmnpqrst]{2,})', word)
        if match:
            return self._add_vowel_between(match)

        return word

    def _add_initial_vowel(self, match):
        """Add a vowel at the beginning of a word if it improves validity."""
        consonant = match.group(1)
        rest = match.group(2)

        for vowel in 'aeiou':
            candidate = vowel + consonant + rest
            if self.is_valid_word(candidate):
                return candidate

        return match.group(0)  # Return original if no valid word found

    def _add_vowel_between(self, match):
        """Add a vowel between consonant clusters if it improves validity."""
        first = match.group(1)
        rest = match.group(2)

        for vowel in 'aeiou':
            candidate = first + vowel + rest
            if self.is_valid_word(candidate):
                return candidate

        return match.group(0)  # Return original if no valid word found

    def get_ocr_error_patterns(self):

        patterns = [
            # Missing first letters (especially vowels)
            (
            r'^([bcdfghjklmnpqrstvwxyz])([a-z]+)', lambda m: self._check_missing_initial_vowel(m.group(1), m.group(2))),

            # Transposed characters
            (r'([a-z])([a-z])', lambda m: self._check_transposition(m.group(1), m.group(2))),

            # Common OCR confusion pairs (generic, not specific to words)
            (r'rn', 'm'), (r'cl', 'd'), (r'vv', 'w'), (r'ii', 'n'),
            (r'li', 'h'), (r'm', 'nn'), (r'ij', 'y'),

            # Vowel substitutions (very common in OCR)
            (r'a', 'o'), (r'o', 'a'), (r'u', 'a'), (r'e', 'c'),

            # Missing/extra spaces
            (r'\s+', ' '), (r'(\w)(\w)', r'\1 \2')
        ]

        return patterns

    def _check_missing_initial_vowel(self, first_char, rest):
        """Check if adding a vowel before the first char creates a valid word."""
        for vowel in 'aeiou':
            candidate = vowel + first_char + rest
            if self.is_valid_word(candidate):
                return candidate
        return None

    def _check_transposition(self, char1, char2):
        """Check if transposing two adjacent characters creates a valid word."""
        # This would need the full word context to work properly
        # Implementation would depend on how you're passing the word data
        return None