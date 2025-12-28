"""
OCR-based text extraction for certificate verification.
Extracts key information like names, grades, exam numbers, and dates.
"""

import re
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# Try to import OCR libraries
try:
    import pytesseract
    import os

    # Set Tesseract path only for Windows (Railway/Docker uses system tesseract)
    if os.name == "nt":  # Windows
        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )
    # Linux/Docker: tesseract is in PATH, no need to set

    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    logger.warning("pytesseract not available")

try:
    import easyocr

    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("easyocr not available")


class CertificateOCRExtractor:
    """Extract and validate text content from WAEC certificates."""

    def __init__(self, use_easyocr: bool = True):
        """
        Initialize OCR extractor.

        Args:
            use_easyocr: Whether to use EasyOCR (more accurate but slower)
        """
        self.use_easyocr = use_easyocr and EASYOCR_AVAILABLE
        self.reader = None

        if self.use_easyocr:
            try:
                # Initialize EasyOCR reader for English with model download
                import os

                model_dir = os.path.join(
                    os.path.dirname(__file__), "..", "..", "models", "easyocr"
                )
                os.makedirs(model_dir, exist_ok=True)

                logger.info(
                    "Initializing EasyOCR (this may download models on first run)..."
                )
                self.reader = easyocr.Reader(
                    ["en"],
                    gpu=False,
                    verbose=True,
                    model_storage_directory=model_dir,
                    download_enabled=True,
                )
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
                self.use_easyocr = False
                logger.info("Falling back to Pytesseract if available")

        if not self.use_easyocr and not PYTESSERACT_AVAILABLE:
            logger.warning("No OCR engine available!")

    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract all text from certificate image.

        Args:
            image: Certificate image as numpy array

        Returns:
            Extracted text as string
        """
        try:
            if self.use_easyocr and self.reader:
                # Use EasyOCR
                results = self.reader.readtext(image)
                # Combine all detected text
                text = " ".join([result[1] for result in results])
                return text
            elif PYTESSERACT_AVAILABLE:
                # Use Tesseract OCR with optimal preprocessing
                import cv2

                # Convert PIL Image to numpy if needed
                if isinstance(image, Image.Image):
                    image = np.array(image)

                # Convert to grayscale
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image

                # Try multiple preprocessing strategies and pick best result
                texts = []

                # Strategy 1: Upscale 2x + OTSU (BEST for certificates)
                try:
                    upscaled = cv2.resize(
                        gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
                    )
                    _, upscaled_otsu = cv2.threshold(
                        upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                    text1 = pytesseract.image_to_string(upscaled_otsu, lang="eng")
                    texts.append(text1)
                except:
                    pass

                # Strategy 2: Direct OTSU
                try:
                    _, otsu = cv2.threshold(
                        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                    text2 = pytesseract.image_to_string(otsu, lang="eng")
                    texts.append(text2)
                except:
                    pass

                # Strategy 3: Simple thresholding (fallback)
                try:
                    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                    text3 = pytesseract.image_to_string(thresh, lang="eng")
                    texts.append(text3)
                except:
                    pass

                # Return longest text (more text = better extraction)
                valid_texts = [t for t in texts if t.strip()]
                if valid_texts:
                    return max(valid_texts, key=len)

                return ""
            else:
                logger.error("No OCR engine available")
                return ""
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            return ""

    def extract_certificate_data(self, image: np.ndarray) -> Dict:
        """
        Extract structured data from WAEC certificate.

        Args:
            image: Certificate image as numpy array (can be from PIL Image)

        Returns:
            Dictionary with extracted certificate data:
            {
                'full_name': str,
                'exam_number': str,
                'center_number': str,
                'exam_year': str,
                'exam_month': str,
                'subjects': List[Dict],
                'serial_number': str,
                'candidate_number': str,
                'raw_text': str,
                'confidence': float
            }
        """
        # Convert PIL Image to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Extract raw text
        raw_text = self.extract_text(image)

        if not raw_text:
            return {
                "full_name": None,
                "exam_number": None,
                "center_number": None,
                "exam_year": None,
                "exam_month": None,
                "subjects": [],
                "serial_number": None,
                "candidate_number": None,
                "raw_text": "",
                "confidence": 0.0,
                "ocr_available": False,
            }

        # Extract structured information
        data = {
            "full_name": self._extract_name(raw_text),
            "exam_number": self._extract_exam_number(raw_text),
            "center_number": self._extract_center_number(raw_text),
            "exam_year": self._extract_year(raw_text),
            "exam_month": self._extract_month(raw_text),
            "subjects": self._extract_subjects(raw_text),
            "serial_number": self._extract_serial_number(raw_text),
            "candidate_number": self._extract_candidate_number(raw_text),
            "raw_text": raw_text,
            "confidence": self._calculate_confidence(raw_text),
            "ocr_available": True,
        }

        return data

    def _extract_name(self, text: str) -> Optional[str]:
        """Extract candidate name from text."""
        # Look for patterns like "NAME: JOHN DOE" or "Full Name: JOHN DOE"
        patterns = [
            r"(?:NAME|FULL\s*NAME|CANDIDATE\s*NAME)[\s:]+([A-Z\s]+?)(?:\n|EXAM|CENTER|SEX)",
            r"NAME\s*:\s*([A-Z][A-Z\s]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Clean up name (remove extra spaces, numbers)
                name = re.sub(r"\s+", " ", name)
                name = re.sub(r"\d+", "", name).strip()
                if len(name) > 3:  # Minimum name length
                    return name

        return None

    def _extract_exam_number(self, text: str) -> Optional[str]:
        """Extract examination number."""
        # Normalize text - handle line breaks and multiple spaces
        text_normalized = re.sub(r"[\r\n]+", " ", text)
        text_normalized = re.sub(r"\s+", " ", text_normalized)

        patterns = [
            # Explicit "Examination Number" label (most reliable)
            r"EXAMINATION\s+NUMBER\s+(\d{7,15})",
            r"EXAM(?:INATION)?\s+(?:NO\.?|NUMBER|NUM)[\s:.-]*(\d{7,15})",
            # Candidate/Index number as fallback
            r"(?:CANDIDATE|INDEX)\s+(?:NO\.?|NUMBER)[\s:.-]*(\d{7,15})",
            # Generic number after "Number" keyword
            r"NUMBER\s+(\d{7,15})",
            # Standalone long numbers (7-15 digits) - last resort
            r"\b(\d{10,15})\b",
            r"\b(\d{7,9})\b",  # Shorter numbers too
        ]

        for pattern in patterns:
            match = re.search(pattern, text_normalized, re.IGNORECASE)
            if match:
                exam_no = match.group(1).strip()
                # Remove any remaining spaces/dashes
                exam_no = re.sub(r"[\s-]", "", exam_no)

                if len(exam_no) >= 7 and exam_no.isdigit():
                    return exam_no

        return None

    def _extract_center_number(self, text: str) -> Optional[str]:
        """Extract center number."""
        # Normalize text
        text_normalized = re.sub(r"\s+", " ", text)

        patterns = [
            r"(?:CENTER|CENTRE|CTR)\s*(?:NO\.?|NUMBER|NUM)[\s:.-]+(\d[\s-]?){4,8}",
            r"(?:CENTER|CENTRE)[\s:.-]+(\d{4,8})",
            r"CTR[\s:.-]+(\d{4,8})",
            # School code pattern
            r"SCHOOL\s*(?:CODE|NO)[\s:.-]+(\d{4,8})",
        ]

        for pattern in patterns:
            match = re.search(pattern, text_normalized, re.IGNORECASE)
            if match:
                center_no = match.group(1) if match.lastindex == 1 else match.group(0)
                # Remove spaces and dashes
                center_no = re.sub(r"[\s-]", "", center_no)
                # Extract only digits
                center_no = re.sub(r"\D", "", center_no)

                if len(center_no) >= 4:
                    return center_no

        return None

    def _extract_year(self, text: str) -> Optional[str]:
        """Extract examination year."""
        # Normalize text
        text_normalized = re.sub(r"[\r\n]+", " ", text)
        text_normalized = re.sub(r"\s+", " ", text_normalized)

        # Look for years in various formats
        patterns = [
            r"WASSCE\s*(\d{2})",  # WASSCE20, WASSCE23
            r"20(\d{2})",  # Extract from WASSCE20XX
            r"\b(19[9]\d|20[0-2]\d|203[0])\b",  # Full year
        ]

        matches = []
        for pattern in patterns:
            found = re.findall(pattern, text_normalized, re.IGNORECASE)
            for year in found:
                # Convert 2-digit to 4-digit if needed
                if len(year) == 2:
                    year = "20" + year
                if len(year) == 4 and year.isdigit():
                    matches.append(year)

        if matches:
            # Filter to most recent years (2000+)
            recent_years = [y for y in matches if int(y) >= 2000]
            if recent_years:
                return max(recent_years)
            # Fallback to any year found
            return max(matches)

        return None

    def _extract_month(self, text: str) -> Optional[str]:
        """Extract examination month."""
        months = [
            "JANUARY",
            "FEBRUARY",
            "MARCH",
            "APRIL",
            "MAY",
            "JUNE",
            "JULY",
            "AUGUST",
            "SEPTEMBER",
            "OCTOBER",
            "NOVEMBER",
            "DECEMBER",
            "JAN",
            "FEB",
            "MAR",
            "APR",
            "MAY",
            "JUN",
            "JUL",
            "AUG",
            "SEP",
            "OCT",
            "NOV",
            "DEC",
        ]

        text_upper = text.upper()
        for month in months:
            if month in text_upper:
                return month

        return None

    def _extract_subjects(self, text: str) -> List[Dict]:
        """
        Extract subject grades.

        Returns list of dicts: [{'subject': 'MATHEMATICS', 'grade': 'A1'}, ...]
        """
        subjects = []

        # Normalize text - preserve line structure for subject detection
        text_normalized = text.upper()
        # Split into lines for better subject detection
        lines = text_normalized.split("\n")

        # Common WAEC subjects with variations
        subject_patterns = [
            (r"MATH(?:EMATICS)?(?:\s*\(?(?:CORE|ELECTIVE|GENERAL)\)?)?", "MATHEMATICS"),
            (r"ENGLISH(?:\s+LANGUAGE)?", "ENGLISH LANGUAGE"),
            (r"PHYSICS", "PHYSICS"),
            (r"CHEMISTRY", "CHEMISTRY"),
            (r"BIOLOG(?:Y)?", "BIOLOGY"),
            (r"ECONOMICS?", "ECONOMICS"),
            (r"GEOGRAPHY", "GEOGRAPHY"),
            (r"GOVERN?MENT", "GOVERNMENT"),
            (r"LIT(?:ERATURE)?(?:\s+IN\s+ENGLISH)?", "LITERATURE"),
            (r"HISTORY", "HISTORY"),
            (r"COMMERCE", "COMMERCE"),
            (r"ACCOUNT(?:ING|ANCY)?", "ACCOUNTING"),
            (r"AGRIC(?:ULTURAL)?(?:\s+SC(?:IENCE)?)?", "AGRICULTURAL SCIENCE"),
            (r"TECH(?:NICAL)?(?:\s+DRAWING)?", "TECHNICAL DRAWING"),
            (r"COMPUTER\s+STUD(?:IES)?", "COMPUTER STUDIES"),
            (r"MARKETING", "MARKETING"),
            (r"CIVIC\s+EDU(?:CATION)?", "CIVIC EDUCATION"),
            (r"FRENCH", "FRENCH"),
            (r"YORUBA", "YORUBA"),
            (r"IGBO", "IGBO"),
            (r"HAUSA", "HAUSA"),
            (r"C\.?R\.?S\.?|CHRISTIAN\s+RELIGIOUS\s+STUDIES", "CRS"),
            (r"I\.?R\.?S\.?|ISLAMIC\s+RELIGIOUS\s+STUDIES", "IRS"),
            (r"INTE(?:GRATED)?\s+SC(?:IENCE)?", "INTEGRATED SCIENCE"),
            (r"BASIC\s+SC(?:IENCE)?", "BASIC SCIENCE"),
            (r"ANIMAL\s+HUSBAND(?:RY)?", "ANIMAL HUSBANDRY"),
        ]

        # Enhanced grade patterns - WAEC uses A1-F9
        grade_patterns = [
            r"\b([A-F][1-9])\b",  # A1, B2, C3, etc.
            r"\b([A-F])\s*[:-]?\s*([1-9])\b",  # A 1, A:1, A-1
        ]

        # First pass: Look for subjects with grades on same line or nearby
        for subject_pattern, subject_name in subject_patterns:
            # Check if subject exists in text
            if not re.search(subject_pattern, text_normalized):
                continue

            # Try to find grade near the subject
            for grade_pattern in grade_patterns:
                # Look for subject followed by grade within reasonable distance
                pattern = f"{subject_pattern}[\\s:.-]{{0,30}}({grade_pattern})"
                match = re.search(pattern, text_normalized)

                if match:
                    # Extract grade and normalize
                    if match.lastindex >= 2 and match.group(match.lastindex - 1):
                        grade = match.group(match.lastindex - 1) + match.group(
                            match.lastindex
                        )
                    else:
                        grade = match.group(match.lastindex)

                    grade = re.sub(r"[\s:-]", "", grade).upper()

                    # Validate grade format (A1-F9)
                    if re.match(r"^[A-F][1-9]$", grade):
                        if not any(s["subject"] == subject_name for s in subjects):
                            subjects.append({"subject": subject_name, "grade": grade})
                            break

            # If no grade found but subject is present, add it without grade
            # This helps with detection even if grade is not OCR'd
            if not any(s["subject"] == subject_name for s in subjects):
                # Only add if we're confident (subject name is complete)
                if re.search(f"\\b{subject_pattern}\\b", text_normalized):
                    subjects.append({"subject": subject_name, "grade": "N/A"})

        return subjects

    def _extract_serial_number(self, text: str) -> Optional[str]:
        """Extract certificate serial number."""
        patterns = [
            r"(?:SERIAL|S/N)[\s:]+([A-Z0-9]{6,20})",
            r"SERIAL\s*NO[\s.:]+([A-Z0-9]{6,20})",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _extract_candidate_number(self, text: str) -> Optional[str]:
        """Extract candidate number (different from exam number)."""
        # Normalize text
        text_normalized = re.sub(r"\s+", " ", text)

        patterns = [
            r"(?:CANDIDATE|CAND)\s*(?:NO\.?|NUMBER|NUM)[\s:.-]+(\d[\s-]?){7,15}",
            r"CAND\.?\s*(?:NO\.?|NUMBER)[\s:.-]+(\d{7,15})",
            r"CANDNO[\s:.-]+(\d{7,15})",
        ]

        for pattern in patterns:
            match = re.search(pattern, text_normalized, re.IGNORECASE)
            if match:
                cand_no = match.group(1) if match.lastindex == 1 else match.group(0)
                # Remove spaces and dashes
                cand_no = re.sub(r"[\s-]", "", cand_no)
                # Extract only digits
                cand_no = re.sub(r"\D", "", cand_no)

                if len(cand_no) >= 7:
                    return cand_no

        return None

    def _calculate_confidence(self, text: str) -> float:
        """
        Calculate confidence score based on extracted data quality.

        Returns:
            Confidence score between 0 and 1
        """
        if not text or len(text) < 50:
            return 0.0

        confidence = 0.0

        # Check for key WAEC identifiers
        waec_keywords = ["WAEC", "WEST AFRICAN EXAMINATIONS COUNCIL", "SENIOR SCHOOL"]
        for keyword in waec_keywords:
            if keyword in text.upper():
                confidence += 0.2

        # Check for structured data presence
        if re.search(r"\d{10,}", text):  # Has exam number-like pattern
            confidence += 0.15

        if re.search(r"20[0-2]\d", text):  # Has year
            confidence += 0.1

        if re.search(r"[A-F][1-9]", text):  # Has grade pattern
            confidence += 0.15

        if re.search(r"(?:NAME|CANDIDATE)", text, re.IGNORECASE):
            confidence += 0.1

        # Check text length (longer = more content extracted)
        if len(text) > 200:
            confidence += 0.1
        elif len(text) > 100:
            confidence += 0.05

        return min(confidence, 1.0)

    def validate_certificate_data(self, data: Dict) -> Dict:
        """
        Validate extracted certificate data for authenticity.

        Args:
            data: Extracted certificate data dictionary

        Returns:
            Validation results with anomalies and confidence score
        """
        validation = {
            "is_valid": True,
            "anomalies": [],
            "confidence": data.get("confidence", 0.0),
            "validation_score": 1.0,
        }

        # Check if OCR worked
        if not data.get("ocr_available"):
            validation["anomalies"].append("OCR not available")
            validation["is_valid"] = False
            validation["validation_score"] = 0.0
            return validation

        # Check for minimum required fields
        if not data.get("full_name"):
            validation["anomalies"].append("Missing candidate name")
            validation["validation_score"] -= 0.2

        if not data.get("exam_number") and not data.get("candidate_number"):
            validation["anomalies"].append("Missing examination/candidate number")
            validation["validation_score"] -= 0.2

        if not data.get("exam_year"):
            validation["anomalies"].append("Missing examination year")
            validation["validation_score"] -= 0.15

        # Validate year range
        if data.get("exam_year"):
            try:
                year = int(data["exam_year"])
                if year < 2000 or year > 2030:
                    validation["anomalies"].append(f"Invalid year: {year}")
                    validation["validation_score"] -= 0.15
            except ValueError:
                validation["anomalies"].append("Invalid year format")
                validation["validation_score"] -= 0.15

        # Check for subjects - CRITICAL for authenticity
        subjects = data.get("subjects", [])
        if not subjects or len(subjects) < 3:
            validation["anomalies"].append(
                "Insufficient subjects found (minimum 3 expected)"
            )
            validation["validation_score"] -= 0.15

        # Validate grades - CRITICAL AUTHENTICITY INDICATOR
        # Authentic certificates MUST have valid grades
        valid_grades_count = 0
        missing_grades_count = 0

        for subject in subjects:
            grade = subject.get("grade", "")
            if re.match(r"^[A-F][1-9]$", grade):
                valid_grades_count += 1
            else:
                missing_grades_count += 1
                if grade == "N/A":
                    validation["anomalies"].append(
                        f"Missing grade for {subject.get('subject', 'Unknown')}"
                    )

        # Grade presence is a STRONG indicator
        if len(subjects) > 0:
            grade_ratio = valid_grades_count / len(subjects)

            if grade_ratio == 0:
                # NO GRADES AT ALL - Very suspicious (likely forged)
                validation["anomalies"].append(
                    "⚠️ CRITICAL: No valid grades found (strong forgery indicator)"
                )
                validation["validation_score"] -= 0.5  # Heavy penalty

            elif grade_ratio < 0.5:
                # Less than half have grades - Suspicious
                validation["anomalies"].append(
                    f"⚠️ WARNING: Only {valid_grades_count}/{len(subjects)} subjects have valid grades"
                )
                validation["validation_score"] -= 0.3

            elif grade_ratio < 0.8:
                # Most but not all have grades - Slightly suspicious
                validation["anomalies"].append(
                    f"Some subjects missing grades ({missing_grades_count} of {len(subjects)})"
                )
                validation["validation_score"] -= 0.1

        # Check raw text quality
        raw_text = data.get("raw_text", "")
        if "WAEC" not in raw_text.upper() and "WEST AFRICAN" not in raw_text.upper():
            validation["anomalies"].append("Missing WAEC identifier")
            validation["validation_score"] -= 0.2

        # Check for grade patterns in raw text (additional verification)
        grade_patterns_found = len(re.findall(r"\b[A-F][1-9]\b", raw_text))
        if grade_patterns_found == 0 and len(subjects) > 0:
            validation["anomalies"].append(
                "⚠️ No grade patterns detected in certificate text"
            )
            # Already penalized above, but note it

        # Final validation decision
        validation["validation_score"] = max(0.0, validation["validation_score"])
        validation["is_valid"] = (
            validation["validation_score"] >= 0.5 and len(validation["anomalies"]) < 3
        )

        return validation


# Global OCR extractor instance
_ocr_extractor = None


def get_ocr_extractor() -> CertificateOCRExtractor:
    """Get or create global OCR extractor instance."""
    global _ocr_extractor
    if _ocr_extractor is None:
        _ocr_extractor = CertificateOCRExtractor()
    return _ocr_extractor
