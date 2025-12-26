"""Extract text from authentic PDF to check if grades are present"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import cv2
import numpy as np

# Configure tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def extract_from_pdf(pdf_path):
    """Extract text directly from PDF and also from rendered images"""
    print(f"\n{'='*70}")
    print(f"📄 Analyzing PDF: {os.path.basename(pdf_path)}")
    print(f"{'='*70}")
    
    doc = fitz.open(pdf_path)
    print(f"📑 Total Pages: {len(doc)}")
    
    for page_num in range(min(2, len(doc))):  # Check first 2 pages
        page = doc[page_num]
        
        print(f"\n{'='*70}")
        print(f"📄 Page {page_num + 1}")
        print(f"{'='*70}")
        
        # Method 1: Extract text directly from PDF
        print("\n🔹 Method 1: Direct PDF Text Extraction")
        print("-" * 70)
        text = page.get_text()
        print(text[:500])  # First 500 chars
        
        # Look for grades in direct text
        import re
        grades = re.findall(r'\b([A-F][1-9])\b', text)
        if grades:
            print(f"\n✅ GRADES FOUND in PDF text: {grades}")
        else:
            print("\n❌ No grades found in direct PDF text")
        
        # Method 2: Render page as image and OCR
        print("\n🔹 Method 2: Render PDF Page → OCR")
        print("-" * 70)
        
        # Render page to image (2x scale for better quality)
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Convert to numpy for OpenCV
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Apply best preprocessing
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR
        ocr_text = pytesseract.image_to_string(binary)
        print(ocr_text[:500])  # First 500 chars
        
        # Look for grades in OCR text
        grades_ocr = re.findall(r'\b([A-F][1-9])\b', ocr_text)
        if grades_ocr:
            print(f"\n✅ GRADES FOUND in OCR: {grades_ocr}")
        else:
            print("\n❌ No grades found in OCR text")


if __name__ == "__main__":
    test_pdfs = [
        "training_data/val/authentic/Waec Original result.pdf",
        "training_data/train/authentic/CamScanner 10-23-2025 20.15.pdf",
    ]
    
    for pdf_path in test_pdfs:
        if os.path.exists(pdf_path):
            extract_from_pdf(pdf_path)
        else:
            print(f"❌ File not found: {pdf_path}")
