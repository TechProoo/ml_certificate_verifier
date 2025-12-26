"""Detailed OCR analysis to see full extracted text with best preprocessing"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image
import cv2
import numpy as np
import pytesseract
import re

# Configure tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def upscale_otsu_ocr(img_path):
    """Apply best preprocessing strategy: Upscale 2x + OTSU"""
    print(f"\n{'='*70}")
    print(f"📄 Analyzing: {os.path.basename(img_path)}")
    print(f"{'='*70}")
    
    # Load image
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Upscale 2x
    upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Apply OTSU threshold
    _, binary = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Extract text
    text = pytesseract.image_to_string(binary)
    
    print("\n📝 FULL OCR TEXT:")
    print("=" * 70)
    print(text)
    print("=" * 70)
    
    # Look for grade patterns
    print("\n🔍 Searching for Grade Patterns:")
    print("-" * 70)
    
    # Pattern 1: Single letter followed by single digit
    pattern1 = re.findall(r'\b([A-F])\s*([1-9])\b', text)
    if pattern1:
        print(f"✓ Found letter-digit pairs: {pattern1}")
    else:
        print("✗ No letter-digit pairs found (A1, B2, etc.)")
    
    # Pattern 2: Letter and digit together
    pattern2 = re.findall(r'\b([A-F][1-9])\b', text)
    if pattern2:
        print(f"✓ Found combined grades: {pattern2}")
    else:
        print("✗ No combined grades found")
    
    # Pattern 3: Look for grade context
    pattern3 = re.findall(r'(?:Grade|GRADE|grade)[\s:.-]*([A-Z0-9\s]+)', text)
    if pattern3:
        print(f"✓ Found grade context: {pattern3}")
    else:
        print("✗ No 'Grade' label found")
    
    # Check for subject-grade table structure
    print("\n📊 Looking for Subject-Grade Structure:")
    print("-" * 70)
    lines = text.split('\n')
    for i, line in enumerate(lines):
        # Look for common subjects
        if re.search(r'MATH|ENGLISH|PHYSICS|CHEMISTRY|BIOLOGY', line, re.IGNORECASE):
            # Show context (previous line, current, next line)
            context_start = max(0, i-1)
            context_end = min(len(lines), i+2)
            print(f"Line {i}: {line}")
            for j in range(context_start, context_end):
                if j != i:
                    print(f"  [{j}]: {lines[j]}")
            print()
    
    return text


if __name__ == "__main__":
    test_images = [
        "training_data/val/forged/forged_0001.jpg",
        "training_data/val/forged/forged_0003.jpg",
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            text = upscale_otsu_ocr(img_path)
        else:
            print(f"❌ File not found: {img_path}")
