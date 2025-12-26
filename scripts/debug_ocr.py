# -*- coding: utf-8 -*-
"""Debug OCR to see what text is actually extracted"""

import sys
import os

# Set UTF-8 encoding for console output
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.ocr_extractor import CertificateOCRExtractor
from PIL import Image
import numpy as np

def debug_ocr():
    print("Initializing OCR extractor...")
    extractor = CertificateOCRExtractor(use_easyocr=False)  # Force Tesseract
    
    # Test on forged certificates
    print("\n" + "="*70)
    print("RAW OCR TEXT EXTRACTION DEBUG")
    print("="*70)
    
    test_dir = "training_data/val/forged"
    test_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))][:3]
    
    for img_file in test_files:
        img_path = os.path.join(test_dir, img_file)
        print(f"\n📄 File: {img_file}")
        print("-" * 70)
        
        try:
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # Extract raw text
            raw_text = extractor.extract_text(img_array)
            
            print(f"✅ Text Length: {len(raw_text)} characters")
            print(f"\n📝 RAW TEXT OUTPUT:")
            print("=" * 70)
            print(raw_text)
            print("=" * 70)
            
            # Now try extraction
            print(f"\n🔍 EXTRACTED FIELDS:")
            data = extractor.extract_certificate_data(img)
            print(f"  Name: {data['full_name']}")
            print(f"  Exam Number: {data['exam_number']}")
            print(f"  Center: {data['center_number']}")
            print(f"  Year: {data['exam_year']}")
            print(f"  Month: {data['exam_month']}")
            print(f"  Subjects: {len(data['subjects'])}")
            for subj in data['subjects'][:5]:
                print(f"    - {subj['subject']}: {subj['grade']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_ocr()
