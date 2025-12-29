# -*- coding: utf-8 -*-
"""
Test OCR extraction and validation on training data (authentic vs forged)
"""

import sys
import os
from pathlib import Path
import sys
import os

ml_service_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ml_service")
)
if ml_service_path not in sys.path:
    sys.path.insert(0, ml_service_path)
from app.models.ocr_extractor import get_ocr_extractor
from PIL import Image
import traceback

TRAIN_DIR = Path(__file__).parent.parent / "training_data" / "train"
AUTH_DIR = TRAIN_DIR / "authentic"
FORGED_DIR = TRAIN_DIR / "forged"


def test_ocr_on_folder(folder, label):
    extractor = get_ocr_extractor()
    files = [
        f
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".png", ".webp", ".jpeg"))
    ]
    print(f"\n{'='*80}\nTesting {label} certificates: {len(files)} files\n{'='*80}")
    for img_file in files:
        img_path = os.path.join(folder, img_file)
        print(f"\n📄 Processing: {img_file}")
        print("-" * 70)
        try:
            img = Image.open(img_path)
            data = extractor.extract_certificate_data(img)
            print(f"✓ Name: {data['full_name']}")
            print(f"✓ Exam Number: {data['exam_number']}")
            print(f"✓ Center Number: {data['center_number']}")
            print(f"✓ Exam Year: {data['exam_year']}")
            print(f"✓ Serial Number: {data['serial_number']}")
            print(f"✓ Subjects Found: {len(data['subjects'])}")
            print(f"✓ OCR Confidence: {data['confidence']:.2%}")
            validation = extractor.validate_certificate_data(data)
            print(f"  Valid: {validation['is_valid']}")
            print(f"  Validation Score: {validation['validation_score']:.2%}")
            if validation["anomalies"]:
                print(f"  Anomalies: {', '.join(validation['anomalies'])}")
        except Exception as e:
            print(f"❌ Error: {e}")
            traceback.print_exc()


def main():
    test_ocr_on_folder(AUTH_DIR, "AUTHENTIC")
    test_ocr_on_folder(FORGED_DIR, "FORGED")


if __name__ == "__main__":
    main()
