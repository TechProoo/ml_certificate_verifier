# -*- coding: utf-8 -*-
"""Test OCR extraction on certificate images"""

import sys
import os

# Set UTF-8 encoding for console output
if sys.platform == "win32":
    import codecs

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.ocr_extractor import get_ocr_extractor
from PIL import Image
import json


def test_ocr():
    print("Initializing OCR extractor...")
    extractor = get_ocr_extractor()

    # Test on an authentic certificate
    print("\n" + "=" * 70)
    print("Testing OCR on AUTHENTIC certificates:")
    print("=" * 70)

    auth_dir = "training_data/val/authentic"
    auth_files = [
        f for f in os.listdir(auth_dir) if f.endswith((".jpg", ".png", ".webp"))
    ][:2]

    for img_file in auth_files:
        img_path = os.path.join(auth_dir, img_file)
        print(f"\n📄 Processing: {img_file}")
        print("-" * 70)

        try:
            # Load image
            img = Image.open(img_path)

            # Extract data
            data = extractor.extract_certificate_data(img)

            # Display extracted information
            print(f"✓ Name: {data['full_name']}")
            print(f"✓ Exam Number: {data['exam_number']}")
            print(f"✓ Center Number: {data['center_number']}")
            print(f"✓ Exam Year: {data['exam_year']}")
            print(f"✓ Exam Month: {data['exam_month']}")
            print(f"✓ Serial Number: {data['serial_number']}")
            print(f"✓ Subjects Found: {len(data['subjects'])}")
            for subject in data["subjects"][:5]:  # Show first 5
                print(f"    - {subject['subject']}: {subject['grade']}")
            print(f"✓ OCR Confidence: {data['confidence']:.2%}")

            # Validate
            validation = extractor.validate_certificate_data(data)
            print(f"\n📊 Validation Results:")
            print(f"    Valid: {validation['is_valid']}")
            print(f"    Validation Score: {validation['validation_score']:.2%}")
            if validation["anomalies"]:
                print(f"    Anomalies: {', '.join(validation['anomalies'])}")
            # Show sample of raw text
            raw_text = data["raw_text"]
            print(f"\n📝 Raw Text (length: {len(raw_text)} chars):")
            if raw_text:
                # Show first 500 chars
                print(f"{raw_text[:500]}")
                if len(raw_text) > 500:
                    print(f"... ({len(raw_text)-500} more characters)")
            else:
                print("    (No text extracted)")
            # Show sample of raw text
            raw_text = data["raw_text"]
            print(f"\n📝 Raw Text (length: {len(raw_text)} chars):")
            if raw_text:
                # Show first 500 chars
                print(f"{raw_text[:500]}")
                if len(raw_text) > 500:
                    print(f"... ({len(raw_text)-500} more characters)")
            else:
                print("    (No text extracted)")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()

    # Test on a forged certificate
    print("\n" + "=" * 70)
    print("Testing OCR on FORGED certificates:")
    print("=" * 70)

    forged_dir = "training_data/val/forged"
    forged_files = [f for f in os.listdir(forged_dir) if f.endswith((".jpg", ".png"))][
        :2
    ]

    for img_file in forged_files:
        img_path = os.path.join(forged_dir, img_file)
        print(f"\n📄 Processing: {img_file}")
        print("-" * 70)

        try:
            # Load image
            img = Image.open(img_path)

            # Extract data
            data = extractor.extract_certificate_data(img)

            # Display extracted information
            print(f"✓ Name: {data['full_name']}")
            print(f"✓ Exam Number: {data['exam_number']}")
            print(f"✓ Center Number: {data['center_number']}")
            print(f"✓ Exam Year: {data['exam_year']}")
            print(f"✓ Exam Month: {data['exam_month']}")
            print(f"✓ Serial Number: {data['serial_number']}")
            print(f"✓ Subjects Found: {len(data['subjects'])}")
            for subject in data["subjects"][:5]:  # Show first 5
                print(f"    - {subject['subject']}: {subject['grade']}")
            print(f"✓ OCR Confidence: {data['confidence']:.2%}")

            # Validate
            validation = extractor.validate_certificate_data(data)
            print(f"\n📊 Validation Results:")
            print(f"    Valid: {validation['is_valid']}")
            print(f"    Validation Score: {validation['validation_score']:.2%}")
            if validation["anomalies"]:
                print(f"    Anomalies: {', '.join(validation['anomalies'])}")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    test_ocr()
