"""Debug script to check specific problematic certificates"""

import sys
import os

if sys.platform == "win32":
    import codecs

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.ocr_extractor import get_ocr_extractor
from PIL import Image


def debug_certificate(filename: str):
    """Debug a specific certificate"""
    print(f"\n{'='*80}")
    print(f"Debugging: {filename}")
    print(f"{'='*80}")

    extractor = get_ocr_extractor()
    img_path = f"training_data/val/forged/{filename}"

    if not os.path.exists(img_path):
        print(f"❌ File not found: {img_path}")
        return

    image = Image.open(img_path)
    data = extractor.extract_certificate_data(image)

    print(f"\n📋 Extracted Data:")
    print(f"  Name: {data.get('full_name')}")
    print(f"  Exam Number: {data.get('exam_number')}")
    print(f"  Year: {data.get('exam_year')}")
    print(f"  Subjects Found: {len(data.get('subjects', []))}")

    subjects = data.get("subjects", [])
    print(f"\n📚 Subjects & Grades:")
    for s in subjects:
        grade = s.get("grade", "N/A")
        grade_icon = "✅" if grade != "N/A" else "❌"
        print(f"  {grade_icon} {s.get('subject')}: {grade}")

    validation = extractor.validate_certificate_data(data)
    print(f"\n🔍 Validation:")
    print(f"  Score: {validation.get('validation_score'):.0%}")
    print(f"  Anomalies: {len(validation.get('anomalies', []))}")
    for anomaly in validation.get("anomalies", [])[:5]:
        print(f"    - {anomaly}")

    print(f"\n📝 Raw Text (first 500 chars):")
    print(data.get("raw_text", "")[:500])


if __name__ == "__main__":
    # Check the problematic ones
    debug_certificate("forged_0008.jpg")  # Has some grades (unusual)
    debug_certificate("forged_0010.jpg")  # Almost authentic score
