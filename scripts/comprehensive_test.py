"""
Comprehensive test showing complete OCR-based certificate verification
Demonstrates forged vs authentic certificate detection
"""

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


def test_certificate(img_path: str, expected_type: str = "unknown"):
    """Test a single certificate and display results"""
    print(f"\n{'='*80}")
    print(f"📋 Certificate: {os.path.basename(img_path)}")
    print(f"📁 Expected Type: {expected_type.upper()}")
    print(f"{'='*80}")

    try:
        # Load image
        image = Image.open(img_path)
        print(f"📏 Image Size: {image.size[0]}x{image.size[1]} pixels")

        # Get extractor
        extractor = get_ocr_extractor()

        # Extract data
        print("\n🔄 Extracting certificate data...")
        data = extractor.extract_certificate_data(image)

        # Display extracted data
        print(f"\n📋 EXTRACTED INFORMATION:")
        print(f"  ├─ Candidate Name: {data.get('full_name') or '❌ NOT FOUND'}")
        print(f"  ├─ Exam Number: {data.get('exam_number') or '❌ NOT FOUND'}")
        print(f"  ├─ Center: {data.get('center_number') or '❌ NOT FOUND'}")
        print(f"  ├─ Year: {data.get('exam_year') or '❌ NOT FOUND'}")
        print(f"  ├─ Month: {data.get('exam_month') or '❌ NOT FOUND'}")
        print(f"  └─ OCR Confidence: {data.get('confidence', 0):.1%}")

        # Display subjects
        subjects = data.get("subjects", [])
        print(f"\n📚 SUBJECTS ({len(subjects)} found):")
        if subjects:
            grades_present = sum(1 for s in subjects if s.get("grade") != "N/A")
            for subject in subjects[:5]:  # Show first 5
                grade_status = "✅" if subject.get("grade") != "N/A" else "❌"
                print(
                    f"  {grade_status} {subject.get('subject')}: {subject.get('grade', 'N/A')}"
                )
            if len(subjects) > 5:
                print(f"  ... and {len(subjects) - 5} more subjects")
            print(
                f"\n  📊 Grade Summary: {grades_present}/{len(subjects)} subjects have valid grades"
            )
        else:
            print("  ❌ No subjects found")

        # Validate
        print("\n🔍 VALIDATION RESULTS:")
        validation = extractor.validate_certificate_data(data)

        print(f"  ├─ Validation Score: {validation.get('validation_score', 0):.1%}")
        print(f"  ├─ Valid: {validation.get('is_valid')}")
        print(f"  └─ Anomalies Found: {len(validation.get('anomalies', []))}")

        if validation.get("anomalies"):
            print(f"\n⚠️  ANOMALIES DETECTED:")
            for anomaly in validation.get("anomalies", []):
                if "CRITICAL" in anomaly:
                    print(f"  🚨 {anomaly}")
                elif "WARNING" in anomaly:
                    print(f"  ⚠️  {anomaly}")
                else:
                    print(f"  • {anomaly}")

        # Final verdict
        print(f"\n{'='*80}")
        score = validation.get("validation_score", 0)

        if score > 0.7:
            verdict = "✅ LIKELY AUTHENTIC"
            confidence = "HIGH"
        elif score > 0.4:
            verdict = "⚠️  UNCERTAIN (Review Required)"
            confidence = "MEDIUM"
        else:
            verdict = "❌ LIKELY FORGED"
            confidence = "HIGH"

        print(f"🎯 VERDICT: {verdict}")
        print(f"📊 Confidence: {confidence}")
        print(f"📈 Authenticity Score: {score:.1%}")
        print(f"{'='*80}")

        return {
            "file": os.path.basename(img_path),
            "verdict": verdict,
            "score": score,
            "is_authentic": score > 0.7,
            "data": data,
            "validation": validation,
        }

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Run comprehensive certificate verification tests"""
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print(
        "█"
        + "  OCR-Based Certificate Verification System - Complete Test".center(78)
        + "█"
    )
    print("█" + " " * 78 + "█")
    print("█" * 80)

    extractor = get_ocr_extractor()
    print(f"\n✅ OCR Extractor initialized successfully")

    # Test forged certificates
    print(f"\n\n{'='*80}")
    print("🔴 TESTING FORGED CERTIFICATES")
    print(f"{'='*80}")

    forged_results = []
    forged_dir = "training_data/val/forged"
    if os.path.exists(forged_dir):
        forged_files = sorted(
            [f for f in os.listdir(forged_dir) if f.endswith((".jpg", ".png", ".webp"))]
        )[
            :3
        ]  # Test first 3

        for file in forged_files:
            result = test_certificate(os.path.join(forged_dir, file), "FORGED")
            if result:
                forged_results.append(result)

    # Test authentic certificates
    print(f"\n\n{'='*80}")
    print("🟢 TESTING AUTHENTIC CERTIFICATES")
    print(f"{'='*80}")

    authentic_results = []
    authentic_dir = "training_data/val/authentic"
    if os.path.exists(authentic_dir):
        authentic_files = sorted(
            [
                f
                for f in os.listdir(authentic_dir)
                if f.endswith((".jpg", ".png", ".webp"))
            ]
        )[
            :3
        ]  # Test first 3

        for file in authentic_files:
            result = test_certificate(os.path.join(authentic_dir, file), "AUTHENTIC")
            if result:
                authentic_results.append(result)

    # Summary
    print(f"\n\n{'='*80}")
    print("📊 TEST SUMMARY")
    print(f"{'='*80}")

    print(f"\n🔴 Forged Certificates: {len(forged_results)} tested")
    if forged_results:
        forged_correct = sum(1 for r in forged_results if not r.get("is_authentic"))
        print(
            f"   ✅ Correctly Identified as Forged: {forged_correct}/{len(forged_results)}"
        )
        avg_score = sum(r.get("score", 0) for r in forged_results) / len(forged_results)
        print(f"   📊 Average Score: {avg_score:.1%}")

    print(f"\n🟢 Authentic Certificates: {len(authentic_results)} tested")
    if authentic_results:
        authentic_correct = sum(1 for r in authentic_results if r.get("is_authentic"))
        print(
            f"   ✅ Correctly Identified as Authentic: {authentic_correct}/{len(authentic_results)}"
        )
        avg_score = sum(r.get("score", 0) for r in authentic_results) / len(
            authentic_results
        )
        print(f"   📊 Average Score: {avg_score:.1%}")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
