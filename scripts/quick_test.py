"""Quick test script to evaluate trained model"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.detector import CertificateDetector
from PIL import Image
import json


def test_model():
    print("Loading model...")
    detector = CertificateDetector()

    print(f"\nModel Details:")
    print(f"Number of classes: {detector.num_classes}")
    print(f"Class names: {detector.class_names}")

    # Load class indices
    with open("models/class_indices.json", "r") as f:
        class_indices = json.load(f)
    print(f"\nClass indices mapping: {class_indices}")

    # Test on some authentic images
    print("\n" + "=" * 70)
    print("Testing on AUTHENTIC certificates:")
    print("=" * 70)

    authentic_dir = "training_data/val/authentic"
    authentic_files = [
        f for f in os.listdir(authentic_dir) if f.endswith((".jpg", ".png", ".webp"))
    ][:5]

    for img_file in authentic_files:
        img_path = os.path.join(authentic_dir, img_file)
        print(f"\n{img_file}:")
        try:
            img = Image.open(img_path)
            result = detector.predict(img)
            if "authenticity" in result:
                print(f"  Prediction: {result['authenticity']}")
                print(f"  Confidence: {result['confidence']:.2f}%")
                print(f"  Probabilities: {result['class_probabilities']}")
            else:
                print(f"  Result keys: {result.keys()}")
                print(f"  Full result: {result}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback

            traceback.print_exc()

    # Test on forged images
    print("\n" + "=" * 70)
    print("Testing on FORGED certificates:")
    print("=" * 70)

    forged_dir = "training_data/val/forged"
    forged_files = [f for f in os.listdir(forged_dir) if f.endswith((".jpg", ".png"))][
        :5
    ]

    for img_file in forged_files:
        img_path = os.path.join(forged_dir, img_file)
        print(f"\n{img_file}:")
        try:
            img = Image.open(img_path)
            result = detector.predict(img)
            if "authenticity" in result:
                print(f"  Prediction: {result['authenticity']}")
                print(f"  Confidence: {result['confidence']:.2f}%")
                print(f"  Probabilities: {result['class_probabilities']}")
            else:
                print(f"  Result keys: {result.keys()}")
                print(f"  Full result: {result}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    test_model()
