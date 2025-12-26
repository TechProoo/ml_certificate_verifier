"""
Model Evaluation Script

Evaluates trained CNN model on test dataset and generates performance metrics.

Usage:
    python scripts/evaluate_model.py --test-dir training_data/test --model-path models/certificate_detector_v1.h5
"""

import os
import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

from app.models.detector import CertificateDetector


def evaluate_model(args):
    """
    Evaluate trained model on test dataset.

    Args:
        args: Command line arguments
    """
    print("=" * 80)
    print("Certificate Verification CNN Evaluation")
    print("=" * 80)
    print(f"Model Path: {args.model_path}")
    print(f"Test Directory: {args.test_dir}")
    print("=" * 80)

    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ ERROR: Model not found at {args.model_path}")
        print("\nTrain a model first using:")
        print("  python scripts/train_model.py")
        sys.exit(1)

    # Check if test data exists
    test_path = Path(args.test_dir)
    if not test_path.exists():
        print(f"❌ ERROR: Test directory not found: {args.test_dir}")
        print("\nPlease create test data directory with structure:")
        print("test/")
        print("├── authentic/")
        print("├── forged/")
        print("└── suspicious/")
        sys.exit(1)

    # Load model
    print("\n🤖 Loading model...")
    detector = CertificateDetector(model_path=str(model_path))

    if not detector.model_loaded:
        print("⚠️  Warning: Model not loaded properly, creating new model")
    else:
        print("✅ Model loaded successfully")

    # Create test data generator
    print("\n📊 Loading test data...")
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_generator = test_datagen.flow_from_directory(
        args.test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=False,  # Important: don't shuffle for accurate evaluation
    )

    print(f"✅ Found {test_generator.samples} test samples")
    print(f"✅ Class indices: {test_generator.class_indices}")

    # Get predictions
    print("\n🔍 Evaluating model...")
    predictions = detector.model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Calculate metrics
    accuracy = accuracy_score(true_classes, predicted_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_classes, predicted_classes, average="weighted"
    )

    # Print results
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")
    print("=" * 80)

    # Detailed classification report
    print("\n📋 Detailed Classification Report:")
    print("-" * 80)
    report = classification_report(
        true_classes, predicted_classes, target_names=class_labels, digits=4
    )
    print(report)

    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)

    print("\n📊 Confusion Matrix:")
    print("-" * 80)
    print(f"{'':>15}", end="")
    for label in class_labels:
        print(f"{label:>15}", end="")
    print()

    for i, label in enumerate(class_labels):
        print(f"{label:>15}", end="")
        for j in range(len(class_labels)):
            print(f"{cm[i][j]:>15}", end="")
        print()
    print("-" * 80)

    # Calculate per-class accuracies
    print("\n🎯 Per-Class Accuracy:")
    print("-" * 80)
    for i, label in enumerate(class_labels):
        class_total = np.sum(true_classes == i)
        class_correct = cm[i][i]
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        print(
            f"{label:>15}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%) - {class_correct}/{class_total} correct"
        )
    print("-" * 80)

    # Save metrics to JSON
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    metrics_path = output_dir / "evaluation_metrics.json"
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(model_path),
        "test_samples": int(test_generator.samples),
        "overall_metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        },
        "per_class_metrics": {},
        "confusion_matrix": cm.tolist(),
    }

    # Add per-class metrics
    for i, label in enumerate(class_labels):
        class_total = int(np.sum(true_classes == i))
        class_correct = int(cm[i][i])
        class_accuracy = float(class_correct / class_total if class_total > 0 else 0)

        metrics["per_class_metrics"][label] = {
            "accuracy": class_accuracy,
            "correct": class_correct,
            "total": class_total,
        }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Metrics saved to: {metrics_path}")

    # Plot confusion matrix if matplotlib available
    if args.plot:
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_labels,
                yticklabels=class_labels,
            )
            plt.title("Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")

            plot_path = output_dir / "confusion_matrix.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"✅ Confusion matrix plot saved to: {plot_path}")

            if args.show:
                plt.show()
        except Exception as e:
            print(f"⚠️  Could not generate plot: {str(e)}")

    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Certificate Verification CNN"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="models/certificate_detector_v1.h5",
        help="Path to trained model file (default: models/certificate_detector_v1.h5)",
    )

    parser.add_argument(
        "--test-dir",
        type=str,
        default="training_data/test",
        help="Directory containing test data (default: training_data/test)",
    )

    parser.add_argument(
        "--plot", action="store_true", help="Generate confusion matrix plot"
    )

    parser.add_argument(
        "--show", action="store_true", help="Display plots (requires --plot)"
    )

    args = parser.parse_args()

    # Evaluate model
    evaluate_model(args)


if __name__ == "__main__":
    main()
