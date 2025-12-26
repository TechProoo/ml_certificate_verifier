"""
Certificate Verification CNN Model Training Script

This script trains a MobileNetV2-based CNN to classify certificates as:
- AUTHENTIC: Real, unmodified certificates
- SUSPICIOUS: Modified or questionable certificates
- FORGED: Clearly fake or forged certificates

Usage:
    python scripts/train_model.py --epochs 50 --batch-size 32 --learning-rate 0.001
"""

import os
import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)

from app.models.detector import CertificateDetector


def create_data_generators(
    train_dir: str, val_dir: str, batch_size: int = 32, img_size: tuple = (224, 224)
):
    """
    Create data generators with augmentation for training and validation.

    Args:
        train_dir: Directory containing training data
        val_dir: Directory containing validation data
        batch_size: Batch size for training
        img_size: Image size (width, height)

    Returns:
        Tuple of (train_generator, validation_generator)
    """
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,  # Random rotation ±15 degrees
        width_shift_range=0.1,  # Random horizontal shift
        height_shift_range=0.1,  # Random vertical shift
        shear_range=0.1,  # Shear transformation
        zoom_range=0.15,  # Random zoom 10-20%
        brightness_range=[0.8, 1.2],  # Brightness adjustment ±20%
        horizontal_flip=False,  # Don't flip (certificates have fixed orientation)
        fill_mode="nearest",
    )

    # Validation data (no augmentation, only rescaling)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        seed=42,
    )

    return train_generator, validation_generator


def train_model(args):
    """
    Train the certificate verification CNN model.

    Args:
        args: Command line arguments
    """
    print("=" * 80)
    print("Certificate Verification CNN Training")
    print("=" * 80)
    print(f"Training Directory: {args.train_dir}")
    print(f"Validation Directory: {args.val_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print("=" * 80)

    # Check if training data exists
    train_path = Path(args.train_dir)
    if not train_path.exists():
        print(f"❌ ERROR: Training directory not found: {args.train_dir}")
        print("\nPlease create the following directory structure:")
        print("training_data/")
        print("├── train/")
        print("│   ├── authentic/")
        print("│   ├── forged/")
        print("│   └── suspicious/")
        print("└── val/")
        print("    ├── authentic/")
        print("    ├── forged/")
        print("    └── suspicious/")
        sys.exit(1)

    # Create output directories
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    logs_dir = Path("logs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create data generators
    print("\n📊 Creating data generators...")
    train_gen, val_gen = create_data_generators(
        args.train_dir, args.val_dir, batch_size=args.batch_size
    )

    print(f"✅ Found {train_gen.samples} training samples")
    print(f"✅ Found {val_gen.samples} validation samples")
    print(f"✅ Class indices: {train_gen.class_indices}")

    # Create model
    print("\n🤖 Creating CNN model...")

    # Detect number of classes from data generator
    num_classes = len(train_gen.class_indices)
    print(f"Detected {num_classes} classes: {list(train_gen.class_indices.keys())}")

    detector = CertificateDetector()
    model = detector._create_model(num_classes=num_classes)

    # Always recompile with fresh optimizer to avoid cached state issues
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
    )

    print("✅ Model created successfully")
    print(f"Total parameters: {model.count_params():,}")

    # Callbacks
    callbacks = [
        # Save best model
        ModelCheckpoint(
            str(models_dir / "certificate_detector_v1.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        # Early stopping
        EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
        ),
        # TensorBoard logging
        TensorBoard(log_dir=str(logs_dir), histogram_freq=1, write_graph=True),
    ]

    # Train model
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print("=" * 80)

    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model
    final_model_path = models_dir / "certificate_detector_final.h5"
    model.save(str(final_model_path))
    print(f"\n✅ Final model saved to: {final_model_path}")

    # Save training history
    history_path = models_dir / "training_history.json"
    history_dict = {
        "accuracy": [float(x) for x in history.history["accuracy"]],
        "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
        "loss": [float(x) for x in history.history["loss"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
    }

    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)

    print(f"✅ Training history saved to: {history_path}")

    # Save class indices
    class_indices_path = models_dir / "class_indices.json"
    with open(class_indices_path, "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    print(f"✅ Class indices saved to: {class_indices_path}")

    # Print final metrics
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"\n📁 Models saved to: {models_dir.absolute()}")
    print(f"📊 TensorBoard logs: {logs_dir.absolute()}")
    print("\nTo view training progress in TensorBoard:")
    print(f"  tensorboard --logdir {logs_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Train Certificate Verification CNN")

    parser.add_argument(
        "--train-dir",
        type=str,
        default="training_data/train",
        help="Directory containing training data (default: training_data/train)",
    )

    parser.add_argument(
        "--val-dir",
        type=str,
        default="training_data/val",
        help="Directory containing validation data (default: training_data/val)",
    )

    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs (default: 50)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Initial learning rate (default: 0.001)",
    )

    args = parser.parse_args()

    # Train model
    train_model(args)


if __name__ == "__main__":
    main()
