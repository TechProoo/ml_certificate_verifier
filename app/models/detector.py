import os
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. CNN detection will use fallback mode.")

logger = logging.getLogger(__name__)


class CertificateDetector:
    """
    CNN-based certificate forgery detector using transfer learning.
    Uses MobileNetV2 for efficient inference on certificate images.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the certificate detector.

        Args:
            model_path: Path to trained model file (.h5 or .keras)
                       If None, looks for default model in models/ directory
        """
        self.input_size = (224, 224)
        self.num_classes = None  # Will be set when model loads
        self.class_names = None  # Will be detected from training data
        self.model = None
        self.model_loaded = False

        # Confidence thresholds for SUSPICIOUS classification
        self.suspicious_threshold = 0.6  # If max confidence < 60%, mark as suspicious

        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available. Using fallback predictions.")
            return

        # Determine model path
        if model_path is None:
            # Look for default model in models/ directory
            base_dir = Path(__file__).parent.parent.parent
            models_dir = base_dir / "models"
            model_path = str(models_dir / "certificate_detector_v1.h5")

        # Try to load existing model
        if os.path.exists(model_path):
            try:
                self.load_model(model_path)
                logger.info(f"Loaded trained model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {str(e)}")
                logger.info(
                    "Using untrained model. Train the model for better results."
                )
                self.model = self._create_model()
        else:
            logger.warning(f"No trained model found at {model_path}")
            logger.info("Using untrained model. Train the model for better results.")
            self.model = self._create_model()

    def _create_model(self, num_classes: int = 2) -> Model:
        """
        Create CNN model using MobileNetV2 transfer learning.

        Args:
            num_classes: Number of output classes (2 for AUTHENTIC/FORGED, 3 for AUTHENTIC/SUSPICIOUS/FORGED)

        Returns:
            Compiled Keras model
        """
        if not TF_AVAILABLE:
            return None

        self.num_classes = num_classes
        if num_classes == 2:
            self.class_names = ["AUTHENTIC", "FORGED"]
        else:
            self.class_names = ["AUTHENTIC", "SUSPICIOUS", "FORGED"]

        # Load pre-trained MobileNetV2 (without top classification layer)
        base_model = MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet",  # Use ImageNet pre-trained weights
        )

        # Freeze base model layers initially
        base_model.trainable = False

        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation="relu", name="dense_1")(x)
        x = Dropout(0.5, name="dropout_1")(x)
        x = Dense(128, activation="relu", name="dense_2")(x)
        x = Dropout(0.3, name="dropout_2")(x)
        predictions = Dense(num_classes, activation="softmax", name="output")(x)

        # Create final model
        model = Model(inputs=base_model.input, outputs=predictions)

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
        )

        return model

    def load_model(self, model_path: str):
        """
        Load trained model from file.

        Args:
            model_path: Path to saved model file
        """
        if not TF_AVAILABLE:
            logger.error("Cannot load model: TensorFlow not available")
            return

        try:
            self.model = keras.models.load_model(model_path)
            self.model_loaded = True

            # Detect number of classes from model output
            output_shape = self.model.output_shape[-1]
            self.num_classes = output_shape

            if self.num_classes == 2:
                self.class_names = ["AUTHENTIC", "FORGED"]
                logger.info("Loaded 2-class model (AUTHENTIC/FORGED)")
            else:
                self.class_names = ["AUTHENTIC", "SUSPICIOUS", "FORGED"]
                logger.info("Loaded 3-class model (AUTHENTIC/SUSPICIOUS/FORGED)")

            logger.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def save_model(self, save_path: str):
        """
        Save trained model to file.

        Args:
            save_path: Path where model should be saved
        """
        if self.model is None:
            raise ValueError("No model to save")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)
        logger.info(f"Model saved to {save_path}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for CNN input.

        Args:
            image: Input image as numpy array (can be grayscale or color)

        Returns:
            Preprocessed image ready for model input (224x224x3, normalized)
        """
        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        # Resize to model input size
        import cv2

        image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LANCZOS4)

        # Normalize pixel values to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Ensure shape is (224, 224, 3)
        if image.shape != (224, 224, 3):
            raise ValueError(f"Invalid image shape after preprocessing: {image.shape}")

        return image

    def predict(self, image: np.ndarray) -> Dict:
        """
        Predict certificate authenticity.

        Args:
            image: Certificate image - can be PIL Image or numpy array (grayscale or color)

        Returns:
            Dictionary with prediction results:
            {
                'confidence': float (0-100),
                'authenticity': str ('AUTHENTIC', 'SUSPICIOUS', 'FORGED'),
                'class_probabilities': dict,
                'model_trained': bool
            }
        """
        if not TF_AVAILABLE or self.model is None:
            # Fallback to random predictions
            return self._fallback_prediction()

        try:
            # Handle PIL Image conversion
            from PIL import Image

            if isinstance(image, Image.Image):
                image = np.array(image)

            # Preprocess image
            processed = self.preprocess_image(image)

            # Add batch dimension
            batch = np.expand_dims(processed, axis=0)

            # Get predictions
            predictions = self.model.predict(batch, verbose=0)[0]

            # Handle 2-class vs 3-class models
            if self.num_classes == 2:
                # 2-class model: AUTHENTIC, FORGED
                authentic_prob = float(predictions[0] * 100)
                forged_prob = float(predictions[1] * 100)

                class_probs = {
                    "authentic": authentic_prob,
                    "forged": forged_prob,
                    "suspicious": 0.0,  # Not trained separately
                }

                # Get max confidence
                max_confidence = max(authentic_prob, forged_prob)

                # If confidence is low, mark as SUSPICIOUS
                if max_confidence < (self.suspicious_threshold * 100):
                    authenticity = "SUSPICIOUS"
                    confidence = 100 - max_confidence  # Uncertainty score
                    class_probs["suspicious"] = confidence
                else:
                    # High confidence prediction
                    predicted_class_idx = np.argmax(predictions)
                    authenticity = self.class_names[predicted_class_idx]
                    confidence = max_confidence
            else:
                # 3-class model: AUTHENTIC, SUSPICIOUS, FORGED
                class_probs = {
                    "authentic": float(predictions[0] * 100),
                    "suspicious": float(predictions[1] * 100),
                    "forged": float(predictions[2] * 100),
                }

                predicted_class_idx = np.argmax(predictions)
                authenticity = self.class_names[predicted_class_idx]
                confidence = float(predictions[predicted_class_idx] * 100)

            return {
                "confidence": round(confidence, 2),
                "authenticity": authenticity,
                "class_probabilities": class_probs,
                "model_trained": self.model_loaded,
                "prediction_method": f"CNN (MobileNetV2) - {self.num_classes}-class",
                "model_type": f"{self.num_classes}-class",
            }

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return self._fallback_prediction()

    def _fallback_prediction(self) -> Dict:
        """
        Generate random predictions when model is not available.
        Used for testing before model training.

        Returns:
            Dictionary with random prediction results
        """
        import random

        # Generate random probabilities
        probs = np.random.dirichlet([1, 1, 1])

        class_probs = {
            "authentic": float(probs[0] * 100),
            "suspicious": float(probs[1] * 100),
            "forged": float(probs[2] * 100),
        }

        # Pick class with highest probability
        predicted_idx = np.argmax(probs)
        authenticity = self.class_names[predicted_idx]
        confidence = float(probs[predicted_idx] * 100)

        return {
            "confidence": round(confidence, 2),
            "authenticity": authenticity,
            "class_probabilities": class_probs,
            "model_trained": False,
            "prediction_method": "Random (Model not trained)",
            "warning": "Using untrained model. Train CNN for accurate predictions.",
        }

    def predict_batch(self, images: list) -> list:
        """
        Predict authenticity for multiple certificates.

        Args:
            images: List of preprocessed certificate images

        Returns:
            List of prediction dictionaries
        """
        return [self.predict(img) for img in images]

    def get_model_summary(self) -> str:
        """
        Get model architecture summary.

        Returns:
            String representation of model architecture
        """
        if self.model is None:
            return "No model loaded"

        import io

        buffer = io.StringIO()
        self.model.summary(print_fn=lambda x: buffer.write(x + "\n"))
        return buffer.getvalue()


# Global detector instance (lazy loading)
_detector_instance = None


def get_detector() -> CertificateDetector:
    """
    Get global detector instance (singleton pattern).

    Returns:
        CertificateDetector instance
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = CertificateDetector()
    return _detector_instance
