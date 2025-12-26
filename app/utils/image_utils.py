import cv2
import numpy as np
from typing import Union, Tuple, List
from io import BytesIO

try:
    from pdf2image import convert_from_bytes

    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: pdf2image not installed. PDF support disabled.")


def pdf_to_images(pdf_bytes: bytes, dpi: int = 300) -> List[np.ndarray]:
    """
    Convert PDF to list of images (one per page).

    Args:
        pdf_bytes: PDF file as bytes
        dpi: Resolution for conversion (higher = better quality but slower)

    Returns:
        List of images as numpy arrays
    """
    if not PDF_SUPPORT:
        raise ImportError("pdf2image not installed. Run: pip install pdf2image")

    try:
        # Convert PDF pages to PIL images
        pil_images = convert_from_bytes(pdf_bytes, dpi=dpi)

        # Convert PIL images to OpenCV format (numpy arrays)
        cv_images = []
        for pil_img in pil_images:
            # Convert PIL to numpy array
            img_array = np.array(pil_img)
            # Convert RGB to BGR (OpenCV format)
            if len(img_array.shape) == 3:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_array
            cv_images.append(img_bgr)

        return cv_images
    except Exception as e:
        raise ValueError(f"Failed to convert PDF: {str(e)}")


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Load image from uploaded file bytes."""
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image data")

    return img


def load_image(source: Union[str, bytes]) -> np.ndarray:
    """Load image from file path or uploaded bytes."""
    if isinstance(source, bytes):
        return load_image_from_bytes(source)

    if isinstance(source, str):
        img = cv2.imread(source)
        if img is None:
            raise ValueError(f"Failed to load image from {source}")
        return img

    raise TypeError("Source must be file path (str) or bytes")


def preprocess_uploaded_certificate(
    image_bytes: bytes,
    target_size: Tuple[int, int] = (1024, 1024),
) -> np.ndarray:
    """
    Comprehensive preprocessing pipeline for certificate images.

    Args:
        image_bytes: Raw image bytes
        target_size: Target dimensions (width, height)

    Returns:
        Preprocessed image ready for ML models
    """

    # 1️⃣ Load image
    image = load_image_from_bytes(image_bytes)

    # 2️⃣ Resize while maintaining aspect ratio
    image = resize_with_aspect_ratio(image, target_size)

    # 3️⃣ Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 4️⃣ Denoise with advanced method
    denoised = cv2.fastNlMeansDenoising(
        gray, None, h=10, templateWindowSize=7, searchWindowSize=21
    )

    # 5️⃣ Enhance contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 6️⃣ Correct lighting and shadows
    enhanced = correct_lighting(enhanced)

    # 7️⃣ Deskew if needed
    deskewed = deskew_image(enhanced)

    return deskewed


def resize_with_aspect_ratio(
    image: np.ndarray, target_size: Tuple[int, int]
) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio, padding if necessary.

    Args:
        image: Input image
        target_size: Target dimensions (width, height)

    Returns:
        Resized image with padding
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize image with high-quality interpolation
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Create white canvas
    canvas = np.full(
        (target_h, target_w, 3) if len(image.shape) == 3 else (target_h, target_w),
        255,
        dtype=np.uint8,
    )

    # Center the image
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2

    if len(image.shape) == 3:
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    else:
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return canvas


def correct_lighting(image: np.ndarray) -> np.ndarray:
    """
    Correct uneven lighting and shadows in grayscale image.

    Args:
        image: Grayscale image

    Returns:
        Image with corrected lighting
    """
    # Apply morphological closing to estimate background
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    background = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=3)
    background = cv2.GaussianBlur(background, (0, 0), sigmaX=15, sigmaY=15)

    # Subtract background to normalize lighting
    corrected = cv2.divide(image, background, scale=255)

    return corrected


def deskew_image(image: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    """
    Detect and correct image skew using Hough Line Transform.

    Args:
        image: Grayscale image
        max_angle: Maximum rotation angle to attempt (degrees)

    Returns:
        Deskewed image
    """
    try:
        # Edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        # Detect lines using Hough Transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None or len(lines) == 0:
            return image

        # Calculate angles of detected lines
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            if abs(angle) <= max_angle:
                angles.append(angle)

        if not angles:
            return image

        # Use median angle to avoid outliers
        median_angle = np.median(angles)

        # Rotate image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return rotated
    except:
        # If deskewing fails, return original image
        return image


def normalize_for_ml(image: np.ndarray) -> np.ndarray:
    """
    Normalize image for ML model input (0-1 range).

    Args:
        image: Grayscale image (0-255)

    Returns:
        Normalized image (0-1)
    """
    return image.astype(np.float32) / 255.0


def preprocess_for_ml_model(
    image: Union[bytes, np.ndarray],
    target_size: Tuple[int, int] = (1024, 1024),
    normalize: bool = True,
) -> np.ndarray:
    """
    Complete preprocessing pipeline for ML models.
    Handles both raw bytes and numpy arrays.

    Args:
        image: Image as bytes or numpy array
        target_size: Target dimensions
        normalize: Whether to normalize to 0-1 range

    Returns:
        Preprocessed image ready for ML inference
    """
    # Load and preprocess if bytes
    if isinstance(image, bytes):
        processed = preprocess_uploaded_certificate(image, target_size)
    else:
        # Already a numpy array
        if len(image.shape) == 3:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            processed = image
        processed = cv2.resize(processed, target_size)

    # Normalize if requested
    if normalize:
        processed = normalize_for_ml(processed)

    return processed
