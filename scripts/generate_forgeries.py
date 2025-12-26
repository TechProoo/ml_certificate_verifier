"""
Automated Forgery Generator

Creates synthetic forged certificates from authentic ones by applying
realistic forgery techniques:
- Text overlays (simulating edited names/dates/grades)
- Watermark degradation
- Color adjustments
- Noise and compression artifacts
- Rotation and distortion
- Blur effects

Usage:
    python scripts/generate_forgeries.py --input training_data/train/authentic --output training_data/train/forged --count 40
"""

import os
import sys
from pathlib import Path
import argparse
import random
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance


def apply_text_overlay(image: np.ndarray) -> np.ndarray:
    """
    Add fake text overlays to simulate edited names/grades.

    Args:
        image: Input image as numpy array

    Returns:
        Image with text overlays
    """
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Try to use a common font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", random.randint(20, 40))
    except:
        font = ImageFont.load_default()

    # Random text overlays (simulating edits)
    fake_texts = ["JOHN DOE", "A1", "PASS", "2024", "DISTINCTION", "MODIFIED", "EDITED"]

    # Add 1-3 random text overlays
    for _ in range(random.randint(1, 3)):
        text = random.choice(fake_texts)
        x = random.randint(50, pil_img.width - 150)
        y = random.randint(50, pil_img.height - 100)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.text((x, y), text, font=font, fill=color)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def degrade_watermark(image: np.ndarray) -> np.ndarray:
    """
    Remove or degrade watermarks by applying selective blurring.

    Args:
        image: Input image

    Returns:
        Image with degraded watermark
    """
    # Apply Gaussian blur to random regions (simulating watermark removal)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Create 2-4 random circular regions
    for _ in range(random.randint(2, 4)):
        center = (random.randint(0, image.shape[1]), random.randint(0, image.shape[0]))
        radius = random.randint(30, 100)
        cv2.circle(mask, center, radius, 255, -1)

    # Blur the masked regions
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    result = np.where(mask[:, :, np.newaxis] == 255, blurred, image)

    return result.astype(np.uint8)


def add_noise(image: np.ndarray) -> np.ndarray:
    """
    Add random noise to simulate low-quality scans.

    Args:
        image: Input image

    Returns:
        Noisy image
    """
    noise_type = random.choice(["gaussian", "salt_pepper", "speckle"])

    if noise_type == "gaussian":
        mean = 0
        sigma = random.uniform(10, 30)
        gaussian = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
        noisy = cv2.add(image, gaussian)

    elif noise_type == "salt_pepper":
        prob = random.uniform(0.01, 0.05)
        noisy = image.copy()
        # Salt
        salt_mask = np.random.random(image.shape[:2]) < prob / 2
        noisy[salt_mask] = 255
        # Pepper
        pepper_mask = np.random.random(image.shape[:2]) < prob / 2
        noisy[pepper_mask] = 0

    else:  # speckle
        noise = np.random.randn(*image.shape) * random.uniform(0.1, 0.3)
        noisy = image + image * noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return noisy


def adjust_colors(image: np.ndarray) -> np.ndarray:
    """
    Randomly adjust colors to simulate editing.

    Args:
        image: Input image

    Returns:
        Color-adjusted image
    """
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Random brightness adjustment
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(random.uniform(0.7, 1.3))

    # Random contrast adjustment
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(random.uniform(0.8, 1.4))

    # Random color adjustment
    enhancer = ImageEnhance.Color(pil_img)
    pil_img = enhancer.enhance(random.uniform(0.5, 1.5))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def apply_compression_artifacts(image: np.ndarray) -> np.ndarray:
    """
    Add JPEG compression artifacts.

    Args:
        image: Input image

    Returns:
        Compressed image
    """
    quality = random.randint(20, 60)

    # Encode as JPEG with low quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", image, encode_param)

    # Decode back
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return decoded


def apply_rotation_distortion(image: np.ndarray) -> np.ndarray:
    """
    Apply slight rotation and perspective distortion.

    Args:
        image: Input image

    Returns:
        Distorted image
    """
    h, w = image.shape[:2]

    # Random rotation
    angle = random.uniform(-5, 5)
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE
    )

    # Random perspective distortion
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32(
        [
            [random.randint(0, 20), random.randint(0, 20)],
            [w - random.randint(0, 20), random.randint(0, 20)],
            [random.randint(0, 20), h - random.randint(0, 20)],
            [w - random.randint(0, 20), h - random.randint(0, 20)],
        ]
    )

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    distorted = cv2.warpPerspective(
        rotated, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE
    )

    return distorted


def apply_blur(image: np.ndarray) -> np.ndarray:
    """
    Apply random blur effects.

    Args:
        image: Input image

    Returns:
        Blurred image
    """
    blur_type = random.choice(["gaussian", "motion", "median"])

    if blur_type == "gaussian":
        ksize = random.choice([3, 5, 7])
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    elif blur_type == "motion":
        size = random.randint(5, 15)
        kernel = np.zeros((size, size))
        kernel[int((size - 1) / 2), :] = np.ones(size)
        kernel = kernel / size
        return cv2.filter2D(image, -1, kernel)

    else:  # median
        ksize = random.choice([3, 5, 7])
        return cv2.medianBlur(image, ksize)


def generate_forgery(image_path: str, techniques: List[str]) -> np.ndarray:
    """
    Generate a forged certificate by applying multiple techniques.

    Args:
        image_path: Path to authentic certificate
        techniques: List of forgery techniques to apply

    Returns:
        Forged certificate image
    """
    # Load image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Apply selected techniques
    technique_functions = {
        "text_overlay": apply_text_overlay,
        "watermark_removal": degrade_watermark,
        "noise": add_noise,
        "color_adjustment": adjust_colors,
        "compression": apply_compression_artifacts,
        "distortion": apply_rotation_distortion,
        "blur": apply_blur,
    }

    for technique in techniques:
        if technique in technique_functions:
            image = technique_functions[technique](image)

    return image


def generate_forgeries(input_dir: str, output_dir: str, count: int):
    """
    Generate multiple forged certificates from authentic ones.

    Args:
        input_dir: Directory containing authentic certificates
        output_dir: Directory to save forged certificates
        count: Number of forgeries to generate
    """
    print("=" * 80)
    print("Automated Forgery Generator")
    print("=" * 80)
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Target Count: {count}")
    print("=" * 80)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get list of authentic certificates
    input_path = Path(input_dir)
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    authentic_images = [
        str(f) for f in input_path.iterdir() if f.suffix.lower() in image_extensions
    ]

    if not authentic_images:
        print(f"❌ ERROR: No images found in {input_dir}")
        sys.exit(1)

    print(f"✅ Found {len(authentic_images)} authentic certificates")
    print()

    # Define forgery technique combinations
    technique_sets = [
        ["text_overlay", "noise"],
        ["watermark_removal", "compression"],
        ["color_adjustment", "blur"],
        ["distortion", "noise", "compression"],
        ["text_overlay", "watermark_removal", "blur"],
        ["color_adjustment", "compression", "distortion"],
        ["noise", "blur", "compression"],
    ]

    # Generate forgeries
    generated = 0
    while generated < count:
        # Select random authentic certificate
        authentic_path = random.choice(authentic_images)

        # Select random technique combination
        techniques = random.choice(technique_sets)

        try:
            # Generate forgery
            forged_image = generate_forgery(authentic_path, techniques)

            # Save forged certificate
            output_filename = f"forged_{generated + 1:04d}.jpg"
            output_filepath = output_path / output_filename
            cv2.imwrite(str(output_filepath), forged_image)

            generated += 1
            techniques_str = ", ".join(techniques)
            print(f"[{generated}/{count}] Generated: {output_filename}")
            print(f"           Techniques: {techniques_str}")
            print(f"           Source: {Path(authentic_path).name}")
            print()

        except Exception as e:
            print(
                f"⚠️  Warning: Failed to generate forgery from {authentic_path}: {str(e)}"
            )
            continue

    print("=" * 80)
    print(f"✅ Successfully generated {generated} forged certificates")
    print(f"📁 Saved to: {output_path.absolute()}")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Review generated forgeries in the output directory")
    print("2. Manually adjust/delete any unrealistic forgeries")
    print("3. Retrain model with balanced dataset:")
    print(
        "   python scripts/train_model.py --train-dir training_data/train --val-dir training_data/val --epochs 30"
    )
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic forged certificates"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="training_data/train/authentic",
        help="Directory containing authentic certificates (default: training_data/train/authentic)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="training_data/train/forged",
        help="Directory to save forged certificates (default: training_data/train/forged)",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=40,
        help="Number of forgeries to generate (default: 40)",
    )

    args = parser.parse_args()

    # Generate forgeries
    generate_forgeries(args.input, args.output, args.count)


if __name__ == "__main__":
    main()
