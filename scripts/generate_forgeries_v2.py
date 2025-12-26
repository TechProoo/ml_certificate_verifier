"""
Enhanced Automated Forgery Generator V2

Creates more challenging synthetic forged certificates with advanced techniques:
- Subtle text modifications (matching fonts/colors)
- Smart watermark cloning/removal
- Realistic document scanning artifacts
- Copy-paste forgery (cloning parts from other certificates)
- Edge artifacts from content-aware fill
- Micro-pattern disruption
- Subtle color inconsistencies
- Mixed authentic/forged regions

Usage:
    python scripts/generate_forgeries_v2.py --input training_data/train/authentic --output training_data/train/forged --count 40
"""

import os
import sys
from pathlib import Path
import argparse
import random
from typing import List, Tuple, Optional
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps


def extract_document_colors(image: np.ndarray) -> dict:
    """Extract dominant colors from document for realistic text matching."""
    # Convert to RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Sample common text regions (middle sections)
    h, w = image.shape[:2]
    sample_region = rgb[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]

    # Get dark colors (likely text)
    dark_mask = cv2.cvtColor(sample_region, cv2.COLOR_RGB2GRAY) < 100
    if dark_mask.any():
        dark_pixels = sample_region[dark_mask]
        text_color = np.median(dark_pixels, axis=0).astype(int)
    else:
        text_color = np.array([0, 0, 0])

    # Get light colors (likely background)
    light_mask = cv2.cvtColor(sample_region, cv2.COLOR_RGB2GRAY) > 200
    if light_mask.any():
        light_pixels = sample_region[light_mask]
        bg_color = np.median(light_pixels, axis=0).astype(int)
    else:
        bg_color = np.array([255, 255, 255])

    return {"text": tuple(text_color), "background": tuple(bg_color)}


def apply_subtle_text_edit(image: np.ndarray) -> np.ndarray:
    """
    Apply subtle text edits that match document style.
    More realistic than obvious overlays.
    """
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img, "RGBA")

    # Extract document colors
    colors = extract_document_colors(image)

    # Try multiple fonts for better matching
    font_names = ["arial.ttf", "times.ttf", "calibri.ttf", "consola.ttf"]
    font = None
    for font_name in font_names:
        try:
            font = ImageFont.truetype(font_name, random.randint(14, 28))
            break
        except:
            continue
    if font is None:
        font = ImageFont.load_default()

    # Realistic altered fields
    realistic_edits = [
        ("A1", "B2"),
        ("A2", "C4"),
        ("PASS", "FAIL"),
        ("2023", "2024"),
        ("CREDIT", "PASS"),
        ("80", "45"),
        ("90", "55"),
        ("75", "60"),
    ]

    # Apply 1-2 subtle edits
    for _ in range(random.randint(1, 2)):
        old_text, new_text = random.choice(realistic_edits)

        # Random position (more focused on central regions)
        x = random.randint(pil_img.width // 4, 3 * pil_img.width // 4)
        y = random.randint(pil_img.height // 4, 3 * pil_img.height // 4)

        # First, try to "erase" by drawing background color rectangle
        bbox = draw.textbbox((x, y), old_text, font=font)
        draw.rectangle(bbox, fill=colors["background"] + (255,))

        # Then draw new text with slight opacity variation
        text_color = colors["text"] + (random.randint(230, 255),)
        draw.text((x, y), new_text, font=font, fill=text_color)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def apply_copy_paste_forgery(image: np.ndarray, source_images: List[str]) -> np.ndarray:
    """
    Clone regions from other certificates (realistic forgery technique).
    """
    if not source_images or random.random() > 0.5:
        return image

    try:
        # Load a random source certificate
        source_path = random.choice(source_images)
        source = cv2.imread(source_path)

        if source is None:
            return image

        # Resize source to match target dimensions
        source = cv2.resize(source, (image.shape[1], image.shape[0]))

        # Copy 1-2 rectangular regions
        for _ in range(random.randint(1, 2)):
            # Source region
            src_h, src_w = source.shape[:2]
            region_w = random.randint(src_w // 8, src_w // 4)
            region_h = random.randint(src_h // 10, src_h // 6)
            src_x = random.randint(0, src_w - region_w)
            src_y = random.randint(0, src_h - region_h)

            # Target position
            dst_x = random.randint(0, image.shape[1] - region_w)
            dst_y = random.randint(0, image.shape[0] - region_h)

            # Copy region
            copied_region = source[
                src_y : src_y + region_h, src_x : src_x + region_w
            ].copy()

            # Blend edges for seamlessness
            mask = np.ones((region_h, region_w), dtype=np.float32)
            feather = min(10, region_h // 4, region_w // 4)
            for i in range(feather):
                alpha = i / feather
                mask[i, :] *= alpha
                mask[-i - 1, :] *= alpha
                mask[:, i] *= alpha
                mask[:, -i - 1] *= alpha

            # Apply blended copy
            for c in range(3):
                image[dst_y : dst_y + region_h, dst_x : dst_x + region_w, c] = (
                    image[dst_y : dst_y + region_h, dst_x : dst_x + region_w, c]
                    * (1 - mask[:, :, np.newaxis])
                    + copied_region[:, :, c : c + 1] * mask[:, :, np.newaxis]
                ).astype(np.uint8)[:, :, 0]

        return image
    except Exception as e:
        return image


def apply_smart_watermark_removal(image: np.ndarray) -> np.ndarray:
    """
    Smart watermark removal using inpainting (more realistic).
    """
    # Detect potential watermark regions (semi-transparent areas)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create mask for potential watermark areas
    # Look for areas with specific intensity ranges
    mask = np.zeros(gray.shape, dtype=np.uint8)

    # Multiple small circular regions (simulating watermark removal attempts)
    for _ in range(random.randint(3, 8)):
        center = (
            random.randint(50, image.shape[1] - 50),
            random.randint(50, image.shape[0] - 50),
        )
        radius = random.randint(15, 50)
        cv2.circle(mask, center, radius, 255, -1)

    # Apply inpainting to "remove" watermark
    inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    # Blend with original for subtle effect
    alpha = random.uniform(0.6, 0.9)
    result = cv2.addWeighted(inpainted, alpha, image, 1 - alpha, 0)

    return result


def add_scanning_artifacts(image: np.ndarray) -> np.ndarray:
    """
    Add realistic document scanning artifacts.
    """
    # Add slight scan lines
    if random.random() > 0.5:
        for i in range(0, image.shape[0], random.randint(20, 50)):
            noise_line = np.random.randint(-5, 5, image.shape[1])
            if i < image.shape[0]:
                image[i, :] = np.clip(
                    image[i, :].astype(int) + noise_line[:, np.newaxis], 0, 255
                ).astype(np.uint8)

    # Add slight non-uniform lighting
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2

    y, x = np.ogrid[:h, :w]
    # Create vignette effect (darker at edges)
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    vignette = 1 - (distance / max_distance) * random.uniform(0.1, 0.3)

    # Apply vignette
    for c in range(3):
        image[:, :, c] = np.clip(image[:, :, c] * vignette, 0, 255).astype(np.uint8)

    return image


def add_jpeg_ghosting(image: np.ndarray) -> np.ndarray:
    """
    Add JPEG compression artifacts that appear when documents are
    repeatedly saved (evidence of editing).
    """
    # Multiple compression cycles
    for _ in range(random.randint(2, 4)):
        quality = random.randint(40, 70)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode(".jpg", image, encode_param)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return image


def add_edge_inconsistencies(image: np.ndarray) -> np.ndarray:
    """
    Add subtle inconsistencies at edges of edited regions.
    """
    # Find edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Dilate edges slightly
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Add slight noise/artifacts along edges
    noise = np.random.randint(-15, 15, image.shape).astype(np.int16)
    edge_mask = edges[:, :, np.newaxis] > 0

    # Apply noise only along edges
    image = image.astype(np.int16)
    image[edge_mask] += noise[edge_mask]
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def add_color_inconsistency(image: np.ndarray) -> np.ndarray:
    """
    Add subtle color inconsistencies in different regions.
    """
    # Divide image into regions
    h, w = image.shape[:2]
    num_regions = random.randint(3, 6)

    # Apply slight color shifts to random regions
    for _ in range(num_regions):
        # Random rectangular region
        x1 = random.randint(0, w // 2)
        y1 = random.randint(0, h // 2)
        x2 = random.randint(x1 + w // 4, w)
        y2 = random.randint(y1 + h // 4, h)

        # Random color shift
        shift = np.random.randint(-10, 10, (1, 1, 3)).astype(np.int16)

        # Create smooth transition mask
        mask = np.zeros((h, w), dtype=np.float32)
        mask[y1:y2, x1:x2] = 1.0
        mask = cv2.GaussianBlur(mask, (51, 51), 20)

        # Apply shift
        image = image.astype(np.int16)
        for c in range(3):
            image[:, :, c] += (shift[0, 0, c] * mask).astype(np.int16)
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def add_subtle_noise_patterns(image: np.ndarray) -> np.ndarray:
    """
    Add very subtle noise that's hard for models to detect.
    """
    # Add perlin-like noise (more organic)
    noise = np.random.randn(*image.shape) * random.uniform(2, 8)
    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    # Add random micro-patterns
    if random.random() > 0.5:
        pattern = np.random.randint(0, 3, image.shape[:2], dtype=np.int8)
        for c in range(3):
            noisy[:, :, c] = np.clip(
                noisy[:, :, c].astype(np.int16) + pattern, 0, 255
            ).astype(np.uint8)

    return noisy


def apply_selective_blur(image: np.ndarray) -> np.ndarray:
    """
    Apply blur only to suspicious regions (simulating content-aware editing).
    """
    # Create random mask for regions to blur
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Add 2-4 regions
    for _ in range(random.randint(2, 4)):
        # Elliptical regions
        center = (
            random.randint(w // 4, 3 * w // 4),
            random.randint(h // 4, 3 * h // 4),
        )
        axes = (random.randint(30, 100), random.randint(20, 60))
        angle = random.randint(0, 180)
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)

    # Smooth mask edges
    mask = cv2.GaussianBlur(mask, (31, 31), 15)
    mask_normalized = mask.astype(np.float32) / 255.0

    # Apply blur
    blurred = cv2.GaussianBlur(image, (11, 11), 0)

    # Blend using mask
    result = (
        image * (1 - mask_normalized[:, :, np.newaxis])
        + blurred * mask_normalized[:, :, np.newaxis]
    ).astype(np.uint8)

    return result


def generate_forgery_v2(
    image_path: str, all_images: List[str], difficulty: str = "medium"
) -> np.ndarray:
    """
    Generate a challenging forged certificate.

    Args:
        image_path: Path to authentic certificate
        all_images: List of all authentic images (for copy-paste forgery)
        difficulty: 'easy', 'medium', 'hard' - determines number/subtlety of techniques

    Returns:
        Forged certificate image
    """
    # Load image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Define technique pools
    subtle_techniques = [
        add_subtle_noise_patterns,
        add_color_inconsistency,
        add_jpeg_ghosting,
        add_scanning_artifacts,
    ]

    moderate_techniques = [
        apply_smart_watermark_removal,
        apply_selective_blur,
        add_edge_inconsistencies,
    ]

    aggressive_techniques = [
        apply_subtle_text_edit,
        lambda img: apply_copy_paste_forgery(img, all_images),
    ]

    # Select techniques based on difficulty
    if difficulty == "easy":
        # More obvious forgeries
        num_techniques = random.randint(2, 4)
        selected = (
            random.sample(subtle_techniques, min(1, len(subtle_techniques)))
            + random.sample(moderate_techniques, min(1, len(moderate_techniques)))
            + random.sample(aggressive_techniques, min(1, len(aggressive_techniques)))
        )

    elif difficulty == "hard":
        # Very subtle forgeries (hardest to detect)
        num_techniques = random.randint(1, 3)
        selected = random.sample(
            subtle_techniques + moderate_techniques,
            min(num_techniques, len(subtle_techniques) + len(moderate_techniques)),
        )

    else:  # medium (default)
        # Mix of techniques
        num_techniques = random.randint(2, 4)
        all_techniques = subtle_techniques + moderate_techniques + aggressive_techniques
        selected = random.sample(
            all_techniques, min(num_techniques, len(all_techniques))
        )

    # Apply selected techniques
    for technique in selected:
        try:
            image = technique(image)
        except Exception as e:
            print(f"Warning: Technique failed: {e}")
            continue

    return image


def generate_forgeries_v2(
    input_dir: str, output_dir: str, count: int, difficulty: str = "medium"
):
    """
    Generate multiple challenging forged certificates.

    Args:
        input_dir: Directory containing authentic certificates
        output_dir: Directory to save forged certificates
        count: Number of forgeries to generate
        difficulty: 'easy', 'medium', 'hard'
    """
    print("=" * 80)
    print("Enhanced Forgery Generator V2")
    print("=" * 80)
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Target Count: {count}")
    print(f"Difficulty Level: {difficulty}")
    print("=" * 80)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get list of authentic certificates
    input_path = Path(input_dir)
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    authentic_images = []
    for ext in image_extensions:
        authentic_images.extend(list(input_path.glob(f"*{ext}")))
        authentic_images.extend(list(input_path.glob(f"*{ext.upper()}")))

    authentic_images = [str(p) for p in authentic_images]

    if not authentic_images:
        print(f"❌ No images found in {input_dir}")
        return

    print(f"✅ Found {len(authentic_images)} authentic certificates")
    print("\n🔨 Generating forgeries...")

    # Generate forgeries
    metadata = []

    for i in range(count):
        # Select random authentic image
        source_image = random.choice(authentic_images)

        # Randomly select difficulty for variety
        if difficulty == "mixed":
            selected_difficulty = random.choice(["easy", "medium", "hard"])
        else:
            selected_difficulty = difficulty

        try:
            # Generate forgery
            forged = generate_forgery_v2(
                source_image, authentic_images, selected_difficulty
            )

            # Save forgery
            output_file = output_path / f"forged_{i+1:04d}.jpg"
            cv2.imwrite(str(output_file), forged, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Save metadata
            metadata.append(
                {
                    "forged_file": str(output_file.name),
                    "source_file": Path(source_image).name,
                    "difficulty": selected_difficulty,
                }
            )

            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{count} forgeries...")

        except Exception as e:
            print(f"  ❌ Failed to generate forgery {i+1}: {e}")
            continue

    # Save metadata
    metadata_file = output_path / "forgery_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Successfully generated {len(metadata)} forgeries")
    print(f"📊 Metadata saved to: {metadata_file}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Generate challenging forged certificates V2"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input directory with authentic certificates",
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output directory for forged certificates"
    )
    parser.add_argument(
        "--count", "-c", type=int, default=40, help="Number of forgeries to generate"
    )
    parser.add_argument(
        "--difficulty",
        "-d",
        choices=["easy", "medium", "hard", "mixed"],
        default="medium",
        help="Forgery difficulty level",
    )

    args = parser.parse_args()

    generate_forgeries_v2(args.input, args.output, args.count, args.difficulty)


if __name__ == "__main__":
    main()
