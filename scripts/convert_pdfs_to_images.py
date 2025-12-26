"""Convert PDF certificates to images for training"""

import os
import sys
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io


def convert_pdfs_in_directory(pdf_dir: str, delete_pdfs: bool = False):
    """
    Convert all PDFs in a directory to JPG images.

    Args:
        pdf_dir: Directory containing PDF files
        delete_pdfs: Whether to delete PDFs after conversion
    """
    pdf_dir = Path(pdf_dir)

    if not pdf_dir.exists():
        print(f"❌ Directory not found: {pdf_dir}")
        return

    # Find all PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return

    print(f"\n{'='*70}")
    print(f"Converting {len(pdf_files)} PDFs to images")
    print(f"Directory: {pdf_dir}")
    print(f"{'='*70}\n")

    converted = 0
    failed = 0

    for pdf_file in pdf_files:
        try:
            print(f"Converting: {pdf_file.name}...", end=" ")

            # Open PDF with PyMuPDF
            pdf_document = fitz.open(str(pdf_file))

            if len(pdf_document) == 0:
                print("❌ No pages found")
                failed += 1
                continue

            # Get the first page
            page = pdf_document[0]

            # Render page to pixmap at 200 DPI
            zoom = 200 / 72  # 72 is default DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Convert pixmap to PIL Image
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))

            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Save as JPG with same base name
            output_path = pdf_file.with_suffix(".jpg")
            image.save(output_path, "JPEG", quality=95, optimize=True)

            print(f"✅ Saved as {output_path.name}")
            converted += 1

            # Close PDF
            pdf_document.close()

            # Delete PDF if requested
            if delete_pdfs:
                pdf_file.unlink()
                print(f"   🗑️  Deleted {pdf_file.name}")

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            failed += 1

    print(f"\n{'='*70}")
    print(f"Conversion Complete!")
    print(f"✅ Converted: {converted}")
    print(f"❌ Failed: {failed}")
    print(f"{'='*70}\n")


def main():
    # Convert PDFs in training forged directory
    train_forged = "training_data/train/forged"
    val_forged = "training_data/val/forged"

    print("\n🔄 Converting Training Forged PDFs...")
    convert_pdfs_in_directory(train_forged, delete_pdfs=True)

    print("\n🔄 Converting Validation Forged PDFs...")
    convert_pdfs_in_directory(val_forged, delete_pdfs=True)

    # Count final dataset
    print("\n📊 Final Dataset Summary:")
    print("=" * 70)

    train_auth = Path("training_data/train/authentic")
    train_forge = Path("training_data/train/forged")
    val_auth = Path("training_data/val/authentic")
    val_forge = Path("training_data/val/forged")

    train_auth_count = len(
        [
            f
            for f in train_auth.glob("*")
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
        ]
    )
    train_forge_count = len(
        [
            f
            for f in train_forge.glob("*")
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
        ]
    )
    val_auth_count = len(
        [
            f
            for f in val_auth.glob("*")
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
        ]
    )
    val_forge_count = len(
        [
            f
            for f in val_forge.glob("*")
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
        ]
    )

    print(f"Training Set:")
    print(f"  Authentic: {train_auth_count}")
    print(f"  Forged: {train_forge_count}")
    print(f"  Total: {train_auth_count + train_forge_count}")
    print(f"\nValidation Set:")
    print(f"  Authentic: {val_auth_count}")
    print(f"  Forged: {val_forge_count}")
    print(f"  Total: {val_auth_count + val_forge_count}")
    print(
        f"\nGrand Total: {train_auth_count + train_forge_count + val_auth_count + val_forge_count}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
