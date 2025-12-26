"""
Create a simple REST API endpoint for certificate verification using OCR
This will be integrated into the main ML service
"""

from fastapi import APIRouter, UploadFile, File
from typing import Dict, Any
import numpy as np
from PIL import Image
import io
import sys
import os

# Assuming this runs in the ml_service directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.models.ocr_extractor import get_ocr_extractor

router = APIRouter(prefix="/api/ocr", tags=["OCR Verification"])


@router.post("/verify-certificate")
async def verify_certificate(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Verify certificate authenticity using OCR text extraction.

    Args:
        file: Certificate image file (JPG, PNG, etc.)

    Returns:
        {
            "authenticity_score": 0.0-1.0,
            "is_authentic": True/False,
            "confidence": 0.0-1.0,
            "extracted_data": {
                "candidate_name": str,
                "exam_number": str,
                "exam_year": str,
                "subjects": [...],
                ...
            },
            "validation": {
                "is_valid": True/False,
                "validation_score": 0.0-1.0,
                "anomalies": [...]
            }
        }
    """
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Get OCR extractor
        extractor = get_ocr_extractor()

        # Extract certificate data
        data = extractor.extract_certificate_data(image)

        # Validate data
        validation = extractor.validate_certificate_data(data)

        # Calculate authenticity score based on validation
        # Mapping validation score to authenticity score (inverse relationship)
        validation_score = validation.get("validation_score", 0)

        if validation_score > 0.7:
            authenticity_score = 0.9  # Likely authentic
            is_authentic = True
        elif validation_score > 0.4:
            authenticity_score = 0.5  # Uncertain
            is_authentic = None  # Review needed
        else:
            authenticity_score = 0.1  # Likely forged
            is_authentic = False

        return {
            "authenticity_score": authenticity_score,
            "is_authentic": is_authentic,
            "confidence": data.get("confidence", 0),
            "extracted_data": {
                "candidate_name": data.get("full_name"),
                "exam_number": data.get("exam_number"),
                "center_number": data.get("center_number"),
                "exam_year": data.get("exam_year"),
                "exam_month": data.get("exam_month"),
                "subjects": data.get("subjects", []),
                "serial_number": data.get("serial_number"),
                "candidate_number": data.get("candidate_number"),
            },
            "validation": {
                "is_valid": validation.get("is_valid"),
                "validation_score": validation.get("validation_score"),
                "anomalies": validation.get("anomalies", []),
            },
            "raw_text": data.get("raw_text", "")[:500],  # First 500 chars for debugging
        }

    except Exception as e:
        return {
            "error": str(e),
            "authenticity_score": 0,
            "is_authentic": False,
            "confidence": 0,
        }


@router.post("/extract-text")
async def extract_text(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Extract raw text from certificate image.

    Args:
        file: Certificate image file

    Returns:
        {
            "raw_text": str,
            "text_length": int,
            "confidence": 0.0-1.0
        }
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        extractor = get_ocr_extractor()
        text = extractor.extract_text(np.array(image))
        confidence = extractor._calculate_confidence(text)

        return {
            "raw_text": text,
            "text_length": len(text),
            "confidence": confidence,
        }

    except Exception as e:
        return {"error": str(e), "raw_text": "", "text_length": 0, "confidence": 0}


if __name__ == "__main__":
    # This file can be imported and used as a router in the main FastAPI app
    print("OCR Verification Router created successfully")
    print("Use: from scripts.ocr_router import router")
    print("Then add to app: app.include_router(router)")
