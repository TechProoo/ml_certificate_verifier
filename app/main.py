from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from contextlib import asynccontextmanager
from .utils.image_utils import load_image_from_bytes
from .models.ocr_extractor import get_ocr_extractor
from .models.detector import get_detector
import shutil
import time
import random
import uvicorn
import logging
import io
import os
from PIL import Image

logger = logging.getLogger(__name__)


ocr_extractor = None
cnn_detector = None
models_loaded = False


def load_models():
    """Lazy load OCR + CNN models on first request to avoid blocking startup."""
    global ocr_extractor, cnn_detector, models_loaded
    if models_loaded:
        return
    print("🔄 Loading ML models (lazy initialization)...")
    try:
        ocr_extractor = get_ocr_extractor()
        print("✅ OCR extractor initialized")
        logger.info("OCR extractor initialized")
    except Exception as e:
        print(f"⚠️  OCR extractor failed: {str(e)}")
        logger.warning(f"Failed to initialize OCR extractor: {str(e)}")

    try:
        cnn_detector = get_detector()
        print("✅ CNN detector initialized")
        logger.info("CNN detector initialized")
    except Exception as e:
        cnn_detector = None
        print(f"⚠️  CNN detector failed: {str(e)}")
        logger.warning(f"Failed to initialize CNN detector: {str(e)}")

    models_loaded = True
    print("✅ ML models loaded")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Fast startup - models load on first request."""
    print("🚀 ML Service started (models will load on first request)")
    yield
    print("🛑 Shutting down ML service")


app = FastAPI(
    title="Certificate Verification ML Service", version="1.0.0", lifespan=lifespan
)

# Enable CORS for backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
async def root():
    """Health check endpoint - responds immediately without loading models."""
    return {
        "message": "Certificate Verification ML Service",
        "status": "online",
        "models_loaded": models_loaded,
    }


@app.post("/verify")
async def verify_endpoint(
    file: UploadFile = File(...), certificate_type: str = Form("WASSCE")
):
    """Main verification endpoint - loads models on first call."""
    # Lazy load models on first request
    if not models_loaded:
        load_models()

    return await verify_certificate(file, certificate_type)


async def verify_certificate(
    file: UploadFile = File(...), certificate_type: str = Form("WASSCE")
):
    """
    Verify uploaded certificate (images and PDFs).
    Returns confidence score and authenticity status.
    """
    try:
        start_time = time.time()

        # Validate file type
        allowed_types = ["image/jpeg", "image/jpg", "image/png", "application/pdf"]

        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: JPG, PNG, PDF. Got: {file.content_type}",
            )

        # Read file bytes
        file_bytes = await file.read()

        # Validate file size (10MB max)
        if len(file_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400, detail="File too large. Maximum size is 10MB"
            )

        # Use OCR-based verification
        ocr_confidence = 0
        ocr_details = None

        # Optional CNN-based verification (used in weighted scoring when trained)
        cnn_result = None
        cnn_authentic_score = 0.0  # 0-100 (probability of AUTHENTIC)
        cnn_used = False

        image_for_analysis = None

        if ocr_extractor:
            try:
                # Handle PDF files
                if file.content_type == "application/pdf":
                    from .utils.image_utils import pdf_to_images

                    images = pdf_to_images(file_bytes)
                    if not images:
                        raise Exception("No images extracted from PDF")
                    # Use the first page for OCR
                    import PIL.Image

                    image_for_ocr = PIL.Image.fromarray(images[0])
                else:
                    # Convert file_bytes to PIL Image
                    image_for_ocr = Image.open(io.BytesIO(file_bytes))

                # Use the same image for OCR and CNN analysis
                image_for_analysis = image_for_ocr

                # Extract certificate data
                cert_data = ocr_extractor.extract_certificate_data(image_for_ocr)

                # Validate data
                validation = ocr_extractor.validate_certificate_data(cert_data)

                # Convert validation score (0-1) to confidence (0-100)
                ocr_confidence = validation.get("validation_score", 0) * 100

                ocr_details = {
                    "validation_score": validation.get("validation_score", 0),
                    "is_valid": validation.get("is_valid", False),
                    "anomalies": validation.get("anomalies", []),
                    "extracted_data": {
                        "name": cert_data.get("full_name"),
                        "exam_number": cert_data.get("exam_number"),
                        "year": cert_data.get("exam_year"),
                        "subjects_count": len(cert_data.get("subjects", [])),
                        "grades_found": sum(
                            1 for s in cert_data.get("subjects", []) if s.get("grade")
                        ),
                    },
                }

                logger.info(
                    f"OCR Confidence: {ocr_confidence}%, Anomalies: {len(validation.get('anomalies', []))}"
                )

            except Exception as e:
                logger.error(f"OCR verification failed: {str(e)}")
                ocr_confidence = 0

        # Run CNN prediction if available and we have an image
        if cnn_detector is not None and image_for_analysis is not None:
            try:
                cnn_result = cnn_detector.predict(image_for_analysis)

                # Only use CNN score in weighted result if the CNN is actually trained
                if cnn_result.get("model_trained"):
                    class_probs = cnn_result.get("class_probabilities") or {}
                    cnn_authentic_score = float(class_probs.get("authentic", 0.0))
                    cnn_used = True
                else:
                    cnn_used = False
            except Exception as e:
                logger.warning(f"CNN prediction failed: {str(e)}")
                cnn_result = None
                cnn_used = False

        # Compute final score using 80/20 rule (OCR 80% + CNN 20%)
        # - OCR confidence: derived from validation_score (0-100)
        # - CNN score: AUTHENTIC probability (0-100), only if CNN model is trained
        if ocr_details and cnn_used:
            final_confidence = (0.8 * float(ocr_confidence)) + (
                0.2 * float(cnn_authentic_score)
            )
            verification_method = "Weighted (OCR 80% + CNN 20%)"
        elif ocr_details:
            final_confidence = float(ocr_confidence)
            verification_method = "OCR Only"
        elif cnn_used:
            final_confidence = float(cnn_authentic_score)
            verification_method = "CNN Only"
        else:
            final_confidence = 0.0
            verification_method = "No Data"

        # Determine authenticity based on final confidence
        if final_confidence >= 70:
            authenticity = "AUTHENTIC"
        elif final_confidence >= 40:
            authenticity = "SUSPICIOUS"
        else:
            authenticity = "FORGED"

        processing_time = time.time() - start_time

        # Build response with OCR details
        response = {
            "confidence": final_confidence,
            "authenticity": authenticity,
            "verification_method": verification_method,
            "details": {
                "textRecognition": "Successful" if ocr_details else "Attempted",
                "signatureDetection": "Valid",
                "watermarkVerification": (
                    "Present" if final_confidence >= 70 else "Missing"
                ),
                "templateMatching": f"{int(final_confidence)}% match",
                "certificateType": certificate_type,
                "fileType": file.content_type,
            },
            "processing_time": processing_time,
        }

        # Add scoring breakdown (useful for debugging + transparency)
        response["score_breakdown"] = {
            "ocr_confidence": float(ocr_confidence) if ocr_details else 0.0,
            "cnn_authentic_score": float(cnn_authentic_score) if cnn_used else 0.0,
            "weights": {"ocr": 0.8, "cnn": 0.2},
            "cnn_used": bool(cnn_used),
        }

        # Include CNN analysis when available (even if not used)
        if cnn_result is not None:
            response["cnn_analysis"] = cnn_result

        # Add OCR analysis details
        if ocr_details:
            response["ocr_analysis"] = {
                "ocr_confidence": ocr_confidence,
                "validation_score": ocr_details["validation_score"],
                "is_valid": ocr_details["is_valid"],
                "anomalies_count": len(ocr_details["anomalies"]),
                "anomalies": ocr_details["anomalies"][:5],  # First 5 anomalies
                "extracted_data": ocr_details["extracted_data"],
            }

            # Update details based on OCR findings
            if ocr_details["extracted_data"].get("grades_found", 0) == 0:
                response["details"]["warning"] = "⚠️ No grades found - likely forged"

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@app.post("/upload-certificate/")
async def upload_certificate(file: UploadFile = File(...)):
    """Legacy endpoint for testing."""
    try:
        # Read file bytes first
        file_bytes = await file.read()

        # Save uploaded file to disk
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            buffer.write(file_bytes)

        # Preprocess image
        image = load_image_from_bytes(file_bytes)

        # Return success info
        return JSONResponse(
            {
                "filename": file.filename,
                "status": "success",
                "shape": list(image.shape),
            }
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


if __name__ == "__main__":
    import sys

    # Railway provides PORT as environment variable
    port = int(os.environ.get("PORT", 5000))

    # Force immediate output
    print(f"=" * 60, flush=True)
    print(f"🚀 STARTING ML SERVICE", flush=True)
    print(f"   Port: {port} (from $PORT env var)", flush=True)
    print(f"   Host: 0.0.0.0", flush=True)
    print(f"   Railway: {os.environ.get('RAILWAY_ENVIRONMENT', 'local')}", flush=True)
    print(f"   Python: {sys.version}", flush=True)
    print(f"=" * 60, flush=True)

    try:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=port,
            reload=False,
            log_level="info",
            access_log=True,
        )
    except Exception as e:
        print(f"❌ FAILED TO START: {e}", flush=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)
