
import os
import base64
import logging
import tempfile
from pathlib import Path
from io import BytesIO

import cv2
from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from starlette.concurrency import run_in_threadpool

from backend.services.analysis_service import (
    analyze_image,
    InvalidRetinalImageError
)


# ==================================================
# APPLICATION
# ==================================================

app = FastAPI(
    title="AI-Powered Diabetic Retinopathy Detection API",
    description="AI-assisted diabetic retinopathy screening prototype.",
    version="1.0.0"
)

logger = logging.getLogger(__name__)


# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


# ==================================================
# FRONTEND
# ==================================================

@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/style.css", include_in_schema=False)
def serve_css():
    return FileResponse(
        FRONTEND_DIR / "style.css",
        media_type="text/css"
    )


@app.get("/script.js", include_in_schema=False)
def serve_js():
    return FileResponse(
        FRONTEND_DIR / "script.js",
        media_type="application/javascript"
    )


# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/health")
def health_check():
    return {"status": "healthy"}


# ==================================================
# IMAGE ANALYSIS
# ==================================================

@app.post("/analyze")
async def analyze_retinal_image(
    file: UploadFile = File(...)
):

    temp_file_path = None

    try:

        # Validate file extension
        extension = Path(
            file.filename or ""
        ).suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="Only JPG, JPEG and PNG images are supported."
            )

        # Limit upload size
        contents = await file.read(MAX_FILE_SIZE + 1)

        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Image size must not exceed 10 MB."
            )

        # Validate image contents
        try:

            with Image.open(BytesIO(contents)) as image:

                if image.format not in ALLOWED_FORMATS:
                    raise ValueError("Unsupported image format.")

                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Image resolution is too large.")

                image.verify()
                

        except (
            UnidentifiedImageError,
            Image.DecompressionBombError,
            OSError,
            ValueError
        ):
            raise HTTPException(
                status_code=400,
                detail="Invalid or unsupported image."
            )

        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_file_path = temp_file.name
            temp_file.write(contents)

        # Perform complete analysis
        result = await run_in_threadpool(
            analyze_image,
            temp_file_path
        )

        # Convert RGB image to OpenCV BGR format
        overlay = cv2.cvtColor(
            result["gradcam_overlay"],
            cv2.COLOR_RGB2BGR
        )

        # Encode Grad-CAM in memory
        success, buffer = cv2.imencode(
            ".jpg",
            overlay
        )

        if not success:
            raise RuntimeError(
                "Grad-CAM encoding failed."
            )

        encoded_image = base64.b64encode(
            buffer.tobytes()
        ).decode("utf-8")

        gradcam_url = (
            f"data:image/jpeg;base64,{encoded_image}"
        )

        # Return complete analysis
        response = {
            "class_id": result["class_id"],
            "severity": result["severity"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "recommendations": result["recommendations"],
            "disclaimer": result["disclaimer"],
            "gradcam_generated": True,
            "gradcam_url": gradcam_url
        }

        return JSONResponse(
            content=response,
            headers={"Cache-Control": "no-store"}
        )

    except HTTPException:
        raise

    except InvalidRetinalImageError as error:

        logger.warning("Invalid retinal image uploaded.")

        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    except Exception:

        logger.exception("Retinal image analysis failed.")

        raise HTTPException(
            status_code=500,
            detail="Image analysis failed. Please try another image."
        )

    finally:

        # Remove temporary uploaded image
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        await file.close()
