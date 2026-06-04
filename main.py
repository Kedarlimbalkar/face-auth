"""
Face Authentication (Face Verification) Service
FastAPI app that accepts two face images, detects faces,
extracts embeddings, computes similarity, and returns verification result.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from deepface import DeepFace
import base64
from io import BytesIO
from PIL import Image
import tempfile
import os

app = FastAPI(
    title="Face Authentication API",
    description="Verify if two face images belong to the same person.",
    version="1.0.0"
)

SIMILARITY_THRESHOLD = 0.60  # Cosine similarity threshold


def read_image_from_upload(file_bytes: bytes) -> np.ndarray:
    """Convert uploaded file bytes to OpenCV image (BGR)."""
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Please upload a valid JPG/PNG.")
    return img


def detect_faces_and_boxes(img_bgr: np.ndarray) -> list:
    """
    Detect faces using OpenCV Haar Cascade and return bounding boxes.
    Returns list of dicts: {x, y, w, h}
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return []
    return [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in faces]


def get_embedding(img_bgr: np.ndarray, temp_path: str) -> np.ndarray:
    """
    Save image temporarily and extract face embedding using DeepFace (Facenet model).
    """
    cv2.imwrite(temp_path, img_bgr)
    try:
        result = DeepFace.represent(
            img_path=temp_path,
            model_name="Facenet",
            enforce_detection=True,
            detector_backend="opencv"
        )
        embedding = np.array(result[0]["embedding"])
        return embedding
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Face not detected or embedding extraction failed: {str(e)}"
        )


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors."""
    dot = np.dot(emb1, emb2)
    norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    if norm == 0:
        return 0.0
    return float(dot / norm)


@app.get("/")
def root():
    return {
        "message": "Face Authentication API is running.",
        "endpoints": {
            "POST /verify": "Upload two face images to verify if they are the same person.",
            "GET /health": "Check service health."
        }
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/verify")
async def verify_faces(
    image1: UploadFile = File(..., description="First face image (JPG/PNG)"),
    image2: UploadFile = File(..., description="Second face image (JPG/PNG)")
):
    """
    Verify if two face images belong to the same person.

    - **image1**: First face image
    - **image2**: Second face image

    Returns:
    - **verification_result**: "same person" or "different person"
    - **similarity_score**: Cosine similarity (0 to 1)
    - **bounding_boxes**: Detected face boxes for each image
    """
    # Read uploaded images
    img1_bytes = await image1.read()
    img2_bytes = await image2.read()

    img1 = read_image_from_upload(img1_bytes)
    img2 = read_image_from_upload(img2_bytes)

    # Detect faces & get bounding boxes
    boxes1 = detect_faces_and_boxes(img1)
    boxes2 = detect_faces_and_boxes(img2)

    if len(boxes1) == 0:
        raise HTTPException(status_code=422, detail="No face detected in image1.")
    if len(boxes2) == 0:
        raise HTTPException(status_code=422, detail="No face detected in image2.")

    # Save to temp files for DeepFace
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as t1, \
         tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as t2:
        tmp1, tmp2 = t1.name, t2.name

    try:
        emb1 = get_embedding(img1, tmp1)
        emb2 = get_embedding(img2, tmp2)
    finally:
        os.unlink(tmp1)
        os.unlink(tmp2)

    # Compute similarity
    similarity = cosine_similarity(emb1, emb2)
    result = "same person" if similarity >= SIMILARITY_THRESHOLD else "different person"

    return JSONResponse(content={
        "verification_result": result,
        "similarity_score": round(similarity, 4),
        "threshold_used": SIMILARITY_THRESHOLD,
        "bounding_boxes": {
            "image1": boxes1,
            "image2": boxes2
        },
        "faces_detected": {
            "image1": len(boxes1),
            "image2": len(boxes2)
        }
    })
