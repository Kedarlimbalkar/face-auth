"""
test.py — Face Authentication Inference
========================================
This file contains only:
  1. load_model()  — loads the FaceNet model
  2. predict()     — accepts two image paths and returns verification result

Usage:
    python test.py --image1 path/to/face1.jpg --image2 path/to/face2.jpg
"""

import os
import argparse
import numpy as np
import cv2
from deepface import DeepFace

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME          = "Facenet"
DETECTOR_BACKEND    = "opencv"
SIMILARITY_THRESHOLD = 0.60   # Adjust: higher = stricter match
# ─────────────────────────────────────────────────────────────────────────────

_model = None  # Global model cache (loaded once)


def load_model():
    """
    Load and return the FaceNet model.
    The model is cached globally so it loads only once per session.
    """
    global _model
    if _model is None:
        print(f"[INFO] Loading '{MODEL_NAME}' model ...")
        _model = DeepFace.build_model(MODEL_NAME)
        print(f"[INFO] Model loaded successfully.")
    return _model


def predict(image1_path: str, image2_path: str) -> dict:
    """
    Verify whether two face images belong to the same person.

    Parameters:
        image1_path (str): File path to the first face image.
        image2_path (str): File path to the second face image.

    Returns:
        dict with keys:
            - verification_result  : "same person" or "different person"
            - similarity_score     : float (0.0 to 1.0)
            - threshold_used       : float
            - bounding_boxes       : dict with 'image1' and 'image2' box lists
    """
    # ── Validate inputs ──────────────────────────────────────────────────────
    if not os.path.isfile(image1_path):
        raise FileNotFoundError(f"Image 1 not found: {image1_path}")
    if not os.path.isfile(image2_path):
        raise FileNotFoundError(f"Image 2 not found: {image2_path}")

    # ── Load model ───────────────────────────────────────────────────────────
    load_model()

    # ── Extract embeddings ───────────────────────────────────────────────────
    def get_embedding(path):
        result = DeepFace.represent(
            img_path=path,
            model_name=MODEL_NAME,
            enforce_detection=True,
            detector_backend=DETECTOR_BACKEND
        )
        return np.array(result[0]["embedding"])

    try:
        emb1 = get_embedding(image1_path)
    except Exception as e:
        raise ValueError(f"Could not extract face from image1: {e}")

    try:
        emb2 = get_embedding(image2_path)
    except Exception as e:
        raise ValueError(f"Could not extract face from image2: {e}")

    # ── Cosine similarity ────────────────────────────────────────────────────
    dot  = np.dot(emb1, emb2)
    norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    similarity = float(dot / norm) if norm != 0 else 0.0

    verification_result = "same person" if similarity >= SIMILARITY_THRESHOLD else "different person"

    # ── Detect bounding boxes (OpenCV Haar Cascade) ──────────────────────────
    def get_boxes(path):
        img  = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in faces]

    boxes1 = get_boxes(image1_path)
    boxes2 = get_boxes(image2_path)

    return {
        "verification_result": verification_result,
        "similarity_score":    round(similarity, 4),
        "threshold_used":      SIMILARITY_THRESHOLD,
        "bounding_boxes": {
            "image1": boxes1,
            "image2": boxes2
        }
    }


# ── CLI Entry Point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Authentication — Predict")
    parser.add_argument("--image1", type=str, required=True, help="Path to first face image")
    parser.add_argument("--image2", type=str, required=True, help="Path to second face image")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("   Face Authentication — Prediction")
    print("=" * 50)
    print(f"  Image 1 : {args.image1}")
    print(f"  Image 2 : {args.image2}")
    print("-" * 50)

    try:
        result = predict(args.image1, args.image2)
        print(f"  Result          : {result['verification_result'].upper()}")
        print(f"  Similarity Score: {result['similarity_score']}")
        print(f"  Threshold Used  : {result['threshold_used']}")
        print(f"  Faces (img1)    : {result['bounding_boxes']['image1']}")
        print(f"  Faces (img2)    : {result['bounding_boxes']['image2']}")
    except Exception as e:
        print(f"  [ERROR] {e}")

    print("=" * 50)
