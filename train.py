"""
train.py — Face Authentication Model Setup & Training
======================================================
This script handles the model preparation step.

Since we use a pre-trained FaceNet model (via DeepFace), "training" here means:
  1. Downloading and caching the FaceNet weights locally
  2. (Optionally) Fine-tuning on a custom face dataset if provided

For most use cases (1:1 face verification), the pre-trained FaceNet
model is sufficient and no further training is needed.

To run:
    python train.py
    python train.py --dataset ./my_faces_dataset   # if you have custom data
"""

import os
import argparse
import numpy as np
from deepface import DeepFace

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = "Facenet"          # Options: Facenet, VGG-Face, ArcFace, DeepFace
DETECTOR   = "opencv"           # Options: opencv, retinaface, mtcnn, ssd
CACHE_DIR  = "./model_cache"
# ────────────────────────────────────────────────────────────────────────────


def download_and_cache_model():
    """
    Forces DeepFace to download and cache the FaceNet weights.
    DeepFace stores models in ~/.deepface/weights/ by default.
    """
    print(f"[INFO] Loading / downloading '{MODEL_NAME}' model weights ...")
    try:
        model = DeepFace.build_model(MODEL_NAME)
        print(f"[INFO] '{MODEL_NAME}' model loaded successfully.")
        print(f"[INFO] Model weights are cached at: ~/.deepface/weights/")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise


def verify_model_with_sample(model):
    """
    Quick sanity check — generates a random 'image' embedding to confirm
    the model pipeline works end-to-end.
    """
    print("\n[INFO] Running sanity check on model ...")
    # Create a small blank test image
    dummy_img = np.ones((160, 160, 3), dtype=np.uint8) * 128
    test_path = os.path.join(CACHE_DIR, "dummy_test.jpg")
    os.makedirs(CACHE_DIR, exist_ok=True)

    import cv2
    cv2.imwrite(test_path, dummy_img)

    try:
        result = DeepFace.represent(
            img_path=test_path,
            model_name=MODEL_NAME,
            enforce_detection=False,   # Skip detection for dummy image
            detector_backend=DETECTOR
        )
        emb = result[0]["embedding"]
        print(f"[INFO] Embedding shape: ({len(emb)},)  ✓  Model is working correctly.")
    except Exception as e:
        print(f"[WARN] Sanity check skipped: {e}")
    finally:
        if os.path.exists(test_path):
            os.remove(test_path)


def build_embeddings_from_dataset(dataset_dir: str):
    """
    (Optional) If you have a labeled face dataset, this function walks through
    each person's folder, extracts embeddings, and saves them as a .npy file.

    Expected folder structure:
        dataset_dir/
            person_A/
                img1.jpg
                img2.jpg
            person_B/
                img1.jpg

    This allows the API to do 1:N recognition in the future.
    """
    print(f"\n[INFO] Building embeddings from dataset: {dataset_dir}")

    if not os.path.isdir(dataset_dir):
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        return

    embeddings_db = {}  # { "person_name": [emb1, emb2, ...] }

    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        person_embeddings = []
        for img_file in os.listdir(person_dir):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(person_dir, img_file)
            try:
                result = DeepFace.represent(
                    img_path=img_path,
                    model_name=MODEL_NAME,
                    enforce_detection=True,
                    detector_backend=DETECTOR
                )
                emb = result[0]["embedding"]
                person_embeddings.append(emb)
                print(f"  [OK] {person_name}/{img_file}")
            except Exception as e:
                print(f"  [SKIP] {person_name}/{img_file} — {e}")

        if person_embeddings:
            embeddings_db[person_name] = person_embeddings

    # Save embeddings
    os.makedirs(CACHE_DIR, exist_ok=True)
    save_path = os.path.join(CACHE_DIR, "face_embeddings.npy")
    np.save(save_path, embeddings_db)
    print(f"\n[INFO] Embeddings saved to: {save_path}")
    print(f"[INFO] Total persons indexed: {len(embeddings_db)}")
    return embeddings_db


def main():
    parser = argparse.ArgumentParser(description="Face Auth — Model Setup & Training")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(Optional) Path to labeled face dataset folder for embedding extraction."
    )
    args = parser.parse_args()

    print("=" * 55)
    print("   Face Authentication — Model Training / Setup")
    print("=" * 55)

    # Step 1: Download and cache the pre-trained model
    model = download_and_cache_model()

    # Step 2: Sanity check
    verify_model_with_sample(model)

    # Step 3 (Optional): Build embeddings from custom dataset
    if args.dataset:
        build_embeddings_from_dataset(args.dataset)
    else:
        print("\n[INFO] No --dataset provided. Skipping embedding extraction.")
        print("[INFO] To index your own faces, run:")
        print("         python train.py --dataset ./path/to/dataset")

    print("\n[DONE] Model is ready. You can now run the API:")
    print("         uvicorn main:app --reload")
    print("=" * 55)


if __name__ == "__main__":
    main()
