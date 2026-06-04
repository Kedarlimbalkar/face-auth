# Face Authentication API

A Python + FastAPI service that verifies whether two face images belong to the same person using **FaceNet** embeddings via [DeepFace](https://github.com/serengil/deepface).

---

## Features

- Accepts two face images via REST API
- Detects faces using OpenCV Haar Cascade
- Extracts 128-d embeddings using **FaceNet** (pre-trained)
- Computes **cosine similarity** between embeddings
- Returns:
  - `verification_result` → `"same person"` or `"different person"`
  - `similarity_score` → float between 0 and 1
  - `bounding_boxes` → detected face locations in both images

---

## Project Structure

```
face_auth/
├── main.py           # FastAPI app (inference endpoint)
├── train.py          # Model download/setup + optional dataset embedding
├── test.py           # CLI predict function (load model + predict)
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Setup & Installation

### 1. Clone / navigate to the project folder
```bash
cd face_auth
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Step 1 — Prepare the Model (train.py)

Downloads and caches the FaceNet model weights locally:

```bash
python train.py
```

Optional — if you have a labeled face dataset to build an embeddings index:
```bash
python train.py --dataset ./my_dataset
```

Expected dataset structure:
```
my_dataset/
    Alice/
        img1.jpg
        img2.jpg
    Bob/
        img1.jpg
```

---

## Step 2 — Run the API (main.py)

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Step 3 — Test via CLI (test.py)

```bash
python test.py --image1 samples/face1.jpg --image2 samples/face2.jpg
```

Sample output:
```
==================================================
   Face Authentication — Prediction
==================================================
  Image 1 : samples/face1.jpg
  Image 2 : samples/face2.jpg
--------------------------------------------------
  Result          : SAME PERSON
  Similarity Score: 0.8723
  Threshold Used  : 0.6
  Faces (img1)    : [{'x': 45, 'y': 30, 'w': 120, 'h': 120}]
  Faces (img2)    : [{'x': 60, 'y': 25, 'w': 115, 'h': 115}]
==================================================
```

---

## API Reference

### `POST /verify`

| Field    | Type | Description                  |
|----------|------|------------------------------|
| image1   | file | First face image (JPG/PNG)   |
| image2   | file | Second face image (JPG/PNG)  |

**Response:**
```json
{
  "verification_result": "same person",
  "similarity_score": 0.8723,
  "threshold_used": 0.6,
  "bounding_boxes": {
    "image1": [{"x": 45, "y": 30, "w": 120, "h": 120}],
    "image2": [{"x": 60, "y": 25, "w": 115, "h": 115}]
  },
  "faces_detected": {
    "image1": 1,
    "image2": 1
  }
}
```

### `GET /health`
Returns `{"status": "healthy"}` — useful for Docker/K8s health checks.

---

## Model Details

| Property        | Value                     |
|-----------------|---------------------------|
| Model           | FaceNet (pre-trained)     |
| Embedding size  | 128-d vector              |
| Face detector   | OpenCV Haar Cascade       |
| Similarity      | Cosine similarity         |
| Threshold       | 0.60 (adjustable)         |

---

## Tuning the Threshold

Edit `SIMILARITY_THRESHOLD` in `main.py` and `test.py`:

| Threshold | Behavior                            |
|-----------|-------------------------------------|
| 0.50      | Lenient — more "same person" hits   |
| 0.60      | Balanced (default)                  |
| 0.75      | Strict — fewer false positives      |

---

## Sample Images

Place two face images in a `samples/` folder to test quickly:
```bash
mkdir samples
# Copy face1.jpg and face2.jpg into samples/
python test.py --image1 samples/face1.jpg --image2 samples/face2.jpg
```
