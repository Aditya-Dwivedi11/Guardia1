# Face Recognition System using FaceNet + MTCNN

This project implements a full deep-learning based face recognition system using:

- **MTCNN** for face detection & alignment
- **FaceNet (InceptionResnetV1)** for generating embeddings
- **Google Colab Webcam** for capturing test images
- **Mean Embedding Database** for stable person matching
- Supports **masked & unmasked** datasets

---

## Features

- Capture images in real‑time through webcam
- Auto‑align faces using MTCNN
- Build embedding database from:
  ```
  person_name/
      mask/
      unmask/
  ```
- Normalize FaceNet embeddings
- Create mean embeddings per identity
- Identify faces with bounding boxes & labels

---

## Project Structure

```
├── create_database.py
├── identify_face.py
├── photo.jpg
├── face_embeddings_mean.pkl
├── mask_output_final.jpg
└── README.md
```

---

## Installation

```bash
pip install facenet-pytorch mtcnn torch torchvision pillow numpy opencv-python
```

---

## Step 1 — Capture Image

```python
filename = take_photo()
```

---

## Step 2 — Build Database

```bash
python create_database.py
```

---

## Step 3 — Identify Faces

```bash
python identify_face.py
```

Output:

```
mask_output_final.jpg
```

---

## Threshold Tuning

```
THRESHOLD = 0.9
```

Recommended: **0.85 – 1.05**

---

## Dataset Format

```
face_dataset/
    person1/
        mask/
        unmask/
    person2/
        mask/
        unmask/
```

---
