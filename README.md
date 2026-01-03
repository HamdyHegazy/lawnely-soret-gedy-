# 🧓 lawnely-soret-gedy-👴  
### Face Colorization Using Deep Learning

**lawnely-soret-gedy-👴** is a deep learning project that automatically
colorizes grayscale human face images and converts them into realistic
RGB images using a **U-Net–based autoencoder** built with
**TensorFlow / Keras**.

<p align="center">
  <img src="results/result_1.png" width="500">
</p>

---

## 🚀 Project Overview

- Converts grayscale face images into color (RGB)
- Uses a **U-Net architecture as an autoencoder**
- Trained end-to-end using TensorFlow / Keras
- Includes preprocessing, training, evaluation, and visualization
- Produces visually realistic face colorizations

---

## 📂 Project Structure

```
lawnely-soret-gedy-👴/
│
├── src/
│   ├── face_colorization.ipynb
│   └── image_colorization_model.keras
│
├── data/
│   └── README.md
│
├── results/
│   └── sample_output.png
│
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture (U-Net Autoencoder)

The model is based on a **U-Net architecture**, adapted to function as an
**autoencoder for image colorization**.

### How it works

- The **encoder** takes a single-channel grayscale face image and
  progressively downsamples it using convolutional layers, extracting
  high-level facial features such as edges, structure, and texture.
- The **decoder** upsamples the encoded representation back to the
  original image resolution, predicting **three color channels (RGB)**.
- **Skip connections** are used between corresponding encoder and decoder
  layers to preserve spatial details, improving color consistency and
  facial structure reconstruction.

This design allows the network to combine **global context** with
**fine-grained facial details**, resulting in more realistic and stable
colorization results.

**Input:**  
- Grayscale image `(224 × 224 × 1)`

**Output:**  
- Color image `(224 × 224 × 3)`

---

## ▶️ How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Load the Trained Model

```python
from tensorflow.keras.models import load_model

model = load_model("src/image_colorization_model.keras")
```

---

### 3️⃣ Colorize a Grayscale Image

```python
import numpy as np
import cv2

img = cv2.imread("path/to/grayscale.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=(0, -1))

pred = model.predict(img)
pred = np.clip(pred[0], 0, 1)

cv2.imwrite("results/colorized_output.jpg", pred * 255)
```

---

## 📌 Notes & Limitations

- Designed primarily for **human face images**
- The model predicts **plausible colors**, not exact ground truth
- Performance depends heavily on dataset quality and diversity

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

---

## 📜 License

This project is intended for **educational and research purposes only**.
