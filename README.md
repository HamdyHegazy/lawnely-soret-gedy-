# Face Colorization Using Deep Learning

This project takes a grayscale image of a human face and predicts a
realistic RGB (colorized) version using a deep neural network built with
TensorFlow/Keras.

<p align="center">
  <img src="results/result_1.png" width="500">
</p>

## 🚀 Project Overview

-   Convert grayscale human face images to RGB color images\
-   Uses a deep convolutional encoder--decoder architecture\
-   Trained using TensorFlow/Keras\
-   Includes preprocessing, model training, evaluation, and
    visualization\
-   Outputs realistic colorized face images

## 📂 Project Structure

    face-colorization-project/
    │
    ├── src/
    │   └── face_colorization.ipynb     
    │   └── image_colorization_model.keras    
    │                   
    │
    ├── data/
    │   └── README.md                   
    │
    ├── results/
    │   └── sample_output.png           
    │
    ├── requirements.txt                
    └── README.md                       

---

## 🧠 Model Summary

The model is a convolutional encoder–decoder network that maps:

**Input:** 1-channel grayscale image  
**Output:** 3-channel RGB image

The notebook contains:

- Data preprocessing  
- Model creation and training  
- Loss/accuracy visualization  
- Sample predictions and output visualization  

---

## ▶️ How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```
2. Load the trained model:

```python
from tensorflow.keras.models import load_model
model = load_model("src/model.keras")
```

3. Use the model to colorize a grayscale image:

```python
import numpy as np
import cv2

# Load grayscale image
img = cv2.imread("path/to/grayscale.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=(0, -1))

# Predict colorization
pred = model.predict(img)
pred = np.clip(pred[0], 0, 1)

# Save output
cv2.imwrite("results/colorized_output.jpg", pred * 255)
```

