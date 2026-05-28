# MNIST Digit Recognizer

CNN-powered handwritten digit recognition system built with TensorFlow, Keras, and Flask with real-time canvas prediction and image upload support.

---

## Live Demo

https://mnist-recognizer.jainparichay.in/

---

## Overview

MNIST Digit Recognizer is a deep learning web application that classifies handwritten digits using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

The project combines deep learning inference, image preprocessing, real-time prediction, and a lightweight Flask deployment pipeline into an interactive computer vision application.

---

## Features

* Real-time handwritten digit prediction
* Interactive canvas drawing interface
* Image upload prediction support
* CNN model trained on MNIST dataset
* 99%+ test accuracy
* Flask-based inference API
* Lightweight frontend interface
* Docker deployment support

---

## Tech Stack

### Frontend

* HTML
* CSS
* JavaScript

### Backend

* Flask
* Python

### Deep Learning

* TensorFlow
* Keras
* CNN Architecture

### Deployment

* Docker
* Render

---

## System Flow

```text
Canvas Drawing / Uploaded Image
                ↓
Image Preprocessing
                ↓
CNN Inference
                ↓
Digit Prediction
                ↓
Prediction Output
```

---

## CNN Architecture

The model uses a Convolutional Neural Network trained on the MNIST handwritten digit dataset.

### Model Components

* Convolutional layers
* Max pooling layers
* ReLU activation
* Dense fully connected layers
* Softmax output layer

### Training Details

* Dataset: MNIST
* Training Samples: 60,000
* Test Samples: 10,000
* Framework: TensorFlow/Keras
* Optimizer: Adam
* Loss Function: Sparse Categorical Crossentropy

---

## Model Performance

| Metric        | Value            |
| ------------- | ---------------- |
| Test Accuracy | 99%+             |
| Dataset       | MNIST            |
| Classes       | 10 Digits (0–9)  |
| Framework     | TensorFlow/Keras |

---

## Project Structure

```text
mnist-recognizer/
├── app.py
├── train.py
├── requirements.txt
├── Dockerfile
├── mnist_model.h5
├── model/
├── templates/
├── index.html
└── README.md
```

---

## Local Setup

### Clone Repository

```bash
git clone https://github.com/mokshijain22/mnist-recognizer.git
cd mnist-recognizer
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Run Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## Training the Model

```bash
python train.py
```

This generates the trained CNN model file used during inference.

---

## Key Learnings

* CNNs are highly effective for image classification tasks
* Image preprocessing significantly affects inference quality
* Lightweight deep learning deployments require dependency optimization
* Real-time inference systems require balancing speed and accuracy

---

## Future Improvements

* Multi-digit recognition
* Handwritten equation solving
* Improved mobile drawing support
* Confidence visualization graphs
* Better preprocessing pipeline
* ONNX/TensorRT optimization

---

## Notes

* Optimized for MNIST-style handwritten digits
* Best results achieved with centered and clearly drawn digits
* Docker setup included for deployment portability

---

## Author

Mokshi Jain

GitHub: https://github.com/mokshijain22

---


ai
```
