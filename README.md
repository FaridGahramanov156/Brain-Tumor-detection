# Brain Tumor Detection using Deep Learning 🧠🧬

This project implements a deep learning-based classification system to detect the presence of **brain tumors** in MRI scans using **TensorFlow**. It uses grayscale MR images and classifies them into **tumor** or **non-tumor** classes.

---

## 🔍 Problem Statement

Early and accurate diagnosis of brain tumors is critical. This project provides a computer vision pipeline to:
- Detect if a brain MRI contains a tumor
- Classify the tumor type (**glioma**, **meningioma**, **pituitary**) using a second-stage model

---

## ⚙️ Tools & Technologies

- Python
- TensorFlow & Keras
- NumPy, Matplotlib
- CNN architectures
- Transfer Learning (MobileNetV2)
- Scikit-learn for evaluation

---

## 🧠 Model Overview

### ✅ Stage 1: Binary Classification
- Input: Brain MRI (grayscale)
- Output: `Tumor` or `Notumor`
- Model: CNN with Conv2D → MaxPooling → Dense layers

### ✅ Stage 2: Tumor Type Classification
- Classes: `glioma`, `meningioma`, `pituitary`
- Model: MobileNetV2-based transfer learning model
- Label Mode: `categorical` with softmax activation
- Evaluation: F1-score, accuracy, confusion matrix

---

## 📈 Sample Metrics (Multiclass Classifier)

| Class       | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| Glioma      | 0.95      | 0.84   | 0.89     |
| Meningioma  | 0.84      | 0.89   | 0.86     |
| Pituitary   | 0.93      | 0.98   | 0.95     |

- **Accuracy**: 90%
- **F1 (macro avg)**: 0.90

---
