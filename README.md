# ECG Signal Classification for Arrhythmia Detection using 1D CNN

This project implements a **1D Convolutional Neural Network (CNN)** in **PyTorch** to classify ECG signals into three categories:

- **Normal Sinus Rhythm**
- **Ventricular Tachycardia/Fibrillation (VT/VF)**
- **Noise**

The model is trained on multiple **PhysioNet ECG databases** for robustness and real-world applicability.

---

## 📖 Table of Contents
- [Overview](#overview)
- [Methodology](#methodology)
- [Datasets](#datasets)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

---

## 📌 Overview
The goal is to develop a **robust ECG arrhythmia detection system** that:
- Differentiates normal rhythms from **life-threatening arrhythmias (VT/VF)**.
- Rejects noisy signals for clinical reliability.
- Provides a **deep learning pipeline** from data preprocessing to evaluation.

---

## ⚙️ Methodology

### 1. Data Preprocessing
- **Resampling:** signals → 250 Hz
- **Filtering:** Butterworth high-pass filter (remove baseline wander)
- **Normalization:** Z-score scaling
- **Segmentation:** fixed **5s windows (1250 samples)**

### 2. Model Architecture
- **5-layer 1D CNN** with:
  - Conv1D + BatchNorm + ReLU + MaxPooling
  - Fully connected layers with Dropout
  - Softmax output → 3 classes

**Initialization:**
- Conv layers → Kaiming Normalization  
- Linear layers → Normal distribution  

### 3. Training
- **Optimizer:** Adam (lr=0.001, weight_decay=1e-4)  
- **Loss:** CrossEntropyLoss  
- **Splits:** 70% Train | 15% Validation | 15% Test  
- **Epochs:** 100  
- **Device:** CUDA (if available)  

---

## 📂 Datasets
ECG signals are sourced from **PhysioNet**:

1. **MIT-BIH Arrhythmia Database** → Normal rhythms (Class 0)  
2. **Malignant Ventricular Ectopy Database** → VT/VF arrhythmias (Class 1)  
3. **Creighton University Ventricular Tachyarrhythmia Database** → VF signals (Class 1)  
4. **MIT-BIH Noise Stress Test Database** → Noisy signals (Class 2)  

---

## 🛠 Setup

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (optional but recommended)
- Access to PhysioNet ECG databases

### Installation
```bash
# Clone repo
git clone https://github.com/your-username/ecg-cnn-classification.git
cd ecg-cnn-classification

# Install dependencies
pip install -r requirements.txt
