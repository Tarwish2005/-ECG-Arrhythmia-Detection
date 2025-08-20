# ECG Signal Classification for Arrhythmia Detection using a 1D CNN

This project implements a **1D Convolutional Neural Network (CNN)** using **PyTorch** to classify ECG signals into three categories:  

- **Normal Sinus Rhythm**  
- **Ventricular Tachycardia/Fibrillation (VT/VF)**  
- **Noise**  

The model is trained and validated on data from multiple publicly available ECG databases.

---

## 📝 Table of Contents
- [Project Overview](#-project-overview)
- [Methodology](#-methodology)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Model Architecture](#2-model-architecture)
  - [3. Training](#3-training)
- [Datasets](#-datasets)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Results](#-results)

---

## 📖 Project Overview
The primary goal of this project is to build a **robust classifier for detecting life-threatening ventricular arrhythmias (VT/VF)** from ECG signals.  

The model distinguishes critical arrhythmias from normal rhythms and noisy signals, addressing a real-world clinical challenge.  

The notebook demonstrates the full workflow:
- Loading and preprocessing raw ECG data  
- Defining a custom 1D CNN architecture for time-series data  
- Training and validating the model on a composite dataset  

---

## 🛠 Methodology

### 1. Data Preprocessing
To prepare the raw ECG signals, the following steps are applied:
- **Resampling** → signals are resampled to **250 Hz**  
- **Filtering** → high-pass Butterworth filter removes baseline wander  
- **Normalization** → Z-score normalization (mean subtraction & division by std)  
- **Segmentation** → signals segmented into **5s windows (1250 samples)**  

---

### 2. Model Architecture
The model is a **5-layer 1D CNN** optimized for ECG time-series data.  

- **Input:** 1D vector of 1250 samples (5s ECG)  
- **Convolutional Blocks:**  
  - Conv1d (kernel size=3, ReLU)  
  - BatchNorm1d  
  - MaxPool1d (kernel size=2)  
- **Fully Connected Layers:** Flattened features → 3 dense layers with dropout  
- **Output Layer:** 3 units for classification  
  - Class 0 → Normal Rhythm  
  - Class 1 → VT/VF  
  - Class 2 → Noise  

**Initialization:**  
- Convolutional layers → Kaiming Normalization  
- Linear layers → Normal distribution  

---

### 3. Training
- **Optimizer:** Adam (lr=0.001, weight_decay=1e-4)  
- **Loss Function:** CrossEntropyLoss  
- **Data Split:** 70% train, 15% validation, 15% test  
- **Epochs:** 100  
- **Device:** CUDA GPU (if available)  

---

## 📚 Datasets
The model uses **4 PhysioNet ECG databases** for diversity and robustness:

1. **MIT-BIH Arrhythmia Database (ADB_DIR)** → Normal rhythms & arrhythmias (Class 0)  
2. **Malignant Ventricular Ectopy Database (VFDB_DIR)** → VT/VF arrhythmias (Class 1)  
3. **Creighton University Ventricular Tachyarrhythmia Database (CUVTDB_DIR)** → VF recordings (Class 1)  
4. **MIT-BIH Noise Stress Test Database (NSTDB_DIR)** → Noisy signals (Class 2)  

---

## 🚀 Getting Started

### Prerequisites
- Python 3.x  
- Access to PhysioNet ECG databases  

### Installation
Clone the repository (or download the notebook):

```bash
git clone https://your-repository-url.git
cd your-repository-url
