# ECG Arrhythmia Classification using 1D CNN

This project implements a **1D Convolutional Neural Network (CNN)** in PyTorch to classify ECG signals into three categories:  

- **Normal Sinus Rhythm (Class 0)**  
- **Ventricular Tachycardia/Fibrillation (VT/VF, Class 1)**  
- **Noise (Class 2)**  

The model is trained on a **composite dataset aggregated from four PhysioNet databases**, ensuring robustness and generalization for real-world ECG monitoring.  

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)  
- [Methodology](#-methodology)  
  - [Data Preprocessing](#data-preprocessing)  
  - [Model Architecture](#model-architecture)  
- [Datasets](#-datasets)  
- [Getting Started](#-getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
  - [Usage](#usage)  
- [Results](#-results)  
- [Contributing](#-contributing)  
- [License](#-license)  

---

## 📖 Project Overview
The primary goal is to **detect life-threatening ventricular arrhythmias (VT/VF)** and differentiate them from:  
1. Normal heart rhythms, and  
2. Noisy/corrupted signals (a common issue in clinical ECG monitoring).  

The workflow includes:  
- Loading and preprocessing raw ECG waveform data  
- Defining a **custom 1D CNN** architecture for time-series classification  
- Training and validating the model  
- Visualizing performance (loss/accuracy curves)  

---

## 🛠️ Methodology

### Data Preprocessing
All ECG signals undergo a series of steps for consistency:  
- **Resampling** → All signals resampled to **250 Hz**  
- **Filtering** → High-pass Butterworth filter to remove baseline wander  
- **Normalization** → Z-score normalization (zero mean, unit variance)  
- **Segmentation** → Signals segmented into **5-second windows** (1250 samples each)  

### Model Architecture
A **5-layer 1D CNN** captures temporal features from ECG signals.  

- **Input Layer**: `(batch_size, 1, 1250)`  
- **Conv Blocks (x5)**:  
  - `Conv1d (kernel=3)` → `BatchNorm1d` → `ReLU` → `MaxPool1d`  
- **Fully Connected Layers**:  
  - Flatten → Dense Layers with `ReLU` + Dropout (0.5, 0.3)  
- **Output Layer**: 3 neurons for classification  
  - **0** → Normal Rhythm  
  - **1** → VT/VF  
  - **2** → Noise  

---

## 📚 Datasets
Training combines four **PhysioNet** databases:  

| Database | Variable | Purpose | Class Label |
|----------|----------|---------|-------------|
| MIT-BIH Arrhythmia Database | `ADB_DIR` | Normal rhythms & other arrhythmias | 0 |
| Malignant Ventricular Ectopy Database | `VFDB_DIR` | Life-threatening VT/VF | 1 |
| Creighton Univ. Ventricular Tachyarrhythmia DB | `CUVTDB_DIR` | Extra VT/VF examples | 1 |
| MIT-BIH Noise Stress Test DB | `NSTDB_DIR` | Realistic noise artifacts | 2 |  

📌 **Note**: All datasets are publicly available on [PhysioNet](https://physionet.org).  

---

## 🚀 Getting Started

### Prerequisites
- Python **3.7+**  
- PyTorch  
- CUDA-enabled GPU (recommended)  

### Installation
Clone the repository:  
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
