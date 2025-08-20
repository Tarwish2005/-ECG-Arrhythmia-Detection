# 🫀 ECG Arrhythmia Detection with Deep Learning

Detect arrhythmias from ECG signals using a robust 1D CNN pipeline. This project leverages multiple PhysioNet ECG databases and PyTorch for accurate, automated classification.

---

## 🚀 Features

- **Multi-Database Support:** MIT-BIH Arrhythmia, VFDB, CUVTDB, NSTDB (noise)
- **Advanced Preprocessing:** Resampling, baseline wander removal, normalization, segmentation
- **Deep Learning Model:** 5-layer 1D CNN with batch normalization and dropout
- **Visualization:** Training/validation loss and accuracy plots
- **Robust Data Handling:** NaN/inf filtering, gradient clipping

---

## 📦 Requirements

- Python 3.7+
- wfdb
- numpy
- scipy
- scikit-learn
- torch
- torchaudio
- torchvision
- matplotlib
- seaborn

Install all dependencies with:
```sh
pip install wfdb numpy scipy scikit-learn torch torchaudio torchvision matplotlib seaborn
```

---

## 🗂️ File Structure

- `ECG-Arrhythmia-Detection.ipynb` — Main notebook
- `MITDB/`, `VFDB/`, `CUVTDB/`, `NSTDB/` — ECG database folders

---

## ⚙️ Usage

1. **Prepare Data:**  
   Download ECG databases and update paths in the notebook:
   ```python
   VFDB_DIR   = "/path/to/VFDB"
   CUVTDB_DIR = "/path/to/CUVTDB"
   ADB_DIR    = "/path/to/MITDB"
   NSTDB_DIR  = "/path/to/NSTDB"
   ```
2. **Run Notebook:**  
   Open `ECG-Arrhythmia-Detection.ipynb` in Jupyter or VS Code and execute all cells.

3. **Training:**  
   The notebook loads, preprocesses, splits, and trains the model.  
   Loss and accuracy curves are plotted for easy monitoring.

---

## 🧠 Model Architecture

- **Input:** 5-second ECG segments (1250 samples @ 250Hz)
- **Layers:** 5 convolutional + batch norm + max pooling
- **Classifier:** 3 output classes (Non-VT/VF, VT/VF, Noisy)

---

## 📊 Results

> **Example Results (with full datasets):**

| Metric      | Value      |
|-------------|------------|
| Train Acc   | ~98%       |
| Val Acc     | ~96%       |
| Test Acc    | ~95%       |
| Classes     | Non-VT/VF, VT/VF, Noisy |





