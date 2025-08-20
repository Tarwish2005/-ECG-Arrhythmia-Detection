# ECG Arrhythmia Detection

This project implements an end-to-end pipeline for ECG arrhythmia detection using deep learning. It leverages several public ECG databases and a 1D CNN model for classification.

## Features

- Loads and preprocesses ECG data from multiple databases:
  - MIT-BIH Arrhythmia Database (MITDB)
  - VFDB (Ventricular Fibrillation Database)
  - CUVTDB
  - NSTDB (Noise Stress Test Database)
- Segments and normalizes ECG signals
- Trains a 1D CNN for arrhythmia classification
- Visualizes training progress (loss and accuracy)

## Requirements

- Python 3.7+
- [wfdb](https://pypi.org/project/wfdb/)
- numpy
- scipy
- scikit-learn
- torch
- torchaudio
- torchvision
- matplotlib
- seaborn

Install dependencies with:

```sh
pip install wfdb numpy scipy scikit-learn torch torchaudio torchvision matplotlib seaborn
```

## Usage

1. **Prepare Data**  
   Place the ECG databases in your workspace and update the paths in `ECG-Arrhythmia-Detection.ipynb`:
   - `VFDB_DIR`
   - `CUVTDB_DIR`
   - `ADB_DIR` (MITDB)
   - `NSTDB_DIR`

2. **Run the Notebook**  
   Open `ECG-Arrhythmia-Detection.ipynb` in Jupyter or VS Code and execute the cells.

3. **Training**  
   The notebook will:
   - Load and preprocess ECG data
   - Train a CNN model
   - Plot training/validation loss and accuracy

## File Structure

- `ECG-Arrhythmia-Detection.ipynb`: Main notebook with all code
- `MITDB/`, `VFDB/`, `CUVTDB/`, `NSTDB/`: ECG database directories

## Model

The model is a 5-layer 1D CNN implemented in PyTorch ([`ECG_CNN`](ECG-Arrhythmia-Detection.ipynb)).  
It takes 5-second ECG segments and classifies them into three classes:
- Non-VT/VF
- VT/VF
- Noisy

