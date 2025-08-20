ECG Signal Classification for Arrhythmia Detection using a 1D CNN
This project implements a 1D Convolutional Neural Network (CNN) using PyTorch to classify ECG signals into three categories: Normal Sinus Rhythm, Ventricular Tachycardia/Fibrillation (VT/VF), and Noise. The model is trained and validated on data from multiple publicly available ECG databases.

📝 Table of Contents
Project Overview

Methodology

1. Data Preprocessing

2. Model Architecture

3. Training

Datasets

Getting Started

Prerequisites

Installation

Usage

Results

📖 Project Overview
The primary goal of this project is to build a robust classifier for detecting life-threatening ventricular arrhythmias (VT/VF) from ECG signals. The model distinguishes these critical signals from normal rhythms and noisy segments, which is a common challenge in real-world clinical applications.

The notebook covers the entire workflow:

Loading and preprocessing raw ECG data.

Defining a custom 1D CNN architecture suitable for time-series data.

Training and validating the model on a composite dataset.

🛠 Methodology
1. Data Preprocessing
To prepare the raw ECG signals for the neural network, the following preprocessing steps are applied to each record:

Resampling: All signals are resampled to a uniform sampling frequency of 250 Hz.

Filtering: A high-pass Butterworth filter is used to remove baseline wander.

Normalization: The signal is normalized using Z-score normalization (mean subtraction and division by standard deviation).

Segmentation: The continuous signal is segmented into non-overlapping windows of 5 seconds each (1250 samples).

2. Model Architecture
The project uses a 5-layer 1D CNN. The architecture is designed to capture hierarchical features from the ECG time-series data.

Input: A 1D vector of 1250 samples (5 seconds of ECG data).

Convolutional Blocks: The model consists of five convolutional blocks. Each block contains:

Conv1d layer with a kernel size of 3 and ReLU activation.

BatchNorm1d for stabilizing learning.

MaxPool1d with a kernel size of 2 for downsampling.

Fully Connected Layers: The features extracted by the convolutional layers are flattened and passed through three fully connected (dense) layers with dropout for regularization.

Output Layer: The final layer has 3 units corresponding to the classes:

Class 0: Normal Rhythm

Class 1: VT/VF

Class 2: Noise

The model weights are initialized using Kaiming Normalization for convolutional layers and Normal distribution for linear layers.

3. Training
Optimizer: Adam with a learning rate of 0.001 and weight decay of 1e-4.

Loss Function: Cross-Entropy Loss.

Data Split: The combined dataset is split into training (70%), validation (15%), and testing (15%) sets.

Epochs: The model is trained for 100 epochs.

Environment: The training is configured to run on a CUDA-enabled GPU if available.

📚 Datasets
This model is trained on a combination of four well-known ECG databases to ensure diversity and robustness.

MIT-BIH Arrhythmia Database (ADB_DIR): Provides examples of normal sinus rhythm and other non-VT/VF arrhythmias (Class 0).

Malignant Ventricular Ectopy Database (VFDB_DIR): Contains recordings of life-threatening ventricular arrhythmias (Class 1).

Creighton University Ventricular Tachyarrhythmia Database (CUVTDB_DIR): Provides additional examples of ventricular fibrillation (Class 1).

MIT-BIH Noise Stress Test Database (NSTDB_DIR): Used to train the model to recognize and reject noisy signals (Class 2).

🚀 Getting Started
Follow these instructions to set up and run the project locally.

Prerequisites
Python 3.x

Access to the required ECG databases. You can download them from PhysioNet.

Installation
Clone the repository (or download the notebook):

Bash

git clone https://your-repository-url.git
cd your-repository-url
Install the required Python libraries:

Bash

pip install wfdb numpy scipy scikit-learn torch torchaudio torchvision matplotlib
Usage
Download the Datasets: Download the four datasets mentioned above and place them in your desired location.

Configure Paths: Open the Jupyter Notebook (The_Best_one.ipynb) and update the directory paths in the Parameter cell to point to your dataset locations.

Python

VFDB_DIR   = "/path/to/your/VFDB"
CUVTDB_DIR = "/path/to/your/CUVTDB"
ADB_DIR    = "/path/to/your/MITDB"
NSTDB_DIR  = "/path/to/your/NSTDB"
Run the Notebook: Execute the cells in the notebook sequentially. The main() function will handle data loading, preprocessing, model training, and validation.

📈 Results
After 100 epochs of training, the model achieves high performance on the validation set.

Training Accuracy: ~97.92%

Validation Accuracy: ~97.62%

The consistent performance between the training and validation sets suggests that the model generalizes well without significant overfitting. Further evaluation on the hold-out test set is recommended to confirm its real-world performance.
