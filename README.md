# MNIST Handwritten Digit Classification using MLP in TensorFlow

This repository contains a Jupyter Notebook that implements a Multi-Layer Perceptron (MLP) model using TensorFlow/Keras to classify images from the MNIST dataset.

## Overview

- **Dataset**: MNIST (28x28 grayscale images of handwritten digits)
- **Model**: Multi-Layer Perceptron (MLP) with 3 hidden layers (128 units each) and ReLU activation.
- **Output Layer**: Softmax activation for multi-class classification.
- **Optimizer**: Adam with a learning rate of 1e-3.
- **Loss Function**: Categorical Cross-Entropy.
- **Training**: 10 epochs with a batch size of 64.
- **Validation**: 20% of the training data is used for validation.
- **Test Accuracy**: ~97.74%

## Code Structure

1. **Data Loading and Preprocessing**:
   - Load MNIST dataset.
   - Normalize and reshape images.
   - One-hot encode labels.

2. **Model Definition**:
   - Custom `MLP` class with configurable layers and units.
   - Sequential model with input, hidden, and output layers.

3. **Training**:
   - Train the model for 10 epochs.
   - Measure and print training time.

4. **Evaluation**:
   - Evaluate the model on the test set.
   - Print test accuracy and loss.

5. **Visualization**:
   - Display the first image from the training set.

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/mohammedobaidraza/MNIST-MLP-TensorFlow.git
   cd MNIST-MLP-TensorFlow


2. Install dependencies:
    ```bash
    pip install -r requirements.txt

3. Run the Jupyter Notebook:
    ```bash
    jupyter notebook
4. Open and run Untitled2.ipynb.


## Results: 
```bash
       Total params: 134,794 (526.54 KB)
       Trainable params: 134,794 (526.54 KB)
       Non-trainable params: 0 (0.00 B) 


