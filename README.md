# CNN-fashion-mnist-pytorch
A Convolutional Neural Network (CNN) built using **PyTorch** to classify clothing images from the Fashion MNIST dataset. This project trains a deep learning model and evaluates its accuracy on both training and test sets.

---

## 📌 Dataset

The model uses the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset, which is a drop-in replacement for the classic MNIST dataset. It contains:
- 60,000 training images
- 10,000 test images
- 10 classes of fashion items (T-shirt, Trouser, Pullover, etc.)
- Grayscale images of size 28x28 pixels

---

## 🚀 Features

- Custom PyTorch `Dataset` class for loading CSV data
- CNN with 2 convolutional layers, ReLU, BatchNorm, MaxPooling
- Fully connected layers with dropout for classification
- Accuracy evaluation on training and test sets
- GPU support (CUDA)

---

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
bash
git clone https://github.com/Ace007-0/CNN-fashion-mnist-pytorch.git
cd CNN-fashion-mnist-pytorch


### 2. Install Requirements
Make sure you have Python ≥3.8. Then install dependencies:
bash
pip install torch torchvision pandas scikit-learn

### 3. Download Dataset
Download fashion-mnist_train.csv:- (https://www.kaggle.com/datasets/zalando-research/fashionmnist)

### 4. Model Architecture
Input: [1 x 28 x 28] grayscale image
Feature Extractor:
    - Conv2d(1, 32, kernel_size=3, padding='same') + ReLU + BatchNorm + MaxPool
    - Conv2d(32, 64, kernel_size=3, padding='same') + ReLU + BatchNorm + MaxPool
Classifier:
    - Flatten
    - Linear(64*7*7 → 128) + ReLU + Dropout
    - Linear(128 → 64) + ReLU + Dropout
    - Linear(64 → 10)  → class scores

Please feel free to provide feedback.



