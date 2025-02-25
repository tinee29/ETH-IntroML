# Machine Learning Project: ETH IntroML

This repository contains solutions for multiple tasks as part of the ETH Introduction to Machine Learning course. Each task focuses on a specific machine learning problem, including linear regression, ridge regression, feature engineering, medical data prediction, and food image triplet prediction.

## Tasks Overview

### Task 0: Linear Regression with Gradient Descent
- **Description**: Implements linear regression from scratch using gradient descent.
- **Input**: `train.csv`, `test.csv`
- **Output**: `sample.csv`
- **Key Features**:
  - Gradient descent optimization.
  - Root Mean Squared Error (RMSE) tracking.

### Task 1a: Ridge Regression with Cross-Validation
- **Description**: Implements ridge regression with k-fold cross-validation.
- **Input**: `train.csv`
- **Output**: `sample.csv`
- **Key Features**:
  - Closed-form solution for ridge regression.
  - 10-fold cross-validation.

### Task 1b: Linear Regression with Feature Engineering
- **Description**: Implements linear regression with polynomial, exponential, and trigonometric features.
- **Input**: `train.csv`
- **Output**: `sample.csv`
- **Key Features**:
  - Feature engineering (polynomial, exponential, cosine).
  - Gradient descent optimization.

### Task 2: Medical Data Prediction
- **Description**: Predicts medical outcomes using Random Forest Classifier and Ridge Regression.
- **Input**: `train_features.csv`, `test_features.csv`, `train_labels.csv`
- **Output**: `prediction.zip`
- **Key Features**:
  - Feature engineering for time-series data.
  - Random Forest Classifier for classification.
  - Ridge Regression for regression.

### Task 3: Food Image Triplet Prediction
- **Description**: Predicts relationships between triplets of food images using a pre-trained deep learning model.
- **Input**: `food.zip`, `train_triplets.txt`, `test_triplets.txt`
- **Output**: `output.txt`
- **Key Features**:
  - Pre-trained InceptionResNetV2 for feature extraction.
  - Triplet generation and neural network training.

### Task 4: Molecular Property Prediction
- **Description**: Predicts molecular properties using a deep learning model.
- **Input**: `pretrain_features.csv.zip`, `pretrain_labels.csv.zip`, `train_features.csv.zip`, `train_labels.csv.zip`, `test_features.csv.zip`, `sample.csv`
- **Output**: `output.csv`
- **Key Features**:
  - Pre-trained feature extraction.
  - Fine-tuning with additional layers.

## Getting Started

### Dependencies
All tasks require the following Python libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `tensorflow` (for Tasks 3 and 4)
- `opencv-python` (for Task 3)

Install the dependencies using:
```bash
pip install numpy pandas scikit-learn tensorflow opencv-python
