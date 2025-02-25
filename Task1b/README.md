# Linear Regression Model with Feature Engineering

This repository contains a Python implementation of a linear regression model with feature engineering. The model is trained on a dataset (`train.csv`) and outputs the learned weights to a file (`sample.csv`).

## Overview

The code performs the following steps:

1. **Data Loading**: Loads the dataset from `train.csv`.
2. **Data Preprocessing**: Drops the `Id` column and splits the data into features (`X`) and target (`Y`).
3. **Feature Engineering**: Enhances the feature set by adding polynomial features (squares), exponential features, and trigonometric features (cosine).
4. **Model Training**: Implements a gradient descent algorithm to train a linear regression model.
5. **Output**: Saves the learned weights to `sample.csv`.

## Code Details

### Dependencies

The code requires the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

You can install the dependencies using:

```bash
pip install numpy pandas matplotlib scikit-learn
