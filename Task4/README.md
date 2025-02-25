# Task 4: Molecular Property Prediction

This project involves predicting molecular properties using a deep learning model. The task uses pre-trained features and labels, as well as training and testing datasets, to build and evaluate a neural network model. The goal is to predict properties for test molecules and generate a submission file.

## Overview

The project consists of the following steps:

1. **Data Loading**: Unzip and load pre-trained features, labels, training features, training labels, and test features.
2. **Model Building**: Construct a neural network model using pre-trained features and fine-tune it with training data.
3. **Prediction**: Use the trained model to predict properties for test molecules.
4. **Submission**: Save the predictions to a CSV file for submission.

## Code Details

### Dependencies

The code requires the following Python libraries:

- `numpy`
- `pandas`
- `tensorflow`

You can install the dependencies using:

```bash
pip install numpy pandas tensorflow
