# Food Image Triplet Prediction

This project involves processing food images, extracting features using a pre-trained deep learning model, and training a model to predict relationships between triplets of images. The goal is to determine whether a given triplet of images follows a specific relationship (e.g., A is more similar to B than to C).

## Overview

The project consists of the following steps:

1. **Preprocessing**: Unzip the food image archive and rename the image files.
2. **Feature Extraction**: Use a pre-trained InceptionResNetV2 model to extract features from the images.
3. **Triplet Generation**: Create feature tensors for training and testing triplets.
4. **Model Training**: Train a neural network to predict relationships between triplets.
5. **Prediction**: Generate predictions for test triplets and save the results to a submission file.

## Code Details

### Dependencies

The code requires the following Python libraries:

- `numpy`
- `pandas`
- `tensorflow`
- `opencv-python`
- `glob`
- `zipfile`

You can install the dependencies using:

```bash
pip install numpy pandas tensorflow opencv-python
