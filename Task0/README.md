# Linear Regression with Gradient Descent

## Overview

This project implements **linear regression** from scratch using **gradient descent** in Python. It reads training and test data from CSV files, trains a linear model, and generates predictions for the test dataset. The predictions are saved in an output CSV file.

## How It Works

1. **Data Loading**:
   - The script reads `train.csv` and `test.csv` as input files.
   - The `Id` column is extracted and preserved for the output.
   - The first column in the training data is treated as the target variable **Y**, while the remaining columns are used as features **X**.

2. **Model Training**:
   - A **linear regression model** is trained using **gradient descent**.
   - The model minimizes the **Root Mean Squared Error (RMSE)**.
   - Model parameters are updated iteratively based on the computed gradient.

3. **Prediction & Output**:
   - Predictions are generated for the test set using the learned model.
   - The results are stored in `sample.csv`, which maps each test `Id` to its corresponding predicted value.

## Dependencies

Ensure the following Python libraries are installed:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

To install these dependencies, run:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

1. Ensure `train.csv` and `test.csv` are present in the same directory as `main.py`.

2. Run the script using:

```bash
python main.py
```

3. The predictions will be saved in `sample.csv`.

## Output

The output CSV file (`sample.csv`) will have the following format:

```
Id,y
1,5.67
2,4.32
3,6.45
```

## Parameters

- **Learning Rate**: `0.0000005`
- **Iterations**: `10000`

## Notes

- RMSE is printed every 10% of the training iterations to track progress.
- Modify the learning rate and number of iterations to fine-tune model performance.

## Example

Sample terminal output during training:

```
RMSE is: 8.1234
RMSE is: 5.7890
RMSE is: 4.5678
```
