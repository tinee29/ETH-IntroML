# Ridge Regression with Cross-Validation

## Overview

This project implements **Ridge Regression** using the **closed-form solution** and performs **k-fold cross-validation** to evaluate model performance. The model is trained on a dataset from `train.csv`, and the **Root Mean Squared Error (RMSE)** for different regularization strengths (**alpha values**) is computed and saved in `sample.csv`.

## How It Works

1. **Data Loading**:
   - The script reads `train.csv` for training data.
   - The first column represents the target variable (**y**), and columns 2 to 14 represent the features (**X**).

2. **Model Training (Ridge Regression)**:
   - Ridge regression applies L2 regularization to the linear regression model.
   - Formula used:
     $`
\theta = (X^T X + \alpha I)^{-1} X^T y
`$


3. **Cross-Validation**:
   - **10-fold cross-validation** is used to split the data.
   - For each **alpha** value in `[0.1, 1, 10, 100, 200]`, the model is trained and evaluated.
   - The average **RMSE** across all folds is calculated.

4. **Output Generation**:
   - The average RMSE values for each **alpha** are saved in `sample.csv`.

## Dependencies

Ensure you have the following libraries installed:

- `numpy`
- `pandas`
- `scikit-learn`

Install them using:

```bash
pip install numpy pandas scikit-learn
```

## Usage

1. Ensure `train.csv` is in the same directory as `main.py`.

2. Run the script using:

```bash
python main.py
```

3. Check the RMSE output in `sample.csv`.

## Output

The `sample.csv` file contains RMSE values corresponding to each alpha value, one per line. Example output:

```
5.4321
4.9876
4.7654
4.6543
4.5432
```

## Parameters

- **Alpha values**: `[0.1, 1, 10, 100, 200]` (controls L2 regularization strength)
- **K-Folds**: `10` (for cross-validation)

## Notes

- The model uses the closed-form solution for efficiency.
- Increasing **alpha** increases regularization, reducing model complexity but potentially increasing bias.
- Modify the `alphaList` or `folds_number` variables to tune model performance.

## Example Output

Terminal output during training:

```
Alpha: 0.1, Average RMSE: 5.4321
Alpha: 1, Average RMSE: 4.9876
Alpha: 10, Average RMSE: 4.7654
Alpha: 100, Average RMSE: 4.6543
Alpha: 200, Average RMSE: 4.5432
```
