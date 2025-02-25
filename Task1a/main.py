import csv
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

#theta = inv(xTx + lam*I) * xTy
def fit(alpha, X, y):
  XTX = np.dot(X.T, X) + alpha * np.identity(X.shape[1])
  omega = np.dot(np.dot(np.linalg.inv(XTX), X.T), y_train)
  return omega 

#read csv file
data = pd.read_csv("train.csv")

#X and Y matrices for train data
data = data.values
X = data[:, 1:14]
y = data[:,0].reshape(data.shape[0],1)

#generate folds
folds_number = 10
kf = KFold(n_splits=folds_number)
kf.get_n_splits(X)


alphaList = np.array([0.1,1,10,100,200])
RMSE_list = []
#iterate over lambda values
for alpha in alphaList:
  sumRMSE = 0
  #iterate over folds
  for train_index, test_index in kf.split(X):
    #get X train and test folds
    X_train, X_test = X[train_index], X[test_index]
    #get y train and test folds
    y_train, y_test = y[train_index], y[test_index]
    #prepend ones column to X train and test
    X_train = np.c_[np.ones((X_train.shape[0],)),X_train]
    X_test = np.c_[np.ones((X_test.shape[0],)),X_test]
    #train model
    omega = fit(alpha, X_train, y_train);
    y_t = np.dot(X_test,omega)
    #sum up RMSE
    sumRMSE += mean_squared_error(y_t, y_test)**0.5
  #avarage of RMSE per lambda value
  RMSE = sumRMSE / folds_number
  RMSE_list.append(RMSE)

#write RMSE values as column in sample.csv
with open('sample.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    writer.writerows([RMSE_list[index]] for index in range(0, len(RMSE_list)))
