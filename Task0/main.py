import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


#read csv file
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#get id colum(we need this later)
id_test = test["Id"]

#drop 'id' column
train = train.drop(["Id"], axis = 1)
test = test.drop(["Id"], axis = 1)

#X and Y matrices for train data
train = train.values
Y = train[:,0].reshape(train.shape[0],1)
X = train[:, 1:11]

#X and Y matrices for test data
test = test.values
Y_test = np.zeros((test.shape[0],)).reshape(test.shape[0],1)
X_test = test[:, 0:10]

def model(X, Y, learning_rate, iteration):

  m = Y.size
  omega = np.zeros((X.shape[1], 1))
  RMSE_list = []
  for i in range(iteration):

    y_pred = np.dot(X, omega)

#compute error on predicted data
    RMSE = mean_squared_error(Y, y_pred)**0.5

#compute gradient
    d_omega = (2/m)*np.dot(X.T, y_pred - Y)
    omega = omega - learning_rate*d_omega

#print error every 10 iterations
    RMSE_list.append(RMSE)
    if(i%(iteration/10) == 0):
      print("RMSE is:", RMSE)

  return omega, RMSE_list

#train model on train data
iteration = 10000
learning_rate = 0.0000005
omega, RMSE_list = model(X,Y,learning_rate, iteration)

#compute y of test data
Y_test = np.dot(X_test,omega)

#put id and y in dictionary
mydict = {'Id': 'y'}
i = 0
for id in id_test:
  mydict[id] = Y_test[i][0]
  i += 1

#write dictonary to csv file
with open('sample.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in mydict.items():
       writer.writerow([key, value])
          
