import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def model(X, Y, learning_rate, iteration):

  m = Y.size
  omega = np.zeros((X.shape[1], 1))
  for i in range(iteration):

    y_pred = np.dot(X, omega)

#compute gradient
    d_omega = (2/m)*np.dot(X.T, y_pred - Y)
    omega = omega - learning_rate*d_omega

  return omega

data = pd.read_csv("train.csv")
data = data.drop(["Id"], axis=1)

train = data.values
Y = train[:,0].reshape(train.shape[0],1)
X = train[:, 1:6]

x1 = X[:,0]
x2 = X[:,1]
x3 = X[:,2]
x4 = X[:,3]
x5 = X[:,4]
ones = np.ones((X.shape[0],))

X_train = np.vstack([x1,x2,x3,x4,x5,np.square(x1), np.square(x2), np.square(x3)
, np.square(x4), np.square(x5), np.exp(x1), np.exp(x2), np.exp(x3), np.exp(x4)
, np.exp(x5), np.cos(x1), np.cos(x2), np.cos(x3), np.cos(x4), np.cos(x5)
, ones]).T


iteration= 10000
learning_rate = 0.001
omega = model(X_train,Y,learning_rate, iteration)

with open('sample.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    writer.writerows([omega[index][0]] for index in range(0, len(omega)))


