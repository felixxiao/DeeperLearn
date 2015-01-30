# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 17:41:41 2015

@author: Felix
"""

import numpy as np
from matplotlib import pyplot
from sklearn import preprocessing

def predict(X, alpha, beta):
    Z = 1 / (1.0 + np.exp(- np.dot(X, alpha)))
    Z[:,0] = 1.0
    return np.dot(Z, beta)

"------------------ Data import and preprocessing ----------------------------"
addr = "..\\bike_share_train.csv"
data = np.genfromtxt(addr, skiprows = 1, delimiter = ',')
data = data[:,1:-1]
P = data.shape[1] - 1
data = np.hstack((np.ones((len(data),1)), data))
preprocessing.normalize(data)
np.random.shuffle(data)
test = data[:int(0.2*len(data)),:]
data = data[int(0.2*len(data)):,:]

#%%
"------------------ Learning parameters --------------------------------------"
M      = 3        # Number of hidden units (excluding bias unit)
rate   = 0.00001  # Descent rate
epochs = 100      # Maximum number of passes through the data set
weight = 0.0      # Higher means more regularized

#%%
"------------------ Gradient Descent -----------------------------------------"
beta  = np.random.rand(M+1) - 0.5
alpha = np.random.rand(P+1, M+1) - 0.5

train_errors = []
test_errors = []
for t in range(epochs):
    Z = 1 / (1.0 + np.exp(- np.dot(data[:,:-1], alpha)))
    Z[:,0] = 1.0
    delta = np.dot(Z, beta) - data[:,-1]
    
    grad_beta = np.dot(delta, Z)
    grad_alpha = np.dot(data[:,:-1].T, delta[:,None] * Z * (1.0-Z) * beta)
    
    beta  -= rate * grad_beta  - weight * beta
    alpha -= rate * grad_alpha - weight * alpha
    
    train_errors.append(np.linalg.norm(delta) / len(delta))
    test_pred = predict(test[:,:-1], alpha, beta)
    test_errors.append(np.linalg.norm(test[:,-1] - test_pred) / len(test))

pyplot.plot(train_errors, c = 'b')
pyplot.plot(test_errors, c = 'r')
pyplot.xlabel("Training epoch")
pyplot.ylabel("Mean Squared error")

#%%
"------------------ Benchmarking ---------------------------------------------"
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

print "Best test error:   ", np.min(test_errors)

print "OLS:               ",
ols = LinearRegression()
ols.fit(data[:,:-1], data[:,-1:])
print np.linalg.norm(ols.predict(test[:,:-1]) - test[:,-1:]) / len(test)
print "OLS R^2:           ", ols.score(test[:,:-1], test[:,-1:])

print "Nearest Neighbors: ",
nearest = KNeighborsRegressor(n_neighbors = 20)
nearest.fit(data[:,:-1], data[:,-1:])
print np.linalg.norm(nearest.predict(test[:,:-1]) - test[:,-1:]) / len(test)