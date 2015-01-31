# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 17:41:41 2015

@author: Felix
"""

import numpy as np
from matplotlib import pyplot
from sklearn import preprocessing

"------------------ Data import and preprocessing ----------------------------"
addr = "wine.data"
data = np.genfromtxt(addr, skiprows = 0, delimiter = ',')
P = data.shape[1] - 1
#data = np.hstack((np.ones((len(data),1)), data))
data = preprocessing.scale(data, axis = 0)
data[:,0] = 1.0
np.random.shuffle(data)

#%%
"------------------ Learning parameters --------------------------------------"
M      = 5        # Number of hidden units (excluding bias unit)
rate   = 0.001    # Descent rate
epochs = 1000     # Maximum number of passes through the data set
weight = 0.0      # Higher means more regularized

#%%
"------------------ Gradient Descent -----------------------------------------"
beta  = np.random.rand(P+1, M+1) - 0.5
alpha = np.random.rand(P+1, M+1) - 0.5

train_errors = []
test_errors = []
for t in range(epochs):
    Z = 1 / (1.0 + np.exp(- np.dot(data, alpha)))
    Z[:,0] = 1.0    # Bias unit
    delta = np.dot(Z, beta.T) - data
    
    grad_beta = np.dot(delta.T, Z)
    grad_alpha = np.dot(data.T, np.dot(delta, beta) * Z * (1.0 - Z))
    
    beta  -= rate * grad_beta  - weight * beta
    alpha -= rate * grad_alpha - weight * alpha
    
    train_errors.append(np.linalg.norm(delta) / len(delta))

pyplot.plot(train_errors, c = 'b')
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