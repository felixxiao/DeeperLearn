# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 17:41:41 2015

@author: Felix
"""

import numpy as np
from matplotlib import pyplot
from sklearn import preprocessing

"------------------ Data import and preprocessing ----------------------"
addr = "wine.data"
data = np.genfromtxt(addr, skiprows = 0, delimiter = ',')
P = data.shape[1] - 1
#data = np.hstack((np.ones((len(data),1)), data))
data = preprocessing.scale(data, axis = 0)
data[:,0] = 1.0
np.random.shuffle(data)

#%%
"------------------ Learning parameters --------------------------------"
M      = 10       # Number of hidden units (excluding bias unit)
rate   = 0.001    # Descent rate
epochs = 1000     # Maximum number of passes through the data set
weight = 0.0      # Higher means more regularized

#%%
"------------------ Gradient Descent -----------------------------------"
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
    
    train_errors.append(np.linalg.norm(delta[:,1:])**2 / len(delta))

pyplot.plot(train_errors, c = 'b')
pyplot.xlabel("Training Epoch")
pyplot.ylabel("Mean Squared Error")

print "Mean squared encoding errors for each feature (0th feature is dummy)"
print np.linalg.norm(delta, axis = 0)**2 / len(delta)
print
print "Activation frequency of each hidden unit (0th unit is bias unit)"
print np.sum(Z > 0.5, axis = 0) / float(len(Z))