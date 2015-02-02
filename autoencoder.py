# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 17:41:41 2015

@author: Felix
"""

import numpy as np
from matplotlib import pyplot
from sklearn import preprocessing

def cost(data, alpha, beta, weight, p):
    Z = 1 / (1.0 + np.exp(- np.dot(data, alpha)))
    Z[:,0] = 1.0    # Bias unit
    delta = np.dot(Z, beta.T) - data
    fit_cost = np.linalg.norm(delta[:,1:])**2 / len(data) / 2.0
    
    freq = np.mean(Z, axis = 0)
    freq[0] = p
    kl_one = p*np.log(p / freq)
    kl_one_sum = np.sum(kl_one)
    if np.isnan(kl_one_sum) or np.isinf(kl_one_sum):
        print "one", kl_one
        print freq
        raise ArithmeticError
        
    kl_two = (1-p)*np.log((1-p) / (1-freq))
    kl_two_sum = np.sum(kl_two)
    if np.isnan(kl_two_sum) or np.isinf(kl_two_sum):
        print "two", kl_two
        print freq
        raise ArithmeticError
    
    sparse_cost = kl_one_sum + kl_two_sum
    return fit_cost + weight * sparse_cost


def empiricalGradients(data, alpha, beta, P, M, epsilon, weight, sparse):
    grad_beta = np.zeros((P+1, M+1))
    grad_alpha = np.zeros((P+1, M+1))
    for p in range(P+1):
        for m in range(M+1):
            alpha[p,m] += epsilon
            J_plus = cost(data, alpha, beta, weight, sparse)
            alpha[p,m] -= epsilon + epsilon
            J_minus = cost(data, alpha, beta, weight, sparse)
            grad_alpha[p,m] = (J_plus - J_minus) / 2 / epsilon
            alpha[p,m] += epsilon
            
            beta[p,m] += epsilon
            J_plus = cost(data, alpha, beta, weight, sparse)
            beta[p,m] -= epsilon + epsilon
            J_minus = cost(data, alpha, beta, weight, sparse)
            grad_beta[p,m] = (J_plus - J_minus) / 2 / epsilon
            beta[p,m] += epsilon            
            
    return (grad_alpha, grad_beta)

#%%
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
M       = 3        # Number of hidden units (excluding bias unit)
rate    = 0.03     # Descent rate
epochs  = 20000    # Maximum number of passes through the data set
weight  = 5.0      # Higher means larger penalty for non-sparsity
sparse  = 0.10     # Constrain hidden units to fire about this often
epsilon = 0.00001  # For gradient checking

#%%
"------------------ Gradient Descent -----------------------------------"
beta  = np.random.rand(P+1, M+1) - 0.5
alpha = np.random.rand(P+1, M+1) - 0.5

train_errors = []
for t in range(epochs):
    Z = 1 / (1.0 + np.exp(- np.dot(data, alpha)))
    Z[:,0] = 1.0    # Bias unit
    delta = np.dot(Z, beta.T) - data
    
    grad_beta = np.dot(delta.T, Z) / len(data)
    grad_alpha = np.dot(data.T, np.dot(delta, beta) * Z * (1.0 - Z)) \
        / len(data)
    
    freq = np.mean(Z, axis = 0)
    freq[0] = sparse
    grad_alpha += weight * np.dot(data.T, Z * (1.0 - Z)) * \
        (- sparse / freq + (1.0 - sparse) / (1.0 - freq)) / len(data)
    
    if t < epochs - 1:
        beta  -= rate * grad_beta
        alpha -= rate * grad_alpha
    
        train_errors.append(np.linalg.norm(delta[:,1:])**2 / len(delta))

#%%
"------------------ Diagnostics ----------------------------------------"
empirical_grad = empiricalGradients(data, alpha, beta, P, M,epsilon,
                                    weight, sparse)
print "---------------- alpha grad error -------------------"
print grad_alpha - empirical_grad[0]
print
print "---------------- beta grad error --------------------"
print grad_beta - empirical_grad[1]
print

pyplot.plot(train_errors, c = 'b')
pyplot.xlabel("Training Epoch")
pyplot.ylabel("Mean Squared Error")

print "Mean squared encoding errors for each feature (0th feature is dummy)"
print np.linalg.norm(delta, axis = 0)**2 / len(delta)
print
print "Activation frequency of each hidden unit (0th unit is bias unit)"
print np.sum(Z > 0.5, axis = 0) / float(len(Z))
print
print "Mean activations"
print freq
