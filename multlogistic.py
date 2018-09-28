#!/usr/bin/env python

import numpy as np
from scipy.optimize import minimize

def logist_nexp(theta, x):

    arg = -np.dot(x, theta)
    arg[arg > 100] = 100   # Handle overflow.
    return np.exp(arg)

def logist_reg(data, labels, cost_func, t0, lamb=1e-4, nc=2, tol=None, maxiter=100, disp=True):

    res = minimize(cost_func, t0, jac=True, method='L-BFGS-B', args=(data, labels, lamb, nc), tol=tol, options={'maxiter': maxiter, 'disp': disp})
    return res.x

def mult_logist_predict(theta, x, nc):

    norm = logist_nexp(-theta.T, x)
    return norm.T / np.sum(norm, axis=1)

# Assumes class labels are 0, 1, 2, ...
def mult_logist_cost(theta, x, y, lamb, nc):

    theta = theta.reshape(nc, x.shape[1])
    prob = mult_logist_predict(theta, x, nc)

    sm = 0
    for i in range(prob.shape[0]):
        wh = (np.where( y == i ))[0]
        sm -= np.sum(np.log(prob[i,wh]))
    sm += 0.5*lamb*np.sum(np.multiply(theta, theta))   # Weight decay term.

    indic = np.zeros(prob.shape, dtype=int)
    for i in range(prob.shape[0]):
        wh = (np.where( y == i ))[0]
        indic[i,wh] = 1
    grad = np.dot(prob - indic, x)
    grad += lamb*theta   # Weight decay term.
    grad = grad.reshape(grad.size)

    return [ sm, grad ]

