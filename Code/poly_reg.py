# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
from implementations import *

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    size = len(y)
    np.random.seed(seed)
    n = int(ratio*size)
    index = np.random.permutation(size)

    x_train = x[index[:n]]
    y_train = y[index[:n]]
    
    x_test = x[index[n:]]
    y_test = y[index[n:]]
    
    
    return x_train, y_train, x_test, y_test

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi = np.ones((x.shape[0],degree+1))
    for i in range(1, degree+1):
        phi[:,i] = np.power(x,i)
    return phi

def build_poly_f(x, degrees):
    tx = np.ones(len(x))
    d = degrees
    for i in range(x.shape[1]):
        feature = x[:,i]
        tx_f = build_poly(feature,d)[:,1:]
        tx = np.c_[tx,tx_f]
    return tx

