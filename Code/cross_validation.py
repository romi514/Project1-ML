# -*- coding: utf-8 -*-

"""some helper functions"""
import csv
import numpy as np
from implementations import *
from helpers import build_poly_f


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_,degree):
    """return the loss of ridge regression."""

    test_indices = k_indices[k]
    train_indices =k_indices[~(np.arange(k_indices.shape[0]) == k)].reshape(-1)

    x_test = x[test_indices]
    x_train = x[train_indices]
    
       
    y_test = y[test_indices]
    y_train = y[train_indices]
    
    tx_train = build_poly_f(x_train, degree)
    tx_test = build_poly_f(x_test, degree)


    # ridge regression
    loss, w= ridge_regression(y_train,tx_train, lambda_)

    # calculate the loss for train and test data
    loss_tr = np.sqrt(2 * compute_loss(y_train, tx_train, w))
    loss_te = np.sqrt(2 * compute_loss(y_test, tx_test, w))  
    
    return w,loss_tr, loss_te

def best_lambda(y,tx,deg):
    """return the lambda for which the model fits best in terms on rmse"""
    
    seed = 10
    k_fold = 4
    lambdas = [0.0000001, 0.000001,0.00001,0.0001,0.001,0.01,0.1]
    degree =deg
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    weights = []
    
    # For each lambda, calculate the mean of the rmse for k_fold batches
    for ind, lambda_ in enumerate(lambdas):
        rmse_train = []
        rmse_test = []
        for k in range(k_fold):
            w,loss_train, loss_test = cross_validation(y, tx, k_indices, k, lambda_, degree)
            rmse_train.append(loss_train)
            rmse_test.append(loss_test)
            weights.append(w)
        rmse_tr.append(np.mean(rmse_train))
        rmse_te.append(np.mean(rmse_test))
    return lambdas[rmse_te.index(np.min(rmse_te))]


def best_degree(y,tx, l):
    """return the degree for which the model fits best in terms on rmse"""

    seed = 10
    k_fold = 10
    lambda_ = l
    degrees = np.linspace(1,7,7)
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    
    # For each polynomial degree, calculate the mean of the rmse for k_fold batches
    for ind, d in enumerate(degrees):
        rmse_train = []
        rmse_test = []
        for k in range(k_fold):
            w,loss_train, loss_test = cross_validation(y, tx, k_indices, k, lambda_, int(d))
            rmse_train.append(loss_train)
            rmse_test.append(loss_test)
        rmse_tr.append(np.mean(rmse_train))
        rmse_te.append(np.mean(rmse_test))

    return int(degrees[rmse_te.index(np.min(rmse_te))])

def find_score(y,y_pred):
    """Score the predictions accuracy (from 0 to 1) """

    count = 0
    for i in range(len(y_pred)):
        if(y_pred[i] == y[i]):
            count = count + 1       
    return (count/len(y_pred))
