# -*- coding: utf-8 -*-
"""some helper functions """

import csv
import numpy as np

def split_data(x, y, ids, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    size = len(y)
    np.random.seed(seed)
    n = int(ratio*size)
    index = np.random.permutation(size)

    x_train = x[index[:n]]
    y_train = y[index[:n]]
    ids_train= ids[index[:n]]
    
    x_test = x[index[n:]]
    y_test = y[index[n:]]
    ids_test = ids[index[n:]]
    
    
    return x_train, y_train, x_test, y_test, ids_train, ids_test

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi = np.ones((x.shape[0],degree+1))
    for i in range(1, degree+1):
        phi[:,i] = np.power(x,i)
    return phi

def build_poly_f(x, degrees):
    """builds polynomial basis for each feature"""
    tx = np.ones(len(x))
    d = degrees
    for i in range(x.shape[1]):
        feature = x[:,i]
        tx_f = build_poly(feature,d)[:,1:]
        tx = np.c_[tx,tx_f]
    return tx

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data, logistic_reg = False): # For regression
    """Generates class predictions given weights, and a test data matrix"""
 
    y_pred = np.dot(data, weights)

    if (logistic_reg):    
        y_pred[np.where(y_pred <= .5)] = 0
        y_pred[np.where(y_pred > .5)] = 1
    else:
        y_pred[np.where(y_pred <= 0)] = -1
        y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
