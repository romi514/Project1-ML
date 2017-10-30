# Useful starting lines
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
%load_ext autoreload
%autoreload 2

#Load data
from helpers import *
from clean_data import *

# load data.
y, train_data, ids = load_csv_data("train.csv", sub_sample=False)
test_y, test_data, ids_t = load_csv_data("test.csv", sub_sample=False)

y[np.where(y == -1)] = 0
test_y[np.where(test_y == -1)] = 0

#Cleaned data
tx0, tx1, tx2, tx3, y0, y1, y2, y3, ids0, ids1, ids2, ids3 = clean_data(train_data,y,ids)
tx_test0, tx_test1, tx_test2, tx_test3, y_test0, y_test1, y_test2, y_test3, ids0_t, ids1_t, ids2_t, ids3_t = clean_data(test_data,test_y, ids_t, test = True) # Note: Didn't use ids, no use

def logistic_regression_poly(x_train, y_train, x_test, y_test, degree):
    gamma = 0.00001
    max_iters = 200
    
    
    tx_poly_tr = build_poly_f(x_train, degree)
    tx_poly_te = build_poly_f(x_test, degree)

    initial_w = np.zeros((1,tx_poly_tr.shape[1]))[0]
    losses, ws = logistic_regression(y_train, tx_poly_tr, initial_w, max_iters, gamma)
    w = ws[-1]
    return w
    
    
    
degree = 2 # Best degree found manually.
#degree = best_degree(y_train, x_train, gamma, max_iters)
w0 = logistic_regression_poly(tx0, y0, tx_test0, y_test0, degree)
w1 = logistic_regression_poly(tx1, y1, tx_test1, y_test1, degree)
w2 = logistic_regression_poly(tx2, y2, tx_test2, y_test2, degree)
w3 = logistic_regression_poly(tx3, y3, tx_test3, y_test3, degree)
y_pred = construct_predictions(w0, w1, w2, w3, tx0, tx1, tx2, tx3, ids_t,ids0_t, ids1_t,ids2_t,ids3_t)



def cross_validation_logistic(y, x, k_indices, k, lambda_,degree, max_iters):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train: TODO
    test_indices = k_indices[k]
    train_indices =k_indices[~(np.arange(k_indices.shape[0]) == k)].reshape(-1)

    x_test = x[test_indices]
    x_train = x[train_indices]
    
       
    y_test = y[test_indices]
    y_train = y[train_indices]
    
    print(x_train.shape)
    tx_train = build_poly_f(x_train, degree)
    tx_test = build_poly_f(x_test, degree)
    print(tx_train.shape)
    
    initial_w = np.zeros((1,tx_train.shape[1]))[0]
    # ridge regression
    loss, w= logistic_regression(y_train,tx_train,initial_w, max_iters, lambda_)

    # calculate the loss for train and test data
    loss_tr = np.sqrt(2 * compute_loss(y_train, tx_train, w[-1]))
    loss_te = np.sqrt(2 * compute_loss(y_test, tx_test, w[-1]))  
    return w,loss_tr, loss_te
    
def best_degree_logistic(y,tx, l, max_iters):
    seed = 1
    k_fold = 10
    lambda_ = l
    degrees = np.linspace(1,7,7)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    for ind, d in enumerate(degrees):
        rmse_train = []
        rmse_test = []
        
        for k in range(k_fold):
            w,loss_train, loss_test = cross_validation_logistic(y, tx, k_indices, k, lambda_, int(d), max_iters)
            rmse_train.append(loss_train)
            rmse_test.append(loss_test)
        rmse_tr.append(np.mean(rmse_train))
        rmse_te.append(np.mean(rmse_test))

    return int(degrees[rmse_te.index(np.min(rmse_te))])