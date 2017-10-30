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
    
    
    
degree = 2
w0 = logistic_regression_poly(tx0, y0, tx_test0, y_test0, degree)
w1 = logistic_regression_poly(tx1, y1, tx_test1, y_test1, degree)
w2 = logistic_regression_poly(tx2, y2, tx_test2, y_test2, degree)
w3 = logistic_regression_poly(tx3, y3, tx_test3, y_test3, degree)
y_pred = construct_predictions(w0, w1, w2, w3, tx0, tx1, tx2, tx3, ids_t,ids0_t, ids1_t,ids2_t,ids3_t)