import numpy as np
from helpers import *
from implementations import *
from clean_data import clean_data
from visualize_features import visualize_features
from cross_validation import *
%load_ext autoreload
%autoreload 2

 #####################################################

submission = False
train_ratio = 0.7
seed = 1

max_iters = 200
gamma = 0.00001
degree = 2

print("_-_-_-_-_- Model Construction using Logistic Regression _-_-_-_-_-\n")

print("Loading Data ...\n")
# Load Data
path_dataset_train = "train.csv"
yb, input_data, ids = load_csv_data(path_dataset_train)

path_dataset_test = "test.csv"
yb_t, input_data_t, ids_t = load_csv_data(path_dataset_test)

yb[np.where(yb == -1)] = 0
yb_test[np.where(yb_test == -1)] = 0

 #####################################################
print("Cleaning Data ...\n")
    
# Data Cleaning
batches = clean_data(input_data,yb,ids)
batches_test = clean_data(input_data_t,yb_t,ids_t, test=True) #Doesn't delete data points (only filters features\

 #####################################################


if (submission):
    y_pred_sub = np.ones(len(ids_t))
else :
    score = 0;

i = 0;
for [tx, y, ids],[tx_test,y_test,ids_test] in zip(batches,batches_test) :
    
    print("------ Computing weights for batch {} -------\n".format(i))
    
             #### Feature Visualization (uncomment to test) ####
        
    # features = np.array([1,4,5]) # Features to compare
    # visualize_features(tx0,y0,features,'Feature_comparision')
    
    if (not(submission)) : # We overwrite the test data
        tx_train, y_train, tx_test, y_test, ids_train, ids_test = split_data(tx, y, ids, train_ratio, seed=1)
    else :
        y_train = y
        tx_train = tx
        ids_train = ids
    
        
    tx_poly_train = build_poly_f(tx_train, degree)
    
    initial_w = np.zeros(tx_poly_train.shape[1])

    _, w = logistic_regression(y_train, tx_poly_train, initial_w, max_iters, gamma)
    w = w[-1]
    
    tx_poly_test = build_poly_f(tx_test, degree)
    
    print("Predicting labels ...")
    
    y_pred = predict_labels(w,tx_poly_test)
    
    if (submission):
        y_pred_sub[ids_test] = y_pred
    else :
        score_aux = find_score(y_pred,y_test)
        score += score_aux/4.
        
        print("Score for batch {} = {} \n".format(i,score_aux))

    i +=1
    
    
if (submission):
    create_csv_submission(ids_t, y_pred_sub, 'submission.csv')
else :
    print("Final score = {} \n".format(score))
    
