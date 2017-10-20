# -*- coding: utf-8 -*-
"""
Ridge Regression
"""

import numpy as np

def ridge_regression(y, tx, lambda_):
    
    I = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    txT = np.transpose(tx)
    
    mat1 = txT.dot(tx)+I
    mat2 = txT.dot(y)
    
    w = np.linalg.solve(mat1,mat2)    
    loss = compute_loss(y,tx,w)
    
    return loss, w


