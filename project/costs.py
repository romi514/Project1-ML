# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np

def compute_loss(y, tx, w):
    """Calculate the loss using MSE."""
    e = y - tx.dot(w)
    N = tx.shape[0]
    et = np.transpose(e)
    L = et.dot(e)/(2*N)
    return L
