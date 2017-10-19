import numpy as np
from costs import *
from numpy.linalg import *


# Least squares regression using normal equations
def least_squares(y, tx):
    """calculate the least squares solution."""
    gram = (tx.transpose()).dot(tx)
    b = (tx.transpose()).dot(y) 
    w = np.linalg.solve(gram,b)
    mse = compute_mse(y,tx,w)
    return mse, w


#Linear regression using gradient descent
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - tx.dot(w)
    N = tx.shape[0]
    grad = -np.transpose(tx).dot(e)/N
    return grad


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)        
        w = w - (gamma*gradient)
        ws.append(w)
        losses.append(loss)

    return losses, ws
    
# Linear regression using stochastic gradient descent  #BESMA


