import numpy as np
from costs import *
from numpy.linalg import *
from helpers import *


# Least squares regression using normal equations
def least_squares(y, tx):
    """calculate the least squares solution."""
    gram = (tx.transpose()).dot(tx)
    b = (tx.transpose()).dot(y) 
    w = np.linalg.solve(gram,b)
    losses = compute_loss(y,tx,w)
    return losses, w


#Linear regression using gradient descent
def compute_gradient(y, tx, w):
    """Compute the gradient for MSE Loss"""
    e = y - tx.dot(w)
    N = tx.shape[0]
    grad = -np.transpose(tx).dot(e)/N
    return grad


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
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
    
# Linear regression using stochastic gradient descent
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    "Stochastic gradient descent algorithm "
    
    batch_size = 1
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gradient = compute_gradient(minibatch_y,minibatch_tx,w)        
        loss = compute_loss(y, tx, w)        
        w = w - (gamma*gradient)
        ws.append(w)
        losses.append(loss)

    return losses, ws

# Change variable batch_size ? take away loss + weight tracking through algo? implement MAE loss with subgradient descente ?