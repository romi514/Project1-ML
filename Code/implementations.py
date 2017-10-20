# -*- coding: utf-8 -*-

"""Function used to compute the loss."""
import numpy as np
from numpy.linalg import *
from helpers import *



def compute_loss(y, tx, w):
    """Calculate the loss using MSE."""
    e = y - tx.dot(w)
    N = tx.shape[0]
    et = np.transpose(e)
    L = et.dot(e)/(2*N)
    return L


"""
    Least squares
"""


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



"""
    Ridge Regression
"""


def ridge_regression(y, tx, lambda_):
    
    I = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    txT = np.transpose(tx)
    
    mat1 = txT.dot(tx)+I
    mat2 = txT.dot(y)
    
    w = np.linalg.solve(mat1,mat2)
    loss = compute_loss(y,tx,w)
    
    return loss, w


"""
    Logistic regression
"""


def sigmoid(t):
    """apply sigmoid function on t."""
    aux = np.exp(t)
    return aux/(1+aux)

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    return np.sum(np.log(1+np.exp(tx.dot(w))) - np.multiply(y,tx.dot(w)))

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return np.transpose(tx).dot(sigmoid(tx.dot(w))-y)

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    aux = sigmoid(tx.dot(w))
    S = np.multiply(aux,1-aux)
    S = np.eye(S.shape[0])*S
    
    return np.transpose(tx).dot(S).dot(tx)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression using gradient descent"""
    
    # init parameters
    threshold = 1e-8
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for iter in range(max_iters):
        loss = calculate_loss(y,tx,w)
        gradient = calculate_gradient(y,tx,w)
        w = w - gamma*gradient
        
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return losses, ws

def logistic_regression_with_Newton(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression using gradient descent and Newton method"""
    
    # init parameters
    threshold = 1e-8
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for iter in range(max_iters):
        loss = calculate_loss(y,tx,w)
        gradient = calculate_gradient(y,tx,w)
        hessian = calculate_hessian(y,tx,w)
        
        w = w - np.linalg.inv(hess).dot(grad)
        
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return losses, ws

def reg_logistic_regression_with_Newton(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Penalized logistic regression using gradient descent and Newton method"""
    
    # init parameters
    threshold = 1e-8
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for iter in range(max_iters):
        loss = calculate_loss(y,tx,w) + lambda_*np.linalg.norm(w)/2
        gradient = calculate_gradient(y,tx,w) + lambda_*w
        hessian = calculate_hessian(y,tx,w) + lambda_*np.eye(w.shape[0])
        
        w = w - np.linalg.inv(hess).dot(grad)
        
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return losses, ws

# Keep only one regularized function with/without Newton ? (lambda = 0 for normal regression)


