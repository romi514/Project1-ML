import numpy as np

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