from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_stable(x):
    if(x.ndim>1):
        max_x = np.max(x,axis=1)[:,np.newaxis]
        shiftx = x - max_x
        exp = np.exp(shiftx)
        row_sum = exp.sum(axis=1)[:,np.newaxis]
        return exp/row_sum
    else:
        shiftx = x - np.max(x)
        exp = np.exp(shiftx)
        return exp/exp.sum()
def softmax_unstable(x):
    if(x.ndim>1):
        exp = np.exp(x)
        row_sum = exp.sum(axis=1)[:,np.newaxis]
        return exp/row_sum
    else:
        exp = np.exp(x)
        return exp/exp.sum()
def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        log_scores = X[i]@W # shape: 1xD X DxC = (C,)
        prob = softmax_stable(log_scores)
        correct_class_prob = prob[y[i]]
        loss -= np.log(correct_class_prob)
        delta = np.zeros(num_classes)
        # dsoftmax = softmax_yi(delta-softmax_j), where delta = 1 if j=yi, else delta=0.
        # and so dW = softmax_j - delta, delta as above.
        # see https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        # or just derive it yourself.
        delta[y[i]] = 1
        grad = np.outer(X[i],prob - delta)
        dW += grad
    loss /= num_train
    loss += reg * (W**2).sum()
    dW /= num_train
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    log_scores = X@W # shape: NxD X DxC = (NxC)
    prob = softmax_stable(log_scores)
    correct_class_prob = prob[range(num_train),y]
    loss -= (np.log(correct_class_prob)).mean()
    prob[range(num_train),y] -= 1
    # dsoftmax = softmax_yi(delta-softmax_j), where delta = 1 if j=yi, else delta=0.
    # and so dW = softmax_j - delta, delta as above.
    # see https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    # or just derive it yourself.
    dW += X.T@prob # fxn X nxc = fxc.
    loss += reg * (W**2).sum()
    dW /= num_train
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
