from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                # add gradient of example to weights of w_y and w_j
                dW[:,y[i]] -= X[i]
                dW[:,j] += X[i]
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    # don't forget to average out the gradient!
    dW /= num_train
    # also add gradient of regularization factor.
    dW += reg*2*W
    
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # calculated gradient with the loss, so no code here.
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    # get all scores of all exapmles
    scores = X@W # num_train x C
    correct_class_score = scores[range(scores.shape[0]),y,np.newaxis] # take from each row it's corresponding label score.
    # also, make 2d using newaxis to stay a column vector for subtraction.
    # remember that loss = max(0,s_j - s_yi + 1)
    scores -= correct_class_score
    scores += 1
    scores[scores<0] = 0
    scores[range(scores.shape[0]),y] = 0 # loss of for class y_i is defined to be 0.
    loss = scores.sum() # - num_train # since loss of correct class is defined as 0 while we gave it 1, we subtract by num_train.
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # only derivative of non-zero scores are relevant.
    mask = np.zeros_like(scores)
    mask[scores>0] = 1
    # for every non_zero score, we need to subtract that example from its y_i class column weights.
    # reminder: df/dW = sum(1(s_j - s_yi + 1 > 0)x_i for s_j's)
    mask[range(scores.shape[0]),y] -= mask.sum(axis=1)
    # dW.shape = feat X C
    # sum feature values of examples where score wasn't 0.
    dW = X.T@mask
    dW /= num_train
    dW += reg * 2 * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
 # old version, before the idea of summing the amount of non_zero scores... also works, way more complex and slower.
    # and finally, df/dW = -sum(1(s_j-s_yi+1>0)x_i) for s_yi's, and all j!=yi, PER EXAMPLE i.
#     correct_class_multiplier = mask.sum(axis=1)[:,np.newaxis]
#     correct_class_derivative = (X * correct_class_multiplier)
    # now correct_class_derivative has in each row (row is the features of an example)
    # the amount to subtract from that example's correct_class dW column.
#     to_sub = np.zeros((10,dW.shape[0]))
    # sum the derivatives that are supposed to be sub'ed from each class, then sub it.
#     for c in range(10):
#         to_sub[c] = correct_class_derivative[y==c].sum(axis=0)
#     dW -= to_sub.T