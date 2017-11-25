import numpy as np
from random import shuffle
from past.builtins import xrange

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
    
    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    
    for i in range(num_train):
        # compute vector of scores
        f = X[i].dot(W)
        
        # for stability
        f -= np.max(f)
        
        sum_i =  np.sum(np.exp(f))
        p = lambda k:np.exp(f[k])/sum_i
        
        loss += -np.log(p(y[i]))
        
    # computing gradient
        for k in range(num_classes):
            dW[:,k] += (p(k) - (y[i] == k))*X[i]
            
    loss = loss / num_train
    loss += 0.5*reg*np.sum(W*W)
    dW /= num_train
    dW += reg*W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    
    f_X = X.dot(W)
    #print(f_X.shape)
    f_X -= np.max(f_X,axis=1,keepdims=True)
    sum_X = np.sum(np.exp(f_X),axis = 1,keepdims=True)
    
    p = np.exp(f_X) / sum_X
    loss = np.sum(-np.log(p[np.arange(num_train),y])) ##### DOUBT ##### WHY NOT np.arange(num_train).reshape(5,1) worked correct ######
    loss = loss / num_train
    
    ind = np.zeros_like(p)
    ind[np.arange(num_train),y] = 1
    
    dW = X.T.dot(p-ind)
    #f_y = np.exp(f_X[y])
    #sum_X = sum_X.reshape(num_train,1)
    #f_y = f_y.reshape(num_train,1)
   
    
    #loss = np.sum(-np.log(f_y/sum_X))
    
    loss += 0.5 * reg * np.sum(W*W)
    dW /= num_train
    dW += reg*W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

