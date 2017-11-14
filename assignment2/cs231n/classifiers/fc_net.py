import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    b1 = np.zeros((hidden_dim,))
    b2 = np.zeros((num_classes,))
    w1 = weight_scale * np.random.randn(input_dim,hidden_dim)
    w2 = weight_scale * np.random.randn(hidden_dim,num_classes)
    self.params['W1'] = w1
    self.params['W2'] = w2
    self.params['b1'] = b1
    self.params['b2'] = b2
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    w1,w2,b1,b2 = self.params['W1'],self.params['W2'],self.params['b1'],self.params['b2']
    fc_relu1,cache1 = affine_relu_forward(X,w1,b1)
    scores,cache2 = affine_forward(fc_relu1,w2,b2)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    reg = self.reg
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    data_loss,dscores = softmax_loss(scores,y)
    reg_loss = 0.5 * reg * (np.sum(w1*w1) + np.sum(w2*w2))
    loss = data_loss + reg_loss

    grads = {}
    dx1,dw2,db2 = affine_backward(dscores,cache2)
    dx,dw1,db1 = affine_relu_backward(dx1,cache1)
    dw1 += reg * w1
    dw2 += reg * w2
    grads['W1'] = dw1
    grads['W2'] = dw2
    grads['b1'] = db1
    grads['b2'] = db2
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    b = np.zeros((hidden_dims[0],))
    w = weight_scale * np.random.randn(input_dim,hidden_dims[0])
    self.params['W1'] = w
    self.params['b1'] = b

    if self.use_batchnorm:
      gamma = np.ones((hidden_dims[0],))
      beta = np.zeros((hidden_dims[0],))
      self.params['gamma1'] = gamma
      self.params['beta1'] = beta
    
    #print w.shape,b.shape
    for i in range(1,len(hidden_dims)):
      b = np.zeros((hidden_dims[i],))
      w = weight_scale * np.random.randn(hidden_dims[i-1],hidden_dims[i])
      param1 = 'W'+chr(i+49)
      param2 = 'b'+chr(i+49)

      #adding batch normalization
      if self.use_batchnorm:
        gamma = np.ones((hidden_dims[i],))
        beta = np.zeros((hidden_dims[i],))
        param3 = 'gamma' + chr(i+49)
        param4 = 'beta' + chr(i+49)
        self.params[param3] = gamma
        self.params[param4] = beta
      
      self.params[param1] = w
      self.params[param2] = b
      #print w.shape,b.shape
    b = np.zeros((num_classes,))
    w = weight_scale * np.random.randn(hidden_dims[-1],num_classes)
    param1 = 'W'+chr(self.num_layers+48)
    param2 = 'b'+chr(self.num_layers+48)
    self.params[param1] = w
    self.params[param2] = b
    #print w.shape,b.shape
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    hidden = {}
    hidden['h0'] = X.reshape(X.shape[0], np.prod(X.shape[1:]))

    if self.use_dropout:
      hdrop, cache_hdrop = dropout_forward(hidden['h0'], self.dropout_param)
      hidden['hdrop0'], hidden['cache_hdrop0'] = hdrop, cache_hdrop
            
    for i in range(self.num_layers):
      
      w = self.params['W' + chr(i+49)]
      b = self.params['b' + chr(i+49)]
      h = hidden['h' + chr(i+48)]
      if self.use_dropout:
        h = hidden['hdrop' + chr(i+48)]
        
      #print h.shape,w.shape,b.shape
        
      if (i == self.num_layers - 1):
        h,cache_h  = affine_forward(h,w,b)
        
      else:
        if self.use_batchnorm and i != self.num_layers - 1:
          gamma = self.params['gamma' + chr(i+49)]
          beta = self.params['beta' + chr(i+49)]
          h, cache_h = affine_norm_relu_forward(h,w,b,gamma,beta,self.bn_params[i])
          
        else:
          h,cache_h = affine_relu_forward(h,w,b)
          
        if self.use_dropout:
          hdrop,cache_hdrop = dropout_forward(h,self.dropout_param)
          hidden['hdrop' + chr(i+49)] = hdrop
          hidden['cache_hdrop' + chr(i+49)] = cache_hdrop
          
      hidden['h' + chr(i+49)] = h
      hidden['cache_h' + chr(i+49)] = cache_h
    scores = hidden['h' + chr(self.num_layers+48)]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    data_loss,dscores = softmax_loss(scores,y)
    reg_wsum = 0
    for i in range(self.num_layers):
      reg_wsum += np.sum(self.params['W' + chr(i+49)] ** 2)
    reg_loss = 0.5 * self.reg * reg_wsum
    loss = data_loss + reg_loss

    grads = {}
    hidden['dh' + chr(self.num_layers + 48)] = dscores
    for i in range(self.num_layers-1,-1,-1):
      dh = hidden['dh' + chr(i+49)]
      cache_h = hidden['cache_h' + chr(i+49)]
      if (i == self.num_layers - 1):
        dh,dw,db = affine_backward(dh,cache_h)
      else:
        if self.use_dropout:
            dh = dropout_backward(dh,hidden['cache_hdrop' + chr(i+49)])
        if self.use_batchnorm:
          dh, dw, db, dgamma, dbeta = affine_norm_relu_backward(dh,cache_h)
          hidden['dgamma' + chr(i+49)] = dgamma
          hidden['dbeta' + chr(i+49)] = dbeta
          grads['gamma' + chr(i+49)] = dgamma
          grads['beta' + chr(i+49)] = dbeta
        else:
          dh,dw,db = affine_relu_backward(dh,cache_h)
      hidden['dh' + chr(i+48)] = dh
      hidden['dw' + chr(i+49)] = dw
      hidden['db' + chr(i+49)] = db

      reg_wsum = self.reg * self.params['W' + chr(i+49)]
      reg_bsum = self.reg * self.params['b' + chr(i+49)]
      
      grads['W' + chr(i+49)] = dw + reg_wsum
      grads['b' + chr(i+49)] = db + reg_bsum
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

def affine_norm_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that perorms an affine transform followed by a ReLU
    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma, beta : Weight for the batch norm regularization
    - bn_params : Contain variable use to batch norml, running_mean and var
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """

    h, h_cache = affine_forward(x, w, b)
    hnorm, hnorm_cache = batchnorm_forward(h, gamma, beta, bn_param)
    hnormrelu, relu_cache = relu_forward(hnorm)
    cache = (h_cache, hnorm_cache, relu_cache)

    return hnormrelu, cache


def affine_norm_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    h_cache, hnorm_cache, relu_cache = cache

    dhnormrelu = relu_backward(dout, relu_cache)
    dhnorm, dgamma, dbeta = batchnorm_backward(dhnormrelu, hnorm_cache)
    dx, dw, db = affine_backward(dhnorm, h_cache)

    return dx, dw, db, dgamma, dbeta
