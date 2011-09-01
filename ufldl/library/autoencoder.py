# -*- coding: utf-8 -*-
# <script type="text/javascript"
#   src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
# </script>

# Various autoencoder-related routines for the Deep Learning
# exercises.

from library.imports import *

# Flatten W1, W2, b1, b2 into a row vector
def flatten(W1, W2, b1, b2):
  return np.array(np.hstack([W1.ravel('F'), W2.ravel('F'), b1.ravel('F'), b2.ravel('F')]), order='F')

# Expand a row vector back into W1, W2, b1, b2
def unflatten(theta, visible_size, hidden_size):
  hv = hidden_size * visible_size
  W1 = theta[0:hv].reshape([hidden_size, visible_size], order='F')
  W2 = theta[hv:2*hv].reshape([visible_size, hidden_size], order='F')
  b1 = theta[2*hv:2*hv+hidden_size].reshape([hidden_size, 1], order='F')
  b2 = theta[2*hv+hidden_size:].reshape([visible_size, 1], order='F')
  return (W1, W2, b1, b2)

# Initialize a single-layer autoencoder
#
# - `hidden_size` - the number of hidden units
# - `visible_size` - the size of the input (and output)
def initialize_parameters(hidden_size, visible_size):
  r = np.sqrt(6) / np.sqrt(hidden_size + visible_size + 1)
  W1 = np.random.random([hidden_size, visible_size]) * 2 * r - r;
  W2 = np.random.random([visible_size, hidden_size]) * 2 * r - r;
  b1 = np.zeros([hidden_size, 1])
  b2 = np.zeros([visible_size, 1])

  return flatten(W1, W2, b1, b2)

# The sigmoid function, \\( \\frac{1}{1 + e^{-x}} \\).
def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

# Compute sparse autoencoder loss and gradient for a
# single-hidden-layer autoencoder.
#
# - `theta` - position at which to compute loss and gradient
# - `visible_size` - size of input and output layers
# - `hidden_size` - size of the hidden layer
# - `lamb` - weight decay parameter
# - `target_activation` - desired average activation of the hidden
#   units (\\(\\rho\\) in the lecture notes)
# - `beta` - weight of sparsity penalty term
# - `data` - the input data
def sparse_autoencoder_loss(theta, visible_size, hidden_size, lamb,
                            target_activation, beta, data):
  m = data.shape[1]

  W1, W2, b1, b2 = unflatten(theta, visible_size, hidden_size)

  z2 = W1.dot(data) + b1
  a2 = sigmoid(z2)
  z3 = W2.dot(a2) + b2
  a3 = sigmoid(z3)

  rhohats = np.mean(a2,1)[:, np.newaxis]
  rho = target_activation
  KLsum = np.sum(rho * np.log(rho / rhohats) + (1-rho) * np.log((1-rho) / (1-rhohats)))

  squares = (a3 - data) ** 2
  squared_err_J = 0.5 * (1.0/m) * np.sum(squares)
  weight_decay_J = (lamb/2.0) * (np.sum(W1**2) + np.sum(W2**2))
  sparsity_J = beta * KLsum
  loss = squared_err_J + weight_decay_J + sparsity_J

  delta3 = -(data - a3) * a3 * (1-a3)
  beta_term = beta * (-rho / rhohats + (1-rho) / (1-rhohats))
  delta2 = (W2.T.dot(delta3) + beta_term) * a2 * (1-a2)

  W2grad = (1.0/m) * delta3.dot(a2.T) + lamb * W2
  b2grad = (1.0/m) * np.sum(delta3, 1)[:, np.newaxis]
  W1grad = (1.0/m) * delta2.dot(data.T) + lamb * W1
  b1grad = (1.0/m) * np.sum(delta2, 1)[:, np.newaxis]

  grad = flatten(W1grad, W2grad, b1grad, b2grad)

  return (loss, grad)

# Compute the output of the hidden layer for a trained autoencoder.
#
# - `theta` - trained weights from the autoencoder
# - `visible_size` - the number of input units
# - `hidden_size` the number of hidden units
# - `data` - matrix containing the training data as columns. So,
#   `data(:,i)` is the i-th training example.
def feedforward_autoencoder(theta, hidden_size, visible_size, data):
  W1, W2, b1, b2 = unflatten(theta, visible_size, hidden_size)
  return sigmoid(W1.dot(data) + b1)
