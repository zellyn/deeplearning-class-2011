# -*- coding: utf-8 -*-
# <script type="text/javascript"
#   src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
# </script>
# Functions for implementing Stacked Autoencoder + Softmax classification. See
# [stackedae_exercise.py](stackedae_exercise.html) for a running example.

from imports import *
import util
import autoencoder

# Convert a "stack" structure into a flattened parameter vector and
# also store the network configuration. This is useful when working
# with optimization toolboxes.
#
# `params, netconfig = stack2params(stack)`
#
# `stack` - the stack structure, where:
#
#  - `stack[1].w` = weights of first layer
#  - `stack[1].b` = weights of first layer
#  - `stack[2].w` = weights of second layer
#  - `stack[2].b` = weights of second layer
#  - ... etc.
def stack2params(stack):
  params = np.empty([1,0], order='F')
  netconfig = util.Empty()
  netconfig.layersizes = []
  netconfig.inputsize = 0
  if stack:
    netconfig.inputsize = stack[0].w.shape[1]
    for entry in stack:
      params = np.hstack([params, entry.w.reshape([1, -1], order='F'), entry.b.reshape([1, -1], order='F')])
      netconfig.layersizes.append(entry.w.shape[0])

  return params.ravel('F'), netconfig

# Convert a flattened parameter vector into a nice "stack" structure
# for us to work with. This is seful when building multilayer
# networks.
#
# `stack = params2stack(params, netconfig)`
#
# - `params` - flattened parameter vector
# - `netconfig` - auxiliary variable containing the configuration of
#   the network
def params2stack(params, netconfig):
  depth = len(netconfig.layersizes)
  stack = []
  prev_layersize = netconfig.inputsize
  cur_pos = 0

  for d in xrange(depth):
    entry = util.Empty()

    # Extract weights
    layersize = netconfig.layersizes[d]
    wlen = layersize * prev_layersize
    entry.w = params[cur_pos:cur_pos + wlen].reshape([layersize, prev_layersize], order='F')
    cur_pos += wlen

    # Extract bias
    blen = layersize
    entry.b = params[cur_pos:cur_pos + blen].reshape([layersize, 1], order='F')
    cur_pos += blen

    prev_layersize = layersize
    stack.append(entry)

  return stack

# Take a trained (autoencoder network and softmax) theta, and a
# training data set with labels, and return cost and gradient using a
# stacked autoencoder model. Used for finetuning.
#
# - `theta` - trained weights from the autoencoder
# - `hidden_size` - the number of hidden units *at the last layer*
# - `num_classes` -  the number of categories
# - `netconfig` - the network configuration of the stack
# - `lamb` - the weight regularization penalty
# - `data` - Our matrix containing the training data as columns.  So,
#   `data[:,i]` is the i-th training example.
# - `labels` - A vector containing labels, where `labels[i]` is the
#   label for the i-th training example
def cost(theta, input_size, hidden_size, num_classes, netconfig, lamb, data, labels):
  # We first extract the part which compute the softmax gradient
  softmax_theta = theta[:hidden_size * num_classes].reshape([num_classes, hidden_size], order='F')

  # Extract out the "stack"
  stack = params2stack(theta[hidden_size * num_classes:], netconfig)
  depth = len(stack)
  num_cases = data.shape[1]
  ground_truth = np.zeros([num_classes, num_cases])
  ground_truth[labels.ravel(), np.arange(num_cases)] = 1

  # Compute the cost function and gradient vector for the stacked
  # autoencoder.
  #
  # `stack` is a cell-array of the weights and biases for every
  # layer. In particular, the weights of layer d are `stack[d].w` and
  # the biases are `stack[d].b`.
  #
  # The last layer of the network is connected to the softmax
  # classification layer, `softmax_theta`.
  #
  # Compute the gradients for the softmax_theta, storing that in
  # `softmax_theta_grad`. Similarly, compute the gradients for each
  # layer in the stack, storing the gradients in `stack_grad[d].w` and
  # `stack_grad[d].b`.  Note that the size of the matrices in stackgrad
  # should match exactly that of the size of the matrices in stack.
  z = [0]
  a = [data]

  for layer in xrange(depth):
    z.append(stack[layer].w.dot(a[layer]) + stack[layer].b)
    a.append(autoencoder.sigmoid(z[layer+1]))

  M = softmax_theta.dot(a[depth])
  M = M - M.max(0)
  p = np.exp(M) / np.exp(M).sum(0)

  gt_vec = ground_truth.reshape([1, -1], order='F')
  p_vec = p.reshape([-1, 1], order='F')
  cost = (-1.0/num_cases * gt_vec.dot(np.log(p_vec)) + lamb/2 * (softmax_theta**2).sum())
  softmax_theta_grad = -1.0/num_cases * (ground_truth - p).dot(a[depth].T) + lamb * softmax_theta

  d = [0 for _ in xrange(depth+1)]

  d[depth] = -(softmax_theta.T.dot(ground_truth - p)) * a[depth] * (1-a[depth])

  for layer in range(depth-1, 0, -1):
    d[layer] = stack[layer].w.T.dot(d[layer+1]) * a[layer] * (1-a[layer])

  stack_grad = [util.Empty() for _ in xrange(depth)]
  for layer in range(depth-1, -1, -1):
    stack_grad[layer].w = (1.0/num_cases) * d[layer+1].dot(a[layer].T)
    stack_grad[layer].b = (1.0/num_cases) * np.sum(d[layer+1], 1)[:, np.newaxis]

  grad = np.append(softmax_theta_grad.ravel('F'), stack2params(stack_grad)[0])

  assert (grad.shape==theta.shape)
  assert grad.flags['F_CONTIGUOUS']
  return cost, grad

# Take a trained theta and a test data set, and return the predicted
# labels for each example.
#
# - `theta` - trained weights from the autoencoder
# - `visibleSize` - the number of input units
# - `hidden_size` - the number of hidden units *at the 2nd layer*
# - `num_classes` - the number of categories
# - `data` - matrix containing the training data as columns.  So,
#   `data[:,i]` is the i-th training example.
#
# returns the prediction matrix
# \\( pred_i = \\mathop{\\arg\\!\\max}\\limits_c \\,P(y^{(i)} = c | x^{(i)}) \\).
def predict(theta, input_size, hidden_size, num_classes, netconfig, data):
  # We first extract the part which compute the softmax gradient
  softmax_theta = theta[:hidden_size * num_classes].reshape([num_classes, hidden_size], order='F')

  # Extract out the "stack"
  stack = params2stack(theta[hidden_size * num_classes:], netconfig)

  depth = len(stack)
  z = [0]
  a = [data]

  for layer in xrange(depth):
    z.append(stack[layer].w.dot(a[layer]) + stack[layer].b)
    a.append(autoencoder.sigmoid(z[layer+1]))

  return softmax_theta.dot(a[depth]).argmax(0)
