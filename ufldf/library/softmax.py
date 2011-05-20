# -*- coding: utf-8 -*-
# <script type="text/javascript"
#   src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
# </script>
# Functions for implementing Softmax Regression. See
# [softmax_exercise.py](softmax_exercise.html) for a running example.

from imports import *

# Compute the cost function \\(J(\\theta)\\) and gradient \\( \\nabla_\\theta J(\\theta) \\).
#
# - `theta` - flattened (numClasses × inputSize) parameters
# - `num_classes` - the number of classes
# - `input_size` - the size N of the input vector
# - `lamb` - weight decay parameter
# - `data` - the N × M input matrix, where each column data(:, i) corresponds to
#   a single test set
# - `labels` - an M × 1 matrix containing the labels corresponding for the input data
def cost(theta, num_classes, input_size, lamb, data, labels):
  theta = theta.reshape([num_classes, input_size], order='F')
  num_cases = data.shape[1]
  ground_truth = np.zeros([num_classes, num_cases])
  ground_truth[labels.ravel(), np.arange(num_cases)] = 1

  M = theta.dot(data)
  M = M - M.max(0)

  p = np.exp(M) / np.exp(M).sum(0)

  # <p>$$
  # J(\theta) = -\frac{1}{m} \left[
  #   \sum_{i=1}^m \sum_{j=1}^k 1\{y^{(i)} = j\} \log \; p(y^{(i)} = j | x^{(i)} ; \theta)
  # \right]
  # + \frac{\lambda}{2} \sum_{i=1}^k \sum_{j=0}^n \theta_{ij}^2
  # $$</p>
  gt_vec = ground_truth.reshape([1, -1], order='F')
  p_vec = p.reshape([-1, 1], order='F')
  cost = (-1.0/num_cases * gt_vec.dot(np.log(p_vec)) + lamb/2 * (theta**2).sum())

  # <p>$$
  # \nabla_{\theta_j} J(\theta) =
  # - \frac{1}{m} \sum_{i=1}^{m}{ \left[
  #   x^{(i)} ( 1\{ y^{(i)} = j\}  - p(y^{(i)} = j | x^{(i)}; \theta) )
  # \right] }
  # + \lambda \theta_j
  # $$</p>
  # …but we vectorize by transposing \\(x\\).
  theta_grad = -1.0/num_cases * (ground_truth - p).dot(data.T) + lamb * theta

  # Unroll into a vector
  grad = theta_grad.ravel('F')
  return cost[0][0], grad

# Use `cost()` to train a network using l-bfgs.
#
# - `input_size` - the size N of the input vector
# - `num_classes` - the number of classes
# - `lamb` - weight decay parameter
# - `input_data` - the N × M input matrix, where each column data(:, i) corresponds to
#   a single test set
# - `labels` - an M × 1 matrix containing the labels corresponding for the input data
# - `maxfun` - maximum number of iterations for l-bfgs
def train(input_size, num_classes, lamb, input_data, labels, maxfun=400):
  theta = 0.005 * np.random.randn(num_classes * input_size, 1)
  theta = np.asfortranarray(theta)

  fn = lambda theta: cost(theta, num_classes, input_size, lamb, input_data, labels)

  softmax_opt_theta, f, d = scipy.optimize.fmin_l_bfgs_b(fn, theta, maxfun=maxfun, iprint=25, m=20)

  return dict(opt_theta = softmax_opt_theta.reshape([num_classes, input_size], order='F'),
              input_size = input_size,
              num_classes = num_classes)

# Given a trained model and data, return the prediction matrix
# \\( pred_i = \\mathop{\\arg\\!\\max}\\limits_c \\,P(y^{(i)} = c | x^{(i)}) \\):
#
# - `model` - model trained using train()
# - `input_data` - the N × M input matrix, where each column data(:, i)
#   corresponds to a single test set
def predict(model, input_data):
  theta = model['opt_theta']
  return (theta.dot(input_data)).argmax(0)
