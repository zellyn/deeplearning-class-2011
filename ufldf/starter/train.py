#! /Users/zellyn/.ve/np/bin/python

import time
import sys
import numpy as np
import scipy as sp
import matplotlib as mpl
if sys.platform == 'darwin': mpl.use('TkAgg')

import matplotlib.pyplot as plt
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import scipy.optimize

def sampleIMAGES(patchsize, num_patches):
  IMAGES = sio.loadmat('IMAGES')['IMAGES']
  patches = np.zeros([patchsize * patchsize, num_patches])
  [ydim, xdim, num_images] = IMAGES.shape

  for i in range(num_patches):
    img = np.random.randint(num_images)
    y_start = np.random.randint(ydim - patchsize + 1)
    x_start = np.random.randint(xdim - patchsize + 1)
    patch = IMAGES[y_start:y_start+patchsize, x_start:x_start+patchsize, img]
    patches[:,i] = patch.reshape([patchsize * patchsize])

  return patches

def display_network(arr):
  # TODO(zellyn): implement
  print "display_network: arr.shape=%s" % (arr.shape,)

def flatten(W1, W2, b1, b2):
  return np.array(np.hstack([W1.ravel(), W2.ravel(), b1.ravel(), b2.ravel()]), order='F')

def unflatten(theta, visible_size, hidden_size):
  hv = hidden_size * visible_size
  W1 = theta[0:hv].reshape([hidden_size, visible_size])
  W2 = theta[hv:2*hv].reshape([visible_size, hidden_size])
  b1 = theta[2*hv:2*hv+hidden_size].reshape([hidden_size, 1])
  b2 = theta[2*hv+hidden_size:].reshape([visible_size, 1])
  return (W1, W2, b1, b2)

def initialize_parameters(hidden_size, visible_size):
  r = np.sqrt(6) / np.sqrt(hidden_size + visible_size + 1)
  W1 = np.random.random([hidden_size, visible_size]) * 2 * r - r;
  W2 = np.random.random([visible_size, hidden_size]) * 2 * r - r;
  b1 = np.zeros([hidden_size, 1])
  b2 = np.zeros([visible_size, 1])

  return flatten(W1, W2, b1, b2)

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

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

def compute_numerical_gradient(fn, theta):
  epsilon = 1e-4
  numgrad = np.zeros(theta.shape)

  for i in range(theta.size):
    theta_minus = theta.copy()
    theta_plus = theta.copy()
    theta_minus[i] = theta_minus[i] - epsilon
    theta_plus[i] = theta_plus[i] + epsilon
    numgrad[i] = (fn(theta_plus) - fn(theta_minus)) / (2 * epsilon)

  return numgrad

def main(testing=False):
  # STEP 0: Parameters

  patchsize = 8
  num_patches = testing and 10 or 10000
  visible_size = patchsize * patchsize
  hidden_size = testing and 3 or 25
  target_activation = 0.01
  lamb = 0.0001
  beta = 3

  # STEP 1: Get data

  patches = sampleIMAGES(patchsize, num_patches)
  display_network(patches[:,np.random.randint(0, num_patches, 200)])
  theta = initialize_parameters(hidden_size, visible_size)

  # STEP 2: Implement sparseAutoencoderLoss

  def sal(theta):
    return sparse_autoencoder_loss(theta, visible_size, hidden_size, lamb,
                                   target_activation, beta, patches)

  loss, grad = sal(theta)


  # # STEP 3: Gradient Checking
  # numgrad = compute_numerical_gradient(lambda x: sal(x)[0], theta)
  #
  # # Eyeball the gradients
  # print np.hstack([numgrad, grad])
  #
  # diff = linalg.norm(numgrad-grad) / linalg.norm(numgrad+grad)
  # print "Normed difference: %f" % diff

  # STEP 4: Run sparse_autoencoder_loss with L-BFGS

  # Initialize random theta
  theta = initialize_parameters(hidden_size, visible_size)

  print "Starting..."
  x, f, d = scipy.optimize.fmin_l_bfgs_b(sal, theta, maxfun=400, iprint=1)
  print "Done"
  print x
  print f
  print d

if __name__ == '__main__':
  main(testing=False)
