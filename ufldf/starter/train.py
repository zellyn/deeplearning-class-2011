#! /Users/zellyn/.ve/np/bin/python

import time
import sys

sys.path.append('..')
from library.imports import *
from library import util

def normalize_data(data):
  data = data - np.mean(data)
  pstd = 3 * np.std(data)
  data = np.fmax(np.fmin(data, pstd), -pstd) / pstd
  data = (data + 1) * 0.4 + 0.1;
  return data

def sampleIMAGES(patchsize, num_patches):
  IMAGES = sio.loadmat('../data/IMAGES')['IMAGES']
  patches = np.zeros([patchsize * patchsize, num_patches])
  [ydim, xdim, num_images] = IMAGES.shape

  for i in range(num_patches):
    img = np.random.randint(num_images)
    y_start = np.random.randint(ydim - patchsize + 1)
    x_start = np.random.randint(xdim - patchsize + 1)
    patch = IMAGES[y_start:y_start+patchsize, x_start:x_start+patchsize, img]
    patches[:,i] = patch.reshape([patchsize * patchsize])

  return normalize_data(patches)

def sparse_autoencoder_loss(theta, visible_size, hidden_size, lamb,
                            target_activation, beta, data):
  m = data.shape[1]

  W1, W2, b1, b2 = util.unflatten(theta, visible_size, hidden_size)

  z2 = W1.dot(data) + b1
  a2 = util.sigmoid(z2)
  z3 = W2.dot(a2) + b2
  a3 = util.sigmoid(z3)

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

  grad = util.flatten(W1grad, W2grad, b1grad, b2grad)

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
  theta = util.initialize_parameters(hidden_size, visible_size)

  # STEP 2: Implement sparseAutoencoderLoss

  def sal(theta):
    return sparse_autoencoder_loss(theta, visible_size, hidden_size, lamb,
                                   target_activation, beta, patches)

  loss, grad = sal(theta)


  # STEP 3: Gradient Checking
  if testing:
    numgrad = compute_numerical_gradient(lambda x: sal(x)[0], theta)

    # Eyeball the gradients
    print np.hstack([numgrad, grad])

    diff = linalg.norm(numgrad-grad) / linalg.norm(numgrad+grad)
    print "Normed difference: %f" % diff

  # STEP 4: Run sparse_autoencoder_loss with L-BFGS

  # Initialize random theta
  theta = util.initialize_parameters(hidden_size, visible_size)

  print "Starting..."
  x, f, d = scipy.optimize.fmin_l_bfgs_b(sal, theta, maxfun=400, iprint=25, m=20)
  print "Done"
  print x
  print f
  print d

  W1, W2, b1, b2 = util.unflatten(x, visible_size, hidden_size)
  print "W1.shape=%s" % (W1.shape,)
  display_network(W1.T)

def test_loss():
  visible_size = 4
  hidden_size = 2
  rho = 0.01
  lamb = 0.0001
  beta = 3

  W1 = np.array([
      [9, 3, 6, -1],
      [7, 5, 3, 2]])

  W2 = np.array([
      [0.2, -0.9],
      [-0.3, 1.1],
      [0.5, 0.6],
      [0.7, -0.4]])

  b1 = np.array([
      [-0.7],
      [-0.5]])

  b2 = np.array([
      [0.2],
      [1.3],
      [-0.7],
      [0.6]])

  theta = util.flatten(W1,W2,b1,b2)

  data = np.array([
      [0.2, -0.7,  0.8, -0.1, -0.8],
      [0.3,  0.4, -0.7,  0.2, -0.9],
      [0.1, -0.3, -0.6,  1.0,  0.7],
      [0.8,  0.5, -0.7, -0.9, -0.7],
  ])

  print sparse_autoencoder_loss(theta, visible_size, hidden_size, lamb, rho, beta, data)

if __name__ == '__main__':
  # test_loss()
  main(testing=False)
