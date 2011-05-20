#! /Users/zellyn/.ve/np/bin/python

import time
import sys

sys.path.append('..')
from library.imports import *
from library import util
from library import autoencoder

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
  util.display_network(patches[:,np.random.randint(0, num_patches, 200)])
  theta = autoencoder.initialize_parameters(hidden_size, visible_size)

  # STEP 2: Implement sparseAutoencoderLoss

  def sal(theta):
    return autoencoder.sparse_autoencoder_loss(theta, visible_size, hidden_size, lamb,
                                               target_activation, beta, patches)

  loss, grad = sal(theta)


  # STEP 3: Gradient Checking
  if testing:
    numgrad = util.compute_numerical_gradient(lambda x: sal(x)[0], theta)

    # Eyeball the gradients
    print np.hstack([numgrad, grad])

    diff = linalg.norm(numgrad-grad) / linalg.norm(numgrad+grad)
    print "Normed difference: %f" % diff

  # STEP 4: Run sparse_autoencoder_loss with L-BFGS

  # Initialize random theta
  theta = autoencoder.initialize_parameters(hidden_size, visible_size)

  print "Starting..."
  x, f, d = scipy.optimize.fmin_l_bfgs_b(sal, theta, maxfun=400, iprint=25, m=20)
  print "Done"
  print x
  print f
  print d

  W1, W2, b1, b2 = autoencoder.unflatten(x, visible_size, hidden_size)
  print "W1.shape=%s" % (W1.shape,)
  util.display_network(W1.T)

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

  theta = autoencoder.flatten(W1,W2,b1,b2)

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
