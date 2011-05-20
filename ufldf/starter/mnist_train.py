#! /Users/zellyn/.ve/np/bin/python

import struct
import sys
sys.path.append('..')
from library.imports import *
from library import autoencoder
from library import util
from library import mnist

def main(testing=True):
  images = mnist.load_images('../data/train-images-idx3-ubyte')  # 784 x 60000
  labels = mnist.load_labels('../data/train-labels-idx1-ubyte')  # 60000 x 1
  util.display_network(images[:,0:100])  # Show the first 100 images

  visible_size = 28*28
  hidden_size = 196
  sparsity_param = 0.1
  lamb = 3e-3
  beta = 3
  patches = images[:,0:10000]
  theta = autoencoder.initialize_parameters(hidden_size, visible_size)
  def sal(theta):
    return autoencoder.sparse_autoencoder_loss(theta, visible_size, hidden_size, lamb,
                                               sparsity_param, beta, patches)
  x, f, d = scipy.optimize.fmin_l_bfgs_b(sal, theta, maxfun=400, iprint=1, m=20)
  W1, W2, b1, b2 = autoencoder.unflatten(x, visible_size, hidden_size)
  util.display_network(W1.T)

if __name__ == '__main__':
  main(testing=False)
