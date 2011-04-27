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

def flatten(W1, b1, W2, b2):
  return np.hstack([W1.ravel(), W2.ravel(), b1.ravel(), b2.ravel()])[:,np.newaxis]

def initialize_parameters(hidden_size, visible_size):
  r = np.sqrt(6) / np.sqrt(hidden_size + visible_size + 1)
  W1 = np.random.random([hidden_size, visible_size]) * 2 * r - r;
  W2 = np.random.random([visible_size, hidden_size]) * 2 * r - r;
  b1 = np.zeros([hidden_size, 1])
  b2 = np.zeros([visible_size, 1])

  return flatten(W1, b1, W2, b2)

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

if __name__ == '__main__':
  main(testing=True)
