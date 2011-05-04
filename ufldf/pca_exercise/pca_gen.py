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

def sample_IMAGES_RAW():
  IMAGES = sio.loadmat('IMAGES_RAW.mat')['IMAGESr']
  patch_size = 12
  num_patches = 10000
  ydim, xdim, num_images = IMAGES.shape
  patches = np.zeros([patch_size * patch_size, num_patches])
  p = 0
  for im in range(0, num_images):
    num_samples = num_patches / num_images
    for s in range(0, num_samples):
      y = np.random.randint(ydim - patch_size + 1)
      x = np.random.randint(xdim - patch_size + 1)
      sample = IMAGES[y:y+patch_size, x:x+patch_size, im]
      patches[:,p] = sample.reshape([patch_size * patch_size])
      p = p + 1
  return patches

def display_network(arr, title, show=True):
  arr = arr - np.mean(arr)
  L, M = arr.shape
  sz = np.sqrt(L)
  buf = 1

  # Figure out pleasant grid dimensions
  if M == np.floor(np.sqrt(M))**2:
    n = m = np.sqrt(M)
  else:
    n = np.ceil(np.sqrt(M))
    while (M%n) and n < 1.2*np.sqrt(M):
      n += 1
    m = np.ceil(M/n)

  array = np.zeros([buf+m*(sz+buf), buf+n*(sz+buf)])

  k = 0
  for i in range(0, int(m)):
    for j in range(0, int(n)):
      if k>=M:
        continue
      cmax = np.max(arr[:,k])
      cmin = np.min(arr[:,k])
      r = buf+i*(sz+buf)
      c = buf+j*(sz+buf)
      array[r:r+sz, c:c+sz] = (arr[:,k].reshape([sz,sz], order='F') - cmin) / (cmax-cmin)
      k = k + 1
  plt.figure()
  plt.title(title)
  plt.imshow(array, interpolation='nearest', cmap=plt.cm.gray)
  if show:
    plt.show()

# STEP 0a: Load data
x = sample_IMAGES_RAW()
m, n = x.shape
randsel = np.random.randint(0, n, 200)
# display_network(x[:, randsel], 'Raw data')

# STEP 0b: Zero-mean the data (by column)
avg = np.mean(x, 0)[np.newaxis, :]
x = x - avg

# STEP 1a: Implement PCA to obtain xRot

sigma = x.dot(x.T) / n
U, S, Vh = linalg.svd(sigma)
xRot = U.T.dot(x)

# STEP 1b: Check your implementation of PCA
covar = xRot.dot(xRot.T) / n

plt.figure()
plt.imshow(covar)
plt.title('Visualization of covariance matrix')
plt.show()

# STEP 2: Find k, the number of components to retain

SD = S
k = SD[(np.cumsum(SD) / np.sum(SD)) < 0.99].size

# STEP 3: Implement PCA with dimension reduction
xRot = U[:,0:k].T.dot(x)
xHat = U[:,0:k].dot(xRot)

display_network(x[:, randsel], 'Raw images', show=False)
display_network(xHat[:, randsel], 'PCA processed images')

# STEP 4a: Implement PCA with whitening and regularisation

epsilon = 0.1
xPCAWhite = np.diag(1.0 / np.sqrt(S + epsilon)).dot(U.T).dot(x)

# STEP 4b: Check your implementation of PCA whitening

covar = xPCAWhite.dot(xPCAWhite.T) / n

plt.figure()
plt.imshow(covar)
plt.title('Visualization of covariance matrix')
plt.show()

# STEP 5: Implement ZCA whitening

xZCAWhite = U.dot(xPCAWhite)

display_network(x[:, randsel], 'Raw images', show=False)
display_network(xZCAWhite[:, randsel], 'ZCA whitened images')
