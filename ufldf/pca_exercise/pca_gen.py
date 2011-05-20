#! /Users/zellyn/.ve/np/bin/python

import sys
sys.path.append('..')
from library.imports import *
from library import util

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

# STEP 0a: Load data
x = sample_IMAGES_RAW()
m, n = x.shape
randsel = np.random.randint(0, n, 200)
# util.display_network(x[:, randsel], 'Raw data')

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

util.display_network(x[:, randsel], 'Raw images', show=False)
util.display_network(xHat[:, randsel], 'PCA processed images')

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

util.display_network(x[:, randsel], 'Raw images', show=False)
util.display_network(xZCAWhite[:, randsel], 'ZCA whitened images')
