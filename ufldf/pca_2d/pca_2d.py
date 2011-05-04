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

# STEP 0: Load data

x = np.loadtxt('pcaData.txt')
plt.figure()
plt.plot(x[0,:], x[1,:], 'o', mec='blue', mfc='none')
plt.title('Raw data')
plt.show()

# STEP 1a: Implement PCA to obtain the rotation matrix, U, which is
# the eigenbases sigma.

sigma = x.dot(x.T) / x.shape[1]
U, S, Vh = linalg.svd(sigma)

plt.figure()
plt.plot([0, U[0,0]], [0, U[1,0]])
plt.plot([0, U[0,1]], [0, U[1,1]])
plt.plot(x[0,:], x[1,:], 'o', mec='blue', mfc='none')
plt.show()

# STEP 1b: Compute xRot, the projection on to the eigenbasis

xRot = U.T.dot(x)

plt.figure()
plt.plot(xRot[0,:], xRot[1,:], 'o', mec='blue', mfc='none')
plt.title('xRot')
plt.show()

# STEP 2: Reduce the number of dimensions from 2 to 1

k = 1
xRot = U[:,0:k].T.dot(x)
xHat = U[:,0:k].dot(xRot)

plt.figure()
plt.plot(xHat[0,:], xHat[1,:], 'o', mec='blue', mfc='none')
plt.title('xHat')
plt.show()

# STEP 3: PCA Whitening

epsilon = 1e-5
xPCAWhite = np.diag(1.0 / np.sqrt(S + epsilon)).dot(U.T).dot(x)

plt.figure()
plt.plot(xPCAWhite[0,:], xPCAWhite[1,:], 'o', mec='blue', mfc='none')
plt.title('xPCAWhite')
plt.show()

# STEP 4: ZCA Whitening

xZCAWhite = U.dot(xPCAWhite)

plt.figure()
plt.plot(xZCAWhite[0,:], xZCAWhite[1,:], 'o', mec='blue', mfc='none')
plt.title('xZCAWhite')
plt.show()
