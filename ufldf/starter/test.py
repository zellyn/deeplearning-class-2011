#! /Users/zellyn/.ve/np/bin/python

from train import *

patchsize = 8
visible_size = patchsize * patchsize
hidden_size = 25
target_activation = 0.01
lamb = 0.0001
beta = 3

theta = sio.loadmat('testdata/save_theta.mat')['theta']
patches = sio.loadmat('testdata/save_patches.mat')['patches']

def sal(theta):
  return sparse_autoencoder_loss(theta, visible_size, hidden_size, lamb, target_activation, beta, patches)
opttheta, f, d = scipy.optimize.fmin_l_bfgs_b(sal, theta, maxfun=3000, iprint=1, m=theta.size)

sio.savemat('testdata/numpy-opttheta.mat', {'opttheta' : opttheta})
