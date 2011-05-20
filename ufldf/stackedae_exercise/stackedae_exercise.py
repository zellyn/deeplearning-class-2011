#! /Users/zellyn/.ve/np/bin/python
# -*- coding: utf-8 -*-
# <script type="text/javascript"
#   src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
# </script>

# My implementation of the [Stacked autoencoder deep network exercise](http://ufldl.stanford.edu/wiki/index.php/Exercise:_Implement_deep_networks_for_digit_classification)
# for the Deep Learning class.

DISPLAY = True

import time
import sys
sys.path.append('..')
from library.imports import *
from library import autoencoder
from library import mnist
from library import softmax
from library import util

# === Step 0: Initialize constants and parameters ===
#
# - `hidden_size_l1` - layer 1 hidden size
# - `hidden_size_l2` - layer 2 hidden size
# - `sparsity_param` - desired average activation of the hidden units
#   (\\(\\rho\\) in the lecture notes)
# - `lamb` - weight decay parameter
# - `beta` - weight of sparsity penalty term
input_size = 28 * 28
num_classes = 10
hidden_size_l1 = 200
hidden_size_l2 = 200
sparsity_param = 0.1
lamb = 3e-3
beta = 3

# === Step 1: Load data from the MNIST database ===
train_data = mnist.load_images('../data/train-images-idx3-ubyte')
train_labels = mnist.load_labels('../data/train-labels-idx1-ubyte')

# === Step 2: Train the first sparse autoencoder ===
#
# Train the first sparse autoencoder on the unlabelled STL training
# images.

# Randomly initialize the parameters
sae1_theta = autoencoder.initialize_parameters(hidden_size_l1, input_size)

# Train the first layer sparse autoencoder. This layer has a hidden
# size of `hidden_size_l1`.

# sae1_opt_theta, loss = minFunc( @(p) sparseAutoencoderLoss(p, ...
#     inputSize, hiddenSizeL1, ...
#     lambda, sparsityParam, ...
#     beta, trainData), ...
#     sae1Theta, options);

fn = lambda theta: autoencoder.sparse_autoencoder_loss(
  theta, input_size, hidden_size_l1, lamb, sparsity_param, beta, train_data)
sae1_opt_theta, loss, d = (
  scipy.optimize.fmin_l_bfgs_b(fn, sae1_theta, maxfun=400, iprint=1, m=20))

if DISPLAY:
  W1, W2, b1, b2 = autoencoder.unflatten(sae1_opt_theta, input_size, hidden_size_l1)
  util.display_network(W1.T)

# === Step 3: Train the second sparse autoencoder ===
#
# Train the second sparse autoencoder on the first autoencoder features.
sae1_features = autoencoder.feedforward_autoencoder(sae1_opt_theta, hidden_size_l1, input_size, train_data)

# Randomly initialize the parameters
sae2_theta = autoencoder.initialize_parameters(hidden_size_l2, hidden_size_l1)

fn = lambda theta: autoencoder.sparse_autoencoder_loss(
  theta, hidden_size_l1, hidden_size_l2, lamb, sparsity_param, beta, sae1_features)
sae2_opt_theta, loss, d = (
  scipy.optimize.fmin_l_bfgs_b(fn, sae2_theta, maxfun=400, iprint=1, m=20))

if DISPLAY:
  W11, W21, b11, b21 = autoencoder.unflatten(sae1_opt_theta, input_size, hidden_size_l1)
  W12, W22, b12, b22 = autoencoder.unflatten(sae2_opt_theta, hidden_size_l1, hidden_size_l2)
  # TODO(zellyn): figure out how to display a 2-level network
  # display_network(log(W11' ./ (1-W11')) * W12');

# === Step 4: Train the softmax classifier ===
# Train the sparse autoencoder on the second autoencoder features.

sae2_features = autoencoder.feedforward_autoencoder(sae2_opt_theta, hidden_size_l2, hidden_size_l1, sae1_features)
sae_Softmax_theta = 0.005 * np.random.randn(hidden_size_l2 * num_Classes, 1)

# Train the softmax classifier. The classifier takes in input of
# dimension `hidden_size_l2` corresponding to the hidden layer size of
# the 2nd layer.
softmax_model = softmax.train(hidden_size_l2, num_classes, 1e-4, sae2_features, train_labels)
sae_softmax_opt_theta = softmax_model['opt_theta'].ravel(order='F')

