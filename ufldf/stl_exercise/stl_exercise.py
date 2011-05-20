#! /Users/zellyn/.ve/np/bin/python
# -*- coding: utf-8 -*-

# <script type="text/javascript"
#   src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
# </script>

# My implementation of the [Self-taught learning
# exercise](http://ufldl.stanford.edu/wiki/index.php/Exercise:Self-Taught_Learning)
# for the Deep Learning class.

import sys
sys.path.append('..')
from library.imports import *
from library import mnist
from library import softmax
from library import util
from library import autoencoder

# === Step 0: Initialize constants and parameters ===
#
# - `sparsity_param` - desired average activation of the hidden units
#   (\\(\\rho\\) in the lecture notes)
# - `lamb` - weight decay parameter
# - `beta` - weight of sparsity penalty term
input_size = 28 * 28
num_labels = 5
hidden_size = 200
sparsity_param = 0.1
lamb = 3e-3
beta = 3
maxfun = 400

# === Step 1: Load data ===
#

print 'Loading raw MNIST data...'
mnist_data = mnist.load_images('../data/train-images-idx3-ubyte')
mnist_labels = mnist.load_labels('../data/train-labels-idx1-ubyte')

# Simulate a Labeled and Unlabeled set

print 'Splitting MNIST data...'
labeled_set = mnist_labels <= 4
unlabeled_set = mnist_labels >= 5

unlabeled_data = mnist_data[:, unlabeled_set]
labeled_data = mnist_data[:, labeled_set]
labels = mnist_labels[labeled_set]

num_train = labels.size // 2

train_data = labeled_data[:, :num_train]
train_labels = labels[:num_train]

test_data = labeled_data[:, num_train:]
test_labels = labels[num_train:]

# Output some statistics
print '# examples in unlabeled set: %d' % unlabeled_data.shape[1]
print '# examples in supervised training set: %d' % train_data.shape[1]
print '# examples in supervised testing set: %d' % test_data.shape[1]

# === Step 2: Train the sparse autoencoder ===
#
# This trains the sparse autoencoder on the unlabeled training images.

# Randomly initialize the parameters
theta = autoencoder.initialize_parameters(hidden_size, input_size)

# The single-parameter function to minimize
fn = lambda theta: autoencoder.sparse_autoencoder_loss(
  theta, input_size, hidden_size,lamb, sparsity_param, beta, unlabeled_data)
# Find `opt_theta` by running the sparse autoencoder on unlabeled
# training images.
opt_theta, loss, d = (
  scipy.optimize.fmin_l_bfgs_b(fn, theta, maxfun=maxfun, iprint=1, m=20))

# Visualize weights
W1, W2, b1, b2 = autoencoder.unflatten(opt_theta, input_size, hidden_size)
util.display_network(W1.T)

# === Step 3: Extract Features from the Supervised Dataset ===
train_features = autoencoder.feedforward_autoencoder(
  opt_theta, hidden_size, input_size, train_data)
test_features = autoencoder.feedforward_autoencoder(
  opt_theta, hidden_size, input_size, test_data)

# === Step 4: Train the softmax classifier ===
lamb = 1e-4
num_classes = len(set(train_labels))
softmax_model = softmax.train(hidden_size, num_classes, lamb,
                              train_features, train_labels, maxfun=100)

# === Step 5: Testing ===
#
# Compute Predictions on the test set (testFeatures) using
# `softmax.predict`.
pred = softmax.predict(softmax_model, test_features)
acc = (test_labels == pred).mean()
print 'Accuracy: %0.3f' % (acc * 100)
