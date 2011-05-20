#! /Users/zellyn/.ve/np/bin/python
# -*- coding: utf-8 -*-
# <script type="text/javascript"
#   src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
# </script>

# My implementation of the [Stacked autoencoder deep network exercise](http://ufldl.stanford.edu/wiki/index.php/Exercise:_Implement_deep_networks_for_digit_classification)
# for the Deep Learning class.

import time
import sys
sys.path.append('..')
from library.imports import *
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
sae1_theta = util.initialize_parameters(hidden_size_l1, input_size)

# Train the first layer sparse autoencoder. This layer has a hidden
# size of `hidden_size_l1`.

# sae1_opt_theta, loss = minFunc( @(p) sparseAutoencoderLoss(p, ...
#     inputSize, hiddenSizeL1, ...
#     lambda, sparsityParam, ...
#     beta, trainData), ...
#     sae1Theta, options);
