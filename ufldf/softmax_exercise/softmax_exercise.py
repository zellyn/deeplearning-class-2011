#! /Users/zellyn/.ve/np/bin/python
# -*- coding: utf-8 -*-

# <script type="text/javascript"
#   src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
# </script>

# My implementation of the [Softmax regression
# exercise](http://ufldl.stanford.edu/wiki/index.php/Exercise:Softmax_Regression)
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
#  Here we define and initialise some constants which allow your code
#  to be used more generally on any arbitrary input.  We also
#  initialise some parameters used for tuning the model.

input_size = 28 * 28
num_classes = 10
lamb = 1e-4

# === Step 1: Load data ===
#
#  In this section, we load the input and output data.  For softmax
#  regression on MNIST pixels, the input data is the images, and the
#  output data is the labels.

images = mnist.load_images('../data/train-images-idx3-ubyte')
labels = mnist.load_labels('../data/train-labels-idx1-ubyte')
input_data = images

# For debugging purposes, you may wish to reduce the size of the input
# data in order to speed up gradient checking.  Here, we consider only
# 8 pixels of the images, and only the first 100 images.

DEBUG = True
if DEBUG:
  input_size = 8
  # only 100 datapoints
  input_data = input_data[:, :100]
  labels = labels[:100]
  # only top input_size most-varying input elements (pixels)
  indices = input_data.var(1).argsort()[-input_size:]
  input_data = input_data[indices, :]

# Randomly initialise theta
theta = 0.005 * np.random.randn(num_classes * input_size, 1)

# === Step 2: Implement softmaxCost ===
#
# Implement softmaxCost in [softmax.cost()](softmax.html#section-1)

cost, grad = softmax.cost(theta, num_classes, input_size, lamb, input_data, labels)

# === Step 3: Gradient checking ===
#
#  As with any learning algorithm, you should always check that your
#  gradients are correct before learning the parameters.

# Cost function
def cost_func(x):
  return softmax.cost(x, num_classes, input_size, lamb, input_data, labels)[0]

# For testing…
if False:
  num_grad = util.compute_numerical_gradient(cost_func, theta)
  num_grad = num_grad.ravel('F')

  # Visually compare the gradients side by side
  print np.vstack([grad, num_grad]).T

  # Compare numerically computed gradients with those computed analytically
  diff = linalg.norm(num_grad - grad) / linalg.norm(num_grad + grad);
  print(diff)

# === Step 4: Learning parameters ===
#
#  Once you have verified that your gradients are correct, you can
#  start training your softmax regression code using softmax.train

options.maxIter = 100;
softmax_model = softmax.train(input_size, num_classes, lamb,
                              input_data, labels, max_iter=100)

