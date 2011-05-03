#! /Users/zellyn/.ve/np/bin/python

import struct
from train import *

def load_MNIST_images(filename):
  with open(filename, 'rb') as f:
    magic = struct.unpack('>i', f.read(4))[0]
    assert magic == 2051, ("Bad magic number(%s) in filename '%s'" % (magic, filename))
    num_images, num_rows, num_cols = struct.unpack('>3i', f.read(12))
    num_bytes = num_images * num_rows * num_cols
    images = np.fromstring(f.read(), dtype='uint8')
  assert images.size == num_bytes, 'Mismatch in dimensions vs data size'
  images = images.reshape([num_cols, num_rows, num_images], order='F')
  images = images.swapaxes(0,1)

  # Reshape to #pixels x #examples
  images = images.reshape([num_cols*num_rows, num_images], order='F')
  # Convert to double and rescale to [0,1]
  images = images / 255.0
  return images

def load_MNIST_labels(filename):
  with open(filename, 'rb') as f:
    magic = struct.unpack('>i', f.read(4))[0]
    assert magic == 2049, ("Bad magic number(%s) in filename '%s'" % (magic, filename))
    num_labels = struct.unpack('>i', f.read(4))[0]
    labels = np.fromstring(f.read(), dtype='uint8')
  assert labels.size == num_labels, 'Mismatch in label count'
  return labels

def main(testing=True):
  images = load_MNIST_images('data/train-images-idx3-ubyte')  # 784 x 60000
  labels = load_MNIST_labels('data/train-labels-idx1-ubyte')  # 60000 x 1
  display_network(images[:,0:100])  # Show the first 100 images

  visible_size = 28*28
  hidden_size = 196
  sparsity_param = 0.1
  lamb = 3e-3
  beta = 3
  patches = images[:,0:10000]
  theta = initialize_parameters(hidden_size, visible_size)
  def sal(theta):
    return sparse_autoencoder_loss(theta, visible_size, hidden_size, lamb,
                                   sparsity_param, beta, patches)
  x, f, d = scipy.optimize.fmin_l_bfgs_b(sal, theta, maxfun=400, iprint=1, m=20)
  W1, W2, b1, b2 = unflatten(x, visible_size, hidden_size)
  display_network(W1.T)

if __name__ == '__main__':
  main(testing=False)
