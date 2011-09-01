import numpy as np
import struct

def load_images(filename):
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

def load_labels(filename):
  with open(filename, 'rb') as f:
    magic = struct.unpack('>i', f.read(4))[0]
    assert magic == 2049, ("Bad magic number(%s) in filename '%s'" % (magic, filename))
    num_labels = struct.unpack('>i', f.read(4))[0]
    labels = np.fromstring(f.read(), dtype='uint8')
  assert labels.size == num_labels, 'Mismatch in label count'
  return labels
