import numpy as np

def compute_numerical_gradient(fn, theta, epsilon=1e-4):
  numgrad = np.zeros(theta.shape)

  for i in range(theta.size):
    theta_minus = theta.copy()
    theta_plus = theta.copy()
    theta_minus[i] = theta_minus[i] - epsilon
    theta_plus[i] = theta_plus[i] + epsilon
    numgrad[i] = (fn(theta_plus) - fn(theta_minus)) / (2 * epsilon)

  return numgrad
