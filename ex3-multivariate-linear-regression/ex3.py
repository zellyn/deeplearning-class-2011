#! ~/.ve/np/bin/python

import time
import sys
import numpy as np
import scipy as sp
import matplotlib as mpl
if sys.platform == 'darwin': mpl.use('TkAgg')

import matplotlib.pyplot as plt
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D

x = np.loadtxt('ex3x.dat')
y = np.loadtxt('ex3y.dat')[:, np.newaxis]

x = np.hstack([np.ones([x.shape[0], 1]), x])
m, n = x.shape

# plt.figure()
# plt.plot(x[:,1], y, 'x')
# plt.ylabel('Price in \\$')
# plt.xlabel('Living area in feet$^2$')

# plt.figure()
# plt.plot(x[:,2], y, 'x')
# plt.ylabel('Price in \\$')
# plt.xlabel('Number of bedrooms')

sigma = np.std(x, 0, ddof=1)
mu = np.mean(x, 0)
x = (x - (mu - [1,0,0])) / (sigma + [1,0,0])

num_iterations = 50
alphas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 1.3, 3, 10]

J = np.zeros([num_iterations, len(alphas)])

for i, alpha in enumerate(alphas):
    theta = np.zeros([n, 1])
    for j in range(num_iterations):
        J[j, i] = (1.0 / 2*m) * (x.dot(theta) - y).T.dot(x.dot(theta) - y)
        theta = theta - alpha * (1.0/m) * x.T.dot(x.dot(theta) - y)

plt.figure()
for i, alpha in enumerate(alphas[:-2]):
    plt.plot(np.arange(0, num_iterations), J[:,i], '-')

alpha = 1
theta = np.zeros([n, 1])
for j in range(100):
    theta = theta - alpha * (1.0/m) * x.T.dot(x.dot(theta) - y)

print theta

x1 = (np.array([2, 1650, 3]) - mu) / (sigma + [1,0,0])
y_predicted_1 = x1.dot(theta)[0]
print 'First prediction: %s' % y_predicted_1


# Normal Equations

# Reload data

x = np.loadtxt('ex3x.dat')
y = np.loadtxt('ex3y.dat')[:, np.newaxis]

x = np.hstack([np.ones([x.shape[0], 1]), x])
m, n = x.shape

theta = linalg.solve(x.T.dot(x), x.T.dot(y))

x2 = np.array([1, 1650, 3])
y_predicted_2 = x2.dot(theta)[0]
print 'Second prediction: %s' % y_predicted_2



plt.show()
