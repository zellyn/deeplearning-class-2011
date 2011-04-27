#! ~/.ve/np/bin/python

import time
import sys
import numpy as np
import scipy as sp
import matplotlib as mpl
if sys.platform == 'darwin':
  mpl.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D

X = np.loadtxt('ex2x.dat')[:, np.newaxis]
y = np.loadtxt('ex2y.dat')[:, np.newaxis]

plt.figure()
plt.plot(X, y, 'o')
plt.ylabel('Height in meters')
plt.xlabel('Age in years')

X = np.hstack([np.ones([X.shape[0], 1]), X])
m, n = X.shape

alpha = 0.07
theta = np.zeros([n, 1])

delta = - alpha * (1.0/m) * X.T.dot(X.dot(theta) - y)
theta += delta

print theta

while linalg.norm(delta) > 1e-6:
  delta = - alpha * (1.0/m) * X.T.dot(X.dot(theta) - y)
  theta += delta

print theta

plt.plot(X[:, 1:], X.dot(theta), '-')
# plt.legend('Training data', 'Linear regression')

J_vals = np.zeros([100, 100])
theta0_vals = np.linspace(-3, 3, 100)
theta1_vals = np.linspace(-1, 1, 100)

for i in range(0, theta0_vals.size):
  for j in range(0, theta1_vals.size):
      t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
      J_vals[i, j] = (1.0 / (2 * m)) * np.sum( (X.dot(t) - y) ** 2)

J_vals = J_vals.T

fig = plt.figure()
ax = fig.gca(projection='3d')
xx, yy = np.meshgrid(theta0_vals, theta1_vals)

surf = ax.plot_surface(xx, yy, J_vals, rstride=1, cstride=1, cmap=mpl.cm.jet,
                       linewidth=0, antialiased=False)



plt.figure()
plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 2, 15));





plt.show()
