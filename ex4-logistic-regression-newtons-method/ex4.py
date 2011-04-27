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

x = np.loadtxt('ex4x.dat')
y = np.loadtxt('ex4y.dat')[:, np.newaxis]

x = np.hstack([np.ones([x.shape[0], 1]), x])
m, n = x.shape

def g(z):
  return 1.0 / (1.0 + np.exp(-z))

theta = np.zeros([n, 1])
diff = np.ones([n, 1])
Js = np.array([])

while linalg.norm(diff) > 1e-6:
  h = g(x.dot(theta))
  J = (1.0/m) * np.sum(-y * np.log(h) - (1-y) * np.log(1-h))
  Js = np.append(Js, J)
  grad = (1.0/m) * x.T.dot(h-y)
  H = (1.0/m) * (h * (1-h) * x).T.dot(x)
  diff = linalg.solve(H, grad)
  theta = theta - diff

print "Theta:\n%s" % theta
print "Iterations: %d" % len(Js)
prob = 1 - g(np.array([1, 20, 80]).dot(theta))
print "Prob: %f" % prob

plot_x = np.array([np.min(x[:,1])-2, np.max(x[:,1])+2])
plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])
plt.figure()
p1 = plt.plot(x[:,1:2][y==1], x[:,2:3][y==1], '+')
p2 = plt.plot(x[:,1:2][y==0], x[:,2:3][y==0], 'o', mec='blue', mfc='none')
p3 = plt.plot(plot_x, plot_y, '-')
plt.ylabel('Exam 1 score')
plt.xlabel('Exam 2 score')
plt.legend((p1, p2, p3), ('Admitted', 'Not admitted', 'Decision boundary'), 'upper right')

plt.figure()
plt.plot(Js, 'o--')
plt.ylabel(r'$J(\theta)$')
plt.xlabel('Iteration')

plt.show()
