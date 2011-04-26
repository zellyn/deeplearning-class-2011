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

x = np.loadtxt('ex5Linx.dat')[:, np.newaxis]
y = np.loadtxt('ex5Liny.dat')[:, np.newaxis]

def f1(x):
  return np.hstack([x**0, x, x**2, x**3, x**4, x**5])

x = f1(x)
m, n = x.shape
L = np.diag(np.ones(n)); L[0, 0] = 0

theta_0 = linalg.solve(x.T.dot(x), x.T.dot(y))
theta_1 = linalg.solve(x.T.dot(x) + L, x.T.dot(y))
theta_10 = linalg.solve(x.T.dot(x) + 10 * L, x.T.dot(y))

xs = np.linspace(-1, 1, 100)[:, np.newaxis]
xs = f1(xs)

plt.figure()
p1 = plt.plot(x[:, 1:2], y, 'o')
p2 = plt.plot(xs[:, 1:2], xs.dot(theta_0), 'r--')
p3 = plt.plot(xs[:, 1:2], xs.dot(theta_1), 'g--')
p4 = plt.plot(xs[:, 1:2], xs.dot(theta_10), 'b--')
plt.legend((p1, p2, p3, p4), ('Training data', '$\lambda=0$', '$\lambda=1$', '$\lambda=10$'), 'upper right')



# Logistic Regression ########################################################

x = np.loadtxt('ex5Logx.dat', delimiter=',')
y = np.loadtxt('ex5Logy.dat')[:, np.newaxis]
u = x[:, 0:1]; v = x[:, 1:2]

def g(z):
    return 1.0 / (1.0 + np.exp(-z))

def map_feature(u, v):
  if isinstance(u, (int, long, float)):
    u = np.array([[u]])
    v = np.array([[v]])
  ret = np.empty([u.size, 0])
  degree = 6
  for i in range(0, degree+1):
    for j in range(0, degree-i+1):
      ret = np.hstack([ret, u**i * v**j])
  return ret

X = map_feature(u, v)
m, n = X.shape

eye_0 = np.eye(n); eye_0[0, 0] = 0

lambdas = [0.0, 1.0, 10.0]

thetas = np.empty([n, 0])
for lam in lambdas:
  theta = np.zeros([n,1])
  diff = np.ones([n,1])
  iterations = 0
  while linalg.norm(diff) > 1e-6:
    iterations += 1
    h = g(X.dot(theta))
    theta_0 = eye_0.dot(theta)
    J = -(1.0/m) * np.sum(y * np.log(h) + (1-y) * np.log(1-h)) + (lam/(2*m)) * theta_0.T.dot(theta_0)
    grad = (1.0/m) * X.T.dot(h-y) + ((lam/m) * theta_0)
    H = (1.0/m) * (h * (1-h) * X).T.dot(X) + ((lam/m) * eye_0)
    diff = linalg.solve(H, grad)
    theta = theta - diff
  print "Iterations: %d" % iterations
  thetas = np.hstack([thetas, theta])

# Plotting contours

grid_size = 200
uu = np.linspace(-1, 1.5, grid_size)[:, np.newaxis]
vv = np.linspace(-1, 1.5, grid_size)

z0 = np.empty([grid_size, 0])
z1 = np.empty([grid_size, 0])
z10= np.empty([grid_size, 0])

for vvj in vv:
  z0 = np.hstack([z0, map_feature(uu, np.array([[vvj]])).dot(thetas[:, 0:1])])
  z1 = np.hstack([z1, map_feature(uu, np.array([[vvj]])).dot(thetas[:, 1:2])])
  z10 = np.hstack([z10, map_feature(uu, np.array([[vvj]])).dot(thetas[:, 2:3])])

z0 = z0.T
z1 = z1.T
z10 = z10.T

plt.figure()
p1 = plt.plot(x[:,0:1][y==1], x[:,1:2][y==1], '+')
p2 = plt.plot(x[:,0:1][y==0], x[:,1:2][y==0], 'o', mec='blue', mfc='none')
p3 = plt.contour(uu[:, 0], vv, z0, [0], colors='r')
p4 = plt.contour(uu[:, 0], vv, z1, [0], colors='g')
p5 = plt.contour(uu[:, 0], vv, z10, [0], colors='b')
plt.xlabel('u')
plt.ylabel('v')
p1[0].set_label('y=1')
p2[0].set_label('y=0')
p3.collections[0].set_label(r'$\lambda=0$')
p4.collections[0].set_label(r'$\lambda=1$')
p5.collections[0].set_label(r'$\lambda=10$')
plt.legend()
plt.show()
