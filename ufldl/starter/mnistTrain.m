% MNIST training

addpath '../library/'

images = loadMNISTImages('../data/train-images-idx3-ubyte');  % 784 x 60000
labels = loadMNISTLabels('../data/train-labels-idx1-ubyte');  % 60000 x 1

display_network(images(:,1:100)); % Show the first 100 images
disp(labels(1:10));

visibleSize = 28*28;
hiddenSize = 196;
sparsityParam = 0.1;
lambda = 3e-3;
beta = 3;
patches = images(:,1:10000);
theta = initializeParameters(hiddenSize, visibleSize);
addpath '../library/minFunc/'
options.Method = 'lbfgs';
options.maxIter = 400;
options.display = 'on';
[opttheta, loss] = minFunc( @(p) sparseAutoencoderLoss(p, ...
    visibleSize, hiddenSize, ...
    lambda, sparsityParam, ...
    beta, patches), ...
    theta, options);

W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
display_network(W1', 12);
% print -djpeg 'mnist_weights.jpg';
