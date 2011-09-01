patchSize = 8;
visibleSize = patchSize * patchSize;
hiddenSize = 25;
targetActivation = 0.01;
lambda = 0.0001;
beta = 3;
load 'testdata/save_patches.mat';
load 'testdata/save_theta.mat';

% [loss, grad] = sparseAutoencoderLoss(theta, visibleSize, hiddenSize, lambda, target_activation, beta, data);

addpath '../library/minFunc/'
options.Method = 'lbfgs';
options.maxIter = 400;
options.display = 'on';
[opttheta, loss] = minFunc( @(p) sparseAutoencoderLoss(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, targetActivation, ...
                                   beta, patches), ...
                                   theta, options);

save 'testdata/matlab-opttheta.mat' opttheta;
