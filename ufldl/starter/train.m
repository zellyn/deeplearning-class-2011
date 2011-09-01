%% CS294a Introductory Homework Starter Code

addpath '../library/'

%  Instructions
%  ------------
%
%  This starter file contains code that helps you get started on the
%  homework. You are required to complete the code in sampleIMAGES.m,
%  sparseAutoencoderLoss.m and computeNumericalGradient.m
%  For the purpose of completing the assignment, you do not need to
%  change the code in this file.
%
%%======================================================================
%% STEP 0: We are giving you all relevent parameters to get good filters,
% So you do not need to change the parameters below
visibleSize = 8*8;
hiddenSize = 25;
% hiddenSize = 3;  % (For testing)
targetActivation = 0.01;   % Try between 0.01 to 0.05
lambda = 0.0001;           % Try small values about 0.0001
beta = 3;                  % Try between 1 to 10

%%======================================================================
%% STEP 1: Implement sampleIMAGES
%
%  After implementing sampleIMAGES, the display_network command should
%  display a random sample of 200 patches from the dataset

patches = sampleIMAGES;
% patches = patches(:,1:10);  % (For testing)
display_network(patches(:,randi(size(patches,2),200,1)),8);

%  Obtain random theta
theta = initializeParameters(hiddenSize, visibleSize);

%%======================================================================
%% STEP 2: Implement sparseAutoencoderLoss
%
%  You can implement all the components in the loss function at once but it
%  may be easier to do it step-by-step and run gradient checking (see STEP
%  3) for every component. We suggest implementing the sparseAutoencoderLoss
%  function using the following steps :
%
%  (a) Implement the loss for forward propagation, reconstruction
%      cost and back propagation then run Gradient Checking
%
%  (b) Implement Weight Regularization then run Gradient Checking
%
%  (c) Implement Sparsity Cost then Gradient Checking
%
%  Hint: You are free to change the training settings when debugging your
%  code. However, in your final submission of visualized weights, use the
%  following settings for the parameters.

[loss, grad] = sparseAutoencoderLoss(theta, visibleSize, hiddenSize, lambda, ...
                                     targetActivation, beta, patches);

% %%======================================================================
% %% STEP 3: Gradient Checking
% %  Hint: Perform gradient checks on small models and datasets
% %        (e.g. use only first 10 patches and 1 hidden unit)
% numgrad = computeNumericalGradient( @(x) sparseAutoencoderLoss(x, visibleSize, ...
%                                                   hiddenSize, lambda, ...
%                                                   targetActivation, beta, ...
%                                                   patches), theta);
%
% % Use this to eyeball the gradients
% disp([numgrad grad]);
%
% % Compare numerical gradients and analytical
% diff = norm(numgrad-grad)/norm(numgrad+grad);
% disp(diff); % should be small e.g. < 1e-6

%%======================================================================
%% STEP 4: Run your sparseAutoencoderLoss function with minFunc and
%      visualize your results

%  Initialize random theta
theta = initializeParameters(hiddenSize, visibleSize);

%  Use minFunc to minimize the function
addpath '../library/minFunc/'
options.Method = 'lbfgs';
options.maxIter = 400;
options.display = 'on';


[opttheta, loss] = minFunc( @(p) sparseAutoencoderLoss(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, targetActivation, ...
                                   beta, patches), ...
                              theta, options);


%%======================================================================
%% STEP 5: Visualization
W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
display_network(W1', 12);
