function [] = checkMlpCost()

% Check the gradients for the MLP
%

addpath '../library/'

%% Setup random data / small model

visibleDim = 9;  % width/height of input
visibleSize = visibleDim^2;

hiddenDimL1 = 4;  % width/height of hidden layer 1
hiddenSizeL1 = hiddenDimL1^2;
hiddenViewDimL1 = 3;  % width/height of local receptive area for hidden layer 1
hiddenViewSizeL1 = hiddenViewDimL1^2;  % local receptive area size for hidden layer 1
hiddenViewStepL1 = 2;  % step for receptive area moving across input
assert (((hiddenDimL1-1) * hiddenViewStepL1 + hiddenViewDimL1) == visibleDim);

hiddenDimL2 = 2;  % width/height of hidden layer 2
hiddenSizeL2 = hiddenDimL2^2;

hiddenViewDimL2 = 3;  % width/height of local receptive area for hidden layer 2
hiddenViewSizeL2 = hiddenViewDimL2^2;  % local receptive area size for hidden layer 2
hiddenViewStepL2 = 1;  % step for receptive area moving across hidden layer 1
assert (((hiddenDimL2-1) * hiddenViewStepL2 + hiddenViewDimL2) == hiddenDimL1);

[W1, b1] = initializeOneLayerParams(hiddenSizeL1, hiddenViewSizeL1);
[W2, b2] = initializeOneLayerParams(hiddenSizeL2, hiddenViewSizeL2);
[W3, b3] = initializeOneLayerParams(1, hiddenSizeL2);

W1Indices = buildIndices(visibleDim, hiddenDimL1, hiddenViewDimL1, hiddenViewStepL1);
W2Indices = buildIndices(hiddenDimL1, hiddenDimL2, hiddenViewDimL2, hiddenViewStepL2);

assert(all(size(W1Indices) == [hiddenViewSizeL1, hiddenSizeL1]));
assert(all(size(W2Indices) == [hiddenViewSizeL2, hiddenSizeL2]));

theta = [W1(:);b1(:);W2(:);b2(:);W3(:);b3(:)];

lambda = 0.01;
data   = randn(visibleSize, 7);
labels = [ 0 1 0 1 0 1 0];

[cost, grad] = mlpCost(theta, visibleSize, ...
                       hiddenSizeL1, hiddenViewSizeL1, ...
                       hiddenSizeL2, hiddenViewSizeL2, ...
                       W1Indices, W2Indices, ...
                       lambda, data, labels);

% Check that the numerical and analytic gradients are the same
numgrad = computeNumericalGradient( @(x) mlpCost(x, visibleSize, ...
                                                 hiddenSizeL1, hiddenViewSizeL1, ...
                                                 hiddenSizeL2, hiddenViewSizeL2, ...
                                                 W1Indices, W2Indices, ...
                                                 lambda, data, labels), ...
                                   theta);

% Use this to visually compare the gradients side by side
% disp([numgrad grad (numgrad-grad)]);

% Compare numerically computed gradients with the ones obtained from backpropagation
% disp('Norm between numerical and analytical gradient (should be less than 1e-9)');
diff = norm(numgrad-grad)/norm(numgrad+grad);

% disp(diff); % Should be small. In our implementation, these values are
              % usually less than 1e-9.
assert (diff < 1e-9);

end
