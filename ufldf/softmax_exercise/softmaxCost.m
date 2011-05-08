function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% theta - numClasses*k
% numClasses - the number of classes
% inputSize - the size k of the input vector
% lambda - weight decay parameter
% data - the kxm input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an 1xm matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

% Compute theta^T x
theta_x = theta * data;
% Subtract out the max by column so we don't overflow (this'll affect numerator
% and denominator equally)
theta_x = bsxfun(@minus, theta_x, max(theta_x, [], 1));
% e^(theta^T x)
e_theta_x = e .^ theta_x;
% Normalize
h_x = bsxfun(@rdivide, e_theta_x, sum(e_theta_x));


%% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

