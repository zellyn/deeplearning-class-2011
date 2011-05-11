function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
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

l_theta_by_col = sum(theta_x .* groundTruth) - log(sum(e_theta_x));
l_theta = sum(l_theta_by_col);


cost = -l_theta + (lambda/2) * sum(sum(theta .^ 2));

dl_theta_by_col =
%% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

