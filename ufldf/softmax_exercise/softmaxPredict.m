function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix
% pred, where pred(i) is argmax_c P(y(c) | x(i)).

% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
pred = zeros(1, size(data, 2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start
%                from 1.

inputSize = softmaxModel.inputSize
numClasses = softmaxModel.numClasses
size(theta)
size(data)
theta_x = theta * data;
theta_x = bsxfun(@minus, theta_x, max(theta_x, [], 1));
e_theta_x = exp(theta_x);
h_x = bsxfun(@rdivide, e_theta_x, sum(e_theta_x));

[nop, pred] = max(h_x);

% ---------------------------------------------------------------------

end

