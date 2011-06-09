function [cost, grad] = mlpCost(theta, ...
                                visibleSize, ...
                                hiddenSizeL1, hiddenViewSizeL1, ...
                                hiddenSizeL2, hiddenViewSizeL2, ...
                                W1Indices, W2Indices, ...
                                lambda, data, labels)
  cost = 0;
  grad = zeros(size(theta));

  m = size(data, 2);
  [W1, b1, W2, b2, W3, b3] = unpackTheta(theta, ...
                                         hiddenSizeL1, hiddenViewSizeL1, ...
                                         hiddenSizeL2, hiddenViewSizeL2);

  % z2 = W1 * data + repmat(b1, [1, m]);
  z2 = multiplyStripes(W1, b1, W1Indices, data);
  a2 = sigmoid(z2);
  % z3 = W2 * a2 + repmat(b2, [1, m]);
  z3 = multiplyStripes(W2, b2, W2Indices, a2);
  a3 = sigmoid(z3);

  z4 = W3 * a3 + repmat(b3, [1, m]);
  a4 = sigmoid(z4);

  squares = (a4 - labels).^2;
  squared_err_J = (1/2) * (1/m) * sum(squares(:));
  weight_decay_J = (lambda/2) * (sum(W1(:).^2) + sum(W2(:).^2) + sum(W3(:).^2));

  cost = squared_err_J + weight_decay_J;

  % ============== EVERYTHING FROM HERE DOWN IS WRONG =====================
  % Taken from sparseAutoencoderLoss.m, with sparsity removed:
  % It assumes a fully-connected, two-layer network.
  % =======================================================================

  % delta4 = -(labels - a4) .* fprime(z4);
  % but fprime(z4) = a4 * (1-a4)
  delta4 = -(labels - a4) .* a4 .* (1-a4);
  delta3 = (W3' * delta4) .* a3 .* (1-a3);
  delta2 = backMultiplyStripes(W2, W2Indices, delta3) .* a2 .* (1-a2);

  % data: 81x7
  % W1: 16x9
  % W2: 4x9
  % W3: 1x4
  % delta2, a2: 16x7
  % delta3, a3: 4x7
  % delta4, a4: 1x7

  % 16x7, 4x9 -> 4x7
  % 4x7, 7x16 -> 4x9

  W3grad = (1/m) * delta4 * a3' + lambda * W3;
  b3grad = (1/m) * sum(delta4, 2);
  W2grad = (1/m) * indexedWeightSubset(delta3 * a2', W2Indices) + lambda * W2;
  b2grad = (1/m) * sum(delta3, 2);
  W1grad = (1/m) * indexedWeightSubset(delta2 * data', W1Indices) + lambda * W1;
  b1grad = (1/m) * sum(delta2, 2);

  grad = [W1grad(:) ; b1grad(:); W2grad(:) ; b2grad(:) ; W3grad(:) ; b3grad(:)];
end

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
