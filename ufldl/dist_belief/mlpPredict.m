function [pred] = mlpPredict(theta, ...
                             visibleSize, ...
                             hiddenSizeL1, hiddenViewSizeL1, ...
                             hiddenSizeL2, hiddenViewSizeL2, ...
                             W1Indices, W2Indices, ...
                             data)

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

  pred = z4 >= 0;
end

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
