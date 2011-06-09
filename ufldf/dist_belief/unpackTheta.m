function [W1, b1, W2, b2, W3, b3] = unpackTheta(theta, hiddenSizeL1, hiddenViewSizeL1, hiddenSizeL2, hiddenViewSizeL2)
  % W1: hiddenSizeL1, hiddenViewSizeL1
  % b1: hiddenSizeL1, 1
  % W2: hiddenSizeL2, hiddenViewSizeL2
  % b2: hiddenSizeL2, 1
  % W3: 1, hiddenSizeL2
  % b3: 1, 1

  W1size = hiddenSizeL1 * hiddenViewSizeL1;
  W2start = 1 + W1size + hiddenSizeL1;
  W2size = hiddenSizeL2 * hiddenViewSizeL2;
  W3start = W2start + W2size + hiddenSizeL2;

  W1 = reshape(theta(1:W1size), hiddenSizeL1, hiddenViewSizeL1);
  b1 = reshape(theta(1+W1size:W1size+hiddenSizeL1), hiddenSizeL1, 1);
  W2 = reshape(theta(W2start:W2start+W2size-1), hiddenSizeL2, hiddenViewSizeL2);
  b2 = reshape(theta(W2start+W2size:W2start+W2size+hiddenSizeL2-1), hiddenSizeL2, 1);
  W3 = reshape(theta(W3start:W3start+hiddenSizeL2-1), 1, hiddenSizeL2);
  b3 = reshape(theta(end), 1, 1);
end
