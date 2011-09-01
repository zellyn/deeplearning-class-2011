function product = backMultiplyStripes(W, indices, deltas)
  n = size(W,1);
  w = size(W,2);

  originalSize = max(indices(:));  % 10
  multiplier = zeros(originalSize, n);

  for i = 1:n
    multiplier(indices(:, i), i) = W(i,:);
  end

  product = multiplier * deltas;
end
