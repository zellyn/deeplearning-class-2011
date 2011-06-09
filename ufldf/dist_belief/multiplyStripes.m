function product = multiplyStripes(W, b, indices, data)
  m = size(W,1);
  n = size(data, 2);
  product = zeros(m, n);
  for i = 1:m
    product(i, :) = W(i,:) * data(indices(:, i), :);
  end
  product = product + repmat(b, 1, n);
end
