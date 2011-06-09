function product = multiplyStripes(W, b, indices, data)
  n = size(W,1);
  m = size(data, 2);
  product = zeros(n, m);
  for i = 1:n
    product(i, :) = W(i,:) * data(indices(:, i), :);
  end
  product = product + repmat(b, 1, m);
end
