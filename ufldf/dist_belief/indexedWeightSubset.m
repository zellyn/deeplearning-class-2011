function subweights = indexedWeightSubset(weights, indices)
  [numOutputs, fullSize] = size(weights);
  smallSize = size(indices, 1);
  assert (size(indices, 2) == numOutputs);

  subweights = zeros(numOutputs, smallSize);

  for i = (1:numOutputs)
    subweights(i,:) = weights(i, indices(:, i));
  end
end
