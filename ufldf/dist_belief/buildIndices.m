function indices = buildIndices(inputDim, layerDim, viewDim, viewStep)
  inputSize = inputDim^2;
  layerSize = layerDim^2;

  assert (((layerDim-1) * viewStep + viewDim) == inputDim);
  mask = zeros(layerSize, inputSize);
  indices = zeros(viewDim^2, layerSize);

  index = 1;
  for c = (1:layerDim)
    viewStartC = (c-1) * viewStep + 1;
    for r = (1:layerDim)
      viewStartR = (r-1) * viewStep + 1;
      mask1 = zeros(inputDim, inputDim);
      mask1(viewStartR:viewStartR+viewDim-1, viewStartC:viewStartC+viewDim-1) = 1;
      mask(index, :) = mask1(:);
      index = index + 1;
    end
  end

  for i = 1:layerSize
    indices(:,i) = find(mask(i,:));
  end
end
