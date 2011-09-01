function [W, b] = initializeOneLayerParams(layerSize, inputSize)
%% Initialize Parameters Randomly Based on Layer Sizes
r  = sqrt(6) / sqrt(layerSize+inputSize+1);
W = rand(layerSize, inputSize) * 2 * r - r;
b = zeros(layerSize, 1);
end
