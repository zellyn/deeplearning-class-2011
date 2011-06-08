%%
%% Matlab version of dist_belief exercise for UFLDF class.
%%

addpath '../library/'
addpath '../library/minFunc/'

% X: 5000 x 27648
% y: 5000 x 1
% class_names: 1 x 10
% size(fold_indices): 1 x 10
% size(fold_indices{1}): 1000 x 1


% load '../data/stl10_matlab/train.mat';
% trainImages = reshape(X,5000,96,96,3);
% trainImagesBw = squeeze(mean(trainImages, 4));

% TODO(zellyn): pull out just horses/planes or etc.

visibleDim = 96;  % width/height of input
visibleSize = visibleDim^2;

hiddenDimL1 = 18;  % width/height of hidden layer 1
hiddenSizeL1 = hiddenDimL1^2;
hiddenViewDimL1 = 11;  % width/height of local receptive area for hidden layer 1
hiddenViewSizeL1 = hiddenViewDimL1^2;  % local receptive area size for hidden layer 1
hiddenViewStepL1 = 5;  % step for receptive area moving across input
assert (((hiddenDimL1-1) * hiddenViewStepL1 + hiddenViewDimL1) == visibleDim);

hiddenDimL2 = 8;  % width/height of hidden layer 2
hiddenSizeL2 = hiddenDimL2^2;

hiddenViewDimL2 = 11;  % width/height of local receptive area for hidden layer 2
hiddenViewSizeL2 = hiddenViewDimL2^2;  % local receptive area size for hidden layer 2
hiddenViewStepL2 = 1;  % step for receptive area moving across hidden layer 1
assert (((hiddenDimL2-1) * hiddenViewStepL2 + hiddenViewDimL2) == hiddenDimL1);

[W1, b1] = initializeOneLayerParams(hiddenSizeL1, hiddenViewSizeL1);
[W2, b2] = initializeOneLayerParams(hiddenSizeL2, hiddenViewSizeL2);
[W3, b3] = initializeOneLayerParams(1, hiddenSizeL2);

W1Indices = buildIndices(hiddenSizeL1, visibleSize, hiddenDimL1, visibleDim, hiddenViewDimL1, hiddenViewStepL1);
W2Indices = buildIndices(hiddenSizeL2, hiddenSizeL1, hiddenDimL2, hiddenDimL1, hiddenViewDimL2, hiddenViewStepL2);

assert (size(W1Indices) == [hiddenViewSizeL1, hiddenSizeL1]);
assert (size(W2Indices) == [hiddenViewSizeL2, hiddenSizeL2]);
