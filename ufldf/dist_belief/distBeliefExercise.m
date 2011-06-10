%%
%% Matlab version of dist_belief exercise for UFLDF class.
%%

addpath '../library/'
addpath '../library/minFunc/'

% 3e-4
% 1e-3:
% 3e-3: 75.438%
% 1e-2: 50% (doesn't work)
lambda = 1e-3;

% X: 5000 x 27648
% y: 5000 x 1
% class_names: 1 x 10

labelHorse = 7;
labelPlane = 1;
labelDog = 6;
labelCat = 4;

disp('Loading training data');
load '../data/stl10_matlab/train.mat';
disp('Reformatting training data');
X = double(X)/255;
trainImages = reshape(X,5000,96,96,3);
trainImagesBw = squeeze(mean(trainImages, 4));
X = reshape(trainImagesBw, 5000, 96^2);

mask_hp = (y == labelHorse) | (y == labelPlane);
mask_dc = (y == labelDog) | (y == labelCat);

y_hp = y(mask_hp);
Xtrain_hp = X(mask_hp, :)';
y_dc = y(mask_dc);
Xtrain_dc = X(mask_dc, :)';
labelsTrain_hp = (y_hp == labelHorse)';  % Horse = 1, Plane = 0
labelsTrain_dc = (y_dc == labelDog)';    % Dog = 1, Cat = 0


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

W1Indices = buildIndices(visibleDim, hiddenDimL1, hiddenViewDimL1, hiddenViewStepL1);
W2Indices = buildIndices(hiddenDimL1, hiddenDimL2, hiddenViewDimL2, hiddenViewStepL2);

assert(all(size(W1Indices) == [hiddenViewSizeL1, hiddenSizeL1]));
assert(all(size(W2Indices) == [hiddenViewSizeL2, hiddenSizeL2]));

theta = [W1(:);b1(:);W2(:);b2(:);W3(:);b3(:)];

options.Method = 'lbfgs';
options.maxIter = 400;
options.display = 'on';

[optTheta, loss] = minFunc( @(x) mlpCost(x, visibleSize, ...
                                         hiddenSizeL1, hiddenViewSizeL1, ...
                                         hiddenSizeL2, hiddenViewSizeL2, ...
                                         W1Indices, W2Indices, ...
                                         lambda, Xtrain_hp, labelsTrain_hp), ...
                           theta, options);

[pred] = mlpPredict(optTheta, visibleSize, ...
                    hiddenSizeL1, hiddenViewSizeL1, ...
                    hiddenSizeL2, hiddenViewSizeL2, ...
                    W1Indices, W2Indices, ...
                    Xtrain_hp);
acc = mean(labelsTrain_hp(:) == pred(:));
fprintf('Accuracy on input data: %0.3f%%\n', acc * 100);


disp('Loading test data');
load '../data/stl10_matlab/test.mat';
disp('Reformatting test data');
X = double(X)/255;
testImages = reshape(X,8000,96,96,3);
testImagesBw = squeeze(mean(testImages, 4));
X = reshape(testImagesBw, 8000, 96^2);

mask_hp = (y == labelHorse) | (y == labelPlane);
mask_dc = (y == labelDog) | (y == labelCat);

y_hp = y(mask_hp);
Xtest_hp = X(mask_hp, :)';
y_dc = y(mask_dc);
Xtest_dc = X(mask_dc, :)';
labelsTest_hp = (y_hp == labelHorse)';  % Horse = 1, Plane = 0
labelsTest_dc = (y_dc == labelDog)';    % Dog = 1, Cat = 0

[pred] = mlpPredict(optTheta, visibleSize, ...
                    hiddenSizeL1, hiddenViewSizeL1, ...
                    hiddenSizeL2, hiddenViewSizeL2, ...
                    W1Indices, W2Indices, ...
                    Xtest_hp);
acc = mean(labelsTest_hp(:) == pred(:));
fprintf('Test Accuracy: %0.3f%%\n', acc * 100);
