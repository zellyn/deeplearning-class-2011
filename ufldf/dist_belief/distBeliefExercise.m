%%
%% Matlab version of dist_belief exercise for UFLDF class.
%%

addpath '../library/'
addpath '../library/minFunc/'

% 0: (82.188 - 164/187), (81.188 - 273/351)
% 1e-4: (82.375 - 1000)
% 3e-4: (82.938 - 1000), (83.062 - 1000), (83.000 with 1591/1767)
% 1e-3: (78.188 - 400), (78.375 - 481/525)
% 3e-3: (75.438)
% 1e-2: 50%
lambda = 3e-4;
maxFunEvals = 1000;
maxIter = 10000;
minGrad = 1e-6;

% 10: explodes
% 3: explodes
% 1: alright (plummet, then slope)
alpha = 2;
alphaGrow = 1.01;
alphaShrink = 0.5;

USE_MINFUNC = false;

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

if USE_MINFUNC
  options.Method = 'lbfgs';
  options.maxIter = maxIter;
  options.maxFunEvals = maxFunEvals;
  options.display = 'on';
  [optTheta, loss] = minFunc( @(x) mlpCost(x, visibleSize, ...
                                           hiddenSizeL1, hiddenViewSizeL1, ...
                                           hiddenSizeL2, hiddenViewSizeL2, ...
                                           W1Indices, W2Indices, ...
                                           lambda, Xtrain_hp, labelsTrain_hp), ...
                             theta, options);
else
  js = zeros(1,1);
  h = plot(js, 'YDataSource', 'js');
  js = zeros(0,1);
  drawnow;
  n = 1;
  i = 0;
  lastCost = 1e10;
  while ((i < maxIter) && n >= minGrad)
    i = i + 1;
    [cost, grad] = mlpCost(theta, visibleSize, ...
                           hiddenSizeL1, hiddenViewSizeL1, ...
                           hiddenSizeL2, hiddenViewSizeL2, ...
                           W1Indices, W2Indices, ...
                           lambda, Xtrain_hp, labelsTrain_hp);
    if (cost > lastCost)
      alpha = alpha * alphaShrink;
    else
      alpha = alpha * alphaGrow;
    end
    lastCost = cost;
    n = norm(grad);
    fprintf('%d: cost=%e  norm(grad)=%e  alpha=%e\n', i, cost, n, alpha);
    if (i > 0)
      js = [js; cost];
    end
    theta = theta - alpha * grad;
    if (mod(i,25)==0)
      refreshdata(h, 'caller')
      drawnow;
    end
  end
  optTheta = theta;
end

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
fprintf('(lambda = %0.0e\n)', lambda);
