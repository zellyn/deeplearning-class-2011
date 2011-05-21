%% CS294A/CS294W Stacked Autoencoder Exercise

%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  sstacked autoencoder exercise. You will need to complete code in
%  stackedAECost.m
%  You will also need to have implemented sparseAutoencoderCost.m and
%  softmaxCost.m from previous exercises.
%
%  For the purpose of completing the assignment, you do not need to
%  change the code in this file.
%

addpath '../library/'
addpath '../library/minFunc/'

SKIPTO = 6;
DISPLAY = true;

options.Method = 'lbfgs';
options.maxIter = 400;
options.display = 'on';

%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to
%  change the parameters below.

inputSize = 28 * 28;
numClasses = 10;
hiddenSizeL1 = 200;    % Layer 1 Hidden Size
hiddenSizeL2 = 200;    % Layer 2 Hidden Size
sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes).
lambda = 3e-3;         % weight decay parameter
beta = 3;              % weight of sparsity penalty term

%%======================================================================
%% STEP 1: Load data from the MNIST database
%
%  This loads our training data from the MNIST database files.

% Load MNIST database files
trainData = loadMNISTImages('../data/train-images-idx3-ubyte');
trainLabels = loadMNISTLabels('../data/train-labels-idx1-ubyte');

trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1

if SKIPTO <= 2
  %%======================================================================
  %% STEP 2: Train the first sparse autoencoder
  %  This trains the first sparse autoencoder on the unlabelled STL training
  %  images.
  %  If you've correctly implemented sparseAutoencoderCost.m, you don't need
  %  to change anything here.


  %  Randomly initialize the parameters
  sae1Theta = initializeParameters(hiddenSizeL1, inputSize);

  %% ---------------------- YOUR CODE HERE  ---------------------------------
  %  Instructions: Train the first layer sparse autoencoder, this layer has
  %                an hidden size of "hiddenSizeL1"
  %                You should store the optimal parameters in sae1OptTheta

  [sae1OptTheta, loss] = minFunc( @(p) sparseAutoencoderLoss(p, ...
      inputSize, hiddenSizeL1, ...
      lambda, sparsityParam, ...
      beta, trainData), ...
      sae1Theta, options);

  save('saves/step2.mat', 'sae1OptTheta');

  % -------------------------------------------------------------------------
else
  load('saves/step2.mat');
end

if DISPLAY
  W1 = reshape(sae1OptTheta(1:hiddenSizeL1 * inputSize), hiddenSizeL1, inputSize);
  display_network(W1');
end

%%======================================================================
%% STEP 3: Train the second sparse autoencoder
%  This trains the second sparse autoencoder on the first autoencoder
%  featurse.
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.

[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainData);

if SKIPTO <= 3
  %  Randomly initialize the parameters
  sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);

  %% ---------------------- YOUR CODE HERE  ---------------------------------
  %  Instructions: Train the second layer sparse autoencoder, this layer has
  %                an hidden size of "hiddenSizeL2" and an inputsize of
  %                "hiddenSizeL1"
  %
  %                You should store the optimal parameters in sae2OptTheta

  [sae2OptTheta, loss] = minFunc( @(p) sparseAutoencoderLoss(p, ...
      hiddenSizeL1, hiddenSizeL2, ...
      lambda, sparsityParam, ...
      beta, sae1Features), ...
      sae2Theta, options);

  save('saves/step3.mat', 'sae2OptTheta');

  % -------------------------------------------------------------------------
else
  load('saves/step3.mat')
end

if DISPLAY
  W11 = reshape(sae1OptTheta(1:hiddenSizeL1 * inputSize), hiddenSizeL1, inputSize);
  W12 = reshape(sae2OptTheta(1:hiddenSizeL2 * hiddenSizeL1), hiddenSizeL2, hiddenSizeL1);
  % TODO(zellyn): figure out how to display a 2-level network
  % display_network(log(W11' ./ (1-W11')) * W12');
end

%%======================================================================
%% STEP 4: Train the softmax classifier
%  This trains the sparse autoencoder on the second autoencoder features.
%  If you've correctly implemented softmaxCost.m, you don't need
%  to change anything here.

[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);
if SKIPTO <= 4
  %  Randomly initialize the parameters
  saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);


  %% ---------------------- YOUR CODE HERE  ---------------------------------
  %  Instructions: Train the softmax classifier, the classifier takes in
  %                input of dimension "hiddenSizeL2" corresponding to the
  %                hidden layer size of the 2nd layer.
  %
  %                You should store the optimal parameters in saeSoftmaxOptTheta
  %
  %  NOTE: If you used softmaxTrain to complete this part of the exercise,
  %        set saeSoftmaxOptTheta = softmaxModel.optTheta(:);

  softmaxModel = softmaxTrain(hiddenSizeL2, numClasses, 1e-4, ...
                              sae2Features, trainLabels, options);
  saeSoftmaxOptTheta = softmaxModel.optTheta(:);

  save('saves/step4.mat', 'saeSoftmaxOptTheta');

  % -------------------------------------------------------------------------
else
  load('saves/step4.mat');
end

%%======================================================================
%% STEP 5: Finetune softmax model

% Implement the stackedAECost to give the combined cost of the whole model
% then run this cell.

% Initialize the stack using the parameters learned
stack = cell(2,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the deep network, hidden size here refers to the '
%                dimension of the input to the classifier, which corresponds
%                to "hiddenSizeL2".
%
%

if SKIPTO <= 5
  [stackedAEOptTheta, loss] = minFunc( @(x) stackedAECost(x, ...
      inputSize, hiddenSizeL2, numClasses, netconfig, ...
      lambda, trainData, trainLabels), ...
      stackedAETheta, options);

  save('saves/step5.mat', 'stackedAEOptTheta');
else
  load('saves/step5.mat');
end

% -------------------------------------------------------------------------

if DISPLAY
  optStack = params2stack(stackedAEOptTheta(hiddenSizeL2*numClasses+1:end), netconfig);
  W11 = optStack{1}.w;
  W12 = optStack{2}.w;
  % TODO(zellyn): figure out how to display a 2-level network
  % display_network(log(1 ./ (1-W11')) * W12');
end


%%======================================================================
%% STEP 6: Test
%  Instructions: You will need to complete the code in stackedAEPredict.m
%                before running this part of the code
%

% Get labelled test images
% Note that we apply the same kind of preprocessing as the training set
testData = loadMNISTImages('../data/t10k-images-idx3-ubyte');
testLabels = loadMNISTLabels('../data/t10k-labels-idx1-ubyte');

testLabels(testLabels == 0) = 10; % Remap 0 to 10

[pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

% Accuracy is the proportion of correctly classified images
% The results for our implementation were:
%
% Before Finetuning Test Accuracy: XX.XXXX%
% After Finetuning Test Accuracy: XX.XXXX%
%
% If your values are too low (accuracy less than X.XX), you should check
% your code for errors, and make sure you are training on the
% entire data set of 60000 28x28 training images
% (unless you modified the loading code, this should be the case)

