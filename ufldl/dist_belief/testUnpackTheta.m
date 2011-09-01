function [] = testUnpackTheta()

visibleDim = 5;  % width/height of input
visibleSize = visibleDim^2;

hiddenDimL1 = 3;  % width/height of hidden layer 1
hiddenSizeL1 = hiddenDimL1^2;
hiddenViewDimL1 = 3;  % width/height of local receptive area for hidden layer 1
hiddenViewSizeL1 = hiddenViewDimL1^2;  % local receptive area size for hidden layer 1
hiddenViewStepL1 = 1;  % step for receptive area moving across input
assert (((hiddenDimL1-1) * hiddenViewStepL1 + hiddenViewDimL1) == visibleDim);

hiddenDimL2 = 2;  % width/height of hidden layer 2
hiddenSizeL2 = hiddenDimL2^2;

hiddenViewDimL2 = 2;  % width/height of local receptive area for hidden layer 2
hiddenViewSizeL2 = hiddenViewDimL2^2;  % local receptive area size for hidden layer 2
hiddenViewStepL2 = 1;  % step for receptive area moving across hidden layer 1
assert (((hiddenDimL2-1) * hiddenViewStepL2 + hiddenViewDimL2) == hiddenDimL1);

[W1, b1] = initializeOneLayerParams(hiddenSizeL1, hiddenViewSizeL1);
[W2, b2] = initializeOneLayerParams(hiddenSizeL2, hiddenViewSizeL2);
[W3, b3] = initializeOneLayerParams(1, hiddenSizeL2);

theta = [W1(:);b1(:);W2(:);b2(:);W3(:);b3(:)];
[tW1, tb1, tW2, tb2, tW3, tb3] = unpackTheta(theta, hiddenSizeL1, hiddenViewSizeL1, hiddenSizeL2, hiddenViewSizeL2);

assert(all(W1(:) == tW1(:)));
assert(all(b1(:) == tb1(:)));
assert(all(W2(:) == tW2(:)));
assert(all(b2(:) == tb2(:)));
assert(all(W3(:) == tW3(:)));
assert(all(b3(:) == tb3(:)));

% Real sizes

visibleDim = 96;  % width/height of input
visibleSize = visibleDim^2;

hiddenDimL1 = 18;  % width/height of hidden layer 1
hiddenSizeL1 = hiddenDimL1^2;
hiddenViewDimL1 = 11;  % width/height of local receptive area for hidden layer 1
hiddenViewSizeL1 = hiddenViewDimL1^2;  % local receptive area size for hidden layer 1
hiddenViewStepL1 = 5;  % step for receptive area moving across input
assert(all(((hiddenDimL1-1) * hiddenViewStepL1 + hiddenViewDimL1) == visibleDim));

hiddenDimL2 = 8;  % width/height of hidden layer 2
hiddenSizeL2 = hiddenDimL2^2;

hiddenViewDimL2 = 11;  % width/height of local receptive area for hidden layer 2
hiddenViewSizeL2 = hiddenViewDimL2^2;  % local receptive area size for hidden layer 2
hiddenViewStepL2 = 1;  % step for receptive area moving across hidden layer 1
assert(all(((hiddenDimL2-1) * hiddenViewStepL2 + hiddenViewDimL2) == hiddenDimL1));

[W1, b1] = initializeOneLayerParams(hiddenSizeL1, hiddenViewSizeL1);
[W2, b2] = initializeOneLayerParams(hiddenSizeL2, hiddenViewSizeL2);
[W3, b3] = initializeOneLayerParams(1, hiddenSizeL2);

theta = [W1(:);b1(:);W2(:);b2(:);W3(:);b3(:)];
[tW1, tb1, tW2, tb2, tW3, tb3] = unpackTheta(theta, hiddenSizeL1, hiddenViewSizeL1, hiddenSizeL2, hiddenViewSizeL2);

assert(all(W1(:) == tW1(:)));
assert(all(b1(:) == tb1(:)));
assert(all(W2(:) == tW2(:)));
assert(all(b2(:) == tb2(:)));
assert(all(W3(:) == tW3(:)));
assert(all(b3(:) == tb3(:)));

end
