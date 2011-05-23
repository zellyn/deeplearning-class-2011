function [numExamples, images, labels] = loadSTL10Data(setType, maxExamples)
%loadSTL10Data Load data (images and labels) for STL10 dataset
% Parameters:
%  setType - one of 'train', 'trainSmall', 'trainSmall64',
%            'test','testSmall', 'testSmall64'
%  (optional) maxExamples - truncate the examples to at most this number
%
% Returns:
%   numExamples - number of examples loaded
%   images - images, such that images(r, c, channel, image number) is the
%            intensity of the (r, c) pixel of the given channel for the
%            given image
%   labels - labels (ranges from 1 - 10 for the STL10 dataset)
%

% Properties of STL10 images

% If maxExamples is not set, set it to -1
if nargin < 2
    maxExamples = -1;
end

assert(isequal(setType, 'train') | isequal(setType, 'trainSmall') | isequal(setType, 'trainSmall64') | ...
       isequal(setType, 'test') | isequal(setType, 'testSmall') | isequal(setType, 'testSmall64'), ...
       'setType should be one of "train", "trainSmall", "trainSmall64", "test", "testSmall" or "testSmall64" ');

imageChannels = 3;
if isequal(setType(end-1:end), '64')
    imageDim = 64;
else
    imageDim = 96;
end

filename = sprintf('../data/stl10_matlab/%s.mat', setType);
images = load(filename, 'X');
images = double(images.X);
labels = load(filename, 'y');
labels = labels.y;

% If maxExamples was not set by the user (and hence is set to -1)
% or was set to a negative value by the user, take all examples
if maxExamples < 0
    maxExamples = size(labels, 1);
end

maxExamples = min(maxExamples, size(labels, 1));
fprintf('Using %s set, %d out of %d examples\n', upper(setType), maxExamples, size(labels, 1));

numExamples = size(labels, 1);
images = reshape(images, numExamples, imageDim, imageDim, imageChannels);
images = permute(images, [2 3 4 1]); % now images(r, c, channel, which image)

images = images(:, :, :, 1:maxExamples);
labels = labels(1:maxExamples);

end

