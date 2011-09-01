function patches = sampleIMAGES
% sampleIMAGES
% Returns 10000 patches for training

load '../data/IMAGES';

patchsize = 8;
numpatches = 10000;

% Initialize patches with zeros
patches = zeros(patchsize*patchsize, numpatches);
%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Fill patches using data from IMAGES
%  IMAGES is a 3D array containing 10 images
%  For instance, IMAGES(21:30,21:30,1) is an image patch corresponding
%  to the pixels in the block (21,21) to (30,30) of Image 1

[ydim, xdim, num_images] = size(IMAGES);

for i = 1:numpatches
  img = randi(num_images);
  y_start = randi(ydim - patchsize + 1);
  x_start = randi(xdim - patchsize + 1);
  patch = IMAGES(y_start:y_start+patchsize-1, x_start:x_start+patchsize-1, img);
  patches(:,i) = patch(:);
end

%% ---------------------------------------------------------------
% For the autoencoder to work well we need to normalize the data
% Specifically, since the output of the network is bounded between [0,1]
% (due to the sigmoid activation function)
% It is important to make sure the input is also bounded between [0,1]
patches = normalizeData(patches);

end

%% ---------------------------------------------------------------
function patches = normalizeData(patches)

% Squash data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer

% Remove DC
patches = bsxfun(@minus, patches, mean(patches));

% Truncate to +/-3 STD and scale to -1 to 1
pstd = 3 * std(patches(:));
patches = max(min(patches, pstd), -pstd) / pstd;

% Rescale to 0.1 to 0.9
patches = (patches + 1) * 0.4 + 0.1;

end
