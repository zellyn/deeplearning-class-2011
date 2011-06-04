%%
%% Matlab version of dist_belief exercise for UFLDF class.
%%

addpath '../library/'
addpath '../library/minFunc/'

imageChannels = 3;     % number of channels (rgb, so 3)
load '../data/stl10_matlab/stlTrainSubset.mat' % loads numTrainImages, trainImages, trainLabels
