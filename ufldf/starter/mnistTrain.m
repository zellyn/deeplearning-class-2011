% MNIST training

images = loadMNISTImages('data/train-images-idx3-ubyte');  % 784 x 60000
labels = loadMNISTLabels('data/train-labels-idx1-ubyte');  % 60000 x 1

display_network(images(:,1:100)); % Show the first 100 images
disp(labels(1:10));
