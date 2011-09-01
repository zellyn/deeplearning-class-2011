visible_size = 4;
hidden_size = 2;
rho = 0.01;
lamb = 0.0001;
beta = 3;

W1 = [9, 3, 6, -1; 7, 5, 3, 2];
W2 = [0.2, -0.9; -0.3, 1.1; 0.5, 0.6; 0.7, -0.4];
b1 = [-0.7; -0.5];
b2 = [0.2; 1.3; -0.7; 0.6];

theta = [W1(:); W2(:); b1(:); b2(:)];
data = [0.2, -0.7,  0.8, -0.1, -0.8; 0.3,  0.4, -0.7,  0.2, -0.9; 0.1, -0.3, -0.6,  1.0,  0.7; 0.8,  0.5, -0.7, -0.9, -0.7 ];

[loss, grad] = sparseAutoencoderLoss(theta, visible_size, hidden_size, lamb, rho, beta, data)
