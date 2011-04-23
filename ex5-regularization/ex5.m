x = load('ex5Linx.dat');
y = load('ex5Liny.dat');
[m, n] = size(x);
x = [ones(m, 1), x, x.^2, x.^3, x.^4, x.^5];
[m, n] = size(x);


L = diag([0, ones(1,n-1)]);

theta_0 = (x' * x) \ (x' * y);
theta_1 = (x' * x + L) \ (x' * y);
theta_10 = (x' * x + 10 * L) \ (x' * y);

xs = linspace(-1, 1, 100)';
xs = [xs.^0, xs, xs.^2, xs.^3, xs.^4, xs.^5];

figure;
hold on;
plot(x(:,2),y, 'o');
plot(xs(:,2), xs * theta_0, 'r--')
plot(xs(:,2), xs * theta_1, 'g--')
plot(xs(:,2), xs * theta_10, 'b--')
legend('Training data', 'lambda=0', 'lambda=1', 'lambda=10')
hold off;


% Logistic Regression %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all;

x = load('ex5Logx.dat');
u = x(:,1); v = x(:,2);
y = load('ex5Logy.dat');

g = inline('1.0 ./ (1.0 + exp(-z))')

X = map_feature(u, v);
[m, n] = size(X);

eye_0 = diag([0; ones(n-1,1)]);

lambdas = [0, 1, 10];

thetas = [];
for i = 1:length(lambdas)
  theta = zeros(n, 1);
  lambda = lambdas(i);
  Js = [];
  diff = ones(n, 1);
  while norm(diff) > 1e-6
    h = g(X * theta);
    theta_0 = [0; theta(2:end,1)];
    J = -(1/m) * sum(y.*log(h) + (1-y).*log(1 - h)) + (lambda/2*m) * theta_0' * theta_0;
    Js = [Js, J];
    grad = ((1/m) * X' * (h - y)) + ((lambda/m) * theta_0);
    H = ((1/m) * (repmat(h .* (1 - h), 1, n) .* X)' * X) + ((lambda/m) * eye_0);
    diff = H \ grad;
    theta = theta - diff;
  end
  iterations = length(Js)
  thetas = [thetas, theta];
end

% Define the ranges of the grid
uu = linspace(-1, 1.5, 200);
vv = linspace(-1, 1.5, 200);

% Initialize space for the values to be plotted
z0 = zeros(length(uu), length(vv));
z1 = zeros(length(uu), length(vv));
z10 = zeros(length(uu), length(vv));

% Evaluate z = theta*x over the grid
for i = 1:length(uu)
    for j = 1:length(vv)
        % Notice the order of j, i here!
        z0(j,i) = map_feature(uu(i), vv(j))*thetas(:,1);
        z1(j,i) = map_feature(uu(i), vv(j))*thetas(:,2);
        z10(j,i) = map_feature(uu(i), vv(j))*thetas(:,3);
    end
end

% Because of the way that contour plotting works
% in Matlab, we need to transpose z, or
% else the axis orientation will be flipped!
%
% But not in octave...

if (exist ('octave_config_info') ~= 0)
  % this is octave
else
  % this is not octave
  z0 = z0';
  z1 = z1';
  z10 = z10';
end

figure;
hold on;
plot(x(y==0,1), x(y==0,2), 'o');
plot(x(y==1,1), x(y==1,2), '+');

% Plot z = 0 by specifying the range [0, 0]
contour(uu,vv,z0, [0, 0], 'LineWidth', 2, 'r-');
contour(uu,vv,z1, [0, 0], 'LineWidth', 2, 'g-');
contour(uu,vv,z10, [0, 0], 'LineWidth', 2, 'b-');

xlabel('u'); ylabel('v');

legend('y = 0', 'y = 1', 'lambda=0', 'lambda=1', 'lambda=10');
