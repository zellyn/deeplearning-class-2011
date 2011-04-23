x = load('ex3x.dat');
y = load('ex3y.dat');
m = length(y);
x = [ones(m, 1), x];

# figure;
# plot(x(:,2), y, 'x');
# ylabel('Price in $');
# xlabel('Living area in square feet');

# figure;
# plot(x(:,3), y, 'x');
# ylabel('Price in $');
# xlabel('Number of bedrooms');

sigma = std(x);
mu = mean(x);
x(:,2) = (x(:,2) - mu(2)) ./ sigma(2);
x(:,3) = (x(:,3) - mu(3)) ./ sigma(3);

num_iterations = 50;

alphas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 1.3, 3, 10];
J = zeros(num_iterations, length(alphas));

for iter = 1:length(alphas)
  theta = zeros(size(x(1,:)))';
  alpha = alphas(iter);
  for num_iterations = 1:50
    J(num_iterations, iter) = (1/2*m) * (x * theta - y)' * (x * theta - y);
    theta = theta - alpha * (1/m) * x' * (x * theta - y);
  end
end

figure;
hold on;

for iter = 1:length(alphas)-2
  alpha = alphas(iter);

  % now plot J
  % technically, the first J starts at the zero-eth iteration
  % but Matlab/Octave doesn't have a zero index
  plot(0:49, J(1:50, iter), '-');
  xlabel(['Number of iterations, alpha = ' num2str(alpha)]);
  ylabel('Cost J');
end

alpha = 1;
theta = zeros(size(x(1,:)))';
for num_iterations = 1:100
  theta = theta - alpha * (1/m) * x' * (x * theta - y);
end

theta

x1 = ([2, 1650, 3] - mu) ./ (sigma + [1,0,0]);
y_predicted_1 = x1 * theta


% Normal Equations

% Reload data
x = load('ex3x.dat');
y = load('ex3y.dat');
m = length(y);
x = [ones(m, 1), x];

theta = (x' * x) \ (x' * y)

x2 = [1 1650 3];
y_predicted_2 = x2 * theta

