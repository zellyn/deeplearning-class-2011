x = load('ex4x.dat');
y = load('ex4y.dat');
[m, n] = size(x);
x = [ones(m, 1), x];
[m, n] = size(x);

figure;
hold on;
plot(x(y==1, 2), x(y==1, 3), '+')
plot(x(y!=1, 2), x(y!=1, 3), 'o')
xlabel('Exam 1 score')
ylabel('Exam 2 score')

g = inline('1.0 ./ (1.0 + exp(-z))')

theta = zeros(n, 1);
diff = ones(n, 1);
Js = []

while norm(diff) > 1e-6
  h = g(x * theta);
  Js(length(Js)+1) = (1/m) * sum(-y.*log(h) - (1-y).*log(1 - h));
  grad = (1/m) * x' * (h - y);
  H = (1/m) * (repmat(h .* (1 - h), 1, n) .* x)' * x;
  diff = H \ grad;
  theta = theta - diff;
end

iterations = length(Js)
theta
prob = 1 - g([1, 20, 80] * theta)

plot_x = [min(x(:,2))-2, max(x(:,2))+2];
plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
plot(plot_x, plot_y);
legend('Admitted', 'Not admitted', 'Decision boundary')
hold off;

figure;
plot(0:iterations-1, Js, 'o--', 'MarkerFaceColor', 'r', 'MarkerSize', 8);
ylabel('J(\theta)');
xlabel('Iteration')
