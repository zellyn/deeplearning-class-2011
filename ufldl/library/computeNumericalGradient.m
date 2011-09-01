function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: is a vector of parameters
% J:  is a scalar real-valued function; calling J(theta) will return the
% function value at theta

% Initialize numgrad with zeros
numgrad = zeros(size(theta));

epsilon = 1e-4;

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Implement numerical gradient checking for your code
%                to compute numgrad (gradient from finite differences)
%                (Check Section 2.3 in the lecture notes)
%
% Hint: You may want to compute each element of numgrad at a time.

for i = 1:numel(theta)
  theta_minus = theta;
  theta_plus = theta;
  theta_minus(i) = theta_minus(i) - epsilon;
  theta_plus(i) = theta_plus(i) + epsilon;
  numgrad(i) = (J(theta_plus) - J(theta_minus)) / (2*epsilon);
end

%% ---------------------------------------------------------------
end