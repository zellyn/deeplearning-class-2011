function checkNumericalGradient()

addpath '../library/'

for i = 1:100
  x = rand([2,1]) * 10 - 5;
  grad = simple_grad(x);
  numgrad = computeNumericalGradient(@simple_func, x);
  diff = norm(numgrad-grad)/norm(numgrad+grad);
  if (diff > 1e-4)
    disp('Problem with numerical gradient!')
    disp([x numgrad grad]);
  end
end

% Use this to eyeball the gradients
disp([numgrad grad]);

endfunction

function y = simple_func(x)
  y = x(1)^2 + 3 * x(1) * x(2);
endfunction

function dy = simple_grad(x)
  dy = [2 * x(1) + 3 * x(2); 3 * x(1)];
endfunction
