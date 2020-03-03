function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters,
	k = 1:m;
     th1 = sum((theta(1) + theta(2) .* X(k,2)) - y(k)); 
     th2 = sum(((theta(1) + theta(2) .* X(k,2)) - y(k)) .* X(k,2)); 
    
     theta(1) = theta(1) - (alpha/m) * (th1);
     theta(2) = theta(2) - (alpha/m) * (th2);
        
     J_history(iter) = computeCost(X, y, theta);

end

end
