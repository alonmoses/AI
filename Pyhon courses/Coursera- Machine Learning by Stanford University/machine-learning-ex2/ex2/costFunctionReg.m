function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

hTheta2 = sigmoid(X * theta);
firtExp = y .* log(hTheta2);
secondExp = (1-y) .* log(1-hTheta2);
regExp = theta(2:length(theta))' * theta(2:length(theta)) * (lambda/(2*m));
J = sum(-firtExp -secondExp)/m + regExp;

inerSub = hTheta2 - y;
dJ = X' * inerSub / m;
grad = dJ + theta * lambda/m;
grad(1) = (1/m) * inerSub' * ones(m,1);

% =============================================================

end
