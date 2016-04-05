function [weights] = logistic_train(data, labels, epsilon, maxiter)
%
% code to train a logistic regression classifier
%
% INPUTS:
% data = n * (d+1) matrix withn samples and d features, where
% column d+1 is all ones (corresponding to the intercept term)
% labels = n * 1 vector of class labels (taking values 0 or 1)
% epsilon = optional argument specifying the convergence
% criterion - if the change in the absolute difference in
% predictions, from one iteration to the next, averaged across
% input features, is less than epsilon, then halt
% (if unspecified, use a default value of 1e-5)
% maxiter = optional argument that specifies the maximum number of
% iterations to execute (useful when debugging in case your
% code is not converging correctly!)
% (if unspecified can be set to 1000)
%
% OUTPUT:
% weights = (d+1) * 1 vector of weights where the weights correspond to
% the columns of "data"
%

% read parameters
if nargin < 2
    fprintf('Need at least two parameters!');
    return;
end
if nargin < 3
    epsilon = 1e-5;
end
if nargin < 4
    maxiter = 1000;
end

n = size(data,2);
% initialize weights
weights_old = zeros(n,1);
y_old = logsig(data*weights_old);
iter = 0;
% using Newton-Raphson (IRLS) iterative procedure
while iter < maxiter
    % avoid the issue y(1-y) become 0, which lead to sigular matrix H
    if sum(y_old==0)>0 || sum(y_old==1)>0
        break;
    end
    R = diag(y_old.*(1-y_old));
    dE = data'*(y_old-labels);
    H = data'*R*data;
    weights = weights_old-H\dE;
    y = logsig(data*weights);
    % convergence criterion
    if mean(abs(y-y_old)) < epsilon
        break;
    end
    y_old = y;
    weights_old = weights;
    iter = iter+1;
end

end