% @author Varda Laud
% This funtion finds the weights using Linear Regression method
% X: N*2 vector of points in coordinate plane
% Y: N*1 vector of labels +1 or -1
% w: Weight vector of dimension 3*1
function [w] = pseudoinverse(X, Y)
if size(X, 2) ~= 2 || size(Y, 2) ~= 1 || size(X, 1) ~= size(Y, 1) || size(X, 1) < 1 || size(Y, 1) < 1
    error('Error in Input Arguments')
end
% Set the 1st column values = 1 which are the bias terms. X is now a N*3 matrix
X = [ones(size(X, 1), 1), X];
% Get pseudoinverse of X
Xdagger = pinv(X);
% Get the weight vector w
w = Xdagger * Y;