% @author Varda Laud
% This funtion generates N points in interval (-1,1)*(-1,1) and labels them
% positive or negative according to their position above or below a
% randomly picked up line in that plane
% N: Number of points in coordinate plane to generate
% X: N*2 vector of points in coordinate plane [?1, 1] × [?1, 1]
% Y: N*1 vector of labels +1 or -1

function [X, Y] = generateData(N)
if N < 1
    error('Error in Input Arguments')
end

% Generate a N-by-2 vector as N points in the interval(-1,1)
X = -1 + (1 + 1) * rand(N, 2);
% Generate a 1-by-2 vector as point A in the interval(-1,1)
A = -1 + (1 + 1) * rand(1, 2);
% Generate a 1-by-2 vector as point B in the interval(-1,1)
B = -1 + (1 + 1) * rand(1, 2);
% Find if the N points lie above or below the line passing through points
% A and B
Y = sign((B(1, 1) - A(1, 1)) * (X(1:N, 2) - A(1, 2)) - (B(1, 2) - A(1, 2)) * (X(1:N, 1) - A(1, 1)));