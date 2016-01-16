% @author Varda Laud
% This funtion predicts the labels for testing set using k-NN
% classification technique
% trainX - training samples
% testX - testing samples
% trainY - training sample labels
% k - number of nearest neighbors to consider
% testY - predicted labels of testing samples
function [testY] = testknn(trainX, trainY, testX, k)
disp('--- Running k-NN Function ---')
if mod(k, 2) == 0 || size(trainX, 1) < 1 || size(testX, 1) < 1 || size(trainY, 2) ~= 1 || size(trainX, 1) ~= size(trainY, 1) || size(trainX, 2) ~= size(testX, 2)
    error('Error in Input Arguments')
end

tic

% Find Euclidean distance between tuples in testing and training set.
testSamples = testX(1:end, :);
trainSamples = trainX(1:end, :);
% Rows of distanceMatrix are samples in testX (testing set), Columns are
% samples in trainX (training set) and the values are Euclidean distance
% between the two.
distanceMatrix = pdist2(testSamples, trainSamples);

% Sort the distanceMatrix row wise
[~, trainXIndices] = sort(distanceMatrix, 2);
% Get the labels of k trainX samples for which the distance is minimum
neighbors = trainY(trainXIndices(:, 1:k));
% The maximum occuring labels is set as the labels of testing data.
testY = mode(neighbors, 2);

toc