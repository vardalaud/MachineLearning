% @author Varda Laud
% This funtion computes the row numbers of condensed training set from
% input training set using condensed 1-NN algorithm.
% trainX - training samples
% trainY - training sample labels
% condensedIdx - row numbers of condensed set of training samples
function [condensedIdx] = condensedata(trainX, trainY)
disp('--- Running condensed 1-NN Function ---')
if size(trainX, 1) < 1 || size(trainY, 2) ~= 1 || size(trainX, 1) ~= size(trainY, 1)
    error('Error in Input Arguments')
end

tic

% Create masterDistanceMatrix which is the euclidean distance between all
% pairs of trainX samples
testRowIndex = trainX(1:end, :);
trainRowIndex = trainX(1:end, :);
masterDistanceMatrix = pdist2(testRowIndex, trainRowIndex);

nTrain = size(trainX, 1);

% Preallocate condensedIdx to the size of trainX which is the max size it
% can get. This is done to improve efficiency. condensedIdx keeps track of
% condensed set of rows from trainX.
condensedIdx = zeros(1, nTrain);
% Keeps track of current index in condensedIdx
i=0;

% Initialize nearestNeighborDistance to Infinite for all trainX rows. This
% keeps track of current distance from nearest neighbor of a row.
nearestNeighborDistance = Inf(1, nTrain);
% Initialize nearestNeighborIndex to zeros for all trainX rows. This keeps
% track of index in trainX of nearest neighbor of a row
nearestNeighborIndex = condensedIdx;

% This keeps track of wrongly classified samples in each iteration.
wronglyClassifiedSamples = 1:nTrain;

% This keeps track of samples not in condensedIdx in each iteration.
notClassifiedSamples = 1:nTrain;

% Continue till either the wronglyClassifiedSamples gets empty or all the
% training samples are put in condensedIdx
while ~isempty(wronglyClassifiedSamples) && i < nTrain
    
    % Pick up a sample randomly from wronglyClassifiedSamples
    wronglyClassifiedRowNumber = datasample(wronglyClassifiedSamples, 1);
    
    % Increment the current index in condensedIdx and add the newly sampled
    % wrongly classified row number to it
    i = i + 1;
    condensedIdx(i) = wronglyClassifiedRowNumber;
    
    % Remove the newly sampled wrongly classified row number from
    % notClassifiedSamples
    notClassifiedSamples(notClassifiedSamples == wronglyClassifiedRowNumber) = [];
    
    % Used as an index to traverse notClassifiedSamples
    testRowIndex = notClassifiedSamples(1, :);
    
    % Compare the current nearest neighbor distance to the newly added
    % training data in condensedIdx. If the former is greater set
    % distanceComparisonResult to 1 else 0.
    distanceComparisonResult = nearestNeighborDistance(testRowIndex) > masterDistanceMatrix(condensedIdx(i), testRowIndex);
    
    % Update the distances for distanceComparisonResult = 1
    nearestNeighborDistance(notClassifiedSamples(distanceComparisonResult)) = masterDistanceMatrix(condensedIdx(i), notClassifiedSamples(distanceComparisonResult));
    % Update the index of the new nearest neighbor for
    % distanceComparisonResult = 1
    nearestNeighborIndex(notClassifiedSamples(distanceComparisonResult)) = i;
    
    % Find the predictedLabels for all notClassifiedSamples which is the
    % label of nearestNeighborIndex row
    predictedLabel = trainY(condensedIdx(nearestNeighborIndex(testRowIndex)));
    
    % Create the comparisonResult vector comparing predicted and actual
    % labels. Set 1 if lables not matching else 0.
    comparisonResult = predictedLabel ~= trainY(testRowIndex);
    
    % Add all row numbers for which lables do not match (comparisonResult
    % having value one) from notClassifiedSamples to
    % wronglyClassifiedSamples.
    wronglyClassifiedSamples = notClassifiedSamples(comparisonResult);
end

condensedIdx = nonzeros(condensedIdx);
toc