% @author Varda Laud
% This funtion is used for testing testknn and condensedata functions
function driverMethod()
clc
inputData = readtable('letter-recognition.csv', 'ReadVariableNames', false);
% Read 1st 15000 samples as training set
trainX = inputData{1:15000, 2:end};
% Read last 5000 samples as testing set
trainY = cell2mat(inputData{1:15000, 1:1});
testX = inputData{15001:end, 2:end};
expectedTestY = cell2mat(inputData{15001:end, 1:1});
disp('***** Running k-NN Experiments *****')
% Run knn for each combination of k and N
for k = 1:2:9
    for N = [100, 1000, 2000, 5000, 10000, 15000]
        % Randomly sample N data samples from trainX without replacement
        [sampledTrainX, sampledIndexes] = datasample(trainX, N, 'Replace', false);
        % Build the sampledTrainY vector based on the indices of samples in
        % trainX
        sampledTrainY = trainY(sampledIndexes);
        
        testY = testknn(sampledTrainX, sampledTrainY, testX, k);
        
        % Compare predicted and actual labels and compute accuracy
        correctPredictions = sum(testY == expectedTestY);
        accuracy = correctPredictions/5000;
        op = ['k = ', num2str(k), ', N = ', num2str(N), ', Accuracy = ',num2str(accuracy)];
        disp(op)
    end
end
disp('***** Running condensed 1-NN Experiments *****')
for N = [100, 1000, 2000, 5000, 10000, 15000]
    [sampledTrainX, sampledIndexes] = datasample(trainX, N, 'Replace', false);
    sampledTrainY = trainY(sampledIndexes);
    % Passed the sampled training data to condensedata
    condensedIdx = condensedata(sampledTrainX, sampledTrainY);
    condensedTrainX = trainX(condensedIdx, :);
    condensedTrainY = trainY(condensedIdx);
    for k = 1:2:9
        % Run testknn using the condensed training set
        testY = testknn(condensedTrainX, condensedTrainY, testX, k);
        correctPredictions = sum(testY == expectedTestY);
        accuracy = correctPredictions/5000;
        op = ['k = ', num2str(k), ', N = ', num2str(N), ', |condensedIdx| = ', num2str(size(condensedIdx, 1)), ', Accuracy = ',num2str(accuracy)];
        disp(op)
    end
end