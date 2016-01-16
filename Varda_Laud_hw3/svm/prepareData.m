load('imdb.mat','images')

% Convert the 4D data to 2D
reshapedData = reshape(images.data,[784,20000]);

trainLabel = images.labels(1:10000).';
% Sparcify training data matrix
trainDataSparse = sparse(double(reshapedData(:,1:10000).'));
% Write the training data in SVM format
libsvmwrite('mnist.train', trainLabel, trainDataSparse);

testLabel = images.labels(10001:end).';
% Sparcify testing data matrix
testDataSparse = sparse(double(reshapedData(:,10001:end).'));
% Write the testing data in SVM format
libsvmwrite('mnist.test', testLabel, testDataSparse);

% Run this on command line to scale the data in the range 0-1
%svm-scale -l 0 -s range mnist.train > mnist_scale.train
%svm-scale -r range mnist.test > mnist_scale.test