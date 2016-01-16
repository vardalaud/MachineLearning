[trainLabel, trainData] = libsvmread('mnist_scale.train');
[testLabel, testData] = libsvmread('mnist_scale.test');

% Build and test Linear SVM model
model_linear = svmtrain(trainLabel, trainData, '-t 0 -c 0.05 -m 1024 -q');
[predict_label_linear, accuracy_linear, dec_values_linear] = svmpredict(testLabel, testData, model_linear, []);

% Build and test Poly2 SVM model
model_pol2 = svmtrain(trainLabel, trainData, '-t 1 -d 2 -c 0.125 -g 0.125 -r 0.6 -m 1024 -q');
[predict_label_pol2, accuracy_pol2, dec_values_pol2] = svmpredict(testLabel, testData, model_pol2, []);

% Build and test Poly4 SVM model
model_pol4 = svmtrain(trainLabel, trainData, '-t 1 -d 4 -c 0.5 -g 0.03125 -r 1.2 -m 1024 -q');
[predict_label_pol4, accuracy_pol4, dec_values_pol4] = svmpredict(testLabel, testData, model_pol4, []);

% Build and test RBF SVM model
model_rbf = svmtrain(trainLabel, trainData, '-t 2 -c 2 -g 0.03125 -m 1024 -q');
[predict_label_rbf, accuracy_rbf, dec_values_rbf] = svmpredict(testLabel, testData, model_rbf, []);