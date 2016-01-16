[trainLabel, trainData] = libsvmread('mnist_scale.train');
bestcv = 0;
% Run 5-fold cross validation on log2c values in the range -1 to 3
for log2c = -1:3,
    cmd = ['-v 5 -t 0 -c ', num2str(2^log2c), ' -m 1024 -q'];
    cv = svmtrain(trainLabel, trainData, cmd);
    if (cv >= bestcv),
        bestcv = cv; bestc = 2^log2c;
    end
    fprintf('%g %g (best c=%g, rate=%g)\n', log2c, cv, bestc, bestcv);
end