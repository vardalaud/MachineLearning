[trainLabel, trainData] = libsvmread('mnist_scale.train');
bestcv = 0;
% Run 5-fold cross validation using log2c values in the range -1 to 3 and log2g value in the range -4 to 1
for log2c = -1:3,
    for log2g = -4:1,
        cmd = ['-v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g), ' -m 1024 -q'];
        cv = svmtrain(trainLabel, trainData, cmd);
        if (cv >= bestcv),
            bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
        end
        fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
    end
end