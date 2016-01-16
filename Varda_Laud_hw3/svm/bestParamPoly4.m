[trainLabel, trainData] = libsvmread('mnist_scale.train');
bestcv = 0;
% Run 5-fold cross validation using log2c values in the range -5 to 15
% and log2g value in the range -15 to 3 both with step size 2
for log2c = -5:2:15,
    for log2g = -15:2:3,
        cmd = ['-v 5 -t 1 -d 4 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g), ' -m 1024 -q'];
        cv = svmtrain(trainLabel, trainData, cmd);
        if (cv >= bestcv),
            bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
        end
        fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
    end
end