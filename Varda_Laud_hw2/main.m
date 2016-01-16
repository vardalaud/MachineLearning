% @author Varda Laud
% This funtion is used for testing pla and pseudoinverse functions
function main()
clc
close all
plaAvgIters = zeros(1, 6);
plaAvgTime = zeros(1, 6);
pseudoAvgIters = zeros(1, 6);
pseudoAvgTime = zeros(1, 6);
nIter = 1;
totalTime = 0;
% Run pla & pseudoinverse for each combination of N and trial
for N = [10, 50, 100, 200, 500, 1000]
    op = ['N = ',num2str(N)];
    disp(op)
    
    plaIters = zeros(1, 100);
    pseudoIters = zeros(1, 100);
    plaTime = zeros(1, 100);
    pseudoTime = zeros(1, 100);
    for trial = 1:100
        % Generate the Samples and Labels
        [X, Y] = generateData(N);
        
        % Run pla
        tic;
        [w, iters] = pla(X, Y);
        plaTime(trial) = toc;
        plaIters(trial) = iters;
        
        % Run pla followed by pseudoinverse
        tic;
        [w] = pseudoinverse(X, Y);
        [w, iters] = pla(X, Y, w);
        pseudoTime(trial) = toc;
        pseudoIters(trial) = iters;
    end
    
    plaAvgIters(nIter) = mean(plaIters);
    plaAvgTime(nIter) = mean(plaTime);
    totalTime = totalTime + sum(plaTime);
    op = ['PLA iterations = ', num2str(plaAvgIters(nIter)), ', Avg Time = ', num2str(plaAvgTime(nIter))];
    disp(op)
    
    pseudoAvgIters(nIter) = mean(pseudoIters);
    pseudoAvgTime(nIter) = mean(pseudoTime);
    totalTime = totalTime + sum(pseudoTime);
    op = ['PLA iterations after Linear Regression = ', num2str(pseudoAvgIters(nIter)), ', Avg Time = ', num2str(pseudoAvgTime(nIter))];
    disp(op)
    nIter = nIter + 1;
    
    trial = 1:100;
    figure
    
    % Plot iterations vs trial & time vs trial graph for pla
    ax1 = subplot(2, 1, 1);
    plot(ax1, trial, plaIters, 'b-o', trial, pseudoIters, 'r-s')
    title(ax1, ['PLA iterations for different trials, N = ', num2str(N)])
    xlabel(ax1, 'Trial')
    ylabel(ax1, 'Iteration')
    legend(ax1, 'pla', 'pla after linear regression')
    
    % Plot iterations vs trial & time vs trial graph for pla after
    % pseudoinverse
    ax2 = subplot(2, 1, 2);
    plot(ax2, trial, plaTime, 'b-o', trial, pseudoTime, 'r-s')
    title(ax2, ['PLA execution time for different trials, N = ', num2str(N)])
    xlabel(ax2, 'Trial')
    ylabel(ax2, 'Time')
    legend(ax2, 'pla','pla after linear regression')
end
op = ['Total time taken for execution = ', num2str(totalTime)];
disp(op)

N = [10, 50, 100, 200, 500, 1000];
figure

% Plot iterations vs N & time vs N graph for pla
ax1 = subplot(2, 1, 1);
plot(ax1, N, plaAvgIters, 'b-o', N, pseudoAvgIters, 'r-s')
title(ax1, 'Average PLA iterations for different sample sizes')
xlabel(ax1, 'N')
ylabel(ax1, 'Iteration')
legend(ax1, 'pla', 'pla after linear regression')

% Plot iterations vs N & time vs N graph for pla after pseudoinverse
ax2 = subplot(2, 1, 2);
plot(ax2, N, plaAvgTime, 'b-o', N, pseudoAvgTime, 'r-s')
title(ax2, 'Average PLA execution time for different sample sizes')
xlabel(ax2, 'N')
ylabel(ax2, 'Time')
legend(ax2, 'pla','pla after linear regression')