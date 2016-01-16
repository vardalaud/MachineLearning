% @author Varda Laud
% This funtion is used to see the difference between the lines plotted with 
% the weights generated by pla and pla after linear regression functions
function plotScatter()
figure
N = 1000;
[X, Y] = generateData(N);
i = 1:N;
positive = Y(i)==+1;
positiveIndices = find(positive);
negativeIndices = setdiff(i,positiveIndices);

scatter(X(positiveIndices,1),X(positiveIndices,2), 'g.')
hold on
scatter(X(negativeIndices,1),X(negativeIndices,2), 'r.')
hold off

[w, ~] = pla(X, Y);
plaXintersect = -(w(1)/w(2));
plaYintersect = -(w(1)/w(3));
plaLine = [plaXintersect,0;0,plaYintersect];

[w] = pseudoinverse(X, Y);
pseudoXintersect = -(w(1)/w(2));
pseudoYintersect = -(w(1)/w(3));
pseudoLine = [pseudoXintersect,0;0,pseudoYintersect];

hold on
plot(plaLine(1:2,1),plaLine(1:2,2), 'b', pseudoLine(1:2,1),pseudoLine(1:2,2), 'r')
hold off