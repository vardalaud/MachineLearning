function calaculate_filters(W, P, F, S)
% Calculates Size of output after data passes through Conv Layer
result = ((W + (2 * P) - F) / S) + 1 ; 

disp('Size:')
disp(result);

end

