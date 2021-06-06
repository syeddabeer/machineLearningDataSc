function [MSE] = LRError(ytrue,PredictionMatrix)

%fprintf('outputting from the LRError Function');
%fprintf('%9.4f\n', [PredictionMatrix(1:2,:)]);
%fprintf('%9.4f\n', [ytrue(1:2,:)]);

error = PredictionMatrix - ytrue;
squareerror = error.^2;
MSE = (sum(squareerror(:)))/(size(squareerror,1));

end
