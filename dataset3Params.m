function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

test_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

ind = 1;
error = 100 * ones(length(test_vec)^2, 1);
for i = 1:length(test_vec)
    C = test_vec(i);
    for j = 1:length(test_vec)
       sigma = test_vec(j);
       model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
       predictions = svmPredict(model, Xval);
       error(ind) = mean(double(predictions ~= yval));
       if min(error) == error(ind)
           C_ind = i;
           sigma_ind = j;
       end
       ind = ind + 1;
    end
end
C = test_vec(C_ind);
sigma = test_vec(sigma_ind);
pause;
% results = eye(64,3);
% errorRow = 0;
% 
% for C_test = [0.01 0.03 0.1 0.3 1, 3, 10 30]
%     for sigma_test = [0.01 0.03 0.1 0.3 1, 3, 10 30]
%         errorRow = errorRow + 1;
%         model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
%         predictions = svmPredict(model, Xval);
%         prediction_error = mean(double(predictions ~= yval));
% 
%         results(errorRow,:) = [C_test, sigma_test, prediction_error];     
%     end
% end
% 
% sorted_results = sortrows(results, 3); % sort matrix by column #3, the error, ascending
% 
% C = sorted_results(1,1);
% sigma = sorted_results(1,2);


end
