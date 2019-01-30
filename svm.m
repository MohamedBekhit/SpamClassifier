%% Initialization
clear ; close all; clc

%% =============== Part 1: Loading and Visualizing Data ================

fprintf('Loading and Visualizing Data ...\n')

% You will have X, y in your environment
load('data1.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ==================== Part 2: Training Linear SVM ====================

% You will have X, y in your environment
load('data1.mat');

fprintf('\nTraining Linear SVM ...\n')

C = 100;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;
% 
%% =============== Part 3: Implementing Gaussian Kernel ===============

fprintf('\nEvaluating the Gaussian Kernel ...\n')

x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);

fprintf(['Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f :' ...
         '\n\t%f\n'], sigma, sim);
fprintf('Program paused. Press enter to continue.\n');
pause;


%% =============== Part 4: Visualizing Dataset 2 ================

fprintf('Loading and Visualizing Data ...\n')

% Load from data2: 
load('data2.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========

fprintf('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');


load('data2.mat');

% SVM Parameters
C = 1; sigma = 0.1;

% the tolerance and max_passes are set lower here so that the code will run
% faster. However, in practice, it is prefered to run the training to
% convergence.
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
visualizeBoundary(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =============== Part 6: Visualizing Dataset 3 ================

fprintf('Loading and Visualizing Data ...\n')


load('data3.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

load('data3.mat');

% Try different SVM Parameters here
[C, sigma] = dataset3Params(X, y, Xval, yval);
fprintf('C = %f\tsigma = %f\n', C, sigma);
% Train the SVM
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;

