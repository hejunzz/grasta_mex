%%
% % % % % % % % % % % % % % % % % % % % % %
% Sample code to test grasta_mex: Robust subspace recovery despite outliers and missing data
% % % % % % % % % % % % % % % % % % % % % %
%
% Jun He, June. 2015
%


%% First we generate the data matrix and incomplete sample vector.
close all;
clear all;
clc;

L1_NORM = 1;

outlier_magnitude = 10;

MAX_MU = 10:5:20;

marker_set = ['+', 'o', '*', 'x', 's','d','^','v','>','<','p','h'];

NORM_TYPE = L1_NORM;  % 1 for L1_NORM

% addpath(genpath('/Users/jhe/Documents/MATLAB/myCode/GRASTA/grasta.1.2.0/'))

nMonteCarlo = 10;

for iTest = 1: nMonteCarlo,
    
    outlierFac = 0.1;
    SAMPLING   = 0.5;
    noiseFac   = 1 * 1e-10;
    
    % Number of rows and columns
    numr = 300;
    numc = 300;
    
    probSize = [numr,numc];
    % Rank of the underlying matrix.
    truerank = 5;
    
    % Size of vectorized matrix
    N = numr*numc;
    % Number of samples that we will reveal.
    M = round(SAMPLING * N);
    
    
    YL = randn(numr,truerank);
    YR = randn(numc,truerank);
    
    A = YL*YR';
    
    % Select a random set of M entries of Y.
    p = randperm(N);
    idx = p(1:M);
    clear p;
    
    [I,J] = ind2sub([numr,numc],idx(1:M));
    [J, inxs]=sort(J'); I=I(inxs)';
    
    % S denotes the values of Y at the locations indexed by I and J.
    S = sum(YL(I,:).*YR(J,:),2);
    
    S_true = S;
    
    
    % Add Gaussian noise.
    noise = noiseFac*max(S)*randn(size(S));
    S = S + noise;
    
    % Add sparse outliers
    outlier_magnitude = 10*max(abs(S));
    idx = randperm(length(S));
    sparseIdx = idx(1:ceil(outlierFac*length(S)))';
    Outlier_part = outlier_magnitude * randn(size(sparseIdx));    
    S(sparseIdx) = S(sparseIdx) + Outlier_part;
    
    D = sparse(I,J,S,numr,numc);
    D = full(D);
    missing_idx = find(D==0);
    D(missing_idx) = inf;
    
    D_true = sparse(I,J,S_true,numr,numc);
    D_true = full(D_true);
    %
    % Now we set parameters for the algorithm.
    % We set the number of cycles and put the necessary parameters into OPTIONS
    
    OPTIONS.ADAPTIVE            = 1;
    OPTIONS.STEP_SCALE          = 0.1;%0.25;
    OPTIONS.MAX_STEPSIZE        = 10;
    OPTIONS.maxCycles           = round(10000/numc);    % the max cycles of robust mc
    OPTIONS.QUIET               = 0;     % =1 will suppress the debug information
    
    OPTIONS.MAX_LEVEL           = 50;    % For multi-level step-size,
    OPTIONS.MAX_MU              = 50;     % For multi-level step-size depends on rank
    OPTIONS.MIN_MU              = 0;     % For multi-level step-size
    
    OPTIONS.DIM_M               = numr;  % your data's ambient dimension
    OPTIONS.DIM                 = numr;  % your data's ambient dimension
    
    OPTIONS.RANK                = truerank; % give your estimated rank
    
    OPTIONS.MIN_ITER            = 30;    % the min iteration allowed for ADMM at the beginning
    OPTIONS.MAX_ITER            = 30;    % the max iteration allowed for ADMM
    OPTIONS.rho                 = 2;   % ADMM penalty parameter for acclerated convergence
    OPTIONS.TOL                 = 1e-10;   % ADMM convergence tolerance
    
    OPTIONS.USE_MEX             = 1;     % If you do not have the mex-version of Alg 2
    % please set Use_mex = 0.
    
    OPTIONS.convergeLevel       = 55;    % If status.level >= CONVERGE_LEVLE, robust mc converges
    OPTIONS.NORM_TYPE           = L1_NORM;  
    
    [U0 S0 V0] = svd(YL*YR','econ');
    
    OPTIONS.GT_mat              = U0(:,1:OPTIONS.RANK); %YL*YR';
    
    % % % % % % % % % % % % % % % % % % % % %
    % Now run robust matrix completion.
    if ~OPTIONS.QUIET,
        figure(1); 
    end
    
    axis_step = 500;
    legend_cell = cell(length(MAX_MU)+1,1);
    
    %% Experiments on multi-level adaptive step-size
    for k = 1:length(MAX_MU),                
        OPTIONS.MAX_MU = MAX_MU(k);
        Uhat1 = [nan];
        status1 = struct();
        status1.init = 0;
                
        t1 = tic;
        
        OPTIONS.NORM_TYPE = L1_NORM;
        [Uhat1, ~, ~, status1] = grasta_mex( D, Uhat1, status1, OPTIONS);
        t1 = toc(t1);
        
        len = length(status1.hist_rel);
        semilogy(1:axis_step:len,status1.hist_rel(1:axis_step:end),['r-',marker_set(k)],'MarkerSize',8); hold on;grid on;
        pause(0.1);
        fprintf('GRASTA-Mu-%d --- [Uhat, Utrue]: %.2e, %.4f sec\n', MAX_MU(k),subspace(Uhat1, OPTIONS.GT_mat ), t1);
        legend_cell{k} = ['GRASTA-MU-' num2str(MAX_MU(k))];
    end
    
    %% Diminishing step-size
    OPTIONS.ADAPTIVE            = 0;
    OPTIONS.STEP_SCALE          = 5;
    
    Uhat2 = [nan];
    status2 = struct();
    status2.init = 0;
    
    t1 = tic;
    [Uhat2, W2, Outlier2, status2] = grasta_mex( D, Uhat2, status2, OPTIONS);
    t1 = toc(t1);
    
    len = length(status2.hist_rel);
    semilogy(1:axis_step:len,status2.hist_rel(1:axis_step:end),['r-',marker_set(k+1)],'MarkerSize',8); hold on;grid on;
    fprintf('GRASTA_Diminishing --- [Uhat, Utrue]: %.2e, %.4f sec\n', subspace(Uhat2, OPTIONS.GT_mat ), t1);
    legend_cell{k+1} = 'GRASTA-DM';        

    legend(legend_cell);

    hold off;
    
    title(['Trial-' num2str(iTest)]);
    pause(1);
end
