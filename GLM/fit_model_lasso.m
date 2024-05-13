function [testFit,trainFit,param_mean,mdls,paramMat] = ...
    fit_model_lasso(A,spiketrain,numFolds)


% A is the matrix of the dependent varibles
[~,numCol] = size(A);

% Divide the data up into n=num_folds sections
sections = numFolds;
edges = round(linspace(1,numel(spiketrain)+1,sections+1));

% Initialize matrices
testFit = nan(numFolds,6); 
% var ex, correlation, llh increase, mse
trainFit = nan(numFolds,6); 
% var ex, correlation, llh increase, mse
paramMat = nan(numFolds,numCol);
mdls=cell(1,numFolds);

% Compute a filter, which will be used to smooth the dff hat
    filter = gaussmf(-4:4,[8 0]); filter = filter/sum(filter);


% Perform k-fold Cross Validation
for k = 1:numFolds
    % fprintf('\t\t- Cross validation fold %d of %d\n', k, numFolds);
    
    test_ind = edges(k):(edges(k+1)-1);
    test_spikes = spiketrain(test_ind);
    test_A = A(test_ind,:);

    train_ind = setdiff(1:numel(spiketrain),test_ind);
    train_spikes = spiketrain(train_ind);
    train_A = A(train_ind,:);

    %%%%%%%%%%%%%%% Fitting Mdoel (Gaussian lasso) %%%%%%%%%%%%%%%%%%%%%%
       [B,FitInfo] = lassoglm(train_A, train_spikes,'normal','Alpha',1);
%        BestLamda = find(FitInfo.MSE == min(FitInfo.MSE),1,'last');
        BestLamda = find(FitInfo.Deviance == min(FitInfo.Deviance),1,'first');
       param = B(:,BestLamda);
       FitInfo.BestLamda=BestLamda;
       mdls{k}=FitInfo;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    

    %%%%%%% Compute Predicted Firing Rates and inspect model %%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%% TEST DATA %%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute the predicted firing rate (smoothed)
    fr_hat_test = test_A*param;
    fr_hat_test = conv(fr_hat_test,filter,'same');

    
    % Compute varExplain (smoothed)
    sse = sum((fr_hat_test-test_spikes).^2,"omitnan");
    sst = sum((test_spikes-mean(test_spikes,"omitnan")).^2,"omitnan");
    varExplain_test = 1-(sse/sst);

    % Compute Pearson Correlation (smoothed)
    correlation_test = corr(test_spikes,fr_hat_test,'type','Pearson');
    
    % Compute log-likelihood increase from "mean firing rate model" 
    [log_llh_test,log_llh_diff_test,lh_test] = llh(test_spikes,fr_hat_test);
 
    % Compute MSE (smoothed)
    mse_test = mean((fr_hat_test-test_spikes).^2,"omitnan");
    
    % Fill in all the relevant values for the test fit cases
    testFit(k,:) = [varExplain_test correlation_test...
        log_llh_test log_llh_diff_test mse_test lh_test];


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%% TRAINING DATA %%%%%%%%%%%%%%%%%%%%%%%
    % compute the firing rate (smoothed)
    fr_hat_train = train_A*param;
    fr_hat_train = conv(fr_hat_train,filter,'same');
    

    % Compute varExplain (smoothed)
    sse = sum((fr_hat_train-train_spikes).^2,"omitnan");
    sst = sum((train_spikes-mean(train_spikes,"omitnan")).^2,"omitnan");
    varExplain_train = 1-(sse/sst);
  
    
    % Compute Pearson Correlation (smoothed)
    correlation_train = corr(train_spikes,fr_hat_train,'type','Pearson');
    
    % Compute log-likelihood increase from "mean firing rate model"  
    [log_llh_train,log_llh_diff_train,lh_train] = llh(train_spikes,fr_hat_train);
   
    % Compute MSE (smoothed)
    mse_train = mean((fr_hat_train-train_spikes).^2,"omitnan");
    
    % Fill in all the relevant values for the train fit cases
    trainFit(k,:) = [varExplain_train correlation_train...
        log_llh_train log_llh_diff_train mse_train lh_train];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%% Save the parameters %%%%%%%%%%%%%%%%%%%%%%
    paramMat(k,:) = param;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

% Calculate mean parapeters from k-fold Cross Validation
param_mean = mean(paramMat,"omitnan");

end
