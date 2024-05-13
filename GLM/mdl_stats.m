function [varExplain,correlation,log_llh,log_llh_diff,mse] = mdl_stats(Fit)

varExplain = [];
correlation =[];
log_llh=[];
log_llh_diff = [];
mse=[];
for k=1:3
for   i=1:size(Fit{k},1) % go over cells
    for j = 1:size(Fit{k},2) % go over models
        % Get a kx5 matrix for k-fold cross validation and 5 parameters 
        % whitch are 
        % varExplain, correlation,log_llh, log_llh_diff, MSE

        varExplain{k}(i,j) = mean(Fit{k}{i,j}(:,1),"omitnan"); 
        correlation{k}(i,j) = mean(Fit{k}{i,j}(:,2),"omitnan");
        log_llh{k}(i,j) = mean(Fit{k}{i,j}(:,3),"omitnan");
        log_llh_diff{k}(i,j) = mean(Fit{k}{i,j}(:,4),"omitnan");
        mse{k}(i,j) = mean(Fit{k}{i,j}(:,5),"omitnan");

      
   
    end
end
end







end