function [log_llh_model,log_llh_diff,lh_model] = llh(FR,Fr_hat)
% compute log-likelihood increase from "mean firing rate model"
% r is predicted Firing Rate 
% n is real Firing Rate
% meanFR is mean real Firing Rate

N = length(FR);
sigma2 = var(FR,"omitnan");
meanFR = mean(FR,"omitnan");

lh_model = (2*pi*sigma2)^(-N/2)*exp(-sum((FR-Fr_hat).^2,"omitnan")/(2*sigma2));
lh_model =lh_model/N;

% keyboard()
% lh_mean = (2*pi*sigma2)^(-N/2)*exp(-sum((FR-meanFR).^2,"omitnan")/(2*sigma2));
% lh_mean =lh_mean/N;


log_llh_model = -0.5*N*log(2*pi*sigma2)-sum((FR-Fr_hat).^2,"omitnan")/(2*sigma2); 
log_llh_model = 60*30.98*log_llh_model/N; % normalize to time (min)

log_llh_mean = -0.5*N*log(2*pi*sigma2)-sum((FR-meanFR).^2,"omitnan")/(2*sigma2); 
log_llh_mean = 60*30.98*log_llh_mean/N; 

log_llh_diff = log_llh_model - log_llh_mean;
% log_llh_diff = log(2)*log_llh_diff;

      
% % Original code
%     log_llh_test_model = nansum(r-n.*log(r)+log(factorial(n)))/sum(n); 
%     % note: log(gamma(n+1)) will be unstable if n is large (which it isn't here)
%     log_llh_test_mean = nansum(meanFR_test-n.*log(meanFR_test)+log(factorial(n)))/sum(n);
%     log_llh_test = (-log_llh_test_model + log_llh_test_mean);
%     log_llh_test = log(2)*log_llh_test;




end