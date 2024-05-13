function [all_mdl,sig_mdl] = SelectBestModels(testFit) 


% Select all the log likelihood values
log_llh_diff_test = cell(0);


for i=1:size(testFit,1) % go over cells
    for j = 1:size(testFit,2) % go over models
        log_llh_diff_test{i,j} = testFit{i,j}(:,4); 
    end
end

selected_model = [];
mdl_max_llh=[];
sig_mdl =[];all_mdl=[];
singleModels = 5:7;

for i=1:size(testFit,1) % num of cells

% find the best single model
[~,top1] = max(mean(cell2mat(log_llh_diff_test(i,singleModels)),"omitnan")); 
top1 = top1 + singleModels(1)-1;

% find the best double model that includes the single model
if top1 == 5 % Time -> Pos & Time, lick & Time
    [~,top2] = max(mean(cell2mat(log_llh_diff_test(i,[2,3])),"omitnan")); 
    vec = [2,3]; top2 = vec(top2);
elseif top1 == 6 % pos -> pos & Time, Pos & lick
    [~,top2] = max(mean(cell2mat(log_llh_diff_test(i,[2,4])),"omitnan")); 
    vec = [2,4]; top2 = vec(top2);
elseif top1 == 7 % lick -> lick & Time, lick & pos
    [~,top2] = max(mean(cell2mat(log_llh_diff_test(i,[3,4])),"omitnan")); 
    vec = [3,4]; top2 = vec(top2);
   
else 
    keyboard
end

top3 = 1; % full model

% Find the best fit model
LLH1 = cell2mat(log_llh_diff_test(i,top1));
LLH2 = cell2mat(log_llh_diff_test(i,top2));
LLH3 = cell2mat(log_llh_diff_test(i,top3));

% if sum(isnan(LLH1))==5 %||  sum(isnan(LLH2))==5 || sum(isnan(LLH3))==5
%     selected_model(i) = nan;
%     selected_LLH(i) = nan;selected_LLH_diff(i)=nan;best_LLH_diff(i)=nan;
% else
    [p_llh_12,~] = signrank(LLH2,LLH1,'tail','right');
    [p_llh_23,~] = signrank(LLH3,LLH2,'tail','right');
    
    P=0.05;
    if p_llh_12 < P % double model is sig. better
             if p_llh_23 < P  % full model is sig. better 
                 selected_model(i) = top3; % full model
             else
             selected_model(i) = top2; % double model
             end    
    else
        selected_model(i) = top1; % single model
    end

% Save the mean LLH of the selected model
selected_LLH_diff(i) = mean(cell2mat(log_llh_diff_test(i,selected_model(i))),"omitnan");


% Find mdl with largest LLH
[best_LLH_diff(i) ,mdl_max_llh(i)] = max(mean(cell2mat(log_llh_diff_test(i,:)),"omitnan")); 

% end


if selected_LLH_diff(i) > 0 
    sig_mdl = [ sig_mdl;i,best_LLH_diff(i),mdl_max_llh(i),selected_LLH_diff(i),selected_model(i)];
end
end

all_mdl = [selected_LLH_diff' selected_model' ];
 
% re-set if selected model is not above baseline
% pval_baseline(i) = ...
%     signrank(cell2mat(log_llh_diff_test(i,mdl_max_llh(i))),0,'tail','right');
% end

% if pval_baseline(i) < 0.1
%    sig_mdl = [ sig_mdl;i,best_LLH_diff(i),mdl_max_llh(i),selected_LLH_diff(i),selected_model(i)];
% end



end