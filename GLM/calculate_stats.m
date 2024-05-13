function [sample_size,spike_count,spike_count_ratio ,...
    r2_good,corr_good,log_llh_good,log_llh_diff_good,mse_good,...
    r2_adj] = ...
    calculate_stats(testFit,FiringRate,trial_count,good_cell,d)

% Calculate num of data points
sample_size = [size(FiringRate{1},1),size(FiringRate{2},1),size(FiringRate{3},1)];

% Calculate spike counts
spike_count=[];
for trial=1:3
    for cellID = 1:size(FiringRate{trial},2)
        FR=[];
        FR = FiringRate{trial}(:,cellID);
        spike_count(cellID,trial) = max(bwlabel(FR));
    end
end

% Select cells based on spike count
spike_count_ratio = spike_count./trial_count;
% good_spike_cells=cell(0);
% for  trial=1:3
%    good_spike_cells{trial} = find(spike_count_ratio(:,trial)>1.5);
% end

% Get other results
[r2,corr,log_llh,log_llh_diff,mse] = mdl_stats(testFit);

for k=1:3 % go over trial types
if ~isempty(good_cell{k})
    for cells=1:length(good_cell{k}) % go over cells
        i = good_cell{k}(cells);
    
        r2_good{k}(cells,:)= r2{k}(i,:);
        corr_good{k}(cells,:)= corr{k}(i,:);
        mse_good{k}(cells,:)= mse{k}(i,:);
        log_llh_good{k}(cells,:) = log_llh{k}(i,:);
        log_llh_diff_good{k}(cells,:) = log_llh_diff{k}(i,:);    
    end
else 
    r2_good{k}= NaN(1,7);
    corr_good{k}= NaN(1,7);
    mse_good{k}= NaN(1,7);
    log_llh_good{k}= NaN(1,7);
    log_llh_diff_good{k}= NaN(1,7); 
end
end

% Calculate adjusted r2
r2_adj=cell(0);
for k=1:3 % go over trial types
    if ~isnan(r2_good{k})
        for mdl_num = 1:7 % go over mdls
        r2_col = r2_good{k}(:,mdl_num);
        r2_adj{k}(:,mdl_num) = 1-(1-r2_col)*(sample_size(k)-1)/(sample_size(k)-d(mdl_num)-1);
        end
    else 
        r2_adj{k}=NaN(1,7);
    end

end

% Calculate llh_base2
% log2_llh=cell(0);
% for k=1:3 % go over trial types
%     log2_llh{k}=log2(exp(sample_size(k)*log_llh_good{k}/(60*31)));
%     log2_llh{k}=log2_llh{k}/sample_size(k);
% end

% original_log_llh=cell(0);
% for k=1:3 % go over trial types
%     original_log_llh{k}=log_llh_good{k}/(60*31);
% end

end