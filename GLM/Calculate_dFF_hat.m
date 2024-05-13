function [dFF_hat,...
    dFF_hat_time_by_trial,dFF_hat_pos_by_trial,dFF_hat_lick_by_trial,...
    FiringRate_by_trial] = Calculate_dFF_hat(param,Behavior_Variables,behaviorOI,FiringRate,selected_cells,numModels)

dFF_hat =cell(0);

% GLMfit
for k=1:3 % Go over trial types
for i=1:length(selected_cells{k})% Go over all cells
    cell_num = selected_cells{k}(i);
       for n = 1:numModels 
            dFF_hat{k}{i,n} = param{k}{ cell_num,n}(1)+...
                Behavior_Variables{k}{n}*(param{k}{cell_num,n}(2:end)');    
       end
end
end

% Calculate dF/F hat by trial
dFF_hat_time_by_trial=cell(0);
dFF_hat_pos_by_trial=cell(0);
dFF_hat_lick_by_trial=cell(0);
FiringRate_by_trial=cell(0);
for k=1:3 % Go over trial types
for j = 1:length(selected_cells{k}) % Go over cells

    dFF_hat_time = dFF_hat{k}{j,5};
    dFF_hat_pos = dFF_hat{k}{j,6};
    dFF_hat_lick = dFF_hat{k}{j,7};
  
    for i = 1:size(behaviorOI{k},1) % Go over each trial
    dFF_hat_pos_by_trial{k}(i,:,j)  = dFF_hat_pos(500*(i-1)+1:500*(i));
    dFF_hat_time_by_trial{k}(i,:,j)  = dFF_hat_time(500*(i-1)+1:500*(i));
    dFF_hat_lick_by_trial{k}(i,:,j)  = dFF_hat_lick(500*(i-1)+1:500*(i));
    FiringRate_by_trial{k}(i,:,j)  = FiringRate{k}(500*(i-1)+1:500*(i),selected_cells{k}(j));
    end

end
end




end