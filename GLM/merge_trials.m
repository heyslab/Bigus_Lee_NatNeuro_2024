function [FiringRate,Position,Time,Speed] = ...
    merge_trials(selected_trials,selected_cells,...
    parsed_behavior,parsed_flevel_only_all,frame_rate)

% Make example data
Position=[];
Time=[];
Speed=[];
FiringRate =[];

% Real time settings
time_interval = 1/frame_rate;
realTime = 0:time_interval:499*time_interval; % seconds

% Piece together all selected trails
for  j=1:length(selected_trials) % go over trails
i=selected_trials(j);
Position = [Position squeeze(parsed_behavior(i,:,2))];


Time =[Time realTime];
Speed =[Speed squeeze(parsed_behavior(i,:,1))];
end

% Get dF/F from time cells in selected trails
for k=1:length(selected_cells)
    FR=[];
    for  j=1:length(selected_trials)
    i=selected_trials(j);
    FR = [FR squeeze(parsed_flevel_only_all(i,:,selected_cells(k)))];
    end
    FiringRate(:,k)= FR';
end

% FiringRate is a m*n matrix
% m = data length
% n = time cell number




end