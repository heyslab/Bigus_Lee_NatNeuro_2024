%% Load F file
clear all;
clc
[Ffile,Ffilepath]=uigetfile('*.mat','pick the F file','MultiSelect','on');
fullFfile =[Ffilepath Ffile];
load(fullFfile);
clear parsed_behavior
%%
% Column index for pico_data
% reward_pd = 1;
% tpsignal_pd = 2;
% ttl_pd = 3;
% buzzer_pd = 4;
licking_pd = 5;
light_pd = 6;
velocity_pd = 7;
odor_pd = 8;
% timestamp_pd = 9;

% Settings
frame_rate = 30.98;
time_interval = 1/frame_rate;
realTime = 0:time_interval:499*time_interval; % seconds

% Find each trial 
% (Find trail start, then cut each train into 500 frames)
light_index = find(pico_data(:, light_pd) == 1);
trial_index = [];
iter = 1;

while iter < length(light_index)

    if light_index(iter+1) - light_index(iter) < 5
        iter = iter+1;
        continue;
    end

    trial_index(end+1, 1) = light_index(iter);        
    iter = iter+1;
end
trial_index(:, 2) = trial_index(:, 1) + 499;
%%
if size(parsed_trial,1) ~= size(trial_index,1)
if trial_index(end,1)==light_index(end)
else
    trial_index = [trial_index;light_index(end),light_index(end)+499];
end
end

%% Get trial information & data parsing
 
parsed_flevel_only_all = []; 
% dF/F by trial, 3d matrix:1: trial, 2: frame, 3: cell

parsed_behavior = []; 
% this is just all behavior data by trial vars in one matrix
% 3d matrix:1: trial, 2: frame, 3: type of behavior

for trial_iter = 1 : size(trial_index,1)
   V =[];
   V = pico_data(trial_index(trial_iter,1) : trial_index(trial_iter,2), velocity_pd);

    % Speed cm/s
    parsed_behavior(trial_iter, :, 1) = V*10;
    % Position cm
    parsed_behavior(trial_iter, :, 2) = Speed_to_Position(V,frame_rate);
    % Odor 0 or 1
    parsed_behavior(trial_iter, :, 3) = ...
        pico_data(trial_index(trial_iter,1) : trial_index(trial_iter,2), odor_pd);
    % Licking 0 or 1
    parsed_behavior(trial_iter, :, 4) = ...
        pico_data(trial_index(trial_iter,1) : trial_index(trial_iter,2), licking_pd);

    for cell_iter = 1 : size(Fc3_DF, 2)
        parsed_flevel_only_all(trial_iter, :, cell_iter) = ...
        Fc3_DF(trial_index(trial_iter,1) : trial_index(trial_iter,2), cell_iter);
    end
    
end


% Select all correct trails
% INfo on condition type & correctness index:
% col#1:
    % cell number
% col#2: 
    % ss_type = 1;
    % ls_type = 2; 
    % sl_type = 3;
% col#3: 
    % correct_type = 1;
    % incorrect_type = 2;

% Select all correct trails
correct_trials = [];
for i = 1:size(parsed_trial,1)
if parsed_trial(i,3)==1 
    correct_trials = [correct_trials i];
end
end

% Select different correct trial types
c_ss_trials =[];c_ls_trials =[];c_sl_trials =[];
for i = 1:size(parsed_trial,1)
    if parsed_trial(i,3)==1 % correct trials
    
        if parsed_trial(i,2)==1 
            c_ss_trials = [c_ss_trials i];
        elseif parsed_trial(i,2)==2
            c_ls_trials = [c_ls_trials i];
        elseif parsed_trial(i,2)==3
            c_sl_trials = [c_sl_trials i];
        else
            keyboard
        end

    end
end

% Select time cells
selected_cells = 1:size(Fc3_DF,2);

% Piece together all selected trials
c_trails{1}=c_ss_trials;
c_trails{2}=c_ls_trials;
c_trails{3}=c_sl_trials;

for i =1:3
[Time{i},Speed{i},Position{i},Odor{i},Licking{i},...
    FiringRate{i},...
    behaviorOI{i},dFFOI{i}]...
    = SelectTrial(c_trails{i},parsed_behavior,parsed_flevel_only_all,selected_cells,realTime);
end

% Save data 
fullFname=[Ffilepath [Ffile '_Correct_Trials.mat']];
save(fullFname,'parsed_flevel_only_all','parsed_trial','parsed_behavior',...
    'pico_data','masks_NoDups','Fc3_DF','com_roi',...
    'Speed','Position','Time','Odor','Licking','FiringRate',...
    'behaviorOI','dFFOI',...
    'c_ls_trials','c_sl_trials','c_ss_trials','c_trails','correct_trials');

%% Function
function [Time,Speed,Position,Odor,Licking,FiringRate,behaviorOI,dFFOI]...
    = SelectTrial(trialOI,parsed_behavior,parsed_flevel_only_all,allcells,realTime)
Time=[];Speed=[];Position=[];Odor=[];Licking=[];
% Get behavior data in selected trials
for  j=1:length(trialOI) % Go over trials
i=trialOI(j);
Time =[Time realTime];
Speed =[Speed squeeze(parsed_behavior(i,:,1))];
Position = [Position squeeze(parsed_behavior(i,:,2))];
Odor = [Odor squeeze(parsed_behavior(i,:,3))];
Licking =[Licking squeeze(parsed_behavior(i,:,4))];
end

% Get dF/F in selected trials
for k=1:length(allcells)   % go over cells
    FR=[];
    for  j=1:length(trialOI) % go over trials
    i=trialOI(j);
    FR = [FR squeeze(parsed_flevel_only_all(i,:,allcells(k)))];
    end
    FiringRate(:,k)= FR';
end

behaviorOI = parsed_behavior(trialOI,:,:);
dFFOI = parsed_flevel_only_all(trialOI,:,:);

end
