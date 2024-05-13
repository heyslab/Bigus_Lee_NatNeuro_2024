%% Use data organized to remove trials that are just empty
% Need time cell data

% Load time file
clear all;
clc

timefile = '/Users/sparrowsally/Desktop/parsed_session_reproduce/Time_cells.mat';
load(timefile);

% Load data
[Ffile,Ffilepath]=uigetfile('*.mat','pick the F file','MultiSelect','on');
fullFfile =[Ffilepath Ffile];
load(fullFfile);

timecell= TimeCellAll(10,:);
% Select all empty trails
not_empty =zeros(size(parsed_flevel_only_all,1),size(parsed_flevel_only_all,3));
for cellnum = 1:size(parsed_flevel_only_all,3) % go over cells
for trialnum = 1:size(parsed_flevel_only_all,1) % go over trials
    if sum(parsed_flevel_only_all(trialnum,:,cellnum)) >0
        not_empty(trialnum,cellnum)=1;
    end
end
end

% Settings
frame_rate = 30.98;
time_interval = 1/frame_rate;
realTime = 0:time_interval:499*time_interval; % seconds


clear Time Speed Position Licking FiringRate Odor

for k =1:3
[Time{k},Speed{k},Position{k},Licking{k},FiringRate{k}]...
    = SelectTrial(c_trails{k},parsed_behavior,parsed_flevel_only_all,realTime,not_empty,timecell{k});
end

% Save data 
fullFname=[Ffilepath [Ffile '_Skip_Empty.mat']];
save(fullFname,'parsed_flevel_only_all','parsed_trial','parsed_behavior',...
    'pico_data','masks_NoDups','Fc3_DF','com_roi',...
    'Speed','Position','Time','Licking','FiringRate',...
    'c_ls_trials','c_sl_trials','c_ss_trials','c_trails','correct_trials','not_empty');

%% Function
function [Time,Speed,Position,Licking,FiringRate]...
    = SelectTrial(c_trails,parsed_behavior,parsed_flevel_only_all,realTime,not_empty,timecell)



for j =1: length(timecell)   % go over cells
    cellID = timecell(j);
    for  i = 1:length(c_trails) % go over trials
    trialID = c_trails(i); 
    FiringRate{j,i} = NaN(1,500);
    Time{j,i} = NaN(1,500);
    Speed{j,i} = NaN(1,500);
    Position{j,i} = NaN(1,500);
    Licking{j,i} = NaN(1,500);

        if not_empty(trialID,cellID)==1
        FiringRate{j,i} = squeeze(parsed_flevel_only_all(trialID,:,cellID));
        Time{j,i}= realTime;
        Speed{j,i} = squeeze(parsed_behavior(trialID,:,1));
        Position{j,i} = squeeze(parsed_behavior(trialID,:,2));
        Licking{j,i}= squeeze(parsed_behavior(trialID,:,4));
        end

    end
end

end
