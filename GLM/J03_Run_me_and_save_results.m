%% Load F file
clear all;
clc
[Ffile3,Ffilepath3]=uigetfile('*.mat','pick the F file','MultiSelect','on');

IG_time_all =[];IG_pos_all =[];IG_licking_all =[];

sig_mdl_all=cell(0);all_mdl_all=cell(0);good_cell_all=cell(0);
trail_count_all=[];cell_count =[];
sample_size_all=[];spike_count_all=[];spike_count_ratio_all=[];
log_llh_all= [];log_llh_diff_all = [];mse_all =[];corr_all=[];
r2_all=[];r2_adj_all=[];

% number of parameters for each model
d = [23,21,21,21,11,11,11,3];

    for i=1:size(Ffile3,2)      
   
    load([Ffilepath3 Ffile3{i}],'testFit','FR',...
        'c_ls_trials','c_sl_trials','c_ss_trials');

    % Select best model
    for k=1:3
    [all_mdl{k},sig_mdl{k}] = SelectBestModels(testFit{k}) ;
    end

    % Find cells with significant models for each trial type
    good_cell=cell(1,3);good_cell=cell(1,3);
    for k=1:3
    if  ~isempty(sig_mdl{k})
    good_cell{k}=sig_mdl{k}(:,1);
    mdl_max_llh{k}=sig_mdl{k}(:,3);
    else
    good_cell{k}=[];
    mdl_max_llh{k}=[];
    end
    end
    %
  
    % Calculate informaton gained for mdl with LLH>0
    [IG_time,IG_pos,IG_licking] = Information_Gained(testFit,good_cell,mdl_max_llh);

    % Count the number of trials in each session (ss, ls, sl)
    trial_count = [length(c_ss_trials),length(c_ls_trials),length(c_sl_trials)];
    
    % Get information of mdl with LLH>0
    [sample_size,spike_count_all{i},spike_count_ratio_all{i},...
    r2,corr,log_llh,log_llh_diff,mse,r2_adj] = ...
    calculate_stats(testFit,FR,trial_count,good_cell,d);
    % Get information of all models
    [r2_allcell{i,1},corr_allcell{i,1},...
        log_llh_allcell{i,1},log_llh_diff_allcell{i,1},mse_allcell{i,1}] ...
        = mdl_stats(testFit);
    
    % Count cell
    cell_count_s=[];
    for k=1:3
    cell_count_s = [cell_count_s size(testFit{k},1)];
    end

     % Combine data
    cell_count = [cell_count; cell_count_s];
    sample_size_all = [sample_size_all; sample_size];
    trail_count_all = [trail_count_all; trial_count];

    IG_time_all = [IG_time_all; IG_time];
    IG_pos_all = [IG_pos_all; IG_pos];
    IG_licking_all = [IG_licking_all ;IG_licking];

    sig_mdl_all = [sig_mdl_all ;sig_mdl]; all_mdl_all = [all_mdl_all ;all_mdl]; 
    good_cell_all = [good_cell_all;good_cell];

    r2_all = [r2_all;r2];
    corr_all = [corr_all;corr];
    log_llh_all= [log_llh_all ;log_llh];
    log_llh_diff_all = [log_llh_diff_all; log_llh_diff];
    mse_all =[mse_all ;mse];
    r2_adj_all =[r2_adj_all;r2_adj];
    end

    clear IG_time IG_pos IG_licking cell_count_s
    clear sig_mdl all_mdl good_cell trial_count mdl_max_llh
    clear sample_size good_spike_cells 
    clear r2 r2_adj corr log_llh log_llh_diff mse 
    clear testFit FiringRate c_ss_trials c_sl_trials c_ls_trials
    clear Ffile3 Ffilepath3 i k 

