

clear
close all

mother_root = 'G:\Hyuwnoo\OneDrive - SNU\Heys lab folder\1. projects\1. tDNMT\';

% % set input
input_root = 'G:\Hyuwnoo\OneDrive - SNU\Heys lab folder\1. projects\1. tDNMT\1. Analysis\cell_id\';

input_file_name = 'cell_id.csv';

input_root2 = [mother_root '1. Analysis\a\7\mat files2_reproduce\'];
% % 

% % set output
% output_root = 'C:\Users\Alex\OneDrive - SNU\Heys lab folder\1. projects\1. tDNMT\1. Analysis\a\3\pop_ratemap2_5a3';
% output_file_name 
% % 


% % set parameters

% % 


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% % set index

% column index for parsed_behavior, parsed_trial
licking_pb = 1;
odor_pb = 2;
reward_pb = 3;
velocity_pb = 4;

trialn_pt = 1;
condition_pt = 2;
correctness_pt = 3;
%

% condition type, correctness index
ss_type = 1;
ls_type = 2;
sl_type = 3;

correct_type = 1;
incorrect_type = 2;
%

% % 


% % load input
cell_list = readmatrix([input_root input_file_name]);
% % 


% % load all ratemap

cov_time_all = []; % 2d matrix. 1: cell iter, 2: condition
cov_dist_all = []; % 2d matrix. 1: cell iter, 2: condition

for cell_iter = 1 : size(cell_list, 1)

    % load data
    if length(num2str(cell_list(cell_iter, 2))) < 4
        session_id = [num2str(cell_list(cell_iter, 1)) '_0' num2str(cell_list(cell_iter, 2))];
    else
        session_id = [num2str(cell_list(cell_iter, 1)) '_' num2str(cell_list(cell_iter, 2))];
    end
    cell_id = [session_id '_' num2str(cell_list(cell_iter, 3))];
    
    load([input_root2 cell_id '.mat'], 'coef_var_time', 'coef_var_dist');
    cov_time_all(cell_iter, :) = coef_var_time;
    cov_dist_all(cell_iter, :) = coef_var_dist;
    %

    clear coef_var_time coef_var_dist;

end

% % 


