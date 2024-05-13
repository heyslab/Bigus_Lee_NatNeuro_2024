
% % 2022 Nov 10

clear;
close all;


% % set input
% cell_id = '31_1029_13';
% session_id = '31_1029';

cell_id = '31_1029_178';
session_id = '31_1029';

% input_root = 'G:\Hyuwnoo\OneDrive - SNU\Heys lab folder\1. projects\1. tDNMT\1. Analysis\a\5\parsed_cell\';
% input_root2 = 'G:\Hyuwnoo\OneDrive - SNU\Heys lab folder\1. projects\1. tDNMT\1. Analysis\a\5\parsed_session\';

input_root = 'G:\Hyuwnoo\OneDrive - SNU\Heys lab folder\1. projects\1. tDNMT\1. Analysis\a\5\parsed_cell_reproduce\';
input_root2 = 'G:\Hyuwnoo\OneDrive - SNU\Heys lab folder\1. projects\1. tDNMT\1. Analysis\a\5\parsed_session_reproduce\';

% input_root = '/Users/jennifer/Library/CloudStorage/OneDrive-SNU/Heys lab folder/1. projects/1. tDNMT/1. Analysis/a/5/parsed_cell/';
% input_root2 = '/Users/jennifer/Library/CloudStorage/OneDrive-SNU/Heys lab folder/1. projects/1. tDNMT/1. Analysis/a/5/parsed_session/';
% %

% % set parameters
target_condition = 1;

smoothing_window = 10;
% %


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

% % initialization

% column index for pico data
reward_pd = 1;
tpsignal_pd = 2;
ttl_pd = 3;
buzzer_pd = 4;
licking_pd = 5;
light_pd = 6;
velocity_pd = 7;
odor_pd = 8;
timestamp_pd = 9;
% 

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


% load data
load([input_root cell_id '.mat'], 'parsed_flevel_only_shifted2');
load([input_root2 session_id '.mat'], 'pico_data', 'parsed_trial', 'parsed_behavior', 'Fc3_DF');
parsed_flevel_only = parsed_flevel_only_shifted2;

temp = find(cell_id == '_');
f_data = Fc3_DF(:, str2num(cell_id(temp(2)+1 : end)));
%



% % trial parsing

ttl_index = find(pico_data(:, ttl_pd) == 1);

trial_index = [];
iter = 1;

temp_odor_index = [];
while iter < length(ttl_index)

    if ttl_index(iter+1) - ttl_index(iter) < 5
        iter = iter+1;
        continue;
    end

    temp_index = pico_data(:, ttl_pd) == 1;

    current_range = [];
    current_range(1 : size(pico_data, 1)) = false;
    current_range(ttl_index(iter) : ttl_index(iter+1)) = true;
    current_range = logical(current_range);

    temp_light = bwlabel(pico_data(current_range, light_pd));
    temp_odor = bwlabel(pico_data(current_range, odor_pd));

    if max(temp_light) == 1 && max(temp_odor) == 2
        current_odor_start = find(pico_data(current_range, odor_pd) == 1, 1, 'first');
        trial_index(end+1, 1) = ttl_index(iter);
        trial_index(end, 2) = ttl_index(iter+1);
    end

    iter = iter+1;
end

% %


% get target trials
temp_index = parsed_trial(:, correctness_pt) == correct_type & parsed_trial(:, condition_pt) == target_condition;
trial_numbers = find(temp_index == 1);
%


% get trial ratemaps in time & distance dimension

ratemap_time = [];
ratemap_dist = [];

for trial_iter = 1 : length(trial_numbers)

    current_trial_number = trial_numbers(trial_iter);
    
    current_light = pico_data(trial_index(current_trial_number, 1) : trial_index(current_trial_number, 1)+600, light_pd);
    current_flevel = f_data(trial_index(current_trial_number, 1) : trial_index(current_trial_number, 1)+600);
    current_velocity = pico_data(trial_index(current_trial_number, 1) : trial_index(current_trial_number, 1)+600, velocity_pd);

    current_light_start = find(current_light == 1, 1, 'first');

    ratemap_time(trial_iter, :) = smoothdata(current_flevel(current_light_start : current_light_start+499), 'movmean', smoothing_window);
%     ratemap_time(trial_iter, :) = ratemap_time(trial_iter, :) / max(ratemap_time(trial_iter, :));
    ratemap_time(trial_iter, :) = ratemap_time(trial_iter, :);

    % get distance dimension
    current_distance = 0;
    for iter = 2 : 500
        current_distance(iter, 1) = current_distance(iter-1, 1) + current_velocity(current_light_start + iter-1);
    end

    temp_x = current_distance;
    temp_y = current_flevel(current_light_start : current_light_start+499);

    temp_xq = 0 : 1 : 1000;
%     temp_xq = 0 : 2 : 1500;
%     temp_xq = 0 : 2 : 2000;
    temp_xq(end) = [];

    temp_yq = interp1(temp_x, temp_y, temp_xq);
    temp_yq(isnan(temp_yq)) = 0;

    ratemap_dist(trial_iter, :) = smoothdata(temp_yq, 'movmean', smoothing_window*2);
%     ratemap_dist(trial_iter, :) = ratemap_dist(trial_iter, :) / max(ratemap_dist(trial_iter, :));
    ratemap_dist(trial_iter, :) = ratemap_dist(trial_iter, :);
    %

end

%

% % normalization

ratemap_time_norm = ratemap_time;
ratemap_dist_norm = ratemap_dist;

for iter = 1 : size(ratemap_time, 1)
    ratemap_time_norm(iter, :) = ratemap_time_norm(iter, :) / max(ratemap_time_norm(iter, :));
    ratemap_dist_norm(iter, :) = ratemap_dist_norm(iter, :) / max(ratemap_dist_norm(iter, :));
end

% % 

figure
% imagesc(ratemap_time)
imagesc(ratemap_time_norm)
colormap('jet');
clim([0 1]);
colorbar

figure
% imagesc(ratemap_dist)
imagesc(ratemap_dist_norm)
colormap('jet');
clim([0 1]);
colorbar
