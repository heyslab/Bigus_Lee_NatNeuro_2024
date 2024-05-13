
% % 2023 Sep 4

clear;

rng('shuffle');

% % set input

% cluster id root
input_root = 'G:\Hyuwnoo\OneDrive - University of Utah\Heys lab folder\1. projects\1. tDNMT\1. Analysis\cell_id\';
% input_file_name = 'dayN_LS_20221107_4mice.csv';
input_file_name = 'cell_id.csv';

% day1 = {'30_1014', '31_1014', '33_1005', '35_1014', '32_1018', '34_1014', '25_0624'};
% dayN = {'20_0519', '25_0630', '30_1017', '30_1028', '31_1021', '31_1029', '33_1025', '33_1027', '35_1025', ...
%     '30_10281', '30_10282', '33_10271', '33_10272'};

session_id = '35_1025';

% parsed data root
input_root2 = 'G:\Hyuwnoo\OneDrive - University of Utah\Heys lab folder\1. projects\1. tDNMT\1. Analysis\a\5\parsed_session_reproduce\';
input_root3 = 'G:\Hyuwnoo\OneDrive - University of Utah\Heys lab folder\1. projects\1. tDNMT\1. Analysis\a\5\parsed_cell_reproduce\';


% mutual information root
% input_root4 = 'G:\Hyuwnoo\OneDrive - SNU\Heys lab folder\1. projects\1. tDNMT\1. Analysis\a\4\mat files\';
% input_root4 = '/Users/jennifer/Library/CloudStorage/OneDrive-SNU/Heys lab folder/1. projects/1. tDNMT/1. Analysis/a/4/mat files/';
% % 

% % set output
output_root1 = 'G:\Hyuwnoo\OneDrive - University of Utah\Heys lab folder\1. projects\1. tDNMT\1. Analysis\a\8\mat files7_non time cells\';
output_root2 = 'G:\Hyuwnoo\OneDrive - University of Utah\Heys lab folder\1. projects\1. tDNMT\1. Analysis\a\8\figures7_non time cells\';
% output_root = '/Users/jennifer/Library/CloudStorage/OneDrive-SNU/Heys lab folder/1. projects/1. tDNMT/1. Analysis/a/7/mat files2/';
% % 

% % set parameters
time_bin = 500; % unit in ms
% time_bin = 3000; % unit in ms
% time_bin = 100; % unit in ms
% time_window = 20000; % unit in ms
frame_rate = 30.98; % unit in Hz
% % 

% add path
addpath(genpath('G:\Hyuwnoo\OneDrive - University of Utah\Heys lab folder\1. projects\1. tDNMT\2. Analysis programs'));
%

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% column index for parsed_trial
trialn_pt = 1;
condition_pt = 2;
correctness_pt = 3;

ss_type = 1;
ls_type = 2;
sl_type = 3;
%

% % load input
cell_list = readmatrix([input_root input_file_name]);

valid_list = [];
for cell_iter = 1 : size(cell_list, 1)

    if length(num2str(cell_list(cell_iter, 2))) < 4
        temp_session_id = [num2str(cell_list(cell_iter, 1)) '_0' num2str(cell_list(cell_iter, 2))];
    else
        temp_session_id = [num2str(cell_list(cell_iter, 1)) '_' num2str(cell_list(cell_iter, 2))];
    end

    if strcmp(temp_session_id, session_id)
        valid_list(cell_iter, 1) = 1;
    else
        valid_list(cell_iter, 1) = 0;
    end
end

cell_list(~valid_list, :) = [];
% % 


% % making cell pairs
session_id_list = {};
cell_id_list = {};
cell_peak_list = [];

% make cell id list
for cell_iter = 1 : size(cell_list, 1)
    if length(num2str(cell_list(cell_iter, 2))) < 4
        session_id_list{cell_iter, 1} = [num2str(cell_list(cell_iter, 1)) '_0' num2str(cell_list(cell_iter, 2))];
    else
        session_id_list{cell_iter, 1} = [num2str(cell_list(cell_iter, 1)) '_' num2str(cell_list(cell_iter, 2))];
    end

    cell_id_list{cell_iter, 1} = [session_id_list{cell_iter, 1} '_' num2str(cell_list(cell_iter, 3))];
    if ~strcmp(session_id_list{cell_iter, 1}, session_id)
        error('!');
    end

    % get peak location
    load([input_root3 cell_id_list{cell_iter, 1} '.mat'], 'parsed_flevel_only_shifted2');
    parsed_flevel_only_shifted = parsed_flevel_only_shifted2;
    load([input_root2 session_id_list{cell_iter} '.mat'], 'parsed_trial');

    temp_index = parsed_trial(:, correctness_pt) == 1;
%     temp_index = parsed_trial(:, correctness_pt) == 1 & parsed_trial(:, condition_pt) == sl_type;
    temp = mean(parsed_flevel_only_shifted(temp_index, 1:341), 1);
    [~, cell_peak_list(cell_iter, 1)] = max(temp);

    clear parsed_flevel_only_shifted2 parsed_trial;
    %
end
% 

% sorting cell id list
[~, peak_sort_index] = sort(cell_peak_list, 'ascend');
cell_id_list_sorted = {};

for iter = 1 : length(cell_id_list)
    cell_id_list_sorted{iter, 1} = cell_id_list{peak_sort_index(iter)};
end
%

% make cell pairs
cell_pairs = {};
cell_pairs_index = [];

for i = 1 : size(cell_id_list_sorted, 1)-1
    for j = i+1 : size(cell_id_list_sorted, 1)
        if strcmp(session_id_list{i}, session_id_list{j})
            cell_pairs{end+1, 1} = cell_id_list_sorted{i};
            cell_pairs{end, 2} = cell_id_list_sorted{j};

            cell_pairs_index(end+1, 1) = i;
            cell_pairs_index(end, 2) = j;
        end
    end
end
% 

temp_index = randperm(size(cell_pairs_index, 1), 1000);
cell_pairs_index = cell_pairs_index(temp_index, :);
cell_pairs = cell_pairs(temp_index, :);

% % 

% % batch
for pair_iter = 1 : size(cell_pairs, 1)

    [rat_id, ss_id, ~] = disassemble_id_5zz(cell_pairs{pair_iter, 1});

    load([input_root2 '' rat_id '_' ss_id '.mat'], 'pico_data', 'Fc3_DF', 'parsed_trial');

    
    [~, ~, cl_id1] = disassemble_id_5zz(cell_pairs{pair_iter, 1});
    f_data1 = Fc3_DF(:, str2num(cl_id1));

    [~, ~, cl_id2] = disassemble_id_5zz(cell_pairs{pair_iter, 2});
    f_data2 = Fc3_DF(:, str2num(cl_id2));

    [Kendall_r_trial_list(pair_iter, 1), Kendall_p_trial_list(pair_iter, 1), Kendall_r_iti_list(pair_iter, 1), Kendall_p_iti_list(pair_iter, 1)] = cross_correlation_function7_5a8(cell_pairs{pair_iter, 1}, cell_pairs{pair_iter, 2}, pico_data, parsed_trial, f_data1, f_data2, time_bin, frame_rate);
    
    clear pico_data Fc3_DF parsed_trial;
    disp(['cell pair #' num2str(pair_iter) ' is done: ' num2str(Kendall_r_trial_list(pair_iter)) ', ' num2str(Kendall_r_iti_list(pair_iter))]);
end

% %

% % display & save
if 0
corr_mat_trial = [];
corr_mat_iti = [];

for iter = 1 : size(cell_pairs_index, 1)
    corr_mat_trial(cell_pairs_index(iter, 1), cell_pairs_index(iter, 2)) = Kendall_r_trial_list(iter);
    corr_mat_iti(cell_pairs_index(iter, 1), cell_pairs_index(iter, 2)) = Kendall_r_iti_list(iter);
end

save([output_root1 session_id '.mat'], 'cell_id_list', 'cell_id_list_sorted', 'cell_peak_list', 'peak_sort_index', ...
    'cell_pairs', 'cell_pairs_index', 'Kendall_r_trial_list', 'Kendall_p_trial_list', 'Kendall_r_iti_list', 'Kendall_p_iti_list', ...
    'corr_mat_trial', 'corr_mat_iti');

fh = figure('color', [1 1 1]);
imagesc(corr_mat_trial)
% colormap('hot');
colormap('jet');
clim([-0.3 0.6])
% clim([-0.2 0.3])
colorbar;
title([session_id ' trial'])
saveas(fh, [output_root2 session_id ' trial'], 'png');
close(fh);

fh = figure('color', [1 1 1]);
imagesc(corr_mat_iti)
% colormap('hot');
colormap('jet');
clim([-0.3 0.6])
% clim([-0.2 0.3])
colorbar;
title([session_id ' ITI'])
saveas(fh, [output_root2 session_id ' ITI'], 'png');
close(fh);

fh = figure('color', [1 1 1]);
scatter(Kendall_r_trial_list, Kendall_r_iti_list)
xlabel('Trial');
ylabel('ITI');

[r, p] = corrcoef(Kendall_r_trial_list, Kendall_r_iti_list);
r = r(1, 2); p = p(1, 2);
title([session_id ', r = ' num2str(r) ', p = ' num2str(p)]);
saveas(fh, [output_root2 session_id ' trial vs ITI'], 'png');
close(fh);
end
% % 


