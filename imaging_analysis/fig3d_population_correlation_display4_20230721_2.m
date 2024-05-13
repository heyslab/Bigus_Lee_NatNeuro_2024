
% 2023 Sep 12

rng('shuffle');

clear
close all

mother_root = 'G:\Hyuwnoo\OneDrive - University of Utah\Heys lab folder\1. projects\1. tDNMT\';
% mother_root = '/Users/jennifer/Library/CloudStorage/OneDrive-SNU/Heys lab folder/1. projects/1. tDNMT/';

% % set input
% input_root = [mother_root '1. Analysis\all\20221107\'];
% input_root = [mother_root '1. Analysis\all\20221129\'];
input_root = 'G:\Hyuwnoo\OneDrive - University of Utah\Heys lab folder\1. projects\1. tDNMT\1. Analysis\cell_id\';
% input_root = '/Users/jennifer/Library/CloudStorage/OneDrive-SNU/Heys lab folder/1. projects/1. tDNMT/1. Analysis/cell_id/';

% input_file_name = 'day1_all_20221107.csv';
% input_file_name = 'dayN_SS_20221107_4mice.csv';
% input_file_name = 'probe3_SS_20221129_E30.csv';
% input_file_name = 'sig_tuned_dayN_20221109.csv';
input_file_name = 'cell_id.csv';

input_root2 = [mother_root '1. Analysis\a\5\parsed_session_reproduce\'];
input_root3 = [mother_root '1. Analysis\a\5\parsed_cell_reproduce\'];
% input_root2 = [mother_root '1. Analysis/a/5/parsed_session/'];
% input_root3 = [mother_root '1. Analysis/a/5/parsed_cell/'];
% % 

% % set output
% output_root = 'C:\Users\Alex\OneDrive - SNU\Heys lab folder\1. projects\1. tDNMT\1. Analysis\a\3\pop_ratemap2_5a3';
% output_file_name 
% % 


% % set parameters

normalization_method = 1; % only 1 option available

target_condition1 = 1;
target_condition2 = 2;

% target_window = [8 11]*31;

% bin_size = 1; % unit = frame. 1 frames ~ 32 ms
bin_size = 8; % unit = frame. 8 frames ~ 258 ms

% shuffle_N = 100;
shuffle_N = 10000;
% % 

addpath(genpath('G:\Hyuwnoo\OneDrive - University of Utah\Heys lab folder\1. projects\1. tDNMT\2. Analysis programs'));
% addpath(genpath('/Users/jennifer/Library/CloudStorage/OneDrive-SNU/Heys lab folder/1. projects/1. tDNMT/2. Analysis programs'));

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
probe_type = 4;

correct_type = 1;
incorrect_type = 2;
%

% % 


% % load input
cell_list = readmatrix([input_root input_file_name]);
% % 


% % load all ratemap

ratemap_all = []; % 3d matrix. 1: cell iter, 2: frame, 3: condition
com_all = [];  % 2d matrix. 1: cell iter, 2: condition
corr_r_all = []; % 2d matrix. 1: cell iter, 2: condition (1st column: SS vs LS; 2nd column: SS vs SL; 3rd column: LS vs SL)
corr_p_all = []; % 2d matrix. 1: cell iter, 2: condition (1st column: SS vs LS; 2nd column: SS vs SL; 3rd column: LS vs SL)

ratemap_cond1_shuffle = [];
ratemap_cond2_shuffle = [];

for cell_iter = 1 : size(cell_list, 1)

    % get mean rate map for each condition (only correct trials)
    if length(num2str(cell_list(cell_iter, 2))) < 4
        session_id = [num2str(cell_list(cell_iter, 1)) '_0' num2str(cell_list(cell_iter, 2))];
    else
        session_id = [num2str(cell_list(cell_iter, 1)) '_' num2str(cell_list(cell_iter, 2))];
    end
    cell_id = [session_id '_' num2str(cell_list(cell_iter, 3))];
    
    load([input_root2 session_id '.mat'],'parsed_trial', 'parsed_behavior');
    load([input_root3 cell_id '.mat'], 'parsed_flevel_only_shifted2');
    parsed_flevel_only = parsed_flevel_only_shifted2;

    % shuffle
%     temp_index = parsed_trial(:, condition_pt) == target_condition1 & parsed_trial(:, correctness_pt) == correct_type;
    temp_index = parsed_trial(:, condition_pt) == target_condition1;
    trial_set1 = find(temp_index == 1);
%     temp_index = parsed_trial(:, condition_pt) == target_condition2 & parsed_trial(:, correctness_pt) == correct_type;
    temp_index = parsed_trial(:, condition_pt) == target_condition2;
    trial_set2 = find(temp_index == 1);

    trial_set = [trial_set1; trial_set2];

    for shuffle_iter = 1 : shuffle_N+1

        if shuffle_iter == shuffle_N+1
            trial_set1_shuffle = trial_set1;
            trial_set2_shuffle = trial_set2;
        else
            trial_set_shuffle = trial_set(randperm(length(trial_set)));
            trial_set1_shuffle = trial_set_shuffle(1 : length(trial_set1));
            trial_set2_shuffle = trial_set_shuffle(length(trial_set1)+1 : end);
        end

        ratemap_cond1_shuffle(cell_iter, :, shuffle_iter) = mean(parsed_flevel_only(trial_set1_shuffle, :), 1);
        ratemap_cond2_shuffle(cell_iter, :, shuffle_iter) = mean(parsed_flevel_only(trial_set2_shuffle, :), 1);
    end
    %

end

% % 


% % normalization

if normalization_method == 1 % make each cell's peak as 1

    for cell_iter = 1 : size(ratemap_cond1_shuffle, 1)

        for shuffle_iter = 1 : shuffle_N+1

            peak_fr1 = max(ratemap_cond1_shuffle(cell_iter, :, shuffle_iter));            
            peak_fr2 = max(ratemap_cond2_shuffle(cell_iter, :, shuffle_iter));
            peak_fr = max([peak_fr1, peak_fr2]);
%             peak_fr = 1;

            if peak_fr > 0
                ratemap_cond1_shuffle(cell_iter, :, shuffle_iter) = ratemap_cond1_shuffle(cell_iter, :, shuffle_iter) / peak_fr;
                ratemap_cond2_shuffle(cell_iter, :, shuffle_iter) = ratemap_cond2_shuffle(cell_iter, :, shuffle_iter) / peak_fr;
            end
        end
    end

elseif normalization_method == 2 % make each condition's peak as 1

    for cell_iter = 1 : size(ratemap_all, 1)
        for cond_iter = 1 : 6
            temp = squeeze(ratemap_all(cell_iter, :, cond_iter));
            peak_fr = max(temp);

            if peak_fr > 0
                ratemap_all(cell_iter, :, cond_iter) = ratemap_all(cell_iter, :, cond_iter) / peak_fr;
            else
            end
        end
    end
end

% % 

% % Binning
for iter = 1 : ceil(size(ratemap_cond1_shuffle, 2) / bin_size)

    if iter == ceil(size(ratemap_cond1_shuffle, 2) / bin_size)
        current_range = [(iter-1) * bin_size+1, size(ratemap_cond1_shuffle, 2)];
    else
        current_range = [(iter-1) * bin_size+1, iter * bin_size];
    end

    temp1(:, iter, :) = mean(ratemap_cond1_shuffle(:, current_range(1) : current_range(2), :), 2);
    temp2(:, iter, :) = mean(ratemap_cond2_shuffle(:, current_range(1) : current_range(2), :), 2);
end

ratemap_cond1_shuffle = temp1;
ratemap_cond2_shuffle = temp2;
% % 

% % Population correlation matrix

% pop_corr_mat = [];
% pop_corp_mat = [];

pop_corr_vect = [];
pop_pcorr_vect = [];
fr_diff_vect = [];

for shuffle_iter = 1 : shuffle_N+1
    for iter1 = 1 : size(ratemap_cond1_shuffle, 2)
%         for iter2 = 1 : size(ratemap_cond1_shuffle, 2)

            temp1 = ratemap_cond1_shuffle(:, iter1, shuffle_iter);
            temp2 = ratemap_cond2_shuffle(:, iter1, shuffle_iter);

            nan_index = isnan(temp1) | isnan(temp2);
            temp1(nan_index) = [];
            temp2(nan_index) = [];

%             [r, p] = corrcoef(temp1, temp2);

%             pop_corr_mat(iter1, iter2, shuffle_iter) = r(1, 2);
%             pop_corp_mat(iter1, iter2, shuffle_iter) = p(1, 2);

%             if iter1 == iter2
%                 pop_corr_vect(iter1, shuffle_iter) = r(1, 2);
%                 pop_pcorr_vect(iter1, shuffle_iter) = p(1, 2);

                fr_diff_vect(iter1, shuffle_iter, :) = abs(temp1 - temp2);
%             end
%         end
    end
end

% % 


% % Display

fr_diff_vect_mean = [];
fr_diff_vect_sem = [];

for iter = 1 : size(fr_diff_vect, 1)
    fr_diff_vect_mean(iter, 1) = mean(fr_diff_vect(iter, shuffle_N+1, :));
    fr_diff_vect_sem(iter, 1) = std(fr_diff_vect(iter, shuffle_N+1, :)) / sqrt(size(fr_diff_vect, 3));
end

figure('color', [1 1 1]);
hold on;

% for iter = 1 : shuffle_N
%     plot(fr_diff_vect(:, iter), 'color', [.5 .5 .5]);
% end

for iter = 1 : size(fr_diff_vect, 1)
    plot([iter, iter], [fr_diff_vect_mean(iter) - fr_diff_vect_sem(iter), fr_diff_vect_mean(iter) + fr_diff_vect_sem(iter)], 'color', [.5 .5 .5])
end
plot(fr_diff_vect_mean, 'color', 'k');

set(gca, 'xlim', [0 341], 'ylim', [0.1 0.3]);
set(gca, 'xlim', [0 43], 'ylim', [0 0.3]);
% set(gca, 'xlim', [0 341]);
ylabel('Mean fr difference');

% set(gca, 'xtick', 0:1:260, 'xticklabels', {'0','1','2','3','4','5','6','7','8','9','10','11','12','13','14'});
% set(gca, 'xtick', 0:31:434, 'xticklabels', {'0','1','2','3','4','5','6','7','8','9','10','11','12','13','14'});
xlabel('Time (sec)');

title('Day N, SS vs LS');

% fr_diff_shuffle_median = median(fr_diff_vect(:, 1:shuffle_N), 2);
% pop_corr_shuffle_median = median(pop_corr_vect(:, 1:shuffle_N), 2);

% save('SS vs LS, day 1.mat', 'fr_diff_vect', 'fr_diff_vect_mean', 'fr_diff_vect_sem');
