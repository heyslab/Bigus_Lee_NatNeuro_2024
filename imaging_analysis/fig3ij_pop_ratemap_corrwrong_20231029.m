

clear
% close all

rng('shuffle');

mother_root = 'G:\Hyuwnoo\OneDrive - University of Utah\Heys lab folder\1. projects\1. tDNMT\';
% mother_root = '/Users/jennifer/Library/CloudStorage/OneDrive-SNU/Heys lab folder/1. projects/1. tDNMT/';

% % set input
input_root = 'G:\Hyuwnoo\OneDrive - University of Utah\Heys lab folder\1. projects\1. tDNMT\1. Analysis\all\20231029\';
% input_root = '/Users/jennifer/Library/CloudStorage/OneDrive-SNU/Heys lab folder/1. projects/1. tDNMT/1. Analysis/cell_id/';
% input_root = [mother_root '1. Analysis\all\20221107\'];
% input_root = [mother_root '1. Analysis/all/20221222/'];

input_file_name = 'SS_dayN_20231029.csv';
% input_file_name = 'day1_SS_20221107.csv';
% input_file_name = 'dayN_SL_4mice_20221222.csv';
% input_file_name = 'sig_tuned_dayN_20221109.csv';

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

normalization_method = 2;

target_condition = 1; % 1:S-S, 2:L-S, 3:S-L

% N_shuffle = 1000;
N_shuffle = 1;

smoothing_window = 15;

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

ratemap_all = []; % 3d matrix. 1: cell iter, 2: frame, 3: condition (1st column: correct1; 2nd column: correct2; 3rd column: wrong)
corr_r_all = []; % 2d matrix. 1: cell iter, 2: condition (1st column: correct1 vs correct2; 2nd column: correct1 vs wrong)
corr_p_all = []; % 2d matrix. 1: cell iter, 2: condition (1st column: correct1 vs correct2; 2nd column: correct1 vs wrong)

for cell_iter = 1 : size(cell_list, 1)

    % load data
    if length(num2str(cell_list(cell_iter, 2))) < 4
        session_id = [num2str(cell_list(cell_iter, 1)) '_0' num2str(cell_list(cell_iter, 2))];
    else
        session_id = [num2str(cell_list(cell_iter, 1)) '_' num2str(cell_list(cell_iter, 2))];
    end
    cell_id = [session_id '_' num2str(cell_list(cell_iter, 3))];
    
    load([input_root2 session_id '.mat'],'parsed_trial', 'parsed_behavior');
    load([input_root3 cell_id '.mat'], 'parsed_flevel_only_shifted2');
    parsed_flevel_only = parsed_flevel_only_shifted2;
    %

    % get trial numbers
    temp_index = parsed_trial(:, condition_pt) == target_condition & parsed_trial(:, correctness_pt) == correct_type;
    correct_trial_index = find(temp_index == 1);
    
    temp_index = parsed_trial(:, condition_pt) == target_condition & parsed_trial(:, correctness_pt) == incorrect_type;
    incorrect_trial_index = find(temp_index == 1);

    if floor(length(correct_trial_index)/2) < length(incorrect_trial_index)
        N_subsample = floor(length(correct_trial_index)/2);
    else
        N_subsample = length(incorrect_trial_index);
    end
    %

    temp_ratemap = [];
    for shuffle_iter = 1 : N_shuffle

        % shuffle trial numbers
        correct_trial_index_rand = correct_trial_index(randperm(length(correct_trial_index)));
        incorrect_trial_index_rand = incorrect_trial_index(randperm(length(incorrect_trial_index)));

        correct_trials1 = correct_trial_index_rand(N_subsample+1 : end);
        correct_trials2 = correct_trial_index_rand(1 : N_subsample);
        incorrect_trials = incorrect_trial_index_rand(1 : N_subsample);
        %

        % get ratemaps
        temp_ratemap(shuffle_iter, :, 1) = mean(parsed_flevel_only(correct_trials1, :), 1);
        temp_ratemap(shuffle_iter, :, 2) = mean(parsed_flevel_only(correct_trials2, :), 1);
        temp_ratemap(shuffle_iter, :, 3) = mean(parsed_flevel_only(incorrect_trials, :), 1);
        %

        % get correlation coefficient
%         [a, b] = corrcoef(temp_ratemap(shuffle_iter, :, 1), temp_ratemap(shuffle_iter, :, 2));
        [a, b] = corrcoef(temp_ratemap(shuffle_iter, 1:341, 1), temp_ratemap(shuffle_iter, 1:341, 2));
        corr_r_all(cell_iter, 1, shuffle_iter) = a(1, 2);
        corr_p_all(cell_iter, 1, shuffle_iter) = b(1, 2);

%         [a, b] = corrcoef(temp_ratemap(shuffle_iter, :, 1), temp_ratemap(shuffle_iter, :, 3));
        [a, b] = corrcoef(temp_ratemap(shuffle_iter, 1:341, 1), temp_ratemap(shuffle_iter, 1:341, 3));
        corr_r_all(cell_iter, 2, shuffle_iter) = a(1, 2);
        corr_p_all(cell_iter, 2, shuffle_iter) = b(1, 2);
        %
    end

    % get ratemaps
    ratemap_all(cell_iter, :, 1) = mean(squeeze(temp_ratemap(:, :, 1)), 1);
    ratemap_all(cell_iter, :, 2) = mean(squeeze(temp_ratemap(:, :, 2)), 1);
    ratemap_all(cell_iter, :, 3) = mean(squeeze(temp_ratemap(:, :, 3)), 1);
    %

end

% % 


% % get peak index
peak_index = [];

for cell_iter = 1 : size(ratemap_all, 1)
    for cond_iter = 1 : 3
        [~, peak_index(cell_iter, cond_iter)] = max(ratemap_all(cell_iter, :, cond_iter));
    end
end
% %

% % smoothing

if smoothing_window == 0 % no smoothing

else % smoothing with moving average

    for cell_iter = 1 : size(ratemap_all, 1)
        for cond_iter = 1 : 3
            temp = squeeze(ratemap_all(cell_iter, :, cond_iter));
            temp = smoothdata(temp, 'movmean', smoothing_window);

            ratemap_all(cell_iter, :, cond_iter) = temp;
        end
    end
end

% % 

% % normalization

if normalization_method == 1 % make each cell's peak as 1

    for cell_iter = 1 : size(ratemap_all, 1)

        temp = squeeze(ratemap_all(cell_iter, :, :));
        peak_fr = max(max(temp));

        if peak_fr > 0
            ratemap_all(cell_iter, :, :) = ratemap_all(cell_iter, :, :) / peak_fr;
        else
        end
    end

elseif normalization_method == 2 % make each condition's peak as 1

    for cell_iter = 1 : size(ratemap_all, 1)
        for cond_iter = 1 : 3
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



% sort & display - original
plot_name = {'correct 1st half', 'correct 2nd half', 'incorrect'};

for cond_iter = 1 : 1

    % sort
    [~, sort_index] = sort(peak_index(:, cond_iter));
    sorted_sort_index = ratemap_all(sort_index, :, :);
    %

    % display
    sheet_position = [200 200 400 400];
    figure('position', sheet_position, 'color', [1 1 1]);

    for iter = 1 : 3
        subplot(1, 3, iter);
        imagesc(sorted_sort_index(:, :, iter));
        colormap('jet');
        clim([0 1.1]);

        hold on
%         plot([87, 87], [1, size(ratemap_all, 1)], '-', 'color', 'r', 'lineWidth', 2);
%         plot([242, 242], [1, size(ratemap_all, 1)], '-', 'color', 'r', 'lineWidth', 2);
%         plot([338, 338], [1, size(ratemap_all, 1)], '-', 'color', 'r', 'lineWidth', 2);

        set(gca, 'xtick', 0:31:434, 'xticklabels', {'0','1','2','3','4','5','6','7','8','9','10','11','12','13','14'});
        xlabel('Time (sec)');
        ylabel('Cell number');
        title(plot_name{iter});
    end
    %

end
% 

corr_r_all_mean = squeeze(mean(corr_r_all, 3));

% nanmean(corr_r_all_mean(:,1))
nanmean(corr_r_all_mean(:,2))

% signrank(corr_r_all_mean(:, 1), corr_r_all_mean(:, 2))

% save("day1_SS_20231029.mat", 'corr_r_all', 'corr_p_all');
% save("dayN_SS_example plot 20231029.mat", 'corr_r_all', 'corr_p_all');
