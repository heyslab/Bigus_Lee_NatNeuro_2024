

clear
close all

mother_root = 'G:\Hyuwnoo\OneDrive - SNU\Heys lab folder\1. projects\1. tDNMT\';
% mother_root = '/Users/jennifer/Library/CloudStorage/OneDrive-SNU/Heys lab folder/1. projects/1. tDNMT/';

% % set input
% input_root = [mother_root '1. Analysis\all\20221107\'];
% input_root = [mother_root '1. Analysis/all/20221222/'];
input_root = 'G:\Hyuwnoo\OneDrive - SNU\Heys lab folder\1. projects\1. tDNMT\1. Analysis\all\20230916\';
% input_root = '/Users/jennifer/Library/CloudStorage/OneDrive-SNU/Heys lab folder/1. projects/1. tDNMT/1. Analysis/cell_id/';

% input_file_name = 'day1_all_20221107.csv';
% input_file_name = 'dayN_all_20221107_4mice.csv';
% input_file_name = 'day1_SS_4mice_20221222.csv';
input_file_name = 'cell_id_SL_day all_20230916.csv';

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
smoothing_window = 15;
% smoothing_window = 0;

% % 

addpath(genpath('G:\Hyuwnoo\OneDrive - SNU\Heys lab folder\1. projects\1. tDNMT\2. Analysis programs'));
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

correct_type = 1;
incorrect_type = 2;
%

% % 


% % load input
cell_list = readmatrix([input_root input_file_name]);
% % 


% % load all ratemaps

ratemap_all = []; % 3d matrix. 1: cell iter, 2: frame, 3: condition
com_all = [];  % 2d matrix. 1: cell iter, 2: condition
corr_r_all = []; % 2d matrix. 1: cell iter, 2: condition (1st column: SS vs LS; 2nd column: SS vs SL; 3rd column: LS vs SL)
corr_p_all = []; % 2d matrix. 1: cell iter, 2: condition (1st column: SS vs LS; 2nd column: SS vs SL; 3rd column: LS vs SL)

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

    for cond_iter = 1 : 3

        % correct trials
        temp_index = parsed_trial(:, condition_pt) == cond_iter & parsed_trial(:, correctness_pt) == correct_type;
        temp = mean(parsed_flevel_only(temp_index, :), 1);
        ratemap_all(cell_iter, :, cond_iter) = temp;
        com_all(cell_iter, cond_iter) = get_com_5zz(temp(1:341));
        %

        % incorrect trials
        temp_index = parsed_trial(:, condition_pt) == cond_iter & parsed_trial(:, correctness_pt) == incorrect_type;
        temp = mean(parsed_flevel_only(temp_index, :), 1);
        ratemap_all(cell_iter, :, cond_iter+3) = temp;
        com_all(cell_iter, cond_iter+3) = get_com_5zz(temp(1:341));
        %

    end
    %

    % compute correlation coefficient
%     [a, b] = corrcoef(ratemap_all(cell_iter, :, ss_type), ratemap_all(cell_iter, :, ls_type));
    [a, b] = corrcoef(ratemap_all(cell_iter, 1:341, ss_type), ratemap_all(cell_iter, 1:341, ls_type));
    corr_r_all(cell_iter, 1) = a(1, 2);
    corr_p_all(cell_iter, 1) = b(1, 2);

%     [a, b] = corrcoef(ratemap_all(cell_iter, :, ss_type), ratemap_all(cell_iter, :, sl_type));
    [a, b] = corrcoef(ratemap_all(cell_iter, 1:341, ss_type), ratemap_all(cell_iter, 1:341, sl_type));
    corr_r_all(cell_iter, 2) = a(1, 2);
    corr_p_all(cell_iter, 2) = b(1, 2);

%     [a, b] = corrcoef(ratemap_all(cell_iter, :, ls_type), ratemap_all(cell_iter, :, sl_type));
    [a, b] = corrcoef(ratemap_all(cell_iter, 1:341, ls_type), ratemap_all(cell_iter, 1:341, sl_type));
    corr_r_all(cell_iter, 3) = a(1, 2);
    corr_p_all(cell_iter, 3) = b(1, 2);
    %

end

% % 


% % get peak index
peak_index = [];

for cell_iter = 1 : size(ratemap_all, 1)
    for cond_iter = 1 : 3
%         [~, peak_index(cell_iter, cond_iter)] = max(ratemap_all(cell_iter, :, cond_iter));
        [~, peak_index(cell_iter, cond_iter)] = max(ratemap_all(cell_iter, 1:341, cond_iter));
    end
end
% %


% % smoothing

if smoothing_window == 0 % no smoothing

else % smoothing with moving average

    for cell_iter = 1 : size(ratemap_all, 1)
        for cond_iter = 1 : 6
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


% sort & display - original
plot_name = {'S-S correct', 'L-S correct', 'S-L correct', 'S-S wrong', 'L-S wrong', 'S-L wrong'};

for cond_iter = 1 : 3

    % sort
    [~, sort_index] = sort(peak_index(:, cond_iter));
    sorted_sort_index = ratemap_all(sort_index, :, :);
    %

    % display
    sheet_position = [100 100 800 400];
    figure('position', sheet_position, 'color', [1 1 1]);

    for iter = 1 : 6
        subplot(1, 6, iter);
        imagesc(sorted_sort_index(:, :, iter));
        colormap('jet');
        clim([0 1.1]);

        hold on
%         plot([87, 87], [1, size(ratemap_all, 1)], '-', 'color', 'r', 'lineWidth', 2);
%         plot([242, 242], [1, size(ratemap_all, 1)], '-', 'color', 'r', 'lineWidth', 2);
%         plot([338, 338], [1, size(ratemap_all, 1)], '-', 'color', 'r', 'lineWidth', 2);
        
        set(gca, 'xtick', 0:31:434, 'xticklabels', {'0','1','2','3','4','5','6','7','8','9','10','11','12','13','14'});
        set(gca, 'ytick', 0:50:size(sorted_sort_index, 1));
        xlabel('Time (sec)');
        ylabel('Cell number');
        title(plot_name{iter});
    end
    %

end
% 

figure
colorbar
colormap('jet');
clim([0 1.1]);
