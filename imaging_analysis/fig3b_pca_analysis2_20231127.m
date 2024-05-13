
% % 2023 Dec 10

clear
close all

mother_root = 'G:\Hyuwnoo\OneDrive - University of Utah\Heys lab folder\1. projects\1. tDNMT\';

% % set input

% day1 = {'30_1014', '31_1014', '33_1005', '35_1014', '32_1018', '34_1014', '25_0624'};
% dayN = {'20_0519', '25_0630', '30_1017', '30_1028', '31_1021', '31_1029', '33_1025', '33_1027', '35_1025', ...
%     '30_10281', '30_10282', '33_10271', '33_10272'};

% session_id = '25_0630';
% input_root = 'G:\Hyuwnoo\OneDrive - SNU\Heys lab folder\1. projects\1. tDNMT\1. Analysis\cell_id\';
input_root = 'G:\Hyuwnoo\OneDrive - University of Utah\Heys lab folder\1. projects\1. tDNMT\1. Analysis\all\20231127 temp working\4. pca analysis\';

% input_file_name = 'dayN time cells.csv';
% input_file_name = 'dayN time cells (4mice).csv';
input_file_name = 'day1 time cells.csv';

input_root2 = [mother_root '1. Analysis\a\5\parsed_session_reproduce\'];
input_root3 = [mother_root '1. Analysis\a\5\parsed_cell_reproduce\'];
% input_root2 = [mother_root '1. Analysis/a/5/parsed_session/'];
% input_root3 = [mother_root '1. Analysis/a/5/parsed_cell/'];
% % 

% % set output
% output_root = 'G:\Hyuwnoo\OneDrive - SNU\Heys lab folder\1. projects\1. tDNMT\1. Analysis\b\1\mat files9\';
% output_root = 'G:\Hyuwnoo\OneDrive - SNU\Heys lab folder\1. projects\1. tDNMT\1. Analysis\b\1\mat files9_2\';
% % 


% % set parameters
% bin_size = 1; % unit = frame. 1 frames ~ 32 ms
% bin_size = 8; % unit = frame. 8 frames ~ 258 ms
% bin_size = 16; % unit = frame. 8 frames ~ 258 ms
bin_size = 31; % unit = frame. 8 frames ~ 258 ms
% % 

addpath(genpath('G:\Hyuwnoo\OneDrive - SNU\Heys lab folder\1. projects\1. tDNMT\2. Analysis programs'));

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

ratemap_all = []; % 3d matrix. cell # x frame # x condition

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
        % temp_index = parsed_trial(:, condition_pt) == cond_iter & parsed_trial(:, correctness_pt) == correct_type;
        temp_index = parsed_trial(:, condition_pt) == cond_iter;
        temp = mean(parsed_flevel_only(temp_index, :), 1);
        ratemap_all(cell_iter, :, cond_iter) = temp;
        %
    end
    %

end
% %

% % Binning
temp = [];
for iter = 1 : ceil(size(ratemap_all, 2) / bin_size)

    if iter == ceil(size(ratemap_all, 2) / bin_size)
        current_range = [(iter-1) * bin_size+1, size(ratemap_all, 2)];
    else
        current_range = [(iter-1) * bin_size+1, iter * bin_size];
    end

    temp(:, iter, :) = mean(ratemap_all(:, current_range(1) : current_range(2), :), 2);
end

ratemap_all_binned = temp;
% % 


% % run pca

% get pca axis for all trials
mean_fr_all = squeeze(mean(ratemap_all_binned, 3))'; % 2d matrix. bin # x cell #

mean_fr_ss = ratemap_all_binned(:, :, ss_type)';
mean_fr_ls = ratemap_all_binned(:, :, ls_type)';
mean_fr_sl = ratemap_all_binned(:, :, sl_type)';


% normalization
for cell_iter = 1 : size(mean_fr_all, 2)

    peak_fr = max(mean_fr_all(:, cell_iter));

    mean_fr_ss(:, cell_iter) = mean_fr_ss(:, cell_iter) / peak_fr;
    mean_fr_ls(:, cell_iter) = mean_fr_ls(:, cell_iter) / peak_fr;
    mean_fr_sl(:, cell_iter) = mean_fr_sl(:, cell_iter) / peak_fr;
    mean_fr_all(:, cell_iter) = mean_fr_all(:, cell_iter) / peak_fr;
end
% 

[coef_all, score_all, ~, ~, explained, ~] = pca(mean_fr_all);

score_ss = mean_fr_ss * coef_all;
score_sl = mean_fr_sl * coef_all;
score_ls = mean_fr_ls * coef_all;
% 

% get explained proportion
total_var_ss = 0;
total_var_sl = 0;
total_var_ls = 0;
total_var_all = 0;

for iter = 1 : size(mean_fr_ss, 2)
    total_var_ss = total_var_ss + var(mean_fr_ss(:, iter));
    total_var_sl = total_var_sl + var(mean_fr_sl(:, iter));
    total_var_ls = total_var_ls + var(mean_fr_ls(:, iter));
    total_var_all = total_var_all + var(mean_fr_all(:, iter));
end

explained_ss = [];
explained_sl = [];
explained_ls = [];
explained_all = [];

% for iter = 1 : size(mean_fr_ss, 2)
for iter = 1 : size(score_ss, 2)
    explained_ss(iter, 1) = var(score_ss(:,iter)) / total_var_ss * 100;
    explained_sl(iter, 1) = var(score_sl(:,iter)) / total_var_sl * 100;
    explained_ls(iter, 1) = var(score_ls(:,iter)) / total_var_ls * 100;
    explained_all(iter, 1) = var(score_all(:,iter)) / total_var_all * 100;
end
%

% smoothed score values
% smoothing_window = 10;
smoothing_window = 1;
score_ss_smooth = smoothdata(score_ss, 1, 'movmean', smoothing_window);
score_sl_smooth = smoothdata(score_sl, 1, 'movmean', smoothing_window);
score_ls_smooth = smoothdata(score_ls, 1, 'movmean', smoothing_window);
%

% % 

% % display

% color setting
ss_color = [255 54 54]/255; % red
sl_color = [18 102 255]/255; % blue
ls_color = [0 183 0]/255; % green
%


% smoothed trajectory

% set colors (CYMK)
% condition_color = [0 49.22 97.66 0; 1.17 98.05 1.56 0; 69.14 14.84 0 0]; % SS, LS, SL
% ss_color = condition_color(1, :);
% ls_color = condition_color(2, :);
% sl_color = condition_color(3, :);
% 
% gradual_ss_color = [];
% gradual_ls_color = [];
% gradual_sl_color = [];
% 
% for iter = 1 : 4
%     gradual_ss_color(:, iter) = linspace(ss_color(iter), 0, size(score_ss_smooth, 1));
%     gradual_ls_color(:, iter) = linspace(ls_color(iter), 0, size(score_ss_smooth, 1));
%     gradual_sl_color(:, iter) = linspace(sl_color(iter), 0, size(score_ss_smooth, 1));
% end
% 

% set colors (RGB)
condition_color = [248 149 34; 233 22 139; 45 171 226]/255;
ss_color = condition_color(1, :);
ls_color = condition_color(2, :);
sl_color = condition_color(3, :);

gradual_ss_color = [];
gradual_ls_color = [];
gradual_sl_color = [];
for iter = 1 : 3
    gradual_ss_color(:, iter) = ss_color(iter) : (0.9 - ss_color(iter)) / (size(score_ss_smooth, 1)-1) : 0.9;
    gradual_ls_color(:, iter) = ls_color(iter) : (0.9 - ls_color(iter)) / (size(score_ss_smooth, 1)-1) : 0.9;
    gradual_sl_color(:, iter) = sl_color(iter) : (0.9 - sl_color(iter)) / (size(score_ss_smooth, 1)-1) : 0.9;
end
% 

% plot color scale bar
temp = [];

figure
subplot(1, 3, 1)
temp(:, 1, :) = gradual_ss_color;
imshow(temp)
title('S-S')

subplot(1, 3, 2)
temp(:, 1, :) = gradual_sl_color;
imshow(temp)
title('S-L')

subplot(1, 3, 3)
temp(:, 1, :) = gradual_ls_color;
imshow(temp)
title('L-S')
% 

figure('color', [1 1 1])
hold on


if 1
linewidth = 3;
markersize = 13;
linewidth2 = 1.5;
for bin_iter = 1 : size(score_ss_smooth, 1)

    if bin_iter < size(score_ss_smooth, 1)
        plot([score_ss_smooth(bin_iter : bin_iter+1, 1)], [score_ss_smooth(bin_iter : bin_iter+1, 2)], ...
            '-', 'color', gradual_ss_color(bin_iter, :), 'LineWidth', linewidth2)
        plot([score_ls_smooth(bin_iter : bin_iter+1, 1)], [score_ls_smooth(bin_iter : bin_iter+1, 2)], ...
            '-', 'color', gradual_ls_color(bin_iter, :), 'LineWidth', linewidth2)
        plot([score_sl_smooth(bin_iter : bin_iter+1, 1)], [score_sl_smooth(bin_iter : bin_iter+1, 2)], ...
            '-', 'color', gradual_sl_color(bin_iter, :), 'LineWidth', linewidth2)
    end

    plot(score_ss_smooth(bin_iter, 1), score_ss_smooth(bin_iter, 2), '.', ...
        'MarkerFaceColor', gradual_ss_color(bin_iter, :), 'MarkerEdgeColor', gradual_ss_color(bin_iter, :), ...
        'LineWidth', linewidth, 'MarkerSize', markersize);

    plot(score_ls_smooth(bin_iter, 1), score_ls_smooth(bin_iter, 2), '.', ...
        'MarkerFaceColor', gradual_ls_color(bin_iter, :), 'MarkerEdgeColor', gradual_ls_color(bin_iter, :), ...
        'LineWidth', linewidth, 'MarkerSize', markersize);

    plot(score_sl_smooth(bin_iter, 1), score_sl_smooth(bin_iter, 2), '.', ...
        'MarkerFaceColor', gradual_sl_color(bin_iter, :), 'MarkerEdgeColor', gradual_sl_color(bin_iter, :), ...
        'LineWidth', linewidth, 'MarkerSize', markersize);

    % plot(score_ss_smooth(bin_iter, 1), score_ss_smooth(bin_iter, 2), 'o', ...
    %     'MarkerFaceColor', gradual_ss_color(bin_iter, :), 'MarkerEdgeColor', gradual_ss_color(bin_iter, :), ...
    %     'LineWidth', linewidth, 'MarkerSize', markersize);
    % 
    % plot(score_ls_smooth(bin_iter, 1), score_ls_smooth(bin_iter, 2), 'o', ...
    %     'MarkerFaceColor', gradual_ls_color(bin_iter, :), 'MarkerEdgeColor', gradual_ls_color(bin_iter, :), ...
    %     'LineWidth', linewidth, 'MarkerSize', markersize);
    % 
    % plot(score_sl_smooth(bin_iter, 1), score_sl_smooth(bin_iter, 2), 'o', ...
    %     'MarkerFaceColor', gradual_sl_color(bin_iter, :), 'MarkerEdgeColor', gradual_sl_color(bin_iter, :), ...
    %     'LineWidth', linewidth, 'MarkerSize', markersize);
end
end

if 0
linewidth = 3;
markersize = 15;
linewidth2 = 5;
for bin_iter = 1 : size(score_ss_smooth, 1)

    if bin_iter < size(score_ss_smooth, 1)
        plot([score_ss_smooth(bin_iter : bin_iter+1, 1)], [score_ss_smooth(bin_iter : bin_iter+1, 2)], ...
            '-', 'color', gradual_color(bin_iter, :), 'LineWidth', linewidth2)
        plot([score_ls_smooth(bin_iter : bin_iter+1, 1)], [score_ls_smooth(bin_iter : bin_iter+1, 2)], ...
            '-', 'color', gradual_color(bin_iter, :), 'LineWidth', linewidth2)
        plot([score_sl_smooth(bin_iter : bin_iter+1, 1)], [score_sl_smooth(bin_iter : bin_iter+1, 2)], ...
            '-', 'color', gradual_color(bin_iter, :), 'LineWidth', linewidth2)
    end

    plot(score_ss_smooth(bin_iter, 1), score_ss_smooth(bin_iter, 2), 'o', ...
        'MarkerFaceColor', condition_color(ss_type, :), 'MarkerEdgeColor', gradual_color(bin_iter, :), ...
        'LineWidth', linewidth, 'MarkerSize', markersize);

    plot(score_ls_smooth(bin_iter, 1), score_ls_smooth(bin_iter, 2), 'o', ...
        'MarkerFaceColor', condition_color(ls_type, :), 'MarkerEdgeColor', gradual_color(bin_iter, :), ...
        'LineWidth', linewidth, 'MarkerSize', markersize);

    plot(score_sl_smooth(bin_iter, 1), score_sl_smooth(bin_iter, 2), 'o', ...
        'MarkerFaceColor', condition_color(sl_type, :), 'MarkerEdgeColor', gradual_color(bin_iter, :), ...
        'LineWidth', linewidth, 'MarkerSize', markersize);
end
end

if 0
plot(score_ss_smooth(:, 1), score_ss_smooth(:, 2), '-', 'color', ss_color)
plot(score_sl_smooth(:, 1), score_sl_smooth(:, 2), '-', 'color', sl_color)
plot(score_ls_smooth(:, 1), score_ls_smooth(:, 2), '-', 'color', ls_color)

linewidth = 1.5;
markersize = 10;
for bin_iter = 1 : size(score_ss_smooth, 1)

    plot(score_ss_smooth(bin_iter, 1), score_ss_smooth(bin_iter, 2), 'o', ...
        'MarkerEdgeColor', condition_color(ss_type, :), 'MarkerFaceColor', gradual_color(bin_iter, :), ...
        'LineWidth', linewidth, 'MarkerSize', markersize);

    plot(score_ls_smooth(bin_iter, 1), score_ls_smooth(bin_iter, 2), 'o', ...
        'MarkerEdgeColor', condition_color(ls_type, :), 'MarkerFaceColor', gradual_color(bin_iter, :), ...
        'LineWidth', linewidth, 'MarkerSize', markersize);

    plot(score_sl_smooth(bin_iter, 1), score_sl_smooth(bin_iter, 2), 'o', ...
        'MarkerEdgeColor', condition_color(sl_type, :), 'MarkerFaceColor', gradual_color(bin_iter, :), ...
        'LineWidth', linewidth, 'MarkerSize', markersize);
end
end


% start point
plot(score_ss_smooth(1,1), score_ss_smooth(1,2), '.', 'markersize', 30, 'color', ss_color)
plot(score_sl_smooth(1,1), score_sl_smooth(1,2), '.', 'markersize', 30, 'color', sl_color)
plot(score_ls_smooth(1,1), score_ls_smooth(1,2), '.', 'markersize', 30, 'color', ls_color)

text(score_ss_smooth(1,1), score_ss_smooth(1,2), '  start', 'color', ss_color);
text(score_sl_smooth(1,1), score_sl_smooth(1,2), '  start', 'color', sl_color);
text(score_ls_smooth(1,1), score_ls_smooth(1,2), '  start', 'color', ls_color);
%

% frame 93
current_bin = round(93/bin_size);
plot(score_ss_smooth(current_bin,1), score_ss_smooth(current_bin,2), '.', 'markersize', 15, 'color', ss_color)
plot(score_sl_smooth(current_bin,1), score_sl_smooth(current_bin,2), '.', 'markersize', 15, 'color', sl_color)
plot(score_ls_smooth(current_bin,1), score_ls_smooth(current_bin,2), '.', 'markersize', 15, 'color', ls_color)

text(score_ss_smooth(current_bin,1), score_ss_smooth(current_bin,2), '  3', 'color', ss_color);
text(score_sl_smooth(current_bin,1), score_sl_smooth(current_bin,2), '  3', 'color', sl_color);
text(score_ls_smooth(current_bin,1), score_ls_smooth(current_bin,2), '  3', 'color', ls_color);
%

% frame 248
current_bin = round(248/bin_size);
plot(score_ss_smooth(current_bin,1), score_ss_smooth(current_bin,2), '.', 'markersize', 15, 'color', ss_color)
plot(score_sl_smooth(current_bin,1), score_sl_smooth(current_bin,2), '.', 'markersize', 15, 'color', sl_color)
plot(score_ls_smooth(current_bin,1), score_ls_smooth(current_bin,2), '.', 'markersize', 15, 'color', ls_color)

text(score_ss_smooth(current_bin,1), score_ss_smooth(current_bin,2), '  8', 'color', ss_color);
text(score_sl_smooth(current_bin,1), score_sl_smooth(current_bin,2), '  8', 'color', sl_color);
text(score_ls_smooth(current_bin,1), score_ls_smooth(current_bin,2), '  8', 'color', ls_color);
% 

% frame 341
current_bin = round(341/bin_size);
plot(score_ss_smooth(current_bin,1), score_ss_smooth(current_bin,2), '.', 'markersize', 15, 'color', ss_color)
plot(score_sl_smooth(current_bin,1), score_sl_smooth(current_bin,2), '.', 'markersize', 15, 'color', sl_color)
plot(score_ls_smooth(current_bin,1), score_ls_smooth(current_bin,2), '.', 'markersize', 15, 'color', ls_color)

text(score_ss_smooth(current_bin,1), score_ss_smooth(current_bin,2), '  11', 'color', ss_color);
text(score_sl_smooth(current_bin,1), score_sl_smooth(current_bin,2), '  11', 'color', sl_color);
text(score_ls_smooth(current_bin,1), score_ls_smooth(current_bin,2), '  11', 'color', ls_color);
%

% frame end
current_bin = size(score_ss_smooth, 1);
plot(score_ss_smooth(current_bin,1), score_ss_smooth(current_bin,2), '.', 'markersize', 15, 'color', ss_color)
plot(score_sl_smooth(current_bin,1), score_sl_smooth(current_bin,2), '.', 'markersize', 15, 'color', sl_color)
plot(score_ls_smooth(current_bin,1), score_ls_smooth(current_bin,2), '.', 'markersize', 15, 'color', ls_color)

text(score_ss_smooth(current_bin,1), score_ss_smooth(current_bin,2), '  end', 'color', ss_color);
text(score_sl_smooth(current_bin,1), score_sl_smooth(current_bin,2), '  end', 'color', sl_color);
text(score_ls_smooth(current_bin,1), score_ls_smooth(current_bin,2), '  end', 'color', ls_color);
%

title('smoothed trajectories');
legend('S-S', 'L-S', 'S-L')
grid on
%

% % 
