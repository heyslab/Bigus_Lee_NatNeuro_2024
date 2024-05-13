
% % 2023 Oct 3

clear
close all

mother_root = 'G:\Hyuwnoo\OneDrive - SNU\Heys lab folder\1. projects\1. tDNMT\';

% % set input

% day1 = {'30_1014', '31_1014', '33_1005', '35_1014', '32_1018', '34_1014', '25_0624'};
% dayN = {'20_0519', '25_0630', '30_1017', '30_1028', '31_1021', '31_1029', '33_1025', '33_1027', '35_1025', ...
%     '30_10281', '30_10282', '33_10271', '33_10272'};

session_id = '35_1025';
% input_root = [mother_root '1. Analysis\all\20221107\'];
% input_root = [mother_root '1. Analysis/all/20221107/'];
input_root = 'G:\Hyuwnoo\OneDrive - SNU\Heys lab folder\1. projects\1. tDNMT\1. Analysis\cell_id\';

% input_file_name = 'day1_all_20221107.csv';
% input_file_name = 'dayN_SS_20221107_4mice.csv';
input_file_name = 'cell_id.csv';

input_root2 = [mother_root '1. Analysis\a\5\parsed_session_reproduce\'];
input_root3 = [mother_root '1. Analysis\a\5\parsed_cell_reproduce\'];
% input_root2 = [mother_root '1. Analysis/a/5/parsed_session/'];
% input_root3 = [mother_root '1. Analysis/a/5/parsed_cell/'];
% % 

% % set output
% output_root = 'G:\Hyuwnoo\OneDrive - SNU\Heys lab folder\1. projects\1. tDNMT\1. Analysis\b\1\mat files9\';
output_root = 'G:\Hyuwnoo\OneDrive - SNU\Heys lab folder\1. projects\1. tDNMT\1. Analysis\b\1\mat files9_2\';
% % 


% % set parameters
% target_window = [8 11] * 31;
target_window = [9 11] * 31;
% target_window = [1/31 2] * 31;

shuffle_N = 1000;
% shuffle_N = 0;

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

% function [] = lda_decoding_function_5b1(cell_list, input_root1, input_root2)

if sum(diff(cell_list(:, 1))) > 0 || sum(diff(cell_list(:, 2))) > 0
    error('It is not an ensemble data');
end

% % load input
mean_fr = [];
peak_loc = [];

load([input_root2 session_id '.mat'], 'parsed_trial');

for cell_iter = 1 : size(cell_list, 1)
    cell_id = [session_id '_' num2str(cell_list(cell_iter, 3))];
    load([input_root3 cell_id '.mat'], 'parsed_flevel_only_shifted2');
    parsed_flevel_only = parsed_flevel_only_shifted2;

    mean_fr(:, cell_iter) = mean(parsed_flevel_only(:, target_window(1) : target_window(2)), 2);
    [~, peak_loc(:, cell_iter)] = max(parsed_flevel_only(:, target_window(1) : target_window(2)), [], 2);

    temp_index = mean_fr(:, cell_iter) == 0;
    peak_loc(temp_index, cell_iter) = 0;
end

% %

% % Run LDA

response_list_real = parsed_trial(:, condition_pt);
% response_list(response_list == 3) = 2;

% remove incorrect trials
temp_index = parsed_trial(:, correctness_pt) == incorrect_type;
mean_fr(temp_index, :) = [];
peak_loc(temp_index, :) = [];
response_list_real(temp_index, :) = [];
%

% remove variance = 0 cells
nan_index = [];
for iter = 1 : size(mean_fr, 2)    
    if var(mean_fr(:, iter)) == 0
        nan_index(end+1) = iter;
    end
end
mean_fr(:, nan_index) = [];
peak_loc(:, nan_index) = [];
% 

% using leave-one-out method

correctness1 = [];
correctness2 = [];
correctness_prob = [];

for shuffle_iter = 1 : shuffle_N + 1

    if shuffle_iter == shuffle_N + 1
        response_list = response_list_real;
    else
        response_list = response_list_real(randperm(length(response_list_real)));
    end

    parfor trial_iter = 1 : length(response_list)

        train_x = mean_fr; train_x(trial_iter, :) = [];
        train_y = response_list; train_y(trial_iter) = [];

        test_x = mean_fr(trial_iter, :);
        test_y = response_list(trial_iter);

        % model = fitcdiscr(train_x, train_y);
        model = fitcecoc(train_x, train_y);
        [label_list(trial_iter, 1), correctness_prob(trial_iter, :), cost] = predict(model, test_x);

        % correctness_mean(trial_iter, :) = correctness_prob(trial_iter, :);
        % [~, temp_index] = max(correctness_mean(trial_iter, :));
        % zero_index = false(size(correctness_mean(trial_iter, :)));
        % zero_index(temp_index) = 1;
        % correctness_mean(trial_iter, zero_index) = 1;
        % correctness_mean(trial_iter, ~zero_index) = 0;

        if test_y == label_list(trial_iter, 1)
            correctness1(trial_iter, shuffle_iter) = 1;
        else
            correctness1(trial_iter, shuffle_iter) = 0;
        end

        if test_y == ss_type
            if label_list(trial_iter, 1) == ss_type
                correctness2(trial_iter, shuffle_iter) = 1;
            else
                correctness2(trial_iter, shuffle_iter) = 0;
            end

        else
            if label_list(trial_iter, 1) == ss_type
                correctness2(trial_iter, shuffle_iter) = 0;
            else
                correctness2(trial_iter, shuffle_iter) = 1;
            end
        end
    end

    disp(['done ' num2str(shuffle_iter)]);
end
%

% % 

% % mean correctness

% correctness_mean = [];
% 
% temp = label_list(response_list == ss_type);
% correctness_mean()

% % 


% % output

correctness1_real = mean(correctness1(:, shuffle_N+1));
correctness1_shuffle = mean(correctness1, 1); correctness1_shuffle(end) = [];
correctness1_p = sum(correctness1_real <= correctness1_shuffle) / shuffle_N;

correctness2_real = mean(correctness2(:, shuffle_N+1));
correctness2_shuffle = mean(correctness2, 1); correctness2_shuffle(end) = [];
correctness2_p = sum(correctness2_real <= correctness2_shuffle) / shuffle_N;

save([output_root session_id '.mat'], 'correctness1', 'correctness2', 'correctness1_real', 'correctness1_shuffle' ...
    , 'correctness1_p', 'correctness2_real', 'correctness2_shuffle', 'correctness2_p');

if 0

behavior_correctness = [];
behavior_correctness(2) = sum(parsed_trial(:, condition_pt) == ss_type & parsed_trial(:, correctness_pt) == correct_type) / sum(parsed_trial(:, condition_pt) == ss_type);
behavior_correctness(3) = sum(parsed_trial(:, condition_pt) == ls_type & parsed_trial(:, correctness_pt) == correct_type) / sum(parsed_trial(:, condition_pt) == ls_type);
behavior_correctness(4) = sum(parsed_trial(:, condition_pt) == sl_type & parsed_trial(:, correctness_pt) == correct_type) / sum(parsed_trial(:, condition_pt) == sl_type);
behavior_correctness(1) = sum(parsed_trial(:, correctness_pt) == correct_type) / size(parsed_trial, 1); % overall

correct_trial_count = [];
correct_trial_count(2) = sum(parsed_trial(:, condition_pt) == ss_type & parsed_trial(:, correctness_pt) == correct_type);
correct_trial_count(3) = sum(parsed_trial(:, condition_pt) == ls_type & parsed_trial(:, correctness_pt) == correct_type);
correct_trial_count(4) = sum(parsed_trial(:, condition_pt) == sl_type & parsed_trial(:, correctness_pt) == correct_type);
correct_trial_count(1) = sum(parsed_trial(:, correctness_pt) == correct_type); % overall

decoding_correctness = [mean(correctness1), mean(correctness2)];

decoding_prob = [];

temp_index = response_list == ss_type;
% decoding_prob(1, :) = mean(correctness_prob(temp_index, :), 1);
decoding_prob(1, :) = mean(correctness_mean(temp_index, :), 1);

temp_index = response_list == ls_type;
% decoding_prob(2, :) = mean(correctness_prob(temp_index, :), 1);
decoding_prob(2, :) = mean(correctness_mean(temp_index, :), 1);

temp_index = response_list == sl_type;
% decoding_prob(3, :) = mean(correctness_prob(temp_index, :), 1);
decoding_prob(3, :) = mean(correctness_mean(temp_index, :), 1);

output_table = [behavior_correctness, correct_trial_count, decoding_correctness, decoding_prob(1,:), decoding_prob(2,:), decoding_prob(3,:)];
% output_table = [decoding_prob(1,:), decoding_prob(2,:), decoding_prob(3,:)];

% % 

end

if 0

% % pca display, method 1

% color setting
ss_color = [255 54 54]/255; % red
sl_color = [18 102 255]/255; % blue
ls_color = [0 183 0]/255; % green
%

% [coef_pca, score_pca, ~, ~, explained_pca, ~] = pca([mean_fr, peak_loc]);
[coef_pca, score_pca, ~, ~, explained_pca, ~] = pca(mean_fr);

figure
hold on
for iter = 1 : size(score_pca, 1)

    if response_list(iter) == ss_type
        plot3(score_pca(iter, 1), score_pca(iter, 2), score_pca(iter, 3), '.', 'color', ss_color)
    elseif response_list(iter) == ls_type
        plot3(score_pca(iter, 1), score_pca(iter, 2), score_pca(iter, 3), '.', 'color', ls_color)
    elseif response_list(iter) == sl_type
        plot3(score_pca(iter, 1), score_pca(iter, 2), score_pca(iter, 3), '.', 'color', sl_color)
    end
end
% % 


% % pca display, method 2

% color setting
ss_color = [255 54 54]/255; % red
sl_color = [18 102 255]/255; % blue
ls_color = [0 183 0]/255; % green
%

temp_index = response_list == ss_type;
% ss_mean = mean([mean_fr(temp_index, :), peak_loc(temp_index, :)], 1);
ss_mean = mean(mean_fr(temp_index, :));

temp_index = response_list == ls_type;
% ls_mean = mean([mean_fr(temp_index, :), peak_loc(temp_index, :)], 1);
ls_mean = mean(mean_fr(temp_index, :));

temp_index = response_list == sl_type;
% sl_mean = mean([mean_fr(temp_index, :), peak_loc(temp_index, :)], 1);
sl_mean = mean(mean_fr(temp_index, :));

[coef_pca, score_pca, ~, ~, explained_pca, ~] = pca([ss_mean; ls_mean; sl_mean]);

% score_ss2 = [mean_fr(response_list == ss_type, :), peak_loc(response_list == ss_type, :)] * coef_pca;
% score_ls2 = [mean_fr(response_list == ls_type, :), peak_loc(response_list == ls_type, :)] * coef_pca;
% score_sl2 = [mean_fr(response_list == sl_type, :), peak_loc(response_list == sl_type, :)] * coef_pca;
score_ss2 = mean_fr(response_list == ss_type, :) * coef_pca;
score_ls2 = mean_fr(response_list == ls_type, :) * coef_pca;
score_sl2 = mean_fr(response_list == sl_type, :) * coef_pca;

score2 = [score_ss2; score_ls2; score_sl2];

score_ss2 = score_ss2 - mean(score2, 1);
score_ls2 = score_ls2 - mean(score2, 1);
score_sl2 = score_sl2 - mean(score2, 1);

figure
hold on
plot(score_ss2(:, 1), score_ss2(:, 2), '.', 'color', ss_color);
plot(score_ls2(:, 1), score_ls2(:, 2), '.', 'color', ls_color);
plot(score_sl2(:, 1), score_sl2(:, 2), '.', 'color', sl_color);

% k means clustering
x_k = [score_ss2; score_ls2; score_sl2];

y_k = [];
y_k(1 : size(score_ss2, 1), 1) = 1;
y_k(end+1 : end + size(score_ls2, 1), 1) = 2;
y_k(end+1 : end + size(score_sl2, 1), 1) = 3;

ind_k = kmeans(x_k, 3);

figure
hold on
color_list = [ss_color; ls_color; sl_color];
for iter = 1 : 3
    plot(x_k(ind_k == iter, 1), x_k(ind_k == iter, 2), '.', 'color', color_list(iter, :));
end
% 

% % 

end