
% % 2023 Feb 23

clear
close all

session_list = {'20_0519', '25_0630', '30_1017', '31_1029', '33_1025', '35_1025'};

% input_root = 'G:\Hyuwnoo\OneDrive - SNU\Heys lab folder\1. projects\1. tDNMT\1. Analysis\b\1\mat files5\';
input_root = 'G:\Hyuwnoo\OneDrive - SNU\Heys lab folder\1. projects\1. tDNMT\1. Analysis\b\1\mat files9\';
% input_root = 'G:\Hyuwnoo\OneDrive - SNU\Heys lab folder\1. projects\1. tDNMT\1. Analysis\b\1\mat files9_2\';

mean_list = [];
sem_list = [];
real_list = [];
p_list = [];
all_shuffle = [];

for session_iter = 1 : length(session_list)
    load([input_root session_list{session_iter} '.mat'], 'correctness1_real', 'correctness1_shuffle');
    % correctness1: trial type, correctness2: match vs nonmatch
    correctness_real = correctness1_real;
    correctness_shuffle = correctness1_shuffle;
    mean_list(session_iter, 1) = mean(correctness_shuffle);
    sem_list(session_iter, 1) = std(correctness_shuffle) / sqrt(length(correctness_shuffle));
    range_list(session_iter, :) = [prctile(correctness_shuffle, 2.5), prctile(correctness_shuffle, 97.5)];
    real_list(session_iter, 1) = correctness_real;
    p_list(session_iter, 1) = sum(correctness_real <= correctness_shuffle) / length(correctness_shuffle);
    all_shuffle = [all_shuffle, correctness_shuffle];
end

session_colors = [255 54 54; 255 130 36; 196 183 59; 11 201 4; 37 36 255; 135 72 225]/255;

figure;
hold on

for iter = 1 : length(mean_list)
    plot([1 2], [mean_list(iter), real_list(iter)], '.', 'color', session_colors(iter, :));
    plot([1 2], [mean_list(iter), real_list(iter)], 'color', session_colors(iter, :));
    plot([1 1], range_list(iter, :), 'color', session_colors(iter, :));
end
set(gca, 'xlim', [.7 2.3])
set(gca, 'ylim', [.2 1])

% % % % % % 

[p, ~, stat] = ranksum(all_shuffle, real_list)

