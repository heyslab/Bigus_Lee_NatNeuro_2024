


ss_1 = [];
ls_1 = [];
sl_1 = [];

ss_n = [];
ls_n = [];
sl_n = [];

% load('G:\Hyuwnoo\OneDrive - SNU\Heys lab folder\1. projects\1. tDNMT\1. Analysis\all\20221108_3\data_20221108_3.mat', 'ss_1', 'ls_1', 'sl_1', 'ss_n', 'ls_n', 'sl_n');

ss_1 = ss_1 / 30.98;
ss_n = ss_n / 30.98;
ls_1 = ls_1 / 30.98;
ls_n = ls_n / 30.98;
sl_1 = sl_1 / 30.98;
sl_n = sl_n / 30.98;

day1 = [ss_1; ls_1; sl_1];
dayn = [ss_n; ls_n; sl_n];

ss = [ss_1; ss_n];
ls = [ls_1; ls_n];
sl = [sl_1; sl_n];
% 

[p, ~, stat] = ranksum(ss_1, ss_n)
[p, ~, stat] = ranksum(ls_1, ls_n)
[p, ~, stat] = ranksum(sl_1, sl_n)
[p, ~, stat] = ranksum(day1, dayn)

[~, p, ~, stat] = ttest2(ss_1, ss_n)
[~, p, ~, stat] = ttest2(ls_1, ls_n)
[~, p, ~, stat] = ttest2(sl_1, sl_n)

mean(day1) 
std(day1) / sqrt(length(day1))

mean(dayn) 
std(dayn) / sqrt(length(dayn))

mean(ss_1) 
std(ss_1) / sqrt(length(ss_1))
median(ss_1)

mean(ss_n)
std(ss_n) / sqrt(length(ss_n))
median(ss_n)

mean(ls_1) 
std(ls_1) / sqrt(length(ls_1))

mean(ls_n)
std(ls_n) / sqrt(length(ls_n))

mean(sl_1) 
std(sl_1) / sqrt(length(sl_1))
median(sl_1)

mean(sl_n)
std(sl_n) / sqrt(length(sl_n))
median(sl_n)



[~, p, stat] = kstest2(ss_1, ss_n)

sum(day1 < 1)
length(day1)
prop1 = sum(day1 < 1) / length(day1)

pie([prop, 1-prop])

sum(dayn < 1)
length(dayn)
prop2 = sum(dayn < 1) / length(dayn)

bar([prop1 prop2])

[p, ~, stat] = ranksum(sl_n, ls_n)
[p, ~, stat] = ranksum(sl_1, ls_1)
[p, ~, stat] = ranksum(sl_1, ss_1)

% % % % % % % % % % % % % % % % % % % % % % 

% hist_edge = 0 : 1 : 14;
hist_edge = 0 : 1 : 12;

figure('color', [1 1 1])
histogram(ss_1, hist_edge, 'normalization', 'probability');
hold on
histogram(ss_n, hist_edge, 'normalization', 'probability');
title('peak SS')
%set(gca, 'ylim', [0 0.6]);

figure('color', [1 1 1])
histogram(ls_1, hist_edge, 'normalization', 'probability');
hold on
histogram(ls_n, hist_edge, 'normalization', 'probability');
title('peak LS')
%set(gca, 'ylim', [0 0.6]);

figure('color', [1 1 1])
histogram(sl_1, hist_edge, 'normalization', 'probability');
hold on
histogram(sl_n, hist_edge, 'normalization', 'probability');
title('peak SL')
%set(gca, 'ylim', [0 0.6]);

figure('color', [1 1 1])
histogram(day1, hist_edge, 'normalization', 'probability');
hold on
histogram(dayn, hist_edge, 'normalization', 'probability');
title('peak all')
%set(gca, 'ylim', [0 0.6]);


figure('color', [1 1 1])
histogram(ss, hist_edge, 'normalization', 'probability');
title('SS')

figure('color', [1 1 1])
histogram(ls, hist_edge, 'normalization', 'probability');
title('LS')

figure('color', [1 1 1])
histogram(sl, hist_edge, 'normalization', 'probability');
title('SL')

hold on
histogram(dayn, hist_edge, 'normalization', 'probability');
title('peak all')