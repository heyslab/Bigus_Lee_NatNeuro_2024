

r_trial = [];
r_iti = [];

r_trial = [r_trial; Kendall_r_trial_list];
r_iti = [r_iti; Kendall_r_iti_list];

figure
plot(r_trial, r_iti, '.')
xlabel('r trial')
ylabel('r iti')

[r, p] = corrcoef(r_trial, r_iti)

% % % % % % % % % % % % % % % % % % % % % % % % % 
% % fig 5a

a = [];
b = [];

load_root = 'G:\Hyuwnoo\OneDrive - University of Utah\Heys lab folder\1. projects\1. tDNMT\1. Analysis\a\8\mat files9\';
load_root = 'G:\Hyuwnoo\OneDrive - University of Utah\Heys lab folder\1. projects\1. tDNMT\1. Analysis\a\8\mat files9_shuffle\';

load([load_root '20_0519.mat'])
load([load_root '25_0630.mat'])
load([load_root '30_1017.mat'])
load([load_root '31_1029.mat'])
load([load_root '33_1025.mat'])
load([load_root '35_1025.mat'])

% load data
a = [a; Kendall_r_trial_list];
b = [b; Kendall_r_iti_list];

% a = [a; Kendall_r_SS_list];
% b = [b; Kendall_r_LS_list];

% a = [a; Kendall_r_SS_list];
% b = [b; Kendall_r_SL_list];

% a = [a; Kendall_r_LS_list];
% b = [b; Kendall_r_SL_list];
%

length(a)

figure
scatter(a, b)
plot(a, b, '.')

xlabel('Kendall correlation r (L-S)')
ylabel('Kendall correlation r (S-L)')

set(gca, 'xlim', [-.4 1], 'ylim', [-.4 1])
[r, p] = corrcoef(a, b)
r = r(1, 2), p = p(1, 2);
title(['Day N, all time cells, r = ' num2str(r) ', p = ' num2str(p)])

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % fig 5b - run for every session

a = [];
b = [];

peak_sort = cell_peak_list(peak_sort_index);
peak_diff_list = [];
for iter = 1 : size(cell_pairs_index, 1)
    peak_diff_list(iter, 1) = diff(peak_sort(cell_pairs_index(iter, :)));
end

a = [a; peak_diff_list];
b = [b; Kendall_r_iti_list];

% % % % % % % % % % % % % % % % % % % 

figure
plot(a, b, '.')
xlabel('Difference in peak time (frame)');
ylabel('Pairwise correlation during ITI');

[r, p] = corrcoef(a, b)

[r, p] = corr(a, b, 'Type', 'Kendall')

% % % % % % % % % % % % % % % % % % % 
% % fig 5b - run one time
bin_size = 31;

mean_list = [];
sem_list = [];

for iter = 1 : bin_size : 350

    current_range = [iter iter+bin_size-1];
    temp_index = a >= current_range(1) & a <= current_range(2);
    temp = b(temp_index);

    mean_list(end+1, 1) = mean(temp);
    sem_list(end+1, 1) = std(temp) / sqrt(sum(temp_index));

end
mean_list(end) = [];
sem_list(end) = [];

figure
plot(mean_list, 'color', 'k')
hold on
for iter = 1 : length(mean_list)
%     plot(iter, mean_list(iter), 'o', 'color', 'k')
    plot([iter iter], [mean_list(iter)-sem_list(iter), mean_list(iter)+sem_list(iter)], 'color', 'k');
end
xlabel('Difference in peak time (sec)');
ylabel('Pairwise correlation during ITI');

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 


load([input_root2 session_id_list{cell_iter} '.mat']);

figure
temp = squeeze(sum(masks_NoDups, 1))/5;
temp = temp + squeeze(masks_NoDups(52, :, :)) + squeeze(masks_NoDups(200, :, :));
temp = temp + squeeze(masks_NoDups(125, :, :)) + squeeze(masks_NoDups(245, :, :));
imagesc(temp)



imagesc(squeeze(masks_NoDups(29, :, :)))
figure
imagesc(squeeze(masks_NoDups(39, :, :)))

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% example drawing

plot_range = [98200, 103200];
% plot_range = [22000, 27000];

figure
subplot(2, 1, 1)
plot(f_data1)
hold on
plot(f_data2)
set(gca, 'xlim', plot_range)

for iter = 1 : size(trial_index, 1)
    plot([trial_index(iter, 1) trial_index(iter, 1)], [0 1], 'color', 'r')
    plot([trial_index(iter, 2) trial_index(iter, 2)], [0 1], 'color', 'r')
end

for iter = 1 : size(iti_index, 1)
    plot([iti_index(iter, 1) iti_index(iter, 1)], [0 1], 'color', 'k')
    plot([iti_index(iter, 2) iti_index(iter, 2)], [0 1], 'color', 'k')
end

subplot(2, 1, 2)
plot(pico_data(:, odor_pd))
hold on
% plot(pico_data(:, licking_pd))
plot(pico_data(:, reward_pd))
set(gca, 'xlim', plot_range)


figure
subplot(3, 1, 1)
plot(f_data1_trial)
hold on
plot(f_data2_trial)
set(gca, 'xlim', plot_range)

subplot(3, 1, 2)
plot(f_data1_iti)
hold on
plot(f_data2_iti)
set(gca, 'xlim', plot_range)

subplot(3, 1, 3)
plot(pico_data(:, odor_pd))
hold on
% plot(pico_data(:, licking_pd))
plot(pico_data(:, reward_pd))
set(gca, 'xlim', plot_range)



figure
plot(f_data1_iti)
hold on
plot(f_data2_iti)

f_data1_trial_norm = f_data1_trial / max(f_data1_trial);
f_data2_trial_norm = f_data2_trial / max(f_data2_trial);

f_data1_iti_norm = f_data1_iti / max(f_data1_iti);
f_data2_iti_norm = f_data2_iti / max(f_data2_iti);

figure
imagesc([f_data1_trial_norm, f_data2_trial_norm]')
clim([0 1]);
colormap('hot')

figure
imagesc([f_data1_iti_norm, f_data2_iti_norm]')
clim([0 1]);
colormap('hot')


figure
imagesc([f_data1_trial_norm, f_data2_trial_norm, f_data1_iti_norm, f_data2_iti_norm]')
clim([0 1]);
colormap('hot')

figure
plot(f_data1_trial_norm)
hold on
plot(f_data2_trial_norm)

plot(pico_data(:, odor_pd))
plot(pico_data(:, reward_pd))

figure
plot(f_data1)
hold on
plot(f_data2)