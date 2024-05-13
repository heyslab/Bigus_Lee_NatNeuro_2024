
% % 2023 Feb 6

function fh = cell_ratemap_function9_5a3(cell_id, input_root4, parsed_trial, parsed_flevel_only, smoothing_window)


% % initialization

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


% % get ratemap for each condition & correctness

% load data
% load([input_root4 cell_id '.mat'], 'MI_real', 'MI_p', 'mean_fr');
%

cond_ratemap = {}; % column S-S, L-S, S-L;
% 1st row: correct trials; 2nd row: incorrect trials.

for cond_iter = 1 : 3

    % correct trials
    temp_index = parsed_trial(:, condition_pt) == cond_iter & parsed_trial(:, correctness_pt) == correct_type;
    cond_ratemap{1, cond_iter} = parsed_flevel_only(temp_index, :);
    %

    % incorrect trials
    temp_index = parsed_trial(:, condition_pt) == cond_iter & parsed_trial(:, correctness_pt) == incorrect_type;
    cond_ratemap{2, cond_iter} = parsed_flevel_only(temp_index, :);
    %

end

% %


% % display

condition_name = {'S-S', 'S-L', 'L-S'};
session_name = get_session_type_5zz(cell_id);

sheet_position = [100 100 500 650];
fh = figure('color', [1 1 1], 'position', sheet_position, 'units', 'pixels');

% write figure title
font_size = 15;

subplot('position', pixel_norm_5zz([50 25 400 25], sheet_position))
temp = cell_id; temp(temp == '_') = '-';
text(0, 0.5, temp, 'FontSize', font_size);
text(0.5, 0.5, session_name, 'FontSize', font_size);

axis off;
%

% write condition names
subplot('position', pixel_norm_5zz([50 70 400 25], sheet_position))
text(0.1, 0.5, 'S-S');
text(0.5, 0.5, 'L-S');
text(0.9, 0.5, 'S-L');
axis off;
%

peak_fr = 0;

% 1st row (correct trials. normalized in each plot)
plot_pos_1 = [50 200 100 130];

for cond_iter = 1 : 3

    current_mat = cond_ratemap{1, cond_iter};
    
    if peak_fr < max(max(current_mat))
        peak_fr = max(max(current_mat));
    end

    for iter = 1 : size(current_mat, 1)
        if max(current_mat(iter, :)) > 0
            current_mat(iter, :) = smoothdata(current_mat(iter, :), 'movmean', smoothing_window);
            current_mat(iter, :) = current_mat(iter, :) / max(current_mat(iter, :));            
        end
    end

    temp_pos = plot_pos_1 + [160 0 0 0] * (cond_iter-1);
    subplot('Position', pixel_norm_5zz(temp_pos, sheet_position));
    imagesc(current_mat);
    clim([0 1]);
    colormap('jet');
    hold on
    plot([248 248], [0 size(current_mat, 1)], 'y');
    plot([341 341], [0 size(current_mat, 1)], 'y');
    ylabel('Trial # (correct)');
end
%

% 2nd row (incorrect trials. normalized in each plot)
plot_pos_2 = [50 280 100 60];

for cond_iter = 1 : 3

    current_mat = cond_ratemap{2, cond_iter};
    
    % if peak_fr < max(max(current_mat))
    %     peak_fr = max(max(current_mat));
    % end

    for iter = 1 : size(current_mat, 1)
        if max(current_mat(iter, :)) > 0
            current_mat(iter, :) = smoothdata(current_mat(iter, :), 'movmean', smoothing_window);
            current_mat(iter, :) = current_mat(iter, :) / max(current_mat(iter, :));
        end
    end
    
    temp_pos = plot_pos_2 + [160 0 0 0] * (cond_iter-1);
    subplot('Position', pixel_norm_5zz(temp_pos, sheet_position));
    imagesc(current_mat);
    clim([0 1]);
    hold on
    plot([248 248], [0 size(current_mat, 1)], 'y');
    plot([341 341], [0 size(current_mat, 1)], 'y');
    colormap('jet');
    ylabel('Trial # (wrong)');
end
%

% 3rd row (correct trials. normalized in correct plots)
plot_pos_3 = [50 450 100 130];

for cond_iter = 1 : 3

    current_mat = cond_ratemap{1, cond_iter};
    
    for iter = 1 : size(current_mat, 1)
        current_mat(iter, :) = smoothdata(current_mat(iter, :), 'movmean', smoothing_window);
    end

    if peak_fr > 0
        current_mat = current_mat / peak_fr;
    end

    temp_pos = plot_pos_3 + [160 0 0 0] * (cond_iter-1);
    subplot('Position', pixel_norm_5zz(temp_pos, sheet_position));
    imagesc(current_mat);
    clim([0 1]);
    hold on
    plot([248 248], [0 size(current_mat, 1)], 'y');
    plot([341 341], [0 size(current_mat, 1)], 'y');
    colormap('jet');
    ylabel('Trial # (correct)');
end
%

% 4th row (incorrect trials. normalized in correct plots)
plot_pos_4 = [50 530 100 60];

for cond_iter = 1 : 3

    current_mat = cond_ratemap{2, cond_iter};    
    for iter = 1 : size(current_mat, 1)
        current_mat(iter, :) = smoothdata(current_mat(iter, :), 'movmean', smoothing_window);
    end

    if peak_fr > 0
        current_mat = current_mat / peak_fr;
    end

    temp_pos = plot_pos_4 + [160 0 0 0] * (cond_iter-1);
    subplot('Position', pixel_norm_5zz(temp_pos, sheet_position));
    imagesc(current_mat);
    clim([0 1]);
    hold on
    plot([248 248], [0 size(current_mat, 1)], 'y');
    plot([341 341], [0 size(current_mat, 1)], 'y');
    colormap('jet');
    ylabel('Trial # (wrong)');
end
%

% write mean fr, MI information
plot_pos_5 = [50 630 300 130];

subplot('Position', pixel_norm_5zz(plot_pos_5, sheet_position));
hold on
peak_fr_mean = 0;
for cond_iter = 1 : 3
    plot(smoothdata(mean(cond_ratemap{1, cond_iter}, 1),'movmean', smoothing_window));
    peak_fr_mean = max(peak_fr_mean, max(mean(cond_ratemap{1, cond_iter})));
end

plot([248 248], [0 peak_fr_mean], 'color', 'r')
plot([341 341], [0 peak_fr_mean], 'color', 'r')
legend('S-S', 'L-S', 'S-L', 'location', 'eastoutside')
set(gca, 'xlim', [0 434]);
% for cond_iter = 1 : 3
% 
%     temp_pos = plot_pos_5 + [160 0 0 0] * (cond_iter-1);
%     subplot('Position', pixel_norm_5zz(temp_pos, sheet_position));
%     
%     text(0, 0.9, ['FR: ' num2str(mean_fr(cond_iter))], 'FontSize', font_size);
%     text(0, 0.6, ['MI: ' num2str(MI_real(cond_iter))], 'FontSize', font_size);
%     text(0, 0.3, ['MI p: ' num2str(MI_p(cond_iter))], 'FontSize', font_size);
% 
%     axis off;
% end
%

% %
