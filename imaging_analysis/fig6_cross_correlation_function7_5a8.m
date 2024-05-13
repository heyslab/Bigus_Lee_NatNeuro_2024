
% % 2023 Mar 2

function [Kendall_r_trial, Kendall_p_trial, Kendall_r_iti, Kendall_p_iti] = cross_correlation_function7_5a8(cell_id1, cell_id2, pico_data, parsed_trial, f_data1, f_data2, time_bin, frame_rate)

% use follow for shuffle data
% shifting_amount = randi(length(f_data2), 1) - 1; % 0 ~ frame_length-1
% 
% temp = [];
% temp(1+shifting_amount : length(f_data2)) = f_data2(1 : length(f_data2) - shifting_amount);
% temp(1 : shifting_amount) = f_data2(length(f_data2) - shifting_amount + 1 : length(f_data2));
% f_data2 = temp;
% 

% % initialization

% column index for pico data
reward_pd = 1;
tpsignal_pd = 2;
ttl_pd = 3;
buzzer_pd = 4;
licking_pd = 5;
light_pd = 6;
velocity_pd = 7;
odor_pd = 8;
timestamp_pd = 9;
% 

% column index for parsed_trial
trialn_pt = 1;
condition_pt = 2;
correctness_pt = 3;
%

ss_type = 1;
ls_type = 2;
sl_type = 3;

correct_type = 1;
incorrect_type = 2;
% % 

% % pico data parsing

% % trial parsing

ttl_index = find(pico_data(:, ttl_pd) == 1);

trial_index = [];
iti_index = [];
iter = 1;

temp_odor_index = [];
while iter < length(ttl_index)

    if ttl_index(iter+1) - ttl_index(iter) < 5
        iter = iter+1;
        continue;
    end

    temp_index = pico_data(:, ttl_pd) == 1;

    current_range = [];
    current_range(1 : size(pico_data, 1)) = false;
    current_range(ttl_index(iter) : ttl_index(iter+1)) = true;
    current_range = logical(current_range);

    temp_light = bwlabel(pico_data(current_range, light_pd));
    temp_odor = bwlabel(pico_data(current_range, odor_pd));

    if max(temp_light) == 1 && max(temp_odor) == 2
        current_odor_start = find(pico_data(current_range, odor_pd) == 1, 1, 'first');
        current_odor_end = find(pico_data(current_range, odor_pd) == 1, 1, 'last');
        
        trial_index(end+1, 1) = ttl_index(iter) + current_odor_start - 31;
        trial_index(end, 2) = trial_index(end, 1) + 340;
        
        if size(trial_index, 1) > 1
            iti_index(end+1, 1) = trial_index(end-1, 2) + 31*5;
            iti_index(end, 2) = trial_index(end, 1) - 31*3;

            if iti_index(end, 2) < iti_index(end, 1)
                error('!!');
            end
        end
    end

    iter = iter+1;
end
% %

% remove incorrect trials
if size(trial_index, 1) ~= size(parsed_trial, 1)
    error('!');
end

temp_index = parsed_trial(:, correctness_pt) == incorrect_type;
trial_index(temp_index, :) = [];

% temp_index = parsed_trial(:, correctness_pt) == correct_type & parsed_trial(:, condition_pt) == sl_type;
% trial_index(~temp_index, :) = [];

% 

% % cross correlogram

% make f data
f_data1_trial = nan(size(f_data1));
f_data2_trial = nan(size(f_data1));
f_data1_iti = nan(size(f_data1));
f_data2_iti = nan(size(f_data1));

for trial_iter = 1 : size(trial_index, 1)
    f_data1_trial(trial_index(trial_iter, 1) : trial_index(trial_iter, 2)) = f_data1(trial_index(trial_iter, 1) : trial_index(trial_iter, 2));
    f_data2_trial(trial_index(trial_iter, 1) : trial_index(trial_iter, 2)) = f_data2(trial_index(trial_iter, 1) : trial_index(trial_iter, 2));
end

for trial_iter = 1 : size(iti_index, 1)
    f_data1_iti(iti_index(trial_iter, 1) : iti_index(trial_iter, 2)) = f_data1(iti_index(trial_iter, 1) : iti_index(trial_iter, 2));
    f_data2_iti(iti_index(trial_iter, 1) : iti_index(trial_iter, 2)) = f_data2(iti_index(trial_iter, 1) : iti_index(trial_iter, 2));
end
% 

% compute cofiring
cofiring_trial = [];
cofiring_iti = [];

% for trial epoch
bin_size = round(frame_rate * time_bin * 0.001);
current_bin = 1;

temp_flag = true;
while temp_flag

    if current_bin + bin_size - 1 > length(f_data1_trial)
        if sum(isnan(f_data1_trial(current_bin : end))) == length(f_data1_trial(current_bin : end))
            cofiring_trial(end+1, 1) = nan;
            cofiring_trial(end, 2) = nan;
        else
            cofiring_trial(end+1, 1) = sum(f_data1_trial(current_bin : end), 'omitnan');
            cofiring_trial(end, 2) = sum(f_data2_trial(current_bin : end), 'omitnan');
        end
    else
        if sum(isnan(f_data1_trial(current_bin : current_bin + bin_size -1))) == bin_size
            cofiring_trial(end+1, 1) = nan;
            cofiring_trial(end, 2) = nan;
        else            
            cofiring_trial(end+1, 1) = sum(f_data1_trial(current_bin : current_bin + bin_size -1), 'omitnan');
            cofiring_trial(end, 2) = sum(f_data2_trial(current_bin : current_bin + bin_size -1), 'omitnan');
        end
    end

    current_bin = current_bin + bin_size;

    if current_bin > length(f_data1_trial)
        temp_flag = false;
    end
end

cofiring_trial(isnan(cofiring_trial(:, 1)), :) = [];
% 

% for iti epoch
bin_size = round(frame_rate * time_bin * 0.001);
current_bin = 1;

temp_flag = true;
while temp_flag

    if current_bin + bin_size - 1 > length(f_data1_iti)
        if sum(isnan(f_data1_iti(current_bin : end))) == length(f_data1_iti(current_bin : end))
            cofiring_iti(end+1, 1) = nan;
            cofiring_iti(end, 2) = nan;
        else
            cofiring_iti(end+1, 1) = sum(f_data1_iti(current_bin : end), 'omitnan');
            cofiring_iti(end, 2) = sum(f_data2_iti(current_bin : end), 'omitnan');
        end
    else
        if sum(isnan(f_data1_iti(current_bin : current_bin + bin_size -1))) == bin_size
            cofiring_iti(end+1, 1) = nan;
            cofiring_iti(end, 2) = nan;
        else
            cofiring_iti(end+1, 1) = sum(f_data1_iti(current_bin : current_bin + bin_size -1), 'omitnan');
            cofiring_iti(end, 2) = sum(f_data2_iti(current_bin : current_bin + bin_size -1), 'omitnan');
        end
    end

    current_bin = current_bin + bin_size;

    if current_bin > length(f_data1_iti)
        temp_flag = false;
    end
end

cofiring_iti(isnan(cofiring_iti(:, 1)), :) = [];
% 

% compute correlation
[Kendall_r_trial, Kendall_p_trial] = corr(cofiring_trial(:, 1), cofiring_trial(:, 2), 'Type', 'Kendall');
[Kendall_r_iti, Kendall_p_iti] = corr(cofiring_iti(:, 1), cofiring_iti(:, 2), 'Type', 'Kendall');
% 



% % display
% cross_correlation_display3_5a8;
% % 



