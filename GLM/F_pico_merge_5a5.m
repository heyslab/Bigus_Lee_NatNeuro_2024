
%% 2022 Oct 24

clear;

%% Set input

% session id
session_id = '30_1030';
% 

% load pico data
pico_root = 'F:\E30\20221030';
pico_file_names = {'10.30.E30SS_1.mat', '10.30.E30SS_2.mat'}; % for manual selection
% pico_file_name = '10.14.E35SS';   % for auto selection
% 

% load F data
f_root = 'F:\E30\20221030';
f_file_name = 'E30_1030_Post_Fc3.mat';
% 



%% Set output
output_root = 'G:\Hyuwnoo\OneDrive - SNU\Heys lab folder\1. projects\1. tDNMT\1. Analysis\a\5';
% output_file_name = 'E32_1018_Post_Pico new.mat';
% % 

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% pico data structure
% 	▪ A: water (reward)
% 	▪ B: 2p signal
% 	▪ C: TTL pulse
% 	▪ D: buzzer
% 	▪ E: licking
% 	▪ F: green light
% 	▪ G: velocity
% 	▪ H: odor
% % 

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

% column index for parsed_trial, parsed_behavior
trialn_pt = 1;
condition_pt = 2;
correctness_pt = 3;

licking_pb = 1;
odor_pb = 2;
reward_pb = 3;
velocity_pb = 4;
%

% condition type, correctness index
ss_type = 1;
ls_type = 2;
sl_type = 3;

correct_type = 1;
incorrect_type = 2;
%

%
response_window = 93; % 93 frames = 3 seconds
%

% % 

% % load data

% load pico data
if 0
    temp = dir(pico_root);

    pico_file_names = {};
    for iter = 1 : size(temp, 1)
        if strfind(temp(iter).name, pico_file_name)
            pico_file_names{end+1} = temp(iter).name;
        end
    end

    disp('Found following pico files: ');
end
disp(pico_file_names);

raw_pico_data = [];
pico_index = 1;

for iter = 1 : size(pico_file_names, 2)

    load([pico_root '\' pico_file_names{iter}])
    pico_length = Length;
    pico_sampling_interval = Tinterval;

    raw_pico_data(pico_index : pico_index + pico_length-1, reward_pd) = A;
    raw_pico_data(pico_index : pico_index + pico_length-1, tpsignal_pd) = B;
    raw_pico_data(pico_index : pico_index + pico_length-1, ttl_pd) = C;
    raw_pico_data(pico_index : pico_index + pico_length-1, buzzer_pd) = D;
    raw_pico_data(pico_index : pico_index + pico_length-1, licking_pd) = E;
    raw_pico_data(pico_index : pico_index + pico_length-1, light_pd) = F;
    raw_pico_data(pico_index : pico_index + pico_length-1, velocity_pd) = G;
    raw_pico_data(pico_index : pico_index + pico_length-1, odor_pd) = H;

    pico_index = pico_index + pico_length;
end

raw_pico_data(1:end, timestamp_pd) = 0 : pico_sampling_interval : pico_sampling_interval * (size(raw_pico_data, 1) - 1);
% 

% load F data
load([f_root '\' f_file_name], 'Fc3_DF');

f_data = Fc3_DF;
% 

% % 


% % get pico 2p signal & downsampling pico data

% filtering 2p signal & get beginning and end timestamp of recording
temp = raw_pico_data(:, tpsignal_pd);
temp(temp < 4) = 0;
temp(temp >= 4) = 1;

temp_index = find(temp == 1);
temp_gap = diff(temp_index) * pico_sampling_interval;

if sum(temp_gap > 0.5) ~= 1 % if there is a gap between 2p signal larger than 0.5s
    error('need to revise code')
end

recording_range_index = [temp_index(1) temp_index(find(temp_gap > 0.5)); temp_index(find(temp_gap > 0.5)+1) temp_index(end)];
%

% check data length
diff_duration(1) = (diff(recording_range_index(1,:)) * pico_sampling_interval) - (size(Fc3_DF, 1) / 30.98 / 2);
diff_duration(2) = (diff(recording_range_index(2,:)) * pico_sampling_interval) - (size(Fc3_DF, 1) / 30.98 / 2);

disp(['length difference between pico and F is ' num2str(diff_duration)]);

if sum(diff_duration > 3)
    disp('error!!!');
end
% 

% downsampling pico data

pico_data = nan(size(Fc3_DF, 1), size(raw_pico_data, 2));

for iter = 1 : 2
    
    bin_size = diff(recording_range_index(iter,:)) / (size(Fc3_DF, 1)/2) * pico_sampling_interval; % bin size in time (sec)
    frame_range = [iter-1 iter] * size(Fc3_DF, 1)/2 + [1 0];
    start_timestamp = raw_pico_data(recording_range_index(iter, 1), timestamp_pd);

    tic;
    parfor frame_iter = frame_range(1) : frame_range(2)

        current_range = [frame_iter-frame_range(1) frame_iter-frame_range(1)+1] * bin_size + start_timestamp;
        temp_index = raw_pico_data(:, timestamp_pd) >= current_range(1) & raw_pico_data(:, timestamp_pd) < current_range(2);

        pico_data(frame_iter, :) = mean(raw_pico_data(temp_index, :), 1);
    end
    toc;
end
% 

% % 

% % convert analog to digial
analog_index = [reward_pd tpsignal_pd ttl_pd buzzer_pd light_pd odor_pd];

for iter = analog_index
    pico_data(pico_data(:, iter) < 2.5, iter) = 0;
    pico_data(pico_data(:, iter) >= 2.5, iter) = 1;
end

pico_data(pico_data(:, licking_pd) < 2, licking_pd) = 0;
pico_data(pico_data(:, licking_pd) >= 2, licking_pd) = 1;
% % 


%% trial parsing

% get trial start & end index
ttl_index = find(pico_data(:, ttl_pd) == 1);

% temp = bwlabel(pico_data(:, ttl_pd));

trial_index = [];
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
        trial_index(end+1, 1) = ttl_index(iter) + current_odor_start - 25;        
    end

    iter = iter+1;
end

trial_index(:, 2) = trial_index(:, 1) + 499;
%

% get trial information & data parsing
parsed_trial = []; % 2d matrix. 1: trial, 2: info
parsed_behavior = []; % 3d matrix. 1: trial, 2: frame, 3: info
parsed_flevel_only_all = []; % 3d matrix -> save as 2d matrix. 1: trial, 2: frame, 3: cell

for trial_iter = 1 : size(trial_index, 1)

    parsed_behavior(trial_iter, :, licking_pb) = pico_data(trial_index(trial_iter,1) : trial_index(trial_iter,2), licking_pd);
    parsed_behavior(trial_iter, :, odor_pb) = pico_data(trial_index(trial_iter,1) : trial_index(trial_iter,2), odor_pd);
    parsed_behavior(trial_iter, :, reward_pb) = pico_data(trial_index(trial_iter,1) : trial_index(trial_iter,2), reward_pd);
    parsed_behavior(trial_iter, :, velocity_pb) = pico_data(trial_index(trial_iter,1) : trial_index(trial_iter,2), velocity_pd);

    parsed_trial(trial_iter, trialn_pt) = trial_iter;
    
    % decide condition (trial type)
    temp_odor = bwlabel(parsed_behavior(trial_iter, :, odor_pb));

    if sum(temp_odor == 1) < 100 && sum(temp_odor == 2) < 100       % s-s condition
        parsed_trial(trial_iter, condition_pt) = ss_type;
    elseif sum(temp_odor == 1) > 100 && sum(temp_odor == 2) < 100   % l-s condition
        parsed_trial(trial_iter, condition_pt) = ls_type;
    elseif sum(temp_odor == 1) < 100 && sum(temp_odor == 2) > 100   % s-l condition
        parsed_trial(trial_iter, condition_pt) = sl_type;
    else
        error('!!');
    end
    %

    % decide correctness
    if parsed_trial(trial_iter, condition_pt) == ss_type

        temp = parsed_behavior(trial_iter, :, odor_pb);
        if sum(parsed_behavior(trial_iter, find(temp == 1, 1, 'first') : find(temp == 1, 1, 'last')+response_window, licking_pb))
            parsed_trial(trial_iter, correctness_pt) = incorrect_type;
        else
            parsed_trial(trial_iter, correctness_pt) = correct_type;
        end

    else
        if sum(parsed_behavior(trial_iter, :, reward_pb))
            parsed_trial(trial_iter, correctness_pt) = correct_type;
        else
            parsed_trial(trial_iter, correctness_pt) = incorrect_type;
        end
    end
    %

    % parsed_flevel_only_all
    for cell_iter = 1 : size(Fc3_DF, 2)
        parsed_flevel_only_all(trial_iter, :, cell_iter) = Fc3_DF(trial_index(trial_iter,1) : trial_index(trial_iter,2), cell_iter);
    end
    %

end
%

% % 

% % save
save([output_root '\parsed_session\' session_id '.mat'], 'parsed_behavior', 'parsed_trial', 'pico_data', 'Fc3_DF');

for cell_iter = 1 : size(Fc3_DF, 2)
    parsed_flevel_only = squeeze(parsed_flevel_only_all(:, :, cell_iter));
    save([output_root '\parsed_cell\' session_id '_' num2str(cell_iter) '.mat'], 'parsed_flevel_only');
end
% % 


% % display behavior performance
All_T = size(parsed_trial, 1);
All_T_c = sum(parsed_trial(:, correctness_pt) == correct_type) / All_T;
All_LS = sum(parsed_trial(:, condition_pt) == ls_type);
All_LS_c = sum(parsed_trial(:, condition_pt) == ls_type & parsed_trial(:, correctness_pt) == correct_type) / All_LS;
All_SL = sum(parsed_trial(:, condition_pt) == sl_type);
All_SL_c = sum(parsed_trial(:, condition_pt) == sl_type & parsed_trial(:, correctness_pt) == correct_type) / All_SL;
All_SS = sum(parsed_trial(:, condition_pt) == ss_type);
All_SS_c = sum(parsed_trial(:, condition_pt) == ss_type & parsed_trial(:, correctness_pt) == correct_type) / All_SS;

% table(All_T,All_T_c,All_SL,All_SL_c,All_LS,All_LS_c,All_SS,All_SS_c)
table(All_T_c,All_SS_c,All_LS_c,All_SL_c)
% % 