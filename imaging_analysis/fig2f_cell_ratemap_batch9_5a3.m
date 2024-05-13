
% % 2023 Feb 6

clear;


% % set input

% cluster id root
input_root = 'G:\Hyuwnoo\OneDrive - University of Utah\Heys lab folder\1. projects\1. tDNMT\1. Analysis\cell_id\';
% input_root = '/Users/jennifer/Library/CloudStorage/OneDrive-SNU/Heys lab folder/1. projects/1. tDNMT/1. Analysis/cell_id/';
input_file_name = 'cell_id.csv';

% input_root = '/Users/jennifer/Library/CloudStorage/OneDrive-SNU/Heys lab folder/1. projects/1. tDNMT/1. Analysis/all/20221222/';
% input_file_name = 'dayN_SS_4mice_20221222.csv';


% parsed data root
% input_root2 = 'G:\Hyuwnoo\OneDrive - University of Utah\Heys lab folder\1. projects\1. tDNMT\1. Analysis\a\5\parsed_session\';
% input_root3 = 'G:\Hyuwnoo\OneDrive - University of Utah\Heys lab folder\1. projects\1. tDNMT\1. Analysis\a\5\parsed_cell\';
input_root2 = 'G:\Hyuwnoo\OneDrive - University of Utah\Heys lab folder\1. projects\1. tDNMT\1. Analysis\a\5\parsed_session_reproduce\';
input_root3 = 'G:\Hyuwnoo\OneDrive - University of Utah\Heys lab folder\1. projects\1. tDNMT\1. Analysis\a\5\parsed_cell_reproduce\';
% input_root2 = '/Users/jennifer/Library/CloudStorage/OneDrive-SNU/Heys lab folder/1. projects/1. tDNMT/1. Analysis/a/5/parsed_session/';
% input_root3 = '/Users/jennifer/Library/CloudStorage/OneDrive-SNU/Heys lab folder/1. projects/1. tDNMT/1. Analysis/a/5/parsed_cell/';

% mutual information root
input_root4 = 'G:\Hyuwnoo\OneDrive - University of Utah\Heys lab folder\1. projects\1. tDNMT\1. Analysis\a\4\mat files8\';
% input_root4 = '/Users/jennifer/Library/CloudStorage/OneDrive-SNU/Heys lab folder/1. projects/1. tDNMT/1. Analysis/a/4/mat files7/';
% % 

% % set output
% output_root = 'G:\Hyuwnoo\OneDrive - University of Utah\Heys lab folder\1. projects\1. tDNMT\1. Analysis\a\3\figures9_reproduce\';
% % 

% set parameter
smoothing_window = 5;
% 

% add path
addpath(genpath('G:\Hyuwnoo\OneDrive - University of Utah\Heys lab folder\1. projects\1. tDNMT\2. Analysis programs'));
% addpath(genpath('/Users/jennifer/Library/CloudStorage/OneDrive-SNU/Heys lab folder/1. projects/1. tDNMT/2. Analysis programs'));
%

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 



% % load input
cell_list = readmatrix([input_root input_file_name]);
% % 


% % batch

for cell_iter = 1 : size(cell_list, 1)

    % load cell data
    if length(num2str(cell_list(cell_iter, 2))) < 4
        session_id = [num2str(cell_list(cell_iter, 1)) '_0' num2str(cell_list(cell_iter, 2))];
    else
        session_id = [num2str(cell_list(cell_iter, 1)) '_' num2str(cell_list(cell_iter, 2))];
    end
    cell_id = [session_id '_' num2str(cell_list(cell_iter, 3))];
    
    % skip if there's existing file
%     if exist([output_root cell_id '.png'], "file")
%         continue;
%     end
    %

    load([input_root2 session_id '.mat'],'parsed_trial');
    load([input_root3 cell_id '.mat'], 'parsed_flevel_only_shifted2');
    parsed_flevel_only = parsed_flevel_only_shifted2;
    %

    fh = cell_ratemap_function9_5a3(cell_id, input_root4, parsed_trial, parsed_flevel_only, smoothing_window);

    % save
    % saveas(fh, [output_root cell_id], 'png');
    %

    % close(fh);
    clear parsed_trial parsed_flevel_only;

end

% % 


