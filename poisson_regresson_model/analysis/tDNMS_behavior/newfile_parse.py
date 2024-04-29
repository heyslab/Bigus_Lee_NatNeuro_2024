import sys
sys.path.append('../')
import os
import argparse
import glob

import read_csv
import read_mat
import database
import expt_classes

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='folder storing behavior .csv files')
    parser.add_argument(
        'mouse_name', help='identifier of the mouse for this experiment')
    parser.add_argument(
        '--project_name', default='tDNMS',
        help='name of the project that this data is recorded for')
    parser.add_argument(
        '--experiment_type', default='tDNMS',
        help='name of the project that this data is recorded for')
    parser.add_argument('start_time', help='experiment start time')
    parser.add_argument('experimentor_setup',
                        help='Erin or Hyunwoo configuration for data columns')
    args = parser.parse_args(argv)

    data_folder = os.path.abspath(args.folder)
    folder_check = database.fetch_trials(data_folder=data_folder)
    if len(folder_check) > 1:
        raise Exception("multiple experiments subdirectory")

    if len(folder_check) == 1:
        database.update_trial(
            data_folder, args.mouse_name, args.project_name,
            args.experiment_type, args.start_time,
            trial_id=folder_check[0])
    else:
        database.update_trial(
            data_folder, args.mouse_name, args.project_name,
            args.experiment_type, args.start_time)

    if len(glob.glob(os.path.join(data_folder, '*.csv'))):
        data = read_csv.format_csvs(data_folder)
    if len(glob.glob(os.path.join(data_folder, '*.mat'))):
        data = read_mat.format_mats(data_folder, args.experimentor_setup)
    else:
        raise Exception("no behavior file found")

    expt = expt_classes.Experiment(
        database.fetch_trials(data_folder=data_folder)[0])
    expt.store_data(data, 'behavior.h5', 'behavior')
    print(f'File saved: {data_folder}')
    print(expt)


if __name__ == '__main__':
    main(sys.argv[1:])
