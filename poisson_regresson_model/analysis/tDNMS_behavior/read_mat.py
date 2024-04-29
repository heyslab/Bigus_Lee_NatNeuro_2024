import sys
sys.path.append('../')

import os
import scipy.io
import glob
import pandas as pd
import numpy as np
import itertools as it

import pd_helpers as pdH


def read_pico_mat(filename, experimentor_setup):
    data = scipy.io.loadmat(filename)
    if experimentor_setup == 'Hyunwoo':
        column_names = ['Water', '2p', 'TTL', 'Buzzer', 'Lick', 'Light',
                        'Velocity', 'Odor']
        dataframe = pd.DataFrame(
            np.concatenate(list(map(data.get,
                                    ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])),
        axis=1), columns=column_names)
    elif experimentor_setup == 'Erin':
        column_names = ['Lick', 'Water', 'Odor', 'Light']
        dataframe = pd.DataFrame(
            np.concatenate(list(map(data.get,
                                    ['C', 'D', 'E', 'F'])),
        axis=1), columns=column_names)
    else:
        raise Exception('Valid experimentor_setup is "Erin" or "Hyunwoo"')

    dataframe['Time'] = np.linspace(
        data['Tstart'][0][0], len(dataframe)*data['Tinterval'][0][0],
        len(dataframe))
    dataframe.attrs['Tinterval'] = data['Tinterval'][0][0]
    return dataframe

def format_mats(folder, experimentor_setup, threshold=2.1, file_offset=0.001):
    filenames = pd.Series(sorted(glob.glob(os.path.join(folder, '*.mat'))))
    data = filenames.apply(read_pico_mat, experimentor_setup=experimentor_setup)
    time_offsets = pd.Series(
        [0] + data.apply(lambda x: x['Time'].iloc[-1]).tolist()[:-1])
    time_offsets[1:] += file_offset
    time_index = data.apply(
        lambda x, t=iter(time_offsets.cumsum()): x['Time'] + next(t)
        ).stack().values
    data = pd.concat(data.values, keys=filenames.apply(lambda x: x[-6:-4]))
    data.index = pd.Index(time_index, name='Time')
    data = data.drop('Time', axis=1)

    data = data > threshold
    trial_times = data.index.get_level_values(0)[
                data['Light'].astype(int).diff() > 0].tolist()
    trial_times = [0] + trial_times + [data.index[-1] + 0.1]
    trials_index = pd.IntervalIndex.from_tuples(
        list(zip(trial_times[:-1], trial_times[1:])), closed='left')

    trial_nums = data.index.to_series().apply(
        trials_index.contains).apply(np.where).apply(
            it.chain.from_iterable).apply(next) - 1
    trial_nums[trial_nums < 0] = None
    data.index = pdH.add_level(data.index, trial_nums, 'trial')

    data['Lick_Start'] = data['Lick'].astype(int).diff() > 0
    data['Water_Start'] = data['Water'].astype(int).diff() > 0

    X = data.groupby('trial').apply(
        lambda x: x.index.get_level_values('Time').to_series() - x.index[0][0])
    dat = data.reorder_levels((1, 0)).reindex(X.index)
    dat.index = pdH.add_level(
        dat.index, np.round(X.values, decimals=4), 'trial_time')
    return dat

