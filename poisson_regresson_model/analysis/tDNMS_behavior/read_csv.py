import sys
sys.path.append('../')

import os
import glob
import pandas as pd
import numpy as np
import itertools as it

import pd_helpers as pdH


def read_pico_csv(filename):
    data = pd.read_csv(filename, skiprows=[1]).drop('Channel G', axis=1)
    data.columns = ['Time', 'Lick', 'Water', 'Odor', 'Light']
    return data


def format_csvs(folder, threshold=2.1, file_offset=0.001):
    filenames = pd.Series(sorted(glob.glob(os.path.join(folder, '*.csv'))))
    data = filenames.apply(read_pico_csv)
    time_offsets = pd.Series(
        [0] + data.apply(lambda x: x['Time'].iloc[-1]).tolist()[:-1])
    time_offsets[1:] += file_offset
    time_index = data.apply(
        lambda x, t=iter(time_offsets.cumsum()): x['Time'] + next(t)
        ).stack().values
    data = pd.concat(data.values)
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
        dat.index, np.round(X.values, decimals=3), 'trial_time')
    return dat

