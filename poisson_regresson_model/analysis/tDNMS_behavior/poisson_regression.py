import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import re
import sys
sys.path.append('../')
from sklearn import preprocessing
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import d2_tweedie_score
import scipy
import numpy as np
import math
import pandas as pd
import itertools as it
import argparse
import functools
import json

from mpl_helpers import PdfPlotter
import expt_classes
import jPlots as jP
import pd_helpers as pdH
import predictors

def determine_consumption_window(data):
    def func(x):
        return x[:-1].to_frame().set_index(
            x[1:].index).iloc[:, 0].reindex(x.index).fillna(0).droplevel(0)
    return data['Water_Start'].groupby('trial').cumsum().groupby(
        'trial').apply(func).astype(bool)


def sort_model_parts(model):
    return ' + '.join(
        pd.Series(model.split(' + ')).apply(
            lambda x: sorted(x.split(' x '))).apply(' x '.join).sort_values())


def previous_lickrate(expt, data):
    prev_lr = data['Lick_Start'].groupby('trial').mean().shift(
        periods=-1).fillna(0)
    prev_lr = prev_lr.reindex(data.index.get_level_values('trial'))
    prev_lr.index = X.index
    prev_lr.name = 'PrevLR'
    return prev_lr


def response_window(expt, data):
    trial_starts = expt.behavior_data.reset_index(
        level='Time')['Time'].groupby('trial').head(1)
    response_interval = expt.odors_demixed['Odor2'].apply(lambda x: (x[0], x[1] + 3))
    trial_response_time = response_interval.to_frame().apply(
        lambda x, ts=trial_starts.droplevel([1, 2, 3]):
            tuple(np.round(np.array(x[0])-ts[x.name], decimals=1)), axis=1)
    times = data.reset_index(level='trial_time')['trial_time'].droplevel([1, 2])

    response_window = trial_response_time.to_frame().apply(
        lambda x, times=times:
            pd.Series(
                True, index=pd.IntervalIndex.from_tuples(x.values)
                ).reindex(times[x.name]), axis=1
        ).fillna(False).stack()
    return response_window[pd.MultiIndex.from_frame(times.reset_index())]


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('trial_id')
    parser.add_argument('model')
    args = parser.parse_args(argv)
    save = True
    mask_consumption_window = True
    alpha = 1e-5
    models = json.load(open('models.json'))

    expt = expt_classes.Experiment.create(args.trial_id)
    save_path = os.path.join(
        expt.data_folder, 'analysis', 'tDMNS_models', 'logreg')
    jP.make_folder(save_path)
    filename = os.path.join(
        save_path,
        f'{expt.mouse_name}_{expt.start_time.replace("/", ".")}' + '_{}.pdf')

    jP.set_rcParams(plt)

    print(f'[{args.model}]')
    if args.model == 'all':
        list(map(main, zip(it.repeat(argv[0]), models.keys())))
        return

    if args.model in models.keys():
        model = sort_model_parts(models[args.model])
    else:
        model = sort_model_parts(args.model)

    expt = expt_classes.Experiment.create(args.trial_id)
    data = expt.downsample_behavior(0.1)
    data = data.groupby('trial').apply(lambda x: x.iloc[0:200]).droplevel(0)

    print(f'[{model}]')
    X = predictors.PredictorFactory.parse(model, expt, data)
    if mask_consumption_window:
        consumption_window = determine_consumption_window(data)
        y = data['Lick_Start'].mask(
            consumption_window).fillna(0).values.reshape((-1, 1))
    else:
        y = data['Lick_Start'].values.reshape((-1, 1))

    kf = KFold(n_splits=10)
    trials = X.index.unique('trial').to_series().reset_index(drop=True)
    groups = list(map(
        lambda x, trials=trials: trials.loc[x].values,
        [x[1] for x in kf.split(trials.index.values)]))
    splits = pd.Series(groups).apply(
        lambda g, X=X, count=it.count():
            pd.Series(next(count),
                      index=np.arange(len(X.reindex(g, level='trial'))))
        ).stack()

    X.index = pdH.add_level(X.index, splits, 'split', 0)
    y_frame = pd.DataFrame(y, index=splits)
    def train_predict(split, X, y, alpha=1e-5):
        print(split)
        clf = PoissonRegressor(alpha=alpha, max_iter=1000)
        clf.fit(X.drop(split).values, y.drop(split).values.flatten())
        test = X.xs(split).copy()
        prediction = clf.predict(test.values)
        return pd.Series(prediction.flatten())

    predictions = pd.Series(splits.unique()).apply(
        lambda x, X=X, y=y_frame: train_predict(x, X, y)
        ).apply(pd.Series).stack()
    predictions.index=data.index
    predictions.attrs['r2'] = r2_score(y.flatten(), predictions)
    predictions.attrs['d2'] = d2_tweedie_score(
        y_frame.values.flatten(), predictions.values, power=1)

    clf = PoissonRegressor(alpha=alpha, max_iter=1000)
    clf.fit(X.values, y.flatten())

    if False:
        dat = data['Odor'].groupby(['type', 'trial_time']).mean() > 0.5
        dat.index = pdH.add_level(
            dat.index, dat.index.get_level_values('type'), 'trial', 0)
        dat = dat.to_frame()
        dat.columns = ['Odor']
        res = predictors.PredictorFactory.parse(model, expt, dat)
        PdfPlotter('test.pdf')
        prediction.groupby('type').apply(lambda x: plt.plot(x.values))
        plt.show()

    print(f'r2: {predictions.attrs["r2"]}')
    print(f'd2: {predictions.attrs["d2"]}')

    if save:
        if mask_consumption_window:
            expt.save_model_predictions(f'PoissonMCW:{model}', predictions)
            expt.save_model(f'PoissonMCW:{model}', clf, X.columns)
        else:
            expt.save_model_predictions(f'Poisson:{model}', predictions)
            expt.save_model(f'Poisson:{model}', clf, X.columns)

if __name__ == '__main__':
    main(sys.argv[1:])
