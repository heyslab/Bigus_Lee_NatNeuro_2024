import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
import os
import pandas as pd
import scipy
import numpy as np
import itertools as it
import json
import math
import functools

import expt_classes
import database

from mpl_helpers import PdfPlotter
import jPlots as jP
import pd_helpers as pdH
import predictors


def determine_consumption_window(data):
    def func(x):
        return x[:-1].to_frame().set_index(
            x[1:].index).iloc[:, 0].reindex(x.index).fillna(0).droplevel(0)
    return data['Water_Start'].groupby('trial').cumsum().groupby(
        'trial').apply(func).astype(bool)


def load_data(trial_id, models, keys):
    sampling_rate = 0.1
    expt = expt_classes.Experiment.create(trial_id)
    if not expt.get('use', True):
        return

    log2 = lambda x: np.log(x)/np.log(2)

    d2_dict = {}
    pred_dict = {}
    for model, key in zip(models, keys):
        predictions = expt.model_predictions(f'PoissonMCW:{model}')

        data = expt.downsample_behavior(
            sampling_rate).reindex(predictions.index)
        y = data['Lick_Start'].mask(
            determine_consumption_window(data)).fillna(0).astype(int)
        llh = (-predictions.apply(np.exp).apply(log2) +
               y.apply(np.exp).apply(log2) * predictions.apply(log2) -
               y.apply(math.factorial).apply(log2)).sum()

        # convert to bits/sec
        llh_conv = llh / len(y) / sampling_rate
        d2 = predictions.attrs["d2"]
        d2_dict[key] = {'llh': llh_conv, 'd2': d2}
        pred_dict[key] = pd.concat(
            (predictions, y, data['Odor']), keys=('y_', 'y', 'odor'))
    predictions = pd.Series(pred_dict).apply(pd.Series).T.unstack(0)
    stats = pd.Series(d2_dict).apply(pd.Series)
    predictions.attrs[expt.mouse_name] = stats
    predictions.index.name = 'model'
    predictions.index = pdH.add_level(
        predictions.index, it.repeat(expt.mouse_name), 'mouse_name')
    predictions.index = pdH.add_level(
        predictions.index, it.repeat(expt.get('day', None)), 'day')
    return predictions


def plot_licks(ax, y, **kwargs):
    lick_events = y.reset_index(drop=True)[y.reset_index(drop=True) > 0]
    ax.eventplot(np.expand_dims(lick_events.index.values.T, 1), color='k',
                     lw=0.5, linelengths=lick_events.values,
                     lineoffsets=lick_events.values/2, **kwargs)
    ax.spines['left'].set_position(('axes', -0.01))
    ax.set_ylim(0, 3)
    ax.set_yticks([0, 3])
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_bounds(0, 3)
    ax.set_yticks([0, 3])


def plot_color_model(ax, y_):
    ax.plot(y_.mask(y_.index.get_level_values('type') != 'LS').values,
                    zorder=1e3, color='#eb0d8c')
    ax.plot(y_.mask(y_.index.get_level_values('type') != 'SL').values,
            zorder=1e3, color='#2bace2')
    ax.plot(y_.mask(y_.index.get_level_values('type') != 'SS').values,
            zorder=1e3, color='#f89521')

    trial_times = np.where(y_.index.get_level_values('trial_time') == 0)[0]
    ax.set_xticks(trial_times)
    tick_labels =  [f'Trial {int(x)}' for x in y_.index.get_level_values('trial')[trial_times]]
    tick_labels = [t if i % 3 == 0 else '' for i, t in enumerate(tick_labels)]
    ax.set_xticklabels(tick_labels, ha='left')

    ax.spines['left'].set_position(('axes', -0.01))
    ax.set_yticks([0, 0.5])
    ax.spines['left'].set_bounds(0.05, 0.3)
    ax.text(-0.01, np.mean(ax.spines['left'].get_bounds()), '2.5\nPred./s',
            ha='right', va='center',
            transform=ax.get_yaxis_transform(), rotation=90, fontsize=5,
            ma='center', color='k')
    ax.set_ylim(-0.02, 0.7)
    ax.set_yticks([])
    # color for scale bars
    if False:
        ax.spines['left'].set_color('#eb0d8c')

        tempax = ax.twinx()
        [s.set_visible(False) for s in tempax.spines.values()]
        tempax.set_ylim(ax.get_ylim())
        tempax.spines['left'].set_visible(True)
        tempax.spines['left'].set_bounds(0.05, 0.3)
        tempax.spines['left'].set_position(('axes', -0.0075))
        tempax.spines['left'].set_color('#2bace2')

        tempax = ax.twinx()
        [s.set_visible(False) for s in tempax.spines.values()]
        tempax.set_ylim(ax.get_ylim())
        tempax.spines['left'].set_visible(True)
        tempax.spines['left'].set_bounds(0.05, 0.3)
        tempax.spines['left'].set_position(('axes', -0.005))
        tempax.spines['left'].set_color('#f89521')



def trial_colored_plot(axs, data, attrs):
    y = data['notshort']['y'].xs('B6', level='mouse_name')
    llh = attrs[0]['B6']['llh']['notshort'] - attrs[0]['B6']['llh']['none']
    list(map(functools.partial(jP.configure_spines, fix_ylabel=True), axs))

    ax_iter = iter(axs)
    ax = next(ax_iter)
    y_none = data['none']['y_'].xs('B6', level='mouse_name')
    plot_color_model(ax, y_none)
    ax.text(0.01, 0.99,
        r'd$^2$: ' + f'{np.round(data.attrs["B6"]["d2"]["none"], 3)}\n ',
        ha='left', va='top', transform=ax.transAxes, zorder=1e5)
    ax.set_xticks([])
    jP.set_ylabel_position(ax, 3.1)
    ax.set_ylabel('Cue-Based\nModel')
    ax2 = ax.twinx()
    ax.set_zorder(2)
    ax.set_facecolor([1, 1, 1, 0])
    ax2.set_zorder(1)
    plot_licks(ax2, y, alpha=0.7)
    ax2.set_ylim(-4, 3)
    ax2.text(1.02, 1.5, 'Observed\nLicks', fontsize=5, rotation=-90, ma='center',
             ha='left', va='center', transform=ax2.get_yaxis_transform())

    ax = next(ax_iter)
    y_ = data['notshort']['y_'].xs('B6', level='mouse_name')
    plot_color_model(ax, y_)
    ax.text(0.01, 0.99,
            r'd$^2$: ' +
                f'{np.round(data.attrs["B6"]["d2"]["notshort"], 3)}\n' +
                f'LLHi: {np.round(llh, 3)}',
            ha='left', va='top', transform=ax.transAxes, zorder=1e5)
    ax.set_ylabel('Strategy 3\nModel')
    jP.set_ylabel_position(ax, 3.1)
    ax2 = ax.twinx()
    ax.set_zorder(2)
    ax.set_facecolor([1, 1, 1, 0])
    ax2.set_zorder(1)
    plot_licks(ax2, y, alpha=0.7)
    ax2.set_ylim(-4, 3)
    ax2.text(1.02, 1.5, 'Observed\nLicks', fontsize=5, rotation=-90, ma='center',
             ha='left', va='center', transform=ax2.get_yaxis_transform())

    list(map(lambda x: x.spines['bottom'].set_visible(False), axs))

    odor = data['notshort']['odor'].reset_index(drop=True)
    odor_start = odor[odor.astype(int).diff() > 0].index.values
    odor_stop = odor[odor.astype(int).diff() < 0].index.values
    for ax in axs:
        list(map(functools.partial(ax.axvspan, color='orange', alpha=0.2),
                 odor_start, odor_stop))


def main(argv):
    jP.set_rcParams(plt)
    margins = jP.default_margins()
    dpi = 300

    xlim = (3000, 6990)
    with open('models.json') as f:
        models = json.load(f)

    trials = [35, 50]
    save_path = os.path.join(jP.analysis_folder(), 'example_models')
    jP.make_folder(save_path)
    info_file = os.path.join(save_path, 'info.txt')
    filename = os.path.join(save_path, '{}.pdf')
    with open(info_file, 'w') as f:
        pass

    del models['long']
    del models['longtrial']
    models_sorted = list(map(expt_classes.tDNMS_Experiment._sort_model_parts,
                             models.values()))
    data = pd.Series(trials).apply(
        load_data, models=models_sorted, keys=models.keys())
    attrs = list(map(lambda x: x.attrs, data))
    functools.reduce(lambda a, b: a.update(b), attrs)
    data = pd.concat(data.values)
    data.attrs = attrs[0]

    margins = jP.default_margins()
    adj_margins = margins.copy()
    adj_margins['left'] = 140
    PdfPlotter(filename.format('examples'), fixed_margins=adj_margins)
    plt.figure(figsize=(6.5, 2), dpi=dpi)
    gs = matplotlib.gridspec.GridSpec(2, 1, wspace=0.0,
                                      hspace=0.1)
    axs = list(map(plt.subplot, gs))
    trial_colored_plot(axs, data, attrs)
    list(map(lambda x, xlim=xlim: x.set_xlim(*xlim), axs))
    plt.show()
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main(sys.argv[1:])

