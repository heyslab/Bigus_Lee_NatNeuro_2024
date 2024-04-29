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
    for model, key in zip(models, keys):
        predictions = expt.model_predictions(f'PoissonMCW:{model}')

        data = expt.downsample_behavior(sampling_rate).reindex(predictions.index)
        y = data['Lick_Start'].mask(determine_consumption_window(data)).fillna(0).astype(int)
        llh = (-predictions.apply(np.exp).apply(log2) + y.apply(np.exp).apply(log2) * predictions.apply(log2) - y.apply(math.factorial).apply(log2)).sum()
        # convert to bits/sec
        llh_conv = llh / len(y) / sampling_rate
        d2 = predictions.attrs["d2"]
        d2_dict[key] = {'llh': llh_conv, 'd2': d2}
    d2 = pd.Series(d2_dict)
    d2.index.name = 'model'
    d2.index = pdH.add_level(d2.index, it.repeat(trial_id), 'trial')
    d2.index = pdH.add_level(d2.index, it.repeat(expt.mouse_name), 'mouse_name')
    d2.index = pdH.add_level(d2.index, it.repeat(
        expt.get('decoding_accuracy', None)), 'decoding')
    d2.index = pdH.add_level(d2.index, it.repeat(expt.get('day', None)), 'day')
    return d2.to_frame()


def main(argv):
    jP.set_rcParams(plt)
    margins = jP.default_margins()
    dpi = 300
    with open('models.json') as f:
        models = json.load(f)

    trials = database.fetch_trials(
        project_name='tDNMS', experiment_type='tDNMS_isi_probe')
    trials.extend(
        database.fetch_trials(
            project_name='tDNMS', experiment_type='tDNMS_longISI_probe'))
    save_path = os.path.join(jP.analysis_folder(), 'probe_regression_metrics')
    jP.make_folder(save_path)
    info_file = os.path.join(save_path, 'info.txt')
    filename = os.path.join(save_path, '{}.pdf')
    with open(info_file, 'w') as f:
        pass

    del models['long']
    del models['longtrial']
    models_sorted = list(map(expt_classes.tDNMS_Experiment._sort_model_parts,
                             models.values()))
    data = pd.concat(pd.Series(trials).apply(
        load_data, models=models_sorted, keys=models.keys()).values
        )[0].apply(pd.Series)

    def box_plotter(y, x, ax):
        ax.boxplot(y, showfliers=False,
            meanprops={'color': 'k', 'linestyle': ''}, patch_artist=True,
            medianprops={'linestyle': '-', 'color': 'k'}, widths=0.4,
            flierprops={},
            boxprops={'facecolor': 'w'},
            positions=(next(x),))

    def point_plotter(y, x, ax):
        colors = {
            'E20': 'red',
            'E25': 'orange',
            'E30': 'yellow',
            'E31': 'green',
            'E33': 'blue',
            'E35': 'purple'
        }
        xs = next(x) + np.linspace(-len(y)/2 / len(y)*0.25, len(y)/2 / \
            len(y)*0.25, len(y))
        for mouse, x_ in zip(y.index.unique('mouse_name'),  xs):
            c = colors.get(mouse, 'gray')
            zorder = 1e3
            if mouse in colors.keys():
                zorder = 1e5
            ax.plot([x_],
                    [y.xs(mouse, level='mouse_name')], ls='', marker='o', c=c,
                    ms=2.5, zorder=zorder, mec='k')

    best_strat = data['d2'].reindex(
        data.index.unique('model').difference(['none']), level='model'
        ).groupby(['trial', 'mouse_name']).max()
    plot_data = pd.concat(
        (data['d2']['none'].droplevel('decoding').droplevel('day'), best_strat),
        axis=1, keys=('none', 'best'))
    plot_data.columns.name = 'model'

    adj_margins = margins.copy()
    adj_margins['bottom'] = 125
    PdfPlotter(filename.format('d2_boxplot'), fixed_margins=adj_margins)
    plt.figure(figsize=(2, 2), dpi=300)
    ax = plt.gca()
    plot_data.stack().groupby('model').apply(list).apply(
            box_plotter, x=iter((1, 0)), ax=ax)
    plot_data.apply(point_plotter, x=it.count(), ax=ax)
    jP.configure_spines(ax)
    ax.set_ylim(-0.1, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.set_xticks([0, 1])
    ax.set_ylabel(r'Deviance Explained (d$^2$)')
    label_pad = jP.annotation_padding(ax, 0.05)
    labely = plot_data.max().max() + 2*label_pad
    ax.annotate('', xy=(0, labely), xytext=(1, labely),
                arrowprops=dict(arrowstyle='-', shrinkA=0, shrinkB=0))

    ttest = scipy.stats.ttest_rel(plot_data['none'], plot_data['best'])
    annotation = jP.significance_symbols([ttest[1]]).values[0]
    ax.text(0.5, labely + label_pad, annotation, va='center', ha='center')

    ax.set_xticklabels(
        ['Cue\nBased', 'Cognitive\nStrategy'], rotation=45, ha='right',
        rotation_mode='anchor')
    plt.show()
    print(str(ttest))

    info_str = f"{plt._pdf_plotter.__dict__['filepath']}\n{str(ttest)}\n\n"
    print(info_str)
    with open(info_file, 'a') as f:
        f.write(info_str)

    model_llh = data['llh'].unstack('model')
    llhi = model_llh.subtract(model_llh['none'], axis=0).drop('none', axis=1)
    llhi = llhi.reindex(['long_wait', 'triallength', 'notshort', 'compare'], axis=1)

    adj_margins = margins.copy()
    adj_margins['bottom'] = 125
    PdfPlotter(filename.format('strategy_comparison'), fixed_margins=adj_margins)
    plt.figure(figsize=(3, 2), dpi=300)
    ax = plt.gca()
    llhi.apply(box_plotter, x=it.count(), ax=ax, axis=0)
    llhi.apply(point_plotter, x=it.count(), ax=ax, axis=0)
    jP.configure_spines(ax)
    ax.set_ylabel('LLHi (bits/second)')
    ax.set_xticklabels(
        ['Strategy 1', 'Strategy 2', 'Strategy 3', 'Strategy 4'], rotation=45,
        rotation_mode='anchor', ha='right')

    ttest_1sided = llhi.apply(scipy.stats.ttest_1samp, popmean=0, axis=0).T[1] / 2
    anova = scipy.stats.f_oneway(*llhi.values.T)
    tukey = scipy.stats.tukey_hsd(*llhi.values.T)

    ax.set_ylim(-2.2, 4)
    label_pad = jP.annotation_padding(ax, 0.05)
    tab_length = jP.annotation_padding(ax, 0.025)
    labely = 1.2 + 2*label_pad

    idxs = pd.Series([a for a in it.combinations(np.arange(len(tukey.pvalue)), 2)])
    idxs = idxs[idxs.apply(np.diff).apply(pd.Series)[0].sort_values().index]
    pvalues = list(map(lambda x, tukey=tukey: tukey.pvalue[x], idxs))
    annotations = jP.significance_symbols(pvalues)
    for i, (x1, x2) in enumerate(idxs):
        if pvalues[i] > 0.05:
            continue

        if i == 0:
            x2 -= 0.01
        elif i == 1:
            x1 += 0.01
            x2 -= 0.01
        elif i == 2:
            x1 += 0.01

        ax.annotate('', xy=(x1, labely), xytext=(x2, labely),
                    arrowprops=dict(arrowstyle='-', shrinkA=0, shrinkB=0))
        ax.annotate('', xy=(x1, labely), xytext=(x1, labely-tab_length),
                    arrowprops=dict(arrowstyle='-', shrinkA=0, shrinkB=0))
        ax.annotate('', xy=(x2, labely), xytext=(x2, labely-tab_length),
                    arrowprops=dict(arrowstyle='-', shrinkA=0, shrinkB=0))

        va = 'center'
        if annotations.iloc[i] == 'n.s.':
            va = 'bottom'
        ax.text(x1 + (x2-x1) / 2, labely + label_pad/2, annotations.iloc[i], ha='center', va=va, fontsize=5)
        if i > 1:
            labely += 2.5 * label_pad

    ax.set_yticks([-2, 0, 2, 4])
    plt.show()

    adj2_margins = adj_margins.copy()
    adj2_margins['left'] = 90
    PdfPlotter(filename.format('strategy_comparison_2'),
               fixed_margins=adj2_margins)
    llhi = llhi.reindex(['triallength', 'notshort'], axis=1)
    plt.figure(figsize=(1, 2.25), dpi=300)
    ax = plt.gca()
    ax.axes.set_facecolor([235/255., 220/255., 235/255.])
    llhi.apply(box_plotter, x=it.count(), ax=ax, axis=0)
    llhi.apply(point_plotter, x=it.count(), ax=ax, axis=0)
    jP.configure_spines(ax)
    ax.set_ylabel('ISI Probe - LLHi (bits/second)', labelpad=-1)
    ax.set_xticklabels(
        ['Strategy 2', 'Strategy 3'], rotation=45,
        rotation_mode='anchor', ha='right')

    ttest_1sided = llhi.apply(scipy.stats.ttest_1samp, popmean=0, axis=0).T[1] / 2
    anova = scipy.stats.f_oneway(*llhi.values.T)
    tukey = scipy.stats.tukey_hsd(*llhi.values.T)

    ax.set_ylim(-2.1, 4)
    label_pad = jP.annotation_padding(ax, 0.05)
    tab_length = jP.annotation_padding(ax, 0.025)
    #labely = 1.2 + 2*label_pad
    labely = 2

    idxs = pd.Series([a for a in it.combinations(np.arange(len(tukey.pvalue)), 2)])
    idxs = idxs[idxs.apply(np.diff).apply(pd.Series)[0].sort_values().index]
    pvalues = list(map(lambda x, tukey=tukey: tukey.pvalue[x], idxs))
    ttest = scipy.stats.ttest_rel(*llhi.values.T)
    pvalues = [ttest.pvalue]
    annotations = jP.significance_symbols(pvalues)
    for i, (x1, x2) in enumerate(idxs):
        if pvalues[i] > 0.05:
            continue

        if i == 0:
            x2 -= 0.01
        elif i == 1:
            x1 += 0.01
            x2 -= 0.01
        elif i == 2:
            x1 += 0.01

        ax.annotate('', xy=(x1, labely), xytext=(x2, labely),
                    arrowprops=dict(arrowstyle='-', shrinkA=0, shrinkB=0))
        ax.annotate('', xy=(x1, labely), xytext=(x1, labely-tab_length),
                    arrowprops=dict(arrowstyle='-', shrinkA=0, shrinkB=0))
        ax.annotate('', xy=(x2, labely), xytext=(x2, labely-tab_length),
                    arrowprops=dict(arrowstyle='-', shrinkA=0, shrinkB=0))

        va = 'center'
        if annotations.iloc[i] == 'n.s.':
            va = 'bottom'
        ax.text(x1 + (x2-x1) / 2, labely + label_pad/2, annotations.iloc[i], ha='center', va=va, fontsize=5)
        if i > 1:
            labely += 2.5 * label_pad

    ax.set_yticks([-2, 0, 2, 4])
    ax.annotate('', xy=(ax.get_xlim()[0], ax.get_ylim()[0]), xytext=(ax.get_xlim()[0], ax.get_ylim()[1]),
                arrowprops=dict(arrowstyle='-',linestyle='--', shrinkA=0,
                shrinkB=0, color='purple'))
    ax.annotate('', xy=(ax.get_xlim()[1], ax.get_ylim()[0]), xytext=(ax.get_xlim()[1], ax.get_ylim()[1]),
                arrowprops=dict(arrowstyle='-',linestyle='--', shrinkA=0,
                shrinkB=0, color='purple'))
    ax.annotate('', xy=(ax.get_xlim()[0], ax.get_ylim()[0]), xytext=(ax.get_xlim()[1], ax.get_ylim()[0]),
                arrowprops=dict(arrowstyle='-',linestyle='--', shrinkA=0,
                shrinkB=0, color='purple'))
    ax.annotate('', xy=(ax.get_xlim()[0], ax.get_ylim()[1]), xytext=(ax.get_xlim()[1], ax.get_ylim()[1]),
                arrowprops=dict(arrowstyle='-',linestyle='--', shrinkA=0,
                shrinkB=0, color='purple'))
    plt.show()

    info_str = f"{plt._pdf_plotter.__dict__['filepath']}\n{str(ttest)}\n\n"
    print(info_str)
    with open(info_file, 'a') as f:
        f.write(info_str)


if __name__ == '__main__':
    main(sys.argv[1:])

