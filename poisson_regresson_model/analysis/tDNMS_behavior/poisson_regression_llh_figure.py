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
from sklearn.mixture import GaussianMixture

import expt_classes
import database

from mpl_helpers import PdfPlotter
import jPlots as jP
import pd_helpers as pdH
import predictors
colors = {
    'E20': 'red',
    'E25': 'orange',
    'E30': 'yellow',
    'E31': 'green',
    'E33': 'blue',
    'E35': 'purple'
}

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

        data = expt.downsample_behavior(sampling_rate).reindex(
            predictions.index)
        y = data['Lick_Start'].mask(
            determine_consumption_window(data)).fillna(0).astype(int)
        llh = (-predictions.apply(np.exp).apply(log2) +
               y.apply(np.exp).apply(log2) * predictions.apply(log2) -
               y.apply(math.factorial).apply(log2)).sum()

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
    d2.index = pdH.add_level(d2.index, it.repeat(
        expt.get('percent_correct', None)), 'percent_correct')
    d2.index = pdH.add_level(d2.index, it.repeat(expt.get('day', None)), 'day')
    return d2.to_frame()


def main(argv):
    jP.set_rcParams(plt)
    margins = jP.default_margins()
    dpi = 300
    with open('models.json') as f:
        models = json.load(f)

    trials = database.fetch_trials(
        project_name='tDNMS', experiment_type='tDNMS', day='N')
    save_path = os.path.join(jP.analysis_folder(), 'poission_metrics')
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


    def box_plotter(y, x, ax, widths=0.4, **kwargs):
        ax.boxplot(y, showfliers=False,
            meanprops={'color': 'k', 'linestyle': ''}, patch_artist=True,
            medianprops={'linestyle': '-', 'color': 'k'}, widths=widths,
            flierprops={},
            boxprops={'facecolor': 'w'},
            positions=(next(x),), **kwargs)

    def point_plotter(y, x, ax, colors=colors):

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

    best_strat = data['d2'].unstack('model').drop('none', axis=1).max(axis=1)
    plot_data = pd.concat(
        (data['d2']['none'], best_strat),
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
    jP.configure_spines(ax, fix_xlabel=False)
    ax.set_ylim(-0.05, 0.7)
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

    PdfPlotter(filename.format('d2_scatter'), fixed_margins=margins)
    plt.figure(figsize=(2, 2), dpi=300)
    ax = plt.gca()
    c = list(map(colors.get, plot_data.index.get_level_values('mouse_name'),
                 it.repeat('gray')))
    plt.scatter(plot_data['none'], plot_data['best'], marker='o', c=c,
                edgecolor='k', s=10)
    jP.configure_spines(ax)
    ax.set_ylabel(r'Best Cognitive Strategy (d$^2$)')
    ax.set_xlabel(r'Cue Based (d$^2$)')
    ax.set_xticks([0.2, 0.4, 0.6])
    ax.set_yticks([0.2, 0.4, 0.6])
    plt.show()

    PdfPlotter(filename.format('d2_inset_scatter'), fixed_margins=adj_margins)
    plt.figure(figsize=(2, 2.25), dpi=300)
    ax = plt.gca()
    c = list(map(colors.get, plot_data.index.get_level_values('mouse_name'),
                 it.repeat('gray')))
    plt.scatter(plot_data['none'], plot_data['best'], marker='o', c=c,
                edgecolor='k', s=10)
    jP.configure_spines(ax, fix_xlabel=False)
    ax.set_ylabel(r'Cognitive Strategy (d$^2$)')
    ax.set_xlabel(r'Cue Based (d$^2$)')
    ax.set_xticks([0.2, 0.4, 0.6])
    ax.set_yticks([0.2, 0.4, 0.6])
    ax.set_title('Deviance Explained')

    ax2 = ax.inset_axes([0.6, 0.125, 0.45, 0.4])
    plot_data.stack().groupby('model').apply(list).apply(
        box_plotter, x=iter((1, 0)), ax=ax2)
    #plot_data.apply(point_plotter, x=it.count(), ax=ax2)
    jP.configure_spines(ax2, fix_ylabel=False)
    ax2.set_ylim(-0.05, 0.8)
    ax2.set_yticks([0.0,  0.8])
    ax2.set_xticks([0, 1])
    ax2.set_ylabel(r'd$^2$', labelpad=-5, rotation=0)
    label_pad = jP.annotation_padding(ax2, 0.05)
    labely = plot_data.max().max() + 2*label_pad
    ax2.annotate('', xy=(0, labely), xytext=(1, labely),
                arrowprops=dict(arrowstyle='-', shrinkA=0, shrinkB=0))

    ttest = scipy.stats.ttest_rel(plot_data['none'], plot_data['best'])
    annotation = jP.significance_symbols([ttest[1]]).values[0]
    ax2.text(0.5, labely + label_pad, annotation, va='center', ha='center')

    ax2.set_xticklabels(
        ['Cue\nBased', 'Cognitive\nStrategy'],
        rotation_mode='anchor', fontsize=5)
    plt.show()

    adj_margins = margins.copy()
    adj_margins['bottom'] = 125
    PdfPlotter(filename.format('strategy_comparison'),
               fixed_margins=adj_margins)
    plt.figure(figsize=(3, 2), dpi=300)
    ax = plt.gca()
    plt.show()

    model_llh = data['llh'].unstack('model')
    llhi = model_llh.subtract(model_llh['none'], axis=0).drop('none', axis=1)
    llhi = llhi.reindex(['long_wait', 'triallength', 'notshort', 'compare'],
                        axis=1)

    adj_margins = margins.copy()
    adj_margins['bottom'] = 125
    adj2_margins = adj_margins.copy()
    adj2_margins['right'] = 50
    PdfPlotter(filename.format('strategy_comparison'),
               fixed_margins=adj2_margins)
    plt.figure(figsize=(1.5, 2.25), dpi=300)
    ax = plt.gca()
    llhi.apply(box_plotter, x=it.count(), ax=ax, axis=0, widths=0.5)
    llhi.apply(point_plotter, x=it.count(), ax=ax, axis=0)
    jP.configure_spines(ax)
    ax.set_ylabel('LLHi (bits/second)')
    ax.set_xticklabels(
        ['Strategy 1', 'Strategy 2', 'Strategy 3', 'Strategy 4'], rotation=45,
        rotation_mode='anchor', ha='right')

    ttest_1sided = llhi.apply(
        scipy.stats.ttest_1samp, popmean=0, axis=0).T[1] / 2
    anova = scipy.stats.f_oneway(*llhi.values.T)
    tukey = scipy.stats.tukey_hsd(*llhi.values.T)

    label_pad = jP.annotation_padding(ax, 0.05)
    tab_length = jP.annotation_padding(ax, 0.025)
    labely = llhi.max().max() + 2*label_pad

    idxs = pd.Series(
        [a for a in it.combinations(np.arange(len(tukey.pvalue)), 2)])
    idxs = idxs[idxs.apply(np.diff).apply(pd.Series)[0].sort_values().index]
    pvalues = list(map(lambda x, tukey=tukey: tukey.pvalue[x], idxs))

    ax.set_ylim(-0.1, 1.3)
    annotations = jP.significance_symbols(pvalues)
    for i, (x1, x2) in enumerate(idxs):
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
        ax.text(x1 + (x2-x1) / 2, labely + label_pad/2, annotations.iloc[i],
                ha='center', va=va, fontsize=5)
        if i > 1:
            labely += 2.5 * label_pad

    ax.axvspan(0.5, 2.5, color='purple', alpha=0.15)
    ax.set_yticks([0.2, 0.6, 1.0])
    ax.annotate('', xy=(0.5, ax.get_ylim()[0]), xytext=(0.5, ax.get_ylim()[1]),
                arrowprops=dict(arrowstyle='-',linestyle='--', shrinkA=0,
                shrinkB=0, color='purple'))
    ax.annotate('', xy=(2.5, ax.get_ylim()[0]), xytext=(2.5, ax.get_ylim()[1]),
                arrowprops=dict(arrowstyle='-',linestyle='--', shrinkA=0,
                shrinkB=0, color='purple'))
    ax.annotate('', xy=(0.5, ax.get_ylim()[0]), xytext=(2.5, ax.get_ylim()[0]),
                arrowprops=dict(arrowstyle='-',linestyle='--', shrinkA=0,
                shrinkB=0, color='purple'))
    ax.annotate('', xy=(0.5, ax.get_ylim()[1]), xytext=(2.5, ax.get_ylim()[1]),
                arrowprops=dict(arrowstyle='-',linestyle='--', shrinkA=0,
                shrinkB=0, color='purple'))
    plt.show()

    info_str = f"{plt._pdf_plotter.__dict__['filepath']}\nANOVA:\n{str(anova)}\n" +\
               f"Tukey\n:{str(tukey)}\n{str(tukey.pvalue)}\n\n"
    print(info_str)
    with open(info_file, 'a') as f:
        f.write(info_str)

    PdfPlotter(filename.format('decoding'), fixed_margins=adj_margins)
    plt.figure(figsize=(2, 2.25), dpi=300)
    ax = plt.gca()

    decoding_info = llhi.max(1)[
        llhi.index.get_level_values('decoding').notna()].reset_index('decoding')
    decoding_info['decoding'] *= 100
    c = list(map(colors.get,
                 decoding_info.index.get_level_values('mouse_name')))
    plt.scatter(decoding_info['decoding'], decoding_info[0], s=10,
                edgecolor='k', marker='o', c=c)
    jP.plot_reg_line(ax, decoding_info['decoding'], decoding_info[0],
                     label_pad=(100, -0.1), ls='--', c='k')
    jP.configure_spines(ax)
    ax.set_xlim(60, 100)
    ax.set_ylim(-0.1, 1)
    ax.set_yticks([0, 0.3, 0.6, 0.9])
    ax.set_xticks([65, 80, 95])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    ax.set_xlabel('Decoding Accuracy (SVM)')
    ax.set_ylabel('LLHi (bits/second)')

    stats = scipy.stats.spearmanr(*decoding_info.values.T)
    ax.text(0.5, 1.0,
            r"(Spearman's $\rho$: " +
            f'{stats.statistic:0.3f}, p={stats.pvalue:0.3f})', fontsize=5,
            ha='center', transform=ax.transAxes)
    plt.show()

    info_str = f"{plt._pdf_plotter.__dict__['filepath']}\n" +\
               f"spearman correlation:\n{str(stats)}\n\n"
    print(info_str)
    with open(info_file, 'a') as f:
        f.write(info_str)

    PdfPlotter(filename.format('decoding_d2'), fixed_margins=margins)
    plt.figure(figsize=(2.4, 2), dpi=300)
    ax = plt.gca()
    d2_data = data['d2'].unstack('model').max(axis=1)
    d2_data = d2_data[d2_data.index.get_level_values('decoding').notna()]
    c = list(map(colors.get, d2_data.index.get_level_values('mouse_name'),
                 it.repeat('gray')))
    plt.scatter(d2_data.index.get_level_values('decoding') * 100, d2_data, marker='o', c=c,
                edgecolor='k', s=10)
    p, r = jP.plot_reg_line(
        ax, d2_data.index.get_level_values('decoding') * 100, d2_data,
        ls='--', c='k', label_pad=(-800, 0.02))
    jP.configure_spines(ax)
    ax.set_ylabel(r'Best Fit Model (d$^2$)')
    ax.set_xlabel('Decoding Accuracy (SVM)')
    ax.set_xticks([50, 70, 90])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    ax.set_yticks([0.2,  0.4, 0.6])
    ax.set_ylim(0, 0.65)
    ax.set_xlim(60, 100)
    ax.set_xticks([65, 80, 95])
    stats = scipy.stats.spearmanr(
        d2_data.index.get_level_values('decoding') * 100, d2_data)

    ax.text(0.5, 1.0,
            r"(Spearman's $\rho$: " +
            f'{stats.statistic:0.3f}, p={stats.pvalue:0.3f})', fontsize=5,
            ha='center', transform=ax.transAxes)
    plt.show()
    info_str = f"{plt._pdf_plotter.__dict__['filepath']}\n" +\
               f"pearsonsr, r:{r:.4f}, p:{p:.4f}\n" + \
               f"spearman's\n{str(stats)}\n\n"
    print(info_str)
    with open(info_file, 'a') as f:
        f.write(info_str)


    PdfPlotter(filename.format('percent_correct'), fixed_margins=margins)
    plt.figure(figsize=(2, 2), dpi=300)
    ax = plt.gca()
    c = list(map(colors.get, llhi.index.get_level_values('mouse_name'),
                 it.repeat('gray')))
    p, r = jP.plot_reg_line(
        ax, llhi.index.get_level_values('percent_correct'), llhi.max(1),
        label_pad=(-5, 0.04), ls='--', c='k')
    plt.scatter(llhi.index.get_level_values('percent_correct'), llhi.max(1), marker='o', c=c,
                edgecolor='k', s=10, zorder=0)
    jP.configure_spines(ax)
    ax.set_ylabel(r'LLHi (bits/second)')
    ax.set_xlabel(r'Percent Correct')
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    ax.set_xticks([50, 70, 90])
    ax.set_yticks([0, 0.2, 0.6, 1.0])
    ax.set_ylim(-0.1, 1)
    plt.show()

    corr = scipy.stats.spearmanr(
        llhi.index.get_level_values('percent_correct'), llhi.max(1))
    info_str = f"{plt._pdf_plotter.__dict__['filepath']}\n" +\
               f"pearsonsr, r:{r:.4f}, p:{p:.4f}\n" + \
               f"spearman's\n{str(corr)}\n\n"
    print(info_str)
    with open(info_file, 'a') as f:
        f.write(info_str)

    PdfPlotter(filename.format('percent_correct_d2'), fixed_margins=margins)
    plt.figure(figsize=(2, 2), dpi=300)
    ax = plt.gca()
    d2_data = data['d2'].unstack('model')

    c = list(map(colors.get, d2_data.index.get_level_values('mouse_name'),
                 it.repeat('gray')))
    d2_data = data['d2'].unstack('model').max(1)
    plt.scatter(d2_data.index.get_level_values('percent_correct'), d2_data, marker='o', c=c,
                edgecolor='k', s=10)
    p, r = jP.plot_reg_line(
        ax, d2_data.index.get_level_values('percent_correct'), d2_data,
        label_pad=(-4, 0.02), ls='--', c='k')
    jP.configure_spines(ax)
    ax.set_ylabel(r'Best Fit Model (d$^2$)')
    ax.set_xlabel(r'Percent Correct')
    ax.set_xticks([50, 70, 90])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    ax.set_yticks([0.2, 0.4, 0.6])
    ax.set_ylim(0, 0.65)
    plt.show()

    corr = scipy.stats.spearmanr(
        d2_data.index.get_level_values('percent_correct'), d2_data)
    info_str = f"{plt._pdf_plotter.__dict__['filepath']}\n" +\
               f"pearsonsr, r:{r:.4f}, p:{p:.4f}\n" + \
               f"spearman's\n{str(corr)}\n\n"
    print(info_str)
    with open(info_file, 'a') as f:
        f.write(info_str)

    adj2_margins = margins.copy()
    adj2_margins['left'] = 125
    PdfPlotter(filename.format('percent_correct_decoding'), fixed_margins=adj2_margins)
    plt.figure(figsize=(2, 2), dpi=300)
    ax = plt.gca()
    c = list(map(colors.get,
                 decoding_info.index.get_level_values('mouse_name')))
    plt.scatter(decoding_info.index.get_level_values('percent_correct'),
                decoding_info['decoding'], s=10,
                edgecolor='k', marker='o', c=c)
    jP.plot_reg_line(
        ax, decoding_info.index.get_level_values('percent_correct'),
        decoding_info['decoding'], label_pad=(-1, 2), ls='--', c='k')
    jP.configure_spines(ax)
    ax.set_xlim(65, 90)
    ax.set_ylim(65, 100)
    ax.set_xticks([70, 80, 90])
    ax.set_yticks([70, 80, 90, 100])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    ax.set_ylabel('Decoding Accuracy (SVM)')
    ax.set_xlabel('Percent Correct')
    plt.show()

    gmm = GaussianMixture(n_components=2)
    y = gmm.fit_predict(llhi.max(1).values[:, None])
    min_group = np.where(gmm.means_ == np.min(gmm.means_))[0][0]
    c = {k:'gray' for k in
         llhi.index.get_level_values('mouse_name')[y != min_group]}
    c.update(
        {k:'red' for k in
         llhi.index.get_level_values('mouse_name')[y == min_group]})

    PdfPlotter(filename.format('gmm_classification'),
               fixed_margins=adj_margins)
    plt.figure(figsize=(1, 2), dpi=300)
    ax = plt.gca()
    llhi[['notshort']].apply(box_plotter, x=it.count(), ax=ax)
    llhi[['notshort']].apply(point_plotter, x=it.count(), colors=c, ax=ax)
    jP.configure_spines(ax)
    ax.set_ylabel('LLHi (bits/second)')
    ax.set_xticklabels(['Strategy 3'], rotation=45, rotation_mode='anchor',
                       ha='right')
    ax.text(1, gmm.means_[int(not min_group)][0], 'Strategy\nUsing',
            transform=ax.get_yaxis_transform(), color='black', rotation=-90,
            ha='left', va='center', fontsize=5, ma='center')
    ax.text(1, gmm.means_[min_group][0], 'Cue\nBased',
            transform=ax.get_yaxis_transform(), color='red', rotation=-90,
            ha='left', va='center', fontsize=5, ma='center')
    ax.set_ylim(-0.1, 1.3)
    ax.set_yticks([0.2, 0.6, 1.0])
    plt.show()
    import pdb; pdb.set_trace()

    gmm = GaussianMixture(n_components=2)
    y = gmm.fit_predict(llhi.max(1).values[:, None])
    min_group = np.where(gmm.means_ == np.min(gmm.means_))[0][0]
    c = {k:'gray' for k in
         llhi.index.get_level_values('mouse_name')[y != min_group]}
    c.update(
        {k:'red' for k in
         llhi.index.get_level_values('mouse_name')[y == min_group]})

    adj2_margins = margins.copy()
    adj2_margins['top'] = 15
    adj2_margins['bottom'] = 60
    PdfPlotter(filename.format('gmm_classification_hor'),
               fixed_margins=adj2_margins)
    plt.figure(figsize=(2.4, 0.75), dpi=300)
    ax = plt.gca()

    box_plotter(llhi.max(1), x=iter((0,)), ax=ax, vert=False)
    n = len(llhi)
    y_vals = np.linspace(-n/2 / n*0.4, n/2 / n*0.4, n)
    plt.scatter(llhi.max(1), y_vals, zorder=10, c=list(map(c.get, llhi.index.get_level_values('mouse_name'))), edgecolor='k')
    jP.configure_spines(ax)
    ax.set_xlabel('LLHi (bits/second)')
    ax.set_xticks([0, 0.9])
    ax.set_yticks([])
    ax.text(gmm.means_[min_group][0], 0.9, 'Cue',
            transform=ax.get_xaxis_transform(), color='red',
            ha='center', va='center', fontsize=5, ma='center')
    ax.text(gmm.means_[int(not min_group)][0], 0.9, 'Strategy',
            transform=ax.get_xaxis_transform(), color='black',
            ha='center', va='center', fontsize=5, ma='center')
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])

