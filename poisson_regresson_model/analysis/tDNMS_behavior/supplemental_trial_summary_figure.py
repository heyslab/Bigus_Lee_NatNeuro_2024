import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sys
sys.path.append('../')
import os
import pandas as pd
import scipy
import numpy as np
import itertools as it
import json

import expt_classes
import database

from mpl_helpers import PdfPlotter
import jPlots as jP
import pd_helpers as pdH

from poisson_regression import determine_consumption_window

colors = ('#eb0d8c', '#2bace2', '#f89521')

def plot_example_response(axs, odor_times, i, model, predictions, data):
    odor_times['Odor1'].dropna().groupby('type').apply(
        lambda x: x.index.get_level_values('trial_time')).apply(
            lambda x, axs=iter(axs): next(axs).axvspan(*x, color='orange',
                                                       alpha=0.2))
    odor_times['Odor2'].dropna().groupby('type').apply(
        lambda x: x.index.get_level_values('trial_time')).apply(
            lambda x, axs=iter(axs): next(axs).axvspan(*x, color='orange',
                                                       alpha=0.2))

    def plot_func(X, axs, colors, ylims=(-0.1, 1.2), alpha=1, lw=1):
        trial_type = X.index.unique(0)[0]
        ax = next(axs)
        ax.plot(X.index.get_level_values('trial_time'), X.values,
                c=next(colors), lw=lw, alpha=alpha)
        jP.configure_spines(ax, fix_xlabel=False)
        ax.set_xlim(0, 20)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        ax.set_ylim(*ylims)
        ax.set_ylabel(f'{trial_type} Trial')

    consumption_window = determine_consumption_window(data)
    twinaxs = list(map(lambda x: x.twinx(), axs))
    data['Lick_Start'].mask(consumption_window).fillna(0).groupby(
        ['type', 'trial_time']).mean().groupby('type').apply(
            lambda x: x[x.index[0][0]]).groupby('type').apply(
                plot_func, axs=iter(axs), colors=it.repeat('k'), lw=0.75)
    predictions.groupby(['type', 'trial_time']).mean().groupby('type').apply(
        lambda x: x[x.index[0][0]]).groupby('type').apply(
            plot_func, axs=iter(twinaxs), colors=iter(colors))
    list(map(lambda x: x.spines['bottom'].set_visible(False), twinaxs))
    list(map(lambda x: x.spines['left'].set_visible(False), twinaxs))
    list(map(lambda x: x.set_ylabel(''), twinaxs))
    list(map(lambda x, c: x.yaxis.label.set_color(c), axs, colors))

    axs[0].spines['bottom'].set_visible(False)
    axs[0].text(1, 1, r'd$^2$: ' + f'{predictions.attrs["d2"]:.3f}', ha='right',
                va='top', transform=axs[0].transAxes, fontsize=5)
    axs[1].spines['bottom'].set_visible(False)
    axs[0].set_xticks([])
    axs[1].set_xticks([])
    axs[2].set_xlabel('Time (s)')
    axs[2].set_xticks(np.linspace(0, 20, 5)[1:])



def main(argv):
    jP.set_rcParams(plt)
    margins = jP.default_margins()
    dpi = 300

    #trials = sorted(database.fetch_trials(experiment_type='tDNMS'))
    trials = (39, 7)
    expts = list(map(expt_classes.Experiment.create, trials))
    expts = list(filter(lambda x: x.get('use', True), expts))
    order = pd.DataFrame(
        list(map(lambda x: {'mouse_name': x.mouse_name,
                            'id': x.trial_id,
                            'start_date': x.start_time}, expts))).set_index(
            ['mouse_name', 'start_date']).sort_index()
    save_path = os.path.join(jP.analysis_folder(),
                             'trial_summary_supp')
    jP.make_folder(save_path)
    filename = os.path.join(save_path, '{}.pdf')

    models = json.load(open('models.json'))
    for j, trial in enumerate(order.values.flatten()):
        PdfPlotter(filename.format(f'summary_{trial}'), fixed_margins=margins)
        plt.figure(figsize=(4, 2), dpi=300)
        gs1 = matplotlib.gridspec.GridSpec(1, 1, hspace=0.35)

        gs0 = gs1[0].subgridspec(1, 5)
        expt = expt_classes.Experiment.create(trial)
        print(f'[{expt}]')
        model_1 = models['none']
        model_1 = expt_classes.tDNMS_Experiment._sort_model_parts(model_1)
        predictions_1 = expt.model_predictions(f'PoissonMCW:{model_1}')

        model_2 = models['long_wait']
        model_2 = expt_classes.tDNMS_Experiment._sort_model_parts(model_2)
        predictions_2 = expt.model_predictions(f'PoissonMCW:{model_2}')

        model_3 = models['triallength']
        model_3 = expt_classes.tDNMS_Experiment._sort_model_parts(model_3)
        predictions_3 = expt.model_predictions(f'PoissonMCW:{model_3}')

        model_4 = models['notshort']
        model_4= expt_classes.tDNMS_Experiment._sort_model_parts(model_4)
        predictions_4 = expt.model_predictions(f'PoissonMCW:{model_4}')

        model_5 = models['compare']
        model_5= expt_classes.tDNMS_Experiment._sort_model_parts(model_5)
        predictions_5 = expt.model_predictions(f'PoissonMCW:{model_5}')
        plot_models = (model_1, model_2, model_3, model_4, model_5)
        plot_predictions = (predictions_1, predictions_2, predictions_3,
                            predictions_4, predictions_5)

        odor_times = expt.odor_times.groupby('type').apply(
            lambda x: x.loc[x.index[0][0]]).apply(
                lambda x: x[x.astype(int).diff().fillna(0) != 0])
        plot_labels = ('Cue Based', 'Strategy 1', 'Stragegy 2', 'Strategy 3',
                       'Strategy 4', 'Strategy 5')

        data = expt.downsample_behavior(0.1)
        data = data.groupby('trial').apply(lambda x: x.iloc[0:200]).droplevel(0)
        for i, (model, predictions, title) in enumerate(
                zip(plot_models, plot_predictions, plot_labels)):

            gs = gs0[i].subgridspec(3, 1)
            axs = list(map(plt.subplot, gs))
            plot_example_response(axs, odor_times, i, model, predictions, data)
            axs[0].set_title(title)
            if i == 0:
                list(map(lambda x: x.spines['left'].set_bounds(0.1, 0.35), axs))
                list(map(lambda x: x.spines['left'].set_visible(True), axs))
                list(map(lambda x: x.text(
                    0, 0.225, '2.5\nLick/s', ha='right', va='center',
                    transform=x.get_yaxis_transform(), rotation=90, fontsize=5,
                    ma='center'), axs))
            if i == 4:
                list(map(lambda x: x.spines['right'].set_bounds(0.1, 0.35),
                         axs))
                list(map(lambda x: x.spines['right'].set_visible(True), axs))
                list(map(
                    lambda x, cs=iter(colors): x.spines['right'].set_color(
                        next(cs)), axs))
                list(map(lambda x, cs=iter(colors): x.text(
                    1, 0.225, '2.5\nPred./s', ha='left', va='center',
                    transform=x.get_yaxis_transform(), rotation=-90, fontsize=5,
                    ma='center', color=next(cs)), axs))
        plt.show()
    import pdb; pdb.set_trace()

    PdfPlotter(filename.format('summary_2row'), fixed_margins=margins)
    plt.figure(figsize=(3, 5), dpi=300)
    gs1 = matplotlib.gridspec.GridSpec(1, 1, hspace=0.5)
    for j, trial in enumerate(order.values.flatten()):
        gs0 = gs1[j].subgridspec(4, 7, hspace=0.5)
        expt = expt_classes.Experiment.create(trial)
        print(f'[{expt}]')
        for i, (model, predictions, title) in enumerate(
                zip(plot_models, plot_predictions, plot_labels)):

            #gs = gs0[i + int(i > 2)].subgridspec(3, 1)
            if i == 0:
                gs = gs0[0:2, 0:2].subgridspec(3, 1)
            elif i < 3:
                gs = gs0[0:2, (1 + 2*i):(1 + 2*i+2)].subgridspec(3, 1)
            else:
                gs = gs0[2:4, (3 + 2 * (i % 3)):(5 + 2 * (i % 3))].subgridspec(3, 1)

            axs = list(map(plt.subplot, gs))
            plot_example_response(axs, odor_times, i, model, predictions, data)
            if i != 0:
                list(map(lambda x: x.set_ylabel(''), axs))
            axs[0].set_title(title)
            if i in (1, 2):
                axs[-1].set_xticks([])
                axs[-1].set_xlabel('')
            if i in (0, 1, 3):
                list(map(lambda x: x.spines['left'].set_bounds(0.1, 0.35), axs))
                list(map(lambda x: x.spines['left'].set_visible(True), axs))
                list(map(lambda x: x.text(
                    0, 0.225, '2.5\nLick/s', ha='right', va='center',
                    transform=x.get_yaxis_transform(), rotation=90, fontsize=5,
                    ma='center'), axs))
            if i in (2, 4):
                list(map(lambda x: x.spines['right'].set_bounds(0.1, 0.35),
                         axs))
                list(map(lambda x: x.spines['right'].set_visible(True), axs))
                list(map(
                    lambda x, cs=iter(colors): x.spines['right'].set_color(
                        next(cs)), axs))
                list(map(lambda x, cs=iter(colors): x.text(
                    1, 0.225, '25%\nProb.', ha='left', va='center',
                    transform=x.get_yaxis_transform(), rotation=-90, fontsize=5,
                    ma='center', color=next(cs)), axs))
    plt.show()
    import pdb; pdb.set_trace()





    trials = (39,)
    expts = list(map(expt_classes.Experiment.create, trials))
    expts = list(filter(lambda x: x.get('use', True), expts))
    order = pd.DataFrame(
        list(map(lambda x: {'mouse_name': x.mouse_name,
                            'id': x.trial_id,
                            'start_date': x.start_time}, expts))).set_index(
            ['mouse_name', 'start_date']).sort_index()
    save_path = os.path.join(jP.analysis_folder(),
                             'trial_summary_figure')
    jP.make_folder(save_path)
    filename = os.path.join(save_path, '{}.pdf')

    models = json.load(open('models.json'))
    print(filename.format(f'summary_{trials[0]}'))
    PdfPlotter(filename.format(f'summary_{trials[0]}'), fixed_margins=margins)
    plt.figure(figsize=(2, 2), dpi=300)
    gs1 = matplotlib.gridspec.GridSpec(1, 1, hspace=0.35)
    for j, trial in enumerate(order.values.flatten()):
        gs0 = gs1[j].subgridspec(1, 2)
        expt = expt_classes.Experiment.create(trial)
        print(f'[{expt}]')
        model_1 = models['none']
        model_1 = expt_classes.tDNMS_Experiment._sort_model_parts(model_1)
        predictions_1 = expt.model_predictions(f'PoissonMCW:{model_1}')

        model_2 = models['long_wait']
        model_2 = expt_classes.tDNMS_Experiment._sort_model_parts(model_2)
        predictions_2 = expt.model_predictions(f'PoissonMCW:{model_2}')

        model_3 = models['triallength']
        model_3 = expt_classes.tDNMS_Experiment._sort_model_parts(model_3)
        predictions_3 = expt.model_predictions(f'PoissonMCW:{model_3}')

        model_4 = models['notshort']
        model_4= expt_classes.tDNMS_Experiment._sort_model_parts(model_4)
        predictions_4 = expt.model_predictions(f'PoissonMCW:{model_4}')

        model_5 = models['compare']
        model_5= expt_classes.tDNMS_Experiment._sort_model_parts(model_5)
        predictions_5 = expt.model_predictions(f'PoissonMCW:{model_5}')
        plot_models = (model_1, model_4)
        plot_predictions = (predictions_1, predictions_4)

        odor_times = expt.odor_times.groupby('type').apply(
            lambda x: x.loc[x.index[0][0]]).apply(
                lambda x: x[x.astype(int).diff().fillna(0) != 0])
        plot_labels = ('Cue Based', 'Strategy 3')

        data = expt.downsample_behavior(0.1)
        data = data.groupby('trial').apply(lambda x: x.iloc[0:200]).droplevel(0)
        for i, (model, predictions, title) in enumerate(
                zip(plot_models, plot_predictions, plot_labels)):

            gs = gs0[i].subgridspec(3, 1)
            axs = list(map(plt.subplot, gs))
            plot_example_response(axs, odor_times, i, model, predictions, data)
            axs[0].set_title(title)
            if i == 0:
                list(map(lambda x: x.spines['left'].set_bounds(0.1, 0.35), axs))
                list(map(lambda x: x.spines['left'].set_visible(True), axs))
                list(map(
                    lambda x:
                        x.text(0, 0.225, '2.5\nLick/s', ha='right', va='center',
                               transform=x.get_yaxis_transform(), rotation=90,
                               fontsize=5, ma='center'), axs))
            if i == 1:
                list(map(lambda x: x.spines['right'].set_bounds(0.1, 0.35), axs))
                list(map(lambda x: x.spines['right'].set_visible(True), axs))
                list(map(lambda x, cs=iter(colors): x.spines['right'].set_color(next(cs)), axs))
                list(map(
                    lambda x, cs=iter(colors):
                        x.text(1, 0.225, '25%\nProb.', ha='left', va='center',
                               transform=x.get_yaxis_transform(), rotation=-90,
                               fontsize=5, ma='center', color=next(cs)), axs))
    plt.show()

    print(filename)

if __name__ == '__main__':
    main(sys.argv[1:])
