import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np
import itertools as it
import functools
import scipy.stats
import pandas as pd

import glob
import os


def analysis_folder():
    return '/Users/jbowler/Lab2/analysis'

def make_folder(path):
    if not (os.path.exists(path)):
        os.makedirs(path)

def rm_pdfs(path):
    pdfs = glob.glob(os.path.join(path, '*.pdf'))
    list(map(os.remove, pdfs))


def cmap(trial_type):
    if trial_type == 'LEC':
        return plt.get_cmap('Blues')
    elif trial_type == 'MEC':
        return plt.get_cmap('Oranges')
    elif trial_type == 'CA1':
        return plt.get_cmap('Greens')
    elif trial_type == 'dreadd_mec':
        return plt.get_cmap('Oranges')
    elif trial_type == 'dreadd_mec_cno':
        return plt.get_cmap('Reds')
    elif trial_type == 'dreadd_lec':
        return plt.get_cmap('Blues')
    elif trial_type == 'dreadd_lec_cno':
        return plt.get_cmap('Purples')
    elif trial_type is None:
        return plt.get_cmap('gray_r')


def color_map(trial_type, n):
    norm = matplotlib.colors.Normalize(vmin=0, vmax=n)
    func = lambda norm, x: cmap(trial_type)(norm(x + 1))
    return functools.partial(func, norm)


def set_rcParams(_plt):
    _plt.rcParams['axes.labelsize'] = 7
    _plt.rcParams['font.size'] = 6
    _plt.rcParams['axes.titlesize'] = 7
    _plt.rcParams['figure.titlesize'] = 7
    _plt.rcParams['legend.fontsize'] = 6
    _plt.rcParams['ytick.labelsize'] = 6
    _plt.rcParams['xtick.labelsize'] = 6
    _plt.rcParams['lines.markersize'] = 2
    _plt.rcParams['lines.markeredgewidth'] = 0.5
    _plt.rcParams['lines.linewidth'] = 1
    _plt.rcParams['patch.linewidth'] = 0.5
    _plt.rcParams['axes.linewidth'] = 1
    _plt.rcParams['axes.unicode_minus'] = False
    _plt.rcParams['xtick.major.pad']= '1'
    _plt.rcParams['ytick.major.pad']= '1'
    _plt.rcParams["legend.fancybox"] = False
    _plt.rcParams["legend.edgecolor"] = 'k'
    _plt.rcParams["legend.framealpha"] = 1.0

    _plt.rcParams['ps.useafm'] = True
    _plt.rcParams['pdf.fonttype'] = 42


def default_margins(cbar=False):
    if cbar:
        raise Exception('need to implement')
        return dict(
            left=0.3125 * dpi, right=0.4861 * dpi, top=0.2778 * dpi,
            bottom=0.2778 * dpi)
    return dict(left=110, right=90, top=90, bottom=90)


def percent_y(ax, decimals=0):
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=decimals))


def hide_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])


def configure_spines(ax, lw=1, fix_ylabel=True, fix_xlabel=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(lw)
    ax.spines['bottom'].set_linewidth(lw)
    ax.tick_params(width=lw, length=lw*2)

    if fix_ylabel:
        y_transform = matplotlib.transforms.blended_transform_factory(
            ax.figure.dpi_scale_trans, ax.transAxes)
        ax.yaxis.set_label_coords(
            0.15, 0.5, transform=y_transform)
    if fix_xlabel:
        x_transform = matplotlib.transforms.blended_transform_factory(
            ax.transAxes, ax.figure.dpi_scale_trans)
        ax.xaxis.set_label_coords(
            0.5, 0.11, transform=x_transform)


def set_ylabel_position(ax, nlines=1):
    y_transform = matplotlib.transforms.blended_transform_factory(
        ax.figure.dpi_scale_trans, ax.transAxes)
    ax.yaxis.set_label_coords(
        0.04 + 0.07 * nlines, 0.5, transform=y_transform)


def set_axes_label_positions(ax, fix_ylabel=True, fix_xlabel=True):
    if fix_ylabel:
        y_transform = matplotlib.transforms.blended_transform_factory(
            ax.figure.dpi_scale_trans, ax.transAxes)
        ax.yaxis.set_label_coords(
            14/ax.figure.dpi, 0.5, transform=y_transform)
    if fix_xlabel:
        x_transform = matplotlib.transforms.blended_transform_factory(
            ax.transAxes, ax.figure.dpi_scale_trans)
        ax.xaxis.set_label_coords(
            0.5, 14/ax.figure.dpi, transform=x_transform)


def tick_fontsize(ax, fontsize, axis=-1):
    if axis < 1:
        list(map(lambda l: l.set_fontsize(fontsize), ax.get_xticklabels()))
    if axis != 0:
        list(map(lambda l: l.set_fontsize(fontsize), ax.get_yticklabels()))


def configure_scale(ax, xbar=None, ybar=None, xlabel=None, ylabel=None,
                    pad_x=0, pad_y=0, fontsize=6, linewidth=2, float_ybar=True):

    ylim = ax.get_ylim()
    xlim = ax.get_xlim()

    if ybar is None:
        ybar = int(ylim[1]/2.)
    if xbar is None:
        xbar = int((xlim[1] - xlim[0])/10.)
    if ylabel is None:
        ylabel = str(ybar)
    if xlabel is None:
        xlabel = str(xbar)

    ax.axis('on')

    ax.spines['bottom'].set_bounds(xlim[0], xlim[0]+xbar)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['bottom'].set_zorder(100)
    ax.spines['bottom'].set_position(('outward', pad_x))
    if not float_ybar:
        ax.spines['left'].set_bounds(ylim[0], ylim[0] + ybar)
    else:
        ax.spines['left'].set_bounds(0, ybar)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['left'].set_zorder(100)
    ax.spines['left'].set_position(('outward', pad_y))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks([xlim[0]+xbar])
    ax.set_xticklabels([xlabel], ha='left', fontsize=fontsize)
    ax.tick_params(bottom=False, left=False, pad=2, right=False, top=False)
    ax.tick_params(axis='x', pad=0)
    if float_ybar:
        ax.set_yticks([ybar])
    else:
        ax.set_yticks([ylim[0] + ybar])

    ax.set_yticklabels([ylabel], fontsize=fontsize)


def scale_bar(ax, expt, pad_label=0, fontsize=8, box_bottom=0.04,
              label_scale=True, loc='lower left', lw=4, length_um=None):
    microns_per_pixel = expt.imaging_parameters['element_size_um'][-1]
    ax.axis('on')
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position(('axes', 0.1))
    axis_to_data = ax.transAxes + ax.transData.inverted()

    if loc == 'lower left':
        spine_left = axis_to_data.transform((0.1, 0))[0]
        if length_um is not None:
            spine_right = length_um/microns_per_pixel + spine_left
        else:
            spine_right = np.round(axis_to_data.transform((0.25, 0))[0], -1)
    elif loc == 'lower right':
        spine_right = np.round(axis_to_data.transform((0.9, 0))[0], -1)
        if length_um is not None:
            spine_left = spine_right - length_um/microns_per_pixel
        else:
            spine_left = axis_to_data.transform((0.75, 0))[0]

    ax.spines['bottom'].set_bounds(spine_left, spine_right)
    ax.spines['bottom'].set_linewidth(lw)
    ax.spines['bottom'].set_color('w')
    ax.set_xticks([spine_right])
    ax.tick_params(bottom=False, left=False, pad=-5, right=False, top=False)
    um = int((spine_right-spine_left)*microns_per_pixel)

    if label_scale:
        ax.set_xticklabels(['\n%s%sum' % (' '*pad_label, um)], ha='right',
                           color='w', fontsize=fontsize, fontweight='bold')
                           #bbox=dict(facecolor='k', boxstyle='square,pad=0.25'))
        ax.axvspan(axis_to_data.transform((0.075, 0))[0],
                   axis_to_data.transform((0.2825, 0))[0],
                   ymin=box_bottom, ymax=0.125, color='k')
    else:
        ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    return um


def data_to_axis(ax, x):
    axis_to_data = ax.transAxes + ax.transData.inverted()
    return axis_to_data.inverted().transform(x)



def event_plot(ax, data, scale=1, **ep_kwargs):
    rows, cols = np.where(data > 0)
    cols = cols * scale
    event_times = []
    for i in range(data.shape[0]):
        event_times.append(
            [c for r, c, j in zip(rows, cols, it.repeat(i)) if r == j])

    ax.eventplot(event_times, **ep_kwargs)


def color_bar(ax, im, label=None, fontsize=5, labelpad=12, width=20,
              spacing=0.15, height='100%'):
    axins = inset_axes(
        ax, width=width/ax.figure.dpi, height=height, loc='lower left',
        bbox_to_anchor=(1 + spacing, 0., 1, 1), bbox_transform=ax.transAxes,
        borderpad=0)

    cbar = ax.figure.colorbar(im, cax=axins)
    cbar.solids.set_rasterized(True)

    if label is not None:
        cbar.ax.set_ylabel(
            label, fontsize=fontsize, rotation=270, labelpad=labelpad)

    return cbar


def roi_overlay(ax, expt, label='rois', roi_filter=None, alpha=0.4):
    from sima import ROI

    ds = expt.imaging_dataset
    rois = ds.ROIs[label]
    if roi_filter is not None:
        rois = [r for r, rf in zip(rois, it.repeat(roi_filter)) if r.id in rf]

    im_shape = ds.frame_shape[1:-1]
    #width = im_shape[1]*height/im_shape[0]
    time_avg = ds.time_averages[0,:,:,0]
    ax.imshow(
        np.clip(time_avg, np.nanpercentile(time_avg, 10),
                np.nanpercentile(time_avg, 95)),
        plt.get_cmap('gist_gray'), origin='lower')

    #p = ProgressBar(len(rois))
    for j, roi in enumerate(rois):
        #p.update(j)
        try:
            roi.polygons
        except:
            #TODO: this only saves the fist mask as a polygon
            roi.im_shape = im_shape
            bool_masks = []
            for mask in roi.mask:
                bool_masks.append(mask.astype(bool))
            roi.polygons = ROI.mask2poly(bool_masks)

        for poly in roi.polygons:
            poly = Polygon(np.array(poly.exterior.coords)[:,:2])
            patch = PolygonPatch(
                poly, fc=color(j%20), ec=color(j%20), alpha=alpha, zorder=2)
            ax.add_patch(patch)
    #p.end()
    ax.set_xlim(0, ds.frame_shape[2])
    ax.set_ylim(ds.frame_shape[1], 0)
    ax.axis('off')


def lap_correlation_plot(ax, corrs, group, labels={}, number_groups=True,
                         vmax=0.16, cmap=None, line_color='k', line_weight=1,
                         **cbar_kwargs):
    # index on corrs should be (group, context_count, lap)

    index = corrs.index
    #ax = plt.gca()
    im = ax.imshow(
        corrs.mask(np.eye(len(corrs), dtype=bool)), interpolation='nearest',
        vmin=0, vmax=vmax, cmap=cmap)
    cbar = color_bar(ax, im, 'Correlation Coefficient', **cbar_kwargs)
    #cbar.ax.tick_params(labelsize=5)
    cbar.ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.set_facecolor('k')

    group_index = index.get_level_values(group)
    lap_index = index.get_level_values('lap')
    lap_index = lap_index[group_index.notna()]
    group_index = group_index.dropna()
    blocks = corrs.index.droplevel('lap').unique()

    group_idx = blocks.names.index(group)
    context_count_idx = blocks.names.index('context_count')
    for i, block in enumerate(blocks):
        laps = corrs.loc[block, block].index.values
        lim = list(corrs.index.get_level_values('lap')).index(laps[-1])
        if i != len(blocks):
            ax.axhline(y=lim + 0.5, c=line_color, lw=line_weight)
            ax.axvline(x=lim + 0.5, c=line_color, lw=line_weight)

            if len(blocks) > 2 and i < len(blocks) - 1:
                label_text = labels.get(block[group_idx], block[group_idx][:5])
                if number_groups:
                    label = f'{label_text} {int(block[context_count_idx] + 1)}'
                else:
                    label = f'{label_text}'
                label_space = len(laps)/2
                ax.text(
                    data_to_axis(ax, (lim - label_space, 0))[0], 1, label,
                    horizontalalignment='center', verticalalignment='bottom',
                    color='k',transform=ax.transAxes)
                ax.text(
                    1, 1 - data_to_axis(ax, (lim - label_space, 0))[0], label,
                    horizontalalignment='left', verticalalignment='center',
                    rotation=270, color='k',transform=ax.transAxes)
    ax.set_ylabel('Lap')
    ax.set_xlabel('Lap')
    ax.set_aspect('equal', anchor='SW')
    #figsize = ax.figure.get_size_inches()

    #TODO:
    #ax.yaxis.set_label_coords(
    #    14/ax.figure.dpi, figsize[1]/2, transform=ax.figure.dpi_scale_trans)
    #ax.xaxis.set_label_coords(
    #    figsize[0]/2, 14/ax.figure.dpi, transform=ax.figure.dpi_scale_trans)

    return group_index, lap_index


def context_correlation_plot(ax, context_corrs, group, labels={}, vmax=0.25,
                             cmap=matplotlib.cm.get_cmap('viridis'),
                             number_labels=True, fontsize=6, **cbar_kwargs):

    im = ax.imshow(
        context_corrs.mask(np.eye(context_corrs.shape[0]).astype(bool)),
        interpolation='nearest', vmin=0, vmax=vmax, cmap=cmap)

    cbar = color_bar(ax, im, 'Correlation Coefficient', **cbar_kwargs)
    #cbar.ax.tick_params(labelsize=8)

    if number_labels:
        labels_fixed = [f'{labels.get(a, a[:5])}{int(b) + 1}' for labels, (b, a) in
                  zip(it.repeat(labels), context_corrs.columns.unique())]
    else:
        labels_fixed = [f'{labels.get(a, a[:5])}' for labels, (b, a) in
                  zip(it.repeat(labels), context_corrs.columns.unique())]

    ax.set_facecolor('k')
    ax.set_yticks(range(len(labels_fixed)))
    ax.set_xticks(range(len(labels_fixed)))
    ax.set_ylim(context_corrs.shape[0] - 0.5, -0.5)
    ax.set_xlim(-0.5, context_corrs.shape[0] - 0.5)

    if len(labels_fixed[0]) > 3:
        ax.set_yticklabels(labels_fixed, fontsize=fontsize, fontweight='bold',
                           rotation=90, verticalalignment='center')
    else:
        ax.set_yticklabels(labels_fixed, fontsize=fontsize, fontweight='bold')

    ax.set_xticklabels(labels_fixed, fontsize=fontsize, fontweight='bold')


def position_correlation_plot(
        ax, corr, group, labels={}, number_groups=True,
        cmap=matplotlib.cm.get_cmap('viridis'), vmax=0.5,
        line_color='k', **cbar_kwargs):
    position_bins = len(corr.columns.unique('position'))

    im = ax.imshow(corr, interpolation='nearest',
                   vmin=0, vmax=vmax, cmap=cmap)
    ax.set_yticks(np.arange(position_bins/2, corr.shape[0],
                  position_bins))
    ax.set_ylim(corr.shape[0], 0)
    cbar = color_bar(ax, im, 'Correlation Coefficient', **cbar_kwargs)

    for i in range(position_bins, corr.shape[0], position_bins):
        ax.axhline(y=i, c=line_color, lw=1)
        ax.axvline(x=i, c=line_color, lw=1)
    ax.set_facecolor('k')

    corr.columns.droplevel('position').unique()
    ticks = np.linspace(
        position_bins/2, corr.shape[1] - position_bins/2,
        len(corr.columns.droplevel('position').unique()))

    if number_groups:
        labels_fixed = [f'{labels.get(a, a[:5])}{int(b) + 1}' for labels, (b, a) in
                        zip(it.repeat(labels),
                            corr.columns.droplevel('position').unique())]
    else:
        ctx_labels = corr.columns.droplevel('position').unique()
        if len(ctx_labels[0]) == 2:
            labels_fixed = [f'{labels.get(a, a[:5])}' for labels, (b, a) in
                            zip(it.repeat(labels), ctx_labels)]
        else:
            labels_fixed = [f'{labels.get(a, a[:5])}' for labels, a in
                            zip(it.repeat(labels), ctx_labels)]

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels_fixed, fontweight='bold')
    ax.set_yticks(ticks)
    if len(labels_fixed[0]) > 3:
        ax.set_yticklabels(labels_fixed, fontweight='bold', rotation=90,
                           verticalalignment='center')
    else:
        ax.set_yticklabels(labels_fixed, fontweight='bold')
    ax.set_ylim(corr.shape[0], 0)


def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None,
                              dh=.05, barh=.05, fs=None, maxasterix=None):
    """
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr is not None:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)


def mean_std_plot(ax, x, Y, axis=1, **plot_kwargs):
    m = Y.mean(axis)
    std = Y.std(axis)
    ax.fill_between(x, m + std, m - std, alpha=0.15, **plot_kwargs)
    ax.plot(x, m, lw=2, **plot_kwargs)


def bar_point_plot(ax, bar_data, point_data, trial_types, ctxs, colors,
                   width=0.25, plot_lines=True, labels=None, xlabels=None,
                   spread_mice=True, nogap=False, labels_fontsize=5):

    n = bar_data.groupby(bar_data.index.names[0]).count().max()
    all_x = []
    for i, trial_type in enumerate(trial_types):
        xvals = ()
        if nogap:
            xvals = np.arange(0, n*width, width) - width/2 * \
                (n-1) + 1 + i + (n-1) * width * i
        else:
            xvals = np.arange(0, n*(2*width/1.5), (2*width/1.5)) - \
                width/1.5 * (n-1) + 1 + i + (n-1) * width * i
        all_x.extend(xvals)
        all_mousex = []
        sorted_ctxs = [a for a, b in
                       zip(ctxs,
                           it.repeat(bar_data[trial_type].index.unique(0)))
                       if a in b]
        for j, ctx in enumerate(sorted_ctxs):
            x = xvals[j]
            ax.bar(
                [x],
                bar_data[trial_type, ctx], width=width,
                color=colors[i * n + j], edgecolor='k',
                alpha=0.75)

            if point_data is not None:
                if spread_mice:
                    mouse_x = np.linspace(
                        x+width/2, x-width/2,
                        len(point_data[trial_type, ctx].index.unique('mouse')) + 2)[1:-1]
                else:
                    mouse_x = [x] * \
                        len(point_data[trial_type, ctx].index.unique('mouse'))
                ax.plot(
                    mouse_x,
                    point_data[trial_type, ctx].values, 'o',
                    color=colors[i * n + j], zorder=10,
                    markeredgecolor='k', clip_on=False)
                all_mousex.append(list(mouse_x))

        if plot_lines:
            all_mousex = np.array(all_mousex).T
            for j, mouse in enumerate(
                    point_data[trial_type].index.unique('mouse')):
                ax.plot(
                    all_mousex[j],
                    point_data.loc[trial_type][:, mouse][list(ctxs)].values,
                    c='k')
        if labels is not None:
            label_xs = pd.Series(xvals).rolling(
                2, min_periods=2, center=True).mean().dropna()
            #label_ys = bar_data[trial_type].rolling(
            #    2, center=True, min_periods=2).max().dropna()
            label_ys = point_data[trial_type].groupby(point_data.index.names[1]).max().rolling(
                2, center=True, min_periods=2).max().dropna()
            label_pad = 0.08 * ax.figure.dpi
            pad = (ax.transData.inverted().transform([[0, label_pad]]) - \
                   ax.transData.inverted().transform([[0, 0]]))[0][1]
            label_ys = label_ys + pad

            def draw_line(x, ax, y, labels_itr):
                try:
                    label = next(labels_itr)
                except StopIteration:
                    label = ''

                if label != '':
                    ax.annotate('', (x.iloc[0], y), xytext=(x.iloc[1], y), textcoords='data',
                                arrowprops={'arrowstyle': '-', 'shrinkA': 0, 'shrinkB': 0})
                return 0

            dl = functools.partial(draw_line, ax=ax, y=label_ys, labels_itr=iter(labels.get(trial_type, [])))
            pd.DataFrame(xvals).rolling( 2, min_periods=2, center=True).apply(dl)
            for label, x, y in zip(labels.get(trial_type, []), label_xs,
                                   label_ys):
                if label == 'n.s.':
                    y = y + pad/2
                ax.text(x, y, label, fontsize=labels_fontsize, ha='center',
                        zorder=25, va='bottom')

    if xlabels is None:
        ax.set_xticks(
            np.arange(0, (1 + width * (n - 1)) * len(trial_types),
                      1 + width * (n - 1)) + 1)
        ax.set_xticklabels(trial_types)
    else:
        ax.set_xticks(all_x)
        ax.set_xticklabels(list(xlabels) * (i + 1), rotation=45, ha='right',
                           rotation_mode='anchor')
    configure_spines(ax)
    x0 = 1 - width/1.5 * (n-1)
    ax.set_xlim(0, xvals[-1] + x0)
    plt.show()


def bar_point_legend(ax, trial_type, labels, colors=None, loc='upper right',
                     **legend_kwargs):
    if labels is not None:
        if colors is None:
            cm = color_map(trial_type, len(labels))
            colors = map(cm, range(len(labels)))
        patches = [matplotlib.patches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    else:
        patches = []

    #circle = matplotlib.lines.Line2D([], [], color='w', marker='o', markeredgecolor='k',
    #                                 label='Mouse Avg')
    #patches.append(circle)
    leg = ax.legend(handles=patches, loc=loc, **legend_kwargs)
    leg.set_zorder(1e3)
    return leg


def bar_point_info(bar_data, point_data, trial_types, ctxs, percents=True):
    type_name = point_data.index.names[0]
    ctx_name = point_data.index.names[1]

    ttest0_ = scipy.stats.ttest_rel(
        point_data[trial_types[0], ctxs[0]], point_data[trial_types[0], ctxs[1]])
    ttest1_ = scipy.stats.ttest_rel(
        point_data[trial_types[1], ctxs[0]], point_data[trial_types[1], ctxs[1]])
    ttest_0 = scipy.stats.ttest_ind(
        point_data[trial_types[0], ctxs[0]], point_data[trial_types[1], ctxs[0]],
        equal_var=False)
    ttest_1 = scipy.stats.ttest_ind(
        point_data[trial_types[0], ctxs[1]], point_data[trial_types[1], ctxs[1]],
        equal_var=False)
    info = []
    for trial_type in trial_types:
        info.append(f'{trial_type} (bar):\n')
        bar_percents = bar_data.groupby(point_data.index.names[:-1]).mean()
        if percents:
            bar_percents = bar_percents * 100
        info.append(
            '\t' + ', '.join(map(str, bar_percents[trial_type].to_frame().apply(
                lambda x: (x.name, f'{x[0]:.4f}'), 1).values)) + '\n')
    for trial_type, ctx in it.product(trial_types, ctxs):
        info.append(f'{trial_type} - {ctx}:\n')
        info.append(
            '\t' + ', '.join(map(str, point_data[trial_type, ctx].to_frame().apply(
                lambda x: (x.name, f'{x[0]:.4f}'), 1).values)) + '\n')

    info.extend([
        'two-sample paired t-test (points):\n',
        f'\t{trial_types[0]} {ctxs} statistic: {ttest0_.statistic:.4f}, ' +
            f'p-value: {ttest0_.pvalue:.4f}\n',
        f'\t{trial_types[1]} {ctxs} statistic: {ttest1_.statistic:.4f}, ' +
            f'p-value: {ttest1_.pvalue:.4f}\n',
        'Welch\'s t-test (mice):\n',
        f'\t{ctxs[0]} {trial_types} statistic: {ttest_0.statistic:.4f}, ' +
            f'p-value: {ttest_0.pvalue:.4f}\n',
        f'\t{ctxs[1]} {trial_types} statistic: {ttest_1.statistic:.4f}, ' +
            f'p-value: {ttest_1.pvalue:.4f}\n'])
    if percents:
        ztest0_ = proportion.proportions_ztest(
            bar_data[trial_types[0]].groupby(ctx_name).sum(),
            bar_data[trial_types[0]].groupby(ctx_name).count())
        ztest1_ = proportion.proportions_ztest(
            bar_data[trial_types[1]].groupby(ctx_name).sum(),
            bar_data[trial_types[1]].groupby(ctx_name).count())
        ztest_0 = proportion.proportions_ztest(
            bar_data.drop(ctxs[1], level=ctx_name).groupby(type_name).sum(),
            bar_data.drop(ctxs[1], level=ctx_name).groupby(type_name).count())
        ztest_1 = proportion.proportions_ztest(
            bar_data.drop(ctxs[0], level=ctx_name).groupby(type_name).sum(),
            bar_data.drop(ctxs[0], level=ctx_name).groupby(type_name).count())

        info.extend([
            '2-sample proportion z-test (bars):\n',
            f'\t{trial_types[0]} {ctxs} statistic: {ztest0_[0]:.4f}, ' +
                f'p-value: {ztest0_[1]:.4f}\n',
            f'\t{trial_types[1]} {ctxs} statistic: {ztest1_[0]:.4f}, ' +
                f'p-value: {ztest1_[1]:.4f}\n',
            f'\t{ctxs[0]} {trial_types} statistic: {ztest_0[0]:.4f}, ' +
                f'p-value: {ztest_0[1]:.4f}\n',
            f'\t{ctxs[1]} {trial_types} statistic: {ztest_1[1]:.4f}, ' +
                f'p-value: {ztest_1[1]:.4f}\n',
            '\n\n'
        ])

    return info


def bar_point_info_2bar(bar_data, point_data, trial_types):
    type_name = point_data.index.names[0]
    ctx_name = point_data.index.names[1]

    ttest = scipy.stats.ttest_ind(
        point_data[trial_types[0]], point_data[trial_types[1]],
        equal_var=False)
    info = []
    for trial_type in trial_types:
        info.append(f'{trial_type} (bar):\n')
        bar_percents = bar_data.groupby(point_data.index.names[:-1]).mean()
        info.append(
            f'\t{trial_type} {bar_data[trial_type].values[0]:.4f} \n')

    info.extend([
        'Welch\'s t-test (mice):\n',
        f'\t{trial_types} statistic: {ttest.statistic:.4f}, ' +
            f'p-value: {ttest.pvalue:.4f}\n'])
    return info


def barplot(ax, bar_data, colors, width=0.25):
    n = len(bar_data)
    for i, bar in enumerate(bar_data):
        ax.bar(
            [i + 1],
            bar_data[i], width=width,
            color=colors[i], edgecolor='k',
            alpha=0.75)

    ax.set_xticks(np.arange(1, n+1))
    configure_spines(ax)
    x0 = 1 - width/1.5 * (n-1)
    ax.set_xlim(0, n + x0)
    plt.show()


def draw_brace(ax, yspan, xx,  text):
    """Draws an annotated brace on the axes."""
    ymin, ymax = yspan
    yspan = ymax - ymin
    ax_ymin, ax_ymax = ax.get_ylim()
    yax_span = ax_ymax - ax_ymin
    xmin, xmax = ax.get_xlim()
    xspan = xmax - xmin
    resolution = int(yspan/yax_span*100)*2+1 # guaranteed uneven
    beta = 300./yax_span # the higher this is, the smaller the radius

    axis_to_data = ax.transAxes + ax.transData.inverted()
    xcoord = axis_to_data.transform((xx, 1))[0]

    y = np.linspace(ymin, ymax, resolution)
    y_half = y[:int(resolution/2)+1]
    x_half_brace = (1/(1.+np.exp(-beta*(y_half-y_half[0])))
                    + 1/(1.+np.exp(-beta*(y_half-y_half[-1]))))
    x = np.concatenate((x_half_brace, x_half_brace[-2::-1]))
    x = xcoord + (.05*x - .01)*xspan # adjust vertical position

    ax.autoscale(False)
    ax.plot(x, y, color='black', lw=1, clip_on=False)

    ax.text(xcoord + .07*xspan, (ymax + ymin)/2., text, ha='left', va='center',
            rotation=-90)

def invisible_subfig(fig, loc, fix_spines=True, **spines_kwargs):
    ax1 = fig.add_subplot(loc, frameon=False)
    if fix_spines:
        configure_spines(ax1, **spines_kwargs)
    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.set_yticks([])
    ax1.set_xticks([])

    return ax1


def twoline_info(X, groupby, labels, proportions=False):
    if not proportions:
        info = ['Welch\'s t-test:\n',]
        info.append(str(
            X.groupby(groupby).apply(
                lambda x: scipy.stats.ttest_ind(x[labels[0]], x[labels[1]],
                                                equal_var=False)
            ).apply(pd.Series, index=('statistic', 'p'))))
        info.append(
            '\nn = \n' + str(X.unstack(0).groupby(groupby).count().iloc[0]) +
            '\n')
    else:
        info = ['Proportions normal z-test:\n',]
        info.append(str(
            X.groupby(groupby).apply(
                lambda x: proportion.proportions_ztest(
                    x.groupby('type').sum(), x.groupby('type').count())
            ).apply(pd.Series, index=('statistic', 'p'))))
    return info


def annotation(ax, x, y, text, fontsize=5, va='center'):
    ax.annotate('', (x[0], y), xytext=(x[1], y),
                textcoords='data',
                arrowprops={'arrowstyle': '-', 'shrinkA': 0,
                            'shrinkB': 0})

    ax.text((x[0] + x[1]) / 2, y, text, ha='center', va=va,
            fontsize=fontsize)


def annotation_padding(ax, x, axis='y'):
    transform = lambda x, ax: ax.transData.inverted().transform(
        ax.figure.dpi_scale_trans.transform(x))
    if axis == 'y':
        pad = (transform((0, x), ax) - transform((0, 0), ax))[1]
    else:
        pad = (transform((x, 0), ax) - transform((0, 0), ax))[0]

    return pad


def plot_reg_line(ax, x, y, label_pad=(0, 0), **plot_kwargs):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    reg_func = functools.partial(
        lambda x, slope, intercept: slope * x + intercept, slope=slope,
        intercept=intercept)
    ax.plot(
        (x.min(), x.max()), list(map(reg_func, (x.min(), x.max()))),
        **plot_kwargs)

    labelx = x.max() + annotation_padding(ax, label_pad[0], axis='x')
    labely = reg_func(x.max()) + annotation_padding(ax, label_pad[1])

    if p_value > 0.0001:
        text = f'p={p_value:.4f}'
    else:
        text = f'p={p_value:.2}'
    ax.text(
        labelx, labely,
        text, fontsize=5,
        c=plot_kwargs.get('c', 'k'), va='bottom')

    return p_value, r_value

def anova_table(ax, aov_text, fig_pad=(60, 5)):
    aov_labels = '\n' + '\n'.join([f'{k:>18s}:{"":>8s}' for k, v in aov_text.items()])
    aov_values = '\n' + '\n'.join(aov_text.values())

    fig_size_pix = ax.figure.get_size_inches() * ax.figure.dpi
    text_loc = ax.figure.transFigure.inverted().transform(
        (fig_size_pix[0] - fig_pad[0], fig_size_pix[1] - fig_pad[1]))
    text_loc_values = ax.figure.transFigure.inverted().transform(
        (fig_size_pix[0] - fig_pad[0] - 25, fig_size_pix[1] - fig_pad[1]))

    bbox_style = {'facecolor':'white', 'pad': 1.0, 'alpha': 0.25}
    ax.text(
        *text_loc , aov_labels, ha='right', ma='right',
        transform=ax.figure.transFigure, fontsize=5, va='top', bbox=bbox_style)
    ax.text(
        *text_loc_values, aov_values, ha='center', ma='center',
        transform=ax.figure.transFigure, fontsize=5, va='top')
    ax.text(
        *text_loc, 'ANOVA Table'.center(26), ha='right',
        transform=ax.figure.transFigure, fontsize=5, va='top',
        fontweight='bold')

def significance_symbols(x):
    sig_symbols = pd.Series(
        ('***', '**', '*', 'n.s.'),
        index=pd.IntervalIndex.from_tuples(
            ((-1, 0.001), (0.001, 0.01), (0.01, 0.05), (0.05, 1))))
    symbols = sig_symbols.reindex(x)
    try:
        symbols.index = x.index
    except:
        pass
    return symbols
