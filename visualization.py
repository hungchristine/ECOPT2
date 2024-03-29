# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 16:03:55 2021

@author: chrishun
"""
import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import (MultipleLocator, IndexLocator, FuncFormatter, PercentFormatter)
from matplotlib.patches import Patch
import numpy as np
from cycler import cycler

from matplotlib.backends.backend_pdf import PdfPages
import os
import itertools
import logging

log = logging.getLogger(__name__)

#### Define custom colormaps
# Paired colormap for comparing tecs
paired = LinearSegmentedColormap.from_list('paired', colors=['indigo', 'thistle', 'mediumblue', 'lightsteelblue',
                                                             'darkgreen', 'yellowgreen', 'olive', 'lightgoldenrodyellow',
                                                             'darkorange', 'navajowhite', 'darkred', 'salmon'], N=12)
light = LinearSegmentedColormap.from_list('light', colors=['thistle', 'lightsteelblue', 'yellowgreen',
                                                           'lightgoldenrodyellow', 'navajowhite', 'salmon'], N=6)
dark = LinearSegmentedColormap.from_list('dark', colors=['indigo', 'mediumblue', 'darkgreen',
                                                         'olive', 'darkorange', 'darkred'], N=6)
paired_tec = LinearSegmentedColormap.from_list('paired_by_tec', colors=['indigo', 'mediumblue', 'darkgreen',
                                                                        'olive', 'darkorange', 'darkred', 'thistle',
                                                                        'lightsteelblue', 'yellowgreen', 'lightgoldenrodyellow',
                                                                        'navajowhite', 'salmon'], N=12)
paired_dict = {'A':LinearSegmentedColormap.from_list('A',colors=['indigo','thistle']),
               'B':LinearSegmentedColormap.from_list('B',colors=['mediumblue','lightsteelblue']),
               'C':LinearSegmentedColormap.from_list('C',colors=['darkgreen','yellowgreen']),
               'D':LinearSegmentedColormap.from_list('D',colors=['olive','lightgoldenrodyellow']),
               'E':LinearSegmentedColormap.from_list('E',colors=['darkorange','navajowhite']),
               'F':LinearSegmentedColormap.from_list('F',colors=['darkred','salmon'])}

paired_cycler = cycler(color=['indigo', 'mediumblue', 'darkgreen', 'olive', 'darkorange', 'darkred'])

# Colormap for lifecycle phases
cmap_em = LinearSegmentedColormap.from_list('impacts', ['lightsteelblue', 'midnightblue',
                                                          'silver', 'grey', 'lemonchiffon', 'gold'], N=6)
# Colormap for technology stocks
tec_cm = LinearSegmentedColormap.from_list('tec', ['xkcd:burgundy', 'xkcd:light mauve'])
tec_cm = cm.get_cmap(tec_cm, 5)
tec_cm_blue = LinearSegmentedColormap.from_list('tec', ['xkcd:dark grey blue', 'xkcd:light grey blue'])
tec_cm_blue = cm.get_cmap(tec_cm_blue, 5)
tec_cm4 = ListedColormap(np.vstack((tec_cm(np.linspace(0, 1, 5)),
                                     tec_cm_blue(np.linspace(0, 1, 5)))), name='tec')

hatch = ['','.','/','|','x','-']

#%% Helper functions
def plot_arrange(fleet):
    """Figure out subplot arrangement based on number of regions."""

    ord_reg = [reg for reg in fleet.sets.reg]
    cat_type = CategoricalDtype(categories=ord_reg, ordered=True)

    ord_fleetreg = [reg for reg in fleet.sets.fleetreg]

    if len(ord_fleetreg) == 1:
        plt_array = (1,1)
    else:
        if len(ord_fleetreg) % 3 == 0:
            plt_array = (int(len(ord_fleetreg) / 3), 3)
            empty_spots = 0
        elif len(ord_fleetreg) % 2 == 0:
            plt_array = (int(len(ord_fleetreg) / 2), 2)
            empty_spots = 0
        else:
            plt_array = (int(len(ord_fleetreg) / 3) + 1, 3)
            empty_spots = (plt_array[0] * plt_array[1]) - len(ord_fleetreg)
    return plt_array, empty_spots, cat_type


def remove_subplots(ax, empty_spots):
    """Remove extra (empty) subplots from layout."""

    for i in range(empty_spots):
        ax[-1, -1+i].remove()  # remove from bottom row, rightmost subplot first


def fix_age_legend(ax, pp, cropx, max_year, title='Vehicle ages'):
    """Customize legend formatting for stock figures."""

    patches, labels = ax.get_legend_handles_labels()

    if len(labels) == 12:
        order = [11, 9, 7, 5, 3, 1, 10, 8, 6, 4, 2, 0]
        labels = [x + ', ' + y for x, y in itertools.product(['BEV', 'ICEV'], ['mini', 'small', 'medium', 'large', 'executive', 'luxury and SUV'])]
        ax.legend([patches[idx] for idx in order], [labels[idx] for idx in range(11, -1, -1)],
                  bbox_to_anchor=(1.05, 1.0), loc='upper left', ncol=2, title=title, borderaxespad=0.)
    elif len(labels) == 6:
        order = [5, 3, 1, 4, 2, 0]
        ax.legend([patches[idx] for idx in order], [labels[idx] for idx in order], bbox_to_anchor=(1.05, 1.0),
                  loc='upper left', ncol=2, title=title, borderaxespad=0.)
    elif len(labels) > 34:
        ax.legend(patches, labels, bbox_to_anchor=(1.05, 1.0), loc='upper left', ncol=3, title=title, borderaxespad=0.)
    else:
        ax.legend(patches, labels, bbox_to_anchor=(1.05, 1.0), loc='upper left', ncol=2, title=title, borderaxespad=0.)

    if cropx and ax.get_xlim()[1] == 80:
        ax.set_xlim(right=max_year)


def plot_subplots(fig, axes, grouped_df, title, labels=None, cmap='jet', xlabel='year'):
    """Plot input parameters by segment."""

    for (key, ax) in zip(grouped_df.groups.keys(), axes.flatten()):
        d = grouped_df.get_group(key)
        if d.index.nlevels == 3:
            if title == 'initial stock of each cohort':
                d = grouped_df.get_group(key).reset_index(level=[0], drop=True)
                d = d.unstack('fleetreg')
            else:
                d = grouped_df.get_group(key).reset_index(level=[0, 1], drop=True)

        elif d.index.nlevels == 2:
            d = grouped_df.get_group(key).reset_index(level=[0], drop=True)

        d.plot(ax=ax, cmap=cmap, legend=False)

        ax.set_xlabel(xlabel)
        ax.set_title(key, fontsize=10, fontweight='bold')

        ax.xaxis.set_minor_locator(IndexLocator(2, 0))
        ax.grid(which='minor', axis='x', c='lightgrey', alpha=0.55, linestyle=':', lw=0.3)
        ax.grid(which='major', axis='x', c='darkgrey', alpha=0.75, linestyle=':', lw=1)

        ax.grid(which='minor', axis='y', c='lightgrey', alpha=0.55, linestyle=':', lw=0.3)
        ax.grid(which='major', axis='y', c='darkgrey', alpha=0.75, linestyle=':', lw=1)

        plt.subplots_adjust(hspace=0.3, wspace=0.12)
        fig.suptitle(title, y=0.95)

        ax.set_xbound(0, 80)
    if labels:
        ax.legend(labels=labels, bbox_to_anchor=(0.2, -0.3), ncol=2, fontsize='large', borderaxespad=0.)

    return ax


def sort_ind(df, cat_type, fleet):
    """Sort Index by region using CategoricalIndex.

    Parameters
    ----------
    ind : pd.Index or pd.MultiIndex
        Index to sort.
    cat_type : CategoricalDType
        Ordered CategoricalDType to use as sort pattern.
    fleet : FleetModel
        FleetModel instantiation.

    Returns
    -------
    ind : pd.Index or pd.MultiIndex
        Reordered index.

    """
    if isinstance(df.index, pd.MultiIndex):
        ind = df.index.names
        # find levels with reg or fleetreg
        for i, lvl in enumerate(df.index.levels):
            if (df.index.levels[i].name == 'reg') or (df.index.levels[i].name == 'fleetreg'):
                col = df.index.levels[i].name
                df = df.reset_index()  # move index to columns to conver type
                df[col] = df[col].astype(cat_type)
                break
        df = df.set_index(ind)
        df.sort_index(inplace=True)
    else:
        df.index = df.index.astype(cat_type)

    return df


def fix_tuple_axis_labels(fig, axes, axis_label, label_level=1, isAxesSubplot=False):
    """
    Create simple axis tick labels from tuples.

    Parameters
    ----------
    fig : Pyplot figure
        Figure containing subplots.
    axes : array of AxesSubplots or AxesSubplot
        Subplot(s) to label.
    axis_label : str
        Desired axis label.
    label_level : int, optional
        Index of tuple element to use as axis tick labels. The default is 1.
    isAxesSubplot : bool, optional
        Indicates "axes" is a single subplot. The default is False.

    Returns
    -------
    None.

    """
    fig.canvas.draw()

    def reduce_tuple(ax):
        new_labels = [x.get_text().strip("()").split(",")[label_level] for x in ax.get_xticklabels() if len(x.get_text()) > 0]
        ax.xaxis.set_ticklabels(new_labels)
        ax.set_xlabel(axis_label)

    if isAxesSubplot:
        reduce_tuple(axes)
    elif axes.ndim == 2:
        for ax in axes[-1, :]:
            reduce_tuple(ax)
    elif axes.ndim == 1:
        for ax in axes:
            reduce_tuple(ax)


def get_ref_ax(axes):
    if axes.ndim == 2:
        return axes[0,-1]
    elif axes.ndim == 1:
        return axes[-1]


def flip(items, ncol):
    """Make legend entries fill by row rather than columns."""
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])


def trunc_colormap(cmap, min_val=0, max_val=1, n=50):
    """
    Truncate built-in matplotlib colourmaps.

    From:
    https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib

    """
    new_cmap = LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=min_val, b=max_val),
                                                 cmap(np.linspace(min_val, max_val, n)))
    return new_cmap


def export_fig(fp, ax, pp, export_pdf=True, export_png=False, png_name=None):
    """ Export figures to pdf or png. """

    if export_pdf:
        pp.savefig(bbox_inches='tight')
    if export_png:
        if not png_name:
            png_name = ax.get_title()
        plt.savefig(os.path.join(fp, png_name+'.png'), format='png', bbox_inches='tight')


#%% for exporting values

from pprint import pprint
from inspect import getmembers
from types import FunctionType

def attributes(obj):
    disallowed_names = {
      name for name, value in getmembers(type(obj))
        if isinstance(value, FunctionType)}
    return {
      name: getattr(obj, name) for name in dir(obj)
        if name[0] != '_' and name not in disallowed_names and hasattr(obj, name)}

def print_attributes(obj):
    pprint(attributes(obj))


#%%
# TODO: pass in dict of experiment specific labels, e.g., new component names, and/or longform names for segments, etc
def vis_GAMS(fleet, fp, filename, param_values, export_png=False, export_pdf=True, max_year=2050, cropx=True, suppress_vis=False):
    """
    Visualize model results and input.

    Parameters
    ----------
    fleet : FleetModel
        Contains experiment input and results.
    fp : str
        Filepath for saving files.
    filename : str
        Experiment identifier.
    param_values : dict of {str: int|float|dict|list}
        Dictionary of parameter values for experiment.
    export_png : bool
        Toggle for exporting figures a png.
    export_pdf : bool, optional
        Toggle for exporting figures in a pdf. The default is True.
    max_year : int, optional
        If cropping figures, new value for max x-limit. The default is 50.
    cropx : bool, optional
        Toggle for cropping figures to max_year. The default is True.
    suppress_vis : bool, optional
        Turn off interactive mode for pyplot. The default is False.
    """

    pp = PdfPages(os.path.join(fp, 'output_vis_' + filename + '.pdf'))
    plt.rcParams.update({'figure.max_open_warning': 0})  # suppress max 20 figures warning
    if suppress_vis:
        plt.ioff()

    plt_array, empty_spots, cat_type = plot_arrange(fleet)

    #### Make summary page describing parameters for model run
    if param_values:
        div_page = plt.figure(figsize=(25, 8))
        ax = plt.subplot(111)
        ax.axis('off')
        plt.text(0.05, 0.9, f'Run name: {filename}')
        for i, (key, value) in enumerate(param_values.items()):
            plt.text(0.05, 0.85-i*(0.05), key, fontsize=14)
            plt.text(0.15, 0.85-i*(0.05), str(value), fontsize=14)

        export_fig(fp, ax, pp, export_pdf, export_png, 'tec-seg-cohort')
    else:
        print('Could not make parameter table in export PDF')

#%% Begin figure plotting

    #%% Stock additions by segment, technology and region (absolute)
    """--- Plot stock additions by segment, technology and region ---"""
    try:
        fig, axes = plt.subplots(plt_array[0], plt_array[1],
                                 sharex=True, sharey='row', dpi=300)

        tmp = fleet.stock_add.sum(axis=1).unstack('seg').unstack('tec')
        tmp.reset_index(level='prodyear', inplace=True)
        tmp['prodyear'] = tmp['prodyear'].astype(int)
        tmp.set_index('prodyear', drop=True, append=True, inplace=True)
        tmp = tmp.loc(axis=0)[:, 2020:]

        # Check if scaling of stock additions required
        level_2050 = tmp.groupby(['fleetreg', 'prodyear']).sum().sum(axis=1).unstack('prodyear')[2050].mean()
        if level_2050 > 1e6:
            tmp /= 1e6
            units = 'millions'
        elif level_2050 > 15e3:
            tmp /= 1e3
            units = 'thousands'
        else:
            units = ''

        tmp = sort_ind(tmp, cat_type, fleet)

        tmp = tmp.groupby(['fleetreg'], observed=True)
        tmp = {key: group for key, group in tmp if len(group) > 0}

        for (key, ax) in zip(tmp.keys(), axes.flatten()):
            tmp[key].plot(ax=ax, kind='area', cmap=paired, lw=0, legend=False, title=f'{key}')
            ax.xaxis.set_major_locator(IndexLocator(10, 0))
            ax.xaxis.set_tick_params(rotation=45)
            ax.set_xbound(0, 50)

        fix_tuple_axis_labels(fig, axes, 'year')
        remove_subplots(axes, empty_spots)

        ref_ax = get_ref_ax(axes)
        fix_age_legend(ref_ax, pp, cropx, max_year, 'Technology and segment')
        fig.text(0, 0.5, f'Stock additions \n {units} ', rotation='vertical', ha='center', va='center')

        plt.ylim(bottom=0)

        fig.suptitle('Stock additions, by segment, technology and region', y=0.995)

        export_fig(fp, ax, pp, export_pdf, export_png, png_name=fig._suptitle.get_text())

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: Market share by seg, tec, reg')
        print(e)

    #%% market shares by segment and technology (normalized)
    """--- Plot stock addition shares by segment and technology ---"""
    try:
        fig, axes = plt.subplots(plt_array[0], plt_array[1],
                                 sharex=True, sharey=True, dpi=300)
        tmp = fleet.add_share
        tmp = sort_ind(tmp, cat_type, fleet)
        tmp.reset_index(level='prodyear', inplace=True)
        tmp['prodyear'] = tmp['prodyear'].astype(int)
        tmp.set_index('prodyear', drop=True, append=True, inplace=True)
        tmp = tmp.groupby(['fleetreg'], sort=False, observed=True)
        tmp = {key: group for key, group in tmp if len(group) > 0}

        for (key, ax) in zip(tmp.keys(), axes.flatten()):
            tmp[key].plot(ax=ax, kind='area', cmap=paired, lw=0, legend=False, title=f'{key}')
            ax.set_xbound(0,50)
            ax.xaxis.set_major_locator(IndexLocator(10, 0))
            ax.xaxis.set_minor_locator(IndexLocator(2, 0))
            ax.xaxis.set_tick_params(rotation=45)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.set_ybound(0, 1)

        fix_tuple_axis_labels(fig, axes, 'year')
        remove_subplots(axes, empty_spots)
        ref_ax = get_ref_ax(axes)
        fix_age_legend(ref_ax, pp, cropx, max_year, 'Technology and segment')
        fig.text(0, 0.5, 'Total market share', rotation='vertical', ha='center', va='center')
        fig.suptitle('Stock additions, by technology, segment and region,\n as share of total stock additions', y=1.05)

        export_fig(fp, ax, pp, export_pdf, export_png, png_name= 'share stock_add_tec, seg, reg')

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: Market share, by seg, tec')
        print(e)


    #%% Total stock shares by segment and technology (un-normalized)
    """--- Plot tech split of stock additions by segment ---"""
    try:
        tmp = fleet.add_share.div(fleet.add_share.sum(axis=1, level='seg'), axis=1, level='seg')
        tmp = sort_ind(tmp, cat_type, fleet)
        tmp.reset_index(level='prodyear', inplace=True)
        tmp['prodyear'] = tmp['prodyear'].astype(int)
        tmp.set_index('prodyear', drop=True, append=True, inplace=True)
        tmp = tmp.groupby(['fleetreg'], sort=False, observed=True)
        tmp = {key: group for key, group in tmp if len(group) > 0}

        fig, axes = plt.subplots(len(fleet.sets.seg), len(tmp.keys()),
                                 sharex=True, sharey=True, dpi=300)

        for col, reg in enumerate(tmp.keys()):
            for row, seg in enumerate(fleet.sets.seg):
                tmp[reg][seg].plot(ax=axes[row,col], kind='area', cmap=paired_dict[seg], lw=0, legend=False)
                axes[row,col].set_xbound(0, 50)
                axes[row,col].xaxis.set_major_locator(IndexLocator(10, 0))
                axes[row,col].xaxis.set_minor_locator(IndexLocator(2, 0))
                axes[row,col].xaxis.set_tick_params(rotation=45)
                axes[row,col].set_ybound(0,1)
                axes[row,col].yaxis.set_ticklabels([])
                if not col:
                    axes[row,col].text(-10, 0.5, seg, va='center')
                if not row:
                    # annotations for top row
                    axes[row,col].text(25, 1.1, reg, ha='center')
        fix_tuple_axis_labels(fig, axes, 'year')

        ref_ax = get_ref_ax(axes)
        fix_age_legend(ref_ax,  pp, cropx, max_year, 'Technology')
        fig.text(0.05, 0.5, 'Segment market share \n(0-100%)', rotation='vertical', ha='center', va='center')
        fig.suptitle('Stock additions, by technology, segment and region, \n as share of segment stock', y=1.05)

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: Technology market share, by seg')
        print(e)


    #%% Segment shares of new tec by region
    """--- Plot market share of new tecs by segment and region ---"""
    try:
        fig, axes = plt.subplots(plt_array[0], plt_array[1],
                                 sharex=True, sharey=True, dpi=300)

        tmp = fleet.add_share.div(fleet.add_share.sum(axis=1, level='seg'), axis=1, level='seg')
        tmp = tmp.drop('ICE', axis=1, level='tec')
        tmp.reset_index(level='prodyear', inplace=True)
        tmp['prodyear'] = tmp['prodyear'].astype(int)
        tmp.set_index('prodyear', drop=True, append=True, inplace=True)
        tmp = sort_ind(tmp, cat_type, fleet)
        tmp = tmp.groupby(['fleetreg'], observed=True)
        tmp = {key: group for key, group in tmp if len(group) > 0}

        for (key, ax) in zip(tmp.keys(), axes.flatten()):
            tmp[key].plot(ax=ax, cmap=dark, legend=False, title=f'{key}')
            ax.set_xbound(0, 50)
            ax.set_ybound(0, 1)
            ax.xaxis.set_major_locator(IndexLocator(10, 0))
            ax.xaxis.set_minor_locator(IndexLocator(2, 0))
            ax.xaxis.set_tick_params(rotation=45)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))

        fix_tuple_axis_labels(fig, axes, 'year')
        remove_subplots(axes, empty_spots)  # remove extraneous/empty subplots

        ref_ax = get_ref_ax(axes)
        fix_age_legend(ref_ax, pp, cropx, max_year, 'Segment')
        handles, labels = ref_ax.get_legend_handles_labels()
        labels = [label.strip('()').split(',')[0] for label in labels]
        ref_ax.legend(handles, labels, loc=2, bbox_to_anchor=(1.01, 1))
        fig.suptitle('Market share of new tec by segment and region', y=0.995)

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: new tec market share, by seg, reg')
        print(e)

    #%% Regional shares of new tecs by segment
    """--- Plot market share of new tec by segment and region ---"""
    try:
        fig, axes = plt.subplots(3, 2, figsize=(9,9),
                                 sharex=True, sharey='row', dpi=300)
        plt.subplots_adjust(top=0.95, hspace=0.25, wspace=0.05)

        tmp = fleet.stock_add.div(fleet.stock_add.sum(level=['seg', 'tec', 'prodyear']))
        tmp.dropna(axis=1, how='all', inplace=True)
        tmp = tmp.drop('ICE', axis=0, level='tec')
        tmp = sort_ind(tmp, cat_type, fleet)
        tmp.reset_index(level='prodyear', inplace=True)
        tmp['prodyear'] = tmp['prodyear'].astype(int)
        tmp.set_index('prodyear', drop=True, append=True, inplace=True)
        tmp = tmp.unstack('fleetreg').droplevel('age', axis=1).droplevel('tec', axis=0).dropna(how='all', axis=1).groupby(['seg'])

        alpha = np.linspace(1,0.1, len(fleet.sets.fleetreg))

        i = 0
        for (key, ax) in zip(tmp.groups.keys(), axes.flatten()):
            cmap = (dark(i/dark.N))  # use new tec segment colours
            plot_data = tmp.get_group(key).droplevel('seg')
            pl = ax.stackplot(plot_data.index.map(float).values,
                               plot_data.T.values,
                               colors=[cmap]*len(fleet.sets.fleetreg),
                               ec='w',
                               lw=1)
            for country_area, al in zip(pl, alpha):
                country_area.set_alpha(al)  # set alpha for each region

            ax.set_xbound(2020, 2050)

            ax.xaxis.set_tick_params(rotation=45)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.set_ybound(0, 1)
            ax.set_title(key)
            i += 1

        ref_ax = get_ref_ax(axes)
        legend_elem = []
        for i in range(len(fleet.sets.fleetreg)):
            legend_elem.append(Patch(facecolor=dark(0), alpha=alpha[i], ec='w',label=fleet.sets.fleetreg[i]))
        leg = axes[0,-1].legend(handles=legend_elem, title='Region',
                                title_fontsize='medium', borderpad=1,
                                loc=2, labelspacing=1.5, handlelength=4, bbox_to_anchor=(1.01, 1))

        for patch in leg.get_patches():
            patch.set_height(20)
            patch.set_y(-6)
        fig.suptitle('Regional share of new tec market shares by segment ', y=0.995)

        export_fig(fp, ax, pp, export_pdf, export_png, png_name=fig._suptitle.get_text())

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: new tec market share, by seg, reg 2')
        print(e)

    #%% Total impacts by tec and lifecycle phase, against total stock
    """--- Plot total impacts by tec and lifecycle phase---"""
    try:
        fleet.all_impacts.sort_index(axis=1, level=0, ascending=False, inplace=True)

        fig = plt.figure(figsize=(14,9), dpi=300)

        gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1,3], hspace=0.05)
        ax2 = fig.add_subplot(gs[0])  # stock plot
        ax1 = fig.add_subplot(gs[1], sharex=ax2)  # impacts plot

        plot_ax2 = (fleet.tot_stock_plot.sum(axis=1).unstack('seg').sum(axis=1).unstack('tec').sum(level='year')/1e6)
        plot_ax2.plot(ax=ax2, kind='area', cmap=tec_cm, lw=0)
        (fleet.all_impacts/1e6).plot(ax=ax1, kind='area', lw=0, cmap=cmap_em)
        (fleet.bau_impacts/1e6).unstack(level='imp').plot(ax=ax1, kind='line')

        ax1.set_ylabel('Lifecycle climate impacts \n Mt $CO_2$-eq', fontsize=13)
        ax2.set_ylabel('Vehicles, millions', fontsize=13, labelpad=25)

        if cropx:
            ax1.set_xlim(right=max_year)
            ax2.set_xlim(right=max_year)
        handles, labels = ax1.get_legend_handles_labels()
        lvl0_lab = fleet.all_impacts.columns.get_level_values(0).unique()
        lvl1_lab = fleet.all_impacts.columns.get_level_values(1).unique()
        labels = [x+', '+y for x, y in itertools.product(lvl0_lab, lvl1_lab)] #['Production', 'Operation', 'End-of-life'], ['ICEV', 'BEV'])]
        labels.insert(0, 'Impacts, 0% electrification')
        ax1.legend(handles, labels, loc=0, fontsize=14)
        handles, labels = ax2.get_legend_handles_labels()
        print(labels)
        ax2.legend(handles, labels, loc=0, fontsize=14, framealpha=1)#['BEV', 'ICEV'], loc=0, fontsize=14, framealpha=1)

        ax2.set_xbound(0, 50)
        ax1.set_ylim(bottom=0)

        plt.setp(ax2.get_yticklabels(), fontsize=14)  # modify x-tick label size

        fix_tuple_axis_labels(fig, ax1, 'year', 0, isAxesSubplot=True)
        plt.xlabel('year', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        export_fig(fp, ax, pp, export_pdf, export_png, png_name='LC_impacts_vs_stock')

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: Total stocks vs lifecycle impacts by tec')
        print(e)

    #%% Comparison of emission impacts
    # plot 2020, 2035, 2050?
    # regions vs tec and lifecycle phase
    # regions vs tec and seg?

    # try:
    #     fig, axes = plt.subplots(plt_array[0], plt_array[1],
    #                              sharex=True, sharey=True, dpi=300)

    #     plt_groups = fleet.impacts.groupby('fleetreg')
         # for (reg, ax) in zip(plt_groups.keys(), axes.flatten()):
         #     plt_groups(reg)

         # make a line graph with all impact categories, normalized to base year and stock size

    #%% Regionalized fleet impacts
    try:
        fig, ax = plt.subplots(1,1, dpi=300)

        plot_impacts = fleet.impacts.sum(level=['fleetreg', 'tec'])
        plot_impacts.index = plot_impacts.index.swaplevel('fleetreg', 'tec')
        plot_impacts = sort_ind(plot_impacts, cat_type, fleet)
        plot_impacts.columns = plot_impacts.columns.astype(int)

        ax.set_prop_cycle(paired_cycler)

        (plot_impacts/1e6).T.plot(ax=ax, kind='area', lw=0, cmap=tec_cm4)
        ax.set_ylabel('Lifecycle climate impacts \n Mt $CO_2$-eq', fontsize=13)

        if cropx:
            ax.set_xlim(right=max_year)

        ax.set_xbound(2000, 2050)
        ax.set_ybound(0)

        fix_tuple_axis_labels(fig, ax, 'year', 0, isAxesSubplot=True)
        plt.xlabel('year', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        fix_age_legend(ax, pp, cropx, max_year, 'Technology and region')

        fig.suptitle('Fleet impacts by region and technology', y=0.95)

        export_fig(fp, ax, pp, export_pdf, export_png, png_name='LC_impacts')

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: Lifecycle impacts by tec and region')
        print(e)

    #%% Operation impacts by tec, benchmarked against regulation impacts targets
    """--- Plot operation impacts by tec ---"""
    try:
        if 'EUR' in fleet.sets.fleetreg:
            fig, ax = plt.subplots(1,1, dpi=300)

            plot_data = fleet.operation_impacts.sum(level=['tec','fleetreg'], axis=0).loc(axis=0)[:,'EUR'] / 1e6
            plot_data.index = plot_data.index.droplevel('fleetreg')
            plot_data = plot_data.T
            plot_data.index = plot_data.index.astype(int)
            cmap = LinearSegmentedColormap.from_list('temp', colors=['silver', 'grey'])
            cmap_cols = {'BEV':'silver', 'ICE': 'grey'}

            plot_data.plot.area(stacked=True, ax=ax, color=cmap_cols, lw=0)

            # plot regulation levels
            plt.hlines(442, xmin=0.16, xmax=0.6, linestyle='dotted', color='darkslategrey', label='EU 2030 target, \n 20% reduction from 2008 impacts', transform=ax.get_yaxis_transform())
            plt.hlines(185, xmin=0.6, xmax=1, linestyle='-.', color='darkslategrey', label='EU 2050 target, \n 60% reduction from 1990 impacts', transform=ax.get_yaxis_transform())
            plt.ylabel('Fleet operation impacts \n Mt $CO_2$-eq')

            if cropx:
                plt.xlim(right=max_year)

            ax.set_xbound(2000, 2050)
            ax.set_ybound(0)

            handles, labels = ax.get_legend_handles_labels()

            order = [1, 0, 2, 3]
            ax.legend([handles[x] for x in order],
                      [labels[x] for x in order],
                                loc=2, bbox_to_anchor= (1.01, 1.01))
            export_fig(fp, ax, pp, export_pdf, export_png, png_name='operation_impacts')

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: Operation impacts by tec')
        print(e)

    #%% Total stocks by segment
    """--- Plot total stocks by segment ---"""
    try:
        fig, ax = plt.subplots(1,1, dpi=300)
        plot_data = fleet.tot_stock_plot.sum(axis=1).unstack('seg').sum(axis=0, level=['year'])
        plot_data.index = plot_data.index.astype(int)
        plot_data.plot(kind='area', ax=ax, cmap='jet', lw=0, title='Total stocks by segment')
        ax.set_xbound(2000, 2050)
        ax.set_ybound(lower=0)
        ax.legend(loc=2, bbox_to_anchor=(1.01,1), title='Vehicle segment')
        export_fig(fp, ax, pp, export_pdf, export_png, png_name=ax.get_title())

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: Total stocks by seg')
        print(e)

    #%% Total stocks by region
    """--- Plot total stocks by region ---"""
    try:
        fig, ax = plt.subplots(1,1, dpi=300)

        tmp = fleet.tot_stock_plot.sum(axis=1).unstack('fleetreg').sum(axis=0, level=['year']).T
        tmp = sort_ind(tmp, cat_type, fleet).sort_index()
        tmp = tmp.T
        tmp.index = tmp.index.astype(int)
        tmp.plot(kind='area', ax=ax, cmap='jet', lw=0, title='Total stocks by region')
        ax.set_xbound(2000, 2050)
        ax.set_ybound(lower=0)
        ax.legend(loc=2, bbox_to_anchor=(1.01,1), title='Region')
        export_fig(fp, ax, pp, export_pdf, export_png, png_name=ax.get_title())

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: Total stocks by reg')
        print(e)

    #%% Total stock by technology and region
    """--- Plot total stock share of new tec ---"""
    try:
        fig, ax = plt.subplots(1,1, dpi=300)
        plot_data = fleet.tot_stock_plot.sum(axis=1).unstack('tec').unstack('fleetreg').sum(axis=0, level='year')
        plot_data.index = plot_data.index.astype(int)
        plot_data.plot(kind='area', ax=ax, lw=0, cmap=paired_tec,
                            title='Total stocks by region and technology')
        ax.set_xbound(2000, 2050)
        ax.set_ybound(lower=0)
        fix_age_legend(ax, pp, cropx, max_year, 'Vehicle region and technology')
        export_fig(fp, ax, pp, export_pdf, export_png, png_name=ax.get_title())

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: Total stocks by reg, tec')
        print(e)


    #%% Total stocks by segment and technology
    """--- Plot total stocks by age, segment and technology ---"""
    try:
        fig, ax = plt.subplots(1,1, dpi=300)
        plot_data = fleet.tot_stock_plot.sum(axis=1).unstack('seg').unstack('tec').sum(axis=0, level='year')
        plot_data.index = plot_data.index.astype(int)
        plot_data.plot(kind='area', ax=ax, cmap=paired, lw=0,
                            title='Total stocks by segment and technology')
        ax.set_xbound(2000, 2050)
        ax.set_ybound(lower=0)
        fix_age_legend(ax, pp, cropx, max_year, 'Vehicle segment and technology')
        export_fig(fp, ax, pp, export_pdf, export_png, png_name=ax.get_title())

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: Total stocks by age, seg, tec')
        print(e)

    #%% Total stocks by age, segment and technology
    """--- Plot total stocks by age, segment and technology ---"""

    try:
        fig, ax = plt.subplots(1,1, dpi=300)

        stock_tec_seg_reg = fleet.tot_stock_plot.sum(axis=1).unstack('seg').unstack('tec').unstack('fleetreg').T
        stock_tec_seg_reg = sort_ind(stock_tec_seg_reg, cat_type, fleet)
        stock_tec_seg_reg = stock_tec_seg_reg.T
        stock_tec_seg_reg.columns = stock_tec_seg_reg.columns.swaplevel('fleetreg', 'tec')
        stock_tec_seg_reg.index = stock_tec_seg_reg.index.astype(int)

        stock_tec_seg_reg.plot(kind='area', ax=ax, cmap='jet', lw=0,
                               title='Total stocks by segment, technology and region')
        ax.set_xbound(2000, 2050)
        ax.set_ybound(lower=0)

        fix_age_legend(ax, pp, cropx, max_year, 'Vehicle segment, technology and region')
        export_fig(fp, ax, pp, export_pdf, export_png, png_name=ax.get_title())

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: Total stocks, by age, seg and tec')
        print(e)


    #%% Addition to stocks by segment, technology and region
    """--- Plot addition to stocks by segment, technology and region ---"""
    try:
        fig, ax = plt.subplots(1, 1, dpi=300)
        plot_stock_add = fleet.stock_add.sum(level=['tec', 'fleetreg', 'prodyear'])
        plot_stock_add = plot_stock_add.loc[:, (plot_stock_add != 0).any(axis=0)]
        plot_stock_add = plot_stock_add.unstack(['tec', 'fleetreg']).droplevel(axis=1, level=0).T
        plot_stock_add = sort_ind(plot_stock_add, cat_type, fleet)
        plot_stock_add = plot_stock_add.T
        plot_stock_add.index = plot_stock_add.index.astype(int)

        plot_stock_add.plot(ax=ax, kind='area', cmap=tec_cm4, lw=0, legend=True, title='Stock additions by technology and region')
        ax.set_xbound(2000, 2050)
        ax.set_ybound(lower=0)

        fix_age_legend(ax, pp, cropx, max_year, 'Vehicle technology and region')

        plt.xlabel('year')
        plt.ylabel('Vehicles added to stock')
        export_fig(fp, ax, pp, export_pdf, export_png, png_name=ax.get_title())


    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: addition to stocks, by tec and reg')
        print(e)

    #%% Total stocks by segment, technology and region
    """--- Plot total stocks by segment, technology and region ---"""
    try:
        fig, ax = plt.subplots(1, 1, dpi=300)

        plot_stock = fleet.tot_stock.sum(axis=1).unstack(['tec', 'fleetreg']).sum(level=['year'])

        plot_stock = plot_stock.stack('tec').T  # remove MultiIndex to set Categorical type for regions
        plot_stock = sort_ind(plot_stock, cat_type, fleet)
        plot_stock = plot_stock.T
        plot_stock = plot_stock.unstack('tec').swaplevel('tec', 'fleetreg', axis=1).sort_index(axis=1, level='tec')
        plot_stock.index = plot_stock.index.astype(int)
        plot_stock.plot(ax=ax, kind='area', cmap=tec_cm4, lw=0, legend=True, title='Total stock by technology and region')
        ax.set_xbound(2000, 2050)
        ax.set_ybound(lower=0)

        fix_age_legend(ax, pp, cropx, max_year, 'Vehicle technology and region')

        ax.set_xlabel('year')
        ax.set_ylabel('Vehicles in stock')
        export_fig(fp, ax, pp, export_pdf, export_png, png_name=ax.get_title())
        plt.show()  # without this line, legend shows as "BEV" and "ICEV" ...????

    except Exception as e:
        print('\n *****************************************')
        print(f'Error with stock additions, {key}')
        print(e)

    #%% Total resource use - against total stock
    """--- Plot total resource use ---"""
    try:
        for resource in fleet.resources.columns.get_level_values(1).unique():
            fig = plt.figure(figsize=(14,9), dpi=300)

            gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1,3], hspace=0.15)
            ax2 = fig.add_subplot(gs[0])
            ax1 = fig.add_subplot(gs[1], sharex=ax2)

            plot_resources = fleet.resources.loc[:, (slice(None), resource)].copy()
            plot_resources.index = plot_resources.index.astype(int)
            plot_virg_mat = (fleet.parameters.virg_mat_supply.copy().filter(like=resource, axis=0)*1000).T  # primary material supply is in tons; convert to kg
            plot_virg_mat.index = plot_virg_mat.index.astype(int)

            if plot_resources.max().mean() >= 1e6:
                plot_resources /= 1e6
                plot_virg_mat /= 1e6
                units = 'Mt'
            elif plot_resources.max().mean() >= 1e4:
                plot_resources /= 1e3
                plot_virg_mat /= 1e3
                units = 't'
            else:
                units = 'kg'

            if (plot_resources.values < 0).any():
                log.warning('----- Some resource demand is negative! Setting to 0. \n')
                print(plot_resources.loc[plot_resources.values < 0])  # print negative values for troubleshooting; may be very small or boundary condition
                print('\n')
                plot_resources.where(plot_resources>=0, 0, inplace=True)   # set value to 0 for plotting

            # plot use of materials
            plot_resources.loc[2020] = np.nan # add 2020 to properly align plots
            plot_resources.sort_index(ascending=True, inplace=True)
            plot_resources.plot(ax=ax1, kind='area', lw=0, cmap='jet')

            plot_virg_mat = plot_virg_mat.loc[2020:].sum(axis=1)
            plot_virg_mat.plot(ax=ax1, lw=4, kind='line', alpha=0.7)

            # plot fleet evolution
            fleet_evol = (fleet.tot_stock_plot.sum(axis=1).unstack('seg').sum(axis=1).unstack('tec').sum(level='year')/1e6).loc['2020':]
            fleet_evol.index = fleet_evol.index.astype(int)
            fleet_evol.plot(ax=ax2, kind='area', cmap=tec_cm, lw=0)
            lab = fleet_evol.columns

            ax1.set_ylabel(f'{resource} used in new batteries \n {units} {resource}', fontsize=14)
            ax2.set_ylabel('Stock, millions', fontsize=14, labelpad=25)
            if cropx:
                ax1.set_xlim(right=max_year)
                ax2.set_xlim(right=max_year)

            handles, labels = ax1.get_legend_handles_labels()
            new_labels = []
            for label in labels:
                new_labels.append(str(label)[1:-3].replace(",", "").replace("'","").capitalize())
            new_labels[0] = 'Total primary material available'
            ax1.legend(handles, new_labels, title=f'{resource} source used')

            handles, labels = ax2.get_legend_handles_labels()
            ax2.legend(handles, lab,  loc=4, fontsize=14, framealpha=1)# ['BEV', 'ICEV'], loc=4, fontsize=14, framealpha=1)

            ax1.set_xbound(2020, 2050)
            ax2.set_xbound(2020, 2050)
            ax1.set_ybound(lower=0, upper=plot_resources.sum(axis=1).max()*1.1)
            ax2.set_ybound(lower=0)

            plt.setp(ax2.get_yticklabels(), fontsize=14)
            plt.xlabel('year', fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

        export_fig(fp, ax, pp, export_pdf, export_png, png_name='Total resource use')

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: Total resource use')
        print(e)

    #%% Total resource use
    """--- Plot total resource use ---"""
    try:
        gpby_class = {supp: mat for mat, li in fleet.sets.mat_prod.items() for supp in li}

        resource_use = fleet.resources.copy()
        resource_use.index = resource_use.index.astype(int)
        resource_use.loc[2020] = np.nan
        resource_use.sort_index(ascending=True, inplace=True)

        # get total available primary supplies of each material category
        prim_supply = (fleet.parameters.virg_mat_supply.T.groupby(gpby_class, axis=1).sum() * fleet.parameters.raw_data.eur_batt_share)*1000 # primary supply is in t, convert to kg
        prim_supply.index = prim_supply.index.astype(int)
        prim_supply = prim_supply.loc[2020:]

        fig, axes = plt.subplots(len(fleet.sets.mat_cat), 1, sharex=True, dpi=300)
        plt.subplots_adjust(top=0.85, hspace=0.4)

        for i, mat in enumerate(fleet.sets.mat_prod.keys()):
            plot_resources = resource_use.loc[:, (slice(None), mat)]
            plot_prim_supply = prim_supply[mat]

            if (plot_resources.values < 0).any():
                log.warning('----- Some resource demand is negative! Setting to 0. \n')
                print(plot_resources.loc[plot_resources.values < 0])  # print negative values for troubleshooting; may be very small or boundary condition
                print('\n')
                plot_resources.where(plot_resources>=0, 0, inplace=True)  # set value to 0 for plotting
            # scale y-axis if necessary
            if plot_resources.max().mean() >= 1e6:
                plot_resources /= 1e6
                plot_prim_supply /= 1e6
                units = 'Mt'
                ylabel = f'Mt {mat}'
            elif plot_resources.max().mean() >= 1e4:
                plot_resources /= 1e3
                plot_prim_supply /= 1e3
                units = 't'
                ylabel = f't {mat}'
            else:
                units = 'kg'
                ylabel = f'kg {mat}'

            plot_resources.plot(ax=axes[i], kind='area', lw=0, stacked=True,
                                cmap='jet', legend=False, alpha=0.9)
            plot_prim_supply.plot(ax=axes[i], ls='-.', lw=2, alpha=0.7)

            axes[i].set_title(f'{mat} used in new batteries',
                              fontsize=10, fontweight='bold')

            axes[i].set_ylabel(ylabel)
            axes[i].set_ybound(lower=0, upper=plot_resources.sum(axis=1).max()*1.1)

            if cropx:
                axes[i].set_xlim(left=2020, right=max_year)
            else:
                axes[i].set_xlim(2020, 2050)

        handles, labels = axes[0].get_legend_handles_labels()
        new_labels = []
        for label in labels:
            new_labels.append(str(label)[1:-3].replace(",", "").replace("'","").capitalize())
        new_labels[0] = 'Total primary material available'

        # rearrange legend entries
        label_order = [1, 2, 0]
        handles = [handles[i] for i in label_order]
        new_labels= [new_labels[i] for i in label_order]

        axes[-1].legend(handles, new_labels, loc=2, bbox_to_anchor=(0, -0.7, 1, 0.2),
                        ncol=2, mode='expand', borderaxespad=0, handlelength=3,
                       title='Material source used for stock additions')

        plt.xlabel('year', fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        fig.align_ylabels()

        fig.suptitle('Primary and secondary material use', y=0.998)

        export_fig(fp, ax, pp, export_pdf, export_png, png_name=fig._suptitle.get_text())

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: Primary vs secondary material use')
        print(e)

    #%% Production mixes for primary critical materials
    """--- Plot production mixes for primary critical materials ---"""
    try:
        fig, axes = plt.subplots(len(fleet.sets.mat_cat), 1, sharex=True, dpi=300)
        plt.subplots_adjust(top=0.85, hspace=0.4)
        plot_data = fleet.mat_mix.copy()
        plot_data.index = plot_data.index.astype(int)
        level_2020 = plot_data.loc[2020].mean() # use as baseline value to scale y-axis
        plot_data = plot_data.loc[2020:2050]  # restrict to optimization period
        fleet.parameters.virg_mat_supply.columns = fleet.parameters.virg_mat_supply.columns.astype(int)
        supply_constr = fleet.parameters.virg_mat_supply.loc(axis=1)[2020:2050] * 1000 # virg_mat_supply is in t, not kg
        supply_constr = supply_constr.T
        plot_data.sort_index(axis=1, inplace=True)  # make sure material mixes and supply constraints are in the same order
        supply_constr.sort_index(axis=1, inplace=True)

        mat_cmaps = ['Purples','Blues', 'Greens', 'Oranges', 'Reds']
        # manually make yellow gradient colourmap
        yellows = ListedColormap(['lemonchiffon',
                                  'xkcd:faded yellow',
                                  'xkcd:dandelion',
                                  'gold',
                                  ], 'Yellows')
        supply_cmaps = [yellows, 'Oranges', 'Reds', 'Blues', 'Greens']

        # scale values for y-axis if necessary
        if plot_data.max().mean() >= 1e6:
            plot_data = plot_data / 1e6
            supply_constr = supply_constr / 1e6
            y_label = f'Mt {mat}'
        elif plot_data.max().mean() >= 1e4:
            plot_data = plot_data / 1e3
            supply_constr = supply_constr / 1e3
            y_label = f't {mat}'
        else:
            y_label = f'kg {mat}'

        n = len(fleet.sets.mat_prod.items())
        tmp = pd.DataFrame()

        for i, (mat, prods) in enumerate(fleet.sets.mat_prod.items()):
            # make supply constraint lines relative to actual materials used
            for j, prod in enumerate(prods):
                if j > 0:
                    tmp[prod] = plot_data[prods].iloc(axis=1)[0:j].sum(axis=1) + supply_constr[prods].iloc(axis=1)[j]
                elif j == 0:
                    tmp[prod] = supply_constr[prod]

            cmap = plt.get_cmap(mat_cmaps[i])

            # FIXME: Using the linestyle cycler 'breaks' the PDF export.
            # This is also the case if the cycler is used directly in the plot
            # specification, or if the linestyle cycler is implemented via rcParams.
            # Find workaround?

            axes[i].set_prop_cycle(cycler('linestyle', ['solid',':','-.','--']))

            # plot primary material mixes
            plot_data[prods].plot(ax=axes[i], kind='area', stacked=True, lw=0,
                                  cmap=trunc_colormap(cmap, 0.35, 0.35+1/n, n=n),
                                  legend=False)
            y_lims = axes[i].get_ylim()
            s_cmap = plt.get_cmap(supply_cmaps[i])

            # plot supply constraints
            tmp[prods].plot(ax=axes[i], stacked=False, legend=False, lw=2.5,
                                      color='k', alpha=0.7)

            axes[i].set_title(f'{mat} production mix', fontsize=10, fontweight='bold')

            axes[i].set_ylabel(y_label)
            axes[i].set_ylim(0, y_lims[1])
            axes[i].set_xlim(2020, 2050)
            axes[i].xaxis.set_major_locator(MultipleLocator(10))

        # construct legend for each subplot
        lines = []
        labels = []
        max_len = max([len(i) for i in fleet.sets.mat_prod.values()])

        for ax, mat in zip(axes, fleet.sets.mat_prod.keys()):
            ind = []
            labels.extend([f'$\\bf{mat}$' + ' $\\bfproducers$']) # add title for each material
            lines.extend([plt.plot([],marker="", ls="")[0]]) # adds empty handle for title for centering
            axLine, axLabel = ax.get_legend_handles_labels()

            # customize labels for max primary supply constraints
            for i, (line, label) in enumerate(zip(axLine, axLabel)):
                if isinstance(line, matplotlib.lines.Line2D):
                    axLabel[i] = label + " max capacity"
                    ind.extend([i])

            # manual spacing for legend entries by adding empty plots
            if len(axLabel) < (max_len*2):
                start = max(ind) + 1
                end = int(start + (max_len - len(axLabel)/2))

                # add blank entries to match greatest number of material producers
                axLine[start:start] = [plt.plot([], marker="", ls="")[0]]* (end-start)
                axLabel[start:start] = [''] * (end-start)
                for i in range(end-start):
                    axLine.extend([plt.plot([], marker="", ls="")[0]])
                    axLabel.extend([''])

            lines.extend(axLine)
            labels.extend(axLabel)

        leg = axes[-1].legend(lines, labels, loc=9, bbox_to_anchor=(0, -2.2, 1, 1.5),
                              bbox_transform=axes[-1].transAxes,
                              mode='expand', ncol=len(axes), borderaxespad=0, handlelength=3)

        for vpack in leg._legend_handle_box.get_children():
            for hpack in vpack.get_children()[:1]:
                hpack.get_children()[0].set_width(0)  # "remove" title handle to center text

        plt.xlabel('year', fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        fig.align_ylabels()

        fig.suptitle('Demand shares for primary critical materials', y=0.995)

        export_fig(fp, ax, pp, export_pdf, export_png, png_name=fig._suptitle.get_text())

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: Primary production mixes for critical materials')
        print(e)

    #%% Manufacturing constraint
    """--- Plot total new battery capacity and manfuacturing constraint ---"""
    try:
        fig, ax = plt.subplots(1, 1, dpi=300)

        if fleet.new_capac_demand.max()[0] <= 100:
            plot_resources *= 1e3
            plot_prim_supply *= 1e3
            units = 'MWh'
            ylabel = 'MWh new batteries per year'
        else:
            units = 'GWh'
            ylabel = 'GWh new batteries per year'

        fleet.new_capac_demand.index = fleet.new_capac_demand.index.astype(int)
        fleet.parameters.manuf_cnstrnt.columns = fleet.parameters.manuf_cnstrnt.columns.astype(int)

        dem = fleet.new_capac_demand.plot(ax=ax, kind='area', lw=0,)
        constr = fleet.parameters.manuf_cnstrnt.T.plot(ax=ax,
                         lw=2.5, ls='--', color='k', alpha=0.7,)

        handles, labels = constr.get_legend_handles_labels()

        labels = [label + ' manufacturing capacity' if label in fleet.sets.newtec else label for label in labels]

        ax.set_ylabel(ylabel)
        ax.set_ybound(lower=0, upper=fleet.parameters.manuf_cnstrnt.max().max()*1.1)

        ax.xaxis.set_major_locator(MultipleLocator(10))

        if cropx:
            ax.set_xlim(right=max_year)
        else:
            ax.set_xlim(2000, 2050)

        ax.legend(handles, labels, loc=2, bbox_to_anchor=(0, -0.4, 1, 0.2),
                  ncol=2, mode='expand', borderaxespad=0, handlelength=3)

        plt.xlabel('year', fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        fig.align_ylabels()

        fig.suptitle('New battery demand', y=0.95)

        export_fig(fp, ax, pp, export_pdf, export_png, png_name=fig._suptitle.get_text())

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: New battery constraints')
        print(e)


    pp.close()
    plt.show()

#%%  Input parameter checking
def vis_input(fleet, fp, filename, param_values, export_png, export_pdf=True, max_year=50, cropx=False, suppress_vis=False):
    """Visualize model results and input.

    Parameters
    ----------
    fleet : FleetModel
        Contains experiment input and results.
    fp : str
        Filepath for saving files.
    filename : str
        Experiment identifier.
    param_values : dict of {str: int|float|dict|list}
        Dictionary of parameter values for experiment.
    export_png : bool
        Toggle for exporting figures a png.
    export_pdf : bool, optional
        Toggle for exporting figures in a pdf. The default is True.
    max_year : int, optional
        If cropping figures, new value for max x-limit. The default is 50.
    cropx : bool, optional
        Toggle for cropping figures to max_year. The default is True.
    suppress_vis : bool, optional
        Turn off interactive mode for pyplot. The default is False.
    """

    """Future work"""
    # Todo: plot newtec cohorts as share of total fleet
    # Todo: plot crossover in LC impacts between incumbent and new tecs by segment

    """--- Cover page for input parameter checking ---"""
    pp = PdfPages(fp+'input_params_vis_' + filename + '.pdf')
    plt.rcParams.update({'figure.max_open_warning': 0})  # suppress max 20 figures warning
    if suppress_vis:
        plt.ioff()
        div_page = plt.figure(figsize=(11.69, 8.27))
        txt = 'Plotting of input parameters for checking'
        div_page.text(0.5, 0.5, txt, transform=div_page.transFigure, size=30, ha="center")
        pp.savefig()

    plt_array, empty_spots, cat_type = plot_arrange(fleet)

    """ Plot total fleet"""
    try:
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(5,6))
        plot_data = {}
        plot_data['fleetreg'] = fleet.tot_stock.groupby(['fleetreg', 'year']).sum().sum(axis=1)
        plot_data['seg'] = fleet.tot_stock.groupby(['seg', 'year']).sum().sum(axis=1)
        plot_data['tec'] = fleet.tot_stock.groupby(['tec', 'year']).sum().sum(axis=1)
        cms = ['tab10', dark , tec_cm]

        for key, cmap, ax in zip(plot_data.keys(), cms, axes.flatten()):
            plot_data[key].unstack(key).plot(ax=ax, kind='area', stacked=True, cmap=cmap)
            ax.set_title(f'{key}')
            ax.set_xbound(0, 50)
            ax.xaxis.set_tick_params(rotation=45)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.suptitle('Vehicle stock, by region, segment or tec', y=0.995)

    except Exception as e:
        print('\n *****************************************')
        print('Error with input figure: fleet stock')
        print(e)

    """ Plot fleet dynamics"""
    try:
        fig, axes = plt.subplots(plt_array[0], plt_array[1], sharex=True, sharey='row')
        plot_data = pd.DataFrame(columns=['stock added', 'stock removed', 'total stock'])
        plot_data['stock added'] = fleet.stock_add.unstack(['seg','tec']).sum(axis=1)
        plot_data['stock removed'] = -fleet.stock_rem.unstack(['seg', 'tec']).sum(axis=1)
        plot_data['total stock'] = fleet.tot_stock.unstack(['seg', 'tec']).sum(axis=1)
        grouped_plot_data = plot_data.groupby('fleetreg')

        for key, ax in zip(grouped_plot_data.groups.keys(), axes.flatten()):
            df = grouped_plot_data.get_group(key)
            df.index = df.index.droplevel('fleetreg')
            plot_lines = ax.plot(df)
            ax.set_title(f'{key}')
            ax.xaxis.set_major_locator(IndexLocator(10, 0))
            ax.set_xbound(0, 30)
            ax.xaxis.set_tick_params(rotation=45)
        labels = ['Added', 'Removed', 'Total stock']
        fig.legend(plot_lines, labels, loc='upper left', bbox_to_anchor=(0.85, 0.91), )
        remove_subplots(axes, empty_spots)

    except Exception as e:
        print('\n *****************************************')
        print('Error with input figure: fleet dynamics')
        print(e)

    """ Plot fleet """
    try:
        for reg in fleet.sets.fleetreg:
            fig, axes = plt.subplots(len(fleet.sets.seg), len(fleet.sets.tec), sharex=True, sharey=True, figsize=(9.5,20))
            plt.subplots_adjust(top=0.85, hspace=0.25, wspace=0.05)
            fig.set_tight_layout(True)
            plot_data = fleet.tot_stock.stack().reset_index(['year', 'age'])
            plot_data['year'] = plot_data['year'].astype(int)
            plot_data['age'] = plot_data['age'].astype(int)
            plot_data['cohort'] = plot_data['year'] - plot_data['age']
            plot_data.set_index(['year', 'cohort'], append=True, drop=True, inplace=True)
            plot_data.drop(columns='age', inplace=True)
            plot_data = plot_data.groupby(['seg','tec'])

            for key, ax in zip(plot_data.groups.keys(), axes.flatten()):
                df = plot_data.get_group(key)
                df.index = df.index.droplevel(['tec', 'seg'])
                df = df.unstack()
                df.columns = df.columns.droplevel(0)
                df.loc[reg].plot(kind='area', stacked=True, ax=ax, legend=None)
                ax.set_title(f'{key}')
                ax.xaxis.set_major_locator(IndexLocator(10, 0))
                ax.xaxis.set_tick_params(rotation=45)
            fig.suptitle(f'Vehicle dynamics in {reg}, by technology and cohort year', y=1.02)
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, ncol=3, title='Cohort year',
                       loc='upper left', bbox_to_anchor=(1.01, 0.985))


        for reg in fleet.sets.fleetreg:
            fig, axes = plt.subplots(len(fleet.sets.seg), len(fleet.sets.tec), sharex=True, sharey=True, figsize=(9.5,20))
            plt.subplots_adjust(top=0.85, hspace=0.25, wspace=0.05)
            fig.set_tight_layout(True)

            grouped_plot_data = fleet.tot_stock.groupby(['seg', 'tec'])
            for key, ax in zip(grouped_plot_data.groups.keys(), axes.flatten()):
                df = grouped_plot_data.get_group(key)
                df.index = df.index.droplevel(['tec', 'seg'])
                df.loc[reg].plot(kind='area', stacked=True, ax=ax, legend=None)
                ax.set_title(f'{key}')
                ax.xaxis.set_major_locator(IndexLocator(10, 0))
                ax.xaxis.set_tick_params(rotation=45)
            fig.suptitle(f'Vehicle dynamics in {reg}, by technology and age', y=1.02)
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, ncol=2, title='age',
                       loc='upper left', bbox_to_anchor=(1.01, 0.985))


    except Exception as e:
        print('\n *****************************************')
        print('Error with input figures: fleet dynamics')
        print(e)

    """ Plot evolution of lifecycle impacts """
    try:
        fig, axes = plt.subplots(len(fleet.sets.fleetreg), len(fleet.sets.tec),
                                 sharey=True, sharex=True,
                                 figsize=(2*len(fleet.sets.tec)+1, 2*len(fleet.sets.fleetreg)+2))
        plt.subplots_adjust(top=0.85, hspace=0.25, wspace=0.05)

        for i, reg in enumerate(fleet.sets.fleetreg):
            for j, tec in enumerate(fleet.sets.tec):
                plot_data = fleet.LC_impacts.unstack('seg').loc(axis=0)[fleet.sets.optimp, reg, tec, '2000':'2050']
                plot_data.index = plot_data.index.droplevel(['imp','fleetreg','tec'])
                plot_data.plot(ax=axes[i,j], legend=False)
                axes[i,j].set_title(f'{tec}, {reg}', fontsize=10, fontweight='bold')
                x_labels = [label for label in plot_data.index.tolist()]
                axes[i,j].xaxis.set_major_locator(IndexLocator(10, 0))
                axes[i,j].xaxis.set_minor_locator(IndexLocator(2, 0))

                axes[i,j].xaxis.set_major_formatter(FuncFormatter(lambda x, _: dict(zip(range(len(x_labels)), x_labels)).get(x,"")))
                axes[i,j].xaxis.set_tick_params(rotation=45)
                axes[i,j].set_xbound(0, 50)

            axes[0,0].yaxis.set_major_locator(MultipleLocator(20))

        for ax in axes[-1, :]:
            ax.set_xlabel('Cohort year')

        fig.text(0, 0.5, f'Lifecycle impacts over {fleet.sets.age[-1]}-year lifetime, in t CO2-eq', rotation='vertical', ha='center', va='center')

        fig.suptitle('Evolution of lifecycle impacts by \n cohort, segment, region and technology', y=0.975)
        patches, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(patches, labels, bbox_to_anchor=(1.0, 0.8), loc='upper left', title='Segment', borderaxespad=0.)
        pp.savefig()
        plt.show()

    except Exception as e:
        print('\n *****************************************')
        print('Error with input figure: evolution of lifecycle impacts')
        print(e)


    """--- Plot production impacts by tec and seg ---"""
    prod = fleet.production_impacts.stack().unstack('tec').sum(level=['seg', 'year'])
    labels = prod.columns.to_list()
    prod_int = prod / fleet.stock_add.sum(axis=1).unstack('tec').sum(level=['seg', 'prodyear'])  # production emission intensity
    try:
        fig, axes = plt.subplots(3, 2, figsize=(9,9), sharex=True, sharey=True)
        title='Total production impacts by technology and segment'
        plot_subplots(fig, axes, prod.groupby(['seg']), title=title, labels=labels, cmap='jet_r')
        fig.text(0.04, 0.5, 'Production impacts \n(Mt CO2-eq)', ha='center', va='center', rotation='vertical')
        export_fig(fp, fig, pp, export_pdf, export_png, png_name='tot_prod_impacts')
        pp.savefig(bbox_inches='tight')

    except Exception as e:
        print('\n *****************************************')
        print('Error with input figure: production impacts by tec and seg')
        print(e)

    try:
        fig, axes = plt.subplots(3, 2, figsize=(9,9), sharex=True, sharey=True)
        title = 'Production emission intensities by technology and segment'
        plot_subplots(fig, axes, prod_int.groupby(['seg']),title=title,labels=labels, cmap='jet_r')
        fig.text(0.04, 0.5, 'Production impacts intensity \n(t CO2/unit)', ha='center', va='center', rotation='vertical')
        export_fig(fp, ax, pp, export_pdf, export_png, png_name='prod_intensity_out')
        pp.savefig(bbox_inches='tight')
    except Exception as e:
        print('\n *****************************************')
        print('Error with input figure: production impacts by tec and seg (2)')
        print(e)

    try:
        fig, axes = plt.subplots(3, 2, figsize=(9,9), sharex=True, sharey=True)
        title = 'TEC_PROD_IMPACT_INT'
        plot_subplots(fig, axes, fleet.tec_prod_impact_int.unstack('tec').groupby(['seg']), title=title, labels=labels, cmap='jet_r')
        fig.text(0.04, 0.5, 'Production impacts intensity \n(t CO2/unit)', ha='center', va='center', rotation='vertical')
        export_fig(fp, ax, pp, export_pdf, export_png, png_name='TEC_PROD_IMPACT_INT')
        pp.savefig(bbox_inches='tight')
    except Exception as e:
        print('\n *****************************************')
        print('Error with input figure: TEC_PROD_IMPACT_INT')
        print(e)

    try:
        fig, axes = plt.subplots(3, 2, figsize=(9,9), sharex=True, sharey=True)
        title = 'TEC_OPER_EINT - check ICE sigmoid function'
        plot_subplots(fig, axes, fleet.veh_oper_eint.unstack('tec').groupby(['seg']), title=title, labels=labels, cmap='jet_r')
        fig.text(0.04, 0.5, 'Operation energy intensity \n(kWh/km)', ha='center', va='center', rotation='vertical')
        export_fig(fp, ax, pp, export_pdf, export_png, png_name='TEC_OPER_EINT')
        pp.savefig(bbox_inches='tight')
    except Exception as e:
        print('\n *****************************************')
        print('Error with input figure: TEC_OPER_EINT')
        print(e)

    try:
        fig, ax = plt.subplots(1, 1)
        tmp = (fleet.enr_impact_int * 1000).unstack(['reg']).T
        tmp = sort_ind(tmp, cat_type, fleet)
        tmp = tmp.T
        tmp = tmp.unstack(['enr']).swaplevel('enr', 'reg', axis=1).sort_index(level='enr', axis=1).loc[fleet.sets.optimp[0]]
        tmp['ELC'].plot(ax=ax, title=f'ENR_IMPACT_INT for {fleet.sets.optimp[0]}')
        tmp['FOS'].plot(ax=ax, color='darkslategrey', linestyle='dashed', label='FOS (all countries)')

        plt.ylabel('Fuel chain (indirect) impacts intensity, \n g CO2-eq/kWh')
        ax.set_xbound(0, 50)

        handles, labels = ax.get_legend_handles_labels()
        labels = ['ELC, '+ label for label in labels[:-len(fleet.sets.reg)]]
        labels += ['FOS (all countries)']
        handles = handles[:-len(fleet.sets.reg)+1]
        ax.legend(flip(handles, 4), flip(labels, 4), bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=4)
        export_fig(fp, ax, pp, export_pdf, export_png, png_name='ENR_IMPACT_INT')
        pp.savefig(bbox_inches='tight')
        plt.show()

    except Exception as e:
        print('\n *****************************************')
        print('Error with input figure: ENR_IMPACT_INT')
        print(e)

    try:
        fig, axes = plt.subplots(3, 2, figsize=(9,9), sharex=True, sharey=True)
        title = 'Stock introduced in each cohort'
        tmp = fleet.stock_add.replace(0, np.nan).dropna(axis=1, how='all')
        tmp = (tmp.unstack('tec').droplevel('age', axis=1) / 1e6)
        tmp = tmp.unstack('fleetreg').T
        tmp = sort_ind(tmp, cat_type, fleet)
        tmp = tmp.T

        # tmp = tmp.stack('fleetreg').reorder_levels(['seg', 'fleetreg', 'prodyear'])
        tmp = tmp.groupby(['seg'])

        axe = axes.ravel()
        for i, ax in enumerate(axe):
            tmp.get_group(fleet.sets.seg[i]).droplevel(0).plot(ax=ax, kind='area', stacked=True,
                                                  legend=False, lw=0, cmap=tec_cm4)

            # plot_data = tmp.get_group(fleet.sets.seg[i]).sum(level=['seg','prodyear']).sum(axis=1).loc[fleet.sets.seg[i]]
            # plot_data.name='Total fleet'
            # plot_data.plot(ax=ax, legend=False, color='k',ls=':')

        # plot_subplots(fig, axes, tmp, cmap=tec_cm4, title=title)

        fig.suptitle(title)
        fig.text(0.04, 0.5, 'Total stock, by segment and technology \n(millions)', ha='center', va='center', rotation='vertical')
        patches, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(flip(patches, 5), flip(labels, 5), bbox_to_anchor=(0.5, 0), loc='lower center', ncol=5, borderaxespad=0.)
        plt.xlabel('Cohort year')
        pp.savefig(bbox_inches='tight')
        plt.show()

    except Exception as e:
        print('\n *****************************************')
        print('Error with input figure: Total stock by segment and technology')
        print(e)

    pp.close()
    plt.show()