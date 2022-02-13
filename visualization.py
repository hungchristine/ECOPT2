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
from matplotlib.ticker import (MultipleLocator, IndexLocator, IndexFormatter, PercentFormatter)
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
cmap_em = LinearSegmentedColormap.from_list('emissions', ['lightsteelblue', 'midnightblue',
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

        # if title == 'initial stock of each cohort':
        #     print(d)
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


def sort_ind(ind, cat_type, fleet):
    """Sort Index by region.

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
    if isinstance(ind, pd.MultiIndex):
        # find levels with reg or fleetreg
        for i, lvl in enumerate(ind.levels):
            if (ind.levels[i].name == 'reg') or (ind.levels[i].name == 'fleetreg'):
                lvl_num = i
                break
        df = ind.to_frame()
        df['fleetreg'] = pd.Categorical(df['fleetreg'], categories=fleet.sets.fleetreg, ordered=True)  # for simplified, 2-region test case
        # df['reg'] = pd.Categorical(df['reg'], categories=['LOW', 'II', 'MID', 'IV', 'HIGH'], ordered=True)
        ind = pd.MultiIndex.from_frame(df)
        # ind.set_levels(ind.levels[lvl_num].astype(cat_type), 'reg', inplace=True)
    else:
        ind = ind.astype(cat_type)
    return ind


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


#%%
def vis_GAMS(fleet, fp, filename, param_values, export_png=False, export_pdf=True, max_year=50, cropx=True, suppress_vis=False):
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
        fig, axes = plt.subplots(plt_array[0], plt_array[1], sharex=True, sharey='row')

        tmp = fleet.stock_add.sum(axis=1).unstack('seg').unstack('tec').loc['2020':]

        # Check if scaling of stock additions required
        level_2050 = tmp.groupby(['fleetreg', 'prodyear']).sum().sum(axis=1).unstack('prodyear')['2050'].mean()
        if level_2050 > 1e6:
            tmp /= 1e6
            units = 'millions'
        elif level_2050 > 15e3:
            tmp /= 1e3
            units = 'thousands'
        else:
            units = ''
        tmp.index = sort_ind(tmp.index, cat_type, fleet)
        tmp = tmp.groupby('fleetreg', sort=False)

        for (key, ax) in zip(tmp.groups.keys(), axes.flatten()):
            tmp.get_group(key).plot(ax=ax, kind='area', cmap=paired, lw=0, legend=False, title=f'{key}')
            ax.xaxis.set_major_locator(IndexLocator(10, 0))
            ax.xaxis.set_tick_params(rotation=45)
            ax.set_xbound(0, 50)


        fix_tuple_axis_labels(fig, axes, 'year')
        remove_subplots(axes, empty_spots)

        ref_ax = get_ref_ax(axes)
        fix_age_legend(ref_ax, pp, cropx, max_year, 'Vehicle technology and segment')
        fig.text(0, 0.5, f'Vehicles added to stock \n {units} of vehicles', rotation='vertical', ha='center', va='center')

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
        fig, axes = plt.subplots(plt_array[0], plt_array[1], sharex=True, sharey=True)
        tmp = fleet.add_share
        tmp.index = sort_ind(tmp.index, cat_type, fleet)
        tmp = tmp.groupby(['fleetreg'], sort=False)

        for (key, ax) in zip(tmp.groups.keys(), axes.flatten()):
            tmp.get_group(key).plot(ax=ax, kind='area', cmap=paired, lw=0, legend=False, title=f'{key}')
            ax.set_xbound(0, 80)
            ax.xaxis.set_major_locator(IndexLocator(10, 0))
            ax.xaxis.set_minor_locator(IndexLocator(2, 0))
            ax.xaxis.set_tick_params(rotation=45)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.set_ybound(0, 1)

        fix_tuple_axis_labels(fig, axes, 'year')
        remove_subplots(axes, empty_spots)
        # axes[1, 2].remove()  # remove 6th subplot
        ref_ax = get_ref_ax(axes)
        fix_age_legend(ref_ax, pp, cropx, max_year, 'Vehicle technology and segment')
        fig.text(0, 0.5, 'Total market share', rotation='vertical', ha='center', va='center')
        fig.suptitle('Stock additions, by technology, vehicle segment and region,\n as share of total stock additions', y=1.05)

        export_fig(fp, ax, pp, export_pdf, export_png, png_name= 'share stock_add_tec, seg, reg')

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: Market share, by seg, tec')
        print(e)

    #%% market shares by segment and technology (un-normalized)
    """--- Plot tech split of stock additions by segment ---"""
    try:
        fig, axes = plt.subplots(len(fleet.sets.seg), len(tmp.groups.keys()), sharex=True, sharey=True)
        tmp = fleet.add_share.div(fleet.add_share.sum(axis=1, level='seg'), axis=1, level='seg')
        tmp.index = sort_ind(tmp.index, cat_type, fleet)
        tmp = tmp.groupby(['fleetreg'], sort=False)

        for col, reg in enumerate(tmp.groups.keys()):
            for row, seg in enumerate(fleet.sets.seg):
                tmp.get_group(reg)[seg].plot(ax=axes[row,col], kind='area', cmap=paired_dict[seg], lw=0, legend=False)
                axes[row,col].set_xbound(0, 80)
                axes[row,col].xaxis.set_major_locator(IndexLocator(10, 0))
                axes[row,col].xaxis.set_minor_locator(IndexLocator(2, 0))
                axes[row,col].xaxis.set_tick_params(rotation=45)
                axes[row,col].set_ybound(0,1)
                axes[row,col].yaxis.set_ticklabels([])
                if not col:
                    # annotations for top row
                    axes[row,col].text(-7, 0.5, seg, va='center')
                if not row:
                    axes[row,col].text(25, 1.1, reg, ha='center')
        fix_tuple_axis_labels(fig, axes, 'year')
        remove_subplots(axes, empty_spots) # remove extraneous/empty subplots

        ref_ax = get_ref_ax(axes)
        fix_age_legend(ref_ax,  pp, cropx, max_year, 'Vehicle technology')
        fig.text(0, 0.5, 'Segment market share \n(0-100%)', rotation='vertical', ha='center', va='center')
        fig.suptitle('Stock additions, by technology, vehicle segment and region, \n as share of segment stock', y=1.05)

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: Technology market share, by seg')
        print(e)

    #%% Segment shares of BEVs by region
    """--- Plot market share of BEVs by segment and region ---"""
    try:
        fig, axes = plt.subplots(plt_array[0], plt_array[1], sharex=True, sharey=True)

        tmp = fleet.add_share.div(fleet.add_share.sum(axis=1, level='seg'), axis=1, level='seg')
        tmp = tmp.drop('ICE', axis=1, level='tec')
        tmp.index = sort_ind(tmp.index, cat_type, fleet)
        tmp = tmp.groupby(['fleetreg'])

        for (key, ax) in zip(tmp.groups.keys(), axes.flatten()):
            tmp.get_group(key).plot(ax=ax, cmap=dark, legend=False, title=f'{key}')
            ax.set_xbound(0, 80)
            ax.set_ybound(0, 1)
            ax.xaxis.set_major_locator(IndexLocator(10, 0))
            ax.xaxis.set_minor_locator(IndexLocator(2, 0))
            ax.xaxis.set_tick_params(rotation=45)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))

        fix_tuple_axis_labels(fig, axes, 'year')
        remove_subplots(axes, empty_spots)  # remove extraneous/empty subplots

        ref_ax = get_ref_ax(axes)
        fix_age_legend(ref_ax, pp, cropx, max_year, 'Vehicle segment')
        handles, labels = ref_ax.get_legend_handles_labels()
        labels = [label.strip('()').split(',')[0] for label in labels]
        ref_ax.legend(handles, labels, loc=2, bbox_to_anchor=(1.01, 1))
        fig.suptitle('Market share of BEVs by segment and region', y=0.995)

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: BEV market share, by seg, reg')
        print(e)

    #%% Regional shares of BEVs by segment
    """--- Plot market share of BEVs by segment and region ---"""
    try:
        fig, axes = plt.subplots(3, 2, figsize=(9,9), sharex=True, sharey='row')
        plt.subplots_adjust(top=0.95, hspace=0.25, wspace=0.05)

        tmp = fleet.stock_add.div(fleet.stock_add.sum(level=['seg', 'tec', 'prodyear']))
        tmp.dropna(axis=1, how='all', inplace=True)
        tmp = tmp.drop('ICE', axis=0, level='tec')
        tmp.index = sort_ind(tmp.index, cat_type, fleet)
        tmp = tmp.unstack('fleetreg').droplevel('age', axis=1).droplevel('tec', axis=0).dropna(how='all', axis=1).groupby(['seg'])

        # TODO: fix region order
        i = 0
        for (key, ax) in zip(tmp.groups.keys(), axes.flatten()):
            cmap = (dark(i/dark.N))  # use BEV segment colours
            plot_data = tmp.get_group(key).droplevel('seg')
            pl = ax.stackplot(plot_data.index.map(float).values,
                               plot_data.T.values,
                               colors=[cmap]*len(fleet.sets.fleetreg),
                               ec='w',
                               lw=1)
            for country_area, patt in zip(pl, hatch[:len(fleet.sets.fleetreg)]):
                country_area.set_hatch(2*patt)  # set hatching for each region

            ax.set_xbound(2020, 2050)
            ax.xaxis.set_major_locator(IndexLocator(10, 0))
            ax.xaxis.set_minor_locator(IndexLocator(2, 0))
            ax.xaxis.set_tick_params(rotation=45)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.set_ybound(0, 1)
            ax.set_title(key)
            i += 1

        ref_ax = get_ref_ax(axes)
        legend_elem = []
        for i in range(len(fleet.sets.fleetreg)):
            legend_elem.append(Patch(facecolor=dark(0), hatch=2*hatch[i], ec='w',label=fleet.sets.fleetreg[i]))
        leg = axes[0,-1].legend(handles=legend_elem, title='Region',
                                title_fontsize='medium', borderpad=1,
                                loc=2, labelspacing=1.5, handlelength=4, bbox_to_anchor=(1.01, 1))

        for patch in leg.get_patches():
            patch.set_height(20)
            patch.set_y(-6)
        fig.suptitle('Regional share of BEVs by segment ', y=0.995)

        export_fig(fp, ax, pp, export_pdf, export_png, png_name=fig._suptitle.get_text())

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: BEV market share, by seg, reg 2')
        print(e)

    #%% Total emissions by tec and lifecycle phase, against total stock
    """--- Plot total emissions by tec and lifecycle phase---"""
    try:
        fleet.emissions.sort_index(axis=1, level=0, ascending=False, inplace=True)

        fig = plt.figure(figsize=(14,9))

        gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1,3], hspace=0.05)
        ax2 = fig.add_subplot(gs[0])  # vehicle plot
        ax1 = fig.add_subplot(gs[1], sharex=ax2)  # emissions plot

        plot_ax2 = (fleet.stock_df_plot.sum(axis=1).unstack('seg').sum(axis=1).unstack('tec').sum(level='year')/1e6)
        plot_ax2.plot(ax=ax2, kind='area', cmap=tec_cm, lw=0)
        (fleet.emissions/1e6).plot(ax=ax1, kind='area', lw=0, cmap=cmap_em)
        (fleet.bau_emissions/1e6).plot(ax=ax1, kind='line')

        ax1.set_ylabel('Lifecycle climate emissions \n Mt $CO_2$-eq', fontsize=13)
        ax2.set_ylabel('Vehicles, millions', fontsize=13, labelpad=25)

        if cropx:
            ax1.set_xlim(right=max_year)
            ax2.set_xlim(right=max_year)
        handles, labels = ax1.get_legend_handles_labels()
        labels = [x+', '+y for x, y in itertools.product(['Production', 'Operation', 'End-of-life'], ['ICEV', 'BEV'])]
        labels.insert(0, 'Emissions, 0% electrification')
        ax1.legend(handles, labels, loc=0, fontsize=14)
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(handles, ['BEV', 'ICEV'], loc=0, fontsize=14, framealpha=1)

        ax2.set_xbound(0, 50)
        ax1.set_ylim(bottom=0)

        plt.setp(ax2.get_yticklabels(), fontsize=14)  # modify x-tick label size

        fix_tuple_axis_labels(fig, ax1, 'year', 0, isAxesSubplot=True)
        plt.xlabel('year', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        export_fig(fp, ax, pp, export_pdf, export_png, png_name='LC_emissions_vs_stock')

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: Total vehicle stocks vs lifecycle emissions by tec')
        print(e)

    #%% Regionalized fleet emissions
    try:
        fig, ax = plt.subplots(1,1, figsize=(14,9), dpi=300)

        plot_emiss = fleet.veh_totc.sum(level=['fleetreg', 'tec'])
        plot_emiss.index = plot_emiss.index.swaplevel('fleetreg', 'tec')
        plot_emiss.index = sort_ind(plot_emiss.index, cat_type, fleet).sortlevel(0, sort_remaining=True)[0]

        ax.set_prop_cycle(paired_cycler)

        (plot_emiss/1e6).T.plot(ax=ax, kind='area', lw=0) #'Dark2')
        ax.set_ylabel('Lifecycle climate emissions \n Mt $CO_2$-eq', fontsize=13)

        if cropx:
            ax.set_xlim(right=max_year)

        ax.set_xbound(0, 50)
        ax.set_ybound(0)

        fix_tuple_axis_labels(fig, ax, 'year', 0, isAxesSubplot=True)
        plt.xlabel('year', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        export_fig(fp, ax, pp, export_pdf, export_png, png_name='LC_emissions')

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: Lifecycle emissions by tec and region')
        print(e)

    #%% Operation emissions by tec, benchmarked against regulation emissions targets
    """--- Plot operation emissions by tec ---"""
    try:
        ax = (fleet.emissions.loc[:, 'Operation'] / 1e6).plot(kind='area', cmap=LinearSegmentedColormap.from_list('temp', colors=['silver', 'grey']), lw=0)

        # plot regulation levels
        plt.hlines(442, xmin=0.16, xmax=0.6, linestyle='dotted', color='darkslategrey', label='EU 2030 target, \n 20% reduction from 2008 emissions', transform=ax.get_yaxis_transform())
        plt.hlines(185, xmin=0.6, xmax=1, linestyle='-.', color='darkslategrey', label='EU 2050 target, \n 60% reduction from 1990 emissions', transform=ax.get_yaxis_transform())
        plt.ylabel('Fleet operation emissions \n Mt $CO_2$-eq')

        if cropx:
            plt.xlim(right=max_year)

        ax.set_xbound(0, 50)
        ax.set_ybound(0)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, ['ICEV',
                            'BEV',
                            'EU 2030 target, \n20% reduction from 2008 emissions',
                            'EU 2050 target, \n60% reduction from 1990 emissions'
                            ], loc=2, bbox_to_anchor= (1.05, 1.02))
        export_fig(fp, ax, pp, export_pdf, export_png, png_name='operation_emissions')

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: Operation emissions by tec')
        print(e)

    #%% Total stocks by segment
    """--- Plot total stocks by segment ---"""
    # TODO: fix legend order of segments? (one column?)
    try:
        ax = fleet.stock_df_plot.sum(axis=1).unstack('seg').sum(axis=0, level=['year']).plot(kind='area', cmap='jet', lw=0, title='Total stocks by segment')
        ax.set_xbound(0, 80)
        ax.set_ybound(lower=0)
        fix_age_legend(ax, pp, cropx, max_year, 'Vehicle segments')
        export_fig(fp, ax, pp, export_pdf, export_png, png_name=ax.get_title())

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: Total stocks by seg')
        print(e)

    #%% Total stocks by region
    """--- Plot total stocks by region ---"""
    try:
        tmp = fleet.stock_df_plot.sum(axis=1).unstack('fleetreg').sum(axis=0, level=['year'])
        tmp.columns = sort_ind(tmp.columns, cat_type, fleet).sort_values()
        ax.set_xbound(0, 80)
        ax.set_ybound(lower=0)
        fix_age_legend(ax, pp, cropx, max_year, 'Region')
        export_fig(fp, ax, pp, export_pdf, export_png, png_name=ax.get_title())

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: Total stocks by reg')
        print(e)

    #%% Total stocks by segment and technology
    """--- Plot total stocks by age, segment and technology ---"""
    try:
        fig, ax = plt.subplots(1,1, dpi=300)
        plot_data = fleet.stock_df_plot.sum(axis=1).unstack('seg').unstack('tec').sum(axis=0, level='year')
        plot_data.plot(kind='area', ax=ax, cmap=paired, lw=0,
                            title='Total stocks by segment and technology')
        ax.set_xbound(0, 80)
        ax.set_ybound(lower=0)
        fix_age_legend(ax, pp, cropx, max_year, 'Vehicle segment and technology')
        export_fig(fp, ax, pp, export_pdf, export_png, png_name=ax.get_title())

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: Total stocks by age, seg, tec')
        print(e)

    #%% Total stocks by age, segment and technology
    """--- Plot total stocks by age, segment and technology ---"""

    #        ax = fleet.stock_df_plot.sum(axis=1).unstack('seg').unstack('tec').unstack('reg').plot(kind='area',cmap=paired,title='Total stocks by segment, technology and region')
    try:
        fig, ax = plt.subplots(1,1, dpi=300)

        stock_tec_seg_reg = fleet.stock_df_plot.sum(axis=1).unstack('seg').unstack('tec').unstack('fleetreg')
        stock_tec_seg_reg.columns = sort_ind(stock_tec_seg_reg.columns, cat_type, fleet).sortlevel(0, sort_remaining=True)[0]  # sortlevel returns one-element tuple
        stock_tec_seg_reg.columns = stock_tec_seg_reg.columns.swaplevel('fleetreg', 'tec')

        stock_tec_seg_reg.plot(kind='area', ax=ax, cmap='jet', lw=0,
                               title='Total stocks by segment, technology and region')
        ax.set_xbound(0, 80)
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
        plot_stock_add = plot_stock_add.unstack(['tec', 'fleetreg']).droplevel(axis=1, level=0)
        plot_stock_add.columns = sort_ind(plot_stock_add.columns, cat_type, fleet).sortlevel(0, sort_remaining= True)[0]
        plot_stock_add.plot(ax=ax, kind='area', cmap=tec_cm4, lw=0, legend=True, title='Stock additions by technology and region')
        ax.set_xbound(0, 50)
        ax.set_ybound(lower=0)
        plt.xlabel('year')
        plt.ylabel('Vehicles added to stock')
        pp.savefig()

    except Exception as e:
        print('\n *****************************************')
        print('Error with figure: addition to stocks, by tec and reg')
        print(e)

    #%% Total stocks by segment, technology and region
    """--- Plot total stocks by segment, technology and region ---"""
    try:
        fig, ax = plt.subplots(1, 1, dpi=300)

        plot_stock = fleet.veh_stck.sum(axis=1).unstack(['tec', 'fleetreg']).sum(level=['year'])

        plot_stock = plot_stock.stack('tec')  # remove MultiIndex to set Categorical type for regions
        plot_stock.columns = sort_ind(plot_stock.columns, cat_type, fleet)
        plot_stock = plot_stock.unstack('tec').swaplevel('tec', 'fleetreg', axis=1).sort_index(axis=1, level='tec')
        plot_stock.plot(ax=ax, kind='area', cmap=tec_cm4, lw=0, legend=True, title='Total stock by technology and region')
        ax.set_xbound(0, 80)
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

    #%% Total resource use - against total vehicle fleet
    """--- Plot total resource use ---"""
    try:
        for resource in fleet.resources.columns.get_level_values(1).unique():
            fig = plt.figure(figsize=(14,9), dpi=300)

            gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1,3], hspace=0.15)
            ax2 = fig.add_subplot(gs[0])
            ax1 = fig.add_subplot(gs[1], sharex=ax2)

            if fleet.resources.max().mean() >= 1e6:
                fleet.resources /= 1e6
                units = 'Mt'
            elif fleet.resources.max().mean() >= 1e4:
                fleet.resources /= 1e3
                units = 't'
            else:
                units = 'kg'

            # plot_df = pd.concat([fleet.resources['primary', resource], fleet.resources['recycled', resource]], axis=1)
            if (fleet.resources.values < 0).any():
                log.warning('----- Some resource demand is negative! Setting to 0. \n')
                print(fleet.resources.loc[fleet.resources.values < 0])  # print negative values for troubleshooting; may be very small or boundary condition
                print('\n')
                fleet.resources[fleet.resources < 0] = 0  # set value to 0 for plotting

            # plot use of materials
            plot_resources = fleet.resources.loc[:, (slice(None), resource)]
            plot_resources.loc['2020'] = np.nan # add 2020 to properly align plots
            plot_resources.sort_index(ascending=True, inplace=True)
            plot_resources.plot(ax=ax1, kind='area', lw=0, cmap='jet')

            plot_virg_mat = fleet.parameters.virg_mat_supply.filter(like=resource, axis=1)
            plot_virg_mat = plot_virg_mat.loc['2020':].sum(axis=1)
            plot_virg_mat.plot(ax=ax1, lw=4, kind='line', alpha=0.7)

            # plot fleet evolution
            fleet_evol = (fleet.stock_df_plot.sum(axis=1).unstack('seg').sum(axis=1).unstack('tec').sum(level='year')/1e6).loc['2020':]
            fleet_evol.plot(ax=ax2, kind='area', cmap=tec_cm, lw=0)

            ax1.set_ylabel(f'{resource} used in new batteries \n {units} {resource}', fontsize=14)
            ax2.set_ylabel('Vehicles, millions', fontsize=14, labelpad=25)
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
            ax2.legend(handles, ['BEV', 'ICEV'], loc=4, fontsize=14, framealpha=1)

            ax1.set_xbound(0, 30)
            ax2.set_xbound(0, 30)
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
        fig, axes = plt.subplots(len(fleet.sets.mat_cat), 1, sharex=True, dpi=300)
        plt.subplots_adjust(top=0.85, hspace=0.4)

        gpby_class = {supp: mat for mat, li in fleet.sets.mat_prod.items() for supp in li}

        resource_use = fleet.resources * 1000 # convert to kg
        resource_use.loc['2020'] = np.nan
        resource_use.sort_index(ascending=True, inplace=True)

        # get total available primary supplies of each material category
        prim_supply = (fleet.parameters.virg_mat_supply.groupby(gpby_class, axis=1).sum() * fleet.parameters.raw_data.eur_batt_share)*1000 # priamry supply is in t
        prim_supply = prim_supply.loc['2020':]

        for i, mat in enumerate(fleet.sets.mat_prod.keys()):
            plot_resources = resource_use.loc[:, (slice(None), mat)]
            plot_prim_supply = prim_supply[mat]
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
            axes[i].set_xlim(0, 30)
            # if cropx:
            #     axes[i].set_xlim(right=max_year)

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
        fig, axes = plt.subplots(len(fleet.sets.mat_cat), 1, sharex=True, dpi=300)#, figsize=(8,8))
        plt.subplots_adjust(top=0.85, hspace=0.4)
        plot_data = fleet.mat_mix.copy()
        level_2020 = plot_data.loc['2020'].mean() # use as baseline value to scale y-axis
        plot_data = plot_data.loc['2020':'2050']  # restrict to optimization period
        supply_constr = fleet.parameters.virg_mat_supply.loc['2020':'2050'] * 1000 # virg_mat_supply in t, not kg
        plot_data.sort_index(axis=1, inplace=True)  #make sure material mixes and supply constraints are in the same order
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
                    tmp[prod] = plot_data[prods].iloc(axis=1)[j-1] + supply_constr[prods].iloc(axis=1)[j]
                elif j == 0:
                    tmp[prod] = supply_constr[prod]

            cmap = plt.get_cmap(mat_cmaps[i])
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
            axes[i].set_xlim(0, 30)
            axes[i].xaxis.set_major_locator(MultipleLocator(10))

        # construct legend for each subplot
        lines = []
        labels = []
        for ax, mat in zip(axes, fleet.sets.mat_prod.keys()):
            labels.extend([f'$\\bf{mat}$' + ' $\\bfproducers$']) # add title for each material
            lines.extend([plt.plot([],marker="", ls="")[0]]) # adds empty handle for title for centering
            axLine, axLabel = ax.get_legend_handles_labels()
            # customize labels for max primary supply constraints
            for i, (line, label) in enumerate(zip(axLine, axLabel)):
                if isinstance(line, matplotlib.lines.Line2D):
                    axLabel[i] = label + " max capacity"
            lines.extend(axLine)
            labels.extend(axLabel)

        w = .42 * len(fleet.sets.mat_cat)
        h = .15 * len(labels)
        leg = axes[-1].legend(lines, labels, loc=9, bbox_to_anchor=((1-w)/2, -0.4-h, w, h),
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

        if fleet.batt_demand.max()[0] <= 1:
            plot_resources *= 1e3
            plot_prim_supply *= 1e3
            units = 'MWh'
            ylabel = 'MWh new batteries per year'
        else:
            units = 'GWh'
            ylabel = 'GWh new batteries per year'

        dem = ax.stackplot(fleet.batt_demand.index, fleet.batt_demand[0],
                           lw=0, labels=['New battery demand'])
        constr = ax.plot(fleet.parameters.manuf_cnstrnt.index,
                         fleet.parameters.manuf_cnstrnt,
                         lw=2.5, ls='--', color='k', alpha=0.7,
                         label='Manufacturing Capacity')

        ax.set_ylabel(ylabel)
        ax.set_ybound(lower=0, upper=fleet.parameters.manuf_cnstrnt.max()[0]*1.1)
        ax.set_xlim(0, 50)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        # if cropx:
        #     axes[i].set_xlim(right=max_year)

        ax.legend(loc=2, bbox_to_anchor=(0, -0.4, 1, 0.2),
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
    # Todo: plot BEV cohorts as share of total fleet
    # Todo:  plot crossover in LC emissions between ICEV and BEV by segment

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
        plot_data['fleetreg'] = fleet.veh_stck.groupby(['fleetreg', 'year']).sum().sum(axis=1)
        plot_data['seg'] = fleet.veh_stck.groupby(['seg', 'year']).sum().sum(axis=1)
        plot_data['tec'] = fleet.veh_stck.groupby(['tec', 'year']).sum().sum(axis=1)
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
        plot_data = pd.DataFrame(columns=['vehicles added', 'vehicles removed', 'total vehicles'])
        plot_data['vehicles added'] = fleet.stock_add.unstack(['seg','tec']).sum(axis=1)
        plot_data['vehicles removed'] = -fleet.stock_rem.unstack(['seg', 'tec']).sum(axis=1)
        plot_data['total vehicles'] = fleet.veh_stck.unstack(['seg', 'tec']).sum(axis=1)
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
        fig.suptitle('Vehicle dynamics, technology and region', y=0.9)
        remove_subplots(axes, empty_spots)

    except Exception as e:
        print('\n *****************************************')
        print('Error with input figure: fleet dynamics')
        print(e)

    """ Plot fleet """
    try:
        fig, axes = plt.subplots(len(fleet.sets.seg), len(fleet.sets.fleetreg), sharex=True, sharey=True, figsize=(8,20))
        fig.suptitle('Vehicle dynamics, technology and region', y=0.85)
        plt.subplots_adjust(top=0.85, hspace=0.25, wspace=0.05)
        grouped_plot_data = fleet.veh_stck.groupby(['seg', 'fleetreg'])
        for key, ax in zip(grouped_plot_data.groups.keys(), axes.flatten()):
            df = grouped_plot_data.get_group(key)
            df.index = df.index.droplevel(['fleetreg', 'seg'])
            df.plot(kind='area', stacked=True, ax=ax, legend=None)
            ax.set_title(f'{key}')
            ax.xaxis.set_major_locator(IndexLocator(10, 0))
            ax.xaxis.set_tick_params(rotation=45)

    except Exception as e:
        print('\n *****************************************')
        print('Error with input figure: fleet dynamics')
        print(e)

    """ Plot evolution of lifecycle emissions """
    try:
        fig, axes = plt.subplots(len(fleet.sets.fleetreg), len(fleet.sets.tec),
                                 sharey=True, sharex=True,
                                 figsize=(2*len(fleet.sets.tec)+1, 2*len(fleet.sets.fleetreg)+2))
        plt.subplots_adjust(top=0.85, hspace=0.25, wspace=0.05)

        for i, reg in enumerate(fleet.sets.fleetreg):
            for j, tec in enumerate(fleet.sets.tec):
                plot_data = fleet.LC_emissions.unstack('seg').loc(axis=0)[tec, '2000':'2050', reg]
                plot_data.plot(ax=axes[i,j], legend=False)
                axes[i,j].set_title(f'{tec}, {reg}', fontsize=10, fontweight='bold')
                x_labels = [label[1] for label in plot_data.index.tolist()]
                axes[i,j].xaxis.set_major_locator(IndexLocator(10, 0))
                axes[i,j].xaxis.set_minor_locator(IndexLocator(2, 0))
                axes[i,j].xaxis.set_major_formatter(IndexFormatter(x_labels))

                axes[i,j].xaxis.set_tick_params(rotation=45)
                axes[i,j].set_xbound(0, 50)

            axes[0,0].yaxis.set_major_locator(MultipleLocator(20))

        for ax in axes[-1, :]:
            ax.set_xlabel('Cohort year')

        fig.text(0, 0.5, 'Lifecycle emissions, in t CO2-eq', rotation='vertical', ha='center', va='center')

        fig.suptitle('Evolution of lifecycle emissions by \n cohort, segment, region and technology', y=0.975)
        patches, labels = axes[0, 0].get_legend_handles_labels()
        labels = [lab[4] for lab in labels]
        fig.legend(patches, labels, bbox_to_anchor=(1.0, 0.8), loc='upper left', title='Vehicle segment', borderaxespad=0.)
        pp.savefig()
        plt.show()

    except Exception as e:
        print('\n *****************************************')
        print('Error with input figure: evolution of lifecycle emissions')
        print(e)


    """--- Plot production emissions by tec and seg ---"""
    prod = fleet.veh_prod_totc.stack().unstack('tec').sum(level=['seg', 'year'])#/1e9
    prod_int = prod / fleet.stock_add.sum(axis=1).unstack('tec').sum(level=['seg', 'prodyear'])  # production emission intensity
    try:
        fig, axes = plt.subplots(3, 2, figsize=(9,9), sharex=True, sharey=True)
        labels=['BEV', 'ICEV']
        title='Total production emissions by technology and segment'
        plot_subplots(fig, axes, prod.groupby(['seg']), title=title, labels=labels)
        fig.text(0.04, 0.5, 'Production emissions \n(Mt CO2-eq)', ha='center', va='center', rotation='vertical')
        export_fig(fp, fig, pp, export_pdf, export_png, png_name='tot_prod_emissions')
        pp.savefig(bbox_inches='tight')

    except Exception as e:
        print('\n *****************************************')
        print('Error with input figure: production emissions by tec and seg')
        print(e)

    try:
        fig, axes = plt.subplots(3, 2, figsize=(9,9), sharex=True, sharey=True)
        title = 'Production emission intensities by technology and segment'
        plot_subplots(fig, axes, prod_int.groupby(['seg']),title=title,labels=labels)
        fig.text(0.04, 0.5, 'Production emissions intensity \n(t CO2/vehicle)', ha='center', va='center', rotation='vertical')
        export_fig(fp, ax, pp, export_pdf, export_png, png_name='prod_intensity_out')
        pp.savefig(bbox_inches='tight')
    except Exception as e:
        print('\n *****************************************')
        print('Error with input figure: production emissions by tec and seg (2)')
        print(e)

    try:
        fig, axes = plt.subplots(3, 2, figsize=(9,9), sharex=True, sharey=True)
        title = 'VEH_PROD_CINT'
        plot_subplots(fig, axes, fleet.veh_prod_cint.unstack('tec').groupby(['seg']), title=title, labels=labels)
        fig.text(0.04, 0.5, 'Production emissions intensity \n(t CO2/vehicle)', ha='center', va='center', rotation='vertical')
        export_fig(fp, ax, pp, export_pdf, export_png, png_name='VEH_PROD_CINT')
        pp.savefig(bbox_inches='tight')
    except Exception as e:
        print('\n *****************************************')
        print('Error with input figure: VEH_PROD_CINT')
        print(e)

    try:
        fig, axes = plt.subplots(3, 2, figsize=(9,9), sharex=True, sharey=True)
        title = 'VEH_OPER_EINT - check ICE sigmoid function'
        plot_subplots(fig, axes, fleet.veh_oper_eint.unstack('tec').groupby(['seg']), title=title, labels=labels)
        fig.text(0.04, 0.5, 'Operation energy intensity \n(kWh/km)', ha='center', va='center', rotation='vertical')
        export_fig(fp, ax, pp, export_pdf, export_png, png_name='VEH_OPER_EINT')
        pp.savefig(bbox_inches='tight')
    except Exception as e:
        print('\n *****************************************')
        print('Error with input figure: VEH_OPER_EINT')
        print(e)

    try:
        fig, ax = plt.subplots(1, 1)
        tmp = (fleet.enr_cint * 1000).unstack(['reg'])
        tmp.columns = sort_ind(tmp.columns, cat_type, fleet)
        tmp = tmp.unstack(['enr']).swaplevel('enr', 'reg', axis=1).sort_index(level='enr', axis=1)
        tmp['ELC'].plot(ax=ax, title='ENR_CINT')
        tmp[('FOS', 'LOW')].plot(ax=ax, color='darkslategrey', linestyle='dashed', label='FOS (all countries)')

        plt.ylabel('Fuel chain (indirect) emissions intensity, \n g CO2-eq/kWh')
        ax.set_xbound(0, 50)

        handles, labels = ax.get_legend_handles_labels()
        labels[:-1] = ['ELC, '+ label for label in labels[:-1]]
        ax.legend(flip(handles, 4), flip(labels, 4), bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=4)
        export_fig(fp, ax, pp, export_pdf, export_png, png_name='ENR_CINT')
        pp.savefig(bbox_inches='tight')
        plt.show()

    except Exception as e:
        print('\n *****************************************')
        print('Error with input figure: ENR_CINT')
        print(e)

    try:
        fig, axes = plt.subplots(3, 2, figsize=(9,9), sharex=True, sharey=True)
        title = 'initial stock of each cohort'
        tmp = fleet.stock_add.replace(0, np.nan).dropna(axis=1, how='all')
        tmp = (tmp.unstack('tec').droplevel('age', axis=1) / 1e6)
        tmp = tmp.unstack('fleetreg')
        tmp.columns = sort_ind(tmp.columns, cat_type, fleet)

        tmp = tmp.stack('fleetreg').reorder_levels(['seg', 'fleetreg', 'prodyear'])
        tmp = tmp.groupby(['seg'])
        plot_subplots(fig, axes, tmp, cmap=tec_cm4, title=title)
        fig.text(0.04, 0.5, 'Total vehicles, by segment and technology \n(millions)', ha='center', va='center', rotation='vertical')
        patches, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(flip(patches, 5), flip(labels, 5), bbox_to_anchor=(0.5, 0), loc='lower center', ncol=5, borderaxespad=0.)
        pp.savefig(bbox_inches='tight')
        plt.show()

    except Exception as e:
        print('\n *****************************************')
        print('Error with input figure: Total vehicles by segment and technology')
        print(e)

    try:
        fig, axes = plt.subplots(3, 2, figsize=(9,9), sharex=True, sharey=True)
        fig.text(0.04, 0.5, 'Vehicle operating emissions intensity, by region and segment \n (kg CO2-eq/km)', ha='center', va='center', rotation='vertical')
        title = 'VEH_OPER_CINT for BEVs, by region and segment'
        veh_op_cint_plot = fleet.veh_oper_cint.droplevel(['prodyear', 'enr']).drop_duplicates().unstack(['fleetreg']).loc(axis=0)['BEV']
        veh_op_cint_plot = (veh_op_cint_plot.swaplevel(-2, -1, axis=0) * 1000).unstack('age')

        k_r_cmap = ListedColormap(['k' for i in np.arange(0, (len(veh_op_cint_plot.columns) / 2))] +
                                  ['r' for i in np.arange(0, (len(veh_op_cint_plot.columns) / 2))])

        plot_subplots(fig, axes, veh_op_cint_plot.groupby(['seg']), title=title, cmap=k_r_cmap)
        axes[0, 0].set_xbound(0, 80)

        red_patch = matplotlib.patches.Patch(color='r', label='Low values by region')
        blk_patch = matplotlib.patches.Patch(color='k', label='High values by region')
        fig.legend(handles=[blk_patch, red_patch], bbox_to_anchor=(0.5, 0), loc='lower center',
                   ncol=2, fontsize='large', borderaxespad=0.)

        pp.savefig(bbox_inches='tight')

    except Exception as e:
        print('\n *****************************************')
        print('Error with input figure: BEV BEH_OPER_CINT by region and segment')
        print(e)

        """ Check each vintage cohort's LC emissions (actually just production and lifetime operation emissions) for various assumptions of lifetime (in years)"""
    #        fig, axes = plt.subplots(3,2,figsize=(9,9),sharey=True)
    #        title = 'Average Lifetime emissions per vehicle (11 years)'
    #        ax = plot_subplots(fleet.LC_emissions_avg.unstack('tec').groupby(['seg']),title=title,labels=labels)
    #        ax.set_xlabel('Vintage cohort')
    #        fig.text(0.04, 0.5, 'Lifetime emissions intensity (without EOL) \n(t/average vehicle)', ha='center', rotation='vertical')
    #        pp.savefig(bbox_inches='tight')

        """for i in range(0,len(fleet.LC_emissions_avg)):
            fig, axes = plt.subplots(3,2,figsize=(9,9),sharey=True)
            if i==27:
                title = f'Full lifetime emissions per vehicle ({i} year lifetime)'
            else:
                title = f'Average lifetime emissions per vehicle ({i} year lifetime)'
            ax = plot_subplots(fleet.LC_emissions_avg[i].unstack('tec').groupby(['seg']),title=title,labels=labels,xlabel='Vintage cohort')
            fig.text(0.04, 0.5,f'Lifetime emissions intensity (without EOL) \n(t/{i}-year-old vehicle)', ha='center', rotation='vertical')
            pp.savefig(bbox_inches='tight') """

        """------- Calculate lifecycle emissions (actually production + operation) by cohort for QA  ------- """
        """ See figure_calculations for calculation of these dataframes """
    #        fig, axes = plt.subplots(3,2,figsize=(9,9),sharey=True)
    #        title = 'Operating emissions, back calculated from fleet emissions'
    #        plot_subplots(op_intensity.unstack('tec').groupby(['seg']),title=title,labels=labels)
    #        fig.text(0.04, 0.5, 'Operation emissions intensity  \n(t/vehicle)', ha='center', rotation='vertical')
    #        pp.savefig(bbox_inches='tight')

        """ See figure_calculations for calculation of these dataframes """
    #        fig, axes = plt.subplots(3,2,figsize=(9,9),sharey=True)
    #        title = 'Lifetime operation emissions by cohort for whole fleet'
    #        plot_subplots(fleet.LC_intensity.unstack('tec').groupby(['seg']),title=title,labels=labels)
    #        fig.text(0.04, 0.5, 'Operation emissions \n(t)', ha='center', rotation='vertical')
    #        pp.savefig(bbox_inches='tight')
    #
        """ Need to fix! """
    #        fig, axes = plt.subplots(3,2,figsize=(9,9),sharey=True)
    #        title = 'VEH_OPER_CINT'
    ##        plot_subplots((fleet.veh_oper_cint.unstack(['tec', 'enr'])*1e6).groupby(['seg']),title=title,labels=labels)
    #        plot_subplots((fleet.veh_oper_cint*1e6).groupby(['seg']),title=title,labels=labels)
    #        fig.text(0.04, 0.5, 'Operation emissions intensity \n(g CO2/vkm)', ha='center', rotation='vertical')
    #        pp.savefig(bbox_inches='tight')

        """kept commented"""
        #for (key,ax) in zip(fleet.veh_prod_totc.unstack('tec').groupby(['seg']).groups.keys(),axes.flatten()):
    #        for (key,ax) in zip(fleet.veh_prod_totc.unstack('tec').groupby(['seg']).groups.keys(),axes.flatten()):
    #            fleet.veh_prod_totc.unstack('tec').groupby(['seg']).get_group(key).plot(ax=ax,cmap='jet',legend=False)
    #            ax.set_xticklabels([2000,2010,2020,2030,2040,2050])
    #            ax.set_xlabel('year')
    #            ax.set_title(key,fontsize=10,fontweight='bold')
    #            ax.set_label('segment')
    #
    #            ax.xaxis.set_minor_locator(MultipleLocator(1))
    #            ax.grid(which='minor',axis='x',c='lightgrey',alpha=0.55,linestyle=':',lw=0.3)
    #            ax.grid(which='major',axis='x',c='darkgrey',alpha=0.75,linestyle='--',lw=1)
    #
    #
    #        ax.legend(labels=['BEV', 'ICE'],bbox_to_anchor=(0.2,-0.3),ncol=2,fontsize='large')
    #        plt.subplots_adjust(hspace=0.45)
    #        pp.savefig(bbox_inches='tight')
    #


        #stock_df_grouped =stock_df.groupby(level=[0])
    #        for name, group in stock_df_grouped:
    #            ax=group.plot(kind='area',cmap='Spectral_r',title=name+' stock by age')
    #            fix_age_legend(ax)
    #            pp.savefig()

        #stock_df.columns = stock_df.columns.astype(int)
        #stock_df.sort_index(axis=1,inplace=True)
        #tot_stock_df=stock_df.sum(axis=0,level=1)
        #ax = tot_stock_df.plot.area(cmap='Spectral_r',title='Total stocks by vehicle age',figsize = (10,6))
        #fix_age_legend(ax)
        #plt.savefig('total_stocks_by_age.png',pad_inches=2)
        #pp.savefig()

        # Plot total stocks by technology
    #        stock_df.sum(axis=1).unstack().T.plot(kind='area', title='Total stocks by technology')
    #        stock_df.sum(axis=1).unstack().T.plot(title='Total stocks by technology')
    #        plt.savefig('total_stocks_by_tec.png',dpi=600)
    #        pp.savefig()

        # Plot stock additions and removals by technology
    #        temp_vdict_a = reorder_age_headers(v_dict['VEH_STCK_REM']).stack()
    #        temp_vdict_b = reorder_age_headers(v_dict['VEH_STCK_ADD']).stack()
    #        add_rem_df = pd.concat([temp_vdict_a, temp_vdict_b],axis=1,keys=('VEH_STCK_REM', 'VEH_STCK_ADD'))
    #
    #        add_rem_df_2=add_rem_df.stack().unstack(level=[0,3])
    #
    #        for column,variable in add_rem_df_2:
    #            ax = add_rem_df_2[column][variable].unstack().plot(kind='area',cmap='Spectral_r',title=column+" "+variable)
    #            fix_age_legend(ax)
    #
    #        add_rem_df_2.plot(subplots=True,title='Stock removal and addition variables')
    #        pp.savefig()
    #
    #        # Plot carbon emissions by technology and lifecycle phase
    #        totc_df=pd.concat((v_dict['VEH_PROD_TOTC'],v_dict['VEH_OPER_TOTC'],v_dict['VEH_EOLT_TOTC'],v_dict['VEH_TOTC']),axis=0,keys=('VEH_PROD_TOTC', 'VEH_OPER_TOTC', 'VEH_EOLT_TOTC', 'VEH_TOTC'))
    #        totc_df=totc_df.T.swaplevel(0,1,axis=1)
    #        ax = totc_df.plot(figsize = (10,6))
    #        fix_age_legend(ax)
    #        plt.savefig('CO2.png',pad_inches=2, dpi=600)
    #        pp.savefig()

    pp.close()
    plt.show()
