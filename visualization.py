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
from matplotlib.ticker import (MultipleLocator, IndexLocator, IndexFormatter)
import numpy as np

# import seaborn
from matplotlib.backends.backend_pdf import PdfPages
import os
import itertools

def vis_GAMS(fleet, fp, filename, param_values, export_png, export_pdf=True, max_year=50, cropx=True, suppress_vis=False):
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

    # TODO: split into input/output visualization; add plotting of CO2 and stocks together


    os.chdir(fp)
    pp = PdfPages('output_vis_' + filename + '.pdf')
    plt.rcParams.update({'figure.max_open_warning': 0})  # suppress max 20 figures warning
    if suppress_vis:
        plt.ioff()

#%% Helper functions

    def fix_age_legend(ax, title='Vehicle ages'):
        # Customize legend formatting for stock figures

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
        export_fig(ax.get_title)
        pp.savefig(bbox_inches='tight')

    def plot_subplots(grouped_df, title, labels=None, cmap='jet', xlabel='year'):
        # plotting of input parameters by segment
        for (key, ax) in zip(grouped_df.groups.keys(), axes.flatten()):
            d = grouped_df.get_group(key)
            if d.index.nlevels == 3:
                if title == 'initial stock of each cohort':
                    d = grouped_df.get_group(key).reset_index(level=[0], drop=True)
                    d = d.unstack('reg')
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


    def sort_ind(ind):
        if isinstance(ind, pd.MultiIndex):
            # TODO: fix for multiindex - making one level categorical does not fix the order of elements in that level
            #  maybe need to rebuild the index from scratch after converting to categorical?
            for i, lvl in enumerate(ind.levels):
                if ind.levels[i].name == 'reg':
                    lvl_num = i
                    break
            df = ind.to_frame()
            df['reg'] = pd.Categorical(df['reg'], categories=['LOW', 'HIGH'], ordered=True)  # for simplified, 2-region test case
            # df['reg'] = pd.Categorical(df['reg'], categories=['LOW', 'II', 'MID', 'IV', 'HIGH'], ordered=True)
            ind = pd.MultiIndex.from_frame(df)
            # ind.set_levels(ind.levels[lvl_num].astype(cat_type), 'reg', inplace=True)
        else:
            ind = ind.astype(cat_type)
        return ind


    def fix_tuple_axis_labels(axes, axis_label, label_level=1):
        fig.canvas.draw()

        def reduce_tuple(ax):
            new_labels = [x.get_text().strip("()").split(", ")[label_level] for x in ax.get_xticklabels() if len(x.get_text()) > 0]
            ax.xaxis.set_ticklabels(new_labels)
            ax.set_xlabel(axis_label)

        if axes.ndim == 2:
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
        # for making legend entries fill by row rather than columns
        return itertools.chain(*[items[i::ncol] for i in range(ncol)])


    def export_fig(png_name=None):
        if export_pdf:
            pp.savefig(bbox_inches='tight')
        if export_png:
            png_name = ax.get_title()
            plt.savefig(png_name, format='png', bbox_inches='tight')
#
#        def crop_x(ax,max_year,cropx):
#            if cropx:
#                ax.xlim(right=max_year)

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
   #tec_cm = LinearSegmentedColormap.from_list('tec',['xkcd:aubergine', 'lavender'])

    #### Make summary page describing parameters for model run
    if param_values:
        div_page = plt.figure(figsize=(25, 8))
        ax = plt.subplot(111)
        ax.axis('off')
        for i, (key, value) in enumerate(param_values.items()):
            plt.text(0.05, 0.9-i*(0.05), key, fontsize=14)
            plt.text(0.15, 0.9-i*(0.05), str(value), fontsize=14)
        "df_param = pd.DataFrame.from_dict(param_values)"
        "df_param = df_param.T"

        # param_table = plt.table(cellText=df_param.values, colLabels=['scenario \n name', 'values'], rowLabels=df_param.index, colWidths=[0.1, 0.9], cellLoc='left', loc=8)
        # param_table.auto_set_font_size(False)
        # param_table.set_fontsize(14)
        # param_table.scale(1, 2.5)
        export_fig('tec-seg-cohort')
    else:
        print('Could not make parameter table in export PDF')

    """--- Plot total stocks by age, technology, and segment---"""
    """
    for region in self.reg[:-1]:
        fig, axes = plt.subplots(4, 3, figsize=(12,12), sharey=True, sharex=True)
        plt.ylim(0, np.ceil(self.stock_df_plot.sum(axis=1).max()))#/5e7)*5e7)

        if cropx:
            plt.xlim(right=max_year)

        for (key, ax) in zip(self.stock_df_plot_grouped.groups.keys(), axes.flatten()):
#            if(key==('BEV', 'B')):
#                fix_age_legend(ax)
            d = self.stock_df_plot_grouped.get_group(key).reset_index(level=[0, 1], drop=True)
            ax = d.loc[region].plot(ax=ax, kind='area', cmap='Spectral_r', legend=False)
#             self.stock_df_plot_grouped.get_group(key).plot(ax=ax,kind='area',cmap='Spectral_r',legend=False)
            #handles,labels = ax.get_legend_handles_labels()
            ax.xaxis.set_major_locator(MultipleLocator(10))
            ax.xaxis.set_minor_locator(MultipleLocator(5))
            ax.xaxis.set_tick_params(rotation=45)
#            ax.set_xticklabels([2000,2010,2020,2030,2040,2050],fontsize=9, rotation=45)
#            ax.set_xticklabels(self.stock_df_plot_grouped.groups[key].get_level_values('year'))
            ax.set_xlabel('year')
            ax.text(0.5, 0.9, key, horizontalalignment='center', transform=ax.transAxes, fontweight='bold')
#            plt.xticks(rotation=45)
#            ax.set_title(key,fontsize=10,fontweight='bold')
"""
#        patches, labels = ax.get_legend_handles_labels()
#        ax.legend(patches,labels,bbox_to_anchor=(1.62,4.35), ncol=2, title='Age')
#        """" Does not work, need a non-'aliased' reference to datapoint """
#        #ax.axvline(x=2020,ls='dotted',color='k')
#        fig.suptitle('Vehicle stock by technology and segment')
#        plt.subplots_adjust(hspace=0.12,wspace=0.1)
#        pp.savefig(bbox_inches='tight')

#        for (key, ax) in zip(self.stock_df_plot_grouped.groups.keys(), axes.flatten()):
#           self.stock_df_plot_grouped.get_group(key).plot(ax=ax,kind='area',cmap='Spectral_r')

#
#        ax = self.stock_df_plot.loc['BEV'].groupby('seg').plot(kind='area',cmap='Spectral_r',title='BEV stocks by age and segment')
#        ax = self.stock_df_plot.loc['BEV'].plot(kind='area',cmap='Spectral_r',title='BEV stocks by age and segment')
#        fix_age_legend(ax)
#        ax = self.stock_df_plot.loc['ICE'].groupby('seg').plot(kind='area',cmap='Spectral_r',title='ICE stocks by age and segment')
#        ax = self.stock_df_plot.loc['ICE'].plot(kind='area',cmap='Spectral_r',title='ICE stocks by age and segment')
#        fix_age_legend(ax)

    # trunc = lambda x: x.strip("()").split(",")[1]  # function for using only year as x-axis labels
    # print(trunc)
    # reg_dict = {'HIGH': 5, 'MID': 3, 'LOW': 1, 'II': 2, 'IV': 4, 'PROD': 6}

    # cat_type = CategoricalDtype(categories=['LOW', 'II', 'MID', 'IV', 'HIGH', 'PROD'], ordered=True)
    # cat_type = CategoricalDtype(categories=['LOW', 'HIGH', 'PROD'], ordered=True)  # simplified two-region system

    ord_reg = [reg for reg in fleet.sets.reg]
    cat_type = CategoricalDtype(categories = ord_reg, ordered=True)

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

    def remove_subplots(ax, empty_spots):
        for i in range(empty_spots):
            ax[-1, -1+i].remove()  # remove from bottom row, rightmost subplot first
    # ord_reg_ind = pd.CategoricalIndex(type(cat_type))

#%% Begin figure plotting

    #%% Stock additions by segment, technology and region (absolute)
    """--- Plot stock additions by segment, technology and region ---"""
    try:
        fig, axes = plt.subplots(plt_array[0], plt_array[1], sharex=True, sharey='row')

        tmp = fleet.stock_add.sum(axis=1).unstack('seg').unstack('tec').loc['2020':] / 1e6
        tmp.index = sort_ind(tmp.index)
        tmp = tmp.groupby('reg', sort=False)

        for (key, ax) in zip(tmp.groups.keys(), axes.flatten()):
            tmp.get_group(key).plot(ax=ax, kind='area', cmap=paired, lw=0, legend=False, title=f'{key}')
            ax.set_xbound(0, 80)
            ax.xaxis.set_major_locator(IndexLocator(10, 0))
            ax.xaxis.set_tick_params(rotation=45)

        # plot_subplots(tmp, 'Stock additions, by segment, technology and region', ['BEV', 'iCEV'])
        # (fleet.stock_add.sum(axis=1).unstack('seg').unstack('tec').groupby(['reg']).plot(kind='area', cmap=paired, title=f'Stock additions, by segment, technology and region')
        fix_tuple_axis_labels(axes, 'year')
        remove_subplots(axes, empty_spots)
        # axes[1, 2].remove()  # remove 6th subplot

        ref_ax = get_ref_ax(axes)
        fix_age_legend(ref_ax, 'Vehicle technology and segment')
        ax.set_ylabel('Vehicles added to stock \n millions of vehicles')
        plt.ylim(0, np.ceil((fleet.stock_add.sum(axis=1)).max() / 1e6))

        fig.suptitle('Stock additions, by segment, technology and region', y=0.995)

        #%% market shares by segment and technology (normalized)
        """--- Plot stock addition shares by segment and technology ---"""
        fig, axes = plt.subplots(plt_array[0], plt_array[1], sharex=True, sharey=True)
        tmp = fleet.add_share
        tmp.index = sort_ind(tmp.index)
        tmp = tmp.groupby(['reg'], sort=False)

        for (key, ax) in zip(tmp.groups.keys(), axes.flatten()):
            tmp.get_group(key).plot(ax=ax, kind='area', cmap=paired, lw=0, legend=False, title=f'{key}')
            ax.set_xbound(0, 80)
            ax.xaxis.set_major_locator(IndexLocator(10, 0))
            ax.xaxis.set_minor_locator(IndexLocator(2, 0))
            ax.xaxis.set_tick_params(rotation=45)

        fix_tuple_axis_labels(axes, 'year')
        remove_subplots(axes, empty_spots)
        # axes[1, 2].remove()  # remove 6th subplot
        ref_ax = get_ref_ax(axes)
        fix_age_legend(ref_ax, 'Vehicle technology and segment')
        fig.suptitle('Stock additions, by technology, vehicle segment and region,\n as share of total stock', y=1.05)

        #%% market shares by segment and technology (un-normalized)
        """--- Plot tech split of stock additions by segment ---"""
        fig, axes = plt.subplots(plt_array[0], plt_array[1], sharex=True, sharey=True)
        tmp = fleet.add_share.div(fleet.add_share.sum(axis=1, level='seg'), axis=1, level='seg')
        tmp.index = sort_ind(tmp.index)
        tmp = tmp.groupby(['reg'], sort=False)

        for (key, ax) in zip(tmp.groups.keys(), axes.flatten()):
            tmp.get_group(key).plot(ax=ax, kind='area', cmap=paired, lw=0, legend=False, title=f'{key}')
            ax.set_xbound(0, 80)
            ax.xaxis.set_major_locator(IndexLocator(10, 0))
            ax.xaxis.set_minor_locator(IndexLocator(2, 0))
            ax.xaxis.set_tick_params(rotation=45)

        fix_tuple_axis_labels(axes, 'year')
        remove_subplots(axes, empty_spots)

        # axes[1, 2].remove()  # remove 6th subplot
        ref_ax = get_ref_ax(axes)
        fix_age_legend(ref_ax, 'Vehicle technology and segment')
        fig.suptitle('Stock additions, by technology, vehicle segment and region, \n as share of segment stock', y=1.05)

    except Exception as e:
        print(f'Error with stock additions, {key}')
        print(e)

    #%% Segment shares of BEVs by region
    """--- Plot market share of BEVs by segment and region ---"""
    # TODO: check if this calculation is correct (cross-check from first principles)
    fig, axes = plt.subplots(plt_array[0], plt_array[1], sharex=True, sharey=True)
    # tmp = (fleet.add_share.div(fleet.add_share.sum(axis=1, level=['seg']), axis=1, level=['seg','reg']))

    tmp = fleet.add_share.div(fleet.add_share.sum(axis=1, level='seg'), axis=1, level='seg')
    tmp = tmp.drop('ICE', axis=1, level='tec')
    tmp.index = sort_ind(tmp.index)
    tmp = tmp.groupby(['reg'])

    for (key, ax) in zip(tmp.groups.keys(), axes.flatten()):
        tmp.get_group(key).plot(ax=ax, cmap=dark, legend=False, title=f'{key}')
        ax.set_xbound(0, 80)
        ax.xaxis.set_major_locator(IndexLocator(10, 0))
        ax.xaxis.set_minor_locator(IndexLocator(2, 0))
        ax.xaxis.set_tick_params(rotation=45)

    fix_tuple_axis_labels(axes, 'year')
    remove_subplots(axes, empty_spots)
    # axes[1, 2].remove()  # remove 6th subplot

    ref_ax = get_ref_ax(axes)
    fix_age_legend(ref_ax, 'Vehicle segment')
    handles, labels = ref_ax.get_legend_handles_labels()
    labels = [label.strip('()').split(',')[0] for label in labels]
    ref_ax.legend(handles, labels)
    fig.suptitle('Market share of BEVs by segment and region', y=0.995)

    #%% Regional shares of BEVs by segment
    """--- Plot market share of BEVs by segment and region ---"""
    fig, axes = plt.subplots(plt_array[0], plt_array[1], sharex=True, sharey='row')
    tmp = fleet.stock_add.div(fleet.stock_add.sum(level=['seg', 'tec', 'prodyear']))
    tmp = tmp.drop('ICE', axis=0, level='tec')#.drop('PROD', axis=0, level='reg')
    # tmp = (fleet.add_share.div(fleet.add_share.sum(axis=1,level='seg'), axis=1, level='seg')).drop('ICE', axis=1, level='tec')
    # tmp = fleet.add_share.div(fleet.add_share.sum(axis=1, level='seg'), axis=1, level='seg')
    tmp.index = sort_ind(tmp.index)
    tmp = tmp.unstack('reg').droplevel('age', axis=1).groupby(['seg'])  #.drop('PROD', axis=1)
    # TODO: fix region order

    for (key, ax) in zip(tmp.groups.keys(), axes.flatten()):
        tmp.get_group(key).plot(ax=ax, kind='area', stacked=True, cmap=dark, lw=0, legend=False, title=f'{key}')
        ax.set_xbound(0, 80)
        ax.xaxis.set_major_locator(IndexLocator(10, 0))
        ax.xaxis.set_minor_locator(IndexLocator(2, 0))
        ax.xaxis.set_tick_params(rotation=45)

    fix_tuple_axis_labels(axes, 'year', label_level=2)
    ref_ax = get_ref_ax(axes)
    fix_age_legend(ref_ax, 'Region')
    fig.suptitle('Regional share of BEVs by segment ', y=0.995)

#         for region in fleet.reg[:-1]:
#             """  change this to be subplots with regions """
#             ax = (fleet.stock_add.sum(axis=1).unstack('seg').unstack('tec').loc[region]).plot(kind='area', cmap=paired, title=f'Stock additions, by segment and technology in region {region}')
# #        ax = (fleet.stock_add.sum(axis=1).unstack('seg').unstack('tec')/1e6).groupby('reg').plot(kind='area',cmap=paired,title='Stock additions, by segment and technology')
#             fix_age_legend(ax, 'Vehicle technology and segment')
#             ax.set_ylabel('Vehicles added to stock \n millions of vehicles')
#             plt.ylim(0, np.ceil(fleet.stock_df_plot.sum(axis=1).max()))#/5e7)*5e7)
#             #axes = fleet.stock_add.unstack('seg').groupby('tec').plot(kind='area',cmap='jet',title='Stock additions by segment and technology')
#             #ax.set_xticklabels([2000,2010,2020,2030,2040,2050])
#             #ax.set_xlabel('year')
#             #ax.axvline(x=2020,ls='dotted')

#             """--- Plot stock addition shares by segment and technology ---"""
#             ax = fleet.add_share.loc[region].plot(kind='area', cmap=paired, title=f'Share of stock additions, by technology and vehicle segment in region {region}')
#             ax.xaxis.set_minor_locator(MultipleLocator(1))
#             ax.grid(which='minor', axis='x', c='w', alpha=0.6, linestyle=(0, (5,10)), lw=0.1)
#             ax.grid(which='major', axis='x', c='darkgrey', alpha=0.75, linestyle='--', lw=0.5,)
#             fix_age_legend(ax, 'Vehicle technology and segment')

#             """--- Plot tech split of stock additions by segment ---"""
#             temp_df = fleet.add_share/fleet.add_share.sum(axis=1, level=0)
#             ax = temp_df.loc[region].plot(kind='area', cmap=paired, title=f'Technological split of total segment additions in region {region}')
#             ax.xaxis.set_minor_locator(MultipleLocator(1))
#             ax.yaxis.set_minor_locator(MultipleLocator(0.25))
#             ax.grid(which='minor', axis='x', c='w', alpha=0.6, linestyle=(0,(5,10)), lw=0.1)
#             ax.grid(which='major', axis='x', c='darkgrey', alpha=0.75, linestyle='--', lw=0.5)
#             ax.grid(which='minor', axis='y', c='w', alpha=0.6, linestyle='dotted', lw=0.1)
#             fix_age_legend(ax, 'Vehicle technology and segment')

#             """--- Plot share of BEVs in stock additions ---"""
#             temp_df = (fleet.add_share/fleet.add_share.sum(axis=1,level=0)).drop('ICE', axis=1, level=1)
#             ax = temp_df.loc[region].plot(kind='line', cmap=paired, title=f'Share of BEVs in stock additions in region {region}')
#             ax.xaxis.set_minor_locator(MultipleLocator(1))
#             ax.yaxis.set_minor_locator(MultipleLocator(0.25))
#             ax.grid(which='minor', axis='x', c='w', alpha=0.6, linestyle=(0,(5,10)), lw=0.1)
#             ax.grid(which='major', axis='x', c='darkgrey', alpha=0.75, linestyle='--', lw=0.5)
#             ax.grid(which='minor', axis='y', c='w', alpha=0.6, linestyle='dotted', lw=0.1)
#             fix_age_legend(ax, 'Vehicle technology and segment')

    #%% Total emissions by tec and lifecycle phase, against total stock
    """--- Plot total emissions by tec and lifecycle phase---"""
    fleet.emissions.sort_index(axis=1, level=0, ascending=False, inplace=True)
#        fleet.emissions = fleet.emissions/1e6

    fig = plt.figure(figsize=(14,9))

    gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1,3], hspace=0.05)
    ax2 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax2)
    (fleet.emissions/1e6).plot(ax=ax1, kind='area', lw=0, cmap=cmap_em)
    (fleet.stock_df_plot.sum(axis=1).unstack('seg').sum(axis=1).unstack('tec').sum(level='year')/1e6).plot(ax=ax2, kind='area', cmap=tec_cm, lw=0)

    ax1.set_ylabel('Lifecycle climate emissions \n Mt $CO_2$-eq', fontsize=13)
    ax2.set_ylabel('Vehicles, millions', fontsize=13, labelpad=25)
    if cropx:
        ax1.set_xlim(right=max_year)
        ax2.set_xlim(right=max_year)
#        patches, labels = ax1.get_legend_handles_labels()
#        order = [5, 3, 1, 4, 2, 0]
#        ax1.legend([patches[idx] for idx in order],[labels[idx] for idx in order], loc=1, fontsize=12)
    handles, labels = ax1.get_legend_handles_labels()
    labels = [x+', '+y for x, y in itertools.product(['Production', 'Operation', 'End-of-life'], ['ICEV', 'BEV'])]
    ax1.legend(handles, labels, loc=1, fontsize=14)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, ['BEV', 'ICEV'], loc=4, fontsize=14, framealpha=1)

    ax1.set_xbound(0, 50)
    ax2.set_xbound(0, 50)

    plt.setp(ax2.get_yticklabels(), fontsize=14)
    plt.xlabel('year', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    export_fig('LC_emissions_vs_stock')
#        pp.savefig(bbox_inches='tight')

    #%% Operation emissions by tec, benchmarked against regulation emissions targets
    """--- Plot operation emissions by tec ---"""

    ax = (fleet.emissions.loc[:, 'Operation'] / 1e6).plot(kind='area', cmap=LinearSegmentedColormap.from_list('temp', colors=['silver', 'grey']), lw=0)
    ax.set_xbound(0, 50)

    # plot regulation levels
    plt.hlines(442, xmin=0.16, xmax=0.6, linestyle='dotted', color='darkslategrey', label='EU 2030 target, \n 20% reduction from 2008 emissions', transform=ax.get_yaxis_transform())
    plt.hlines(185, xmin=0.6, xmax=1, linestyle='-.', color='darkslategrey', label='EU 2050 target, \n 60% reduction from 1990 emissions', transform=ax.get_yaxis_transform())
    plt.ylabel('Fleet operation emissions \n Mt $CO_2$-eq')
    if cropx:
        plt.xlim(right=max_year)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['ICEV', 'BEV', 'EU 2030 target, \n20% reduction from 2008 emissions', 'EU 2050 target, \n60% reduction from 1990 emissions'], bbox_to_anchor= (1.05, 1.02))#loc=1
    export_fig('operation_emissions')
#        pp.savefig(bbox_inches='tight')

    #%% Total stocks by segment
    """--- Plot total stocks by segment ---"""
    # TODO: fix legend order of segments? (one column?)
    ax = fleet.stock_df_plot.sum(axis=1).unstack('seg').sum(axis=0, level=['year']).plot(kind='area', cmap='jet', lw=0, title='Total stocks by segment')
    ax.set_xbound(0, 80)
    fix_age_legend(ax, 'Vehicle segments')

    #%% Total stocks by region
    """--- Plot total stocks by region ---"""
    # TODO: set to categorical index for legend
    tmp = fleet.stock_df_plot.sum(axis=1).unstack('fleetreg').sum(axis=0, level=['year'])
    tmp.columns = sort_ind(tmp.columns)
    ax = tmp.plot(kind='area', cmap='jet', lw=0, title='Total stocks by region')
    ax.set_xbound(0, 80)
    fix_age_legend(ax, 'Region')

    #%% Total stocks by segment and technology
    """--- Plot total stocks by age, segment and technology ---"""
    ax = fleet.stock_df_plot.sum(axis=1).unstack('seg').unstack('tec').sum(axis=0, level='year').plot(kind='area', cmap=paired, lw=0, title='Total stocks by segment and technology')
    ax.set_xbound(0, 80)
    fix_age_legend(ax, 'Vehicle segment and technology')

    #%% Total stocks by age, segment and technology
    """--- Plot total stocks by age, segment and technology ---"""

#        ax = fleet.stock_df_plot.sum(axis=1).unstack('seg').unstack('tec').unstack('reg').plot(kind='area',cmap=paired,title='Total stocks by segment, technology and region')
    stock_tec_seg_reg = fleet.stock_df_plot.sum(axis=1).unstack('seg').unstack('tec').unstack('fleetreg')
    stock_tec_seg_reg = stock_tec_seg_reg.stack(['seg', 'tec'])
    stock_tec_seg_reg.columns = sort_ind(stock_tec_seg_reg.columns)
    stock_tec_seg_reg = stock_tec_seg_reg.unstack(['seg','tec']).reorder_levels(['seg','tec','fleetreg'], axis=1).sort_index(axis=1, level=['seg','tec'])
    ax = stock_tec_seg_reg.plot(kind='area', cmap='jet', lw=0, title='Total stocks by segment, technology and region')
    ax.set_xbound(0, 80)

    # TODO: fix region order
    fix_age_legend(ax, 'Vehicle segment, technology and region')

    #%% Total stocks by technology and segment
    """--- Plot total stocks by technology and segment ---"""
#        ax = fleet.veh_stck.unstack(['tec', 'seg', 'year']).sum().unstack(['tec', 'seg']).stack().unstack(['seg']).plot(kind='area',cmap=paired_tec, title='Total stocks by technology and segment')
#        fix_age_legend(ax, 'Vehicle segment and technology')

    #%% Total stocks by age
    """--- Plot total stocks by age ---"""
    #stock_df_plot = stock_df_plot.sum(axis=1,level=1) # aggregates segments
#        ax = fleet.stock_df_plot.sum(level=2).plot(kind='area',cmap='Spectral_r',title='Total stocks by age')
#        fix_age_legend(ax)


    #ax = fleet.stock_df_plot.sum(level=2).plot.barh()
    """--- Plot total stocks by age and technology ---"""

    """ Plot BEVs by cohort and region """
    """
    stock_cohort = fleet.stock_cohort.copy()
    stock_cohort.sort_index(level=['reg', 'modelyear'],ascending=[1,2], inplace=True)
    temp_stock_cohort = (stock_cohort/1e6).loc['BEV'].loc(axis=0)[:, '2020':'2050']
    temp_stock_cohort[temp_stock_cohort < 1e-9] = 0#0.4] = 0 # Drops very small vehicle stocks in earlier years
    fig, axes = plt.subplots(len(fleet.reg[:-1]), 1, sharex=True, sharey=True)
    plt.subplots_adjust(top=0.85, hspace=0.25)

    for i, reg in enumerate(fleet.reg[:-1]):
        plot_stock = (temp_stock_cohort.loc[reg].replace(0,np.nan).dropna(how='all', axis=1))
        plot_stock.plot(ax=axes[i], kind='bar', stacked=True, width=1, cmap='Spectral', title=f'BEV stock by vintage cohort in region {reg}', legend=False)

#        for reg, group in plot_stock_cohort:
#            group = group.replace(0, np.nan).dropna(how='all', axis=1)
#            ax = group.plot(kind='bar', stacked=True, width=1, cmap='Spectral', title=f'BEV stock by vintage cohort in region {reg}')
    axes[-1].xaxis.set_major_locator(IndexLocator(5, 0))
    x_labels = [label[1] for label in temp_stock_cohort.index.tolist()]
    axes[-1].xaxis.set_major_formatter(IndexFormatter(x_labels))
    axes[-1].xaxis.set_minor_locator(IndexLocator(5, 0))
#        ax.set_xticklabels([2015,2020,2025,2030,2035,2040,2045,2050])
    plt.ylabel('BEV stock, in millions of vehicles', y=0.6)
    plt.xlabel('year')
    fix_age_legend(axes[0], title='Vehicle vintage')
    pp.savefig()
    """
    """temp_stock_cohort = (fleet.stock_cohort/1e6).loc['ICE'].loc[:'2050']
    temp_stock_cohort[temp_stock_cohort<0.4] = 0 # Drops very small vehicle stocks in earlier years
    temp_stock_cohort = temp_stock_cohort.replace(0,np.nan).dropna(how='all',axis=1)
    try:
        ax = temp_stock_cohort.plot(kind='bar',stacked=True,width=1,cmap='Spectral',title='ICEV stock by vintage cohort')
        ax.xaxis.set_major_locator(IndexLocator(10,0))
        ax.xaxis.set_major_formatter(IndexFormatter(temp_stock_cohort.index))
        ax.xaxis.set_tick_params(rotation=45)
        ax.xaxis.set_minor_locator(IndexLocator(5,0))
#        ax.set_xticklabels([1995,2000,2010,2020,2030,2040,2050])
#        ax.set_xticklabels([2015,2020,2025,2030,2035,2040,2045,2050])
        plt.ylabel('ICEV stock, in millions of vehicles')
        plt.xlabel('year')
        fix_age_legend(ax,title='Vehicle vintage')
    except TypeError:
        print('No ICEVs!')"""

    #%% Addition to stocks by segment, technology and region
    """--- Plot addition to stocks by segment, technology and region ---"""
    fig, ax = plt.subplots(1, 1)
    plot_stock_add = fleet.stock_add.sum(level=['tec', 'reg', 'prodyear']).unstack(['tec', 'reg']).droplevel(axis=1, level=0)
    plot_stock_add = plot_stock_add.stack('tec')
    plot_stock_add.columns = sort_ind(plot_stock_add.columns)
    plot_stock_add = plot_stock_add.unstack('tec').swaplevel('tec', 'reg', axis=1).sort_index(axis=1, level='tec')
    plot_stock_add.plot(ax=ax, kind='area', cmap=tec_cm4, lw=0, legend=True, title='Stock additions by technology and region')
    ax.set_xbound(0, 80)

    # TODO: fix region order workaround
    fix_age_legend(ax, 'Vehicle technology and region')
    plt.xlabel('year')
    plt.ylabel('Vehicles added to stock')
    pp.savefig()

    #%% Total stocks by segment, technology and region
    """--- Plot total stocks by segment, technology and region ---"""
    fig, ax = plt.subplots(1, 1)
    plot_stock = fleet.veh_stck.sum(axis=1).unstack(['tec', 'fleetreg']).sum(level=['year'])

    plot_stock = plot_stock.stack('tec')  # remove MultiIndex to set Categorical type for regions
    plot_stock.columns = sort_ind(plot_stock.columns)
    plot_stock = plot_stock.unstack('tec').swaplevel('tec', 'fleetreg', axis=1).sort_index(axis=1, level='tec')
    plot_stock.plot(ax=ax, kind='area', cmap=tec_cm4, lw=0, legend=True, title='Total stock by technology and region')
    ax.set_xbound(0, 80)
    # TODO: fix region order workaround

    fix_age_legend(ax, 'Vehicle technology and region')

    ax.set_xlabel('year')
    ax.set_ylabel('Vehicles in stock')
    pp.savefig()
    plt.show()  # without this line, legend shows as "BEV" and "ICEV" ...????

#        patches, labels  = ax.get_legend_handles_labels()
#        plt.legend(handles=patches, labels=labels, bbox_to_anchor=(-0.05, 0.6), title='Vehicle technology and region')

#        fig,axes = plt.subplots(1,2,figsize=(6,3))
#        stock_add_grouped = fleet.stock_add.unstack('seg').groupby('tec')
#        for (key,ax) in zip(stock_add_grouped.groups.keys(),axes.flatten()):
#            stock_add_grouped.get_group(key).plot(ax=ax,kind='area',cmap='jet',legend=False)
#            ax.set_xticklabels([2000,2010,2020,2030,2040,2050])
#            ax.set_xlabel('year')
#            ax.set_title(key,fontsize=10,fontweight='bold')
#            #ax.axvline(x=('BEV',2020),ls='dotted')
#        fig.suptitle('Additions to stock by segment and technology')
#        ax.legend(labels=fleet.seg,title='Segment',markerscale=15)

    #%% Total resource use - against total vehicle fleet
    """--- Plot total resource use ---"""
    for resource in fleet.resources.columns.get_level_values(1).unique():
        fig = plt.figure(figsize=(14,9))

        gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1,3], hspace=0.05)
        ax2 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1], sharex=ax2)
        plot_df = pd.concat([fleet.resources['primary', resource], fleet.resources['recycled', resource]], axis=1)
        plot_df[plot_df < 0] = 0  # replace with fleet.resources[total required]
        (plot_df/1e6).plot(ax=ax1, kind='area', lw=0, cmap='jet')
        (fleet.stock_df_plot.sum(axis=1).unstack('seg').sum(axis=1).unstack('tec').sum(level='year')/1e6).plot(ax=ax2, kind='area', cmap=tec_cm, lw=0)

        ax1.set_ylabel(f'{resource} used in new batteries \n Mt {resource}', fontsize=13)
        ax2.set_ylabel('Vehicles, millions', fontsize=13, labelpad=25)
        if cropx:
            ax1.set_xlim(right=max_year)
            ax2.set_xlim(right=max_year)
    #        patches, labels = ax1.get_legend_handles_labels()
    #        order = [5, 3, 1, 4, 2, 0]
    #        ax1.legend([patches[idx] for idx in order],[labels[idx] for idx in order], loc=1, fontsize=12)
        handles, labels = ax1.get_legend_handles_labels()
        new_labels = []
        for label in labels:
            new_labels.append(str(label)[1:-3].replace(",", "").replace("'","").capitalize())
        ax1.legend(handles, new_labels, title=f'{resource} source used')

        # labels = [x+', '+y for x,y in itertools.product(['Production', 'Operation', 'End-of-life'], ['ICEV', 'BEV'])]
        # ax1.legend(handles, labels, loc=1, fontsize=14)
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(handles, ['BEV', 'ICEV'], loc=4, fontsize=14, framealpha=1)


        ax1.set_xbound(0, 50)
        ax2.set_xbound(0, 50)

        plt.setp(ax2.get_yticklabels(), fontsize=14)
        plt.xlabel('year', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

    """--- Plot production mixes for virgin critical materials ---"""
    fig, axes = plt.subplots(len(fleet.sets.mat_cats), 1, sharex=True)
    plot_data = fleet.mat_mix.copy()
    plot_data.drop(index=plot_data.loc[str(max_year + 2001):].index.tolist(), inplace=True)
    plot_data = plot_data / 1e6
    for i, (mat, prods) in enumerate(fleet.sets.mat_prod.items()):
        plot_data[prods].plot(ax=axes[i], kind='area', stacked=True, cmap='jet')
        axes[i].set_title(f'{mat} production mix', fontsize=10, fontweight='bold')
        axes[i].legend(title=f'{mat} producer')
        axes[i].set_ylabel('Mt material')

    fig.suptitle('Production shares for virgin critical materials', y=0.995)

    pp.savefig()

    #%%  Input parameter checking
    """--- Divider page for input parameter checking ---"""
    div_page = plt.figure(figsize=(11.69, 8.27))
    txt = 'Plotting of input parameters for checking'
    div_page.text(0.5, 0.5, txt, transform=div_page.transFigure, size=30, ha="center")
    pp.savefig()


    """ Plot evolution of lifecycle emissions """
    fig, axes = plt.subplots(len(fleet.sets.fleetreg), len(fleet.sets.tecs), sharey=True, sharex=True, figsize=(5, 12))
    plt.subplots_adjust(top=0.85, hspace=0.25, wspace=0.05)

    for i, reg in enumerate(fleet.sets.fleetreg):
        for j, tec in enumerate(fleet.sets.tecs):
            plot_data = fleet.LC_emissions.unstack('seg').loc(axis=0)[tec, '1995':'2050', reg]
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

#        fig.text('Cohort year', 0.5, 0, ha='center')
#        full_fig.set_xlabel('Cohort year')
    fig.suptitle('Evolution of lifecycle emissions by \n cohort, segment, region and technology', y=0.925)
#        fix_age_legend(axes[0,1], title='Vehicle segment')
    patches, labels = axes[0, 0].get_legend_handles_labels()
    labels = [lab[4] for lab in labels]
    fig.legend(patches, labels, bbox_to_anchor=(1.0, 0.8), loc='upper left', title='Vehicle segment', borderaxespad=0.)
    pp.savefig()
    plt.show()

    """--- to do: plot BEV cohorts as share of total fleet ---"""
    """--- to do: plot crossover in LC emissions between ICEV and BEV by segment---"""

    """--- Plot production emissions by tec and seg ---"""
#        fig,axes = plt.subplots(1,2)
#        for (key,ax) in zip(fleet.veh_prod_totc.groupby(['tec', 'seg']).groups.keys(),axes.flatten()):
#            fleet.veh_prod_totc.groupby(['tec', 'seg']).get_group(key).plot(ax=ax,kind='area',cmap='jet',legend=False)
#            ax.set_xticklabels([2000,2010,2020,2030,2040,2050])
#            ax.set_xlabel('year')
#            ax.set_title(key,fontsize=10,fontweight='bold')
#            #ax.axvline(x=('BEV',2020),ls='dotted')
#            ax.set_label('segment')
#        ax.legend()
#        pp.savefig(bbox_inches='tight')

    """--- Plot production emissions by tec and seg ---"""
    prod = fleet.veh_prod_totc.stack().unstack('tec').sum(level=['seg', 'year'])#/1e9
    prod_int = prod / fleet.stock_add.sum(axis=1).unstack('tec').sum(level=['seg', 'prodyear'])  # production emission intensity

    fig, axes = plt.subplots(3, 2, figsize=(9,9), sharex=True, sharey=True)
    labels=['BEV', 'ICEV']
    title='Total production emissions by technology and segment'
    #plot_subplots((fleet.veh_prod_totc.unstack('tec').groupby(['seg'])),title=title,labels=labels)
    plot_subplots(prod.groupby(['seg']), title=title, labels=labels)
#        ax.legend(labels=['BEV', 'ICE'],bbox_to_anchor=(0.2,-0.3),ncol=2,fontsize='large')
    fig.text(0.04, 0.5, 'Production emissions \n(Mt CO2-eq)', ha='center', va='center', rotation='vertical')
    export_fig('tot_prod_emissions')
    pp.savefig(bbox_inches='tight')

    fig, axes = plt.subplots(3, 2, figsize=(9,9), sharex=True, sharey=True)
    #ax.legend(labels=['BEV', 'ICE'],bbox_to_anchor=(0.2,-0.3),ncol=2,fontsize='large')
    title = 'Production emission intensities by technology and segment'
    plot_subplots(prod_int.groupby(['seg']),title=title,labels=labels)
    fig.text(0.04, 0.5, 'Production emissions intensity \n(t CO2/vehicle)', ha='center', va='center', rotation='vertical')
    export_fig('prod_intensity_out')
    pp.savefig(bbox_inches='tight')

    fig, axes = plt.subplots(3, 2, figsize=(9,9), sharex=True, sharey=True)
    title = 'VEH_PROD_CINT'
    plot_subplots(fleet.veh_prod_cint.unstack('tec').groupby(['seg']), title=title, labels=labels)
    fig.text(0.04, 0.5, 'Production emissions intensity \n(t CO2/vehicle)', ha='center', va='center', rotation='vertical')
    export_fig('VEH_PROD_CINT')
    pp.savefig(bbox_inches='tight')

    fig, axes = plt.subplots(3, 2, figsize=(9,9), sharex=True, sharey=True)
    title = 'VEH_OPER_EINT - check ICE sigmoid function'
    plot_subplots(fleet.veh_oper_eint.unstack('tec').groupby(['seg']), title=title, labels=labels)
    fig.text(0.04, 0.5, 'Operation energy intensity \n(kWh/km)', ha='center', va='center', rotation='vertical')
    export_fig('VEH_OPER_EINT')
    pp.savefig(bbox_inches='tight')

    fig, ax = plt.subplots(1, 1)
    tmp = (fleet.enr_cint * 1000).unstack(['reg'])
    tmp.columns = sort_ind(tmp.columns)
    tmp = tmp.unstack(['enr']).swaplevel('enr', 'reg', axis=1).sort_index(level='enr', axis=1)
    # tmp.drop([('FOS', 'II'):('FOS','PROD')], inplace=True)
    tmp['ELC'].plot(ax=ax, title='ENR_CINT')
    tmp[('FOS', 'LOW')].plot(ax=ax, color='darkslategrey', linestyle='dashed', label='FOS (all countries)')

    plt.ylabel('Fuel chain emissions intensity, \n g CO2-eq/kWh')
    # if cropx:
    #     plt.xlim(right=max_year)
    ax.set_xbound(0, 50)

    handles, labels = ax.get_legend_handles_labels()
    labels[:-1] = ['ELC, '+ label for label in labels[:-1]]
    ax.legend(flip(handles, 4), flip(labels, 4), bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=4)
    export_fig('ENR_CINT')
    pp.savefig(bbox_inches='tight')
    plt.show()


#        fig, axes = plt.subplots(3,2,figsize=(9,9),sharey=True)
#        title = 'VEH_OPER_CINT'
#        temp_oper_cint = fleet.veh_oper_cint.loc[~fleet.veh_oper_cint.index.duplicated(keep='first')]*1e6
#        plot_subplots(temp_oper_cint.unstack('tec').groupby(['seg']),title=title,labels=labels)
#        fig.text(0.04, 0.5, 'Operation emissions intensity \n(g CO2-eq/km)', ha='center', rotation='vertical')
#        pp.savefig(bbox_inches='tight')
#
#        fig, axes = plt.subplots(3,2,figsize=(9,9),sharey=True)
#        title = 'Lifetime operation emissions by cohort for whole fleet'
#        plot_subplots(fleet.veh_op_cohort.unstack('tec').groupby(['seg']),title=title,labels=labels)
#        fig.text(0.04, 0.5, 'Operation emissions \n(t)', ha='center', rotation='vertical')
#        pp.savefig(bbox_inches='tight')

#        fig, axes = plt.subplots(3,2,figsize=(9,9),sharey=True)
#        title = 'Operating emissions, back calculated from fleet emissions'
#        plot_subplots(fleet.veh_op_intensity.unstack('tec').groupby(['seg']),title=title,labels=labels)
#        fig.text(0.04, 0.5, 'Lifetime operation emissions intensity \n(t/vehicle)', ha='center', rotation='vertical')
#        pp.savefig(bbox_inches='tight')

    fig, axes = plt.subplots(3, 2, figsize=(9,9), sharex=True, sharey=True)
    title = 'initial stock of each cohort'
    tmp = (fleet.stock_add.unstack('tec').droplevel('age', axis=1) / 1e6)
    # tmp.drop('PROD', level='reg', axis=0, inplace=True)
    tmp = tmp.unstack('reg')
    tmp.columns = sort_ind(tmp.columns)

    tmp = tmp.stack('reg').reorder_levels(['seg', 'reg', 'prodyear'])
    tmp = tmp.groupby(['seg'])
    plot_subplots(tmp, cmap=tec_cm4, title=title)#, labels=labels)
    fig.text(0.04, 0.5, 'Total vehicles, by segment and technology \n(millions)', ha='center', va='center', rotation='vertical')
    patches, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(flip(patches, 5), flip(labels, 5), bbox_to_anchor=(0.5, 0), loc='lower center', ncol=5, borderaxespad=0.)
    pp.savefig(bbox_inches='tight')
    plt.show()

    fig, axes = plt.subplots(3, 2, figsize=(9,9), sharex=True, sharey=True)
    fig.text(0.04, 0.5, 'Vehicle operating emissions intensity, by region and segment \n (kg CO2-eq/km)', ha='center', va='center', rotation='vertical')
    title = 'VEH_OPER_CINT for BEVs, by region and segment'
    veh_op_cint_plot = fleet.veh_oper_cint.droplevel(['prodyear', 'enr']).drop_duplicates().unstack(['reg']).loc(axis=0)['BEV']
    veh_op_cint_plot = (veh_op_cint_plot.swaplevel(-2, -1, axis=0) * 1000).unstack('age')

    k_r_cmap = ListedColormap(['k' for i in np.arange(0, (len(veh_op_cint_plot.columns) / 2))] +
                              ['r' for i in np.arange(0, (len(veh_op_cint_plot.columns) / 2))])

    plot_subplots(veh_op_cint_plot.groupby(['seg']), title=title, cmap=k_r_cmap)
    axes[0, 0].set_xbound(0, 80)

    red_patch = matplotlib.patches.Patch(color='r', label='Low values by region')
    blk_patch = matplotlib.patches.Patch(color='k', label='High values by region')
    fig.legend(handles=[blk_patch, red_patch], bbox_to_anchor=(0.5, 0), loc='lower center',
               ncol=2, fontsize='large', borderaxespad=0.)
#        ax = veh_op_cint_plot.groupby(['seg']).plot(cmap=k_r_cmap)
    # ax.legend(labels=labels, bbox_to_anchor=(0.2, -0.3), ncol=2, fontsize='large')

    pp.savefig(bbox_inches='tight')

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

#        plt.clf()
    """For later: introduce figure plotting vehicle stock vs emissions"""

    # Plot parameter values for quality assurance
#        ax= p_df.plot(subplots=True,title='Parameter values')
