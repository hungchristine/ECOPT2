# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:26:45 2019

@author: chrishun
"""
import logging
import sys

import fleet_model
import gams_runner
#import sigmoid
#import test_gams
from itertools import product
from datetime import datetime
import yaml
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcolors

import numpy as np

import copy
import pickle
import os

fp = r'C:\Users\chrishun\Box Sync\YSSP_temp\visualization output\Run_2019-09-27T19_37\Sensitivity_analysis'

os.chdir(fp)


def fix_legend(ax,title='Vehicle ages'):
    patches, labels = ax.get_legend_handles_labels()
    ax.legend(patches,labels,bbox_to_anchor=(1.05,1.02), ncol=2, title=title)
#
#
##new values
#with open('sensitivity_analysis_2019-09-29T17_37.pkl','rb') as f:
#    new = pickle.load(f)
#
##original run
#with open('sensitivity_analysis_2019-09-29T16_49.pkl','rb') as f:
#    original = pickle.load(f)

#updated=[]
#for i,df in enumerate(original):
#    if isinstance(df,pd.DataFrame):
#        updated.append(df.join(new[i],lsuffix='old'))
#        try:
#            updated.append(df.update(new[i]))
#        except IndexError:
#            for index,row in new[i].iterrows():
#                temp=original[5].update(row)
##            temp = df.join(new[i],rsuffix='keep')
##            temp.dropna
#            updated.append()
##            updated.append(df.T.update(new[i].T))
#        except Exception as e:
#            print(f'fail on {i}')
#            print(e)
            
#with pd.ExcelWriter('aggregate_runs.xlsx') as writer:
#    original[0].to_excel(writer,sheet_name='original_0')
#    new[0].to_excel(writer,sheet_name='new_0')
#    original[1].to_excel(writer,sheet_name='original_1')
#    new[1].to_excel(writer,sheet_name='new_1')
#    original[2].to_excel(writer,sheet_name='original_2')
#    new[2].to_excel(writer,sheet_name='new_2')
#    original[3].to_excel(writer,sheet_name='original_3')
#    new[3].to_excel(writer,sheet_name='new_3')
#    original[4].to_excel(writer,sheet_name='original_4')
#    new[4].to_excel(writer,sheet_name='new_4')
#    original[5].to_excel(writer,sheet_name='original_5')
#    new[5].to_excel(writer,sheet_name='new_5')

##
#shares_2030=updated[0]
#shares_2050=updated[1]
#add_share = updated[2]
#stock_comp = updated[3]
#full_BEV_yr = original[4]

#scenario_totcs = pd.read_excel('final_cumulative_output.xlsx',sheet_name='totc_opt1',index='A',header=0)
#scenario_totcs.set_index(scenario_totcs.columns[0],inplace=True)

#with pd.ExcelWriter('final_cumulative_output.xlsx') as writer:
#    shares_2030.to_excel(writer,sheet_name='tec_shares_in_2030')
#    shares_2050.to_excel(writer,sheet_name='tec_shares_in_2050')
#    add_share.to_excel(writer,sheet_name='add_shares')
#    stock_comp.to_excel(writer,sheet_name='total_stock')
#    full_BEV_yr.to_excel(writer,sheet_name='1st_year_full_BEV')
#    scenario_totcs.to_excel(writer,sheet_name='totc_opt')
#    scenariot_tots.to_excel(writer,sheet_name='totc_opt1')
##    default_fleet.__getattribute__('veh_partab').to_excel(writer,sheet_name='orig_vehpartab')


"""Visualization code assumes that sensitivity analysis has just been run """
"""Prepwork for baseline comparisons"""
#ff=default_fleet.__getattribute__('veh_stck')
#dd =default_fleet.__getattribute__('stock_comp')


for i,df in shares_2030.groupby('tec'):
    if i=='BEV':
        shares_2030_BEV=df
        d=default_fleet.__getattribute__('shares_2030')
        shares_2030_BEV['baseline']=d[d.index.get_level_values('tec').isin(['BEV'])]
for i,df in shares_2050.groupby('tec'):
    if i=='BEV':
        shares_2050_BEV = df
        e=default_fleet.__getattribute__('shares_2050')
        shares_2050_BEV['baseline']=d[d.index.get_level_values('tec').isin(['BEV'])]


keys = shares_2030_BEV.columns.to_list()
values = ['avg veh lifetime',
 'std dev veh lifetime',
 'market increase constraint',
 'occupancy rate',
 'passenger demand',
 'veh stck tot',
 'EOL emissions, BEV, u',
 'EOL emissions, BEV, A',
 'EOL emissions, BEV, B',
 'EOL emissions, BEV, r',
 'EOL emissions, ICE, u',
 'EOL emissions, ICE, A',
 'EOL emissions, ICE, B',
 'EOL emissions, ICE, r',
 'Operational energy, BEV, u',
 'Operational energy, BEV, A',
 'Operational energy, BEV, B',
 'Operational energy, BEV, r',
 'Operational energy, ICE, u',
 'Operational energy, ICE, A',
 'Operational energy, ICE, B',
 'Operational energy, ICE, r',
 'Production emissions, BEV, u',
 'Production emissions, BEV, A',
 'Production emissions, BEV, B',
 'Production emissions, BEV, r',
 'Production emissions, ICE, u',
 'Production emissions, ICE, A',
 'Production emissions, ICE, B',
 'Production emissions, ICE, r',
 'Production energy req., BEV, u',
 'Production energy req., BEV, A',
 'Production energy req., BEV, B',
 'Production energy req., BEV, r',
 'Production energy req., ICE, u',
 'Production energy req., ICE, A',
 'Production energy req., ICE, B',
 'Production energy req., ICE, r',
 'Electricity CO2 intensity, CINT, u',
 'Electricity CO2 intensity, CINT, A',
 'Electricity CO2 intensity, CINT, B',
 'Electricity CO2 intensity, CINT, r',
 'Fossil fuel CO2 intensity, CINT, u',
 'Fossil fuel CO2 intensity, CINT, A',
 'Fossil fuel CO2 intensity, CINT, B',
 'Fossil fuel CO2 intensity, CINT, r']

scen_names = dict(zip(keys,values))

# Fix names of scenarios in the dataframes

# First, fix mislabelled columns (this should be temporary)
stock_comp.rename(columns={'level':'avg_age_sensitivity','levelc':'EOLT_CINT_BEV_u-term'},inplace=True)

shares_2030_BEV.rename(columns=scen_names,inplace=True)
shares_2050_BEV.rename(columns=scen_names,inplace=True)

stock_comp.rename(columns=scen_names,inplace=True)
scenario_totcs.rename(index=scen_names,inplace=True)


pp = PdfPages('sensitivity_vis_'+now+'.pdf')
fig=plt.subplots(1,1,figsize=(9,18))

run_id_list2 = full_BEV_yr.index.values
ax = plt.scatter(full_BEV_yr,run_id_list2)
ax.axes.set_xbound(lower=2020,upper=2050)
pp.savefig(bbox_inches='tight')

# Make custom colormap for vehicle shares, by scenarios
#colors1 = plt.cm.Greys(np.linspace(0., 1, 6))
#colors2 = plt.cm.BuGn(np.linspace(0, 1, 8))
#colors3 = plt.cm.BuPu(np.linspace(0., 1, 8))
#colors4 = plt.cm.YlGn_r(np.linspace(0, 1, 8))
#colors5 = plt.cm.YlOrBr(np.linspace(0., 1, 8))
#colors6 = plt.cm.Reds(np.linspace(0, 1, 8))

colors1 = plt.cm.Greys(np.linspace(0., 1, 6))
colors2 = plt.cm.Red(np.linspace(0, 1, 8))
colors3 = plt.cm.BuPu(np.linspace(0., 1, 8))
colors4 = plt.cm.YlGn_r(np.linspace(0, 1, 8))
colors5 = plt.cm.YlOrBr(np.linspace(0., 1, 8))
colors6 = plt.cm.Reds(np.linspace(0, 1, 8))
colors = np.vstack((colors1, colors2,colors3,colors4,colors5,colors6))

scen_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
#Make custom colormap for vehicle shares, by sinusoidal terms
colors1 = plt.cm.Greys(np.linspace(0., 1, 6))
colors2 = plt.cm.BuGn(np.linspace(0, 1, 8))
colors3 = plt.cm.BuPu(np.linspace(0., 1, 8))
colors4 = plt.cm.YlGn_r(np.linspace(0, 1, 8))
colors5 = plt.cm.YlOrBr(np.linspace(0., 1, 8))
colors6 = plt.cm.Reds(np.linspace(0, 1, 8))

colors = np.vstack((colors1, colors2,colors3,colors4,colors5,colors6))

scen_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)


#colors1 = plt.cm.Oranges(np.linspace(0., 1, 6))
#colors2 = plt.cm.Blues(np.linspace(0, 1, 6))

#ax=plt.bar(x=shares_2030_BEV.columns.values,stacked=True)
# Normalize market share datafraes for easier visualization
shares_2030_plot = shares_2030_BEV.divide(shares_2030_BEV.loc[:,'baseline'],axis='index')
shares_2050_plot = shares_2050_BEV.divide(shares_2050_BEV.loc[:,'baseline'],axis='index')
#stockcomp_plot = stock_comp.sum(level=[0,1]).T
stockcomp_plot = stockcomp_plot.divide(stockcomp_plot.loc['Baseline'].sum(),level=0)
        
ax = shares_2030_plot.plot(kind='bar',cmap='Dark2', figsize=(12,12),width=0.95)
## OBS: FIX THIS TO BE RELATIVE TO MINIMUM
ax.axes.set_ybound(lower=0.95,upper=1.05)
fix_legend(ax,'2030 market shares')
pp.savefig(bbox_inches='tight')

ax = shares_2050_plot.plot(kind='bar',cmap='Dark2',figsize=(12,12))
fix_legend(ax,'2050 market shares')
pp.savefig(bbox_inches='tight')

# Make custom colormap for stock composition
colors1 = plt.cm.Oranges(np.linspace(0., 1, 6))
colors2 = plt.cm.Blues(np.linspace(0, 1, 6))

# combine them and build a new colormap
colors = np.vstack((colors1, colors2))

mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

stockcomp_plot.plot(kind='bar',stacked=True,width=1,figsize=(18,8),cmap=mymap)
#stock_comp.sum(level=[0,1]).T.plot(kind='bar',stacked=True,width=1,figsize=(18,8),cmap=mymap)
pp.savefig(bbox_inches='tight')

fig=plt.subplots(1,1,figsize=(8,18))
plt.scatter(x=scenario_totcs.iloc[:,2],y=scenario_totcs.index.values)
#pp.savefig(bbox_inches='tight')

            