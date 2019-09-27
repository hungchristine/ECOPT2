# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:26:23 2019

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


import copy
import pickle
import os


# Log to screen
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

# Make output YAML less ugly, see https://stackoverflow.com/a/30682604
yaml.SafeDumper.ignore_aliases = lambda *args: True

# Make timestamp and directory for scenario run results for output files
#now = datetime.now().isoformat(timespec='minutes').replace(':','_')
run_name = r'Run_2019-09-27T19_37'
fp = r"C:\Users\chrishun\Box Sync\YSSP_temp\visualization output\Run_2019-09-27T19_37"
save_dir = fp+r'\Sensitivity_analysis'
try:
    os.chdir(fp)
    os.mkdir(save_dir)
except:
    print("cannot find folder!")
        
# 'fleet_vkm', 'veh_oper_dist',  'seg_batt_caps', 'A_terms_raw', 'prod_df', 'batt_list', 'temp_prod_df', 'temp_oper_df', 'temp_df', 'b_prod', 'temp_a', 'b_oper', 'A', 'B', 'B_prod', 'B_oper',
params = ['tec_add_gradient', 'occupancy_rate','passenger_demand','veh_stck_tot','avg_age','std_dev_age']
complex_params = ['enr_partab', 'veh_partab']
veheq = ['EOLT_CINT', 'OPER_EINT', 'PROD_CINT_CSNT', 'PROD_EINT']
enr = ['ELC','FOS']
function_terms = ['A','B','r','u']
now = datetime.now().isoformat(timespec='minutes').replace(':','_')
gams_run = gams_runner.GAMSRunner()

def complex_parameters(equation_list):
    i=0
    for cparam in complex_params:
        if cparam == 'enr_partab':
            tec = ['CINT','CINT']
            equation_list = enr
        elif cparam == 'veh_partab':
            equation_list = veheq
            tec = ['BEV','ICE']
        for eq,tec,term in product(equation_list,tec,function_terms):
            run_id = f'{eq}_{term}_term'
            run_id_list.append(run_id)
            log.info(f'Starting run {run_id}')
            print('Starting run '+str(i)+' of '+str(len(complex_params)*len(equation_list)*len(function_terms))+'\n\n')
            new_fleet = run_sensitivity_analysis(cparam,default_fleet,run_id,tec,eq,term)
            i+=1
                    
            if isinstance(default_fleet.__getattribute__(cparam),pd.DataFrame) or isinstance(default_fleet.__getattribute__(cparam),pd.Series):
                def_param_value = default_fleet.__getattribute__(cparam).to_string()
                new_param_value = new_fleet.__getattribute__(cparam).to_string()
            else:
                def_param_value = default_fleet.__getattribute__(cparam)
                new_param_value = new_fleet.__getattribute__(cparam)
         
            info[run_id] = {
                    'parameter value':{
                            'old value ': def_param_value,
                            'new value ': new_param_value
                    },
                    'output': {
                         'first year of 100% BEV market share': new_fleet.full_BEV_year,
                         'totc': new_fleet.totc,
                         'totc_opt':new_fleet.totc_opt,
                         'BEV shares in 2030': new_fleet.shares_2030.loc[:,'BEV'].to_string()
                    }
                }
            
    print('\n\n\n ********** End of run ' + str(i)+ ' ************** \n\n\n')

    # Write log to file
    with open(f'output_{now}.yaml', 'w') as f:
        yaml.safe_dump(['original totc_opt: ',default_totc_opt],f)
        yaml.safe_dump(info, f)

def run_sensitivity_analysis(param, default_fleet,run_id,tec=None, eq=None, term=None):
    fleet = copy.copy(default_fleet)
    if eq==None and term==None:
        fleet.__setattr__(param, fleet.__getattribute__(param)*0.99)
    else:
        temp_attr = fleet.__getattribute__(param)
#        fleet.veh_partab[veheq,tec][term]=original_fleet.loc[veheq,tec][term]*0.99
        temp_attr[eq,tec][term] = default_fleet.__getattribute__(param).loc[eq,tec][term]*0.99
        fleet.__setattr__(param, temp_attr)
    gams_run.run_GAMS(fleet,run_id)
    
    exceptions = gams_run.db.get_database_dvs()
    if len(exceptions) > 1:
        print(exceptions[0].symbol.name)
        dunno = exceptions[0].symbol_dvs

        dunno2 = exceptions[0].symbol
        print(fleet.db.number_symbols)    
        
#      Pickle the new fleet object
    with open(run_id+'.pkl','wb') as f:
        pickle.dump(fleet,f)
        
#    fleet.vis_GAMS(save_di1r,run_id)
    return fleet

def load_default():
    with open('run_run_def_baseline_def_def_def_def_iTEM2-Base2019-09-27T19_37.pkl','rb') as f:
        d = pickle.load(f)
        return d
    
default_fleet = load_default()
default_totc_opt = default_fleet.__getattribute__('totc_opt')
os.chdir(save_dir)
i=0

# Dictionary for logging
info = {}

shares_2030 = None
shares_2050 = None
add_share = None
stock_comp = None
fleet_dict = {}
run_id_list = []
totc_list = []
full_BEV_yr_list = []

#for param in params:
#    run_id = f'{param}_sensitivity'
#    run_id_list.append(run_id)
#    log.info(f'Starting run {run_id}')
#    
#    i+=1
#    
#    print('Starting run '+str(i)+' of '+str(len(params))+'\n\n')
#    print(run_id)
#    print(default_fleet.__getattribute__(param))
#    print('old totc')
#    print(default_fleet.totc_opt)
#    new_fleet = run_sensitivity_analysis(param, default_fleet,run_id)
##    gams_run.run_GAMS(new_fleet,run_id)
#    print(new_fleet.__getattribute__(param))
#    print('new totc \n')
#    print(new_fleet.totc_opt)
#    ## pickle new_fleet
#    
#    if isinstance(default_fleet.__getattribute__(param),pd.DataFrame) or isinstance(default_fleet.__getattribute__(param),pd.Series):
#        def_param_value = default_fleet.__getattribute__(param).to_string()
#        new_param_value = new_fleet.__getattribute__(param).to_string()
#    else:
#        def_param_value = default_fleet.__getattribute__(param)
#        new_param_value = new_fleet.__getattribute__(param)
#         
#    info[run_id] = {
#            'parameter value':{
#                    'old value ': def_param_value,
#                    'new value ': new_param_value
#            },
#            'output': {
#                 'first year of 100% BEV market share': new_fleet.full_BEV_year,
#                 'totc': new_fleet.totc,
#                 'totc_opt':new_fleet.totc_opt,
#                 'BEV shares in 2030': new_fleet.shares_2030.loc[:,'BEV'].to_string()
#            }
#        }
##        'old value ': default_fleet.__getattribute__(param)
##        'new value ': new_fleet.__getattribute__(param)
#    # Save pertinent info to compare across scenarios in dataframe
#    new_fleet.shares_2030.name = run_id
#    new_fleet.shares_2050.name = run_id
#    new_fleet.add_share.name = run_id
#    new_fleet.veh_stck.name = run_id
#    
#    if shares_2030 is None:
#        shares_2030 = pd.DataFrame(new_fleet.shares_2030)
#    else:
#        shares_2030[run_id] = new_fleet.shares_2030
#    
#    if shares_2050 is None:
#        shares_2050 = pd.DataFrame(new_fleet.shares_2050)
#    else:
#        shares_2050[run_id] = new_fleet.shares_2050   
#    
#    if add_share is None:
#        add_share = pd.DataFrame(new_fleet.add_share.stack().stack())
#    else:
#        add_share[run_id] = new_fleet.add_share.stack().stack()
#    
#    if stock_comp is None:
#        stock_comp = pd.DataFrame(new_fleet.veh_stck)
#    else:
#        stock_comp[run_id] = new_fleet.veh_stck
#    
#    try:
#        int(new_fleet.full_BEV_year)
#    except:
#        pass
#    
#    full_BEV_yr_list.append(new_fleet.full_BEV_year)
#    totc_list.append(new_fleet.totc_opt)
#
#    # Display the info for this run
#    log.info(repr(info[run_id]))
#    
#    print('\n\n\n ********** End of run ' + str(i)+ ' ************** \n\n\n')


complex_parameters(enr)

#
#full_BEV_yr = pd.DataFrame(full_BEV_yr_list,index = run_id_list)
#scenario_totcs = pd.DataFrame(totc_list, index = run_id_list, columns=['totc_opt'])
#scenario_totcs['Abs. difference from totc_opt'] = default_totc_opt - scenario_totcs['totc_opt']
#scenario_totcs['%_change_in_totc_opt'] = scenario_totcs['totc_opt']/default_totc_opt  
#
## Export potentially helpful output for analyzing across scenarios
#with open('sensitivity_analysis_'+now+'.pkl', 'wb') as f:
#    pickle.dump([shares_2030, shares_2050, add_share, stock_comp, full_BEV_yr, scenario_totcs], f)
#
#with pd.ExcelWriter('cumulative_scenario_output'+now+'.xlsx') as writer:
#    shares_2030.to_excel(writer,sheet_name='tec_shares_in_2030')
#    shares_2050.to_excel(writer,sheet_name='tec_shares_in_2050')
#    add_share.to_excel(writer,sheet_name='add_shares')
#    stock_comp.to_excel(writer,sheet_name='total_stock')
#    full_BEV_yr.to_excel(writer,sheet_name='1st_year_full_BEV')
#    scenario_totcs.to_excel(writer,sheet_name='totc_opt')
#
#pp = PdfPages('sensitivity_vis_'+now+'.pdf')
#ax = plt.scatter(full_BEV_yr,run_id_list)
#ax.axes.set_xbound(lower=2020,upper=2050)
#pp.savefig(bbox_inches='tight')
#
#
#shares_2030.groupby('tec').plot(kind='bar',cmap='viridis')
#pp.savefig(bbox_inches='tight')
#shares_2050.groupby('tec').plot(kind='bar',cmap='viridis')
#pp.savefig(bbox_inches='tight')
#
#"""with open('run_2019-09-22T14_50.pkl','rb') as f:
#    d=pickle.load(f)
#d[0][run_id_list[0]].add_share"""
#
#pp.close()