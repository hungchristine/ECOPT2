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

import pickle
import os
import traceback

"""
#fleet.read_all_sets("C:\\Users\\chrishun\\Box Sync5\\YSSP_temp\\EVD4EUR_input.gdx")
#fleet.add_to_GAMS()

#fleet._read_all_final_parameters("C:\\Users\\chrishun\\Box Sync\\YSSP_temp\\EVD4EUR_input.gdx")
"""
#a=[1/6 for i in range(6)]
#b=[1/6 for i in range(6)]
#c=0.3
#.
#fleet = fleet_model.FleetModel(a,b,c)
#sigm = sigmoid.Sigmoid
#fun = getattr(sigmoid, f'make_{sigmoid_case}')
#
#values = fun()
##values = sigmoid.make_values(**sigmoid_case[1])
##values = sigmoid.make_values(A_batt_size=30, F_batt_size=100)
#asdf = getattr(fleet,'veh_partab')
#vals = asdf()
##asdf = getattr(sigmoid,'genlogfnc')([2000, 2010, 2020, 2030])
#print(asdf)
##fleet.run_GAMS('run_x')
##fleet.vis_GAMS('run_x')


# Log to screen
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

# Make output YAML less ugly, see https://stackoverflow.com/a/30682604
yaml.SafeDumper.ignore_aliases = lambda *args: True

# Make timestamp and directory for scenario run results for output files
now = datetime.now().isoformat(timespec='minutes').replace(':','_')
#yaml_name = 'unit_test'#.yaml'
yaml_name = 'GAMS_input'#.yaml'

if yaml_name == 'unit_test.yaml':
    fp = r'C:\Users\chrishun\Box Sync\YSSP_temp\visualization output\unit_test_'+now
    input_file = r'C:\Users\chrishun\Box Sync\YSSP_temp\unit_test.yaml'
else:
    fp = r'C:\Users\chrishun\Box Sync\YSSP_temp\visualization output\Run_'+now
    input_file = r'C:\Users\chrishun\Box Sync\YSSP_temp\GAMS_input.yaml'

try:
    os.mkdir(fp)
    os.chdir(fp)
#    with open(fp+'\failed.txt','w+') as f:
#        f.write('If I exist, run failed!')
except:
    print("cannot make folder!")
        
def run_experiment():
    """--- Read YAML for key parameter values for each experiment,
    create a GAMSRunner object, and loop through each experiment
    
    
    
    ---"""
    # Load parameter values for each experiment from YAML
    # r'C:\Users\chrishun\Box Sync\YSSP_temp\temp_input.yaml'
    # r'C:\Users\chrishun\Box Sync\YSSP_temp\temp_input_presubmission.yaml'
#    with open(r'C:\Users\chrishun\Box Sync\YSSP_temp\GAMS_input.yaml', 'r') as stream:
    
    with open(input_file, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
            print('finished reading parameter values')
        except yaml.YAMLError as exc:
            print(exc)

    # Dictionary for logging
    info = {}

    # Explicit list of parameters
    param_names = ['veh_stck_int_seg',
                   'tec_add_gradient',
                   'seg_batt_caps',
                   'B_term_prod',
                   'B_term_oper_EOL',
                   'r_term_factors',
                   'u_term_factors',
                   'eur_batt_share',
                   'pkm_scenario']#,'seg_batt_caps']

    # Run experiments
    id_and_value = [params[p].items() for p in param_names]

    # NB could also change the names here
    
    # Calculate total number of runs
    count = 1
    for x in params:
        temp = len(params[x])
        #temp += len(id_and_value[x])
        count = temp * count
        
    # Create data structures for comparing results from multiple runs
    shares_2030 = None
    shares_2050 = None
    add_share = None
    stock_comp = None
    fleet_dict = {}
    run_id_list = []
    totc_list = []
    full_BEV_yr_df = pd.DataFrame()
    
    # Create a GAMSRunner object to run the experiments
    gams_run = gams_runner.GAMSRunner()
    
    for i, run_params in enumerate(product(*id_and_value)):
        print('Starting run '+str(i+1)+' of '+str(count)+'\n\n')
#        veh_seg_shr, tec_add_gradient, seg_batt_caps = run_params
        # Unpack the defined run parameters
        veh_stck_int_seg, tec_add_gradient, seg_batt_caps, B_term_prod, B_term_oper_EOL, r_term_factors, u_term_factors, eur_batt_share, pkm_scenario = run_params

        # Make run ID
        now = datetime.now().isoformat(timespec='minutes').replace(':','_')
        run_id = f'run_{tec_add_gradient[0]}_{seg_batt_caps[0]}_{B_term_prod[0]}_{B_term_oper_EOL[0]}_{r_term_factors[0]}_{u_term_factors[0]}_{eur_batt_share[0]}_{pkm_scenario[1]}' #'_{seg_batt_caps[0]}'
        run_tag = run_id + now
        run_id_list.append(run_id)

        # run_id = f'run_{i}'  # alternate format

        # Run the appropriate function to make sigmoid parameters/values
#        sigm = sigmoid.Sigmoid
#        fun = getattr(sigm, 'batt_cap')(seg_batt_caps[1])
#        values = fun()
        
#        values = sigmoid.make_values(**sigmoid_case[1])
#        values = sigmoid.make_values(A_batt_size=30, F_batt_size=100)
                

        log.info(f'Starting run {run_id}')

        # need to pass in run ID tag for saving gdx/csv
        # NB here, use explicit names to avoid any confusion
        fm = fleet_model.FleetModel(veh_stck_int_seg = veh_stck_int_seg[1],
                                    tec_add_gradient = tec_add_gradient[1],
                                    seg_batt_caps = seg_batt_caps[1],
                                    B_term_prod = B_term_prod[1],
                                    B_term_oper_EOL = B_term_oper_EOL[1],
                                    r_term_factors = r_term_factors[1],
                                    u_term_factors = u_term_factors[1],
                                    eur_batt_share = eur_batt_share[1],
                                    pkm_scenario = pkm_scenario[1])#,
#                                    growth_constraint = growth_constraint[1]) 
        
        """fm.run_GAMS(run_tag)"""
        try:
            gams_run.run_GAMS(fm, run_tag, yaml_name) # run the GAMS model
        except Exception:
            print("failed run, deleting folder")
            traceback.print_exc()
            os.chdir('..')
            os.rmdir(fp)
            # Force quit if single run has failed
            if count == 1:
                sys.exit()
            
        exceptions = gams_run.db.get_database_dvs()
        if len(exceptions) > 1:
            print(exceptions[0].symbol.name)
            dunno = exceptions[0].symbol_dvs

            dunno2 = exceptions[0].symbol
            print(fm.db.number_symbols)

        # Pickle the scenario fleet object
#        os.chdir(fp)
        
        with open('run_'+run_tag+'.pkl','wb') as f:
            pickle.dump(fm,f)
        
        # Save log info
        info[run_tag] = {
            'params': {
                'veh_stck_int_seg ': veh_stck_int_seg,
                'tec_add_gradient ': tec_add_gradient,
                'seg_batt_caps ': seg_batt_caps,
                'B_term_prod ': B_term_prod,
                'B_term_oper_EOL ': B_term_oper_EOL,
                'r_term_factors ': r_term_factors,
                'u_term_factors ': u_term_factors,
                'pkm_scenario': pkm_scenario,
                'European share of global batter manuf. capacity': eur_batt_share
            },
            'output': {
#                'totc': 42,   # life, the universe, and everything…
#                 'first year of 100% BEV market share': fm.full_BEV_year,
                 'totc': fm.totc,
#                 'BEV shares in 2030': fm.shares_2030.loc[:,'BEV'].to_string(),
#                 'totc in optimization period':fm.totc_opt # collect these from all runs into a dataframe...ditto with shares of BEV/ICE
            }
        }
        
#        with open(fp+'\failed.txt','a+') as f:
#            f.write('Successful run. Next: visualization!')
            
        try:
            fm.figure_calculations()  # run extra calculations for cross-experiment figures
            fm.vis_GAMS(fp, run_id, info[run_tag]['params'], export_png=False)
        except Exception:
            print("failed visualization, deleting folder")
            traceback.print_exc()
            os.chdir('..')
            if os.path.exists(fp):
                os.rmdir(fp)        
        
#        os.rename(fp,fp+'_success_vis')
        
        # Save pertinent info to compare across scenarios in dataframe
        fm.shares_2030.name = run_id
        fm.shares_2050.name = run_id
        fm.add_share.name = run_id
        fm.veh_stck.name = run_id
        
        if shares_2030 is None:
            shares_2030 = pd.DataFrame(fm.shares_2030)
        else:
            shares_2030[run_id] = fm.shares_2030
        
        if shares_2050 is None:
            shares_2050 = pd.DataFrame(fm.shares_2050)
        else:
            shares_2050[run_id] = fm.shares_2050   
        
        if add_share is None:
            add_share = pd.DataFrame(fm.add_share.stack().stack())
        else:
            add_share[run_id] = fm.add_share.stack().stack()
        
        if stock_comp is None:
            stock_comp = pd.DataFrame(fm.veh_stck)
        else:
            stock_comp[run_id] = fm.veh_stck
        
        full_BEV_yr_df.append(fm.full_BEV_year, ignore_index=True)
        totc_list.append(fm.totc_opt)

        # Display the info for this run
        log.info(repr(info[run_tag]))
        print('\n\n\n ********** End of run ' + str(i+1)+ ' ************** \n\n\n')

    # Write log to file
    with open(f'output_{now}.yaml', 'w') as f:
        yaml.safe_dump(info, f)
    
    # Return last fleet object for troubleshooting
    return fm, run_id_list, shares_2030,  shares_2050, add_share, stock_comp, full_BEV_yr_df, totc_list#, fleet_dict


""" Run the full experiment portfolio; also returns last instance of FleetModel object for troubleshooting"""
fm, run_id_list, shares_2030, shares_2050, add_share, stock_comp, full_BEV_yr_df, totc_list = run_experiment()

#with open(fp+'\failed.txt','a+') as f:
#    f.write('Successfully completed all runs!')

""" Perform calculations across the experiments"""
full_BEV_yr = pd.DataFrame(full_BEV_yr_df, index = run_id_list)

scenario_totcs = pd.DataFrame(totc_list, index=run_id_list)
scenario_totcs = pd.DataFrame(totc_list, index = run_id_list, columns=['totc_opt'])
# Load a "baseline" fleet and extract parameters for comparison
#with open('run_2019-09-22T14_50.pkl','rb') as f:
#    d=pickle.load(f)
#    default_totc_opt = d[0][run_id_list[0]].totc_opt

try:
    scenario_totcs['Abs. difference from totc_opt'] = default_totc_opt - scenario_totcs['totc_opt']
    scenario_totcs['%_change_in_totc_opt'] = scenario_totcs['totc_opt']/default_totc_opt  
except:
    print("No comparison to default performed")

# Export potentially helpful output for analyzing across scenarios
with open('run_'+now+'.pkl', 'wb') as f:
    pickle.dump([shares_2030, shares_2050, add_share, stock_comp, full_BEV_yr_df, scenario_totcs], f)

"""with open('run_2019-09-22T14_50.pkl','rb') as f:
    d=pickle.load(f)
d[0][run_id_list[0]].add_share"""

#with pd.ExcelWriter('cumulative_scenario_output'+now+'.xlsx') as writer:
#    shares_2030.to_excel(writer,sheet_name='tec_shares_in_2030')
#    shares_2050.to_excel(writer,sheet_name='tec_shares_in_2050')
#    add_share.to_excel(writer,sheet_name='add_shares')
#    stock_comp.to_excel(writer,sheet_name='total_stock')
#    full_BEV_yr.to_excel(writer,sheet_name='1st_year_full_BEV')
#    scenario_totcs.to_excel(writer,sheet_name='totc')
   
#full_BEV_yr.plot()

#os.remove(fp+'\failed.txt')


#        bounds = ['high','baseline','low']

#    for element in itertools.product(experiment_list,bounds):
#        fleet = fleet_model.FleetModel()

#    for experiment in experiment_list:
#        keeper
#        for i, bound in enumerate(bounds):
#            keeper[i] = + bound
