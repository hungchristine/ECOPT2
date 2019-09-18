# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:26:23 2019

@author: chrishun
"""
import logging
import sys

import fleet_model
#import sigmoid
#import test_gams
from itertools import product
from datetime import datetime
import yaml
import pandas as pd
"""
#fleet.read_all_sets("C:\\Users\\chrishun\\Box Sync5\\YSSP_temp\\EVD4EUR_input.gdx")
#fleet.add_to_GAMS()

#fleet._read_all_final_parameters("C:\\Users\\chrishun\\Box Sync\\YSSP_temp\\EVD4EUR_input.gdx")
"""
#a=[1/6 for i in range(6)]
#b=[1/6 for i in range(6)]
#c=0.3
#
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

def run_experiment():
    # Load parameter values from YAML
    with open('temp_input.yaml', 'r') as stream:
        try:
            params = yaml.safe_load(stream)
            print('finished')
        except yaml.YAMLError as exc:
            print(exc)

    # Dictionary for logging
    info = {}

    # Explicit list of parameters
    param_names = ['veh_stck_int_seg','tec_add_gradient','seg_batt_caps','B_term_prod','B_term_oper_EOL','r_term_factors','u_term_factors','pkm_scenario']#,'seg_batt_caps']

    # Run experiments
    id_and_value = [params[p].items() for p in param_names]
    # NB could also change the names here
    
    shares_2030 = None
    run_id_list = []
    full_BEV_yr_list = []
    
    for i, run_params in enumerate(product(*id_and_value)):
        # Same order as in param_names. Each is a tuple of (id, values)
#        veh_seg_shr, tec_add_gradient, seg_batt_caps = run_params
        veh_stck_int_seg, tec_add_gradient, seg_batt_caps, B_term_prod, B_term_oper_EOL, r_term_factors, u_term_factors, pkm_scenario = run_params

        # Make run ID
        now = datetime.now().isoformat(timespec='minutes').replace(':','_')
        run_id = f'run_{tec_add_gradient[0]}_{seg_batt_caps[0]}_{pkm_scenario[1]}' #'_{seg_batt_caps[0]}'
        run_tag = run_id+now
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
                                    pkm_scenario = pkm_scenario[1])#,
#                                    growth_constraint = growth_constraint[1])
        
        fm.run_GAMS(run_tag)

        exceptions = fm.db.get_database_dvs()
        if len(exceptions) > 1:
            print(exceptions[0].symbol.name)
            dunno = exceptions[0].symbol_dvs

            dunno2 = exceptions[0].symbol
            print(fm.db.number_symbols)

        fm.vis_GAMS(run_id)
        
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
                'pkm_scenario': pkm_scenario
            },
            'output': {
#                'totc': 42,   # life, the universe, and everything…
                 'first year of 100% BEV market share': fm.full_BEV_year,
                 'totc': fm.totc,
                 'BEV shares in 2030': fm.shares_2030.loc[:,'BEV'].to_string()
#                 'totc in optimization period':fm.totc_opt # collect these from all runs into a dataframe...ditto with shares of BEV/ICE
            }
        }
        
            # Save pertinent info to compare across scenarios in dataframe
        fm.shares_2030.name = run_id
        if shares_2030 is None:
            shares_2030 = pd.DataFrame(fm.shares_2030)
        else:
            shares_2030[run_id] = fm.shares_2030
            
        full_BEV_yr_list.append(fm.full_BEV_year)

        # Display the info for this run
        log.info(repr(info[run_tag]))

    # Write log to file
    now = datetime.now().isoformat(timespec='seconds').replace(':','_')
    with open(f'output_{now}.yaml', 'w') as f:
        yaml.safe_dump(info, f)
    
    return fm, shares_2030, run_id_list, full_BEV_yr_list

shares_2030 = pd.DataFrame()
full_BEV_years = pd.DataFrame()

fleet,shares_2030,run_id_list, full_BEV_yr_list = run_experiment()
shares_2030.plot(kind='bar')
full_BEV_yr = pd.DataFrame(full_BEV_yr_list,index = run_id_list)
full_BEV_yr.plot()

#        bounds = ['high','baseline','low']

#    for element in itertools.product(experiment_list,bounds):
#        fleet = fleet_model.FleetModel()

#    for experiment in experiment_list:
#        keeper
#        for i, bound in enumerate(bounds):
#            keeper[i] = + bound
