# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:26:23 2019

@author: chrishun
  """
import logging
import sys

import fleet_model
import gams_runner
from fleet_model_init import SetsClass, ParametersClass
import visualization as vis

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
import gzip
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

# Housekeeping: set up experiment options
demo = True
export = False

# unit test consists of static LCA factors
# (straight line in lieu of logistic function)
if demo:
    yaml_name = 'GAMS_input_demo'
else:
    #yaml_name = 'unit_test'#.yaml'
    yaml_name = 'GAMS_input'#.yaml'
    # yaml_name = 'GAMS_input_demo'

# Log to screen
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

# Make output YAML less ugly, see https://stackoverflow.com/a/30682604
yaml.SafeDumper.ignore_aliases = lambda *args: True

# Make timestamp and directory for scenario run results for output files
now = datetime.now().isoformat(timespec='minutes').replace(':','_')

if 'unit_test' in yaml_name:
    fp = os.path.join(os.path.curdir, 'output', 'unit_test_' + now)
else:
    fp = os.path.join(os.path.curdir, 'output','Run_'+now)

input_file = os.path.join(os.path.curdir, yaml_name + '.yaml')


try:
    os.mkdir(fp)
    # os.chdir(fp)
except:
    print('\n *****************************************')
    print("cannot make folder!")

def run_experiment():
    """
    Parse all experiment parameters and run model.

    Read YAML for key parameter values, create a GAMSRunner object,
    loop through each experiment (Cartesian product of all
    parameter combinations)

    """
    # Load parameter values for each experiment from YAML
    # r'C:\Users\chrishun\Box Sync\YSSP_temp\temp_input.yaml'
    # r'C:\Users\chrishun\Box Sync\YSSP_temp\temp_input_presubmission.yaml'
#    with open(r'C:\Users\chrishun\Box Sync\YSSP_temp\GAMS_input.yaml', 'r') as stream:

    with open(input_file, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
            print('finished reading parameter values')
        except yaml.YAMLError as exc:
            print('\n *****************************************')
            print(exc)

    params_dict = {}  # dict with parameter names key values, and dict of experiments as values
    for key, item in params.items():
        params_dict[key] = item

    # Dictionary for logging
    info = {}

    # Calculate total number of runs, for progress updates
    count = 1
    for x in params:
        temp = len(params[x])
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
    gams_run = gams_runner.GAMSRunner(fp)

    all_exp_list = []  # list of all exp_dicts
    exp_dict = {}  # dict containing current experiment parameters
    param_vals = [params_dict[p].items() for p in params_dict.keys()]  # list containing all parameter experiment values (for Cartesian product)
    exp_id_list = []  # list of experiment IDs

    # unpack/create all experiments as Cartesian product of all parameter options
    now = datetime.now().isoformat(timespec='minutes').replace(':', '_')  # timestamp for run ID
    for i, experiment in enumerate(product(*param_vals)):
        id_string = "run"

        for key, exp in zip(params_dict.keys(), experiment):
            # build dict describing each experiment - {parameter: experiment value}
            exp_dict[key] = exp[1]
            id_string = id_string + '_' + exp[0]  # make run ID
        all_exp_list.append(exp_dict.copy())
        exp_id_list.append(id_string + '_' + now)

    # start iterating and running experiments
    for i, experiment in enumerate(all_exp_list):
        print('Starting run ' + str(i+1) + ' of ' + str(count) + '\n\n')
        log.info(f'Starting run {exp_id_list[i]}')
        run_tag = exp_id_list[i]
        run_id = f'{exp_id_list[i]}'
        run_id_list.append(run_id)

        if demo:
            sets = SetsClass.from_file(os.path.join(os.path.curdir, 'data', 'sets_demo.xlsx'))
            params = ParametersClass.from_file(os.path.join(os.path.curdir, 'data', 'GAMS_input_demo.xlsx'), experiment=experiment)
        else:
            sets = SetsClass.from_file(os.path.join(os.path.curdir, 'data', 'sets.xlsx'))
            params = ParametersClass.from_file(os.path.join(os.path.curdir, 'data', 'GAMS_input_demo_test.xlsx'), experiment=experiment)

        fm = fleet_model.FleetModel(sets, params)

        """
        # need to pass in run ID tag for saving gdx/csv
        # instantiate FleetModel object
        # NB here, use explicit names to avoid any confusion """
        # sets = SetsClass.from_file(r'C:\Users\chrishun\Box Sync\YSSP_temp\Data\load_data\sets.xlsx')

        try:
            gams_run.run_GAMS(fm, run_tag, yaml_name, now)  # run the GAMS model
        except Exception:
            print('\n *****************************************')
            log.warning("Failed run")
            traceback.print_exc()
            # os.chdir('..')
            if not os.listdir(fp):  # check folder is empty before deleting
                log.warning(f'Deleting folder {fp}')
                os.rmdir(fp)
            # Force quit if single run has failed
            if count == 1:
                sys.exit()

        exceptions = gams_run.db.get_database_dvs()
        if len(exceptions) > 1:
            print(exceptions[0].symbol.name)
            print(fm.db.number_symbols)

        # Pickle the scenario fleet object
        pickle_fp = os.path.join(fp, run_tag + '.pkl')
        with gzip.GzipFile(pickle_fp, 'wb') as f:
            pickle.dump(fm, f)

        # Save log info
        info[run_tag] = {
            'params': experiment,
            'output': {
#                 'first year of 100% BEV market share': fm.full_BEV_year
                 'totc_opt': fm.totc_opt,
                 'solver status': gams_run.ss,
                 'model status': gams_run.ms
#                 'BEV shares in 2030': fm.shares_2030.loc[:,'BEV'].to_string(),
#                 'totc in optimization period':fm.totc_opt # collect these from all runs into a dataframe...ditto with shares of BEV/ICE
            }
        }

#        with open(fp+'\failed.txt','a+') as f:
#            f.write('Successful run. Next: visualization!')

        # convert dict values in experiment to list (drop keys)
        exp_params = {}
        for key, value in experiment.items():
            if isinstance(value, dict):
                tmp = [val for key, val in value.items()]
                exp_params[key] = tmp
            else:
                exp_params[key] = value

        # Run visualization
        try:
            fm.figure_calculations()  # run extra calculations for cross-experiment figures
            vis.vis_GAMS(fm, fp, run_id, experiment, export_png=False, export_pdf=False)
            # vis.vis_input(fm, fp, run_id, experiment, export_png=False, export_pdf=False, max_year=50, cropx=True, suppress_vis=False)
        except Exception:
            print('\n *****************************************')
            log.warning("Failed visualization, deleting folder")
            traceback.print_exc()
            # os.chdir('..')
            if (os.path.exists(fp)) and (not os.listdir(fp)):
                os.rmdir(fp)

        # Save pertinent info to compare across scenarios in dataframe
        fm.shares_2030.name = run_id
        fm.shares_2050.name = run_id
        fm.add_share.name = run_id
        fm.veh_stck.name = run_id

        if shares_2030 is None:
            shares_2030 = pd.DataFrame(fm.shares_2030.stack().stack(), columns=[run_id])
        else:
            shares_2030[run_id] = fm.shares_2030.stack().stack()

        if shares_2050 is None:
            shares_2050 = pd.DataFrame(fm.shares_2050.stack().stack(), columns=[run_id])
        else:
            shares_2050[run_id] = fm.shares_2050.stack().stack()

        if add_share is None:
            add_share = pd.DataFrame(fm.add_share.stack().stack(), columns=[run_id])
        else:
            add_share[run_id] = fm.add_share.stack().stack()

        if stock_comp is None:
            stock_comp = pd.DataFrame(fm.veh_stck.stack(), columns=[run_id])
        else:
            stock_comp[run_id] = fm.veh_stck.stack()

        full_BEV_yr_df.append(fm.full_BEV_year, ignore_index=True)
        totc_list.append(fm.totc_opt)

        # Display the info for this run
        log.info(repr(info[run_tag]))
        print('\n\n\n ********** End of run ' + str(i+1) + ' ************** \n\n\n')

    # Write log to file
    output_fp = os.path.join(fp, f'output_{now}.yaml')
    with open(output_fp, 'w') as f:
        yaml.safe_dump(info, f)

    # Return last fleet object for troubleshooting
    return fm, run_id_list, shares_2030, shares_2050, add_share, stock_comp, full_BEV_yr_df, totc_list  #, fleet_dict


""" Run the full experiment portfolio; also returns last instance of FleetModel object for troubleshooting"""
fm, run_id_list, shares_2030, shares_2050, add_share, stock_comp, full_BEV_yr_df, totc_list = run_experiment()

#with open(fp+'\failed.txt','a+') as f:
#    f.write('Successfully completed all runs!')

""" Perform calculations across the experiments"""
print('\n ********** Performing cross-experiment calculations ************** \n')
full_BEV_yr = pd.DataFrame(full_BEV_yr_df, index=run_id_list)

scenario_totcs = pd.DataFrame(totc_list, index=run_id_list)
scenario_totcs = pd.DataFrame(totc_list, index=run_id_list, columns=['totc_opt'])

# Load a "baseline" fleet and extract parameters for comparison
baseline_file = None
for i, exp in enumerate(run_id_list):
    new_list = [j for j in exp.split('_')]
    if len(set(new_list))==1:
        new_list[0] =='def'
        print(exp)
        print(os.path.abspath(os.path.curdir))
        baseline_file = 'run_' + run_id_list[i] + '.pkl'

if baseline_file is None:
    baseline_file = run_id_list[0] + '.pkl' # use first experiment performed as baseline experiment

with gzip.open(baseline_file, 'rb') as f:
    d = pickle.load(f)
    default_totc_opt = d.totc_opt
#with open('run_2019-09-22T14_50.pkl','rb') as f:
#    d=pickle.load(f)
#    default_totc_opt = d[0][run_id_list[0]].totc_opt

try:
    scenario_totcs['Abs. difference from totc_opt'] = default_totc_opt - scenario_totcs['totc_opt']
    scenario_totcs['%_change_in_totc_opt'] = (default_totc_opt - scenario_totcs['totc_opt'])/default_totc_opt
except:
    print('\n *****************************************')
    print("No comparison to default performed")

# Export potentially helpful output for analyzing across scenarios
if export:
    print('\n ********** Exporting cross-experiment results to pickle and Excel ************** \n')
    with open('overview_run_' + now + '.pkl', 'wb') as f:
        pickle.dump([shares_2030, shares_2050, add_share, stock_comp, full_BEV_yr_df, scenario_totcs], f)

    """with open('run_2019-09-22T14_50.pkl','rb') as f:
        d=pickle.load(f)
    d[0][run_id_list[0]].add_share"""
    """
    "with open(os.path.join(os.path.curdir, 'Run_2020-09-02T22_57', 'run_run_def_baseline_def_def_def_def_def_iTEM2-Base2020-09-02T22_57.pkl','rb') as f:
        fm=pickle.load(f)
    """
    print('Exporting to Excel')
    with pd.ExcelWriter('cumulative_scenario_output'+now+'.xlsx') as writer:
        shares_2030.to_excel(writer,sheet_name='tec_shares_in_2030')
        shares_2050.to_excel(writer,sheet_name='tec_shares_in_2050')
        add_share.to_excel(writer,sheet_name='add_shares')
        stock_comp.to_excel(writer,sheet_name='total_stock')
        full_BEV_yr.to_excel(writer,sheet_name='1st_year_full_BEV')
        scenario_totcs.to_excel(writer,sheet_name='totc')

# plotting
# rename = {baseline_file.split('.pkl')[0]: 'baseline'}
# shares_2030.rename(columns=rename, inplace=True)
# shares_2050.rename(columns=rename, inplace=True)
# add_share.rename(columns=rename, inplace=True)
# stock_comp.rename(columns=rename, inplace=True)
# full_BEV_yr.rename(index=rename, inplace=True)
# scenario_totcs.rename(index=rename, inplace=True)

try:
    full_BEV_yr.plot()
except Exception as e:
    print(f'Error plotting figure: {e}')
# except TypeError:
#     print('Regions do not reach 100% market share during model period')

# (scenario_totcs['%_change_in_totc_opt']*100).plot(kind='bar')
# import numpy as np
# (shares_2030.replace(0, np.nan).dropna(how='all', axis=0)).unstack(['reg', 'tec']).T.plot(kind='bar', stacked=True)


#os.remove(fp+'\failed.txt')


#        bounds = ['high','baseline','low']

#    for element in itertools.product(experiment_list,bounds):
#        fleet = fleet_model.FleetModel()

#    for experiment in experiment_list:
#        keeper
#        for i, bound in enumerate(bounds):
#            keeper[i] = + bound