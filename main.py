"""
Created on Thu Apr 25 17:26:23 2019

@author: chrishun
  """
import logging
import sys
from importlib import reload
reload(logging)  # override built-in logging config
from enum import Enum

from itertools import product
from datetime import datetime
import yaml
import pandas as pd

import pickle
import gzip
import os
import traceback

import fleet_model
import gams_runner
from fleet_model_init import SetsClass, ParametersClass
import visualization as vis


# Make timestamp and directory for scenario run results for output files
now = datetime.now().isoformat(timespec='minutes').replace(':', '_')  # timestamp for run IDs


# Housekeeping: set up experiment options of filepaths
class Experiment(Enum):
    """ Define experiment being run and therefore YAML input file name. """

    DEMO = 'GAMS_input_demo'
    UNIT = 'unit_test' # unit test consists of static LCA factors # (straight line in lieu of logistic function)
    NORMAL = 'GAMS_input'
    WORLD = 'GAMS_input_world'

experiment_type = Experiment.DEMO #NORMAL # must be one of 'DEMO', 'UNIT', 'NORMAL'
export_png = True  # visualization file format
export_pdf = True  # visualization file format
export = False  # whether to export cross-scenario results
visualize_input = False # visualize input factors for debugging

yaml_name = experiment_type.value

if experiment_type == Experiment.UNIT:
    fp = os.path.join(os.path.curdir, 'output', 'Unit_test_' + now)
elif experiment_type == Experiment.DEMO:
    fp = os.path.join(os.path.curdir, 'output','Demo_'+now)
else:
    fp = os.path.join(os.path.curdir, 'output','Run_'+now)

input_file = os.path.join(os.path.curdir, yaml_name + '.yaml')
log_fp = os.path.join(fp, now+'_log.log')
formatter = logging.Formatter('%(levelname)s [%(name)s] - %(message)s')
log = logging.getLogger()
log.setLevel(logging.INFO)

try:
    if os.path.isdir('output'):
        os.mkdir(fp)
    else:
        os.mkdir('output')
        os.mkdir(fp)
    file_log = logging.FileHandler(log_fp)
except:
    print('\n *****************************************')
    log.error(f'Could not make output folder {fp}!')

# Set up logging - both stream to console and to log file

file_log.setLevel(logging.INFO)
file_log.setFormatter(formatter)
stream_log = logging.StreamHandler(sys.stdout)
stream_log.setLevel(logging.INFO)
stream_log.setFormatter(formatter)
log.addHandler(file_log)
log.addHandler(stream_log)

# Make output YAML less ugly, see https://stackoverflow.com/a/30682604
yaml.SafeDumper.ignore_aliases = lambda *args: True

def run_experiment():
    """
    Parse all experiment parameters and run model.

    Read YAML for key parameter values, create a GAMSRunner object,
    loop through each experiment (Cartesian product of all
    parameter combinations)

    """

    with open(input_file, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
            log.info(f'Successfully read experiment parameter values from {input_file} file')
        except yaml.YAMLError as exc:
            print('\n *****************************************')
            log.error(f'----- Could not read parameter values from YAML file. {exc}')


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
    run_id_list = []

    # Create a GAMSRunner object to run the experiments
    gams_run = gams_runner.GAMSRunner(fp)
    all_exp_list = []  # list of all exp_dicts
    exp_dict = {}  # dict containing current experiment parameters
    param_vals = [params_dict[p].items() for p in params_dict.keys()]  # list containing all parameter experiment values (for Cartesian product)
    exp_id_list = []  # list of experiment IDs

    # DataFrames to store key cross-experiment results
    shares_2030 = pd.DataFrame()
    shares_2050 = pd.DataFrame()
    add_share = pd.DataFrame()
    stock_comp = pd.DataFrame()
    full_newtec_yr_df = pd.DataFrame()
    totc_df = pd.DataFrame()

    # unpack/create all experiments as Cartesian product of all parameter options
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
        log.info(f'Starting run {exp_id_list[i]}, {i+1} of {count}')
        run_tag = exp_id_list[i]
        run_id = f'{exp_id_list[i]}'
        run_id_list.append(run_id)

        if experiment_type == Experiment.DEMO:
            sets = SetsClass.from_file(os.path.join(os.path.curdir, 'data', 'sets_demo.xlsx'))
            params = ParametersClass.from_file(os.path.join(os.path.curdir, 'data', 'GAMS_input_demo.xlsx'), experiment=experiment)
        elif experiment_type == Experiment.WORLD:
            sets = SetsClass.from_file(os.path.join(os.path.curdir, 'data', 'sets_world.xlsx'))
            params = ParametersClass.from_file(os.path.join(os.path.curdir, 'data', 'GAMS_input_world.xlsx'), experiment=experiment)
        else:
            sets = SetsClass.from_file(os.path.join(os.path.curdir, 'data', 'sets.xlsx'))
            params = ParametersClass.from_file(os.path.join(os.path.curdir, 'data', 'GAMS_input.xlsx'), experiment=experiment)

        fm = fleet_model.FleetModel(sets, params)

        try:
            gams_run.run_GAMS(fm, run_tag, yaml_name, now)  # run the GAMS model
        except Exception:
            print('\n *****************************************')
            log.warning("Failed run")
            traceback.print_exc()
            if not os.listdir(fp):  # check folder is empty before deleting
                log.warning(f'Deleting folder {fp}')
                os.rmdir(fp)
            # Force quit if single run has failed
            if count == 1:
                sys.exit()

        exceptions = gams_run.db.get_database_dvs()
        if len(exceptions) > 0:
            print("GAMS errors as follows:")
            print(exceptions[0].symbol.name)
            print(fm.db.number_symbols)

        # Pickle the scenario fleet object
        pickle_fp = os.path.join(fp, run_tag + '.pkl')
        with gzip.GzipFile(pickle_fp, 'wb') as f:
            pickle.dump(fm, f)

        # convert dict values in experiment to list (drop keys)
        exp_params = {}
        for key, value in experiment.items():
            if isinstance(value, dict):
                tmp = [val for key, val in value.items()]
                exp_params[key] = tmp
            else:
                exp_params[key] = value

        # Run visualization if GAMS ran without problems
        if len(exceptions) == 0:
            try:
                fm.figure_calculations()  # run extra calculations for cross-experiment figures
                fig_fp = os.path.join(fp, f'exp_{i}_figs')
                os.mkdir(fig_fp)
                log.info("Starting visualization of results")
                vis.vis_GAMS(fm, fig_fp, run_id, experiment, export_png=export_png, export_pdf=export_pdf)
                if visualize_input:
                    log.info("Starting visualization of input parameters")
                    vis.vis_input(fm, fp, run_id, experiment, export_png=export_png, export_pdf=export_pdf, max_year=2050, cropx=True, suppress_vis=False)
            except Exception:
                print('\n *****************************************')
                log.error("Failed visualization, deleting folder")
                traceback.print_exc()
                if (os.path.exists(fp)) and (not os.listdir(fp)):
                    os.rmdir(fp)

        # Save log info
        try:
            info[run_tag] = {
                'input_params': experiment,
                'output': {
                     'tot_impacts_opt': fm.tot_impacts_opt,
                     'solver status': gams_run.ss,
                     'model status': gams_run.ms,
                     'first year of 100% newtec market share': fm.full_newtec_year.to_dict(),
                     'newtec shares in 2030': fm.shares_2030.to_dict(),
                     'totc in optimization period':fm.tot_impacts_opt # collect these from all runs into a dataframe...ditto with shares of BEV/ICE
                }
            }

            # Save pertinent info to compare across scenarios in dataframe
            fm.shares_2030.name = run_id
            fm.shares_2050.name = run_id
            fm.add_share.name = run_id
            fm.tot_stock.name = run_id

            totc_df.loc['tot_impacts_opt', run_id] = fm.tot_impacts_opt
            stock_comp[run_id] = fm.tot_stock.stack() # fleet stock composition
            shares_2030[run_id] = fm.shares_2030.stack().stack() # market shares in 2030
            shares_2050[run_id] = fm.shares_2050.stack().stack()  # market shares in 2050
            add_share[run_id] = fm.add_share.stack().stack()  # market shares
            full_newtec_yr_df[run_id] = fm.full_newtec_year  # first year of full newtec penetration

            # Display the info for this run
            log.info('\n'+repr(info[run_tag])+'\n')
            log.info(f'End of run {str(i+1)}')
            print('\n\n\n ********** End of run ' + str(i+1) + ' ************** \n\n\n')
        except AttributeError as e:
            log.error(f'Could not export run info {e}')

    # Write log to file
    output_fp = os.path.join(fp, f'output_{now}.yaml')
    with open(output_fp, 'w') as f:
        yaml.safe_dump(info, f)

    # Return last fleet object for troubleshooting
    return fm, run_id_list, shares_2030, shares_2050, add_share, stock_comp, full_newtec_yr_df, totc_df


""" Run the full experiment portfolio; also returns last instance of FleetModel object for troubleshooting"""
fm, run_id_list, shares_2030, shares_2050, add_share, stock_comp, full_newtec_yr_df, totc_df = run_experiment()


""" Perform calculations across the experiments"""
if len(run_id_list) > 1:
    print('\n ********** Performing cross-experiment calculations ************** \n')

    # Load a "baseline" fleet and extract parameters for comparison
    baseline_file = None
    for i, exp in enumerate(run_id_list):
        new_list = [j for j in exp.split('_')]
        new_list = new_list[1:-2]  # drop 'run' and timestamp from run tag
        # find experiment of all "def"; by default, this run is the baseline
        if len(set(new_list))==1:
            new_list[0] =='def'
            log.info(f'Using {exp} as baseline scenario in cross-experiment calculations')
            baseline_file = os.path.join(fp, run_id_list[i] + '.pkl')

    # if baseline file not found, use the first experiment as baseline instead
    if baseline_file is None:
        baseline_file = os.path.join(fp, run_id_list[0] + '.pkl') # use first experiment performed as baseline experiment
        log.info(f'Could not find baseline scenario. Using {run_id_list[0]} as baseline scenario in cross-experiment calculations')

    with gzip.open(baseline_file, 'rb') as f:
        d = pickle.load(f)  # load FleetModel instance
        default_totc_opt = d.tot_impacts_opt

    try:
        totc_df.loc['Abs. difference from tot_impacts_opt'] = default_totc_opt - totc_df.loc['tot_impacts_opt']
        totc_df.loc['%_change_in_totc_opt'] = (default_totc_opt - totc_df.loc['tot_impacts_opt'])/default_totc_opt
    except:
        print('\n *****************************************')
        log.info("No comparison to default performed")

    # Plotting
    if not full_newtec_yr_df.isnull().all().all():
        full_newtec_yr_df.plot()
    else:
        log.info('100% market share of BEVs is not achieved in any scenario')

# Export potentially helpful output for analyzing across scenarios
if export:
    print('\n ********** Exporting cross-experiment results to pickle and Excel ************** \n')
    pickle_name = 'overview_run_' + now + '.pkl'
    with open(pickle_name, 'wb') as f:
        pickle.dump([shares_2030, shares_2050, add_share, stock_comp, full_newtec_yr_df, totc_df], f)

    excel_name = 'cumulative_scenario_output'+now+'.xlsx'
    with pd.ExcelWriter(excel_name) as writer:
        shares_2030.to_excel(writer,sheet_name='tec_shares_in_2030')
        shares_2050.to_excel(writer,sheet_name='tec_shares_in_2050')
        add_share.to_excel(writer,sheet_name='add_shares')
        stock_comp.to_excel(writer,sheet_name='total_stock')
        full_newtec_yr_df.to_excel(writer,sheet_name='1st_year_full_newtec')
        totc_df.to_excel(writer,sheet_name='totc')

    log.info(f'Exported cross-experiment results to {pickle_name} and {excel_name}')

# Close logging handlers
for handler in log.handlers:
    handler.close()