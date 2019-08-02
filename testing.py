# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:26:23 2019

@author: chrishun
"""
import logging
import sys

import fleet_model
#import test_gams
from itertools import product
from datetime import datetime
import yaml

"""
#fleet.read_all_sets("C:\\Users\\chrishun\\Box Sync5\\YSSP_temp\\EVD4EUR_input.gdx")
#fleet.add_to_GAMS()

#fleet._read_all_final_parameters("C:\\Users\\chrishun\\Box Sync\\YSSP_temp\\EVD4EUR_input.gdx")
#"""
a=[1/6 for i in range(6)]
b=[1/6 for i in range(6)]

fleet = fleet_model.FleetModel(a,b)
fleet.run_GAMS('run_x')
#fleet.vis_GAMS('run_x')


# Log to screen
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

# Make output YAML less ugly, see https://stackoverflow.com/a/30682604
yaml.SafeDumper.ignore_aliases = lambda *args: True


def run_experiment():
    # Load parameter values from YAML
    with open('input.yaml', 'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Dictionary for logging
    info = {}

    # Explicit list of parameters
    param_names = ['veh_seg_int', 'veh_seg_shr']

    # Run experiments
    id_and_value = [params[p].items() for p in param_names]
    # NB could also change the names here
    for i, run_params in enumerate(product(*id_and_value)):
        # Same order as in param_names. Each is a tuple of (id, values)
        veh_seg_int, veh_seg_shr = run_params

        # Make run ID
        run_id = f'run_{veh_seg_int[0]}_{veh_seg_shr[0]}'
        # run_id = f'run_{i}'  # alternate format

        log.info(f'Starting run {run_id}')

        # need to pass in run ID tag for saving gdx/csv
        # NB here, use explicit names to avoid any confusion
        fm = fleet_model.FleetModel(veh_seg_shr=veh_seg_shr[1],
                                    veh_seg_int=veh_seg_int[1])
        fm.run_GAMS(run_id)

        # fm.vis_GAMS(filename)
        exceptions = fm.db.get_database_dvs()
        if len(exceptions) > 1:
            print(exceptions[0].symbol.name)
            domain_violations = exceptions[0].symbol_dvs

            symbol_exc = exceptions[0].symbol
            print(fm.db.number_symbols)

        # Save log info
        info[run_id] = {
            'params': {
                'veh_seg_int': veh_seg_int,
                'veh_seg_shr': veh_seg_shr,
            },
            'output': {
                'totc': 42,   # life, the universe, and everything…
                # 'totc': fm.totc,
            }
        }

        # Display the info for this run
        log.info(repr(info[run_id]))

    # Write log to file
    now = datetime.now().isoformat(timespec='seconds').replace(':','_')
    with open(f'testing_{now}.yaml', 'w') as f:
        yaml.safe_dump(info, f)


#run_experiment()

"""
GDX export from within GAMS:
       execute 'gdxviewer.exe i=inputfile.gdx type=outputfile id=x';

execute 'Gdxviewer.exe Trnsport.gdx';
execute_unload 'Result.gdx', i,x; * where x is variable, i is set

* XLS writing
execute 'gdxviewer.exe i=Result.gdx xls=Result.xls id=x';

* Excel Pivot Table writing
execute 'gdxviewer.exe i=Result.gdx pivot=ResultPivot.xls id=x';

* CSV file writing
execute 'gdxviewer.exe i=Result.gdx csv=Result.csv id=x';
"""

#        bounds = ['high','baseline','low']

#    for element in itertools.product(experiment_list,bounds):
#        fleet = fleet_model.FleetModel()

#    for experiment in experiment_list:
#        keeper
#        for i, bound in enumerate(bounds):
#            keeper[i] = + bound
