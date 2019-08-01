# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:26:23 2019

@author: chrishun
"""

import fleet_model
#import test_gams
import itertools
from datetime import datetime
import yaml

"""
#fleet.read_all_sets("C:\\Users\\chrishun\\Box Sync5\\YSSP_temp\\EVD4EUR_input.gdx")
#fleet.add_to_GAMS()

#fleet._read_all_final_parameters("C:\\Users\\chrishun\\Box Sync\\YSSP_temp\\EVD4EUR_input.gdx")
#"""
#a=[1/6 for i in range(6)]
#b=[1/6 for i in range(6)]
#
#fleet = fleet_model.FleetModel(a,b)
#fleet.run_GAMS('run_x')      
#fleet.vis_GAMS('run_x') 

def run_experiment():
    # Load parameter values from YAML
    with open('input.yaml','r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    f = open('experiment_log.txt','a')
    i=1
    
    # Run experiments
    id_and_value = [p.items() for p in params.values()]
    
    id_and_value = [
            [('A', [ ....]), ('B', [ .... ])],
            [('X', [.....]), ('Y', [.....])],
            ]
    
    for (a_id, a), (b_id, b) in itertools.product(*id_and_value):
    for a, b in itertools.product(*params.values()):
        #### Make run ID tag
        filename = 'run '+str(i)
        run_id = f'run_{a_id}_{b_id}'
        print(filename)
        print(a)
        print(b)
        fm = fleet_model.FleetModel(veh_seg_shr=a, veh_seg_int=b) #need to pass in run ID tag for saving gdx/csv
        i+=1
        fm.run_GAMS(filename)
        #fm.vis_GAMS(filename)
        exceptions = fm.db.get_database_dvs()
        if len(exceptions)>1:
            print(exceptions[0].symbol.name)
            dunno=exceptions[0].symbol_dvs
            
            dunno2=exceptions[0].symbol
            print(fleet.db.number_symbols)
        
        
        f.write('Experiment run: '+str(i)+'\n')
        f.write('Time of run: '+"{:%d-%m-%y, %H_%M_%S}".format(datetime.now())+'\n')
        #### Add stuff about var_a, var_b, __
        f.write(str(a))
        f.write('\n')
        f.write(str(b))
        f.write('\n')
        f.write('TOTC: '+str(fm.totc))
        f.write('\n')
        f.write('End experiment *************** \n')
        f.write('\n')
    f.close()
 
run_experiment()
   
    
#        bounds = ['high','baseline','low']

#    for element in itertools.product(experiment_list,bounds):
#        fleet = fleet_model.FleetModel()
    
#    for experiment in experiment_list:
#        keeper
#        for i, bound in enumerate(bounds):
#            keeper[i] = + bound