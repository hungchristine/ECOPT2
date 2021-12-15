# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 12:37:00 2019

@author: chrishun

This code initializes the GAMS workspace and database for running a scenario using
the fleet model object.
"""

import os
import logging
import shutil
import traceback
import glob
import pandas as pd
import numpy as np


import gams
import gmspy

log = logging.getLogger(__name__)

class GAMSRunner:
    """
    Control the GAMS environment for running experiments.

    Set up parameters for the GAMS workspace and export filepaths, set up

    Attributes
    ----------
    current_path : str
                Working directory filepath
    export_fp : str
                Export directory filepath
    ws : gams.GamsWorkspace
        GAMS workspace for experiments
    opt : gams.GamsOptions
        Options for GAMS run
    db : gams.GamsDatabase
        Database containing all GAMS symbols for experiment
    ms: int
        model solve state from GAMS
    ss: int
        solution solve state from GAMS

    Methods
    -------
    _load_input_to_GAMS
        Utility method to load input to GAMS workspace and create .gdx file
    get_output_from_GAMS(gams_db, output_var)
        Load output from GAMS .gdx file
    run_GAMS(fleet, run_tag, filename)
        Run experiment in GAMS
    """

    def __init__(self, fp):
        """Initialize GAMS workspace for set of experiments."""
        self.current_path = os.path.dirname(os.path.realpath(__file__))
        self.export_fp = fp
        self.ws = gams.GamsWorkspace(working_directory=self.current_path, debug=2)
        # gams.execution.SymbolUpdateType = 1  # if record does not exist, use values from instantiation

    def _load_input_to_GAMS(self, fleet, filename, timestamp):
        """Create input gdx file for GAMS experiment.

        Add database to workspace, update FleetModel, then load database with
        experiment parameters.

        Parameters
        ----------
        fleet : FleetModel object
            FleetModel containing run input.
        filename : str
            YAML filename with scenario definition.
        timestamp : str
            Runtime ID for filenames.

        Returns
        -------
        None.
        """
        try:
            if hasattr(self.db, 'name'):
                print('\n Database exists, clearing values from previous run')
                self.db.clear()  # remove entry values from database for subsequent runs
        except AttributeError:
            # for first run, add database to GAMS workspace
            print('\n Creating new GAMS database')
            self.db = self.ws.add_database(database_name='pyGAMS_input')

        years = gmspy.list2set(self.db, fleet.sets.year,'year')
        modelyear = gmspy.list2set(self.db, fleet.sets.modelyear,'modelyear')
        optyear = gmspy.list2set(self.db, fleet.sets.optyear, 'optyear')
        inityear = gmspy.list2set(self.db, fleet.sets.inityear, 'inityear')
        tec = gmspy.list2set(self.db, fleet.sets.tec, 'tec')
        newtec= gmspy.list2set(self.db, fleet.sets.newtec, 'newtec')
        age = gmspy.list2set(self.db, fleet.sets.age, 'age')
        new = gmspy.list2set(self.db, fleet.sets.new, 'new')
        enr = gmspy.list2set(self.db, fleet.sets.enr, 'enr')
        seg = gmspy.list2set(self.db, fleet.sets.seg, 'seg')
        reg = gmspy.list2set(self.db, fleet.sets.reg, 'reg')
        fleetreg = gmspy.list2set(self.db, fleet.sets.fleetreg, 'fleetreg')
        demeq =  gmspy.list2set(self.db, fleet.sets.demeq, 'demeq')
        grdeq = gmspy.list2set(self.db, fleet.sets.grdeq, 'grdeq')
        sigvar = gmspy.list2set(self.db, fleet.sets.sigvar, 'sigvar')
        veheq = gmspy.list2set(self.db, fleet.sets.veheq, 'veheq')

        mat_cat = gmspy.list2set(self.db, fleet.sets.mat_cat, 'mat_cat')
        mat_prod = gmspy.list2set(self.db, sum(fleet.sets.mat_prod.values(), []), 'mat_prod')  # concatenate all material producers from all material categories

        # create "mat" multidimensional superset
        try:
            mat = self.db.get_set('mat')
        except:
            mat = self.db.add_set("mat", 2, "")

        for key, item in fleet.sets.mat_prod.items():
            for producer in item:
                mat.add_record((key, producer))

        veh_oper_dist = gmspy.df2param(self.db, fleet.parameters.veh_oper_dist, ['year'], 'VEH_OPER_DIST')
        veh_stck_tot = gmspy.df2param(self.db, fleet.parameters.veh_stck_tot, ['year', 'fleetreg'], 'VEH_STCK_TOT')
        veh_stck_int_seg = gmspy.df2param(self.db, fleet.parameters.veh_stck_int_seg, ['seg'], 'VEH_STCK_INT_SEG')
        bev_capac = gmspy.df2param(self.db, fleet.parameters.bev_capac, ['seg'], 'BEV_CAPAC')

        veh_lift_cdf = gmspy.df2param(self.db, fleet.parameters.veh_lift_cdf, ['age'], 'VEH_LIFT_CDF')
        veh_lift_pdf = gmspy.df2param(self.db, fleet.parameters.veh_lift_pdf, ['age'], 'VEH_LIFT_PDF')
        veh_lift_age = gmspy.df2param(self.db, fleet.parameters.veh_lift_age, ['age'], 'VEH_LIFT_AGE')
        veh_lift_mor = gmspy.df2param(self.db, fleet.parameters.veh_lift_mor, ['age'], 'VEH_LIFT_MOR')

        veh_stck_int_tec = gmspy.df2param(self.db, fleet.parameters.veh_stck_int_tec,['tec'],'VEH_STCK_INT_TEC')

        enr_veh = gmspy.df2param(self.db, fleet.parameters.enr_veh, ['enr', 'tec'], 'ENR_VEH')
        enr_cint = gmspy.df2param(self.db, fleet.parameters.enr_cint, ['enr', 'reg', 'year'], 'ENR_CINT')

        veh_pay = gmspy.df2param(self.db, fleet.parameters.veh_pay, ['prodyear', 'age', 'year'], 'VEH_PAY')

        year_par = gmspy.df2param(self.db, fleet.parameters.year_par, ['year'], 'YEAR_PAR')
        veh_partab = gmspy.df2param(self.db, fleet.parameters.veh_partab, ['veheq', 'tec', 'seg', 'sigvar'], 'VEH_PARTAB')

        try:
            veh_add_grd = self.db.get_parameter('VEH_ADD_GRD')
        except:
            veh_add_grd = self.db.add_parameter_dc('VEH_ADD_GRD', ['grdeq', 'newtec'])

        # Prep work making add gradient df from given rate constraint
        # adding growth constraint for each (new/emerging) tec
        for keys, value in iter(fleet.parameters.veh_add_grd.items()):
            veh_add_grd.add_record(keys).value = value

        # gro_cnstrnt = gmspy.df2param(self.db, fleet.gro_cnstrnt, ['year'], 'GRO_CNSTRNT')

        manuf_cnstrnt = gmspy.df2param(self.db, fleet.parameters.manuf_cnstrnt, ['year'], 'MANUF_CNSTRNT')

        mat_content = gmspy.df2param(self.db, fleet.parameters.mat_content, ['year','mat_cat'], 'MAT_CONTENT')
        mat_cint = gmspy.df2param(self.db, fleet.parameters.mat_cint, ['year', 'mat_prod'], 'MAT_CINT')
        virg_mat = gmspy.df2param(self.db, fleet.parameters.virg_mat_supply, ['year','mat_prod'], 'VIRG_MAT_SUPPLY')
        recovery_pct = gmspy.df2param(self.db, fleet.parameters.recovery_pct, ['year','mat_cat'], 'RECOVERY_PCT')

        try:
            self.db.export(os.path.join(self.export_fp, filename + '_'+timestamp))
            log.info('Exported input database...' + filename + '_'+timestamp)
        except Exception as e:
            print('\n *****************************************')
            log.error(f'Error in exporting input database. {e}')
            self.db.suppress_auto_domain_checking = 1
            self.db.export(os.path.join(self.export_fp, filename + '_FAILED_'+timestamp))


    def get_output_from_GAMS(self, gams_db, output_var):
        """
        Retrieve symbol values from gams_db.

        Parameters
        ----------
        gams_db : gams.database.GamsDatabase
            Database containing experiment run results from GAMS.
        output_var : str
            Symbol name for values.

        Returns
        -------
        temp_output_df : Pandas DataFrame
            Contains results from GAMS experiment.

        """
        temp_GMS_output = []
        temp_index_list = []

        for rec in gams_db[output_var]:
            if gams_db[output_var].number_records == 1:  # special case for totc
                temp_output_df = gams_db[output_var].first_record().level
                return temp_output_df

            dict1 = {}
            dict1.update({'level': rec.level})
            temp_GMS_output.append(dict1)
            temp_index_list.append(rec.keys)
        temp_domain_list = list(gams_db[output_var].domains_as_strings)
        temp_index = pd.MultiIndex.from_tuples(temp_index_list, names=temp_domain_list)
        temp_output_df = pd.DataFrame(temp_GMS_output, index = temp_index)

        return temp_output_df


    def run_GAMS(self, fleet, run_tag, filename, timestamp):
        """
        Load FleetModel data to GAMS and initiate model solve.

        Parameters
        ----------
        fleet : FleetModel instance
                FleetModel containing run input.
        run_tag : str
                Unique experiment run name.
        filename : str
                YAML filename with scenario definition.
        timestamp: str
                Runtime ID for filenames.

        Raises
        ------
        exceptions
            Print all GamsDatabaseDomainViolations, including relevant symbols

        """
        # Pass to GAMS all necessary sets and parameters
        print('\n')
        print('Loading data to GAMS database')
        self._load_input_to_GAMS(fleet, filename, timestamp)

        # Run GAMS Optimization
        try:
            model_run = self.ws.add_job_from_file(fleet.gms_file, job_name='EVD4EUR_'+run_tag) # model_run is type GamsJob
            self.opt = self.ws.add_options() # opt is of type gams.options.GamsOptions
            self.opt.keep = 0  # Controls keeping or deletion of process directory and scratch files.
            self.opt.logline = 0  # Amount of line tracing to the log file; values of 0-2
            self.opt.trace = os.path.join(self.export_fp, 'trace.trc')
            self.opt.traceopt = 2  # Trace file format option; 1-3: Solver trace only in format used for GAMS performance world.
            self.opt.dumpparms = 0 # GAMS parameter logging; 1: accepted parameters/sets, 2; log of file operations + accepted sets/parameters
            self.opt.forcework = 0  # Force GAMS to process a save file created with a newer GAMS version or with execution errors.

            lst_fp = os.path.abspath(os.path.join(self.export_fp, 'EVD4EUR_A_' + run_tag + '.lst'))
            self.opt.set_output(lst_fp)
            self.opt.putdir = self.export_fp
            self.opt.defines["gdxincname"] = self.db.name  # for auto-loading of database in GAMS model

            print('\n' + f'Using input gdx file: {self.db.name}')
            print('Running GAMS model, please wait...')
            print('\n')

            model_run.run(gams_options=self.opt, databases=self.db)

            self.ms = model_run.out_db['ms'].find_record().value
            self.ss = model_run.out_db['ss'].find_record().value

            model_stat_dict = {1 : 'Optimal',
                               2 : 'Locally Optimal',
                               3 : 'Unbounded',
                               4 : 'Infeasible',
                               5 : 'Locally Infeasible',
                               6 : 'Intermediate Infeasible',
                               7 : 'Intermediate Nonoptimal',
                               8 : 'Integer Solution',
                               9 : 'Intermediate Non-Integer',
                               10 : 'Integer Infeasible',
                               11 : 'Licensing Problems - No Solution',
                               12 : 'Error Unknown',
                               13 : 'Error No Solution',
                               14 : 'No Solution Returned',
                               15 : 'Solved Unique',
                               16 : 'Solved',
                               17 : 'Solved Singular',
                               18 : 'Unbounded - No Solution',
                               19 : 'Infeasible - No Solution'
                               }
            solve_stat_dict = {1 : 'Normal Completion',
                               2 : 'Iteration Interrupt',
                               3 : 'Resource Interrupt',
                               4 : 'Terminated By Solver',
                               5 : 'Evaluation Interrupt',
                               6 : 'Capability Problems',
                               7 : 'Licensing Problems',
                               8 : 'User Interrupt',
                               9 : 'Setup Failure',
                               10 : 'Solver Failure',
                               11 : 'Internal Solver Failure',
                               12 : 'Solve Processing Skipped',
                               13 : 'System Failure'
                               }
            print('\n \n \n')
            log.info('Ran GAMS model: ' + fleet.gms_file)

            # provide model solve report
            if self.ms in model_stat_dict.keys():
                print(f'Model status: {self.ms}, {model_stat_dict[self.ms]}')
                print(f'Solve status: {self.ss}, {solve_stat_dict[self.ss]}' + '\n')

            gams_db = model_run.out_db  # get the solution .gdx database
            export_model = os.path.join(self.export_fp, run_tag + '_solution.gdx')
            gams_db.export(export_model)  # export the solution .gdx database
            print('\n')
            log.info(f'Completed export of solution database to {export_model}')


            # Workaround to manually move .lst and .pf files to run output folder
            # The entire filepath for these filepath can be (are) defined
            # in self.opt above, but the API has hardcoded these filenames to match
            # the job name and working directory of the workspace and therefore
            # ignores any changes made to these options. (See execution.py lines ~847)
            file_ext = ['.lst', '.pf']

            for file in file_ext:
                shutil.move(os.path.join(os.path.curdir, 'EVD4EUR_'+ run_tag+file),
                            self.export_fp)

            # remove the temp opt files the GAMS API creates (without file extension)
            for file in glob.glob('_gams_py_'+('[0-9a-z_]'*8)):
                log.info(f'Removing temp opt file {file}')
                os.remove(file)

        except Exception as e:
            print('\n *****************************************')
            log.error(f'----- Error in running model {fleet.gms_file}')
            try:
                exceptions = self.db.get_database_dvs()
                if len(exceptions) > 0:
                    log.error(f'----- GAMS database exceptions: \n{exceptions.symbol.name}')
                else:
                    log.error(f'{e} \n {traceback.format_exc()}')

            except Exception as ee:
                log.error(f'----- Error running GAMS model, no database. \n {ee}')

        # Fetch model outputs and retrieve key values for scenario comparisons
        try:
            fleet.read_gams_db(gams_db)  # retrieve results from GAMS run (.gdx file)
        except Exception as e:
            log.error(f'----- Error in reading GAMS database. {e}')

        try:
            fleet.import_model_results()  # load key results as FleetModel attributes
            log.info('Model results loaded to FleetModel object')
        except Exception as e:
            log.error(f'----- Error in loading results from GAMS database to FleetModel object. {e}')
