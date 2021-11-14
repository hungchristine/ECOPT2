# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:27:57 2019

@author: chrishun
"""

import os
import sys
import traceback
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import (MultipleLocator, IndexLocator, IndexFormatter)

# import seaborn
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

import itertools

import gams
import gmspy
from fleet_model_init import SetsClass, ParametersClass

class FleetModel:
    """
    Instance of a fleet model experiment.

    Contains all input to GAMS, stores all results from GAMS, and performs
    visualization of inputs and results.

    Attributes
    ----------
        home_fp : str
            filepath
        gms_file : str
            filepath to .gms file with LP model
        gdx_file : str
            Optional, filepath to gdx file to use as input for building FleetModel instance
        import_fp : str
            filepath to .csv file containing input values
        export_fp : str
            filepath to folder to save result files
        keeper : str
            timestamp identifier for tagging results

    Methods
    -------
        read_gams_db()
            description
        import_model_results()
            description
        build_BEV()
            description
        BEV_weight()
            description
        build_veh_partab()
            description
        figure_calculations()
            DE

    Placeholder methods
    -------------------
        read_all_sets()
            DEPRECATED
        get_output_from_GAMS()
            DEPRECATED
        calc_op_emissions()
        calc_EOL_emissions()
        calc_cint_operation()
        calc_eint_oper()
        calc_veh_mass()
        vehicle_builder()
        calc_crit_materials()
        post_processing()
        import_from_MESSAGE()
        elmix()
    """

    def __init__(self,
                 sets: SetsClass=None,
                 parameters: ParametersClass=None,
                 pkm_scenario='iTEM2-Base',
                 data_from_message=None,
                 gdx_file=None
                 ):
        """
        Initialize with experiment values.

        If .gdx filepath given, intialize from gdx, otherwise, initialize from YAML file.

        Parameters
        ----------
        sets: SetsClass
            Contains all sets and indices for GAMS model.
        parameters: ParametersClass
            Contains all parameter values for CMAS model.
        pkm_scenario : str, optional
            Name of scenario for total p-km travelled. The default is 'iTEM2-Base'.
        data_from_message : Pandas DataFrame, optional
            Contains output (eg., total transport demand or electricity mix
            intensity) from MESSAGE/integrate assessment model in
            time series form. The default is None.
        gdx_file : str, optional
            Filepath of .gdx file to read from. The default is None.

        Returns
        -------
        None.
        """

        # Instantiate filepaths
        self.home_fp = os.path.dirname(os.path.realpath(__file__))

#       self.gdx_file = 'C:\\Users\\chrishun\\Box Sync\\YSSP_temp\\EVD4EUR_input.gdx'#EVD4EUR_ver098.gdx'  # for reading in inputs from gdx file
        self.gms_file = os.path.join(self.home_fp, r'EVD4EUR.gms') # GAMS model file
        self.import_fp = os.path.join(self.home_fp, r'GAMS_input_new.xls')
        self.export_fp = os.path.join(self.home_fp, r'Model run data')
        self.keeper = "{:%d-%m-%y, %H_%M}".format(datetime.now())

        if gdx_file is not None:
            self._from_gdx(gdx_file)
        else:
            self.sets = sets
            self.parameters = parameters
            try:
                self._from_python(data_from_message)
            except AttributeError as err:
                print('\n *****************************************')
                print(f"Exception: {err}")
                print(traceback.format_exc())
                print("Generating empty fleet object")

    def _from_python(self, data_from_message):
        """
        Instantiate FleetModel object from scratch via Excel input files.

        Parameters
        ----------
        data_from_message : Pandas DataFrame, optional
            Contains output (eg., total transport demand or electricity mix
            intensity) from MESSAGE/integrate assessment model in
            time series form. The default is None.

        Returns
        -------
        None.
        """
        # battery_specs, fuelcell_specs and lightweighting are not (yet) in use
        # self.battery_specs = pd.DataFrame() # possible battery_sizes (and acceptable segment assignments, CO2 production emissions, critical material content, mass)
        # self.fuelcell_specs = pd.DataFrame() # possible fuel cell powers (and acceptable segment assignments, CO2 production emissions, critical material content, fuel efficiency(?), mass)
        # self.lightweighting = pd.DataFrame() # lightweighting data table - lightweightable materials and coefficients for corresponding lightweighting material(s)

        if data_from_message is not None:
            # Not currently implemented
            self.el_intensity = data_from_message # regional el-mix intensities as time series from MESSAGE
            self.trsp_dem = data_from_message # EUR transport demand as time series from MESSAGE

        self.parameters.calculate_from_raw_data(self.sets)  # calculate parameters from intermediate/raw data
        self.parameters.validate_data(self.sets)
        """ boundary conditions for constraints, e.g., electricity market supply constraints, crit. material reserves? could possibly belong in experiment specifications as well..."""

        #### filters and parameter aliases ####
        self.parameters.enr_veh = self._process_df_to_series(self.parameters.enr_veh)
        self.parameters.veh_pay = self._process_df_to_series(self.parameters.veh_pay)

        # Temporary introduction of seg-specific VEH_PARTAB from Excel; will later be read in from YAML
#        self.veh_partab = pd.DataFrame(pd.read_excel(self.import_fp,sheet_name = 'genlogfunc',usecols='A:G',index_col=[0,1,2],skipfooter=6)).stack()
        if type(self.parameters.raw_data.r_term_factors) == float:
            self.parameters.raw_data.r_term_factors = {'BEV': self.parameters.raw_data.r_term_factors, 'ICE': self.parameters.raw_data.r_term_factors}
        if type(self.parameters.raw_data.u_term_factors) == float or type(self.parameters.raw_data.u_term_factors) == int:
            self.parameters.raw_data.u_term_factors = {'BEV': self.parameters.raw_data.u_term_factors, 'ICE': self.parameters.raw_data.u_term_factors}

        """# ACEA.be has segment division for Western Europe
        # https://www.acea.be/statistics/tag/category/segments-body-country
        # More detailed age distribution (https://www.acea.be/uploads/statistic_documents/ACEA_Report_Vehicles_in_use-Europe_2018.pdf)"""

        self.growth_constraint = 0  # growth_constraint
        self.gro_cnstrnt = [self.growth_constraint for i in range(len(self.sets.modelyear))]
        self.gro_cnstrnt = pd.Series(self.gro_cnstrnt, index=self.sets.modelyear)
        self.gro_cnstrnt.index = self.gro_cnstrnt.index.astype('str')

        self.manuf_cnstrnt = self._process_df_to_series(self.parameters.manuf_cnstrnt)
        self.mat_content = [[0.11, 0.05] for year in range(len(self.sets.modelyear))]
        self.mat_content = pd.DataFrame(self.parameters.mat_content, index=self.sets.modelyear, columns=self.sets.mat_cats)
        self.mat_content.index = self.mat_content.index.astype('str')

        self.parameters.mat_cint.columns = self.parameters.mat_cint.columns.astype(str)
        self.parameters.mat_cint = self.parameters.mat_cint.T
        self.parameters.mat_cint.columns = self.parameters.mat_cint.columns.droplevel(['mat_cat'])

        self.parameters.recovery_pct.index = self.parameters.recovery_pct.index.astype('str')

        #TODO: remove hardcoding of European share
        self.parameters.virg_mat_supply.index = self.parameters.virg_mat_supply.index.astype('str')
        self.parameters.virg_mat_supply.columns = self.parameters.virg_mat_supply.columns.droplevel(['mat_cat'])

        # --------------- Expected GAMS Outputs ------------------------------
        self.totc = None
        self.BEV_fraction = pd.DataFrame()
        self.ICEV_fraction = pd.DataFrame()
        self.BEV_ADD_blaaaah = pd.DataFrame()
        self.VEH_STCK = pd.DataFrame()

        """ experiment specifications """
        self.recycling_losses = pd.DataFrame() # vector of material-specific recycling loss factors
        self.fossil_scenario = pd.DataFrame() # adoption of unconventional sources for fossil fuel chain
        self.hydrogen_scenario = pd.DataFrame()

        self.battery_density = None # time series of battery energy densities
        self.lightweighting_scenario = None # lightweighting scenario - yes/no (or gradient, e.g., none/mild/aggressive?)

    @classmethod
    def _from_gdx(self, gdx_file):
        """
        Instantiate FleetModel object via an existing GAMS .gdx file.

        Use gmpsy module to retrieve set values from .gdx file. Load
        database using GAMS API, generate dicts containing parameter and
        variable values, and assign to FleetModel attributes.

        Parameters
        ----------
        gdx_file : str
            Filepath of .gdx file to read from.

        Returns
        -------
        None.
        """
        # TODO: generalize to allow loading input OR results .gdx
        # Build fleet object from gdx file (contains model inputs and outputs)
        # For visualization

        self.sets = gmspy.ls(gdx_filepath=gdx_file, entity='Set')  # list of set names
        ws = gams.GamsWorkspace()
        db = ws.add_database_from_gdx(gdx_file)

        # attempt to generalize set intro
        # for i, name in enumerate(self.sets):
        #     set_dict = {name: gmspy.set2list(self.sets[i], db=db, ws=ws)}

        self.year = gmspy.set2list(self.sets[0], db=db, ws=ws)
        self.age = gmspy.set2list(self.sets[4], db=db, ws=ws)
        self.tecs = gmspy.set2list(self.sets[5], db=db, ws=ws)
        self.enr = gmspy.set2list(self.sets[6], db=db, ws=ws)
        self.reg = gmspy.set2list(self.sets[7], db=db, ws=ws)
        self.seg = gmspy.set2list(self.sets[8], db=db, ws=ws)

        self.read_gams_db(db)  # generate p_dict and v_dict

        # Retrieve parameters required for visualization and calculations
        self.veh_oper_dist = self._p_dict['VEH_OPER_DIST']

        # Retrieve GAMS-calculated parameters and variables
        self.import_model_results()


    @staticmethod
    def _process_df_to_series(df):
        """
        Process DataFrames to MultIndexed Series for exporting to GAMS.

        Parameters
        ----------
        df : Pandas DataFrame
            Unstacked DataFrame.

        Returns
        -------
        df : Pandas Series
            Stacked (Series) form of df, with MultiIndex.
        """
        dims = df.shape[1] - 1 # assumes unstacked format
        if dims > 0:
            if isinstance(df.columns, pd.MultiIndex):
                df = df.stack(level=[i for i in range(df.columns.nlevels)])
            else:
                # make MultiIndex from columns
                indices = df.columns[:-1].tolist()
                df.set_index(indices, inplace=True)

                temp = []
                # convert index values to string (requirement for GAMS)
                for i in range(dims):
                    temp.append(df.index.get_level_values(i).astype(str))
                df.index = temp
                df.index.names = [''] * dims

                df.columns = ['']
                df = pd.Series(df.iloc[:, 0])
            return df
        else:
            # case of df is DataFrame with 1 column in series form (nx1)
            df.columns = ['']
            return df

    def read_gams_db(self, gams_db):
        """
        Fetch all symbols from GAMS database.

        Extract all parameters, variables and equations from GAMS database.
        Stores symbols in a dictionary of each symbol type.
        Raise exceptions if errros in unpacking the database occurs.

        Parameters
        ----------
        gams_db : gams.database.GamsDatabase
            Database containing experiment run results from GAMS.

        Returns
        -------
        None.
        """
        # List all symbol contents for each symbol type
        sets = gmspy.ls(db=gams_db, entity='Set')
        parameters = gmspy.ls(db=gams_db, entity='Parameter')
        variables = gmspy.ls(db=gams_db, entity='Variable')
        equations = gmspy.ls(db=gams_db, entity='Equation')

        # Export parameters
        self._p_dict = {}
        for p in parameters:
            # Skip model status and solver status scalar parameters
            if (p != 'ms') and (p != 'ss'):
                try:
                    self._p_dict[p] = gmspy.param2df(p, db=gams_db)
                except ValueError as e:
                    print('\n *****************************************')
                    print(f'p_dict ValueError in {p}')
                except AttributeError:
                    print('\n *****************************************')
                    print(f'-----Warning!: p_dict AttributeError in {p}! Probably no records for this parameter.')
                    pass
                except Exception as E:
                    print(E)

        # Export variables
        self._v_dict = {}
        for v in variables:
            try:
                self._v_dict[v] = gmspy.var2df(v, db=gams_db)
            except ValueError:
                if len(gams_db[v]) == 1: # special case for totc
                    self._v_dict[v] = gams_db[v].first_record().level
                else:
                    print(f'Warning!: v_dict ValueError in {v}!')
                pass
            except TypeError: # This error is specifically for seg_add
                print('\n *****************************************')
                print(f'-----Warning! v-dict TypeError in {v}! Probably no records for this variable.')
                print('\n')
                pass
            except Exception as E:
                print(E)

        self._e_dict = {}
        for e in equations:
            try:
                self._e_dict[e] = gmspy.eq2series(e, db=gams_db)
            except ValueError:
                if len(gams_db[e]) == 1: # special case for totc
                    self._e_dict[e] = gams_db[e].first_record().level
                else:
                    print(f'-----Warning: e_dict ValueError in {e}!')
                    print('\n')
            except:
                print('\n *****************************************')
                print(f'-----Warning: Error in {e}')
                print(sys.exc_info()[0])
                print('\n')
                pass


    def import_model_results(self):
        """
        Unpack parameters and variables for visualization.

        Returns
        -------
        None.
        """

        def reorder_age_headers(df_unordered):
            """
            Reorder age index in ascending order.

            Parameters
            ----------
            df_unordered : Pandas DataFrame
                DataFrame with out-of-order age index.

            Returns
            -------
            temp : Pandas DataFrame
                DataFrame with age index sorted in ascending order.
            """
            temp = df_unordered
            temp.columns = temp.columns.astype(int)
            temp.sort_index(inplace=True, axis=1)
            return temp

        # Import the parameters that are calculated within the GAMS model
        self.veh_prod_cint = self._p_dict['VEH_PROD_CINT']
        self.veh_prod_cint = self.veh_prod_cint.stack().to_frame()
        self.veh_prod_cint.index.rename(['tec', 'seg', 'prodyear'], inplace=True)

        self.veh_oper_eint = self._p_dict['VEH_OPER_EINT']
        self.veh_oper_eint = self.veh_oper_eint.stack().to_frame()
        self.veh_oper_eint.index.rename(['tec', 'seg', 'year'], inplace=True)

        self.veh_oper_cint = self._p_dict['VEH_OPER_CINT']
        self.veh_oper_cint = self.veh_oper_cint.stack().to_frame()
        self.veh_oper_cint.index.names = ['tec', 'enr', 'seg', 'fleetreg', 'age', 'modelyear', 'prodyear']


        # Import model results
        self.veh_stck_delta = self._v_dict['VEH_STCK_DELTA']
        self.veh_stck_add = self._v_dict['VEH_STCK_ADD']
        self.veh_stck_rem = self._v_dict['VEH_STCK_REM']
        self.veh_stck = self._v_dict['VEH_STCK']
        self.veh_totc = self._v_dict['VEH_TOTC']
        self.annual_totc = self.veh_totc.sum(axis=0)

        self.totc_opt = self._v_dict['TOTC_OPT']

        self.veh_prod_totc = self._v_dict['VEH_PROD_TOTC']
        self.veh_oper_totc = self._v_dict['VEH_OPER_TOTC']
        self.total_op_emissions = self.veh_oper_totc.sum(axis=0)
        self.veh_eolt_totc = self._v_dict['VEH_EOLT_TOTC']

        self.emissions = pd.concat([self.veh_prod_totc.stack(), self.veh_oper_totc.stack(), self.veh_eolt_totc.stack()], axis=1)
        self.emissions.columns = ['Production', 'Operation', 'End-of-life']
        self.emissions = self.emissions.unstack(['tec', 'year']).sum().unstack([None, 'tec'])

        try:
            self.recycled_batt = self._v_dict['RECYCLED_BATT']
            self.recycled_mat = self._v_dict['RECYCLED_MAT']
            self.primary_mat = self._v_dict['TOT_PRIMARY_MAT']
            self.resources = pd.concat([self.primary_mat, self.recycled_mat], axis=1, keys=['primary', 'recycled'])
            self.mat_mix = self._v_dict['MAT_MIX']
            self.mat_co2 = self._v_dict['MAT_CO2']
        except Exception as e:
            print('INFO: could not load material related variables. Check model.')

        # Prepare model output dataframes for visualization
        self.stock_df = self._v_dict['VEH_STCK']
        self.stock_df = reorder_age_headers(self.stock_df)
        self.stock_add = self._v_dict['VEH_STCK_ADD']
        self.stock_add = reorder_age_headers(self.stock_add)
        self.stock_add = self.stock_add.dropna(axis=1, how='any')
        self.stock_add.index.rename(['tec', 'seg', 'fleetreg', 'prodyear'], inplace=True)
        self.stock_rem = self._v_dict['VEH_STCK_REM']
        self.stock_rem = reorder_age_headers(self.stock_rem)
        self.stock_df_plot = self.stock_df.stack().unstack('age')
        self.stock_df_plot = reorder_age_headers(self.stock_df_plot)
        if (self.stock_df_plot.values < 0).any():
            # check for negative values in VEH_STCK; throw a warning for large values.
            if self.stock_df_plot.where((self.stock_df_plot < 0) & (np.abs(self.stock_df_plot) > 1e-1)).sum().sum() < 0:
                print('-----Warning: Large negative values in VEH_STCK found')
                print('\n')
            else:
                # for smaller values, set 0 and print warning
                print('-----Warning: Small negative values in VEH_STCK found. Setting to 0')
                self.stock_df_plot.where(~(self.stock_df_plot < 0) & ~(np.abs(self.stock_df_plot) <= 1e-1), other=0, inplace=True)
                print('\n')

        self.stock_df_plot_grouped = self.stock_df_plot.groupby(['tec', 'seg'])

        # Import post-processing parameters
        try:
            self.veh_oper_cohort = self._p_dict['VEH_OPER_COHORT']
            self.veh_oper_cohort.index.rename(['tec', 'seg', 'fleetreg', 'prodyear', 'modelyear' 'age'], inplace=True)
            self.stock_cohort = self._p_dict['VEH_STCK_COHORT']
            self.stock_cohort.index.rename(['tec', 'seg', 'fleetreg', 'prodyear', 'age'], inplace=True)
            self.stock_cohort.columns.rename('modelyear', inplace=True)
            self.stock_cohort = self.stock_cohort.droplevel(level='age', axis=0)
            self.stock_cohort = self.stock_cohort.stack().unstack('prodyear').sum(axis=0, level=['tec', 'fleetreg', 'modelyear'])
            self.bau_emissions = self._p_dict['BAU_EMISSIONS']
            self.bau_emissions.index.rename(['modelyear'], inplace=True)

            # self.mat_req_virg = self._p_dict['MAT_REQ_VIRG']
            # self.mat_req_virg = pd.concat([self.mat_req_virg], axis=1, keys=['primary'])
            # self.mat_req_virg.where(cond=self.mat_req_virg>=0, other=np.nan, inplace=True)  # replace negative values with np.nan
            # if self.mat_req_virg.isnull().sum().sum() > 0:
            #     print('-----INFO: supply of secondary materials greater than demand of materials')
            #     print('\n')
            # self.mat_recycled = self._p_dict['MAT_RECYCLED']
            # self.mat_recycled = pd.concat([self.mat_recycled], axis=1, keys=['recycled'])
            self.mat_demand = self._p_dict['MAT_REQ_TOT']
            self.mat_demand = pd.concat([self.mat_demand], axis=1, keys=['total'])
            # self.resources_pp = pd.concat([self.mat_demand, self.mat_req_virg, self.mat_recycled], axis=1)
        except TypeError:
            print("-----Warning: Could not find post-processing parameter(s)")


        self.veh_oper_dist = self._p_dict['VEH_OPER_DIST']
        self.veh_oper_dist.index = self.veh_oper_dist.index.get_level_values(0) # recast MultiIndex as single index

        self.full_oper_dist = self.veh_oper_dist.reindex(self.veh_oper_cint.index, level='modelyear')
        self.op_emissions = self.veh_oper_cint.multiply(self.full_oper_dist)
        self.op_emissions.index = self.op_emissions.index.droplevel(level=['enr', 'age']) # these columns are unncessary/redundant
        self.op_emissions = self.op_emissions.sum(level=['tec', 'seg', 'fleetreg', 'prodyear']) # sum the operating emissions over all model years for each cohort
        self.op_emissions = self.op_emissions.reorder_levels(order=['tec', 'seg', 'fleetreg', 'prodyear']) # reorder MultiIndex to add production emissions

        self.LC_emissions = self.op_emissions.add(self.veh_prod_cint)

        add_gpby = self.stock_add.sum(axis=1).unstack('seg').unstack('tec')
        self.add_share = add_gpby.div(add_gpby.sum(axis=1), axis=0)

        # Export technology shares in 2030 to evaluate speed of uptake
        self.shares_2030 = self.add_share.loc(axis=0)[:, '2030']
        self.shares_2050 = self.add_share.loc(axis=0)[:, '2050']

        self.enr_cint = self._p_dict['ENR_CINT']
        self.enr_cint = self.enr_cint.stack()
        self.enr_cint.index.rename(['enr', 'reg', 'year'], inplace=True)

        print('\n Finished importing results from GAMS run')

    def figure_calculations(self):
        """
        Calculate lifecycle intensity by cohort for quality check.

        Calculate average operating emissions intensity by using total
        lifetime operating emissions by cohort, divided by original stock
        of the cohort

        Returns
        -------
        None.
        """
        ####  Used by main.py to aggregate key indicators across experiments.

        # Calculate operating emissions by cohort (i.e., prodyear)
        try:
            operation_em = self.veh_oper_cohort.sum(level=['prodyear', 'fleetreg', 'tec', 'seg']).sum(axis=1)
            operation_em.sort_index(axis=0, level=0, inplace=True)
            op = operation_em.loc['2000':'2050']

            # Calculate stock
            init_stock = self.veh_stck_add.loc(axis=1)[0]
            init_stock.replace(0, np.nan)
            init_stock.dropna(axis=0, inplace=True)
            init_stock.index = init_stock.index.reorder_levels([3, 2, 0, 1])
            init_stock.sort_index(inplace=True)

            self.op_intensity = op / init_stock
            self.op_intensity.sort_index(inplace=True)
        except Exception as e:
            print('\n *****************************************')
            print('Error in calculating operating emissions by cohort. Perhaps post-processing parameters not loaded?')
            print(e)

    #### Placeholder methods below
    """
        el_intensity: pandas Series
            according to given MESSAGE climate scenario
        trsp_dem:
            projected transport demand from MESSAGE, consistent with climate scenario

        lightweighting : Pandas DataFrame
                        Currently not in use. lightweighting correspondance matrix
        battery_specs : Pandas DataFrame
                        Currently not in use. Static battery specifications from inventories
        fuelcell_specs : Pandas DataFrame
                        Currently not in use. PEMFC specifications from inventories

        ?? recycling losses: material-specific manufacturing losses (?)
        fuel scenarios: fuel chain scenarios (fossil, hydrogen)
        occupancy rate: consumer preferences (vehicle ownership) and modal shifts
        battery_density: energy density for traction batteries
        lightweighting_scenario: whether (how aggressively) LDVs are lightweighted in the experiment
    """
    def main(self):
        #
        pass
    def calc_op_emissions(self):
        """ calculate operation emissions from calc_cint_operation and calc_eint_operation """
        pass

    def calc_prod_emissions(self):
        """ calculate production vehicle emissions"""
        """ glider, powertrain, RoV"""
        pass

    def calc_EOL_emissions(self):
        """ calculate EOL emissions"""
        pass

    def calc_cint_operation(self):
        # carbon intensity factors from literature here
        # can later update to include in modified A, F matrices
        # either use Kim's physics models or linear regression à la size & range
        pass

    def calc_eint_oper(self):
        # calculate the energy intensity of driving, kWh/km; function of mass
        pass

    def calc_veh_mass(self):
        # use factors to calculate vehicle total mass.
        # used in calc_eint_oper()
        pass

    def vehicle_builder(self):
        # Assembles vehicle from powertrain, glider and BoP and checks that vehicles makes sense (i.e., no Tesla motors in a Polo or vice versa)
        # used in calc_veh_mass()
        pass

    def add_to_GAMS(self):
        # Adding sets
        # def build_set(set_list=None, name=None, desc=None):
        #     i = self.db.add_set(name, 1, desc)
        #     for s in set_list:
        #         i.add_record(str(s))

        # NOTE: Check that 'cohort', 'year' and 'prodyear' work nicely together
#        cohort = build_set(self.cohort, 'year', 'cohort')
#        tec = build_set(self.tecs, 'tec', 'technology')
#        age = build_set(self.age, 'age', 'age')
#        enr = build_set(self.enr, 'enr', 'energy types')

        # Export for troubleshooting
        #self.db.export('add_sets.gdx')
        pass

    def calc_crit_materials(self):
        # performs critical material mass accounting
        pass

    def post_processing(self):
        # make pretty figures?
        pass

    def import_from_MESSAGE(self):
        pass

    """
    Intermediate methods
    """

    def elmix(self):
        # produce time series of elmix intensities, regions x year
        pass

    def _read_all_final_parameters(self, a_file):
        # deprecated
        # will become unnecessary as we start internally defining all parameters
        db = gmspy._iwantitall(None, None, a_file)


class EcoinventManipulator:
    """ Placeholder class. Generate time series of ecoinvent using MESSAGE inputs """

    def __init__(self, data_from_message, A, F):
        """
        Parameters
        ----------
        data_from_message : Pandas DataFrame
            Time series for input from MESSAGE/IAM; e.g., electricity mix.
        A : Numpy array
            Requirements matrix from ecoinvent.
        F : Numpy array
            Stressor matrix from ecoinvent.

        Returns
        -------
        None.

        """
        self.A = A  # default ecoinvent A matrix
        self.F = F  # default ecionvent F matrix

    def elmix_subst(self):
        """
        # substitute MESSAGE el mixes into ecoinvent
        """
        pass