"""
Created on Sun Apr 21 13:27:57 2019

@author: chrishun
"""

import os
import logging
import sys
import traceback
import pandas as pd
import numpy as np
from scipy.stats import norm

from datetime import datetime

import gams
import gmspy
from fleet_model_init import SetsClass, ParametersClass

log = logging.getLogger(__name__)


class FleetModel:
    """Instance of a fleet model experiment.

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
            Imports all entries in GAMS database
        import_model_results()
            Import results from a .gdx file to FleetModel instance and cleans data for visualization
        figure_calculations()
            Exports key indicators across experiments
    """

    def __init__(self,
                 sets: SetsClass=None,
                 parameters: ParametersClass=None,
                 pkm_scenario='iTEM2-Base',
                 data_from_message=None,
                 gdx_file=None
                 ):
        """Initialize with experiment values.

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
                log.error(f"Exception: {err}. {traceback.format_exc()}")
                log.warning('----- Generating empty fleet object')
                # print("Generating empty fleet object")

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
        if data_from_message is not None:
            # Not currently implemented
            self.el_intensity = data_from_message # regional el-mix intensities as time series from MESSAGE
            self.trsp_dem = data_from_message # EUR transport demand as time series from MESSAGE

        self.parameters.calculate_from_raw_data(self.sets)  # calculate parameters from intermediate/raw data
        """ boundary conditions for constraints, e.g., electricity market supply constraints, crit. material reserves? could possibly belong in experiment specifications as well..."""
        self.parameters.validate_data(self.sets)

        #### filters and parameter aliases ####
        self.parameters.enr_veh = self._process_df_to_series(self.parameters.enr_veh)
        self.parameters.veh_pay = self._process_df_to_series(self.parameters.veh_pay)

        if type(self.parameters.raw_data.r_term_factors) == float:
            self.parameters.raw_data.r_term_factors = {'BEV': self.parameters.raw_data.r_term_factors, 'ICE': self.parameters.raw_data.r_term_factors}
        if type(self.parameters.raw_data.u_term_factors) == float or type(self.parameters.raw_data.u_term_factors) == int:
            self.parameters.raw_data.u_term_factors = {'BEV': self.parameters.raw_data.u_term_factors, 'ICE': self.parameters.raw_data.u_term_factors}

        self.growth_constraint = 0  # growth_constraint
        self.gro_cnstrnt = [self.growth_constraint for i in range(len(self.sets.modelyear))]
        self.gro_cnstrnt = pd.Series(self.gro_cnstrnt, index=self.sets.modelyear)
        self.gro_cnstrnt.index = self.gro_cnstrnt.index.astype('str')

        self.manuf_cnstrnt = self._process_df_to_series(self.parameters.manuf_cnstrnt)
        self.mat_content = [[0.11, 0.05] for year in range(len(self.sets.modelyear))]
        self.mat_content = pd.DataFrame(self.parameters.mat_content, index=self.sets.modelyear, columns=self.sets.mat_cat)
        self.mat_content.index = self.mat_content.index.astype('str')

        self.parameters.mat_cint.columns = self.parameters.mat_cint.columns.astype(str)
        self.parameters.mat_cint = self.parameters.mat_cint.T
        self.parameters.mat_cint.columns = self.parameters.mat_cint.columns.droplevel(['mat_cat'])

        self.parameters.recovery_pct.index = self.parameters.recovery_pct.index.astype('str')

        self.parameters.virg_mat_supply.index = self.parameters.virg_mat_supply.index.astype('str')
        self.parameters.virg_mat_supply.columns = self.parameters.virg_mat_supply.columns.droplevel(['mat_cat'])

        # --------------- Expected GAMS Outputs ------------------------------
        self.totc = None
        self.VEH_STCK = pd.DataFrame()

        log.info('Imported parameters')

    def _from_gdx(self, gdx_file):
        """
        Instantiate FleetModel object via an existing GAMS .gdx file.

        Use gmpsy module to retrieve set values from .gdx file. Load
        database using GAMS API, generate dicts containing parameter and
        variable values, and assign to FleetModel attributes. Useful for
        visualizing.

        Parameters
        ----------
        gdx_file : str
            Filepath of .gdx file to read from.

        Returns
        -------
        None.
        """

        # Build fleet object from gdx file (contains model inputs and outputs)
        # For visualization

        ws = gams.GamsWorkspace()
        db = ws.add_database_from_gdx(gdx_file)

        self.read_gams_db(db, sets=True) # generate s_dict, p_dict and v_dict
        self.sets = SetsClass(**self._s_dict)
        self._p_dict = {key.lower(): value for key, value in self._p_dict.items()}
        self.parameters = ParametersClass(**self._p_dict)

        # Retrieve GAMS-calculated variables, if present in .gdx file
        if self._v_dict:
            self.import_model_results()

        log.info(f'Loaded FleetModel object from {gdx_file}')


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

    def read_gams_db(self, gams_db, sets=False):
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
        log.info('Loading parameters...')
        for p in parameters:
            # Skip model status and solver status scalar parameters
            if (p != 'ms') and (p != 'ss'):
                try:
                    self._p_dict[p] = gmspy.param2df(p, db=gams_db)
                except ValueError as e:
                    log.error(f'p_dict ValueError in {p}. {e}')
                except AttributeError:
                    log.warning(f'p_dict AttributeError in {p}! Probably no records for this parameter.')
                    pass
                except Exception as E:
                    log.error(f'Error in loading parameters. {E}')
                    # print(E)

        # Export variables
        self._v_dict = {}
        print('\n')
        log.info('Loading variables...')
        for v in variables:
            try:
                self._v_dict[v] = gmspy.var2df(v, db=gams_db)
            except ValueError:
                if len(gams_db[v]) == 1: # special case for totc
                    self._v_dict[v] = gams_db[v].first_record().level
                else:
                    log.error('-----  v_dict ValueError in {v}!')
                pass
            except TypeError: # This error is specifically for seg_add
                log.warning(f'----- v-dict TypeError in {v}! Probably no records for this variable.')
                pass
            except Exception as E:
                print(E)

        print('\n')
        log.info('Loading equations...')
        self._e_dict = {}
        for e in equations:
            try:
                self._e_dict[e] = gmspy.eq2series(e, db=gams_db)
            except ValueError:
                if len(gams_db[e]) == 1: # special case for totc
                    self._e_dict[e] = gams_db[e].first_record().level
                else:
                    log.warning(f'-----  e_dict ValueError in {e}!')
            except:
                log.error(f'----- Error in {e}. {sys.exc_info()[0]}')
                pass

        if sets:
            log.info('Loading sets...')
            self._s_dict = {}
            for s in sets:
                try:
                    if s != 'mat':
                        self._s_dict[s] = gmspy.set2list(s, db=gams_db, ws=None, gdx_filepath=None)
                except Exception as e:
                    log.warning(f'----- Error loading sets. {e}.')

            mat_dict = {}
            for m in self._s_dict['mat_cat']:
                mat_dict[m] = [prod for prod in self._s_dict['mat_prod'] if m in prod]  # NB: relies on mat_cat being in the producer names
            self._s_dict['mat_prod'] = mat_dict

        log.info('Finished loading')

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

        try:
            self.veh_oper_cint = self._p_dict['VEH_OPER_CINT']
            self.veh_oper_cint = self.veh_oper_cint.stack().to_frame()
            self.veh_oper_cint.index.names = ['tec', 'enr', 'seg', 'fleetreg', 'age', 'modelyear', 'prodyear']
        except:
            log.warning('Invalid input for veh_oper_cint')

        # Import model results
        try:
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
                log.warning(f'Could not load material related variables. Check model. {e}')

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
                    log.warning('----- Large negative values in VEH_STCK found')
                    print('\n')
                else:
                    # for smaller values, set 0 and print warning
                    print('-----Warning: Small negative values in VEH_STCK found. Setting to 0')
                    log.warning('----- Small negative values in VEH_STCK found. Setting to 0')
                    self.stock_df_plot.where(~(self.stock_df_plot < 0) & ~(np.abs(self.stock_df_plot) <= 1e-1), other=0, inplace=True)
                    print('\n')

            self.stock_df_plot_grouped = self.stock_df_plot.groupby(['tec', 'seg'])
        except ValueError as e:
            log.info(f'Could not load variables from .gdx. Perhaps input database was used? {e}')
        except Exception as e:
            log.warning(f'----- Error loading variables from .gdx. {e}')

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

            self.mat_demand = self._p_dict['MAT_REQ_TOT']
            self.mat_demand = pd.concat([self.mat_demand], axis=1, keys=['total'])

            self.batt_demand = self._p_dict['TOT_BATT_MANUF']
            self.batt_demand.index.rename(['modelyear'], inplace=True)
            # reset index (which is currently a single-level MultiIndex)
            self.batt_demand.index = self.batt_demand.index.get_level_values(0)
        except TypeError as e:
            log.warning(f"Could not find post-processing parameter(s). {e}")


        self.veh_oper_dist = self._p_dict['VEH_OPER_DIST']
        self.veh_oper_dist.index = self.veh_oper_dist.index.get_level_values(0) # recast MultiIndex as single index

        self.full_oper_dist = self.veh_oper_dist.reindex(self.veh_oper_cint.index, level='modelyear')
        self.op_emissions = self.veh_oper_cint.multiply(self.full_oper_dist)
        self.op_emissions.index = self.op_emissions.index.droplevel(level=['enr', 'age']) # these columns are unncessary/redundant
        self.op_emissions = self.op_emissions.sum(level=['tec', 'seg', 'fleetreg', 'prodyear']) # sum the operating emissions over all model years for each cohort
        self.op_emissions = self.op_emissions.reorder_levels(order=['tec', 'seg', 'fleetreg', 'prodyear']) # reorder MultiIndex to add production emissions

        self.LC_emissions = self.op_emissions.add(self.veh_prod_cint)

        self.enr_cint = self._p_dict['ENR_CINT']
        self.enr_cint = self.enr_cint.stack()
        self.enr_cint.index.rename(['enr', 'reg', 'year'], inplace=True)

        log.info('Finished importing results from GAMS run')

    def figure_calculations(self):
        """
        Calculate key results for cross-experiment compraisons.

        Used by main.py to aggregate key indicators across experiments.
        Calculate technology shares in 2030 and 2050, first year of 100% market
        share and years of highest market share for new technologies

        Returns
        -------
        None.
        """
        ####

        try:
            self.LC_emissions_avg = [self.avg_LCemiss(i) for i in range(0, 28)]  # quality check - estimate lifecycle emissions of vehicles

            add_gpby = self.stock_add.sum(axis=1).unstack('seg').unstack('tec')
            self.add_share = add_gpby.div(add_gpby.sum(axis=1), axis=0)
            self.add_share.dropna(how='all', axis=0, inplace=True) # drop production country (no fleet)

            """ Export technology shares in 2030 to evaluate speed of uptake"""
            self.shares_2030 = self.add_share.loc(axis=0)[:,'2030']
            self.shares_2050 = self.add_share.loc(axis=0)[:,'2050']

            """ Export first year of 100% BEV market share """
            tec_shares = self.add_share.stack().stack().sum(level=['prodyear', 'tec','fleetreg'])
            bev_shares = tec_shares.loc[:, 'BEV',:]
            bev_shares = bev_shares.unstack('fleetreg')
            bev_shares = bev_shares.loc['2020':'2050']

            self.highest_BEV_marketshare = bev_shares.idxmax()

            self.full_BEV_year = bev_shares[bev_shares>=0.99]
            self.full_BEV_year = self.full_BEV_year.apply(pd.Series.first_valid_index)

        except Exception as e:
            print('\n ******************************')
            log.error(f'----- Error in calculating scenario results. {e}')


    def avg_LCemiss(self, age=12):
        """ Calculate lifecycle intensity by cohort for quality check.

        Calculate average operating emissions intensity by using total
        lifetime operating emissions by cohort, divided by original stock
        of the cohort

        Returns
        -------
        Average lifecycle emissions.
        """

        """Test calculation for average lifetime vehicle (~12 years)."""
        self.veh_oper_cint_avg = self.veh_oper_cint.index.levels[4].astype(int)
        ind = self.veh_oper_cint.index
        self.veh_oper_cint.index = self.veh_oper_cint.index.set_levels(ind.levels[4].astype(int), level=4) # set ages as int
        self.veh_oper_cint.sort_index(level='age', inplace=True)
        self.veh_oper_cint.sort_index(level='age', inplace=True)
        self.veh_oper_cint_avg = self.veh_oper_cint.reset_index(level='age')
        self.veh_oper_cint_avg = self.veh_oper_cint_avg[self.veh_oper_cint_avg.age<=age] # then, drop ages over lifetime
        self.veh_oper_cint_avg = self.veh_oper_cint_avg.set_index([self.veh_oper_cint_avg.index, self.veh_oper_cint_avg.age])
        self.veh_oper_cint_avg.drop(columns='age', inplace=True)
        self.veh_oper_cint_avg = self.veh_oper_cint_avg.reorder_levels(['tec','enr','seg','fleetreg','age','modelyear','prodyear'])

        self.avg_oper_dist = self.full_oper_dist.reset_index(level='age')
        self.avg_oper_dist = self.avg_oper_dist.astype({'age': 'int32'})
        self.avg_oper_dist = self.avg_oper_dist[self.avg_oper_dist.age <= age]  # again, drop ages over lifetime
        self.avg_oper_dist = self.avg_oper_dist.set_index([self.avg_oper_dist.index, self.avg_oper_dist.age]) # make same index for joining with self.veh_oper_cint_avg
        self.avg_oper_dist.drop(columns='age', inplace=True)
        self.avg_oper_dist = self.avg_oper_dist.reorder_levels(['tec','enr','seg','fleetreg','age','modelyear','prodyear'])
        self.d = self.avg_oper_dist.join(self.veh_oper_cint_avg, lsuffix='_dist')
        self.d.columns=['dist','intensity']
        self.op_emissions_avg = self.d.dist * self.d.intensity
        self.op_emissions_avg.index = self.op_emissions_avg.index.droplevel(level=['enr']) # these columns are unncessary/redundant
        self.op_emissions_avg.to_csv('op_emiss_avg_with_duplicates.csv')
        self.op_emissions_avg = self.op_emissions_avg.reset_index().drop_duplicates().set_index(['tec','seg','fleetreg','age','modelyear','prodyear'])
        self.op_emissions_avg.to_csv('op_emiss_avg_without_duplicates.csv')
        self.op_emissions_avg = self.op_emissions_avg.sum(level=['tec','seg','fleetreg','prodyear']) # sum the operating emissions over all model years
        self.op_emissions_avg = self.op_emissions_avg.reorder_levels(order=['tec','seg','fleetreg','prodyear']) # reorder MultiIndex to add production emissions
        self.op_emissions_avg.columns = ['']


     # Estimate operating emissions by cohort (i.e., prodyear)
        try:
            # sum total operating emissions for each cohort, by region, technology and segment
            operation_em = self.veh_oper_cohort.sum(level=['prodyear', 'fleetreg', 'tec', 'seg']).sum(axis=1)
            operation_em.sort_index(axis=0, level=0, inplace=True)
            op = operation_em.loc['2000':'2050']

            # Calculate initial stock
            init_stock = self.veh_stck_add.loc(axis=1)[0]  # retrieve all stock added (age=0) in cohort year
            init_stock.replace(0, np.nan)
            init_stock.dropna(axis=0, inplace=True)
            init_stock.index = init_stock.index.reorder_levels([3, 2, 0, 1])
            init_stock.sort_index(inplace=True)

            self.op_intensity = op / init_stock
            self.op_intensity.sort_index(inplace=True)
        except Exception as e:
            print('\n*****************************************')
            log.error(f'----- Error in calculating operating emissions by cohort. Perhaps post-processing parameters not loaded? {e}')

        return self.op_emissions_avg.add(self.veh_prod_cint, axis=0)
