# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:27:57 2019

@author: chrishun
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator, IndexLocator,IndexFormatter,LinearLocator)

import seaborn
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

import itertools

import gams
import gmspy

"""class Car:
    def __init__(self, color):
        self.color = color

    @classmethod
    def from_gdx(cls, filepath):
        with open(filepath, "r") as f:
            color = file.read()
        return cls(color)

car = Car.from_gdx("path")"""

class FleetModel:
    """
    Instance of a fleet model experiment
    Attributes:
        el-mix intensities: according to given MESSAGE climate scenario
        transport demand: projected transport demand from MESSAGE, consistent with climate scenario
        A, F: matrices from ecoinvent
        lightweighting: lightweighting correspondance matrix
        battery_specs: static battery specifications from inventories
        fuelcell_specs: ditto for PEMFC

        ?? recycling losses: material-specific manufacturing losses (?)
        fuel scenarios: fuel chain scenarios (fossil, hydrogen)
        occupancy rate: consumer preferences (vehicle ownership) and modal shifts
        battery_density: energy density for traction batteries
        lightweighting_scenario: whether (how aggressively) LDVs are lightweighted in the experiment
    """
    def __init__(self, veh_stck_int_seg=None, tec_add_gradient=None, seg_batt_caps=None, B_term_prod=None, B_term_oper_EOL=None, r_term_factors=0.2,
                 u_term_factors=2025, pkm_scenario='iTEM2-Base', eur_batt_share=0.5, occupancy_rate=1.643, data_from_message=None, gdx_file=None):
#        self.current_path = os.path.dirname(os.path.realpath(__file__))
        " Instantiate filepaths "
        self.home_fp = os.path.dirname(os.path.realpath(__file__))

#        os.chdir(home_fp)

#       self.gdx_file = 'C:\\Users\\chrishun\\Box Sync\\YSSP_temp\\EVD4EUR_input.gdx'#EVD4EUR_ver098.gdx'  # for reading in inputs from gdx file
        self.gms_file = os.path.join(self.home_fp, r'EVD4EUR.gms') # GAMS model file
        self.import_fp = os.path.join(self.home_fp, r'GAMS_input_new.xls')
        self.export_fp = os.path.join(self.home_fp, r'Model run data')
        self.keeper = "{:%d-%m-%y, %H_%M}".format(datetime.now())


        if gdx_file is not None:
            self._from_gdx(gdx_file)
        else:
            try:
                self._from_python(veh_stck_int_seg, tec_add_gradient, seg_batt_caps,
                                  B_term_prod, B_term_oper_EOL, r_term_factors, u_term_factors,
                                  pkm_scenario, eur_batt_share, occupancy_rate, data_from_message)
#                self.B_prod = B_term_prod # not currently used
#                self.B_oper = B_term_oper_EOL # not currently used
            except AttributeError as err:
                print(f"Exception: {err}")
                print("Generating empty fleet object")

    def _from_python(self, veh_stck_int_seg, tec_add_gradient, seg_batt_caps,
                     B_term_prod, B_term_oper_EOL, r_term_factors, u_term_factors,
                     pkm_scenario, eur_batt_share, occupancy_rate, data_from_message):

        """--- For instantiating Fleet object using Python ---"""
        """ static input data.....hardcoded and/or read in from Excel? """
        self.battery_specs = pd.DataFrame() # possible battery_sizes (and acceptable segment assignments, CO2 production emissions, critical material content, mass)
        self.fuelcell_specs = pd.DataFrame() # possible fuel cell powers (and acceptable segment assignments, CO2 production emissions, critical material content, fuel efficiency(?), mass)
        self.lightweighting = pd.DataFrame() # lightweighting data table - lightweightable materials and coefficients for corresponding lightweighting material(s)
        self.tec_add_gradient = tec_add_gradient

        if data_from_message is not None:
            self.el_intensity = data_from_message # regional el-mix intensities as time series from MESSAGE
            self.trsp_dem = data_from_message # EUR transport demand as time series from MESSAGE
        """ boundary conditions for constraints, e.g., electricity market supply constraints, crit. material reserves? could possibly belong in experiment specifications as well..."""

        """ GAMS-relevant attributes"""
        #  --------------- GAMS sets / domains -------------------------------
        self.tecs = ['ICE', 'BEV']                               # drivetrain technologies; can include e.g., alternative battery chemistries
        self.modelyear = [str((2000)+i) for i in range(81)]
        self.inityear=[str(2000+i) for i in range(21)]          # reduce to one/five year(s)? Originally 2000-2020
        self.cohort = [str((2000-28)+i) for i in range(81+28)]  # vehicle cohorts (production year)
        self.optyear = [str(2020+i) for i in range(61)]
        self.age = [str(i) for i in range(28)]                  # vehicle age, up to 27 years old
#        self.age = [str(i) for i in range(11)]
        self.enr = ['ELC', 'FOS']                                # fuel types; later include H2,
        self.seg = ['A', 'B', 'C', 'D', 'E', 'F']                    # From ACEA: Small, lower medium, upper medium, executive
        self.reg = ['LOW','II','MID','IV', 'HIGH', 'PROD']#['1', '2', '3', '4', '5', '6'] # study regions
        self.demeq= ['STCK_TOT', 'OPER_DIST', 'OCUP']             # definition of
        self.dstvar=['mean', 'stdv']
        self.enreq=['CINT']
        self.grdeq=['IND', 'ALL']
        self.veheq = ['PROD_EINT', 'PROD_CINT_CSNT', 'OPER_EINT', 'EOLT_CINT']
        self.lfteq = ['LFT_DISTR', 'AGE_DISTR']
        self.sigvar = ['A', 'B', 'r', 'u']                         # S-curve terms
        self.critmats = ['Cu', 'Li', 'Co', 'Pt', '', '']           # critical elements to count for; to incorporate later
        self.age_int = list(map(int,self.age))

        # --------------- GAMS Parameters -------------------------------------

        """needs to be made in terms of tec as well??"""
        """Currently uses generalized logistic growth curve"""

        """ Currently uses smoothed total vehicle stock instead of stock from MESSAGE-Transport, which swings widely """
        # self.veh_stck_tot = pd.DataFrame(pd.read_excel(self.import_fp, sheet_name='VEH_STCK_TOT', header=None, usecols='A,C', skiprows=[0])) # usecols='A:B' for MESSAGE data, usecols='A,C' for old data
        self.veh_stck_tot = pd.DataFrame(pd.read_excel(self.import_fp, sheet_name='temp_veh_stck_tot', header=None, usecols='A:C', skiprows=[0]))
#        self.veh_stck_tot = self._process_series(self.veh_stck_tot)
        self.veh_stck_tot = self._process_df_to_series(self.veh_stck_tot)

        # "Functional unit" # TODO: this is redundant
        # Eurostat road_tf_veh [vkm]
        # Eurostat road_tf_vehage [vkm, cohort] NB: very limited geographic spectrum
        # Eurostat road_pa_mov [pkm]
        # http://www.odyssee-mure.eu/publications/efficiency-by-sector/transport/distance-travelled-by-car.html
        self.occupancy_rate = occupancy_rate or 1.643 #convert to time-dependent parameter #None # vkm -> pkm conversion
        self.all_pkm_scenarios = pd.DataFrame(pd.read_excel(self.import_fp, sheet_name='pkm', header=[0], index_col=[0])).T
        self.passenger_demand = self.all_pkm_scenarios[pkm_scenario] # retrieve pkm demand from selected scenario
        self.passenger_demand.reset_index()
#        self.passenger_demand = pd.DataFrame(pd.read_excel(self.import_fp,sheet_name='VEH_STCK_TOT',header=None,usecols='A,G',skiprows=[0])) #hardcoded retrieval of pkm demand
#        self.passenger_demand = self._process_df_to_series(self.passenger_demand)
        self.passenger_demand = self.passenger_demand * 1e9
        self.passenger_demand.name = ''
        # self.passenger_demand.index = self.veh_stck_tot.index

        self.fleet_vkm = self.passenger_demand/self.occupancy_rate

#        self.veh_oper_dist = self.fleet_vkm/self.veh_stck_tot
        self.veh_oper_dist = pd.Series([10000 for i in range(0,len(self.fleet_vkm))], index=[str(i) for i in range(2000,2081)])
#        self.veh_oper_dist = pd.Series([-97.052*i+207474 for i in range(2000,2051)],index=[str(i) for i in range(2000,2051)])
        self.veh_oper_dist.index.name='year'
        # [years] driving distance each year # TODO: rename?

        self.veh_stck_int_seg = veh_stck_int_seg or [0.08, 0.21, 0.27, 0.08, 0.03, 0.34]  # Shares from 2017, ICCT report
        self.veh_stck_int_seg= pd.Series(self.veh_stck_int_seg,index=self.seg)

        self.seg_batt_caps = pd.Series(seg_batt_caps,index = self.seg) # For battery manufacturing capacity constraint
        self.eur_batt_share = eur_batt_share or 0.5

        ################ Life cycle intensities ################
#        """These factors are usually calculated using the general logistic function"""
#        self.veh_prod_cint_csnt = pd.DataFrame(pd.read_excel(self.import_fp,sheet_name='VEH_PROD_CINT_CSNT',header=None,usecols='A:D',skiprows=[0]))
#        self.veh_prod_cint_csnt = self._process_df_to_series(self.veh_prod_cint_csnt)

#        self.veh_prod_eint = pd.DataFrame(pd.read_excel(self.import_fp,sheet_name='VEH_PROD_EINT',header=None,usecols='A:D',skiprows=[0]))
#        self.veh_prod_eint = self._process_df_to_series(self.veh_prod_eint)

#        self.veh_prod_cint = pd.DataFrame(pd.read_excel('GAMS_input_new.xls',sheet_name='VEH_PROD_CINT',header=None,usecols='A:D',skiprows=[0]))  # [tecs, cohort]
#        self.veh_prod_cint = self._process_df_to_series(self.veh_prod_cint)
#
#        self.veh_oper_eint = pd.DataFrame(pd.read_excel(self.import_fp,sheet_name='VEH_OPER_EINT',header=None,usecols='A:D',skiprows=[0]))  # [[tecs, enr], cohort]
#        self.veh_oper_eint = self._process_df_to_series(self.veh_oper_eint)
#        pd.read_excel(self.import_fp,sheet_name='fuel economy', usecols='A:G',skiprows=23,index_col=[0],nrows=14)
#        self.veh_oper_cint = pd.DataFrame(pd.read_excel('GAMS_input_new.xls',sheet_name='VEH_OPER_CINT',header=None,usecols='A:E',skiprows=[0]))  # [[tecs, enr], cohort]
#        self.veh_oper_cint = self._process_df_to_series(self.veh_oper_cint)
#
        """Trial for calculating general logistic function in-code"""
        """self.veh_partab = pd.DataFrame(pd.read_excel(self.import_fp,sheet_name='VEH_PARTAB',header=None,usecols='A:D',skiprows=[0]))
        self.veh_partab = self._process_df_to_series(self.veh_partab)
        print(self.veh_partab)
        self.trial_oper_eint = self.veh_partab['OPER_EINT']
        self.oper_eint = self.trial_oper_eint['A']+(trial_oper_eint['B']-trial_oper_eint['A'])/(1+exp(-trial_oper_eint['r']*(self.year-trial_oper_eint['u'])))
        #A, B, r, u"""
#
#        self.veh_eolt_cint = pd.DataFrame(pd.read_excel(self.import_fp,sheet_name='VEH_EOLT_CINT',header=None,usecols='A:D',skiprows=[0]))  # [[tecs, enr], cohort]
#        self.veh_eolt_cint = self._process_df_to_series(self.veh_eolt_cint)  # [tecs, cohort]

        ################ Fleet dynamics ################
        """VEH_LIFT_CDF(age) = cdfnormal(AGE_PAR(age),LFT_PARTAB('mean'),LFT_PARTAB('stdv'));
        VEH_LIFT_AGE(age) = (1 - VEH_LIFT_CDF(age))/sum(agej, VEH_LIFT_CDF(agej)) ;
        VEH_LIFT_MOR(age)$(ord(age)< 20) = 1 - VEH_LIFT_AGE(age+1)/VEH_LIFT_AGE(age);
        VEH_LIFT_MOR(age)$(ord(age)= 20) = 1"""
        self.avg_age = 11.1 # From ACEA 2019-2020 report
        self.std_dev_age = 2.21
#        self.avg_age = 11
#        self.std_dev_age = 0
        self.veh_lift_cdf = pd.Series(norm.cdf(self.age_int, self.avg_age, self.std_dev_age), index=self.age)#pd.Series(pd.read_pickle(self.import_fp+'input.pkl'))#pd.DataFrame()  # [age] TODO Is it this one we feed to gams?
        self.veh_lift_cdf.index = self.veh_lift_cdf.index.astype('str')

        self.veh_lift_age = pd.Series(1 - self.veh_lift_cdf)     # [age] # probability of car of age x to die in current year

        #lifetime = [1-self.veh_lift_age[i+1]/self.veh_lift_age[i] for i in range(len(self.age)-1)]
        self.veh_lift_pdf = pd.Series(calc_steadystate_vehicle_age_distributions(self.age_int, self.avg_age, self.std_dev_age), index=self.age)   # idealized age PDF given avg fleet age and std dev
        self.veh_lift_pdf.index = self.veh_lift_pdf.index.astype('str')

        self.veh_lift_mor = pd.Series(calc_probability_of_vehicle_retirement(self.age_int, self.veh_lift_pdf), index=self.age)
        self.veh_lift_mor.index = self.veh_lift_mor.index.astype('str')


        # Initial stocks
        """# Eurostat road_eqs_carpda[tec]
        # Eurostat road_eqs_carage [age - <2, 2-5, 5-10, 10-20];
        # ACEA [age in year divisions up to 10 years]
        # Also check pb2018-section2016.xls for more cohesive, EU28++ data"""
        self.veh_stck_int = pd.DataFrame(pd.read_excel(self.import_fp, sheet_name='VEH_STCK_INT', header=None, usecols='A:D', skiprows=[0]))  # [tec, age]
        self.veh_stck_int = self._process_df_to_series(self.veh_stck_int)

        BEV_int_shr = 0.0018  # from Eurostat; assume remaining is ICE
        self.veh_stck_int_tec = pd.Series([1-BEV_int_shr, BEV_int_shr], index=['ICE', 'BEV'])

        ################ filters and parameter aliases ################
        self.enr_veh = pd.DataFrame(pd.read_excel(self.import_fp, sheet_name='ENR_VEH', header=None, usecols='A:C', skiprows=[0]))            # [enr, tec]
        self.enr_veh = self._process_df_to_series(self.enr_veh)

        self.veh_pay = pd.DataFrame(pd.read_excel(self.import_fp, sheet_name='VEH_PAY', header=None, usecols='A:D', skiprows=[0]))            # [cohort, age, year]
        self.veh_pay = self._process_df_to_series(self.veh_pay)

        self.age_par = pd.Series([float(i) for i in self.age])
        self.age_par.index = self.age_par.index.astype('str')

        self.year_par = pd.Series([float(i) for i in self.cohort], index=self.cohort)
        self.year_par.index = self.year_par.index.astype('str')

        self.prodyear_par = pd.Series([int(i) for i in self.cohort], index=self.cohort)
        self.prodyear_par.index = self.prodyear_par.index.astype('str')

        # Temporary introduction of seg-specific VEH_PARTAB from Excel; will later be read in from YAML
#        self.veh_partab = pd.DataFrame(pd.read_excel(self.import_fp,sheet_name = 'genlogfunc',usecols='A:G',index_col=[0,1,2],skipfooter=6)).stack()
        self.veh_partab = self.build_veh_partab(B_term_prod, B_term_oper_EOL, r_term_factors, u_term_factors)#.stack()
        """" if modify_b_ice or modify_b_bev:
                self.veh_partab.loc[:, 'ICE',:, 'B']=self.veh_partab.loc[:, 'ICE',:, 'A'].values*modify_b_ice
                self.veh_partab.loc[:, 'BEV',:, 'B'] = self.veh_partab.loc[:, 'BEV',:, 'A'].values*modify_b_bev"""

        """"if BEV_batt ==30:
            self.veh_partab.loc['PROD_EINT', 'BEV',:,:]=pd.DataFrame(array, index=['A', 'B', 'r', 'u'])
            self.veh_partab.loc['PROD_CINT_CSNT', 'BEV',:,:]=pd.DataFrame(array, index=['A', 'B', 'r', 'u'])"""
#        self.veh_partab.index = self.veh_partab.index.astype('str')
#        self.veh_partab = self._process_df_to_series(self.veh_partab)
        """# ACEA.be has segment division for Western Europe
        # https://www.acea.be/statistics/tag/category/segments-body-country
        # More detailed age distribution (https://www.acea.be/uploads/statistic_documents/ACEA_Report_Vehicles_in_use-Europe_2018.pdf)"""

        self.tec_add_gradient = tec_add_gradient or 0.2

        self.growth_constraint = 0#growth_constraint
        self.gro_cnstrnt = [self.growth_constraint for i in range(len(self.modelyear))]
        self.gro_cnstrnt = pd.Series(self.gro_cnstrnt, index=self.modelyear)
        self.gro_cnstrnt.index = self.gro_cnstrnt.index.astype('str')

        self.manuf_cnstrnt = pd.read_excel(self.import_fp, sheet_name='MANUF_CONSTR', header=None, usecols='A,B', skiprows=[0])  # Assumes stabilized manufacturing capacity post-2030ish
#        self.manuf_cnstrnt = pd.read_excel(self.import_fp,sheet_name='MANUF_CONSTR',header=None,usecols='A,C',skiprows=[0]) # Assumes continued (linear) growth in manufacturing capacity until end of model period
#        self.manuf_cnstrnt = pd.read_excel(self.import_fp,sheet_name='MANUF_CONSTR',header=None,usecols='A,D',skiprows=[0]) # Assumes continued (linear) growth in manufacturing capacity until 2050

        self.manuf_cnstrnt = self._process_df_to_series(self.manuf_cnstrnt)
#        self.manuf_cnstrnt.index = self.manuf_cnstrnt.index.astype('str')
        self.manuf_cnstrnt = self.manuf_cnstrnt * self.eur_batt_share

        self.enr_partab = pd.read_excel(self.import_fp, sheet_name='ENR_PARTAB', usecols='A:G', index_col=[0, 1, 2]) #enr,reg,X

        # read in electricity pathways from electricity_clustering.py (from MESSSAGE)
        self.enr_cint = pd.read_csv(os.path.join(self.home_fp, 'Data', 'el_footprints_pathways.csv'), index_col=[0, 1])
        self.enr_cint.columns = self.enr_cint.columns.astype('int64')
        # insert years and interpolate between decades to get annual resolution
        for decade in self.enr_cint.columns:
            for i in np.arange(1, 10):
                self.enr_cint[decade + i] = np.nan
        self.enr_cint[2019] = self.enr_cint[2020]  # # to-do: fill with historical data
        self.enr_cint = self.enr_cint.astype('float64').sort_index(axis=1).interpolate(axis=1) # sort year orders and interpolate
        self.enr_cint.columns = self.enr_cint.columns.astype(str)  # set to strings for compatibility with GAMS

        self.enr_cint = self.enr_cint.stack()  # reg, enr, year

        # complete enr_cint parameter with fossil fuel chain and electricity in production regions
        # using terms for general logisitic function
        for label, row in self.enr_partab.iterrows():
            A = row['A']
            B = row['B']
            r = row['r']
            u = row['u']
            reg = label[1]
            enr = label[0]
            for t in [((2000)+i) for i in range(81)]:
                self.enr_cint.loc[(reg, enr, str(t))] = A + (B-A)/(1+np.exp(-r*(t-u)))
        print(self.enr_cint)

        self.enr_cint = self.enr_cint.swaplevel(0,1) # enr, reg, year

        # --------------- Expected GAMS Outputs ------------------------------

        self.totc = 0
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

        """ Optimization Initialization """
        """self.ws = gams.GamsWorkspace(working_directory=self.current_path,debug=2)
        self.db = self.ws.add_database()#database_name='pyGAMSdb')
        self.opt = self.ws.add_options()
#        self.opt.DumpParms = 2
        self.opt.ForceWork = 1"""
#        self.opt.SysOut = 1

    def _from_gdx(self, gdx_file):
        # Build fleet object from gdx file (contains model inputs and outputs)
        # For visualization
        self.sets = gmspy.ls(gdx_filepath=gdx_file, entity='Set')
        ws = gams.GamsWorkspace()
        db = ws.add_database_from_gdx(gdx_file)
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

    def main(self):
        #
        pass

    @staticmethod
    def _process_df_to_series(df):
        """"--- process DataFrames to MultIndexed Series for exporting to GAMS ---"""
        dims = df.shape[1] - 1 # assumes unstacked format
        indices = df.columns[:-1].tolist()
        df.set_index(indices, inplace=True)

        temp=[]
        for i in range(dims):
            temp.append(df.index.get_level_values(i).astype(str))
        df.index = temp
        df.columns = ['']
        df.index.names = [''] * dims
        df = pd.Series(df.iloc[:, 0])
        return df

    @staticmethod
    def _process_series(ds):
        """--- currently not used; deprecate? ---"""
        ds.index = ds.index.astype(str)
        ds.name = ''
        ds.columns = ['']
        ds.index.rename('', inplace=True)
        return ds

#        dims = df.shape[1]-1 # assumes unstacked format
#        indices = df.columns[:-1].tolist()
#        df.set_index(indices,inplace=True)
#
#        temp=[]
#        for i in range(dims):
#            temp.append(df.index.get_level_values(i).astype(str))
#        df.index = temp
#        df.columns=['']
#        df.index.names = ['']*dims
#        df = pd.Series(df.iloc[:, 0])
#        return df


    def read_all_sets(self, gdx_file):
        """--- No longer used after commit c941039 as sets are now internally defined ---"""

#        db = gmspy._iwantitall(None, None, gdx_file)
#        self.tecs = gmspy.set2list('tec', db)
#        self.cohort = gmspy.set2list('year', db)
#        self.age = gmspy.set2list('age', db)
#        self.enr = gmspy.set2list('enr', db)

#         spy.param2series('VEH_PAY', db) # series, otherwise makes giant sparse dataframe
        pass

    def get_output_from_GAMS(self, gams_db, output_var):
        """ DEPRECATED. Replaced by v_dict functionality in import_model_results
        Import variable results from GAMS database
        Follows variable structure in GAMS, i.e., uses Multi-Index Series"""
        temp_GMS_output = []
        temp_index_list = []

        for rec in gams_db[output_var]:
           if gams_db[output_var].number_records == 1: # special case for totc
               temp_output_df = gams_db[output_var].first_record().level
               return temp_output_df

           dict1 = {}
           dict1.update({'level': rec.level})
           temp_GMS_output.append(dict1)
           temp_index_list.append(rec.keys)
        temp_domain_list = list(gams_db[output_var].domains_as_strings)
        temp_index = pd.MultiIndex.from_tuples(temp_index_list, names=temp_domain_list)
        temp_output_df = pd.DataFrame(temp_GMS_output, index=temp_index)

        return temp_output_df

    def read_gams_db(self, gams_db):
        """ Fetch sets, parameters, variables and equations from GAMS database"""
        sets = gmspy.ls(db=gams_db, entity='Set')
        parameters = gmspy.ls(db=gams_db, entity='Parameter')
        variables = gmspy.ls(db=gams_db, entity='Variable')
        equations = gmspy.ls(db=gams_db, entity='Equation')

        # Export parameters
        self._p_dict = {}
        for p in parameters:
            try:
                self._p_dict[p] = gmspy.param2df(p, db=gams_db)
#            except ValueError:
#                try:
#                    self._p_dict[p] = gmspy.param2series(p, db=gams_db)
##                    print('param2series')
#                except:
#                    pass
#                print(f'Warning!: p_dict ValueError in {p}!')
#                pass
            except AttributeError:
                print(f'Warning!: p_dict AttributeError in {p}! Probably no records for this parameter.')
                pass

        # Export variables
        self._v_dict = {}
        for v in variables:
            try:
                self._v_dict[v] = gmspy.var2df(v, db=gams_db)
#            except ValueError:
#                try:
#                    self._v_dict[v] = gmspy.var2series(v, db=gams_db)
            except ValueError:
                if len(gams_db[v]) == 1: # special case for totc
                   self._v_dict[v] = gams_db[v].first_record().level
                else:
                    print(f'Warning!: v_dict ValueError in {v}!')
                pass
            except TypeError: # This is specifically for seg_add
                print(f'Warning! v-dict TypeError in {v}! Probably no records for this variable.')
                pass

        self._e_dict={}
        for e in equations:
            try:
                self._e_dict[e] = gmspy.eq2series(e, db=gams_db)
            except ValueError:
                if len(gams_db[e]) == 1: # special case for totc
                       self._e_dict[e] = gams_db[e].first_record().level
                else:
                    print(f'Warning!: e_dict ValueError in {e}!')
            except:
                print(f'Warning!: Error in {e}')
                print(sys.exc_info()[0])
                pass

    def import_model_results(self):
        """--- Import calculated parameters and variables from GDX for visualization ---"""
        def reorder_age_headers(df_unordered):
            temp = df_unordered
            temp.columns = temp.columns.astype(int)
            temp.sort_index(inplace=True, axis=1)
            return temp

        """--- Import the parameters that are calculated within the GAMS model ---"""
        self.veh_prod_cint = self._p_dict['VEH_PROD_CINT']
        self.veh_prod_cint = self.veh_prod_cint.stack().to_frame()
        self.veh_prod_cint.index.rename(['tec', 'seg', 'prodyear'], inplace=True)

        self.veh_oper_eint = self._p_dict['VEH_OPER_EINT']
        self.veh_oper_eint = self.veh_oper_eint.stack().to_frame()
        self.veh_oper_eint.index.rename(['tec', 'seg', 'year'], inplace=True)

        self.veh_oper_cint = self._p_dict['VEH_OPER_CINT']
        self.veh_oper_cint = self.veh_oper_cint.stack().to_frame()
        self.veh_oper_cint.index.names = ['tec', 'enr', 'seg', 'reg', 'age', 'modelyear', 'prodyear']

        """--- Import post-processing parameters ---"""
        self.veh_oper_cohort = self._p_dict['VEH_OPER_COHORT']
        self.veh_oper_cohort.index.rename(['tec', 'seg', 'reg', 'prodyear', 'modelyear'], inplace=True)
        self.veh_stock_cohort = self._p_dict['VEH_STCK_CHRT']

        """--- Import model results ---"""
        self.veh_stck_delta = self._v_dict['VEH_STCK_DELTA']
        self.veh_stck_add = self._v_dict['VEH_STCK_ADD']
        self.veh_stck_rem = self._v_dict['VEH_STCK_REM']
        self.veh_stck = self._v_dict['VEH_STCK']
        self.veh_totc = self._v_dict['VEH_TOTC']
        self.annual_totc = self.veh_totc.sum(axis=0)#self.veh_totc.unstack('year').sum()

#        self.totc = self._v_dict['TOTC']
        self.totc_opt = self._v_dict['TOTC_OPT']

        self.veh_prod_totc = self._v_dict['VEH_PROD_TOTC']
        self.veh_oper_totc = self._v_dict['VEH_OPER_TOTC']
        self.total_op_emissions = self.veh_oper_totc.sum(axis=0)#.unstack('year').sum()
        self.veh_eolt_totc = self._v_dict['VEH_EOLT_TOTC']

        self.emissions = pd.concat([self.veh_prod_totc.stack(), self.veh_oper_totc.stack(), self.veh_eolt_totc.stack()], axis=1)
        self.emissions.columns = ['Production', 'Operation', 'End-of-life']
        self.emissions = self.emissions.unstack(['tec', 'year']).sum().unstack([None, 'tec'])

        """--- Prepare model output dataframes for visualization ---"""
        self.stock_df = self._v_dict['VEH_STCK']
        self.stock_df2 = self._v_dict['VEH_STCK']
        self.stock_df = reorder_age_headers(self.stock_df)
        self.stock_add = self._v_dict['VEH_STCK_ADD']
        self.stock_add = reorder_age_headers(self.stock_add)
        self.stock_add = self.stock_add.dropna(axis=1, how='any')
        self.stock_add.index.rename(['tec', 'seg', 'reg', 'prodyear'], inplace=True)
        self.stock_df_plot = self.stock_df.stack().unstack('age')
        self.stock_df_plot = reorder_age_headers(self.stock_df_plot)

        self.stock_df_plot_grouped = self.stock_df_plot.groupby(['tec', 'seg'])

        self.stock_cohort = self._p_dict['VEH_STCK_CHRT']
        self.stock_cohort.index.rename(['tec', 'seg', 'reg', 'prodyear', 'age'], inplace=True)
        self.stock_cohort.columns.rename('modelyear', inplace=True)
        self.stock_cohort = self.stock_cohort.droplevel(level='age', axis=0)
        self.stock_cohort = self.stock_cohort.stack().unstack('prodyear').sum(axis=0, level=['tec', 'reg', 'modelyear'])#.unstack('modelyear')

        self.veh_oper_dist = self._p_dict['VEH_OPER_DIST']
        self.veh_oper_dist.index = self.veh_oper_dist.index.get_level_values(0) # recast MultiIndex as single index

        self.full_oper_dist = self.veh_oper_dist.reindex(self.veh_oper_cint.index, level='modelyear')
        self.op_emissions = self.veh_oper_cint.multiply(self.full_oper_dist)#.to_frame())
        self.op_emissions.index = self.op_emissions.index.droplevel(level=['enr', 'age']) # these columns are unncessary/redundant
        self.op_emissions = self.op_emissions.sum(level=['tec', 'seg', 'reg', 'prodyear']) # sum the operating emissions over all model years
        self.op_emissions = self.op_emissions.reorder_levels(order=['tec', 'seg', 'reg', 'prodyear']) # reorder MultiIndex to add production emissions

        self.LC_emissions = self.op_emissions.add(self.veh_prod_cint)

        add_gpby = self.stock_add.sum(axis=1).unstack('seg').unstack('tec')
        self.add_share = add_gpby.div(add_gpby.sum(axis=1), axis=0)

        " Export technology shares in 2030 to evaluate speed of uptake"
        self.shares_2030 = self.add_share.loc(axis=0)[:, '2030']#.to_string()
        self.shares_2050 = self.add_share.loc(axis=0)[:, '2050']

        self.enr_cint = self._p_dict['ENR_CINT']
        self.enr_cint = self.enr_cint.stack()
        self.enr_cint.index.rename(['enr', 'reg', 'year'], inplace=True)

        print('\n Finished importing results from GAMS run')

    def build_BEV(self):
        """ Specify battery size for each segment and calculate resulting production emissions"""
        self.lookup_table = pd.read_excel(self.import_fp, sheet_name='Sheet6', header=[0, 1], index_col=0, nrows=3) # fetch battery portfolio
        self.prod_df = pd.DataFrame()

        # assemble production emissions for battery for defined battery capacities
        for key,value in self.seg_batt_caps.items():
            self.prod_df[key] = self.lookup_table[key, value]
        mi = pd.MultiIndex.from_product([self.prod_df.index.to_list(), ['BEV'], ['batt']])
        self.prod_df.index = mi
        self.prod_df = self.prod_df.stack()
        self.prod_df.index.names = ['veheq', 'tec', 'comp', 'seg']
        self.prod_df.index = self.prod_df.index.swaplevel(i=-2, j=-1)

#        self.oper_df =
#        body_weight = [923,np.average(923,1247),1247,1407,average(1407,1547),1547]
        return self.prod_df

    def BEV_weight(self):
        """ Use this function to calculate weight-based operation energy requirements"""
        pass


    def build_veh_partab(self,B_term_prod,B_term_oper_EOL,r_term_factors,u_term_factors):
        """ Builds the VEH_PARTAB parameter for GAMS, which contains the A, B, r and u terms for the sigmoid functions"""
        """ TO DO: separate A-terms for battery and rest-of-vehicle and apply different b-factors"""
        self.A_terms_raw = pd.read_excel(self.import_fp, sheet_name='genlogfunc', header=[0], index_col=[0,1,2], usecols='A:F', nrows=48)
        self.A_terms_raw.columns.names=['comp']
        self.A_terms_raw = self.A_terms_raw.stack().to_frame('a')

        # Retrieve production emission factors for chosen battery capacities and place in raw A factors (with component resolution)
        self.batt_list = self.build_BEV()
        for index, value in self.prod_df.iteritems():
            self.A_terms_raw.loc[index, 'a'] = value


        # Get input for B-multiplication factors (relative to A) from YAML file
        reform = {(firstKey, secondKey, thirdKey): values for firstKey, secondDict in B_term_prod.items() for secondKey, thirdDict in secondDict.items() for thirdKey, values in thirdDict.items()}
        mi = pd.MultiIndex.from_tuples(reform.keys())
        self.temp_prod_df = pd.DataFrame()
        self.temp_oper_df = pd.DataFrame()
        self.temp_df = pd.DataFrame()

        self.b_prod = pd.DataFrame(reform.values(), index = mi)
        self.b_prod.index.names = ['veheq', 'tec', 'comp']

        # Apply B-multiplication factors to production A-factors (with component resolution)
        self.temp_a = self.A_terms_raw.join(self.b_prod,on=['veheq', 'tec', 'comp'], how='left')
        self.temp_prod_df['B'] = self.temp_a['a'] * self.temp_a[0]
        self.temp_prod_df.dropna(how='any', axis=0, inplace=True)

        # Apply B-multiplication factors for operation and EOL A-factors
        reform = {(firstKey, secondKey): values for firstKey, secondDict in B_term_oper_EOL.items() for secondKey, values in secondDict.items()}
        mi = pd.MultiIndex.from_tuples(reform.keys())
        self.b_oper = pd.DataFrame(reform.values(), index=mi, columns=['b'])
#
        self.temp_oper_df = self.A_terms_raw.join(self.b_oper,on=['veheq', 'tec'], how='left')
        self.temp_oper_df['B'] = self.temp_oper_df['a'] * self.temp_oper_df['b']
        self.temp_oper_df.dropna(how='any', axis=0, inplace=True)
        self.temp_oper_df.drop(columns=['a', 'b'], inplace=True)

        # Aggregate component A values for VEH_PARTAB parameter
        self.A = self.A_terms_raw.sum(axis=1)
        self.A = self.A.unstack(['comp']).sum(axis=1)
        self.A.columns = ['A']

        # Begin building final VEH_PARTAB parameter table
        self.temp_df['A'] = self.A
        self.B = pd.concat([self.temp_prod_df, self.temp_oper_df], axis=0).dropna(how='any', axis=1)
        self.B = self.B.unstack(['comp']).sum(axis=1)
        self.temp_df['B'] = self.B

        # Add same r values across all technologies...can add BEV vs ICE resolution here
        temp_r = pd.DataFrame.from_dict(r_term_factors, orient='index', columns=['r'])
        self.temp_df = self.temp_df.join(temp_r, on=['tec'], how='left')
#        self.temp_df['r'] = r_term_factors

        # Add technology-specific u values
        temp_u = pd.DataFrame.from_dict(u_term_factors, orient='index', columns=['u'])
        self.temp_df = self.temp_df.join(temp_u, on=['tec'], how='left')

#        self.temp_df.drop(labels=0,axis=1,inplace=True)
#        self.temp_df.index.names=[None,None,None]

        return self.temp_df

### NB:  _load_exeperiment_data_in_gams moved to gams_runner
        """
    def _load_experiment_data_in_gams(self,filename): # will become unnecessary as we start calculating/defining sets and/or parameters within the class
        years = gmspy.list2set(self.db,self.cohort, 'year')
        modelyear = gmspy.list2set(self.db,self.modelyear, 'modelyear')
        tecs = gmspy.list2set(self.db, self.tecs, 'tec')
        #cohort = gmspy.list2set(self.db, self.cohort, 'prodyear') ## prodyear is an alias of year, not a set of its own
        age = gmspy.list2set(self.db, self.age, 'age')
        enr = gmspy.list2set(self.db, self.enr, 'enr')
        seg = gmspy.list2set(self.db, self.seg, 'seg')
        demeq =  gmspy.list2set(self.db, self.demeq, 'demeq')
        dstvar = gmspy.list2set(self.db,self.dstvar, 'dstvar')
        enreq = gmspy.list2set(self.db,self.enreq, 'enreq')
        grdeq = gmspy.list2set(self.db,self.grdeq, 'grdeq')
        inityear = gmspy.list2set(self.db,self.inityear, 'inityear')
        lfteq = gmspy.list2set(self.db,self.lfteq, 'lfteq')
        sigvar = gmspy.list2set(self.db,self.sigvar, 'sigvar')
        veheq = gmspy.list2set(self.db, self.veheq, 'veheq')
        optyear = gmspy.list2set(self.db,self.optyear, 'optyear')

        veh_oper_dist = gmspy.df2param(self.db, self.veh_oper_dist, ['year'], 'VEH_OPER_DIST')
        veh_stck_tot = gmspy.df2param(self.db, self.veh_stck_tot, ['year'], 'VEH_STCK_TOT')
        veh_stck_int_seg = gmspy.df2param(self.db,self.veh_stck_int_seg,['seg'], 'VEH_STCK_INT_SEG')
        bev_capac = gmspy.df2param(self.db,self.seg_batt_caps,['seg'], 'BEV_CAPAC')
#        veh_seg_int = gmspy.df2param(self.db,self.veh_seg_int,['seg'], 'VEH_SEG_INT')

#        veh_prod_cint = gmspy.df2param(self.db, self.veh_prod_cint, ['tec', 'seg', 'prodyear'], 'VEH_PROD_CINT')
#        veh_prod_cint_csnt = gmspy.df2param(self.db,self.veh_prod_cint_csnt,['tec', 'seg', 'prodyear'], 'VEH_PROD_CINT_CSNT')
#        veh_prod_eint = gmspy.df2param(self.db,self.veh_prod_eint,['tec', 'seg', 'prodyear'], 'VEH_PROD_EINT')

#        veh_oper_eint = gmspy.df2param(self.db, self.veh_oper_eint, ['tec', 'seg', 'prodyear'], 'VEH_OPER_EINT')
#        veh_oper_cint = gmspy.df2param(self.db, self.veh_oper_cint, ['tec', 'enr', 'seg', 'prodyear'], 'VEH_OPER_CINT')
#        veh_eolt_cint = gmspy.df2param(self.db, self.veh_eolt_cint, ['tec', 'seg', 'prodyear'], 'VEH_EOLT_CINT')

        veh_lift_cdf = gmspy.df2param(self.db, self.veh_lift_cdf, ['age'], 'VEH_LIFT_CDF')
        veh_lift_pdf = gmspy.df2param(self.db, self.veh_lift_pdf, ['age'], 'VEH_LIFT_PDF')
        veh_lift_age = gmspy.df2param(self.db, self.veh_lift_age, ['age'], 'VEH_LIFT_AGE')
        veh_lift_mor = gmspy.df2param(self.db, self.veh_lift_mor, ['age'], 'VEH_LIFT_MOR' )

        ######  OBS: Originally calculated using VEH_STCK_INT_TEC, VEH_LIFT_AGE, VEH_STCK_TOT
        veh_stck_int = gmspy.df2param(self.db, self.veh_stck_int, ['tec', 'seg', 'age'], 'VEH_STCK_INT')
        veh_stck_int_tec = gmspy.df2param(self.db,self.veh_stck_int_tec,['tec'], 'VEH_STCK_INT_TEC')

        enr_veh = gmspy.df2param(self.db, self.enr_veh, ['enr', 'tec'], 'ENR_VEH')

        veh_pay = gmspy.df2param(self.db, self.veh_pay, ['prodyear', 'age', 'year'], 'VEH_PAY')

        #age_par = gmspy.df2param(self.db,self.age_par, ['age'], 'AGE_PAR')
        year_par = gmspy.df2param(self.db,self.year_par, ['year'], 'YEAR_PAR')
        veh_partab = gmspy.df2param(self.db,self.veh_partab,['veheq', 'tec', 'seg', 'sigvar'], 'VEH_PARTAB')

        veh_add_grd = self.db.add_parameter_dc('VEH_ADD_GRD', ['grdeq', 'tec'])
        for keys,value in iter(self.veh_add_grd.items()):
            veh_add_grd.add_record(keys).value = value

#        veh_add_grd = gmspy.df2param(self.db,self.veh_add_grd, ['grdeq', 'tec'], 'VEH_ADD_GRD')

        gro_cnstrnt = gmspy.df2param(self.db, self.gro_cnstrnt,['year'], 'GRO_CNSTRNT')

        enr_partab = gmspy.df2param(self.db,self.enr_partab,['enr', 'enreq', 'sigvar'], 'ENR_PARTAB')

        print('exporting database...'+filename+'_input')
        self.db.suppress_auto_domain_checking = 1
        self.db.export(os.path.join(self.current_path,filename+'_input'))
        """

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
    """
    def run_GAMS(self,filename):
        # Pass to GAMS all necessary sets and parameters
        self._load_experiment_data_in_gams(filename)
        #self.db.export(' _custom.gdx')

        def reorder_age_headers(df_unordered):
            temp = df_unordered
            temp.columns = temp.columns.astype(int)
            temp.sort_index(inplace=True,axis=1)
            return temp

        #Run GMS Optimization
        try:
            model_run = self.ws.add_job_from_file(self.gms_file) # model_run is type GamsJob

            opt = self.ws.add_options()
            opt.defines["gdxincname"] = self.db.name
            model_run.run(opt,databases=self.db)#,create_out_db = True)
            print("Ran GAMS model: "+self.gms_file)

            gams_db = model_run.out_db
            self.export_fp = os.path.join(self.export_fp,filename+'_solution.gdx')
            gams_db.export(self.export_fp)
            print("Completed export of solution database")# + self.export_fp)

            "" Fetch model outputs""
            self.totc = self.get_output_from_GAMS(gams_db, 'TOTC')
            self.totc_opt = self.get_output_from_GAMS(gams_db, 'TOTC_OPT')
            self.veh_stck_delta = self.get_output_from_GAMS(gams_db, 'VEH_STCK_DELTA')
            self.veh_stck_add = self.get_output_from_GAMS(gams_db, 'VEH_STCK_ADD')
            self.veh_stck_rem = self.get_output_from_GAMS(gams_db, 'VEH_STCK_REM')
            self.veh_stck = self.get_output_from_GAMS(gams_db, 'VEH_STCK')
            self.veh_totc = self.get_output_from_GAMS(gams_db, 'VEH_TOTC')
            self.annual_totc = self.veh_totc.unstack('year').sum()

            self.veh_prod_totc = self.get_output_from_GAMS(gams_db, 'VEH_PROD_TOTC')
            self.veh_oper_totc = self.get_output_from_GAMS(gams_db, 'VEH_OPER_TOTC')
            self.total_op_emissions = self.veh_oper_totc.unstack('year').sum()
            self.veh_eolt_totc = self.get_output_from_GAMS(gams_db, 'VEH_EOLT_TOTC')

            self.emissions = self.veh_prod_totc.join(self.veh_oper_totc,rsuffix='op').join(self.veh_eolt_totc,rsuffix='eolt')
            self.emissions.columns = ['Production', 'Operation', 'End-of-life']
            self.emissions = self.emissions.unstack(['tec', 'year']).sum().unstack([None, 'tec'])


            "" Fetch variable and stock compositions""
            sets = gmspy.ls(gdx_filepath=self.export_fp, entity='Set')
            parameters = gmspy.ls(gdx_filepath=self.export_fp,entity='Parameter')
            variables = gmspy.ls(gdx_filepath=self.export_fp,entity='Variable')
            years = gmspy.set2list(sets[0], gdx_filepath=self.export_fp)

    #        plt.xkcd()
            matplotlib.rcParams.update({'font.size': 13})

            # Export parameters
            p_dict = {}
            for p in parameters:
                try:
                    p_dict[p] = gmspy.param2df(p,gdx_filepath=self.export_fp)
                except ValueError:
                    try:
                        p_dict[p] = gmspy.param2series(p,gdx_filepath=self.export_fp)
                    except:
                        pass
                except AttributeError:
                    pass

            # Export variables
            v_dict = {}
            for v in variables:
                try:
                    v_dict[v] = gmspy.var2df(v,gdx_filepath=self.export_fp)
                except ValueError:
                    try:
                        v_dict[v] = gmspy.var2series(v,gdx_filepath=self.export_fp)
                    except:
                        pass
                except TypeError: # This is specifically for seg_add
                    pass


            # Prepare dataframes
            self.stock_df = v_dict['VEH_STCK']
            self.stock_df = reorder_age_headers(self.stock_df)
            self.stock_add = v_dict['VEH_STCK_ADD']
            self.stock_add = reorder_age_headers(self.stock_add)
            self.stock_add = self.stock_add.dropna(axis=1,how='any')
            #stock_df.loc['BEV'].sum(axis=1).unstack('age').plot(kind='area',cmap='Spectral_r') # aggregate segments, plot by age
            #stock_df.loc['ICE'].sum(axis=1).unstack('age').plot(kind='area',cmap='Spectral_r') # aggregate segments, plot by age
            self.stock_df_plot = self.stock_df.stack().unstack('age')
            self.stock_df_plot = reorder_age_headers(self.stock_df_plot)

            #stock_df_plot.unstack('seg')
            self.stock_df_plot_grouped = self.stock_df_plot.groupby(['tec', 'seg'])

            self.stock_cohort = v_dict['VEH_STCK_CHRT']
            self.stock_cohort = self.stock_cohort.droplevel(level='age',axis=0)
            self.stock_cohort = self.stock_cohort.stack().unstack('prodyear').sum(axis=0,level=['tec', 'modelyear'])
    #        self.stock_cohort = reorder_age_headers(self.stock_cohort)#VEH_STCK_CHRT(tec,seg,prodyear,age,modelyear)


            self.veh_prod_cint = p_dict['VEH_PROD_CINT']
            self.veh_prod_cint = self.veh_prod_cint.stack()
            self.veh_prod_cint.index.rename(['tec', 'seg', 'year'],inplace=True)

            self.veh_oper_eint = p_dict['VEH_OPER_EINT']
            self.veh_oper_eint = self.veh_oper_eint.stack()
            self.veh_oper_eint.index.rename(['tec', 'seg', 'year'],inplace=True)

            self.veh_oper_cint = p_dict['VEH_OPER_CINT']
            self.veh_oper_cint = self.veh_oper_cint.stack()
            self.veh_oper_cint.index.rename(['tec', 'enr', 'seg', 'cohort', 'age', 'year'],inplace=True)
            self.veh_oper_cint.index = self.veh_oper_cint.index.droplevel(['year', 'age', 'enr'])
    #        self.veh_oper_cint.index = self.veh_oper_cint.index.droplevel('enr')

            self.enr_cint = p_dict['ENR_CINT']
            self.enr_cint = self.enr_cint.stack()
            self.enr_cint.index.rename(['enr', 'year'],inplace=True)

            add_gpby = self.stock_add.sum(axis=1).unstack('seg').unstack('tec')
            self.add_share = add_gpby.div(add_gpby.sum(axis=1),axis=0)
            " Export technology shares in 2030 to evaluate speed of uptake"
            self.shares_2030 = self.add_share.loc['2030']#.to_string()
            self.shares_2050 = self.add_share.loc['2050']

            " Export first year of 100% BEV market share "
            tec_shares = self.add_share.stack().stack().sum(level=['year', 'tec'])
            self.full_BEV_year = int((tec_shares.loc[:, 'BEV']==1).idxmax()) - 1
            if self.full_BEV_year=='1999':
                self.full_BEV_year = np.nan
            temp = self.veh_stck.unstack(['year', 'tec']).sum()
#            self.stock_tot['percent_BEV'] = (temp)/temp.sum()
#            self.time_10 = stock_tot['percent_BEV'].between(0.9, 0.11)
        except:
            exceptions = self.db.get_database_dvs()
            try:
                print(exceptions.symbol.name)
            except:
                print(exceptions)
           # self.db.export(os.path.join(self.current_path, 'troubleshooting_tryexcept'))
        """
    def add_to_GAMS(self):
        # Adding sets
        def build_set(set_list=None, name=None, desc=None):
            i = self.db.add_set(name, 1, desc)
            for s in set_list:
                i.add_record(str(s))

        # NOTE: Check that 'cohort', 'year' and 'prodyear' work nicely together
#        cohort = build_set(self.cohort, 'year', 'cohort')
#        tec = build_set(self.tecs, 'tec', 'technology')
#        age = build_set(self.age, 'age', 'age')
#        enr = build_set(self.enr, 'enr', 'energy types')

        # Export for troubleshooting
        #self.db.export('add_sets.gdx')

    def calc_crit_materials(self):
        # performs critical material mass accounting
        pass

    def post_processing(self):
        # make pretty figures?
        pass

    def import_from_MESSAGE(self):
        pass
    def figure_calculations(self):
        """--- Used by main.py to aggregate key indicators across experiments ---"""
        # Attempt to calculate average operating intensity by using total lifetime operating emissions by cohort, divided by original stock of the cohort

        # Calculate operating emissions by cohort (i.e., prodyear)
        operation_em = self.veh_oper_cohort.sum(level=['prodyear', 'reg', 'tec', 'seg']).sum(axis=1)
        operation_em.sort_index(axis=0, level=0, inplace=True)
        op = operation_em.loc['2000':'2050']

        # Calculate stock
        init_stock = self.veh_stck_add.loc(axis=1)[0]
        init_stock.replace(0, np.nan)
        init_stock.dropna(axis=0, inplace=True)
#        init_stock.index.rename('prodyear', level=3, inplace=True)
        init_stock.index = init_stock.index.reorder_levels([3, 2, 0, 1])
        init_stock.sort_index(inplace=True)

        self.op_intensity = op/init_stock

        temp_prod = self.veh_prod_cint.copy(deep=True)
        temp_prod.index = temp_prod.index.reorder_levels([2, 0, 1])
        temp_prod.sort_index(inplace=True)

        self.op_intensity.sort_index(inplace=True)
#        self.LC_intensity - self.op_intensity.add(temp_prod,axis='index')

    def vis_GAMS(self, fp, filename, param_values, export_png, export_pdf=True, max_year=50, cropx=True, suppress_vis=False):
        """ visualize key GAMS parameters for quality checks"""
        """To do: split into input/output visualization; add plotting of CO2 and stocks together"""
        """To do: """

        os.chdir(fp)
        pp = PdfPages('output_vis_' + filename + '.pdf')
        plt.rcParams.update({'figure.max_open_warning': 0}) # suppress max 20 figures warning
        if suppress_vis:
            plt.ioff()

        def fix_age_legend(ax, title='Vehicle ages'):
            patches, labels = ax.get_legend_handles_labels()

            if len(labels) == 12:
                order = [11,9,7,5,3,1,10,8,6,4,2,0]
                labels = [x + ', ' + y for x,y in itertools.product(['BEV', 'ICEV'],['mini', 'small', 'medium', 'large', 'executive', 'luxury and SUV'])]
                ax.legend([patches[idx] for idx in order], [labels[idx] for idx in range(11,-1,-1)], bbox_to_anchor = (1.05,1.02), loc='upper left', ncol=2,title=title)
            elif len(labels) == 6:
                order = [5,3,1,4,2,0]
                ax.legend([patches[idx] for idx in order], [labels[idx] for idx in order], bbox_to_anchor=(1.05,1.02), loc='upper left', ncol=2, title=title)
            elif len(labels) > 34:
                ax.legend(patches,labels, bbox_to_anchor=(1.05, 1.02), loc='upper left', ncol=3, title=title)
            else:
                ax.legend(patches,labels, bbox_to_anchor=(1.05, 1.02), loc='upper left', ncol=2, title=title)

            if cropx and ax.get_xlim()[1] == 80:
                ax.set_xlim(right=max_year)
            export_fig(ax.get_title)
            pp.savefig(bbox_inches='tight')

        def plot_subplots(grouped_df, title, labels, cmap='jet', xlabel='year'):
            for (key,ax) in zip(grouped_df.groups.keys(), axes.flatten()):
                d = grouped_df.get_group(key)
                if d.index.nlevels == 3:
                    d = grouped_df.get_group(key).reset_index(level=[0, 1], drop=True)
                elif d.index.nlevels == 2:
                    d = grouped_df.get_group(key).reset_index(level=[0], drop=True)

                d.plot(ax=ax, cmap=cmap, legend=False)

                ax.set_xlabel(xlabel)
                ax.set_title(key, fontsize=10, fontweight='bold')

                ax.xaxis.set_minor_locator(LinearLocator(4))
                ax.grid(which='minor', axis='x', c='lightgrey', alpha=0.55, linestyle=':', lw=0.3)
                ax.grid(which='major', axis='x', c='darkgrey', alpha=0.75, linestyle=':', lw=1)

                ax.grid(which='minor', axis='y', c='lightgrey', alpha=0.55, linestyle=':', lw=0.3)
                ax.grid(which='major', axis='y', c='darkgrey', alpha=0.75, linestyle=':', lw=1)

                plt.subplots_adjust(hspace=0.45)
                fig.suptitle(title)
            ax.legend(labels=labels, bbox_to_anchor=(0.2,-0.3), ncol=2, fontsize='large')
            return ax

        def export_fig(png_name=None):
            if export_pdf:
                pp.savefig(bbox_inches='tight')
            if export_png:
                png_name = ax.get_title()
                plt.savefig(png_name, format='png', bbox_inches='tight')
#
#        def crop_x(ax,max_year,cropx):
#            if cropx:
#                ax.xlim(right=max_year)

        ## Make paired colormap for comparing tecs
        paired = LinearSegmentedColormap.from_list('paired', colors=['indigo', 'thistle', 'mediumblue', 'lightsteelblue', 'darkgreen', 'yellowgreen', 'olive', 'lightgoldenrodyellow', 'darkorange', 'navajowhite', 'darkred', 'salmon'], N=12)
        light = LinearSegmentedColormap.from_list('light', colors=['thistle', 'lightsteelblue', 'yellowgreen', 'lightgoldenrodyellow', 'navajowhite', 'salmon'], N=6)
        dark = LinearSegmentedColormap.from_list('dark', colors=['indigo', 'mediumblue', 'darkgreen', 'olive', 'darkorange', 'darkred'], N=6)
        paired_tec = LinearSegmentedColormap.from_list('paired_by_tec', colors=['indigo', 'mediumblue', 'darkgreen', 'olive', 'darkorange', 'darkred', 'thistle', 'lightsteelblue', 'yellowgreen', 'lightgoldenrodyellow', 'navajowhite', 'salmon'], N=12)

        if param_values:
            div_page = plt.figure(figsize=(25, 8))
            ax = plt.subplot(111)
            ax.axis('off')
            df_param = pd.DataFrame.from_dict(param_values)
            df_param = df_param.T
#        df_param_text = df_param.values
#        for item in df_param:
#            item.replace('{', '')
#            item.replace('}} , ', '\n')
#            item.replace('}', '')

            param_table = plt.table(cellText=df_param.values, colLabels=['scenario \n name', 'values'], rowLabels=df_param.index, colWidths=[0.1, 0.9], cellLoc='left', loc=8)
            param_table.auto_set_font_size(False)
            param_table.set_fontsize(14)
            param_table.scale(1, 2.5)
            export_fig('tec-seg-cohort')
            #           pp.savefig(bbox_inches='tight')
        else:
            print('Could not make parameter table in export PDF')

        """--- Plot total stocks by age, technology, and segment---"""
        for region in self.reg[:-1]:
            fig, axes = plt.subplots(4, 3, figsize=(12,12), sharey=True, sharex=True)
            plt.ylim(0, np.ceil(self.stock_df_plot.sum(axis=1).max()))#/5e7)*5e7)

            if cropx:
                plt.xlim(right=max_year)

            for (key, ax) in zip(self.stock_df_plot_grouped.groups.keys(), axes.flatten()):
    #            if(key==('BEV', 'B')):
    #                fix_age_legend(ax)
                d = self.stock_df_plot_grouped.get_group(key).reset_index(level=[0, 1], drop=True)
                ax = d.loc[region].plot(ax=ax, kind='area', cmap='Spectral_r', legend=False)
    #             self.stock_df_plot_grouped.get_group(key).plot(ax=ax,kind='area',cmap='Spectral_r',legend=False)
                #handles,labels = ax.get_legend_handles_labels()
                ax.xaxis.set_major_locator(MultipleLocator(10))
                ax.xaxis.set_minor_locator(MultipleLocator(5))
                ax.xaxis.set_tick_params(rotation=45)
    #            ax.set_xticklabels([2000,2010,2020,2030,2040,2050],fontsize=9, rotation=45)
    #            ax.set_xticklabels(self.stock_df_plot_grouped.groups[key].get_level_values('year'))
                ax.set_xlabel('year')
                ax.text(0.5, 0.9, key, horizontalalignment='center', transform=ax.transAxes, fontweight='bold')
    #            plt.xticks(rotation=45)
    #            ax.set_title(key,fontsize=10,fontweight='bold')

#        patches, labels = ax.get_legend_handles_labels()
#        ax.legend(patches,labels,bbox_to_anchor=(1.62,4.35), ncol=2, title='Age')
#        """" Does not work, need a non-'aliased' reference to datapoint """
#        #ax.axvline(x=2020,ls='dotted',color='k')
#        fig.suptitle('Vehicle stock by technology and segment')
#        plt.subplots_adjust(hspace=0.12,wspace=0.1)
#        pp.savefig(bbox_inches='tight')

#        for (key, ax) in zip(self.stock_df_plot_grouped.groups.keys(), axes.flatten()):
 #           self.stock_df_plot_grouped.get_group(key).plot(ax=ax,kind='area',cmap='Spectral_r')

#
#        ax = self.stock_df_plot.loc['BEV'].groupby('seg').plot(kind='area',cmap='Spectral_r',title='BEV stocks by age and segment')
#        ax = self.stock_df_plot.loc['BEV'].plot(kind='area',cmap='Spectral_r',title='BEV stocks by age and segment')
#        fix_age_legend(ax)
#        ax = self.stock_df_plot.loc['ICE'].groupby('seg').plot(kind='area',cmap='Spectral_r',title='ICE stocks by age and segment')
#        ax = self.stock_df_plot.loc['ICE'].plot(kind='area',cmap='Spectral_r',title='ICE stocks by age and segment')
#        fix_age_legend(ax)
        for region in self.reg[:-1]:
            """  change this to be subplots with regions """
            ax = (self.stock_add.sum(axis=1).unstack('seg').unstack('tec').loc[region]).plot(kind='area', cmap=paired, title=f'Stock additions, by segment and technology in region {region}')
#        ax = (self.stock_add.sum(axis=1).unstack('seg').unstack('tec')/1e6).groupby('reg').plot(kind='area',cmap=paired,title='Stock additions, by segment and technology')
            fix_age_legend(ax, 'Vehicle technology and segment')
            ax.set_ylabel('Vehicles added to stock \n millions of vehicles')
            plt.ylim(0, np.ceil(self.stock_df_plot.sum(axis=1).max()))#/5e7)*5e7)
            #axes = self.stock_add.unstack('seg').groupby('tec').plot(kind='area',cmap='jet',title='Stock additions by segment and technology')
            #ax.set_xticklabels([2000,2010,2020,2030,2040,2050])
            #ax.set_xlabel('year')
            #ax.axvline(x=2020,ls='dotted')

            """--- Plot stock addition shares by segment and technology ---"""
            ax = self.add_share.loc[region].plot(kind='area', cmap=paired, title=f'Share of stock additions, by technology and vehicle segment in region {region}')
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.grid(which='minor', axis='x', c='w', alpha=0.6, linestyle=(0, (5,10)), lw=0.1)
            ax.grid(which='major', axis='x', c='darkgrey', alpha=0.75, linestyle='--', lw=0.5,)
            fix_age_legend(ax, 'Vehicle technology and segment')

            """--- Plot tech split of stock additions by segment ---"""
            temp_df = self.add_share/self.add_share.sum(axis=1, level=0)
            ax = temp_df.loc[region].plot(kind='area', cmap=paired, title=f'Technological split of total segment additions in region {region}')
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.25))
            ax.grid(which='minor', axis='x', c='w', alpha=0.6, linestyle=(0,(5,10)), lw=0.1)
            ax.grid(which='major', axis='x', c='darkgrey', alpha=0.75, linestyle='--', lw=0.5)
            ax.grid(which='minor', axis='y', c='w', alpha=0.6, linestyle='dotted', lw=0.1)
            fix_age_legend(ax, 'Vehicle technology and segment')

            """--- Plot share of BEVs in stock additions ---"""
            temp_df = (self.add_share/self.add_share.sum(axis=1,level=0)).drop('ICE', axis=1, level=1)
            ax = temp_df.loc[region].plot(kind='line', cmap=paired, title=f'Share of BEVs in stock additions in region {region}')
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.25))
            ax.grid(which='minor', axis='x', c='w', alpha=0.6, linestyle=(0,(5,10)), lw=0.1)
            ax.grid(which='major', axis='x', c='darkgrey', alpha=0.75, linestyle='--', lw=0.5)
            ax.grid(which='minor', axis='y', c='w', alpha=0.6, linestyle='dotted', lw=0.1)
            fix_age_legend(ax, 'Vehicle technology and segment')


        """--- Plot total emissions by tec and lifecycle phase---"""

        cmap_em = LinearSegmentedColormap.from_list('emissions', ['lightsteelblue', 'midnightblue', 'silver', 'grey', 'lemonchiffon', 'gold'], N=6)
        tec_cm = LinearSegmentedColormap.from_list('tec', ['xkcd:burgundy', 'xkcd:light mauve'])

        self.emissions.sort_index(axis=1, level=0, ascending=False, inplace=True)
#        self.emissions = self.emissions/1e6

        fig = plt.figure(figsize=(14,9))

        gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1,3], hspace=0.05)
        ax2 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1], sharex=ax2)
        (self.emissions/1e6).plot(ax=ax1, kind='area', cmap=cmap_em)
        (self.stock_df_plot.sum(axis=1).unstack('seg').sum(axis=1).unstack('tec').sum(level='year')/1e6).plot(ax=ax2, kind='area', cmap=tec_cm)

        ax1.set_ylabel('Lifecycle climate emissions \n Mt $CO_2$-eq', fontsize=13)
        ax2.set_ylabel('Vehicles, millions', fontsize=13, labelpad=25)
        if cropx:
            ax1.set_xlim(right=max_year)
            ax2.set_xlim(right=max_year)
#        patches, labels = ax1.get_legend_handles_labels()
#        order = [5, 3, 1, 4, 2, 0]
#        ax1.legend([patches[idx] for idx in order],[labels[idx] for idx in order], loc=1, fontsize=12)
        handles, labels = ax1.get_legend_handles_labels()
        labels = [x+', '+y for x,y in itertools.product(['Production', 'Operation', 'End-of-life'], ['ICEV', 'BEV'])]
        ax1.legend(handles, labels, loc=1, fontsize=14)
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(handles, ['BEV', 'ICEV'], loc=4, fontsize=14, framealpha=1)

        plt.setp(ax2.get_yticklabels(), fontsize=14)
        plt.xlabel('year', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        export_fig('LC_emissions_vs_stock')
#        pp.savefig(bbox_inches='tight')

        """--- Plot operation emissions by tec ---"""
        ax = (self.emissions.loc[:, 'Operation']/1e6).plot(kind='area',cmap=LinearSegmentedColormap.from_list('temp', colors=['silver', 'grey']))
        plt.hlines(442, xmin=0.16, xmax=0.6, linestyle='dotted', color='darkslategrey', label='EU 2030 target, \n 20% reduction from 2008 emissions', transform=ax.get_yaxis_transform())
        plt.hlines(185, xmin=0.6, xmax=1, linestyle='-.', color ='darkslategrey', label='EU 2050 target, \n 60% reduction from 1990 emissions', transform=ax.get_yaxis_transform())
        plt.ylabel('Fleet operation emissions \n Mt $CO_2$-eq')
        if cropx:
            plt.xlim(right=max_year)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, ['ICEV', 'BEV', 'EU 2030 target, \n20% reduction from 2008 emissions', 'EU 2050 target, \n60% reduction from 1990 emissions'], bbox_to_anchor = (1.05,1.02))#loc=1
        export_fig('operation_emissions')
#        pp.savefig(bbox_inches='tight')

        """--- Plot total stocks by segment ---"""
        ax = self.stock_df_plot.sum(axis=1).unstack('seg').sum(axis=0, level=['year']).plot(kind='area', cmap='jet', title='Total stocks by segment')
        fix_age_legend(ax, 'Vehicle segments')

        """--- Plot total stocks by region ---"""
        ax = self.stock_df_plot.sum(axis=1).unstack('reg').sum(axis=0, level=['year']).plot(kind='area', cmap='jet', title='Total stocks by region')
        fix_age_legend(ax, 'Region')

        """--- Plot total stocks by age, segment and technology ---"""
        ax = self.stock_df_plot.sum(axis=1).unstack('seg').unstack('tec').sum(axis=0, level='year').plot(kind='area', cmap=paired, title='Total stocks by segment and technology')
        fix_age_legend(ax, 'Vehicle segment and technology')

        """--- Plot total stocks by age, segment and technology ---"""
#        ax = self.stock_df_plot.sum(axis=1).unstack('seg').unstack('tec').unstack('reg').plot(kind='area',cmap=paired,title='Total stocks by segment, technology and region')
        stock_tec_seg_reg = self.stock_df_plot.sum(axis=1).unstack('seg').unstack('tec').unstack('reg').drop('PROD', level='reg', axis=1)
        ax = stock_tec_seg_reg.plot(kind='area', cmap='jet', title='Total stocks by segment, technology and region')
        fix_age_legend(ax, 'Vehicle segment, technology and region')

        """--- Plot total stocks by technology and segment ---"""
#        ax = self.veh_stck.unstack(['tec', 'seg', 'year']).sum().unstack(['tec', 'seg']).stack().unstack(['seg']).plot(kind='area',cmap=paired_tec, title='Total stocks by technology and segment')
#        fix_age_legend(ax, 'Vehicle segment and technology')


        """--- Plot total stocks by age ---"""
        #stock_df_plot = stock_df_plot.sum(axis=1,level=1) # aggregates segments
#        ax = self.stock_df_plot.sum(level=2).plot(kind='area',cmap='Spectral_r',title='Total stocks by age')
#        fix_age_legend(ax)


        #ax = self.stock_df_plot.sum(level=2).plot.barh()
        """--- Plot total stocks by age and technology ---"""

        """ Plot BEVs by cohort and region """
        stock_cohort = self.stock_cohort.copy()
        stock_cohort.sort_index(level=['reg', 'modelyear'],ascending=[1,2], inplace=True)
        temp_stock_cohort = (stock_cohort/1e6).loc['BEV'].loc(axis=0)[:, '2020':'2050']
        temp_stock_cohort[temp_stock_cohort < 1e-9] = 0#0.4] = 0 # Drops very small vehicle stocks in earlier years
        fig, axes = plt.subplots(len(self.reg[:-1]), 1, sharex=True, sharey=True)
        plt.subplots_adjust(top=0.85, hspace=0.25)

        for i, reg in enumerate(self.reg[:-1]):
            plot_stock = (temp_stock_cohort.loc[reg].replace(0,np.nan).dropna(how='all', axis=1))
            plot_stock.plot(ax=axes[i], kind='bar', stacked=True, width=1, cmap='Spectral', title=f'BEV stock by vintage cohort in region {reg}', legend=False)

#        for reg, group in plot_stock_cohort:
#            group = group.replace(0, np.nan).dropna(how='all', axis=1)
#            ax = group.plot(kind='bar', stacked=True, width=1, cmap='Spectral', title=f'BEV stock by vintage cohort in region {reg}')
        axes[-1].xaxis.set_major_locator(IndexLocator(5, 0))
        x_labels = [label[1] for label in temp_stock_cohort.index.tolist()]
        axes[-1].xaxis.set_major_formatter(IndexFormatter(x_labels))
        axes[-1].xaxis.set_minor_locator(IndexLocator(5, 0))
#        ax.set_xticklabels([2015,2020,2025,2030,2035,2040,2045,2050])
        plt.ylabel('BEV stock, in millions of vehicles', y=0.6)
        plt.xlabel('year')
        fix_age_legend(axes[0], title='Vehicle vintage')
        pp.savefig()

        """temp_stock_cohort = (self.stock_cohort/1e6).loc['ICE'].loc[:'2050']
        temp_stock_cohort[temp_stock_cohort<0.4] = 0 # Drops very small vehicle stocks in earlier years
        temp_stock_cohort = temp_stock_cohort.replace(0,np.nan).dropna(how='all',axis=1)
        try:
            ax = temp_stock_cohort.plot(kind='bar',stacked=True,width=1,cmap='Spectral',title='ICEV stock by vintage cohort')
            ax.xaxis.set_major_locator(IndexLocator(10,0))
            ax.xaxis.set_major_formatter(IndexFormatter(temp_stock_cohort.index))
            ax.xaxis.set_tick_params(rotation=45)
            ax.xaxis.set_minor_locator(IndexLocator(5,0))
    #        ax.set_xticklabels([1995,2000,2010,2020,2030,2040,2050])
    #        ax.set_xticklabels([2015,2020,2025,2030,2035,2040,2045,2050])
            plt.ylabel('ICEV stock, in millions of vehicles')
            plt.xlabel('year')
            fix_age_legend(ax,title='Vehicle vintage')
        except TypeError:
            print('No ICEVs!')"""

        """--- Plot addition to stocks by segment, technology and region ---"""
#        tec_cm = LinearSegmentedColormap.from_list('tec',['xkcd:aubergine', 'lavender'])
        tec_cm = LinearSegmentedColormap.from_list('tec', ['xkcd:burgundy', 'xkcd:light mauve'])
        tec_cm4 = LinearSegmentedColormap.from_list('tec', ['xkcd:burgundy', 'xkcd:light mauve', 'xkcd:dark grey blue', 'xkcd:light grey blue'])
        plot_stock_add = self.stock_add.sum(level=['tec', 'reg', 'prodyear']).unstack(['tec', 'reg']).droplevel(axis=1, level=0).drop(columns='PROD', level=1)

        ax = plot_stock_add.plot(kind='area', cmap=tec_cm4, title='Stock additions by technology and region')

        fix_age_legend(ax, 'Vehicle technology and region')
        plt.xlabel('year')
        plt.ylabel('Vehicles added to stock')

#        patches, labels  = ax.get_legend_handles_labels()
#        plt.legend(handles=patches, labels=labels, bbox_to_anchor=(-0.05, 0.6), title='Vehicle technology and region')

#        fig,axes = plt.subplots(1,2,figsize=(6,3))
#        stock_add_grouped = self.stock_add.unstack('seg').groupby('tec')
#        for (key,ax) in zip(stock_add_grouped.groups.keys(),axes.flatten()):
#            stock_add_grouped.get_group(key).plot(ax=ax,kind='area',cmap='jet',legend=False)
#            ax.set_xticklabels([2000,2010,2020,2030,2040,2050])
#            ax.set_xlabel('year')
#            ax.set_title(key,fontsize=10,fontweight='bold')
#            #ax.axvline(x=('BEV',2020),ls='dotted')
#        fig.suptitle('Additions to stock by segment and technology')
#        ax.legend(labels=self.seg,title='Segment',markerscale=15)

        pp.savefig()

        """--- Plot total stocks by segment, technology and region ---"""
        plot_stock = self.veh_stck.sum(axis=1).unstack(['tec', 'reg']).sum(level=['year']).drop(columns='PROD', level=1)
        ax = plot_stock.plot(kind='area', cmap=tec_cm4, title='Total stock by technology and region')

        fix_age_legend(ax, 'Vehicle technology and region')
        plt.xlabel('year')
        plt.ylabel('Vehicles in stock')
        pp.savefig()

        """ Plot evolution of lifecycle emissions """
        fig, axes = plt.subplots(len(self.reg)-1, len(self.tecs), sharey=True, sharex=True)
        plt.subplots_adjust(top=0.85, hspace=0.25, wspace=0.05)

        for i, reg in enumerate(self.reg[:-1]):
            for j, tec in enumerate(self.tecs):
                plot_data = self.LC_emissions.unstack('seg').loc(axis=0)[tec, '1995':'2050', reg]
                plot_data.plot(ax=axes[i,j], legend=False)
                axes[i,j].set_title(f'{tec}, {reg}', fontsize=10, fontweight='bold')
                x_labels = [label[1] for label in plot_data.index.tolist()]
                axes[i,j].xaxis.set_major_locator(IndexLocator(5, 0))
                axes[i,j].xaxis.set_major_formatter(IndexFormatter(x_labels))

                axes[i,j].xaxis.set_tick_params(rotation=45)
                if not i:
                    axes[i,j].xaxis.label.set_visible(False)
                else:
                    axes[i,j].xaxis.label.set_visible(True)
                    axes[i,j].xaxis.set_label('Cohort year')
                axes[i,j].yaxis.set_major_locator(IndexLocator(25, 0))

        plt.xlabel('Cohort year')
        fig.text(0, 0.5, 'Lifecycle emissions, in t CO2-eq', rotation='vertical', ha='center', va='center')

#        fig.text('Cohort year', 0.5, 0, ha='center')
#        full_fig.set_xlabel('Cohort year')
        fig.suptitle('Evolution of lifecycle emissions by cohort, segment, region and technology')
#        fix_age_legend(axes[0,1], title='Vehicle segment')
        patches, labels = axes[0, 0].get_legend_handles_labels()
        labels = [lab[4] for lab in labels]
        fig.legend(patches, labels, bbox_to_anchor=(1.01, 0.9), loc='upper left', title='Vehicle segment')
        pp.savefig()

        """--- to do: plot BEV cohorts as share of total fleet ---"""
        """--- to do: plot crossover in LC emissions between ICEV and BEV by segment---"""


        """--- Divider page for input parameter checking ---"""
        div_page = plt.figure(figsize=(11.69, 8.27))
        txt = 'Plotting of input parameters for checking'
        div_page.text(0.5, 0.5, txt, transform=div_page.transFigure, size=30, ha="center")
        pp.savefig()

        """--- Plot production emissions by tec and seg ---"""
#        fig,axes = plt.subplots(1,2)
#        for (key,ax) in zip(self.veh_prod_totc.groupby(['tec', 'seg']).groups.keys(),axes.flatten()):
#            self.veh_prod_totc.groupby(['tec', 'seg']).get_group(key).plot(ax=ax,kind='area',cmap='jet',legend=False)
#            ax.set_xticklabels([2000,2010,2020,2030,2040,2050])
#            ax.set_xlabel('year')
#            ax.set_title(key,fontsize=10,fontweight='bold')
#            #ax.axvline(x=('BEV',2020),ls='dotted')
#            ax.set_label('segment')
#        ax.legend()
#        pp.savefig(bbox_inches='tight')

        """--- Plot production emissions by tec and seg ---"""
        prod = self.veh_prod_totc.stack().unstack('tec').sum(level=['seg', 'year'])#/1e9
        prod_int = prod/self.stock_add.sum(axis=1).unstack('tec').sum(level=['seg', 'prodyear']) #production emission intensity

        fig, axes = plt.subplots(3, 2, figsize=(9,9), sharey=True)
        labels=['BEV', 'ICEV']
        title='Total production emissions by technology and segment'
        #plot_subplots((self.veh_prod_totc.unstack('tec').groupby(['seg'])),title=title,labels=labels)
        plot_subplots(prod.groupby(['seg']), title=title, labels=labels)
#        ax.legend(labels=['BEV', 'ICE'],bbox_to_anchor=(0.2,-0.3),ncol=2,fontsize='large')
        fig.text(0.04, 0.5, 'Production emissions \n(Mt CO2-eq)', ha='center', va='center', rotation='vertical')
        export_fig('tot_prod_emissions')
        pp.savefig(bbox_inches='tight')

        fig, axes = plt.subplots(3, 2, figsize=(9,9), sharey=True)
        #ax.legend(labels=['BEV', 'ICE'],bbox_to_anchor=(0.2,-0.3),ncol=2,fontsize='large')
        title = 'Production emission intensities by technology and segment'
        plot_subplots(prod_int.groupby(['seg']),title=title,labels=labels)
        fig.text(0.04, 0.5, 'Production emissions intensity \n(t CO2/vehicle)', ha='center', va='center', rotation='vertical')
        export_fig('prod_intensity_out')
        pp.savefig(bbox_inches='tight')

        fig, axes = plt.subplots(3, 2, figsize=(9,9), sharey=True)
        title = 'VEH_PROD_CINT'
        plot_subplots(self.veh_prod_cint.unstack('tec').groupby(['seg']), title=title, labels=labels)
        fig.text(0.04, 0.5, 'Production emissions intensity \n(t CO2/vehicle)', ha='center', va='center', rotation='vertical')
        export_fig('VEH_PROD_CINT')
        pp.savefig(bbox_inches='tight')

        fig, axes = plt.subplots(3, 2, figsize=(9,9), sharey=True)
        title = 'VEH_OPER_EINT - check ICE sigmoid function'
        plot_subplots(self.veh_oper_eint.unstack('tec').groupby(['seg']), title=title, labels=labels)
        fig.text(0.04, 0.5, 'Operation energy intensity \n(kWh/km)', ha='center', va='center', rotation='vertical')
        export_fig('VEH_OPER_EINT')
        pp.savefig(bbox_inches='tight')

        fig = (self.enr_cint * 1000).unstack(['enr', 'reg']).plot(title='ENR_CINT')
        plt.ylabel('Emissions intensity, fuels \n g CO2-eq/kWh')
        if cropx:
            plt.xlim(right=max_year)
        export_fig('ENR_CINT')
        pp.savefig(bbox_inches='tight')

#        fig, axes = plt.subplots(3,2,figsize=(9,9),sharey=True)
#        title = 'VEH_OPER_CINT'
#        temp_oper_cint = self.veh_oper_cint.loc[~self.veh_oper_cint.index.duplicated(keep='first')]*1e6
#        plot_subplots(temp_oper_cint.unstack('tec').groupby(['seg']),title=title,labels=labels)
#        fig.text(0.04, 0.5, 'Operation emissions intensity \n(g CO2-eq/km)', ha='center', rotation='vertical')
#        pp.savefig(bbox_inches='tight')
#
#        fig, axes = plt.subplots(3,2,figsize=(9,9),sharey=True)
#        title = 'Lifetime operation emissions by cohort for whole fleet'
#        plot_subplots(self.veh_op_cohort.unstack('tec').groupby(['seg']),title=title,labels=labels)
#        fig.text(0.04, 0.5, 'Operation emissions \n(t)', ha='center', rotation='vertical')
#        pp.savefig(bbox_inches='tight')

#        fig, axes = plt.subplots(3,2,figsize=(9,9),sharey=True)
#        title = 'Operating emissions, back calculated from fleet emissions'
#        plot_subplots(self.veh_op_intensity.unstack('tec').groupby(['seg']),title=title,labels=labels)
#        fig.text(0.04, 0.5, 'Lifetime operation emissions intensity \n(t/vehicle)', ha='center', rotation='vertical')
#        pp.savefig(bbox_inches='tight')


        fig, axes = plt.subplots(3, 2, figsize=(9,9), sharey=True)
        title = 'initial stock of each cohort'
        plot_subplots(self.stock_add.unstack('tec').groupby(['seg']), title=title, labels=labels)
        fig.text(0.04, 0.5, 'Total vehicles, by segment and technology \n(t)', ha='center', va='center', rotation='vertical')
        pp.savefig(bbox_inches='tight')

        fig, axes = plt.subplots(3, 2, figsize=(9,9), sharey=True)
        fig.text(0.04, 0.5, 'Vehicle operating emissions intensity, by region and segment \n (kg CO2-eq/km)', ha='center', va='center', rotation='vertical')
        title = 'VEH_OPER_CINT for BEVs, by region and segment'
        veh_op_cint_plot = self.veh_oper_cint.drop(labels='PROD', axis=0, level='reg').droplevel(['prodyear', 'enr']).drop_duplicates().unstack(['reg']).loc(axis=0)['BEV']
        veh_op_cint_plot = (veh_op_cint_plot.swaplevel(-2, -1, axis=0) * 1000).unstack('age')

        k_r_cmap = ListedColormap(['k' for i in np.arange(0, (len(veh_op_cint_plot.columns)/2))] + ['r' for i in np.arange(0, (len(veh_op_cint_plot.columns)/2))])
        plot_subplots(veh_op_cint_plot.groupby(['seg']), title=title, cmap=k_r_cmap, labels=['HIGH', 'LOW'])
#        ax = veh_op_cint_plot.groupby(['seg']).plot(cmap=k_r_cmap)
        ax.legend(labels=labels, bbox_to_anchor=(0.2,-0.3), ncol=2, fontsize='large')
        pp.savefig(bbox_inches='tight')


        """ Check each vintage cohort's LC emissions (actually just production and lifetime operation emissions) for various assumptions of lifetime (in years)"""
#        fig, axes = plt.subplots(3,2,figsize=(9,9),sharey=True)
#        title = 'Average Lifetime emissions per vehicle (11 years)'
#        ax = plot_subplots(self.LC_emissions_avg.unstack('tec').groupby(['seg']),title=title,labels=labels)
#        ax.set_xlabel('Vintage cohort')
#        fig.text(0.04, 0.5, 'Lifetime emissions intensity (without EOL) \n(t/average vehicle)', ha='center', rotation='vertical')
#        pp.savefig(bbox_inches='tight')

        """for i in range(0,len(self.LC_emissions_avg)):
            fig, axes = plt.subplots(3,2,figsize=(9,9),sharey=True)
            if i==27:
                title = f'Full lifetime emissions per vehicle ({i} year lifetime)'
            else:
                title = f'Average lifetime emissions per vehicle ({i} year lifetime)'
            ax = plot_subplots(self.LC_emissions_avg[i].unstack('tec').groupby(['seg']),title=title,labels=labels,xlabel='Vintage cohort')
            fig.text(0.04, 0.5,f'Lifetime emissions intensity (without EOL) \n(t/{i}-year-old vehicle)', ha='center', rotation='vertical')
            pp.savefig(bbox_inches='tight') """

        """------- Calculate lifecycle emissions (actually production + operation) by cohort for QA  ------- """
        """ See figure_calculations for calculation of these dataframes """
#        fig, axes = plt.subplots(3,2,figsize=(9,9),sharey=True)
#        title = 'Operating emissions, back calculated from fleet emissions'
#        plot_subplots(op_intensity.unstack('tec').groupby(['seg']),title=title,labels=labels)
#        fig.text(0.04, 0.5, 'Operation emissions intensity  \n(t/vehicle)', ha='center', rotation='vertical')
#        pp.savefig(bbox_inches='tight')

        """ See figure_calculations for calculation of these dataframes """
#        fig, axes = plt.subplots(3,2,figsize=(9,9),sharey=True)
#        title = 'Lifetime operation emissions by cohort for whole fleet'
#        plot_subplots(self.LC_intensity.unstack('tec').groupby(['seg']),title=title,labels=labels)
#        fig.text(0.04, 0.5, 'Operation emissions \n(t)', ha='center', rotation='vertical')
#        pp.savefig(bbox_inches='tight')
#
        """ Need to fix! """
#        fig, axes = plt.subplots(3,2,figsize=(9,9),sharey=True)
#        title = 'VEH_OPER_CINT'
##        plot_subplots((self.veh_oper_cint.unstack(['tec', 'enr'])*1e6).groupby(['seg']),title=title,labels=labels)
#        plot_subplots((self.veh_oper_cint*1e6).groupby(['seg']),title=title,labels=labels)
#        fig.text(0.04, 0.5, 'Operation emissions intensity \n(g CO2/vkm)', ha='center', rotation='vertical')
#        pp.savefig(bbox_inches='tight')


        """kept commented"""
        #for (key,ax) in zip(self.veh_prod_totc.unstack('tec').groupby(['seg']).groups.keys(),axes.flatten()):
#        for (key,ax) in zip(self.veh_prod_totc.unstack('tec').groupby(['seg']).groups.keys(),axes.flatten()):
#            self.veh_prod_totc.unstack('tec').groupby(['seg']).get_group(key).plot(ax=ax,cmap='jet',legend=False)
#            ax.set_xticklabels([2000,2010,2020,2030,2040,2050])
#            ax.set_xlabel('year')
#            ax.set_title(key,fontsize=10,fontweight='bold')
#            ax.set_label('segment')
#
#            ax.xaxis.set_minor_locator(MultipleLocator(1))
#            ax.grid(which='minor',axis='x',c='lightgrey',alpha=0.55,linestyle=':',lw=0.3)
#            ax.grid(which='major',axis='x',c='darkgrey',alpha=0.75,linestyle='--',lw=1)
#
#
#        ax.legend(labels=['BEV', 'ICE'],bbox_to_anchor=(0.2,-0.3),ncol=2,fontsize='large')
#        plt.subplots_adjust(hspace=0.45)
#        pp.savefig(bbox_inches='tight')
#


        #stock_df_grouped =stock_df.groupby(level=[0])
#        for name, group in stock_df_grouped:
#            ax=group.plot(kind='area',cmap='Spectral_r',title=name+' stock by age')
#            fix_age_legend(ax)
#            pp.savefig()

        #stock_df.columns = stock_df.columns.astype(int)
        #stock_df.sort_index(axis=1,inplace=True)
        #tot_stock_df=stock_df.sum(axis=0,level=1)
        #ax = tot_stock_df.plot.area(cmap='Spectral_r',title='Total stocks by vehicle age',figsize = (10,6))
        #fix_age_legend(ax)
        #plt.savefig('total_stocks_by_age.png',pad_inches=2)
        #pp.savefig()

        # Plot total stocks by technology
#        stock_df.sum(axis=1).unstack().T.plot(kind='area', title='Total stocks by technology')
#        stock_df.sum(axis=1).unstack().T.plot(title='Total stocks by technology')
#        plt.savefig('total_stocks_by_tec.png',dpi=600)
#        pp.savefig()

        # Plot stock additions and removals by technology
#        temp_vdict_a = reorder_age_headers(v_dict['VEH_STCK_REM']).stack()
#        temp_vdict_b = reorder_age_headers(v_dict['VEH_STCK_ADD']).stack()
#        add_rem_df = pd.concat([temp_vdict_a, temp_vdict_b],axis=1,keys=('VEH_STCK_REM', 'VEH_STCK_ADD'))
#
#        add_rem_df_2=add_rem_df.stack().unstack(level=[0,3])
#
#        for column,variable in add_rem_df_2:
#            ax = add_rem_df_2[column][variable].unstack().plot(kind='area',cmap='Spectral_r',title=column+" "+variable)
#            fix_age_legend(ax)
#
#        add_rem_df_2.plot(subplots=True,title='Stock removal and addition variables')
#        pp.savefig()
#
#        # Plot carbon emissions by technology and lifecycle phase
#        totc_df=pd.concat((v_dict['VEH_PROD_TOTC'],v_dict['VEH_OPER_TOTC'],v_dict['VEH_EOLT_TOTC'],v_dict['VEH_TOTC']),axis=0,keys=('VEH_PROD_TOTC', 'VEH_OPER_TOTC', 'VEH_EOLT_TOTC', 'VEH_TOTC'))
#        totc_df=totc_df.T.swaplevel(0,1,axis=1)
#        ax = totc_df.plot(figsize = (10,6))
#        fix_age_legend(ax)
#        plt.savefig('CO2.png',pad_inches=2, dpi=600)
#        pp.savefig()

        pp.close()

#        plt.clf()
        """For later: introduce figure plotting vehicle stock vs emissions"""

        # Plot parameter values for quality assurance
#        ax= p_df.plot(subplots=True,title='Parameter values')


    """
    Intermediate methods
    """

    def elmix(self):
        # produce time series of elmix intensities, regions x year
        pass

    def _read_all_final_parameters(self, a_file):
        # will become unnecessary as we start internally defining all parameters
        db = gmspy._iwantitall(None, None, a_file)


class EcoinventManipulator:
    """ generates time series of ecoinvent using MESSAGE inputs"""

    def __init__(self, data_from_message, A, F):
        self.A = A #default ecoinvent A matrix
        self.F = F #default ecionvent F matrix

    def elmix_subst(self):
        # substitute MESSAGE el mixes into ecoinvent
        pass

def genlogfnc(t, A=0.0, B=1.0, r=None, u=None, r0=10.):
    """ Generalized Logistic function

    Parameters
    ----------
    t : 1-dimensional numpy array, or list
        Time values
    a : float, default 0.0
        Initial asymptote
    b : float, default 1.0
        Final asymptote
    r : float, or None (default)
        Rate of change. If None, defaults to r0 divided by the range of t
    u : float, or None (default)
        Time of maximum growth rate. If None, defaults to the median of t
    r0 : float
        A proportionality constant to help scale default r values


    Returns
    -------
    y : 1-dimensional numpy array
    """
    # Convert t to numpy array (if needed), and calculate t_range at the same time
    try:
        t_range = t.ptp()
    except AttributeError:
        t = np.array(t)
        t_range = t.ptp()

    # Define default inflection point
    if u is None:
        u = np.median(t)

    # Define default rate
    if r is None:
        r = r0 / t_range

    # The actual calculation
    y = A + (B - A) / (1 + np.exp(-r * (t - u)))

    return y

def calc_steadystate_vehicle_age_distributions(ages, average_expectancy=10.0, standard_dev=3.0):
    """ Calc a steady-state age distribution consistent with a normal distribution around an average life expectancy

    Parameters
    ----------
    ages : 1-dimensional numpy array
        The range of ages that we are investigating
    average_expectancy : float
        Average age at which a typical car dies
    standard_deviation: float
        Standard deviation around that average death age

    Returns
    -------
    q : 1-dimensional numpy array
        The fraction of cars for each age

    Example
    -------
    Assuming
    - Average age of death (loc): 10
    - Standard Deviation: 3

                          AGE DISTRIBUTION AT STEADY STATE

      10%  +----------------------------------------------------------------------+
           |*************                                                         |
           |             *******                                                  |
       9%  |                    ***                                               |
           |                       *                                              |
       8%  |                        ***                                           |
           |                                                                      |
           |                           **                                         |
       7%  |                             **                                       |
           |                                                                      |
       6%  |                               *                                      |
           |                                *                                     |
           |                                 *                                    |
       5%  |                                  *                                   |
           |                                   *                                  |
           |                                    *                                 |
       4%  |                                     *                                |
           |                                      **                              |
       3%  |                                        **                            |
           |                                                                      |
           |                                          ***                         |
       2%  |                                                                      |
           |                                             ***                      |
       1%  |                                                *                     |
           |                                                 ***                  |
           |                                                    *******           |
       0%  +----------------------------------------------------------------------+
         0      2      4      6      8       10     12     14     16     18     20
                                        VEHICLE AGE
    """

    # The total (100%) minus the cumulation of all the cars retired by the time they reach a certain age
    h = 1 - norm.cdf(ages, loc=average_expectancy, scale=standard_dev)

    # Normalize to represent a _fraction_ of the total fleet
    q = h / h.sum()
    return q


def calc_probability_of_vehicle_retirement(ages, age_distribution):
    """ Calculate probability of any given car dying during the year, depending on its age.
    This probability is calculated from the age distribution of a population, that is assumed to be and to have been at
    steady state

    This is only valid if we can assume that the population is at steady state.  If in doubt, it is probably best to
    rely on some idealized population distribution, such as the one calculated by
    `calc_steadystate_vehicle_age_distributions()`

    Parameters
    ----------
    ages : 1-dimensional numpy array
        The range of ages that we are investigating

    age_distribution: 1-dimensional numpy array
        The fraction of vehicles that have a certain age


    Returns
    -------
    g : 1-dimensional numpy array
        The probability that a car of a given age will die during the year

    See Also
    --------

    `calc_steadystate_vehicle_age_distributions()`

    Example
    --------

    Given an age distribution consistent with an average life expectancy of 10 years (SD 3 years), we get the following

              PROBABILITY OF DEATH DURING THE YEAR, AS FUNCTION OF AGE

        1 +---------------------------------------------------------------------+
          |                                                                  *  |
          |                                                                 *   |
      0.9 |                                                                 *   |
          |                                                                *    |
      0.8 |                                                                *    |
          |                                                               *     |
          |                                                               *     |
      0.7 |                                                              *      |
          |                                                             **      |
      0.6 |                                                         ****        |
          |                                                      ***            |
          |                                                   ***               |
      0.5 |                                                 **                  |
          |                                               **                    |
          |                                           ****                      |
      0.4 |                                         **                          |
          |                                       **                            |
      0.3 |                                    ***                              |
          |                                  **                                 |
          |                                **                                   |
      0.2 |                             ***                                     |
          |                          ***                                        |
      0.1 |                      ****                                           |
          |                   ***                                               |
          |            *******                                                  |
        0 +---------------------------------------------------------------------+
          0      2      4      6      8      10     12     14     16     18     20
                                         AGE OF CAR

    """
    # Initialize
    g = np.zeros_like(age_distribution)


    for a in ages[:-1]:
        if age_distribution[a] > 0:
            # Probability of dying is 1 minus the factions of cars that make it to the next year
            g[a] = 1 - age_distribution[a + 1] / age_distribution[a]

        else:
            # If no car left, then 100% death (just to avoid NaN)
            g[a] = 1.0

    # At the end of the time window, force exactly 100% probability of death
    g[-1] = 1.0

    return g

