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
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

import seaborn
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime


import gams
import gmspy


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
    def __init__(self, data_from_message=None):
        self.current_path = os.path.dirname(os.path.realpath(__file__))
       #self.gdx_file = 'C:\\Users\\chrishun\\Box Sync\\YSSP_temp\\EVD4EUR_input.gdx'#EVD4EUR_ver098.gdx'
        self.gms_file = 'C:\\Users\\chrishun\\Box Sync\\YSSP_temp\\EVD4EUR.gms' # GAMS model file
        self.import_fp = 'C:\\Users\\chrishun\\Box Sync\\YSSP_temp\\data\\'
        self.export_fp = ''
        self.keeper = "{:%d-%m-%y, %H_%M}".format(datetime.now())
        self.xl_writer = pd.ExcelWriter('output'+self.keeper+'.xlsx')

        """ static input data.....hardcoded and/or read in from Excel? """
        self.battery_specs = pd.DataFrame() # possible battery_sizes (and acceptable segment assignments, CO2 production emissions, critical material content, mass)
        self.fuelcell_specs = pd.DataFrame() # possible fuel cell powers (and acceptable segment assignments, CO2 production emissions, critical material content, fuel efficiency(?), mass)
        self.lightweighting = pd.DataFrame() # lightweighting data table - lightweightable materials and coefficients for corresponding lightweighting material(s)
        
        self.el_intensity = data_from_message # regional el-mix intensities as time series from MESSAGE
        self.trsp_dem = data_from_message # EUR transport demand as time series from MESSAGE
        """ boundary conditions for constraints, e.g., electricity market supply constraints, crit. material reserves? could possibly belong in experiment specifications as well..."""
        
        """ GAMS-relevant attributes"""
        #  --------------- GAMS sets / domains -------------------------------
        self.tecs = ['ICE','BEV']                           # drivetrain technologies; can include e.g., alternative battery chemistries
        self.year = [str(2000+i) for i in range(51)]
        self.cohort = [str(2000+i) for i in range(51)]      # vehicle cohorts (production year)
        self.optyear = [str(2020+i) for i in range(31)]
        self.age = [str(i) for i in range(28)]              # vehicle age, up to 27 years old
        self.enr = ['ELC','FOS']                            # fuel types; later include H2, 
        self.seg = ['A','B','C','D','E','F']                    # From ACEA: Small, lower medium, upper medium, executive
        self.demeq= ['STCK_TOT','OPER_DIST','OCUP']         # definition of 
        self.dstvar=['mean','stdv']
        self.enreq=['CINT']
        self.grdeq=['IND','ALL']
        self.veheq = ['PROD_EINT','PROD_CINT_CSNT','OPER_EINT','EOLT_CINT']
        self.inityear=[str(2000+i) for i in range(21)]      # reduce to one/five year(s)? Originally 2000-2020
        self.lfteq=['LFT_DISTR','AGE_DISTR']
        self.sigvar=['A','B','r','t','u']                   # S-curve terms
        self.critmats = ['Cu','Li','Co','Pt','','']         # critical elements to count for; to incorporate later
        self.age_int = list(map(int,self.age))
        
        # --------------- GAMS Parameters -------------------------------------

        
        # "Functional unit" # TODO: this is redund
        # Eurostat road_tf_veh [vkm]
        # Eurostat road_tf_vehage [vkm, cohort] NB: very limited geographic spectrum
        # Eurostat road_pa_mov [pkm]
        # http://www.odyssee-mure.eu/publications/efficiency-by-sector/transport/distance-travelled-by-car.html
        
        self.veh_oper_dist = pd.Series([-97.052*i+207474 for i in range(2000,2051)],index=[str(i) for i in range(2000,2051)]) 
        #self.veh_oper_dist.index.names=['year']
        # [years] driving distance each year # TODO: rename?
        
        """needs to be made in terms of tec as well??"""
        """Currently uses generalized logistic growth curve"""
        """TODO: make veh_stck_tot a variable with occupancy rate"""
        #self.veh_stck_tot = pd.Series([100500000]*len(self.cohort),index=[str(i) for i in range(2000,2051)])  
        self.veh_stck_tot = pd.DataFrame(pd.read_excel('GAMS_input_new.xls',sheet_name='VEH_STCK_TOT',header=None,usecols='A:B',skiprows=[0]))
        self.veh_stck_tot = self._process_df_to_series(self.veh_stck_tot)
        
        ################ Life cycle intensities ################
        """These factors are usually calculated using the general logistic function"""
        self.veh_prod_cint = pd.DataFrame(pd.read_excel('GAMS_input_new.xls',sheet_name='VEH_PROD_CINT',header=None,usecols='A:D',skiprows=[0]))  # [tecs, cohort]
        self.veh_prod_cint = self._process_df_to_series(self.veh_prod_cint)

        self.veh_oper_cint = pd.DataFrame(pd.read_excel('GAMS_input_new.xls',sheet_name='VEH_OPER_CINT',header=None,usecols='A:E',skiprows=[0]))  # [[tecs, enr], cohort]
        self.veh_oper_cint = self._process_df_to_series(self.veh_oper_cint)
        
        """Trial for calculating general logistic function in-code""" 
        """self.veh_partab = pd.DataFrame(pd.read_excel('GAMS_input_new.xls',sheet_name='VEH_PARTAB',header=None,usecols='A:D',skiprows=[0]))
        self.veh_partab = self._process_df_to_series(self.veh_partab)
        print(self.veh_partab)
        self.trial_oper_eint = self.veh_partab['OPER_EINT']
        self.oper_eint = self.trial_oper_eint['A']+(trial_oper_eint['B']-trial_oper_eint['A'])/(1+exp(-trial_oper_eint['r']*(self.year-trial_oper_eint['u'])))
        #A, B, r, u"""

        self.veh_eolt_cint = pd.DataFrame(pd.read_excel('GAMS_input_new.xls',sheet_name='VEH_EOLT_CINT',header=None,usecols='A:D',skiprows=[0]))  # [[tecs, enr], cohort]
        self.veh_eolt_cint = self._process_df_to_series(self.veh_eolt_cint)  # [tecs, cohort]

        ################ Fleet dynamics ################
        """VEH_LIFT_CDF(age) = cdfnormal(AGE_PAR(age),LFT_PARTAB('mean'),LFT_PARTAB('stdv'));
        VEH_LIFT_AGE(age) = (1 - VEH_LIFT_CDF(age))/sum(agej, VEH_LIFT_CDF(agej)) ;
        VEH_LIFT_MOR(age)$(ord(age)< 20) = 1 - VEH_LIFT_AGE(age+1)/VEH_LIFT_AGE(age);
        VEH_LIFT_MOR(age)$(ord(age)= 20) = 1"""
        avg_age = 11.1 # From ACEA 2019-2020 report
        std_dev_age = 2.21
        self.veh_lift_cdf = pd.Series(norm.cdf(self.age_int,avg_age,std_dev_age),index=self.age)#pd.Series(pd.read_pickle(self.import_fp+'input.pkl'))#pd.DataFrame()  # [age] TODO Is it this one we feed to gams?
        self.veh_lift_cdf.index = self.veh_lift_cdf.index.astype('str')
        
        self.veh_lift_age = pd.Series(1-self.veh_lift_cdf)     # [age] # probability of car of age x to die in current year
        
        #lifetime = [1-self.veh_lift_age[i+1]/self.veh_lift_age[i] for i in range(len(self.age)-1)]
        self.veh_lift_pdf = pd.Series(calc_steadystate_vehicle_age_distributions(self.age_int,avg_age,std_dev_age), index = self.age)   # idealized age PDF given avg fleet age and std dev
        self.veh_lift_pdf.index = self.veh_lift_pdf.index.astype('str')
        
        self.veh_lift_mor = pd.Series(calc_probability_of_vehicle_retirement(self.age_int,self.veh_lift_pdf), index = self.age)
        self.veh_lift_mor.index = self.veh_lift_mor.index.astype('str')

        
        # Initial stocks
        """# Eurostat road_eqs_carpda[tec]
        # Eurostat road_eqs_carage [age - <2, 2-5, 5-10, 10-20]; 
        # ACEA [age in year divisions up to 10 years]
        # Also check pb2018-section2016.xls for more cohesive, EU28++ data"""
        self.veh_stck_int = pd.DataFrame(pd.read_excel('GAMS_input_new.xls',sheet_name='VEH_STCK_INT',header=None,usecols='A:D',skiprows=[0]))  # [tec, age]
        self.veh_stck_int = self._process_df_to_series(self.veh_stck_int)


        ################ filters and parameter aliases ################
        self.enr_veh = pd.DataFrame(pd.read_excel('GAMS_input_new.xls',sheet_name='ENR_VEH',header=None,usecols='A:C',skiprows=[0]))            # [enr, tec]
        self.enr_veh = self._process_df_to_series(self.enr_veh)

        self.veh_pay = pd.DataFrame(pd.read_excel('GAMS_input_new.xls',sheet_name='VEH_PAY',header=None,usecols='A:D',skiprows=[0]))            # [cohort, age, year]
        self.veh_pay = self._process_df_to_series(self.veh_pay)
        
        self.age_par = pd.Series([float(i) for i in self.age])
        self.age_par.index = self.age_par.index.astype('str')
        
        self.year_par = pd.Series([float(i) for i in self.cohort],index = self.cohort)
        self.year_par.index = self.year_par.index.astype('str')
        
        self.prodyear_par = pd.Series([int(i) for i in self.cohort],index = self.cohort)
        self.prodyear_par.index = self.prodyear_par.index.astype('str')
        
        """# ACEA.be has segment division for Western Europe
        # https://www.acea.be/statistics/tag/category/segments-body-country
        # More detailed age distribution (https://www.acea.be/uploads/statistic_documents/ACEA_Report_Vehicles_in_use-Europe_2018.pdf)"""

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
        
        self.occupancy_rate = None # vkm -> pkm conversion
        self.battery_density = None # time series of battery energy densities
        self.lightweighting_scenario = None # lightweighting scenario - yes/no (or gradient, e.g., none/mild/aggressive?)

        """ Optimization Initialization """
        self.ws = gams.GamsWorkspace(working_directory=self.current_path,debug=2)
        self.db = self.ws.add_database(database_name='pyGAMSdb')
        self.opt = self.ws.add_options()
        self.opt.DumpParms = 2
        self.opt.ForceWork = 1

        
    def main(self):
        #
        pass
    
    @staticmethod
    def _process_df_to_series(df):
        """self.veh_prod_cint.set_index([0,1],inplace=True)
        a=self.veh_prod_cint.index.get_level_values(0).astype(str)
        b=self.veh_prod_cint.index.get_level_values(1).astype(str)
        self.veh_prod_cint.index = [a,b]
        self.veh_prod_cint.columns=['']
        self.veh_prod_cint.index.names = ['','']
        self.veh_prod_cint = pd.Series(self.veh_prod_cint.iloc[:,0])"""
        
#        col = self.veh_oper_cint.columns[:-1].tolist()
#        self.veh_oper_cint.set_axis(col,axis=1,inplace=True)
#        self.veh_oper_cint.set_index(col,inplace=True)
#        self.veh_oper_cint.set_index(self.veh_oper_cint.loc[:,col],inplace=True)
        
#        
#        a = self.veh_oper_cint.index.get_level_values(0).astype(str)
#        b = self.veh_oper_cint.index.get_level_values(1).astype(str)
#        c = self.veh_oper_cint.index.get_level_values(2).astype(str)
#        
#        self.veh_oper_cint.index = [a,b,c]
#        self.veh_oper_cint.columns=['']
#        self.veh_oper_cint.index.names = ['','','']
#        self.veh_oper_cint = pd.Series(self.veh_oper_cint.iloc[:,0])
        
        
        dims = df.shape[1]-1 # assumes stacked format
        indices = df.columns[:-1].tolist()
        df.set_index(indices,inplace=True)

        temp=[]
        for i in range(dims):
            temp.append(df.index.get_level_values(i).astype(str))
        df.index = temp
        df.columns=['']
        df.index.names = ['']*dims
        df = pd.Series(df.iloc[:,0])
        return df
    
    def read_all_sets(self, gdx_file):
        # No longer used after commit c941039 as sets are now internally defined
        """db = gmspy._iwantitall(None, None, gdx_file)
        self.tecs = gmspy.set2list('tec', db)
        self.cohort = gmspy.set2list('year', db)
        self.age = gmspy.set2list('age', db)
        self.enr = gmspy.set2list('enr', db)"""

         #spy.param2series('VEH_PAY', db) # series, otherwise makes giant sparse dataframe        


    def _load_experiment_data_in_gams(self): # will become unnecessary as we start calculating/defining sets and/or parameters within the class
        years = gmspy.list2set(self.db,self.year,'year')
        tecs = gmspy.list2set(self.db, self.tecs, 'tec')
        #cohort = gmspy.list2set(self.db, self.cohort, 'prodyear') ## prodyear is an alias of year, not a set of its own
        age = gmspy.list2set(self.db, self.age, 'age')
        enr = gmspy.list2set(self.db, self.enr, 'enr')
        seg = gmspy.list2set(self.db, self.seg, 'seg')
        demeq =  gmspy.list2set(self.db, self.demeq, 'demeq')
        dstvar = gmspy.list2set(self.db,self.dstvar,'dstvar')
        enreq = gmspy.list2set(self.db,self.enreq,'enreq')
        grdeq = gmspy.list2set(self.db,self.grdeq,'grdeq')
        inityear = gmspy.list2set(self.db,self.inityear,'inityear')
        lfteq = gmspy.list2set(self.db,self.lfteq,'lfteq')
        sigvar = gmspy.list2set(self.db,self.sigvar,'sigvar')
        veheq = gmspy.list2set(self.db, self.veheq,'veheq')
        optyear = gmspy.list2set(self.db,self.optyear,'optyear')
        # Add sets to GAMS
        #self.add_to_GAMS()

        veh_oper_dist = gmspy.df2param(self.db, self.veh_oper_dist, ['year'], 'VEH_OPER_DIST')
        veh_stck_tot = gmspy.df2param(self.db, self.veh_stck_tot, ['year'], 'VEH_STCK_TOT')
        
        veh_prod_cint = gmspy.df2param(self.db, self.veh_prod_cint, ['tec','seg', 'prodyear'], 'VEH_PROD_CINT')
        veh_oper_cint = gmspy.df2param(self.db, self.veh_oper_cint, ['tec', 'enr','seg', 'prodyear'], 'VEH_OPER_CINT')
        veh_eolt_cint = gmspy.df2param(self.db, self.veh_eolt_cint, ['tec','seg', 'prodyear'], 'VEH_EOLT_CINT')

        veh_lift_cdf = gmspy.df2param(self.db, self.veh_lift_cdf, ['age'], 'VEH_LIFT_CDF')
        veh_lift_pdf = gmspy.df2param(self.db, self.veh_lift_pdf, ['age'], 'VEH_LIFT_PDF')
        veh_lift_age = gmspy.df2param(self.db, self.veh_lift_age, ['age'], 'VEH_LIFT_AGE')
        veh_lift_mor = gmspy.df2param(self.db, self.veh_lift_mor, ['age'], 'VEH_LIFT_MOR' )
    
        ######  OBS: Originally calculated using VEH_STCK_INT_TEC, VEH_LIFT_AGE, VEH_STCK_TOT
        veh_stck_int = gmspy.df2param(self.db, self.veh_stck_int, ['tec','seg', 'age'], 'VEH_STCK_INT')

        enr_veh = gmspy.df2param(self.db, self.enr_veh, ['enr', 'tec'], 'ENR_VEH')

        veh_pay = gmspy.df2param(self.db, self.veh_pay, ['prodyear', 'age', 'year'], 'VEH_PAY')
        
        age_par = gmspy.df2param(self.db,self.age_par, ['age'], 'AGE_PAR')
        year_par = gmspy.df2param(self.db,self.year_par, ['year'], 'YEAR_PAR')
        
        print('exporting database...troubleshooting_params')
        self.db.export(os.path.join(self.current_path,'troubleshooting_params'))

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
        # calculate the energy intensity of driving, kWh/km
        pass

    def calc_veh_mass(self):
        # use factors to calculate vehicle total mass. 
        # used in calc_eint_oper() 
        pass

    def vehicle_builder(self):
        # Assembles vehicle from powertrain, glider and BoP and checks that vehicles makes sense (i.e., no Tesla motors in a Polo or vice versa)
        # used in calc_veh_mass()
        pass
              
    def run_GAMS(self):

        # Pass to GAMS all necessary sets and parameters
        self._load_experiment_data_in_gams()
        #self.db.export('troubleshooting_custom.gdx')
        
        #Run GMS Optimization
        try:
            model_run = self.ws.add_job_from_file(self.gms_file)
        
            model_run.run(databases=self.db,create_out_db = True)
            print("Ran GAMS model: "+self.gms_file)
            gams_db = model_run.out_db
            self.export_fp = os.path.join(self.current_path,'fleet_model_output.gdx')
            gams_db.export(self.export_fp)
            print("Completed export of " + self.export_fp)
            
            self.totc = self.get_output_from_GAMS(gams_db,'TOTC')
            #self.totc.to_excel(self.xl_writer,sheet_name='TOTC')
            self.veh_stck_add = self.get_output_from_GAMS(gams_db,'VEH_STCK_ADD')
#            self.veh_stck_add.to_excel(self.xl_writer,sheet_name='VEH_STCK_ADD')
            self.veh_stck = self.get_output_from_GAMS(gams_db,'VEH_STCK')
#            self.veh_stck.to_excel(self.xl_writer,sheet_name='TOTC')

            self.veh_totc = self.get_output_from_GAMS(gams_db,'VEH_TOTC')
#            self.veh_totc.to_excel(self.xl_writer,sheet_name='TOTC')

            self.veh_prod_totc = self.get_output_from_GAMS(gams_db,'VEH_PROD_TOTC')
            self.veh_oper_totc = self.get_output_from_GAMS(gams_db,'VEH_OPER_TOTC')
            self.veh_eolt_totc = self.get_output_from_GAMS(gams_db,'VEH_EOLT_TOTC')

        except:
            exceptions = self.db.get_database_dvs()
            try:
                print(exceptions.symbol.name)
            except:
                print(exceptions)
           # self.db.export(os.path.join(self.current_path,'troubleshooting_tryexcept'))
        
    def add_to_GAMS(self):
        # Adding sets
        def build_set(set_list=None,name=None,desc=None):
            i = self.db.add_set(name,1,desc)
            for s in set_list:
                i.add_record(str(s))
                
        # NOTE: Check that 'cohort', 'year' and 'prodyear' work nicely together
#        cohort = build_set(self.cohort, 'year', 'cohort')
#        tec = build_set(self.tecs, 'tec', 'technology')
#        age = build_set(self.age, 'age', 'age')
#        enr = build_set(self.enr, 'enr', 'energy types')
        
        # Export for troubleshooting
        #self.db.export('add_sets.gdx')
        
    def get_output_from_GAMS(self,gams_db,output_var):
         temp_GMS_output = []
         temp_index_list = []
         
         for rec in gams_db[output_var]:
            if gams_db[output_var].number_records ==1: # special case for totc
                temp_output_df = gams_db[output_var].first_record().level
                return temp_output_df
            
            dict1 = {}
            dict1.update({'level':rec.level})
            temp_GMS_output.append(dict1)
            temp_index_list.append(rec.keys)
         temp_domain_list = list(gams_db[output_var].domains_as_strings)
         temp_index = pd.MultiIndex.from_tuples(temp_index_list,names=temp_domain_list)
         temp_output_df = pd.DataFrame(temp_GMS_output,index = temp_index)

         return temp_output_df
        
    def calc_crit_materials(self):
        # performs critical material mass accounting
        pass

    def post_processing(self):
        # make pretty figures?
        pass
    
    def import_from_MESSAGE(self):
        pass

    def vis_GAMS(self):
        """ visualize key GAMS parameters for quality checks"""
        """To do: split into input/output visualization; add plotting of CO2 and stocks together"""
        ch_path = os.path.dirname(os.path.abspath(__file__))+r'\visualization output\ '
        os.chdir(ch_path)
        pp = PdfPages('output_vis_'+self.keeper+'.pdf')
        
        gdx_file = self.export_fp #'C:\\Users\\chrishun\\Box Sync\\YSSP_temp\\EVD4EUR_ver098.gdx'
        sets = gmspy.ls(gdx_filepath=gdx_file, entity='Set')
        parameters = gmspy.ls(gdx_filepath=gdx_file,entity='Parameter')
        variables = gmspy.ls(gdx_filepath=gdx_file,entity='Variable')
        """os.path.join(ws.working_directory,".gdx'0"""
        years = gmspy.set2list(sets[0], gdx_filepath=gdx_file)

        # Export parameters
        p_dict = {}
        for p in parameters:
            try:
                p_dict[p] = gmspy.param2df(p,gdx_filepath=gdx_file)
            except ValueError:
                try:
                    p_dict[p] = gmspy.param2series(p,gdx_filepath=gdx_file)
                except:
                    pass
            except AttributeError:
                pass
            
        
#        p_df = pd.DataFrame(index=years)
#        p_df.index.name='year'
#        for key in p_dict:
#            if len(p_dict[key])==len(years):
#                p_dict[key].rename_axis('year',inplace=True)
#                #p_df=pd.concat([p_df,p_dict[key]],axis=1,join_axes=[p_df.index])
#                #p_df= p_df.join(p_dict[key],how='outer')
#                p_df= pd.merge(p_df,p_dict[key].rename(key),on='year')#left_index=True,right_index=True)
#            else:
#                pass
#                    #print(key)
#        p_df.drop(['YEAR_PAR','PRODYEAR_PAR'],axis=1,inplace=True)
        
        
        # Export variables
        v_dict = {}
        for v in variables:
            try:
                v_dict[v] = gmspy.var2df(v,gdx_filepath=gdx_file)
            except ValueError:
                try:
                    v_dict[v] = gmspy.var2series(v,gdx_filepath=gdx_file)
                except:
                    pass
            except TypeError: # This is specifically for seg_add
                pass

        def reorder_age_headers(df_unordered):
            temp = df_unordered
            temp.columns = temp.columns.astype(int)
            temp.sort_index(inplace=True,axis=1)
            return temp
        
        def fix_age_legend(ax,title='Vehicle ages'):
            patches, labels = ax.get_legend_handles_labels()
            ax.legend(patches,labels,bbox_to_anchor=(1.05,1), ncol=2, title=title)
            pp.savefig(bbox_inches='tight')
            
        def plot_subplots(grouped_df,title,labels):
            for (key,ax) in zip(grouped_df.groups.keys(),axes.flatten()):
                grouped_df.get_group(key).plot(ax=ax,cmap='jet',legend=False)
                ax.set_xticklabels([2000,2010,2020,2030,2040,2050])
                ax.set_xlabel('year')
                ax.set_title(key,fontsize=10,fontweight='bold')
                
                ax.xaxis.set_minor_locator(MultipleLocator(1))
                ax.grid(which='minor',axis='x',c='lightgrey',alpha=0.55,linestyle=':',lw=0.3)
                ax.grid(which='major',axis='x',c='darkgrey',alpha=0.75,linestyle='--',lw=1)

                plt.subplots_adjust(hspace=0.45)
                fig.suptitle(title)
            ax.legend(labels=labels,bbox_to_anchor=(0.2,-0.3),ncol=2,fontsize='large')    

        ## Make paired colormap for comparing tecs
        co = plt.get_cmap('tab20')
        paired = matplotlib.colors.LinearSegmentedColormap.from_list('paired',co.colors[:12],N=12)

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
        self.stock_df_plot_grouped = self.stock_df_plot.groupby(['tec','seg'])

        """--- Plot total stocks by age, technology, and segment---"""   
        fig, axes = plt.subplots(4,2, figsize=(12,12), sharey=True)
        for (key, ax) in zip(self.stock_df_plot_grouped.groups.keys(), axes.flatten()):
            #print(key)
#            if(key==('BEV','B')):
#                fix_age_legend(ax)
            self.stock_df_plot_grouped.get_group(key).plot(ax=ax,kind='area',cmap='Spectral_r',legend=False)
            #handles,labels = ax.get_legend_handles_labels()
            ax.set_xticklabels([2000,2010,2020,2030,2040,2050])
            ax.set_xlabel('year')
            ax.set_title(key,fontsize=10,fontweight='bold')
        patches, labels = ax.get_legend_handles_labels()
        ax.legend(patches,labels,bbox_to_anchor=(1.52,5.32), ncol=2, title='Age')
        """" Does not work, need a non-'aliased' reference to datapoint """
        #ax.axvline(x=2020,ls='dotted',color='k')
        fig.suptitle('Vehicle stock by technology and segment')
        plt.subplots_adjust(hspace=0.42)#right=0.82,
        pp.savefig(bbox_inches='tight')
        
#        for (key, ax) in zip(self.stock_df_plot_grouped.groups.keys(), axes.flatten()):
 #           self.stock_df_plot_grouped.get_group(key).plot(ax=ax,kind='area',cmap='Spectral_r')

    
#        ax = self.stock_df_plot.loc['BEV'].groupby('seg').plot(kind='area',cmap='Spectral_r',title='BEV stocks by age and segment')
#        ax = stock_df_plot.loc['BEV'].plot(kind='area',cmap='Spectral_r',title='BEV stocks by age and segment')
#        fix_age_legend(ax)  
#        ax = self.stock_df_plot.loc['ICE'].groupby('seg').plot(kind='area',cmap='Spectral_r',title='ICE stocks by age and segment')
#        ax = stock_df_plot.loc['ICE'].plot(kind='area',cmap='Spectral_r',title='ICE stocks by age and segment')
#        fix_age_legend(ax) 
         
        """--- Plot total stocks by segment ---"""   
        ax = self.stock_df_plot.sum(axis=1).unstack('seg').sum(axis=0,level=1).plot(kind='area',cmap='jet',title='Total stocks by segment')
        fix_age_legend(ax,'Vehicle segments') 
          
        
        
        """--- Plot total stocks by age, segment and technology ---"""   
        ax = self.stock_df_plot.sum(axis=1).unstack('seg').unstack('tec').plot(kind='area',cmap=paired,title='Total stocks by segment and technology')
        fix_age_legend(ax,'Vehicle segment and technology') 

    
        """--- Plot total stocks by age ---"""   
        #stock_df_plot = stock_df_plot.sum(axis=1,level=1) # aggregates segments
        ax = self.stock_df_plot.sum(level=2).plot(kind='area',cmap='Spectral_r',title='Total stocks by age') 
        fix_age_legend(ax)
        
        
        #ax = self.stock_df_plot.sum(level=2).plot.barh()
        """--- Plot total stocks by age and technology ---"""
        ax = self.stock_df_plot.loc['BEV'].sum(level=1).plot(kind='area',cmap='Spectral_r',title='BEV stocks by age')
        fix_age_legend(ax)  
        
        ax = self.stock_df_plot.loc['ICE'].sum(level=1).plot(kind='area',cmap='Spectral_r',title='ICE stocks by age')
        fix_age_legend(ax)  
        ax.axvline(2020,ls='dotted',color='k')
                
        """--- Plot addition to stocks by segment and technology  ---"""
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
        
        ax = self.stock_add.sum(axis=1).unstack('seg').unstack('tec').plot(kind='area',cmap=paired,title='Stock additions, by segment and technology')
        fix_age_legend(ax,'Vehicle segment and technology') 
        #axes = self.stock_add.unstack('seg').groupby('tec').plot(kind='area',cmap='jet',title='Stock additions by segment and technology')
        #ax.set_xticklabels([2000,2010,2020,2030,2040,2050])
        #ax.set_xlabel('year')
        #ax.axvline(x=2020,ls='dotted')
        
        """--- Plot stock addition shares by segment and technology ---"""
        add_gpby = self.stock_add.sum(axis=1).unstack('seg').unstack('tec')
        add_share = add_gpby.div(add_gpby.sum(axis=1),axis=0)
        
        ax = add_share.plot(kind='area',cmap=paired,title='Share of stock additions, by segment and technology')
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(which='minor',axis='x',c='w',alpha=0.6,linestyle=':',lw=0.3)
        ax.grid(which='major',axis='x',c='darkgrey',alpha=0.75,linestyle='--',lw=1)
        fix_age_legend(ax,'Vehicle segment and technology') 

        """--- Plot production emissions by tec and seg ---"""
#        fig,axes = plt.subplots(1,2)
#        for (key,ax) in zip(self.veh_prod_totc.groupby(['tec','seg']).groups.keys(),axes.flatten()):
#            self.veh_prod_totc.groupby(['tec','seg']).get_group(key).plot(ax=ax,kind='area',cmap='jet',legend=False)
#            ax.set_xticklabels([2000,2010,2020,2030,2040,2050])
#            ax.set_xlabel('year')
#            ax.set_title(key,fontsize=10,fontweight='bold')
#            #ax.axvline(x=('BEV',2020),ls='dotted')
#            ax.set_label('segment')
#        ax.legend()
#        pp.savefig(bbox_inches='tight')
        
        """--- Plot production emissions by tec and seg ---"""
        prod_int = self.veh_prod_totc.unstack('tec')/self.stock_add.sum(axis=1).unstack('tec')
        
        fig,axes = plt.subplots(3,2,figsize=(9,9),sharey=True)
        labels=['BEV','ICE']
        title='Total production emissions by technology and segment'
        plot_subplots(self.veh_prod_totc.unstack('tec').groupby(['seg']),title=title,labels=labels)
#        ax.legend(labels=['BEV','ICE'],bbox_to_anchor=(0.2,-0.3),ncol=2,fontsize='large')    
        
        fig,axes = plt.subplots(3,2,figsize=(9,9),sharey=True)
        #ax.legend(labels=['BEV','ICE'],bbox_to_anchor=(0.2,-0.3),ncol=2,fontsize='large')    
        title = 'Production emission intensities by technology and segment'
        plot_subplots(prod_int.groupby(['seg']),title=title,labels=labels)
        

        
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
#        ax.legend(labels=['BEV','ICE'],bbox_to_anchor=(0.2,-0.3),ncol=2,fontsize='large')    
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
#        add_rem_df = pd.concat([temp_vdict_a, temp_vdict_b],axis=1,keys=('VEH_STCK_REM','VEH_STCK_ADD'))
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
#        totc_df=pd.concat((v_dict['VEH_PROD_TOTC'],v_dict['VEH_OPER_TOTC'],v_dict['VEH_EOLT_TOTC'],v_dict['VEH_TOTC']),axis=0,keys=('VEH_PROD_TOTC','VEH_OPER_TOTC','VEH_EOLT_TOTC','VEH_TOTC'))
#        totc_df=totc_df.T.swaplevel(0,1,axis=1)
#        ax = totc_df.plot(figsize = (10,6))
#        fix_age_legend(ax)
#        plt.savefig('CO2.png',pad_inches=2, dpi=600)
#        pp.savefig()
        pp.close()
        
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

        #self.veh_oper_dist = gmspy.param2series('VEH_OPER_DIST', db)
        #self.veh_stck_tot = gmspy.param2series('VEH_STCK_TOT', db)
        #self.veh_lift_cdf = gmspy.param2series('VEH_LIFT_CDF', db)
        #self.veh_lift_age = gmspy.param2series('VEH_LIFT_AGE', db)

        #self.veh_prod_cint = gmspy.param2df('VEH_PROD_CINT', db)
        #self.veh_oper_cint = gmspy.param2df('VEH_OPER_CINT', db)
        #self.veh_eolt_cint = gmspy.param2df('VEH_EOLT_CINT', db)

        #self.veh_stck_int = gmspy.param2df('VEH_STCK_INT', db)

        #self.enr_veh = gmspy.param2df('ENR_VEH', db)
        #self.veh_pay = gm
    
       

class EcoinventManipulator:
    """ generates time series of ecoinvent using MESSAGE inputs"""

    def __init__(self, data_from_message, A, F):
        self.A = A #default ecoinvent A matrix
        self.F = F #default ecionvent F matrix

    def elmix_subst(self):
        # substitute MESSAGE el mixes into ecoinvent
        pass

def genlogfnc(t, a=0.0, b=1.0, r=None, u=None, r0=10.):
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

    # The actual Calculation
    y = a + (b - a) / (1 + np.exp(-r * (t - u)))

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

