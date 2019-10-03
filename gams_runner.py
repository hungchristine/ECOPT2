# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 12:37:00 2019

@author: chrishun
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import norm

import seaborn
from datetime import datetime

import itertools

import gams
import gmspy

import fleet_model

class GAMSRunner:
    def __init__(self):
        """ Optimization Initialization """
        self.current_path = os.path.dirname(os.path.realpath(__file__))
        self.export_fp = 'C:\\Users\\chrishun\\Box Sync\\YSSP_temp\\Model run data\\'
        self.ws = gams.GamsWorkspace(working_directory=self.current_path,debug=2)
#        self.db = self.ws.add_database()#database_name='pyGAMSdb')
        self.opt = self.ws.add_options()
        self.opt.LogLine = 2
#        self.opt.DumpParms = 2
        self.opt.ForceWork = 1
        gams.execution.SymbolUpdateType =1
        
    def _load_experiment_data_in_gams(self, fleet, filename): # will become unnecessary as we start calculating/defining sets and/or parameters within the class
        # Clear database for new run
#        self.db.clear() # need to add functionality to gmspy --> check if Symbol exists in database, write over
        self.db = self.ws.add_database()#database_name='pyGAMSdb')
        self.update_fleet(fleet)
        years = gmspy.list2set(self.db,fleet.cohort,'year')
        modelyear = gmspy.list2set(self.db,fleet.modelyear,'modelyear')
        tecs = gmspy.list2set(self.db, fleet.tecs, 'tec')
        #cohort = gmspy.list2set(self.db, self.cohort, 'prodyear') ## prodyear is an alias of year, not a set of its own
        age = gmspy.list2set(self.db, fleet.age, 'age')
        enr = gmspy.list2set(self.db, fleet.enr, 'enr')
        seg = gmspy.list2set(self.db, fleet.seg, 'seg')
        demeq =  gmspy.list2set(self.db, fleet.demeq, 'demeq')
        dstvar = gmspy.list2set(self.db,fleet.dstvar,'dstvar')
        enreq = gmspy.list2set(self.db,fleet.enreq,'enreq')
        grdeq = gmspy.list2set(self.db,fleet.grdeq,'grdeq')
        inityear = gmspy.list2set(self.db,fleet.inityear,'inityear')
        lfteq = gmspy.list2set(self.db,fleet.lfteq,'lfteq')
        sigvar = gmspy.list2set(self.db,fleet.sigvar,'sigvar')
        veheq = gmspy.list2set(self.db, fleet.veheq,'veheq')
        optyear = gmspy.list2set(self.db,fleet.optyear,'optyear')

        veh_oper_dist = gmspy.df2param(self.db, fleet.veh_oper_dist, ['year'], 'VEH_OPER_DIST')
        veh_stck_tot = gmspy.df2param(self.db, fleet.veh_stck_tot, ['year'], 'VEH_STCK_TOT')
        veh_stck_int_seg = gmspy.df2param(self.db,fleet.veh_stck_int_seg,['seg'],'VEH_STCK_INT_SEG')
        bev_capac = gmspy.df2param(self.db,fleet.seg_batt_caps,['seg'],'BEV_CAPAC')

        veh_lift_cdf = gmspy.df2param(self.db, fleet.veh_lift_cdf, ['age'], 'VEH_LIFT_CDF')
        veh_lift_pdf = gmspy.df2param(self.db, fleet.veh_lift_pdf, ['age'], 'VEH_LIFT_PDF')
        veh_lift_age = gmspy.df2param(self.db, fleet.veh_lift_age, ['age'], 'VEH_LIFT_AGE')
        veh_lift_mor = gmspy.df2param(self.db, fleet.veh_lift_mor, ['age'], 'VEH_LIFT_MOR' )
    
        ######  OBS: Originally calculated using VEH_STCK_INT_TEC, VEH_LIFT_AGE, VEH_STCK_TOT
        veh_stck_int = gmspy.df2param(self.db, fleet.veh_stck_int, ['tec','seg', 'age'], 'VEH_STCK_INT')
        veh_stck_int_tec = gmspy.df2param(self.db, fleet.veh_stck_int_tec,['tec'],'VEH_STCK_INT_TEC')

        enr_veh = gmspy.df2param(self.db, fleet.enr_veh, ['enr', 'tec'], 'ENR_VEH')

        veh_pay = gmspy.df2param(self.db, fleet.veh_pay, ['prodyear', 'age', 'year'], 'VEH_PAY')
        
        #age_par = gmspy.df2param(self.db,self.age_par, ['age'], 'AGE_PAR')
        year_par = gmspy.df2param(self.db, fleet.year_par, ['year'], 'YEAR_PAR')
        veh_partab = gmspy.df2param(self.db, fleet.veh_partab,['veheq','tec','seg','sigvar'],'VEH_PARTAB')

        veh_add_grd = self.db.add_parameter_dc('VEH_ADD_GRD', ['grdeq','tec'])
        # Prep work making add gradient df from given rate constraint
        
        # adding growth constraint for each tec    
        for keys,value in iter(fleet.veh_add_grd.items()):
            veh_add_grd.add_record(keys).value = value
        print('_load_experiment_data')
        print(veh_add_grd)
#        veh_add_grd = gmspy.df2param(self.db,self.veh_add_grd, ['grdeq','tec'], 'VEH_ADD_GRD')
    
        gro_cnstrnt = gmspy.df2param(self.db, fleet.gro_cnstrnt,['year'],'GRO_CNSTRNT')
        
        enr_partab = gmspy.df2param(self.db, fleet.enr_partab,['enr','enreq','sigvar'],'ENR_PARTAB')
        
        print('exporting database...'+filename+'_input')
        self.db.suppress_auto_domain_checking = 1
        self.db.export(os.path.join(self.current_path,filename+'_input'))

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
     
    def update_fleet(self,fleet):
        """ tec_add_gradient --> veh_add_grd """
        fleet.veh_add_grd = dict()
        for element in itertools.product(*[fleet.grdeq,fleet.tecs]):
            fleet.veh_add_grd[element] = fleet.tec_add_gradient
            
        """ occupancy_rate --> veh_oper_dist """    
        fleet.fleet_vkm = fleet.passenger_demand/fleet.occupancy_rate
        fleet.veh_oper_dist = fleet.fleet_vkm/fleet.veh_stck_tot
        
        """ recalculate age functions"""
        fleet.veh_lift_cdf = pd.Series(norm.cdf(fleet.age_int,fleet.avg_age,fleet.std_dev_age),index=fleet.age)#pd.Series(pd.read_pickle(fleet.import_fp+'input.pkl'))#pd.DataFrame()  # [age] TODO Is it this one we feed to gams?
        fleet.veh_lift_cdf.index = fleet.veh_lift_cdf.index.astype('str')
        
        fleet.veh_lift_age = pd.Series(1-fleet.veh_lift_cdf)     # [age] # probability of car of age x to die in current year
        
        #lifetime = [1-fleet.veh_lift_age[i+1]/fleet.veh_lift_age[i] for i in range(len(fleet.age)-1)]
        fleet.veh_lift_pdf = pd.Series(fleet_model.calc_steadystate_vehicle_age_distributions(fleet.age_int,fleet.avg_age,fleet.std_dev_age), index = fleet.age)   # idealized age PDF given avg fleet age and std dev
        fleet.veh_lift_pdf.index = fleet.veh_lift_pdf.index.astype('str')
        
        fleet.veh_lift_mor = pd.Series(fleet_model.calc_probability_of_vehicle_retirement(fleet.age_int,fleet.veh_lift_pdf), index = fleet.age)
        fleet.veh_lift_mor.index = fleet.veh_lift_mor.index.astype('str')
            
    def run_GAMS(self, fleet, filename):
        # Pass to GAMS all necessary sets and parameters
        self._load_experiment_data_in_gams(fleet, filename)
        #self.db.export(' _custom.gdx')
        
        def reorder_age_headers(df_unordered):
            temp = df_unordered
            temp.columns = temp.columns.astype(int)
            temp.sort_index(inplace=True,axis=1)
            return temp
        
        #Run GMS Optimization
        try:
            model_run = self.ws.add_job_from_file(fleet.gms_file) # model_run is type GamsJob
            
            opt = self.ws.add_options()
            opt.defines["gdxincname"] = self.db.name
            model_run.run(opt,databases=self.db)#,create_out_db = True)
            print("Ran GAMS model: "+fleet.gms_file)

            gams_db = model_run.out_db
            self.export_model = os.path.join(self.export_fp,filename+'_solution.gdx')
            gams_db.export(self.export_model)
            print("Completed export of solution database")# + self.export_fp)
            
            """ Fetch model outputs"""
            fleet.totc = self.get_output_from_GAMS(gams_db,'TOTC')
            fleet.totc_opt = self.get_output_from_GAMS(gams_db,'TOTC_OPT')
            print("TEST")
            print(fleet.totc_opt)
            fleet.veh_stck_delta = self.get_output_from_GAMS(gams_db,'VEH_STCK_DELTA')
            fleet.veh_stck_add = self.get_output_from_GAMS(gams_db,'VEH_STCK_ADD')
            fleet.veh_stck_rem = self.get_output_from_GAMS(gams_db, 'VEH_STCK_REM')
            fleet.veh_stck = self.get_output_from_GAMS(gams_db,'VEH_STCK')
            fleet.veh_totc = self.get_output_from_GAMS(gams_db,'VEH_TOTC')
            fleet.annual_totc = fleet.veh_totc.unstack('year').sum()

            fleet.veh_prod_totc = self.get_output_from_GAMS(gams_db,'VEH_PROD_TOTC')
            fleet.veh_oper_totc = self.get_output_from_GAMS(gams_db,'VEH_OPER_TOTC')
            fleet.total_op_emissions = fleet.veh_oper_totc.unstack('year').sum()
            fleet.veh_eolt_totc = self.get_output_from_GAMS(gams_db,'VEH_EOLT_TOTC')
            
            fleet.emissions = fleet.veh_prod_totc.join(fleet.veh_oper_totc,rsuffix='op').join(fleet.veh_eolt_totc,rsuffix='eolt')
            fleet.emissions.columns = ['Production','Operation','End-of-life']
            fleet.emissions = fleet.emissions.unstack(['tec','year']).sum().unstack([None,'tec'])
            
            
            """ Fetch variable and stock compositions"""
            sets = gmspy.ls(gdx_filepath=self.export_model, entity='Set')
            parameters = gmspy.ls(gdx_filepath=self.export_model,entity='Parameter')
            variables = gmspy.ls(gdx_filepath=self.export_model,entity='Variable')
            equations = gmspy.ls(gdx_filepath=self.export_model, entity='Equation')
            years = gmspy.set2list(sets[0], gdx_filepath=self.export_model)
                
            # Export parameters
            p_dict = {}
            for p in parameters:
                try:
                    p_dict[p] = gmspy.param2df(p,gdx_filepath=self.export_model)
                except ValueError:
                    try:
                        p_dict[p] = gmspy.param2series(p,gdx_filepath=self.export_model)
                    except:
                        print(f'Warning!: p_dict ValueError in {p}!')
                        pass
                except AttributeError:
                    print(f'Warning!: p_dict AttributeError in {p}!')
                    pass
                
            # Export variables
            v_dict = {}
            for v in variables:
                try:
                    v_dict[v] = gmspy.var2df(v,gdx_filepath=self.export_model)
                except ValueError:
                    try:
                        v_dict[v] = gmspy.var2series(v,gdx_filepath=self.export_model)
                    except:
                        print(f'Warning!: v_dict ValueError in {v}!')
                        pass
                except TypeError: # This is specifically for seg_add
                    print(f'Warning! v-dict TypeError in {v}!')
                    pass
            
            e_dict={}
            for e in equations:
                try:
                    e_dict[e] = gmspy.eq2series(e, gdx_filepath=self.export_model)
                except:
                    print(f'Warning!: Error in {e}')
                    pass
            
            # Prepare model output dataframes for visualization
            fleet.stock_df = v_dict['VEH_STCK']
            fleet.stock_df = reorder_age_headers(fleet.stock_df)
            fleet.stock_add = v_dict['VEH_STCK_ADD']
            fleet.stock_add = reorder_age_headers(fleet.stock_add)
            fleet.stock_add = fleet.stock_add.dropna(axis=1,how='any')
            fleet.stock_df_plot = fleet.stock_df.stack().unstack('age')  
            fleet.stock_df_plot = reorder_age_headers(fleet.stock_df_plot)
    
            fleet.stock_df_plot_grouped = fleet.stock_df_plot.groupby(['tec','seg'])
            
            fleet.stock_cohort = v_dict['VEH_STCK_CHRT']
            fleet.stock_cohort = fleet.stock_cohort.droplevel(level='age',axis=0)
            fleet.stock_cohort = fleet.stock_cohort.stack().unstack('prodyear').sum(axis=0,level=['tec','modelyear'])
            
            fleet.veh_prod_cint = p_dict['VEH_PROD_CINT']
            fleet.veh_prod_cint = fleet.veh_prod_cint.stack()
            fleet.veh_prod_cint.index.rename(['tec','seg','year'],inplace=True)
            
            fleet.veh_oper_eint = p_dict['VEH_OPER_EINT']
            fleet.veh_oper_eint = fleet.veh_oper_eint.stack()
            fleet.veh_oper_eint.index.rename(['tec','seg','year'],inplace=True)
            
            fleet.veh_oper_cint = p_dict['VEH_OPER_CINT']
            fleet.veh_oper_cint = fleet.veh_oper_cint.stack()
            fleet.veh_oper_cint.index.rename(['tec','enr','seg','cohort','age','year'],inplace=True)
            fleet.veh_oper_cint.index = fleet.veh_oper_cint.index.droplevel(['year','age','enr'])
            
            fleet.enr_cint = p_dict['ENR_CINT'].stack()
            fleet.enr_cint.index.rename(['enr','year'],inplace=True)
            
            add_gpby = fleet.stock_add.sum(axis=1).unstack('seg').unstack('tec')
            fleet.add_share = add_gpby.div(add_gpby.sum(axis=1),axis=0)
            """ Export technology shares in 2030 to evaluate speed of uptake"""
            fleet.shares_2030 = fleet.add_share.loc['2030']#.to_string()
            fleet.shares_2050 = fleet.add_share.loc['2050']
            fleet.eq = e_dict['EQ_STCK_GRD']
            
            """ Export first year of 100% BEV market share """
            tec_shares = fleet.add_share.stack().stack().sum(level=['year','tec'])
            fleet.full_BEV_year = int((tec_shares.loc[:,'BEV']==1).idxmax()) - 1
            if fleet.full_BEV_year == 1999:
                fleet.full_BEV_year = np.nan
            temp = fleet.veh_stck.unstack(['year','tec']).sum()
        except:
            exceptions = self.db.get_database_dvs()
            try:
                print(exceptions.symbol.name)
            except:
                print(exceptions)        