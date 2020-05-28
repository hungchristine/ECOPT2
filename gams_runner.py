# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 12:37:00 2019

@author: chrishun

This code initializes the GAMS workspace and database for running a scenario using
the fleet model object.
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
    """--- This class sets up the GAMS environment for running experiments ---"""
    def __init__(self):
        """ Optimization Initialization """
        self.current_path = os.path.dirname(os.path.realpath(__file__))
        self.export_fp = os.path.join(self.current_path, 'Model run data') #r'C:\Users\chrishun\Box Sync\YSSP_temp\Model run data\'
        self.ws = gams.GamsWorkspace(working_directory=self.current_path, debug=2)
#        self.db = self.ws.add_database()#database_name='pyGAMSdb')
        self.opt = self.ws.add_options()
        self.opt.LogLine = 2
#        self.opt.DumpParms = 2
        self.opt.ForceWork = 1
        gams.execution.SymbolUpdateType =1
        
    def _load_experiment_data_in_gams(self, fleet, filename): # will become unnecessary as we start calculating/defining sets and/or parameters within the class
        # Clear database for new run
#        self.db.clear() # need to add functionality to gmspy --> check if Symbol exists in database, write over
        self.db = self.ws.add_database(database_name='pyGAMS_input')#database_name='pyGAMSdb')
        if filename.find('unit_test'):
            fleet.veh_add_grd = dict()
            for element in itertools.product(*[fleet.grdeq, fleet.tecs]):
                fleet.veh_add_grd[element] = fleet.tec_add_gradient
        else:
            self.update_fleet(fleet)

        years = gmspy.list2set(self.db, fleet.cohort,'year')
        modelyear = gmspy.list2set(self.db, fleet.modelyear,'modelyear')
        tecs = gmspy.list2set(self.db, fleet.tecs, 'tec')
        #cohort = gmspy.list2set(self.db, self.cohort, 'prodyear') ## prodyear is an alias of year, not a set of its own
        age = gmspy.list2set(self.db, fleet.age, 'age')
        enr = gmspy.list2set(self.db, fleet.enr, 'enr')
        seg = gmspy.list2set(self.db, fleet.seg, 'seg')
        demeq =  gmspy.list2set(self.db, fleet.demeq, 'demeq')
        dstvar = gmspy.list2set(self.db, fleet.dstvar,'dstvar')
        enreq = gmspy.list2set(self.db, fleet.enreq,'enreq')
        grdeq = gmspy.list2set(self.db, fleet.grdeq,'grdeq')
        inityear = gmspy.list2set(self.db, fleet.inityear,'inityear')
        lfteq = gmspy.list2set(self.db, fleet.lfteq,'lfteq')
        sigvar = gmspy.list2set(self.db, fleet.sigvar,'sigvar')
        veheq = gmspy.list2set(self.db, fleet.veheq,'veheq')
        optyear = gmspy.list2set(self.db, fleet.optyear,'optyear')

        veh_oper_dist = gmspy.df2param(self.db, fleet.veh_oper_dist, ['year'], 'VEH_OPER_DIST')
        veh_stck_tot = gmspy.df2param(self.db, fleet.veh_stck_tot, ['year'], 'VEH_STCK_TOT')
        veh_stck_int_seg = gmspy.df2param(self.db, fleet.veh_stck_int_seg, ['seg'], 'VEH_STCK_INT_SEG')
        bev_capac = gmspy.df2param(self.db, fleet.seg_batt_caps, ['seg'], 'BEV_CAPAC')

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
        veh_partab = gmspy.df2param(self.db, fleet.veh_partab, ['veheq', 'tec', 'seg', 'sigvar'], 'VEH_PARTAB')

        veh_add_grd = self.db.add_parameter_dc('VEH_ADD_GRD', ['grdeq','tec'])
        
        # Prep work making add gradient df from given rate constraint
        
        # adding growth constraint for each tec    
        for keys,value in iter(fleet.veh_add_grd.items()):
            veh_add_grd.add_record(keys).value = value

#        veh_add_grd = gmspy.df2param(self.db,self.veh_add_grd, ['grdeq','tec'], 'VEH_ADD_GRD')
    
        gro_cnstrnt = gmspy.df2param(self.db, fleet.gro_cnstrnt, ['year'],'GRO_CNSTRNT')
        
        manuf_cnstrnt = gmspy.df2param(self.db, fleet.manuf_cnstrnt, ['year'],'MANUF_CNSTRNT')
        
        enr_partab = gmspy.df2param(self.db, fleet.enr_partab,['enr','enreq','sigvar'],'ENR_PARTAB')
        
        print('\n exporting database...'+filename+'_input')
        self.db.suppress_auto_domain_checking = 1
        self.db.export(os.path.join(self.current_path, filename+'_input'))

    def get_output_from_GAMS(self, gams_db, output_var):
         temp_GMS_output = []
         temp_index_list = []
         
         for rec in gams_db[output_var]:
            if gams_db[output_var].number_records == 1: # special case for totc
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
     
    def update_fleet(self, fleet):
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
            
    def run_GAMS(self, fleet, run_tag, filename):
        # Pass to GAMS all necessary sets and parameters
        self._load_experiment_data_in_gams(fleet, filename)
        #self.db.export(' _custom.gdx')
        
        #Run GMS Optimization
        
        """"    # create a GAMSModelInstance and solve it with single links in the network blocked
    mi = cp.add_modelinstance()
    x = mi.sync_db.add_variable("x", 2, VarType.Positive)
    xup = mi.sync_db.add_parameter("xup", 2, "upper bound on x")
    # instantiate the GAMSModelInstance and pass a model definition and GAMSModifier to declare upper bound of X mutable
    mi.instantiate("transport use lp min z", GamsModifier(x, UpdateAction.Upper, xup))
    mi.solve() """
        try:
            model_run = self.ws.add_job_from_file(fleet.gms_file) # model_run is type GamsJob
            
            opt = self.ws.add_options()
            opt.defines["gdxincname"] = self.db.name
            print('\n'+ f'Using input gdx file: {self.db.name}')
            print('running GAMS model, please wait...')
            model_run.run(opt, databases=self.db)#,create_out_db = True)
            print('\n Ran GAMS model: ' + fleet.gms_file)

            gams_db = model_run.out_db
            self.export_model = os.path.join(self.export_fp, run_tag+'_solution.gdx')
            gams_db.export(self.export_model)
            print('\n' + f'Completed export of solution database to {self.export_model}')
            
            """ Fetch model outputs"""
#            fleet.totc = self.get_output_from_GAMS(gams_db,'TOTC')
#            fleet.totc_opt = self.get_output_from_GAMS(gams_db,'TOTC_OPT')
            
        except:
            print('\n' + f'ERROR in running model {fleet.gms_file}')
            exceptions = self.db.get_database_dvs()
            try:
                print(exceptions.symbol.name)
            except:
                print(exceptions) 
        
        fleet.read_gams_db(gams_db)
        fleet.import_model_results()
                
        
        """" Test calculation for average lifetime vehicle (~12 years)"""
        def quality_check(age=12):
            fleet.veh_oper_cint_avg = fleet.veh_oper_cint.index.levels[4].astype(int)
            ind = fleet.veh_oper_cint.index
            fleet.veh_oper_cint.index.set_levels(ind.levels[4].astype(int),level=4,inplace=True) #set ages as int
            fleet.veh_oper_cint.sort_index(level='age', inplace=True)
            fleet.veh_oper_cint.sort_index(level='age', inplace=True)
            fleet.veh_oper_cint_avg = fleet.veh_oper_cint.reset_index(level='age')
            fleet.veh_oper_cint_avg = fleet.veh_oper_cint_avg[fleet.veh_oper_cint_avg.age<=age] # then, drop ages over lifetime 
            fleet.veh_oper_cint_avg = fleet.veh_oper_cint_avg.set_index([fleet.veh_oper_cint_avg.index, fleet.veh_oper_cint_avg.age])
            fleet.veh_oper_cint_avg.drop(columns='age', inplace=True)
            fleet.veh_oper_cint_avg = fleet.veh_oper_cint_avg.reorder_levels(['tec','enr','seg','reg','age','modelyear','prodyear'])
    
            fleet.avg_oper_dist = fleet.full_oper_dist.reset_index(level='age') 
            fleet.avg_oper_dist = fleet.avg_oper_dist[fleet.avg_oper_dist.age <= age]  # again, drop ages over lifetime 
            fleet.avg_oper_dist = fleet.avg_oper_dist.set_index([fleet.avg_oper_dist.index, fleet.avg_oper_dist.age]) # make same index for joining with fleet.veh_oper_cint_avg
            fleet.avg_oper_dist.drop(columns='age', inplace=True)
            fleet.avg_oper_dist = fleet.avg_oper_dist.reorder_levels(['tec','enr','seg','reg','age','modelyear','prodyear'])
    #        fleet.op_emissions_avg = fleet.veh_oper_cint_avg.multiply(fleet.avg_oper_dist)
            fleet.d = fleet.avg_oper_dist.join(fleet.veh_oper_cint_avg, lsuffix='_dist')
            fleet.d.columns=['dist','intensity']
            fleet.op_emissions_avg = fleet.d.dist * fleet.d.intensity
            fleet.op_emissions_avg.index = fleet.op_emissions_avg.index.droplevel(level=['enr']) # these columns are unncessary/redundant
            fleet.op_emissions_avg.to_csv('op_emiss_avg_with_duplicates.csv')
            fleet.op_emissions_avg = fleet.op_emissions_avg.reset_index().drop_duplicates().set_index(['tec','seg','reg','age','modelyear','prodyear'])
            fleet.op_emissions_avg.to_csv('op_emiss_avg_without_duplicates.csv')
#            fleet.op_emissions_avg = fleet.op_emissions_avg.drop_duplicates() # replaced by reset_index/drop_duplicates/set_index above
            fleet.op_emissions_avg = fleet.op_emissions_avg.sum(level=['tec','seg','reg','prodyear']) # sum the operating emissions over all model years
            fleet.op_emissions_avg = fleet.op_emissions_avg.reorder_levels(order=['tec','seg','reg','prodyear']) # reorder MultiIndex to add production emissions
            fleet.op_emissions_avg.columns = ['']
            return fleet.op_emissions_avg.add(fleet.veh_prod_cint, axis=0) 
        
        fleet.LC_emissions_avg = [quality_check(i) for i in range(0, 28)]

#            fleet.LC_emissions_avg = fleet.op_emissions_avg.add(fleet.veh_prod_cint) 
               
#        with pd.ExcelWriter('troubleshooting_output_1.xlsx') as writer:
#            fleet.veh_oper_dist.to_excel(writer,sheet_name='veh_oper_dist')
#            fleet.veh_oper_cint.to_excel(writer,sheet_name='veh_oper_cint')
#            fleet.veh_prod_cint.to_excel(writer,sheet_name='veh_prod_cint')
#            fleet.LC_emissions.to_excel(writer,sheet_name='LC_emissions')

#        fleet.enr_cint = fleet._p_dict['ENR_CINT'].stack()
#        fleet.enr_cint.index.rename(['enr', 'reg', 'year'], inplace=True)
        
        add_gpby = fleet.stock_add.sum(axis=1).unstack('seg').unstack('tec')
        fleet.add_share = add_gpby.div(add_gpby.sum(axis=1), axis=0)
        fleet.add_share.dropna(how='all', axis=0, inplace=True) # drop production country (no fleet)
        """ Export technology shares in 2030 to evaluate speed of uptake"""
        fleet.shares_2030 = fleet.add_share.loc(axis=0)[:,'2030']#.to_string()
        fleet.shares_2050 = fleet.add_share.loc(axis=0)[:,'2050']
        
        try:
            fleet.eq = fleet._e_dict['EQ_STCK_GRD']
        except:
            print('No equation EQ_STCK_GRD')
            
        """ Export first year of 100% BEV market share """
        tec_shares = fleet.add_share.stack().stack().sum(level=['prodyear', 'tec','reg'])
        temp_full_year = ((tec_shares.unstack('reg').loc(axis=0)[:, 'BEV']==1).idxmax()).tolist()
        fleet.full_BEV_year = [int(i[0]) if int(i[0])>1999 else np.nan for i in temp_full_year]
#        if fleet.full_BEV_year == 1999:
#            fleet.full_BEV_year = np.nan
        temp = fleet.veh_stck.unstack(['year', 'tec']).sum()