# -*- coding: utf-8 -*-

import gams
import pandas as pd
import numpy as np
import gmspy

import matplotlib
import matplotlib.pyplot as plt

import os

"""
Created on Sun Apr 21 13:27:57 2019

@author: chrishun
"""

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
        self.gdx_file = 'C:\\Users\\chrishun\\Box Sync\\YSSP_temp\\EVD4EUR_ver098.gdx'
        self.gms_file = 'C:\\Users\\chrishun\\Box Sync\\YSSP_temp\\EVD4EUR.gms'#_test.gms'#EVD4EUR.gms'
        self.export_fp = ''
        
        """ static input data.....hardcoded and/or read in from Excel? """
        self.battery_specs = pd.DataFrame() # possible battery_sizes (and acceptable segment assignments, CO2 production emissions, critical material content, mass)
        self.fuelcell_specs = pd.DataFrame() # possible fuel cell powers (and acceptable segment assignments, CO2 production emissions, critical material content, fuel efficiency(?), mass)
        self.lightweighting = pd.DataFrame() # lightweighting data table - lightweightable materials and coefficients for corresponding lightweighting material(s)
        
        self.el_intensity = data_from_message # regional el-mix intensities as time series from MESSAGE
        self.trsp_dem = data_from_message # EUR transport demand as time series from MESSAGE
        """ boundary conditions for constraints, e.g., electricity market supply constraints, crit. material reserves? could possibly belong in experiment specifications as well..."""
        
        """ GAMS-relevant attributes"""
        #  --------------- GAMS sets / domains -------------------------------
        self.tecs = []
        self.cohort = []
        self.age = []
        self.enr = []

        # --------------- GAMS Parameters -------------------------------------

        # "Functional unit" # TODO: this is redud
        self.veh_oper_dist = pd.Series()     # [years] driving distance each year # TODO: rename?
        self.veh_stck_tot = pd.Series()      # [years]

        # Life cycle intensities
        self.veh_prod_cint = pd.DataFrame()  # [tecs, cohort]
        self.veh_oper_cint = pd.DataFrame()  # [[tecs, enr], cohort]
        self.veh_eolt_cint = pd.DataFrame()  # [tecs, cohort]

        # Fleet dynamics
        self.veh_lift_cdf = pd.DataFrame()  # [age] TODO Is it this one we feed to gams?
        self.veh_lift_age = pd.Series()     # [age]

        # Initial stocks
        self.veh_stck_int = pd.DataFrame()  # [tech, age] TODO Is this the right one

        # filters
        self.enr_veh = pd.DataFrame()       # [enr, tec]
        self.veh_pay = pd.DataFrame()       # [cohort, age, year]

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
        self.ws = gams.GamsWorkspace()
        self.db = self.ws.add_database()

        
    def main(self):
        #
        pass
        
    def read_all_sets(self, gdx_file):
        db = gmspy._iwantitall(None, None, gdx_file)
        self.tecs = gmspy.set2list('tec', db)
        self.cohort = gmspy.set2list('year', db)
        self.age = gmspy.set2list('age', db)
        self.enr = gmspy.set2list('enr', db)

    def _read_all_final_parameters(self, a_file):
        db = gmspy._iwantitall(None, None, a_file)

        self.veh_oper_dist = gmspy.param2series('VEH_OPER_DIST', db)
        self.veh_stck_tot = gmspy.param2series('VEH_STCK_TOT', db)

        self.veh_prod_cint = gmspy.param2df('VEH_PROD_CINT', db)
        self.veh_oper_cint = gmspy.param2df('VEH_OPER_CINT', db)
        self.veh_eolt_cint = gmspy.param2df('VEH_EOLT_CINT', db)

        self.veh_lift_cdf = gmspy.param2series('VEH_LIFT_CDF', db)
        self.veh_lift_age = gmspy.param2series('VEH_LIFT_AGE', db)

        self.veh_stck_int = gmspy.param2df('VEH_STCK_INT', db)

        self.enr_veh = gmspy.param2df('ENR_VEH', db)
        self.veh_pay = gmspy.param2series('VEH_PAY', db)

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
        model_run = self.ws.add_job_from_file(self.gms_file)
        model_run.run(create_out_db = True)
        print("Ran GAMS model: "+self.gms_file)
        gams_db=model_run.out_db
        gams_db.export(os.path.join(self.current_path,'test_db.gdx'))
        self.export_fp = os.path.join(self.current_path,'test_db.gdx')
        print("Completed export of " + self.export_fp)
            #print("x(" + rec.keys[0] + "," + rec.keys[1] + "): level=" + str(rec.level) + " marginal=")# + str(rec.marginal)

    def add_to_GAMS(self):
        # Adding sets
        # NOTE: Check that 'cohort', 'year' and 'prodyear' work nicely together
        cohort = build_set(self.cohort, 'year', 'year')
        tec = build_set(self.tecs, 'tec', 'technology')
        age = build_set(self.age, 'age', 'age')
        enr = build_set(self.enr, 'enr', 'energy types')
        
    def calc_crit_materials(self):
        # performs critical material mass accounting
        pass

    def post_processing(self):
        # make pretty figures?
        pass
    

    def vis_GAMS(self):
        """ visualize key GAMS parameters for quality checks"""
        """To do: split into input/output visualization; add plotting of CO2 and stocks together"""
        
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

        def reorder_age_headers(df_unordered):
            temp = df_unordered
            temp.columns = temp.columns.astype(int)
            temp.sort_index(inplace=True,axis=1)
            return temp
        
        def fix_age_legend(ax):
            patches, labels = ax.get_legend_handles_labels()
            ax.legend(bbox_to_anchor=(1.1,1), ncol=2, title='Vehicle ages')

        # Plot total stocks by age
        stock_df=v_dict['VEH_STCK']
        stock_df.columns = stock_df.columns.astype(int)
        stock_df.sort_index(axis=1,inplace=True)
                
        stock_df_grouped =stock_df.groupby(level=[0])
        for name, group in stock_df_grouped:
            ax=group.plot(kind='area',cmap='Spectral_r',title=name+' stock by age')
            fix_age_legend(ax)

        stock_df.columns = stock_df.columns.astype(int)
        stock_df.sort_index(axis=1,inplace=True)
        tot_stock_df=stock_df.sum(axis=0,level=1)
        ax = tot_stock_df.plot.area(cmap='Spectral_r',title='Total stocks by vehicle age')
        fix_age_legend(ax)
        
        # Plot total stocks by technology
        stock_df.sum(axis=1).unstack().T.plot(kind='area', title='Total stocks by technology')  
        stock_df.sum(axis=1).unstack().T.plot(title='Total stocks by technology')

        # Plot stock additions and removals by technology
        temp_vdict_a = reorder_age_headers(v_dict['VEH_STCK_REM']).stack()
        temp_vdict_b = reorder_age_headers(v_dict['VEH_STCK_ADD']).stack()
        add_rem_df = pd.concat([temp_vdict_a, temp_vdict_b],axis=1,keys=('VEH_STCK_REM','VEH_STCK_ADD'))
        
        add_rem_df_2=add_rem_df.stack().unstack(level=[0,3])
        
        for column,variable in add_rem_df_2:
            ax = add_rem_df_2[column][variable].unstack().plot(kind='area',cmap='Spectral_r',title=column+" "+variable)
            fix_age_legend(ax)
        
        add_rem_df_2.plot(subplots=True,title='Stock removal and addition variables')
        

        # Plot carbon emissions by technology and lifecycle phase
        totc_df=pd.concat((v_dict['VEH_PROD_TOTC'],v_dict['VEH_OPER_TOTC'],v_dict['VEH_EOLT_TOTC'],v_dict['VEH_TOTC']),axis=0,keys=('VEH_PROD_TOTC','VEH_OPER_TOTC','VEH_EOLT_TOTC','VEH_TOTC'))
        totc_df=totc_df.T.swaplevel(0,1,axis=1)
        totc_df.plot()
        
        """For later: introduce figure plotting vehicle stock vs emissions"""
        
        # Plot parameter values for quality assurance
#        ax= p_df.plot(subplots=True,title='Parameter values')
                

    """
    Intermediate methods
    """

    def elmix(self):
        # produce time series of elmix intensities, regions x year 
        pass

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
