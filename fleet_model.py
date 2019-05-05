# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt

import gams

import gmspy
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
       #self.gdx_file = 'C:\\Users\\chrishun\\Box Sync\\YSSP_temp\\EVD4EUR_input.gdx'#EVD4EUR_ver098.gdx'
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
        self.tecs = ['ICE','BEV']
        self.cohort = [2000+i for i in range(51)]
        self.age = [i for i in range(21)]
        self.enr = ['ELC','FOS']
        self.demeq= ['STCK_TOT','OPER_DIST','OCUP']
        self.dstvar=['mean','stdv']
        self.enreq=['CINT']
        self.grdeq=['IND','ALL']
        self.inityear=[2000+i for i in range(21)]
        self.lfteq=['LFT_DISTR','AGE_DISTR']
        self.sigvar=['A','B','r','t','u']
        # --------------- GAMS Parameters -------------------------------------

        # "Functional unit" # TODO: this is redud
        # Eurostat road_tf_veh [vkm]
        # Eurostat road_tf_vehage [vkm, cohort] NB: very limited geographic spectrum
        # Eurostat road_pa_mov [pkm]
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
        # Eurostat road_eqs_carpda[tec]
        # Eurostat road_eqs_carage [age - <2, 2-5, 5-10, 10-20]; 
        self.veh_stck_int = pd.DataFrame()  # [tech, age] TODO Is this the right one

        # filters
        self.enr_veh = pd.DataFrame()       # [enr, tec]
        self.veh_pay = pd.DataFrame()       # [cohort, age, year]
        
        #self.age_par = pd.DataFrame()
        #self.year_par = pd.DataFrame()

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
        self.veh_pay = gmspy.param2series('VEH_PAY', db) # series, otherwise makes giant sparse dataframe


    def _load_experiment_data_in_gams(self): # will become unnecessary as we start calculating/defining sets and/or parameters within the class
        tecs = gmspy.list2set(self.db, self.tecs, 'tec')
        cohort = gmspy.list2set(self.db, self.cohort, 'year')
        age = gmspy.list2set(self.db, self.age, 'age')
        enr = gmspy.list2set(self.db, self.enr, 'enr')
        demeq=  gmspy.list2set(self.db, self.demeq, 'demeq')
        dstvar = gmspy.list2set(self.db,self.dstvar,'dstvar')
        enreq = gmspy.list2set(self.db,self.enreq,'enreq')
        grdeq = gmspy.list2set(self.db,self.grdeq,'grdeq')
        inityear = gmspy.list2set(self.db,self.inityear,'inityear')
        lfteq = gmspy.list2set(self.db,self.lfteq,'lfteq')
        sigvar = gmspy.list2set(self.db,self.sigvar,'sigvar')
        
      
        #self.add_to_GAMS()

        veh_oper_dist = gmspy.df2param(self.db, self.veh_oper_dist, ['year'], 'VEH_OPER_DIST')
        veh_stck_tot = gmspy.df2param(self.db, self.veh_stck_tot, ['year'], 'VEH_STCK_TOT')

        veh_prod_cint = gmspy.df2param(self.db, self.veh_prod_cint, [tecs, cohort], 'VEH_PROD_CINT')
        veh_oper_cint = gmspy.df2param(self.db, self.veh_oper_cint, [tecs, enr, cohort], 'VEH_OPER_CINT')
        veh_eolt_cint = gmspy.df2param(self.db, self.veh_eolt_cint, [tecs, cohort], 'VEH_EOLT_CINT')

        veh_lift_cdf = gmspy.df2param(self.db, self.veh_lift_cdf, [age], 'VEH_LIFT_CDF')
        veh_lift_age = gmspy.df2param(self.db, self.veh_lift_age, [age], 'VEH_LIFT_AGE')

        veh_stck_int = gmspy.df2param(self.db, self.veh_stck_int, [tecs, age], 'VEH_STCK_INT')#cohort], 'VEH_STCK_INT')

        enr_veh = gmspy.df2param(self.db, self.enr_veh, [enr, tecs], 'ENR_VEH')

        veh_pay = gmspy.df2param(self.db, self.veh_pay, [cohort, age, cohort], 'VEH_PAY')
        
        #age_par = gmspy.df2param(self.db,self.age_par, [age], 'AGE_PAR')
        #year_par = gmspy.df2param(self.db,self.year_par, [year], 'YEAR_PAR')
        
        print('exporting database...')
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
        #self.db.export('troubleshooting.gdx')
        #Run GMS Optimization
        try:
            model_run = self.ws.add_job_from_file(self.gms_file)
        
            model_run.run(databases=self.db,create_out_db = True)
            print("Ran GAMS model: "+self.gms_file)
            gams_db=model_run.out_db
            gams_db.export(os.path.join(self.current_path,'test_v2_db.gdx'))
            self.export_fp = os.path.join(self.current_path,'test_v2_db.gdx')
            print("Completed export of " + self.export_fp)
        #print("x(" + rec.keys[0] + "," + rec.keys[1] + "): level=" + str(rec.level) + " marginal=")# + str(rec.marginal)
        except:
            exceptions = fleet.db.get_database_dvs()
            print(exceptions.symbol.name)
           # self.db.export(os.path.join(self.current_path,'troubleshooting_tryexcept'))

    def add_to_GAMS(self):
        # Adding sets
        def build_set(set_list=None,name=None,desc=None):
            i = self.db.add_set(name,1,desc)
            for s in set_list:
                i.add_record(str(s))
                
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

    # The total (100%) minus the cummulation of all the cars retired by the time they reach a certain age
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

