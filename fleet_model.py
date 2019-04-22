# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:27:57 2019

@author: chrishun
"""

class fleetModel:
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
    def __init__(self,data_from_message):
        """ static input data.....hardcoded and/or read in from Excel? """
        self.battery_specs = pd.DataFrame() # possible battery_sizes (and acceptable segment assignments, CO2 production emissions, critical material content, mass)
        self.fuelcell_specs = pd.DataFrame() # possible fuel cell powers (and acceptable segment assignments, CO2 production emissions, critical material content, fuel efficiency(?), mass)
        self.lightweighting = pd.DataFrame() # lightweighting data table - lightweightable materials and coefficients for corresponding lightweighting material(s)
        
        self.el_intensity = data_from_message # regional el-mix intensities as time series from MESSAGE
        self.trsp_dem = data_from_message # EUR transport demand as time series from MESSAGE
        """ boundary conditions for constraints, e.g., electricity market supply constraints, crit. material reserves? could possibly belong in experiment specifications as well..."""
        
        """ GAMS-relevant attributes"""
        # first, sets
        self.tecs = [] # list of techs
        self.cohort = []
        self.crit_mtls = []
        
        # second, intializing data for GAMS
        self.init_stock = []
        self.lifetime_distr = []
    
        # second, expected GAMS outputs
        self.BEV_fraction = pd.DataFrame()
        self.ICEV_fraction = pd.DataFrame()            
        self.totc = 0
        self.BEV_ADD_blaaaah = pd.DataFrame()
        self.VEH_STCK = pd.DataFrame()
        
        """ experiment specifications """
        self.recycling_losses = pd.DataFrame() # vector of material-specific recycling loss factors
        self.fossil_scenario = pd.DataFrame() # adoption of unconventional sources for fossil fuel chain
        self.hydrogen_scenario = pd.DataFrame()
        
        self.occupancy_rate = None # vkm -> pkm conversion
        self.battery_density = None # time series of battery energy densities
        self.lightweighting_scenario = None # lightweighting scenario - yes/no (or gradient, e.g., none/mild/aggressive?)
        
"""
'Real' methods
"""
    def main(self):
        #
        
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
        # send stuff to GAMS and run AHS code
        pass

    def calc_crit_materials(self):
        # performs critical material mass accounting
        pass

    def post_processing(self):
        # make pretty figures?
        pass

    def vis_GAMS(self):
        """ visualize key GAMS parameters for quality checks"""
        pass


"""
Intermediate methods
"""

    def elmix(self):
        # produce time series of elmix intensities, regions x year 
        pass

class ecoinvent_manipulator(data_from_message, A, F):
    """ generates time series of ecoinvent using MESSAGE inputs"""

    def __init__(self):
        self.A = A #default ecoinvent A matrix
        self.F = F #default ecionvent F matrix

    def elmix_subst(self):
        # substitute MESSAGE el mixes into ecoinvent
        pass

