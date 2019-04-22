# -*- coding: utf-8 -*-

import gdxpds
import pandas as pd
import matplotlib
from plotly import tools
import plotly.plotly as py
import plotly.offline as pyo
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.tools as tls
import dash
import plotly_express as px
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
from plotly.offline import init_notebook_mode, iplot
pyo.init_notebook_mode()



"""
Created on Sun Apr 21 13:27:57 2019

@author: chrishun
"""
import pandas as pd
import numpy as np
import logging
import gams

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
        gdx_file = 'C:\\Users\\chrishun\\Box Sync\\YSSP_temp\\EVD4EUR_ver098.gdx'
        dataframes = gdxpds.to_dataframes(gdx_file)
        
        # Make vehicle stock dataframe
        stock_data = pd.concat((dataframes['ICE_STCK'],dataframes['BEV_STCK'].drop(columns=['age','year'])),axis=1)
        stock_data =stock_data.set_index(['age','year'])
        stock_data.drop(columns=['Lower','Upper','Scale','Marginal'],inplace=True)
        stock_data.columns=['ICE STCK','BEV STCK']
        
        #stock_data.groupby(level=[1]).sum().plot(kind='area')
        
        stock_data = stock_data.sum(axis=1).unstack()
        stock_data.index = stock_data.index.astype(int)
        stock_data = stock_data.sort_index()
        
        # Plot age distribution of fleet as time series
        ax = stock_data.T.plot.area(cmap='Spectral_r')
        patches, labels = ax.get_legend_handles_labels()
        ax.legend(bbox_to_anchor=(1.1,1), ncol=2, title='Vehicle ages')
        
        # Plot additions to stock, by technology
        pl1=go.Bar(x=dataframes['BEV_STCK_ADD']['year'],y=dataframes['BEV_STCK_ADD']['Level'], name='BEV_STCK_ADD')
        pl2=go.Bar(x=dataframes['ICE_STCK_ADD']['year'],y=dataframes['ICE_STCK_ADD']['Level'], name='ICE_STCK_ADD')
        data=[pl1, pl2]
        iplot(data)
        
        # Plot removals from stock by technology
        pl1=go.Bar(x=dataframes['BEV_STCK_REM']['year'],y=dataframes['BEV_STCK_REM']['Level'], name='BEV_STCK_REM')
        pl2=go.Bar(x=dataframes['ICE_STCK_REM']['year'],y=dataframes['ICE_STCK_REM']['Level'], name='ICE_STCK_REM')
        data=[pl1, pl2]
        iplot(data)
        
        # Plot total carbon emissions by technology
        pl1=go.Bar(x=dataframes['BEV_TOTC']['year'],y=dataframes['BEV_TOTC']['Level'], name='BEV_TOTC')
        pl2=go.Bar(x=dataframes['ICE_TOTC']['year'],y=dataframes['ICE_TOTC']['Level'], name='ICE_TOTC')
        pl3=go.Scatter(x=dataframes['BEV_TOTC']['year'],y=dataframes['BEV_TOTC']['Level']+dataframes['ICE_TOTC']['Level'],name='Total CO2')
        data=[pl1, pl2, pl3]
        iplot(data)
        
        # Plot total vehicle production emissions by technology
        pl1=go.Scatter(x=dataframes['ICE_PROD_TOTC']['year'],y=dataframes['ICE_PROD_TOTC']['Level'], name='ICE_PROD_TOTC')
        pl2=go.Scatter(x=dataframes['BEV_PROD_TOTC']['year'],y=dataframes['BEV_PROD_TOTC']['Level'], name='BEV_PROD_TOTC')
        data=[pl1, pl2]
        iplot(data)
        
        # Plot vehicle peroduction emissions without electricity-related emissions
        pl1=go.Scatter(x=dataframes['BEV_PROD_CINT']['prodyear'],y=dataframes['BEV_PROD_CINT']['Value'], name='BEV_PROD_CINT')
        pl2=go.Scatter(x=dataframes['ICE_PROD_CINT']['prodyear'],y=dataframes['ICE_PROD_CINT']['Value'],name='ICE_PROD_CINT')
        data=[pl1, pl2]
        iplot(data)
        
        # Plot decision variables as time series to check
        pl1=go.Scatter(x=dataframes['VEH_OPER_DIST']['year'],y=dataframes['VEH_OPER_DIST']['Value'], name='VEH_OPER_DIST')
        pl2=go.Scatter(x=dataframes['ELC_CINT']['year'],y=dataframes['ELC_CINT']['Value'], yaxis='y2', name='ELC_CINT')
        #pl2=go.Scatter(x=dataframes['VEH_LIFT_DIST']['age'],y=dataframes['VEH_LIFT_DIST']['Level'], name='VEH_LIFT_DIST')
        #pl1=go.Scatter(x=dataframes['VEH_OPER_DIST']['age'],y=dataframes['VEH_OPER_DIST']['Level'], name='VEH_OPER_DIST')
        data=[pl1, pl2]
        layout = go.Layout(
            yaxis=dict(
                title='VEH_OPER_DIST'
            ),
            yaxis2=dict(
                title='ELC_CINT (g CO2-eq/kWh)',
                overlaying='y',
                side='right')
        )
        fig=go.Figure(data,layout=layout)
        py.iplot(fig)
                

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

