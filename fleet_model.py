# -*- coding: utf-8 -*-

import gdxpds
import gams
import pandas as pd
import numpy as np
import gmspy

import matplotlib
import matplotlib.pyplot as plt
"""from plotly import tools
import cufflinks as cf
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
from plotly.offline import init_notebook_mode, iplot"""




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
        gdx_file = 'C:\\Users\\chrishun\\Box Sync\\YSSP_temp\\EVD4EUR_ver098.gdx'
        dataframes = gdxpds.to_dataframes(gdx_file)
        
        """ static input data.....hardcoded and/or read in from Excel? """
        self.battery_specs = pd.DataFrame() # possible battery_sizes (and acceptable segment assignments, CO2 production emissions, critical material content, mass)
        self.fuelcell_specs = pd.DataFrame() # possible fuel cell powers (and acceptable segment assignments, CO2 production emissions, critical material content, fuel efficiency(?), mass)
        self.lightweighting = pd.DataFrame() # lightweighting data table - lightweightable materials and coefficients for corresponding lightweighting material(s)
        
        self.el_intensity = data_from_message # regional el-mix intensities as time series from MESSAGE
        self.trsp_dem = data_from_message # EUR transport demand as time series from MESSAGE
        """ boundary conditions for constraints, e.g., electricity market supply constraints, crit. material reserves? could possibly belong in experiment specifications as well..."""
        
        """ GAMS-relevant attributes"""
        # first, sets

        # NOTE: as sets are not supposed to be modified over time, it might be safer to define them as tupples instead
        # of lists... immutable
        self.tecs = [] # list of techs
        self.cohort = []
        self.age = []
        self.enr = []
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

        """ Optimization Initialization """
        self.ws = gams.GamsWorkspace()
        self.db = self.ws.add_database()

        
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

        def build_set(var, name, comment):
            """ Simple convenience insert sets"""
            a_set = self.db.add_set(name, 1, comment)
            for v in var:
                a_set.add_record(v)
            return a_set

        def build_param(var, domains, name, comment):
            a_param = db.add_parameter_dc(name, domains, comment)
            for keys, data in var.items():
                a_param.add_record(keys).value = data
            return a_param

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
        gdx_file = 'C:\\Users\\chrishun\\Box Sync\\YSSP_temp\\EVD4EUR_ver098.gdx'
        sets = gmspy.ls(gdx_filepath=gdx_file, entity='Set')
        parameters = gmspy.ls(gdx_filepath=gdx_file,entity='Parameter')
        variables = gmspy.ls(gdx_filepath=gdx_file,entity='Variable')
        
        years = gmspy.set2list(sets[0], gdx_filepath=gdx_file)

        # Export parameters
        p_dict = {}
        for p in parameters:
            p_dict[p] = gmspy.param2series(p,gdx_filepath=gdx_file)
            
            
        p_df = pd.DataFrame(index=years)
        p_df.index.name='year'
        for key in p_dict:
            if len(p_dict[key])==len(years):
                p_dict[key].rename_axis('year',inplace=True)
                #p_df=pd.concat([p_df,p_dict[key]],axis=1,join_axes=[p_df.index])
                #p_df= p_df.join(p_dict[key],how='outer')
                p_df= pd.merge(p_df,p_dict[key].rename(key),on='year')#left_index=True,right_index=True)
            else:
                pass
                    #print(key)
        p_df.drop(['YEAR_PAR','PRODYEAR_PAR'],axis=1,inplace=True)
        
        
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

        # Plot total stocks by age
        stock_df = pd.concat((v_dict['ICE_STCK'].unstack(),v_dict['BEV_STCK'].unstack()),axis=1)
        stock_df.columns=['ICE_STCK','BEV_STCK']
        tot_stock_df = stock_df.sum(axis=1).unstack().T
        tot_stock_df.index = tot_stock_df.index.astype(int)
        tot_stock_df.sort_index(axis=0,inplace=True)
        ax = tot_stock_df.T.plot.area(cmap='Spectral_r',title='Total stocks by vehicle age')
        patches, labels = ax.get_legend_handles_labels()
        ax.legend(bbox_to_anchor=(1.1,1), ncol=2, title='Vehicle ages')
        
        # Plot total stocks by technology
        stock_df.groupby(level=[0]).sum(axis=1).plot(kind='area',title='Total stocks by technology')
        
        stock_df = pd.concat([p_dict['ICE_STCK_TOT'],p_dict['BEV_STCK_TOT']],axis=1)
        stock_df.columns=['ICE_STCK_TOT','BEV_STCK_TOT']
        stock_df.plot(title='Total stocks by technology')
        
        # Plot stock additions and removals by technology
        add_rem_df = pd.concat((v_dict['ICE_STCK_REM'].unstack(),v_dict['BEV_STCK_REM'].unstack(),v_dict['ICE_STCK_ADD'].unstack(),v_dict['BEV_STCK_ADD'].unstack()),axis=1)
        add_rem_df.columns = ['ICE_STCK_REM','BEV_STCK_REM','ICE_STCK_ADD','BEV_STCK_ADD']
        #add_rem_df.plot(subplots=True,title='Stock removal and addition variables')
        for column in add_rem_df:
            ax=add_rem_df[column].unstack().plot(kind='area',cmap='Spectral_r',title=column)
            patches, labels = ax.get_legend_handles_labels()
            ax.legend(bbox_to_anchor=(1.1,1), ncol=2, title='Vehicle ages')

        # Plot carbon emissions by technology and lifecycle phase
        totc_df=pd.concat((v_dict['ICE_PROD_TOTC'],v_dict['ICE_OPER_TOTC'],v_dict['ICE_EOLT_TOTC'],v_dict['ICE_TOTC'],v_dict['BEV_PROD_TOTC'],v_dict['BEV_OPER_TOTC'],v_dict['BEV_EOLT_TOTC'],v_dict['BEV_TOTC']),axis=1)
        totc_df.columns=['ICE_PROD_TOTC','ICE_OPER_TOTC','ICE_EOLT_TOTC','ICE_TOTC','BEV_PROD_TOTC','BEV_OPER_TOTC','BEV_EOLT_TOTC','BEV_TOTC']
        totc_df.plot(title='Total carbon emissions by technology and lifecycle phase')
        
        # Plot parameter values for quality assurance
        ax= p_df.plot(subplots=True,figsize=(15,50),title='Parameter values')
        
        #dataframes = gdxpds.to_dataframes(gdx_file)
        #pyo.init_notebook_mode()
        
        # Make vehicle stock dataframe
        """stock_data = pd.concat((dataframes['ICE_STCK'],dataframes['BEV_STCK'].drop(columns=['age','year'])),axis=1)
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
        ax.legend(bbox_to_anchor=(1.1,1), ncol=2, title='Vehicle ages') """
        
        # Plot additions to stock, by technology
        """pl1=go.Bar(x=dataframes['BEV_STCK_ADD']['year'],y=dataframes['BEV_STCK_ADD']['Level'], name='BEV_STCK_ADD')
        pl2=go.Bar(x=dataframes['ICE_STCK_ADD']['year'],y=dataframes['ICE_STCK_ADD']['Level'], name='ICE_STCK_ADD')
        data=[pl1, pl2]
        iplot(data)"""
        
        
        # Plot removals from stock by technology
        """pl1=go.Bar(x=dataframes['BEV_STCK_REM']['year'],y=dataframes['BEV_STCK_REM']['Level'], name='BEV_STCK_REM')
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
        py.iplot(fig)"""
                

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
