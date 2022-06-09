# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:54:01 2020

@author: chrishun

This script calculates disaggregates the transition pathways for regions in IAMs
(in this case, MESSAGE) to country level, and uses this to project prospective
electricity mixes at the country level and calculate corresponding consumption
mixes.

This script also clusters countries by their electricity mix intensity.

"""


import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import jenkspy
import country_converter as coco
import re
import os
import openpyxl
import logging

data_fp = os.path.join(os.path.curdir, 'data')

log = logging.getLogger(__name__)

#%% Load electricity footprints from regionalized footprints paper - these are used to determine clustering

import_fp = os.path.join(data_fp, 'el_footprints.csv')
df = pd.read_csv(import_fp, index_col=[0], usecols=[0, 2])
df.sort_values(by='Consumption mix intensity', inplace=True)
df.dropna(axis=0, inplace=True)

# Make lists of countries part of each MESSAGE region

EEU = ['Albania',
       'Bosnia and Herzegovina',
       'Bulgaria',
       'Croatia',
       'Czech Republic',
       'Estonia',
       'The former Yugoslav Rep. of Macedonia',
       'Latvia',
       'Lithuania',
       'Hungary',
       'Poland',
       'Romania',
       'Slovak Republic',
       'Slovenia',
       'Serbia',  # added (in lieu of Yugoslavia)
       'Kosovo',  # added (in lieu of Yugoslavia)
       'Montenegro',  # added (in lieu of Yugoslavia)
       'Yugoslavia']

WEU = ['Andorra',
       'Austria',
       'Azores',
       'Belgium',
       'Canary Islands',
       'Channel Islands',
       'Cyprus',
       'Denmark',
       'Faeroe Islands',
       'Finland',
       'France',
       'Germany',
       'Gibraltar',
       'Greece',
       'Greenland',
       'Iceland',
       'Ireland',
       'Isle of Man',
       'Italy',
       'Liechtenstein',
       'Luxembourg',
       'Madeira',
       'Malta',
       'Monaco',
       'Netherlands',
       'Norway',
       'Portugal',
       'Spain',
       'Sweden',
       'Switzerland',
       'Turkey',
       'United Kingdom']

message_countries = EEU + WEU

EEU_ISO2 = coco.convert(EEU, to='ISO2', not_found=None)
WEU_ISO2 = coco.convert(WEU, to='ISO2', not_found=None)

reg_dict = {country: ('EEU' if country in EEU_ISO2 else 'WEU') for country in (EEU_ISO2 + WEU_ISO2)}

missing_countries = list(set(EEU_ISO2 + WEU_ISO2) - set(df.index.tolist()))

#%% Import country data and shapefile from Natural Earth, filter for the countries in our study

fp_map = os.path.join(os.path.curdir, 'data', 'maps', 'ne_10m_admin_0_countries.shp')
country_shapes = gpd.read_file(fp_map)

# Replace ISO_A2 code for France and Norway
country_dict = {'France': 'FR', 'Norway': 'NO'}  # need to manually add these due to no ISO_A2 entry in .shp file
for index, country_row in country_shapes[country_shapes['ISO_A2'] == '-99'].iterrows():
    country = country_row['NAME_SORT']
    if country in list(country_dict.keys()):
        country_shapes.at[index, 'ISO_A2'] = country_dict[country]

# europe_ind = country_shapes[country_shapes.iloc(axis=1)[:-1].isin(message_countries)].dropna(axis=0, how='all').index.tolist()

europe_shapes = country_shapes.cx[-19:34, 32:75]  # filter to only the countries within the bounds of our map figures; NB: this excludes Greenland, Iceland, Canary Islands


#%% Find relevant countries missing consumption mix data and add proxies

europe_shapes = europe_shapes.join(df, on='ISO_A2')  # add cluster and consumption mix intensity data to mapping dataframe

# Determine share of total European population from countries that require proxies
missing_pop = europe_shapes[europe_shapes['Consumption mix intensity'].isnull()]['POP_EST'].sum()
missing_share = missing_pop / europe_shapes['POP_EST'].sum()

missing_count = europe_shapes[europe_shapes['Consumption mix intensity'].isna()].NAME_CIAWF.values

proxy_CI = {'LU': 505,   # LU/HR from Moro and Lonza
            'HR': 465,
            'AL': df.at['NO', 'Consumption mix intensity'],
            'XK': df.at['PL', 'Consumption mix intensity'],
            'LI': df.at['NO', 'Consumption mix intensity'],
            'AD': df.at['ES', 'Consumption mix intensity'],
            'MC': df.at['FR', 'Consumption mix intensity']}

# kosovo = mostly coal (IEA), albania = mostly hydro (IEA), faroe = 50% fossil fuels/50% hydro, wind
# AD/MC assumed to depend on imports from surroudning countries
# LI mostly from hydro (?)
# XK mostly from fossil/coal

for iso, value in proxy_CI.items():
    ind = europe_shapes[europe_shapes['ISO_A2'] == iso].index.values[0]
    europe_shapes.at[ind, 'Consumption mix intensity'] = value


#%% Import MESSAGE results and visualize el pathways

message_fp = import_fp = os.path.join(data_fp, 'MESSAGE_SSP2_400_ppm.xlsx')  # 'MESSAGE_SSP2_electricity pathways.xlsx')
message_el = pd.read_excel(message_fp, index_col=[0, 1, 2], header=[0], usecols='C:P', skipfooter=7)
message_el.index.rename(['reg', 'MESSAGE tec', 'units'], inplace=True)

# Aggregate onshore and offshore wind due to discrepencies with empirical data
message_wind = message_el.xs('Secondary Energy|Electricity|Wind|Offshore', level=1) + message_el.xs('Secondary Energy|Electricity|Wind|Onshore', level=1)
message_wind['MESSAGE tec'] = 'Secondary Energy|Electricity|Wind'
message_wind.set_index('MESSAGE tec', drop=True, append=True, inplace=True)
message_wind.index = message_wind.index.swaplevel(-1, -2)
message_el = message_el.append(message_wind)
message_el.drop(index=['Secondary Energy|Electricity|Wind|Offshore', 'Secondary Energy|Electricity|Wind|Onshore'], level='MESSAGE tec', inplace=True)
message_el.sort_index(level=0, inplace=True)

# Plot MESSAGE pathways by region
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 5), gridspec_kw={'wspace': 0.05}, dpi=600)

for ax, reg in zip(axes, message_el.index.unique(level=0)):
    message_el.loc[reg].stack().unstack(level=0).plot(ax=ax, kind='bar', stacked=True, legend=False, title=f'{reg}')
    ax.set_xticklabels(message_el.columns.tolist(), rotation=45)

handles, labels = axes[0].get_legend_handles_labels()
labels = [re.split(r'\|', label, maxsplit=2)[-1] for label in labels]  # reformat legend labels
plt.legend(handles, labels, bbox_to_anchor=(1, 1))

axes[0].set_ylabel('Secondary Energy | Electricity \n (EJ/yr)')
plt.title('Electricity technology pathways from MESSAGEix')

#%% Calculate electricity pathways from MESSAGE normalized to 2020 values

message_el_shares = pd.DataFrame()

for reg in message_el.index.unique(level=0):
    temp_df = message_el.loc[reg].div(message_el[2020][reg], level=0, axis=0)
    temp_df = pd.concat([temp_df], keys=[reg], names=['reg'])
    message_el_shares = message_el_shares.append(temp_df)

message_el_shares.index = message_el_shares.index.droplevel(-1)  # drop 'units' column of index
message_el_shares.columns = message_el_shares.columns.astype(int)

# Plot shares
"""fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 5), gridspec_kw={'wspace': 0.05}, dpi=600)

for ax, reg in zip(axes, message_el_shares.index.unique(level=0)):
    plot_shares = (message_el_shares.loc[reg].loc(axis=1)[2020:].stack().unstack(level=0))
    plot_shares.plot(ax=ax, kind='area', legend=False, label=message_el_shares.loc[reg].index, title=f'{reg}')
    ax.set_xticklabels(plot_shares.index.values, rotation=45)

handles, labels = axes[1].get_legend_handles_labels()
labels = [re.split(r'\|', label, maxsplit=2)[-1] for label in labels]  # reformat legend labels
plt.legend(handles, labels, bbox_to_anchor=(1, 1))

axes[0].set_ylabel('Share of electricity technology, \n normalized to 2020 shares')"""

#%% Import and clean trade data from Eurostat (2018 values)

# NB: Eurostat publishes both import and export tables; however, these are not internally consistent (i.e., do not balance),
# and have different geographical resolution.
# Solution: Take mean of values from both tables where these exist, otherwise take value where present in one of the two
# tables.

imp_fp = os.path.join(data_fp, 'nrg_ti_eh.xls')
exp_fp = os.path.join(data_fp, 'nrg_te_eh.xls')

eurostat_import = pd.read_excel(imp_fp, header=0, index_col=[0], skiprows=[0,1,2,3,4,5,6,7,8,9], skipfooter = 3, usecols="A:AS", na_values=":")
eurostat_export = pd.read_excel(exp_fp, header=0, index_col=[0], skiprows=[0,1,2,3,4,5,6,7,8,9], skipfooter = 5, usecols="A:AU", na_values=":")

eurostat_import = eurostat_import.T  # make matrix in producer-receiver format

eurostat_import.index = pd.Index(coco.convert(eurostat_import.index.tolist(), to='ISO2'))
eurostat_import.columns = pd.Index(coco.convert(eurostat_import.columns.tolist(), to='ISO2'))
eurostat_export.index = pd.Index(coco.convert(eurostat_export.index.tolist(), to='ISO2'))
eurostat_export.columns = pd.Index(coco.convert(eurostat_export.columns.tolist(), to='ISO2'))

eurostat_trade = pd.concat([eurostat_import.stack(), eurostat_export.stack()], axis=1).mean(axis=1).unstack()
trades_df = eurostat_trade / 1000

#%% Import electricity data from Eurostat - to be used as 2020 baseline

""" Load electricity mixes, regionalized LCA/hybrid LCA factors from BEV footprints """

# mix_fp = os.path.join(data_fp, 'prod_mixes.csv')  # from ENTSO-E (see extract_bentso.py)
mix_fp = os.path.join(data_fp, 'nrg_bal_peh.xls')  # from Eurostat (2018 data)
hydro_fp = os.path.join(data_fp, 'nrg_ind_pehnf.xls')  # from Eurostat (2018 data)

trades_fp = os.path.join(data_fp, 'el_trades.csv')  # from ENTSO-E (see extract_bentso.py)
tec_int_fp = os.path.join(data_fp, 'tec_intensities.csv')  # hybridized, regionalized LCA factors for electricity generation

mix_df = pd.read_excel(mix_fp, header=0, index_col=[0], skiprows=[0, 1, 2, 3, 4, 5], skipfooter=3, na_values=':')  # 2018 production mix, from Eurostat
hydro_df = pd.read_excel(hydro_fp, header=0, index_col=[0], usecols='A:G', skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], skipfooter=3, na_values=':')  # 2018 production mix, from Eurostat
# trades_df = pd.read_csv(trades_fp, index_col=[0], na_values='-')  # 2019 production, in TWh

mix_df = mix_df / 1000  # Eurostat data in GWh; convert to TWh
hydro_df = hydro_df / 1000


#%% perform calculations on hydropower (for Eurostat data)

hydro_df.replace(np.nan, 0, inplace=True)
hydro_df['Hydro Water Reservoir'] = (hydro_df['Pure hydro power'] - hydro_df['Run-of-river hydro power']) + (hydro_df['Mixed hydro power'] - hydro_df['Mixed hydro power - pumping'])
hydro_df['Hydro Pumped Storage'] = hydro_df['Mixed hydro power - pumping'] + hydro_df['Pumped hydro power']
hydro_df['Hydro Run-of-river and poundage'] = hydro_df['Run-of-river hydro power']

hydro_df.drop(columns=['Hydro', 'Pure hydro power', 'Run-of-river hydro power', 'Mixed hydro power', 'Mixed hydro power - pumping', 'Pumped hydro power'], inplace=True)
mix_df = mix_df.join(hydro_df)
mix_df.drop(columns=['Hydro'], inplace=True)

"""eurostat_dict = {'Anthracite': 'Fossil Hard coal w/o CCS',
                 'Coking coal': 'Fossil Hard coal w/o CCS',
                 'Other bituminous coal': 'Fossil Hard coal w/o CCS',
                 'Sub-bituminous coal': 'Fossil Hard coal w/o CCS',
                 'Lignite': 'Fossil Brown coal/Lignite w/o CCS',
                 'Coke oven coke': 'Fossil Brown coal/Lignite w/o CCS',
                 'Gas coke': 'Fossil Brown coal/Lignite w/o CCS',
                 'Patent fuel': 'Fossil Brown coal/Lignite w/o CCS',
                 'Brown coal briquettes': 'Fossil Brown coal/Lignite w/o CCS',
                 'Coal tar': 'Fossil Brown coal/Lignite w/o CCS',
                 'Manufactured gases':'Fossil Coal-derived gas w/o CCS',
                 'Oil and petroleum products (excluding biofuel portion)': 'Fossil Oil',
                 'Oil shale and oil sands': 'Fossil Oil shale',
                 'Peat and peat products': 'Fossil Peat w/o CCS',
                 'Natural gas': 'Fossil Gas w/o CCS',
                 'Nuclear heat': 'Nuclear',
                 'Wind': 'Wind Onshore',
                 'Solar thermal': 'Solar CSP',
                 'Solar photovoltaic': 'Solar PV',
                 'Primary solid biofuels': 'Biomass w/o CCS',
                 'Charcoal': 'Biomass w/o CCS',
                 'Pure biogasoline': 'Biomass w/o CCS',
                 'Blended biogasoline': 'Biomass w/o CCS',
                 'Pure biodiesels': 'Biomass w/o CCS',
                 'Blended biodiesels': 'Biomass w/o CCS',
                 'Pure bio jet kerosene': 'Biomass w/o CCS',
                 'Blended bio jet kerosene': 'Biomass w/o CCS',
                 'Other liquid biofuels': 'Biomass w/o CCS',
                 'Biogases': 'Biomass w/o CCS',
                 'Geothermal': 'Geothermal',
                 'Tide, wave, ocean': 'Marine'
                 }"""

#%% Replace Eurostat nomenclature with ENTSO for integration with existing code

eurostat_dict = {'Fossil Hard coal': ['Anthracite', 'Coking coal',
                                      'Other bituminous coal', 'Sub-bituminous coal'],
                 'Fossil Brown coal/Lignite': ['Lignite', 'Coke oven coke', 'Gas coke',
                                               'Patent fuel', 'Brown coal briquettes', 'Coal tar'],
                 'Fossil Coal-derived gas': 'Manufactured gases',
                 'Fossil Oil': 'Oil and petroleum products (excluding biofuel portion)',
                 'Fossil Oil shale': 'Oil shale and oil sands',
                 'Fossil Peat': 'Peat and peat products',
                 'Fossil Gas': 'Natural gas',
                 'Nuclear': 'Nuclear heat',
                 'Wind': 'Wind',
                 'Solar CSP': 'Solar thermal',
                 'Solar PV': 'Solar photovoltaic',
                 'Biomass': ['Primary solid biofuels', 'Charcoal', 'Pure biogasoline',
                                     'Blended biogasoline', 'Pure biodiesels', 'Blended biodiesels',
                                     'Pure bio jet kerosene', 'Blended bio jet kerosene', 'Other liquid biofuels', 'Biogases'],
                 'Geothermal': 'Geothermal',
                 'Marine': 'Tide, wave, ocean',
                 'Waste': ['Renewable municipal waste', 'Non-renewable waste']
                 }

for new_tec, eurostat_tec in eurostat_dict.items():
    if isinstance(eurostat_tec, list):
        mix_df[new_tec] = mix_df.loc(axis=1)[eurostat_tec].sum(axis=1)
        mix_df.drop(columns=eurostat_tec, inplace=True)
    else:
        mix_df.rename(columns={eurostat_tec: new_tec}, inplace=True)

# Technologies to remove
drop_tecs = ['Total', 'Solid fossil fuels', 'Coke oven gas', 'Gas works gas', 'Blast furnace gas',
             'Other recovered gases', 'Peat', 'Peat products', 'Crude oil', 'Natural gas liquids',
             'Refinery gas', 'Liquefied petroleum gases', 'Naphtha', 'Aviation gasoline',
             'Motor gasoline (excluding biofuel portion)', 'Gasoline-type jet fuel',
             'Kerosene-type jet fuel (excluding biofuel portion)', 'Other kerosene',
             'Gas oil and diesel oil (excluding biofuel portion)', 'Fuel oil',
             'White spirit and special boiling point industrial spirits', 'Lubricants', 'Paraffin waxes',
             'Petroleum coke', 'Bitumen', 'Other oil products n.e.c.', 'Renewables and biofuels',
             'Ambient heat (heat pumps)', 'Non-renewable municipal waste', 'Industrial waste (non-renewable)',
             'Electricity', 'Heat']

mix_df.drop(columns=drop_tecs, inplace=True)

# Perform label cleaning before converting to ISO A2 country codes
mix_df.drop(index='European Union - 28 countries (2013-2020)', inplace=True)
mix_df.rename(index={'Germany (until 1990 former territory of the FRG)': 'Germany',
                     'Kosovo (under United Nations Security Council Resolution 1244/99)': 'Kosovo'}, inplace=True)

mix_df.index = pd.Index(coco.convert(mix_df.index.tolist(), to='ISO2'))

# remove Eurostat countries not covered in EEU/WEU regions in MESSAGE
drop_countries = list(set(mix_df.index.tolist()) - set(list(reg_dict.keys())))
mix_df.drop(index=drop_countries, inplace=True)

missing_countries_eurostat = list(set(EEU_ISO2 + WEU_ISO2) - set(mix_df.index.tolist()))

#%%

tec_int_df = pd.read_csv(tec_int_fp, index_col=[0], na_values='-')  # regionalized (hybridized) carbon intensity factors of generation (g COw-e/kWh)
tec_int_df.rename(columns={'other': 'Other', 'other renewable': 'Other renewable'}, inplace=True)

# Make list of ISO-A2 country codes of relevant countries
iso_a2 = europe_shapes[europe_shapes['Consumption mix intensity'].notna()].ISO_A2
iso_a2.rename('country', inplace=True)


def reformat_el_df(df):
    """ Replaces N/A values from csv files (as '-') to np.nan and sets index to countries """
    df.replace('-', np.nan, inplace=True)
    try:
        if df.columns.tolist() == df.index.tolist():
            # special case for trade matrix, where labels for both columns and indices are replaced
            df = df.reindex(index=iso_a2, columns=iso_a2)
    except ValueError:
        df = df.reindex(iso_a2)
    return df


mix_df = reformat_el_df(mix_df)
trades_df = reformat_el_df(trades_df)
tec_int_df = reformat_el_df(tec_int_df)

#%% Deal with proxy countries to fill gaps from ENTSO-E Transparency Portal

""" Need proxy regionalized factors for missing countries """
# Data from IEA, for year 2018 and expressed in TWh unless otherwise noted

CH = {'Solar': 1.884,
      'Wind': 0.121,
      'Hydro Pumped Storage': 1.554,
      'Hydro Water Reservoir': 17.20767,
      'Hydro Run-of-river and poundage': 17.687,
      'Nuclear': 25.513,
      'Waste': 2.445,
      'Biomass': 0.685,
      'Fossil Gas': 0.729,
      'Fossil Oil': 0.036
}
# NB: hydro from 2018; statistics
# according to IEA; hydro total is 37.802 TWh (CH stats 36.448 TWh)
# according to entso-e data from 2019, 55% reservoir, 9% run-of-river and 35.7% pumped
#CH = 1567 GWh pumped [2015], run-of-river 36 TWh [2015]

proxy_prod_mix = pd.DataFrame([CH], index=['CH'], columns=mix_df.columns)
# proxy_prod_mix = pd.concat([proxy_prod_mix], keys=[2020], names=['year', 'technology'], axis=1)
mix_df = mix_df.append(proxy_prod_mix, sort=True)

trades_df.loc['CH', 'DE'] = 16.524  # data from STATISTIQUE SUISSE DE L’ÉLECTRICITÉ 2018, tab 29
2018
trades_df.loc['DE', 'CH'] = 4.423
trades_df.loc['CH', 'LI'] = 0.015
trades_df.loc['LI', 'CH'] = 0.034

# TODO: need prod mixes for russia, ukraine, belarus, morocco

"""
LU = {'Wind Onshore': 0.245,
      'Hydro Pumped Storage': 1.337,
      'Fossil Gas': 0.194,
      'Waste': 0.124,
      'Solar': 0.112,
      'Biomass': 0.170}  # in GWh IEA statistics 2018 https://www.iea.org/data-and-statistics?country=LUXEMBOU&fuel=Electricity%20and%20heat&indicator=Electricity%20generation%20by%20source;
# biomass == biofuels
# LU: 1431 GWh in pumped, 0.1 TWh natural flow  [https://www.vgb.org/hydropower_fact_sheets_2018.rss?dfid=91827]

HR = {'Wind Onshore': 1.204,
      'Hydro Water Reservoir': 5.508,
      'Fossil gas': 3.090,
      'Waste': 0.124,
      'Solar PV': 0.079,
      'Biomass': 0.526,
      'Fossil Oil': 0.210,
      'Fossil Hard coal': 1.367}
# HR: 165 GWh in pumped, 6.39 TWh natural flow  [https://www.vgb.org/hydropower_fact_sheets_2018.rss?dfid=91827]

AL = {'Hydro Water Reservoir': 4.525}  # unconfirmed hydro PP type...

TR = {'Fossil Hard coal': 113.249 * (68.2 / (68.2 + 45.1)),
      'Fossil Brown coal/Lignite': 113.249 * (45.1 / (68.2 + 45.1)),
      'Fossil Oil': 0.329,
      'Fossil gas': 92.434,
      'Hydro Water Reservoir': 59.755,
      'Geothermal': 6.906,
      'Biomass': 2.636,
      'Wind': 19.882,
      'Solar': 7.477  # PV
      # waste 23
      }  # hard coal:lignite ratio from https://euracoal.eu/info/country-profiles/turkey/#:~:text=In%202018%2C%20Turkey%20imported%2038.3,and%20South%20Africa%20(4.2%25).


CY = {'Fossil Oil': 4.569,
      'Biomass': 0.052,
      'Solar': 0.172,  # PV
      'Wind': 0.211
      }  # 2017 https://www.iea.org/data-and-statistics/data-tables?country=CYPRUS&energy=Electricity&year=2017

MT = {'Fossil Oil': 0.193,
      'Fossil gas': 1.287,
      'Biomass': 0.010,
      'Solar': 0.155
      }  # 2017

IS = {'Fossil Oil': 0.002,
      'Hydro': 13.814,
      'Geothermal': 6.010,
      'Wind': 0.004
      }

XK = {'Fossil Hard coal': 5726,
      'Fossil Oil': 12,
      'Hydro Water Reservoir': 179,
      'Solar': 1
      }  # 2017

proxy_prod_mix = pd.DataFrame([LU, HR, AL, TR, CY, MT, IS, XK], index=['LU', 'HR', 'AL', 'TR', 'CY', 'MT', 'IS', 'XK'], columns=mix_df.columns)
# proxy_prod_mix = pd.concat([proxy_prod_mix], keys=[2020], names=['year', 'technology'], axis=1)
mix_df = mix_df.append(proxy_prod_mix, sort=True)
"""



# Add placeholders for missing countries
# TODO: make better proxies for LI, AD, MC
mix_df = mix_df.append(pd.DataFrame(index=['LI', 'AD', 'MC']))
# mix_df.loc['XK'] = mix_df.loc['PL'] * (5.726 + 0.179 + 0.012) / (mix_df.loc['PL'].sum()) # scale production to keep shares
mix_df.loc['LI'] = mix_df.loc['NO'] * (80 / 1e6) / mix_df.loc['NO'].sum()
mix_df.loc['AD'] = mix_df.loc['ES'] * (99 / 1e3) / mix_df.loc['ES'].sum()  #https://www.worlddata.info/europe/andorra/energy-consumption.php
mix_df.loc['MC'] = mix_df.loc['FR'] * (536 / 1e3) / mix_df.loc['FR'].sum()  #https://en.wikipedia.org/wiki/Energy_in_Monaco#:~:text=Monaco%20has%20no%20domestic%20sources,gas%20and%20fuels%20from%20France.&text=In%202018%2C%20the%20country%20used,it%20was%20used%20tertiary%20services.


# Data for proxy countries to fill gaps from ENTSO-E Transparency Portal

"""
LU: {NG: 3580.3 TJ, oil: 16.2 TJ} #https://statistiques.public.lu/
HR: {wind: 1204, hydro: 5508, natural gas: 3090, waste: 124, PV: 79, biofuels: 526, oil: 210, coal: 1367}  # in GWh, IEA statistics
AL: {hydro: 4525}  # in GWh, IEA statistics 2018
XK: {coal: 5726, hydro: 179, oil: 12}
AD: total prod: 99.48e9 kWh
MC: {depends on france? wiki}

# total consumption in ktoe,
LU: 8.2 TWh
XK: 5 TWh
AL: 6 TWh
HR: 17 TWh

# IMPORTS/EXPORTS in ktoe
XK: {imports: 107, exports: 76}
AL: {imports: 251, exports: 4} # we have these data from entso-e....
HR: {imports: 1045, exports: 447} # we have these data from entso-e ... trade with hungary and bosnia and hercegovina
LU: {imports: 649, exports: -120} #from entsoe....wrong
LU: {imports: {BE: 386, FR: 1302, DE: 5865}, exports:{BE: 147, DE: 1245}} # in GWh, from https://statistiques.public.lu/
AD: {imports: 471.3m KWh, exports: 6000 kWh} #wlrdata.info
TR: {imports: 2466, exportS: 3074} # IEA 2018, in GWh
MT: {imports: 897, exports: 36} #IEA, 2017, in GWh
XK: {imports: 1242, exports: 880} # IEA, 2017, in GWh
"""

# no longer necessary as we re-calculate intensities (rather than reusing those calculated in BEV footprints)
proxy_prod_int = {'LU': 505,   # LU/HR from Moro and Lonza
                  'HR': 465,
                  'AL': df.at['NO', 'Consumption mix intensity'],
                  'XK': df.at['PL', 'Consumption mix intensity'],
                  'LI': df.at['NO', 'Consumption mix intensity'],
                  'AD': df.at['ES', 'Consumption mix intensity'],
                  'MC': df.at['FR', 'Consumption mix intensity']}

#%%

# OBS: prod_shares not used
prod_shares = mix_df.div(mix_df.sum(axis=1), axis=0)
# col_labels = pd.MultiIndex.from_product([message_el.loc(axis=1)[2020:].columns, prod_shares.columns], names=['year', 'technology'])

# prod_shares = pd.concat([prod_shares], keys=[2020], names=['year', 'technology'], axis=1)
# prod_shares = prod_shares.reindex(columns=col_labels)

message_el.columns = message_el.columns.astype(int)
col_labels = pd.MultiIndex.from_product([message_el.loc(axis=1)[2020:].columns, prod_shares.columns], names=['year', 'technology'])

mix_df = pd.concat([mix_df], keys=[2020], names=['year', 'technology'], axis=1)
mix_df = mix_df.reindex(columns=col_labels)
mix_df = mix_df.sort_index(axis=0)


#%% Introduce proxy carbon intensity factors for missing countries

tec_int_df = tec_int_df.reindex(mix_df.index)
proxy_prod = tec_int_df.loc[tec_int_df.isnull().sum(axis=1) == len(tec_int_df.columns)].index  # get index of rows with all np.nans

# Simplification of wind technologies; assume only onshore
tec_int_df.rename(columns={'Wind Onshore': 'Wind'}, inplace=True)
tec_int_df.drop(columns='Wind Offshore', inplace=True)

mix_df.rename(columns={'Solar PV': 'Solar'}, inplace=True)  # for Eurostat data

# Assumption: "other renewable" constitutes tidal, marine tec, and renewable waste
# note waste also includes non-renewable waste (e.g., industrial waste). From Eurostat, approx 50/50 renewable:non renewable
tec_int_df['Marine'] = tec_int_df['Other renewable']
tec_int_df['Waste'] = tec_int_df[['Other', 'Other renewable']].mean(axis=1)

# use regional mean for countries missing intensity factors
for country in proxy_prod:
    tec_list = mix_df.loc[country][2020].index[mix_df.loc[country][2020].notnull()]  # get list of relevant tecs for the country, i.e., where there is production activity
    for tec in tec_list:
        # TODO: make this EEU/WEU specific
        print(f'{country}, {tec}')
        if tec in tec_int_df.columns.tolist():
            tec_int_df.loc[country][tec] = tec_int_df[tec].mean()  # use technology mean of other regions as proxy
        else:
            print(tec + ' is not in tec_int_df')


#%%
# Introduce WEU/EEU regions to multiindex in prodshares for applying MESSAGE transformation pathways

reg_mi = pd.MultiIndex.from_tuples([(reg_dict[country], country) for country in mix_df.index.tolist()], names=['reg', 'country'])
reg_mi = reg_mi.sort_values()
mix_df = mix_df.reindex(reg_mi, level=1)

#%%
# ###### Begin data harmonization (disaggregation) work

#%%
# Disaggregate ENTSO-E production to Solar PV/CSP, hydro  and biomass/fossil with/without CCS

disagg_tec_dict = {'Solar': ['Secondary Energy|Electricity|Solar|CSP', 'Secondary Energy|Electricity|Solar|PV'],
                   'Biomass': ['Secondary Energy|Electricity|Biomass|w/o CCS', 'Secondary Energy|Electricity|Biomass|w/ CCS'],
                   'Fossil Coal-derived gas': ['Secondary Energy|Electricity|Gas|w/o CCS', 'Secondary Energy|Electricity|Gas|w/ CCS'],
                   'Fossil Gas': ['Secondary Energy|Electricity|Gas|w/o CCS', 'Secondary Energy|Electricity|Gas|w/ CCS'],
                   'Fossil Hard coal': ['Secondary Energy|Electricity|Coal|w/o CCS', 'Secondary Energy|Electricity|Coal|w/ CCS'],
                   'Fossil Brown coal/Lignite': ['Secondary Energy|Electricity|Coal|w/o CCS', 'Secondary Energy|Electricity|Coal|w/ CCS'],
                   'Fossil Peat': ['Secondary Energy|Electricity|Coal|w/o CCS', 'Secondary Energy|Electricity|Coal|w/ CCS']
                   }

# new_keys currently unused
new_keys = {#'Secondary Energy|Electricity|Solar|CSP': 'Solar CSP',
            #'Secondary Energy|Electricity|Solar|PV': 'Solar PV',
            'Secondary Energy|Electricity|Biomass|w/o CCS': 'Biomass w/o CCS',
            'Secondary Energy|Electricity|Biomass|w/ CCS': 'Biomass w/ CCS',
            'Secondary Energy|Electricity|Gas|w/o CCS': 'Gas w/o CCS',
            'Secondary Energy|Electricity|Gas|w/ CCS': 'Gas w/ CCS',
            'Secondary Energy|Electricity|Coal|w/o CCS': 'Coal w/o CCS',
            'Secondary Energy|Electricity|Coal|w/ CCS': 'Coal w/ CCS'
            }

# TODO: refine these?
new_CFs = {'Solar CSP': 0.8,   # guesstimate
           'Solar PV': 1,
           'Biomass w/o CCS': 1,
           'Biomass w/ CCS': 0.3,  # guesstimate
           'Fossil Gas w/o CCS': 1,
           'Fossil Gas w/ CCS': 0.35,  # factor from IPCC AR5
           'Fossil Hard coal w/o CCS': 1,
           'Fossil Hard coal w/ CCS': 0.27,  # factor from IPCC AR5
           'Fossil Coal-derived gas w/ CCS': 1,
           'Fossil Coal-derived gas w/o CCS': .35,
           'Fossil Peat w/o CCS': 1,
           'Fossil Peat w/ CCS': 0.27,
           'Fossil Brown coal/Lignite w/ CCS': 0.27,
           'Fossil Brown coal/Lignite w/o CCS': 1
           }

match_tec_dict = {}
tec_shares = pd.DataFrame()


# for testing with MESSAGE electricity mix in 2020
def disagg_entsoe(mix_df, tec_shares):
    """ Disaggregate 2020 data to MESSAGE tec categories, i.e., solar and w/, w/o CCS fossil and update corresponding regionalized emission factors. """
    # mix_df = country_disagg_message
    for key, item in disagg_tec_dict.items():
        # calculate relative split of MESSAGE sub-technolgies in the list
        msg_tecs = message_el[message_el.index.get_level_values(1).isin(item)]
        # tec_shares = pd.DataFrame(msg_tecs[2020] / msg_tecs[2020].sum(level='reg'))
        tmp_tec_shares = (msg_tecs / msg_tecs.sum(level='reg')).droplevel('units')
        tec_shares = tec_shares.append(tmp_tec_shares)
        # tec_shares = pd.DataFrame(msg_tecs / msg_tecs.sum(level='reg'))

        # split ENTSO-E categories accordingly
        for tec in item:
            # make new ENTSO-E style technology name
            lvl_name = key + ' ' + tec.rsplit('|', 1)[1]
            mix_df[2020, lvl_name] = (mix_df[2020, key]).mul(tmp_tec_shares[2020].loc[:, tec], level='reg')
            match_tec_dict[lvl_name] = tec  # add entry for disaggregated entso tec to dictionary

            # update the impact factors for the technology using the dictionary
            if tec == 'Secondary Energy|Electricity|Biomass|w/ CCS':
                tec_int_df[lvl_name] = tec_int_df[key].mul(0.28).subtract(tec_int_df[key] * 0.83 * 2)  # value chain emissions ~28% of LC emissions
            else:
                tec_int_df[lvl_name] = tec_int_df[key].mul(new_CFs[lvl_name])
        mix_df.drop(columns=[(2020, key)], inplace=True)
        tec_int_df.drop(columns=key, inplace=True)

    tec_shares = tec_shares.drop_duplicates()
    return mix_df, tec_shares


#%%
"""--- Calculate change in technology shares according to MESSAGE ---"""
#  Make dict that maps MESSAGE technology groupings to ENTSO-E technology groupings

# Match ENTSO-E technologies to MESSAGE technologies
# NB no oil with CCS
match_tec_dict.update({'Fossil Oil': 'Secondary Energy|Electricity|Oil|w/o CCS',
                       'Fossil Oil shale': 'Secondary Energy|Electricity|Oil|w/o CCS',
                       'Geothermal': 'Secondary Energy|Electricity|Geothermal',
                       'Hydro Pumped Storage': 'Secondary Energy|Electricity|Hydro',
                       'Hydro Run-of-river and poundage': 'Secondary Energy|Electricity|Hydro',
                       'Hydro Water Reservoir': 'Secondary Energy|Electricity|Hydro',
                       'Marine': 'Secondary Energy|Electricity|Other',
                       'Nuclear': 'Secondary Energy|Electricity|Nuclear',
                       'Other': 'Secondary Energy|Electricity|Other',
                       'Other renewable': 'Secondary Energy|Electricity|Other',
                       'Waste': 'Secondary Energy|Electricity|Other',
                       'Wind': 'Secondary Energy|Electricity|Wind'
                       #'Wind Offshore': 'Secondary Energy|Electricity|Wind|Offshore',
                       #'Wind Onshore': 'Secondary Energy|Electricity|Wind|Onshore'
                       })

# Disaggregate MESSAGE to country level using shares from ENTSO (by region and technology)
mix_df, tec_shares = disagg_entsoe(mix_df, tec_shares)


#%%
""" Fill NAs in regionalized emission factors matrix with mean value of technology """

tec_int_df = tec_int_df.reindex(reg_mi, level='country')

for reg in tec_int_df.index.get_level_values('reg'):
    tec_int_df.loc[reg].fillna(tec_int_df.loc[reg].mean(), inplace=True)
tec_int_df.index = tec_int_df.index.droplevel('reg')

#%%
# Map MESSAGE technology classifications to technologies from ENTSO-E
mix_df = mix_df.stack('technology')
tec_mapping = pd.DataFrame.from_dict(match_tec_dict, orient='index', columns=['MESSAGE tec'])
tec_mapping.index.rename('technology', inplace=True)

# Add MESSAGE technology mappings to ENTSO-E/empirical data
mix_df = mix_df.join(tec_mapping, on='technology')
mix_df.set_index(['MESSAGE tec'], append=True, inplace=True)


entso = mix_df.copy()
message = message_el.loc(axis=1)[2020:].copy()
message = message * 277.78  # convert from EJ/year to TWh/year
message = message.droplevel('units')

# calculate equalization factor to address discrepancy in 2020 production amounts between MESSAGE and ENTSO-E
# TODO: thought experiment - should equalization be regional total values, or, region-technology levels?
""" First, match relative shares of each technology in each region for 2020 to ENTSO E"""
""" For post-2020, use year-to-year relationship within MESSAGE"""
equalization = mix_df[2020].sum(level='reg') / message[2020].sum(level='reg')
# ents_sum = entso.loc['EEU'].sum(level='technology')

#%% Make test with only MESSAGE mixes (instead of using ENTSO)


# """ WHAT IS REG_TEC_SHARES SUPPOSED TO BE? """
# reg_tec_shares = entso[2020].div(entso[2020].groupby(['reg', 'technology']).sum())


# """ OBBBBBS"""
# " check the below for whichever is better."

# # reg_tec_shares = entso[2020].div(entso[2020].groupby(['reg', 'country', 'technology']).sum())
# reg_tec_shares = reg_tec_shares.div(reg_tec_shares.sum(level=['reg', 'MESSAGE tec']))  # normalize the MESSAGE categories that are aggregated

# """END OBS"""

# reg_tec_shares = reg_tec_shares.reorder_levels(['reg', 'country', 'technology', 'MESSAGE tec'])

# # Manually add geothermal production in EEU (only when using ENTSO data)
# """geo_mi = pd.MultiIndex.from_product([['EEU'], ['BG', 'HR', 'CZ', 'HU', 'PL', 'RO', 'SK'], ['Geothermal'], ['Secondary Energy|Electricity|Geothermal']]) #countries from https://www.stjornarradid.is/media/atvinnuvegaraduneyti-media/media/frettir/080119_geothermal_europe_memo_for_ossur.pdf
# geothermal_df = pd.DataFrame([1 / len(geo_mi)] * len(geo_mi), index=geo_mi, columns=[2020])
# reg_tec_shares = pd.DataFrame(reg_tec_shares).append(geothermal_df)"""

# # Manually insert offshore wind, which MESSAGE has in EEU, but ENTSO-E does not (???)
# """ The below becomes obsolete when the MESSAGE onshore/offshore wind categories are aggregated """
# """offshore_wind = reg_tec_shares[reg_tec_shares.index.get_level_values('technology').str.contains('Wind')]
# offshore_wind = offshore_wind.drop(index='WEU')  #.to_frame()
# offshore_wind.reset_index(inplace=True)
# offshore_wind['technology'] = offshore_wind['technology'].str.replace('Onshore', 'Offshore')
# offshore_wind['MESSAGE tec'] = offshore_wind['MESSAGE tec'].str.replace('Onshore', 'Offshore')
# offshore_wind.set_index(['reg', 'country', 'technology', 'MESSAGE tec'], inplace=True)"""

# # reg_tec_shares = pd.DataFrame(reg_tec_shares).append(offshore_wind)
# """reg_tec_shares = reg_tec_shares.append(offshore_wind)"""

# """---- NEED TO FIX THE OFFSHORE vs ONSHORE ratios -----"""
# # sub_tec_shares = tec_mapping.join(pd.DataFrame([1]*len(tec_mapping), index=tec_mapping.index, columns=[2020])).set_index('MESSAGE tec', append=True)
# # sub_tec_shares = sub_tec_shares.div(sub_tec_shares.sum(level=['MESSAGE tec']))
# # sub_tec_shares = sub_tec_shares[~sub_tec_shares.index.get_level_values(1).duplicated()].droplevel('technology')
# # disagg_message = sub_tec_shares.mul(reg_tec_shares, axis=0) * (message)

# disagg_message = (message[2020] * reg_tec_shares).reorder_levels(['reg', 'country', 'technology', 'MESSAGE tec'])
# check_disagg = disagg_message.sum(level=['reg', 'MESSAGE tec'])  # check sums add up to what is reported by MESSAGE

# """add_ind = disagg_message.index.difference(entso.index, sort=False)
# entso = entso.append(pd.DataFrame(index=add_ind, columns=entso.columns))
# entso[2020] = disagg_message
# entso = entso.sort_index(level=['reg', 'country', 'technology'])"""

#%%

# disagg_entsoe(country_disagg_message, tec_shares)
# disagg_entsoe(mix_df, tec_shares)

#%% Disaggregate MESSAGE to country level

# prep mapping dataframe for joining
tec_mi = tec_mapping.reset_index().set_index('MESSAGE tec')
"""
message['MSG tec'] = message.index.get_level_values('MESSAGE tec')
message = message.join(tec_mi, on='MSG tec', rsuffix='_other')
message.set_index('technology', append=True, inplace=True)
message.drop(columns='MSG tec', inplace=True)
message.drop(index=('WEU', 'Secondary Energy|Electricity|Oil|w/o CCS', 'Fossil Oil shale'), inplace=True)
message.index = message.index.swaplevel(-1, -2)
message.columns = message.columns.astype(int)

# mi = message.index.difference(entso.droplevel('country').reorder_levels(['reg', 'technology','MESSAGE tec']).index, sort=False)
# message = message.drop(index=mi)

message = message.mul(message.div(message.sum(level=['reg', 'MESSAGE tec'])))
"""
#%% Disaggregate MESSAGE to country level according to ENTSO


# calculate average change in total mix for use in 'Other' category (0 for all periods, regions in MESSAGE )
msg_sum = message.sum(level='reg')
avg_chg = 1 + ((msg_sum - msg_sum.shift(1, axis=1)) / msg_sum.shift(1, axis=1))
message.other = avg_chg

# calculate shares of each sub-technology for disaggregating MESSAGE
# categories
x = entso[2020].unstack('MESSAGE tec').sum(level=['reg', 'technology'])  # entso[2020].unstack('MESSAGE tec').sum('country')
entso_shares = (x.div(x.sum(level='reg'))).stack()
entso_shares.dropna(axis=0, inplace=True)
tmp = entso_shares.filter(like='w/o CCS', axis=0).reset_index()
lvl = entso_shares.index.names[1:]

for ind_lvl in lvl:
    tmp[ind_lvl] = tmp[ind_lvl].str.replace('w/o CCS','w/ CCS')
tmp.set_index(['reg','technology','MESSAGE tec'], inplace=True)
entso_shares = entso_shares.append(tmp.iloc(axis=1)[0])  #entso_shares = entso_shares.append(tmp).iloc(axis=1)[0]

#entso_shares.replace(1, np.nan, inplace=True)

# scale MESSAGE 2020 to equal entso/Eurostat production, technology wise
scale = pd.DataFrame(index=message.index, columns=message.columns)
scale[2020] = message[2020] * (entso[2020].sum(level=['reg', 'MESSAGE tec']) / message[2020])

# scale.iloc(axis=1)[1:] = scale.shift(1, axis=1) * message / message.shift(1, axis=1)  # for whatever reason, shift function doesn't work on scale.
for i, year in enumerate(scale.columns[1:]):
    scale[year] = (message / message.shift(1, axis=1))[year] * scale.iloc(axis=1)[i].rename(year)


# calculate ratio of CCS to non-CCS for thermal technologies (relative to previous time period)
filter_terms = ['Biomass', 'Coal', 'Gas']  # technologies with CCS
for term in filter_terms:
    tmp = message.filter(like=term, axis=0)
    tmp = tmp / tmp.sum(level='reg', axis=0).shift(1, axis=1) # share of CCS and non CCS

    scale.update(tmp.iloc(axis=1)[1:])
    for i, year in enumerate(scale.columns[1:]):
        scale[year].loc[tmp.index] = ((scale[year] * scale.iloc(axis=1)[i].filter(like=term, axis=0).sum(level='reg')))#, overwrite=True) #.shift(1, axis=1))


other_ind = (scale.filter(like='Other', axis=0)).index
scale.loc[other_ind, 2020] = entso.sum(level=['reg', 'MESSAGE tec']).filter(like='Other', axis=0)[2020]
scale.update(avg_chg.reindex(scale.loc[other_ind].index, level='reg'))
for i, year in enumerate(scale.columns[1:]):
    scale.loc[other_ind, year] = scale.loc[other_ind, year] * scale.loc[other_ind].iloc(axis=1)[i]

message = scale

# old stuff below
# remove thermal (w/ and w/o CCS) and solar technologies for now as they need to be split; re-append after scaling
# below no longer necessary if using filter...
"""drop_labels = [label for label in message.index.unique('MESSAGE tec') if label in list(new_keys.keys())]
# ccs_labels = [label for label in message.index.get_level_values('MESSAGE tec') if 'w/ CCS' in label]
tmp_message = message.loc[(slice(None), drop_labels), :]  # temporary "holding" dataframe
message = message.drop(index=drop_labels, level='MESSAGE tec')
"""

# now disaggregate across countries, ENTSO/Eurostat technologies
# need to change calculations to adjust for technology aggregation in message
# use tec shares: df.groupby('MESSAGE tec')
# group.div(group.sum(axis=0, level='ENTSOE tec'))
message['MSG tec'] = message.index.get_level_values('MESSAGE tec')
message = message.join(tec_mi, on='MSG tec', rsuffix='_other')
message.set_index('technology', append=True, inplace=True)
message.drop(columns='MSG tec', inplace=True)
message.index = message.index.swaplevel(-1, -2)
# message = (message.replace(0, np.nan)).dropna(how='all', axis=0)

# remove indices from disaggregated MESSAGE that are not physically present (as per ENTSO)
mi = message.index.difference(entso.droplevel('country').reorder_levels(['reg', 'technology', 'MESSAGE tec']).index, sort=False)
message = message.drop(index=mi)
# mi = message.droplevel('units').index.difference(entso.droplevel('country').reorder_levels(['reg','MESSAGE tec', 'technology']).index, sort=False)
# message = message.droplevel('units').drop(index=mi)

tec_shares = tec_shares.join(tec_mi, on='MESSAGE tec')
tec_shares.set_index('technology', append=True, inplace=True)
tec_shares = tec_shares.reorder_levels(['reg', 'technology', 'MESSAGE tec'])

tmp_tec_share = tec_shares.mul(tec_shares.div(tec_shares.sum(level=['reg', 'MESSAGE tec'])))

# calculate disaggregation shares for MESSAGE categories with further resolution in ENTSO (even split)
"""msg_tec_share = message.div(message.sum(level=['reg', 'MESSAGE tec']))
msg_tec_share = msg_tec_share.reorder_levels(['reg', 'technology', 'MESSAGE tec'])"""


message = message.mul(entso_shares.replace(0, np.nan).dropna(axis=0), axis=0)

"""for year in message.columns:
    # sub_tec_shares.columns = [year]
    message[year] = message[year].mul(entso_shares.rename(year))  # disaggregate to sub-technologies in ENTSO/Eurostat
    # tmp_message[year] = tmp_message[year].mul(tec_shares[year]).mul(sub_tec_shares[year])  # use MESSAGE-based future shares for technologies not in ENTSO dataset
    # tmp_message[year] = tmp_message.reset_index('units')[year].mul(tec_shares[year])  # use MESSAGE-based future shares for technologies not in ENTSO dataset
    # tmp_message[year] = tmp_message[year].mul(tmp_tec_share[year])  # use MESSAGE-based future shares for technologies not in ENTSO dataset
"""
# tmp_message = tmp_message.droplevel('units')
"""tmp_message = tmp_message.join(tec_mi, on='MESSAGE tec')
tmp_message.set_index('technology', append=True, inplace=True)
tmp_message = tmp_message.reorder_levels(['reg', 'technology', 'MESSAGE tec'])
# tmp_message = tmp_message.mul(msg_tec_share)
tmp_message = tmp_message.mul(tmp_tec_share)
message = message.append(tmp_message)"""

#%%

message = message[(message.T != 0).any()]  # remove country-technology combos that have 0 production through study period
message.dropna(axis=0, how='all', inplace=True)  # remove irrelevant country-technology combos
message.replace(np.nan, 0, inplace=True)

# message = message * sub_tec_shares

# Scale MESSAGE production to match total regional production (scaled_message not used)
delta = pd.DataFrame()
scaled_message = message.copy()
scaled_message = scaled_message * (message / entso.sum(level=['reg']))

for i, year in enumerate(message.columns):
    if i > 0:
        delta[year] = message[year] - message.iloc(axis=1)[i-1]

# delta.index = delta.index.swaplevel(1, 2)


#%% Begin setup of simplifed test system for checking implementation of calculations
# (see matching electricity_thought ""experiment.xlsx)
"""
reg_mi = pd.MultiIndex.from_product([['1'],['A','B','C']], names=['reg','country'])
test_entso_mi = pd.MultiIndex.from_product([['1'], ['A','B','C'], ['Coal','Nuclear','Gas','Wind']], names=['reg','country','technology'])
msg_mi = pd.MultiIndex.from_product([['1'],['Coal', 'Nuclear', 'Gas', 'Wind']], names=['reg', 'technology'])#, 'units'])
# msg_mi = pd.DataFrame(['EJ/yr']*4, index=new_mixes.iloc[0:4].index.droplevel('country'), columns=['units']).set_index('units', append=True).index
entso = pd.DataFrame([9,8,8.5,0,1,1,6.5,1,0,0,0,1], index=test_entso_mi, columns=[2020])
entso = entso.join(pd.DataFrame(['MCoal','Mnuclear','mgas','mwind'], index=test_entso_mi.unique(2),columns=['MESSAGE tec']),).set_index('MESSAGE tec', append=True)

message = pd.DataFrame([[10,5],[9,5],[15,22],[2,8]], index=msg_mi, columns=[2020,2030])# index=pd.MultiIndex.from_product([[1],['MCoal','Mnuclear','mgas','mwind']                                                                                    ], names=['reg','MESSAGE tec']), columns=[2020,2030])
message = message.join(pd.DataFrame(['MCoal', 'Mnuclear', 'mgas', 'mwind'], index=msg_mi, columns=['MESSAGE tec'])).set_index('MESSAGE tec', append=True)
message = message.reorder_levels(['reg', 'technology', 'MESSAGE tec'])#, 'units'])
delta = pd.DataFrame()

for i, year in enumerate(message.columns):
    if i>0:
        delta[year] = message[year] - message.iloc(axis=1)[i-1]

#### END TEST code

"""

#%% Calculations for future production mixes

"""
DO THIS FIRST, THEN SUBDIVIDE ACCORDING TO MESSAGE?
"""
# Make multiindex with all country-technology combinations (irrelevant combinations removed later)
complete_mi = []
for country in reg_mi.to_list():
    for tec in tec_mi.set_index('technology', append=True).index.to_list():
        complete_mi.append(country + tec)

complete_mi = pd.MultiIndex.from_tuples(complete_mi, names=['reg', 'country', 'MESSAGE tec', 'technology'])
complete_mi = complete_mi.reorder_levels(['reg', 'country', 'technology', 'MESSAGE tec'])

new_mixes = pd.DataFrame(index=complete_mi, columns=message.columns)  # dataframe to contain calcualted country mixes
new_mixes.update(entso)

reduced_prod = pd.DataFrame()

freed_cap = pd.DataFrame(index=reg_mi, columns=message.columns)
shares_tec = pd.DataFrame()
# incr_tecs = pd.DataFrame(index=new_mixes.index)
null_tecs_df = pd.DataFrame(index=new_mixes.index, columns=message.columns)
neg_tecs = pd.DataFrame(index=new_mixes.index, columns=message.columns)
ratio = pd.DataFrame(index=entso.index, columns=message.columns)

# prepare dataframes to contain interim calculations for quality checks and debugging
new_mixes_check = pd.DataFrame(index=pd.MultiIndex.from_product([['EEU', 'WEU'], new_mixes.index.unique('technology')], names=['reg', 'technology']), columns=message.columns)
incr_prod_df = pd.DataFrame(index=new_mixes.index, columns=message.columns)
red_prod_df = pd.DataFrame(index=new_mixes.index, columns=message.columns)
shares_tec_df = pd.DataFrame(index=message.index, columns=message.columns)
repl_df = pd.DataFrame(index=new_mixes.index, columns=message.columns)
repl_df2 = pd.DataFrame(index=new_mixes.index, columns=message.columns)
freed_cap_df = pd.DataFrame(index=reg_mi, columns=message.columns)
decr_prod = pd.DataFrame(index=new_mixes.index, columns=message.columns)
add_prod_df = pd.DataFrame(index=new_mixes.index, columns=message.columns)

for i, year in enumerate(new_mixes.iloc(axis=1)[1:].columns, 1):
    neg_delta = delta[delta[year] < 0][year]
    pos_delta = delta[delta[year] > 0][year]
    prev_year = new_mixes.iloc(axis=1)[i-1].name
    repl_prod = pd.DataFrame(index=new_mixes.index)
    repl_prod2 = pd.DataFrame(index=new_mixes.index)

    # Keep values from previous year for technologies with same production as previous period (delta=0)
    null_tec_ind = delta[delta[year] == 0].index
    if not null_tec_ind.empty:
        null_tecs = ((new_mixes[prev_year].copy()).reset_index('country')).loc[null_tec_ind].set_index('country', append=True)
        null_tecs.rename(columns={prev_year: year}, inplace=True)
        null_tecs = null_tecs.reorder_levels(['reg', 'country', 'technology', 'MESSAGE tec'])
        new_mixes.update(null_tecs)
        null_tecs_df.update(null_tecs)

    # Calculate new values for technologies with decreasing production
    # calculate the country distribution for each technology (and region)
    ratio = new_mixes[prev_year].div(new_mixes[prev_year].sum(level=['reg', 'technology']).replace(0, np.nan))  # calculate tec shares by region
    ratio = ratio.rename(year).reset_index('country')
    # multiply the country-technology shares by the total amount each technology decreases by region
    # decr_prod[year] = decr_prod.mul(neg_delta)
    ratio[year] = ratio[year].mul(neg_delta)
    ratio = ratio.set_index('country', append=True).reorder_levels(['reg', 'country', 'technology', 'MESSAGE tec'], axis=0)
    decr_prod[year] = ratio
    new_mixes.update(ratio.add(new_mixes[prev_year].rename(year), axis=0), overwrite=False)
    neg_tecs.update(ratio.add(new_mixes[prev_year].rename(year), axis=0), overwrite=False)

    # Calculate new values for technologies with increasing production
    # First, calculate how much 'extra' production needs to be supplied from the
    # phasing out of technologies (i.e., delta < 0)

    # retrieve indices for technologies decreeasing in production
    tmp = new_mixes.reset_index(['country', 'MESSAGE tec']).join(delta, how='outer', lsuffix='_mix', sort=False)  # broadcast delta to country-level resolution
    tmp = tmp.set_index(['country'], append=True).reorder_levels(['reg', 'country', 'technology', 'MESSAGE tec'])  # fix index to match other DFs
    tmp = tmp.filter(regex='[0-9]{4}$', axis=1)  # remove the [year] column inherited from new_mixes
    tmp.columns = tmp.columns.astype(int)
    neg_delt_ind = tmp[tmp[year] < 0].index  # find the indices for technologies that go down in production during the decade

    # find the reduction in country-level production for each technology in the period
    reduced_prod = new_mixes[prev_year].loc[neg_delt_ind].sum(level=['reg', 'country', 'technology']) - new_mixes[year].loc[neg_delt_ind].sum(level=['reg', 'country', 'technology'])
    reduced_prod.rename(year, inplace=True)
    red_prod_df[year] = reduced_prod

    sum_neg_delta = reduced_prod.sum(level= 'reg')  # total amount of production decrease by region
    shares_tec = sum_neg_delta * (pos_delta.div(pos_delta.sum(level='reg')))  # mix of technologies with increased production for 'making up' for reduced production
    shares_tec_df[year] = shares_tec

    # add region mappings to freed_cap for operations with shares_tec
    freed_cap = reduced_prod.sum(level=['reg', 'country'])
    freed_cap_df[year] = freed_cap

    # calculate mix of production displacing retired technologies
    repl_prod = (freed_cap.div(freed_cap.sum(level='reg')).mul(shares_tec))

    # calculate mix of production for additional ("surplus") production
    tec_diff = (pos_delta - shares_tec)
    reg_share = (new_mixes[prev_year].sum(level=['reg', 'country']).div(new_mixes[prev_year].sum(level='reg'))).rename(year)

    add_prod = tec_diff.mul(reg_share).reorder_levels(['reg', 'country', 'technology', 'MESSAGE tec']).sort_index(level=['reg', 'country'])

    # total production of technologies that increase in the time period; sum of previous year's production, 'compensation' for reduced production and growth in production
    incr_prod = new_mixes[prev_year].rename(year).replace(np.nan, 0) + repl_prod + add_prod  # may be problematic drops country-tec combos not originally in new_mixes
    repl_df[year] = repl_prod.reorder_levels(['reg', 'country', 'technology', 'MESSAGE tec'])

    add_prod_df[year] = add_prod.sort_index(level=['reg', 'country'])  # growth in production

    incr_prod.dropna(how='all', axis=0, inplace=True)
    new_mixes.update(incr_prod, overwrite=False)

    new_mixes_check[year] = new_mixes[year].sum(level=['reg', 'technology'])  # for checking against total production in MESSAGE

new_mixes = new_mixes.dropna(axis=0, how='all')  # drop 'unused' country-technology combinations

#%%
prod_df = new_mixes.copy()
# mix_df = mix_df.droplevel(['reg','MESSAGE tec']).unstack('technology')

#%% Populate future production mixes using MESSAGE transformation pathways (factors normalized by 2020 production)

# prod_df = pd.DataFrame(index=mix_df.index, columns=mix_df.columns)
# for ind, row in mix_df.iterrows():
#     reg = ind[0]
#     msg_tec = ind[3]
#     prod_df.loc[ind] = (message_el_shares.loc[(reg, msg_tec), 2020:]) * (mix_df.loc[ind][2020])

#%% Adjust trade matrix to match growth in electricity production

""" Calculate new intensities through time [using constant LCA factors for each technology] --> trading stays the same? """

ann_growth = prod_df.groupby(['reg']).sum()
ann_growth_norm = ann_growth.div(ann_growth[2020], axis=0)  # increase in electricity production normalized to 2020
msg_ann_growth_norm = message_el.groupby(['reg']).sum().div(message_el.groupby(['reg']).sum()[2020], axis=0)

""" Temporary approximation: use overall growth across region to calculate increase in trade flows"""
""" todo: find way to divide to regions?"""
ann_growth_norm_temp = prod_df.sum().div(prod_df.sum()[2020])

trades_df = trades_df.reindex(reg_mi, level=1)
trades_df = trades_df.reindex(reg_mi, level=1, axis=1)
trades_df = pd.concat([trades_df], keys=[2020], names=['year', 'reg', 'country'], axis=1)

# Add the remaining years in the study period
df = pd.DataFrame(columns=message_el.loc(axis=1)[2020:].columns, index=reg_mi)
df.rename_axis(columns='year', inplace=True)
trades_mi = df.stack(dropna=False).index.reorder_levels(['year', 'reg', 'country'])  # [2, 0, 1])
trades_mi.rename(names='year', level=0, inplace=True)
trades_mi = trades_mi.sortlevel('year')[0]
trades_df = trades_df.reindex(columns=trades_mi)

for year in ann_growth_norm_temp.index.tolist():
    trades_df[year] = trades_df[2020] * ann_growth_norm_temp[year]

imports = trades_df.sum(axis=0).unstack('year')
exports = trades_df.groupby(['year'], axis=1).sum()

imports = imports.reindex(reg_mi, level=1)  # reindex to include MESSAGE region mappings
exports = exports.reindex(reg_mi, level=1)  # reindex to include MESSAGE region mappings
cons_df = (prod_df.sum(level='country')).add(imports).subtract(exports)

#%% Calculate new consumption mix intensities


def calculate_impact_factors(production, consumption, trades, import_el, export_el):

    g = production.sum(level='country')  # Vector of total electricity production (by hour)
    g = g.reindex(reg_mi, level=1)

    q = g + import_el  # vector of total consumption
    q.replace(np.nan, 0, inplace=True)

    y = consumption  # final demand (consumption) of electricity

    y.replace(np.nan, 0, inplace=True)
    trades.replace(np.nan, 0, inplace=True)
    # ### Generate regionalized tech generation matrix [tec x countries]
    C = tec_int_df
    C = C.reindex(reg_mi, level=1)
    C = C.reindex(tec_mapping.set_index('MESSAGE tec', append=True).index, level=0, axis=1)

    # # Calculate technology characterization factors including transmission and distribution losses

    # # Start electricity calculations
    # ### Calculate production and consumption mixes
    i = consumption.size  # Number of European regions

    # ### Make Leontief production functions (normalize columns of A)
    # normalized trade matrix quadrant
    Atmx = pd.DataFrame(np.matmul(trades, np.linalg.pinv(np.diag(q))))

    # "Trade" Leontief inverse
    # Total imports from region i to j per unit demand on j
    ltmx = np.linalg.pinv(np.identity(i) - Atmx)
    Ltmx = pd.DataFrame(np.linalg.pinv(np.identity(i) - Atmx), trades.columns, trades.index)

    # normalized production matrix quadrant
    Agen = pd.DataFrame(np.diag(g) * np.linalg.pinv(np.diag(q)), index=g.index, columns=g.index)  # coefficient matrix, generation

    # Ltmx = Ltmx.reindex(reg_mi, level=1)
    # Agen = Agen.reindex(reg_mi, level=1)

    # Production in country i for trade to country j
    # Total generation in i (rows) per unit demand j
    Lgen = pd.DataFrame(np.matmul(Agen, Ltmx), index=Agen.index, columns=Ltmx.columns)

    y_diag = pd.DataFrame(np.diag(y), index=g.index, columns=g.index)

    # total imports for given demand (in TWh/h)
    Xtmx = pd.DataFrame(np.matmul(np.linalg.pinv(np.identity(i) - Atmx), y_diag))

    # Total generation to satisfy demand (consumption) (in TWh/h)
    Xgen = np.matmul(np.matmul(Agen, Ltmx), y_diag)
    Xgen.sum(axis=0)
    Xgen_df = pd.DataFrame(Xgen, index=Agen.index, columns=y_diag.columns)

    # ### Check electricity generated matches demand
    totgen = Xgen.sum(axis=1)
    r_gendem = totgen / g  #y  # All countries should be 1

    totcons = Xgen.sum(axis=0)
    r_condem = totcons / y  # All countries should be 1

    prod_by_tec = production / g
    prod_by_tec = prod_by_tec.unstack(['technology', 'MESSAGE tec'])
    # prod_by_tec = prod_by_tec.stack()
    # prod_by_tec.index = prod_by_tec.index.swaplevel(0,1)
    # prod_by_tec.sort_index(inplace=True)

    # ### Generation technology matrix

    # TC is a country-by-generation technology matrix - normalized to share of total domestic generation, i.e., normalized generation/production mix
    # technology generation, kWh/ kWh domestic generated electricity
    prod = production.unstack(['technology', 'MESSAGE tec'])
    prod = prod.T.reset_index(level='MESSAGE tec').T  # set MESSAGE classifications for columns so we can do the sort
    prod.sort_index(axis=1, inplace=True)
    prod = ((prod.T).set_index(['MESSAGE tec'], append=True)).T
    prod.fillna(0, inplace=True)

    # TC = pd.DataFrame(np.matmul(np.linalg.pinv(np.diag(g)), production.unstack(['technology', 'MESSAGE tec'])))#, index=g.index)#, columns=production.unstack(['technology', 'MESSAGE tec']).columns)
    TC = pd.DataFrame(np.matmul(np.linalg.pinv(np.diag(g)), prod))  # , index=g.index)#, columns=production.unstack(['technology', 'MESSAGE tec']).columns)
    TCsum = TC.sum(axis=1)  # Quality assurance - each country should sum to 1

    # # Calculate technology generation mix in GWh based on production in each region
    TGP = pd.DataFrame(np.matmul(TC.transpose(), np.diag(g)))  # , index=TC.columns, columns=g.index)  # .== production

    # # Carbon intensity of production mix
    CFPI_no_TD = pd.DataFrame(prod.multiply(C).sum(axis=1) / prod.sum(axis=1), columns=['Production mix intensity'])  # production mix intensity without losses
    CFPI_no_TD.fillna(0, inplace=True)

    # # Carbon intensity of consumption mix
    CFCI_no_TD = pd.DataFrame(np.matmul(CFPI_no_TD.T, Lgen).T)
    CFCI_no_TD.columns = ['Consumption mix intensity']
    CFCI_no_TD.index = CFPI_no_TD.index

    # # Transpose added after removing country aggregation as data pre-treatment
    # if include_TD_losses:
    #     CFPI_TD_losses = CFPI_no_TD.multiply(TD_losses, axis=0).dropna(how='any', axis=0)  # apply transmission and distribution losses to production mix intensity
    #     CFCI_TD_losses = CFCI_no_TD.multiply(TD_losses, axis=0).dropna(how='any', axis=0)

    #     # CF = CFPI_TD_losses.join(CFCI_TD_losses)
    #     CF_prod = CFPI_TD_losses
    #     CF_cons = CFCI_TD_losses

    # else:
    #     CF_prod = CFPI_no_TD
    #     CF_cons = CFCI_no_TD

    CF_prod = CFPI_no_TD
    CF_cons = CFCI_no_TD
    return Xgen_df, prod_by_tec, CF_prod, CF_cons


#%%
Xgen_df = pd.DataFrame(index=reg_mi, columns=trades_mi)
Xgen_tecs = pd.DataFrame(index=reg_mi, columns=prod_df.stack().index)

carbon_footprints_prod = pd.DataFrame(index=reg_mi, columns=message_el_shares.loc(axis=1)[2020:].columns)
carbon_footprints_cons = pd.DataFrame(index=reg_mi, columns=message_el_shares.loc(axis=1)[2020:].columns)

#%%
for year in cons_df.columns:
    ann_gen, prod_by_tec, cf_prod, cf_cons = calculate_impact_factors(prod_df[year], cons_df[year], trades_df[year], imports[year], exports[year])

    # update the dataframes with calculations from this hour
    Xgen_df[year].update(ann_gen)  # [country x (yearxcountry)]

    # temp_df = ann_gen.reindex(index=prod_by_tec.index, level=[0,1])
    # for country, col in temp_df.iteritems():
    #     temp_df[country] = col.multiply(prod_by_tec)
    # Xgen_tecs.loc(axis=1)[year].update(temp_df)  # [(hourxcountryxtec) x country]
    carbon_footprints_prod[year] = cf_prod
    carbon_footprints_cons[year] = cf_cons

#%%europe_spaes.
prod_df = pd.concat([prod_df], keys=['Total production'], axis=1)

carbon_footprints_cons = pd.concat([carbon_footprints_cons], keys=['Consumption mix intensity'], axis=1)
carbon_footprints_cons['Consumption mix intensity'] = np.where(carbon_footprints_cons['Consumption mix intensity'] < 1e-2,
                                                               np.nan, carbon_footprints_cons['Consumption mix intensity'])
"""temporary - removal of countries with no consumption mix"""
# carbon_footprints_cons.dropna(axis=0, how='any', inplace=True)

carbon_footprints_cons_tmp = carbon_footprints_cons['Consumption mix intensity', 2020].droplevel('reg')
carbon_footprints_cons_tmp.rename('Consumption mix intensity', inplace=True)

# add consumption mix columns to europe_shapes
europe_shapes.drop(columns=['Consumption mix intensity'], inplace=True)  # drop precalculated CF factors in favour of those just calculated which are more complete
# europe_shapes = europe_shapes.join(carbon_footprints_cons_tmp, on='ISO_A2')

#%%
# Perform clustering and add cluster column
def determine_clusters(num_clusters, df):
    tmp_df = df.copy()
    if isinstance(tmp_df, pd.Series):
        tmp_df = pd.DataFrame(tmp_df)
    tmp_df['Cluster'] = np.nan

    thresholds = jenkspy.jenks_breaks(tmp_df['Consumption mix intensity'], nb_class=num_clusters)
    thresholds[0] = thresholds[0] * 0.99

    for i in np.arange(num_clusters):
        tmp_df_bin = tmp_df[(tmp_df['Consumption mix intensity'] > thresholds[i]) &
                               (tmp_df['Consumption mix intensity'] <= thresholds[i + 1])]
        tmp_df['Cluster'][tmp_df_bin.index] = i + 1

    return tmp_df

def clean_clusters(num_clusters, df):
    clustered_df = determine_clusters(num_clusters, df)
    cluster_pop = clustered_df.value_counts(subset='Cluster')
    # if there are any single- or two-country clusters, recalculate the cluster
    # thresholds with n+1 clusters, and merge with the closest
    if (cluster_pop < 2).any():
        cl = cluster_pop.loc[cluster_pop < 2].index
        for cluster in cl:
            df2 = determine_clusters(num_clusters + 1, clustered_df)
            if cluster == (clustered_df['Cluster'].max()):
                # if singleton cluster is at max, combine with next lowest group
                df2.loc[df2['Cluster'] == (cluster + 1), 'Cluster'] = num_clusters
            elif cluster == clustered_df['Cluster'].min():
                # if singleton cluster is at min, combine with next highest group
                # and adjust cluster numbering
                df2['Cluster'] -= 1
                df2.loc[clustered_df['Cluster'] == 0,'Cluster'] = 1
            else:
                # re-run clustering with neighbouring bins to find best position
                pass
                # tmp_df = determine_clusters(2, df2[df2['Cluster'].between(cluster-1, cluster+1, inclusive=True)])
                # df2.loc[~df2['Consumption mix intensity'].isin(tmp_df['Consumption mix intensity'])] -= 1
                # df2.update(tmp_df)
                # # temp_thresholds = jenkspy.jenks_breaks(df2[df2['Cluster'].between(cluster-1, cluster+1, inclusive=True)]['Consumption mix intensity'], nb_class=2)

                # for i in np.arange(3):
                #     df_bin = df2[(df2['Consumption mix intensity'] > temp_thresholds[i]) & (df2['Consumption mix intensity'] <= thresholds[i+1])]
                #     df['Cluster'][df_bin.index]
        return df2
    else:
        return clustered_df


num_clusters = 5
test_df = clean_clusters(num_clusters, carbon_footprints_cons_tmp)
europe_shapes = europe_shapes.join(test_df, on='ISO_A2')
thresholds = [test_df[test_df['Cluster']==i+1]['Consumption mix intensity'].max() for i in range(int(test_df['Cluster'].max()))]
thresholds.insert(0, 0) # prepend lower boundary for cluster thresholds

# test code for clustering
# test_clust = pd.DataFrame([1,2,3,100,1000,1001,1002], columns=['Consumption mix intensity'])
# test_df = clean_clusters(3, test_clust)


# thresholds = jenkspy.jenks_breaks(europe_shapes['Consumption mix intensity'], nb_class=num_clusters)
# thresholds = jenkspy.jenks_breaks(carbon_footprints_cons_tmp.values, nb_class=num_clusters)
# print(thresholds[0])
# thresholds[0] = thresholds[0] * 0.99

# Add column to dataframe with cluster values
# europe_shapes['Cluster'] = np.nan

# for i in np.arange(num_clusters):
#     df_bin = europe_shapes[(europe_shapes['Consumption mix intensity'] > thresholds[i]) &
#                            (europe_shapes['Consumption mix intensity'] <= thresholds[i + 1])]
#     europe_shapes['Cluster'][df_bin.index] = i + 1

# europe_shapes['Cluster'].replace({1: 'LOW', 2: 'MID-LOW', 3: 'MID-HIGH', 4:'HIGH'}, inplace=True)
europe_shapes['Cluster'].replace({1: 'LOW', 2: 'II', 3: 'MID', 4: 'IV', 5: 'HIGH'}, inplace=True)

cat_type = pd.CategoricalDtype(categories=["LOW", "II", "MID", "IV", "HIGH"], ordered=True)

cluster_mappings = europe_shapes.loc(axis=1)['Cluster', 'ISO_A2', 'ADMIN'].copy()
cluster_mappings['Cluster'] = cluster_mappings['Cluster'].astype(cat_type)
cluster_mappings.dropna(axis=0, how='any', inplace=True)
cluster_mappings.sort_values(['Cluster']).to_csv('cluster_mappings.csv')

#%%

from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

timestamp = datetime.now().strftime("%d_%b_%y,%H_%M")
output_fp = os.path.join(os.path.curdir, 'visualization output', 'electricity clustering')
pp = PdfPages(os.path.join(output_fp ,'output_el_clusters_' + timestamp + '.pdf'))

export_pdf = True
export_png = True

def export_fig(png_name=None):
    if export_pdf:
        pp.savefig(bbox_inches='tight')
    if export_png:
        if not png_name:
            png_name = ax.get_title()
        print(os.path.abspath(output_fp))
        print(png_name)
        plt.savefig(os.path.join(output_fp, png_name + '.png'), format='png', bbox_inches='tight')

#%%
# Plot maps by clusters

fig, ax = plt.subplots(1, 1, figsize=(11, 12), dpi=600)

thresholds = [i*1.0001 for i in thresholds]
norm = colors.BoundaryNorm(thresholds, num_clusters)
cmap = colors.ListedColormap(['midnightblue',
                              #'slategrey',
                              'lightseagreen',
                              'goldenrod',
                              'indigo',
                              'darkred'])
europe_shapes.plot(column='Consumption mix intensity', ax=ax, cmap=cmap, norm=norm, legend=True, edgecolor='w')
europe_shapes[europe_shapes['Cluster'].isna()].plot(column='Cluster', ax=ax, color='lightgrey', edgecolor='darkgrey')

# Set plot limits and format axes
plt.xlim((-19, 34))
plt.ylim((32, 75))
plt.yticks([])
plt.xticks([])

export_fig('cluster_map' + timestamp)

#%%
el_footprints = carbon_footprints_cons.join(europe_shapes.set_index('ISO_A2')['Cluster'], on='country')

# el_footprints = el_footprints.join(prod_df.sum(level='country'), on='country')
el_footprints = el_footprints.join(prod_df.stack().sum(level=['country', -1]).unstack(), on='country')

el_footprints.rename({'Cluster': ('Cluster', '')}, axis=1, inplace=True)
el_footprints.columns = pd.MultiIndex.from_tuples(el_footprints.columns)

#%%
clusters = np.unique(el_footprints['Cluster'].values)
cluster_footprints = pd.DataFrame(index=pd.Index(clusters, name='Cluster'),
                                  columns=carbon_footprints_cons.columns.get_level_values(1).unique())

#%%  Calculate carbon intensity of cluster electricity mix, as weighted average

for name, group in el_footprints.groupby(['Cluster']):
    a = (group['Consumption mix intensity'].mul(group['Total production'])).sum()
    b = group['Total production'].sum()
    cluster_footprints.loc[name] = a.div(b)

cluster_footprints.sort_values(by=2020, axis=0, inplace=True)

#%%


fig, ax = plt.subplots(1, 1, figsize=(8, 7), dpi=600)

# Plot 2020 country-level footprints
plt_df = (el_footprints[[('Consumption mix intensity', 2020), ('Cluster','')]])
plt_df.columns = ['y', 'Cluster']
plt_df.loc(axis=1)['year'] = 2020
# plt_df.set_index('year', inplace=True)
# clrs = {'LOW': 'midnightblue', 'MID-LOW': 'lightseagreen', 'MID-HIGH': 'goldenrod', 'IV': 'indigo', 'HIGH': 'darkred'}
clrs = {'LOW': 'midnightblue', 'II': 'lightseagreen', 'MID': 'goldenrod', 'IV': 'indigo', 'HIGH': 'darkred'}

# plt_df.plot(ax=ax, kind='scatter', x='year', y='y', c=plt_df['Cluster'].apply(lambda x: colors[x]), legend=True)
cluster_footprints.T.plot(ax=ax, cmap=cmap)
plt_df.plot(ax=ax, kind='scatter', x='year', y='y', c=plt_df['Cluster'].apply(lambda x: clrs[x]), legend=True)
# plt_df.plot(ax=ax, y='y', c=plt_df['Cluster'].apply(lambda x: colors[x]), legend=True)


plt.ylabel('Carbon intensity consumption mix \n (weighted average, g CO2/kWh)')
export_fig('electricity_intensity' + timestamp)

plt.show()
pp.close()

#%%
new_mixes = new_mixes.join(europe_shapes.set_index('ISO_A2')['Cluster'], on='country')
groupby_new_mixes = new_mixes.set_index('Cluster', append=True).groupby(['Cluster', 'technology']).sum()
groupby_new_mixes.to_csv('new_mixes_grouped.csv')


#%% Insert cluster footprints into ECOPT2 input file
# prepare cluster footprints for feeding into GAMS
cluster_footprints.dropna(axis=1, how='all', inplace=True)
cluster_footprints.index = ['LOW', 'II', 'MID', 'IV', 'HIGH']
cluster_footprints.index.name = 'reg'
cluster_footprints['enr'] = 'ELC'
cluster_footprints['imp'] = 'GHG'
cluster_footprints = cluster_footprints.set_index(['imp', 'enr'], append=True)
cluster_footprints.index = cluster_footprints.index.reorder_levels(['imp', 'reg', 'enr'])
cluster_footprints = cluster_footprints / 1000

cluster_footprints.to_csv(os.path.join(data_fp, 'el_footprints_pathways.csv'))
fp = os.path.join(data_fp, 'GAMS_input.xlsx')
with pd.ExcelWriter(fp, mode='a') as writer:
    workBook = writer.book
    try:
        workBook.remove(workBook['enr_emiss_int_IAM'])
    except:
            print("Worksheet does not exist")
    finally:
        cluster_footprints.to_excel(writer, 'enr_emiss_int_IAM', startrow=1)
        writer.save()

#%%

cluster_classification = europe_shapes.loc(axis=1)['ISO_A2', 'Cluster']
cluster_classification.sort_values(by='Cluster', inplace=True)

#%% Export for troubleshooting

# output_fp = os.path.join(os.path.curdir, 'calculation output', 'electricity_clustering_output_' + timestamp + '.xlsx')
# with pd.ExcelWriter(output_fp) as writer:
#     new_mixes.to_excel(writer, sheet_name='new_mixes')
#     el_footprints.to_excel(writer, sheet_name='country footprints')
#     tec_int_df.to_excel(writer, sheet_name='tec intensities')

#%% Run check for trade data from UNdata

export_fp = os.path.join(data_fp, 'UNdata', 'UNdata_export_el.csv')
import_fp = os.path.join(data_fp, 'UNdata', 'UNdata_import_el.csv')

exp_check = pd.read_csv(export_fp, usecols=[0, 4], index_col=0, skipfooter=2)
imp_check = pd.read_csv(import_fp, usecols=[0, 4], index_col=0, skipfooter=2)

exp_check.index = coco.convert(exp_check.index.tolist(), to='ISO2')
imp_check.index = coco.convert(imp_check.index.tolist(), to='ISO2')

country_imports = trades_df[2020].sum(axis=0).droplevel('reg')
country_exports = trades_df[2020].sum(axis=1).droplevel('reg')

def check_trades(calc_df, stat_df):
    calc_df = calc_df.to_frame().join(stat_df)
    calc_df['pct diff'] = ((calc_df.iloc[:,0] - calc_df.iloc[:,1]) / calc_df.iloc[:,0]) * 100
    calc_df.sort_values(by='pct diff', inplace=True)
    return calc_df

imp = check_trades(country_imports, imp_check/1000)
exp = check_trades(country_exports, exp_check/1000)