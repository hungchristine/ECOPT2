# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:54:01 2020

@author: chrishun
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

data_fp = os.path.join(os.path.curdir, 'Data', 'load_data')

#%% Load electricity footprints from regionalized footprints paper - these are used to determine clustering

import_fp = os.path.join(data_fp, 'el_footprints.csv')
df = pd.read_csv(import_fp, index_col=[0], usecols=[0, 2])
df.sort_values(by='Consumption mix intensity', inplace=True)
df.dropna(axis=0, inplace=True)

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
       'Serbia', #added
       'Kosovo', #added
       'Montenegro', #added
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

# reg_dict = {'EEU': EEU_ISO2, 'WEU': WEU_ISO2}
reg_dict = {country: ('EEU' if country in EEU_ISO2 else 'WEU') for country in (EEU_ISO2 + WEU_ISO2)}

missing_countries = list(set(EEU_ISO2 + WEU_ISO2) - set(df.index.tolist()))

#%% Import country data and shapefile from Natural Earth, filter for the countries in our study

fp_map = os.path.join(os.path.curdir, 'Data', 'maps', 'ne_10m_admin_0_countries.shp')
country_shapes = gpd.read_file(fp_map)

# Replace ISO_A2 code for France and Norway
country_dict = {'France': 'FR', 'Norway': 'NO'}  # need to manually add these due to no ISO_A2 entry in .shp file
for index, country_row in country_shapes[country_shapes['ISO_A2'] == '-99'].iterrows():
    country = country_row['NAME_SORT']
    if country in list(country_dict.keys()):
        country_shapes.at[index, 'ISO_A2'] = country_dict[country]

europe_ind = country_shapes[country_shapes.iloc(axis=1)[:-1].isin(message_countries)].dropna(axis=0, how='all').index.tolist()
europe_shapes = country_shapes.cx[-12:34, 32:75]  # filter to only the countries within the bounds of our map figures


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

#%%  Perform clustering and add cluster column
num_clusters = 6
thresholds = jenkspy.jenks_breaks(europe_shapes['Consumption mix intensity'], nb_class=num_clusters)

# Add column to dataframe with cluster values
europe_shapes['Cluster'] = np.nan

for i in np.arange(num_clusters):
    df_bin = europe_shapes[(europe_shapes['Consumption mix intensity'] >= thresholds[i]) &
                           (europe_shapes['Consumption mix intensity'] <= thresholds[i+1])]
    europe_shapes['Cluster'][df_bin.index] = i + 1

#%%
# Plot maps by clusters
"""
fig, ax = plt.subplots(1, 1, figsize=(12, 11), dpi=600)

# europe_shapes.plot(column='Cluster', ax=ax, cmap='Dark2', legend=True)
norm = colors.BoundaryNorm(thresholds, 6)
cmap = colors.ListedColormap(['midnightblue',
                              'slategrey',
                              'lightseagreen',
                              'goldenrod',
                              'indigo',
                              'darkred'])
europe_shapes.plot(column='Consumption mix intensity', ax=ax, cmap=cmap, norm=norm, legend=True, edgecolor='w')
europe_shapes[europe_shapes['Cluster'].isna()].plot(column='Cluster', ax=ax, color='lightgrey', edgecolor='darkgrey')

# Set plot limits and format axes
plt.xlim((-12, 34))
plt.ylim((32, 75))
plt.yticks([])
plt.xticks([])
"""

#%%

message_fp = import_fp = os.path.join(os.path.curdir, 'Data', 'MESSAGE_SSP2_electricity pathways.xlsx')
message_el = pd.read_excel(message_fp, index_col=[0, 1, 2], header=[0], usecols='C:P', skipfooter=2)

#%% Calculate electricity pathways from MESSAGE normalized to 2020 values
message_el.index.rename(['reg', 'MESSAGE tec', 'units'], inplace=True)

message_el_shares = pd.DataFrame()
for reg in message_el.index.unique(level=0):
    temp_df = message_el.loc[reg].div(message_el[2020][reg], level=0, axis=0)
    temp_df = pd.concat([temp_df], keys=[reg], names=['reg'])
    message_el_shares = message_el_shares.append(temp_df)

message_el_shares.index = message_el_shares.index.droplevel(-1)  # drop 'units' column of index

#%%
"""
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 5), gridspec_kw={'wspace': 0.05}, dpi=600)


for ax, reg in zip(axes, message_el.index.unique(level=0)):
    message_el.loc[reg].stack().unstack(level=0).plot(ax=ax, kind='bar', stacked=True, legend=False, title=f'{reg}')
    ax.set_xticklabels(message_el.columns.tolist(), rotation=45)

handles, labels = axes[0].get_legend_handles_labels()
labels = [re.split(r'\|', label, maxsplit=2)[-1] for label in labels]  # reformat legend labels
plt.legend(handles, labels, bbox_to_anchor=(1, 1))

axes[0].set_ylabel('Secondary Energy | Electricity \n (EJ/yr)')
plt.title('Electricity technology pathways from MESSAGEix')

#%% Plot shares
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 5), gridspec_kw={'wspace': 0.05}, dpi=600)

for ax, reg in zip(axes, message_el_shares.index.unique(level=0)):
    plot_shares = (message_el_shares.loc[reg].loc(axis=1)[2020:].stack().unstack(level=0))
    plot_shares.plot(ax=ax, kind='area', legend=False, label=message_el_shares.loc[reg].index, title=f'{reg}')
    ax.set_xticklabels(plot_shares.index.values, rotation=45)

handles, labels = axes[1].get_legend_handles_labels()
labels = [re.split(r'\|', label, maxsplit=2)[-1] for label in labels]  # reformat legend labels
plt.legend(handles, labels, bbox_to_anchor=(1, 1))

axes[0].set_ylabel('Share of electricity technology, \n normalized to 2020 shares')
"""
#%%

""" Load electricity mixes, regionalized LCA/hybrid LCA factors from BEV footprints """
mix_fp = os.path.join(data_fp, 'prod_mixes.csv')  # from ENTSO-E (see extract_bentso.py)
trades_fp = os.path.join(data_fp, 'el_trades.csv')  # from ENTSO-E (see extract_bentso.py)
tec_int_fp = os.path.join(data_fp, 'tec_intensities.csv')  # hybridized, regionalized LCA factors for electricity generation

mix_df = pd.read_csv(mix_fp, index_col=[0], na_values='-')  # 2019 production mix by technology, in TWh
trades_df = pd.read_csv(trades_fp, index_col=[0], na_values='-')  # 2019 production
tec_int_df = pd.read_csv(tec_int_fp, index_col=[0], na_values='-')

iso_a2 = europe_shapes[europe_shapes['Consumption mix intensity'].notna()].ISO_A2
iso_a2.rename('country', inplace=True)

def reformat_el_df(df):
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

#%%
""" Need proxy regionalized factors for missing countries """

LU = {'Wind Onshore': 0.245,
      'Hydro Pumped Storage': 1.337,
      'Fossil Gas': 0.194,
      'Waste': 0.124,
      'Solar': 0.112,
      'Biomass': 0.170} # in GWhIEA statistics 2018 https://www.iea.org/data-and-statistics?country=LUXEMBOU&fuel=Electricity%20and%20heat&indicator=Electricity%20generation%20by%20source;
# biomass == biofuels
# LU: 1431 GWh in pumped, 0.1 TWh natural flow  [https://www.vgb.org/hydropower_fact_sheets_2018.rss?dfid=91827]

HR = {'Wind Onshore': 1.204,
      'Hydro Water Reservoir':5.508,
      'Fossil gas': 3.090,
      'Waste': 0.124,
      'Solar': 0.079,
      'Biomass': 0.526,
      'Fossil Oil': 0.210,
      'Fossil Hard coal': 1.367}
# HR: 165 GWh in pumped, 6.39 TWh natural flow  [https://www.vgb.org/hydropower_fact_sheets_2018.rss?dfid=91827]

AL = {'Hydro Water Reservoir': 4.525}  # unconfirmed hydro PP type...

proxy_prod_mix = pd.DataFrame([LU, HR, AL], index=['LU', 'HR', 'AL'], columns=mix_df.columns)
# proxy_prod_mix = pd.concat([proxy_prod_mix], keys=[2020], names=['year', 'technology'], axis=1)
mix_df = mix_df.append(proxy_prod_mix, sort=True)

# Add placeholders for mixxing countries
""" todo: make better proxies """
mix_df = mix_df.append(pd.DataFrame(index=['XK', 'LI', 'AD', 'MC']))
mix_df.loc['XK'] = mix_df.loc['PL']
mix_df.loc['LI'] = mix_df.loc['NO']
mix_df.loc['AD'] = mix_df.loc['ES']
mix_df.loc['MC'] = mix_df.loc['FR']


#%%

prod_shares = mix_df.div(mix_df.sum(axis=1), axis=0)
# col_labels = pd.MultiIndex.from_product([message_el.loc(axis=1)[2020:].columns, prod_shares.columns], names=['year', 'technology'])

# prod_shares = pd.concat([prod_shares], keys=[2020], names=['year', 'technology'], axis=1)
# prod_shares = prod_shares.reindex(columns=col_labels)

col_labels = pd.MultiIndex.from_product([message_el.loc(axis=1)[2020:].columns, prod_shares.columns], names=['year', 'technology'])

mix_df = pd.concat([mix_df], keys=[2020], names=['year', 'technology'], axis=1)
mix_df = mix_df.reindex(columns=col_labels)

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
AD: {imports: 471.3m KWh, exports: 6000 kWh} #wlrdata.info """

proxy_prod_int = {'LU': 505,   # LU/HR from Moro and Lonza
            'HR': 465,
            'AL': df.at['NO', 'Consumption mix intensity'],
            'XK': df.at['PL', 'Consumption mix intensity'],
            'LI': df.at['NO', 'Consumption mix intensity'],
            'AD': df.at['ES', 'Consumption mix intensity'],
            'MC': df.at['FR', 'Consumption mix intensity']}

#%%
"""--- Calculate change in technology shares according to MESSAGE ---"""
#  Make dict that translates MESSAGE technologies to our technologies


""" for year in years:
        generate new production mix (change )
"""

""" Calculate new intensities through time [using constant LCA factors for each technology] --> trading stays the same? """

""" """
# Match ENTSO-E "technologies to MES"SAGE technologies
"""match_tec_dict: {'Biomass': ['Secondary Energy|Electricity|Biomass|w/o CCS'],
                 'Fossil Brown coal/Lignite': ['Secondary Energy|Electricity|Coal|w/o CCS'],
                 'Fossil Coal-derived gas': ['Secondary Energy|Electricity|Gas|w/o CCS'],
                 'Fossil Gas': ['Secondary Energy|Electricity|Gas|w/o CCS'],
                 'Fossil Hard coal': 'Secondary Energy|Electricity|Coal|w/o CCS'],
                 'Fossil Oil': ['Secondary Energy|Electricity|Oil'],
                 'Fossil Oil shale': ['Secondary Energy|Electricity|Oil'],
                 'Fossil Peat': ['Secondary Energy|Electricity|Coal|w/o CCS'],
                 'Geothermal': ['Secondary Energy|Electricity|Geothermal'],
                 'Hydro Pumped Storage': ['Secondary Energy|Electricity|Hydro'],
                 'Hydro Run-of-river and poundage': ['Secondary Energy|Electricity|Hydro'],
                 'Hydro Water Reservoir': ['Secondary Energy|Electricity|Hydro'],
                 'Marine': ['Secondary Energy|Electricity|Other'],
                 'Nuclear': ['Secondary Energy|Electricity|Nuclear'],
                 'Other': ['Secondary Energy|Electricity|Other'],
                 'Other renewable': ['Secondary Energy|Electricity|Other'],
                 'Solar': ['Secondary Energy|Electricity|Solar'],
                 'Waste': ['Secondary Energy|Electricity|Other'],
                 'Wind Offshore': ['Secondary Energy|Electricity|Wind'],
                 'Wind Onshore': ['Secondary Energy|Electricity|Wind']}"""

match_tec_dict = {'Biomass': 'Secondary Energy|Electricity|Biomass|w/o CCS',
                 'Fossil Brown coal/Lignite': 'Secondary Energy|Electricity|Coal|w/o CCS',
                 'Fossil Coal-derived gas': 'Secondary Energy|Electricity|Gas|w/o CCS',
                 'Fossil Gas': 'Secondary Energy|Electricity|Gas|w/o CCS',
                 'Fossil Hard coal': 'Secondary Energy|Electricity|Coal|w/o CCS',
                 'Fossil Oil': 'Secondary Energy|Electricity|Oil',
                 'Fossil Oil shale': 'Secondary Energy|Electricity|Oil',
                 'Fossil Peat': 'Secondary Energy|Electricity|Coal|w/o CCS',
                 'Geothermal': 'Secondary Energy|Electricity|Geothermal',
                 'Hydro Pumped Storage': 'Secondary Energy|Electricity|Hydro',
                 'Hydro Run-of-river and poundage': 'Secondary Energy|Electricity|Hydro',
                 'Hydro Water Reservoir': 'Secondary Energy|Electricity|Hydro',
                 'Marine': 'Secondary Energy|Electricity|Other',
                 'Nuclear': 'Secondary Energy|Electricity|Nuclear',
                 'Other': 'Secondary Energy|Electricity|Other',
                 'Other renewable': 'Secondary Energy|Electricity|Other',
                 'Solar': 'Secondary Energy|Electricity|Solar',
                 'Waste': 'Secondary Energy|Electricity|Other',
                 'Wind Offshore': 'Secondary Energy|Electricity|Wind',
                 'Wind Onshore': 'Secondary Energy|Electricity|Wind'}


#%%
# Introduce WEU/EEU regions to multiindex in prodshares for applying MESSAGE transformation pathways

reg_mi = pd.MultiIndex.from_tuples([(reg_dict[country], country) for country in mix_df.index.tolist()], names=['reg', 'country'])
reg_mi = reg_mi.sort_values()
mix_df = mix_df.reindex(reg_mi, level=1)
mix_df = mix_df.stack('technology')

tec_mapping = pd.DataFrame.from_dict(match_tec_dict, orient='index', columns=['MESSAGE tec'])
mix_df = mix_df.join(tec_mapping, on='technology')  # add MESSAGE technology mappings
mix_df.set_index(['MESSAGE tec'], append=True, inplace=True)

#%% Populate future production mixes using MESSAGE transformation pathways (factors normalized by 2020 production)

prod_df = pd.DataFrame(index=mix_df.index, columns=mix_df.columns)
for ind, row in mix_df.iterrows():
    reg = ind[0]
    msg_tec = ind[3]
    prod_df.loc[ind] = (message_el_shares.loc[(reg, msg_tec), 2020:]) * (mix_df.loc[ind][2020])

#%% Adjust trade matrix to match growth in electricity production

ann_growth = prod_df.groupby(['reg']).sum()
ann_growth_norm = ann_growth.div(ann_growth[2020], axis=0)  # increase in electricity production normalized to 2020

""" Temporary approximation: use overall growth across region to calculate increase in trade flows"""
""" todo: find way to divide to regions?"""
ann_growth_norm_temp = prod_df.sum().div(prod_df.sum()[2020])

trades_df = trades_df.reindex(reg_mi, level=1)
trades_df = trades_df.reindex(reg_mi, level=1, axis=1)
trades_df = pd.concat([trades_df], keys=[2020], names=['year', 'reg','country'], axis=1)

# Add the remaining years in the study period
df = pd.DataFrame(columns=message_el.loc(axis=1)[2020:].columns, index=reg_mi)
trades_mi = df.stack(dropna=False).index.reorder_levels([2,0,1])
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
    r_gendem = totgen / g #y  # All countries should be 1

    totcons = Xgen.sum(axis=0)
    r_condem = totcons / y  # All countries should be 1

    prod_by_tec = production/g
    prod_by_tec = prod_by_tec.unstack(['technology', 'MESSAGE tec'])
    # prod_by_tec = prod_by_tec.stack()
    # prod_by_tec.index = prod_by_tec.index.swaplevel(0,1)
    # prod_by_tec.sort_index(inplace=True)

    # ### Generation technology matrix

    # TC is a country-by-generation technology matrix - normalized to share of total domestic generation, i.e., normalized generation/production mix
    # technology generation, kWh/ kWh domestic generated electricity
    prod = production.unstack(['technology', 'MESSAGE tec'])
    prod = prod.T.reset_index(level='MESSAGE tec').T  #set MESSAGE classifications for columns so we can do the sort
    prod.sort_index(axis=1, inplace=True)
    prod = ((prod.T).set_index(['MESSAGE tec'], append=True)).T
    prod.fillna(0, inplace=True)

    # TC = pd.DataFrame(np.matmul(np.linalg.pinv(np.diag(g)), production.unstack(['technology', 'MESSAGE tec'])))#, index=g.index)#, columns=production.unstack(['technology', 'MESSAGE tec']).columns)
    TC = pd.DataFrame(np.matmul(np.linalg.pinv(np.diag(g)), prod))#, index=g.index)#, columns=production.unstack(['technology', 'MESSAGE tec']).columns)
    TCsum = TC.sum(axis=1)  # Quality assurance - each country should sum to 1

    # # Calculate technology generation mix in GWh based on production in each region
    TGP = pd.DataFrame(np.matmul(TC.transpose(), np.diag(g)))#, index=TC.columns, columns=g.index)  # .== production

    # # Carbon intensity of production mix
    CFPI_no_TD = pd.DataFrame(prod.multiply(C).sum(axis=1) / prod.sum(axis=1), columns=['Production mix intensity'])  # production mix intensity without losses
    CFPI_no_TD.fillna(0, inplace=True)

    # # Carbon intensity of consumption mix
    CFCI_no_TD = pd.DataFrame(np.matmul(CFPI_no_TD.T, Lgen).T)
    CFCI_no_TD.columns = ['Consumption mix intensity']

    CF = pd.DataFrame()
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

carbon_footprints_prod = pd.DataFrame(index=reg_mi, columns=message_el_shares[2020:].columns)
carbon_footprints_cons = pd.DataFrame(index=reg_mi, columns=message_el_shares[2020:].columns)

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