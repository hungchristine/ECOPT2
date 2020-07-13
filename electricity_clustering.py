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
## TODO: fix code to not need pre-calculated production mix intensities (df, europe_shapes)
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


#%% Import MESSAGE results and visualize el pathways

message_fp = import_fp = os.path.join(data_fp, 'MESSAGE_SSP2_400_ppm.xlsx')  #'MESSAGE_SSP2_electricity pathways.xlsx')
message_el = pd.read_excel(message_fp, index_col=[0, 1, 2], header=[0], usecols='C:P', skipfooter=7)

#%% Calculate electricity pathways from MESSAGE normalized to 2020 values
message_el.index.rename(['reg', 'MESSAGE tec', 'units'], inplace=True)

message_el_shares = pd.DataFrame()

for reg in message_el.index.unique(level=0):
    temp_df = message_el.loc[reg].div(message_el[2020][reg], level=0, axis=0)
    temp_df = pd.concat([temp_df], keys=[reg], names=['reg'])
    message_el_shares = message_el_shares.append(temp_df)

message_el_shares.index = message_el_shares.index.droplevel(-1)  # drop 'units' column of index


#%% Plot MESSAGE pathways for EEU and WEU

fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 5), gridspec_kw={'wspace': 0.05}, dpi=600)

for ax, reg in zip(axes, message_el.index.unique(level=0)):
    message_el.loc[reg].stack().unstack(level=0).plot(ax=ax, kind='bar', stacked=True, legend=False, title=f'{reg}')
    ax.set_xticklabels(message_el.columns.tolist(), rotation=45)

handles, labels = axes[0].get_legend_handles_labels()
labels = [re.split(r'\|', label, maxsplit=2)[-1] for label in labels]  # reformat legend labels
plt.legend(handles, labels, bbox_to_anchor=(1, 1))

axes[0].set_ylabel('Secondary Energy | Electricity \n (EJ/yr)')
plt.title('Electricity technology pathways from MESSAGEix')

#%%

# tmp = message_el.loc['WEU'].stack().unstack(level=0)
# tmp.div(tmp.sum(axis=0)).plot(kind='area', legend=False)

#%% Plot shares
"""fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 5), gridspec_kw={'wspace': 0.05}, dpi=600)

for ax, reg in zip(axes, message_el_shares.index.unique(level=0)):
    plot_shares = (message_el_shares.loc[reg].loc(axis=1)[2020:].stack().unstack(level=0))
    plot_shares.plot(ax=ax, kind='area', legend=False, label=message_el_shares.loc[reg].index, title=f'{reg}')
    ax.set_xticklabels(plot_shares.index.values, rotation=45)

handles, labels = axes[1].get_legend_handles_labels()
labels = [re.split(r'\|', label, maxsplit=2)[-1] for label in labels]  # reformat legend labels
plt.legend(handles, labels, bbox_to_anchor=(1, 1))

axes[0].set_ylabel('Share of electricity technology, \n normalized to 2020 shares')"""

#%% Import electricity data from ENTSO - to be used as 2020 baseline

""" Load electricity mixes, regionalized LCA/hybrid LCA factors from BEV footprints """
mix_fp = os.path.join(data_fp, 'prod_mixes.csv')  # from ENTSO-E (see extract_bentso.py)
trades_fp = os.path.join(data_fp, 'el_trades.csv')  # from ENTSO-E (see extract_bentso.py)
tec_int_fp = os.path.join(data_fp, 'tec_intensities.csv')  # hybridized, regionalized LCA factors for electricity generation

mix_df = pd.read_csv(mix_fp, index_col=[0], na_values='-')  # 2019 production mix by technology, in TWh, from ENTSO-E
trades_df = pd.read_csv(trades_fp, index_col=[0], na_values='-')  # 2019 production, in TWh
tec_int_df = pd.read_csv(tec_int_fp, index_col=[0], na_values='-') # regionalized (hybridized) carbon intensity factors of generation (g COw-e/kWh)
tec_int_df.rename(columns={'other':'Other', 'other renewable':'Other renewable'}, inplace=True)

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

#%% Deal with proxy countries to fill gaps from ENTSO-E Transparency Portal
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
mix_df.loc['XK'] = mix_df.loc['PL'] * (5.726+0.179+0.012)/(mix_df.loc['PL'].sum()) # scale production to keep shares
mix_df.loc['LI'] = mix_df.loc['NO'] * (80/1e6)/mix_df.loc['NO'].sum()
mix_df.loc['AD'] = mix_df.loc['ES'] * (99/1e3)/mix_df.loc['ES'].sum() #https://www.worlddata.info/europe/andorra/energy-consumption.php
mix_df.loc['MC'] = mix_df.loc['FR'] * (536/1e3)/mix_df.loc['FR'].sum() #https://en.wikipedia.org/wiki/Energy_in_Monaco#:~:text=Monaco%20has%20no%20domestic%20sources,gas%20and%20fuels%20from%20France.&text=In%202018%2C%20the%20country%20used,it%20was%20used%20tertiary%20services.

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
AD: {imports: 471.3m KWh, exports: 6000 kWh} #wlrdata.info """

proxy_prod_int = {'LU': 505,   # LU/HR from Moro and Lonza
            'HR': 465,
            'AL': df.at['NO', 'Consumption mix intensity'],
            'XK': df.at['PL', 'Consumption mix intensity'],
            'LI': df.at['NO', 'Consumption mix intensity'],
            'AD': df.at['ES', 'Consumption mix intensity'],
            'MC': df.at['FR', 'Consumption mix intensity']}

#%%

prod_shares = mix_df.div(mix_df.sum(axis=1), axis=0)
# col_labels = pd.MultiIndex.from_product([message_el.loc(axis=1)[2020:].columns, prod_shares.columns], names=['year', 'technology'])

# prod_shares = pd.concat([prod_shares], keys=[2020], names=['year', 'technology'], axis=1)
# prod_shares = prod_shares.reindex(columns=col_labels)

col_labels = pd.MultiIndex.from_product([message_el.loc(axis=1)[2020:].columns, prod_shares.columns], names=['year', 'technology'])

mix_df = pd.concat([mix_df], keys=[2020], names=['year', 'technology'], axis=1)
mix_df = mix_df.reindex(columns=col_labels)
mix_df = mix_df.sort_index(axis=0)


#%% Introduce proxy carbon intensity factors for missing countries
tec_int_df = tec_int_df.reindex(mix_df.index)
proxy_prod = tec_int_df.loc[tec_int_df.isnull().sum(axis=1) == len(tec_int_df.columns)].index  # get index of rows with all np.nans
for country in proxy_prod:
    tec_list = mix_df.loc[country][2020].index[mix_df.loc[country][2020].notnull()] # get list of relevant tecs for the country, i.e., where there is production activity
    for tec in tec_list:
        # TODO: make this EEU/WEU specific
        print(f'{country}, {tec}')
        tec_int_df.loc[country][tec] = tec_int_df[tec].mean() # use technology mean of other regions as proxy


#%%
# Introduce WEU/EEU regions to multiindex in prodshares for applying MESSAGE transformation pathways

reg_mi = pd.MultiIndex.from_tuples([(reg_dict[country], country) for country in mix_df.index.tolist()], names=['reg', 'country'])
reg_mi = reg_mi.sort_values()
mix_df = mix_df.reindex(reg_mi, level=1)

#%%
####### Begin data harmonization (disaggregation) work


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

new_keys = {'Secondary Energy|Electricity|Solar|CSP': 'Solar CSP',
            'Secondary Energy|Electricity|Solar|PV': 'Solar PV',
            'Secondary Energy|Electricity|Biomass|w/o CCS': 'Biomass w/o CCS',
            'Secondary Energy|Electricity|Biomass|w/ CCS': 'Biomass w/ CCS',
            'Secondary Energy|Electricity|Gas|w/o CCS': 'Gas w/o CCS',
            'Secondary Energy|Electricity|Gas|w/ CCS': 'Gas w/ CCS',
            'Secondary Energy|Electricity|Coal|w/o CCS': 'Coal w/o CCS',
            'Secondary Energy|Electricity|Coal|w/ CCS': 'Coal w/ CCS'
            }

# Todo: refine these?
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
            tec_int_df[lvl_name] = tec_int_df[key].mul(new_CFs[lvl_name])
        mix_df.drop(columns=[(2020, key)], inplace=True)
        tec_int_df.drop(columns=key, inplace=True)

    tec_shares = tec_shares.drop_duplicates()
    return mix_df, tec_shares

#%%
"""--- Calculate change in technology shares according to MESSAGE ---"""
#  Make dict that maps MESSAGE technology groupings to ENTSO-E technology groupings

# Match ENTSO-E "technologies to MESSAGE technologies
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
                       'Wind Offshore': 'Secondary Energy|Electricity|Wind|Offshore',
                       'Wind Onshore': 'Secondary Energy|Electricity|Wind|Onshore'
                       })

# Disaggregate MESSAGE to country level using shares from ENTSO (by region and technology)
mix_df, tec_shares = disagg_entsoe(mix_df, tec_shares)

#%%
# Add MESSAGE technology mapping to production matrix from ENTSO-E
mix_df = mix_df.stack('technology')
tec_mapping = pd.DataFrame.from_dict(match_tec_dict, orient='index', columns=['MESSAGE tec'])
tec_mapping.index.rename('technology', inplace=True)
mix_df = mix_df.join(tec_mapping, on='technology')  # add MESSAGE technology mappings
mix_df.set_index(['MESSAGE tec'], append=True, inplace=True)


entso = mix_df.copy()
# entso = message_el[]
message = message_el.loc(axis=1)[2020:].copy()
message = message * 277.78  # convert from EJ/year to TWh/year

# ents_sum = entso.loc['EEU'].sum(level='technology')

#%% Make test with only MESSAGE mixes (instead of using ENTSO)

message = message.droplevel('units')
reg_tec_shares = entso[2020].div(entso[2020].groupby(['reg', 'technology']).sum())
reg_tec_shares = reg_tec_shares.div(reg_tec_shares.sum(level=['reg', 'MESSAGE tec']))  # normalize the MESSAGE categories that are aggregated
reg_tec_shares = reg_tec_shares.reorder_levels(['reg', 'country', 'technology', 'MESSAGE tec'])
geo_mi = pd.MultiIndex.from_product([['EEU'],['BG', 'HR', 'CZ', 'HU', 'PL', 'RO', 'SK'],['Geothermal'], ['Secondary Energy|Electricity|Geothermal']]) #countries from https://www.stjornarradid.is/media/atvinnuvegaraduneyti-media/media/frettir/080119_geothermal_europe_memo_for_ossur.pdf
geothermal_df = pd.DataFrame([1/len(geo_mi)]*len(geo_mi), index=geo_mi, columns=[2020])
reg_tec_shares = pd.DataFrame(reg_tec_shares).append(geothermal_df)

# Manually insert offshore wind, which MESSAGE has in EEU, but ENTSO-E does not (???)
offshore_wind = reg_tec_shares[reg_tec_shares.index.get_level_values('technology').str.contains('Wind')]
offshore_wind = offshore_wind.drop(index='WEU')#.to_frame()
offshore_wind.reset_index(inplace=True)
offshore_wind['technology'] = offshore_wind['technology'].str.replace('Onshore', 'Offshore')
offshore_wind['MESSAGE tec'] = offshore_wind['MESSAGE tec'].str.replace('Onshore','Offshore')
offshore_wind.set_index(['reg', 'country', 'technology', 'MESSAGE tec'], inplace=True)

# reg_tec_shares = pd.DataFrame(reg_tec_shares).append(offshore_wind)
reg_tec_shares = reg_tec_shares.append(offshore_wind)
# sub_tec_shares = tec_mapping.join(pd.DataFrame([1]*len(tec_mapping), index=tec_mapping.index, columns=[2020])).set_index('MESSAGE tec', append=True)
# sub_tec_shares = sub_tec_shares.div(sub_tec_shares.sum(level=['MESSAGE tec']))
# sub_tec_shares = sub_tec_shares[~sub_tec_shares.index.get_level_values(1).duplicated()].droplevel('technology')
# disagg_message = sub_tec_shares.mul(reg_tec_shares, axis=0) * (message)

disagg_message = (message[2020] * reg_tec_shares[2020]).reorder_levels(['reg', 'country', 'technology', 'MESSAGE tec'])
# disagg_message = disagg_message[2020]
check_disagg = disagg_message.sum(level=['reg', 'MESSAGE tec'])  # check sums add up to what is reported by MESSAGE

# country_disagg_message =  (reg_tec_shares * message[2020]).reorder_levels(['reg', 'country', 'technology', 'MESSAGE tec'])

#NEED TO FIND A WAY TO ADD THE OFFSHORE WIND TO ENTSO
add_ind = disagg_message.index.difference(entso.index, sort=False)
entso = entso.append(pd.DataFrame(index=add_ind, columns=entso.columns))
entso[2020] = disagg_message
entso = entso.sort_index(level=['reg', 'country', 'technology'])

#%%

# disagg_entsoe(country_disagg_message, tec_shares)
# disagg_entsoe(mix_df, tec_shares)

#%% Attempt distributing MESSAGE to country level

# prep mapping dataframe for joining
tec_mi = tec_mapping.reset_index().set_index('MESSAGE tec')

message['MSG tec'] = message.index.get_level_values('MESSAGE tec')
message = message.join(tec_mi, on='MSG tec', rsuffix='_other')
message.set_index('technology', append=True, inplace=True)
message.drop(columns='MSG tec', inplace=True)
message.drop(index=('WEU', 'Secondary Energy|Electricity|Oil|w/o CCS', 'Fossil Oil shale'), inplace=True)
message.index = message.index.swaplevel(-1, -2)


# mi = message.index.difference(entso.droplevel('country').reorder_levels(['reg', 'technology','MESSAGE tec']).index, sort=False)
# message = message.drop(index=mi)

message = message.mul(message.div(message.sum(level=['reg', 'MESSAGE tec'])))

#%%
# # remove CCS technologies for now
# drop_labels = [label for label in message.index.unique('MESSAGE tec') if label in list(new_keys.keys())]
# # ccs_labels = [label for label in message.index.get_level_values('MESSAGE tec') if 'w/ CCS' in label]
# tmp_message = message.loc[(slice(None), drop_labels),:]
# message = message.drop(index=drop_labels, level='MESSAGE tec')

# # calculate shares of each sub-technology for disaggregating MESSAGE
# # categories
# x = entso[2020].unstack('MESSAGE tec').sum(level=['reg','technology'])#entso[2020].unstack('MESSAGE tec').sum('country')
# entso_shares = x.div(x.sum(level='reg'))

# # need to change calculations to adjust for technology aggregation in message
# # use tec shares: df.groupby('MESSAGE tec')
# # group.div(group.sum(axis=0, level='ENTSOE tec'))
# message['MSG tec'] = message.index.get_level_values('MESSAGE tec')
# message = message.join(tec_mi, on='MSG tec', rsuffix='_other')
# message.set_index('technology', append=True, inplace=True)
# message.drop(columns='MSG tec', inplace=True)
# message.index = message.index.swaplevel(-1, -2)
# # message = (message.replace(0, np.nan)).dropna(how='all', axis=0)

# # remove indices from disaggregated MESSAGE that are not physically present (as per ENTSO)
# mi = message.index.difference(entso.droplevel('country').reorder_levels(['reg', 'technology','MESSAGE tec']).index, sort=False)
# message = message.drop(index=mi)
# # mi = message.droplevel('units').index.difference(entso.droplevel('country').reorder_levels(['reg','MESSAGE tec', 'technology']).index, sort=False)
# # message = message.droplevel('units').drop(index=mi)

# tec_shares = tec_shares.join(tec_mi, on='MESSAGE tec')
# tec_shares.set_index('technology', append=True, inplace=True)
# tec_shares = tec_shares.reorder_levels(['reg', 'technology', 'MESSAGE tec'])

# # tmp_tec_share = tec_shares.mul(tec_shares.div(tec_shares.sum(level=['reg', 'MESSAGE tec'])))

# # calculate disaggregation shares for MESSAGE categories with further resolution in ENTSO
# msg_tec_share = message.div(message.sum(level=['reg', 'MESSAGE tec']))

# for year in message.columns:
#     # sub_tec_shares.columns = [year]
#     message[year] = message[year].mul(entso_shares.stack())
#     # tmp_message[year] = tmp_message[year].mul(tec_shares[year]).mul(sub_tec_shares[year])  # use MESSAGE-based future shares for technologies not in ENTSO dataset
#     # tmp_message[year] = tmp_message.reset_index('units')[year].mul(tec_shares[year])  # use MESSAGE-based future shares for technologies not in ENTSO dataset
#     # tmp_message[year] = tmp_message[year].mul(tmp_tec_share[year])  # use MESSAGE-based future shares for technologies not in ENTSO dataset

# # tmp_message = tmp_message.droplevel('units')
# tmp_message = tmp_message.join(tec_mi, on='MESSAGE tec')
# tmp_message.set_index('technology', append=True, inplace=True)
# tmp_message = tmp_message.reorder_levels(['reg', 'technology', 'MESSAGE tec'])
# tmp_message = tmp_message.mul(msg_tec_share)
# # tmp_message = tmp_message.mul(tmp_tec_share)
# message = message.append(tmp_message)


message = message[(message.T !=0).any()]  # remove country-technology combos that have 0 production through study period
message.dropna(axis=0, how='all', inplace=True)  # remove irrelevant country-technology combos
message.replace(np.nan, 0, inplace=True)

# message = message * sub_tec_shares

delta = pd.DataFrame()

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

new_mixes = pd.DataFrame(index=entso.index, columns = message.columns)  # dataframe to contain calcualted country mixes
new_mixes[2020] = entso[2020]
# new_mixes[2020] = country_disagg_message

# new_mixes.replace(0, np.nan, inplace=True)
ind = pd.MultiIndex.from_product([new_mixes.index.unique('country'), new_mixes.index.unique('technology')])
reduced_prod = pd.DataFrame()

freed_cap = pd.DataFrame(index=reg_mi, columns=message.columns)
shares_tec = pd.DataFrame()
# incr_tecs = pd.DataFrame(index=new_mixes.index)
null_tecs_df = pd.DataFrame(index=new_mixes.index, columns=message.columns)
neg_tecs = pd.DataFrame(index=new_mixes.index, columns=message.columns)
# freed_cap = pd.DataFrame(index=ind.unique(level='reg'), columns=message.columns)
new_mixes_check = pd.DataFrame(index=pd.MultiIndex.from_product([['EEU', 'WEU'],new_mixes.index.unique('technology')], names=['reg','technology']), columns=message.columns)
ratio = pd.DataFrame(index=entso.index, columns = message.columns)
# Calculate new amounts of electricity generated by technologies getting phased out
incr_prod_df = pd.DataFrame(index=new_mixes.index, columns=message.columns)
red_prod_df = pd.DataFrame(index=new_mixes.index, columns=message.columns)
shares_tec_df = pd.DataFrame(index=message.index, columns=message.columns)
repl_df = pd.DataFrame(index=new_mixes.index, columns=message.columns)
repl_df2 = pd.DataFrame(index=new_mixes.index, columns=message.columns)
freed_cap_df = pd.DataFrame(index=reg_mi, columns=message.columns)
ratio_df = pd.DataFrame(index=new_mixes, columns=message.columns)
decr_prod = pd.DataFrame(index=new_mixes.index, columns=message.columns)
check_free_cap = pd.DataFrame(index=['EEU', 'WEU'], columns=message.columns)
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
        null_tecs = (new_mixes[prev_year].copy()).reset_index('country').loc[null_tec_ind].set_index('country', append=True)
        null_tecs.rename(columns={prev_year: year}, inplace=True)
        null_tecs = null_tecs.reorder_levels(['reg', 'country', 'technology', 'MESSAGE tec'])
        new_mixes.update(null_tecs)
        null_tecs_df.update(null_tecs)

    # Calculate new values for technologies with decreasing production
    # calculate the country distribution for each technology (and region)

    ratio = new_mixes[prev_year].div(new_mixes[prev_year].sum(level=['reg', 'technology']).replace(0, np.nan)) # calculate tec shares by region
    ratio = ratio.rename(year).reset_index('country')
    # ratio_df[year] = ratio[year]
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

    tmp = new_mixes.reset_index(['country', 'MESSAGE tec']).join(delta, how='outer', lsuffix='_mix', sort=False)  # broadcast delta to country-level resolution
    tmp = tmp.set_index(['country'], append=True).reorder_levels(['reg', 'country', 'technology', 'MESSAGE tec'])  # fix index to match other DFs
    tmp = tmp.filter(regex='[0-9]{4}$', axis=1)  # remove the [year] column inherited from new_mixes
    tmp.columns = tmp.columns.astype(int)
    neg_delt_ind = tmp[tmp[year] < 0].index  # find the indices for technologies that go down in production during the decade
    # neg_delt_ind = delta[delta[year]<0].index.droplevel(['MESSAGE tec', 'units'])

    # find the reduction in country-level production for each technology in the period
    reduced_prod = new_mixes[prev_year].loc[neg_delt_ind].sum(level=['reg', 'country', 'technology']) - new_mixes[year].loc[neg_delt_ind].sum(level=['reg','country', 'technology'])
    reduced_prod.rename(year, inplace=True)
    red_prod_df[year] = reduced_prod
    # freed_cap[year] = new_mixes[prev_year].sum(level=['reg', 'technology']).sub(new_mixes[year].sum(level=['reg', 'technology'])) # hydro EEU 2030 ==0....

    sum_neg_delta = reduced_prod.sum(level='reg')
    " sum_neg_delta: reg"
    " pos_delta: reg, technology, MESSAGE tec"
    shares_tec = sum_neg_delta * (pos_delta.div(pos_delta.sum(level='reg')))  # distribution of technologies with increased production for 'making up' for reduced production
    shares_tec_df[year] = shares_tec

    # add region mappings to freed_cap for operations with shares_tec
    # freed_cap['reg'] = [reg_dict[country] for country in freed_cap.index.get_level_values(0)]
    freed_cap = reduced_prod.sum(level=['reg','country'])#.reset_index('country')
    # freed_cap = freed_cap.reset_index(['country']).set_index(['reg'], append=True).reorder_levels(['reg', 'technology'])
    freed_cap_df[year] = freed_cap
    check_free_cap[year] =( message.loc[neg_delta.index][year] - message.loc[neg_delta.index][prev_year]).sum(level='reg')

    # calculate mix of production displacing retired technologies
    """ tmp = ((freed_cap.div(np.abs(neg_delta.sum(level='reg'))))[year]).reindex(repl_prod.index, method='ffill')"""
    # tmp2 = freed_cap.div(np.abs(neg_delta.sum(level='reg')), axis=0, level='reg')
    """print('freed cap')
    print(neg_delta.sum(level='reg'))
    tmp2 = freed_cap.div(freed_cap.sum(level='reg'), level='reg')
    print(tmp2.sum(level='reg'))
    tmp2 = tmp2.reindex(repl_prod.index)  # ffills at country level
    # tmp = freed_cap.reindex(repl_prod.index, method='ffill')
    tmp2.rename(year, inplace=True)
    tmp_df = tmp2
    print('test')
    print(tmp2.mul(shares_tec).sum(level='reg'))
    # repl_prod = tmp.mul(shares_tec)
    repl_prod = pd.DataFrame(tmp2).reset_index('country')
    repl_prod = repl_prod.drop(index=repl_prod.index.difference(shares_tec.index)).sort_index(level=['reg','technology'])  # remove technologies that do not increase in period
    repl_prod[year] = repl_prod[year].mul(shares_tec)
    repl_prod = repl_prod.set_index('country', append=True).reorder_levels(['reg', 'country', 'technology', 'MESSAGE tec'])"""

    #### THERE IS SOMETHING WRONG WITH THE TECHNOLOGIES IN SHARED TEC --> check that the tecs in shares_tec, null_tecs and red_prod are mutually exclusive.
    repl_prod = (freed_cap.div(freed_cap.sum(level='reg')).mul(shares_tec))
    # tmp[year].mul(shares_tec).set_index('country', append=True)
    # repl_prod = repl_prod.reorder_levels(['reg', 'country', 'technology', 'MESSAGE tec']).sort_index(level=['reg', 'country'])
    # repl_prod.replace(np.nan, 0, inplace=True)
    print('replacement production')
    print(freed_cap.sum(level=['reg']))
    print(repl_prod.sum(level='reg'))

    # repl_prod2 = tmp.mul(repl_prod2.join(shares_tec.rename(year)).reorder_levels(['reg', 'country', 'technology', 'MESSAGE tec'])[year])
    # repl_prod2 = repl_prod2.reorder_levels(['reg', 'country', 'technology', 'MESSAGE tec']).sort_index(level=['reg', 'country'])
    # repl_prod2.replace(np.nan, 0, inplace=True)


    # calculate mix of production for additional ("surplus") production
    tec_diff = (pos_delta - shares_tec)
    reg_share = (new_mixes[prev_year].sum(level=['reg', 'country']).div(new_mixes[prev_year].sum(level='reg'))).rename(year)

    add_prod = tec_diff.mul(reg_share).reorder_levels(['reg', 'country', 'technology', 'MESSAGE tec']).sort_index(level=['reg','country'])
    print(add_prod.sum(level='reg'))
    # total production of technologies that increase in the time period
    incr_prod = new_mixes[prev_year].rename(year).replace(np.nan, 0).add(repl_prod, fill_value=0).add(add_prod, fill_value=0)
    # incr_prod = new_mixes[prev_year].rename(year).replace(np.nan, 0) + repl_prod + add_prod # may be problematic drops country-tec combos not originally in new_mixes
    repl_df[year] = repl_prod.reorder_levels(['reg','country','technology','MESSAGE tec'])
    # print('surplus production')

    # repl_df2[year] = repl_prod2
    add_prod_df[year] = add_prod.sort_index(level=['reg','country'])
    incr_prod_df[year] = incr_prod
    incr_prod.dropna(how='all', axis=0, inplace=True)

    new_mixes = new_mixes.reindex(index=incr_prod.index)
    new_mixes.update(incr_prod, overwrite=False)

    new_mixes_check[year] = new_mixes[year].sum(level=['reg', 'technology'])
    print((message[year]-message[prev_year]).sum(level='reg'))
    print(add_prod.sum(level='reg'))
    # incr_prod = (new_mixes[prev_year].sum(level=['reg','country']).div(new_mixes[prev_year].sum(level='reg')))*
    # # Second, calculate the mix of technologies that replaces the 'freed-up' production
    # # shares_tec is the amount of each technlogy where delta>0 increases for each unit of
    # # energy that is reduced (regardless of technology, but region/country specific)
    # # shares_tec = (delta[delta[year] > 0][year].div(np.abs(delta[delta[year] < 0][year].sum(level='reg'))))
    # print('shares_tec')
    # print(shares_tec)
    # incr_cap = freed_cap.join(shares_tec.reset_index(['MESSAGE tec', 'units']), lsuffix='_cap').set_index('MESSAGE tec', append=True)
    # # incr_tecs = new_mixes[prev_year].rename(year) + freed_cap[year].mul(shares_tec.reset_index(['MESSAGE tec', 'units'])[year]) ### problem continued??
    # incr_tecs[year] = new_mixes[prev_year].rename(year) + incr_cap[str(year)+'_cap'].mul(incr_cap[str(year)])
    # incr_tecs = incr_tecs.reorder_levels(['reg', 'country', 'technology', 'MESSAGE tec'])
    # new_mixes.update(incr_tecs, overwrite=False)
    # print('new_mixes[year]')
    # print(new_mixes[year])


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
df.rename_axis(columns='year',inplace=True)
trades_mi = df.stack(dropna=False).index.reorder_levels(['year', 'reg', 'country']) #[2, 0, 1])
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
carbon_footprints_cons['Consumption mix intensity'] = np.where(carbon_footprints_cons['Consumption mix intensity']<1e-2,
                                                               np.nan, carbon_footprints_cons['Consumption mix intensity'])
"""temporary - removal of countries with no consumption mix"""
# carbon_footprints_cons.dropna(axis=0, how='any', inplace=True)

carbon_footprints_cons_tmp = carbon_footprints_cons['Consumption mix intensity', 2020].droplevel('reg')
carbon_footprints_cons_tmp.rename('Consumption mix intensity', inplace=True)

# add consumption mix columns to europe_shapes
europe_shapes.drop(columns=['Consumption mix intensity'], inplace=True)  # drop precalculated CF factors in favour of those just calculated which are more complete
europe_shapes = europe_shapes.join(carbon_footprints_cons_tmp, on='ISO_A2')

#%%
# Perform clustering and add cluster column

num_clusters = 5
thresholds = jenkspy.jenks_breaks(europe_shapes['Consumption mix intensity'], nb_class=num_clusters)

# Add column to dataframe with cluster values
europe_shapes['Cluster'] = np.nan

for i in np.arange(num_clusters):
    df_bin = europe_shapes[(europe_shapes['Consumption mix intensity'] >= thresholds[i]) &
                           (europe_shapes['Consumption mix intensity'] <= thresholds[i + 1])]
    europe_shapes['Cluster'][df_bin.index] = i + 1

#%%
# Plot maps by clusters

fig, ax = plt.subplots(1, 1, figsize=(12, 11), dpi=600)

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
plt.xlim((-12, 34))
plt.ylim((32, 75))
plt.yticks([])
plt.xticks([])

#%%
el_footprints = carbon_footprints_cons.join(europe_shapes.set_index('ISO_A2')['Cluster'], on='country')

# el_footprints = el_footprints.join(prod_df.sum(level='country'), on='country')
el_footprints = el_footprints.join(prod_df.stack().sum(level=['country', -1]).unstack(), on='country')

el_footprints.rename({'Cluster': ('Cluster','')}, axis=1, inplace=True)
el_footprints.columns = pd.MultiIndex.from_tuples(el_footprints.columns)

#%%
clusters = np.unique(el_footprints['Cluster'].values)
cluster_footprints = pd.DataFrame(index=pd.Index(clusters, name='Cluster'),
                                  columns=el_footprints.columns.get_level_values(1).unique())

#%% Calculate carbon intensity of cluster electricity mix, as weighted average

for name, group in el_footprints.groupby(['Cluster']):
    a = (group['Consumption mix intensity'].mul(group['Total production'])).sum()
    b = group['Total production'].sum()
    cluster_footprints.loc[name] = a.div(b)

cluster_footprints.T.plot(cmap=cmap)
plt.ylabel('Carbon intensity consumption mix \n (weighted average, g CO2/kWh)')
plt.show()

#%%
new_mixes = new_mixes.join(europe_shapes.set_index('ISO_A2')['Cluster'], on='country')
groupby_new_mixes = new_mixes.set_index('Cluster',append=True).groupby(['Cluster', 'technology']).sum()
groupby_new_mixes.to_csv('new_mixes.csv')

#%%
# prepare cluster footprints for feeding into GAMS
cluster_footprints.dropna(axis=1, how='all', inplace=True)
cluster_footprints.index = ['LOW', 'II', 'MID', 'IV', 'HIGH']
cluster_footprints[''] = 'ELC'
cluster_footprints = cluster_footprints.set_index('', append=True)
cluster_footprints = cluster_footprints/1000
cluster_footprints.to_csv(os.path.join(os.path.curdir, 'Data', 'el_footprints_pathways.csv'))
# el_footprints.to_csv(os.path.join(os.path.curdir, 'Data', 'el_footprints_pathways.csv'))

