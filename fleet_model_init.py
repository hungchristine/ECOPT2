# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 13:30:42 2021

@author: chrishun
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union
from itertools import product

import pandas as pd
import yaml

@dataclass
class SetsClass:
    tecs: list = field(default_factory=lambda:['BEV', 'ICEV'])
    enr: list = field(default_factory=lambda: ['FOS', 'ELC'])
    seg: list = field(default_factory=lambda: ['A', 'C', 'F'])
    mat_cats: list = field(default_factory=lambda: ['Li', 'Co'])
    # default 2 suppliers per material category
    mat_prod: dict = field(default_factory=lambda: {mat: [mat+str(i) for i in range(1,3)] for mat in ['Li', 'Co']})
    reg: list = field(default_factory=lambda: ['LOW', 'HIGH', 'PROD'])
    fleetreg: list = field(default_factory=lambda: ['LOW', 'HIGH'])
    year: list = field(default_factory=lambda: [str(i) for i in range(2000-28, 2081)])
    cohort: list = field(default_factory=lambda: [str(i) for i in range(2000-28, 2081)])
    inityear: list = field(default_factory=lambda: [str(i) for i in range(2000, 2021)])
    optyear: list = field(default_factory=lambda: [str(i) for i in range(2020, 2081)])
    modelyear: list = field(default_factory=lambda: [str(i) for i in range(2000, 2081)])
    age: list = field(default_factory=lambda: [str(i) for i in range(29)])
    age_int: list = field(default_factory=lambda: [i for i in range(29)])

    new: list = field(default_factory=lambda: ['0'])
    demeq: list = field(default_factory=lambda: ['STCK_TOT', 'OPER_DIST', 'OCUP'])
    dstvar: list = field(default_factory=lambda: ['mean', 'stdv'])
    enreq: list = field(default_factory=lambda: ['CINT'])
    grdeq: list = field(default_factory=lambda: ['IND', 'ALL'])
    veheq: list = field(default_factory=lambda: ['PROD_EINT', 'PROD_CINT_CSNT', 'OPER_EINT', 'EOLT_CINT'])
    lfteq: list = field(default_factory=lambda: ['LFT_DISTR', 'AGE_DISTR'])
    sigvar: list = field(default_factory=lambda: ['A', 'B', 'r', 'u'])

    @classmethod
    def from_file(cls, filepath):
        if filepath.endswith('xlsx') or filepath.endswith('.xls'):
            return cls.from_excel(filepath)
        elif filepath.endswith('yml') or filepath.endswith('.yaml'):
            return cls.from_yaml(filepath)
        else:
            raise ValueError("invalid filetype. only excel or yaml accepted. you suck")

    @classmethod
    def from_yaml(cls, filepath):
        print('do_stuff')

    @classmethod
    def from_excel(cls, filepath, sheet=0):
        set_list = ['tecs', 'enr', 'seg', 'mat_cats', 'reg', 'fleetreg',
                    'year', 'modelyear', 'inityear',
                    'cohort', 'optyear', 'age']

        all_sets = pd.read_excel(filepath, sheet, dtype='str')
        all_sets.columns = all_sets.columns.str.lower()

        # Check all mandatory sets are present
        err = []
        for s in set_list:
            if s not in all_sets.columns:
                err.append(s)
        if len(err):
            #TODO: below raises this error: TypeError: exceptions must derive from BaseException
            raise(f'Set(s) {err} not found in Excel file')

        mat_dict = {}
        sets_dict = {}
        for ind in all_sets.columns:
            if '_prod' in ind:
                key = ind.capitalize().split('_prod')[0]
                mat_dict[key] = all_sets[ind].dropna().to_list()
            else:
                sets_dict[ind] = all_sets[ind].dropna().to_list()
        sets_dict['mat_prod'] = mat_dict
        return cls(**sets_dict)

        # TODO: build check to make sure there are no orphan materials or producer lists in mat_dict

@dataclass
class ParametersClass:
    enr_partab: pd.DataFrame
    # enr_cint: pd.DataFrame
    enr_veh: pd.DataFrame
    # veh_prod_eint: pd.DataFrame
    # veh_prod_cint_csnt: pd.DataFrame
    # veh_oper_eint: pd.DataFrame
    # veh_eolt_cint: pd.DataFrame
    veh_stck_tot: pd.DataFrame
    # veh_oper_dist: pd.DataFrame
    veh_pay: pd.DataFrame
    # veh_stck_int_tec: pd.DataFrame
    # veh_stck_int: pd.DataFrame
    # stock_tot: pd.DataFrame
    all_pkm_scen: pd.DataFrame
    batt_portfolio: pd.DataFrame
    glf_terms: pd.DataFrame
    B_term_prod: Any
    B_term_oper_EOL: Any
    r_term_factors: Any
    u_term_factors: Any

    # constraints
    manuf_cnstrnt: pd.DataFrame

    mat_content: pd.DataFrame
    virg_mat_supply: pd.DataFrame
    mat_cint: pd.DataFrame

    veh_oper_dist: int = 10000
    veh_stck_int_seg: list = field(default_factory=lambda:[0.08, 0.21, 0.27, 0.08, 0.03, 0.34])
    bev_int_shr: float = 0.02
    seg_batt_caps:dict = field(default_factory=lambda:{'A': 26.6, 'B': 42.2, 'C': 59.9, 'D': 75, 'E':95, 'F':100})
    pkm_scenario: str = 'iTEM2-Base'

    exogeneous: ExogeneousClass

    # constraints
    tec_add_gradient: float = 0.2
    eur_batt_share: float = 0.4
    recycle_rate: float = 0.75
    # other parameters
    occupancy_rate: float = 2

    @classmethod
    def from_exp(cls, experiment:dict):
        return cls(**experiment)

    @classmethod
    def from_file(cls, filepath: str, experiment: dict = None):
        if filepath.endswith('xlsx') or filepath.endswith('.xls'):
            return cls.from_excel(filepath, experiment)
        elif filepath.endswith('yml') or filepath.endswith('.yaml'):
            return cls.from_yaml(filepath, experiment)
        else:
            raise ValueError("Invalid filetype for ParametersClass. Only Excel or yaml accepted.")

    @classmethod
    def from_yaml(cls, filepath, experiment):
        with open(filepath, 'r') as stream:
            try:
                params = yaml.safe_load(stream)
                print('Finished reading parameter values from yaml')
            except yaml.YAMLError as exc:
                print(exc)

        params_dict = {}  # dict with parameter names key values, and dict of experiments as values
        for key, item in params.items():
            params_dict[key] = item

        params_dict.update(experiment)  # override duplicate entries with experiment values
        return cls(**params_dict)

        # all_exp_list = []  # list of all exp_dicts
        # exp_dict = {}  # dict containing current experiment parameters
        # param_vals = [params_dict[p].items() for p in params_dict.keys()]  # list containing all parameter experiment values (for Cartesian product)
        # exp_id_list = []  # list of experiment IDs

        # # create all experiments as Cartesian product of all parameter options, and iterate
        # for i, experiment in enumerate(product(*param_vals)):
        #     id_string = "run"
        #     for key, exp in zip(params_dict.keys(), experiment):
        #         # build dict describing each dictionary: {parameter: experiment value}
        #         exp_dict[key] = exp[1]
        #         id_string = id_string + "_" + exp[0]
        #     all_exp_list.append(exp_dict.copy())
        #     exp_id_list.append(id_string)
        # print(all_exp_list)
        # return cls(**exp_dict)
        # return cls(**params_dict)

    @classmethod
    def from_excel(cls, filepath, experiment):
        mi_dict = {
                   'enr_partab':['enr', 'reg', 'enreq'],
                   'virg_mat_supply': ['mat_cat', 'mat_prod'],
                   'mat_cint': ['mat_cat', 'mat_prod'],
                   'batt_portfolio':['seg', 'battery size'],
                   'glf_terms': ['veheq', 'tec', 'seg'],
                   'enr_veh': ['enr', 'tec'],
                   'veh_pay': ['cohort', 'age', 'year']
                   }
        # read parameter values in from Excel
        params_dict = {}
        # by default, skip first row, assume single index/column headers
        # MultiIndex cases fixed later in FleetModel object
        all_params = pd.read_excel(filepath, sheet_name=None, skiprows=[0], header=[0])
        for param, value in all_params.items():
            if (param != 'readme') and (not param.startswith('_')):
                if value.shape == (1,2):
                    # special case: scalar value
                    params_dict[param.lower()] = value.iloc[0,1]  # special case of scalars
                elif param.lower() in mi_dict.keys():
                    # special case: MultiIndex index
                    # fill in nan values for columns that will be the index
                    value.loc(axis=1)[mi_dict[param.lower()]] = value.loc(axis=1)[mi_dict[param.lower()]].fillna(method='ffill')
                    value = value.astype({cat: str for cat in mi_dict[param.lower()]})
                    value.set_index(mi_dict[param.lower()], inplace=True, drop=True)
                    params_dict[param.lower()] = value  # ensure case-insensitivity; FleetModel uses all lower case
                elif value.shape[0] == 0:
                    print(f"Empty values for {param} in Excel file.")
                else:
                    # single-level Index
                    value.set_index(value.iloc(axis=1)[0].name , inplace=True, drop=True)
                    value.index = value.index.map(str)  # ensure all indices are strings (required to work for GAMS)
                    params_dict[param.lower()] = value  # ensure case-insensitivity; FleetModel uses all lower case


        params_dict.update(experiment)  # add experiment values and override duplicate entries with experiment values
        my_obj = cls(**params_dict)
        my_obj.calculate_stuff_from_exogeneous_data()
        return my_obj

@dataclass
class VariablesClass:
    answer1: list = None
    answer2: list = None


# p = ParametersClass.from_file(r"C:\Users\chrishun\Box Sync\YSSP_temp\GAMS_input_test.yml")
# pp = ParametersClass.from_exp({'veh_stck_int_seg': [0.08, 0.21, 0.26, 0.08, 0.03, 0.34], 'tec_add_gradient': 1.2, 'seg_batt_caps': {'A': 17.6, 'B': 42.2, 'C': 42.2, 'D': 59.9, 'E': 75, 'F': 95}})
#######
# experiment = {'veh_stck_int_seg': [0.08, 0.21, 0.26, 0.08, 0.03, 0.34],
#               # 'tec_add_gradient': 1.2,
#               'seg_batt_caps': {'A': 17.6, 'B': 42.2, 'C': 42.2, 'D': 59.9, 'E': 75, 'F': 95}}
# p = ParametersClass.from_file(r"C:\Users\chrishun\Box Sync\YSSP_temp\GAMS_input_demo.xls", experiment)