# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 13:30:42 2021

@author: chrishun
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any

import pandas as pd

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
    param1: int = 1

@dataclass
class VariablesClass:
    answer1: list = None
    answer2: list = None