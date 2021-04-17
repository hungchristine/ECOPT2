# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 13:30:42 2021

@author: chrishun
"""
from dataclasses import dataclass, field, fields
from typing import List, Dict, Any, Union
from itertools import product

import warnings
import pandas as pd
import numpy as np
from scipy.stats import norm
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
    grdeq: list = field(default_factory=lambda: ['IND', 'ALL'])
    veheq: list = field(default_factory=lambda: ['PROD_EINT', 'PROD_CINT_CSNT', 'OPER_EINT', 'EOLT_CINT'])
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
class RawDataClass:
    all_pkm_scen: pd.DataFrame = None
    batt_portfolio: pd.DataFrame = None
    veh_factors: pd.DataFrame = None
    B_term_prod: Any = None
    B_term_oper_EOL: Any = None
    r_term_factors: Any = None
    u_term_factors: Any = None
    enr_glf_terms: Any = None

    veh_avg_age: Union[float, dict] = 11.1 # From ACEA 2019-2020 report
    veh_age_stdev: Union[float, dict] = 2.21

    bev_int_shr: float = 0.02
    # seg_batt_caps: Union[dict, list] = field(default_factory=lambda:{'A': 26.6, 'B': 42.2, 'C': 59.9, 'D': 75., 'E':95., 'F':100.})
    pkm_scenario: str = None

    recycle_rate: float = 0.75
    # other parameters
    occupancy_rate: float = 2
    eur_batt_share: float = None
    tec_add_gradient: float = 0.2

    # enr_cint_src: str = os.path.join('Data','load_data','el_footprints_pathways.csv')

    @classmethod
    def from_dict(cls, src_dict:dict):
        return cls(**src_dict)

    # move the operations from fleet_model here; calculation of veh_partab, glf terms, etc etc


@dataclass
class ParametersClass:
    # enr_partab: pd.DataFrame
    # enr_cint: pd.DataFrame
    veh_stck_tot: pd.DataFrame
    enr_veh: pd.DataFrame
    # veh_prod_eint: pd.DataFrame
    # veh_prod_cint_csnt: pd.DataFrame
    # veh_oper_eint: pd.DataFrame
    # veh_eolt_cint: pd.DataFrame

    # veh_oper_dist: pd.DataFrame
    veh_pay: pd.DataFrame
    # veh_stck_int_tec: pd.DataFrame
    # veh_stck_int: pd.DataFrame
    # stock_tot: pd.DataFrame

    # constraints
    manuf_cnstrnt: pd.DataFrame

    mat_content: Union[list, pd.DataFrame]
    virg_mat_supply: pd.DataFrame
    mat_cint: Union[list, pd.DataFrame]
    veh_add_grd: Union[float, pd.DataFrame] = None


    # gro_cnstrnt: Any

    # bev_capac: Any
    # veh_lift_cdf: Any
    # veh_lift_pdf: Any
    # veh_lift_age: Any
    # veh_lift_mor: Any
    # veh_partab: Any
    # veh_stck_int_seg: Any

    # veh_stck_int_tec: Any
    # "year_par"

    # recovery_pct: Any

    enr_cint: pd.DataFrame = None

    raw_data: RawDataClass = None

    veh_oper_dist: Union[float, list, dict, pd.DataFrame] = 10000
    veh_stck_int_seg: list = field(default_factory=lambda:[0.08, 0.21, 0.27, 0.08, 0.03, 0.34])

    bev_capac: Union[dict, list] = field(default_factory=lambda:{'A': 26.6, 'B': 42.2, 'C': 59.9, 'D': 75., 'E':95., 'F':100.})

    def __post_init__(self):
        if isinstance(self.bev_capac, dict):
            # convert battery capacities to float
            self.bev_capac = {key: float(value) for key, value in self.bev_capac.items()}
        else:
            self.bev_capac = [float(value) for value in self.bev_capac]

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
                   'enr_glf_terms':['enr', 'reg', 'enreq'],
                   'virg_mat_supply': ['mat_cat', 'mat_prod'],
                   'mat_cint': ['mat_cat', 'mat_prod'],
                   'batt_portfolio':['seg', 'battery size'],
                   'veh_factors': ['veheq', 'tec', 'seg'],
                   'enr_veh': ['enr', 'tec'],
                   'veh_pay': ['cohort', 'age', 'year'],
                   'enr_cint': ['fleetreg', 'enr']
                   }
        # read parameter values in from Excel
        params_dict = {}
        raw_data_dict = {}
        # by default, skip first row, assume single index/column headers
        # MultiIndex cases fixed later in FleetModel object
        all_params = pd.read_excel(filepath, sheet_name=None, skiprows=[0], header=[0])
        param_attrs = [f.name for f in fields(cls)]
        for param, value in all_params.items():
            if (param != 'readme') and (not param.startswith('_')):
                # if param.lower() in param_attrs:
                if param.lower() in mi_dict.keys():
                    # special case: MultiIndex index
                    # fill in nan values for columns that will be the index
                    value.loc(axis=1)[mi_dict[param.lower()]] = value.loc(axis=1)[mi_dict[param.lower()]].fillna(method='ffill')
                    value = value.astype({cat: str for cat in mi_dict[param.lower()]})
                    value.set_index(mi_dict[param.lower()], inplace=True, drop=True)
                    # params_dict[param.lower()] = value  # ensure case-insensitivity; FleetModel uses all lower case
                elif value.shape[0] == 0:
                    print(f"Empty values for {param} in Excel file.")
                elif value.shape == (1,2):
                    value = value.iloc[0,1]
                else:
                    # single-level Index
                    value.set_index(value.iloc(axis=1)[0].name, inplace=True, drop=True)
                    value.index = value.index.astype(str)  # ensure all indices are strings (required to work for GAMS)
                    # params_dict[param.lower()] = value  # ensure case-insensitivity; FleetModel uses all lower case
                if param.lower() in param_attrs:
                    params_dict[param.lower()] = value
                    # if value.shape == (1,2):
                    # # special case: scalar value
                    #     params_dict[param.lower()] = value.iloc[0,1]  # special case of scalars in Excel file
                    # else:
                    #     params_dict[param.lower()] = value
                else:
                    raw_data_dict[param.lower()] = value
                    # if value.shape == (1,2):
                    # # special case: scalar value
                    #     raw_data_dict[param.lower()] = value.iloc[0,1]  # special case of scalars in Excel file
                    # else:
                    #     raw_data_dict[param.lower()] = value

        # add user-specified values in experiment dict to params_dict and raw_data_dict
        for exp_param, value in experiment.items():
            # separate ready-to-go parameters from intermediate data
            if exp_param in param_attrs:
                params_dict[exp_param] = value
            else:
                raw_data_dict[exp_param] = value

        # params_dict.update(experiment)  # add experiment values and override duplicate entries with experiment values
        my_obj = cls(**params_dict)
        my_obj.raw_data = RawDataClass.from_dict(raw_data_dict)

        return my_obj

    def calculate_from_raw_data(self, sets):
        attrs = dir(self.raw_data)

        # self.build_veh_pay()  # establish self.veh_pay

        if (self.raw_data.B_term_prod) and (self.raw_data.B_term_oper_EOL) and (self.raw_data.r_term_factors) and (self.raw_data.u_term_factors):
            # TODO: check either the above terms exist, or that there is already a veh_partab in the excel file
            # B_term_prod
            # B_term_oper_EOL
            # r_term_factors
            # u_term_factors
            # self.build_BEV()
            # self.veh_partab = self.raw_data.veh_factors.copy()

            self.veh_partab = self.build_veh_partab()

        if self.raw_data.eur_batt_share:
            # multiply manufacturing constraint, critical material supply by eur_batt_share
            self.virg_mat_supply *= self.raw_data.eur_batt_share
            self.manuf_cnstrnt *= self.raw_data.eur_batt_share

        if isinstance(self.veh_oper_dist, float) or isinstance(self.veh_oper_dist, int):
            # calculate veh_oper_dist
            # TODO: add check for veh_oper_dist OR and(pkm_scenario, all_scenarios)
            self.veh_oper_dist = pd.Series([self.veh_oper_dist for i in range(len(sets.modelyear))], index=sets.modelyear)
            self.veh_oper_dist.index.name = 'year'

        if self.raw_data.pkm_scenario:
            # calculate veh oper dist
            self.raw_data.passenger_demand = self.raw_data.all_pkm_scen.T[self.raw_data.pkm_scenario]
            self.raw_data.passenger_demand.reset_index()
            self.raw_data.passenger_demand *= 1e9
            self.raw_data.passenger_demand.name = ''

            self.raw_data.fleet_vkm  = self.raw_data.passenger_demand / self.raw_data.occupancy_rate

        if self.raw_data.recycle_rate is not None:
            self.recovery_pct = [[self.raw_data.recycle_rate]*len(sets.mat_cats) for year in range(len(sets.modelyear))]
            self.recovery_pct = pd.DataFrame(self.recovery_pct,
                                             index=sets.modelyear,
                                             columns=sets.mat_cats)
        if self.raw_data.enr_glf_terms is not None:
            if self.enr_cint:
                # for building enr_cint from IAM pathways
                self.build_enr_cint()
            else:
                mi = pd.MultiIndex.from_product([sets.reg, sets.enr, sets.modelyear], names=['reg', 'enr', 'modelyear'])
                self.enr_cint = pd.Series(index=mi)
            # NB: we only need enr_cint from electricity pathways, OR we can build enr_cint from enr_partab.
            # TODO: run a check for which one to do in post_init

            # complete enr_cint parameter with fossil fuel chain and electricity in production regions
            # using terms for general logisitic function
            for label, row in self.raw_data.enr_glf_terms.iterrows():
                A = row['A']
                B = row['B']
                r = row['r']
                u = row['u']
                reg = label[1]
                enr = label[0]
                for t in [((2000)+i) for i in range(81)]:
                    self.enr_cint.loc[(reg, enr, str(t))] = A + (B - A) / (1 + np.exp(- r*(t - u)))

            self.enr_cint = self.enr_cint.swaplevel(0, 1) # enr, reg, year"
            self.enr_cint = self.enr_cint.to_frame()
            self.enr_cint.dropna(how='all', axis=0, inplace=True)

        if isinstance(self.raw_data.tec_add_gradient, float) and (self.veh_add_grd is None):
            self.veh_add_grd = {}
            for element in product(*[sets.grdeq, sets.tecs]):
                self.veh_add_grd[element] = self.raw_data.tec_add_gradient

        self.virg_mat_supply = self.virg_mat_supply.T
        self.veh_stck_int_tec = pd.Series([1-self.raw_data.bev_int_shr, self.raw_data.bev_int_shr], index=['ICE', 'BEV'])  # TODO: generalize
        self.year_par = pd.Series([float(i) for i in sets.cohort], index=sets.cohort)
        self.calc_veh_liftime(sets)


    def build_veh_pay(self):
        # get this from parameters.py
        # self.veh_pay = stuff
        pass


    def build_BEV(self):
        # build vehicle impacts table from batt_portfolio
        self.raw_data.batt_portfolio = self.raw_data.batt_portfolio.T
        self.raw_data.prod_df = pd.DataFrame()

        # assemble production emissions for battery for defined battery capacities
        for key, value in self.bev_capac.items():
            self.raw_data.prod_df[key] = self.raw_data.batt_portfolio[key, str(value)]
        mi = pd.MultiIndex.from_product([self.raw_data.prod_df.index.to_list(), ['BEV'], ['batt']])
        self.raw_data.prod_df.index = mi
        self.raw_data.prod_df = self.raw_data.prod_df.stack()
        self.raw_data.prod_df.index.names = ['veheq', 'tec', 'comp', 'seg']
        self.raw_data.prod_df.index = self.raw_data.prod_df.index.swaplevel(i=-2, j=-1)
        self.raw_data.prod_df.drop('battery weight', axis=0, inplace=True)  # remove (currently)


    def build_veh_partab(self):
        """
        Build VEH_PARTAB parameter containing sigmoid function terms.

        Fetch current (A-term) data from Excel spreadsheet and battery
        DataFrame. Upper asymptote (B term values) for production and EOL
        phases and all inflection (r-term) and slope (u-term)
        from YAML file (experiment parameter). Aggregate all values in a
        DataFrame for export to GAMS workspace.

        Parameters
        ----------
        B_term_prod : float
            Upper asymptote for production emissions; expressed
            as a multiple of A-term.
        B_term_oper_EOL : float
            Upper asymptote for operaion and EOL emissions; expressed
            as a multiple of A-term..
        r_term_factors : dict of {str: float}
            Growth rate term.
        u_term_factors : dict of {str: float}
            Inflection point term.

        Returns
        -------
        None.
        """

        # TO DO: separate A-terms for battery and rest-of-vehicle and apply different b-factors

        # Read sigmoid A terms from Excel
        "self.veh_factors = pd.read_excel(self.import_fp, sheet_name='genlogfunc', header=[0], index_col=[0,1,2], usecols='A:F', nrows=48)"
        "self.veh_factors.set_index(['veheq', 'tec', 'seg'])"
        self.raw_data.veh_factors.columns.names = ['comp']
        self.raw_data.veh_factors = self.raw_data.veh_factors.stack().to_frame('a')

        # Retrieve production emission factors for chosen battery capacities and place in raw A factors (with component resolution)
        self.build_BEV()  # update self.prod_df with selected battery capacities
        for index, value in self.raw_data.prod_df.iteritems():
            self.raw_data.veh_factors.loc[index, 'a'] = value

        # Get input for B-multiplication factors (relative to A) from YAML file
        reform = {(firstKey, secondKey, thirdKey): values for firstKey, secondDict in self.raw_data.B_term_prod.items() for secondKey, thirdDict in secondDict.items() for thirdKey, values in thirdDict.items()}
        mi = pd.MultiIndex.from_tuples(reform.keys())
        temp_prod_df = pd.DataFrame()
        temp_oper_df = pd.DataFrame()
        temp_df = pd.DataFrame()

        b_prod = pd.DataFrame(reform.values(), index=mi)
        b_prod.index.names = ['veheq', 'tec', 'comp']

        # Apply B-multiplication factors to production A-factors (with component resolution)
        temp_a = self.raw_data.veh_factors.join(b_prod, on=['veheq', 'tec', 'comp'], how='left')
        temp_prod_df['B'] = temp_a['a'] * temp_a[0]
        temp_prod_df.dropna(how='any', axis=0, inplace=True)

        reform = {(firstKey, secondKey): values for firstKey, secondDict in self.raw_data.B_term_oper_EOL.items() for secondKey, values in secondDict.items()}
        mi = pd.MultiIndex.from_tuples(reform.keys())
        b_oper = pd.DataFrame(reform.values(), index=mi, columns=['b'])

        # Apply B-multiplication factors for operation and EOL A-factors
        temp_oper_df = self.raw_data.veh_factors.join(b_oper, on=['veheq', 'tec'], how='left')
        temp_oper_df['B'] = temp_oper_df['a'] * temp_oper_df['b']
        temp_oper_df.dropna(how='any', axis=0, inplace=True)
        temp_oper_df.drop(columns=['a', 'b'], inplace=True)


        # Aggregate component A values for VEH_PARTAB parameter
        A = self.raw_data.veh_factors.sum(axis=1)
        A = A.unstack(['comp']).sum(axis=1)
        A.columns = ['A']

        # Begin building final VEH_PARTAB parameter table
        temp_df['A'] = A
        B = pd.concat([temp_prod_df, temp_oper_df], axis=0).dropna(how='any', axis=1)
        B = B.unstack(['comp']).sum(axis=1)
        temp_df['B'] = B

        # Add same r values across all technologies...can add BEV vs ICE resolution here
        temp_r = pd.DataFrame.from_dict(self.raw_data.r_term_factors, orient='index', columns=['r'])
        temp_df = temp_df.join(temp_r, on=['tec'], how='left')
        #        self.temp_df['r'] = r_term_factors

        # Add technology-specific u values
        temp_u = pd.DataFrame.from_dict(self.raw_data.u_term_factors, orient='index', columns=['u'])
        temp_df = temp_df.join(temp_u, on=['tec'], how='left')

        #        self.temp_df.drop(labels=0,axis=1,inplace=True)
        #        self.temp_df.index.names=[None,None,None]
        return temp_df


    def calc_veh_liftime(self, sets):
        # TODO: introduce Weibull survival curve estimation?
        self.veh_lift_cdf = pd.Series(norm.cdf(sets.age_int, self.raw_data.veh_avg_age, self.raw_data.veh_age_stdev), index=sets.age)
        self.veh_lift_cdf.index = self.veh_lift_cdf.index.astype('str')

        self.veh_lift_age = pd.Series(1 - self.veh_lift_cdf)

        self.veh_lift_pdf = pd.Series(self.calc_steadystate_vehicle_age_distributions(sets.age_int, self.raw_data.veh_avg_age, self.raw_data.veh_age_stdev), index=sets.age)
        self.veh_lift_pdf.index = self.veh_lift_pdf.index.astype('str')

        self.veh_lift_mor = pd.Series(self.calc_probability_of_vehicle_retirement(sets.age_int, self.veh_lift_pdf), index=sets.age)
        self.veh_lift_mor.index = self.veh_lift_mor.index.astype('str')

    def calc_steadystate_vehicle_age_distributions(self, ages, average_expectancy=10.0, standard_dev=3.0):
        """
        Calc a steady-state age distribution consistent with a normal distribution around an average life expectancy

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

        # The total (100%) minus the cumulation of all the cars retired by the time they reach a certain age
        h = 1 - norm.cdf(ages, loc=average_expectancy, scale=standard_dev)

        # Normalize to represent a _fraction_ of the total fleet
        q = h / h.sum()
        return q

    def calc_probability_of_vehicle_retirement(self, ages, age_distribution):
        """
        Calculate probability of any given car dying during the year, depending on its age.

        This probability is calculated from the age distribution of a population,
        that is assumed to be and to have been at     steady state

        This is only valid if we can assume that the population is at steady
        state.  If in doubt, it is probably best to rely on some idealized
        population distribution, such as the one calculated by
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

        Given an age distribution consistent with an average life expectancy of
        10 years (SD 3 years), we get the following

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

    def build_enr_cint(self):
        # electricity_clustering.py produces electricity pathways with decade resolution
        # this method interpolates between decades for an annual resolution
        self.enr_cint.columns = self.enr_cint.columns.astype('int64')
        # insert year headers
        for decade in self.enr_cint.columns:
            for i in np.arange(1, 10):
                self.enr_cint[decade + i] = np.nan
        self.enr_cint[2019] = self.enr_cint[2020]  # TODO: fill with historical data for pre-2020 period
        # interpolate between decades to get annual resolution
        self.enr_cint = self.enr_cint.astype('float64').sort_index(axis=1).interpolate(axis=1) # sort year orders and interpolate
        self.enr_cint.columns = self.enr_cint.columns.astype(str)  # set to strings for compatibility with GAMS

        self.enr_cint = self.enr_cint.stack()  # reg, enr, year

    def validate_data(self, sets):
        if isinstance(self.veh_stck_int_seg, list):
            self.veh_stck_int_seg = {seg: share for seg, share in zip(sets.seg, self.veh_stck_int_seg)}

        if (self.veh_oper_dist is not None) and (self.raw_data.pkm_scenario is not None):
            warnings.warn('Vehicle operating distance over specified. Both an annual vehicle mileage and an IAM scenario are specified.')

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