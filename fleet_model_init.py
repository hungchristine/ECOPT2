# -*- coding: utf-8 -*-

"""
Defines three dataclasses: SetsClass, ParametersClass and RawDataClass.

The SetsClass and ParametersClass contain the sets and parameters
forwarded to GAMS, respectively. RawDataClass contains the
intermediate data used to calculate parameters.
"""

from dataclasses import dataclass, field, fields
from typing import List, Dict, Union
from itertools import product

import logging
import pandas as pd
import numpy as np
from scipy.stats import norm, weibull_min
import yaml


log = logging.getLogger(__name__)

@dataclass
class SetsClass:
    """Contains values for all GAMS sets."""

    """ Default values initialize with a two-region system with three
    size segments and two critical material classes each with two producers."""

    imp: List = field(default_factory=lambda: ['nrg','GHG', 'AP', 'MTP'])
    optimp:  List = field(default_factory=lambda: ['GHG'])
    imp_cat: List = field(default_factory=lambda: ['GHG', 'AP', 'MTP'])
    imp_int: List = field(default_factory=lambda: ['nrg','GHG', 'AP', 'MTP'])
    tec: List = field(default_factory=lambda: ['BEV', 'ICEV'])
    enr: List = field(default_factory=lambda: ['FOS', 'ELC'])
    seg: List = field(default_factory=lambda: ['A', 'C', 'F'])
    mat_cat: List = field(default_factory=lambda: ['Li', 'Co'])
    mat_prod: Dict = field(default_factory=lambda: {mat: [mat+str(i) for i in range(1,3)] for mat in ['Li', 'Co']})
    reg: List = field(default_factory=lambda: ['LOW', 'HIGH', 'PROD'])
    fleetreg: List = field(default_factory=lambda: ['LOW', 'HIGH'])
    year: List = field(default_factory=lambda: [str(i) for i in range(2000-28, 2081)])
    cohort: List = field(default_factory=lambda: [str(i) for i in range(2000-28, 2081)])
    inityear: List = field(default_factory=lambda: [str(i) for i in range(2000, 2021)])
    optyear: List = field(default_factory=lambda: [str(i) for i in range(2020, 2081)])
    modelyear: List = field(default_factory=lambda: [str(i) for i in range(2000, 2081)])
    age: List = field(default_factory=lambda: [str(i) for i in range(29)])
    age_int: List = field(default_factory=lambda: [i for i in range(29)])

    new: List = field(default_factory=lambda: ['0'])  # static set for new vehicles
    newtec: List = field(default_factory=lambda: ['BEV'])
    demeq: List = field(default_factory=lambda: ['STCK_TOT', 'OPER_DIST', 'OCUP'])
    grdeq: List = field(default_factory=lambda: ['IND', 'ALL'])
    veheq: List = field(default_factory=lambda: ['PROD_EINT', 'PROD_CINT_CSNT', 'OPER_EINT', 'EOLT_CINT'])
    sigvar: List = field(default_factory=lambda: ['A', 'B', 'r', 'u'])
    lcphase: List = field(default_factory=lambda: ['prod', 'oper', 'eol'])
    # imp_int: List = field(default_factory=lambda: ['nrg', 'GHG', 'AP', 'MTP'])


    @classmethod
    def from_file(cls, filepath):
        """
        Create SetsClass with user input file.

        Distinguishes between Excel or YAML input files, and raises an error
        with other filetypes

        Parameters
        ----------
        filepath : str
            Filepath for user-defined set and index declerations.

        Raises
        ------
        ValueError
            Raises error on invalid (non-Excel or -YAML) filetypes.

        Returns
        -------
        SetsClass
            Initialized SetsClass from file.
        """
        if filepath.endswith('xlsx') or filepath.endswith('.xls'):
            return cls.from_excel(filepath)
        elif filepath.endswith('yml') or filepath.endswith('.yaml'):
            return cls.from_yaml(filepath)
        else:
            log.error('Could not load set values. Invalid filetype. Only Excel or YAML formats accepted.')
            raise ValueError("Invalid filetype. Only Excel or YAML formats accepted.")

        log.info('Finished initializing sets')

    @classmethod
    def from_yaml(cls, filepath):
        """Read user-defined sets and indices from YAML file.

        Sets and their indices are to be saved as dicts in YAML.
        Dict keys must match SetsClass field names.

        Parameters
        ----------
        filepath : str
            Filepath to YAML file containing user-defined sets and indices as dicts.


        Returns
        -------
        SetsClass
            Initialized SetsClass from YAML file.
        """

        with open(filepath, 'r') as stream:
            try:
                sets = yaml.safe_load(stream)
                return cls(**sets)
            except Exception as e:
                log.error(f'Could not load sets from YAML. \n{e}')

    @classmethod
    def from_excel(cls, filepath, sheet=0):
        """Read user-defined sets and indices from Excel Worksheet.

        The first row of the sheet is assumed to be the set names, which must
        match the SetsClass field names, with the valid index values below

        Parameters
        ----------
        filepath : str
            Filepath to Excel workbook containing user-defined sets and indices.
        sheet : str, optional
            Sheetname in Excel workbook to read sets and indices from.
            The default is 0 (first worksheet).

        Returns
        -------
        SetsClass
            Initialized SetsClass from Excel file.
        """

        # list of sets required in Excel file for initialization
        set_list = ['tec', 'newtec', 'enr', 'seg', 'mat_cat',
                    'reg', 'fleetreg',
                    'year', 'modelyear', 'inityear',
                    'cohort', 'optyear', 'age']

        # DataFrame containing all user-defined sets
        all_sets = pd.read_excel(filepath, sheet, dtype='str')
        all_sets.columns = all_sets.columns.str.lower()

        # Check all mandatory sets are present
        err = []
        for s in set_list:
            if s not in all_sets.columns:
                err.append(s)
        if len(err):
            print('\n *****************************************')
            log.warning(f'Set(s) {err} not found in Excel file')

        # generate dict of sets (remove nan from dataframe due to differing set lengths)
        mat_dict = {}  # special case for materials
        mat_checklist = all_sets['mat_cat'].dropna(how='all', axis=0).values  # used to validate materials
        sets_dict = {}
        for ind in all_sets.columns:
            if '_prod' in ind:
                # manually generate set of primary material producers
                key = ind.capitalize().split('_prod')[0]
                mat_dict[key] = all_sets[ind].dropna().to_list()
                if key in mat_checklist:
                    mat_checklist = mat_checklist[mat_checklist != key]  # remove material from checklist
                else:
                    log.warning(f'Critical material {key} not in mat_cat, but has primary producers')
            else:
                sets_dict[ind] = all_sets[ind].dropna().to_list()
        if len(mat_checklist) > 0:
            log.warning(f'Critical materials {mat_checklist} do not have primary producers defined')
        sets_dict['mat_prod'] = mat_dict

        # validate initialization period and optimization period
        if len(set(sets_dict['inityear']).intersection(set(sets_dict['optyear']))):
            log.warning('---- Overlap in inityear and optyear members. May result in infeasibility.')

        return cls(**sets_dict)


@dataclass
class RawDataClass:
    """Contains any raw data used to further calculate or construct parameters.

    Converts e.g., dicts and floats to timeseries in pd.Series format
    """

    #TODO: move the operations from fleet_model here; calculation of tec_parameters, glf terms, etc etc

    pkm_scenario: str = None
    all_pkm_scen: pd.DataFrame = None
    veh_pkm: pd.DataFrame = None
    fleet_vkm: pd.DataFrame = None
    batt_portfolio: pd.DataFrame = None
    tec_parameters_raw: pd.DataFrame = None
    B_term_prod: Union[dict, float] = None
    B_term_oper_EOL: Union[dict, float] = None
    r_term_factors: Union[dict, pd.Series] = None  # {str: float}
    u_term_factors: Union[dict, pd.Series] = None  # {str: float}
    prod_df: pd.DataFrame = None

    veh_avg_age: Union[float, dict] = 11.1 # From ACEA 2019-2020 report
    veh_age_stdev: Union[float, dict] = 2.21

    newtec_int_shr: float = 0.0018  # from Eurostat; assume remaining is ICE

    recycle_rate: Union[float, pd.Series] = 0.75

    # other parameters
    occupancy_rate: Union[float, pd.Series] = 2.
    eur_batt_share: Union[float, pd.Series] = 1.  # default assumes primary material supply given is for entire region
    tec_add_gradient: Union[float, pd.Series] = 0.2

    enr_glf_terms: pd.Series= None

@dataclass
class ParametersClass:
    """Contains all parameter values for GAMS model (in ready-to-insert form)."""

    exog_tot_stock: Union[pd.Series, pd.DataFrame] = None
    enr_tec_correspondance: Union[pd.Series, pd.DataFrame] = None
    cohort_age_correspondance: Union[pd.Series, pd.DataFrame] = None
    year_par: Union[pd.Series, pd.DataFrame] = None
    tec_parameters: Union[pd.Series, pd.DataFrame] = None

    # constraints
    manuf_cnstrnt: Union[pd.Series, pd.DataFrame] = None

    mat_content: Union[List, pd.Series, pd.DataFrame] = None
    virg_mat_supply: Union[pd.Series, pd.DataFrame] = None
    mat_impact_int: Union[List, pd.Series, pd.DataFrame] = None
    max_uptake_rate: Union[float, pd.Series, pd.DataFrame] = None

    enr_impact_int: Union[pd.Series, pd.DataFrame] = None
    enr_impact_int_IAM: Union[pd.Series, pd.DataFrame] = None # move to rawdataclass?

    raw_data: RawDataClass = None

    veh_oper_dist: Union[float, int, List, Dict, pd.Series, pd.DataFrame] = None
    initial_seg_shares: Union[Dict, List] = field(default_factory=lambda:[0.08, 0.21, 0.27, 0.08, 0.03, 0.34])  # Shares from 2017, ICCT report
    initial_tec_shares: Union[Dict, pd.Series, pd.DataFrame] = None

    bev_capac: Union[Dict, List] = field(default_factory=lambda:{'A': 26.6, 'B': 42.2, 'C': 59.9, 'D': 75., 'E':95., 'F':100.})
    # veh_lift_cdf: Union[pd.Series, pd.DataFrame] = None
    # veh_lift_pdf: Union[pd.Series, pd.DataFrame] = None
    retirement_function: Union[pd.Series, pd.DataFrame] = None
    lifetime_age_distribution: Union[pd.Series, pd.DataFrame] = None

    recovery_pct: Union[float, pd.Series, pd.DataFrame] = None

    def __post_init__(self):
        """
        Post-initialization processing of parameters.

        Returns
        -------
        None.
        """

        # convert battery capacities to float
        if isinstance(self.bev_capac, dict):
            self.bev_capac = {key: float(value) for key, value in self.bev_capac.items()}
        elif not isinstance(self.bev_capac, pd.DataFrame):
            self.bev_capac = [float(value) for value in self.bev_capac]

        # convert DataFrame to correct format (used in initializing FleetModel from .gdx file)
        if isinstance(self.max_uptake_rate, pd.DataFrame) and not isinstance(self.max_uptake_rate.index, pd.MultiIndex):
            tmp = self.max_uptake_rate.stack()
            self.max_uptake_rate = tmp.to_dict()

        if self.exog_tot_stock.index.name == 'fleetreg':
            self.exog_tot_stock = self.exog_tot_stock.T



    @classmethod
    def from_exp(cls, experiment:dict):
        """
        Initialize parameters from experiment dict.

        Parameters
        ----------
        experiment : dict
            Parameter names and values.

        Returns
        -------
        ParametersClass
            Initialized ParametersClass object.
        """

        return cls(**experiment)

    @classmethod
    def from_file(cls, filepath: str, experiment: dict = None):
        """
        Initialize parameters from user-defined file.

        Parameters
        ----------
        filepath : str
            Filepath for user-defined parameter declerations.
        experiment : dict, optional
            Contains experiment-specific parameter values. The default is None.

        Raises
        ------
        ValueError
            Raises error on invalid (non-Excel or -YAML) filetypes.

        Returns
        -------
        ParametersClass
            Initialized collection of parameters.

        """

        if filepath.endswith('xlsx') or filepath.endswith('.xls'):
            return cls.from_excel(filepath, experiment)
        elif filepath.endswith('yml') or filepath.endswith('.yaml'):
            return cls.from_yaml(filepath, experiment)
        else:
            print('\n *****************************************')
            log.error('Invalid filetype for ParametersClass. Only Excel (.xls or .xlsx) or YAML accepted.')
            raise ValueError("Invalid filetype for ParametersClass. Only Excel or yaml accepted.")

    @classmethod
    def from_yaml(cls, filepath, experiment):
        """
        Initialize ParametersClass object from YAML file.

        YAML should contain

        Parameters
        ----------
        filepath : str
            Filepath for user-defined parameter declerations.
        experiment : dict
            Contains all experiment-specific parameter values.

        Returns
        -------
        ParametersClass
            Initialized ParametersClass object.

        """
        with open(filepath, 'r') as stream:
            try:
                params = yaml.safe_load(stream)
                print('Finished reading parameter values from yaml')
            except yaml.YAMLError as exc:
                log.error(f'Error reading in parameters from {filepath}. {exc}')

        params_dict = {}  # dict with parameter names key values, and dict of experiments as values
        for key, item in params.items():
            params_dict[key] = item

        params_dict.update(experiment)  # override duplicate entries with experiment values
        return cls(**params_dict)

    @classmethod
    def from_excel(cls, filepath, experiment):
        """
        Initialize ParametersClass object from Excel file.

        Read all sheets in the provided Excel sheet, performs some level of
        data validation and sets ParametersClass attributes based on sheet
        names (case-insensitive).

        Creates RawData object with data that needs further manipulating.

        Initialize experiment-specific parameter values.

        Parameters
        ----------
        filepath : str
            Filepath for user-defined parameter declerations.
        experiment : dict
            Contains all experiment-specific parameter values.

        Returns
        -------
        my_obj : ParametersClass
            Initialized ParametersClass object.

        """
        mi_dict = {
                   'enr_glf_terms':['imp', 'enr', 'reg'],#, 'enreq'],
                   'virg_mat_supply': ['mat_cat', 'mat_prod'],
                   'mat_impact_int': ['imp', 'mat_cat', 'mat_prod'],
                   'mat_content': ['newtec','mat_cat'],
                   'batt_portfolio':['seg', 'battery size'],
                   'tec_parameters_raw': ['lcphase','imp', 'tec', 'seg'],
                   'enr_tec_correspondance': ['enr', 'tec'],
                   'cohort_age_correspondance': ['year', 'cohort', 'age'],
                   'enr_impact_int_IAM': ['imp', 'reg', 'enr'],
                   'enr_impact_int': ['imp', 'reg', 'enr'],
                   }

        # read parameter values in from Excel
        params_dict = {}
        raw_data_dict = {}

        # by default, skip first row, assume single index/column headers
        # MultiIndex cases fixed later in FleetModel object
        log.info('Loading parameter values from Excel...')
        all_params = pd.read_excel(filepath, sheet_name=None, skiprows=[0], header=[0])
        param_attrs = [f.name for f in fields(cls)]
        for param, value in all_params.items():
            if (param != 'readme') and (not param.startswith('_')):
                if param.lower() in mi_dict.keys():
                    # special case: MultiIndex index
                    # fill in nan values for columns that will be the index
                    value.loc(axis=1)[mi_dict[param.lower()]] = value.loc(axis=1)[mi_dict[param.lower()]].fillna(method='ffill')
                    value = value.astype({cat: str for cat in mi_dict[param.lower()]})  # convert all labels to strings (requirement from GAMS)
                    value.set_index(mi_dict[param.lower()], inplace=True, drop=True)
                elif value.shape[0] == 0:
                    log.info(f"Empty values for {param} in Excel file.")
                elif value.shape == (1,2):
                    value = value.iloc[0,1]
                else:
                    # single-level Index
                    value.set_index(value.iloc(axis=1)[0].name, inplace=True, drop=True) # set first column as index
                    value.index = value.index.astype(str)  # ensure all indices are strings (required to work for GAMS)
                    value.columns = value.columns.astype(str)

                if param.lower() in param_attrs:
                    if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
                        if not value.empty:
                            value = cls.order_sets(value)
                    params_dict[param.lower()] = value
                else:
                    # any sheets in Excel that are not ParameterClass fields are sent to RawDataClass
                    raw_data_dict[param.lower()] = value

        # add user-specified values in experiment dict to params_dict and raw_data_dict
        # add experiment values and override duplicate entries with experiment values
        for exp_param, value in experiment.items():
            # separate ready-to-go parameters from intermediate data
            if exp_param in param_attrs:
                params_dict[exp_param] = value
            else:
                raw_data_dict[exp_param] = value

        my_obj = cls(**params_dict)
        my_obj.raw_data = RawDataClass(**raw_data_dict)

        return my_obj

    def calculate_from_raw_data(self, sets):
        """
        Set up input data structures for parameters from RawData object.

        Set up data structures (DataFrames) and, where necessary, performs
        basic calculations for insertion in GAMS database as model input.

        Parameters
        ----------
        sets : SetsClass
            For setting up DataFrame indices.

        Returns
        -------
        None.
        """

        attrs = dir(self.raw_data)

        self.exog_tot_stock = self.interpolate_years(self.exog_tot_stock, sets, axis=0)

        if (self.veh_oper_dist is not None) and ((self.raw_data.veh_pkm is not None) or (self.raw_data.pkm_scenario is not None)):
            log.warning('----- Vehicle operating distance overspecified. Both an annual vehicle mileage and an IAM scenario are specified.')

        if self.cohort_age_correspondance is None:
            self.cohort_age_correspondance = self.build_cohort_age_correspondance(sets)  # establish self.cohort_age_correspondance

        has_required_data = all(attr is not None for attr in (self.raw_data.B_term_prod,
                                                              self.raw_data.B_term_oper_EOL,
                                                              self.raw_data.r_term_factors,
                                                              self.raw_data.u_term_factors
                                                             ))
        if self.tec_parameters is not None and has_required_data:
            log.warning('----- tec_parameters is overdefined')
        elif has_required_data:
            self.tec_parameters = self.build_tec_parameters()

        if self.raw_data.eur_batt_share:
            # multiply manufacturing constraint and critical material supply by eur_batt_share
            # allows e.g., expressing constraints as global values, with a regional market share
            self.virg_mat_supply = self.virg_mat_supply.mul(self.raw_data.eur_batt_share, axis=0)
            self.manuf_cnstrnt = self.manuf_cnstrnt.mul(self.raw_data.eur_batt_share, axis=0)


        # TODO: expand veh_oper_dist to be tec and reg specific (also in GAMS)
        if self.raw_data.veh_pkm:
            self.veh_oper_dist = self.interpolate_years(self.raw_data.veh_pkm, sets).div(self.exog_tot_stock)
        # if self.raw_data.pkm_scenario:
        #     # calculate veh oper dist
        #     self.raw_data.passenger_demand = self.raw_data.all_pkm_scen.T[self.raw_data.pkm_scenario]
        #     self.raw_data.passenger_demand.reset_index()
        #     self.raw_data.passenger_demand *= 1e9
        #     self.raw_data.passenger_demand.name = ''

        #     self.raw_data.fleet_vkm  = self.raw_data.passenger_demand / self.raw_data.occupancy_rate
        #     self.raw_data.fleet_vkm.index = sets.modelyear
        #     self.raw_data.fleet_vkm.index.name = 'year'
        #     self.veh_oper_dist = self.raw_data.fleet_vkm / self.exog_tot_stock.T.sum()  # assumes uniform distribution of annual distance travelled vs vehicle age and region
            if self.veh_oper_dist.mean() > 25e3:
                log.warning('Warning, calculated annual vehicle mileage is above 25000 km, check fleet_km and exog_tot_stock')
            elif self.veh_oper_dist.mean() < 1e4:
                log.warning('Warning, calculated annual vehicle mileage is less than 10000 km, check fleet_km and exog_tot_stock')

        if isinstance(self.veh_oper_dist, (float, int)):
            # calculate veh_oper_dist
            # given a single value for veh_oper_dist, assumes that value applies for every region, year and technology
            ind = pd.MultiIndex.from_product([sets.fleetreg, sets.modelyear])
            self.veh_oper_dist = pd.Series([self.veh_oper_dist for i in range(len(ind))], index=ind)
        elif isinstance(self.veh_oper_dist, dict):
            self.veh_oper_dist = pd.Series(self.veh_oper_dist)
        self.veh_oper_dist.index.names = ['fleetreg','year']

        if self.raw_data.recycle_rate is not None and isinstance(self.raw_data.recycle_rate, float):
            self.recovery_pct = [[self.raw_data.recycle_rate]*len(sets.mat_cat) for year in range(len(sets.modelyear))]
            self.recovery_pct = pd.DataFrame(self.recovery_pct,
                                             index=sets.modelyear,
                                             columns=sets.mat_cat)
            self.recovery_pct = self.recovery_pct.T
        if (self.raw_data.enr_glf_terms is not None) and (self.enr_impact_int is not None or self.enr_impact_int_IAM is not None):
            log.warning('----- Source for energy pathways may be overspecified; both enr_glf_terms and enr_impact_int are specified. Using enr_impact_int.')

        if self.enr_impact_int is not None or self.enr_impact_int_IAM is not None:
            if self.enr_impact_int_IAM is not None:
                self.check_region_sets(self.enr_impact_int_IAM.index.get_level_values('reg'), 'enr_impact_int_IAM', sets.reg)
                # for building enr_impact_int directly from IAM pathways (see electricity_clustering.py)
                self.enr_impact_int_IAM = self.interpolate_years(self.enr_impact_int_IAM, sets)
                self.enr_impact_int_IAM.index = self.enr_impact_int_IAM.index.reorder_levels(['imp', 'enr', 'reg', 'year'])  # match correct set order for enr_impact_int

            if self.enr_impact_int is not None:
                self.check_region_sets(self.enr_impact_int.index.get_level_values('reg'), 'enr_impact_int', sets.reg)
                self.enr_impact_int = self.interpolate_years(self.enr_impact_int, sets)
                self.enr_impact_int.index = self.enr_impact_int.index.reorder_levels(['imp', 'enr', 'reg', 'year'])
                self.enr_impact_int = pd.concat([self.enr_impact_int_IAM, self.enr_impact_int])
        elif self.raw_data.enr_glf_terms is not None:
            self.check_region_sets(self.raw_data.enr_glf_terms.index.get_level_values('reg'), 'enr_glf_terms', sets.reg)
            # build enr_impact_int from generalized logistic function
            mi = pd.MultiIndex.from_product([sets.imp, sets.reg, sets.enr, sets.modelyear], names=['imp', 'reg', 'enr', 'modelyear'])
            self.enr_impact_int = pd.Series(index=mi)


            # complete enr_impact_int parameter with fossil fuel chain and electricity in production regions
            # using terms for general logisitic function
            for label, row in self.raw_data.enr_glf_terms.iterrows():
                A = row['A']
                B = row['B']
                r = row['r']
                u = row['u']
                imp = label[0]
                reg = label[2]
                enr = label[1]
                for t in [((2000)+i) for i in range(81)]:
                    self.enr_impact_int.loc[(reg, enr, str(t))] = A + (B - A) / (1 + np.exp(- r*(t - u)))

            self.enr_impact_int = self.enr_impact_int.swaplevel(0, 1) # enr, reg, year
            self.enr_impact_int = self.enr_impact_int.to_frame()
            self.enr_impact_int.dropna(how='all', axis=0, inplace=True)

        if (isinstance(self.raw_data.tec_add_gradient, float)) and (self.max_uptake_rate is None):
            self.max_uptake_rate = {}
            for element in product(*[sets.grdeq, sets.tec]):
                if element[1] in sets.newtec:
                    self.max_uptake_rate[element] = self.raw_data.tec_add_gradient

        if len(self.virg_mat_supply.columns) in [len(sets.year), len(sets.optyear), len(sets.modelyear)]:
            # if years are in columns, transpose for transferring to GAMS
            self.virg_mat_supply = self.virg_mat_supply.T

        oldtec = list(set(sets.tec) - set(sets.newtec))  # get name of incumbent technology; works for single tec
        if len(sets.newtec) == 1:
            self.initial_tec_shares = pd.Series([1-self.raw_data.newtec_int_shr, self.raw_data.newtec_int_shr], index=oldtec + sets.newtec)
        else:
            if isinstance(self.raw_data.newtec_int_shr, dict):
                all_new_tecs = sum(self.raw_data.newtec_int_shr.values())
                self.initial_tec_shares = pd.Series(self.raw_data.newtec_int_shr)
                self.initial_tec_shares.loc[oldtec] = 1- all_new_tecs
            elif isinstance(self.raw_data.newtec_int_shr, pd.DataFrame) or isinstance(self.raw_data.newtec_int_shr, pd.Series):
                self.initial_tec_shares.loc[oldtec] - 1 - self.initial_tec_shares.sum()

        self.year_par = pd.Series([float(i) for i in sets.year], index=sets.year)
        self.calc_veh_lifetime(sets)


    def build_cohort_age_correspondance(self, sets):
        """
        Build production year-cohort-age concordance matrix for use in GAMS (as parameter).

        Returns
        -------
        index: pd.DataFrame
            Correspondence between current year, cohort and age

        """

        top_year = int(sets.optyear[-1])  # the last year we are interested in
        # start_year = int(sets.inityear[0])
        start_year = int(sets.modelyear[0])  # start with the oldest cohort
        prod_year = start_year - int(sets.age[-1])
        ind = []
        for year in range(start_year, top_year + 1):
            for a in sets.age_int:
                prod_year = year - a - 1
                ind.append([year, prod_year, a, 1])

        index = pd.DataFrame(ind)
        #index = index[index[0]<=2050]
        # index = index[index[2] <= (top_year + 1)] # we don't need tecnologies manufactured past our time period
        for ind, col in index.iloc[:,:-1].iteritems():
            index.loc(axis=1)[ind] = index.loc(axis=1)[ind].astype(str) # convert to strings as required by GAMS
        index.columns = ['year','prodyear','age', 'level']

        return index

    def build_BEV(self):
        """
        Fetch BEV production impacts based on battery size.

        Select battery size by segment from size-segment combinations,
        fetch and sum production impacts for battery and rest-of-vehicle.
        Update DataFrame with total production impacts and energy use for
        BEVs by segment.

        Returns
        -------
        None.

        """
        # build vehicle impacts table from batt_portfolio
        self.raw_data.batt_portfolio.dropna(axis=1, how='all', inplace=True)
        self.raw_data.prod_df = pd.DataFrame()

        # assemble production impacts for battery for defined battery capacities
        for key, value in self.bev_capac.items():
            self.raw_data.prod_df[key] = self.raw_data.batt_portfolio.loc[key, str(value)]
        self.raw_data.prod_df = self.raw_data.prod_df.T.set_index('lcphase', append=True, drop=True)

        self.raw_data.prod_df['tec'] = 'BEV'
        self.raw_data.prod_df['comp'] = 'batt'
        self.raw_data.prod_df.set_index(['tec', 'comp'], append=True, inplace=True)

        self.raw_data.prod_df = self.raw_data.prod_df.stack()
        self.raw_data.prod_df.index.rename(['seg', 'lcphase', 'tec','comp','imp'], inplace=True)
        self.raw_data.prod_df.index =  self.raw_data.prod_df.index.reorder_levels(['lcphase', 'imp', 'tec', 'seg', 'comp'])
        self.raw_data.prod_df.sort_index(inplace=True)
        try:
            self.raw_data.prod_df.drop('battery weight', axis=0, level='imp', inplace=True)  # remove battery weight (currently not implemented)
        except KeyError:
            log.info('Could not drop battery weight')


    def build_tec_parameters(self):
        """
        Build TEC_PARAMETERS parameter containing sigmoid function terms.

        Fetch current (A-term) data from Excel spreadsheet and battery
        DataFrame. Upper asymptote (B term values) for production and EOL
        phases and all inflection (r-term) and slope (u-term)
        from YAML file (experiment parameter). Aggregate all values in a
        DataFrame for export to GAMS database.

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
        # TODO: separate A-terms for battery and rest-of-vehicle and apply different b-factors
        # TODO: allow for series of B-term values

        # Fetch sigmoid A terms from RawDataClass
        self.raw_data.tec_parameters_raw.columns.names = ['comp']
        self.raw_data.tec_parameters_raw = self.raw_data.tec_parameters_raw.stack().to_frame('a')

        # Retrieve production emission factors for chosen battery capacities and place in raw A factors (with component resolution)
        self.build_BEV()  # update self.prod_df with selected battery capacities
        self.raw_data.tec_parameters_raw.sort_index(inplace=True)
        for index, value in self.raw_data.prod_df.iteritems():
            self.raw_data.tec_parameters_raw.loc[index, 'a'] = value

        # Get input for B-multiplication factors (relative to A) from YAML file
        reform = {(firstKey, secondKey, thirdKey): values for firstKey, secondDict in self.raw_data.B_term_prod.items() for secondKey, thirdDict in secondDict.items() for thirdKey, values in thirdDict.items()}
        mi = pd.MultiIndex.from_tuples(reform.keys())
        temp_prod_df = pd.DataFrame()
        temp_oper_df = pd.DataFrame()
        temp_df = pd.DataFrame()

        b_prod = pd.DataFrame(reform.values(), index=mi)
        b_prod = b_prod.stack()
        b_prod.index.names = ['lcphase', 'imp', 'tec', 'comp']
        b_prod.name = 'b'

        # Apply B-multiplication factors to production A-factors (with component resolution)
        temp_a = self.raw_data.tec_parameters_raw.join(b_prod, on=['lcphase', 'imp', 'tec', 'comp'], how='left')
        temp_prod_df['B'] = temp_a['a'] * temp_a['b']
        temp_prod_df.dropna(how='any', axis=0, inplace=True)

        reform = {(firstKey, secondKey, thirdKey): values for firstKey, secondDict in self.raw_data.B_term_oper_EOL.items() for secondKey, thirdDict in secondDict.items() for thirdKey, values in thirdDict.items()}
        mi = pd.MultiIndex.from_tuples(reform.keys())
        b_oper = pd.DataFrame(reform.values(), index=mi, columns=['b'])
        b_oper.index.names = ['lcphase', 'imp_int', 'tec']

        # Apply B-multiplication factors for operation and EOL A-factors
        temp_oper_df = self.raw_data.tec_parameters_raw.join(b_oper, on=['lcphase', 'imp', 'tec'], how='left')
        temp_oper_df['B'] = temp_oper_df['a'] * temp_oper_df['b']
        temp_oper_df.dropna(how='any', axis=0, inplace=True)
        temp_oper_df.drop(columns=['a', 'b'], inplace=True)


        # Aggregate component A values for TEC_PARAMETERS parameter
        A = self.raw_data.tec_parameters_raw.sum(axis=1)
        A = A.unstack(['comp']).sum(axis=1)
        A.columns = ['A']

        # Begin building final TEC_PARAMETERS parameter table
        temp_df['A'] = A
        B = pd.concat([temp_prod_df, temp_oper_df], axis=0).dropna(how='any', axis=1)
        B = B.unstack(['comp']).sum(axis=1)
        temp_df['B'] = B

        # Add same r values across all technologies...can add BEV vs ICE resolution here
        temp_r = pd.DataFrame.from_dict(self.raw_data.r_term_factors, orient='index', columns=['r'])
        temp_df = temp_df.join(temp_r, on=['tec'], how='left')

        # Add technology-specific u values
        temp_u = pd.DataFrame.from_dict(self.raw_data.u_term_factors, orient='index', columns=['u'])
        temp_df = temp_df.join(temp_u, on=['tec'], how='left')

        return temp_df



    def calc_veh_lifetime(self, sets):
        """
        Calculate survival curve from vehicle average age and standard deviation.

        Parameters
        ----------
        sets : SetsClass
            Used to retrieve desired age indices.

        Returns
        -------
        None.

        """
        # Placeholder for Weibull calculations
        # Weibull: alpha = scale, beta = shape
        # self.veh_lift_cdf = pd.Series(weibull_min.cdf(x, c, loc=0, scale=1), index=sets.age)
        # self.lifetime_age_distribution = pd.Series()
        # self.raw_data.veh_avg_age = alpha * gamma(1+beta^-1)
        # self.raw_data.veh_age_stdev^2 = alpha^2 * (gamma(1+2*beta^-1) - gamma(1+beta^-1)^2)
        # self.veh_lift_cdf = pd.Series(norm.cdf(sets.age_int, self.raw_data.veh_avg_age, self.raw_data.veh_age_stdev), index=sets.age)
        # self.veh_lift_cdf.index = self.veh_lift_cdf.index.astype('str')

        # Calculate normalized survival function
        self.lifetime_age_distribution = pd.Series(self.calc_steadystate_vehicle_age_distributions(sets.age_int, self.raw_data.veh_avg_age, self.raw_data.veh_age_stdev), index=sets.age)
        # self.veh_lift_sc = pd.Series(norm.sf(sets.age_int, self.raw_data.veh_avg_age, self.raw_data.veh_age_stdev), index=sets.age)
        # self.veh_lift_pdf = pd.Series(norm.pdf(sets.age_int, self.raw_data.veh_avg_age, self.raw_data.veh_age_stdev), index=sets.age)

        # self.veh_lift_pdf.index = self.veh_lift_pdf.index.astype('str')

        self.retirement_function = pd.Series(self.calc_probability_of_vehicle_retirement(sets.age_int, self.lifetime_age_distribution), index=sets.age)
        self.retirement_function.index = self.retirement_function.index.astype('str')


    def calc_steadystate_vehicle_age_distributions(self, ages, average_expectancy=10.0, standard_dev=3.0):
        """
        Calculate a steady-state age distribution consistent with a normal distribution around an average life expectancy.

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
        # offset ages by half timestep for discrete age bins (non-continuous removal)
        average_expectancy = average_expectancy  - (ages[1] - ages[0]) / 2

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

    def interpolate_years(self, df, sets, axis=1):
        """
        Construct DataFrame based on energy carbon intensities time series.

        Interpolate output from electricity carbon intensities (in decades)
        for annual time-step resolution. Also sets up DataFrame for GAMS
        database insertion.

        Parameters
        ----------
        df : pandas DataFrame or Series
            DataFrame or Series
        axis : int
            Axis containing years

        Returns
        -------
        None.

        """
        # electricity_clustering.py produces electricity pathways with decade resolution
        # this method interpolates between decades for an annual resolution
        # if axis:
            # df.columns = df.columns.astype('int64')

            # insert year headers
            # for decade in df.columns:
            #     for i in np.arange(1, 10):
            #         df[decade + i] = np.nan
            # df[2019] = df[2020]  # TODO: fill with historical data for pre-2020 period
            # df[2080] = df[2079]

            # interpolate between decades to get annual resolution
            # df = df.astype('float64').sort_index(axis=1).interpolate(axis=1) # sort year orders and interpolate
            # df.columns = df.columns.astype(str)  # set to strings for compatibility with GAMS

            # df = df.stack()  # reg, enr, year ['enr', 'reg', 'year']
            # df.index.rename('year', level=-1, inplace=True)

        ind = pd.Index(sets.modelyear, name='year')
        if axis:
            df.columns = df.columns.astype('str')
        else:
            df.index = df.index.astype('str')
        df = df.reindex(labels=ind, axis=axis)
        df = df.astype('float64').interpolate(axis=axis, limit_direction='both')
        df = pd.DataFrame(df.stack())
        df = self.order_sets(df)
        # df.index.rename('year', level=-1, inplace=True)

        return df

    def validate_data(self, sets):
        """
        Perform simple checks to validate input data.

        Parameters
        ----------
        sets : SetsClass
            Initialized SetsClass object.

        Returns
        -------
        None.

        """

        if isinstance(self.initial_seg_shares, list):
            # convert to dict with explicit connection to segments
            self.initial_seg_shares = {seg: share for seg, share in zip(sets.seg, self.initial_seg_shares)}
        if isinstance(self.initial_seg_shares, dict):
            if sum(self.initial_seg_shares.values()) != 1:
                print('\n *****************************************')
                log.warning('----- Vehicle segment shares (INITIAL_SEG_SHARES) do not sum to 1!')
        if isinstance(self.initial_tec_shares, (list, pd.Series)):
            tec_sum = sum(self.initial_tec_shares)
        elif isinstance(self.initial_tec_shares, dict):
            tec_sum = sum(self.initial_tec_shares.values())
        else:
            print('\n *****************************************')
            log.warning(f'----- initial_tec_shares is an invalid format. It is {type(self.initial_tec_shares)}; only dict or list allowed')
            tec_sum = np.nan
        if tec_sum != 1:
            print('\n *****************************************')
            log.warning('----- Vehicle powertrain technology shares (initial_tec_shares) do not sum to 1!')
            print(self.initial_tec_shares)
        if any(v is None for k, v in self.__dict__.items()):
            missing = [k for k, v in self.__dict__.items() if v is None]
            print('\n *****************************************')
            log.warning(f'----- The following parameters are missing values: {missing}')

        self.check_region_sets(self.exog_tot_stock.index.get_level_values('fleetreg'), 'exog_tot_stock', sets.fleetreg)

    @staticmethod
    def check_region_sets(par_ind, par_name, reg_set):
        """
        Check a given parameter for extra or missing regions.

        Parameters
        ----------
        par_ind : pandas Index
            Parameter index to check.
        par_name : str
            Name of parameter being checked (for troubleshooting)
        reg_set : list
            Established list of regions or countries to check.

        Returns
        -------
        None.

        """

        missing_regs = set(reg_set) - set(par_ind) # regions in set not represented in parameter
        extra_regs = set(par_ind) - set(reg_set)  # regions in parameter not in intialized set
        if len(missing_regs):
            log.error(f'Missing region {missing_regs} in parameter {par_name}')
        elif len(set(par_ind) - set(reg_set)):
            log.warning(f'Region {extra_regs} are in parameter {par_name}, but not declared as a set')    @staticmethod
    @staticmethod
    def order_sets(df):
        """
        Reorder parameter sets to match LP for feeding to GAMS.

        Parameters
        ----------
        df : pandas DataFrame
            Parameter read in from Excel.

        Returns
        -------
        df : pandas DataFrame
            Parameter with column order consistent with LP.

        """
        ordered_list = ['veheq',
                        'grdeq',
                        'lcphase',
                        'imp',
                        # 'imp_int',
                        'tec',
                        'newtec',
                        'mat_cat',
                        'mat_prod',
                        'enr',
                        'seg',
                        'reg',
                        'prodreg',
                        'fleetreg',
                        'year',
                        'modelyear',
                        'optyear',
                        'prodyear',
                        'age',
                        'age_int',
                        'sigvar',
                        'A',
                        'B',
                        'r',
                        'u']

        # check if years are in index
        if df.index.dtype == int:
            df = df.T
        df.reset_index(inplace=True)
        df_collist = df.columns
        new_collist = [col for col in ordered_list if col in df.columns]
        if len(new_collist)>0:
            df.set_index(new_collist, inplace=True)
        return df