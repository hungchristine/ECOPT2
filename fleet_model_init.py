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

    tec: List = field(default_factory=lambda:['BEV', 'ICEV'])
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
    newtec: List = field(default_factory=lambda:['BEV'])
    demeq: List = field(default_factory=lambda: ['STCK_TOT', 'OPER_DIST', 'OCUP'])
    grdeq: List = field(default_factory=lambda: ['IND', 'ALL'])
    veheq: List = field(default_factory=lambda: ['PROD_EINT', 'PROD_CINT_CSNT', 'OPER_EINT', 'EOLT_CINT'])
    sigvar: List = field(default_factory=lambda: ['A', 'B', 'r', 'u'])


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

    #TODO: move the operations from fleet_model here; calculation of veh_partab, glf terms, etc etc

    pkm_scenario: str = None
    all_pkm_scen: pd.DataFrame = None
    fleet_vkm: pd.DataFrame = None
    batt_portfolio: pd.DataFrame = None
    veh_factors: pd.DataFrame = None
    B_term_prod: Union[dict, float] = None
    B_term_oper_EOL: Union[dict, float] = None
    r_term_factors: Union[dict, pd.Series] = None  # {str: float}
    u_term_factors: Union[dict, pd.Series] = None  # {str: float}
    prod_df: pd.DataFrame = None

    veh_avg_age: Union[float, dict] = 11.1 # From ACEA 2019-2020 report
    veh_age_stdev: Union[float, dict] = 2.21

    bev_int_shr: float = 0.0018  # from Eurostat; assume remaining is ICE

    recycle_rate: float = 0.75

    # other parameters
    occupancy_rate: Union[float, pd.Series] = 2.
    eur_batt_share: Union[float, pd.Series] = 1.  # default assumes primary material supply given is for entire region
    tec_add_gradient: Union[float, pd.Series] = 0.2

    enr_glf_terms: pd.Series= None

@dataclass
class ParametersClass:
    """Contains all parameter values for GAMS model (in ready-to-insert form)."""

    veh_stck_tot: Union[pd.Series, pd.DataFrame] = None
    enr_veh: Union[pd.Series, pd.DataFrame] = None
    veh_pay: Union[pd.Series, pd.DataFrame] = None
    year_par: Union[pd.Series, pd.DataFrame] = None
    veh_partab: Union[pd.Series, pd.DataFrame] = None

    # constraints
    manuf_cnstrnt: Union[pd.Series, pd.DataFrame] = None

    mat_content: Union[List, pd.Series, pd.DataFrame] = None
    virg_mat_supply: Union[pd.Series, pd.DataFrame] = None
    mat_cint: Union[List, pd.Series, pd.DataFrame] = None
    veh_add_grd: Union[float, pd.Series, pd.DataFrame] = None

    enr_cint: Union[pd.Series, pd.DataFrame] = None

    raw_data: RawDataClass = None

    veh_oper_dist: Union[float, int, List, Dict, pd.Series, pd.DataFrame] = None
    veh_stck_int_seg: Union[Dict, List] = field(default_factory=lambda:[0.08, 0.21, 0.27, 0.08, 0.03, 0.34])  # Shares from 2017, ICCT report
    veh_stck_int_tec: Union[Dict, pd.Series, pd.DataFrame] = None

    bev_capac: Union[Dict, List] = field(default_factory=lambda:{'A': 26.6, 'B': 42.2, 'C': 59.9, 'D': 75., 'E':95., 'F':100.})
    veh_lift_cdf: Union[pd.Series, pd.DataFrame] = None
    veh_lift_pdf: Union[pd.Series, pd.DataFrame] = None
    veh_lift_mor: Union[pd.Series, pd.DataFrame] = None
    veh_lift_age: Union[pd.Series, pd.DataFrame] = None

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
        if isinstance(self.veh_add_grd, pd.DataFrame) and not isinstance(self.veh_add_grd.index, pd.MultiIndex):
            tmp = self.veh_add_grd.stack()
            self.veh_add_grd = tmp.to_dict()

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
                   'enr_glf_terms':['enr', 'reg', 'enreq'],
                   'virg_mat_supply': ['mat_cat', 'mat_prod'],
                   'mat_cint': ['mat_cat', 'mat_prod'],
                   'batt_portfolio':['seg', 'battery size'],
                   'veh_factors': ['veheq', 'tec', 'seg'],
                   'enr_veh': ['enr', 'tec'],
                   'veh_pay': ['cohort', 'age', 'year'],
                   'enr_cint': ['fleetreg', 'enr'],
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
                    value = value.astype({cat: str for cat in mi_dict[param.lower()]})
                    value.set_index(mi_dict[param.lower()], inplace=True, drop=True)
                elif value.shape[0] == 0:
                    log.info(f"Empty values for {param} in Excel file.")
                elif value.shape == (1,2):
                    value = value.iloc[0,1]
                else:
                    # single-level Index
                    value.set_index(value.iloc(axis=1)[0].name, inplace=True, drop=True)
                    value.index = value.index.astype(str)  # ensure all indices are strings (required to work for GAMS)
                    value.columns = value.columns.astype(str)

                if param.lower() in param_attrs:
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

        if (self.veh_oper_dist is not None) and (self.raw_data.pkm_scenario is not None):
            log.warning('----- Vehicle operating distance overspecified. Both an annual vehicle mileage and an IAM scenario are specified.')

        if self.veh_pay is None:
            self.veh_pay = self.build_veh_pay(sets)  # establish self.veh_pay

        has_required_data = all(attr is not None for attr in (self.raw_data.B_term_prod,
                                                              self.raw_data.B_term_oper_EOL,
                                                              self.raw_data.r_term_factors,
                                                              self.raw_data.u_term_factors
                                                             ))
        if self.veh_partab is not None and has_required_data:
            log.warning('----- veh_partab is overdefined')
        elif has_required_data:
            self.veh_partab = self.build_veh_partab()

        if self.raw_data.eur_batt_share:
            # multiply manufacturing constraint and critical material supply by eur_batt_share
            # allows e.g., expressing constraints as global values, with a regional market share
            self.virg_mat_supply *= self.raw_data.eur_batt_share
            self.manuf_cnstrnt *= self.raw_data.eur_batt_share

        # TODO: expand veh_oper_dist to be tec and reg specific (also in GAMS)
        if isinstance(self.veh_oper_dist, (float, int)):
            # calculate veh_oper_dist
            # given a single value for veh_oper_dist, assumes that value applies for every region, year and technology
            self.veh_oper_dist = pd.Series([self.veh_oper_dist for i in range(len(sets.modelyear))], index=sets.modelyear)
        elif isinstance(self.veh_oper_dist, dict):
            self.veh_oper_dist = pd.Series(self.veh_oper_dist)
        self.veh_oper_dist.index.name = 'year'

        if self.raw_data.pkm_scenario:
            # calculate veh oper dist
            self.raw_data.passenger_demand = self.raw_data.all_pkm_scen.T[self.raw_data.pkm_scenario]
            self.raw_data.passenger_demand.reset_index()
            self.raw_data.passenger_demand *= 1e9
            self.raw_data.passenger_demand.name = ''

            self.raw_data.fleet_vkm  = self.raw_data.passenger_demand / self.raw_data.occupancy_rate
            self.raw_data.fleet_vkm.index = sets.modelyear
            self.raw_data.fleet_vkm.index.name = 'year'
            self.veh_oper_dist = self.raw_data.fleet_vkm / self.veh_stck_tot.T.sum()  # assumes uniform distribution of annual distance travelled vs vehicle age and region
            if self.veh_oper_dist.mean() > 25e3:
                log.warning('Warning, calculated annual vehicle mileage is above 25000 km, check fleet_km and veh_stck_tot')

        if self.raw_data.recycle_rate is not None:
            self.recovery_pct = [[self.raw_data.recycle_rate]*len(sets.mat_cat) for year in range(len(sets.modelyear))]
            self.recovery_pct = pd.DataFrame(self.recovery_pct,
                                             index=sets.modelyear,
                                             columns=sets.mat_cat)

        if (self.raw_data.enr_glf_terms is not None) and self.enr_cint is not None:
            log.warning('----- Source for energy pathways may be overspecified; both enr_glf_terms and enr_cint are specified')
        elif self.enr_cint is not None:
            # for building enr_cint from IAM pathways (see electricity_clustering.py)
            self.build_enr_cint()
        elif self.raw_data.enr_glf_terms is not None:
            # build enr_cint from generalized logistic function
            mi = pd.MultiIndex.from_product([sets.reg, sets.enr, sets.modelyear], names=['reg', 'enr', 'modelyear'])
            self.enr_cint = pd.Series(index=mi)

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

            self.enr_cint = self.enr_cint.swaplevel(0, 1) # enr, reg, year
            self.enr_cint = self.enr_cint.to_frame()
            self.enr_cint.dropna(how='all', axis=0, inplace=True)

        if (isinstance(self.raw_data.tec_add_gradient, float)) and (self.veh_add_grd is None):
            self.veh_add_grd = {}
            for element in product(*[sets.grdeq, sets.tec]):
                if element[1] in sets.newtec:
                    self.veh_add_grd[element] = self.raw_data.tec_add_gradient

        if len(self.virg_mat_supply.columns) in [len(sets.year), len(sets.optyear), len(sets.modelyear)]:
            # if years are in columns, transpose for transferring to GAMS
            self.virg_mat_supply = self.virg_mat_supply.T

        oldtec = list(set(sets.tec) - set(sets.newtec))  # get name of incumbent technology; works for single tec
        if len(sets.newtec) == 1:
            self.veh_stck_int_tec = pd.Series([1-self.raw_data.bev_int_shr, self.raw_data.bev_int_shr], index=oldtec + sets.newtec)
        else:
            if isinstance(self.raw_data.bev_int_shr, dict):
                all_new_tecs = sum(self.raw_data.bev_int_shr.values())
                self.veh_stck_int_tec = pd.Series(self.raw_data.bev_int_shr)
                self.veh_stck_int_tec.loc[oldtec] = 1- all_new_tecs
            elif isinstance(self.raw_data.bev_int_shr, pd.DataFrame) or isinstance(self.raw_data.bev_int_shr, pd.Series):
                self.veh_stck_int_tec.loc[oldtec] - 1 - self.veh_stck_int_tec.sum()

        self.year_par = pd.Series([float(i) for i in sets.year], index=sets.year)
        self.calc_veh_lifetime(sets)



    def build_veh_pay(self, sets):
        """
        Build production year-cohort concordance matrix for use in GAMS (as parameter).

        Returns
        -------
        index: pd.DataFrame
            Correspondence between current year, cohort and age

        """

        top_year = int(sets.optyear[-1])
        start_year = int(sets.inityear[0])
        prod_year = start_year - int(sets.age[-1])
        ind = []
        for year in range(start_year, top_year + 1):
            for a in sets.age_int:
                prod_year = year - a - 1
                ind.append([prod_year, a, year, 1])

        index = pd.DataFrame(ind)
        #index = index[index[0]<=2050]
        index = index[index[2] <= (top_year + 1)]
        for ind, col in index.iloc[:,:-1].iteritems():
            index.loc(axis=1)[ind] = index.loc(axis=1)[ind].astype(str)
        index.columns = ['prodyear', 'age', 'year', 'level']

        return index


    def build_BEV(self):
        """
        Fetch BEV production emissions based on battery size.

        Select battery size by segment from size-segment combinations,
        fetch and sum production emissions for battery and rest-of-vehicle.
        Update DataFrame with total production emissions and energy use for
        BEVs by segment.

        Returns
        -------
        None.

        """
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
        self.raw_data.prod_df.sort_index(inplace=True)
        try:
            self.raw_data.prod_df.drop('battery weight', axis=0, inplace=True)  # remove (currently not implemented)
        except KeyError:
            log.info('Could not drop battery weight')

    def build_veh_partab(self):
        """
        Build VEH_PARTAB parameter containing sigmoid function terms.

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
        self.raw_data.veh_factors.columns.names = ['comp']
        self.raw_data.veh_factors = self.raw_data.veh_factors.stack().to_frame('a')

        # Retrieve production emission factors for chosen battery capacities and place in raw A factors (with component resolution)
        self.build_BEV()  # update self.prod_df with selected battery capacities
        self.raw_data.veh_factors.sort_index(inplace=True)
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
        # self.veh_lift_age = pd.Series()
        # self.raw_data.veh_avg_age = alpha * gamma(1+beta^-1)
        # self.raw_data.veh_age_stdev^2 = alpha^2 * (gamma(1+2*beta^-1) - gamma(1+beta^-1)^2)
        self.veh_lift_cdf = pd.Series(norm.cdf(sets.age_int, self.raw_data.veh_avg_age, self.raw_data.veh_age_stdev), index=sets.age)
        self.veh_lift_cdf.index = self.veh_lift_cdf.index.astype('str')

        # Calculate normalized survival function
        self.veh_lift_age = pd.Series(self.calc_steadystate_vehicle_age_distributions(sets.age_int, self.raw_data.veh_avg_age, self.raw_data.veh_age_stdev), index=sets.age)
        self.veh_lift_sc = pd.Series(norm.sf(sets.age_int, self.raw_data.veh_avg_age, self.raw_data.veh_age_stdev), index=sets.age)
        self.veh_lift_pdf = pd.Series(norm.pdf(sets.age_int, self.raw_data.veh_avg_age, self.raw_data.veh_age_stdev), index=sets.age)

        self.veh_lift_pdf.index = self.veh_lift_pdf.index.astype('str')

        self.veh_lift_mor = pd.Series(self.calc_probability_of_vehicle_retirement(sets.age_int, self.veh_lift_age), index=sets.age)
        self.veh_lift_mor.index = self.veh_lift_mor.index.astype('str')


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

    def build_enr_cint(self):
        """
        Construct DataFrame based on energy carbon intensities time series.

        Interpolate output from electricity carbon intensities (in decades)
        for annual time-step resolution. Also sets up DataFrame for GAMS
        database insertion.

        Returns
        -------
        None.

        """
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

        if isinstance(self.veh_stck_int_seg, list):
            # convert to dict with explicit connection to segments
            self.veh_stck_int_seg = {seg: share for seg, share in zip(sets.seg, self.veh_stck_int_seg)}
        if isinstance(self.veh_stck_int_seg, dict):
            if sum(self.veh_stck_int_seg.values()) != 1:
                print('\n *****************************************')
                log.warning('----- Vehicle segment shares (VEH_STCK_INT_SEG) do not sum to 1!')
        if isinstance(self.veh_stck_int_tec, (list, pd.Series)):
            tec_sum = sum(self.veh_stck_int_tec)
        elif isinstance(self.veh_stck_int_tec, dict):
            tec_sum = sum(self.veh_stck_int_tec.values())
        else:
            print('\n *****************************************')
            log.warning(f'----- veh_stck_int_tec is an invalid format. It is {type(self.veh_stck_int_tec)}; only dict or list allowed')
            tec_sum = np.nan
        if tec_sum != 1:
            print('\n *****************************************')
            log.warning('----- Vehicle powertrain technology shares (VEH_STCK_INT_TEC) do not sum to 1!')
            print(self.veh_stck_int_tec)
        if any(v is None for k, v in self.__dict__.items()):
            missing = [k for k, v in self.__dict__.items() if v is None]
            print('\n *****************************************')
            log.warning(f'----- The following parameters are missing values: {missing}')