""" gmspy - Simple convenience functions to ease the use of GAMS API

There is nothing very special here. All based on (as of 2019-04-23):
    https://www.gams.com/latest/docs/API_PY_TUTORIAL.html#PY_GAMSDATABASE_GDX_IMPORT

"""

import pandas as pd
import gams
import warnings

def list2set(db, var, name, comment='', verbose=True):
    """
    Insert GAMS set based on python list.

    Parameters
    ----------
    db : Gams Database
        Where the parameter will live
    var : list
        Keys that define the set
    name: str
        Name of the parameter in GAMS
    comment : str
        Optional, comment
    verbose : boolean
        If False, the function will silently convert elements of `var` to string upon insertion in `db`

    Returns
    -------
    a_set : GamsSet instance

    """
    try:
      a_set = db.get_set(name)
    except:
        a_set = db.add_set(name, 1, comment)
    try:
        for v in var:
            a_set.add_record(v)
    except gams.GamsException:
        for v in var:
            a_set.add_record(str(v))
            if verbose:
                warnings.warn('Set added just fine, but with its elements converted to strings.')
    return a_set

def df2param(db, df, domains, name, comment=''):
    """
    Insert pandas dataframe as Parameter in a GAMS database. Revision: also accepts dictionaries.

    Parameters
    ----------
    db : Gams Database
        Where the parameter will live
    df: Pandas DataFrame or Series
        Data to be saved as parameter. Indexes and columns must match with pre-defined GAMS sets/domains
    domains: list of instances of GamsSet
        Must match with the indexes and columns of `df`
    name: str
        Name of the parameter in GAMS
    comment : str
        Optional, comment

    Returns
    -------
    a_param : GamsParameter instance

    """
    ## if df is a single-column dataframe (i.e., series cast as dataframe), the below doesn't work; keys uses the index-column name pair...
    ## NB: this function assumes that the columns are in the correct order
    try:
        a_param = db.get_parameter(name)
    except:
        a_param = db.add_parameter_dc(name, domains, comment)

    if isinstance(df, float) or isinstance(df, int):
        # special case: scalar value
        a_param.add_record().value = df
        return a_param
    if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
        if df.ndim > 1 and df.shape[1] > 1:
            # if dataframe with more than one column, stack (to make series)
            df = df.stack()
        df = df.squeeze()
        df = df.to_dict()
    for keys, data in iter(df.items()):
        a_param.add_record(keys).value = data

    return a_param


def ls(db=None, ws=None, gdx_filepath=None, entity=None):
    """
    List either all content or selected entities within a GAMS database or gdx file.

    Can list everything, or all instances of 'Set', 'Parameter', 'Equation' or 'Variable'

    Parameters
    ----------
    db : Gams Database or None
        If available, read from this pre-existing database
    ws : Gams WorkSpace or None
        If available, use that workspace to read a gdx file, otherwise generate one
    gdx_filepath : str or None
        Path to gdx file
    entity : None or str {'Set' | 'Parameter' | 'Equation' | 'Variable'}
        If None, list all content, otherwise restrict list to the right type of entity

    Returns
    -------
    out: list of strings
        Names of entities in db or gdx file
    """
    # Sort out database access or file reading
    db = _iwantitall(db, ws, gdx_filepath)

    # Entities of interest
    classes = {'Set': gams.GamsSet,
               'Parameter':gams.GamsParameter,
               'Equation':gams.GamsEquation,
               'Variable':gams.GamsVariable}

    if entity is None:
        out = [i.get_name() for i in db]
    else:
        out = [i.get_name() for i in db if isinstance(i, classes[entity])]

    return out


def set2list(name, db=None, ws=None, gdx_filepath=None):
    """
    Read in a Set from a GAMS database or gdx file.

    Parameters
    ----------
    name: str
        Name of Gams Set
    db : Gams Database or None
        If available, read from this pre-existing database
    ws : Gams WorkSpace or None
        If available, use that workspace to read a gdx file, otherwise generate one
    gdx_filepath : str or None
        Path to gdx file

    Returns
    -------
    list
        list of GAMS Set keys
    """
    # Sort out database access or file reading
    db = _iwantitall(db, ws, gdx_filepath)

    # Read keys of set as list
    return [rec.keys[0] for rec in db[name]]


def param2series(name, db=None, ws=None, gdx_filepath=None):
    """--- DEPRECATED ---"""
    """
    Read in a parameter from a GAMS database or gdx file.

    Parameters
    ----------
    name: str
        Name of Gams Parameter
    db : Gams Database or None
        If available, read from this pre-existing database
    ws : Gams WorkSpace or None
        If available, use that workspace to read a gdx file, otherwise generate one
    gdx_filepath : str or None
        Path to gdx file,

    Returns
    -------
    Pandas Series
        Series (possibly multi-index) holding the values of the parameter
    """
    # Sort out database access or file reading
    db = _iwantitall(db, ws, gdx_filepath)

    # Read in data and recast as Pandas Series
    return pd.Series(dict((tuple(rec.keys), rec.value) for rec in db[name]))

def param2df(name, db=None, ws=None, gdx_filepath=None):
    """
    Read in a parameter from a GAMS database or gdx file.

    Parameters
    ----------
    name: str
        Name of Gams Parameter
    db : Gams Database or None
        If available, read from this pre-existing database
    ws : Gams WorkSpace or None
        If available, use that workspace to read a gdx file, otherwise generate one
    gdx_filepath : str or None
        Path to gdx file,

    Returns
    -------
    df : Pandas DataFrame
        Dataframe holding the values of the parameter, with rows potentially multi-index
    """
    # Sort out database access or file reading
    db = _iwantitall(db, ws, gdx_filepath)

    # Read in data and recast as Pandas Series
    ds = pd.Series(dict((tuple(rec.keys), rec.value) for rec in db[name]))

    if ds.index.nlevels >1:
        df = ds.unstack()
        # Ensure that the unstacked columns are ordered as in Data Series
        ix_len = len(ds.index.levels[-1])
        cols = ds.index.get_level_values(-1)[:ix_len]

        if len(cols) == len(ds):
            df = df.reindex(columns=cols)
    else:
        df = ds.to_frame(0)
    return df


def var2series(name, db=None, ws=None, gdx_filepath=None):
    """
    DEPRECATED.

    Read in a variable from a GAMS database or gdx file.

    Parameters
    ----------
    name: str
        Name of Gams Variable
    db : Gams Database or None
        If available, read from this pre-existing database
    ws : Gams WorkSpace or None
        If available, use that workspace to read a gdx file, otherwise generate one
    gdx_filepath : string or None
        Path to gdx file,

    Returns
    -------
    Pandas Series
        Series (possibly multi-index) holding the values of the variable
    """
    # Sort out database access or file reading
    db = _iwantitall(db, ws, gdx_filepath)

    # Read in data and recast as Pandas Series
    data = dict((tuple(rec.keys), rec.level) for rec in db[name])
    df = pd.Series(data)
    df.index.rename(db[name].domains_as_strings,inplace=True)
    return df

def var2df(name, db=None, ws=None, gdx_filepath=None):
    """
    Read in a variable from a GAMS database or gdx file.

    Parameters
    ----------
    name: str
        Name of Gams Variable
    db : Gams Database or None
        If available, read from this pre-existing database
    ws : Gams WorkSpace or None
        If available, use that workspace to read a gdx file, otherwise generate one
    gdx_filepath : string or None
        Path to gdx file

    Returns
    -------
    Pandas DataFrame
        Dataframe holding the values of the variable, with row indexes potentially multiindex

    See Also
    --------
    var2series
    """
    # Sort out database access or file reading
    db = _iwantitall(db, ws, gdx_filepath)

    # Read in data and recast as Pandas Series
    data = dict((tuple(rec.keys), rec.level) for rec in db[name])
    df = pd.Series(data)
    df.index.rename(db[name].domains_as_strings,inplace=True)

    if df.index.nlevels > 1:
        df = df.unstack()
    else:
        df = df.to_frame(0)
    return df

def eq2series(name, db=None, ws=None, gdx_filepath=None):
    """
    Read in a equation from a GAMS database or gdx file.

    Parameters
    ----------
    name: str
        Name of Gams Equation
    db : Gams Database or None
        If available, read from this pre-existing database
    ws : Gams WorkSpace or None
        If available, use that workspace to read a gdx file, otherwise generate one
    gdx_filepath : string or None
        Path to gdx file,

    Returns
    -------
    Pandas Series
        Series holding the values of the variable, with row indexes potentially multiindex

    See Also
    --------
    var2series
    """
    # Sort out database access or file reading
    db = _iwantitall(db, ws, gdx_filepath)

    # Read in data and recast as Pandas Series
    data = dict((tuple(rec.keys), rec.marginal) for rec in db[name])
    df = pd.Series(data)
    df.index.rename(db[name].domains_as_strings, inplace=True)

    return df

def _iwantitall(db, ws, gdx_filepath):
    """
    Read a pre-existing database, or setup one to read a gdx file. Internal method.

    Parameters
    ----------
    db : Gams Database or None
        If available, read from this pre-existing database
    ws : Gams WorkSpace or None
        If available, use that workspace to read a gdx file, otherwise generate one
    gdx_filepath : string
        Path to gdx file

    Returns
    -------
    db : Gams Database
    """
    if not db:
        if not ws:
            ws = gams.GamsWorkspace()
        db = ws.add_database_from_gdx(gdx_filepath)
    return db