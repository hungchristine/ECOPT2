# ECOPT<sup>2</sup>

A Python module for the Environmentally Constrained Optimization of Prospective Technology Transitions (ECOPT<sup>2</sup>). 

## Description
ECOPT<sup>2</sup> is a generalized and adaptable model that combines life cycle assessment (LCA), dynamic stock modelling and linear programming (LP) to assess environmentally optimal widescale deployment strategies for emerging technologies. 

## Installation
It is recommended that you create a [virtual environment](https://docs.python.org/3/library/venv.html#:~:text=A%20virtual%20environment%20is%20a,part%20of%20your%20operating%20system.) to install and run ECOPT<sup>2</sup> in. 

In the virtual environment, install the [GAMS Python API](https://www.gams.com/latest/docs/API_PY_TUTORIAL.html#PY_GETTING_STARTED).

[Clone](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository-from-github/cloning-a-repository)  this repository to your local machine to install ECOPT<sup>2</sup>.

## Dependencies
- gams
- pandas
- scipy
- numpy
- matplotlib
- pyYAML

## Usage

### Data input
Users provide input via Excel spreadsheets and a YAML file:
- <b>sets.xlsx:</b> Defines the members of each set in the LP model. The first line is the set names (must match names as defined in LP). Set members are listed in the column below each set name.
- <b>GAMS_input.xlsx:</b> Provides exogeneous data to feed the LP model. Each tab in the spreadsheet is named to match parameter names in ECOPT<sup>2</sup>.  Further details are provided in the sample file in this repository. 
- <b>GAMS_input.yaml:</b> File defining values for parameters for experiments. Multiple values can be defined for each parameter; ECOPT<sup>2</sup> will perform experiments covering all combinations of all parameter values provided. Parameter values are provided by listing the parameter name (must match name in Python code and/or LP ), an experiment alias, and the parameter values as a scalar, list or dictionary (see e.g., [here](https://realpython.com/python-yaml/) for a tutorial on YAML syntax)



### Running experiment(s)
Run models using main.py. In this file, you can specify the experiment type (e.g., demo, unit test or normal). Included in this repository is a demo experiment for a simple, stylized scenario, which is set up to run by default. Users can also specify what file format to export result figures in (.png or .pdf), whether to visualize input parameters for troubleshooting, and whether to export cross-experiment results.

Each experiment output is saved in its own folder, and includes the visualization output, a log file, the GAMS result file in .gdx format and a pickle containing the experiment's FleetModel object.

### Folder structure
- <code>\data\\</code>: contains input files  for experiment
- <code>\demo\\</code>: contains input files for demo experiment using <code>gmspy</code>
- <code>\output\\</code>: contains results, including visualization, logfile and <code>.gdx</code> files from experiments


### Helper scripts
- <code>electricity_clustering.py</code>: calculate impact intensity of national/regional electricity mixes based on generation technologies and create clusters of regions with similarly intensive electricity mixes
- <code>iam_parser.py</code>: general utility for parsing data from integrated assessment models using the [IAMC data format](https://pyam-iamc.readthedocs.io/en/stable/data.html#:~:text=Over%20the%20past%20decade%2C%20the,of%20the%20Sustainable%20Development%20Goals.) for time series
- <code>gmspy.py</code>: utility code for bilateral data exchange between Python and GAMS
 


## License
This code is licensed under the [BSD 3.0](https://choosealicense.com/licenses/bsd-3-clause/) license.
