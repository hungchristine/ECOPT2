********************************************************************************
********************************************************************************
********************************************************************************
* EVD4EUR Model
*
* April 2019
* Anders Hammer Strømman
* Ver 1.0 - DataLoad - Beta II

********************************************************************************
********************************************************************************
********************************************************************************

$funclibin stolib stodclib
function pdfnormal     /stolib.pdfnormal     /;
function cdfnormal     /stolib.cdfnormal     /;


SETS
year           year
optyear(year)  years for optiomization
inityear(year) years for initialization
age            age
tec            techlogy
enr            energy
sigvar         variables for sigmoid equations
dstvar         variables for statistical distributions
enreq          equations for energy (electricity and fuels) system
veheq          equations for vehicle parameters
demeq          equations for demand parameters
lfteq          equations for fleet lifetime parameters

$GDXIN EVD4EUR_ver101
*EVD4EUR_ver100_PreProcesing_BetaII
$load year
$load optyear
$load inityear
$load age
$load tec
$load enr
$load sigvar
$load dstvar
$load enreq
$load veheq
$load demeq
$load lfteq




*** ABBREIVATIONS USED *********************************************************
* PROD = Production
* OPER = Operation
* ENIT = Enerqy intensity
* CINT = CO2-eq intensity
* CNST = b in y = ax + b



* alias call for prodyear = produyction year is identical set to year
alias (year, prodyear)
alias (prodyear, year)
alias (age, agej)
alias (agej, age)

PARAMETERS
********************************************************************************
********************************************************************************
*
* Parameter Definitions p.t 1a : Parameter Declarations
*
********************************************************************************
********************************************************************************


***TIME*************************************************************************
*Declaraton of year as both a set and a parameter
AGE_PAR(age)                     age
YEAR_PAR(year)                   year
PRODYEAR_PAR(prodyear)           production year
VEH_PARTAB(tec,veheq,sigvar)     variables for each tech and veh equation
DEM_PARTAB(demeq,sigvar)         variables for demand equationions
ENR_PARTAB(enr,enreq,sigvar)     variables for each energy (fos or elc) equation
LFT_PARTAB(dstvar)               variables for fleet lifetime equations

***ENERGY (ELECTRICITY GENERATION and FUEL) ************************************
ENR_CINT(enr,year)               CO2 intensity of the energy  [kg CO2-eq pr kwh]

***ENERGY and VEHICLE TECHONLOGY COMBINATIONS **********************************
ENR_VEH(enr,tec)                 feasbile cominations of vehivle technology and energy (fuel).


***All VEHICLES*****************************************************************

**PRODUCTION
VEH_PROD_EINT(tec,prodyear)        Electricity intensity of vehicle prod [kwh el required per vehicle produced]
VEH_PROD_CINT_CSNT(tec,prodyear)   Constant term for CO2 int. of vehicle production [kg CO2-eq per vehicle produced]
VEH_PROD_CINT(tec,prodyear)        CO2 intensity of vehicle production [kg CO2-eq per vehicle produced]

**OPERATION
VEH_OPER_EINT(tec,prodyear)        Energy intensity of vehicle operation [kwh per km]
VEH_OPER_CINT(tec,enr,prodyear)    CO2 intensity of vehicle operation    [kg CO2 per km]

**EOL
VEH_EOLT_CINT(tec,year)            CO2 intensity of ICE vehicle EOL - [kg CO2-eq per vehicle in EOL treatment]




***FLEET ****************************************************************

**DEMAND


VEH_STCK_TOT(year)               Number of vehicles - #
VEH_OPER_DIST(year)              Annual driving distance per vehicles - km
VEH_OCUP(year)

** LIFETIME

VEH_AGE_DSTR(age)                Initial age distribution of vehicle fleet
VEH_LIFT_DSTR(age)               Share of scrapping for a given age - e.g % in given age class i scrapped

********************************************************************************
********************************************************************************
*
* Parameter Definitions p.t 1b : Parameter Load
*
********************************************************************************
********************************************************************************

$GDXIN EVD4EUR_ver100_PreProcesing_BetaII

$load AGE_PAR
$load YEAR_PAR
$load PRODYEAR_PAR
$load VEH_PARTAB
$load DEM_PARTAB
$load ENR_PARTAB
$load LFT_PARTAB
***ENERGY (ELECTRICITY GENERATION and FUEL) ************************************
$load ENR_CINT
***ENERGY and VEHICLE TECHONLOGY COMBINATIONS **********************************
$load ENR_VEH
***All VEHICLES*****************************************************************
**PRODUCTION
$load VEH_PROD_EINT
$load VEH_PROD_CINT_CSNT
$load VEH_PROD_CINT
**OPERATION
$load VEH_OPER_EINT
$load VEH_OPER_CINT
**EOL
$load VEH_EOLT_CINT
***VEHICLE FLEET****************************************************************
** Demand
$load VEH_STCK_TOT
$load VEH_OPER_DIST
$load VEH_OCUP
** Lifetime
$load VEH_AGE_DSTR
$load VEH_LIFT_DSTR




;

DISPLAY

***TIME*************************************************************************
*Declaraton of year as both a set and a parameter
AGE_PAR
YEAR_PAR
PRODYEAR_PAR
VEH_PARTAB
DEM_PARTAB
ENR_PARTAB
LFT_PARTAB

***ENERGY (ELECTRICITY GENERATION and FUEL) ************************************
ENR_CINT

***ENERGY and VEHICLE TECHONLOGY COMBINATIONS **********************************
ENR_VEH


***All VEHICLES*****************************************************************

**PRODUCTION
VEH_PROD_EINT
VEH_PROD_CINT_CSNT
VEH_PROD_CINT

**OPERATION
VEH_OPER_EINT
VEH_OPER_CINT

**EOL
VEH_EOLT_CINT



***VEHICLE FLEET****************************************************************
** Demand

VEH_STCK_TOT
VEH_OPER_DIST
VEH_OCUP

* Lifetime
VEH_AGE_DSTR
VEH_LIFT_DSTR
;

execute_unload 'EVD4EUR_ver100_DataLoad_BetaII.gdx'


