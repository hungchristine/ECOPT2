********************************************************************************
********************************************************************************
********************************************************************************
* EVD4EUR Model
* Basic Edition
* Januar 2018
* Anders Hammer Str�mman
*
* ver 0.1:       basic parameters and structires
* ver 0.2:       refinement of basic parameters and initiation period
* ver 0.3:       introduction of stock initialisation
* ver 0.4:       refinement of stock initialisation
* ver 0.5:       initialization of stock completed
* ver 0.6:       basic equations added
* ver 0.7:       moving stock initialisation from parameters to model equations - AS separate EQs
* ver 0.8:       stock initialisation loop vs sep equ. index problem fixed
* ver 0.9:       linking initial stock model to BEV and ICE Model
* ver 0.91:      Running version with stock total constraint as equal to or larger than
* ver 0.92:      Running version with total constraint as equal - fixed.
* ver 0.93:      Adding BEV STCK and ADD fractions - to be moved to post processing in 0.94
* ver 0.94:      Post processing as above completed
* ver 0.95:      Output analysis and debogging p.t 1 - Clean up Emissions and Stock e.q.s
* ver 0.96:      Include fix for vehicle removal-add constraint across BEV-ICE. Pref constraint forced BEVs to replace BEVs and ICEs to replace ICEs
* ver 0.97:      Changed DELTA variables from positive to free variables
* ver 0.98       Added gradient of change dampening constraint
* ver 0.99       Added init variables for ICE_ADD and ICE_REM
* ver 1.00       Generalized version with technologies as sets etc (only front end).
* ver 1.01       Generalized version with technologies as sets etc. Running version
* ver 1.02       Tecnology gradient of change constraints added.
* ver 1.03       Tecnology gradient of change constraints ver B added.
* ver 1.04       Initiaton resolved

********************************************************************************
********************************************************************************
********************************************************************************


$funclibin stolib stodclib
function pdfnormal     /stolib.pdfnormal     /;
function cdfnormal     /stolib.cdfnormal     /;



SETS
year           year /2000*2050/
optyear(year)  years for optiomization /2020*2050/
inityear(year) years for initialization /2000*2020/
age            age /0*20/
tec            techlogy /ICE,BEV/
enr            energy /ELC,FOS/
sigvar         variables for sigmoid equations /A,B,r,t,u/
dstvar         variables for statistical distributions /mean,stdv/
enreq          equations for energy (electricity and fuels) system /CINT/
veheq          equations for vehicle parameters /PROD_EINT, PROD_CINT_CSNT, OPER_EINT, EOLT_CINT/
demeq          equations for demand parameters /STCK_TOT, OPER_DIST, OCUP/
lfteq          equations for fleet lifetime parameters /LIFT_DISTR, AGE_DISTR/
grdeq          parameters for gradient of change (fleet additions) - individual (IND) for each tech or related to all tech (ALL) /IND,ALL/

*** ABBREIVATIONS USED *********************************************************
* PROD = Production
* OPER = Operation
* ENIT = Enerqy intensity
* CINT = CO2-eq intensity
* CNST = b in y = ax + b
;

$GDXIN 'EVD4EUR_input'
*$LOAD year optyear inityear age tec enr sigvar dstvar enreq veheq demeq lfteq grdeq prodyear agej tecj
$LOAD
$GDXIN


* alias call for prodyear = production year is identical set to year
alias (year, prodyear)
alias (prodyear, year)
alias (age, agej)
alias (agej, age)
alias (tec, tecj)
alias (tecj, tec)

**** General logistic function *************************************************
$macro genlogfnc(A,B,r,t,u) A + (B-A)/(1+exp(-r*(t-u)));
**
** https://en.wikipedia.org/wiki/Generalised_logistic_function
** y = A + (B-A)/(1+exp(-r*(t-tau)));
** Y  is for example the  CO2 eq. intensity of electricity generation
** t  is time.
** A = Initial CO2 intensity of electricity generation
** B = End CO2 intensity of electricity generation
** r = is the rate of change ;
** (tau) u is time the of maximum gradient of Y
*;
*
*
PARAMETERS
*********************************************************************************
*********************************************************************************
**
** Parameter Definitions p.t 1 : Parameter Declarations
**
*********************************************************************************
*********************************************************************************
*
*
****TIME*************************************************************************
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

VEH_LIFT_PDF(age)                Age PDF
VEH_LIFT_CDF(age)                Age CDF Share of scrapping for a given age - e.g % in given age class i scrapped
VEH_LIFT_AGE(age)                Age distribution = 1 - CDF
VEH_LIFT_MOR(age)                Age distribution = 1 - CDF

** COMPOSITION

VEH_PAY(prodyear,age,year)       Correspondance between a vehicle prododuction year and its age (up to 20) in a given year
VEH_STCK_INT_TEC(tec)            Initial share of vehicles in stock tech
VEH_STCK_INT(tec,age)            Initial size of stock of vehicles by age cohort.

** GRADIENT OF CHANGE
VEH_ADD_GRD(grdeq,tec)           Parameter for gradient of change constraint (fleet additions) - individual (IND) for each tech or related to all tech (ALL)

;

* Load in parameter values from Python-generated .gdx file

$GDXIN 'EVD4EUR_input'
*$LOADR
$LOAD AGE_PAR YEAR_PAR PRODYEAR_PAR VEH_PARTAB DEM_PARTAB ENR_PARTAB LFT_PARTAB ENR_CINT ENR_VEH VEH_PROD_EINT VEH_PROD_CINT_CSNT VEH_PROD_CINT VEH_OPER_EINT VEH_OPER_CINT VEH_EOLT_CINT VEH_STCK_TOT VEH_OPER_DIST VEH_OCUP VEH_LIFT_PDF VEH_LIFT_CDF VEH_LIFT_AGE VEH_LIFT_MOR VEH_PAY VEH_STCK_INT_TEC VEH_STCK_INT VEH_ADD_GRD
$GDXIN


********************************************************************************
********************************************************************************
*
* Model Definition  p.t 1 : Variable Definitions
*
********************************************************************************
********************************************************************************


***FREE VARIABLES***************************************************************
* Objective value to be minimized must be a free variable


FREE VARIABLES
TOTC                                 Total CO2 emissions for the whole system over the whole period
VEH_STCK_DELTA(year)                 Delta stock from one year to the next
;

POSITIVE VARIABLES
VEH_STCK(tec,year,age)               Number of vehicles of a given age in a given year
VEH_STCK_REM(tec,year,age)           Number of vehicles of a given age retired in a given year
VEH_STCK_TOT_CHECK(year)             Check on number of vehicles
VEH_STCK_ADD(tec,year,age)           Stock additions (new car sales)

VEH_TOTC(tec,year)                   Total CO2 emissions of BEV vehicles per year

VEH_PROD_TOTC(tec,year)              Total CO2 emissions from production of BEV vehicles per year
VEH_OPER_TOTC(tec,year)              Total CO2 emissions from operations of BEV vehicles per year
VEH_EOLT_TOTC(tec,year)              Total CO2 emissions from operations of BEV vehicles per year
;


********************************************************************************
********************************************************************************
*
* Model Definition  p.t 2 : Model Equation Declarations
*
********************************************************************************
********************************************************************************

EQUATIONS

***VEHICLE STOCK MODEL  ************************************************************

***  Initiate stock in first year
EQ_STCK_REM_T1
*stock additions
EQ_STCK_ADD_T1
*stock
EQ_STCK_BAL_T1

***  Main Model
EQ_STCK_DELTA
*removals
EQ_STCK_REM
*stock additions
EQ_STCK_ADD
*calculating vehicle stock in a given year
EQ_STCK_BAL
*summing the number of vehicles in for check.
EQ_STCK_CHK

*** Gradient of Change

* Gradient of change constraint - as % of individual veh tec add in previous year
EQ_STCK_GRD

**EMISSION and ENERGY MODELS incl OBJ. FUNCTION ************************************************************

* Objective function
EQ_TOTC
* Calculation of Emissions from all vehivles classes per year
EQ_VEH_TOTC
*producton emissions
EQ_VEH_PROD_TOTC
*operation emissions
EQ_VEH_OPER_TOTC
*eolt emissions
EQ_VEH_EOLT_TOTC

;

********************************************************************************
********************************************************************************
*
* Model Definition  p.t 3 : Model Equations
*
********************************************************************************
********************************************************************************


***VEHICLE STOCK MODEL  ************************************************************



***  Initiate stock in first year

*removals
EQ_STCK_REM_T1(tec,year,age)$(ord(year)=1)..                     VEH_STCK_REM(tec,year,age) =e=  VEH_STCK_TOT(year)*VEH_STCK_INT_TEC(tec)*VEH_LIFT_AGE(age)*VEH_LIFT_MOR(age);

*stock additions
*EQ_STCK_ADD_T1(tec,year,age)$(ord(year)=1 and ord(age)=1)..      VEH_STCK_ADD(tec,year,age) =e=  VEH_STCK_TOT(year)*VEH_STCK_INT_TEC(tec)*VEH_LIFT_AGE(age);
EQ_STCK_ADD_T1(year,age)$(ord(year)=1 and ord(age)=1)..      sum( (tec), VEH_STCK_ADD(tec,year,age) ) =e=  sum( (tec,agej), VEH_STCK_REM(tec,year,agej));

*stock
EQ_STCK_BAL_T1(tec,year,age)$(ord(year)=1)..                     VEH_STCK(tec,year,age) =e=  VEH_STCK_TOT(year)*VEH_STCK_INT_TEC(tec)*VEH_LIFT_AGE(age);


***  Main Model
EQ_STCK_DELTA(year)$(ord(year)>1)..                              VEH_STCK_DELTA(year)  =e=  VEH_STCK_TOT(year)-VEH_STCK_TOT(year-1);

*removals
EQ_STCK_REM(tec,year,age)$(ord(year)>1)..                        VEH_STCK_REM(tec,year,age) =e= VEH_STCK(tec,year-1,age-1)*VEH_LIFT_MOR(age-1);

*stock additions
EQ_STCK_ADD(year,age)$(ord(year)>1 and ord(age)=1)..             sum( (tec), VEH_STCK_ADD(tec,year,age)- sum( (agej), VEH_STCK_REM(tec,year,agej) ) ) =e= VEH_STCK_DELTA(year);

*calculating vehicle stock in a given year
EQ_STCK_BAL(tec,year,age)$(ord(year)>1)..                        VEH_STCK(tec,year,age)  =e=  VEH_STCK(tec,year-1,age-1) + VEH_STCK_ADD(tec,year,age) - VEH_STCK_REM(tec,year,age);

*summing the number of vehicles in for check.
EQ_STCK_CHK(year)..                                              VEH_STCK_TOT_CHECK(year) =e= sum((tec,age), VEH_STCK(tec,year,age));

*** Gradient of change constraint

EQ_STCK_GRD(tec,year,age)$(ord(year)>1 and ord(age)=1)..         VEH_STCK_ADD(tec,year,age) =l= (1 + VEH_ADD_GRD('IND',tec))*VEH_STCK_ADD(tec,year-1,age) + 5e6; ;



***EMISSION and ENERGY MODELS incl OBJ. FUNCTION ************************************************************

* Objective function
EQ_TOTC..                                TOTC =e= SUM( (tec,year), VEH_TOTC(tec,year));


* Calculation of Emissions from all vehivles classes per year
EQ_VEH_TOTC(tec,year)..                  VEH_TOTC(tec,year) =e= VEH_PROD_TOTC(tec,year) + VEH_OPER_TOTC(tec,year) + VEH_EOLT_TOTC(tec,year);

EQ_VEH_PROD_TOTC(tec,year)..             VEH_PROD_TOTC(tec,year) =e= sum( (agej)$(ord(agej)=1), VEH_STCK_ADD(tec,year,agej)*VEH_PROD_CINT(tec,year));
EQ_VEH_OPER_TOTC(tec,year)..             VEH_OPER_TOTC(tec,year) =e= sum( (agej,enr,prodyear), VEH_STCK(tec,year,agej)*VEH_OPER_CINT(tec,enr,prodyear)*ENR_VEH(enr,tec)*VEH_PAY(prodyear,agej,year)*VEH_OPER_DIST(year) );
EQ_VEH_EOLT_TOTC(tec,year)..             VEH_EOLT_TOTC(tec,year) =e= sum( (agej), VEH_STCK_REM(tec,year,agej))*VEH_EOLT_CINT(tec,year);


********************************************************************************
********************************************************************************
*
* Model Execution  p.t 1 : Model Definition and Options
*
********************************************************************************
********************************************************************************

* Defining Name of Model(s) and what Equations are used in each Model

MODEL EVD4EUR_Basic
/ALL/
;
* Defining Run Optoins and solver

*OPTION RESLIM = 2000000;
*OPTION NLP = CONOPT;
*OPTION THREADS = 40;

*OPTION limrow = 0;
*OPTION limcol = 0;
*OPTION solprint = off;
*OPTION sysout = off;


********************************************************************************
********************************************************************************
*
* Model Execution  p.t 2 : Variables Initial Values
*
********************************************************************************
********************************************************************************

*Not required



********************************************************************************
********************************************************************************
*
* Model Execution  p.t 3 : Solve Call
*
********************************************************************************
********************************************************************************

SOLVE EVD4EUR_Basic USING LP MINIMIZING TOTC;

DISPLAY VEH_STCK_TOT
DISPLAY VEH_STCK_TOT_CHECK.l


execute_unload 'EVD4EUR_wo_paramdefs.gdx'
*execute_unload 'EVD4EUR_input.gdx' year optyear inityear age tec enr sigvar dstvar enreq veheq demeq lfteq grdeq prodyear agej tecj AGE_PAR YEAR_PAR PRODYEAR_PAR VEH_PARTAB DEM_PARTAB ENR_PARTAB LFT_PARTAB ENR_CINT ENR_VEH VEH_PROD_EINT VEH_PROD_CINT_CSNT VEH_PROD_CINT VEH_OPER_EINT VEH_OPER_CINT VEH_EOLT_CINT VEH_STCK_TOT VEH_OPER_DIST VEH_OCUP VEH_LIFT_PDF VEH_LIFT_CDF VEH_LIFT_AGE VEH_LIFT_MOR VEH_PAY VEH_STCK_INT_TEC VEH_STCK_INT VEH_ADD_GRD
