********************************************************************************
********************************************************************************
********************************************************************************
* EVD4EUR Model
* Basic Edition
* Januar 2018
* Anders Hammer Strømman
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
* ver 0.98:      Added gradient of change dampening constraint
* ver 1.00:      Redesign of 'front end'
* ver 1.01:      Complete introduction of model in ver 1.00
********************************************************************************
********************************************************************************
********************************************************************************
$funclibin stolib stodclib
function pdfnormal     /stolib.pdfnormal     /;
function cdfnormal     /stolib.cdfnormal     /;


SETS
year           year /2000*2050/
optyear(year)  years for optimization /2020*2050/
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

*** General logistic function *************************************************
$macro genlogfnc(A,B,r,t,u) A + (B-A)/(1+exp(-r*(t-u)));
*
* https://en.wikipedia.org/wiki/Generalised_logistic_function
* y = A + (B-A)/(1+exp(-r*(t-tau)));
* Y  is for example the  CO2 eq. intensity of electricity generation
* t  is time.
* A = Initial CO2 intensity of electricity generation
* B = End CO2 intensity of electricity generation
* r = is the rate of change ;
* (tau) u is time the of maximum gradient of Y



PARAMETERS
********************************************************************************
********************************************************************************
*
* Parameter Definitions p.t 1 : Parameter Declarations
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

VEH_LIFT_PDF(age)                Age PDF
VEH_LIFT_CDF(age)                Age CDF Share of scrapping for a given age - e.g % in given age class i scrapped
VEH_LIFT_AGE(age)                  Age distribution = 1 - CDF

** COMPOSITION

VEH_PAY(prodyear,age,year)       Correspondance between a vehicle prododuction year and its age (up to 20) in a given year
VEH_STCK_INT_TEC(tec)            Initial share of vehicles in stock tech
VEH_STCK_INT(tec,age)            Initial size of stock of vehicles by age cohort.


;
********************************************************************************
********************************************************************************
*
* Parameter Definitions p.t 2 : Parameter Equations
*
********************************************************************************
********************************************************************************


***TIME*************************************************************************

* Declaraton of age also as a parameter

AGE_PAR('0') = 0;
loop (age$(ord(age)>1), AGE_PAR(age) = AGE_PAR(age-1)+1);

* Declaraton of year also as a parameter

YEAR_PAR('2000') = 2000;
loop (year$(ord(year)>=2), YEAR_PAR(year) = YEAR_PAR(year-1)+1);

PRODYEAR_PAR('2000') = 2000;
loop (year$(ord(year)>=2), PRODYEAR_PAR(year) = PRODYEAR_PAR(year-1)+1);


***ENERGY (ELECTRICITY GENERATION and FOSSIL FUEL*******************************
* kg CO2 eg pr kWh
PARAMETER ENR_PARTAB(enr,enreq,sigvar)
/        ELC    .CINT    .A  = 1.3
         ELC    .CINT    .B  = 0.1
         ELC    .CINT    .r  = 0.2
         ELC    .CINT    .u  = 2035
         FOS    .CINT    .A  = 0.26
         FOS    .CINT    .B  = 0.26
         FOS    .CINT    .r  = 0.2
         FOS    .CINT    .u  = 2035
/;


ENR_CINT(enr,year) =  genlogfnc(ENR_PARTAB(enr,'CINT','A'),ENR_PARTAB(enr,'CINT','B'),ENR_PARTAB(enr,'CINT','r'),YEAR_PAR(year),ENR_PARTAB(enr,'CINT','u'));


***ENERGY and VEHICLE TECHONLOGY COMBINATIONS **********************************

PARAMETER ENR_VEH(enr,tec)
/        ELC    .BEV   = 1.0
         FOS    .ICE   = 1.0

/;

***VEHICLES*****************************************************************

PARAMETER VEH_PARTAB(tec,veheq,sigvar)
/        ICE     .PROD_EINT      .A  = 2500
         ICE     .PROD_EINT      .B  = 2000
         ICE     .PROD_EINT      .r  = 0.2
         ICE     .PROD_EINT      .u  = 2025
         ICE     .PROD_CINT_CSNT .A  = 3000
         ICE     .PROD_CINT_CSNT .B  = 1500
         ICE     .PROD_CINT_CSNT .r  = 0.2
         ICE     .PROD_CINT_CSNT .u  = 2025
         ICE     .OPER_EINT      .A  = 0.73
         ICE     .OPER_EINT      .B  = 0.27
         ICE     .OPER_EINT      .r  = 0.2
         ICE     .OPER_EINT      .u  = 2025
         ICE     .EOLT_CINT      .A  = 75
         ICE     .EOLT_CINT      .B  = 30
         ICE     .EOLT_CINT      .r  = 0.15
         ICE     .EOLT_CINT      .u  = 2025
         BEV     .PROD_EINT      .A  = 6500
         BEV     .PROD_EINT      .B  = 2500
         BEV     .PROD_EINT      .r  = 0.2
         BEV     .PROD_EINT      .u  = 2025
         BEV     .PROD_CINT_CSNT .A  = 5000
         BEV     .PROD_CINT_CSNT .B  = 1500
         BEV     .PROD_CINT_CSNT .r  = 0.2
         BEV     .PROD_CINT_CSNT .u  = 2025
         BEV     .OPER_EINT      .A  = 0.21
         BEV     .OPER_EINT      .B  = 0.11
         BEV     .OPER_EINT      .r  = 0.2
         BEV     .OPER_EINT      .u  = 2025
         BEV     .EOLT_CINT      .A  = 100
         BEV     .EOLT_CINT      .B  = 35
         BEV     .EOLT_CINT      .r  = 0.15
         BEV     .EOLT_CINT      .u  = 2025
/;

VEH_PROD_EINT(tec,prodyear) = genlogfnc(VEH_PARTAB(tec,'PROD_EINT','A'),VEH_PARTAB(tec,'PROD_EINT','B'),VEH_PARTAB(tec,'PROD_EINT','r'),YEAR_PAR(prodyear),VEH_PARTAB(tec,'PROD_EINT','u'));

VEH_PROD_CINT_CSNT(tec,prodyear) = genlogfnc(VEH_PARTAB(tec,'PROD_CINT_CSNT','A'),VEH_PARTAB(tec,'PROD_CINT_CSNT','B'),VEH_PARTAB(tec,'PROD_CINT_CSNT','r'),YEAR_PAR(prodyear),VEH_PARTAB(tec,'PROD_CINT_CSNT','u'));

VEH_PROD_CINT(tec,prodyear) = VEH_PROD_CINT_CSNT(tec,prodyear) + VEH_PROD_EINT(tec,prodyear)*ENR_CINT('elc',prodyear);

VEH_OPER_EINT(tec,prodyear) = genlogfnc(VEH_PARTAB(tec,'OPER_EINT','A'),VEH_PARTAB(tec,'OPER_EINT','B'),VEH_PARTAB(tec,'OPER_EINT','r'),YEAR_PAR(prodyear),VEH_PARTAB(tec,'OPER_EINT','u'));

VEH_OPER_CINT(tec,enr,prodyear)$(ENR_VEH(enr,tec)) = VEH_OPER_EINT(tec,prodyear)*ENR_CINT(enr,prodyear);

VEH_EOLT_CINT(tec,prodyear) = genlogfnc(VEH_PARTAB(tec,'EOLT_CINT','A'),VEH_PARTAB(tec,'EOLT_CINT','B'),VEH_PARTAB(tec,'EOLT_CINT','r'),YEAR_PAR(prodyear),VEH_PARTAB(tec,'EOLT_CINT','u'));



***FLEEET****************************************************************

*DEMAND


PARAMETER DEM_PARTAB(demeq,sigvar)
/        STCK_TOT   .A  = 20e6
         STCK_TOT   .B  = 40e6
         STCK_TOT   .r  = 0.1
         STCK_TOT   .u  = 2025
         OPER_DIST  .A  = 15000
         OPER_DIST  .B  = 10000
         OPER_DIST  .r  = 0.1
         OPER_DIST  .u  = 2025
         OCUP       .A  = 1.3
         OCUP       .B  = 1.6
         OCUP       .r  = 0.1
         OCUP       .u  = 2025
/;


* Size of vehicle fleet (demand)
VEH_STCK_TOT(year)  = genlogfnc(DEM_PARTAB('STCK_TOT','A'),DEM_PARTAB('STCK_TOT','B'),DEM_PARTAB('STCK_TOT','r'),YEAR_PAR(year),DEM_PARTAB('STCK_TOT','u'));

* Annual driving distance per vehicles - km
VEH_OPER_DIST(year) = genlogfnc(DEM_PARTAB('OPER_DIST','A'),DEM_PARTAB('OPER_DIST','B'),DEM_PARTAB('OPER_DIST','r'),YEAR_PAR(year),DEM_PARTAB('OPER_DIST','u'));

* Vehicle occupancy
VEH_OCUP(year) = genlogfnc(DEM_PARTAB('OCUP','A'),DEM_PARTAB('OCUP','B'),DEM_PARTAB('OCUP','r'),YEAR_PAR(year),DEM_PARTAB('OCUP','u'));

***LIFETIME
PARAMETER LFT_PARTAB(dstvar)
/        mean  = 10
         stdv  =  3
/;



VEH_LIFT_PDF(age) = pdfnormal(AGE_PAR(age),LFT_PARTAB('mean'),LFT_PARTAB('stdv'));
VEH_LIFT_CDF(age) = cdfnormal(AGE_PAR(age),LFT_PARTAB('mean'),LFT_PARTAB('stdv'));
VEH_LIFT_AGE(age) = (1 - VEH_LIFT_CDF(age))/sum(agej, VEH_LIFT_CDF(agej));

***COMPOSITION

* Defining the correspondance between a vehicle prododuction year and its age (up to 20) in a given year
* VEH_PAY(prodyear,age,year)
loop( (prodyear,age,year)$( ord(year)= ord(prodyear)+ord(age)), VEH_PAY(prodyear,age,year) = 1);

PARAMETER VEH_STCK_INT_TEC(tec)
/        ICE = 0.95
         BEV = 0.05
/;

* Initial size of stock of vehicles by age cohort.
VEH_STCK_INT(tec,age) = VEH_STCK_INT_TEC(tec)*VEH_LIFT_AGE(age)*VEH_STCK_TOT('2000');



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


POSITIVE VARIABLES
VEH_STCK(tec,year,age)               Number of vehicles of a given age in a given year
VEH_STCK_REM(tec,year,age)           Number of vehicles of a given age retired in a given year
VEH_STCK_TOT_CHECK(year)             Check on number of vehicles
VEH_STCK_ADD(tec,year,age)           Stock additions (new car sales)

VEH_TOTC(tec,year)                   Total CO2 emissions of BEV vehicles per year

VEH_PROD_TOTC(tec,year)              Total CO2 emissions from production of BEV vehicles per year
VEH_OPER_TOTC(tec,year)              Total CO2 emissions from operations of BEV vehicles per year
VEH_EOLT_TOTC(tec,year)              Total CO2 emissions from operations of BEV vehicles per year



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
EQ_STCK_REM_T1(tec,year,age)$(ord(year)=1 )..                    VEH_STCK_REM(tec,year,age) =e=  VEH_STCK_TOT(year)*VEH_STCK_INT_TEC(tec)*VEH_LIFT_AGE(age)*VEH_LIFT_CDF(age);

*stock additions
EQ_STCK_ADD_T1(tec,year,age)$(ord(year)=1 and ord(age)=1)..      VEH_STCK_ADD(tec,year,age) =e=  VEH_STCK_TOT(year)*VEH_STCK_INT_TEC(tec)*VEH_LIFT_AGE(age);

*stock
EQ_STCK_BAL_T1(tec,year,age)$(ord(year)=1)..                     VEH_STCK(tec,year,age) =e=  VEH_STCK_TOT(year)*VEH_STCK_INT_TEC(tec)*VEH_LIFT_AGE(age);



***  Main Model
EQ_STCK_DELTA(year)$(ord(year)>1)..                              VEH_STCK_DELTA(year)  =e=  VEH_STCK_TOT(year)-VEH_STCK_TOT(year-1);

*removals
EQ_STCK_REM(tec,year,age)$(ord(year)>1)..                        VEH_STCK_REM(tec,year,age) =e= VEH_STCK(tec,year-1,age-1)*VEH_LIFT_CDF(age);

*stock additions
EQ_STCK_ADD(year,age)$(ord(year)>1 and ord(age)=1)..             sum( (tec), VEH_STCK_ADD(tec,year,age)- sum( (agej), VEH_STCK_REM(tec,year,agej) ) ) =e= VEH_STCK_DELTA(year);

*calculating vehicle stock in a given year
EQ_STCK_BAL(tec,year,age)$(ord(year)>1)..                        VEH_STCK(tec,year,age)  =e=  VEH_STCK(tec,year-1,age-1) + VEH_STCK_ADD(tec,year,age) - VEH_STCK_REM(tec,year,age);

*summing the number of vehicles in for check.
EQ_STCK_CHK(year)..                                              VEH_STCK_TOT_CHECK(year) =e= sum((tec,age), VEH_STCK(tec,year,age));


**EMISSION and ENERGY MODELS incl OBJ. FUNCTION ************************************************************

* Objective function
EQ_TOTC..                                TOTC =e= SUM( (tec,year), VEH_TOTC(tec,year));


* Calculation of Emissions from all vehivles classes per year
EQ_VEH_TOTC(tec,year)..                  VEH_TOTC(tec,year) =e= VEH_PROD_TOTC(tec,year) + VEH_OPER_TOTC(tec,year) + VEH_EOLT_TOTC(tec,year);

EQ_VEH_PROD_TOTC(tec,year)..             VEH_PROD_TOTC(tec,year) =e= sum( (agej)$(ord(agej)=1), VEH_STCK_ADD(tec,year,agej)*VEH_PROD_CINT(tec,year));
EQ_VEH_OPER_TOTC(tec,year)..             VEH_OPER_TOTC(tec,year) =e= sum( (agej,enr,prodyear), VEH_STCK(tec,year,agej)*VEH_OPER_CINT(tec,enr,prodyear)*VEH_PAY(prodyear,agej,year)*ENR_VEH(enr,tec)*VEH_OPER_DIST(year) );
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

* Defining Run Options and solver

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


execute_unload 'EVD4EUR_ver101.gdx'
