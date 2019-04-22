********************************************************************************
********************************************************************************
********************************************************************************
* EVD4EUR Model
*
* April 2019
* Anders Hammer Strømman
* Ver 1.0 - Pre Processing - Beta II

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

VEH_AGE_DSTR(age)                Initial age distribution of vehicle fleet
VEH_LIFT_DSTR(age)               Share of scrapping for a given age - e.g % in given age class i scrapped


***STOCK INITISATION************************************************************


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

***FLEET ****************************************************************
PARAMETER LFT_PARTAB(dstvar)
/        mean  = 10
         stdv  =  3
/;


VEH_AGE_DSTR(age)  = pdfnormal(AGE_PAR(age),LFT_PARTAB('mean'),LFT_PARTAB('stdv'));
VEH_LIFT_DSTR(age) = cdfnormal(AGE_PAR(age),LFT_PARTAB('mean'),LFT_PARTAB('stdv'));




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

execute_unload 'EVD4EUR_ver100_PreProcesing_BetaII.gdx'


