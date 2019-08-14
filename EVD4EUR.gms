*-----------------------------------------------------------------------------------
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

*-----------------------------------------------------------------------------------

$funclibin stolib stodclib
function pdfnormal     /stolib.pdfnormal     /;
function cdfnormal     /stolib.cdfnormal     /;



SETS
year           year
optyear(year)  years for optimization
inityear(year) years for initialization
age            age
tec            techlogy
enr            energy
seg            segment or size class
sigvar         variables for sigmoid equations
dstvar         variables for statistical distributions
enreq          equations for energy (electricity and fuels) system
veheq          equations for vehicle parameters
demeq          equations for demand parameters
lfteq          equations for fleet lifetime parameters
grdeq          parameters for gradient of change (fleet additions) - individual (IND) for each tech or related to all tech (ALL)\

*year           year /2000*2050/
*optyear(year)  years for optimization /2020*2050/
*inityear(year) years for initialization /2000*2020/
*age            age /0*20/
*tec            techlogy /ICE,BEV/
*enr            energy /ELC,FOS/
*sigvar         variables for sigmoid equations /A,B,r,t,u/
*dstvar         variables for statistical distributions /mean,stdv/
*enreq          equations for energy (electricity and fuels) system /CINT/
*veheq          equations for vehicle parameters /PROD_EINT, PROD_CINT_CSNT, OPER_EINT, EOLT_CINT/
*demeq          equations for demand parameters /STCK_TOT, OPER_DIST, OCUP/
*lfteq          equations for fleet lifetime parameters /LIFT_DISTR, AGE_DISTR/
*grdeq          parameters for gradient of change (fleet additions) - individual (IND) for each tech or related to all tech (ALL) /IND,ALL/

*---- ABBREIVATIONS USED *-----------------------------------------------------------------------------------
* PROD = Production
* OPER = Operation
* ENIT = Enerqy intensity
* CINT = CO2-eq intensity
* CNST = b in y = ax + b
;

** Load sets defined in Python class
$if not set gdxincname $abort 'no include file name for data file provided'
$gdxin %gdxincname%
*$GDXIN 'troubleshooting_params'
$LOAD year
$LOAD tec
$LOAD age
$LOAD enr
$LOAD seg
$LOAD demeq
$LOAD dstvar
$LOAD enreq
$LOAD grdeq
$LOAD inityear
$LOAD lfteq
$LOAD sigvar
$LOAD veheq
$LOAD optyear
$GDXIN

* alias call for prodyear = production year is identical set to year
* change prodyear to cohort later
alias (year, prodyear)
alias (prodyear, year)
alias (age, agej)
alias (agej, age)
alias (tec, tecj)
alias (tecj, tec)
alias (segj,seg)
alias (seg,segj)

*--- General logistic function -----------------------------------------------------------------------------------
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
**-----------------------------------------------------------------------------------
**
** Parameter Definitions p.t 1 : Parameter Declarations
**
**-----------------------------------------------------------------------------------

**-- TIME ---------------------------------------------------------------------------
*Declaraton of year as both a set and a parameter
YEAR_PAR(year)                   year
*VEH_PARTAB(tec,veheq,sigvar)     variables for each tech and veh equation
VEH_PARTAB(veheq,tec,seg,sigvar)     variables for each tech and veh equation
DEM_PARTAB(demeq,sigvar)         variables for demand equationions
ENR_PARTAB(enr,enreq,sigvar)     variables for each energy (fos or elc) equation
*LFT_PARTAB(dstvar)               variables for fleet lifetime equations

***ENERGY (ELECTRICITY GENERATION and FUEL) ------------------------------------------
ENR_CINT(enr,year)               CO2 intensity of the energy  [kg CO2-eq pr kwh]

***ENERGY and VEHICLE TECHONLOGY COMBINATIONS ----------------------------------------
ENR_VEH(enr,tec)                 feasible cominations of vehicle technology and energy (fuel).


***All VEHICLES ----------------------------------------------------------------------

**PRODUCTION
VEH_PROD_EINT(tec,seg,prodyear)        Electricity intensity of vehicle prod [kwh el required per vehicle produced]
VEH_PROD_CINT_CSNT(tec,seg,prodyear)   Constant term for CO2 int. of vehicle production [t CO2-eq per vehicle produced]
VEH_PROD_CINT(tec,seg,prodyear)        CO2 intensity of vehicle production [t CO2-eq per vehicle produced]

**OPERATION
VEH_OPER_EINT(tec,seg,prodyear)        Energy intensity of vehicle operation [kwh per km]
VEH_OPER_CINT(tec,enr,seg,prodyear)    CO2 intensity of vehicle operation    [t CO2 per km]

**EOL
VEH_EOLT_CINT(tec,seg,year)            CO2 intensity of ICE vehicle EOL - [t CO2-eq per vehicle in EOL treatment]



***FLEET -------------------------------------------------------------------------------
** INITIAL STOCK ------------
INIT_SEG(seg)
INIT_TEC(tec)
INIT_AGE(age)
SEG_TEC(seg,tec)
SEG_TEC_AGE(seg,tec,age)

**DEMAND --------------------

VEH_STCK_TOT(year)               Number of vehicles - #
VEH_OPER_DIST(year)              Annual driving distance per vehicles - km
VEH_OCUP(year)

** LIFETIME

VEH_LIFT_PDF(age)                Age PDF
VEH_LIFT_CDF(age)                Age CDF Share of scrapping for a given age - e.g % in given age class i scrapped
VEH_LIFT_AGE(age)                Age distribution = 1 - CDF
VEH_LIFT_MOR(age)                Age distribution = 1 - CDF

** COMPOSITION --------------

VEH_PAY(prodyear,age,year)       Correspondance between a vehicle production year and its age (up to 20) in a given year
VEH_STCK_INT_TEC(tec)            Initial share of vehicles in stock tech
VEH_STCK_INT_SEG(seg)            Initial stock distribution by segment
VEH_STCK_INT(tec,seg,age)        Initial size of stock of vehicles by age cohort and segment

** GRADIENT OF CHANGE -------
VEH_ADD_GRD(grdeq,tec)           Parameter for gradient of change constraint (fleet additions) - individual (IND) for each tech or related to all tech (ALL)
VEH_SEG_SHR(seg)                 Parameter for segment share minimums in fleet
GRO_CNSTRNT(year)                 Segment growth rate constraint
;


* Here, A = mini; B=small, C = lower medium; D = medium; E = upper medium F = luxury, SUV, sport, vans and others


*$call ="c:\gams\win64\26.1\xls2gms.exe" I="C:\Users\chrishun\Box Sync\YSSP_temp\GAMS_input.xls" *O="VEH_OPER_DIST.inc" R="Sheet2!A2:B52"
*PARAMETER VEH_OPER_DIST(year)
*/
*$include VEH_OPER_DIST.inc
*/
*$onecho > commands.txt
*I=C:\Users\chrishun\Box Sync\YSSP_temp\GAMS_input.xls
*O=VEH_OPER_DIST.inc
*R=Sheet2!A2:B52
*$offecho

*$call ="c:\gams\win64\26.1\xls2gms.exe" @commands.txt


* Load in parameter values from .gdx file [dummy data]
$ONMULTI
$GDXIN 'EVD4EUR_input'
$LOAD DEM_PARTAB
$LOAD ENR_PARTAB
$LOAD VEH_OCUP
$GDXIN

* Load in parameter values defined in Python class
$if not set gdxincname $abort 'no include file name for data file provided'
$gdxin %gdxincname%
$LOAD VEH_OPER_DIST
$LOAD VEH_STCK_TOT
$LOAD VEH_SEG_SHR
$LOAD VEH_LIFT_CDF
$LOAD VEH_LIFT_AGE
$LOAD YEAR_PAR
$LOAD VEH_PARTAB
$LOAD VEH_PROD_CINT_CSNT
$LOAD VEH_PROD_EINT
$LOAD VEH_OPER_EINT
$LOAD VEH_STCK_INT
$LOAD VEH_STCK_INT_TEC
$LOAD VEH_PAY
$LOAD VEH_EOLT_CINT
$LOAD ENR_VEH
$LOAD VEH_LIFT_PDF
$LOAD VEH_LIFT_MOR
$LOAD VEH_ADD_GRD
$LOAD GRO_CNSTRNT 
$OFFMULTI
$GDXIN
;


* temporary definition; to introduce for all inityears
PARAMETER VEH_STCK_INT_SEG(seg)
/
        A = 0.08
        B = 0.21
        C = 0.26
        D = 0.08
        E = 0.03
        F = 0.34
/;


*VEH_STCK_INT_SEG(seg) = VEH_SEG_SHR(seg);

ENR_CINT(enr,year) =  genlogfnc(ENR_PARTAB(enr,'CINT','A'),ENR_PARTAB(enr,'CINT','B'),ENR_PARTAB(enr,'CINT','r'),YEAR_PAR(year),ENR_PARTAB(enr,'CINT','u'));

*----- Production-related emissions
VEH_PROD_EINT(tec,seg,prodyear) = genlogfnc(VEH_PARTAB('PROD_EINT',tec,seg,'A'),VEH_PARTAB('PROD_EINT',tec,seg,'B'),VEH_PARTAB('PROD_EINT',tec,seg,'r'),YEAR_PAR(prodyear),VEH_PARTAB('PROD_EINT',tec,seg,'u'));

VEH_PROD_CINT_CSNT(tec,seg,prodyear) = genlogfnc(VEH_PARTAB('PROD_CINT_CSNT',tec,seg,'A'),VEH_PARTAB('PROD_CINT_CSNT',tec,seg,'B'),VEH_PARTAB('PROD_CINT_CSNT',tec,seg,'r'),YEAR_PAR(prodyear),VEH_PARTAB('PROD_CINT_CSNT',tec,seg,'u'));

VEH_PROD_CINT(tec,seg,prodyear) = VEH_PROD_CINT_CSNT(tec,seg,prodyear) + VEH_PROD_EINT(tec,seg,prodyear)*ENR_CINT('elc',prodyear)/1000;

*----- Operation phase emissions
VEH_OPER_EINT(tec,seg,prodyear) = genlogfnc(VEH_PARTAB('OPER_EINT',tec,seg,'A'),VEH_PARTAB('OPER_EINT',tec,seg,'B'),VEH_PARTAB('OPER_EINT',tec,seg,'r'),YEAR_PAR(prodyear),VEH_PARTAB('OPER_EINT',tec,seg,'u'));

VEH_OPER_CINT(tec,enr,seg,prodyear)$(ENR_VEH(enr,tec)) = VEH_OPER_EINT(tec,seg,prodyear)*ENR_CINT(enr,prodyear)/1000; 

*----- End-of-life phase emissions
VEH_EOLT_CINT(tec,seg,prodyear) = genlogfnc(VEH_PARTAB('EOLT_CINT',tec,seg,'A'),VEH_PARTAB('EOLT_CINT',tec,seg,'B'),VEH_PARTAB('EOLT_CINT',tec,seg,'r'),YEAR_PAR(prodyear),VEH_PARTAB('EOLT_CINT',tec,seg,'u'));

*----- Initialization of vehicle stock
INIT_SEG(seg) = VEH_STCK_INT_SEG(seg)*VEH_STCK_TOT('2000');
INIT_TEC(tec) = VEH_STCK_INT_TEC(tec)*VEH_STCK_TOT('2000');
INIT_AGE(age) = VEH_LIFT_PDF(age)*VEH_STCK_TOT('2000');
SEG_TEC(seg,tec) = (VEH_STCK_INT_TEC(tec))*(VEH_STCK_INT_SEG(seg));
SEG_TEC_AGE(seg,tec,age) = (VEH_STCK_INT_TEC(tec)*VEH_STCK_INT_SEG(seg))*VEH_LIFT_PDF(age);

VEH_STCK_INT(tec,seg,age) =(VEH_STCK_INT_TEC(tec)*VEH_LIFT_PDF(age)*VEH_STCK_INT_SEG(seg))*VEH_STCK_TOT('2000');


*-----------------------------------------------------------------------------------
*
* Model Definition  p.t 1 : Variable Definitions
*
*-----------------------------------------------------------------------------------


***FREE VARIABLES ------------------------------------------------------------------
* Objective value to be minimized must be a free variable


FREE VARIABLES
TOTC                                    Total CO2 emissions for the whole system over the whole period
VEH_STCK_DELTA(year)                    Delta stock from one year to the next
slack_VEH_ADD(seg,year,age)
;

POSITIVE VARIABLES
VEH_STCK(tec,seg,year,age)              Number of vehicles of a given age in a given year
VEH_STCK_REM(tec,seg,year,age)          Number of vehicles of a given age retired in a given year
VEH_STCK_TOT_CHECK(year)                Check on number of vehicles
VEH_STCK_ADD(tec,seg,year,age)          Stock additions (new car sales)
VEH_TOT_ADD(year)                       Total vehicles added
VEH_TOTC(tec,seg,year)                  Total CO2 emissions of vehicles per year, by technology, in t CO2-eq
VEH_STCK_ADD(tec,seg,year,age)          Stock additions (new car sales)

VEH_PROD_TOTC(tec,seg,year)             Total CO2 emissions from production of vehicles per year, in t CO2-eq
VEH_OPER_TOTC(tec,seg,year)             Total CO2 emissions from operations of vehicles per year, in t CO2-eq
VEH_EOLT_TOTC(tec,seg,year)             Total CO2 emissions from vehicle end of life treatment per year, in t CO2-eq
check_add_tot(year)
VEH_TOT_REM(year)
*VEH_STCK_CHRT(tec,seg,year,age,year)
;


*-----------------------------------------------------------------------------------
*
* Model Definition  p.t 2 : Model Equation Declarations
*
*-----------------------------------------------------------------------------------

EQUATIONS

***VEHICLE STOCK MODEL  ------------------------------------------------------------

***  Initiate stock in first year
EQ_STCK_REM_T1
*stock additions
EQ_STCK_ADD_T1
EQ_STCK_ADD_T11
*stock
EQ_STCK_BAL_T1
EQ_STCK_DELTA_T1
***  Main Model
EQ_STCK_DELTA
*removals
EQ_STCK_REM
*stock additions
EQ_STCK_ADD
*calculating vehicle stock in a given year
EQ_STCK_BAL
*summing the number of vehicles added and in stock for check.
*EQ_TOT_ADD
*EQ_TOT_REM
EQ_STCK_CHK

*** Constraint equations
* For technology-related additions
EQ_STCK_GRD
* Keeping segment shares constant
EQ_SEG_GRD


**EMISSION and ENERGY MODELS incl OBJ. FUNCTION --------------------------------------

* Objective function
EQ_TOTC
* Calculation of emissions from all vehicle classes per year
EQ_VEH_TOTC
*producton emissions
EQ_VEH_PROD_TOTC
*operation emissions
EQ_VEH_OPER_TOTC
*eolt emissions
EQ_VEH_EOLT_TOTC
*EQ_CHECK_ADD
;

*-----------------------------------------------------------------------------------
*
* Model Definition  p.t 3 : Model Equations

*-----------------------------------------------------------------------------------


***VEHICLE STOCK MODEL  ------------------------------------------------------------

***  Initiate stock in first year ----------------------------

*removals
EQ_STCK_REM_T1(tec,seg,inityear,age)..                 VEH_STCK_REM(tec,seg,inityear,age) =e=  VEH_STCK_TOT(inityear) * VEH_STCK_INT_TEC(tec) * VEH_LIFT_PDF(age) * VEH_LIFT_MOR(age) * VEH_STCK_INT_SEG(seg);

*stock additions
*-- Makes assumption that retired segments are replaced by the same segment
*EQ_STCK_ADD_T1(seg,inityear,age)$(ord(age)=1)..         sum(tec, VEH_STCK_ADD(tec,seg,inityear,age)) =e=  sum((tec,agej), VEH_STCK_REM(tec,seg,inityear,agej));
EQ_STCK_ADD_T1(seg,inityear,age)$(ord(age)=1)..         sum(tec, VEH_STCK_ADD(tec,seg,inityear,age)) =e=  sum((tec,agej), VEH_STCK_REM(tec,seg,inityear,agej))+ (VEH_STCK_DELTA(inityear)*VEH_STCK_INT_SEG(seg));

EQ_STCK_ADD_T11('2000')..              VEH_STCK_DELTA('2000') =e= 3637119.13333321;

*EQ_STCK_ADD_T11(seg,'2000',age)$(ord(age)=1)..              sum(tec,VEH_STCK_ADD(tec,seg,'2000',age)) =e= sum((tec,agej),VEH_STCK_REM(tec,seg,'2000',agej))+3637119.13333321;

*stock
EQ_STCK_BAL_T1(tec,seg,inityear,age)..                VEH_STCK(tec,seg,inityear,age) =e=  VEH_STCK_TOT(inityear) * (VEH_STCK_INT_TEC(tec) * VEH_LIFT_PDF(age) * VEH_STCK_INT_SEG(seg));

EQ_STCK_DELTA_T1(inityear)$(ord(inityear)>1)..                              VEH_STCK_DELTA(inityear)  =e=  VEH_STCK_TOT(inityear) - VEH_STCK_TOT(inityear-1);


***  Main Model -----------------------------------------------
EQ_STCK_DELTA(optyear)$(ord(optyear)>1)..                              VEH_STCK_DELTA(optyear)  =e=  VEH_STCK_TOT(optyear) - VEH_STCK_TOT(optyear-1);

*removals
*assumes equal removal across technologies and segments
EQ_STCK_REM(tec,seg,optyear,age)$(ord(optyear)>1)..                    VEH_STCK_REM(tec,seg,optyear,age) =e= VEH_STCK(tec,seg,optyear-1,age-1) * VEH_LIFT_MOR(age-1);

*stock additions
EQ_STCK_ADD(optyear,age)$(ord(optyear)>1 and ord(age)=1)..             sum((tec,seg), VEH_STCK_ADD(tec,seg,optyear,age) - sum(agej, VEH_STCK_REM(tec,seg,optyear,agej))) =e= VEH_STCK_DELTA(optyear);
*EQ_STCK_ADD(optyear)..                                              VEH_TOT_ADD(optyear) - sum((tec,seg,agej), VEH_STCK_REM(tec,seg,optyear,agej)) =e= VEH_STCK_DELTA(optyear);

** make into veh_add_tot?

*EQ_STCK_ADD(optyear,age)$(ord(optyear)>1 and ord(age)=1)..            sum(seg, sum(tec, VEH_STCK_ADD(tec,seg,optyear,age)) - sum((agej,tec), VEH_STCK_REM(tec,seg,optyear,agej))) =e= VEH_STCK_DELTA(optyear);
*EQ_STCK_ADD(optyear,age)$(ord(optyear)>1 and ord(age)=1)..             sum((tec,seg), VEH_STCK_ADD(tec,seg,optyear,age)) =e= VEH_STCK_DELTA(optyear) + sum((tec,seg,agej), VEH_STCK_REM(tec,seg,optyear,agej));
*EQ_STCK_ADD(seg,optyear,age)$(ord(optyear)>1 and ord(age)=1)..             sum((tec), VEH_STCK_ADD(tec,seg,optyear,age)) =e= VEH_STCK_DELTA(optyear) + sum((tec,agej), VEH_STCK_REM(tec,seg,optyear,agej));

*calculating vehicle stock in a given year
EQ_STCK_BAL(tec,seg,optyear,age)$(ord(optyear)>1)..                    VEH_STCK(tec,seg,optyear,age)  =e=  VEH_STCK(tec,seg,optyear-1,age-1) + VEH_STCK_ADD(tec,seg,optyear,age) - VEH_STCK_REM(tec,seg,optyear,age);

*-----calculate segment market shares-----
*** What works:
* 1. original EQ_STCK_ADD, EQ_SEG_GRD with VEH_TOT_ADD (with EQ_TOT_ADD commented out)
* 2. original EQ_STCK_ADD, EQ_TOT_ADD and EQ_SEG_GRD with =l= and VEH_STCK_ADD

* Calculate total addition to stock independent of technology and segment,
*EQ_TOT_ADD(year)..                                    VEH_TOT_ADD(year) =e= sum((tec,seg,age), VEH_STCK_ADD(tec,seg,year,age));
*EQ_TOT_REM(year)..                                                      VEH_TOT_REM(year) =e= sum((tec,seg,age),VEH_STCK_REM(tec,seg,year,age));
* summing the number of vehicles in fleet as check.
EQ_STCK_CHK(year)..                                                    VEH_STCK_TOT_CHECK(year) =e= sum((tec,seg,age), VEH_STCK(tec,seg,year,age));

*** Constraints -----------------------------------------------------------------------

* stock additions by technology
EQ_STCK_GRD(tec,seg,optyear,age)$(ord(optyear)>1 and ord(age)=1)..     VEH_STCK_ADD(tec,seg,optyear,age) =l= ((1 + VEH_ADD_GRD('IND',tec)) * VEH_STCK_ADD(tec,seg,optyear-1,age)) + 1e5;

* Segment share constraint (keep segment shares constant over analysis period)
*** This works and I have no idea why. (EQ_TOT_ADD must be removed for this to work)
*EQ_SEG_GRD(seg,optyear,age)$(ord(optyear)>1 and ord(age)=1)..          sum(tec,VEH_STCK_ADD(tec,seg,optyear,age)) =e= VEH_STCK_INT_SEG(seg)* VEH_TOT_ADD(optyear,age);

*EQ_CHECK_ADD(year)..                                                    check_add_tot(year) =e= sum((tec,seg,age),VEH_STCK_ADD(tec,seg,year,age));

*** This causes infeasibility in the solution (but otherwise works, i.e., keeps segment shares constant).
*EQ_SEG_GRD(seg,optyear,age)$(ord(optyear)>1 and ord(age)=1)..          sum(tec,VEH_STCK_ADD(tec,seg,optyear,age)) + slack_VEH_ADD(seg,optyear,age) =e= VEH_STCK_INT_SEG(seg) * sum((tec,segj), VEH_STCK_ADD(tec,segj,optyear,age));

*EQ_SEG_GRD(seg,optyear,age)$(ord(optyear)>1 and ord(age)=1)..          sum(tec,VEH_STCK_ADD(tec,seg,optyear,age)) =e= VEH_SEG_SHR(seg) * sum((tec,segj), VEH_STCK_ADD(tec,segj,optyear,age));


EQ_SEG_GRD(seg,optyear,age)$(ord(optyear)>1 and ord(age)=1)..          sum(tec,VEH_STCK_ADD(tec,seg,optyear,age)) =l= VEH_STCK_INT_SEG(seg) * sum((tec,segj), VEH_STCK_ADD(tec,segj,optyear,age));

*EQ_SEG_GRD(seg,optyear)$(ord(optyear)>1)..                          sum((tec,age), VEH_STCK(tec,seg,optyear,age)) =e= VEH_STCK_INT_SEG(seg)*sum((tec,segj,age), VEH_STCK(tec,segj,optyear,age));

*EQ_SEG_GRD(seg,optyear,age)$(ord(optyear)>1 and ord(age)=1)..          sum(tec,VEH_STCK_ADD(tec,seg,optyear,age)) =l= 1.2*sum((tec,segj),VEH_STCK_ADD(tec,segj,optyear-1,age));


*** EMISSION and ENERGY MODELS incl OBJ. FUNCTION ------------------------------------
* Objective function
EQ_TOTC..                                   TOTC =e= SUM((tec,seg,year), VEH_TOTC(tec,seg,year));

* Calculation of emissions from all vehicle classes per year
EQ_VEH_TOTC(tec,seg,year)..                  VEH_TOTC(tec,seg,year) =e= VEH_PROD_TOTC(tec,seg,year) + VEH_OPER_TOTC(tec,seg,year) + VEH_EOLT_TOTC(tec,seg,year);
*int_tec
EQ_VEH_PROD_TOTC(tec,seg,year)..             VEH_PROD_TOTC(tec,seg,year) =e= sum( (agej)$(ord(agej)=1), VEH_STCK_ADD(tec,seg,year,agej)*VEH_PROD_CINT(tec,seg,year));
EQ_VEH_OPER_TOTC(tec,seg,year)..             VEH_OPER_TOTC(tec,seg,year) =e= sum( (agej,enr,prodyear), VEH_STCK(tec,seg,year,agej) * VEH_OPER_CINT(tec,enr,seg,prodyear) * ENR_VEH(enr,tec)*VEH_PAY(prodyear,agej,year) * VEH_OPER_DIST(year)) / 1000;
EQ_VEH_EOLT_TOTC(tec,seg,year)..             VEH_EOLT_TOTC(tec,seg,year) =e= sum( (agej), VEH_STCK_REM(tec,seg,year,agej))*VEH_EOLT_CINT(tec,seg,year);


*** Convert VEH_STCK to include cohort for clearer figures ------------------------------------------
*** Doesn't work. 
*EQ_STCK_COHORT(tec,seg,prodyear,agej,year)..    VEH_STCK_CHRT(tec,seg,prodyear,agej,year) =e= VEH_STCK(tec,seg,year,agej)*VEH_PAY(prodyear,agej,year);


*-----------------------------------------------------------------------------------
*
* Model Execution  p.t 1 : Model Definition and Options
*
*-----------------------------------------------------------------------------------

* Defining Name of Model(s) and what Equations are used in each Model

MODEL EVD4EUR_Basic
/ALL/
;
* Defining model run options and solver

*OPTION RESLIM = 2000000;
*OPTION NLP = CONOPT;
*OPTION THREADS = 40;

*OPTION limrow = 0;
*OPTION limcol = 0;
*OPTION solprint = off;
*OPTION sysout = off;

*-----------------------------------------------------------------------------------
*
* Model Execution
*
*-----------------------------------------------------------------------------------

*EVD4EUR_Basic.optfile = 1;
SOLVE EVD4EUR_Basic USING LP MINIMIZING TOTC;

*DISPLAY VEH_STCK_TOT
*DISPLAY VEH_STCK_TOT_CHECK.l


*execute_unload 'EVD4EUR_wo_paramdefs.gdx'
