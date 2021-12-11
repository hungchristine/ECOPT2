***
* EVD4EUR Model
* =============
*
* The EVD4EUR model minimizes total CO2 emissions while satisfying
* transport demand via light duty vehicles. Manufacturing and supply
* chain constraints are considered.
*
*-----------------------------------------------------------------------------------


* Slack variables for debugging; remove asterisk (*) from quotes to activate slack variable
$IF NOT SET SLACK_ADD                 $SETGLOBAL SLACK_ADD "*"
$IF NOT SET SLACK_NEW_BATT_CAP        $SETGLOBAL SLACK_NEW_BATT_CAP "*"
$IF NOT SET SLACK_VIRG_MAT            $SETGLOBAL SLACK_VIRG_MAT "*"

SETS
year            total span of years including production before intialization period - superset
modelyear(year) model years (2000-2050)
optyear(year)   years for optimization (2020-2050) 
inityear(year)  years for initialization (2000-2020)
age             age of vehicle
tec             technology - superset
newtec(tec)    new technologies to replace incumbent
enr             energy carrier
reg             region or country group - superset
fleetreg(reg)   model regions of BEV operation
seg             segment or size class
mat_prod        producers or sources for all critical materials 
mat_cat        critical material categories
sigvar          variables for sigmoid equations
veheq           equations for vehicle parameters
demeq           equations for demand parameters
grdeq           parameters for gradient of change (fleet additions) - individual (IND) for each tech or related to all tech (ALL)
mat(mat_cat, mat_prod)      critical material
*    /
*    Li.(Li1,Li2)
*    Co.(Co1,Co2)
*    /
;
SINGLETON SETS
new(age)
;

*---- ABBREIVATIONS USED *-----------------------------------------------------------------------------------
* PROD = Production
* OPER = Operation
* ENIT = Enerqy intensity
* CINT = CO2-eq intensity
* CNST = b in y = ax + b

** Load sets defined in Python class
* These lines must be uncommented if specific input .gdx is not specified, e.g., $gdxin <filepath>_input.gdx
* By default uses pyGAMS_input.gdx if gdx filename not provided by Python
* Comment out these two lines if running directly in GAMS:
$if not set gdxincname $GDXIN C:\Users\chrishun\Box Sync\YSSP_temp\pyGAMS_input.gdx 
$if set gdxincname $GDXIN %gdxincname%
*$abort 'no include file name for data file provided'

* This .gdx for running via Python-defined sets
*$gdxin C:\Users\chrishun\Box Sync\YSSP_temp\pyGAMS_input.gdx

* This .gdx is for troubleshooting
*$GDXIN 'troubleshooting_params'

$LOAD year
$LOAD modelyear
$LOAD optyear

$LOAD inityear
$LOAD age
$LOAD new
$LOAD tec
$LOAD newtec
$LOAD enr
$LOAD reg
$LOAD fleetreg
$LOAD seg
$LOAD mat_cat
$LOAD mat_prod
$LOAD mat

$LOAD sigvar
$LOAD veheq
$LOAD demeq
$LOAD grdeq

$GDXIN
;
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
YEAR_PAR(year)                       year
VEH_PARTAB(veheq,tec,seg,sigvar)     variables for each tech and veh equation

***ENERGY (ELECTRICITY GENERATION and FUEL) ------------------------------------------
ENR_CINT(enr,reg,year)               CO2 intensity of the regional energy mixes            [kg CO2-eq pr kwh]

***ENERGY and VEHICLE TECHONLOGY COMBINATIONS ----------------------------------------
ENR_VEH(enr,tec)                     feasible combinations of vehicle technology and energy (fuel)


***All VEHICLES ----------------------------------------------------------------------

**PRODUCTION
VEH_PROD_EINT(tec,seg,prodyear)        Electricity intensity of vehicle prod                [kwh el required per vehicle produced]
VEH_PROD_CINT_CSNT(tec,seg,prodyear)   Constant term for CO2 int. of vehicle production     [t CO2-eq per vehicle produced]
VEH_PROD_CINT(tec,seg,prodyear)        CO2 intensity of vehicle production                  [t CO2-eq per vehicle produced]

**OPERATION
VEH_OPER_EINT(tec,seg,prodyear)                               Energy intensity of vehicle operation   [kwh per km]
VEH_OPER_CINT(tec,enr,seg,fleetreg,age,modelyear,prodyear)    CO2 intensity of vehicle operation     [t CO2 per km]
OPER_TRIAL(tec,enr,seg,prodyear,age,modelyear)
**EOL
VEH_EOLT_CINT(tec,seg,year)            CO2 intensity of ICE vehicle EOL                     [t CO2-eq per vehicle in EOL treatment]

** FLEET -------------------------------------------------------------------------------
** INITIAL STOCK ------------
BEV_CAPAC(seg)                   Correspondence of battery capacities used in each segment  [kWh]

**DEMAND --------------------
VEH_STCK_TOT(year,reg)           Number of vehicles                                     [#]
VEH_OPER_DIST(year)              Annual driving distance per vehicles                   [km]

** LIFETIME -----------------
VEH_LIFT_PDF(age)                Age PDF
VEH_LIFT_CDF(age)                Age CDF Share of scrapping for a given age - e.g % in given age class i scrapped
VEH_LIFT_AGE(age)                Age distribution = 1 - CDF
VEH_LIFT_MOR(age)                Age distribution = 1 - CDF

** COMPOSITION --------------
VEH_PAY(prodyear,age,year)       Correspondence between a vehicle production year and its age (up to 20) in a given year
VEH_STCK_INT_TEC(tec)            Initial share of vehicles in stock tech
VEH_STCK_INT_SEG(seg)            Initial stock distribution by segment
VEH_STCK_INT(tec,seg,reg,age)    Initial size of stock of vehicles by age cohort and segment

** CONSTRAINTS -------
VEH_ADD_GRD(grdeq,newtec)       Parameter for gradient of change constraint (fleet additions) - individual (IND) for each tech or related to all tech (ALL)
MANUF_CNSTRNT(year)              Annual manufacturing capacity (for batteries destined for Europe) [GWh]
MAT_CONTENT(year,mat_cat)       Critical material content per kWh by production year [kg kWh-1]
RECOVERY_PCT(year,mat_cat)      Recovery of critical materials from battery recycling processes in wt%
VIRG_MAT_SUPPLY(year,mat_prod)   Primary critical material resources available in a given year [in kt]
MAT_CINT(year,mat_prod)          Carbon intensity of each material by source [kg CO2e kg mat]

** POST-PROCESSING --------
TOT_BEVS(year)                   Total number of BEVs in Europe
TOT_BATT_MANUF(year)             Total battery capacity required by year [MWh]
TOT_BATT_RECYCLED(year)          Total battery capacity retired each year [kWh]
MAT_REQ_TOT(year, mat_cat)      Total resources required by year [kg]
*MAT_RECYCLED(year, mat_cat)     Total critical materials recycled per year [kg]
*MAT_REQ_VIRG(year, mat_cat)     Total critical material from primary resources required [kg]
VEH_STCK_TOT_CHECK(modelyear)
VEH_STCK_COHORT(tec,seg,fleetreg,prodyear,age,modelyear)       Total stock by technology segment region and cohort
VEH_OPER_COHORT(tec,seg,fleetreg,prodyear,modelyear,age)       Total fleet operating emissions by technology segment region and cohort
VEH_EOLT_COHORT(tec,seg,fleetreg,prodyear,modelyear,age)       Total end-of-life emissions by technology segment region and cohort
TOT_OPER_EINT(tec,seg,fleetreg,modelyear)
*ANN_TOTC(modelyear)              Total CO2 emissions from LDVs by year                              [t CO2-eq]
VEH_TOT_ADD(fleetreg, year)      Total vehicles added by region
VEH_TOT_REM(fleetreg, year)
VEH_STCK_CHK(fleetreg,year)
VEH_STCK_GRD(tec,seg,fleetreg,optyear)
BAU_PROD(modelyear)
BAU_OPER(modelyear)
BAU_EOL(modelyear)
BAU_EMISSIONS(modelyear)

test(mat_cat, modelyear)
test2(age, prodyear, modelyear)
FILT_ADD(tec,seg,fleetreg,year,age)
FILT_STCK(tec,seg,fleetreg,year,age)
FILT_REM(tec,seg,fleetreg,year,age)
;

* Load in parameter values from .gdx file [dummy data]

* Default input .gdx built in Python
*$gdxin C:\Users\chrishun\Box Sync\YSSP_temp\pyGAMS_input.gdx 

* Sample input GDX as trial for introducing the region set
*$gdxin C:\Users\chrishun\Box Sync\YSSP_temp\Cleanup April 2020\run_def_baseline_def_def_def_def_aggr_iTEM2-Base2019-10-23T07_40_input.gdx

* Comment out these lines if running directly in GAMS

$if not set gdxincname $GDXIN C:\Users\chrishun\Box Sync\YSSP_temp\pyGAMS_input.gdx
$if set gdxincname $GDXIN %gdxincname%

$LOAD YEAR_PAR
$LOAD VEH_PARTAB

$LOAD ENR_VEH
$LOAD ENR_CINT

$LOAD VEH_STCK_TOT
$LOAD VEH_OPER_DIST

$LOAD VEH_LIFT_PDF
$LOAD VEH_LIFT_CDF
$LOAD VEH_LIFT_AGE
$LOAD VEH_LIFT_MOR

$LOAD VEH_PAY
$LOAD VEH_STCK_INT_TEC
$LOAD VEH_STCK_INT_SEG

$LOAD BEV_CAPAC
$LOAD VEH_ADD_GRD
$LOAD MANUF_CNSTRNT

$LOAD MAT_CONTENT
$LOAD RECOVERY_PCT
$LOAD MAT_CINT
$LOAD VIRG_MAT_SUPPLY

$OFFMULTI
$GDXIN
;


** try removing this??
*ENR_CINT('ELC', reg, inityear) = ENR_CINT('ELC', reg, '2019');

*----- Production-related emissions
* Assume constant for all regions for now
VEH_PROD_EINT(tec,seg,prodyear) = genlogfnc(VEH_PARTAB('PROD_EINT',tec,seg,'A'),VEH_PARTAB('PROD_EINT',tec,seg,'B'),VEH_PARTAB('PROD_EINT',tec,seg,'r'),YEAR_PAR(prodyear),VEH_PARTAB('PROD_EINT',tec,seg,'u'));

VEH_PROD_CINT_CSNT(tec,seg,prodyear) = genlogfnc(VEH_PARTAB('PROD_CINT_CSNT',tec,seg,'A'),VEH_PARTAB('PROD_CINT_CSNT',tec,seg,'B'),VEH_PARTAB('PROD_CINT_CSNT',tec,seg,'r'),YEAR_PAR(prodyear),VEH_PARTAB('PROD_CINT_CSNT',tec,seg,'u'));

* only production emissions from 2000 onwards are relevant despite cohorts going back to 1972
VEH_PROD_CINT(tec,seg,prodyear)$(ord(prodyear)>28) = VEH_PROD_CINT_CSNT(tec,seg,prodyear) + VEH_PROD_EINT(tec,seg,prodyear)*ENR_CINT('elc','prod',prodyear)/1000;

*----- Operation phase emissions
* Assume constant for all regions for now
VEH_OPER_EINT(tec,seg,prodyear) = genlogfnc(VEH_PARTAB('OPER_EINT',tec,seg,'A'),VEH_PARTAB('OPER_EINT',tec,seg,'B'),VEH_PARTAB('OPER_EINT',tec,seg,'r'),YEAR_PAR(prodyear),VEH_PARTAB('OPER_EINT',tec,seg,'u'));

VEH_OPER_CINT(tec,enr,seg,fleetreg,age,modelyear,prodyear)$(ENR_VEH(enr,tec)) = VEH_OPER_EINT(tec,seg,prodyear)*VEH_PAY(prodyear,age,modelyear)*(ENR_CINT(enr,fleetreg,modelyear)/1000);

*----- End-of-life phase emissions
*10.10.2019:  Used to prodyear, rather than modelyear
* Assume constant for all regions for now
VEH_EOLT_CINT(tec,seg,modelyear) = genlogfnc(VEH_PARTAB('EOLT_CINT',tec,seg,'A'),VEH_PARTAB('EOLT_CINT',tec,seg,'B'),VEH_PARTAB('EOLT_CINT',tec,seg,'r'),YEAR_PAR(modelyear),VEH_PARTAB('EOLT_CINT',tec,seg,'u'));


VEH_STCK_INT(tec,seg,fleetreg,age) = (VEH_STCK_INT_TEC(tec)*VEH_LIFT_PDF(age)*VEH_STCK_INT_SEG(seg))*VEH_STCK_TOT('2000',fleetreg);
*mat_content(year,'Li') = 0.11;
*mat_content(year,'Co') = 0.05;

*----- TEMPORARY ASSIGNMENT OF LITHIUM RECYCLING PARAMETERS AS CONSTANTS
$ontext
MAT_CONTENT(modelyear, 'Li') = 0.111;
MAT_CONTENT(modelyear, 'Co') = 0.05;
RECOVERY_PCT(modelyear, mat) = 0.5;
VIRG_MAT(optyear, 'Li') = 5e20;
VIRG_MAT(optyear, 'Co') = 3e20;
$offtext

*-----------------------------------------------------------------------------------
*
* Model Definition  p.t 1 : Variable Definitions
*
*-----------------------------------------------------------------------------------


***FREE VARIABLES ------------------------------------------------------------------
* Objective value to be minimized must be a free variable


FREE VARIABLES
TOTC                                        Total CO2 emissions for the whole system over the whole period
TOTC_OPT                                    Total CO2 emissions for the whole system over optimization period
VEH_STCK_DELTA(year,fleetreg)               Net change in stock from one year to the next
TOT_PRIMARY_MAT(year, mat_cat)             Total primary resources required by year [kg]

SLACK_ADD(tec,seg,fleetreg,year,age)        Slack variable for additions to stock
SLACK_TEC_ADD(newtec, seg,fleetreg,year,age)
SLACK_SEG_ADD(seg,fleetreg,year)
SLACK_RECYCLED_BATT(year, fleetreg, age)
SLACK_VIRG_MAT(year, mat_prod)
SLACK_NEW_BATT_CAP(year)

chck(tec,seg, fleetreg, modelyear) 
;

POSITIVE VARIABLES
VEH_STCK(tec,seg,fleetreg,year,age)              Number of vehicles of a given age in a given year
VEH_STCK_REM(tec,seg,fleetreg,year,age)          Number of vehicles of a given age retired in a given year
VEH_TOTC(tec,seg,fleetreg,year)                  Total CO2 emissions of vehicles per year by technology              [t CO2-eq]
VEH_STCK_ADD(tec,seg,fleetreg,year,age)          Stock additions (new car sales)

VEH_PROD_TOTC(tec,seg,fleetreg,year)             Total CO2 emissions from production of vehicles per year            [t CO2-eq]
VEH_OPER_TOTC(tec,seg,fleetreg,year)             Total CO2 emissions from operations of vehicles per year            [t CO2-eq]
VEH_EOLT_TOTC(tec,seg,fleetreg,year)             Total CO2 emissions from vehicle end of life treatment per year     [t CO2-eq]

RECYCLED_BATT(year,fleetreg, age)           Total battery capacity sent to recycling per year                        [kWh]
RECYCLED_MAT(year, mat_cat)                Materials recovered from recycled batteries
MAT_REQ(year,mat_cat)                      Total amount of critical materials needed for new vehicles               [kg]

MAT_MIX(year, mat_prod)                     Production mixes for virgin materials                                    [kg]
MAT_CO2(year, mat_prod)                     Total CO2 emissions from virgin material production per year             [t CO2e kg^-1]
;



*-----------------------------------------------------------------------------------
*
* Model Definition  p.t 2 : Model Equation Declarations
*
*-----------------------------------------------------------------------------------

EQUATIONS

***VEHICLE STOCK MODEL  ------------------------------------------------------------

***  Fleet model
EQ_STCK_DELTA           Net total stock change
EQ_STCK_REM             Removal from stock
EQ_STCK_ADD             Addition to stock
EQ_STCK_BAL             Fleet balance constraint

*** Constraint equations
EQ_STCK_GRD             Technology uptake constraint
EQ_SEG_GRD              Segment share constraint

EQ_NEW_BATT_CAP         Battery manufacturing constraint

EQ_RECYCLED_BATT
EQ_MAT_REQ
EQ_RECYCLED_MAT
EQ_MAT_TOT_PRIMARY
EQ_MAT_SUPPLY           Critical material balance constraint - total supply   
EQ_VIRG_MAT_SUPPLY      Critical material primary supply constraint

**EMISSION and ENERGY MODELS incl OBJ. FUNCTION --------------------------------------

EQ_OBJ                  Objective function - total fleet lifecycle CO2 over optimization period
EQ_VEH_TOTC             Calculation of emissions from all vehicle classes per year
EQ_VEH_PROD_TOTC        Total production emissions including from materials
EQ_MAT_TOTC             Primary material extraction and processing emissions
EQ_VEH_OPER_TOTC        Total vehicle operating emissions
EQ_VEH_EOLT_TOTC        Total vehicle end of life emissions
;

*-----------------------------------------------------------------------------------
*
* Model Definition  p.t 3 : Model Equations

*-----------------------------------------------------------------------------------


***VEHICLE STOCK MODEL  ------------------------------------------------------------
* First, initialize first year (boundary condition)
VEH_STCK.fx(tec,seg,fleetreg,modelyear,age)$(ord(modelyear)=1) = VEH_STCK_TOT(modelyear, fleetreg) * VEH_LIFT_AGE(age) * VEH_STCK_INT_TEC(tec)  * VEH_STCK_INT_SEG(seg);
* try below with age+1 on LHS to avoid weird offset thing?
VEH_STCK_REM.fx(tec,seg,fleetreg,modelyear,age)$(ord(modelyear)=1) = VEH_STCK.l(tec,seg,fleetreg,modelyear,age-1) * VEH_LIFT_MOR(age-1);
VEH_STCK_ADD.fx(tec,seg,fleetreg,modelyear,new)$(ord(modelyear)=1) = sum(age, VEH_STCK_REM.l(tec,seg,fleetreg,modelyear,age));
* Do not add "new" vehicles with age (e.g., used cars from other regions)
VEH_STCK_ADD.fx(tec,seg,fleetreg,modelyear,age)$(ord(age)>1) = 0;

* Initiate stock in remaining initialization period
loop(modelyear $ (ord(modelyear)>1 and ord(modelyear) <= card(inityear)),
VEH_STCK_DELTA.fx(modelyear, fleetreg) = VEH_STCK_TOT(modelyear, fleetreg) - VEH_STCK_TOT(modelyear-1, fleetreg);
* calculate stock removals as per survival curves
VEH_STCK_REM.fx(tec,seg,fleetreg,modelyear,age) = VEH_STCK.l(tec,seg,fleetreg,modelyear-1,age-1) * VEH_LIFT_MOR(age-1);
* in initialization period, assume that new vehicles maintain the status quo of tec and segment shares
VEH_STCK_ADD.fx(tec,seg,fleetreg,modelyear,new) = sum(age, VEH_STCK_REM.l(tec,seg,fleetreg,modelyear,age)) + (VEH_STCK_DELTA.l(modelyear,fleetreg) * VEH_STCK_INT_TEC(tec) * VEH_STCK_INT_SEG(seg));
VEH_STCK.fx(tec,seg,fleetreg,modelyear,age) = VEH_STCK.l(tec,seg,fleetreg,modelyear-1,age-1) - VEH_STCK_REM.l(tec,seg,fleetreg,modelyear,age) + VEH_STCK_ADD.l(tec,seg,fleetreg,modelyear,age);
);


***  Main Model -----------------------------------------------
EQ_STCK_DELTA(modelyear,fleetreg)$(ord(modelyear)>card(inityear))..             VEH_STCK_DELTA(modelyear,fleetreg)  =e=  VEH_STCK_TOT(modelyear,fleetreg) - VEH_STCK_TOT(modelyear-1,fleetreg);

EQ_STCK_REM(tec,seg,fleetreg,modelyear,age)$(ord(modelyear)>card(inityear))..    VEH_STCK_REM(tec,seg,fleetreg,modelyear,age) =e= VEH_STCK(tec,seg,fleetreg,modelyear-1,age-1)*VEH_LIFT_MOR(age-1);

EQ_STCK_ADD(modelyear,fleetreg)$(ord(modelyear)>card(inityear))..               sum((tec,seg), VEH_STCK_ADD(tec,seg,fleetreg,modelyear,new)) =e= sum((tec,seg,age), VEH_STCK_REM(tec,seg,fleetreg,modelyear, age)) + VEH_STCK_DELTA(modelyear, fleetreg);


EQ_STCK_BAL(tec,seg,fleetreg,modelyear,age)$(ord(modelyear) > card(inityear))..      VEH_STCK(tec,seg,fleetreg,modelyear,age)  =e= VEH_STCK(tec,seg,fleetreg,modelyear-1,age-1) + VEH_STCK_ADD(tec,seg,fleetreg,modelyear,age) - VEH_STCK_REM(tec,seg,fleetreg,modelyear,age);


*-----calculate segment market shares-----
*** What works:
* 1. original EQ_STCK_ADD, EQ_SEG_GRD with VEH_TOT_ADD (with EQ_TOT_ADD commented out)
* 2. original EQ_STCK_ADD, EQ_TOT_ADD and EQ_SEG_GRD with =l= and VEH_STCK_ADD
                                        

*** Constraints -----------------------------------------------------------------------


* B - Battery manufacturing constraint;

* total capacity added per year must be less than the battery manufacturing capacity
* MANUF_CNSTRNT input is in GWh; BEV_CAPAC is in kWh
* dummy manufacturing constraint to trigger on dummy fleet...
EQ_NEW_BATT_CAP(optyear)..                            MANUF_CNSTRNT(optyear)*1e6 =g= sum((seg, fleetreg),VEH_STCK_ADD('BEV',seg,fleetreg,optyear,new)*BEV_CAPAC(seg))
%SLACK_NEW_BATT_CAP% - SLACK_NEW_BATT_CAP(optyear)
;


* E - Technology uptake constraint
* incumbent technology excluded
EQ_STCK_GRD(newtec,seg,fleetreg,modelyear)$(ord(modelyear)>card(inityear))..     VEH_STCK_ADD(newtec,seg,fleetreg,modelyear,new)
    =l= ((1 + VEH_ADD_GRD('IND', newtec)) * VEH_STCK_ADD(newtec,seg,fleetreg,modelyear-1,new)) + 5000
%SLACK_ADD% + SLACK_TEC_ADD(newtec,seg,fleetreg,modelyear,new)
;

* F - Segment share constraint (segments kept constant)
* try to remove - allow for ICEV smart cars (e.g.,) to be replaced by a BEV Model X...
EQ_SEG_GRD(seg,fleetreg,modelyear)$(ord(modelyear)>card(inityear))..          sum(tec, VEH_STCK_ADD(tec,seg,fleetreg,modelyear,new))
    =e= VEH_STCK_INT_SEG(seg) * sum((tec,segj), VEH_STCK_ADD(tec,segj,fleetreg,modelyear,new));


*------ Lithium/elemental resource availability constraints;

* Calculate amount of batteries retired each year in kWh
EQ_RECYCLED_BATT(optyear,fleetreg, age)..             RECYCLED_BATT(optyear,fleetreg, age) =e= sum((seg), VEH_STCK_REM('BEV',seg,fleetreg,optyear, age)* BEV_CAPAC(seg))
*%SLACK_RECYCLED_BATT% + SLACK_RECYCLED_BATT(optyear, fleetreg, age)
;

* Calculate amount of materials recovered from end-of-life batteries for recycling (i.e., to new batteries)
EQ_RECYCLED_MAT(optyear,mat_cat)..                   RECYCLED_MAT(optyear, mat_cat) =e= sum((prodyear,age), sum((fleetreg), RECYCLED_BATT(optyear, fleetreg, age))*VEH_PAY(prodyear, age, optyear)* MAT_CONTENT(optyear, mat_cat))*RECOVERY_PCT(optyear, mat_cat)
*/1000
;
* Calculate total amount of materials available (virgin and recycled)
*eq_tot_mats(mat_cat,year)..          sum(mat_prod$mat(mat_cat,mat_prod), mat_mix(year,mat_prod)) =e= battery(mat_cat,year);

* Material supply balance
* Total amount of material required calculated from new vehicles entering the market, in kg
EQ_MAT_REQ(optyear, mat_cat)..                       MAT_REQ(optyear, mat_cat) =e= sum((seg, fleetreg), VEH_STCK_ADD('BEV',seg,fleetreg,optyear,new)*BEV_CAPAC(seg)) * MAT_CONTENT(optyear, mat_cat)
*/1000
%SLACK_NEW_BATT_CAP% + SLACK_NEW_BATT_CAP(optyear) * MAT_CONTENT(optyear, mat_cat)/1000
;

* Amount of virgin + recycled material to produce new batteries
EQ_MAT_SUPPLY(optyear, mat_cat)..                    TOT_PRIMARY_MAT(optyear, mat_cat) + RECYCLED_MAT(optyear, mat_cat) =e= MAT_REQ(optyear, mat_cat);

EQ_MAT_TOT_PRIMARY(mat_cat, optyear)..               TOT_PRIMARY_MAT(optyear, mat_cat) =e= sum(mat_prod$mat(mat_cat, mat_prod), MAT_MIX(optyear, mat_prod));

* D - Critical material primary supply constraint
* demand of virgin resources from each source must be less than or equal to available supply
EQ_VIRG_MAT_SUPPLY(optyear,mat_prod)..                MAT_MIX(optyear, mat_prod)
                                                        =l= VIRG_MAT_SUPPLY(optyear, mat_prod)*1000
%SLACK_VIRG_MAT% + SLACK_VIRG_MAT(optyear, mat_prod)
;


$ontext
virgin material supply + recycled material must be greater than the material demand
EQ_MAT_REQ(optyear, mat)..                                VIRG_MAT(optyear, mat) + sum((prodyear,age), sum((reg), RECYCLED_BATT(optyear, reg, age))*VEH_PAY(prodyear, age, optyear)*MAT_CONTENT(prodyear, mat))*RECOVERY_PCT(optyear, mat) =g= sum((seg, reg), VEH_STCK_ADD('BEV',seg,reg,optyear,'0')*BEV_CAPAC(seg))*MAT_CONTENT(optyear, mat);
$offtext


*** EMISSION and ENERGY MODELS incl OBJ. FUNCTION -------------------------------------------------
* Objective function
EQ_OBJ..                                              TOTC_OPT =e= sum((tec,seg,fleetreg,optyear), VEH_TOTC(tec,seg,fleetreg,optyear));

* Calculation of emissions from all vehicle classes per year
EQ_VEH_TOTC(tec,seg,fleetreg,modelyear)..                  VEH_TOTC(tec,seg,fleetreg,modelyear) =e= VEH_PROD_TOTC(tec,seg,fleetreg,modelyear) + VEH_OPER_TOTC(tec,seg,fleetreg,modelyear) + VEH_EOLT_TOTC(tec,seg,fleetreg,modelyear);

* Calculate emissions from virgin materials. Currently assumes zero emissions for recycled materials.
EQ_MAT_TOTC(modelyear, mat_prod)..                         MAT_CO2(modelyear, mat_prod) =e= (MAT_MIX(modelyear, mat_prod)  * MAT_CINT(modelyear, mat_prod))/1000
%SLACK_VIRG_MAT% + SLACK_VIRG_MAT(modelyear, mat_prod)*1e6
                                                            ;

EQ_VEH_PROD_TOTC(tec,seg,fleetreg,modelyear)..             VEH_PROD_TOTC(tec,seg,fleetreg,modelyear) =e= VEH_STCK_ADD(tec,seg,fleetreg,modelyear,new)*VEH_PROD_CINT(tec,seg,modelyear) + sum(mat_prod, MAT_CO2(modelyear, mat_prod))
%SLACK_ADD% + SLACK_TEC_ADD(newtec,seg,fleetreg,modelyear,new)*VEH_PROD_CINT(tec,seg,modelyear)*1e6
;

EQ_VEH_OPER_TOTC(tec,seg,fleetreg,modelyear)..             VEH_OPER_TOTC(tec,seg,fleetreg,modelyear) =e= sum( (agej,enr,prodyear), VEH_STCK(tec,seg,fleetreg,modelyear,agej) * VEH_PAY(prodyear,agej,modelyear)* VEH_OPER_CINT(tec,enr,seg,fleetreg,agej,modelyear,prodyear) *  VEH_OPER_DIST(modelyear));

EQ_VEH_EOLT_TOTC(tec,seg,fleetreg,modelyear)..             VEH_EOLT_TOTC(tec,seg,fleetreg,modelyear) =e= sum( (agej), VEH_STCK_REM(tec,seg,fleetreg,modelyear,agej))*VEH_EOLT_CINT(tec,seg,modelyear)
;

chck.fx(newtec,seg, fleetreg, modelyear) = ((1 + VEH_ADD_GRD('IND',newtec)) * VEH_STCK_ADD.l(newtec,seg,fleetreg,modelyear-1,new));

execute_loadpoint 'EVD4EUR_Basic_p.gdx';

*-----------------------------------------------------------------------------------
*
* Model Execution  p.t 1 : Model Definition and Options
*
*-----------------------------------------------------------------------------------

* Defining name of model(s) and what equations are used in each model

MODEL EVD4EUR_Basic "default model run in normal mode" /ALL/
      seg_test "model without segment constraint" /EVD4EUR_Basic - EQ_SEG_GRD/
      tec_test "model without growth constraint" /EVD4EUR_Basic - EQ_STCK_GRD/
 
      no_mat "model with no critical material constraints" /EVD4EUR_Basic - EQ_RECYCLED_MAT - EQ_RECYCLED_BATT - EQ_MAT_REQ - EQ_MAT_TOT_PRIMARY - EQ_MAT_SUPPLY - EQ_VIRG_MAT_SUPPLY/

      manuf_test "model without manufacturing capacity constraint" /EVD4EUR_Basic - EQ_NEW_BATT_CAP/
      fleet_test "model without growth or manufacturing constraint" /tec_test - EQ_NEW_BATT_CAP/
      mat_test "model without material constraint" /EVD4EUR_Basic - EQ_MAT_SUPPLY/
      primary_mat_test "model without primary material supply constraint" /EVD4EUR_Basic - EQ_VIRG_MAT_SUPPLY/
      test_model /no_mat - EQ_SEG_GRD/
*      test /no_mat - EQ_STCK_GRD/
      no_constraints "model without constraints" /no_mat - EQ_SEG_GRD - EQ_STCK_GRD - EQ_NEW_BATT_CAP/
      
;
*      no_contr "model no constraints at all" /unit_test2 - EQ_STCK_ADD0 - EQ_STCK_GRD0/
*;

* Defining model run options and solver

*OPTION RESLIM = 2000000;
*OPTION THREADS = 40;

*OPTION sysout = off;
OPTION limrow = 0;
OPTION limcol = 0;
OPTION PROFILE = 0;
* set to PROFILE = 2 for debugging speed
OPTION Solprint = on;
OPTION savepoint = 1;
Option LP = CPLEX;
EVD4EUR_Basic.optfile = 1;

Scalar ms 'model status', ss 'solve status';

*-----------------------------------------------------------------------------------
*
* Model Execution
*
*-----------------------------------------------------------------------------------

SOLVE EVD4EUR_Basic USING LP MINIMIZING TOTC_OPT;
ms = EVD4EUR_Basic.modelstat;
ss = EVD4EUR_Basic.solvestat;

*SOLVE seg_test USING LP MINIMIZING TOTC_OPT;
*ms = seg_test.modelstat;
*ss = seg_test.solvestat;

*SOLVE tec_test USING LP MINIMIZING TOTC_OPT; 
*ms = tec_test.modelstat;
*ss = tec_test.solvestat;

*SOLVE fleet_test USING LP MINIMIZING TOTC_OPT;

*SOLVE mat_test USING LP MINIMIZING TOTC_OPT;
*ms = mat_test.modelstat;
*ss = mat_test.solvestat;

*SOLVE primary_mat_test USING LP MINIMIZING TOTC_OPT;
*SOLVE test_model USING LP MINIMIZING TOTC_OPT;

*SOLVE no_mat USING LP MINIMIZING TOTC_OPT;
*ms = no_mat.modelstat;
*ss = no_mat.solvestat;

*SOLVE manuf_test USING LP MINIMIZING TOTC_OPT;
*ms = manuf_test.modelstat;
*ss = manuf_test.solvestat;

*SOLVE no_constraints USING LP MINIMIZING TOTC_OPT;
*ms = no_constraints.modelstat;
*ss = no_constraints.solvestat;

*-----------------------------------------------------------------------------------
*
* Post-processing calculations
*
*-----------------------------------------------------------------------------------


TOT_BEVS(modelyear) = sum((seg, fleetreg, age), VEH_STCK.l('BEV', seg, fleetreg, modelyear,age)); 
* total capacity of batteries added by year in MWh
TOT_BATT_MANUF(modelyear) = sum((seg, fleetreg)$VEH_STCK_ADD.l('BEV', seg, fleetreg, modelyear, new), VEH_STCK_ADD.l('BEV',seg,fleetreg,modelyear,new)*BEV_CAPAC(seg))/1e6;
TOT_BATT_RECYCLED(modelyear) = sum((seg), sum((fleetreg, age), VEH_STCK_REM.l('BEV',seg,fleetreg,modelyear, age))* BEV_CAPAC(seg));
MAT_REQ_TOT(modelyear, mat_cat) = sum((seg, fleetreg), VEH_STCK_ADD.l('BEV',seg,fleetreg,modelyear,new)*BEV_CAPAC(seg)*MAT_CONTENT(modelyear,mat_cat));
*shares(inityear, fleetreg, tec,seg,age) = VEH_STCK_TOT(inityear,fleetreg) * (VEH_STCK_INT_TEC(tec) * VEH_LIFT_PDF(age)* VEH_STCK_INT_SEG(seg));
*(((VEH_STCK_TOT(inityear,fleetreg) * VEH_STCK_INT_TEC(tec) / tec.len) * VEH_LIFT_PDF(age)/age.len) * VEH_STCK_INT_SEG(seg)/seg.len);

VEH_STCK_GRD(newtec,seg,fleetreg,optyear) = ((1 + VEH_ADD_GRD('IND',newtec)) * VEH_STCK_ADD.l(newtec,seg,fleetreg,optyear-1,new));
VEH_TOT_ADD(fleetreg,modelyear) = sum((tec,seg), VEH_STCK_ADD.l(tec, seg, fleetreg, modelyear, new));
VEH_TOT_REM(fleetreg, modelyear) = sum((tec,seg,age), VEH_STCK_REM.l(tec, seg, fleetreg, modelyear, age));
VEH_STCK_CHK(fleetreg, modelyear) = sum((tec,seg,age), VEH_STCK.l(tec,seg,fleetreg, modelyear, age));

* summing the number of vehicles in fleet as check.
VEH_STCK_TOT_CHECK(modelyear) = sum((tec,seg,fleetreg,age), VEH_STCK.l(tec,seg,fleetreg,modelyear,age));
VEH_STCK_COHORT(tec,seg,fleetreg,prodyear,agej,modelyear) $ VEH_PAY(prodyear,agej,modelyear) = (VEH_STCK.l(tec,seg,fleetreg,modelyear,agej)*VEH_PAY(prodyear,agej,modelyear));

* total operation emissions by cohort and model year
VEH_OPER_COHORT(tec, seg, fleetreg, prodyear, modelyear ,age) = sum((enr), VEH_STCK.l(tec,seg,fleetreg,modelyear,age) * VEH_PAY(prodyear,age,modelyear)* VEH_OPER_CINT(tec,enr,seg,fleetreg,age,modelyear,prodyear) * VEH_OPER_DIST(modelyear));
VEH_EOLT_COHORT(tec,seg,fleetreg,prodyear,modelyear,age) = (VEH_STCK_REM.l(tec,seg,fleetreg,modelyear,age)* VEH_PAY(prodyear,age,modelyear)) * VEH_EOLT_CINT(tec,seg,modelyear);
TOT_OPER_EINT(tec,seg,fleetreg,modelyear) = sum((prodyear, age), VEH_STCK.l(tec,seg,fleetreg,modelyear,age) * VEH_PAY(prodyear,age,modelyear) * VEH_OPER_EINT(tec,seg,prodyear) * VEH_OPER_DIST(modelyear));

* calculate emissions for 0 electrification (business as usual)
BAU_PROD(modelyear) = sum(seg, sum((tec,fleetreg), VEH_STCK_ADD.l(tec, seg, fleetreg, modelyear, new)) * VEH_PROD_CINT('ICE',seg,modelyear));
BAU_OPER(modelyear) = sum((fleetreg,age,prodyear,seg), sum((tec), VEH_STCK.l(tec,seg,fleetreg,modelyear,age)) * VEH_OPER_CINT('ICE', 'FOS', seg, fleetreg, age, modelyear, prodyear) * VEH_PAY(prodyear,age ,modelyear) * VEH_OPER_DIST(modelyear));
BAU_EOL(modelyear)  = sum(seg, sum((tec, fleetreg,age), VEH_STCK_REM.l(tec,seg,fleetreg,modelyear,age))  * VEH_EOLT_CINT('ICE',seg,modelyear));
BAU_EMISSIONS(modelyear) = BAU_PROD(modelyear) + BAU_OPER(modelyear) + BAU_EOL(modelyear);

*test(mat_cat, modelyear) =  sum((prodyear,age), sum((fleetreg), RECYCLED_BATT.l(modelyear, fleetreg, age))*VEH_PAY(prodyear, age, modelyear)*MAT_CONTENT(prodyear, mat_cat));
*test2(age, prodyear, modelyear) =  sum((fleetreg), RECYCLED_BATT.l(modelyear, fleetreg, age))*VEH_PAY(prodyear, age, modelyear);


execute_unload 'EVD4EUR_addset.gdx'
