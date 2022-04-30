**
* ECOPT2 LP Model
* =============
*
* The ECOPT2 LP model minimizes total CO2 emissions while satisfying
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
newtec(tec)     new technologies to replace incumbent
enr             energy carrier
reg             region or country group - superset
fleetreg(reg)   model regions of BEV operation
seg             segment or size class
mat_prod        producers or sources for all critical materials 
mat_cat         critical material categories
sigvar          variables for sigmoid equations
veheq           equations for vehicle parameters
lcphase         life cycle phase
grdeq           parameters for gradient of change (fleet additions) - individual (IND) for each tech or related to all tech (ALL)
mat(mat_cat, mat_prod)      critical material
imp             impact categories - superset
imp_int(imp)    electricity intensity or impact intensity (without electricity contributions)
imp_cat(imp)    subset of indices for impact categories
;

SINGLETON SETS
new(age)        new vehicles (age 0)
optimp(imp)     optimized impact category

;

*---- ABBREIVATIONS USED *-----------------------------------------------------------------------------------
* PROD = Production
* OPER = Operation
* CNST = b in y = ax + b

** Load sets defined in Python class
* These lines must be uncommented if specific input .gdx is not given, e.g., $gdxin <filepath>_input.gdx
* By default uses pyGAMS_input.gdx if gdx filename not provided by Python
$if not set gdxincname $GDXIN C:\Users\chrishun\Box Sync\YSSP_temp\pyGAMS_input.gdx 
$if set gdxincname $GDXIN %gdxincname%

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
$LOAD grdeq
$LOAD imp
$LOAD imp_int
$LOAD imp_cat
$LOAD optimp
$LOAD lcphase

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

PARAMETERS
**-----------------------------------------------------------------------------------
**
** Parameter Definitions p.t 1 : Parameter Declarations
**
**-----------------------------------------------------------------------------------

**-- TIME ---------------------------------------------------------------------------
*Declaraton of year as both a set and a parameter
YEAR_PAR(year)                                 year
TEC_PARAMETERS(lcphase,imp,tec,seg,sigvar)     variables for each tech and veh equation

***ENERGY (ELECTRICITY GENERATION and FUEL) ------------------------------------------
ENR_IMPACT_INT(imp,enr,reg,year)               impact intensity of the regional energy mixes            [kg CO2-eq pr kWh]

***ENERGY and VEHICLE TECHONLOGY COMBINATIONS ----------------------------------------
ENR_TEC_CORRESPONDANCE(tec, enr)               feasible combinations of vehicle technology and energy (fuel)


***All VEHICLES ----------------------------------------------------------------------

CONST(seg,fleetreg)               Used as "seed" for additions to stock based on initial fleet size

**PRODUCTION
TEC_PROD_EL_INT(tec,seg,prodyear)        Electricity intensity of tec production [kwh el required per unit produced]
TEC_PROD_IMPACT_INT_REST(imp,tec,seg,prodyear)   Impact intensity of tec production without electricity contributions    [t CO2-eq per vehicle produced]
TEC_PROD_IMPACT_INT(imp,tec,seg,prodyear)        Total impact intensity of tec production                  [t CO2-eq per vehicle produced]

**OPERATION
TEC_OPER_EINT(tec,seg,prodyear)                  Energy intensity of tec operation     [kwh per km]
TEC_OPER_IMPACT_INT(imp,tec,enr,seg,fleetreg,modelyear,prodyear,age)    Impact intensity of tec operation        [t CO2 per km]

**EOL
TEC_EOLT_IMPACT_INT(imp,tec,seg,year)            Impact intensity of end-of-life treatment                     [t CO2-eq per unit in EOL treatment]

** FLEET -------------------------------------------------------------------------------
** INITIAL STOCK ------------
BEV_CAPAC(seg)                   Correspondence of battery capacities used in each segment  [kWh]

**DEMAND --------------------
EXOG_TOT_STOCK(reg,year)         Number of tec units                            [#]
VEH_OPER_DIST(reg,year)          Annual driving distance per vehicles          [km]

** LIFETIME -----------------
LIFETIME_AGE_DISTRIBUTION(age)          Age distribution = 1 - CDF
RETIREMENT_FUNCTION(age)                Age distribution = 1 - CDF

** COMPOSITION --------------
COHORT_AGE_CORRESPONDANCE(year,prodyear,age)       Correspondence between a vehicle production year and its age (up to 20) in a given year
INITIAL_TEC_SHARES(tec)            Initial share of vehicles in stock tech
INITIAL_SEG_SHARES(seg)            Initial stock distribution by segment

** CONSTRAINTS -------
MAX_UPTAKE_RATE(grdeq,newtec)        Parameter for gradient of change constraint (fleet additions) - individual (IND) for each tech or related to all tech (ALL)
MANUF_CNSTRNT(newtec,year)           Annual manufacturing capacity          [GWh]
MAT_CONTENT(newtec,mat_cat,year)     Critical material content per kWh by production year                       [kg kWh-1]
RECOVERY_PCT(mat_cat,year)       Recovery of critical materials from recycling processes in wt%
VIRG_MAT_SUPPLY(mat_prod,year)   Primary critical material resources available in a given year              [t]
MAT_IMPACT_INT(imp,mat_prod,year)          Impact intensity of each material by source                                [kg CO2e kg^-1 mat]

** POST-PROCESSING --------
TOT_NEW_TECS(year)                   Total stock of new technology in Europe
TOT_CAPACITY_ADDED(newtec,year)      Total battery capacity required by year         [MWh]
TOT_CAPACITY_RECYCLED(year)          Total battery capacity retired each year        [kWh]
MAT_REQ_TOT(mat_cat,year)            Total resources required by year                [kg]
EXOG_TOT_STOCK_CHECK(modelyear)
STOCK_BY_COHORT(tec,seg,fleetreg,modelyear,prodyear,age)       Total stock by technology segment region and cohort
OPER_IMPACT_COHORT(imp,tec,seg,fleetreg,modelyear,prodyear,age)       Total operating impacts by technology segment region and cohort
EOL_IMPACT_COHORT(imp,tec,seg,fleetreg,modelyear,prodyear,age)       Total end-of-life impacts by technology segment region and cohort
TOTAL_OPERATION_ENERGY(tec,seg,fleetreg,modelyear)                      Total fleet operating energy by technology segment and region
TOT_STOCK_ADDED(fleetreg,year)        Total tec added by region
TOT_STOCK_REMOVED(fleetreg,year)      Total tec retired by region
TOT_STOCK_CHECK(fleetreg,year)
BAU_PROD(imp,modelyear)
BAU_OPER(imp,modelyear)
BAU_EOL(imp,modelyear)
BAU_IMPACTS(imp,modelyear)

;

* Load in parameter values from .gdx file [dummy data]

* Sample input GDX as trial for introducing the region set
$if not set gdxincname $GDXIN pyGAMS_input.gdx
$if set gdxincname $GDXIN %gdxincname%

$LOAD YEAR_PAR
$LOAD TEC_PARAMETERS

$LOAD ENR_TEC_CORRESPONDANCE

$LOAD ENR_IMPACT_INT

$LOAD EXOG_TOT_STOCK
$LOAD VEH_OPER_DIST

$LOAD LIFETIME_AGE_DISTRIBUTION
$LOAD RETIREMENT_FUNCTION

$LOAD COHORT_AGE_CORRESPONDANCE
$LOAD INITIAL_TEC_SHARES
$LOAD INITIAL_SEG_SHARES

$LOAD BEV_CAPAC
$LOAD MAX_UPTAKE_RATE
$LOAD MANUF_CNSTRNT

$LOAD MAT_CONTENT
$LOAD RECOVERY_PCT
$LOAD MAT_IMPACT_INT 
$LOAD VIRG_MAT_SUPPLY

$OFFMULTI
$GDXIN
;

*----- Production-related emissions
* Assume constant for all regions for now
TEC_PROD_EL_INT(tec,seg,prodyear) = genlogfnc(TEC_PARAMETERS('prod','nrg',tec,seg,'A'), TEC_PARAMETERS('prod','nrg',tec,seg,'B'), TEC_PARAMETERS('prod', 'nrg',tec,seg,'r'), YEAR_PAR(prodyear), TEC_PARAMETERS('prod', 'nrg',tec,seg,'u'));

TEC_PROD_IMPACT_INT_REST(imp,tec,seg,prodyear)$(imp_cat(imp)) = genlogfnc(TEC_PARAMETERS('prod',imp,tec,seg,'A'), TEC_PARAMETERS('prod',imp,tec,seg,'B'), TEC_PARAMETERS('prod',imp,tec,seg,'r'), YEAR_PAR(prodyear), TEC_PARAMETERS('prod',imp,tec,seg,'u'));

* only production emissions from 2000 onwards are relevant despite cohorts going back to 1972
TEC_PROD_IMPACT_INT(imp,tec,seg,prodyear)$(ord(prodyear)>28) = TEC_PROD_IMPACT_INT_REST(imp,tec,seg,prodyear) + TEC_PROD_EL_INT(tec,seg,prodyear)*ENR_IMPACT_INT(imp,'elc','prod',prodyear)/1000;

*----- Operation phase emissions
* Assume constant for all regions for now
TEC_OPER_EINT(tec,seg,prodyear) = genlogfnc(TEC_PARAMETERS('oper','nrg',tec,seg,'A'),TEC_PARAMETERS('oper','nrg',tec,seg,'B'),TEC_PARAMETERS('oper','nrg',tec,seg,'r'),YEAR_PAR(prodyear),TEC_PARAMETERS('oper','nrg',tec,seg,'u'));

TEC_OPER_IMPACT_INT(imp,tec,enr,seg,fleetreg,modelyear,prodyear,age)$(ENR_TEC_CORRESPONDANCE(tec,enr)) = TEC_OPER_EINT(tec,seg,prodyear)*COHORT_AGE_CORRESPONDANCE(modelyear,prodyear,age)*(ENR_IMPACT_INT(imp,enr,fleetreg,modelyear)/1000);

*----- End-of-life phase emissions
TEC_EOLT_IMPACT_INT(imp,tec,seg,modelyear)$(imp_cat(imp)) = genlogfnc(TEC_PARAMETERS('eol',imp,tec,seg,'A'), TEC_PARAMETERS('eol',imp,tec,seg,'B'), TEC_PARAMETERS('eol',imp,tec,seg,'r'), YEAR_PAR(modelyear), TEC_PARAMETERS('eol',imp,tec,seg,'u'));


*-----------------------------------------------------------------------------------
*
* Model Definition  p.t 1 : Variable Definitions
*
*-----------------------------------------------------------------------------------


***FREE VARIABLES ------------------------------------------------------------------
* Objective value to be minimized must be a free variable


FREE VARIABLES
TOT_IMPACTS_OPT                             Total impacts for the whole system over optimization period
STOCK_CHANGE(fleetreg,year)                 Net change in stock from one year to the next
TOT_PRIMARY_MAT(mat_cat,year)               Total primary resources required by year [kg]

* Slack variables for debugging
SLACK_ADD(tec,seg,fleetreg,year,age)        
SLACK_TEC_ADD(newtec,seg,fleetreg,year,age)
SLACK_SEG_ADD(seg,fleetreg,year)
SLACK_RECYCLED_BATT(fleetreg,year,age)
SLACK_VIRG_MAT(mat_prod,year)
SLACK_NEW_BATT_CAP(year)
;

POSITIVE VARIABLES
TOT_STOCK(tec,seg,fleetreg,year,age)        Number of vehicles of a given age in a given year
STOCK_REMOVED(tec,seg,fleetreg,year,age)    Number of vehicles of a given age retired in a given year
TOT_IMPACTS(imp,tec,seg,fleetreg,year)      Total impacts per year by technology              [t CO2-eq]
STOCK_ADDED(tec,seg,fleetreg,year,age)      Stock additions (new tec sales)

PRODUCTION_IMPACTS(imp,tec,seg,fleetreg,year)   Total impacts from production of vehicles per year            [t CO2-eq]
OPERATION_IMPACTS(imp,tec,seg,fleetreg,year)    Total impacts from operations of vehicles per year            [t CO2-eq]
EOL_IMPACTS(imp,tec,seg,fleetreg,year)          Total impacts from vehicle end of life treatment per year     [t CO2-eq]

RECYCLED_BATT(newtec,fleetreg,year,age)         Total battery capacity sent to recycling per year                   [kWh]
RECYCLED_MAT(mat_cat,year)                      Materials recovered from recycled batteries                         [kg]
MAT_REQ(mat_cat,year)                      Total amount of critical materials needed for new vehicles          [kg]

MAT_MIX(mat_prod,year)                     Production mixes for virgin materials                               [t]
MAT_IMPACTS(imp,mat_prod,year)                 Total impacts from virgin material production per year        [t CO2e kg^-1]
;



*-----------------------------------------------------------------------------------
*
* Model Definition  p.t 2 : Model Equation Declarations
*
*-----------------------------------------------------------------------------------

EQUATIONS

***VEHICLE STOCK MODEL  ------------------------------------------------------------

***  Fleet model
EQ_STCK_CHANGE              Net total stock change
EQ_STCK_REM                 Removal from stock
EQ_STCK_ADD                 Addition to stock
EQ_STCK_BAL                 Fleet balance constraint

*** Constraint equations
EQ_TEC_UPTAKE_CONSTRAINT    Technology uptake constraint
EQ_SEG_SHARE_CONSTRAINT     Segment share constraint
EQ_NEW_MANUF_CAP            Manufacturing constraint

EQ_RECYCLED_BATT
EQ_MAT_REQ
EQ_RECYCLED_MAT
EQ_MAT_TOT_PRIMARY
EQ_MAT_SUPPLY                   Critical material balance constraint - total supply   
EQ_PRIM_MAT_SUPPLY_CONSTRAINT   Critical material primary supply constraint

**EMISSION and ENERGY MODELS incl OBJ. FUNCTION --------------------------------------

EQ_OBJ                      Objective function - total fleet lifecycle CO2 over optimization period
EQ_TOTAL_IMPACTS            Calculation of emissions from all vehicle classes per year
EQ_PRODUCTION_IMPACTS       Total production emissions including from materials
EQ_PRIMARY_MAT_IMPACTS      Primary material extraction and processing emissions
EQ_OPERATION_IMPACTS        Total vehicle operating emissions
EQ_EOL_IMPACTS              Total vehicle end of life emissions
;

*-----------------------------------------------------------------------------------
*
* Model Definition  p.t 3 : Model Equations

*-----------------------------------------------------------------------------------


***VEHICLE STOCK MODEL  ------------------------------------------------------------
* First, initialize first year (boundary condition)
TOT_STOCK.fx(tec,seg,fleetreg,modelyear,age)$(ord(modelyear)=1) = EXOG_TOT_STOCK(fleetreg,modelyear) * LIFETIME_AGE_DISTRIBUTION(age) * INITIAL_TEC_SHARES(tec)  * INITIAL_SEG_SHARES(seg);
* try below with age+1 on LHS to avoid weird offset thing?
STOCK_REMOVED.fx(tec,seg,fleetreg,modelyear,age)$(ord(modelyear)=1) = TOT_STOCK.l(tec,seg,fleetreg,modelyear,age-1) * RETIREMENT_FUNCTION(age-1);
STOCK_ADDED.fx(tec,seg,fleetreg,modelyear,new)$(ord(modelyear)=1) = sum(age, STOCK_REMOVED.l(tec,seg,fleetreg,modelyear,age));
* Do not add "new" vehicles with age (e.g., used cars from other regions)
STOCK_ADDED.fx(tec,seg,fleetreg,modelyear,age)$(ord(age)>1) = 0;

* Initiate stock in remaining initialization period
loop(modelyear $ (ord(modelyear)>1 and ord(modelyear) <= card(inityear)),
STOCK_CHANGE.fx(fleetreg,modelyear) = EXOG_TOT_STOCK(fleetreg,modelyear) - EXOG_TOT_STOCK(fleetreg,modelyear-1);
* calculate stock removals as per survival curves
STOCK_REMOVED.fx(tec,seg,fleetreg,modelyear,age) = TOT_STOCK.l(tec,seg,fleetreg,modelyear-1,age-1) * RETIREMENT_FUNCTION(age-1);
* in initialization period, assume that new vehicles maintain the status quo of tec and segment shares
STOCK_ADDED.fx(tec,seg,fleetreg,modelyear,new) = sum(age, STOCK_REMOVED.l(tec,seg,fleetreg,modelyear,age)) + (STOCK_CHANGE.l(fleetreg,modelyear) * INITIAL_TEC_SHARES(tec) * INITIAL_SEG_SHARES(seg));
TOT_STOCK.fx(tec,seg,fleetreg,modelyear,age) = TOT_STOCK.l(tec,seg,fleetreg,modelyear-1,age-1) - STOCK_REMOVED.l(tec,seg,fleetreg,modelyear,age) + STOCK_ADDED.l(tec,seg,fleetreg,modelyear,age);
);


***  Fleet balance model -----------------------------------------------
* Net change in stock from year to year - exogeneously defined (see Constraint A)
EQ_STCK_CHANGE(modelyear,fleetreg)$(ord(modelyear)>card(inityear))..             STOCK_CHANGE(fleetreg,modelyear)  =e=  EXOG_TOT_STOCK(fleetreg,modelyear) - EXOG_TOT_STOCK(fleetreg,modelyear-1);

EQ_STCK_REM(tec,seg,fleetreg,modelyear,age)$(ord(modelyear)>card(inityear))..    STOCK_REMOVED(tec,seg,fleetreg,modelyear,age) =e= TOT_STOCK(tec,seg,fleetreg,modelyear-1,age-1)*RETIREMENT_FUNCTION(age-1);

EQ_STCK_ADD(modelyear,fleetreg)$(ord(modelyear)>card(inityear))..               sum((tec,seg), STOCK_ADDED(tec,seg,fleetreg,modelyear,new)) =e= sum((tec,seg,age), STOCK_REMOVED(tec,seg,fleetreg,modelyear, age)) + STOCK_CHANGE(fleetreg,modelyear);


EQ_STCK_BAL(tec,seg,fleetreg,modelyear,age)$(ord(modelyear) > card(inityear))..      TOT_STOCK(tec,seg,fleetreg,modelyear,age)  =e= TOT_STOCK(tec,seg,fleetreg,modelyear-1,age-1) + STOCK_ADDED(tec,seg,fleetreg,modelyear,age) - STOCK_REMOVED(tec,seg,fleetreg,modelyear,age);
                                 

*** Constraints -----------------------------------------------------------------------

*--------------------------------------
* B - Manufacturing constraint;

* total capacity added per year must be less than the battery manufacturing capacity
* MANUF_CNSTRNT input is in GWh; BEV_CAPAC is in kWh
EQ_NEW_MANUF_CAP(newtec,optyear)..                   MANUF_CNSTRNT(newtec,optyear)*1e6 =g= sum((seg, fleetreg), STOCK_ADDED(newtec,seg,fleetreg,optyear,new) * BEV_CAPAC(seg))
%SLACK_NEW_BATT_CAP% - SLACK_NEW_BATT_CAP(optyear)
;

*-----------------------------------------------
* C - Material balance constraint - total supply
* Equation 10 (split into two equations here)
* Amount of virgin + recycled material to produce new batteries, in kg; see intermediate calculations below
EQ_MAT_SUPPLY(mat_cat,optyear)..                    TOT_PRIMARY_MAT(mat_cat,optyear) + RECYCLED_MAT(mat_cat,optyear) =e= MAT_REQ(mat_cat,optyear);

EQ_MAT_TOT_PRIMARY(mat_cat, optyear)..               TOT_PRIMARY_MAT(mat_cat,optyear) =e= sum(mat_prod$mat(mat_cat, mat_prod), MAT_MIX(mat_prod,optyear));

*------------------------------------------------
* D - Critical material primary supply constraint
* Equation 11
* demand of virgin resources from each source must be less than or equal to available supply; see intermediate calculations below
* primary resources constraint provided in t, mat_mix in kg
EQ_PRIM_MAT_SUPPLY_CONSTRAINT(mat_prod,optyear)..                MAT_MIX(mat_prod,optyear) =l= VIRG_MAT_SUPPLY(mat_prod,optyear)*1000
%SLACK_VIRG_MAT% + SLACK_VIRG_MAT(mat_prod,optyear)
;

*---------------------------------
* E - Technology uptake constraint
* Equation 12
* incumbent technology excluded
EQ_TEC_UPTAKE_CONSTRAINT(newtec,seg,fleetreg,modelyear)$(ord(modelyear)>card(inityear))..     STOCK_ADDED(newtec,seg,fleetreg,modelyear,new) =l= ((1 + MAX_UPTAKE_RATE('IND', newtec)) * STOCK_ADDED(newtec,seg,fleetreg,modelyear-1,new)) + 100
%SLACK_ADD% + SLACK_TEC_ADD(newtec,seg,fleetreg,modelyear,new)

;

*------------------------------------------------------
* F - Segment share constraint (segments kept constant)
* Equation 13
* Removing allows for ICEV smart cars (e.g.,) to be replaced by a BEV Model X...
EQ_SEG_SHARE_CONSTRAINT(seg,fleetreg,modelyear)$(ord(modelyear)>card(inityear))..          sum(tec, STOCK_ADDED(tec,seg,fleetreg,modelyear,new))
    =e= INITIAL_SEG_SHARES(seg) * sum((tec,segj), STOCK_ADDED(tec,segj,fleetreg,modelyear,new));


*------ Lithium/elemental resource availability calculations for constraints C and D
* Calculate amount of batteries retired each year in kWh
EQ_RECYCLED_BATT(newtec,fleetreg,optyear,age)..             RECYCLED_BATT(newtec,fleetreg,optyear,age) =e= sum((seg), STOCK_REMOVED(newtec,seg,fleetreg,optyear,age)* BEV_CAPAC(seg))
;

* Calculate amount of materials recovered from end-of-life batteries for recycling (i.e., to new batteries), in kg
EQ_RECYCLED_MAT(mat_cat,optyear)..                   RECYCLED_MAT(mat_cat,optyear) =e= sum((newtec,prodyear,age), sum((fleetreg), RECYCLED_BATT(newtec,fleetreg,optyear,age)) * COHORT_AGE_CORRESPONDANCE(optyear,prodyear,age) * MAT_CONTENT(newtec, mat_cat, optyear)) * RECOVERY_PCT(mat_cat,optyear)
;

* Material supply balance
* Total amount of material required calculated from new vehicles entering the market, in kg
EQ_MAT_REQ(mat_cat,optyear)..                       MAT_REQ(mat_cat,optyear) =e= sum(newtec, sum((seg, fleetreg), STOCK_ADDED(newtec,seg,fleetreg,optyear,new) * BEV_CAPAC(seg)) * MAT_CONTENT(newtec, mat_cat, optyear)
%SLACK_NEW_BATT_CAP% + SLACK_NEW_BATT_CAP(optyear) * MAT_CONTENT(newtec, mat_cat, optyear)
)
;



*** EMISSION and ENERGY MODELS incl OBJ. FUNCTION -------------------------------------------------
* Objective function
EQ_OBJ..                                              TOT_IMPACTS_OPT =e= sum((tec,seg,fleetreg,optyear), TOT_IMPACTS(optimp,tec,seg,fleetreg,optyear));

* Calculation of emissions from all vehicle classes per year
EQ_TOTAL_IMPACTS(imp,tec,seg,fleetreg,modelyear)..    TOT_IMPACTS(imp,tec,seg,fleetreg,modelyear) =e= PRODUCTION_IMPACTS(imp,tec,seg,fleetreg,modelyear) + OPERATION_IMPACTS(imp,tec,seg,fleetreg,modelyear) + EOL_IMPACTS(imp,tec,seg,fleetreg,modelyear);

* Calculate emissions from virgin materials. Currently assumes zero emissions for recycled materials.
EQ_PRIMARY_MAT_IMPACTS(imp,mat_prod,modelyear)..      MAT_IMPACTS(imp,mat_prod,modelyear) =e= (MAT_MIX(mat_prod,modelyear) * MAT_IMPACT_INT(imp,mat_prod,modelyear))/1000
%SLACK_VIRG_MAT% + SLACK_VIRG_MAT(mat_prod,modelyear)*1e6
;

* Calculate emissions from vehicle production
EQ_PRODUCTION_IMPACTS(imp,tec,seg,fleetreg,modelyear)..             PRODUCTION_IMPACTS(imp,tec,seg,fleetreg,modelyear) =e= STOCK_ADDED(tec,seg,fleetreg,modelyear,new)*TEC_PROD_IMPACT_INT(imp,tec,seg,modelyear) + sum(mat_prod, MAT_IMPACTS(imp,mat_prod,modelyear))
%SLACK_ADD% + SLACK_TEC_ADD(newtec,seg,fleetreg,modelyear,new) * TEC_PROD_IMPACT_INT(imp,tec,seg,modelyear)*1e6
;

* Calculate emissions from vehicle operation
EQ_OPERATION_IMPACTS(imp,tec,seg,fleetreg,modelyear)..             OPERATION_IMPACTS(imp,tec,seg,fleetreg,modelyear) =e= sum( (agej,enr,prodyear), TOT_STOCK(tec,seg,fleetreg,modelyear,agej) * COHORT_AGE_CORRESPONDANCE(modelyear,prodyear,agej)* TEC_OPER_IMPACT_INT(imp,tec,enr,seg,fleetreg,modelyear,prodyear,agej) *  VEH_OPER_DIST(fleetreg,modelyear));

* Calculate emissions from vehicle disposal and recycling
EQ_EOL_IMPACTS(imp,tec,seg,fleetreg,modelyear)..             EOL_IMPACTS(imp,tec,seg,fleetreg,modelyear) =e= sum((agej),STOCK_REMOVED(tec,seg,fleetreg,modelyear,agej)) * TEC_EOLT_IMPACT_INT(imp,tec,seg,modelyear)
;


*-----------------------------------------------------------------------------------
*
* Model Execution  p.t 1 : Model Definition and Options
*
*-----------------------------------------------------------------------------------

* Defining name of model(s) and what equations are used in each model

MODEL EVD4EUR_Basic     "default model run in normal mode"      /ALL/
;

* Troubleshooting models
MODEL 
      seg_test          "model without segment constraint"      /EVD4EUR_Basic - EQ_SEG_SHARE_CONSTRAINT/
      tec_test          "model without growth constraint"       /EVD4EUR_Basic - EQ_TEC_UPTAKE_CONSTRAINT/
 
      no_mat            "model with no critical material constraints"       /EVD4EUR_Basic - EQ_RECYCLED_MAT - EQ_RECYCLED_BATT - EQ_MAT_REQ - EQ_MAT_TOT_PRIMARY - EQ_MAT_SUPPLY - EQ_PRIM_MAT_SUPPLY_CONSTRAINT/

      manuf_test        "model without manufacturing capacity constraint"       /EVD4EUR_Basic - EQ_NEW_MANUF_CAP/
      fleet_test        "model without growth or manufacturing constraint"      /tec_test - EQ_NEW_MANUF_CAP/
      mat_test          "model without material constraint"                     /EVD4EUR_Basic - EQ_MAT_SUPPLY/
      primary_mat_test  "model without primary material supply constraint"      /EVD4EUR_Basic - EQ_PRIM_MAT_SUPPLY_CONSTRAINT/
      test_model        "model with only manufacturing and growth constraints"  /no_mat - EQ_SEG_SHARE_CONSTRAINT/
      no_constraints    "model without constraints"                             /no_mat - EQ_SEG_SHARE_CONSTRAINT - EQ_TEC_UPTAKE_CONSTRAINT - EQ_NEW_MANUF_CAP/
;


* Defining model run options and solver

*OPTION RESLIM = 2000000;

*OPTION limrow = 0;
*OPTION limcol = 0;
*OPTION PROFILE = 0;
* set to PROFILE = 2 for debugging 
*tec_test.optfile=1;

Scalar ms 'model status', ss 'solve status';

*------------------------------------------------------------------------------                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     -----
*
* Model Execution
*
*-----------------------------------------------------------------------------------

SOLVE EVD4EUR_Basic USING LP MINIMIZING TOT_IMPACTS_OPT;
ss = EVD4EUR_Basic.solvestat;
ms = EVD4EUR_Basic.modelstat;

*SOLVE seg_test USING LP MINIMIZING TOT_IMPACTS_OPT;
*ms = seg_test.modelstat;
*ss = seg_test.solvestat;

*SOLVE tec_test USING LP MINIMIZING TOT_IMPACTS_OPT; 
*ms = tec_test.modelstat;
*ss = tec_test.solvestat;

*SOLVE no_mat USING LP MINIMIZING TOT_IMPACTS_OPT;
*ms = no_mat.modelstat;
*ss = no_mat.solvestat;

*SOLVE manuf_test USING LP MINIMIZING TOT_IMPACTS_OPT;
*ms = manuf_test.modelstat;
*ss = manuf_test.solvestat;

*SOLVE fleet_test USING LP MINIMIZING TOT_IMPACTS_OPT;

*SOLVE mat_test USING LP MINIMIZING TOT_IMPACTS_OPT;
*ms = mat_test.modelstat;
*ss = mat_test.solvestat;

*SOLVE primary_mat_test USING LP MINIMIZING TOT_IMPACTS_OPT;
*SOLVE test_model USING LP MINIMIZING TOT_IMPACTS_OPT;

*SOLVE no_constraints USING LP MINIMIZING TOT_IMPACTS_OPT;
*ms = no_constraints.modelstat;
*ss = no_constraints.solvestat;

*-----------------------------------------------------------------------------------
*
* Post-processing calculations
*
*-----------------------------------------------------------------------------------


TOT_NEW_TECS(modelyear) = sum((newtec,seg, fleetreg, age), TOT_STOCK.l(newtec, seg, fleetreg, modelyear,age)); 
* total capacity of batteries added by year in MWh
TOT_CAPACITY_ADDED(newtec, modelyear) = sum((seg,fleetreg)$STOCK_ADDED.l(newtec, seg, fleetreg, modelyear, new), STOCK_ADDED.l(newtec,seg,fleetreg,modelyear,new)*BEV_CAPAC(seg))/1e6;
TOT_CAPACITY_RECYCLED(modelyear) = sum((seg), sum((newtec,fleetreg, age), STOCK_REMOVED.l(newtec,seg,fleetreg,modelyear,age))* BEV_CAPAC(seg));
MAT_REQ_TOT(mat_cat,modelyear) = sum((newtec,seg, fleetreg), STOCK_ADDED.l(newtec,seg,fleetreg,modelyear,new)*BEV_CAPAC(seg)*MAT_CONTENT(newtec,mat_cat,modelyear));

TOT_STOCK_ADDED(fleetreg,modelyear) = sum((tec,seg), STOCK_ADDED.l(tec, seg, fleetreg, modelyear, new));
TOT_STOCK_REMOVED(fleetreg, modelyear) = sum((tec,seg,age), STOCK_REMOVED.l(tec, seg, fleetreg, modelyear, age));
TOT_STOCK_CHECK(fleetreg, modelyear) = sum((tec,seg,age), TOT_STOCK.l(tec,seg,fleetreg, modelyear, age));

* summing the number of vehicles in fleet as check.
EXOG_TOT_STOCK_CHECK(modelyear) = sum((tec,seg,fleetreg,age), TOT_STOCK.l(tec,seg,fleetreg,modelyear,age));
STOCK_BY_COHORT(tec,seg,fleetreg,modelyear,prodyear,agej) $ COHORT_AGE_CORRESPONDANCE(modelyear,prodyear,agej) = (TOT_STOCK.l(tec,seg,fleetreg,modelyear,agej)*COHORT_AGE_CORRESPONDANCE(modelyear,prodyear,agej));

* total operation emissions by cohort and model year
OPER_IMPACT_COHORT(imp,tec, seg, fleetreg,modelyear,prodyear,age) = sum((enr), TOT_STOCK.l(tec,seg,fleetreg,modelyear,age) * COHORT_AGE_CORRESPONDANCE(modelyear,prodyear,age)* TEC_OPER_IMPACT_INT(imp,tec,enr,seg,fleetreg,modelyear,prodyear,age) * VEH_OPER_DIST(fleetreg,modelyear));
EOL_IMPACT_COHORT(imp,tec,seg,fleetreg,modelyear,prodyear,age) = (STOCK_REMOVED.l(tec,seg,fleetreg,modelyear,age)* COHORT_AGE_CORRESPONDANCE(modelyear,prodyear,age)) * TEC_EOLT_IMPACT_INT(imp,tec,seg,modelyear);
TOTAL_OPERATION_ENERGY(tec,seg,fleetreg,modelyear) = sum((prodyear, age), TOT_STOCK.l(tec,seg,fleetreg,modelyear,age) * COHORT_AGE_CORRESPONDANCE(modelyear,prodyear,age) * TEC_OPER_EINT(tec,seg,prodyear) * VEH_OPER_DIST(fleetreg,modelyear));

* calculate emissions for 0 electrification (business as usual)
BAU_PROD(imp,modelyear) = sum(seg, sum((tec,fleetreg), STOCK_ADDED.l(tec, seg, fleetreg, modelyear, new)) * TEC_PROD_IMPACT_INT(imp,'ICE',seg,modelyear));
BAU_OPER(imp,modelyear) = sum((fleetreg,age,prodyear,seg), sum((tec), TOT_STOCK.l(tec,seg,fleetreg,modelyear,age)) * TEC_OPER_IMPACT_INT(imp,'ICE', 'FOS', seg, fleetreg, modelyear, prodyear, age) * COHORT_AGE_CORRESPONDANCE(modelyear,prodyear,age) * VEH_OPER_DIST(fleetreg,modelyear));
BAU_EOL(imp,modelyear)  = sum(seg, sum((tec, fleetreg,age), STOCK_REMOVED.l(tec,seg,fleetreg,modelyear,age))  * TEC_EOLT_IMPACT_INT(imp,'ICE',seg,modelyear));
BAU_IMPACTS(imp,modelyear) = BAU_PROD(imp,modelyear) + BAU_OPER(imp,modelyear) + BAU_EOL(imp,modelyear);



execute_unload 'EVD4EUR_addset.gdx'
