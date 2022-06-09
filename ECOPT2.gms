*-----------------------------------------------------------------------------------
* ECOPT2 LP Model
* ===============
*
* The ECOPT2 LP model minimizes total impacts while satisfying
* function demand and fleet dynamics. Manufacturing and supply
* chain constraints are considered.
*
*-----------------------------------------------------------------------------------


* Slack variables for debugging; remove asterisk (*) from quotes to activate slack variable
$IF NOT SET SLACK_TEC_ADD             $SETGLOBAL SLACK_TEC_ADD "*"
$IF NOT SET SLACK_NEW_COMPONENTS      $SETGLOBAL SLACK_NEW_COMPONENTS "*"
$IF NOT SET SLACK_VIRG_MAT            $SETGLOBAL SLACK_VIRG_MAT "*"
$IF NOT SET DYNAMIC_EL                $SETGLOBAL DYNAMIC_EL "*"
$IF NOT SET DYNAMIC_MAT               $SETGLOBAL DYNAMIC_MAT "*"

SETS
year            total span of years including production before intialization period - superset
modelyear(year) model years (2000-2050)
optyear(year)   years for optimization (2020-2050) 
inityear(year)  years for initialization (2000-2020)
age             age of unit
tec             technology - superset
newtec(tec)     new technologies to replace incumbent
enr             energy carrier
reg             region or country group - superset
fleetreg(reg)   model regions of technology operation
seg             segment or size class
mat_prod        producers or sources for all critical materials 
mat_cat         critical material categories
sigvar          variables for sigmoid equations
lcphase         life cycle phase
grdeq           parameters for gradient of change (fleet additions) - individual (IND) for each tech or related to all tech (ALL)
mat(mat_cat, mat_prod)      critical material
imp             impact categories - superset
* NOT CURRENTLY USED: IMP_INT
imp_int(imp)    electricity intensity or impact intensity (without electricity contributions)
imp_cat(imp)    subset of indices for impact categories
;

SINGLETON SETS
new(age)        new units (age 0)
optimp(imp)     optimized impact category

;

** Load sets defined in Python class
* These lines must be uncommented if specific input .gdx is not given, e.g., $gdxin <filepath>_input.gdx
* By default uses pyGAMS_input.gdx if gdx filename not provided by Python
$if not set gdxincname $GDXIN pyGAMS_input.gdx 
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


PARAMETERS

**-- TIME ---------------------------------------------------------------------------
*Declaraton of year as both a set and a parameter
YEAR_PAR(year)                                   Year
TEC_PARAMETERS(lcphase,imp,tec,seg,sigvar)       Generalized logistic function term values for calculating evolving impacts

***ENERGY (ELECTRICITY GENERATION and FUEL) -----------------------------------------
ENR_IMPACT_INT(imp,enr,reg,year)                 Impact intensity of the regional energy mixes            [kg CO2-eq pr kWh]

***ENERGY and TECHONLOGY COMBINATIONS -----------------------------------------------
ENR_TEC_CORRESPONDENCE(tec, enr)                 Feasible combinations of technology and energy (fuel)


***IMPACT INTENSITY -----------------------------------------------------------------
**PRODUCTION ----------------
TEC_PROD_EL_INT(tec,seg,prodyear)                Electricity intensity of tec production                                 [kwh el required per unit produced]
TEC_PROD_IMPACT_INT_REST(imp,tec,seg,prodyear)   Impact intensity of tec production without electricity contributions    [t CO2-eq per unit produced]
TEC_PROD_IMPACT_INT(imp,tec,seg,prodyear)        Total impact intensity of tec production                                [t CO2-eq per unit produced]

**OPERATION -----------------
TEC_OPER_EINT(tec,seg,prodyear)                  Energy intensity of tec operation                                       [kwh per km]
TEC_OPER_IMPACT_INT(imp,tec,enr,seg,fleetreg,modelyear,prodyear,age)    Impact intensity of tec operation                [t CO2 per km]

**EOL -----------------------
TEC_EOLT_IMPACT_INT(imp,tec,seg,year)            Impact intensity of end-of-life treatment                               [t CO2-eq per unit in EOL treatment]

** FLEET ----------------------------------------------------------------------------
** INITIAL STOCK ------------
TEC_SIZE_CORRESPONDENCE(tec,seg)     Correspondence of component sizes used in each segment                  [kWh]

**DEMAND --------------------
EXOG_TOT_STOCK(reg,year)             Number of tec units in stock                   [#]
ANNUAL_USE_INTENSITY(reg,year)       Annual driving distance per unit               [km]

** LIFETIME -----------------
LIFETIME_AGE_DISTRIBUTION(age)       Age distribution 
RETIREMENT_FUNCTION(age)             Normalized retirement function

** COMPOSITION --------------
COHORT_AGE_CORRESPONDENCE(year,prodyear,age)     Correspondence between production year and age (up to 20) in a given model year
INITIAL_TEC_SHARES(tec)              Initial stock distribution by technology
INITIAL_SEG_SHARES(seg)              Initial stock distribution by segment

** CONSTRAINTS ------------------------------------------------------------------------
MAX_UPTAKE_RATE(grdeq,newtec)        Parameter for gradient of change constraint (fleet additions) - individual (IND) for each tech or related to all tech (ALL)
CONST(seg,fleetreg)                  Used as "seed" for additions to stock based on initial fleet size [not yet implemented]
MANUF_CNSTRNT(newtec,year)           Annual manufacturing capacity                                              [GWh or units]
MAT_CONTENT(newtec,mat_cat,year)     Critical material content per kWh by production year                       [kg kWh-1]
RECOVERY_PCT(newtec,year)            Recovery or collection rate of components for recycling                    [%]
RECYCLING_YIELD(mat_cat,year)        Yield of critical materials recovery from recycling processes              [wt%]
VIRG_MAT_SUPPLY(mat_prod,year)       Primary critical material resources available in a given year              [t]
MAT_IMPACT_INT(imp,mat_prod,year)    Impact intensity of each material by source                                [kg CO2e kg^-1 mat]

** POST-PROCESSING --------------------------------------------------------------------
TOT_NEW_TECS(year)                   Total stock of new technology in Europe
TOT_NEWTEC_COMP_ADDED(newtec,year)   Total new components required by year         [MWh]
TOT_NEWTEC_COMP_RECYCLED(year)       Total components retired each year            [kWh]
MAT_REQ_TOT(mat_cat,year)            Total resources required by year              [kg]
EXOG_TOT_STOCK_CHECK(modelyear)      Total fleet size as quality check
TOT_STOCK_CHECK(fleetreg,year)       Total fleet size by region as quality check
STOCK_BY_COHORT(tec,seg,fleetreg,modelyear,prodyear,age)            Total stock by technology segment region and cohort
OPER_IMPACT_COHORT(imp,tec,seg,fleetreg,modelyear,prodyear,age)     Total operating impacts by technology segment region and cohort
EOL_IMPACT_COHORT(imp,tec,seg,fleetreg,modelyear,prodyear,age)      Total end-of-life impacts by technology segment region and cohort
TOTAL_OPERATION_ENERGY(tec,seg,fleetreg,modelyear)                  Total fleet operating energy by technology segment and region
TOT_STOCK_ADDED(fleetreg,year)       Total tec added by region
TOT_STOCK_REMOVED(fleetreg,year)     Total tec retired by region
BAU_PROD(imp,modelyear)              Business as usual (no new techs) production impacts
BAU_OPER(imp,modelyear)              Business as usual (no new techs) operation impacts
BAU_EOL(imp,modelyear)               Business as usual (no new techs) end of life impacts  
BAU_IMPACTS(imp,modelyear)           Business as usual (no new techs) total impacts

;


SCALAR
UPTAKE_CONSTANT                      Seed value for introduction of new technologies (constraint E)
;

* Load in parameter values from .gdx file [dummy data]

* Sample input GDX as trial for introducing the region set
$if not set gdxincname $GDXIN pyGAMS_input.gdx
$if set gdxincname $GDXIN %gdxincname%

$LOAD YEAR_PAR
$LOAD TEC_PARAMETERS

$LOAD ENR_TEC_CORRESPONDENCE

$LOAD ENR_IMPACT_INT

$LOAD EXOG_TOT_STOCK
$LOAD ANNUAL_USE_INTENSITY

$LOAD LIFETIME_AGE_DISTRIBUTION
$LOAD RETIREMENT_FUNCTION

$LOAD COHORT_AGE_CORRESPONDENCE
$LOAD INITIAL_TEC_SHARES
$LOAD INITIAL_SEG_SHARES

$LOAD TEC_SIZE_CORRESPONDENCE
$LOAD MAX_UPTAKE_RATE
$LOAD MANUF_CNSTRNT

$LOAD MAT_CONTENT
$LOAD RECOVERY_PCT
$LOAD RECYCLING_YIELD
$LOAD MAT_IMPACT_INT 
$LOAD VIRG_MAT_SUPPLY

$LOAD UPTAKE_CONSTANT

$OFFMULTI
$GDXIN
;

*--- General logistic function -----------------------------------------------------------------------------------
$macro genlogfnc(A,B,r,t,u) A + (B-A)/(1+exp(-r*(t-u)));

** https://en.wikipedia.org/wiki/Generalised_logistic_function
** y = A + (B-A)/(1+exp(-r*(t-tau)));
** Y  is for example the  CO2 eq. intensity of electricity generation
** t  is time.
** A = Initial CO2 intensity of electricity generation
** B = End CO2 intensity of electricity generation
** r = is the rate of change ;
** (tau) u is time the of maximum gradient of Y

*----- Production-related emissions
* Not yet fully implemented: dynamic electricity background for material and energy cycles
%DYNAMIC_EL% ENR_IMPACT_INT(imp,enr,reg,prodyear) = ENR_IMPACT_INT(imp,enr,reg,prodyear) + ENR_EL_INT(imp,enr,reg,prodyear) * ENR_IMPACT_INT(imp,'el',reg, prodyear)
%DYNAMIC_MAT% MAT_IMPACT_INT(imp,mat,prodyear) = MAT_IMPACT_INT(imp, mat, prodyear) + MAT_EL_INT(imp, enr, reg,prodyear) * ENR_IMPACT_INT(imp,'el',reg,prodyear)

* Assume constant for all regions for now
TEC_PROD_EL_INT(tec,seg,prodyear) = genlogfnc(TEC_PARAMETERS('prod','nrg',tec,seg,'A'), TEC_PARAMETERS('prod','nrg',tec,seg,'B'), TEC_PARAMETERS('prod', 'nrg',tec,seg,'r'), YEAR_PAR(prodyear), TEC_PARAMETERS('prod', 'nrg',tec,seg,'u'));

TEC_PROD_IMPACT_INT_REST(imp,tec,seg,prodyear)$(imp_cat(imp)) = genlogfnc(TEC_PARAMETERS('prod',imp,tec,seg,'A'), TEC_PARAMETERS('prod',imp,tec,seg,'B'), TEC_PARAMETERS('prod',imp,tec,seg,'r'), YEAR_PAR(prodyear), TEC_PARAMETERS('prod',imp,tec,seg,'u'));

* only production emissions from 2000 onwards are relevant despite cohorts going back to 1972
TEC_PROD_IMPACT_INT(imp,tec,seg,prodyear)$(ord(prodyear)>28) = TEC_PROD_IMPACT_INT_REST(imp,tec,seg,prodyear) + TEC_PROD_EL_INT(tec,seg,prodyear)*ENR_IMPACT_INT(imp,'elc','prod',prodyear)/1000;

*----- Operation phase emissions
* Assume constant for all regions for now
TEC_OPER_EINT(tec,seg,prodyear) = genlogfnc(TEC_PARAMETERS('oper','nrg',tec,seg,'A'),TEC_PARAMETERS('oper','nrg',tec,seg,'B'),TEC_PARAMETERS('oper','nrg',tec,seg,'r'),YEAR_PAR(prodyear),TEC_PARAMETERS('oper','nrg',tec,seg,'u'));

TEC_OPER_IMPACT_INT(imp,tec,enr,seg,fleetreg,modelyear,prodyear,age)$(ENR_TEC_CORRESPONDENCE(tec,enr)) = TEC_OPER_EINT(tec,seg,prodyear)*COHORT_AGE_CORRESPONDENCE(modelyear,prodyear,age)*(ENR_IMPACT_INT(imp,enr,fleetreg,modelyear)/1000);

*----- End-of-life phase emissions
TEC_EOLT_IMPACT_INT(imp,tec,seg,modelyear)$(imp_cat(imp)) = genlogfnc(TEC_PARAMETERS('eol',imp,tec,seg,'A'), TEC_PARAMETERS('eol',imp,tec,seg,'B'), TEC_PARAMETERS('eol',imp,tec,seg,'r'), YEAR_PAR(modelyear), TEC_PARAMETERS('eol',imp,tec,seg,'u'));



*-----------------------------------------------------------------------------------
*
* Variable declarations
*
*-----------------------------------------------------------------------------------

FREE VARIABLES
TOT_IMPACTS_OPT                                 Total impacts for the whole system over optimization period []
STOCK_CHANGE(fleetreg,year)                     Net change in stock from one year to the next               [units]
TOT_PRIMARY_MAT(mat_cat,year)                   Total primary resources required by year                    [kg]

* Slack variables for debugging
SLACK_TEC_ADD(newtec,seg,fleetreg,year,age)     slack variable for tech uptake constraint relaxation
SLACK_VIRG_MAT(mat_prod,year)                   slack variable for primary supply constraint relaxation
SLACK_NEW_COMPONENTS(year)                      slack variable manufacturing constraint relaxation
;

POSITIVE VARIABLES
TOT_STOCK(tec,seg,fleetreg,year,age)            Number of units of a given age in a given year             [units]
STOCK_REMOVED(tec,seg,fleetreg,year,age)        Number of units of a given age retired in a given year     [units]
TOT_IMPACTS(imp,tec,seg,fleetreg,year)          Total impacts per year by technology                       [t CO2-eq]
STOCK_ADDED(tec,seg,fleetreg,year,age)          Stock additions (new tec sales)                            [units]

PRODUCTION_IMPACTS(imp,tec,seg,fleetreg,year)   Total impacts from production of units per year            [t CO2-eq]
OPERATION_IMPACTS(imp,tec,seg,fleetreg,year)    Total impacts from operations of units per year            [t CO2-eq]
EOL_IMPACTS(imp,tec,seg,fleetreg,year)          Total impacts from unit end of life treatment per year     [t CO2-eq]

RECYCLED_COMPONENTS(newtec,fleetreg,year,age)   Total components sent to recycling per year                [kWh]
RECYCLED_MAT(mat_cat,year)                      Materials recovered from recycled components               [kg]
MAT_REQ(mat_cat,year)                           Total amount of critical materials needed for new units    [kg]

MAT_MIX(mat_prod,year)                          Production mixes for virgin materials                      [t]
MAT_IMPACTS(imp,mat_prod,year)                  Total impacts from virgin material production per year     [t CO2e kg^-1]
;



*-----------------------------------------------------------------------------------
*
* LP model equation declarations
*
*-----------------------------------------------------------------------------------

EQUATIONS

***DYNAMIC STOCK MODEL  ------------------------------------------------------------

***  Fleet model
EQ_STCK_CHANGE                  Net total stock change
EQ_STCK_REM                     Removal from stock
EQ_STCK_BAL                     Fleet balance constraint

*** Constraint equations
EQ_FLEET_BAL_CONSTRAINT         Fleet balance constraint
EQ_TEC_UPTAKE_CONSTRAINT        Technology uptake constraint
EQ_SEG_SHARE_CONSTRAINT         Segment share constraint
EQ_MANUF_CONSTRAINT             Manufacturing constraint

EQ_RECYCLED_COMPONENTS          Total retired components
EQ_MAT_REQ                      Total required critical materials to satisfy demand
EQ_RECYCLED_MAT                 Total recovered critical materials
EQ_MAT_TOT_PRIMARY              Total required primary critical materials
EQ_MAT_BALANCE                  Critical material balance constraint - total supply   
EQ_PRIM_MAT_SUPPLY_CONSTRAINT   Critical material primary supply constraint

**EMISSION and ENERGY MODELS incl OBJ. FUNCTION --------------------------------------

EQ_OBJ                          Objective function - total fleet lifecycle impacts over optimization period
EQ_TOTAL_IMPACTS                Calculation of impacts from all units per year
EQ_PRODUCTION_IMPACTS           Total production emissions including from materials
EQ_PRIMARY_MAT_IMPACTS          Primary material extraction and processing emissions
EQ_OPERATION_IMPACTS            Total fleet operating emissions by year
EQ_EOL_IMPACTS                  Total fleet end of life emissions by year
;

*-----------------------------------------------------------------------------------
*
* LP model equations
*
*-----------------------------------------------------------------------------------

************************************************************
** INITIALIZE STOCK MODEL                                ***
************************************************************
* First, initialize first year (boundary condition)
TOT_STOCK.fx(tec,seg,fleetreg,modelyear,age)$(ord(modelyear)=1) = EXOG_TOT_STOCK(fleetreg,modelyear) * LIFETIME_AGE_DISTRIBUTION(age) * INITIAL_TEC_SHARES(tec)  * INITIAL_SEG_SHARES(seg);
* try below with age+1 on LHS to avoid weird offset thing?
STOCK_REMOVED.fx(tec,seg,fleetreg,modelyear,age)$(ord(modelyear)=1) = TOT_STOCK.l(tec,seg,fleetreg,modelyear,age-1) * RETIREMENT_FUNCTION(age-1);
STOCK_ADDED.fx(tec,seg,fleetreg,modelyear,new)$(ord(modelyear)=1) = sum(age, STOCK_REMOVED.l(tec,seg,fleetreg,modelyear,age));
* Do not add "new" units with age (e.g., used cars from other regions)
STOCK_ADDED.fx(tec,seg,fleetreg,modelyear,age)$(ord(age)>1) = 0;

* Initiate stock in remaining initialization period
loop(modelyear $ (ord(modelyear)>1 and ord(modelyear) <= card(inityear)),
STOCK_CHANGE.fx(fleetreg,modelyear) = EXOG_TOT_STOCK(fleetreg,modelyear) - EXOG_TOT_STOCK(fleetreg,modelyear-1);
* calculate stock removals as per survival curves
STOCK_REMOVED.fx(tec,seg,fleetreg,modelyear,age) = TOT_STOCK.l(tec,seg,fleetreg,modelyear-1,age-1) * RETIREMENT_FUNCTION(age-1);
* in initialization period, assume that new units maintain the status quo of tec and segment shares
STOCK_ADDED.fx(tec,seg,fleetreg,modelyear,new) = sum(age, STOCK_REMOVED.l(tec,seg,fleetreg,modelyear,age)) + (STOCK_CHANGE.l(fleetreg,modelyear) * INITIAL_TEC_SHARES(tec) * INITIAL_SEG_SHARES(seg));
TOT_STOCK.fx(tec,seg,fleetreg,modelyear,age) = TOT_STOCK.l(tec,seg,fleetreg,modelyear-1,age-1) - STOCK_REMOVED.l(tec,seg,fleetreg,modelyear,age) + STOCK_ADDED.l(tec,seg,fleetreg,modelyear,age);
);

************************************************************
***  Fleet balance model                                 ***
************************************************************

* Net change in stock from year to year - exogeneously defined (see Constraint A)
EQ_STCK_CHANGE(modelyear,fleetreg)$(ord(modelyear)>card(inityear))..             STOCK_CHANGE(fleetreg,modelyear)  =e=  EXOG_TOT_STOCK(fleetreg,modelyear) - EXOG_TOT_STOCK(fleetreg,modelyear-1);

* Equation 2
EQ_STCK_BAL(tec,seg,fleetreg,modelyear,age)$(ord(modelyear) > card(inityear))..  TOT_STOCK(tec,seg,fleetreg,modelyear,age)  =e= TOT_STOCK(tec,seg,fleetreg,modelyear-1,age-1) + STOCK_ADDED(tec,seg,fleetreg,modelyear,age) - STOCK_REMOVED(tec,seg,fleetreg,modelyear,age);
  
* Equation 3
EQ_STCK_REM(tec,seg,fleetreg,modelyear,age)$(ord(modelyear)>card(inityear))..    STOCK_REMOVED(tec,seg,fleetreg,modelyear,age) =e= TOT_STOCK(tec,seg,fleetreg,modelyear-1,age-1) * RETIREMENT_FUNCTION(age-1);                               

************************************************************
*** Constraints                                          ***
************************************************************
*--------------------------------------
* A - Stock balance constraint
*--------------------------------------
* Equation 9
EQ_FLEET_BAL_CONSTRAINT(modelyear,fleetreg)$(ord(modelyear)>card(inityear))..    sum((tec,seg), STOCK_ADDED(tec,seg,fleetreg,modelyear,new)) =e= sum((tec,seg,age), STOCK_REMOVED(tec,seg,fleetreg,modelyear, age)) + STOCK_CHANGE(fleetreg,modelyear);

*--------------------------------------
* B - Manufacturing constraint
*--------------------------------------
* Equation 10
* total capacity added per year must be less than the component manufacturing capacity
* MANUF_CNSTRNT input is in GWh; TEC_SIZE_CORRESPONDENCE is in kWh
EQ_MANUF_CONSTRAINT(newtec,optyear)..                MANUF_CNSTRNT(newtec,optyear)*1e6 =g= sum((seg, fleetreg), STOCK_ADDED(newtec,seg,fleetreg,optyear,new) * TEC_SIZE_CORRESPONDENCE(newtec,seg))
%SLACK_NEW_COMPONENTS% - SLACK_NEW_COMPONENTS(optyear)
;

*-----------------------------------------------
* C - Material balance constraint - total supply
*-----------------------------------------------
* Equation 11 (split into five equations here)
* Amount of virgin + recycled material to produce new components, in kg; see intermediate calculations below
EQ_MAT_TOT_PRIMARY(mat_cat, optyear)..               sum(mat_prod$mat(mat_cat, mat_prod), MAT_MIX(mat_prod,optyear)) =e=  TOT_PRIMARY_MAT(mat_cat,optyear);

* Primary material demand
EQ_MAT_BALANCE(mat_cat,optyear)..                    TOT_PRIMARY_MAT(mat_cat,optyear) + RECYCLED_MAT(mat_cat,optyear) =e= MAT_REQ(mat_cat,optyear);

* Material supply balance
* Total amount of material required calculated from new units entering the market, in kg
EQ_MAT_REQ(mat_cat,optyear)..                        MAT_REQ(mat_cat,optyear) =e= sum(newtec, sum((seg, fleetreg), STOCK_ADDED(newtec,seg,fleetreg,optyear,new) * TEC_SIZE_CORRESPONDENCE(newtec,seg)) * MAT_CONTENT(newtec, mat_cat, optyear)
%SLACK_NEW_COMPONENTS% + SLACK_NEW_COMPONENTS(optyear) * MAT_CONTENT(newtec, mat_cat, optyear)
);

* Calculate amount of materials recovered from end-of-life components for recycling (i.e., to new components), in kg
EQ_RECYCLED_MAT(mat_cat,optyear)..                   RECYCLED_MAT(mat_cat,optyear) =e= sum((newtec,prodyear,age), sum((fleetreg), RECYCLED_COMPONENTS(newtec,fleetreg,optyear,age)) * COHORT_AGE_CORRESPONDENCE(optyear,prodyear,age) * MAT_CONTENT(newtec, mat_cat, optyear)) * RECYCLING_YIELD(mat_cat,optyear)
;

* Calculate amount of components from new technologies retired (and collected) each year in kWh
EQ_RECYCLED_COMPONENTS(newtec,fleetreg,optyear,age)..             RECYCLED_COMPONENTS(newtec,fleetreg,optyear,age) =e= sum((seg), STOCK_REMOVED(newtec,seg,fleetreg,optyear,age) *  TEC_SIZE_CORRESPONDENCE(newtec,seg)) * RECOVERY_PCT(newtec, optyear)
;

*------------------------------------------------
* D - Critical material primary supply constraint
*------------------------------------------------
* Equation 12
* demand of virgin resources from each source must be less than or equal to available supply; see intermediate calculations below
* primary resources constraint provided in t, mat_mix in kg
EQ_PRIM_MAT_SUPPLY_CONSTRAINT(mat_prod,optyear)..                MAT_MIX(mat_prod,optyear) =l= VIRG_MAT_SUPPLY(mat_prod,optyear)*1000
%SLACK_VIRG_MAT% + SLACK_VIRG_MAT(mat_prod,optyear)
;

*---------------------------------
* E - Technology uptake constraint
*---------------------------------
* Equation 13
* incumbent technology excluded
EQ_TEC_UPTAKE_CONSTRAINT(newtec,seg,fleetreg,modelyear)$(ord(modelyear)>card(inityear))..     STOCK_ADDED(newtec,seg,fleetreg,modelyear,new) =l= ((1 + MAX_UPTAKE_RATE('IND', newtec)) * STOCK_ADDED(newtec,seg,fleetreg,modelyear-1,new)) + UPTAKE_CONSTANT
%SLACK_TEC_ADD% + SLACK_TEC_ADD(newtec,seg,fleetreg,modelyear,new)
;

*------------------------------------------------------
* F - Segment share constraint (segments kept constant)
*------------------------------------------------------
* Equation 14
* Removing allows for e.g., smaller segments to be replaced by larger segments...
EQ_SEG_SHARE_CONSTRAINT(seg,fleetreg,modelyear)$(ord(modelyear)>card(inityear))..             sum(tec, STOCK_ADDED(tec,seg,fleetreg,modelyear,new))
    =e= INITIAL_SEG_SHARES(seg) * sum((tec,segj), STOCK_ADDED(tec,segj,fleetreg,modelyear,new));

************************************************************
*** LIFECYCLE IMPACTS AND OBJECTIVE FUNCTION             ***
************************************************************
*===================
* Objective function
*===================
* Equation 1
EQ_OBJ..                                              TOT_IMPACTS_OPT =e= sum((tec,seg,fleetreg,optyear), TOT_IMPACTS(optimp,tec,seg,fleetreg,optyear));

* Calculation of emissions from all units classes per year
EQ_TOTAL_IMPACTS(imp,tec,seg,fleetreg,modelyear)..          TOT_IMPACTS(imp,tec,seg,fleetreg,modelyear) =e= PRODUCTION_IMPACTS(imp,tec,seg,fleetreg,modelyear) +  OPERATION_IMPACTS(imp,tec,seg,fleetreg,modelyear) + EOL_IMPACTS(imp,tec,seg,fleetreg,modelyear);

;

* Calculate emissions from production
* Equation 7 (split into two equations for production and primary materials)
EQ_PRODUCTION_IMPACTS(imp,tec,seg,fleetreg,modelyear)..     PRODUCTION_IMPACTS(imp,tec,seg,fleetreg,modelyear) =e= STOCK_ADDED(tec,seg,fleetreg,modelyear,new)*TEC_PROD_IMPACT_INT(imp,tec,seg,modelyear) + sum(mat_prod, MAT_IMPACTS(imp,mat_prod,modelyear))
%SLACK_TEC_ADD% + SLACK_TEC_ADD(newtec,seg,fleetreg,modelyear,new) * TEC_PROD_IMPACT_INT(imp,tec,seg,modelyear)*1e6
;

* Calculate emissions from virgin materials. Currently assumes zero emissions for recycled materials.
EQ_PRIMARY_MAT_IMPACTS(imp,mat_prod,modelyear)..            MAT_IMPACTS(imp,mat_prod,modelyear) =e= (MAT_MIX(mat_prod, modelyear) * MAT_IMPACT_INT(imp, mat_prod, modelyear))/1000
* for dynamic background in material mix - not yet implemented
*%DYNAMIC_MAT% + MAT_MIX(mat_prod, model_year)*MAT_PROD_EL_INT(tec,seg,prodyear)*ENR_IMPACT_INT(imp,'elc','prod',prodyear)/1000
%SLACK_VIRG_MAT% + SLACK_VIRG_MAT(mat_prod,modelyear)*1e6
;

* Calculate impacts from operation
* Equation 7
EQ_OPERATION_IMPACTS(imp,tec,seg,fleetreg,modelyear)..      OPERATION_IMPACTS(imp,tec,seg,fleetreg,modelyear) =e= sum( (agej,enr,prodyear), TOT_STOCK(tec,seg,fleetreg,modelyear,agej) * COHORT_AGE_CORRESPONDENCE(modelyear,prodyear,agej)* TEC_OPER_IMPACT_INT(imp,tec,enr,seg,fleetreg,modelyear,prodyear,agej) *  ANNUAL_USE_INTENSITY(fleetreg,modelyear))
;

* Calculate emissions from disposal and recycling
* Equation 8
EQ_EOL_IMPACTS(imp,tec,seg,fleetreg,modelyear)..            EOL_IMPACTS(imp,tec,seg,fleetreg,modelyear) =e= sum((agej),STOCK_REMOVED(tec,seg,fleetreg,modelyear,agej)) * TEC_EOLT_IMPACT_INT(imp,tec,seg,modelyear)
;


*-----------------------------------------------------------------------------------
*
* Model Definition and Options
*
*-----------------------------------------------------------------------------------

* Defining name of model(s) and what equations are used in each model

MODEL ECOPT2_Basic     "default model run in normal mode"      /ALL/
;

* Model variants and troubleshooting models
MODEL 
      seg_test          "model without segment constraint"      /ECOPT2_Basic - EQ_SEG_SHARE_CONSTRAINT/
      tec_test          "model without growth constraint"       /ECOPT2_Basic - EQ_TEC_UPTAKE_CONSTRAINT/
 
      no_mat            "model with no critical material constraints"           /ECOPT2_Basic - EQ_RECYCLED_MAT - EQ_RECYCLED_COMPONENTS - EQ_MAT_REQ - EQ_MAT_TOT_PRIMARY -                        EQ_MAT_BALANCE - EQ_PRIM_MAT_SUPPLY_CONSTRAINT/

      manuf_test        "model without manufacturing capacity constraint"       /ECOPT2_Basic - EQ_MANUF_CONSTRAINT/
      fleet_test        "model without growth or manufacturing constraint"      /tec_test - EQ_MANUF_CONSTRAINT/
      mat_test          "model without material constraint"                     /ECOPT2_Basic - EQ_MAT_BALANCE/
      primary_mat_test  "model without primary material supply constraint"      /ECOPT2_Basic - EQ_PRIM_MAT_SUPPLY_CONSTRAINT/
      test_model        "model with only manufacturing and growth constraints"  /no_mat - EQ_SEG_SHARE_CONSTRAINT/
      no_constraints    "model without constraints"                             /no_mat - EQ_SEG_SHARE_CONSTRAINT - EQ_TEC_UPTAKE_CONSTRAINT - EQ_MANUF_CONSTRAINT/
;


* Defining model run options and solver

*OPTION RESLIM = 2000000;
*OPTION limrow = 0;
*OPTION limcol = 0;
*OPTION PROFILE = 0;
* set to PROFILE = 2 for debugging 
*tec_test.optfile=1;

Scalar ms 'model status', ss 'solve status';

*-----------------------------------------------------------------------------------                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
*
* Model Execution
*
*-----------------------------------------------------------------------------------

SOLVE ECOPT2_Basic USING LP MINIMIZING TOT_IMPACTS_OPT;
ss = ECOPT2_Basic.solvestat;
ms = ECOPT2_Basic.modelstat;

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

* setup incumbent technology for calculating business-as-usual cases
set incumb_tec(tec);
incumb_tec(tec) = tec(tec)-newtec(tec);

TOT_NEW_TECS(modelyear) = sum((newtec,seg, fleetreg, age), TOT_STOCK.l(newtec, seg, fleetreg, modelyear,age)); 
* total capacity of new components added by year 
TOT_NEWTEC_COMP_ADDED(newtec, modelyear) = sum((seg,fleetreg)$STOCK_ADDED.l(newtec, seg, fleetreg, modelyear, new), STOCK_ADDED.l(newtec,seg,fleetreg,modelyear,new)*TEC_SIZE_CORRESPONDENCE(newtec,seg))/1e6;
TOT_NEWTEC_COMP_RECYCLED(modelyear) = sum((seg), sum((newtec,fleetreg, age), STOCK_REMOVED.l(newtec,seg,fleetreg,modelyear,age)* TEC_SIZE_CORRESPONDENCE(newtec,seg)));
MAT_REQ_TOT(mat_cat,modelyear) = sum((newtec,seg, fleetreg), STOCK_ADDED.l(newtec,seg,fleetreg,modelyear,new)*TEC_SIZE_CORRESPONDENCE(newtec,seg)*MAT_CONTENT(newtec,mat_cat,modelyear));

TOT_STOCK_ADDED(fleetreg,modelyear) = sum((tec,seg), STOCK_ADDED.l(tec, seg, fleetreg, modelyear, new));
TOT_STOCK_REMOVED(fleetreg, modelyear) = sum((tec,seg,age), STOCK_REMOVED.l(tec, seg, fleetreg, modelyear, age));
TOT_STOCK_CHECK(fleetreg, modelyear) = sum((tec,seg,age), TOT_STOCK.l(tec,seg,fleetreg, modelyear, age));

* summing the number of units in fleet as check.
EXOG_TOT_STOCK_CHECK(modelyear) = sum((tec,seg,fleetreg,age), TOT_STOCK.l(tec,seg,fleetreg,modelyear,age));
STOCK_BY_COHORT(tec,seg,fleetreg,modelyear,prodyear,agej) $ COHORT_AGE_CORRESPONDENCE(modelyear,prodyear,agej) = (TOT_STOCK.l(tec,seg,fleetreg,modelyear,agej)*COHORT_AGE_CORRESPONDENCE(modelyear,prodyear,agej));

* total operation emissions by cohort and model year
OPER_IMPACT_COHORT(imp,tec, seg, fleetreg,modelyear,prodyear,age) = sum((enr), TOT_STOCK.l(tec,seg,fleetreg,modelyear,age) * COHORT_AGE_CORRESPONDENCE(modelyear,prodyear,age)* TEC_OPER_IMPACT_INT(imp,tec,enr,seg,fleetreg,modelyear,prodyear,age) * ANNUAL_USE_INTENSITY(fleetreg,modelyear));
EOL_IMPACT_COHORT(imp,tec,seg,fleetreg,modelyear,prodyear,age) = (STOCK_REMOVED.l(tec,seg,fleetreg,modelyear,age)* COHORT_AGE_CORRESPONDENCE(modelyear,prodyear,age)) * TEC_EOLT_IMPACT_INT(imp,tec,seg,modelyear);
TOTAL_OPERATION_ENERGY(tec,seg,fleetreg,modelyear) = sum((prodyear, age), TOT_STOCK.l(tec,seg,fleetreg,modelyear,age) * COHORT_AGE_CORRESPONDENCE(modelyear,prodyear,age) * TEC_OPER_EINT(tec,seg,prodyear) * ANNUAL_USE_INTENSITY(fleetreg,modelyear));

* calculate emissions for 0 electrification (business as usual)
BAU_PROD(imp,modelyear) = sum(seg, sum((tec,fleetreg), STOCK_ADDED.l(tec, seg, fleetreg, modelyear, new)) * sum(incumb_tec, TEC_PROD_IMPACT_INT(imp,incumb_tec,seg,modelyear)));
BAU_OPER(imp,modelyear) = sum((fleetreg,age,prodyear,seg), sum((tec), TOT_STOCK.l(tec,seg,fleetreg,modelyear,age)) * sum((incumb_tec,enr), TEC_OPER_IMPACT_INT(imp,incumb_tec, enr, seg, fleetreg, modelyear, prodyear, age)$(ENR_TEC_CORRESPONDENCE(incumb_tec,enr))) * COHORT_AGE_CORRESPONDENCE(modelyear,prodyear,age) * ANNUAL_USE_INTENSITY(fleetreg,modelyear));
BAU_EOL(imp,modelyear)  = sum(seg, sum((tec, fleetreg,age), STOCK_REMOVED.l(tec,seg,fleetreg,modelyear,age))  * sum(incumb_tec, TEC_EOLT_IMPACT_INT(imp,incumb_tec,seg,modelyear)));
BAU_IMPACTS(imp,modelyear) = BAU_PROD(imp,modelyear) + BAU_OPER(imp,modelyear) + BAU_EOL(imp,modelyear);



execute_unload 'ECOPT2_addset.gdx'
