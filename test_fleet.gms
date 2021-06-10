SETS
year            total span of years - including production before intialization period /1972*2080/
modelyear(year) model years (2000-2050) /2000*2050/
optyear(year)   years for optimization (2020-2050) /2021*2080/
inityear(year)  years for initialization (2000-2020) /2000*2020/
age             age of vehicle /0*28/
tec             technology /ICE, BEV/
enr             energy carrier
*reg             region or country group /LOW, HIGH, PROD/
*fleetreg(reg)     model regions of BEV operation /LOW, HIGH/

SINGLETON SETS
new(age)    /0/
;

alias (year, prodyear)
alias (prodyear, year)
alias (age, agej)
alias (agej, age)
alias (tec, tecj)
alias (tecj, tec)

;
PARAMETERS
VEH_STCK_TOT(year)           Number of vehicles - # 
VEH_OPER_DIST(year)              Annual driving distance per vehicles                   [km]

** LIFETIME -----------------
VEH_LIFT_CDF(age)                Age CDF Share of scrapping for a given age - e.g % in given age class i scrapped
VEH_LIFT_AGE(age)                Age distribution (PDF)
VEH_LIFT_MOR(age)                Age distribution = 1 - CDF

** COMPOSITION --------------
VEH_PAY(prodyear,age,year)       Correspondence between a vehicle production year and its age (up to 20) in a given year
VEH_STCK_INT_TEC(tec)            Initial share of vehicles in stock tech /BEV 0, ICE 1/
VEH_STCK_INT(tec,age)    Initial size of stock of vehicles by age cohort and segment

;

FREE VARIABLES
TOTC                                        Total CO2 emissions for the whole system over the whole period
TOTC_OPT                                    Total CO2 emissions for the whole system over optimization period
VEH_STCK_DELTA(year)               Net change in stock from one year to the next
;

POSITIVE VARIABLES
VEH_STCK(tec,year,age)              Number of vehicles of a given age in a given year
VEH_STCK_REM(tec,year,age)          Number of vehicles of a given age retired in a given year
VEH_TOTC(tec,year)                  Total CO2 emissions of vehicles per year by technology              [t CO2-eq]
VEH_STCK_ADD(tec,year,age)          Stock additions (new car sales)

VEH_PROD_TOTC(tec,year)             Total CO2 emissions from production of vehicles per year            [t CO2-eq]
VEH_OPER_TOTC(tec,year)             Total CO2 emissions from operations of vehicles per year            [t CO2-eq]
VEH_EOLT_TOTC(tec,year)             Total CO2 emissions from vehicle end of life treatment per year     [t CO2-eq]

;

EQUATIONS
EQ_STCK_DELTA
EQ_STCK_REM
EQ_STCK_ADD
EQ_STCK_BAL
;

$if not set gdxincname $gdxin C:\Users\chrishun\Box Sync\YSSP_temp\pyGAMS_input.gdx
$load VEH_LIFT_CDF
$load VEH_LIFT_AGE
$load VEH_LIFT_MOR
$gdxin

loop(modelyear,
VEH_STCK_TOT(modelyear) = 100000;
);


** First, initialize first year (boundary condition)
VEH_STCK.fx(tec,modelyear,age)$(ord(modelyear)=1) = VEH_STCK_TOT(modelyear) * VEH_STCK_INT_TEC(tec) * VEH_LIFT_AGE(age);
*VEH_STCK_DELTA.fx(modelyear)$(ord(modelyear)=1) = 0;
*VEH_STCK_REM.fx(tec,modelyear,age)$(ord(modelyear)=1) = VEH_STCK.l(tec,modelyear-1,age-1) * VEH_LIFT_MOR(age-1);
*VEH_STCK_REM.fx(tec,modelyear,age)$(ord(age)=1) = 0;
** VEH_LIFT_MOR(age);
**VEH_STCK_REM.fx(tec,seg,fleetreg,inityear,age)$(ord(inityear)=1) = VEH_STCK.l(tec,seg,fleetreg,inityear,age) * VEH_LIFT_MOR(age);
*VEH_STCK_ADD.fx(tec,modelyear,new)$(ord(modelyear)=1) = sum(age, VEH_STCK_REM.l(tec,modelyear,age));
*VEH_STCK.l(tec,seg,fleetreg,modelyear,new);
* Do not add "new" vehicles with age
VEH_STCK_ADD.fx(tec,modelyear,age)$(ord(age)>1) = 0;


loop(modelyear $ (ord(modelyear)<= card(inityear)),
VEH_STCK_DELTA.fx(modelyear) = VEH_STCK_TOT(modelyear) - VEH_STCK_TOT(modelyear-1);
*VEH_STCK_REM.fx(tec,seg,fleetreg,modelyear,age) = VEH_STCK.l(tec, seg,fleetreg,modelyear,age) - VEH_STCK.l(tec,seg,fleetreg,modelyear-1, age-1)
*VEH_STCK_REM.fx(tec, modelyear+1, age+1) = VEH_STCK.l(tec,modelyear, age) * VEH_LIFT_MOR(age);
VEH_STCK_REM.fx(tec,modelyear,age) = VEH_STCK.l(tec,modelyear-1,age-1) * VEH_LIFT_MOR(age-1);
* in initialization period, assume that new vehicles maintain the status quo of tec and segment shares
VEH_STCK_ADD.fx(tec,modelyear,new) = sum(age, VEH_STCK_REM.l(tec,modelyear,age)) + (VEH_STCK_DELTA.l(modelyear)*VEH_STCK_INT_TEC(tec));
VEH_STCK.fx(tec,modelyear,age)$(ord(modelyear)>1) = VEH_STCK.l(tec,modelyear-1,age-1) - VEH_STCK_REM.l(tec,modelyear,age) + VEH_STCK_ADD.l(tec,modelyear,age);
);