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
*$ontext
year            total span of years, including production before intialization period
modelyear(year) model years (2000-2050)
optyear(year)   years for optimization (2020-2050)
inityear(year)  years for initialization (2000-2020)
age             age of vehicle
tec             technology
enr             energy
reg             region or country group /HIGH, LOW, PROD/
seg             segment or size class
sigvar          variables for sigmoid equations
dstvar          variables for statistical distributions
enreq           equations for energy (electricity and fuels) system
veheq           equations for vehicle parameters
demeq           equations for demand parameters
lfteq           equations for fleet lifetime parameters
grdeq           parameters for gradient of change (fleet additions) - individual (IND) for each tech or related to all tech (ALL)

*$offtext

$ontext
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
grdeq          parameters for gradient of change (fleet additions) - individual (IND) for each tech or related to all tech (ALL) /IND,ALL/
$offtext

*---- ABBREIVATIONS USED *-----------------------------------------------------------------------------------
* PROD = Production
* OPER = Operation
* ENIT = Enerqy intensity
* CINT = CO2-eq intensity
* CNST = b in y = ax + b
;

** Load sets defined in Python class
* These lines must be uncommented if specific input .gdx is not specified, e.g., $gdxin <filepath>_input.gdx
$ontext
$if not set gdxincname $abort 'no include file name for data file provided'
$gdxin %gdxincname%
$offtext

*$ontext
*$gdxin C:\Users\chrishun\Box Sync\YSSP_temp\model run data\run_slow_baseline_def_def_def_def_iTEM2_Base2019-09-24T12_50_input.gdx
*$offtext

$ontext
$gdxin C:\Users\chrishun\Box Sync\YSSP_temp\run_def_baseline_def_def_def_def_iTEM2-Base2019-10-04T18_47_input.gdx
$offtext

$ontext
$gdxin C:\Users\chrishun\Box Sync\YSSP_temp\run_def_baseline_def_def_def_def_def_iTEM2-Base2019-10-25T18_50_input.gdx
$offtext

$gdxin C:\Users\chrishun\Box Sync\YSSP_temp\Cleanup April 2020\run_def_baseline_def_def_def_def_aggr_iTEM2-Base2019-10-23T07_40_input.gdx

*$gdxin C:\Users\chrishun\Box Sync\YSSP_temp\pyGAMS_input.gdx

* This .gdx is for troubleshooting unwa
*$GDXIN 'troubleshooting_params'
$LOAD year
$LOAD modelyear
$LOAD optyear
$LOAD inityear
$LOAD age
$LOAD tec
$LOAD enr
$LOAD seg

$LOAD sigvar
$LOAD dstvar
$LOAD enreq
$LOAD veheq
$LOAD demeq
$LOAD lfteq
$LOAD grdeq
*
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
* DEM_PARTAB(demeq,sigvar)         variables for demand equations
ENR_PARTAB(enr,reg,enreq,sigvar)     variables for each energy (fos or elc) equation
*LFT_PARTAB(dstvar)               variables for fleet lifetime equations

***ENERGY (ELECTRICITY GENERATION and FUEL) ------------------------------------------
ENR_CINT(enr,reg,year)               CO2 intensity of the regional energy mixes            [kg CO2-eq pr kwh]

***ENERGY and VEHICLE TECHONLOGY COMBINATIONS ----------------------------------------
ENR_VEH(enr,tec)                 feasible combinations of vehicle technology and energy (fuel).


***All VEHICLES ----------------------------------------------------------------------

**PRODUCTION
VEH_PROD_EINT(tec,seg,prodyear)        Electricity intensity of vehicle prod                [kwh el required per vehicle produced]
VEH_PROD_CINT_CSNT(tec,seg,prodyear)   Constant term for CO2 int. of vehicle production     [t CO2-eq per vehicle produced]
VEH_PROD_CINT(tec,seg,prodyear)        CO2 intensity of vehicle production                  [t CO2-eq per vehicle produced]

**OPERATION
VEH_OPER_EINT(tec,seg,prodyear)                     Energy intensity of vehicle operation   [kwh per km]
VEH_OPER_CINT(tec,enr,seg,reg,age,modelyear,prodyear)    CO2 intensity of vehicle operation     [t CO2 per km]
*VEH_OPER_CINT2(tec,enr,seg,prodyear,age,modelyear)
OPER_TRIAL(tec,enr,seg,prodyear,age,modelyear)
**EOL
VEH_EOLT_CINT(tec,seg,year)            CO2 intensity of ICE vehicle EOL                     [t CO2-eq per vehicle in EOL treatment]
COHORT_OPER_EMISSIONS(tec,enr,seg,modelyear)
VEH_LC_EMISS(tec,seg,prodyear)         Lifecycle emissions for each tec-seg-vintage cohort

** FLEET -------------------------------------------------------------------------------
** INITIAL STOCK ------------
$ontext
INIT_SEG(seg)
INIT_TEC(tec)
INIT_AGE(age)
SEG_TEC(seg,tec)
SEG_TEC_AGE(seg,tec,age)
$offtext
BEV_CAPAC(seg)                  Correspondence of battery capacities used in each segment[in kWh]

**DEMAND --------------------

VEH_STCK_TOT(year,reg)               Number of vehicles - #
VEH_OPER_DIST(year)              Annual driving distance per vehicles                   [km]
*VEH_OCUP(year)                                  Load factor for vehicles                                                               [passengers per vehicle]

** LIFETIME

VEH_LIFT_PDF(age)                Age PDF
VEH_LIFT_CDF(age)                Age CDF Share of scrapping for a given age - e.g % in given age class i scrapped
VEH_LIFT_AGE(age)                Age distribution = 1 - CDF
VEH_LIFT_MOR(age)                Age distribution = 1 - CDF

** COMPOSITION --------------

VEH_PAY(prodyear,age,year)       Correspondence between a vehicle production year and its age (up to 20) in a given year
VEH_STCK_INT_TEC(tec)            Initial share of vehicles in stock tech
VEH_STCK_INT_SEG(seg)            Initial stock distribution by segment
VEH_STCK_INT(tec,seg,reg,age)        Initial size of stock of vehicles by age cohort and segment

** CONSTRAINTS -------
VEH_ADD_GRD(grdeq,tec)           Parameter for gradient of change constraint (fleet additions) - individual (IND) for each tech or related to all tech (ALL)
*VEH_SEG_SHR(seg)                 Parameter for segment share minimums in fleet
GRO_CNSTRNT(year)                Segment growth rate constraint relative to stock additions
MANUF_CNSTRNT(year)             Annual manufacturing capacity (for batteries destined for Europe), in MWh
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
*$ONMULTI
*$GDXIN 'EVD4EUR_input'
*$LOAD DEM_PARTAB
* Vehicle load rate not required due to calculations performed in python
*$LOAD VEH_OCUP
*$GDXIN

* Load in parameter values defined in Python class
*$gdxin C:\Users\chrishun\Box Sync\YSSP_temp\run_def_baseline_def_def_def_def_iTEM2-Base2019-10-04T18_47_input.gdx
*$gdxin C:\Users\chrishun\Box Sync\YSSP_temp\run_slow_baseline_def_def_def_def_iTEM2_Base2019-09-24T12_50_input.gdx
*$gdxin C:\Users\chrishun\Box Sync\YSSP_temp\GAMS_input.gdx
*$gdxin C:\Users\chrishun\Box Sync\YSSP_temp\pyGAMS_input.gdx

$gdxin C:\Users\chrishun\Box Sync\YSSP_temp\Cleanup April 2020\run_def_baseline_def_def_def_def_aggr_iTEM2-Base2019-10-23T07_40_input.gdx

*$if not set gdxincname $abort 'no include file name for data file provided'
*$gdxin %gdxincname%

$LOAD YEAR_PAR
$LOAD VEH_PARTAB
*$LOAD ENR_PARTAB # declared manually below

*$LOAD VEH_PROD_CINT_CSNT
*$LOAD VEH_PROD_EINT
*$LOAD VEH_OPER_EINT
*$LOAD VEH_EOLT_CINT

$LOAD ENR_VEH

*$LOAD VEH_STCK_TOT
$LOAD VEH_OPER_DIST

$LOAD VEH_LIFT_PDF
$LOAD VEH_LIFT_CDF
$LOAD VEH_LIFT_AGE
$LOAD VEH_LIFT_MOR

$LOAD VEH_PAY
$LOAD VEH_STCK_INT_TEC
$LOAD VEH_STCK_INT_SEG

*$LOAD VEH_STCK_INT
$LOAD BEV_CAPAC
$LOAD VEH_ADD_GRD
*$LOAD VEH_SEG_SHR
$LOAD GRO_CNSTRNT
$LOAD MANUF_CNSTRNT
$OFFMULTI
$GDXIN
;

TABLE ENR_PARTAB(enr,reg,enreq,sigvar)
                    A       B       r         u
ELC .HIGH .CINT     1       0.275   0.16    2030
ELC .LOW  .CINT     .5      0.1     0.2     2030
ELC .PROD .CINT     0.7     0.7     0.7     2030
FOS .HIGH .CINT     0.3     0.25    0.2     2035
FOS .LOW  .CINT     0.3     0.25    0.2     2035
;

PARAMETER VEH_STCK_TOT(year, reg)
/
2000	.HIGH	=       500000
2001	.HIGH	=       500000
2002	.HIGH	=       500000
2003	.HIGH	=       500000
2004	.HIGH	=       500000
2005	.HIGH	=       500000
2006	.HIGH	=       500000
2007	.HIGH	=       500000
2008	.HIGH	=       500000
2009	.HIGH	=       500000
2010	.HIGH	=       500000
2011	.HIGH	=       500000
2012	.HIGH	=       500000
2013	.HIGH	=       500000
2014	.HIGH	=       500000
2015	.HIGH	=       500000
2016	.HIGH	=       500000
2017	.HIGH	=       500000
2018	.HIGH	=       500000
2019	.HIGH	=       500000
2020	.HIGH	=       509677.4194
2021	.HIGH	=       519354.8387
2022	.HIGH	=	529032.2581
2023	.HIGH	=	538709.6774
2024	.HIGH	=	548387.0968
2025	.HIGH	=	558064.5161
2026	.HIGH	=	567741.9355
2027	.HIGH	=	577419.3548
2028	.HIGH	=	587096.7742
2029	.HIGH	=	596774.1935
2030	.HIGH	=	606451.6129
2031	.HIGH	=	616129.0323
2032	.HIGH	=	625806.4516
2033	.HIGH	=	635483.871
2034	.HIGH	=	645161.2903
2035	.HIGH	=	654838.7097
2036	.HIGH	=	664516.129
2037	.HIGH	=	674193.5484
2038	.HIGH	=	683870.9677
2039	.HIGH	=	693548.3871
2040	.HIGH	=	703225.8065
2041	.HIGH	=	712903.2258
2042	.HIGH	=	722580.6452
2043	.HIGH	=	732258.0645
2044	.HIGH	=	741935.4839
2045	.HIGH	=	751612.9032
2046	.HIGH	=	761290.3226
2047	.HIGH	=	770967.7419
2048	.HIGH	=	780645.1613
2049	.HIGH	=	790322.5806
2050	.HIGH	=	800000
2051	.HIGH	=	800000
2052	.HIGH	=	800000
2053	.HIGH	=	800000
2054	.HIGH	=	800000
2055	.HIGH	=	800000
2056	.HIGH	=	800000
2057	.HIGH	=	800000
2058	.HIGH	=	800000
2059	.HIGH	=	800000
2060	.HIGH	=	800000
2061	.HIGH	=	800000
2062	.HIGH	=	800000
2063	.HIGH	=	800000
2064	.HIGH	=	800000
2065	.HIGH	=	800000
2066	.HIGH	=	800000
2067	.HIGH	=	800000
2068	.HIGH	=	800000
2069	.HIGH	=	800000
2070    .HIGH   =       800000
2071	.HIGH	=	800000
2072	.HIGH	=	800000
2073	.HIGH	=	800000
2074	.HIGH	=	800000
2075	.HIGH	=	800000
2076	.HIGH	=	800000
2077	.HIGH	=	800000
2078	.HIGH	=	800000
2079	.HIGH	=	800000
2080    .HIGH   =       800000
2000	.LOW	=	200000
2001	.LOW	=       200000
2002	.LOW	=       200000
2003	.LOW	=       200000
2004	.LOW	=       200000
2005	.LOW	=       200000
2006	.LOW	=       200000
2007	.LOW	=       200000
2008	.LOW	=       200000
2009	.LOW	=       200000
2010	.LOW	=       200000
2011	.LOW	=       200000
2012	.LOW	=       200000
2013	.LOW	=       200000
2014	.LOW	=       200000
2015	.LOW	=       200000
2016	.LOW	=       200000
2017	.LOW	=       200000
2018	.LOW	=       200000
2019	.LOW	=	200000
2020	.LOW	=	203225.8065
2021	.LOW	=	206451.6129
2022	.LOW	=	209677.4194
2023	.LOW	=	212903.2258
2024	.LOW	=	216129.0323
2025	.LOW	=	219354.8387
2026	.LOW	=	222580.6452
2027	.LOW	=	225806.4516
2028	.LOW	=	229032.2581
2029	.LOW	=	232258.0645
2030	.LOW	=	235483.871
2031	.LOW	=	238709.6774
2032	.LOW	=	241935.4839
2033	.LOW	=	245161.2903
2034	.LOW	=	248387.0968
2035	.LOW	=	251612.9032
2036	.LOW	=	254838.7097
2037	.LOW	=	258064.5161
2038	.LOW	=	261290.3226
2039	.LOW	=	264516.129
2040	.LOW	=	267741.9355
2041	.LOW	=	270967.7419
2042	.LOW	=	274193.5484
2043	.LOW	=	277419.3548
2044	.LOW	=	280645.1613
2045	.LOW	=	283870.9677
2046	.LOW	=	287096.7742
2047	.LOW	=	290322.5806
2048	.LOW	=	293548.3871
2049	.LOW	=	296774.1935
2050	.LOW	=	300000
2051	.LOW	=	300000
2052	.LOW	=	300000
2053	.LOW	=	300000
2054	.LOW	=	300000
2055	.LOW	=	300000
2056	.LOW	=	300000
2057	.LOW	=	300000
2058	.LOW	=	300000
2059	.LOW	=	300000
2060	.LOW	=	300000
2061	.LOW	=	300000
2062	.LOW	=	300000
2063	.LOW	=	300000
2064	.LOW	=	300000
2065	.LOW	=	300000
2066	.LOW	=	300000
2067	.LOW	=	300000
2068	.LOW	=	300000
2069	.LOW	=	300000
2070	.LOW	=	300000
2071	.LOW	=	300000
2072	.LOW	=	300000
2073	.LOW	=	300000
2074	.LOW	=	300000
2075	.LOW	=	300000
2076	.LOW	=	300000
2077	.LOW	=	300000
2078	.LOW	=	300000
2079	.LOW	=	300000
2080	.LOW	=	300000
/;

* temporary definition; to introduce for all inityears
*PARAMETER VEH_STCK_INT_SEG(seg)
*/
*        A = 0.08
*        B = 0.21
*        C = 0.26
*        D = 0.08
*        E = 0.03
*        F = 0.34
*/;
*
*
*PARAMETER MANUF_CNSTRNT(year)
*/
*2010   =37.3416608
*2011   =37.92857143
*2012   =37.92857143
*2013   =71.96938776
*2014   =71.96938776
*2015   =88.68586006
*2016   =103.8826531
*2017   =165.5816327
*2018=  292.930758
*2019=  403.8673469
*2020=  540.0306122
*2021=  735.7653061
*2022=  999.5816327
*2023=  1186.502187
*2024=  1347.892128
*2025=  1501.116327
*2026=  1671.887755
*2027=  1833.581633
*2028=  1952.72449
*2029=  2000
*2030=  2100
*2031=  2150
*2032=  2200
*2033=  2200
*2034=  2200
*2035=  2200
*2036=  2200
*2037=  2200
*2038=  2200
*2039=  2200
*2040=  2200
*2041=  2200
*2042=  2200
*2043=  2200
*2044=  2200
*2045=  2200
*2046=  2200
*2047=  2200
*2048=  2200
*2049=  2200
*2050=  2200
*/;

*PARAMETER VEH_LIFT_MOR(age)
*/
*0=0
*1=0
*2=0
*3=0
*4=0
*5=0
*6=0
*7=0
*8=0
*9=0
*10=0.5
*11=0.5
*12=0.5
*13=0.5
*14=0.5
*15=0.5
*16=0.5
*17=0.5
*18=1
*19=1
*20=1
*21=1
*22=1
*23=1
*24=1
*25=1
*26=1
*27=1
*/;
*PARAMETER VEH_LIFT_MOR(age)
*/
*0=0
*1=0.037037037
*2=0.074074074
*3=0.111111111
*4=0.148148148
*5=0.185185185
*6=0.222222222
*7=0.259259259
*8=0.296296296
*9=0.333333333
*10=0.37037037
*11=0.407407407
*12=0.444444444
*13=0.481481481
*14=0.518518519
*15=0.555555556
*16=0.592592593
*17=0.62962963
*18=0.666666667
*19=0.703703704
*20=0.740740741
*21=0.777777778
*22=0.814814815
*23=0.851851852
*24=0.888888889
*25=0.925925926
*26=0.962962963
*27=1
*/;
*
ENR_CINT(enr,reg,modelyear) =  genlogfnc(ENR_PARTAB(enr,reg,'CINT','A'),ENR_PARTAB(enr,reg,'CINT','B'),ENR_PARTAB(enr,reg,'CINT','r'),YEAR_PAR(modelyear),ENR_PARTAB(enr,reg,'CINT','u'));

*----- Production-related emissions
* Assume constant for all regions for now
VEH_PROD_EINT(tec,seg,prodyear) = genlogfnc(VEH_PARTAB('PROD_EINT',tec,seg,'A'),VEH_PARTAB('PROD_EINT',tec,seg,'B'),VEH_PARTAB('PROD_EINT',tec,seg,'r'),YEAR_PAR(prodyear),VEH_PARTAB('PROD_EINT',tec,seg,'u'));

VEH_PROD_CINT_CSNT(tec,seg,prodyear) = genlogfnc(VEH_PARTAB('PROD_CINT_CSNT',tec,seg,'A'),VEH_PARTAB('PROD_CINT_CSNT',tec,seg,'B'),VEH_PARTAB('PROD_CINT_CSNT',tec,seg,'r'),YEAR_PAR(prodyear),VEH_PARTAB('PROD_CINT_CSNT',tec,seg,'u'));

VEH_PROD_CINT(tec,seg,prodyear)$(ord(prodyear)>28) = VEH_PROD_CINT_CSNT(tec,seg,prodyear) + VEH_PROD_EINT(tec,seg,prodyear)*ENR_CINT('elc','prod',prodyear)/1000;
* only production emissions from 2000 onwards are relevant despite cohorts going back to 1972

*----- Operation phase emissions
* Assume constant for all regions for now
VEH_OPER_EINT(tec,seg,prodyear) = genlogfnc(VEH_PARTAB('OPER_EINT',tec,seg,'A'),VEH_PARTAB('OPER_EINT',tec,seg,'B'),VEH_PARTAB('OPER_EINT',tec,seg,'r'),YEAR_PAR(prodyear),VEH_PARTAB('OPER_EINT',tec,seg,'u'));

VEH_OPER_CINT(tec,enr,seg,reg,age,modelyear,prodyear)$(ENR_VEH(enr,tec)) = VEH_OPER_EINT(tec,seg,prodyear)*VEH_PAY(prodyear,age,modelyear)*(ENR_CINT(enr,reg,modelyear)/1000);
*VEH_OPER_CINT(tec,enr,seg,prodyear,modelyear)$(ENR_VEH(enr,tec) and prodyear <=modelyear) = VEH_OPER_EINT(tec,seg,prodyear)*(ENR_CINT(enr,modelyear)/1000);
*OPER_TRIAL(tec,enr,seg,'2030',age,modelyear)$(modelyear>=prodyear and modelyear<=(prodyear+11))= VEH_OPER_EINT(tec,seg,prodyear)*VEH_PAY(prodyear,age,modelyear)*(ENR_CINT(enr,modelyear)/1000)*VEH_OPER_DIST(modelyear);
*OPER_TRIAL(tec,enr,seg,'2030',age,modelyear)$(modelyear>=2030 and modelyear<=(2030+11))= VEH_OPER_CINT(tec,enr,seg,'2030',age,modelyear);
*VEH_OPER_EINT(tec,seg,'2030')*VEH_PAY('2030',age,modelyear)*(ENR_CINT(enr,modelyear)/1000)*VEH_OPER_DIST(modelyear);


*VEH_OPER_CINT2(tec,enr,seg,prodyear,age,modelyear) = VEH_OPER_EINT(tec,seg,prodyear)*(ENR_CINT(enr,modelyear)/1000)*VEH_PAY(prodyear,age,modelyear);

*----- End-of-life phase emissions
*10.10.2019:  Used to prodyear, rather than modelyear
* Assume constant for all regions for now
VEH_EOLT_CINT(tec,seg,modelyear) = genlogfnc(VEH_PARTAB('EOLT_CINT',tec,seg,'A'),VEH_PARTAB('EOLT_CINT',tec,seg,'B'),VEH_PARTAB('EOLT_CINT',tec,seg,'r'),YEAR_PAR(modelyear),VEH_PARTAB('EOLT_CINT',tec,seg,'u'));

*----- Calculate lifecycle emissions for each vehicle technology and segment for quality check
* uses average lifetime in years
*for(a=prodyear to prodyear+11 by 1, COHORT_OPER_EMISSIONS(tec,enr,seg,a+1) = COHORT_OPER_EMISSIONS(tec,enr,seg,a)+ VEH_OPER_CINT(tec,enr,seg,prodyear,age,a+1)*VEH_OPER_DIST(a+1));
*loop(prodyear, for(a=prodyear to prodyear+11 by 1, COHORT_OPER_EMISSIONS(tec,enr,seg,a+1) = COHORT_OPER_EMISSIONS(tec,enr,seg,a)+ VEH_OPER_CINT(tec,enr,seg,prodyear,age,a+1)*VEH_OPER_DIST(a+1));
*COHORT_OPER_EMISSIONS(tec,enr,seg,prodyear) = VEH_OPER_CINT(tec,enr,seg,prodyear,age,modelyear)*VEH_OPER_DIST(modelyear)
*VEH_LC_EMISS(tec,enr,seg,prodyear) = VEH_PROD_CINT(tec,seg,prodyear) + COHORT_OPER_EMISSIONS(tec,enr,seg,prodyear) + VEH_EOLT_CINT(tec,seg,prodyear+11);

*----- Initialization of vehicle stock
*These are not used elsewhere
$ontext
INIT_SEG(seg) = VEH_STCK_INT_SEG(seg)*VEH_STCK_TOT('2000');
INIT_TEC(tec) = VEH_STCK_INT_TEC(tec)*VEH_STCK_TOT('2000');
INIT_AGE(age) = VEH_LIFT_PDF(age)*VEH_STCK_TOT('2000');
SEG_TEC(seg,tec) = (VEH_STCK_INT_TEC(tec))*(VEH_STCK_INT_SEG(seg));
SEG_TEC_AGE(seg,tec,age) = (VEH_STCK_INT_TEC(tec)*VEH_STCK_INT_SEG(seg))*VEH_LIFT_PDF(age);
$offtext

VEH_STCK_INT(tec,seg,reg,age) = (VEH_STCK_INT_TEC(tec)*VEH_LIFT_PDF(age)*VEH_STCK_INT_SEG(seg))*VEH_STCK_TOT('2000',reg);


*-----------------------------------------------------------------------------------
*
* Model Definition  p.t 1 : Variable Definitions
*
*-----------------------------------------------------------------------------------


***FREE VARIABLES ------------------------------------------------------------------
* Objective value to be minimized must be a free variable


FREE VARIABLES
TOTC                                    Total CO2 emissions for the whole system over the whole period
TOTC_OPT                                Total CO2 emissions for the whole system over optimization period
VEH_STCK_DELTA(year,reg)                    Delta stock from one year to the next
*slack_VEH_ADD(seg,year,age)
;

POSITIVE VARIABLES
VEH_STCK(tec,seg,reg,year,age)              Number of vehicles of a given age in a given year
VEH_STCK_REM(tec,seg,reg,year,age)          Number of vehicles of a given age retired in a given year
VEH_STCK_TOT_CHECK(year)                Check on number of vehicles
VEH_STCK_ADD(tec,seg,reg,year,age)          Stock additions (new car sales)
VEH_TOT_ADD(year)                       Total vehicles added
VEH_TOTC(tec,seg,reg,year)                  Total CO2 emissions of vehicles per year, by technology             [t CO2-eq]
VEH_STCK_ADD_OPTYEAR1(tec,seg,reg,year,age)

VEH_PROD_TOTC(tec,seg,reg,year)             Total CO2 emissions from production of vehicles per year            [t CO2-eq]
VEH_OPER_TOTC(tec,seg,reg,year)             Total CO2 emissions from operations of vehicles per year            [t CO2-eq]
VEH_EOLT_TOTC(tec,seg,reg,year)             Total CO2 emissions from vehicle end of life treatment per year     [t CO2-eq]
check_add_tot(year)
VEH_TOT_REM(year)
ANN_TOTC(year)                          Total CO2 emissions from LDVs, by year                              [t CO2-eq]
VEH_STCK_CHRT(tec,seg,prodyear,age,modelyear)
VEH_OPER_COHORT(tec,seg,prodyear,modelyear,agej)
*OPER(tec,seg,year,prodyear)
*VEH_LC_EMISS(tec,seg,prodyear)

*TOTAL_NEW_CAP(year)                     Total battery capacity added to fleet                               [MWh]


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
$ontext
EQ_STCK_ADD_T11
$offtext

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
*EQ_STCK_BAL1
*EQ_STCK_BAL2
EQ_STCK_BAL

*summing the number of vehicles added and in stock for check.
*EQ_TOT_ADD
*EQ_TOT_REM
EQ_STCK_CHK

*** Constraint equations
* For technology-related additions
EQ_STCK_ADD0
EQ_STCK_GRD0
EQ_STCK_GRD

* Keeping segment shares constant
EQ_SEG_GRD

* Manufacturing capacity constraint
EQ_NEW_BATT_CAP
*EQ_ADD_CAP
*EQ_BATT_MANU_CONSTRNT

**EMISSION and ENERGY MODELS incl OBJ. FUNCTION --------------------------------------

* Objective function
EQ_TOTC_OPT
* Calculation of emissions from all vehicle classes per year
EQ_VEH_TOTC
*producton emissions
EQ_VEH_PROD_TOTC
*operation emissions
EQ_VEH_OPER_TOTC
*eolt emissions
EQ_VEH_EOLT_TOTC
EQ_TOTC
EQ_ANN_TOTC
*EQ_CHECK_ADD
*EQ_OPER
$ontext
EQ_STCK_COHORT
EQ_VEH_OPER_COHORT
$offtext

*EQ_VEH_LC_EMISSIONS
;

*-----------------------------------------------------------------------------------
*
* Model Definition  p.t 3 : Model Equations

*-----------------------------------------------------------------------------------


***VEHICLE STOCK MODEL  ------------------------------------------------------------

***  Initiate stock in first year ----------------------------

* calculate stock removals as per survival curves
EQ_STCK_REM_T1(tec,seg,reg,inityear,age)..                VEH_STCK_REM(tec,seg,reg,inityear,age) =e=  VEH_STCK(tec,seg,reg,inityear-1,age-1)  * VEH_LIFT_MOR(age-1);
* VEH_STCK_INT_SEG(seg)* VEH_LIFT_PDF(age)  * VEH_STCK_INT_TEC(tec)

* calculate stock additions as sum of stock removals from the same year and increase in total fleet (net stock growth)
*-- Makes assumption that retired segments are replaced by the same segment
*EQ_STCK_ADD_T1(seg,inityear,age)$(ord(age)=1)..         sum(tec, VEH_STCK_ADD(tec,seg,inityear,age)) =e=  sum((tec,agej), VEH_STCK_REM(tec,seg,inityear,agej));
EQ_STCK_ADD_T1(seg,reg,inityear,'0')..                    sum(tec, VEH_STCK_ADD(tec,seg,reg,inityear,'0')) =e=  sum((tec,agej), VEH_STCK_REM(tec,seg,reg,inityear,agej))+ (VEH_STCK_DELTA(inityear,reg)*VEH_STCK_INT_SEG(seg));

$ontext
EQ_STCK_ADD_T11('2000')..                             VEH_STCK_DELTA('2000') =e= 3637119.13333321;
$offtext

*EQ_STCK_ADD_T11(seg,'2000',age)$(ord(age)=1)..              sum(tec,VEH_STCK_ADD(tec,seg,'2000',age)) =e= sum((tec,agej),VEH_STCK_REM(tec,seg,'2000',agej))+3637119.13333321;

* calculate total stock per cohort, segement and technology
EQ_STCK_BAL_T1(tec,seg,reg,inityear,age)..                VEH_STCK(tec,seg,reg,inityear,age) =e=  VEH_STCK_TOT(inityear,reg) * (VEH_STCK_INT_TEC(tec) * VEH_LIFT_PDF(age) * VEH_STCK_INT_SEG(seg));

* calculate total increase (or change) of total fleet (net stock growth)
EQ_STCK_DELTA_T1(inityear,reg)$(ord(inityear)>1)..        VEH_STCK_DELTA(inityear,reg)  =e=  VEH_STCK_TOT(inityear,reg) - VEH_STCK_TOT(inityear-1,reg);


***  Main Model -----------------------------------------------

* calculate stock removals as per survival curves
* assumes equal removal across technologies and segments
*EQ_STCK_REM(tec,seg,optyear,age)$(ord(optyear)>1)..    VEH_STCK_REM(tec,seg,optyear,age) =e= VEH_STCK(tec,seg,optyear,age)*VEH_LIFT_MOR(age);
EQ_STCK_REM(tec,seg,reg,optyear,age)$(ord(optyear)>1)..    VEH_STCK_REM(tec,seg,reg,optyear,age) =e= VEH_STCK(tec,seg,reg,optyear-1,age-1)*VEH_LIFT_MOR(age-1);

* calculate stock additions as sum of stock removals from the same year and increase in total fleet (net stock growth)
* assumes removed vehicles are replaced with like segment and technology
*EQ_STCK_ADD(optyear,age)$(ord(optyear)>1 and ord(age)=1)..             sum((tec,seg), VEH_STCK_ADD(tec,seg,optyear,age) - sum(agej, VEH_STCK_REM(tec,seg,optyear,agej))) =e= VEH_STCK_DELTA(optyear);
EQ_STCK_ADD(optyear,age,reg)$(ord(optyear)>1)..             sum((tec,seg), (VEH_STCK_ADD(tec,seg,reg,optyear,'0') - sum(agej, VEH_STCK_REM(tec,seg,reg,optyear,agej)))) =e= VEH_STCK_DELTA(optyear,reg);
*EQ_STCK_ADD(seg,optyear,age)$(ord(optyear)>1)..             sum((tec), VEH_STCK_ADD(tec,seg,optyear,'0') - sum(agej, VEH_STCK_REM(tec,seg,optyear,agej))) =e= VEH_STCK_DELTA(optyear)*VEH_STCK_INT_SEG(seg);

* calculate total increase (or change) of total fleet (net stock growth)
EQ_STCK_DELTA(optyear,reg)$(ord(optyear)>1)..             VEH_STCK_DELTA(optyear,reg)  =e=  VEH_STCK_TOT(optyear,reg) - VEH_STCK_TOT(optyear-1,reg);

*EQ_STCK_BAL(tec,seg,optyear,age)..                     VEH_STCK(tec,seg,optyear,age) =e=  VEH_STCK(tec,seg,optyear-1,age-1) * (VEH_STCK_INT_TEC(tec) * VEH_LIFT_PDF(age) * VEH_STCK_INT_SEG(seg));

***---***---***---***---***---***---***---***---***---***
*EQ_STCK_ADD(optyear)..                                              VEH_TOT_ADD(optyear) - sum((tec,seg,agej), VEH_STCK_REM(tec,seg,optyear,agej)) =e= VEH_STCK_DELTA(optyear);

** make into veh_add_tot?

*EQ_STCK_ADD(optyear,age)$(ord(optyear)>1 and ord(age)=1)..            sum(seg, sum(tec, VEH_STCK_ADD(tec,seg,optyear,age)) - sum((agej,tec), VEH_STCK_REM(tec,seg,optyear,agej))) =e= VEH_STCK_DELTA(optyear);
*EQ_STCK_ADD(optyear,age)$(ord(optyear)>1 and ord(age)=1)..             sum((tec,seg), VEH_STCK_ADD(tec,seg,optyear,age)) =e= VEH_STCK_DELTA(optyear) + sum((tec,seg,agej), VEH_STCK_REM(tec,seg,optyear,agej));
*EQ_STCK_ADD(seg,optyear,age)$(ord(optyear)>1 and ord(age)=1)..             sum((tec), VEH_STCK_ADD(tec,seg,optyear,age)) =e= VEH_STCK_DELTA(optyear) + sum((tec,agej), VEH_STCK_REM(tec,seg,optyear,agej));

*calculating vehicle stock in a given year
*EQ_STCK_BAL1(tec,seg,optyear)$(ord(optyear)>1)..      VEH_STCK(tec,seg,optyear,'0')  =e=  VEH_STCK_ADD(tec,seg,optyear,'0');

*EQ_STCK_BAL2(tec,seg,optyear,age)$(ord(optyear)>1 and ord(age)>1)..      VEH_STCK(tec,seg,optyear,age)  =e=  VEH_STCK(tec,seg,optyear-1,age-1) - VEH_STCK_REM(tec,seg,optyear,age);

EQ_STCK_BAL(tec,seg,reg,optyear,age)$(ord(optyear)>1)..      VEH_STCK(tec,seg,reg,optyear,age)  =e=  VEH_STCK(tec,seg,reg,optyear-1,age-1) + VEH_STCK_ADD(tec,seg,reg,optyear,age) - VEH_STCK_REM(tec,seg,reg,optyear,age);


*-----calculate segment market shares-----
*** What works:
* 1. original EQ_STCK_ADD, EQ_SEG_GRD with VEH_TOT_ADD (with EQ_TOT_ADD commented out)
* 2. original EQ_STCK_ADD, EQ_TOT_ADD and EQ_SEG_GRD with =l= and VEH_STCK_ADD

* Calculate total addition to stock independent of technology and segment,
*EQ_TOT_ADD(year)..                                    VEH_TOT_ADD(year) =e= sum((tec,seg,age), VEH_STCK_ADD(tec,seg,year,age));
*EQ_TOT_REM(year)..                                                      VEH_TOT_REM(year) =e= sum((tec,seg,age),VEH_STCK_REM(tec,seg,year,age));
* summing the number of vehicles in fleet as check.
EQ_STCK_CHK(modelyear)..                                                    VEH_STCK_TOT_CHECK(modelyear) =e= sum((tec,seg,reg,age), VEH_STCK(tec,seg,reg,modelyear,age));

*** Constraints -----------------------------------------------------------------------

* stock additions by technology; consumer uptake constraint
* sets uptake of BEVs in 2019 to 0
EQ_STCK_ADD0(tec,seg,reg,'2019','0')..                                 VEH_STCK_ADD_OPTYEAR1('BEV',seg,reg,'2019','0') =e= 0;

* restrict addition to stock based on previous year's uptake
EQ_STCK_GRD0(tec,seg,reg,'2020','0')..                                 VEH_STCK_ADD(tec,seg,reg,'2020','0') =l= ((1 + VEH_ADD_GRD('IND',tec)) * VEH_STCK_ADD_OPTYEAR1(tec,seg,reg,'2019','0')) + 5e4;

*EQ_STCK_GRD(tec,seg,optyear,age)$(ord(optyear)>1 and ord(age)=1)..     VEH_STCK_ADD(tec,seg,optyear,age) =l= ((1 + VEH_ADD_GRD('IND',tec)) * VEH_STCK_ADD(tec,seg,optyear-1,age)) + 5e4;
EQ_STCK_GRD(tec,seg,reg,optyear)$(ord(optyear)>1)..     VEH_STCK_ADD(tec,seg,reg,optyear,'0') =l= ((1 + VEH_ADD_GRD('IND',tec)) * VEH_STCK_ADD(tec,seg,reg,optyear-1,'0')) + 5e4;

* Segment share constraint (segments kept constant)
* try to remove - allow for ICEV smart cars (e.g.,) to be replaced by a BEV Model X...
EQ_SEG_GRD(seg,reg,optyear,'0')$(ord(optyear)>1)..          sum(tec,VEH_STCK_ADD(tec,seg,reg,optyear,'0')) =l= VEH_STCK_INT_SEG(seg) * sum((tec,segj), VEH_STCK_ADD(tec,segj,reg,optyear,'0'));


*------ Battery manufacturing constraint;
* calculate total new capacity of batteries in BEVs added to stock each year
*EQ_NEW_BATT_CAP(optyear)..                              TOTAL_NEW_CAP(optyear) =e= sum((seg),VEH_STCK_ADD('BEV',seg,optyear,'0')*BEV_CAPAC(seg))/1000;
*
*EQ_ADD_CAP(optyear)..                                   TOTAL_NEW_CAP(optyear) =l= MANUF_CNSTRNT(optyear)*1000;

* total capacity added per year must be less than the battery manufacturing capacity
EQ_NEW_BATT_CAP(optyear)..                              MANUF_CNSTRNT(optyear)*1000 =g= sum((seg,reg),VEH_STCK_ADD('BEV',seg,reg,optyear,'0')*BEV_CAPAC(seg))/1000;

*EQ_BATT_MANU_CONSTRNT(optyear)..                    MANUF_CNSTRNT(optyear)*1000 =l= sum((tec,seg,age),VEH_STCK_ADD('BEV',seg,optyear,'0')*BEV_CAPAC(seg))/1000;


* lithium refining constraint;


$ontext
* Segment share constraint (keep segment shares constant over analysis period)
** This works and I have no idea why. (EQ_TOT_ADD must be removed for this to work)
EQ_SEG_GRD(seg,optyear,age)$(ord(optyear)>1 and ord(age)=1)..          sum(tec,VEH_STCK_ADD(tec,seg,optyear,age)) =e= VEH_STCK_INT_SEG(seg)* VEH_TOT_ADD(optyear,age);

EQ_CHECK_ADD(year)..                                                    check_add_tot(year) =e= sum((tec,seg,age),VEH_STCK_ADD(tec,seg,year,age));

*** This causes infeasibility in the solution (but otherwise works, i.e., keeps segment shares constant).
EQ_SEG_GRD(seg,optyear,age)$(ord(optyear)>1 and ord(age)=1)..          sum(tec,VEH_STCK_ADD(tec,seg,optyear,age)) + slack_VEH_ADD(seg,optyear,age) =e= VEH_STCK_INT_SEG(seg) * sum((tec,segj), VEH_STCK_ADD(tec,segj,optyear,age));

EQ_SEG_GRD(seg,optyear,age)$(ord(optyear)>1 and ord(age)=1)..          sum(tec,VEH_STCK_ADD(tec,seg,optyear,age)) =e= VEH_SEG_SHR(seg) * sum((tec,segj), VEH_STCK_ADD(tec,segj,optyear,age));

EQ_SEG_GRD(seg,optyear)$(ord(optyear)>1)..                          sum((tec,age), VEH_STCK(tec,seg,optyear,age)) =e= VEH_STCK_INT_SEG(seg)*sum((tec,segj,age), VEH_STCK(tec,segj,optyear,age));

EQ_SEG_GRD(seg,optyear,age)$(ord(optyear)>1 and ord(age)=1)..          sum(tec,VEH_STCK_ADD(tec,seg,optyear,age)) =l= 1.2*sum((tec,segj),VEH_STCK_ADD(tec,segj,optyear-1,age));
$offtext
;



*** EMISSION and ENERGY MODELS incl OBJ. FUNCTION ------------------------------------
* Objective function
EQ_TOTC_OPT..                                TOTC_OPT =e= sum((tec,seg,reg,optyear), VEH_TOTC(tec,seg,reg,optyear));

* Calculation of emissions from all vehicle classes per year
EQ_VEH_TOTC(tec,seg,reg,modelyear)..                  VEH_TOTC(tec,seg,reg,modelyear) =e= VEH_PROD_TOTC(tec,seg,reg,modelyear) + VEH_OPER_TOTC(tec,seg,reg,modelyear) + VEH_EOLT_TOTC(tec,seg,reg,modelyear);
EQ_ANN_TOTC(modelyear)..                          ANN_TOTC(modelyear) =e= sum((tec,seg,reg),VEH_TOTC(tec,seg,reg,modelyear));
*int_tec

*** to do: change production emissions to scale across lifetime
*EQ_VEH_PROD_TOTC(tec,seg,modelyear)..             VEH_PROD_TOTC(tec,seg,modelyear) =e= sum( (agej)$(ord(agej)=1), VEH_STCK_ADD(tec,seg,modelyear,agej)*VEH_PROD_CINT(tec,seg,modelyear));
*EQ_VEH_PROD_TOTC(tec,seg,modelyear)..             VEH_PROD_TOTC(tec,seg,modelyear) =e= sum( (agej), VEH_STCK_ADD(tec,seg,modelyear,'0')*VEH_PROD_CINT(tec,seg,modelyear));

EQ_VEH_PROD_TOTC(tec,seg,reg,modelyear)..             VEH_PROD_TOTC(tec,seg,reg,modelyear) =e= VEH_STCK_ADD(tec,seg,reg,modelyear,'0')*VEH_PROD_CINT(tec,seg,modelyear);

*EQ_VEH_OPER_TOTC(tec,seg,modelyear)..             VEH_OPER_TOTC(tec,seg,modelyear) =e= sum( (agej,enr,prodyear), VEH_STCK(tec,seg,modelyear,agej) * VEH_OPER_CINT(tec,enr,seg,prodyear,modelyear) * ENR_VEH(enr,tec)*VEH_PAY(prodyear,agej,modelyear) * VEH_OPER_DIST(modelyear));
EQ_VEH_OPER_TOTC(tec,seg,reg,modelyear)..             VEH_OPER_TOTC(tec,seg,reg,modelyear) =e= sum( (agej,enr,prodyear), VEH_STCK(tec,seg,reg,modelyear,agej) *VEH_PAY(prodyear,agej,modelyear)* VEH_OPER_CINT(tec,enr,seg,reg,agej,modelyear,prodyear) *  VEH_OPER_DIST(modelyear));

* Init phase operation emissions are 0 because we don't account for non-new cars in 2000! (i.e., prodyear is >=2000)
*EQ_OPER(tec,seg,year,prodyear)..                      OPER(tec,seg,year,prodyear) =e= sum((agej,enr),VEH_STCK(tec,seg,year,agej) * VEH_OPER_CINT(tec,enr,seg,prodyear) * ENR_VEH(enr,tec)*VEH_PAY(prodyear,agej,year) * VEH_OPER_DIST(year));
EQ_VEH_EOLT_TOTC(tec,seg,reg,modelyear)..             VEH_EOLT_TOTC(tec,seg,reg,modelyear) =e= sum( (agej), VEH_STCK_REM(tec,seg,reg,modelyear,agej))*VEH_EOLT_CINT(tec,seg,modelyear);

EQ_TOTC..                                    TOTC =e= SUM((tec,seg,reg,modelyear), VEH_TOTC(tec,seg,reg,modelyear));


*** Convert VEH_STCK to include cohort for clearer figures ------------------------------------------
** move to post-processing calculations
$ontext
* 07.05.2020
EQ_STCK_COHORT(tec,seg,prodyear,agej,modelyear)$VEH_PAY(prodyear,agej,modelyear)..    VEH_STCK_CHRT(tec,seg,prodyear,agej,modelyear) =e= (VEH_STCK(tec,seg,modelyear,agej)*VEH_PAY(prodyear,agej,modelyear)) ;
$offtext


$ontext
** move to post-processing calculations
EQ_VEH_LC_EMISSIONS(tec,seg,prodyear)..     VEH_LC_EMISS(tec,seg,prodyear) =e= VEH_PROD_CINT(tec,seg,prodyear) +  sum( (agej,enr,modelyear), VEH_STCK(tec,seg,modelyear,agej) *VEH_PAY(prodyear,agej,modelyear)* VEH_OPER_CINT(tec,enr,seg,prodyear,agej,modelyear) *  VEH_OPER_DIST(modelyear));
$offtext

** move to post-processing calculations
$ONTEXT
* 07.05.2020
EQ_VEH_OPER_COHORT(tec,seg,prodyear,modelyear,agej)$VEH_PAY(prodyear,agej,modelyear)..   VEH_OPER_COHORT(tec,seg,prodyear,modelyear,agej) =e= sum((enr), VEH_STCK(tec,seg,reg,modelyear,agej) *VEH_PAY(prodyear,agej,modelyear)* VEH_OPER_CINT(tec,enr,seg,reg,agej,modelyear,prodyear) *  VEH_OPER_DIST(modelyear));
$OFFTEXT
;
* move the below to post-processing
*EQ_VEH_LC_EMISSIONS(tec,seg,prodyear)..     VEH_LC_EMISS(tec,seg,prodyear) =e= VEH_PROD_CINT(tec,seg,prodyear) + sum((modelyear,agej), VEH_OPER_COHORT(tec,seg,prodyear,modelyear,agej)/VEH_STCK(tec,seg,modelyear,agej)*VEH_PAY(prodyear,agej,modelyear));

*-----------------------------------------------------------------------------------
*
* Model Execution  p.t 1 : Model Definition and Options
*
*-----------------------------------------------------------------------------------

* Defining name of model(s) and what equations are used in each model

MODEL EVD4EUR_Basic "default model, run in normal mode" /ALL/
      seg_test "model without segment constraint" /EVD4EUR_Basic - EQ_SEG_GRD/
      unit_test "model without growth constraint" /EVD4EUR_Basic - EQ_STCK_GRD/
      unit_test2 "model without growth or manufacturing constraint" /unit_test - EQ_NEW_BATT_CAP/
      neq_stcko_contr "model no constraints at all" /unit_test2 - EQ_STCK_ADD0 - EQ_STCK_GRD0/
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
SOLVE EVD4EUR_Basic USING LP MINIMIZING TOTC_OPT;
*SOLVE unit_test USING LP MINIMIZING TOTC_OPT;
*SOLVE unit_test2 USING LP MINIMIZING TOTC_OPT;
*SOLVE no_contr USING LP MINIMIZING TOTC_OPT;

*DISPLAY VEH_STCK_TOT
*DISPLAY VEH_STCK_TOT_CHECK.l


execute_unload 'EVD4EUR_addset.gdx'
