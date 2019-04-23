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
* ver 0.98       Added gradient of change dampening constraint
********************************************************************************
********************************************************************************
********************************************************************************

SETS
year           year /2000*2050/
optyear(year)  years for optiomization /2020*2050/
inityear(year) years for initialization /2000*2020/
age            age /0*20/
tec            techlogy /ICE,BEV/
enr            energy /ELC,FOS/
veheq          equations for vehicle parameters /PROD_EINT, PROD_CINT_CSNT, OPER_EINT, EOLT_CINT/
flteq          equations for fleet parameters /CSTCK_TOT, OPER_DIST/
enreq          equations for energy (electricity and fuels) system /CINT/
sigvar         variables for sigmoid equations /A,B,r,t,u/

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
YEAR_PAR(year)                   year
PRODYEAR_PAR(prodyear)           production year
VEH_PARTAB(tec,veheq,sigvar)     variables for each tech and veh equation
FLT_PARTAB(tec,flteq,sigvar)     variables for each tech and flt equation
ENR_PARTAB(enr,enreq,sigvar)     variables for each energy (fos or elc) equation

***ENERGY (ELECTRICITY GENERATION and FUEL) ************************************
ENR_CINT(enr,year)               CO2 intensity of the energy  [kg CO2-eq pr kwh]

***ENERGY and VEHICLE TECHONLOGY COMBINATIONS **********************************
ENR_VEH(enr,tec)                 feasbile cominations of vehivle technology and energy (fuel).


***All VEHICLES*****************************************************************

**PRODUCTION
VEH_PROD_EINT(tec,prodyear)        Electricity intensity of vehicle prod [kwh el required per vehicle produced]
VEH_PROD_CINT_CSNT(tec,prodyear)   Constant term for CO2 int. of vehicle production [kg CO2-eq per vehicle produced]
VEH_PROD_CINT(tec,prodyear)        CO2 intensity of ICE vehicle production [kg CO2-eq per vehicle produced]

**OPERATION
VEH_OPER_EINT(tec,prodyear)        Energy intensity of vehicle operation [kwh per km]
VEH_OPER_CINT(tec,enr,prodyear)    CO2 intensity of vehicle operation    [kg CO2 per km]

**EOL
VEH_EOLT_CINT(tec,year)            CO2 intensity of ICE vehicle EOL - [kg CO2-eq per vehicle in EOL treatment]



***VEHICLE FLEET****************************************************************

VEH_STCK_TOT(year)               Number of vehicles - #
VEH_OPER_DIST(year)              Annual driving distance per vehicles - km
VEH_LIFT_DSTR(age)               Share of scrapping for a given age - e.g % in given age class i scrapped
VEH_PAY(prodyear,age,year)       Defining the correspondance between a vehicle prododuction year and its age (up to 20) in a given year
VEH_INIT_AGE_DIST(age)           Initial age distribution of vehicle fleet

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
         FOS    .CINT    .B  = 0
         FOS    .CINT    .r  = 0
         FOS    .CINT    .u  = 0
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

VEH_OPER_EINT(tec,prodyear) = genlogfnc(VEH_PARTAB(tec,'OPER_EINT','A'),VEH_PARTAB(tec,'OPER_EINT','B'),VEH_PARTAB(tec,'OPER_EINT','r'),YEAR_PAR(prodyear),VEH_PARTAB(tec,'OPER_EINT','u'));

VEH_OPER_CINT(tec,enr,prodyear)$(ENR_VEH(enr,tec)) = VEH_OPER_EINT(tec,prodyear)*ENR_CINT(enr,prodyear);

$STOP



***VEHICLE FLEET****************************************************************


* Size of a national EU27 vehicle fleet
* General logistic function
* https://en.wikipedia.org/wiki/Generalised_logistic_function
* y = A + (B-A)/(1+exp(-r*(t-tau)));
* Y  is fleet size
* t  is time.
* A = Initial stock asymptote = 20 000 000
* B = End stock               = 40 000 000
* r  is the the rate of change = 0.2 ;
* tau is time the of maximum gradient of = 2015
VEH_STCK_TOT(year) = 1e6*(20 + (40-20)/(1+exp(-0.1*(YEAR_PAR(year)-2025))));




* Annual driving distance per vehicles - km
* General logistic function
* https://en.wikipedia.org/wiki/Generalised_logistic_function
* y = A + (B-A)/(1+exp(-r*(t-tau)));
* Y  drivint distance
* t  is time.
* A = Initial asymptote annual driving distance [km] = 12 000
* B = End asymptote annual driving distance [km] = 10 000
* r  is the the rate of change = 0.2 ;
* tau is time the of maximum gradient of = 2025
VEH_OPER_DIST(year) = (15000 + (10000-15000)/(1+exp(-0.1*(YEAR_PAR(year)-2025))));


* Share of scrapping in a given year for a given vintage
* Scrapping age just set to 11 yrs
* VEH_LIFT_DSTR(age)
* loop( (prodyear,year)$((ord(year) = ord(prodyear)+11)), VEH_LIFT_DSTR(prodyear,year) = 1);
PARAMETERS
VEH_LIFT_DSTR(age)  /0  0.01,  1  0.01,  2  0.02, 3  0.03, 4  0.04, 5   0.1, 6   0.3, 7   0.4, 8   0.5, 9   0.5, 10  0.5
                     11  0.5,  12 0.5,  13 0.5, 14 0.5, 15 0.5, 16  0.5, 17  0.7, 18  0.7, 19  1.00, 20  1.00/;



* Defining the correspondance between a vehicle prododuction year and its age (up to 20) in a given year
* VEH_PAY(prodyear,age,year)
loop( (prodyear,age,year)$( ord(year)= ord(prodyear)+ord(age)), VEH_PAY(prodyear,age,year) = 1);

PARAMETERS
VEH_INIT_AGE_DIST(age)  /0  0.05,  1  0.05,  2  0.05, 3  0.05, 4  0.05, 5  0.05, 6  0.05, 7  0.05, 8  0.05, 9  0.05, 10  0.05
                         11 0.05,  12 0.05,  13 0.05, 14 0.05, 15 0.05, 16 0.05, 17 0.05, 18 0.05, 19 0.05, 20 0.00/;

