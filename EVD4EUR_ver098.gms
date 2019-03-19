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
optyear(year)  years for optimization /2020*2050/
inityear(year) years for initialization /2000*2020/
age            age /0*20/


* alias call for prodyear = produyction year is identical set to year
alias (year, prodyear)
alias (prodyear, year)
alias (age, agej)
alias (agej, age)

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


***ELECTRICITY GENERATION*******************************************************
ELC_CINT(year)                   CO2 intensity of the electricity mix [kg CO2-eq pr kwh]

***ICE VEHICLES*****************************************************************

**PRODUCTION
ICE_PROD_EINT(prodyear)          Electricity intensity of ICE vehicle prod [kwh el required per vehicle produced]
ICE_PROD_CINT_CSNT(prodyear)     Constant term for CO2 int. of ICE vehicle production [kg CO2-eq per vehicle produced]
ICE_PROD_CINT(prodyear)          CO2 intensity of ICE vehicle production [kg CO2-eq per vehicle produced]

**OPERATION
**CH: Maybe split into fuel consumption and WTT intensity of fuel...
ICE_OPER_CINT(prodyear)          CO2 intensity of ICE vehicle operation (inc. fuel prod) [kg CO2 per km]

**EOL
ICE_EOLT_CINT(year)              CO2 intensity of ICE vehicle EOL - [kg CO2-eq per vehicle in EOL treatment]


***BEV VEHICLES*****************************************************************

**PRODUCTION
BEV_PROD_EINT(prodyear)          Electricity intensity of BEV vehicle prod [kwh el required per vehicle produced]
BEV_PROD_CINT_CSNT(prodyear)     Constant term for CO2 int. of BEV vehicle production [kg CO2-eq per vehicle produced]
BEV_PROD_CINT(prodyear)          CO2 intensity of BEV vehicle production - [kg CO2 per vehicle produced]

**OPERATION
BEV_OPER_EINT(prodyear)          Electricity intensity of BEV vehicle operation [kwh el per km]
BEV_OPER_CINT(prodyear)          CO2 intensity of BEV vehicle operation  [kg CO2 per km]

**EOL
BEV_EOLT_CINT(year)              CO2 intensity of BEV vehicle EOL [kg CO2-eq per vehicle in EOL treatment]


***VEHICLE FLEET****************************************************************

VEH_STCK_TOT(year)               Number of vehicles - #
VEH_OPER_DIST(year)              Annual driving distance per vehicles - km
VEH_LIFT_DSTR(age)               Share of scrapping for a given age - e.g % in given age class i scrapped
VEH_PAY(prodyear,age,year)       Defining the correspondance between a vehicle prododuction year and its age (up to 20) in a given year
VEH_INIT_AGE_DIST(age)           Initial age distribution of vehicle fleet

***STOCK INITIALISATION************************************************************


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


***ELECTRICITY GENERATION*******************************************************

*  CO2 intensity of the electricity mix [kg CO2-eq pr kwh]
* General logistic function
* https://en.wikipedia.org/wiki/Generalised_logistic_function
* y = A + (B-A)/(1+exp(-r*(t-tau)));
* Y  is the resulting CO2 eq. intensity of the electricity mix
* t  is time.
* A = Initial CO2 intensity of el mix = 1.3kg CO2/kwh el
* B = end CO2 intensity of el mix = 0.01kg CO2/kwh el
* r  is the the rate of change = 0.2 ;
* tau is time the of maximum gradient of = 2025
ELC_CINT(year) = 1.3 + (0.1-1.3)/(1+exp(-0.2*(YEAR_PAR(year)-2035)));

**CH: is a logistic function the best suited curve for this??
**perhaps solve for r by making assumptions for the beginning/end years for the asymptotes...


***ICE VEHICLES*****************************************************************

**PRODUCTION

* Electricity intensity of ICE vehicle prod [kwh el required per vehicle produced]
* General logistic function
* https://en.wikipedia.org/wiki/Generalised_logistic_function
* y = A + (B-A)/(1+exp(-r*(t-tau)));
* Y  is the electricity intensity of ICE Vehicle production
* t  is time.
* A = Initial electricity intensity of ICE Vehicle prodution = 2500 kwh
* B = End electricity intensity of ICE Vehicle prodution = 2000 kwh
* r  is the the rate of change = 0.2 ;
* tau is time the of maximum gradient of = 2025
ICE_PROD_EINT(prodyear) = 2500 + (2000-2500)/(1+exp(-0.2*(YEAR_PAR(prodyear)-2025)));


* Constant term for CO2 int. of ICE vehicle production [kg CO2-eq per vehicle produced]
* General logistic function
* https://en.wikipedia.org/wiki/Generalised_logistic_function
* y = A + (B-A)/(1+exp(-r*(t-tau)));
* Y  is the constant term for carbon intensity of ICE Vehicle production
* t  is time.
* A = Initial constant term value for carbon intensity of ICE Vehicle prodution = 3000 kg CO2 eq.
* B = End constant term value for carbon intensity of ICE Vehicle prodution = 1500 kg CO2 eq.
* r  is the the rate of change = 0.2 ;
* tau is time the of maximum gradient of = 2025
ICE_PROD_CINT_CSNT(prodyear) = 3000 + (1500-3000)/(1+exp(-0.2*(YEAR_PAR(prodyear)-2025)));


* CO2 intensity of ICE vehicle production [kg CO2-eq per vehicle produced]
ICE_PROD_CINT(prodyear) = ICE_PROD_CINT_CSNT(prodyear)+ ICE_PROD_EINT(prodyear)*ELC_CINT(prodyear);


**OPERATION

* CO2 intensity of ICE vehicle operation (inc. fuel prod) [kg CO2 per km]
* General logistic function
* https://en.wikipedia.org/wiki/Generalised_logistic_function
* y = A + (B-A)/(1+exp(-r*(t-tau)));
* Y  is the CO2 eq. intensity of ICE Vehicle production
* t  is time.
* A = Initial CO2 intensity of ICE operation  = 0.19 kgCO2 e.g pr/km
* B = End electricity intensity of ICE Vehicle  = 0.12 kgCO2 e.g pr/km
* r  is the the rate of change = 0.15 ;
* tau is time the of maximum gradient of = 2025
ICE_OPER_CINT(prodyear) = 0.19 + (0.12-0.19)/(1+exp(-0.15*(YEAR_PAR(prodyear)-2025)));

**EOL

* CO2 intensity of ICE vehicle EOL - [kg CO2-eq per vehicle in EOL treatment]
* General logistic function
* https://en.wikipedia.org/wiki/Generalised_logistic_function
* y = A + (B-A)/(1+exp(-r*(t-tau)));
* Y  is the CO2 eq. CO2 intensity of ICE vehicle EOL
* t  is time.
* A = Initial CO2 intensity of ICE EOL = 75 kgCO2 e.g pr/km
* B = End electricity intensity of ICE EOL = 30 kgCO2 e.g pr/km
* r  is the the rate of change = 0.15 ;
* tau is time the of maximum gradient of = 2025
ICE_EOLT_CINT(prodyear) = 75 + (30-75)/(1+exp(-0.15*(YEAR_PAR(prodyear)-2025)));


***BEV VEHICLES*****************************************************************


**PRODUCTION

* Electricity intensity of BEV vehicle prod [kwh el required per vehicle produced]
* General logistic function
* https://en.wikipedia.org/wiki/Generalised_logistic_function
* y = A + (B-A)/(1+exp(-r*(t-tau)));
* Y  is the electricity intensity of BEV Vehicle production
* t  is time.
* A = Initial electricity intensity of BEV Vehicle prodution = 6500 kwh
* B = End electricity intensity of BEV Vehicle prodution = 2500 kwh
* r  is the the rate of change = 0.2 ;
* tau is time the of maximum gradient of = 2025
BEV_PROD_EINT(prodyear) = 6500 + (2500-6500)/(1+exp(-0.25*(YEAR_PAR(prodyear)-2025)));

* Constant term for CO2 int. of BEV vehicle production [kg CO2-eq per vehicle produced]
* General logistic function
* https://en.wikipedia.org/wiki/Generalised_logistic_function
* y = A + (B-A)/(1+exp(-r*(t-tau)));
* Y  is the constant term for carbon intensity of BEV Vehicle production
* t  is time.
* A = Initial constant term value for carbon intensity of BEV Vehicle prodution = 5000 kg CO2 eq.
* B = End constant term value for carbon intensity of BEV Vehicle prodution = 1500 kg CO2 eq.
* r  is the the rate of change = 0.2 ;
* tau is time the of maximum gradient of = 2025
BEV_PROD_CINT_CSNT(prodyear) = 5000 + (1500-5000)/(1+exp(-0.2*(YEAR_PAR(prodyear)-2025)));

* CO2 intensity of BEV vehicle production [kg CO2-eq per vehicle produced]
BEV_PROD_CINT(prodyear) = BEV_PROD_CINT_CSNT(prodyear)+ BEV_PROD_EINT(prodyear)*ELC_CINT(prodyear);


**OPERATION

* Electricity intensity of BEV vehicle operation [kwh el required per km]
* General logistic function
* https://en.wikipedia.org/wiki/Generalised_logistic_function
* y = A + (B-A)/(1+exp(-r*(t-tau)));
* Y  is the electricity intensity of BEV vehicle operation [kwh el required per km]
* t  is time.
* A = Initial electricity intensity of BEV Vehicle prodution = 0.21 kwh/km
* B = End electricity intensity of BEV Vehicle prodution = 0.11 kwh/km
* r  is the the rate of change = 0.15 ;
* tau is time the of maximum gradient of = 2030
BEV_OPER_EINT(prodyear) = 0.21 + (0.11-0.21)/(1+exp(-0.15*(YEAR_PAR(prodyear)-2025)));

* CO2 intensity of BEV vehicle operation  [kg CO2 per km]
BEV_OPER_CINT(prodyear) = BEV_OPER_EINT(prodyear)*ELC_CINT(prodyear);


**EOL

* CO2 intensity of BEV vehicle EOL - [kg CO2-eq per vehicle in EOL treatment]
* General logistic function
* https://en.wikipedia.org/wiki/Generalised_logistic_function
* y = A + (B-A)/(1+exp(-r*(t-tau)));
* Y  is the CO2 eq. CO2 intensity of BEV vehicle EOL
* t  is time.
* A = Initial CO2 intensity of BEV EOL = 100 kgCO2 e.g pr/km
* B = End electricity intensity of BEV EOL = 35 kgCO2 e.g pr/km
* r  is the the rate of change = 0.15 ;
* tau is time the of maximum gradient of = 2025
*CO2 intensity of BEV vehicle operation  [kg CO2 per km]
BEV_EOLT_CINT(prodyear) = 100 + (35-100)/(1+exp(-0.15*(YEAR_PAR(prodyear)-2025)));



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
TOTC                             Total CO2 emissions for the whole system over the whole period
BEV_STCK_DELTA(year)             Delta stock from one year to the next
ICE_STCK_DELTA(year)             Delta stock from one year to the next


POSITIVE VARIABLES
VEH_STCK(age,year)               Number of vehicles of a given age in a given year
VEH_STCK_REM(age,year)           Number of vehicles of a given age retired in a given year
VEH_STCK_TOT_CHECK(year)         Test on number of vehicles
VEH_STCK_DELTA(year)             Delta stock from one year to the next
VEH_STCK_ADD(age,year)           Stock additions (new car sales)

ICE_STCK(age,year)               Number of vehicles of a given age in a given year
ICE_STCK_REM(age,year)           Number of vehicles of a given age retired in a given year
ICE_STCK_TOT_CHECK(year)         Test on number of vehicles
ICE_STCK_ADD(age,year)           Stock additions (new car sales)


BEV_STCK(age,year)               Number of vehicles of a given age in a given year
BEV_STCK_REM(age,year)           Number of vehicles of a given age retired in a given year
BEV_STCK_TOT_CHECK(year)         Test on number of vehicles
BEV_STCK_ADD(age,year)           Stock additions (new car sales)

BEV_TOTC(year)                   Total CO2 emissions of BEV vehicles per year
ICE_TOTC(year)                   Total CO2 emissions of ICE vehicles per year

BEV_PROD_TOTC(year)              Total CO2 emissions from production of BEV vehicles per year
BEV_OPER_TOTC(year)              Total CO2 emissions from operations of BEV vehicles per year
BEV_EOLT_TOTC(year)              Total CO2 emissions from operations of BEV vehicles per year

ICE_PROD_TOTC(year)              Total CO2 emissions from production of ICE vehicles per year
ICE_OPER_TOTC(year)              Total CO2 emissions from operations of ICE vehicles per year
ICE_EOLT_TOTC(year)              Total CO2 emissions from operations of ICE vehicles per year




********************************************************************************
********************************************************************************
*
* Model Definition  p.t 2 : Model Equation Declarations
*
********************************************************************************
********************************************************************************

EQUATIONS

***VEH STOCK INITIALISATION************************************************************
EQ_VEH_STCK_DELTA_I        Changes in vehicle stock
EQ_STCK_MOD_O
EQ_STCK_MOD_I
EQ_STCK_MOD_II
EQ_STCK_MOD_III
EQ_STCK_MOD_SC

***ICE and BEV STOCK INITIALISATION************************************************************
EQ_ICE_STCK_INIT
EQ_BEV_STCK_INIT


***ICE STOCK MODEL************************************************************
EQ_ICE_STCK_DELTA
EQ_ICE_STCK_MOD_I
EQ_ICE_STCK_MOD_II
EQ_ICE_STCK_MOD_III
EQ_ICE_STCK_MOD_SC

***BEV STOCK MODEL************************************************************
EQ_BEV_STCK_DELTA
EQ_BEV_STCK_MOD_I
EQ_BEV_STCK_MOD_II
EQ_BEV_STCK_MOD_III
EQ_BEV_STCK_MOD_SC

**COMBINED ICE + BEV STOCK ADD MODEL ************************************************************
EQ_CMB_STCK_MOD_II

**TOTAL BEV+ICE STOCK DEMAND************************************************************
EQ_STOCK
EQ_VEH_STCK_DELTA_II
EQ_BEV_GRAD_I

**EMISSION and ENERGY MODELS incl OBJ. FUNCTION ************************************************************

EQ_TOTC                  Total CO2 emissions of the whole system

EQ_BEV_TOTC              Total CO2 emissions from BEVs per year
EQ_BEV_PROD_TOTC         Total CO2 emissions from production of BEVs
EQ_BEV_OPER_TOTC         Total CO2 emissions from operations of BEVs
EQ_BEV_EOLT_TOTC         Total CO2 emissions from operations of BEVs


EQ_ICE_TOTC              Total CO2 emissions from ICEs per year
EQ_ICE_PROD_TOTC         Total CO2 emissions from production of ICEs
EQ_ICE_OPER_TOTC         Total CO2 emissions from operations of ICEs
EQ_ICE_EOLT_TOTC         Total CO2 emissions from operati
;
********************************************************************************
********************************************************************************
*
* Model Definition  p.t 3 : Model Equations
*
********************************************************************************
********************************************************************************


***VEH STOCK INITIALISATION************************************************************

* Initiate stock age in first year
EQ_VEH_STCK_DELTA_I(inityear)$(ord(inityear)>1)..                        VEH_STCK_DELTA(inityear)  =e=  VEH_STCK_TOT(inityear)-VEH_STCK_TOT(inityear-1);

*initializing stock in starting year
EQ_STCK_MOD_O(inityear,age)$(ord(inityear)=1)..                          VEH_STCK(age,inityear) =e=  VEH_STCK_TOT(inityear)*VEH_INIT_AGE_DIST(age);

*calculating vehicle removals in a given year
EQ_STCK_MOD_I(inityear,age)$(ord(inityear)>1)..                          VEH_STCK_REM(age,inityear) =e= VEH_STCK(age-1,inityear-1)*VEH_LIFT_DSTR(age-1);

*calculating vehicle additions in a given year - all new cars go in age class 1
EQ_STCK_MOD_II(inityear,age)$(ord(inityear)>1 and ord(age)=1)..          VEH_STCK_ADD(age,inityear)  =e=  VEH_STCK_DELTA(inityear) + sum( (agej), VEH_STCK(agej,inityear-1)*VEH_LIFT_DSTR(agej) );

*calculating vehicle stock in a given year
EQ_STCK_MOD_III(inityear,age)$(ord(inityear)>1)..                        VEH_STCK(age,inityear)  =e=  VEH_STCK(age-1,inityear-1) + VEH_STCK_ADD(age,inityear)- VEH_STCK_REM(age,inityear);

*summing the number of vehicles in the init stock.
EQ_STCK_MOD_SC(inityear)..                                               VEH_STCK_TOT_CHECK(inityear) =e= sum((age), VEH_STCK(age,inityear));


***ICE and BEV STOCK INITIALISATION************************************************************

*Defining initial stock of ICE Vehicles
EQ_ICE_STCK_INIT(age,inityear)..                                         ICE_STCK(age,inityear) =e= VEH_STCK(age,inityear);

*Defining initial stock of BEV Vehicles
EQ_BEV_STCK_INIT(age,inityear)..                                         BEV_STCK(age,inityear) =e= 0;


***ICE STOCK MODEL************************************************************

EQ_ICE_STCK_DELTA(optyear)$(ord(optyear)>1)..                            ICE_STCK_DELTA(optyear)  =e=  sum( (agej), ICE_STCK(agej,optyear)-ICE_STCK(agej,optyear-1) );

*calculating vehicle removals in a given year

EQ_ICE_STCK_MOD_I(optyear,age)$(ord(optyear)>1)..                        ICE_STCK_REM(age,optyear) =e= ICE_STCK(age-1,optyear-1)*VEH_LIFT_DSTR(age-1);

*calculating vehicle additions in a given year - all new cars go in age class 1
EQ_ICE_STCK_MOD_II(optyear,age)$(ord(optyear)>1 and ord(age)=1)..        ICE_STCK_ADD(age,optyear)  =e=  ICE_STCK_DELTA(optyear) + sum( (agej), ICE_STCK(agej,optyear-1)*VEH_LIFT_DSTR(agej) );

*calculating vehicle stock in a given year
EQ_ICE_STCK_MOD_III(optyear,age)$(ord(optyear)>1)..                      ICE_STCK(age,optyear)  =e=  ICE_STCK(age-1,optyear-1) + ICE_STCK_ADD(age,optyear)- ICE_STCK_REM(age,optyear);

*summing the number of vehicles in the init stock.
EQ_ICE_STCK_MOD_SC(optyear)..                                            ICE_STCK_TOT_CHECK(optyear) =e= sum((age), ICE_STCK(age,optyear));


**BEV STOCK MODEL************************************************************

EQ_BEV_STCK_DELTA(optyear)$(ord(optyear)>1)..                            BEV_STCK_DELTA(optyear)  =e=  sum( (agej), BEV_STCK(agej,optyear)-BEV_STCK(agej,optyear-1) );

*calculating vehicle removals in a given year
EQ_BEV_STCK_MOD_I(optyear,age)$(ord(optyear)>1)..                        BEV_STCK_REM(age,optyear) =e= BEV_STCK(age-1,optyear-1)*VEH_LIFT_DSTR(age-1);

*calculating vehicle additions in a given year - all new cars go in age class 1
EQ_BEV_STCK_MOD_II(optyear,age)$(ord(optyear)>1 and ord(age)=1)..        BEV_STCK_ADD(age,optyear)  =e=  BEV_STCK_DELTA(optyear) + sum( (agej), BEV_STCK(agej,optyear-1)*VEH_LIFT_DSTR(agej) );

*calculating vehicle stock in a given year
EQ_BEV_STCK_MOD_III(optyear,age)$(ord(optyear)>1)..                      BEV_STCK(age,optyear)  =e=  BEV_STCK(age-1,optyear-1) + BEV_STCK_ADD(age,optyear)- BEV_STCK_REM(age,optyear);

*summing the number of vehicles in the init stock.
EQ_BEV_STCK_MOD_SC(optyear)..                                            BEV_STCK_TOT_CHECK(optyear) =e= sum((age), BEV_STCK(age,optyear));


**COMBINED ICE + BEV STOCK ADD MODEL ************************************************************

EQ_CMB_STCK_MOD_II(optyear,age)$(ord(optyear)>1 and ord(age)=0)..        ICE_STCK_ADD(age,optyear) + BEV_STCK_ADD(age,optyear)  =e=  BEV_STCK_DELTA(optyear) + sum( (agej), BEV_STCK(agej,optyear-1)*VEH_LIFT_DSTR(agej) )+ ICE_STCK_DELTA(optyear) + sum( (agej), ICE_STCK(agej,optyear-1)*VEH_LIFT_DSTR(agej) );


**TOTAL BEV+ICE STOCK DEMAND************************************************************


* Stock demand driven by VEH_STCK_TOT
EQ_STOCK(optyear)..                                                      VEH_STCK_TOT(optyear) =e= sum( (agej), BEV_STCK(agej,optyear) + ICE_STCK(agej,optyear) );

* Stock demand driven by VEH_STCK_TOT - VEH_STCK_DELTA p.t not used - just for check
EQ_VEH_STCK_DELTA_II(optyear)$(ord(optyear)>1)..                         VEH_STCK_DELTA(optyear)  =e=  VEH_STCK_TOT(optyear)-VEH_STCK_TOT(optyear-1);

* Gradient of Change dampening constraint
EQ_BEV_GRAD_I(optyear,age)$(ord(age)=1)..                                BEV_STCK_ADD(age,optyear) =l= BEV_STCK_ADD(age,optyear-1) + VEH_STCK_DELTA(optyear);


**EMISSION and ENERGY MODELS incl OBJ. FUNCTION ************************************************************

* Objective function
EQ_TOTC..                        TOTC =e= SUM( (optyear), BEV_TOTC(optyear)+ICE_TOTC(optyear) );


* Calculation of Emissions from BEVs
EQ_BEV_TOTC(optyear)..           BEV_TOTC(optyear) =e= BEV_PROD_TOTC(optyear) + BEV_OPER_TOTC(optyear);

EQ_BEV_PROD_TOTC(optyear)..      BEV_PROD_TOTC(optyear) =e= sum( (agej), BEV_STCK_ADD(agej,optyear))*BEV_PROD_CINT(optyear);
EQ_BEV_OPER_TOTC(optyear)..      BEV_OPER_TOTC(optyear) =e= sum( (agej), BEV_STCK(agej,optyear)*BEV_OPER_CINT(optyear)*VEH_OPER_DIST(optyear));
EQ_BEV_EOLT_TOTC(optyear)..      BEV_EOLT_TOTC(optyear) =e= sum( (agej), BEV_STCK_REM(agej, optyear))*BEV_EOLT_CINT(optyear);


* Calculation of Emissions from ICEs
EQ_ICE_TOTC(optyear)..           ICE_TOTC(optyear) =e= ICE_PROD_TOTC(optyear) + ICE_OPER_TOTC(optyear);

EQ_ICE_PROD_TOTC(optyear)..      ICE_PROD_TOTC(optyear) =e= sum( (agej), ICE_STCK_ADD(agej, optyear))*ICE_PROD_CINT(optyear);
EQ_ICE_OPER_TOTC(optyear)..      ICE_OPER_TOTC(optyear) =e= sum( (agej), ICE_STCK(agej,optyear)*ICE_OPER_CINT(optyear)*VEH_OPER_DIST(optyear));
EQ_ICE_EOLT_TOTC(optyear)..      ICE_EOLT_TOTC(optyear) =e= sum( (agej), ICE_STCK_REM(agej,optyear))*ICE_EOLT_CINT(optyear);


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


********************************************************************************
********************************************************************************
*
* Model Post Processing  p.t 1 : Parameter calculations
*
********************************************************************************
********************************************************************************

PARAMETER



*share of BEVs in stock for a given year and age
BEV_STCK_FRAC_I

*share of BEVs in stock for a given year
BEV_STCK_FRAC_II

*share of BEVs in stock for a given year
BEV_ADD_FRAC


ICE_STCK_TOT

BEV_STCK_TOT

;

*share of BEVs in stock for a given year and age
BEV_STCK_FRAC_I(age,optyear)$(ord(age)<20)= BEV_STCK.l(age,optyear)/( BEV_STCK.l(age,optyear) + ICE_STCK.l(age,optyear) );

*share of BEVs in stock for a given year
BEV_STCK_FRAC_II(optyear) = sum( (agej), BEV_STCK.l(agej,optyear)) /sum( (agej), BEV_STCK.l(agej,optyear) + ICE_STCK.l(agej,optyear) );

*share of BEVs in stock for a given year
BEV_ADD_FRAC(age,optyear)$(ord(age)=1 and ord(optyear)>1)= BEV_STCK_ADD.l(age,optyear)/( BEV_STCK_ADD.l(age,optyear) + ICE_STCK_ADD.l(age,optyear) );

ICE_STCK_TOT(optyear) = sum( (agej),  ICE_STCK.l(agej,optyear) );

BEV_STCK_TOT(optyear) =sum( (agej), BEV_STCK.l(agej,optyear) );



********************************************************************************
********************************************************************************
*
* Model Post Processing  p.t 2 : Display Calls
*
********************************************************************************
********************************************************************************


display YEAR_PAR
display ELC_CINT

display ICE_PROD_EINT
display ICE_PROD_CINT_CSNT
display ICE_PROD_CINT
display ICE_OPER_CINT
display ICE_EOLT_CINT

display BEV_PROD_EINT
display BEV_PROD_CINT_CSNT
display BEV_PROD_CINT
display BEV_OPER_EINT
display BEV_OPER_CINT
display BEV_EOLT_CINT

display VEH_OPER_DIST
display VEH_LIFT_DSTR

display VEH_INIT_AGE_DIST
display VEH_STCK_TOT
display VEH_STCK_TOT_CHECK.l

display VEH_STCK.l
display VEH_STCK_DELTA.l
display VEH_STCK_REM.l
display VEH_STCK_ADD.l


display ICE_STCK_TOT
display ICE_STCK.l
display ICE_STCK_DELTA.l
display ICE_STCK_REM.l
display ICE_STCK_ADD.l

display BEV_STCK_TOT
display BEV_STCK.l
display BEV_STCK_DELTA.l
display BEV_STCK_REM.l
display BEV_STCK_ADD.l

display BEV_STCK_FRAC_I
display BEV_STCK_FRAC_II
display BEV_ADD_FRAC


display TOTC.l

display BEV_TOTC.l
display BEV_PROD_TOTC.l
display BEV_OPER_TOTC.l
display BEV_EOLT_TOTC.l


display ICE_TOTC.l
display ICE_PROD_TOTC.l
display ICE_OPER_TOTC.l
display ICE_EOLT_TOTC.l

execute_unload 'EVD4EUR_ALL_23Jan2019.gdx'


*$gdxout EVD4EUR
*$unload year
*$unload optyear
*$unload inityear
*$unload age
*$unload ICE_STCK
*$unload BEV_STCK age optyear
*$unload BEV_ADD_FRAC.l age optyear
*$unload BEV_STCK_FRAC.l optyear
*$gdxout