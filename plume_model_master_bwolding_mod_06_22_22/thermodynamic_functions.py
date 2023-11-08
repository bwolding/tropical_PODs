import numpy as np

#### Thermodyanmic constants ###
pref = 100000.
tmelt  = 273.15
es0    = 610.78
grav = 9.81
es02 = 3168.0
t02 = 298.16
CPD=1005.7
CPV=1870.0
CL=4190.0
CPVMCL=2320.0
RV=461.5
RD=287.04
EPS=RD/RV
ALV0=2.501e6


def temp_v_calc (temp, q, ql):

    # get some constants:
    RV=461.5
    RD=287.04
    EPS=RD/RV

    # convert inputs to proper units, forms 
    r = q / (1. - q - ql)
    rl =  ql / (1. - q - ql)
    rt = r + rl

    # calc. temp_v
    temp_v = temp * (1. + (r/EPS)) / (1. + rt)

    return temp_v


def es_calc(temp):


    #get some constants:
    tmelt  = 273.15

    #convert inputs to proper units, forms 
    tempc = temp - tmelt # in C
    tempcorig = tempc
    c=np.array((0.6105851e+03,0.4440316e+02,0.1430341e+01,0.2641412e-01,0.2995057e-03,0.2031998e-05,0.6936113e-08,0.2564861e-11,-.3704404e-13))

    #calc. es in hPa (!!!)
    #es = 6.112*EXP(17.67*tempc/(243.5+tempc))
    es=c[0]+tempc*(c[1]+tempc*(c[2]+tempc*(c[3]+tempc*(c[4]+tempc*(c[5]+tempc*(c[6]+tempc*(c[7]+tempc*c[8])))))))
    es = es/100.

#     ;put in Bolton values for T < -80. C
#     junk = check_math(/print) ;;print out any prev. math errors
    lowtempc = np.where(tempc<-80)[0]
#     es[lowtempc]=es_calc_bolton(temp[lowtempc])
    return es

def esi_calc(temp):

    #get some constants:
    #calc. es
    esi = np.exp(23.33086 - (6111.72784/temp) + (0.15215 * np.log(temp)))
    return esi


def es_calc_bolton(temp):

    # ; get some constants:
    # ;pref = 100000.
    tmelt  = 273.15
    # ; convert inputs to proper units, forms 
    # ;press = press_hPa * 100. ; in Pa
    tempc = temp - tmelt # in C

    #calc. es
    es = 6.112*np.exp(17.67*tempc/(243.5+tempc))


    return es


def qsi_calc(press_hPa, temp):

    #get some constants:
    #pref = 100000.
    tmelt  = 273.15
    RV=461.5
    RD=287.04
    EPS=RD/RV

    #convert inputs to proper units, forms 
    press = press_hPa * 100. # in Pa

    #get esi in Pa
    esi = esi_calc(temp) * 100.

    #calc. qs in g/kg
    qsi = (EPS * esi) / (press + ((EPS-1.)*esi))

    return qsi

def qs_calc(press_hPa, temp):

    #get some constants:
    tmelt  = 273.15
    RV=461.5
    RD=287.04
    EPS=RD/RV

    #convert inputs to proper units, forms 
    press = press_hPa * 100. # in Pa
    tempc = temp - tmelt # in C

    #get es in Pa
    es = es_calc(temp) * 100.

    #calc. qs in kg/kg
    qs = (EPS * es) / (press + ((EPS-1.)*es))
    qs = qs

    return qs

def theta_l_calc(press_hPa, temp, q, ql):

    #get some constants:
    pref = 100000.
    tmelt  = 273.15
    CPD=1005.7
    CPV=1870.0
    CPVMCL=2320.0
    RV=461.5
    RD=287.04
    EPS=RD/RV
    ALV0=2.501E6

    ##convert inputs to proper units, forms 
    press = press_hPa * 100. # in Pa
    tempc = temp - tmelt # in C
    r = q / (1. - q - ql)
    rl =  ql / (1. - q - ql)
    rt = r + rl

    #get Lv
    ALV = ALV0 - (CPVMCL * tempc)

    #calc chi and gam:
    chi = (RD + (rt * RV)) / (CPD + (rt * CPV))
    gam = (rt * RV) / (CPD + (rt * CPV))

    #calc. theta_l
    theta_l = temp * (pref / press)**chi * (1. - (rl/(EPS + rt)))**chi* (1. - (rl/rt))**(-gam) * np.exp((-ALV)*rl/((CPD+(rt*CPV))*temp))

    return theta_l


def theta_v_calc(press_hPa, temp, q, ql):

    #get some constants:
    pref = 100000.
    tmelt  = 273.15
    CPD=1005.7
    RD=287.04

    #convert inputs to proper units, forms 
    press = press_hPa * 100. # in Pa

    #get temp_v
    temp_v = temp_v_calc(temp, q, ql)

    #calc. theta_v
    theta_v = temp_v * ((pref/press)**(RD/CPD))

    return theta_v

def theta_il_calc(press_hPa, temp, q, ql, qi):
    
    #get some constants:
    pref = 100000.
    tmelt  = 273.15
    CPD=1005.7
    CPV=1870.0
    CPVMCL=2320.0
    RV=461.5
    RD=287.04
    EPS=RD/RV
    ALV0=2.501E6
    LS=2.834E6

    #convert inputs to proper units, forms 
    press = press_hPa * 100. # in Pa
    tempc = temp - tmelt # in C
    r = q / (1. - q - ql - qi)
    rl =  ql / (1. - q - ql - qi)
    ri =  qi / (1. - q - ql - qi)
    rt = r + rl + ri

    #get Lv
    ALV = ALV0 - (CPVMCL * tempc)

    #calc chi and gam:
    chi = (RD + (rt * RV)) / (CPD + (rt * CPV))
    gam = (rt * RV) / (CPD + (rt * CPV))
    
    #calc. theta_il
    theta_il = temp * (pref / press)**chi * (1. - ((rl + ri)/(EPS + rt)))**chi* (1. - ((rl + ri)/rt))**(-gam) * np.exp(((-ALV)*rl - LS*ri)/((CPD+(rt*CPV))*temp))


    return theta_il


def theta_e_calc (press_hPa, temp, q):

    #get some constants:
    pref = 100000.
    tmelt  = 273.15
    CPD=1005.7
    CPV=1870.0
    CPVMCL=2320.0
    RV=461.5
    RD=287.04
    EPS=RD/RV
    ALV0=2.501E6

    #convert inputs to proper units, forms 
    press = press_hPa * 100. # in Pa
    tempc = temp - tmelt # in C

    r = q / (1. - q)

    # get ev in hPa 
    ev_hPa = press_hPa * r / (EPS + r)

    #get TL
    TL = (2840. / ((3.5*np.log(temp)) - (np.log(ev_hPa)) - 4.805)) + 55.

    #calc chi_e:
    chi_e = 0.2854 * (1. - (0.28*r))

    theta_e = temp * (pref / press)**chi_e * np.exp(((3.376/TL) - 0.00254) * r * 1000. * (1. + (0.81 * r)))
    return theta_e

def theta_calc(press_hPa, temp):

    #get some constants:
    pref = 100000.
    CPD=1005.7
    RD=287.04

    #convert inputs to proper units, forms 
    press = press_hPa * 100. # in Pa

    #calc. theta_l
    theta = temp * (pref/press)**(RD/CPD)


def temp_calc(press_hPa,theta_l,qt):

    ## Only designed to work on single elements for now ##

    ## get some constants ##
    pref = 100000.
    tmelt  = 273.15
    CPD=1005.7
    CPVMCL=2320.0
    RV=461.5
    RD=287.04
    EPS=RD/RV
    ALV0=2.501E6

    #convert inputs to proper units, forms 
    press = press_hPa * 100. # in Pa
    rt = qt / (1. - qt)

    #get TL, tentative T, rs, ql
    TL = theta_l * ((press / pref)**(RD/CPD))
    qs = qs_calc(press_hPa, TL)
    ql = qt - qs if qt>qs else 0
    rs = qs / (1. - qs - ql)
    rl = rt- rs if rt>rs else 0
    temp = TL

    ## Iterative procedure to get T
    deltaT = 999.
    while (abs(deltaT) >= 1e-3):
        ALV= ALV0
        x1=ALV/(CPD*temp)
        deltaT = (temp - TL*(1.+x1*rl))/(1. + x1*TL*(rl/temp+(1.+rs*EPS)*rs*ALV/(RV*temp*temp)))
        temp-= deltaT
        qs = qs_calc(press_hPa, temp)
        ql = qt - qs if qt>qs else 0
        q  = qt if qt<qs else qs
        rs = qs/(1. - qs - ql)
        rl = rt - rs if rt>rs else 0

    ##do additional iteration to make sure final theta_l
    ##agrees with the initial theta_l
    theta_l_new = theta_l_calc(press_hPa,temp, q,ql)
    diff_tl = theta_l_new - theta_l

    while(abs(diff_tl) > 0.05):
        if (abs(diff_tl) < 1.):
            deltaT = 0.001 * diff_tl/(abs(diff_tl))
        else:
            deltaT = diff_tl/3.

        temp-=deltaT
        qs = qs_calc(press_hPa, temp)
        ql = qt - qs if qt>qs else 0.
        q  = qt if qt < qs else qs

        theta_l_new = theta_l_calc(press_hPa,temp,q,ql)
        diff_tl = theta_l_new - theta_l

    return temp

def temp_i_calc(press_hPa, theta_il, qt):
    ## Get temperature from theta_il and all condensed matter in qt is ice ##

    #get some constants:
    pref = 100000.
    tmelt  = 273.15
    CPD=1005.7
    CPVMCL=2320.0
    RV=461.5
    RD=287.04
    EPS=RD/RV
    ALV0=2.501E6
    LS=2.834E6

    #convert inputs to proper units, forms 
    press = press_hPa * 100. # in Pa
    rt = qt / (1. - qt)

    #get TL, tentative T, rs, qi
    TL = theta_il * ((press / pref)**(RD/CPD))
    qs = qsi_calc(press_hPa, TL) 
    qi = qt - qs if qt>qs else 0.
    rs = qs / (1. - qs - qi)
    ri = rt - rs if rt>rs else 0.
    temp = TL

    deltaT = 999.
    while (abs(deltaT) >= 1e-3):
        x1=LS/(CPD*temp)
        deltaT = (temp - TL*(1.+x1*ri))/(1. + x1*TL*(ri/temp+(1.+rs*EPS)*rs*LS/(RV*temp*temp)))
        temp = temp - deltaT
        qs = qsi_calc(press_hPa, temp)
        qi = qt - qs if qt>qs else 0.
        q =  qt if qt<qs else qs
        rs = qs / (1. - qs - qi)
        ri = rt - rs if rt>rs else 0
    # do additional iteration to make sure final theta_il
    # agrees with the initial theta_il
    theta_il_new = theta_il_calc(press_hPa,temp,q,0.0,qi)
    diff_tl = theta_il_new - theta_il
    while(abs(diff_tl) >= 0.05):
            if (abs(diff_tl) < 1.):
                deltaT = 0.001 * diff_tl/(abs(diff_tl))
            else:
                deltaT = diff_tl/3.

            temp-=deltaT
            qs = qsi_calc(press_hPa, temp)
            qi = qt - qs if qt >qs else 0
            q = qt if qt < qs else qs
            theta_il_new = theta_il_calc(press_hPa,temp,q,0.0,qi)
            diff_tl = theta_il_new - theta_il

    return temp


